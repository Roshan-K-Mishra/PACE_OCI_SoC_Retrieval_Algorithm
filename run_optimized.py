"""
OPTIMIZED PACE SoC retrieval.

Numerics are identical to the original (same NN, same cost function, same
9 initial guesses, same L-BFGS settings, same Rodgers diagnostics, same
solution-selection rule). What changed is HOW it is executed:

1. BATCHED L-BFGS: all pixels in a tile x 9 initial guesses are solved as
   one batch of independent problems by tfp.optimizer.lbfgs_minimize
   (it natively supports batching). The NN forward model is evaluated on
   the whole batch at once instead of one sample at a time.
2. GRAPH COMPILATION: the objective and the entire L-BFGS solve run inside
   @tf.function (graph mode), removing per-op Python dispatch overhead.
3. BATCHED JACOBIAN: uncertainty diagnostics use tape.batch_jacobian once
   per tile instead of tape.jacobian once per pixel.
4. VECTORIZED DIAGNOSTICS: the 3x3 Rodgers linear algebra (Sx, A, DOFS,
   SIC) is done for all pixels at once with stacked NumPy operations.

Usage: python run_optimized.py <inputs.npz> <start_pixel> <n_pixels> <out.npz> [tile]
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, joblib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model

inp_file, start, n_pix, out_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
TILE = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
repo = os.environ.get('REPO_DIR', '.')

d = np.load(inp_file)

Eval_input_all  = d["Eval_input"]
Eval_output_all = d["Eval_output"]

# If n_pix <= 0, process all pixels from start to end
if n_pix <= 0:
    n_pix = Eval_input_all.shape[0] - start

Eval_input  = Eval_input_all[start:start + n_pix]
Eval_output = Eval_output_all[start:start + n_pix]

# Use valid mask from make_inputs.py if available.
# Otherwise build one here.
if "valid" in d.files:
    valid_input = d["valid"][start:start + n_pix].astype(bool)
else:
    valid_input = (
        np.isfinite(Eval_input).all(axis=1) &
        np.isfinite(Eval_output).all(axis=1)
    )

# Extra safety for real data
valid_input = (
    valid_input &
    np.isfinite(Eval_input).all(axis=1) &
    np.isfinite(Eval_output).all(axis=1) &
    (Eval_output > 0).all(axis=1)
)

print(f"Requested pixels: {Eval_input.shape[0]}")
print(f"Valid pixels: {valid_input.sum()}")
print(f"Invalid pixels skipped: {(~valid_input).sum()}")

model  = load_model(os.path.join(repo, "NN_model_cs_V_2_13_reduced_20250808_1218.keras"))
Z_mean = tf.constant(joblib.load(os.path.join(repo, "Z_mean_cs.pkl")), dtype=tf.float32)
Z_std  = tf.constant(joblib.load(os.path.join(repo, "Z_std_cs.pkl")),  dtype=tf.float32)

Xa  = tf.constant([15., 10., 0.4], dtype=tf.float32)
Sig = tf.constant([25., 50., 3.],  dtype=tf.float32)
Sa  = np.diag([25., 50., 3.])**2
Sa  = Sa.astype(np.float64)

INITS = np.array([
    [6.0, 0.3, 0.5], [6.0, 16.0, 2.0], [6.0, 60.0, 4.0],
    [15.0, 0.3, 4.0], [15.0, 16.0, 0.5], [15.0, 60.0, 2.0],
    [25.0, 0.3, 2.0], [25.0, 16.0, 4.0], [25.0, 60.0, 0.5]], dtype=np.float32)
N_INIT = INITS.shape[0]


@tf.function(reduce_retracing=True)
def solve_tile(geom, obs, meas_err, init_log):
    """geom (B,3), obs (B,6), meas_err (B,6), init_log (B,3) -> batched LBFGS."""

    def value_and_gradients(log_state):                       # (B, 3)
        with tf.GradientTape() as tape:
            tape.watch(log_state)
            x = tf.exp(log_state)                             # (B, 3) CER, COT, AOD
            input_raw = tf.stack(
                [geom[:, 0], x[:, 0], x[:, 1], x[:, 2], geom[:, 1], geom[:, 2]],
                axis=1)                                       # (B, 6) [SZA,CER,COT,AOD,VZA,RAA]
            input_vec = (input_raw - Z_mean) / Z_std
            y_hat = model(input_vec, training=False)          # (B, 6)
            r = obs - y_hat
            fit_cost   = tf.reduce_sum((r / meas_err) ** 2, axis=1)
            prior_cost = tf.reduce_sum(((x - Xa) / Sig) ** 2, axis=1)
            cost = 0.5 * fit_cost + 0.5 * prior_cost          # (B,)
        grad = tape.gradient(cost, log_state)                 # (B, 3)
        return cost, grad

    return tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=value_and_gradients,
        initial_position=init_log,
        tolerance=1e-5,
        max_iterations=800,
        max_line_search_iterations=100)


@tf.function(reduce_retracing=True)
def predict_and_jacobian(geom, best_state):
    """geom (P,3), best_state (P,3) -> pred (P,6), K_log (P,6,3)."""
    log_best = tf.math.log(best_state)
    with tf.GradientTape() as tape:
        tape.watch(log_best)
        x = tf.exp(log_best)
        input_raw = tf.stack(
            [geom[:, 0], x[:, 0], x[:, 1], x[:, 2], geom[:, 1], geom[:, 2]], axis=1)
        input_vec = (input_raw - Z_mean) / Z_std
        pred = model(input_vec, training=False)               # (P, 6)
    K_log = tape.batch_jacobian(pred, log_best)               # (P, 6, 3)
    return pred, K_log


P_total = Eval_input.shape[0]
state_out   = np.full((P_total, 3), np.nan, dtype=np.float64)
conv_out    = np.zeros(P_total, dtype=bool)
cost_out    = np.full(P_total, np.nan, dtype=np.float64)
poststd_out = np.full((P_total, 3), np.nan, dtype=np.float64)
dofs_out    = np.full((P_total, 4), np.nan, dtype=np.float64)
sic_out     = np.full((P_total, 4), np.nan, dtype=np.float64)
predR_out   = np.full((P_total, 6), np.nan, dtype=np.float32)
valid_out   = np.zeros(P_total, dtype=bool)

t0 = time.perf_counter()
for t_start in range(0, P_total, TILE):
    t_end = min(t_start + TILE, P_total)

    tile_valid = valid_input[t_start:t_end]
    local_valid_idx = np.where(tile_valid)[0]

    # If no valid pixels in this tile, skip it
    if local_valid_idx.size == 0:
        print(f"Tile {t_start}:{t_end} has no valid pixels. Skipping.")
        continue

    geom_np = Eval_input[t_start:t_end][local_valid_idx]
    obs_np  = Eval_output[t_start:t_end][local_valid_idx]

    P = geom_np.shape[0]

    err_np = np.maximum(0.05 * np.abs(obs_np), 1e-6).astype(np.float32)

    # Tile each pixel over the 9 initial guesses -> batch B = P * 9
    geom_b = np.repeat(geom_np, N_INIT, axis=0)
    obs_b  = np.repeat(obs_np,  N_INIT, axis=0)
    err_b  = np.repeat(err_np,  N_INIT, axis=0)
    init_b = np.log(np.tile(INITS, (P, 1))).astype(np.float32)

    res = solve_tile(tf.constant(geom_b), tf.constant(obs_b),
                     tf.constant(err_b), tf.constant(init_b))

    pos   = np.exp(res.position.numpy()).reshape(P, N_INIT, 3).astype(np.float64)
    objv  = res.objective_value.numpy().reshape(P, N_INIT).astype(np.float64)
    convd = res.converged.numpy().reshape(P, N_INIT)

    # ---- per-pixel selection: prefer converged & finite, else best finite ----
    finite = np.isfinite(pos).all(axis=2) & np.isfinite(objv)
    cand1 = np.where(convd & finite, objv, np.inf)
    cand2 = np.where(finite,         objv, np.inf)
    use1  = np.isfinite(cand1).any(axis=1)
    pick  = np.where(use1, np.argmin(cand1, axis=1), np.argmin(cand2, axis=1))
    valid = use1 | np.isfinite(cand2).any(axis=1)

    rows = np.arange(P)
    best_state = pos[rows, pick]                              # (P, 3)
    best_cost  = objv[rows, pick]
    best_conv  = convd[rows, pick]

    # ---- diagnostics for valid pixels ----
    v = np.where(valid)[0]
    if v.size:
        pred, K_log = predict_and_jacobian(
            tf.constant(geom_np[v]), tf.constant(best_state[v], dtype=tf.float32))
        pred  = pred.numpy()
        K     = K_log.numpy().astype(np.float64) / best_state[v][:, None, :]  # (Pv,6,3)
        sigma = err_np[v].astype(np.float64)                  # (Pv, 6)

        Se_inv_K  = K / (sigma ** 2)[:, :, None]              # rows scaled
        KT_Se_inv = np.swapaxes(Se_inv_K, 1, 2)               # (Pv, 3, 6) = K^T Se^-1
        Sa_inv    = np.linalg.inv(Sa)
        Sx        = np.linalg.inv(KT_Se_inv @ K + Sa_inv)     # (Pv, 3, 3)
        A         = Sx @ KT_Se_inv @ K                        # (Pv, 3, 3)

        dofs_diag  = np.diagonal(A, axis1=1, axis2=2)
        dofs_total = np.trace(A, axis1=1, axis2=2)
        post_stds  = np.sqrt(np.diagonal(Sx, axis1=1, axis2=2))
        _, logdet_a = np.linalg.slogdet(Sa)
        _, logdet_x = np.linalg.slogdet(Sx)
        SIC_total = 0.5 * (logdet_a - logdet_x) / np.log(2.0)
        SIC_per   = 0.5 * np.log2(np.diag(Sa)[None, :] /
                                  np.diagonal(Sx, axis1=1, axis2=2))

        g = t_start + local_valid_idx[v]
        
        state_out[g]   = best_state[v]
        conv_out[g]    = best_conv[v]
        cost_out[g]    = best_cost[v]
        poststd_out[g] = post_stds
        dofs_out[g]    = np.column_stack([dofs_diag, dofs_total])
        sic_out[g]     = np.column_stack([SIC_per, SIC_total])
        predR_out[g]   = pred
        valid_out[g]   = True

t1 = time.perf_counter()
save_dict = dict(
    idx=np.arange(start, start + n_pix),
    state=state_out,
    converged=conv_out,
    cost=cost_out,
    post_stds=poststd_out,
    dofs=dofs_out,
    sic=sic_out,
    pred_R=predR_out,
    valid=valid_out,
    input_valid=valid_input,
    elapsed=np.array([t1 - t0]),
    n=np.array([P_total])
)

# Preserve useful real-scene metadata if present
for key in ["shape", "lat", "lon", "SZA", "VZA", "SAA", "VAA", "RAA_lib"]:
    if key in d.files:
        save_dict[key] = d[key]

np.savez(out_file, **save_dict)

print(f'OPTIMIZED: {P_total} pixels in {t1-t0:.2f}s -> {(t1-t0)/P_total*1000:.2f} ms/pixel')
