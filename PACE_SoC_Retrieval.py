import warnings
warnings.filterwarnings('ignore')

import os
import time
import joblib
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import load_model

t0 = time.perf_counter()

input_dir = '/umbc/rs/pi_zzbatmos/users/kn82613/My_WORK/PACA_SoC_Algorithm/V1/'
nc_path = os.path.join(input_dir, 'PACE_SoC_retrieval.nc')

ds = xr.open_dataset(
    os.path.join(input_dir, 'PACE_OCI.20250822T123146.L1B.V3.nc'),
    engine='netcdf4'
)

obs_data = xr.open_dataset(ds.encoding["source"], group="observation_data")
geo_data = xr.open_dataset(ds.encoding["source"], group="geolocation_data")
wavelength_data = xr.open_dataset(ds.encoding["source"], group="sensor_band_parameters")

# Extract lat/lon
lat2d = geo_data["latitude"]
lon2d = geo_data["longitude"]

# 2. Define your target lat/lon box
lat_min, lat_max = -10, -8    # S is -ve
lon_min, lon_max = 9, 11    # E is +ve

# 3. Build a boolean mask of pixels inside that box
mask = (
    (lat2d >= lat_min) & (lat2d <= lat_max) &
    (lon2d >= lon_min) & (lon2d <= lon_max)
)

# 4. Find the minimal row/col extents that cover the mask
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)

i0, i1 = np.where(rows)[0][[0, -1]]
j0, j1 = np.where(cols)[0][[0, -1]]
i1 += 1  
j1 += 1

# 5. Slice out each bands small patch
blue_sub  = obs_data['rhot_blue'].isel(blue_bands=64)[i0:i1, j0:j1]
red_sub   = obs_data['rhot_red'].isel(red_bands =33)[i0:i1, j0:j1]
green_sub = obs_data['rhot_blue'].isel(blue_bands=92)[i0:i1, j0:j1]  # correct green

# 6. Build an RGB array, scale & clamp to [0,1]
factor = 0.4
rgb_sub = np.stack([
    red_sub   / factor,
    green_sub / factor,
    blue_sub  / factor
], axis=-1)
np.clip(rgb_sub, 0, 1, out=rgb_sub)

R_472 = obs_data['rhot_blue'].isel(blue_bands=65)[i0:i1, j0:j1]
R_553 = obs_data['rhot_blue'].isel(blue_bands=97)[i0:i1, j0:j1]

R_655 = obs_data['rhot_red'].isel(red_bands=28)[i0:i1, j0:j1]
R_862 = obs_data['rhot_red'].isel(red_bands=149)[i0:i1, j0:j1]

R_1250 = obs_data['rhot_SWIR'].isel(SWIR_bands=2)[i0:i1, j0:j1]
R_2130 = obs_data['rhot_SWIR'].isel(SWIR_bands=7)[i0:i1, j0:j1]

SZA = geo_data['solar_zenith'][i0:i1, j0:j1].values
VZA = geo_data['sensor_zenith'][i0:i1, j0:j1].values
VAA = geo_data['sensor_azimuth'][i0:i1, j0:j1].values
SAA = geo_data['solar_azimuth'][i0:i1, j0:j1].values

def pace_to_libradtran(phi_sun_pace, phi_view_pace, half_range=True):
    """
    Convert PACE OCI azimuth geometry to libRadtran convention.

    Parameters
    ----------
    phi_sun_pace : float or array-like
        PACE solar azimuth angle (0 = North, clockwise)
    phi_view_pace : float or array-like
        PACE viewing azimuth angle (0 = North, clockwise)
    half_range : bool, optional
        If True, fold both RAA values into 0 to 180 (physical symmetry)

    Returns
    -------
    results : dict
        {
          'PACE_SAA' : solar azimuth (PACE),
          'PACE_VAA' : viewing azimuth (PACE),
          'PACE_RAA' : relative azimuth (PACE, 0 - 180 if half_range=True),
          'LIB_SAA'  : solar azimuth (libRadtran),
          'LIB_VAA'  : viewing azimuth (libRadtran),
          'LIB_RAA'  : relative azimuth (libRadtran, 0 - 180 if half_range=True)
        }
    """

    # Ensure arrays
    phi_sun_pace = np.asarray(phi_sun_pace, dtype=float)
    phi_view_pace = np.asarray(phi_view_pace, dtype=float)

    # --- 1. PACE geometry ---
    pace_saa = phi_sun_pace % 360.0           # Solar azimuth (north-ref)
    pace_vaa = phi_view_pace % 360.0          # View azimuth (north-ref)
    pace_raa = (pace_vaa - pace_saa) % 360.0  # Relative azimuth

    # --- 2. libRadtran geometry ---
    lib_saa = (pace_saa - 180.0) % 360.0      # 0 = South, clockwise
    lib_vaa = pace_vaa % 360.0                # Same as PACE view azimuth
    lib_raa = (pace_raa + 180.0) % 360.0      # Shift by 180

    # --- 3. Fold RAA to 0 - 180 range if requested ---
    if half_range:
        pace_raa = np.where(pace_raa > 180.0, 360.0 - pace_raa, pace_raa)
        lib_raa  = np.where(lib_raa  > 180.0, 360.0 - lib_raa,  lib_raa)

    # --- 4. Return all results ---
    return pace_saa, pace_vaa, pace_raa, lib_saa, lib_vaa, lib_raa

SAA_pace, VAA_pace, RAA_pace, SAA_lib, VAA_lib, RAA_lib = pace_to_libradtran(SAA, VAA, half_range=True)

Eval_input = np.column_stack([
    SZA.flatten(),
    VZA.flatten(),
    RAA_lib.flatten()
]).astype(np.float32)

Eval_output = np.column_stack([
    R_472.values.flatten(),
    R_553.values.flatten(),
    R_655.values.flatten(),
    R_862.values.flatten(),
    R_1250.values.flatten(),
    R_2130.values.flatten()
]).astype(np.float32)


output_shape = SZA.shape
n_total = np.prod(SZA.shape)

sample_arr        = np.full(output_shape, np.nan, dtype=np.float32)

CER_retr_arr      = np.full(output_shape, np.nan, dtype=np.float32)
COT_retr_arr      = np.full(output_shape, np.nan, dtype=np.float32)
AOD_retr_arr      = np.full(output_shape, np.nan, dtype=np.float32)

converged_arr     = np.full(output_shape, False, dtype=bool)

PostStd_CER_arr   = np.full(output_shape, np.nan, dtype=np.float32)
PostStd_COT_arr   = np.full(output_shape, np.nan, dtype=np.float32)
PostStd_AOD_arr   = np.full(output_shape, np.nan, dtype=np.float32)

DOFS_CER_arr      = np.full(output_shape, np.nan, dtype=np.float32)
DOFS_COT_arr      = np.full(output_shape, np.nan, dtype=np.float32)
DOFS_AOD_arr      = np.full(output_shape, np.nan, dtype=np.float32)
DOFS_total_arr    = np.full(output_shape, np.nan, dtype=np.float32)

SIC_CER_bits_arr  = np.full(output_shape, np.nan, dtype=np.float32)
SIC_COT_bits_arr  = np.full(output_shape, np.nan, dtype=np.float32)
SIC_AOD_bits_arr  = np.full(output_shape, np.nan, dtype=np.float32)
SIC_total_bits_arr= np.full(output_shape, np.nan, dtype=np.float32)

Observed_R_arr    = np.full(output_shape + (6,), np.nan, dtype=np.float32)
Predicted_R_arr   = np.full(output_shape + (6,), np.nan, dtype=np.float32)

# -------------------- Load model & scalers ONCE --------------------
model  = load_model(os.path.join(input_dir, "NN_model_cs_V_2_13_reduced_20250808_1218.keras"))
Z_mean = tf.constant(joblib.load(os.path.join(input_dir, "Z_mean_cs.pkl")), dtype=tf.float32)
Z_std  = tf.constant(joblib.load(os.path.join(input_dir, "Z_std_cs.pkl")),  dtype=tf.float32)

# -------------------- Prior (linear space) --------------------
Xa  = tf.constant([15., 10., 0.4],  dtype=tf.float32)    # CER, COT, AOD means
Sig = tf.constant([25., 50., 3], dtype=tf.float32)    # 1-sigma in linear space
Sa  = np.diag(Sig.numpy()**2).astype(np.float64)         # for diagnostics


# -------------------- Optimizer helper --------------------
def optimise_fxn(sample_number):
    # Observations and fixed geometry
    observed_R = tf.constant(Eval_output[sample_number, :], dtype=tf.float32)          # (m,)
    fixed_geom = tf.constant(Eval_input[sample_number, [0, 1, 2]], dtype=tf.float32)  # (SZA, VZA, RAA)

    # Diagonal Se (add epsilon to avoid near-zero)
    meas_err = tf.maximum(0.05 * tf.abs(observed_R), 1e-6)

    # Rodgers-style objective
    def value_and_gradients(log_state):
        with tf.GradientTape() as tape:
            tape.watch(log_state)

            x = tf.exp(log_state)  # positivity (CER, COT, AOD) in linear space

            # Forward model input: [SZA, CER, COT, AOD, VZA, RAA]
            input_raw = tf.stack([fixed_geom[0], x[0], x[1], x[2], fixed_geom[1], fixed_geom[2]])
            input_vec = (input_raw - Z_mean) / Z_std

            y_hat = model(tf.reshape(input_vec, (1, 6)))[0]  # (m,)
            r = observed_R - y_hat

            # Data term
            fit_cost = tf.reduce_sum((r / meas_err) ** 2)

            # Prior term
            prior_cost = tf.reduce_sum(((x - Xa) / Sig) ** 2)

            cost = 0.5 * fit_cost + 0.5 * prior_cost

        grad = tape.gradient(cost, log_state)  # gradient wrt log_state
        return cost, grad

    # Multiple initial guesses
    initial_guess_list = [
        np.array([6.0, 0.3, 0.5], dtype=np.float32),
        np.array([6.0, 16.0, 2.0], dtype=np.float32),
        np.array([6.0, 60.0, 4.0], dtype=np.float32),
        np.array([15.0, 0.3, 4.0], dtype=np.float32),
        np.array([15.0, 16.0, 0.5], dtype=np.float32),
        np.array([15.0, 60.0, 2.0], dtype=np.float32),
        np.array([25.0, 0.3, 2.0], dtype=np.float32),
        np.array([25.0, 16.0, 4.0], dtype=np.float32),
        np.array([25.0, 60.0, 0.5], dtype=np.float32),
    ]

    all_solutions = []

    for k, init_linear in enumerate(initial_guess_list, start=1):
        state_initial_log = tf.Variable(np.log(init_linear), dtype=tf.float32)

        lbfgs_res = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_gradients,
            initial_position=state_initial_log,
            tolerance=1e-5,
            max_iterations=800,
            max_line_search_iterations=100
        )

        final_cost = float(lbfgs_res.objective_value.numpy())
        current_state = tf.exp(lbfgs_res.position).numpy().astype(np.float64)

        all_solutions.append({
            'cost': final_cost,
            'state_var': current_state,
            'lbfgs_res': lbfgs_res,
            'init': init_linear
        })

    # -------------------- SELECTION: prefer converged=True, else best finite --------------------
    all_solutions.sort(key=lambda x: x['cost'])

    valid_converged = [
        s for s in all_solutions
        if bool(s['lbfgs_res'].converged.numpy())
        and np.all(np.isfinite(s['state_var']))
        and not np.any(np.isnan(s['state_var']))
    ]

    if len(valid_converged) > 0:
        best_solution = min(valid_converged, key=lambda s: s['cost'])
    else:
        valid_any = [
            s for s in all_solutions
            if np.all(np.isfinite(s['state_var'])) and not np.any(np.isnan(s['state_var']))
        ]
        if len(valid_any) == 0:
            print("No valid solution found. Skipping sample.")
            return
        best_solution = min(valid_any, key=lambda s: s['cost'])
    # ------------------------------------------------------------------------------------------

    best_cost       = best_solution['cost']
    best_state_var  = best_solution['state_var']   # np array (3,)
    best_lbfgs_res  = best_solution['lbfgs_res']
    best_init       = best_solution['init']

    # Predict at best state
    best_state_var_tf = tf.constant(best_state_var, dtype=tf.float32)
    input_raw_best = tf.stack([fixed_geom[0], best_state_var_tf[0],
                               best_state_var_tf[1], best_state_var_tf[2],
                               fixed_geom[1], fixed_geom[2]])
    input_vec_best = (input_raw_best - Z_mean) / Z_std
    best_predicted_R = model(tf.reshape(input_vec_best, (1, 6)))[0]

    # ================== Uncertainty diagnostics (Rodgers, 2000) ==================
    # Jacobian wrt log(x) at solution
    log_best = tf.Variable(tf.math.log(best_state_var_tf), dtype=tf.float32)
    with tf.GradientTape() as tape2:
        tape2.watch(log_best)
        x2 = tf.exp(log_best)
        input_raw2 = tf.stack([fixed_geom[0], x2[0], x2[1], x2[2], fixed_geom[1], fixed_geom[2]])
        input_vec2 = (input_raw2 - Z_mean) / Z_std
        pred2 = model(tf.reshape(input_vec2, (1, 6)))[0]  # (m,)

    # K_log shape: (m, 3) = dR/dlog(x)
    K_log = tape2.jacobian(pred2, log_best).numpy()  # (m, 3)

    # Convert to linear-space Jacobian: K = dR/dx = K_log * (1/x)
    K = K_log / best_state_var[np.newaxis, :]  # (m, 3)

    sigma = meas_err.numpy().astype(np.float64)

    # Base inverses (Rodgers)
    Se_inv = np.diag(1.0 / (sigma ** 2))
    Sa_inv = np.linalg.inv(Sa)

    # Posterior covariance with effective inverses
    KT_Se_inv = K.T @ Se_inv
    Sx = np.linalg.inv(KT_Se_inv @ K + Sa_inv)

    # Averaging kernel with effective inverses
    A = Sx @ KT_Se_inv @ K

    # DOFS per parameter and total
    dofs_diag  = np.diag(A).astype(np.float64)     # [CER, COT, AOD]
    dofs_total = float(np.trace(A))

    # Posterior 1-sigma uncertainties
    post_stds = np.sqrt(np.diag(Sx)).astype(np.float64)  # [CER, COT, AOD]

    # for SIC ----
    sign_a, logdet_a = np.linalg.slogdet(Sa)
    sign_x, logdet_x = np.linalg.slogdet(Sx)
    SIC_total_bits = 0.5 * (logdet_a - logdet_x) / np.log(2.0)

    prior_vars     = np.diag(Sa)
    posterior_vars = np.diag(Sx)
    SIC_per_bits   = 0.5 * np.log2(prior_vars / posterior_vars)
    # ---------------------------------------------------------------------------------

    # Save into output arrays at 2D index corresponding to sample_number
    idx = np.unravel_index(sample_number, output_shape)

    sample_arr[idx]         = int(sample_number)

    CER_retr_arr[idx]       = float(best_state_var[0])
    COT_retr_arr[idx]       = float(best_state_var[1])
    AOD_retr_arr[idx]       = float(best_state_var[2])

    converged_arr[idx]      = bool(best_lbfgs_res.converged.numpy())

    PostStd_CER_arr[idx]    = float(post_stds[0])
    PostStd_COT_arr[idx]    = float(post_stds[1])
    PostStd_AOD_arr[idx]    = float(post_stds[2])

    DOFS_CER_arr[idx]       = float(dofs_diag[0])
    DOFS_COT_arr[idx]       = float(dofs_diag[1])
    DOFS_AOD_arr[idx]       = float(dofs_diag[2])
    DOFS_total_arr[idx]     = float(dofs_total)

    SIC_CER_bits_arr[idx]   = float(SIC_per_bits[0])
    SIC_COT_bits_arr[idx]   = float(SIC_per_bits[1])
    SIC_AOD_bits_arr[idx]   = float(SIC_per_bits[2])
    SIC_total_bits_arr[idx] = float(SIC_total_bits)

    Observed_R_arr[idx]     = Eval_output[sample_number, :].astype(np.float32)
    Predicted_R_arr[idx]    = best_predicted_R.numpy().astype(np.float32)

# -------------------- Batch loop --------------------
for sample_number in range(n_total):
    optimise_fxn(sample_number)
    
t1 = time.perf_counter()
runtime_s = t1 - t0
print(" ")
print("######################################################")
print(f"Total runtime (s): {runtime_s:.3f}")
print(f"Avg time per sample (s): {runtime_s / n_total:.6f}")

# -------------------- Save all outputs to NetCDF --------------------
retrieval_ds = xr.Dataset(
    data_vars={
        "sample":         (("y", "x"), sample_arr),

        "CER_retr":       (("y", "x"), CER_retr_arr),
        "COT_retr":       (("y", "x"), COT_retr_arr),
        "AOD_retr":       (("y", "x"), AOD_retr_arr),

        "converged":      (("y", "x"), converged_arr),

        "PostStd_CER":    (("y", "x"), PostStd_CER_arr),
        "PostStd_COT":    (("y", "x"), PostStd_COT_arr),
        "PostStd_AOD":    (("y", "x"), PostStd_AOD_arr),

        "DOFS_CER":       (("y", "x"), DOFS_CER_arr),
        "DOFS_COT":       (("y", "x"), DOFS_COT_arr),
        "DOFS_AOD":       (("y", "x"), DOFS_AOD_arr),
        "DOFS_total":     (("y", "x"), DOFS_total_arr),

        "SIC_CER_bits":   (("y", "x"), SIC_CER_bits_arr),
        "SIC_COT_bits":   (("y", "x"), SIC_COT_bits_arr),
        "SIC_AOD_bits":   (("y", "x"), SIC_AOD_bits_arr),
        "SIC_total_bits": (("y", "x"), SIC_total_bits_arr),

        "Observed_R":     (("y", "x", "wl"), Observed_R_arr),
        "Predicted_R":    (("y", "x", "wl"), Predicted_R_arr),
    },
    coords={
        "y": np.arange(output_shape[0]),
        "x": np.arange(output_shape[1]),
        "wl": np.arange(6),
    }
)

retrieval_ds.to_netcdf(nc_path)
print(f"\nSaved NetCDF file: {nc_path}")