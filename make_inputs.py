"""
Build real PACE OCI input file for optimized SoC retrieval.

This script reads a real PACE OCI L1B file, extracts a lat/lon patch,
selects six reflectance bands, extracts real SZA/VZA/SAA/VAA geometry,
converts PACE azimuths to libRadtran-style RAA, and saves:

    Eval_input  = [SZA, VZA, RAA_lib]
    Eval_output = [R472, R553, R655, R862, R1250, R2130]
    shape       = [ny, nx]

Usage:
    python make_inputs.py <repo_dir> <out_file>

Example:
    python make_inputs.py . eval_inputs_real.npz
"""

import os
import sys
import numpy as np
import xarray as xr


def pace_to_libradtran(phi_sun_pace, phi_view_pace, half_range=True):
    """
    Convert PACE OCI azimuth geometry to libRadtran convention.

    PACE azimuth:
        0 = North, clockwise

    Returns:
        pace_saa, pace_vaa, pace_raa,
        lib_saa, lib_vaa, lib_raa
    """

    phi_sun_pace = np.asarray(phi_sun_pace, dtype=float)
    phi_view_pace = np.asarray(phi_view_pace, dtype=float)

    # PACE geometry
    pace_saa = phi_sun_pace % 360.0
    pace_vaa = phi_view_pace % 360.0
    pace_raa = (pace_vaa - pace_saa) % 360.0

    # libRadtran-style geometry
    lib_saa = (pace_saa - 180.0) % 360.0
    lib_vaa = pace_vaa % 360.0
    lib_raa = (pace_raa + 180.0) % 360.0

    if half_range:
        pace_raa = np.where(pace_raa > 180.0, 360.0 - pace_raa, pace_raa)
        lib_raa  = np.where(lib_raa  > 180.0, 360.0 - lib_raa,  lib_raa)

    return pace_saa, pace_vaa, pace_raa, lib_saa, lib_vaa, lib_raa


# ---------------------------------------------------------------------
# Command-line inputs
# ---------------------------------------------------------------------
repo = sys.argv[1] if len(sys.argv) > 1 else "."
out  = sys.argv[2] if len(sys.argv) > 2 else "eval_inputs_real.npz"

l1b_file = os.path.join(repo, "PACE_OCI.20250822T123146.L1B.V3.nc")

# ---------------------------------------------------------------------
# Open PACE L1B groups
# ---------------------------------------------------------------------
obs_data = xr.open_dataset(l1b_file, group="observation_data", engine="netcdf4")
geo_data = xr.open_dataset(l1b_file, group="geolocation_data", engine="netcdf4")
wavelength_data = xr.open_dataset(l1b_file, group="sensor_band_parameters", engine="netcdf4")

# ---------------------------------------------------------------------
# Define target region
# ---------------------------------------------------------------------
lat_min, lat_max = -10.0, -8.0
lon_min, lon_max = 9.0, 11.0

lat2d = geo_data["latitude"]
lon2d = geo_data["longitude"]

mask = (
    (lat2d >= lat_min) & (lat2d <= lat_max) &
    (lon2d >= lon_min) & (lon2d <= lon_max)
)

rows = np.any(mask.values, axis=1)
cols = np.any(mask.values, axis=0)

if not rows.any() or not cols.any():
    raise ValueError(
        f"No pixels found inside lat/lon box: "
        f"lat {lat_min} to {lat_max}, lon {lon_min} to {lon_max}"
    )

i0, i1 = np.where(rows)[0][[0, -1]]
j0, j1 = np.where(cols)[0][[0, -1]]

i1 += 1
j1 += 1

print(f"Selected patch rows {i0}:{i1}, cols {j0}:{j1}")

# ---------------------------------------------------------------------
# Extract six reflectance bands
# ---------------------------------------------------------------------
# Make sure these band indices match your intended wavelengths.
R_472  = obs_data["rhot_blue"].isel(blue_bands=65)[i0:i1, j0:j1]
R_553  = obs_data["rhot_blue"].isel(blue_bands=97)[i0:i1, j0:j1]
R_655  = obs_data["rhot_red"].isel(red_bands=28)[i0:i1, j0:j1]
R_862  = obs_data["rhot_red"].isel(red_bands=149)[i0:i1, j0:j1]
R_1250 = obs_data["rhot_SWIR"].isel(SWIR_bands=2)[i0:i1, j0:j1]
R_2130 = obs_data["rhot_SWIR"].isel(SWIR_bands=7)[i0:i1, j0:j1]

# ---------------------------------------------------------------------
# Extract geometry
# ---------------------------------------------------------------------
SZA = geo_data["solar_zenith"][i0:i1, j0:j1].values
VZA = geo_data["sensor_zenith"][i0:i1, j0:j1].values
VAA = geo_data["sensor_azimuth"][i0:i1, j0:j1].values
SAA = geo_data["solar_azimuth"][i0:i1, j0:j1].values

SAA_pace, VAA_pace, RAA_pace, SAA_lib, VAA_lib, RAA_lib = pace_to_libradtran(
    SAA, VAA, half_range=True
)

# ---------------------------------------------------------------------
# Build retrieval input arrays
# ---------------------------------------------------------------------
Eval_input = np.column_stack([
    SZA.ravel(),
    VZA.ravel(),
    RAA_lib.ravel()
]).astype(np.float32)

Eval_output = np.column_stack([
    R_472.values.ravel(),
    R_553.values.ravel(),
    R_655.values.ravel(),
    R_862.values.ravel(),
    R_1250.values.ravel(),
    R_2130.values.ravel()
]).astype(np.float32)

ny, nx = SZA.shape

# ---------------------------------------------------------------------
# Optional valid-pixel check
# ---------------------------------------------------------------------
finite_geom = np.isfinite(Eval_input).all(axis=1)
finite_refl = np.isfinite(Eval_output).all(axis=1)
valid = finite_geom & finite_refl

print(f"Patch shape: {ny} x {nx} = {ny * nx} pixels")
print(f"Finite geometry pixels: {finite_geom.sum()}")
print(f"Finite reflectance pixels: {finite_refl.sum()}")
print(f"Finite geometry + reflectance pixels: {valid.sum()}")

# ---------------------------------------------------------------------
# Save file expected by run_optimized.py
# ---------------------------------------------------------------------
np.savez(
    out,
    Eval_input=Eval_input,
    Eval_output=Eval_output,
    shape=np.array([ny, nx]),
    valid=valid,
    lat=lat2d[i0:i1, j0:j1].values.astype(np.float32),
    lon=lon2d[i0:i1, j0:j1].values.astype(np.float32),
    SZA=SZA.astype(np.float32),
    VZA=VZA.astype(np.float32),
    SAA=SAA.astype(np.float32),
    VAA=VAA.astype(np.float32),
    RAA_lib=RAA_lib.astype(np.float32),
)

print(f"Saved {out}")