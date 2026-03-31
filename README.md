# PACE OCI Smoke over Cloud (SoC) Retrieval Algorithm

## Overview
This repository contains a machine-learning-based retrieval algorithm to estimate:

- Cloud Optical Thickness (COT)
- Cloud Effective Radius (CER)
- Above-Cloud Aerosol Optical Depth (AOD)

using spectral reflectance from PACE OCI.

---

## Methodology
- Radiative Transfer Model (libRadtran) used to generate LUT
- Neural Network trained to emulate reflectance
- Optimal Estimation (Rodgers, 2000) used for retrieval

---

## Inputs
- Spectral reflectance from PACE OCI:
  - Visible/NIR/SWIR wavelengths:
    - 0.47 µm
    - 0.55 µm
    - 0.66 µm
    - 0.86 µm
    - 1.25 µm
    - 2.13 µm

- Viewing geometry:
  - Solar Zenith Angle (SZA)
  - Viewing Zenith Angle (VZA)
  - Relative Azimuth Angle (RAA)

---

## Outputs
- Retrieved CER, COT, AOD
- Posterior uncertainty
- Degrees of Freedom for Signal (DOFS)
- Shannon Information Content (SIC)

---

## Files
- `PACE_SoC_Retrieval.py` → main retrieval code
- `Plotting_Retrievals.ipynb` → visualization

---

## Software & Implementation Environment

# Programing Language:
- Python: 3.12.12 | packaged by conda-forge | (main, Oct 22, 2025, 23:25:55) [GCC 14.3.0]

# Core Libraries (Operationally Required):
- tensorflow: 2.18.0
- tensorflow_probability: 0.25.0
- tf.keras: 3.12.0
- scikit-learn: 1.8.0
- joblib: 1.5.3
- numpy: 2.4.1
- scipy: 1.17.0
- pandas: 2.3.3
- xarray: 2025.12.0
- netCDF4: 1.7.4

---

## Notes
Large data files and model outputs are excluded using `.gitignore`.

---

## Author
Roshan Mishra  
PhD Candidate, Atmospheric Physics, UMBC
