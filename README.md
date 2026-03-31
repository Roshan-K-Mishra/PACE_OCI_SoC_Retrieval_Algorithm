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

## Notes
Large data files and model outputs are excluded using `.gitignore`.

---

## Author
Roshan Mishra  
PhD Candidate, Atmospheric Physics, UMBC
