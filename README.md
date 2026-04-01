# PACE OCI Smoke over Cloud (SoC) Retrieval Algorithm. The algorithm is for smoke-over-cloud conditions.

## Overview
This repository implements a machine-learning-based optimal estimation retrieval algorithm to jointly estimate:

- Cloud Optical Thickness (COT)
- Cloud Effective Radius (CER)
- Above-Cloud Aerosol Optical Depth (AOD)

from PACE OCI spectral reflectance observations. 


The algorithm combines:
- Radiative Transfer modeling (libRadtran LUT)
- Neural Network forward model (emulator)
- Bayesian Optimal Estimation (Rodgers, 2000)

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


## Methodology

### 1. Forward Model (Neural Network)
A trained neural network emulates reflectance:
Reflectance = f(CER, COT, AOD, SZA, VZA, RAA})

- Input: 6 variables (state + geometry)
- Output: Reflectance at 6 wavelengths

---

### 2. Retrieval Framework (Optimal Estimation)

State vector:
- CER (µm)
- COT
- AOD


Cost function:
- Measurement term (reflectance residual)
- Prior constraint

Optimization:
- L-BFGS (TensorFlow Probability)
- Multiple initial guesses for robustness

---

### 3. Uncertainty Quantification

The algorithm computes:

- Posterior covariance \( S_x \)
- Averaging kernel \( A \)
- Degrees of Freedom for Signal (DOFS)
- Shannon Information Content (SIC)

This follows Rodgers (2000):

\[
S_x = (K^T S_e^{-1} K + S_a^{-1})^{-1}
\]

---


## Files
- `PACE_SoC_Retrieval.py` → main retrieval code
- `Plotting_Retrievals.ipynb` → sample visualization

---

## Software & Implementation Environment

### Programming Language:
- Python: 3.12.12 | packaged by conda-forge | (main, Oct 22, 2025, 23:25:55) [GCC 14.3.0]

### Core Libraries (Operationally Required):
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
