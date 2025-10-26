# RTS Kalman Smoother (C++ × Python)

The package provides a high-performance **Rauch–Tung–Striebel** Kalman smoother for the **local level** model.
This model is a canonical choice for asset prices and similar data (random walk latent state).  
It smooths each asset (column) independently with **global** process/observation variances `(Q, R)` shared across assets.

### Features
- Handles `NaN` observations (prediction-only step when missing)
- Returns smoothed states as the same object type
- Diffuse initialization

### Install (from PyPI)
```bash
pip install rts_smoother
```

### Install (from source)
```bash
pip install -U pip build twine
pip install -e .
```

### Usage
```
from rts_smoother import smooth

# Provide Q and R manually
df_smoothed = smooth(df, Q=1e-4, R=1e-3)
```

### Model
- State: `x_t = x_{t-1} + w_t`, `w_t ~ N(0, Q)`
- Obs: `y_t = x_t + v_t`, `v_t ~ N(0, R)`

### Notes
- Built with **scikit-build-core** and **pybind11**.
- The compiled extension lives at `rts_smoother/_core.*` inside the wheel.
