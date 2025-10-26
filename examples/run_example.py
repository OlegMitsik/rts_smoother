import numpy as np
import pandas as pd

from rts_smoother import smooth


# Params
T, N = 500, 99
Q, R = 1e-4, 1e-3
min_val = 0.05

# Synthetic data: random-walk latent state + observation noise for assets
eps = np.random.normal(0, np.sqrt(Q), size=(T-1, N))
x = np.vstack([np.zeros((1, N)), np.cumsum(eps, axis=0)])
y = x + np.random.normal(0, np.sqrt(R), size=(T, N))

# Introduce some missing data
y[np.random.random((T, N)) < min_val] = np.nan

# Put proper dates
dates = pd.date_range("2020-01-01", periods=T, freq="B")
df = pd.DataFrame(y, index=dates, columns=[f"Asset{n+1}" for n in range(N)])

# Smooth with chosen Q (process var) and R (observation var)
smoothed = smooth(df, Q, R)
print(smoothed)
