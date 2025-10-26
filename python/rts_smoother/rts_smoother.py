from __future__ import annotations

import numpy as np
import pandas as pd


try:
    from . import _core as _ks
except Exception as e:  # pragma: no cover
    _IMPORT_ERR = e
    _ks = None

def _ensure_module():
    if _ks is None:
        raise RuntimeError("rts_smoother extension not built. Did you `pip install -e .`?\n" + f"Original import error: {_IMPORT_ERR}")

def smooth(df, Q: float, R: float):
    """Run RTS Kalman smoothing column-wise on a T×N DataFrame-like.

    Parameters
    ----------
    df : DataFrame-like
        Observations. Index is timestamps, columns are assets. NaN allowed.
    Q, R : float
        Global process and observation noise variances (must be > 0).

    Returns
    -------
    smoothed : DataFrame-like
        Smoothed latent states with same shape as input.
    """

    _ensure_module()

    if not (Q > 0 and R > 0):
        raise ValueError("Q and R must be positive")

    values = None
    if isinstance(df, pd.DataFrame):
        values = np.asarray(df.values, dtype=np.float64, order="C")
    else:
        values = np.asarray(df, dtype=np.float64, order="C")

    if (values is None) or (values.ndim != 2):
        raise ValueError("Input must be 2D with shape (T, N)")
        
    smoothed = _ks.rts_smoother(values, Q, R)
    
    if isinstance(df, pd.DataFrame):        
        return pd.DataFrame(smoothed, index=df.index, columns=df.columns)
    else:
        return smoothed