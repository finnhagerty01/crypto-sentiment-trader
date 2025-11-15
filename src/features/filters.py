"""Filtering utilities for market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_filter_series(
    series: pd.Series,
    process_variance: float = 1e-5,
    measurement_variance: float = 0.05,
) -> pd.Series:
    """Apply a simple scalar Kalman filter to smooth a time-series.

    Parameters
    ----------
    series:
        Input series (e.g., close prices).
    process_variance:
        Variance of the process noise. Larger values react faster to new data.
    measurement_variance:
        Variance of the measurement noise.
    """

    if series.empty:
        return series

    values = series.astype(float).to_numpy(copy=True)
    n = len(values)
    estimates = np.zeros(n)
    error_cov = np.zeros(n)

    estimates[0] = values[0]
    error_cov[0] = 1.0

    for k in range(1, n):
        # Predict
        pred_state = estimates[k - 1]
        pred_cov = error_cov[k - 1] + process_variance

        # Update
        kalman_gain = pred_cov / (pred_cov + measurement_variance)
        estimates[k] = pred_state + kalman_gain * (values[k] - pred_state)
        error_cov[k] = (1 - kalman_gain) * pred_cov

    return pd.Series(estimates, index=series.index, name=series.name)


__all__ = ["kalman_filter_series"]
