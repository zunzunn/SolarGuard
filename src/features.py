"""Shared past-only flux feature construction for training and live inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "log_flux_mean", "log_flux_median", "log_flux_peak",
    "log_flux_mean_6h", "log_flux_std_6h", "log_flux_max_6h", "log_flux_delta_6h",
    "log_flux_mean_24h", "log_flux_std_24h", "log_flux_max_24h", "log_flux_delta_24h",
    "log_flux_delta_1h", "hour_sin", "hour_cos",
]


def add_flux_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return input rows with model features, using data only through sample time."""
    d = frame.copy()
    t = pd.DatetimeIndex(pd.to_datetime(d.sample_time_utc, utc=True))
    d.index = t
    d["log_flux_mean"] = d.log10_flux_mean
    d["log_flux_median"] = np.log10(d.xrsb_flux_median_w_m2.clip(lower=1e-12))
    d["log_flux_peak"] = np.log10(d.xrsb_flux_max_w_m2.clip(lower=1e-12))
    signal = d.log_flux_mean
    for hours in (6, 24):
        roll = signal.rolling(f"{hours}h", min_periods=max(3, hours // 2), closed="right")
        d[f"log_flux_mean_{hours}h"] = roll.mean()
        d[f"log_flux_std_{hours}h"] = roll.std()
        d[f"log_flux_max_{hours}h"] = roll.max()
        d[f"log_flux_delta_{hours}h"] = signal - signal.shift(freq=pd.Timedelta(hours=hours))
    d["log_flux_delta_1h"] = signal - signal.shift(freq=pd.Timedelta(hours=1))
    d["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
    return d.replace([np.inf, -np.inf], np.nan)
