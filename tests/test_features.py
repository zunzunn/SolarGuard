"""Unit tests for trailing flux feature extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, add_flux_features


def test_add_flux_features_columns_and_past_only():
    base = pd.Timestamp("2026-11-18 00:00:00", tz="UTC")
    rows = []
    for h in range(30):
        t = base + pd.Timedelta(hours=h)
        flux = 1e-6 * (1.0 + 0.1 * (h % 5))
        rows.append({
            "sample_time_utc": t,
            "log10_flux_mean": float(np.log10(flux)),
            "xrsb_flux_median_w_m2": flux,
            "xrsb_flux_max_w_m2": flux * 1.2,
        })
    df = pd.DataFrame(rows)
    engineered = add_flux_features(df)

    # Check all expected features exist
    for col in FEATURES:
        assert col in engineered.columns, f"Missing feature {col}"

    # Verify that the latest row (index 29) has no NaNs because we have >= 24h history
    latest = engineered.iloc[-1][FEATURES]
    assert not latest.isna().any(), f"NaNs in latest feature row: {latest[latest.isna()]}"

    # Verify lookahead prevention: mutating future hypothetical rows does not affect row 25
    df_copy = df.copy()
    df_copy.loc[26:, "log10_flux_mean"] = 999.0
    engineered_mutated = add_flux_features(df_copy)
    np.testing.assert_allclose(
        engineered.loc[df.loc[25, "sample_time_utc"], FEATURES].to_numpy(dtype=float),
        engineered_mutated.loc[df.loc[25, "sample_time_utc"], FEATURES].to_numpy(dtype=float),
    )
