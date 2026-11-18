"""Unit tests for live NOAA SWPC feed processing, aggregation, and coverage rules."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from src.live_feed import (
    process_live_records,
    get_consecutive_hourly_samples,
    MIN_VALID_MINUTES_PER_HOUR,
    MIN_CONSECUTIVE_HOURS,
)


def make_mock_record(time_str: str, flux: float = 1e-6, energy: str = "0.1-0.8nm", sat: int = 18) -> dict:
    return {
        "time_tag": time_str,
        "satellite": sat,
        "flux": flux,
        "observed_flux": flux * 1.05,
        "electron_correction": 1e-8,
        "electron_contaminaton": False,
        "energy": energy,
    }


def test_process_live_records_filters_energy_and_invalid():
    now = datetime(2026, 11, 18, 14, 0, tzinfo=timezone.utc)
    records = [
        make_mock_record("2026-11-18T12:05:00Z", 1e-6, energy="0.05-0.4nm"),  # Should be filtered
        make_mock_record("2026-11-18T12:06:00Z", -1.0, energy="0.1-0.8nm"),   # Negative flux filtered
        make_mock_record("2026-11-18T12:07:00Z", 0.0, energy="0.1-0.8nm"),    # Zero flux filtered
    ]
    with pytest.raises(ValueError, match="No valid positive XRS-B flux observations"):
        process_live_records(records, now_utc=now)


def test_process_live_records_coverage_rule():
    """Verify that completed hours with >= 45 valid minutes are kept, < 45 are omitted."""
    now = datetime(2026, 11, 18, 14, 0, tzinfo=timezone.utc)
    records = []

    # Hour 1: 12:00 to 12:59 -> 50 valid minutes (>= 45 -> should be retained as sample_time 13:00)
    for m in range(50):
        records.append(make_mock_record(f"2026-11-18T12:{m:02d}:00Z", 1e-6))

    # Hour 2: 13:00 to 13:59 -> 30 valid minutes (< 45 -> should be dropped)
    for m in range(30):
        records.append(make_mock_record(f"2026-11-18T13:{m:02d}:00Z", 2e-6))

    hourly, meta = process_live_records(records, now_utc=now)
    assert len(hourly) == 1
    assert hourly["valid_minutes"].iloc[0] == 50
    assert hourly["sample_time_utc"].iloc[0] == pd.Timestamp("2026-11-18 13:00:00+00:00")
    assert meta["source_satellite"] == "GOES-18"
    assert not meta["is_stale"]


def test_process_live_records_drops_incomplete_future_hours():
    """In-progress or future hours must not be retained."""
    now = datetime(2026, 11, 18, 12, 30, tzinfo=timezone.utc)
    records = []
    # Hour 11:00-11:59 completed (sample_time 12:00 <= now)
    for m in range(60):
        records.append(make_mock_record(f"2026-11-18T11:{m:02d}:00Z", 1e-6))
    # Hour 12:00-12:59 in-progress (sample_time 13:00 > now)
    for m in range(25):
        records.append(make_mock_record(f"2026-11-18T12:{m:02d}:00Z", 1e-6))

    hourly, _ = process_live_records(records, now_utc=now)
    assert len(hourly) == 1
    assert hourly["sample_time_utc"].iloc[0] == pd.Timestamp("2026-11-18 12:00:00+00:00")


def test_process_live_records_staleness_flag():
    now = datetime(2026, 11, 18, 18, 0, tzinfo=timezone.utc)
    records = [make_mock_record(f"2026-11-18T12:{m:02d}:00Z", 1e-6) for m in range(50)]
    hourly, meta = process_live_records(records, now_utc=now)
    assert meta["is_stale"] is True
    assert meta["staleness_minutes"] > 120


def test_get_consecutive_hourly_samples_success():
    base = pd.Timestamp("2026-11-18 00:00:00", tz="UTC")
    rows = []
    for h in range(30):
        rows.append({
            "sample_time_utc": base + pd.Timedelta(hours=h),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
            "xrsb_flux_mean_w_m2": 3.2e-7,
            "valid_minutes": 60,
        })
    df = pd.DataFrame(rows)
    consec, meta = get_consecutive_hourly_samples(df, min_hours=25, max_hours=72)
    assert meta["consecutive_hours_found"] == 30
    assert meta["has_sufficient_history"] is True
    assert len(consec) == 30


def test_get_consecutive_hourly_samples_detects_gap():
    base = pd.Timestamp("2026-11-18 00:00:00", tz="UTC")
    rows = []
    # 20 hours, then a 3-hour gap, then 10 hours
    for h in range(20):
        rows.append({
            "sample_time_utc": base + pd.Timedelta(hours=h),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        })
    for h in range(23, 33):  # 10 hours after gap
        rows.append({
            "sample_time_utc": base + pd.Timedelta(hours=h),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        })
    df = pd.DataFrame(rows)
    # The trailing contiguous chunk has only 10 hours, which is < 25
    with pytest.raises(ValueError, match="Insufficient contiguous hourly history"):
        get_consecutive_hourly_samples(df, min_hours=25)
