"""Live NOAA SWPC GOES Primary XRS feed ingestion, coverage checking, and aggregation."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

SWPC_PRIMARY_XRAYS_7DAY_URL = os.getenv(
    "SOLARGUARD_SWPC_URL",
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
)
STALENESS_THRESHOLD_MINUTES = int(os.getenv("SOLARGUARD_STALENESS_MINUTES", "120"))
MIN_VALID_MINUTES_PER_HOUR = 45
MIN_CONSECUTIVE_HOURS = 25
MAX_CONSECUTIVE_HOURS = 72


def fetch_live_records(url: str = SWPC_PRIMARY_XRAYS_7DAY_URL, timeout: int = 15) -> list[dict[str, Any]]:
    """Fetch raw records from NOAA SWPC primary XRS endpoint."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list from {url}, got {type(payload).__name__}")
    return payload


def process_live_records(
    records: list[dict[str, Any]],
    now_utc: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process raw SWPC records: filter 0.1-0.8nm, check staleness, aggregate completed UTC hours."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    # Filter to 0.1-0.8 nm (XRS-B long channel)
    xrsb_records = [r for r in records if r.get("energy") == "0.1-0.8nm"]
    if not xrsb_records:
        raise ValueError("No 0.1-0.8nm records found in SWPC payload.")

    df = pd.DataFrame(xrsb_records)
    df["time"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df["xrsb_flux_w_m2"] = pd.to_numeric(df["flux"], errors="coerce")
    df = df.dropna(subset=["time", "xrsb_flux_w_m2"])
    df = df[df["xrsb_flux_w_m2"] > 0]
    if df.empty:
        raise ValueError("No valid positive XRS-B flux observations found in SWPC payload.")

    df = df.sort_values("time")

    # Satellite identification
    satellites = df["satellite"].dropna().unique().tolist()
    sat_label = f"GOES-{satellites[0]}" if satellites else "GOES-Primary"

    latest_raw_time = df["time"].max().to_pydatetime()
    staleness_seconds = max(0.0, (now_utc - latest_raw_time).total_seconds())
    staleness_minutes = staleness_seconds / 60.0
    is_stale = staleness_minutes > STALENESS_THRESHOLD_MINUTES

    # Hourly aggregation matching build_dataset.py:
    # minute observations are resampled into 1-hour bins [H, H+1), then stamped at H+1.
    minute = df.set_index("time")
    minute["log10_xrsb_flux"] = np.log10(minute["xrsb_flux_w_m2"].clip(lower=1e-12))

    hourly = minute.resample("1h").agg(
        xrsb_flux_mean_w_m2=("xrsb_flux_w_m2", "mean"),
        xrsb_flux_max_w_m2=("xrsb_flux_w_m2", "max"),
        xrsb_flux_median_w_m2=("xrsb_flux_w_m2", "median"),
        log10_flux_mean=("log10_xrsb_flux", "mean"),
        valid_minutes=("xrsb_flux_w_m2", "count"),
    )
    # Stamped at the end of the completed hour
    hourly.index = hourly.index + pd.Timedelta(hours=1)
    hourly.index.name = "sample_time_utc"

    # Retain ONLY completed UTC hours strictly <= now_utc
    hourly = hourly[hourly.index <= pd.Timestamp(now_utc)]

    # Coverage rule: at least 45 valid minutes out of 60
    hourly = hourly[hourly.valid_minutes >= MIN_VALID_MINUTES_PER_HOUR].copy()
    hourly = hourly.reset_index()

    metadata = {
        "data_source": "NOAA_SWPC_OPERATIONAL_GOES_PRIMARY",
        "energy_band": "0.1-0.8nm (XRS-B)",
        "source_satellite": sat_label,
        "satellites": satellites,
        "raw_record_count": len(xrsb_records),
        "valid_minute_count": len(df),
        "latest_observation_utc": latest_raw_time.isoformat().replace("+00:00", "Z"),
        "now_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "staleness_minutes": round(staleness_minutes, 1),
        "is_stale": is_stale,
        "staleness_threshold_minutes": STALENESS_THRESHOLD_MINUTES,
        "completed_hours_retained": len(hourly),
        "latest_completed_hour_utc": (
            hourly["sample_time_utc"].max().isoformat().replace("+00:00", "Z")
            if not hourly.empty
            else None
        ),
        "distribution_shift_note": (
            "Operational SWPC real-time primary data versus NCEI science-quality historical archive. "
            "Real-time operational telemetry is evaluated against the validation-selected model baseline; "
            "distribution shift or unflagged telemetry anomalies may affect calibration."
        ),
    }

    return hourly, metadata


def get_consecutive_hourly_samples(
    hourly: pd.DataFrame,
    min_hours: int = MIN_CONSECUTIVE_HOURS,
    max_hours: int = MAX_CONSECUTIVE_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract the trailing contiguous hourly samples without gaps."""
    if hourly.empty:
        raise ValueError("Hourly dataset is empty; cannot extract consecutive samples.")

    sorted_df = hourly.sort_values("sample_time_utc").reset_index(drop=True)
    needed = ["sample_time_utc", "log10_flux_mean", "xrsb_flux_median_w_m2", "xrsb_flux_max_w_m2"]

    # Traverse backwards from the latest completed hour to find consecutive hourly steps
    consecutive_indices: list[int] = []
    prev_time: pd.Timestamp | None = None

    for idx in reversed(sorted_df.index):
        stamp = pd.Timestamp(sorted_df.loc[idx, "sample_time_utc"])
        if prev_time is not None and prev_time - stamp != pd.Timedelta(hours=1):
            break
        consecutive_indices.append(idx)
        prev_time = stamp
        if len(consecutive_indices) >= max_hours:
            break

    consecutive_indices.reverse()
    consecutive_df = sorted_df.loc[consecutive_indices, needed].copy()

    check_meta = {
        "consecutive_hours_found": len(consecutive_df),
        "min_hours_required": min_hours,
        "max_hours_requested": max_hours,
        "has_sufficient_history": len(consecutive_df) >= min_hours,
        "earliest_sample_utc": (
            consecutive_df["sample_time_utc"].iloc[0].isoformat().replace("+00:00", "Z")
            if not consecutive_df.empty
            else None
        ),
        "latest_sample_utc": (
            consecutive_df["sample_time_utc"].iloc[-1].isoformat().replace("+00:00", "Z")
            if not consecutive_df.empty
            else None
        ),
    }

    if len(consecutive_df) < min_hours:
        raise ValueError(
            f"Insufficient contiguous hourly history: found {len(consecutive_df)} "
            f"consecutive hours, require at least {min_hours}."
        )

    return consecutive_df, check_meta


class LiveFeedManager:
    """Thread-safe TTL-cached manager for NOAA SWPC live telemetry."""

    def __init__(self, ttl_seconds: int = 60, url: str = SWPC_PRIMARY_XRAYS_7DAY_URL) -> None:
        self.ttl_seconds = ttl_seconds
        self.url = url
        self._cached_hourly: pd.DataFrame | None = None
        self._cached_metadata: dict[str, Any] | None = None
        self._cached_at: float = 0.0

    def get_data(self, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
        now = time.time()
        if (
            not force_refresh
            and self._cached_hourly is not None
            and self._cached_metadata is not None
            and (now - self._cached_at) < self.ttl_seconds
        ):
            return self._cached_hourly.copy(), dict(self._cached_metadata)

        records = fetch_live_records(self.url)
        hourly, metadata = process_live_records(records)
        metadata["cached_at_unix"] = now
        metadata["cache_ttl_seconds"] = self.ttl_seconds

        self._cached_hourly = hourly
        self._cached_metadata = metadata
        self._cached_at = now

        return hourly.copy(), dict(metadata)


# Global default instance
live_feed_manager = LiveFeedManager()
