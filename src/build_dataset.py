#!/usr/bin/env python3
"""Build hourly GOES XRS features and forward M+ flare labels from NOAA NCEI.

Downloads the mission-length science-quality flare report and available yearly
science-quality XRS 1-minute NetCDF files. Raw inputs are retained for
reproducibility; outputs are UTC, hourly, and sorted chronologically.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
FLARE_DIR = BASE + "multi/l2/data/xrsf-l2-flrpt_science/csv/"
SATELLITES = range(13, 20)
LEGACY_XRS = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/"
TIME0 = pd.Timestamp("2016-11-18", tz="UTC")
TIME1 = pd.Timestamp("2026-11-18", tz="UTC")  # exclusive; end of last complete UTC day


def get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, timeout=(20, 180), stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def directory_files(url: str) -> list[str]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [urljoin(url, a.get("href", "")) for a in soup.select("a[href]")]


def fetch_flare_report(raw: Path) -> Path:
    names = directory_files(FLARE_DIR)
    candidates = [u for u in names if re.search(r"sci_xrsf-l2-flrpt_geo_s\d+_e\d+_v[\d-]+\.csv$", u)]
    if not candidates:
        raise RuntimeError(f"No mission-length flare CSV found at {FLARE_DIR}")
    # Archive exposes one current mission-length aggregate; choose latest version.
    url = sorted(candidates)[-1]
    dest = raw / Path(url).name
    get(url, dest)
    return dest


def fetch_xrs(raw: Path, years: range) -> list[Path]:
    """Fetch science-quality 1-min archives whose names overlap requested years.

    NCEI may publish a mission-span file at the product root and nested
    year/month/day directories. Prefer the root mission-span file when present;
    otherwise use root yearly files. This avoids downloading redundant daily
    files from the monthly directory hierarchy.
    """
    files: list[Path] = []
    requested = set(years)
    for sat in SATELLITES:
        if sat <= 15:
            satdir = f"{LEGACY_XRS}goes{sat}/xrsf-l2-avg1m_science/"
        else:
            satdir = f"{BASE}goes{sat}/l2/data/xrsf-l2-avg1m_science/"
        try:
            root_files = directory_files(satdir)
        except requests.RequestException:
            continue
        candidates = [u for u in root_files if u.endswith(".nc") and f"_g{sat}_" in Path(u).name]
        spans, annuals = [], []
        for url in candidates:
            name = Path(url).name
            span = re.search(r"_s(\d{8})_e(\d{8})_", name)
            annual = re.search(r"_y(\d{4})_", name)
            if span:
                first, last = int(span.group(1)[:4]), int(span.group(2)[:4])
                if any(first <= year <= last for year in requested):
                    spans.append(url)
            elif annual and int(annual.group(1)) in requested:
                annuals.append(url)
        chosen = spans if spans else annuals
        # If no aggregate exists, enumerate year folders to find daily files.
        if not chosen:
            for year in requested:
                year_url = next((u for u in root_files if u.rstrip("/").endswith(f"/{year}")), None)
                if not year_url:
                    continue
                months = directory_files(year_url.rstrip("/") + "/")
                for month_url in months:
                    if not month_url.rstrip("/").endswith(tuple(f"/{m:02d}" for m in range(1, 13))):
                        continue
                    chosen.extend(u for u in directory_files(month_url.rstrip("/") + "/") if u.endswith(".nc"))
        for url in sorted(set(chosen)):
            dest = raw / "xrs" / f"g{sat}" / Path(url).name
            try:
                get(url, dest)
                files.append(dest)
            except requests.RequestException as exc:
                print(f"WARN download failed {url}: {exc}")
    if not files:
        raise RuntimeError("No XRS NetCDF inputs downloaded; check NOAA archive availability")
    return files


def read_xrs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            with xr.open_dataset(path, decode_times=True) as ds:
                # NOAA names XRS-B long channel flux as xrsb_flux on L2 avg1m.
                time_name = next((x for x in ("time", "t") if x in ds.coords or x in ds), None)
                flux_name = next((x for x in ("xrsb_flux", "xrsa_flux") if x in ds.data_vars), None)
                if not time_name or not flux_name:
                    raise ValueError(f"Unexpected variables in {path.name}: {list(ds.variables)}")
                time = pd.to_datetime(ds[time_name].values, utc=True)
                flux = np.asarray(ds[flux_name].values).reshape(-1)
                frame = pd.DataFrame({"time": time, "xrsb_flux_w_m2": flux,
                                      "source_satellite": re.search(r"_g(\d+)_", path.name).group(1)})
                frames.append(frame)
        except Exception as exc:
            print(f"WARN read failed {path}: {exc}")
    if not frames:
        raise RuntimeError("No readable XRS inputs")
    xrs = pd.concat(frames, ignore_index=True)
    xrs = xrs[(xrs.time >= TIME0) & (xrs.time < TIME1)]
    xrs = xrs.dropna(subset=["xrsb_flux_w_m2"])
    xrs = xrs[xrs.xrsb_flux_w_m2 > 0]
    # Multiple spacecraft overlap. Average contemporaneous valid fluxes; this
    # avoids arbitrary satellite switching while preserving coverage.
    return (xrs.groupby("time", as_index=False)
            .agg(xrsb_flux_w_m2=("xrsb_flux_w_m2", "median"),
                 satellite_count=("source_satellite", "nunique")))


def read_flares(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ("time", "start_time", "end_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # Irradiance-based class threshold is more robust than parsing class strings:
    # M1.0 corresponds to 1e-5 W/m^2 in the GOES 1-8 A band.
    peak_col = next((c for c in ("xrsb_irrad", "peak_flux", "xrsb_flux", "irradiance") if c in df), None)
    if peak_col is None:
        class_col = next((c for c in ("current_class", "flare_class", "class") if c in df), None)
        if class_col is None:
            raise ValueError(f"No peak flux/class column. Columns: {df.columns.tolist()}")
        vals = df[class_col].astype(str).str.upper().str.extract(r"([ABCMX])\s*([\d.]+)")
        mult = vals[0].map({"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4})
        df["_peak_w_m2"] = mult * pd.to_numeric(vals[1], errors="coerce")
        peak_col = "_peak_w_m2"
    df["_peak_w_m2"] = pd.to_numeric(df[peak_col], errors="coerce")
    return df.dropna(subset=["time", "_peak_w_m2"])[["time", "_peak_w_m2"]].drop_duplicates()


def flare_catalog_end(path: Path) -> pd.Timestamp:
    """Return exclusive archive coverage date encoded in NCEI filename."""
    match = re.search(r"_e(\d{8})_", path.name)
    if not match:
        raise ValueError(f"Cannot determine flare catalog end date from {path.name}")
    return pd.Timestamp(match.group(1), tz="UTC") + pd.Timedelta(days=1)


def make_hourly(xrs: pd.DataFrame, flares: pd.DataFrame, catalog_end: pd.Timestamp) -> pd.DataFrame:
    minute = xrs.set_index("time").sort_index()
    minute["log10_xrsb_flux"] = np.log10(minute.xrsb_flux_w_m2.clip(lower=1e-12))
    # Do not bridge long outages: hourly minimum coverage is 45 valid minutes.
    hourly = minute.resample("1h").agg(
        xrsb_flux_mean_w_m2=("xrsb_flux_w_m2", "mean"),
        xrsb_flux_max_w_m2=("xrsb_flux_w_m2", "max"),
        xrsb_flux_median_w_m2=("xrsb_flux_w_m2", "median"),
        log10_flux_mean=("log10_xrsb_flux", "mean"),
        valid_minutes=("xrsb_flux_w_m2", "count"),
        satellite_count=("satellite_count", "max"),
    )
    hourly.index.name = "sample_time_utc"
    hourly = hourly[hourly.valid_minutes >= 45]
    # Label window is (hour boundary, boundary + 24h], using flare PEAK time.
    # searchsorted makes the half-open intervals explicit and efficient.
    ft = np.sort(flares.loc[(flares.time >= TIME0) & (flares.time < TIME1) &
                            (flares._peak_w_m2 >= 1e-5), "time"].astype("int64").to_numpy())
    starts = hourly.index.astype("int64").to_numpy()
    left = np.searchsorted(ft, starts, side="right")
    right = np.searchsorted(ft, starts + pd.Timedelta(hours=24).value, side="right")
    hourly["m_plus_count_next_24h"] = right - left
    hourly["label_m_plus_next_24h"] = (hourly.m_plus_count_next_24h > 0).astype("int8")
    hourly = hourly.reset_index()
    hourly = hourly[(hourly.sample_time_utc >= TIME0) &
                    (hourly.sample_time_utc + pd.Timedelta(hours=24) <= catalog_end)]
    return hourly.sort_values("sample_time_utc").reset_index(drop=True)


def write_report(hourly: pd.DataFrame, minute: pd.DataFrame, flare_path: Path, report: Path) -> None:
    n = len(hourly)
    pos = int(hourly.label_m_plus_next_24h.sum()) if n else 0
    coverage = float(minute.time.nunique() / max(1, int((TIME1 - TIME0).total_seconds() / 60)))
    requested_hours = int((TIME1 - TIME0).total_seconds() / 3600)
    minute_counts = minute.set_index("time").xrsb_flux_w_m2.resample("1h").count()
    n_valid_feature_hours = int((minute_counts >= 45).sum())
    summary = {
        "window_utc": {"start_inclusive": TIME0.isoformat(), "end_exclusive": TIME1.isoformat()},
        "flare_catalog_coverage_end_exclusive_utc": flare_catalog_end(flare_path).isoformat(),
        "hourly_samples": n,
        "positive_hours_next_24h": pos,
        "positive_rate": pos / n if n else None,
        "negative_hours": n - pos,
        "hourly_class_imbalance_negative_to_positive": ((n - pos) / pos if pos else None),
        "minute_observation_coverage_fraction": coverage,
        "hourly_rows_with_at_least_45_valid_minutes": n,
        "hours_with_at_least_45_valid_minutes": n_valid_feature_hours,
        "hours_excluded_for_insufficient_flux_coverage": max(0, requested_hours - n_valid_feature_hours),
        "time_ordered": bool(hourly.sample_time_utc.is_monotonic_increasing),
        "duplicate_sample_hours": int(hourly.sample_time_utc.duplicated().sum()),
        "flare_catalog_file": flare_path.name,
        "label_definition": "Any catalog flare with peak XRS-B >= 1e-5 W/m^2 (M1+) whose peak is in (sample_time, sample_time+24h].",
        "notes": ["Rows need >=45 valid 1-minute observations per hour.",
                  "Overlapping spacecraft measurements are median-combined by timestamp.",
                  "No forward fill or random resampling is used.",
                  "Raw NOAA inputs are retained under data/raw for reproducibility."],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n")
    md = ["# SolarGuard AI — Phase 1 data quality", "", f"Window: {TIME0.date()} through {(TIME1 - pd.Timedelta(days=1)).date()} UTC.", "",
          f"- Hourly samples retained: **{n:,}**", f"- Positive next-24h M+ samples: **{pos:,} ({summary['positive_rate']:.3%})**" if n else "- Positive next-24h M+ samples: unavailable (no rows)",
          f"- Negative:positive ratio: **{summary['hourly_class_imbalance_negative_to_positive']:.1f}:1**" if pos else "- Negative:positive ratio: unavailable",
          f"- Minute-level coverage across the window: **{coverage:.1%}**", f"- Chronological ordering: **{summary['time_ordered']}**; duplicate sample hours: **{summary['duplicate_sample_hours']}**", "",
          "Labels are based on the flare peak time and the M1.0 GOES threshold (10⁻⁵ W/m²). Class prevalence counts overlapping hourly forecast windows, so positive hours are not the same as the number of unique flares.", "",
          "## Caveats", "", "- This window starts before GOES-16 became operational. Historical mission coverage therefore depends on GOES-13–15 science-quality archives; satellite handoffs and calibration differences should be reviewed before modeling.",
          f"- Flare catalog coverage ends {flare_catalog_end(flare_path).date()} UTC (exclusive); rows without a complete future 24-hour catalog window are removed.",
          "- Observation archive coverage can also end before the requested end date. Check the minute coverage and source file spans.",
          "- Valid-hour filtering drops hours with fewer than 45 observations; retained timestamps remain in UTC with no interpolation.", ""]
    (report.parent / "data_quality.md").write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-11-18")
    ap.add_argument("--end", default="2026-11-18", help="exclusive UTC date")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--output", default="data/processed/solarguard_hourly.csv.gz")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()
    global TIME0, TIME1
    TIME0 = pd.Timestamp(args.start, tz="UTC")
    TIME1 = pd.Timestamp(args.end, tz="UTC")
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    flpath = fetch_flare_report(raw)
    xpaths = fetch_xrs(raw, range(TIME0.year, TIME1.year + 1))
    xrs = read_xrs(xpaths)
    flares = read_flares(flpath)
    hourly = make_hourly(xrs, flares, flare_catalog_end(flpath))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%SZ")
    write_report(hourly, xrs, flpath, Path(args.report_dir) / "data_quality.json")
    print(f"Wrote {len(hourly):,} hourly rows to {out}")


if __name__ == "__main__":
    main()
