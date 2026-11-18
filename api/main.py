"""Small local API for the validation-selected SolarGuard baseline and live NOAA SWPC feed."""
from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features import add_flux_features
from src.live_feed import (
    live_feed_manager,
    get_consecutive_hourly_samples,
    MIN_CONSECUTIVE_HOURS,
    MAX_CONSECUTIVE_HOURS,
)

MODEL_PATH = Path(os.getenv("SOLARGUARD_MODEL_PATH", "models/solarguard_logistic.joblib"))
HISTORY_PATH = Path(os.getenv("SOLARGUARD_DATA_PATH", "data/processed/solarguard_hourly.csv.gz"))
EVALUATION_PATH = Path("reports/model_evaluation.json")
FRONTEND_DIST = Path(os.getenv("SOLARGUARD_FRONTEND_DIST", "frontend/dist"))
bundle = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SolarGuard AI",
    version="0.4.0",
    description="Experimental next-24-hour M1+ flare probability API with live NOAA SWPC integration",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class HourlyFlux(BaseModel):
    sample_time_utc: datetime = Field(description="End of this completed UTC hour")
    log10_flux_mean: float = Field(description="Mean log10 XRS-B flux over the completed hour")
    xrsb_flux_median_w_m2: float = Field(gt=0)
    xrsb_flux_max_w_m2: float = Field(gt=0)


class PredictRequest(BaseModel):
    observations: list[HourlyFlux] = Field(min_length=25, max_length=72)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "logistic_regression",
        "artifact": str(MODEL_PATH),
        "live_feed_configured": True,
    }


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    index = FRONTEND_DIST / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard files are unavailable; build the frontend first.")
    return FileResponse(index)


if FRONTEND_DIST.joinpath("assets").is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="dashboard-assets")


@app.get("/history")
def history(hours: int = Query(default=168, ge=24, le=720)) -> dict:
    """Historical archive snapshot observations."""
    if not HISTORY_PATH.exists():
        raise HTTPException(503, "Hourly dataset is unavailable; run the Phase 1 pipeline first.")
    data = pd.read_csv(HISTORY_PATH).tail(hours)
    fields = [
        "sample_time_utc",
        "log10_flux_mean",
        "xrsb_flux_median_w_m2",
        "xrsb_flux_max_w_m2",
        "xrsb_flux_mean_w_m2",
        "label_m_plus_next_24h",
    ]
    return {
        "source": "historical_archive",
        "samples": data[fields].where(pd.notna(data[fields]), None).to_dict(orient="records"),
        "requested_hours": hours,
        "archive_latest_utc": data.sample_time_utc.iloc[-1] if len(data) else None,
    }


@app.get("/evaluation")
def evaluation() -> dict:
    if not EVALUATION_PATH.exists():
        raise HTTPException(503, "Evaluation report is unavailable; run Phase 2 training first.")
    return json.loads(EVALUATION_PATH.read_text())


def forecast_rows(rows: list[dict]) -> dict:
    times = [row["sample_time_utc"] for row in rows]
    if any(t.tzinfo is None or t.utcoffset() is None for t in times):
        raise HTTPException(422, "Every sample_time_utc must include a timezone (use UTC/Z).")
    times = [t.astimezone(timezone.utc) for t in times]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise HTTPException(422, "Observations must be in strictly increasing time order.")
    if any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
        raise HTTPException(422, "Provide consecutive completed hourly samples with no gaps.")
    df = pd.DataFrame(rows)
    df["sample_time_utc"] = times
    engineered = add_flux_features(df)
    latest = engineered.iloc[-1][bundle["features"]]
    if latest.isna().any() or not np.isfinite(latest.to_numpy(dtype=float)).all():
        raise HTTPException(422, "Insufficient history to compute the model's trailing features.")
    raw_probability = float(bundle["model"].predict_proba(latest.to_frame().T)[:, 1][0])
    clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
    logit = np.log(clipped / (1 - clipped))
    probability = float(bundle["calibrator"].predict_proba([[logit]])[:, 1][0])
    if probability < 0.1:
        risk = "Low"
    elif probability < 0.4:
        risk = "Moderate"
    elif probability <= 0.7:
        risk = "High"
    else:
        risk = "Severe"
    start = times[-1]
    end = start + timedelta(hours=24)
    threshold = float(bundle["decision_threshold"])
    return {
        "sample_time_utc": start.isoformat().replace("+00:00", "Z"),
        "forecast_window_utc": {
            "start_exclusive": start.isoformat().replace("+00:00", "Z"),
            "end_inclusive": end.isoformat().replace("+00:00", "Z"),
        },
        "m_plus_probability": probability,
        "risk_level": risk,
        "decision_threshold": threshold,
        "predict_m_plus": bool(probability >= threshold),
        "model": "logistic_regression_sigmoid_calibrated",
        "model_use_note": "Research prototype; evaluation used a chronological holdout, not operational deployment data.",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    return forecast_rows([row.model_dump() for row in request.observations])


@app.get("/predict/latest")
def predict_latest() -> dict:
    """Forecast for the latest completed hour in the historical archive."""
    if not HISTORY_PATH.exists():
        raise HTTPException(503, "Hourly dataset is unavailable; run the Phase 1 pipeline first.")
    data = pd.read_csv(HISTORY_PATH).tail(72)
    needed = ["sample_time_utc", "log10_flux_mean", "xrsb_flux_median_w_m2", "xrsb_flux_max_w_m2"]
    observations = []
    previous = None
    for row in data[needed].iloc[::-1].to_dict(orient="records"):
        stamp = pd.Timestamp(row["sample_time_utc"])
        if previous is not None and previous - stamp != pd.Timedelta(hours=1):
            break
        observations.append({**row, "sample_time_utc": stamp.to_pydatetime()})
        previous = stamp
        if len(observations) == 25:
            break
    if len(observations) < 25:
        raise HTTPException(503, "The archive has fewer than 25 consecutive hourly samples at its end.")
    result = forecast_rows(list(reversed(observations)))
    result["data_source"] = "historical_archive"
    return result


# --- Live NOAA SWPC Feed Endpoints ---


@app.get("/live/status")
def live_status() -> dict:
    """Check connectivity and staleness of NOAA SWPC primary feed."""
    try:
        hourly, meta = live_feed_manager.get_data()
        return {
            "status": "ok" if not meta["is_stale"] else "stale",
            "feed": meta,
        }
    except requests.RequestException as exc:
        raise HTTPException(502, f"Failed to connect to NOAA SWPC feed: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Error inspecting NOAA SWPC feed: {exc}")


@app.get("/live/history")
def live_history(hours: int = Query(default=168, ge=24, le=720)) -> dict:
    """Completed hourly samples from NOAA SWPC primary XRS operational feed."""
    try:
        hourly, meta = live_feed_manager.get_data()
    except requests.RequestException as exc:
        raise HTTPException(502, f"Failed to retrieve NOAA SWPC live feed: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Live feed error: {exc}")

    if hourly.empty:
        raise HTTPException(503, "No completed hourly records currently available from live feed.")

    tail_df = hourly.tail(hours)
    fields = [
        "sample_time_utc",
        "log10_flux_mean",
        "xrsb_flux_median_w_m2",
        "xrsb_flux_max_w_m2",
        "xrsb_flux_mean_w_m2",
        "valid_minutes",
    ]
    # Format sample_time_utc as ISO strings
    samples = []
    for row in tail_df[fields].to_dict(orient="records"):
        row["sample_time_utc"] = pd.Timestamp(row["sample_time_utc"]).isoformat().replace("+00:00", "Z")
        samples.append(row)

    return {
        "source": "noaa_swpc_live_feed",
        "samples": samples,
        "requested_hours": hours,
        "feed_metadata": meta,
    }


@app.get("/live/predict")
@app.get("/predict/live")
def predict_live(force_refresh: bool = False) -> dict:
    """Predict next-24-hour M+ flare probability using current NOAA SWPC GOES primary XRS observations."""
    try:
        hourly, meta = live_feed_manager.get_data(force_refresh=force_refresh)
    except requests.RequestException as exc:
        raise HTTPException(502, f"Failed to fetch live observations from NOAA SWPC: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Live feed processing error: {exc}")

    try:
        consecutive_df, check_meta = get_consecutive_hourly_samples(
            hourly,
            min_hours=MIN_CONSECUTIVE_HOURS,
            max_hours=MAX_CONSECUTIVE_HOURS,
        )
    except ValueError as exc:
        raise HTTPException(503, f"Live feed gap or insufficient history: {exc}")

    rows = []
    for r in consecutive_df.to_dict(orient="records"):
        rows.append({
            "sample_time_utc": pd.Timestamp(r["sample_time_utc"]).to_pydatetime(),
            "log10_flux_mean": float(r["log10_flux_mean"]),
            "xrsb_flux_median_w_m2": float(r["xrsb_flux_median_w_m2"]),
            "xrsb_flux_max_w_m2": float(r["xrsb_flux_max_w_m2"]),
        })

    forecast = forecast_rows(rows)
    forecast["data_source"] = meta["data_source"]
    forecast["source_satellite"] = meta["source_satellite"]
    forecast["latest_observation_utc"] = meta["latest_observation_utc"]
    forecast["is_stale"] = meta["is_stale"]
    forecast["staleness_minutes"] = meta["staleness_minutes"]
    forecast["consecutive_hours_used"] = check_meta["consecutive_hours_found"]
    forecast["distribution_shift_note"] = meta["distribution_shift_note"]
    forecast["model_use_note"] = (
        "Operational SWPC real-time primary data. Research prototype; "
        "not an operational space weather warning service."
    )
    return forecast
