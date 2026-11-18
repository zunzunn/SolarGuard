"""Small local API for the validation-selected SolarGuard baseline."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import add_flux_features

MODEL_PATH = Path(os.getenv("SOLARGUARD_MODEL_PATH", "models/solarguard_logistic.joblib"))
bundle = joblib.load(MODEL_PATH)
app = FastAPI(title="SolarGuard AI", version="0.3.0",
              description="Experimental next-24-hour M1+ flare probability API")


class HourlyFlux(BaseModel):
    sample_time_utc: datetime = Field(description="End of this completed UTC hour")
    log10_flux_mean: float = Field(description="Mean log10 XRS-B flux over the completed hour")
    xrsb_flux_median_w_m2: float = Field(gt=0)
    xrsb_flux_max_w_m2: float = Field(gt=0)


class PredictRequest(BaseModel):
    observations: list[HourlyFlux] = Field(min_length=25, max_length=72)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "logistic_regression", "artifact": str(MODEL_PATH)}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    rows = [row.model_dump() for row in request.observations]
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
    if probability < .1:
        risk = "Low"
    elif probability < .4:
        risk = "Moderate"
    elif probability <= .7:
        risk = "High"
    else:
        risk = "Severe"
    start = times[-1]
    end = start + timedelta(hours=24)
    threshold = float(bundle["decision_threshold"])
    return {
        "sample_time_utc": start.isoformat().replace("+00:00", "Z"),
        "forecast_window_utc": {"start_exclusive": start.isoformat().replace("+00:00", "Z"),
                                "end_inclusive": end.isoformat().replace("+00:00", "Z")},
        "m_plus_probability": probability,
        "risk_level": risk,
        "decision_threshold": threshold,
        "predict_m_plus": bool(probability >= threshold),
        "model": "logistic_regression_sigmoid_calibrated",
        "model_use_note": "Research prototype; evaluation used a chronological holdout, not operational deployment data.",
    }
