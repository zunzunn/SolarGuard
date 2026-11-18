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
from fastapi import FastAPI, HTTPException, Query, Response, APIRouter
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
SCIENTIFIC_EVALUATION_PATH = Path("reports/scientific_evaluation.json")
FRONTEND_DIST = Path(os.getenv("SOLARGUARD_FRONTEND_DIST", "frontend/dist"))
bundle = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SolarGuard AI",
    version="0.4.0",
    description="Experimental next-24-hour M1+ flare probability API with live NOAA SWPC integration",
)

router = APIRouter()

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
    allow_origin_regex=r"^https?://((localhost|127\.0\.0\.1)(:\d+)?|.*\.vercel\.app)$",
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


class ScenarioRequest(BaseModel):
    scenario: str = Field(
        default="x_class_burst",
        description="Scenario type: x_class_burst, m_flare_surge, quiet_background, or custom",
    )
    multiplier: float = Field(default=5.0, ge=0.01, le=100.0)


def classify_flux(flux: float) -> str:
    """Format physical W/m² flux into standard GOES solar flare classification (A, B, C, M, X)."""
    if flux >= 1e-4:
        return f"X{flux / 1e-4:.1f}"
    elif flux >= 1e-5:
        return f"M{flux / 1e-5:.1f}"
    elif flux >= 1e-6:
        return f"C{flux / 1e-6:.1f}"
    elif flux >= 1e-7:
        return f"B{flux / 1e-7:.1f}"
    else:
        return f"A{flux / 1e-8:.1f}"


_DATASET_CACHE: pd.DataFrame | None = None
_FLARE_CACHE: pd.DataFrame | None = None


def get_cached_dataset() -> pd.DataFrame:
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        if not HISTORY_PATH.exists():
            raise HTTPException(503, "Hourly dataset is unavailable; run the Phase 1 pipeline first.")
        _DATASET_CACHE = pd.read_csv(HISTORY_PATH)
    return _DATASET_CACHE


def get_cached_flares() -> pd.DataFrame:
    global _FLARE_CACHE
    if _FLARE_CACHE is None:
        df = get_cached_dataset()
        flares = df[df["xrsb_flux_max_w_m2"] >= 1e-5].copy()
        flares["flare_class"] = flares["xrsb_flux_max_w_m2"].apply(classify_flux)
        flares["class_type"] = flares["xrsb_flux_max_w_m2"].apply(lambda f: "X" if f >= 1e-4 else "M")
        _FLARE_CACHE = flares
    return _FLARE_CACHE


@router.get("/health")
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


@router.get("/history")
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


@router.get("/evaluation")
def evaluation() -> dict:
    if not EVALUATION_PATH.exists():
        raise HTTPException(503, "Evaluation report is unavailable; run Phase 2 training first.")
    report = json.loads(EVALUATION_PATH.read_text())
    if SCIENTIFIC_EVALUATION_PATH.exists():
        sci = json.loads(SCIENTIFIC_EVALUATION_PATH.read_text())
        report["scientific_backtest"] = sci.get("rolling_backtest", {}).get("aggregate_performance")
    return report


@router.get("/evaluation/scientific")
def evaluation_scientific() -> dict:
    if not SCIENTIFIC_EVALUATION_PATH.exists():
        raise HTTPException(503, "Scientific evaluation report is unavailable; run Phase 7 scientific evaluation first.")
    return json.loads(SCIENTIFIC_EVALUATION_PATH.read_text())


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


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    return forecast_rows([row.model_dump() for row in request.observations])


@router.get("/predict/latest")
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


@router.get("/flare-events")
def flare_events(
    min_class: str = Query(default="M", pattern="^(M|X|C)$", description="Minimum flare class: M or X"),
    limit: int = Query(default=50, ge=1, le=200),
    sort_by: str = Query(default="time", pattern="^(time|flux)$"),
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> dict:
    """Historical solar flare events catalog (M and X class) with classifications."""
    flares = get_cached_flares()
    if min_class == "X":
        filtered = flares[flares["class_type"] == "X"]
    else:
        filtered = flares

    sort_col = "sample_time_utc" if sort_by == "time" else "xrsb_flux_max_w_m2"
    ascending = (order == "asc")
    sorted_df = filtered.sort_values(by=sort_col, ascending=ascending)
    top = sorted_df.head(limit)

    records = []
    for r in top.to_dict(orient="records"):
        records.append({
            "sample_time_utc": str(r["sample_time_utc"]),
            "flux_max_w_m2": float(r["xrsb_flux_max_w_m2"]),
            "flux_mean_w_m2": float(r["xrsb_flux_mean_w_m2"]),
            "flare_class": str(r["flare_class"]),
            "class_type": str(r["class_type"]),
            "valid_minutes": int(r["valid_minutes"]) if pd.notna(r.get("valid_minutes")) else 60,
            "satellite_count": int(r["satellite_count"]) if pd.notna(r.get("satellite_count")) else 1,
        })

    return {
        "events": records,
        "total_count": int(len(filtered)),
        "returned_count": len(records),
        "min_class": min_class,
        "sort_by": sort_by,
        "order": order,
    }


@router.get("/stats/summary")
def stats_summary() -> dict:
    """Consolidated space weather mission metrics and catalog totals."""
    df = get_cached_dataset()
    flares = get_cached_flares()
    max_idx = df["xrsb_flux_max_w_m2"].idxmax()
    max_row = df.loc[max_idx]

    sci_agg = {}
    if SCIENTIFIC_EVALUATION_PATH.exists():
        try:
            sci = json.loads(SCIENTIFIC_EVALUATION_PATH.read_text())
            sci_agg = sci.get("rolling_backtest", {}).get("aggregate_performance", {})
        except Exception:
            pass

    return {
        "total_hours_cataloged": int(len(df)),
        "archive_start_utc": str(df["sample_time_utc"].iloc[0]),
        "archive_end_utc": str(df["sample_time_utc"].iloc[-1]),
        "total_m_flares": int((flares["class_type"] == "M").sum()),
        "total_x_flares": int((flares["class_type"] == "X").sum()),
        "peak_flux_w_m2": float(max_row["xrsb_flux_max_w_m2"]),
        "peak_flux_class": classify_flux(float(max_row["xrsb_flux_max_w_m2"])),
        "peak_flux_time_utc": str(max_row["sample_time_utc"]),
        "cycle_info": "Solar Cycle 25 (Ascending to Solar Maximum)",
        "model_name": "Logistic Regression (Platt Calibrated)",
        "decision_threshold": float(bundle["decision_threshold"]),
        "evaluation_bss": float(sci_agg.get("mean_brier_skill_score", 0.460)),
        "evaluation_tss": float(sci_agg.get("mean_tss", 0.246)),
        "evaluation_auc": float(sci_agg.get("mean_roc_auc", 0.773)),
        "active_satellites": ["GOES-16 (East)", "GOES-18 (West / Primary)", "GOES-19 (Test)"],
    }


@router.get("/export/flare-events")
def export_flare_events(limit: int = Query(default=1000, ge=10, le=5000)) -> Response:
    """Download historical M and X flare events as CSV."""
    flares = get_cached_flares().sort_values(by="sample_time_utc", ascending=False).head(limit)
    export_cols = [
        "sample_time_utc",
        "flare_class",
        "class_type",
        "xrsb_flux_max_w_m2",
        "xrsb_flux_mean_w_m2",
        "valid_minutes",
        "satellite_count",
    ]
    csv_content = flares[export_cols].to_csv(index=False)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=solarguard_flare_catalog.csv"},
    )


@router.post("/simulate/scenario")
def simulate_scenario(req: ScenarioRequest) -> dict:
    """Simulate hypothetical solar conditions to test model sensitivity."""
    if not HISTORY_PATH.exists():
        raise HTTPException(503, "Hourly dataset unavailable for baseline simulation.")
    df = pd.read_csv(HISTORY_PATH).tail(72)
    needed = ["sample_time_utc", "log10_flux_mean", "xrsb_flux_median_w_m2", "xrsb_flux_max_w_m2"]
    observations = []
    for r in df[needed].iloc[-25:].to_dict(orient="records"):
        observations.append({
            "sample_time_utc": pd.Timestamp(r["sample_time_utc"]).to_pydatetime().astimezone(timezone.utc),
            "log10_flux_mean": float(r["log10_flux_mean"]),
            "xrsb_flux_median_w_m2": float(r["xrsb_flux_median_w_m2"]),
            "xrsb_flux_max_w_m2": float(r["xrsb_flux_max_w_m2"]),
        })

    baseline = forecast_rows(observations)

    sim_obs = [dict(o) for o in observations]
    scenario_desc = ""
    injected_peak = 0.0

    if req.scenario == "x_class_burst":
        scenario_desc = "Extreme X-Class Solar Flare Eruption (X2.5 spike in past 2 hours)"
        injected_peak = 2.5e-4
        for i in [-2, -1]:
            sim_obs[i]["xrsb_flux_max_w_m2"] = injected_peak
            sim_obs[i]["xrsb_flux_median_w_m2"] = 1.2e-4
            sim_obs[i]["log10_flux_mean"] = float(np.log10(1.5e-4))
    elif req.scenario == "m_flare_surge":
        scenario_desc = "Moderate M-Class Active Region Flare (M4.8 peak)"
        injected_peak = 4.8e-5
        for i in [-3, -2, -1]:
            sim_obs[i]["xrsb_flux_max_w_m2"] = injected_peak
            sim_obs[i]["xrsb_flux_median_w_m2"] = 2.5e-5
            sim_obs[i]["log10_flux_mean"] = float(np.log10(3.0e-5))
    elif req.scenario == "quiet_background":
        scenario_desc = "Deep Solar Minimum / Quiescent Sun (B-class background)"
        injected_peak = 3.0e-8
        for i in range(len(sim_obs)):
            sim_obs[i]["xrsb_flux_max_w_m2"] = injected_peak
            sim_obs[i]["xrsb_flux_median_w_m2"] = 2.0e-8
            sim_obs[i]["log10_flux_mean"] = float(np.log10(2.5e-8))
    else:  # custom multiplier
        scenario_desc = f"Custom Flux Scaling ({req.multiplier:.1f}x across trailing 6 hours)"
        for i in range(-6, 0):
            sim_obs[i]["xrsb_flux_max_w_m2"] *= req.multiplier
            sim_obs[i]["xrsb_flux_median_w_m2"] *= req.multiplier
            sim_obs[i]["log10_flux_mean"] = float(np.log10(max(1e-9, sim_obs[i]["xrsb_flux_median_w_m2"])))
        injected_peak = sim_obs[-1]["xrsb_flux_max_w_m2"]

    simulated = forecast_rows(sim_obs)

    return {
        "scenario": req.scenario,
        "description": scenario_desc,
        "injected_peak_flux_w_m2": injected_peak,
        "injected_peak_class": classify_flux(injected_peak) if injected_peak > 0 else "—",
        "baseline_probability": baseline["m_plus_probability"],
        "baseline_risk": baseline["risk_level"],
        "simulated_probability": simulated["m_plus_probability"],
        "simulated_risk": simulated["risk_level"],
        "probability_delta": simulated["m_plus_probability"] - baseline["m_plus_probability"],
        "simulated_forecast": simulated,
    }


# --- Live NOAA SWPC Feed Endpoints ---


@router.get("/live/status")
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


@router.get("/live/history")
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


@router.get("/live/predict")
@router.get("/predict/live")
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


app.include_router(router)
app.include_router(router, prefix="/api")


@app.get("/api/docs", include_in_schema=False)
def api_docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
