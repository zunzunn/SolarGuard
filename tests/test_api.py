"""Integration and API validation tests."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "logistic_regression"
    assert data["live_feed_configured"] is True


def test_api_evaluation():
    resp = client.get("/evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert "test_metrics" in data
    assert "logistic_regression" in data["test_metrics"]
    assert data["selected_model"] == "logistic_regression"


def test_api_predict_latest_archive():
    resp = client.get("/predict/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["data_source"] == "historical_archive"
    assert 0.0 <= data["m_plus_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Moderate", "High", "Severe"]
    assert "forecast_window_utc" in data
    assert "decision_threshold" in data


def test_api_live_endpoints():
    # Live status
    status_resp = client.get("/live/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] in ["ok", "stale"]
    assert "feed" in status_data

    # Live history
    hist_resp = client.get("/live/history?hours=48")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["source"] == "noaa_swpc_live_feed"
    assert len(hist_data["samples"]) <= 48
    assert len(hist_data["samples"]) >= 24

    # Live predict
    pred_resp = client.get("/live/predict")
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["data_source"] == "NOAA_SWPC_OPERATIONAL_GOES_PRIMARY"
    assert 0.0 <= pred_data["m_plus_probability"] <= 1.0
    assert pred_data["risk_level"] in ["Low", "Moderate", "High", "Severe"]
    assert pred_data["consecutive_hours_used"] >= 25
    assert "distribution_shift_note" in pred_data


def test_api_predict_input_validation():
    # Rejects fewer than 25 rows
    now = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
    few_rows = [
        {
            "sample_time_utc": (now - timedelta(hours=i)).isoformat(),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        }
        for i in range(10)
    ]
    resp = client.post("/predict", json={"observations": few_rows})
    assert resp.status_code == 422

    # Rejects unordered timestamps
    valid_count = 25
    unordered_rows = []
    for i in range(valid_count):
        t = now - timedelta(hours=valid_count - i)
        unordered_rows.append({
            "sample_time_utc": t.isoformat(),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        })
    # Swap two elements to make out of order
    unordered_rows[5], unordered_rows[6] = unordered_rows[6], unordered_rows[5]
    resp = client.post("/predict", json={"observations": unordered_rows})
    assert resp.status_code == 422
    assert "increasing time order" in resp.text

    # Rejects gap
    gap_rows = []
    for i in range(valid_count):
        # Introduce a 2-hour jump at step 10
        hour_offset = i if i < 10 else i + 2
        t = now - timedelta(hours=valid_count - hour_offset)
        gap_rows.append({
            "sample_time_utc": t.isoformat(),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        })
    resp = client.post("/predict", json={"observations": gap_rows})
    assert resp.status_code == 422
    assert "no gaps" in resp.text

    # Accepts valid consecutive 25 rows
    valid_rows = []
    for i in range(valid_count):
        t = now - timedelta(hours=valid_count - 1 - i)
        valid_rows.append({
            "sample_time_utc": t.isoformat(),
            "log10_flux_mean": -6.5,
            "xrsb_flux_median_w_m2": 3e-7,
            "xrsb_flux_max_w_m2": 5e-7,
        })
    resp = client.post("/predict", json={"observations": valid_rows})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["m_plus_probability"] <= 1.0


def test_api_flare_events_and_stats():
    # Stats summary
    resp = client.get("/stats/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_hours_cataloged"] > 80000
    assert data["total_m_flares"] > 1000
    assert data["total_x_flares"] > 50
    assert "X" in data["peak_flux_class"]

    # Flare events catalog
    resp = client.get("/flare-events?limit=10")
    assert resp.status_code == 200
    catalog = resp.json()
    assert catalog["returned_count"] == 10
    assert catalog["total_count"] > 1000
    assert len(catalog["events"]) == 10
    first = catalog["events"][0]
    assert "flare_class" in first
    assert "flux_max_w_m2" in first

    # Flare events with class filter
    resp = client.get("/flare-events?min_class=X&limit=5")
    assert resp.status_code == 200
    x_catalog = resp.json()
    assert all(e["class_type"] == "X" for e in x_catalog["events"])

    # CSV export
    resp = client.get("/export/flare-events?limit=20")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "sample_time_utc" in resp.text
    assert len(resp.text.strip().split("\n")) == 21

    # Scenario simulation
    resp = client.post("/simulate/scenario", json={"scenario": "x_class_burst"})
    assert resp.status_code == 200
    sim = resp.json()
    assert sim["simulated_probability"] > sim["baseline_probability"]
    assert sim["simulated_risk"] in ["High", "Severe"]
