"""Unit and integration tests for Phase 7 scientific evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.scientific_evaluation import (
    best_tss_threshold,
    calibrate_sigmoid,
    apply_calibrator,
    compute_metrics,
    run_activity_stratification,
)


def test_compute_metrics_calculations():
    # 10 samples: 5 positive, 5 negative
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    # Good probabilities
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.15, 0.3, 0.25, 0.85])
    metrics = compute_metrics(y, p, threshold=0.5, reference_rate=0.5)

    assert metrics["sample_count"] == 10
    assert metrics["positive_count"] == 5
    assert metrics["positive_rate"] == 0.5
    assert metrics["true_positives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 5
    assert metrics["false_negatives"] == 0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["tss"] == 1.0
    assert metrics["false_alarm_ratio"] == 0.0
    assert metrics["brier_skill_score"] > 0.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_handles_zero_cases():
    # All negatives
    y = np.zeros(10, dtype=int)
    p = np.full(10, 0.1)
    metrics = compute_metrics(y, p, threshold=0.5, reference_rate=0.0)

    assert metrics["sample_count"] == 10
    assert metrics["positive_count"] == 0
    assert metrics["false_alarm_ratio"] == 0.0
    assert metrics["sensitivity"] == 0.0
    assert metrics["specificity"] == 1.0


def test_calibration_and_tss_threshold():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    cal, cal_val = calibrate_sigmoid(raw, y)
    assert len(cal_val) == 8
    assert np.all(cal_val >= 0.0) and np.all(cal_val <= 1.0)

    # Threshold selection
    thresh, tss = best_tss_threshold(y, cal_val)
    assert tss > 0.5
    assert 0.0 <= thresh <= 1.0


def test_run_activity_stratification():
    rows = []
    for i in range(30):
        # 10 quiet, 10 active, 10 storm
        if i < 10:
            log_flux = -7.5  # Quiet
            label = 0
            prob = 0.05
        elif i < 20:
            log_flux = -6.5  # Active
            label = 1 if i % 2 == 0 else 0
            prob = 0.4
        else:
            log_flux = -5.5  # Storm
            label = 1
            prob = 0.85
        rows.append({
            "log_flux_mean_24h": log_flux,
            "label_m_plus_next_24h": label,
        })
    df = pd.DataFrame(rows)
    y = df["label_m_plus_next_24h"].to_numpy()
    p = np.array([0.05]*10 + [0.4]*10 + [0.85]*10)

    stratified = run_activity_stratification(df, y, p, threshold=0.5)
    assert len(stratified) == 3
    for name, res in stratified.items():
        assert res["sample_count"] == 10
        assert "tss" in res
        assert "brier_skill_score" in res


def test_api_scientific_evaluation_endpoint():
    client = TestClient(app)
    resp = client.get("/evaluation/scientific")
    assert resp.status_code == 200
    data = resp.json()
    assert "rolling_backtest" in data
    assert "activity_regimes" in data
    assert "satellite_eras" in data
    agg = data["rolling_backtest"]["aggregate_performance"]
    assert "mean_brier_skill_score" in agg
    assert "mean_tss" in agg
    assert "mean_roc_auc" in agg

    # Check that main /evaluation endpoint now includes scientific backtest summary
    eval_resp = client.get("/evaluation")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert "scientific_backtest" in eval_data
