#!/usr/bin/env python3
"""Train/evaluate chronological SolarGuard forecasting baselines (Phase 2).

The test period is held out until final scoring. A 24-hour purge prevents labels
from crossing train/validation or validation/test boundaries. Calibration and
the TSS decision threshold are fit on validation data only.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, roc_curve,
                             roc_auc_score, average_precision_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from src.features import FEATURES, add_flux_features
except ModuleNotFoundError:  # Supports the documented `python src/train_baseline.py` command.
    from features import FEATURES, add_flux_features


DATA = Path("data/processed/solarguard_hourly.csv.gz")
REPORT = Path("reports/model_evaluation.json")
PREDICTIONS = Path("data/processed/test_predictions.csv.gz")
PLOT = Path("reports/reliability_diagram.png")
MODEL = Path("models/solarguard_logistic.joblib")
HORIZON = pd.Timedelta(hours=24)


def calibrate_sigmoid(raw: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, np.ndarray]:
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    cal.fit(z, y)
    return cal, cal.predict_proba(z)[:, 1]


def apply_calibrator(cal: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return cal.predict_proba(z)[:, 1]


def best_tss_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, p)
    score = tpr - fpr
    ix = int(np.nanargmax(score))
    return float(thresholds[ix]), float(score[ix])


def score(y: np.ndarray, p: np.ndarray, threshold: float, reference_rate: float) -> dict:
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    bs = float(brier_score_loss(y, p))
    bs_ref = float(brier_score_loss(y, np.full(len(y), reference_rate)))
    return {
        "brier_score": bs,
        "brier_skill_score_vs_training_climatology": 1 - bs / bs_ref if bs_ref else None,
        "tss": sensitivity + specificity - 1,
        "threshold_from_validation": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["sample_time_utc"])
    df["sample_time_utc"] = pd.to_datetime(df.sample_time_utc, utc=True)
    if not df.sample_time_utc.is_monotonic_increasing or df.sample_time_utc.duplicated().any():
        raise ValueError("Hourly dataset must be strictly time-ordered with unique timestamps")
    d = add_flux_features(df).dropna(subset=FEATURES + ["label_m_plus_next_24h"])
    features = FEATURES
    times = d.index
    i70, i85 = int(.70 * len(d)), int(.85 * len(d))
    val_start, test_start = times[i70], times[i85]
    train_mask = (times < val_start) & (times + HORIZON < val_start)
    val_mask = (times >= val_start) & (times < test_start) & (times + HORIZON < test_start)
    test_mask = times >= test_start
    train, val, test = d.loc[train_mask], d.loc[val_mask], d.loc[test_mask]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One or more chronological partitions are empty")
    X_train, y_train = train[features], train.label_m_plus_next_24h.astype(int).to_numpy()
    X_val, y_val = val[features], val.label_m_plus_next_24h.astype(int).to_numpy()
    X_test, y_test = test[features], test.label_m_plus_next_24h.astype(int).to_numpy()
    train_rate = float(y_train.mean())

    estimators = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, class_weight=None)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=.08, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=1.0,
            early_stopping=False, random_state=17),
    }
    results, test_probabilities, thresholds, calibrators = {}, {}, {}, {}
    for name, model in estimators.items():
        model.fit(X_train, y_train)
        raw_val = model.predict_proba(X_val)[:, 1]
        calibrator, calibrated_val = calibrate_sigmoid(raw_val, y_val)
        calibrators[name] = calibrator
        threshold, val_tss = best_tss_threshold(y_val, calibrated_val)
        calibrated_test = apply_calibrator(calibrator, model.predict_proba(X_test)[:, 1])
        results[name] = {
            **score(y_test, calibrated_test, threshold, train_rate),
            "validation_tss_at_selected_threshold": val_tss,
        }
        test_probabilities[name] = calibrated_test
        thresholds[name] = threshold

    prior = np.full(len(y_test), train_rate)
    results["training_climatology"] = {
        **score(y_test, prior, .5, train_rate),
        "threshold_from_validation": None,
        "brier_skill_score_vs_training_climatology": 0.0,
    }

    # Select model using validation metrics only; never use test results to choose.
    chosen = max(estimators, key=lambda n: results[n]["validation_tss_at_selected_threshold"])
    p = test_probabilities[chosen]
    edges = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, 9)
    observed, predicted, counts = [], [], []
    for b in range(10):
        m = bin_ids == b
        if m.any():
            predicted.append(float(p[m].mean()))
            observed.append(float(y_test[m].mean()))
            counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="#777", label="Perfect calibration")
    ax.plot(predicted, observed, "o-", color="#1769aa", label=chosen.replace("_", " "))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed M+ frequency",
           title="Reliability diagram — chronological test period")
    ax.grid(alpha=.25)
    ax.legend(loc="upper left")
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)

    pred = test[["sample_time_utc", "label_m_plus_next_24h"]].copy()
    for name, probs in test_probabilities.items():
        pred[f"probability_{name}"] = probs
        pred[f"risk_{name}"] = np.select([probs < .1, probs < .4, probs <= .7],
                                         ["Low", "Moderate", "High"], default="Severe")
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PREDICTIONS, index=False, compression="gzip", date_format="%Y-%m-%dT%H:%M:%SZ")

    summary = {
        "dataset": str(DATA),
        "feature_definition": "Hourly flux features plus trailing 6h/24h summaries and deltas; all flux windows end at sample_time.",
        "split_method": "Chronological 70%/15%/15% by timestamp; no random split.",
        "purge": "Drop samples whose 24-hour label window reaches or crosses a train/validation or validation/test boundary.",
        "calibration": "Sigmoid calibration fitted on validation probabilities only.",
        "threshold_selection": "TSS-maximizing threshold selected on validation data only; test period is untouched until scoring.",
        "boundaries_utc": {"validation_start": val_start.isoformat(), "test_start": test_start.isoformat()},
        "partition_rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "purged_rows": {"train_boundary": int(((times < val_start) & ~train_mask).sum()),
                        "validation_boundary": int(((times >= val_start) & (times < test_start) & ~val_mask).sum())},
        "training_climatology_rate": train_rate,
        "features": features,
        "test_metrics": results,
        "calibration_plot": str(PLOT),
        "test_predictions": str(PREDICTIONS),
        "selected_model": chosen,
        "model_artifact": str(MODEL),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": estimators[chosen], "calibrator": calibrators[chosen],
                 "features": features, "decision_threshold": thresholds[chosen],
                 "selected_by": "highest validation TSS; test set not used for selection",
                 "validation_start_utc": val_start.isoformat(),
                 "test_start_utc": test_start.isoformat()}, MODEL)
    print(json.dumps({"partitions": summary["partition_rows"], "metrics": results,
                      "plot": str(PLOT), "predictions": str(PREDICTIONS),
                      "model_artifact": str(MODEL), "selected_model": chosen}, indent=2))


if __name__ == "__main__":
    main()
