# SolarGuard AI

## Project roadmap and handoff

**Project goal:** Estimate the probability of at least one M1.0-or-stronger solar flare in the 24 hours after the latest completed hour, using GOES X-ray flux. This is a research prototype, not an operational space-weather warning service.

**Current state (2026-11-18):** Phases 1–4 and Phase 6 are implemented and verified with automated test suites. The frontend production build was verified and enhanced with a dual live/archive telemetry switcher. Relative origin URL resolution was introduced to resolve prior blank-preview issues caused by localhost/127.0.0.1 cross-origin mismatch. Phase 5 deployment files (`Dockerfile`, `compose.yaml`) are present and ready for Docker environments.

| Phase | Status | Deliverable / handoff |
|---|---|---|
| 0. Define scope and evaluation guardrails | Complete | Predict next-24-hour M+ occurrence; use chronological splits; make rarity, BSS/TSS, calibration, and risk bands visible; label outputs experimental. |
| 1. Build the historical dataset | Complete | `src/build_dataset.py` downloads/reads NOAA GOES XRS and flare records, aligns UTC hours, applies coverage checks, creates leakage-safe next-24-hour labels, and writes the dataset and quality report. |
| 2. Establish forecasting baselines | Complete | `src/train_baseline.py` compares logistic regression, histogram gradient boosting, and training climatology with chronological train/validation/test partitions and 24-hour purges. Calibration and threshold selection use validation only. BSS/TSS and the reliability diagram are reported. |
| 3. Serve forecasts through an API | Complete | `api/main.py` exposes `/health`, `/history`, `/evaluation`, `/predict`, and `/predict/latest`. The prediction endpoint accepts completed hourly history and returns a calibrated probability, risk band, threshold, and forecast window. |
| 4. Build the dashboard | Complete & verified | `frontend/` provides interactive viewing of live GOES-18 SWPC telemetry and historical archive snapshot. Seven-day XRS-B chart, next-24-hour forecast, and clearly separated held-out test evaluation benchmarks. |
| 5. Package a local deployment | Implemented; Docker run pending | `Dockerfile` and `compose.yaml` build a single service for the API and dashboard. Run `docker compose up --build` when a Docker daemon is available. |
| 6. Add current NOAA observations | Complete | `src/live_feed.py` consumes NOAA SWPC GOES primary XRS rolling feed, isolates 0.1–0.8 nm band, aggregates completed UTC hours using the 45-min coverage rule, checks staleness, and passes history to `api/main.py` endpoints `/live/status`, `/live/history`, and `/live/predict`. |
| 7. Strengthen scientific evaluation | Planned; next implementation step | Add rolling-origin / multiple-period backtests and inspect performance by satellite era and solar activity level. Preserves untouched chronological test periods; reports BSS, TSS, reliability, and positive prevalence rather than accuracy alone. |
| 8. Harden operations and reproducibility | In Progress (Test suite active) | Automated test suite in `tests/` tests data processing, coverage checks, staleness detection, lookahead prevention, and API validation. Re-run with `pytest -v`. |
| 9. Prepare the final demo and judge materials | Planned | Present the question, sources, label definition, time split and purge, flare imbalance, calibration/reliability, BSS/TSS against climatology, limitations, and a clearly marked experimental forecast example. |

**Next action for a continuing developer/model:** Proceed with Phase 7 (Strengthen scientific evaluation: rolling-origin backtests across activity regimes). Keep the existing historical archive, baseline evaluation artifacts, and live feed endpoints intact.

Primary live-feed reference: [NOAA SWPC GOES primary XRS JSON directory](https://services.swpc.noaa.gov/json/goes/primary/) (includes rolling `xrays-1-day.json`, `xrays-3-day.json`, and `xrays-7-day.json` products).

---

## Phase 1: historical data pipeline

Build an hourly dataset from NOAA GOES X-ray observations with a forward-looking label for whether an M1.0-or-stronger flare peaks in the following 24 hours. All timestamps are UTC, samples are chronological, and hours with substantial observation gaps are omitted.

```bash
python src/build_dataset.py
```

Outputs:
- `data/processed/solarguard_hourly.csv.gz`: hourly feature dataset with binary label.
- `reports/data_quality.md` and `reports/data_quality.json`: coverage, prevalence, missing hours.

---

## Phase 2: chronological baseline evaluation

```bash
python src/train_baseline.py
```

Trains logistic regression and histogram gradient boosting against training climatology on chronological 70%/15%/15% splits with 24-hour boundary purges. Calibration and TSS-selected decision thresholds are determined on validation data only.

Outputs:
- `models/solarguard_logistic.joblib`
- `reports/model_evaluation.json`
- `reports/reliability_diagram.png`

---

## Phase 3 & 6: local prediction API with live SWPC observations

Start the API from the repository root:

```bash
uvicorn api.main:app --reload
```

### Historical Archive Endpoints
- `GET /health`: Service health and model status.
- `GET /history?hours=168`: Trailing hourly observations from the historical archive snapshot.
- `GET /evaluation`: Baseline evaluation metrics (BSS, TSS, ROC-AUC) on the chronological test holdout.
- `GET /predict/latest`: Forecast for the latest completed hour in the historical archive.
- `POST /predict`: Predict on 25–72 custom completed hourly observations.

### Live NOAA SWPC Endpoints
- `GET /live/status`: Checks live telemetry feed connectivity, satellite ID (e.g. GOES-18), staleness minutes, and active status.
- `GET /live/history?hours=168`: Hourly aggregated samples from NOAA SWPC primary XRS operational feed, enforcing the 45-minute coverage check.
- `GET /live/predict`: Computes live next-24-hour M+ flare probability using current GOES-18 primary telemetry. If the live feed is unavailable or has gaps, returns explicit HTTP errors (no silent fallback).

---

## Phase 4: local dashboard

Run Vite in development:

```bash
cd frontend
npm install
npm run dev
```

Or build and serve through FastAPI at `http://localhost:8000`:

```bash
cd frontend && npm run build
uvicorn api.main:app --port 8000
```

The dashboard includes a top-level source switcher:
1. **Live SWPC (GOES-18):** Real-time operational feed with active pulse indicator, freshness display, live 7-day XRS-B chart, and live flare forecast.
2. **Historical Archive:** Benchmark view using the historical dataset snapshot.
3. **Held-Out Test Metrics:** Clearly attributed to the chronological test holdout (2025–2026 science-quality data) to ensure users do not conflate historical test metrics with real-time operational telemetry.

---

## Phase 5: one-command local deployment

Package API and dashboard into a single Docker container:

```bash
docker compose up --build
```

Access at `http://localhost:8000`.

---

## Automated test suite

Run all unit and integration tests:

```bash
python3 -m pytest -v
```

Tests cover:
- Live SWPC minute filtering (0.1–0.8 nm band) and positive flux verification.
- 45/60-minute coverage rule enforcement and completed UTC hour filtering.
- Feed staleness detection and threshold flagging.
- Contiguous sequence and gap detection.
- Feature extraction lookahead prevention.
- API validation and endpoint responses.

---

## Source references

- [NOAA SWPC GOES X-ray Flux data and measurement notes](https://www.spaceweather.gov/products/goes-x-ray-flux)
- [NOAA SWPC GOES Primary XRS JSON products](https://services.swpc.noaa.gov/json/goes/primary/)
- [NOAA NCEI GOES-R EXIS/XRS data access](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance)
- [NOAA NCEI GOES-1–15 science-quality XRS NetCDF archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/)
- [NOAA NCEI GOES Flare Report documentation](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf)
