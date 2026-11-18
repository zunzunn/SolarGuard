# SolarGuard AI

## Project roadmap and handoff

**Project goal:** Estimate the probability of at least one M1.0-or-stronger solar flare in the 24 hours after the latest completed hour, using GOES X-ray flux. This is a research prototype, not an operational space-weather warning service.

**Current state (2026-11-18):** Phases 1–4 are implemented. Phase 5 deployment files are implemented, but the Docker image has not been built or run because Docker was unavailable in the previous development environment. The frontend production build succeeded earlier; its browser preview was blank and still needs a runtime check. The API and dashboard currently use the historical archive snapshot; they do not fetch current SWPC observations.

| Phase | Status | Deliverable / handoff |
|---|---|---|
| 0. Define scope and evaluation guardrails | Complete | Predict next-24-hour M+ occurrence; use chronological splits; make rarity, BSS/TSS, calibration, and risk bands visible; label outputs experimental. |
| 1. Build the historical dataset | Complete | `src/build_dataset.py` downloads/reads NOAA GOES XRS and flare records, aligns UTC hours, applies coverage checks, creates leakage-safe next-24-hour labels, and writes the dataset and quality report. See the data-pipeline instructions below. |
| 2. Establish forecasting baselines | Complete | `src/train_baseline.py` compares logistic regression, histogram gradient boosting, and training climatology with chronological train/validation/test partitions and 24-hour purges. Calibration and threshold selection use validation only. BSS/TSS and the reliability diagram are reported. |
| 3. Serve forecasts through an API | Complete | `api/main.py` exposes `/health`, `/history`, `/evaluation`, `/predict`, and `/predict/latest`. The prediction endpoint accepts completed hourly history and returns a calibrated probability, risk band, threshold, and forecast window. |
| 4. Build the dashboard | Implemented; browser runtime check pending | `frontend/` shows the archive forecast, seven-day XRS-B chart, held-out BSS/TSS, and test label prevalence. Previous browser preview was blank; confirm JavaScript rendering and API loading before calling this phase fully verified. |
| 5. Package a local deployment | Implemented; Docker run pending | `Dockerfile` and `compose.yaml` build one service for the API and dashboard. From the repository root, run `docker compose up --build`; open `http://localhost:8000`. Build and smoke-test when Docker is available. |
| 6. Add current NOAA observations | Planned; next implementation step | Consume NOAA SWPC's public GOES primary XRS JSON feed, inspect the live payload, select the 0.1–0.8 nm band, aggregate minute observations into completed UTC hours using the same 45/60-minute coverage rule, and pass at least 25 consecutive hourly samples through the existing feature and prediction code. Add a source timestamp, coverage/staleness checks, and a separate live endpoint. Keep current telemetry visibly distinct from historical test results; assess the archive-to-live distribution shift before presenting live probabilities as calibrated. |
| 7. Strengthen scientific evaluation | Planned | Add rolling-origin / multiple-period backtests and inspect performance by satellite era and activity level. Any candidate-model comparison must select models, calibrators, and thresholds with training/validation data only; preserve a final untouched chronological test period. Report BSS, TSS, reliability, and positive prevalence rather than accuracy alone. |
| 8. Harden operations and reproducibility | Planned | Add automated checks for data schema, ordering, gaps, label boundaries, API input validation, and frontend/API integration; pin or lock deployment dependencies; verify Docker health and restart behavior; document update cadence and stale-feed behavior. Do not silently fall back from live telemetry to archive data. |
| 9. Prepare the final demo and judge materials | Planned | Present the question, sources, label definition, time split and purge, flare imbalance, calibration/reliability, BSS/TSS against climatology, limitations, and a clearly marked experimental forecast example. Do not imply operational warning capability. |

**Next action for a continuing developer/model:** implement Phase 6 only after checking the official SWPC JSON payload and timestamp/coverage behavior. Keep the existing historical archive and its held-out evaluation artifacts unchanged. Do not retrain the model or redesign the frontend as part of feed ingestion unless a later task asks for it. Once the live API has reliable coverage metadata, update the dashboard to show its live-vs-archive source and freshness explicitly.

Primary live-feed reference: [NOAA SWPC GOES primary XRS JSON directory](https://services.swpc.noaa.gov/json/goes/primary/) (includes rolling `xrays-1-day.json`, `xrays-3-day.json`, and `xrays-7-day.json` products). The directory is a rolling operational feed, separate from the science-quality historical inputs used in Phase 1 and model evaluation.

## Phase 1: historical data pipeline

Build an hourly dataset from NOAA GOES X-ray observations with a forward-looking label for whether an M1.0-or-stronger flare peaks in the following 24 hours. All timestamps are UTC, samples are chronological, and hours with substantial observation gaps are omitted. This phase stops at data preparation; it does not train a model.

## Data sources and format

- **Observations:** NOAA NCEI GOES XRS Level 2 science-quality `avg1m` archives. The input is NetCDF and contains 1-minute X-ray irradiances; the pipeline reads the long wavelength XRS-B flux channel (`xrsb_flux`, W/m²). It checks GOES-13–15 legacy science-quality archives and GOES-16–19 GOES-R archives, discovering yearly or mission-span files from their public directory listings.
- **Flare events:** NOAA NCEI's mission-length science-quality GOES flare report in CSV, with accompanying product documentation. It combines GOES flare-summary data and SWPC event reports. Event times and peak irradiance/class are used to identify M1+ peaks.
- **Window:** Defaults to 2016-11-18 inclusive through 2026-11-17 inclusive. Use `--end` as an exclusive UTC date to change it.

The last-ten-year window crosses the GOES-13–15 to GOES-R transition. Those satellite generations have different calibration histories; NCEI publishes science-quality products and documentation, but inspect the quality report and source satellite coverage before comparing measurements across handoffs. Archive coverage is not uniform, and updates can lag the present date.

## Setup and run

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/build_dataset.py
```

The first run downloads several hundred MB (and may take longer depending on NOAA access). Downloads are cached under `data/raw`, so rerunning skips existing files. For a smaller trial, pass a shorter date range:

```bash
python src/build_dataset.py --start 2024-01-01 --end 2025-01-01
```

Outputs:

- `data/processed/solarguard_hourly.csv.gz`: one row per retained UTC hour, with mean/maximum/median flux, mean log10 flux, valid minute count, satellite count, M+ event count in the next 24 hours, and binary label.
- `reports/data_quality.md` and `reports/data_quality.json`: coverage, sample counts, positive prevalence, imbalance, missing-hour count, and basic ordering checks.
- `data/raw/`: the source inputs used to create the outputs.

Each hourly feature row is timestamped at the end of its completed UTC hour: for sample time `t`, features use measurements from `[t - 1 hour, t)`. The row is positive if a catalog flare peak with XRS-B peak flux >= 1e-5 W/m² occurs in `(t, t + 24 hours]`. This keeps every predictor measurement before the forecast window. A row is retained only when at least 45 of its 60 one-minute observations exist. No flux interpolation or forward filling is done. Concurrent satellites at a timestamp are median-combined.

## Phase 2: chronological baseline evaluation

Run `python src/train_baseline.py` after the data pipeline. It creates trailing 6-hour and 24-hour flux summaries/deltas, then compares logistic regression and histogram gradient boosting against a training-period climatology forecast. The split is chronological (70% train, 15% validation, 15% test), with 24-hour label windows purged when they reach or cross a split boundary. Calibration and the TSS-selected probability threshold use validation data only; the test period is reserved for final metrics. The report includes Brier Skill Score (BSS), TSS, ROC-AUC, average precision, and confusion counts. Risk bands use `<10% Low`, `10–<40% Moderate`, `40–70% High`, and `>70% Severe`.

Outputs are `reports/model_evaluation.json`, `reports/model_evaluation.md`, `reports/reliability_diagram.png`, `data/processed/test_predictions.csv.gz`, and the serialized validation-selected model at `models/solarguard_logistic.joblib`.

## Phase 3: local prediction API

After Phase 2 has created the model artifact, start the API from the repository root:

```bash
uvicorn api.main:app --reload
```

`GET /health` reports whether the model loaded. `POST /predict` accepts 25–72 consecutive completed hourly rows, each containing `sample_time_utc`, `log10_flux_mean`, `xrsb_flux_median_w_m2`, and `xrsb_flux_max_w_m2`. These fields match the hourly dataset. The service returns a calibrated probability, the requested risk band, and the validation-selected decision threshold. The model remains an experimental baseline; this endpoint is not an operational space-weather warning service.

## Phase 4: local dashboard

With the API running, start the React dashboard in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. The page loads the latest archived forecast, a seven-day XRS-B flux chart, test BSS/TSS, and test-period label prevalence from the API. This is an archive snapshot, not live solar telemetry. To point the UI to a different API origin, set `VITE_API_BASE_URL` before starting Vite (for example, `http://127.0.0.1:8000`).

## Phase 5: one-command local deployment

Docker packages the existing dashboard and API into one service. It serves the dashboard at `/` and the API under the same origin, avoiding separate frontend/API startup and browser cross-origin setup. Docker Desktop (or Docker Engine with the Compose plugin) must be installed. Build and start it from the repository root:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). Check [http://localhost:8000/health](http://localhost:8000/health) for service health. Stop it with `Ctrl+C`, or run `docker compose down` from another terminal. The image includes the checked-in hourly archive and evaluation report; rebuild it after regenerating either artifact. Forecasts remain experimental and use the latest archived sample, not live telemetry.

## Source references

- [NOAA SWPC GOES X-ray Flux data and measurement notes](https://www.spaceweather.gov/products/goes-x-ray-flux)
- [NOAA NCEI GOES-R EXIS/XRS data access](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance)
- [NOAA NCEI GOES-1–15 space-weather instrument archives](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments)
- [NOAA NCEI GOES-1–15 science-quality XRS NetCDF archive and README](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/)
- [NOAA NCEI GOES Flare Report README (CSV and NetCDF formats, input sources, event definitions)](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf)
- [NOAA NCEI science-quality flare-report CSV directory](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/)
- [NOAA GOES-R XRS L2 examples: reading the flare-report CSV](https://cires-stp.github.io/goesr-spwx-examples/examples/exis/flrpt_example_csv.html)
