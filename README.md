# SolarGuard AI — Phase 1 data pipeline

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

Label definition: a row at hour boundary `t` is positive if a catalog flare peak with XRS-B peak flux >= 1e-5 W/m² occurs in `(t, t + 24 hours]`. This avoids assigning a peak at or before the sample time to its future label. A row is retained only when at least 45 of its 60 one-minute observations exist. No flux interpolation or forward filling is done. Concurrent satellites at a timestamp are median-combined.

## Source references

- [NOAA SWPC GOES X-ray Flux data and measurement notes](https://www.spaceweather.gov/products/goes-x-ray-flux)
- [NOAA NCEI GOES-R EXIS/XRS data access](https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance)
- [NOAA NCEI GOES-1–15 space-weather instrument archives](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments)
- [NOAA NCEI GOES-1–15 science-quality XRS NetCDF archive and README](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/)
- [NOAA NCEI GOES Flare Report README (CSV and NetCDF formats, input sources, event definitions)](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf)
- [NOAA NCEI science-quality flare-report CSV directory](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/)
- [NOAA GOES-R XRS L2 examples: reading the flare-report CSV](https://cires-stp.github.io/goesr-spwx-examples/examples/exis/flrpt_example_csv.html)
