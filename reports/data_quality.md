# SolarGuard AI — Phase 1 data quality

Sample window: 2016-11-18 through 2026-11-16 UTC. Each sample time is the **end of its completed feature hour**; the first row is at 01:00 UTC.

- Hourly samples retained: **87,588**
- Positive next-24h M+ samples: **19,427 (22.180%)**
- Negative:positive ratio: **3.5:1**
- Minute-level coverage across the requested period: **99.96%**
- Chronological ordering: **True**; duplicate sample hours: **0**

Features use the completed UTC hour `[sample_time - 1h, sample_time)`. Labels are based on flare peak time and the M1.0 GOES threshold (10⁻⁵ W/m²), with a positive if a peak occurs in `(sample_time, sample_time + 24h]`. Class prevalence counts overlapping hourly forecast windows, so positive hours are not the same as the number of unique flares.

## Caveats

- This window starts before GOES-16 became operational. Historical mission coverage therefore depends on GOES-13–15 science-quality archives; satellite handoffs and calibration differences should be reviewed before modeling.
- Flare catalog coverage ends 2026-11-17 UTC (exclusive); rows without a complete future 24-hour catalog window are removed.
- Observation archive coverage can also end before the requested end date. Check the minute coverage and source file spans.
- Valid-hour filtering drops hours with fewer than 45 observations; retained timestamps remain in UTC with no interpolation.
