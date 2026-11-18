import { useCallback, useEffect, useMemo, useState } from 'react'

const API = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')

function shortDate(value, withTime = false) {
  if (!value) return '—'
  const date = new Date(value)
  return new Intl.DateTimeFormat('en', withTime
    ? { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short' }
    : { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date)
}

function FluxChart({ samples, title = "GOES XRS-B mean flux" }) {
  const points = useMemo(() => samples.map(row => ({
    date: new Date(row.sample_time_utc),
    value: Number(row.xrsb_flux_mean_w_m2),
  })).filter(p => p.value > 0 && Number.isFinite(p.value)), [samples])

  if (points.length < 2) {
    return <div className="chart-empty">Not enough valid X-ray observations to draw this view.</div>
  }

  const width = 820, height = 270, left = 66, right = 18, top = 16, bottom = 38
  const logs = points.map(p => Math.log10(p.value))
  const min = Math.floor(Math.min(...logs)), max = Math.ceil(Math.max(...logs))
  const yMin = Math.min(min, -9), yMax = Math.max(max, -5)
  const x = i => left + i * (width - left - right) / (points.length - 1)
  const y = v => top + (yMax - Math.log10(v)) * (height - top - bottom) / (yMax - yMin)
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`).join(' ')
  const ticks = Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i)
  const labels = [0, Math.floor((points.length - 1) / 2), points.length - 1]

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <defs>
          <linearGradient id="flux-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#f6a64b" stopOpacity=".2" />
            <stop offset="100%" stopColor="#f6a64b" stopOpacity="0" />
          </linearGradient>
        </defs>
        {ticks.map(tick => {
          const yPos = top + (yMax - tick) * (height - top - bottom) / (yMax - yMin)
          return (
            <g key={tick}>
              <line x1={left} x2={width - right} y1={yPos} y2={yPos} className="grid-line" />
              <text x={left - 10} y={yPos + 4} textAnchor="end" className="axis-label">
                1e{tick}
              </text>
            </g>
          )
        })}
        <path d={`${path} L ${x(points.length - 1)} ${height - bottom} L ${x(0)} ${height - bottom} Z`} fill="url(#flux-fill)" />
        <path d={path} fill="none" className="flux-line" />
        {labels.map(i => (
          <text
            key={i}
            x={x(i)}
            y={height - 10}
            textAnchor={i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle'}
            className="axis-label"
          >
            {shortDate(points[i].date)}
          </text>
        ))}
      </svg>
      <div className="chart-legend"><span className="legend-dot" />XRS-B mean · 1–8 Å</div>
    </div>
  )
}

function Metric({ label, value, detail }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  )
}

export default function App() {
  const [mode, setMode] = useState('live') // 'live' | 'archive'
  const [state, setState] = useState({
    loading: true,
    error: '',
    health: null,
    history: [],
    forecast: null,
    evaluation: null,
    metadata: null,
  })

  const refresh = useCallback(async (selectedMode = mode) => {
    setState(current => ({ ...current, loading: true, error: '' }))
    try {
      const isLive = selectedMode === 'live'
      const historyUrl = isLive ? `${API}/live/history?hours=168` : `${API}/history?hours=168`
      const forecastUrl = isLive ? `${API}/live/predict` : `${API}/predict/latest`

      const [healthRes, historyRes, forecastRes, evalRes] = await Promise.all([
        fetch(`${API}/health`),
        fetch(historyUrl),
        fetch(forecastUrl),
        fetch(`${API}/evaluation`),
      ])

      if (!forecastRes.ok) {
        const errorData = await forecastRes.json().catch(() => ({}))
        const detail = errorData.detail || `Prediction failed with status ${forecastRes.status}`
        throw new Error(isLive ? `Live SWPC Feed Error: ${detail}` : `Archive Error: ${detail}`)
      }

      if (!historyRes.ok) {
        throw new Error(`Failed to load ${isLive ? 'live' : 'archive'} telemetry history (${historyRes.status})`)
      }

      const [health, historyData, forecast, evaluation] = await Promise.all([
        healthRes.json(),
        historyRes.json(),
        forecastRes.json(),
        evalRes.json(),
      ])

      setState({
        loading: false,
        error: '',
        health,
        history: historyData.samples || [],
        forecast,
        evaluation,
        metadata: isLive ? (historyData.feed_metadata || null) : null,
      })
    } catch (error) {
      setState(current => ({
        ...current,
        loading: false,
        error: error.message || 'Could not reach the SolarGuard API service.',
      }))
    }
  }, [mode])

  const handleModeSwitch = newMode => {
    setMode(newMode)
    refresh(newMode)
  }

  useEffect(() => {
    refresh(mode)
  }, [mode, refresh])

  const metrics = state.evaluation?.test_metrics?.[state.evaluation?.selected_model]
  const latest = state.history.at(-1)
  const probability = state.forecast?.m_plus_probability
  const riskClass = (state.forecast?.risk_level || 'moderate').toLowerCase()
  const updated = state.forecast?.sample_time_utc || latest?.sample_time_utc
  const isLive = mode === 'live'

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SolarGuard AI home">
          <span className="brand-mark"><span /></span>
          <span>SolarGuard<span className="brand-ai"> AI</span></span>
        </a>

        <div className="topbar-right">
          <div className="source-switcher" role="radiogroup" aria-label="Data source selection">
            <button
              className={`source-btn ${isLive ? 'active' : ''}`}
              onClick={() => handleModeSwitch('live')}
              aria-checked={isLive}
              role="radio"
            >
              <span className={`live-dot ${state.metadata?.is_stale ? 'stale' : ''}`} />
              Live SWPC (GOES-18)
            </button>
            <button
              className={`source-btn ${!isLive ? 'active' : ''}`}
              onClick={() => handleModeSwitch('archive')}
              aria-checked={!isLive}
              role="radio"
            >
              <span>◷</span> Historical Archive
            </button>
          </div>

          <button
            className="refresh-button"
            onClick={() => refresh(mode)}
            disabled={state.loading}
            aria-label="Refresh dashboard"
          >
            <span className={state.loading ? 'refresh-icon spinning' : 'refresh-icon'}>↻</span> Refresh
          </button>
        </div>
      </header>

      <section className="page-heading" id="top">
        <div>
          <div className="eyebrow">
            {isLive ? 'SPACE WEATHER · LIVE OPERATIONAL TELEMETRY' : 'SPACE WEATHER · HISTORICAL ARCHIVE BENCHMARK'}
          </div>
          <h1>Solar activity outlook</h1>
          <p>Estimated probability of an M-class or stronger solar flare in the next 24 hours.</p>
        </div>

        <div className="archive-note">
          <span className="note-icon">{isLive ? '📡' : '◷'}</span>
          <div>
            <span>{isLive ? 'Latest Completed Hour' : 'Archive Snapshot'}</span>
            <br />
            <strong>{shortDate(updated, true)}</strong>
          </div>
        </div>
      </section>

      {isLive && (
        <section className="shift-callout" aria-label="Operational telemetry notice">
          <span className={`shift-tag ${state.metadata?.is_stale ? 'stale' : ''}`}>
            {state.metadata?.is_stale
              ? `STALE FEED (${state.metadata?.staleness_minutes}m lag)`
              : `OPERATIONAL FEED · ${state.metadata?.staleness_minutes ?? 0}m lag`}
          </span>
          <span>
            Real-time GOES-18 primary XRS data from NOAA SWPC. Experimental prototype evaluated against science-quality historical archives; operational telemetry distribution shift may affect calibration.
          </span>
        </section>
      )}

      {state.error && (
        <section className="error-banner">
          <span>!</span>
          <div>
            <strong>Dashboard data unavailable</strong>
            <p>{state.error}</p>
            {isLive && (
              <button
                className="banner-action-btn"
                onClick={() => handleModeSwitch('archive')}
              >
                Switch to Historical Archive Snapshot
              </button>
            )}
          </div>
        </section>
      )}

      <section className="overview-grid">
        <article className="forecast-card">
          <div className="forecast-card-top">
            <span className="eyebrow">
              {isLive ? 'NEXT 24 HOURS (LIVE)' : 'NEXT 24 HOURS (ARCHIVE)'}
            </span>
            <span className={`risk-badge ${riskClass}`}>{state.forecast?.risk_level || '—'} risk</span>
          </div>

          <div className="probability-row">
            <strong>{probability == null ? '—' : `${(probability * 100).toFixed(1)}%`}</strong>
            <span className="probability-caption">M+ flare<br />probability</span>
          </div>

          <div className="forecast-track">
            <span style={{ width: `${Math.max(0, Math.min(100, (probability || 0) * 100))}%` }} />
          </div>

          <div className="forecast-foot">
            <span>Forecast window</span>
            <strong>
              {state.forecast
                ? `${shortDate(state.forecast.forecast_window_utc.start_exclusive, true)} → ${shortDate(state.forecast.forecast_window_utc.end_inclusive, true)}`
                : '—'}
            </strong>
          </div>

          <div className="model-decision">
            <span className={state.forecast?.predict_m_plus ? 'decision-dot elevated' : 'decision-dot'} />
            {state.forecast
              ? state.forecast.predict_m_plus
                ? `Above decision threshold (${(state.forecast.decision_threshold * 100).toFixed(1)}%)`
                : `Below decision threshold (${(state.forecast.decision_threshold * 100).toFixed(1)}%)`
              : 'Waiting for forecast'}
          </div>
        </article>

        <article className="chart-card">
          <div className="card-heading">
            <div>
              <div className="eyebrow">{isLive ? 'NOAA SWPC LIVE TELEMETRY' : 'NOAA GOES ARCHIVE'}</div>
              <h2>{isLive ? 'Live 7-day X-ray flux' : 'Archived 7-day X-ray flux'}</h2>
            </div>
            <span className="period-tag">{isLive ? 'PAST 168 HOURS' : 'ARCHIVE 7 DAYS'}</span>
          </div>
          <FluxChart samples={state.history} title="GOES XRS-B 1-8 Angstrom flux" />
          <div className="chart-foot">
            <span>Mean XRS-B flux in W/m² · logarithmic scale</span>
            <span>Source: {isLive ? 'NOAA SWPC GOES-18 Primary' : 'NOAA NCEI GOES'}</span>
          </div>
        </article>
      </section>

      <section className="stats-grid" aria-label="Current activity and model performance">
        <article className="stat-card activity-stat">
          <div className="eyebrow">
            {isLive ? 'LATEST COMPLETED HOUR (LIVE)' : 'LATEST COMPLETED HOUR (ARCHIVE)'}
          </div>
          <div className="activity-value">
            {latest?.xrsb_flux_mean_w_m2 ? Number(latest.xrsb_flux_mean_w_m2).toExponential(2) : '—'}
            <small>W/m²</small>
          </div>
          <div className="stat-sub">
            XRS-B mean · 1–8 Å
            {latest?.valid_minutes != null && ` (${latest.valid_minutes}/60 valid mins)`}
          </div>
        </article>

        <article className="stat-card">
          <div className="eyebrow">
            HELD-OUT HISTORICAL TEST · {state.evaluation?.selected_model?.replaceAll('_', ' ') || 'MODEL'}
          </div>
          <div className="metric-pair">
            <Metric
              label="BSS"
              value={metrics ? metrics.brier_skill_score_vs_training_climatology.toFixed(3) : '—'}
              detail="vs training climatology"
            />
            <Metric
              label="TSS"
              value={metrics ? metrics.tss.toFixed(3) : '—'}
              detail="at validation threshold"
            />
          </div>
        </article>

        <article className="stat-card imbalance-stat">
          <div className="eyebrow">HISTORICAL PREVALENCE</div>
          <div className="activity-value">
            {state.evaluation?.test_metrics?.[state.evaluation?.selected_model]?.positive_rate != null
              ? `${(state.evaluation.test_metrics[state.evaluation.selected_model].positive_rate * 100).toFixed(1)}%`
              : '—'}
          </div>
          <div className="stat-sub">Chronological holdout M+ prevalence (2025–2026)</div>
        </article>
      </section>

      <section className="method-note">
        <span className="method-icon">i</span>
        <p>
          <strong>Research prototype, not an operational warning.</strong> This dashboard computes flare probabilities using a validation-selected logistic regression baseline. Live feeds represent operational SWPC telemetry, which may differ from the science-quality archive used during model training and evaluation.
        </p>
        <a href={`${API}/docs`} target="_blank" rel="noreferrer">
          API docs <span>↗</span>
        </a>
      </section>

      <footer>
        <span>SolarGuard AI <i>·</i> GOES-18 Primary XRS-B <i>·</i> UTC</span>
        <span>Forecasts are experimental research prototypes. Not for operational warning use.</span>
      </footer>
    </main>
  )
}
