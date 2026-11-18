import { useCallback, useEffect, useMemo, useState } from 'react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function shortDate(value, withTime = false) {
  if (!value) return '—'
  const date = new Date(value)
  return new Intl.DateTimeFormat('en', withTime
    ? { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', timeZoneName: 'short' }
    : { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date)
}

function FluxChart({ samples }) {
  const points = useMemo(() => samples.map(row => ({
    date: new Date(row.sample_time_utc),
    value: Number(row.xrsb_flux_mean_w_m2),
  })).filter(p => p.value > 0 && Number.isFinite(p.value)), [samples])

  if (points.length < 2) return <div className="chart-empty">Not enough valid X-ray observations to draw this view.</div>

  const width = 820, height = 270, left = 66, right = 18, top = 16, bottom = 38
  const logs = points.map(p => Math.log10(p.value))
  const min = Math.floor(Math.min(...logs)), max = Math.ceil(Math.max(...logs))
  const yMin = Math.min(min, -9), yMax = Math.max(max, -5)
  const x = i => left + i * (width - left - right) / (points.length - 1)
  const y = v => top + (yMax - Math.log10(v)) * (height - top - bottom) / (yMax - yMin)
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`).join(' ')
  const ticks = Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i)
  const labels = [0, Math.floor((points.length - 1) / 2), points.length - 1]

  return <div className="chart-wrap">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="GOES XRS-B mean flux during the latest archived week">
      <defs><linearGradient id="flux-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#f6a64b" stopOpacity=".2"/><stop offset="100%" stopColor="#f6a64b" stopOpacity="0"/></linearGradient></defs>
      {ticks.map(tick => <g key={tick}>
        <line x1={left} x2={width - right} y1={top + (yMax - tick) * (height - top - bottom) / (yMax - yMin)} y2={top + (yMax - tick) * (height - top - bottom) / (yMax - yMin)} className="grid-line" />
        <text x={left - 10} y={top + (yMax - tick) * (height - top - bottom) / (yMax - yMin) + 4} textAnchor="end" className="axis-label">1e{tick}</text>
      </g>)}
      <path d={`${path} L ${x(points.length - 1)} ${height - bottom} L ${x(0)} ${height - bottom} Z`} fill="url(#flux-fill)" />
      <path d={path} fill="none" className="flux-line" />
      {labels.map(i => <text key={i} x={x(i)} y={height - 10} textAnchor={i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle'} className="axis-label">{shortDate(points[i].date)}</text>)}
    </svg>
    <div className="chart-legend"><span className="legend-dot"/>XRS-B mean · 1–8 Å</div>
  </div>
}

function Metric({ label, value, detail }) {
  return <div className="metric"><span className="metric-label">{label}</span><strong>{value}</strong><span className="metric-detail">{detail}</span></div>
}

export default function App() {
  const [state, setState] = useState({ loading: true, error: '', health: null, history: [], forecast: null, evaluation: null })
  const refresh = useCallback(async () => {
    setState(current => ({ ...current, loading: true, error: '' }))
    try {
      const [healthRes, historyRes, forecastRes, evalRes] = await Promise.all([
        fetch(`${API}/health`), fetch(`${API}/history?hours=168`), fetch(`${API}/predict/latest`), fetch(`${API}/evaluation`),
      ])
      const responses = [healthRes, historyRes, forecastRes, evalRes]
      const bad = responses.find(response => !response.ok)
      if (bad) throw new Error(`API request failed (${bad.status}). Check that the service and its data artifacts are available.`)
      const [health, history, forecast, evaluation] = await Promise.all(responses.map(response => response.json()))
      setState({ loading: false, error: '', health, history: history.samples || [], forecast, evaluation })
    } catch (error) {
      setState(current => ({ ...current, loading: false, error: error.message || 'Could not reach the SolarGuard API.' }))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])
  const metrics = state.evaluation?.test_metrics?.[state.evaluation?.selected_model]
  const latest = state.history.at(-1)
  const probability = state.forecast?.m_plus_probability
  const riskClass = (state.forecast?.risk_level || 'moderate').toLowerCase()
  const updated = latest?.sample_time_utc

  return <main className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="SolarGuard AI home"><span className="brand-mark"><span/></span><span>SolarGuard<span className="brand-ai"> AI</span></span></a>
      <div className="topbar-right"><span className="archive-pill"><span className="pulse-dot"/>HISTORICAL ARCHIVE</span><button className="refresh-button" onClick={refresh} disabled={state.loading} aria-label="Refresh dashboard"><span className={state.loading ? 'refresh-icon spinning' : 'refresh-icon'}>↻</span> Refresh</button></div>
    </header>

    <section className="page-heading" id="top">
      <div><div className="eyebrow">SPACE WEATHER · EXPERIMENTAL FORECAST</div><h1>Solar activity outlook</h1><p>Probability of an M-class or stronger flare in the next 24 hours.</p></div>
      <div className="archive-note"><span className="note-icon">◷</span><span>Archive snapshot<br/><strong>{shortDate(updated, true)}</strong></span></div>
    </section>

    {state.error && <section className="error-banner"><span>!</span><div><strong>Dashboard data unavailable</strong><p>{state.error} Start the API from the project root, then refresh.</p></div></section>}

    <section className="overview-grid">
      <article className="forecast-card">
        <div className="forecast-card-top"><span className="eyebrow">NEXT 24 HOURS</span><span className={`risk-badge ${riskClass}`}>{state.forecast?.risk_level || '—'} risk</span></div>
        <div className="probability-row"><strong>{probability == null ? '—' : `${(probability * 100).toFixed(1)}%`}</strong><span className="probability-caption">M+ flare<br/>probability</span></div>
        <div className="forecast-track"><span style={{ width: `${Math.max(0, Math.min(100, (probability || 0) * 100))}%` }}/></div>
        <div className="forecast-foot"><span>Forecast window</span><strong>{state.forecast ? `${shortDate(state.forecast.forecast_window_utc.start_exclusive, true)} → ${shortDate(state.forecast.forecast_window_utc.end_inclusive, true)}` : '—'}</strong></div>
        <div className="model-decision"><span className={state.forecast?.predict_m_plus ? 'decision-dot elevated' : 'decision-dot'}/>{state.forecast ? (state.forecast.predict_m_plus ? 'Above model decision threshold' : 'Below model decision threshold') : 'Waiting for forecast'}</div>
      </article>

      <article className="chart-card">
        <div className="card-heading"><div><div className="eyebrow">GOES X-RAY FLUX</div><h2>Recent solar activity</h2></div><span className="period-tag">LATEST 7 DAYS</span></div>
        <FluxChart samples={state.history}/>
        <div className="chart-foot"><span>Mean XRS-B flux in W/m² · logarithmic scale</span><span>Source: NOAA GOES</span></div>
      </article>
    </section>

    <section className="stats-grid" aria-label="Current activity and model performance">
      <article className="stat-card activity-stat"><div className="eyebrow">LATEST COMPLETED HOUR</div><div className="activity-value">{latest?.xrsb_flux_mean_w_m2 ? Number(latest.xrsb_flux_mean_w_m2).toExponential(2) : '—'}<small>W/m²</small></div><div className="stat-sub">XRS-B mean · 1–8 Å</div></article>
      <article className="stat-card"><div className="eyebrow">HELD-OUT TEST · {state.evaluation?.selected_model?.replaceAll('_', ' ') || 'MODEL'}</div><div className="metric-pair"><Metric label="BSS" value={metrics ? metrics.brier_skill_score_vs_training_climatology.toFixed(3) : '—'} detail="vs training climatology"/><Metric label="TSS" value={metrics ? metrics.tss.toFixed(3) : '—'} detail="at validation threshold"/></div></article>
      <article className="stat-card imbalance-stat"><div className="eyebrow">LABEL PREVALENCE</div><div className="activity-value">{state.evaluation?.test_metrics?.[state.evaluation?.selected_model]?.positive_rate != null ? `${(state.evaluation.test_metrics[state.evaluation.selected_model].positive_rate * 100).toFixed(1)}%` : '—'}</div><div className="stat-sub">Positive forecast windows in test period</div></article>
    </section>

    <section className="method-note"><span className="method-icon">i</span><p><strong>Research prototype, not an operational warning.</strong> This dashboard uses the latest archived sample, which may lag current conditions. Test performance is measured on a chronological holdout; the test period’s M+ prevalence differs from training.</p><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API details <span>↗</span></a></section>
    <footer><span>SolarGuard AI <i>·</i> GOES XRS-B <i>·</i> UTC</span><span>Forecasts are experimental and for research use.</span></footer>
  </main>
}
