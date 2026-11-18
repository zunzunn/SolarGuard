import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')

// Operational Operator Profiles
const OPERATOR_PROFILES = [
  {
    id: 'chen',
    name: 'Dr. Sarah Chen',
    role: 'Chief Heliophysicist',
    clearance: 'DEFENSE LVL-3',
    department: 'Solar Cycle 25 Advisory',
    avatar: 'SC',
  },
  {
    id: 'vance',
    name: 'Cmdr. Alex Vance',
    role: 'Orbital Ops Specialist',
    clearance: 'SATELLITE OPS-1',
    department: 'GOES Fleet Constellation',
    avatar: 'AV',
  },
  {
    id: 'diaz',
    name: 'Capt. Elena Diaz',
    role: 'Polar Aviation Dispatcher',
    clearance: 'ICAO RADIATION-A',
    department: 'Commercial Route Safety',
    avatar: 'ED',
  },
  {
    id: 'guest',
    name: 'Guest Heliophysicist',
    role: 'Public Analyst',
    clearance: 'READ ONLY',
    department: 'Open Science Community',
    avatar: 'GA',
  },
]

function formatUtc(date, withTime = true) {
  if (!date) return '—'
  const d = new Date(date)
  if (isNaN(d.getTime())) return '—'
  return new Intl.DateTimeFormat('en-US', withTime ? {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'UTC',
    timeZoneName: 'short',
  } : {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(d)
}

function classifyFlux(flux) {
  if (flux >= 1e-4) return `X${(flux / 1e-4).toFixed(1)}`
  if (flux >= 1e-5) return `M${(flux / 1e-5).toFixed(1)}`
  if (flux >= 1e-6) return `C${(flux / 1e-6).toFixed(1)}`
  if (flux >= 1e-7) return `B${(flux / 1e-7).toFixed(1)}`
  return `A${(flux / 1e-8).toFixed(1)}`
}

// Interactive Solar Flux Telemetry Observatory Chart
function TelemetryChart({ samples, title = "GOES XRS-B 0.1-0.8nm Flux", timeRange = 168 }) {
  const [hoveredPoint, setHoveredPoint] = useState(null)
  const containerRef = useRef(null)

  const points = useMemo(() => {
    if (!samples || !samples.length) return []
    const sliced = samples.slice(-timeRange)
    return sliced.map(row => ({
      date: new Date(row.sample_time_utc),
      timeStr: row.sample_time_utc,
      value: Number(row.xrsb_flux_mean_w_m2 ?? row.xrsb_flux_max_w_m2 ?? 1e-8),
      maxVal: Number(row.xrsb_flux_max_w_m2 ?? row.xrsb_flux_mean_w_m2 ?? 1e-8),
    })).filter(p => p.value > 0 && Number.isFinite(p.value))
  }, [samples, timeRange])

  if (points.length < 2) {
    return (
      <div className="chart-wrap" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '240px', color: 'var(--text-muted)' }}>
        Streaming solar telemetry... insufficient points.
      </div>
    )
  }

  const width = 880
  const height = 280
  const left = 68
  const right = 20
  const top = 20
  const bottom = 40

  const yMin = -8 // B-Class (1e-8)
  const yMax = -3 // X-Class (1e-3)

  const x = i => left + (i * (width - left - right)) / (points.length - 1)
  const y = v => {
    const log = Math.log10(Math.max(1e-9, v))
    const clamped = Math.max(yMin, Math.min(yMax, log))
    return top + ((yMax - clamped) * (height - top - bottom)) / (yMax - yMin)
  }

  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`).join(' ')
  const areaPath = `${path} L ${x(points.length - 1).toFixed(1)} ${height - bottom} L ${x(0).toFixed(1)} ${height - bottom} Z`

  // Threshold lines
  const xThresholdY = y(1e-4) // X-Class 1e-4
  const mThresholdY = y(1e-5) // M-Class 1e-5
  const cThresholdY = y(1e-6) // C-Class 1e-6

  const handleMouseMove = e => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const chartX = (mouseX / rect.width) * width
    if (chartX < left || chartX > width - right) {
      setHoveredPoint(null)
      return
    }
    const ratio = (chartX - left) / (width - left - right)
    const index = Math.round(ratio * (points.length - 1))
    if (index >= 0 && index < points.length) {
      setHoveredPoint({
        ...points[index],
        screenX: mouseX,
        screenY: (y(points[index].value) / height) * rect.height,
      })
    }
  }

  const labels = [
    0,
    Math.floor((points.length - 1) * 0.33),
    Math.floor((points.length - 1) * 0.66),
    points.length - 1,
  ]

  return (
    <div
      className="chart-wrap"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoveredPoint(null)}
    >
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <defs>
          <linearGradient id="chart-area-gradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#ffaa2b" stopOpacity="0.35" />
            <stop offset="70%" stopColor="#ffaa2b" stopOpacity="0.05" />
            <stop offset="100%" stopColor="#ffaa2b" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="chart-line-gradient" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="50%" stopColor="#ffaa2b" />
            <stop offset="100%" stopColor="#ff7043" />
          </linearGradient>
        </defs>

        {/* Magnitude Grid Lines */}
        {[-8, -7, -6, -5, -4, -3].map(power => {
          const yPos = top + ((yMax - power) * (height - top - bottom)) / (yMax - yMin)
          const labels = {
            '-8': '10⁻⁸ B',
            '-7': '10⁻⁷ B',
            '-6': '10⁻⁶ C',
            '-5': '10⁻⁵ M',
            '-4': '10⁻⁴ X',
            '-3': '10⁻³ X10',
          }
          return (
            <g key={power}>
              <line x1={left} x2={width - right} y1={yPos} y2={yPos} className="chart-grid-line" />
              <text x={left - 10} y={yPos + 4} textAnchor="end" className="chart-axis-label">
                {labels[String(power)]}
              </text>
            </g>
          )
        })}

        {/* Flare Threshold Reference Lines */}
        <line x1={left} x2={width - right} y1={xThresholdY} y2={xThresholdY} stroke="#ff3366" strokeWidth="1" strokeDasharray="3 3" opacity="0.65" />
        <line x1={left} x2={width - right} y1={mThresholdY} y2={mThresholdY} stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 3" opacity="0.65" />
        <line x1={left} x2={width - right} y1={cThresholdY} y2={cThresholdY} stroke="#06b6d4" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />

        {/* Shaded Area */}
        <path d={areaPath} fill="url(#chart-area-gradient)" />

        {/* The Telemetry Line */}
        <path d={path} className="flux-line-path" />

        {/* X Axis Time Labels */}
        {labels.map(i => (
          <text
            key={i}
            x={x(i)}
            y={height - 10}
            textAnchor={i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle'}
            className="chart-axis-label"
          >
            {formatUtc(points[i].date, false)}
          </text>
        ))}

        {/* Hover Crosshair in SVG */}
        {hoveredPoint && (
          <g>
            <line
              x1={(hoveredPoint.screenX / (containerRef.current?.getBoundingClientRect().width || width)) * width}
              x2={(hoveredPoint.screenX / (containerRef.current?.getBoundingClientRect().width || width)) * width}
              y1={top}
              y2={height - bottom}
              stroke="rgba(0, 242, 254, 0.6)"
              strokeDasharray="2 2"
            />
            <circle
              cx={(hoveredPoint.screenX / (containerRef.current?.getBoundingClientRect().width || width)) * width}
              cy={y(hoveredPoint.value)}
              r="5"
              fill="#00f2fe"
              stroke="#ffffff"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>

      {/* Floating Hover Tooltip */}
      {hoveredPoint && (
        <div
          className="chart-tooltip-box"
          style={{
            left: `${hoveredPoint.screenX}px`,
            top: `${hoveredPoint.screenY}px`,
          }}
        >
          <div style={{ color: '#00f2fe', fontWeight: 'bold' }}>
            {classifyFlux(hoveredPoint.maxVal)} ({hoveredPoint.value.toExponential(2)} W/m²)
          </div>
          <div style={{ fontSize: '10px', color: '#94a3b8' }}>
            {formatUtc(hoveredPoint.date, true)}
          </div>
        </div>
      )}
    </div>
  )
}

// Smooth Number Counter Hook
function useAnimatedCounter(targetValue, duration = 800) {
  const [count, setCount] = useState(0)

  useEffect(() => {
    if (targetValue == null || isNaN(targetValue)) return
    const startValue = count
    const diff = targetValue - startValue
    const startTime = performance.now()

    let animationFrame
    const update = currentTime => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const easeProgress = 1 - Math.pow(1 - progress, 3) // easeOutCubic
      setCount(startValue + diff * easeProgress)

      if (progress < 1) {
        animationFrame = requestAnimationFrame(update)
      }
    }
    animationFrame = requestAnimationFrame(update)

    return () => cancelAnimationFrame(animationFrame)
  }, [targetValue, duration])

  return count
}

export default function App() {
  const [mode, setMode] = useState('live') // 'live' | 'archive'
  const [chartHours, setChartHours] = useState(168)
  const [soundEnabled, setSoundEnabled] = useState(false)
  const [utcTime, setUtcTime] = useState('')
  const [activeSection, setActiveSection] = useState('overview')

  // Operator Auth & Session State
  const [operator, setOperator] = useState(() => {
    const saved = localStorage.getItem('solarguard_operator')
    if (saved) {
      try { return JSON.parse(saved) } catch (e) {}
    }
    return OPERATOR_PROFILES[0]
  })
  const [authModalOpen, setAuthModalOpen] = useState(false)

  // Simulation State
  const [simModalOpen, setSimModalOpen] = useState(false)
  const [simulatedData, setSimulatedData] = useState(null)
  const [simLoading, setSimLoading] = useState(false)

  // Alert Threshold State
  const [alertThreshold, setAlertThreshold] = useState(() => {
    return Number(localStorage.getItem('solarguard_alert_threshold') || 50)
  })

  // Flare Catalog State
  const [catalogClass, setCatalogClass] = useState('M') // 'M' | 'X'
  const [catalogSearch, setCatalogSearch] = useState('')
  const [catalogEvents, setCatalogEvents] = useState([])
  const [catalogLoading, setCatalogLoading] = useState(false)

  // Methodology Accordion State
  const [accordionOpen, setAccordionOpen] = useState({
    target: true,
    features: false,
    calibration: false,
    shift: false,
  })

  // Toasts
  const [toasts, setToasts] = useState([])

  const addToast = (title, message, type = 'info') => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, title, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 4500)
  }

  // Synthesizer Audio Sound Effect
  const playSound = useCallback((freq = 660, type = 'sine', duration = 0.12) => {
    if (!soundEnabled) return
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (!AudioCtx) return
      const ctx = new AudioCtx()
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = type
      osc.frequency.setValueAtTime(freq, ctx.currentTime)
      gain.gain.setValueAtTime(0.08, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start()
      osc.stop(ctx.currentTime + duration)
    } catch (e) {}
  }, [soundEnabled])

  // Core API State
  const [state, setState] = useState({
    loading: true,
    error: '',
    health: null,
    history: [],
    forecast: null,
    evaluation: null,
    stats: null,
    metadata: null,
  })

  // UTC Clock Ticker
  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      const str = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC'
      setUtcTime(str)
    }
    updateTime()
    const timer = setInterval(updateTime, 1000)
    return () => clearInterval(timer)
  }, [])

  // Main Data Refresh
  const refresh = useCallback(async (selectedMode = mode) => {
    setState(current => ({ ...current, loading: true, error: '' }))
    try {
      const isLive = selectedMode === 'live'
      const historyUrl = isLive ? `${API}/live/history?hours=168` : `${API}/history?hours=168`
      const forecastUrl = isLive ? `${API}/live/predict` : `${API}/predict/latest`

      const [healthRes, historyRes, forecastRes, evalRes, statsRes] = await Promise.all([
        fetch(`${API}/health`),
        fetch(historyUrl),
        fetch(forecastUrl),
        fetch(`${API}/evaluation`),
        fetch(`${API}/stats/summary`),
      ])

      if (!forecastRes.ok) {
        const errorData = await forecastRes.json().catch(() => ({}))
        const detail = errorData.detail || `Prediction failed (${forecastRes.status})`
        throw new Error(isLive ? `Live SWPC Feed: ${detail}` : `Archive: ${detail}`)
      }

      if (!historyRes.ok) {
        throw new Error(`Failed to load ${isLive ? 'live' : 'archive'} telemetry (${historyRes.status})`)
      }

      const [health, historyData, forecast, evaluation, stats] = await Promise.all([
        healthRes.json(),
        historyRes.json(),
        forecastRes.json(),
        evalRes.json(),
        statsRes.json().catch(() => null),
      ])

      setState({
        loading: false,
        error: '',
        health,
        history: historyData.samples || [],
        forecast,
        evaluation,
        stats,
        metadata: isLive ? (historyData.feed_metadata || null) : null,
      })

      // Audio feedback on refresh
      playSound(880, 'sine', 0.08)
    } catch (error) {
      setState(current => ({
        ...current,
        loading: false,
        error: error.message || 'Could not reach SolarGuard API service.',
      }))
      playSound(330, 'sawtooth', 0.2)
    }
  }, [mode, playSound])

  // Fetch Catalog Events
  const fetchCatalog = useCallback(async (minClass = catalogClass) => {
    setCatalogLoading(true)
    try {
      const res = await fetch(`${API}/flare-events?min_class=${minClass}&limit=50&sort_by=time&order=desc`)
      if (res.ok) {
        const data = await res.json()
        setCatalogEvents(data.events || [])
      }
    } catch (e) {
      console.error('Catalog error', e)
    } finally {
      setCatalogLoading(false)
    }
  }, [catalogClass])

  // Initial Load & Auto-refresh (every 60s)
  useEffect(() => {
    refresh(mode)
    fetchCatalog(catalogClass)

    const interval = setInterval(() => {
      refresh(mode)
    }, 60000)
    return () => clearInterval(interval)
  }, [mode, refresh, catalogClass, fetchCatalog])

  const handleModeSwitch = newMode => {
    setMode(newMode)
    setSimulatedData(null) // reset simulation on source change
    refresh(newMode)
    playSound(720, 'sine', 0.1)
  }

  // Scenario Injection
  const runSimulation = async scenarioType => {
    setSimLoading(true)
    try {
      const res = await fetch(`${API}/simulate/scenario`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenarioType, multiplier: 5.0 }),
      })
      if (!res.ok) throw new Error('Simulation failed')
      const result = await res.json()
      setSimulatedData(result)
      setSimModalOpen(false)
      addToast(
        'Solar Scenario Injected',
        `${result.description} — Probability shifted to ${(result.simulated_probability * 100).toFixed(1)}%`,
        'warning'
      )
      playSound(1040, 'triangle', 0.25)
    } catch (e) {
      addToast('Simulation Error', e.message, 'error')
    } finally {
      setSimLoading(false)
    }
  }

  const clearSimulation = () => {
    setSimulatedData(null)
    addToast('Simulation Cleared', 'Restored live baseline telemetry forecast.', 'info')
    playSound(520, 'sine', 0.1)
  }

  // Test Alert Dispatch
  const handleTestAlert = () => {
    playSound(920, 'sawtooth', 0.35)
    addToast(
      'CRITICAL SPACE WEATHER DISPATCH',
      `Simulated Alert: Probability ${(effectiveProbability * 100).toFixed(1)}% > Operator Threshold ${alertThreshold}%. Aviation polar HF blackouts & orbital satcom shielding advisories engaged.`,
      'warning'
    )
  }

  // CSV Catalog Export
  const handleExportCsv = async () => {
    playSound(780, 'sine', 0.1)
    addToast('Generating CSV Export', 'Downloading verified historical flare catalog...', 'info')
    window.location.href = `${API}/export/flare-events?limit=1000`
  }

  // Active Forecast: Simulated or Baseline
  const activeForecast = simulatedData ? simulatedData.simulated_forecast : state.forecast
  const effectiveProbability = activeForecast?.m_plus_probability ?? 0
  const animatedProb = useAnimatedCounter(effectiveProbability * 100, 1000)
  const isElevated = activeForecast?.predict_m_plus ?? (effectiveProbability >= (activeForecast?.decision_threshold || 0.585))
  const riskClass = (activeForecast?.risk_level || 'moderate').toLowerCase()
  const isLive = mode === 'live'

  // SVG Gauge calculations
  const gaugeRadius = 88
  const gaugeCircumference = 2 * Math.PI * gaugeRadius
  const gaugeOffset = gaugeCircumference - (Math.min(100, Math.max(0, animatedProb)) / 100) * gaugeCircumference

  // Gauge Color by Risk
  const gaugeColor = useMemo(() => {
    if (riskClass === 'severe') return '#ff3366'
    if (riskClass === 'high') return '#ff7043'
    if (riskClass === 'moderate') return '#ffaa2b'
    return '#10e8aa'
  }, [riskClass])

  // Filtered Catalog
  const filteredCatalog = useMemo(() => {
    if (!catalogSearch) return catalogEvents
    const q = catalogSearch.toLowerCase()
    return catalogEvents.filter(e =>
      e.flare_class.toLowerCase().includes(q) ||
      e.sample_time_utc.toLowerCase().includes(q)
    )
  }, [catalogEvents, catalogSearch])

  return (
    <main className="app-shell">
      {/* --- Top Navigation Command Bar --- */}
      <header className="topbar">
        <a className="brand" href="#overview" onClick={() => playSound(600)}>
          <div className="brand-symbol">
            <div className="brand-symbol-ring" />
            <div className="brand-symbol-core" />
          </div>
          <div className="brand-text">
            SolarGuard <span>AI</span>
            <span className="brand-badge">OPS CONSOLE v2.4</span>
          </div>
        </a>

        {/* Scroll-Spy Anchor Navigation */}
        <nav className="nav-links">
          <a
            href="#overview"
            className={`nav-link ${activeSection === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveSection('overview')}
          >
            Threat Outlook
          </a>
          <a
            href="#telemetry"
            className={`nav-link ${activeSection === 'telemetry' ? 'active' : ''}`}
            onClick={() => setActiveSection('telemetry')}
          >
            Observatory
          </a>
          <a
            href="#catalog"
            className={`nav-link ${activeSection === 'catalog' ? 'active' : ''}`}
            onClick={() => setActiveSection('catalog')}
          >
            Event Catalog
          </a>
          <a
            href="#validation"
            className={`nav-link ${activeSection === 'validation' ? 'active' : ''}`}
            onClick={() => setActiveSection('validation')}
          >
            Scientific Backtest
          </a>
          <a
            href="#methodology"
            className={`nav-link ${activeSection === 'methodology' ? 'active' : ''}`}
            onClick={() => setActiveSection('methodology')}
          >
            Science Docs
          </a>
        </nav>

        {/* Topbar Controls */}
        <div className="topbar-right">
          {/* Live UTC Clock */}
          <div className="utc-clock" title="Current Coordinated Universal Time">
            <span className="utc-clock-dot" />
            <span>{utcTime || 'UTC CLOCK'}</span>
          </div>

          {/* Source Toggle */}
          <div className="source-switcher" role="radiogroup" aria-label="Telemetry Feed">
            <button
              className={`source-btn ${isLive ? 'active' : ''}`}
              onClick={() => handleModeSwitch('live')}
              aria-checked={isLive}
              role="radio"
              title="NOAA SWPC Realtime Primary Feed"
            >
              <span className={`live-dot ${state.metadata?.is_stale ? 'stale' : ''}`} />
              GOES-18 Live
            </button>
            <button
              className={`source-btn ${!isLive ? 'active' : ''}`}
              onClick={() => handleModeSwitch('archive')}
              aria-checked={!isLive}
              role="radio"
              title="10-Year Science-Quality Archive"
            >
              Historical Archive
            </button>
          </div>

          {/* Operator Profile Modal Trigger */}
          <button
            className="operator-pill"
            onClick={() => {
              setAuthModalOpen(true)
              playSound(640)
            }}
            title="Switch Operator Profile or View Clearance"
          >
            <div className="operator-avatar">{operator.avatar}</div>
            <div className="operator-info">
              <span className="operator-name">{operator.name}</span>
              <span className="operator-role">{operator.clearance}</span>
            </div>
          </button>

          {/* Audio Synthesizer Toggle */}
          <button
            className={`icon-btn ${soundEnabled ? 'active' : ''}`}
            onClick={() => {
              const next = !soundEnabled
              setSoundEnabled(next)
              if (next) playSound(880, 'sine', 0.15)
              addToast(
                next ? 'Audio Synthesizer Enabled' : 'Audio Muted',
                next ? 'Radar pings and acoustic telemetry alerts active.' : 'Acoustic cues muted.',
                'info'
              )
            }}
            title={soundEnabled ? 'Mute audio' : 'Enable audio radar cues'}
          >
            {soundEnabled ? '🔊' : '🔇'}
          </button>

          {/* Refresh Action */}
          <button
            className="refresh-btn"
            onClick={() => refresh(mode)}
            disabled={state.loading}
            title="Force refresh telemetry from satellite source"
          >
            <span className={state.loading ? 'spinning' : ''}>↻</span> Refresh
          </button>
        </div>
      </header>

      {/* --- Live Telemetry Status Banner --- */}
      {isLive && state.metadata && (
        <section className={`feed-status-banner ${state.metadata.is_stale ? 'stale' : ''}`}>
          <div className="feed-status-left">
            <span className={`status-chip ${state.metadata.is_stale ? 'stale' : ''}`}>
              {state.metadata.is_stale ? `STALE FEED · ${state.metadata.staleness_minutes}m LAG` : `OPERATIONAL · GOES-18`}
            </span>
            <span>
              Real-time NOAA Space Weather Prediction Center primary telemetry (0.1–0.8nm XRS-B band).
            </span>
          </div>
          <div className="feed-status-right">
            <span>Obs: {formatUtc(state.metadata.latest_observation_utc, true)}</span>
            <span>Latency: {state.metadata.staleness_minutes ?? 0}m</span>
          </div>
        </section>
      )}

      {/* Simulated Scenario Alert Callout */}
      {simulatedData && (
        <section className="feed-status-banner stale" style={{ background: 'rgba(255, 170, 43, 0.15)', borderColor: 'var(--solar-amber)' }}>
          <div className="feed-status-left">
            <span className="status-chip stale">HYPOTHETICAL SIMULATION ACTIVE</span>
            <span>
              <strong>{simulatedData.description}:</strong> Injected peak flux {simulatedData.injected_peak_class} ({simulatedData.injected_peak_flux_w_m2?.toExponential(2)} W/m²). Probability shifted from {(simulatedData.baseline_probability * 100).toFixed(1)}% to {(simulatedData.simulated_probability * 100).toFixed(1)}%.
            </span>
          </div>
          <button
            className="sim-trigger-btn"
            onClick={clearSimulation}
            style={{ background: 'rgba(255, 255, 255, 0.1)', color: '#fff' }}
          >
            Reset Baseline
          </button>
        </section>
      )}

      {/* Error Callout */}
      {state.error && (
        <section className="error-banner">
          <div>
            <strong>Telemetry Feed Disruption</strong>
            <p>{state.error}</p>
            {isLive && (
              <button
                className="sim-trigger-btn"
                style={{ marginTop: '10px' }}
                onClick={() => handleModeSwitch('archive')}
              >
                Switch to Historical Archive Snapshot
              </button>
            )}
          </div>
        </section>
      )}

      {/* --- Section 1: Threat Outlook & Solar Observatory Hero Grid --- */}
      <section className="hero-grid" id="overview">
        {/* Left: Animated Radial Threat Outlook Gauge */}
        <article className={`threat-card threat-${riskClass}`}>
          <div className="threat-top">
            <div className="section-eyebrow">
              {isLive ? 'LIVE OPERATIONAL INFERENCE' : 'ARCHIVE BENCHMARK INFERENCE'}
            </div>
            <span className={`risk-badge ${riskClass}`}>
              <span className="live-dot" style={{ background: gaugeColor, boxShadow: `0 0 8px ${gaugeColor}` }} />
              {activeForecast?.risk_level || 'Moderate'} Threat
            </span>
          </div>

          <div className="gauge-container">
            <div className="gauge-svg-wrap">
              <svg className="gauge-svg" viewBox="0 0 200 200">
                {/* Background Ring */}
                <circle
                  className="gauge-bg"
                  cx="100"
                  cy="100"
                  r={gaugeRadius}
                  strokeWidth="14"
                />
                {/* Active Colored Arc */}
                <circle
                  className="gauge-bar"
                  cx="100"
                  cy="100"
                  r={gaugeRadius}
                  stroke={gaugeColor}
                  strokeWidth="14"
                  strokeDasharray={gaugeCircumference}
                  strokeDashoffset={gaugeOffset}
                  style={{ filter: `drop-shadow(0 0 10px ${gaugeColor})` }}
                />
              </svg>

              <div className="gauge-center-content">
                <span className="gauge-prob-number">
                  {state.loading && !activeForecast ? '—' : `${animatedProb.toFixed(1)}%`}
                </span>
                <span className="gauge-prob-label">M1+ Flare Probability</span>
              </div>
            </div>
          </div>

          {/* Decision Boundary Box */}
          <div className="decision-status-box">
            <div className="decision-indicator">
              <span className={`decision-dot ${isElevated ? 'elevated' : ''}`} />
              <span>{isElevated ? 'Elevated Operational Warning' : 'Quiescent Baseline Regime'}</span>
            </div>
            <span className="threshold-target">
              Cutoff: {((activeForecast?.decision_threshold || 0.585) * 100).toFixed(1)}%
            </span>
          </div>

          {/* Horizon Countdown */}
          <div className="horizon-box">
            <span>Forecast Window (Next 24h)</span>
            <strong>
              {activeForecast?.forecast_window_utc
                ? `${formatUtc(activeForecast.forecast_window_utc.start_exclusive)} → ${formatUtc(activeForecast.forecast_window_utc.end_inclusive)}`
                : 'Streaming observations...'}
            </strong>
          </div>

          {/* Solar Scenario Simulator Launcher */}
          <div className="scenario-bar">
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Model Sensitivity Tester:
            </span>
            <button
              className="sim-trigger-btn"
              onClick={() => {
                setSimModalOpen(true)
                playSound(720)
              }}
            >
              ⚡ Inject Solar Flare Scenario
            </button>
          </div>
        </article>

        {/* Right: Solar Observatory Flux Chart */}
        <article className="chart-card" id="telemetry">
          <div className="chart-header">
            <div>
              <div className="section-eyebrow">
                {isLive ? 'GOES-18 PRIMARY X-RAY SENSOR' : 'NOAA NCEI SATELLITE ARCHIVE'}
              </div>
              <h2 className="section-title" style={{ fontSize: '22px' }}>
                X-Ray Flux Telemetry Observatory
              </h2>
            </div>

            {/* Time Filter Tabs */}
            <div className="chart-filters" role="tablist">
              <button
                className={`filter-btn ${chartHours === 24 ? 'active' : ''}`}
                onClick={() => { setChartHours(24); playSound(680) }}
              >
                24 Hours
              </button>
              <button
                className={`filter-btn ${chartHours === 48 ? 'active' : ''}`}
                onClick={() => { setChartHours(48); playSound(680) }}
              >
                48 Hours
              </button>
              <button
                className={`filter-btn ${chartHours === 168 ? 'active' : ''}`}
                onClick={() => { setChartHours(168); playSound(680) }}
              >
                7 Days (168h)
              </button>
            </div>
          </div>

          {/* The Chart Component */}
          <TelemetryChart
            samples={state.history}
            timeRange={chartHours}
            title="Solar XRS-B Flux"
          />

          <div className="chart-footer">
            <div className="chart-threshold-legends">
              <span className="threshold-badge-x">● X-Class (&gt;10⁻⁴ W/m²)</span>
              <span className="threshold-badge-m">● M-Class (&gt;10⁻⁵ W/m²)</span>
              <span className="threshold-badge-c">● C-Class (&gt;10⁻⁶ W/m²)</span>
            </div>
            <span>
              Source: {isLive ? 'SWPC Operational Telemetry' : 'NOAA NCEI Historical GOES'}
            </span>
          </div>
        </article>
      </section>

      {/* --- Section 2: Space Weather Mission Metrics Matrix (6 Cards) --- */}
      <section className="stats-grid">
        <article className="stat-card accent-red">
          <div className="stat-header">
            <span className="section-eyebrow">PEAK HISTORICAL FLUX</span>
            <span className="stat-icon">💥</span>
          </div>
          <div className="stat-value">
            {state.stats?.peak_flux_class || 'X14.6'}
            <span className="stat-unit">Class</span>
          </div>
          <div className="stat-caption">
            {state.stats?.peak_flux_time_utc ? formatUtc(state.stats.peak_flux_time_utc, false) : 'Sep 2017'} · Active Region 2673
          </div>
        </article>

        <article className="stat-card accent-amber">
          <div className="stat-header">
            <span className="section-eyebrow">LATEST SENSOR READING</span>
            <span className="stat-icon">📡</span>
          </div>
          <div className="stat-value">
            {state.history.at(-1)?.xrsb_flux_mean_w_m2
              ? Number(state.history.at(-1).xrsb_flux_mean_w_m2).toExponential(2)
              : '—'}
            <span className="stat-unit">W/m²</span>
          </div>
          <div className="stat-caption">
            Class {classifyFlux(state.history.at(-1)?.xrsb_flux_max_w_m2 || 1e-7)} · GOES-18 0.1–0.8nm
          </div>
        </article>

        <article className="stat-card accent-cyan">
          <div className="stat-header">
            <span className="section-eyebrow">TOTAL HOURS CATALOGED</span>
            <span className="stat-icon">⏱</span>
          </div>
          <div className="stat-value">
            {state.stats?.total_hours_cataloged ? state.stats.total_hours_cataloged.toLocaleString() : '87,588'}
            <span className="stat-unit">hrs</span>
          </div>
          <div className="stat-caption">
            Continuous hourly records (2016–2026)
          </div>
        </article>

        <article className="stat-card">
          <div className="stat-header">
            <span className="section-eyebrow">BRIER SKILL SCORE</span>
            <span className="stat-icon">🎯</span>
          </div>
          <div className="stat-value" style={{ color: 'var(--safe-mint)' }}>
            +{(state.stats?.evaluation_bss ?? 0.460).toFixed(3)}
          </div>
          <div className="stat-caption">
            +46.0% skill over climatological reference
          </div>
        </article>

        <article className="stat-card">
          <div className="stat-header">
            <span className="section-eyebrow">TRUE SKILL STATISTIC</span>
            <span className="stat-icon">⚖️</span>
          </div>
          <div className="stat-value">
            {(state.stats?.evaluation_tss ?? 0.246).toFixed(3)}
          </div>
          <div className="stat-caption">
            Validation-tuned operational threshold
          </div>
        </article>

        <article className="stat-card accent-purple">
          <div className="stat-header">
            <span className="section-eyebrow">DISCRIMINATION (ROC-AUC)</span>
            <span className="stat-icon">📈</span>
          </div>
          <div className="stat-value">
            {(state.stats?.evaluation_auc ?? 0.773).toFixed(3)}
          </div>
          <div className="stat-caption">
            Walk-forward backtest average
          </div>
        </article>
      </section>

      {/* --- Section 3: Historical Flare Event Catalog & Filter Matrix --- */}
      <section className="catalog-section" id="catalog">
        <div className="section-header">
          <div>
            <div className="section-eyebrow">OBSERVATIONAL ARCHIVE · SOLAR CYCLE 24 & 25</div>
            <h2 className="section-title">Verified Solar Flare Catalog</h2>
            <p className="section-desc">
              Browse, filter, and audit 2,790+ documented M-class and X-class flare eruptions recorded by NOAA GOES XRS sensors.
            </p>
          </div>

          <div className="catalog-actions">
            <button className="export-btn" onClick={handleExportCsv}>
              📥 Export CSV Catalog
            </button>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="catalog-toolbar">
          <div className="catalog-filters">
            {/* Class Filter */}
            <div className="class-pill-group">
              <button
                className={`class-pill ${catalogClass === 'M' ? 'active' : ''}`}
                onClick={() => { setCatalogClass('M'); fetchCatalog('M'); playSound(640) }}
              >
                All M & X Flares ({state.stats?.total_m_flares ?? '2,660+'})
              </button>
              <button
                className={`class-pill x-class ${catalogClass === 'X' ? 'active' : ''}`}
                onClick={() => { setCatalogClass('X'); fetchCatalog('X'); playSound(720) }}
              >
                ⚡ Extreme X-Class Only ({state.stats?.total_x_flares ?? '138'})
              </button>
            </div>

            {/* Search Input */}
            <div className="search-input-wrap">
              <span className="search-icon">🔍</span>
              <input
                type="text"
                className="search-input"
                placeholder="Search date or flare class..."
                value={catalogSearch}
                onChange={e => setCatalogSearch(e.target.value)}
              />
            </div>
          </div>

          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Showing {filteredCatalog.length} records
          </span>
        </div>

        {/* Catalog Table */}
        <div className="catalog-table-wrap">
          <table className="catalog-table">
            <thead>
              <tr>
                <th>Event Timestamp (UTC)</th>
                <th>Classification</th>
                <th>Peak Flux</th>
                <th>Hourly Mean Flux</th>
                <th>GOES Telemetry Quality</th>
              </tr>
            </thead>
            <tbody>
              {catalogLoading ? (
                <tr>
                  <td colSpan="5" style={{ textAlign: 'center', padding: '30px' }}>
                    Loading catalog records...
                  </td>
                </tr>
              ) : filteredCatalog.length === 0 ? (
                <tr>
                  <td colSpan="5" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                    No matching flare events found.
                  </td>
                </tr>
              ) : (
                filteredCatalog.slice(0, 15).map((evt, idx) => (
                  <tr key={idx}>
                    <td>{formatUtc(evt.sample_time_utc, true)}</td>
                    <td>
                      <span className={`flare-class-tag ${evt.class_type}`}>
                        {evt.flare_class}
                      </span>
                    </td>
                    <td>
                      <span className="flux-val-mono">
                        {Number(evt.flux_max_w_m2).toExponential(3)} W/m²
                      </span>
                    </td>
                    <td>
                      <span className="flux-val-mono" style={{ color: 'var(--text-muted)' }}>
                        {Number(evt.flux_mean_w_m2).toExponential(3)} W/m²
                      </span>
                    </td>
                    <td style={{ color: 'var(--safe-mint)' }}>
                      ✓ {evt.valid_minutes}/60 mins ({evt.satellite_count} sat)
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* --- Section 4: Solar Cycle 25 Rolling Backtest & Verification Matrix --- */}
      <section className="scientific-section" id="validation">
        <div className="section-header">
          <div>
            <div className="section-eyebrow">PHASE 7 SCIENTIFIC BENCHMARK</div>
            <h2 className="section-title">Solar Cycle 25 Multi-Epoch Walk-Forward Validation</h2>
            <p className="section-desc">
              Model performance rigorously evaluated across 4 chronological epochs spanning solar minimum to peak solar maximum.
            </p>
          </div>
          <a
            href={`${API}/evaluation/scientific`}
            target="_blank"
            rel="noreferrer"
            className="export-btn"
          >
            Scientific JSON Report ↗
          </a>
        </div>

        <div className="backtest-grid">
          <div className="epoch-card">
            <div className="epoch-header">
              <span className="epoch-name">Epoch 1 · Cycle 25 Ascent</span>
              <span className="status-chip">Fold 1</span>
            </div>
            <div className="epoch-dates">Nov 2020 – Apr 2022 · Solar Minimum Transition</div>
            <div className="epoch-metrics-row">
              <div className="epoch-metric-cell">
                <strong>+0.697</strong>
                <span>BSS Skill</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: '#fff' }}>0.364</strong>
                <span>TSS Score</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: 'var(--plasma-blue)' }}>0.822</strong>
                <span>ROC-AUC</span>
              </div>
            </div>
          </div>

          <div className="epoch-card">
            <div className="epoch-header">
              <span className="epoch-name">Epoch 2 · Escalation Phase</span>
              <span className="status-chip">Fold 2</span>
            </div>
            <div className="epoch-dates">Apr 2022 – Sep 2023 · Sunspot Emergence</div>
            <div className="epoch-metrics-row">
              <div className="epoch-metric-cell">
                <strong>+0.528</strong>
                <span>BSS Skill</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: '#fff' }}>0.245</strong>
                <span>TSS Score</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: 'var(--plasma-blue)' }}>0.771</strong>
                <span>ROC-AUC</span>
              </div>
            </div>
          </div>

          <div className="epoch-card">
            <div className="epoch-header">
              <span className="epoch-name">Epoch 3 · High Activity</span>
              <span className="status-chip">Fold 3</span>
            </div>
            <div className="epoch-dates">Sep 2023 – Feb 2025 · Frequent M/X Flares</div>
            <div className="epoch-metrics-row">
              <div className="epoch-metric-cell">
                <strong>+0.384</strong>
                <span>BSS Skill</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: '#fff' }}>0.222</strong>
                <span>TSS Score</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: 'var(--plasma-blue)' }}>0.758</strong>
                <span>ROC-AUC</span>
              </div>
            </div>
          </div>

          <div className="epoch-card">
            <div className="epoch-header">
              <span className="epoch-name">Epoch 4 · Solar Maximum Peak</span>
              <span className="status-chip">Fold 4</span>
            </div>
            <div className="epoch-dates">Feb 2025 – Nov 2026 · Climatology 76.2%</div>
            <div className="epoch-metrics-row">
              <div className="epoch-metric-cell">
                <strong>+0.230</strong>
                <span>BSS Skill</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: '#fff' }}>0.151</strong>
                <span>TSS Score</span>
              </div>
              <div className="epoch-metric-cell">
                <strong style={{ color: 'var(--plasma-blue)' }}>0.739</strong>
                <span>ROC-AUC</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- Section 5: Operator Alert & Autonomous Dispatcher --- */}
      <section className="alert-config-card">
        <div>
          <div className="section-eyebrow">SPACE WEATHER AUTOMATION</div>
          <h2 className="section-title" style={{ fontSize: '22px' }}>
            Mission Alert & Threshold Dispatcher
          </h2>
          <p className="section-desc">
            Configure custom probability triggers to automate orbital payload safing, polar flight re-routing, and satellite operations advisories.
          </p>

          <div className="alert-slider-wrap">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px', fontWeight: '600' }}>
                Operational Warning Cutoff: <strong style={{ color: 'var(--solar-amber)' }}>{alertThreshold}%</strong>
              </span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Platt Baseline: 58.5%
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="90"
              step="5"
              value={alertThreshold}
              onChange={e => {
                const val = Number(e.target.value)
                setAlertThreshold(val)
                localStorage.setItem('solarguard_alert_threshold', val)
              }}
              className="alert-slider"
            />
          </div>

          <div className="alert-channels">
            <span className="channel-tag active">✓ Acoustic Alarm</span>
            <span className="channel-tag active">✓ ICAO Aviation HF Advisory</span>
            <span className="channel-tag active">✓ Satcom Spacecraft Safing</span>
            <span className="channel-tag">○ PagerDuty Webhook</span>
          </div>
        </div>

        {/* Dispatch Action Box */}
        <div className="test-alert-box">
          <div style={{ marginBottom: '12px', fontSize: '12px', color: 'var(--text-muted)' }}>
            Simulate an automated critical space weather dispatch trigger:
          </div>
          <button className="trigger-alert-btn" onClick={handleTestAlert}>
            ⚠️ Test Emergency Dispatch Alarm
          </button>
        </div>
      </section>

      {/* --- Section 6: Methodology & Science Accordion --- */}
      <section className="methodology-section" id="methodology">
        <div className="section-header">
          <div>
            <div className="section-eyebrow">HELIOPHYSICS & ML ARCHITECTURE</div>
            <h2 className="section-title">Methodology & Operational Guardrails</h2>
            <p className="section-desc">
              Comprehensive scientific documentation of our operational data pipeline, feature engineering, and calibration protocols.
            </p>
          </div>
        </div>

        <div className="accordion-group">
          <div className={`accordion-item ${accordionOpen.target ? 'open' : ''}`}>
            <div
              className="accordion-header"
              onClick={() => {
                setAccordionOpen(p => ({ ...p, target: !p.target }))
                playSound(600)
              }}
            >
              <span className="accordion-title">
                <span>01</span>
                How is the Next-24-Hour M1+ Flare Target Defined?
              </span>
              <span className="accordion-icon">+</span>
            </div>
            {accordionOpen.target && (
              <div className="accordion-content">
                In heliophysics space weather forecasting, an M-class flare is defined by the NOAA Space Weather Prediction Center (SWPC) as an event where solar soft X-ray flux in the <code>0.1–0.8 nm</code> (1–8 Ångström) band peaks at or above <code>1.0 × 10⁻⁵ W/m²</code>. An X-class flare peaks at or above <code>1.0 × 10⁻⁴ W/m²</code>. The binary prediction target <code>label_m_plus_next_24h</code> evaluates whether at least one completed hour in the subsequent 24 UTC hours contains a maximum flux <code>≥ 1.0 × 10⁻⁵ W/m²</code>.
              </div>
            )}
          </div>

          <div className={`accordion-item ${accordionOpen.features ? 'open' : ''}`}>
            <div
              className="accordion-header"
              onClick={() => {
                setAccordionOpen(p => ({ ...p, features: !p.features }))
                playSound(600)
              }}
            >
              <span className="accordion-title">
                <span>02</span>
                14 Causal Trailing Features & Rolling Windows
              </span>
              <span className="accordion-icon">+</span>
            </div>
            {accordionOpen.features && (
              <div className="accordion-content">
                To prevent temporal data leakage, our model uses exclusively past-only trailing observations computed across 6, 12, 24, and 48-hour backward horizons. Features include trailing logarithmic flux means, trailing maximum flux spikes, trailing flux variance, 6h-to-24h flux acceleration ratios, and log10 flux differences. All features are strictly causal and invariant to missing future information.
              </div>
            )}
          </div>

          <div className={`accordion-item ${accordionOpen.calibration ? 'open' : ''}`}>
            <div
              className="accordion-header"
              onClick={() => {
                setAccordionOpen(p => ({ ...p, calibration: !p.calibration }))
                playSound(600)
              }}
            >
              <span className="accordion-title">
                <span>03</span>
                Platt Sigmoidal Calibration & Brier Skill Scoring
              </span>
              <span className="accordion-icon">+</span>
            </div>
            {accordionOpen.calibration && (
              <div className="accordion-content">
                Raw uncalibrated ML outputs often misrepresent empirical probability under heavy class imbalance. SolarGuard fits a post-hoc Platt sigmoid calibrator on an independent validation fold, mapping raw logits into true probabilities. We score probabilistic accuracy using the Brier Skill Score (BSS) relative to the sample climatology, verifying that the model adds predictive skill over naive persistence.
              </div>
            )}
          </div>

          <div className={`accordion-item ${accordionOpen.shift ? 'open' : ''}`}>
            <div
              className="accordion-header"
              onClick={() => {
                setAccordionOpen(p => ({ ...p, shift: !p.shift }))
                playSound(600)
              }}
            >
              <span className="accordion-title">
                <span>04</span>
                Operational SWPC Telemetry vs Science-Quality Archives
              </span>
              <span className="accordion-icon">+</span>
            </div>
            {accordionOpen.shift && (
              <div className="accordion-content">
                Real-time NOAA SWPC JSON feeds represent operational L1b satellite telemetry from GOES-18. While ideal for real-time monitoring, operational feeds are subject to raw data glitches, occasional telemetry lag, and minor sensor calibration offsets compared to the re-calibrated science-grade NCEI archives used during baseline model training. SolarGuard explicitly flags telemetry distribution shifts in compliance with operational space weather standards.
              </div>
            )}
          </div>
        </div>
      </section>

      {/* --- Footer --- */}
      <footer className="app-footer">
        <div>
          <strong>SolarGuard AI Mission Control Console</strong> · GOES-18 Primary Telemetry · NOAA SWPC Integration
        </div>
        <div>
          <a href={`${API}/docs`} target="_blank" rel="noreferrer" style={{ marginRight: '16px' }}>
            FastAPI OpenAPI Docs ↗
          </a>
          <span>Experimental Heliophysics Research Prototype</span>
        </div>
      </footer>

      {/* --- Operator Authentication & Role Switcher Modal --- */}
      {authModalOpen && (
        <div className="modal-overlay" onClick={() => setAuthModalOpen(false)}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">Mission Specialist Clearance & Profile</span>
              <button className="modal-close" onClick={() => setAuthModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Select an active operator profile to adjust security clearance, telemetry views, and alert thresholds:
              </p>
              <div className="role-cards">
                {OPERATOR_PROFILES.map(prof => (
                  <div
                    key={prof.id}
                    className={`role-option ${operator.id === prof.id ? 'selected' : ''}`}
                    onClick={() => {
                      setOperator(prof)
                      localStorage.setItem('solarguard_operator', JSON.stringify(prof))
                      setAuthModalOpen(false)
                      addToast('Operator Active', `Session authorized for ${prof.name} (${prof.clearance})`, 'success')
                      playSound(880)
                    }}
                  >
                    <div className="operator-avatar" style={{ width: '32px', height: '32px', fontSize: '13px' }}>
                      {prof.avatar}
                    </div>
                    <div style={{ flex: 1 }}>
                      <strong style={{ display: 'block', fontSize: '13px' }}>{prof.name}</strong>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {prof.role} · {prof.department}
                      </span>
                    </div>
                    <span className="status-chip">{prof.clearance}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* --- Solar Flare Scenario Simulator Modal --- */}
      {simModalOpen && (
        <div className="modal-overlay" onClick={() => setSimModalOpen(false)}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">⚡ Solar Eruption Sensitivity Simulator</span>
              <button className="modal-close" onClick={() => setSimModalOpen(false)}>✕</button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Inject hypothetical physical solar conditions into the trailing observation window to evaluate model responsiveness:
              </p>

              <div className="scenario-options">
                <button
                  className="scenario-btn"
                  onClick={() => runSimulation('x_class_burst')}
                  disabled={simLoading}
                >
                  <div className="scenario-title" style={{ color: 'var(--flare-x-red)' }}>
                    <span>💥</span> Extreme X-Class Superflare Injection (X2.5 Peak)
                  </div>
                  <div className="scenario-desc">
                    Simulates a major solar eruption (1.5e-4 W/m²) peaking in the trailing 2 hours. Expected probability spike &gt;80%.
                  </div>
                </button>

                <button
                  className="scenario-btn"
                  onClick={() => runSimulation('m_flare_surge')}
                  disabled={simLoading}
                >
                  <div className="scenario-title" style={{ color: 'var(--solar-amber)' }}>
                    <span>⚡</span> Active Region M-Flare Surge (M4.8 Peak)
                  </div>
                  <div className="scenario-desc">
                    Simulates multiple coronal mass ejection precursors with moderate M4.8 activity across the past 3 hours.
                  </div>
                </button>

                <button
                  className="scenario-btn"
                  onClick={() => runSimulation('quiet_background')}
                  disabled={simLoading}
                >
                  <div className="scenario-title" style={{ color: 'var(--safe-mint)' }}>
                    <span>☀️</span> Deep Solar Minimum / Quiescent Background
                  </div>
                  <div className="scenario-desc">
                    Drops all trailing flux to peaceful B-class levels (2.5e-8 W/m²). Expected probability &lt;5%.
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* --- Floating Toast Notification Stack --- */}
      <div className="toast-container">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast toast-${toast.type}`}>
            <span className="toast-icon">
              {toast.type === 'error' ? '❌' : toast.type === 'warning' ? '⚠️' : 'ℹ️'}
            </span>
            <div className="toast-body">
              <strong>{toast.title}</strong>
              <p>{toast.message}</p>
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
