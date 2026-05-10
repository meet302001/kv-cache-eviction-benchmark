import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import MetricCard from '../components/MetricCard'

const POLL_INTERVAL = 2000

export default function LiveMetrics() {
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])
  const [polling, setPolling] = useState(false)
  const [error, setError] = useState(null)
  const startTime = useRef(Date.now())

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/metrics')
      const data = await res.json()
      if (data.status === 'error') {
        setError(data.error)
        return
      }
      setError(null)
      setMetrics(data)
      setHistory(prev => {
        const elapsed = Math.round((Date.now() - startTime.current) / 1000)
        const next = [...prev, {
          time: elapsed,
          cache: +(data.gpu_cache_usage * 100).toFixed(1),
          running: data.requests_running,
          waiting: data.requests_waiting,
        }]
        return next.length > 200 ? next.slice(-200) : next
      })
    } catch (e) {
      setError('Cannot connect to backend')
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [])

  useEffect(() => {
    if (!polling) return
    const id = setInterval(fetchMetrics, POLL_INTERVAL)
    return () => clearInterval(id)
  }, [polling])

  return (
    <div>
      <div className="page-header">
        <h1>Live Metrics</h1>
        <p className="page-subtitle">Real-time KV-cache and request state from vLLM</p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {metrics && (
        <>
          <div className="metrics-grid">
            <MetricCard label="Cache Usage" value={`${(metrics.gpu_cache_usage * 100).toFixed(1)}`} unit="%" />
            <MetricCard label="Running" value={metrics.requests_running} />
            <MetricCard label="Waiting" value={metrics.requests_waiting} />
            <MetricCard label="Swapped" value={metrics.requests_swapped} />
            <MetricCard label="Preemptions" value={metrics.preemptions} />
          </div>

          <div className="cache-bar-container">
            <div className="cache-bar-label">
              <span>KV-Cache Utilization</span>
              <span>{(metrics.gpu_cache_usage * 100).toFixed(1)}%</span>
            </div>
            <div className="cache-bar-track">
              <div
                className={`cache-bar-fill ${metrics.gpu_cache_usage > 0.9 ? 'critical' : metrics.gpu_cache_usage > 0.7 ? 'warning' : ''}`}
                style={{ width: `${Math.min(metrics.gpu_cache_usage * 100, 100)}%` }}
              />
              <div className="cache-bar-threshold" style={{ left: '90%' }} />
            </div>
          </div>
        </>
      )}

      {history.length > 1 && (
        <div className="chart-section">
          <h2>Metrics Over Time</h2>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="time" label={{ value: 'Seconds', position: 'insideBottom', offset: -5 }} />
              <YAxis yAxisId="left" domain={[0, 100]} label={{ value: 'Cache %', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" label={{ value: 'Count', angle: 90, position: 'insideRight' }} />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="cache" name="Cache %" stroke="#4f46e5" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="running" name="Running" stroke="#0ea5e9" strokeWidth={1.5} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="waiting" name="Waiting" stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="poll-toggle">
        <label>
          <input type="checkbox" checked={polling} onChange={e => setPolling(e.target.checked)} />
          <span>Auto-refresh every 2 seconds</span>
        </label>
      </div>
    </div>
  )
}
