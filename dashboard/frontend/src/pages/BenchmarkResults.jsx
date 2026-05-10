import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from 'recharts'

const POLICY_COLORS = {
  FIFO: '#4f46e5',
  LRU: '#0ea5e9',
  SJF: '#f59e0b',
  'Cost-Aware': '#8b5cf6',
  Random: '#94a3b8',
}

const TABS = ['Completion', 'TTFT', 'Latency', 'Throughput']

export default function BenchmarkResults() {
  const [summary, setSummary] = useState([])
  const [policies, setPolicies] = useState([])
  const [workloads, setWorkloads] = useState([])
  const [selPolicies, setSelPolicies] = useState([])
  const [selWorkloads, setSelWorkloads] = useState([])
  const [activeTab, setActiveTab] = useState('Completion')
  const [showRaw, setShowRaw] = useState(false)
  const [rawData, setRawData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/summary')
      .then(r => r.json())
      .then(res => {
        setSummary(res.data)
        setPolicies(res.policies)
        setWorkloads(res.workloads)
        setSelPolicies(res.policies)
        setSelWorkloads(res.workloads)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const filtered = summary.filter(
    r => selPolicies.includes(r.policy) && selWorkloads.includes(r.workload)
  )

  const toggleFilter = (value, selected, setter) => {
    if (selected.includes(value)) {
      if (selected.length > 1) setter(selected.filter(v => v !== value))
    } else {
      setter([...selected, value])
    }
  }

  const chartData = (metric) => {
    const byWorkload = {}
    filtered.forEach(r => {
      if (!byWorkload[r.workload]) byWorkload[r.workload] = { workload: r.workload }
      byWorkload[r.workload][r.policyLabel] = r[metric]
    })
    return Object.values(byWorkload)
  }

  const policyLabels = [...new Set(filtered.map(r => r.policyLabel))]

  const loadRawData = async () => {
    if (rawData.length > 0) {
      setShowRaw(!showRaw)
      return
    }
    const res = await fetch('/api/results')
    const data = await res.json()
    setRawData(data.data.slice(0, 500))
    setShowRaw(true)
  }

  if (loading) return <div className="page-header"><h1>Loading...</h1></div>

  if (summary.length === 0) {
    return (
      <div className="page-header">
        <h1>Benchmark Results</h1>
        <p className="page-subtitle">No results found. Run a benchmark first.</p>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <h1>Benchmark Results</h1>
        <p className="page-subtitle">
          {summary.length} runs across {policies.length} policies and {workloads.length} workloads
        </p>
      </div>

      {/* Filters */}
      <div className="filters">
        <div className="filter-group">
          <span className="filter-label">Policy</span>
          <div className="filter-chips">
            {policies.map(p => {
              const label = summary.find(s => s.policy === p)?.policyLabel || p
              return (
                <button
                  key={p}
                  className={`chip ${selPolicies.includes(p) ? 'active' : ''}`}
                  onClick={() => toggleFilter(p, selPolicies, setSelPolicies)}
                >
                  {label}
                </button>
              )
            })}
          </div>
        </div>
        <div className="filter-group">
          <span className="filter-label">Workload</span>
          <div className="filter-chips">
            {workloads.map(w => (
              <button
                key={w}
                className={`chip ${selWorkloads.includes(w) ? 'active' : ''}`}
                onClick={() => toggleFilter(w, selWorkloads, setSelWorkloads)}
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Table */}
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Workload</th>
              <th>Requests</th>
              <th>Completed</th>
              <th>Rate</th>
              <th>TTFT p50</th>
              <th>TTFT p95</th>
              <th>E2E p50</th>
              <th>E2E p95</th>
              <th>tok/s</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i}>
                <td><span className="policy-dot" style={{ background: POLICY_COLORS[r.policyLabel] || '#999' }} />{r.policyLabel}</td>
                <td>{r.workload}</td>
                <td>{r.requests}</td>
                <td>{r.completed}</td>
                <td>{r.completionRate}%</td>
                <td>{r.ttftP50}s</td>
                <td>{r.ttftP95}s</td>
                <td>{r.e2eP50}s</td>
                <td>{r.e2eP95}s</td>
                <td>{r.tokensPerSec}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {TABS.map(t => (
          <button key={t} className={`tab ${activeTab === t ? 'active' : ''}`} onClick={() => setActiveTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {/* Charts */}
      <div className="chart-section">
        {activeTab === 'Completion' && (
          <>
            <h2>Request Completion Rate</h2>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={chartData('completionRate')}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="workload" />
                <YAxis domain={[0, 105]} unit="%" />
                <Tooltip />
                <Legend />
                {policyLabels.map(p => (
                  <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </>
        )}

        {activeTab === 'TTFT' && (
          <div className="chart-grid">
            <div>
              <h2>TTFT p50</h2>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={chartData('ttftP50')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="workload" />
                  <YAxis unit="s" />
                  <Tooltip />
                  <Legend />
                  {policyLabels.map(p => (
                    <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h2>TTFT p95</h2>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={chartData('ttftP95')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="workload" />
                  <YAxis unit="s" />
                  <Tooltip />
                  <Legend />
                  {policyLabels.map(p => (
                    <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'Latency' && (
          <div className="chart-grid">
            <div>
              <h2>E2E Latency p50</h2>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={chartData('e2eP50')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="workload" />
                  <YAxis unit="s" />
                  <Tooltip />
                  <Legend />
                  {policyLabels.map(p => (
                    <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h2>E2E Latency p95</h2>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={chartData('e2eP95')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="workload" />
                  <YAxis unit="s" />
                  <Tooltip />
                  <Legend />
                  {policyLabels.map(p => (
                    <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'Throughput' && (
          <>
            <h2>Token Throughput</h2>
            <ResponsiveContainer width="100%" height={380}>
              <BarChart data={chartData('tokensPerSec')}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="workload" />
                <YAxis unit=" tok/s" />
                <Tooltip />
                <Legend />
                {policyLabels.map(p => (
                  <Bar key={p} dataKey={p} fill={POLICY_COLORS[p] || '#999'} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>

      {/* Raw Data */}
      <div className="raw-data-section">
        <button className="text-button" onClick={loadRawData}>
          {showRaw ? 'Hide raw data' : 'Show raw request data'}
        </button>
        {showRaw && rawData.length > 0 && (
          <div className="table-container table-scroll">
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Workload</th>
                  <th>ID</th>
                  <th>Prompt</th>
                  <th>Completion</th>
                  <th>TTFT</th>
                  <th>Latency</th>
                  <th>Done</th>
                </tr>
              </thead>
              <tbody>
                {rawData.map((r, i) => (
                  <tr key={i}>
                    <td>{r.policy}</td>
                    <td>{r.workload}</td>
                    <td className="mono">{r.request_id}</td>
                    <td>{r.prompt_tokens}</td>
                    <td>{r.completion_tokens}</td>
                    <td>{r.ttft ? Number(r.ttft).toFixed(3) : '-'}s</td>
                    <td>{Number(r.total_latency).toFixed(2)}s</td>
                    <td>{r.completed ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
