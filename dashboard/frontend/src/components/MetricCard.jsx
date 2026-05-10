export default function MetricCard({ label, value, unit }) {
  return (
    <div className="metric-card">
      <div className="metric-value">
        {value}{unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className="metric-label">{label}</div>
    </div>
  )
}
