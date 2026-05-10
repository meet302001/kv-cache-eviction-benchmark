import { NavLink } from 'react-router-dom'

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <h1 className="sidebar-title">KV-Cache Eviction</h1>
        <p className="sidebar-subtitle">Benchmark Dashboard</p>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/metrics" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Live Metrics
        </NavLink>
        <NavLink to="/results" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Benchmark Results
        </NavLink>
        <NavLink to="/about" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          About
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        <p>RTX 4070 Laptop / 8 GB</p>
        <p>Qwen2.5-3B-AWQ</p>
        <a href="https://github.com/meet302001/kv-cache-eviction-benchmark" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </div>
    </aside>
  )
}
