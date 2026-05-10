import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import LiveMetrics from './pages/LiveMetrics'
import BenchmarkResults from './pages/BenchmarkResults'
import About from './pages/About'

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/metrics" replace />} />
          <Route path="/metrics" element={<LiveMetrics />} />
          <Route path="/results" element={<BenchmarkResults />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  )
}
