export default function About() {
  return (
    <div>
      <div className="page-header">
        <h1>About</h1>
      </div>

      <blockquote className="about-quote">
        When GPU memory is full, which request do you sacrifice -- and what does that decision cost you?
      </blockquote>

      <p className="about-text">
        This project benchmarks five KV-cache eviction policies under GPU memory pressure,
        measuring their impact on latency, throughput, fairness, and wasted compute.
        Built on consumer hardware to make the tradeoffs visible and reproducible.
      </p>

      <section className="about-section">
        <h2>The Problem</h2>
        <p className="about-text">
          When concurrent LLM requests exceed KV-cache capacity, the inference server must
          evict a request's cached Key/Value tensors to make room. The evicted request loses
          all generation progress and must restart from scratch. The choice of which request
          to evict determines user-facing latency, throughput, fairness, and how much GPU
          compute is wasted.
        </p>
      </section>

      <section className="about-section">
        <h2>Eviction Policies</h2>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Policy</th>
                <th>Strategy</th>
                <th>Evicts</th>
                <th>Best For</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>FIFO</td><td>First In, First Out</td><td>Oldest request</td><td>Fairness</td></tr>
              <tr><td>LRU</td><td>Least Recently Used</td><td>Least active request</td><td>Interactive latency</td></tr>
              <tr><td>SJF</td><td>Shortest Job First</td><td>Most remaining work</td><td>Throughput</td></tr>
              <tr><td>Cost-Aware</td><td>Least Work Done</td><td>Cheapest to redo</td><td>Min wasted compute</td></tr>
              <tr><td>Random</td><td>Random selection</td><td>Any request</td><td>Baseline comparison</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="about-section">
        <h2>OS Page Replacement Analogy</h2>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>OS Concept</th>
                <th>GPU Equivalent</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Virtual pages</td><td>Logical KV blocks (per request)</td></tr>
              <tr><td>Physical page frames</td><td>Physical KV blocks (VRAM)</td></tr>
              <tr><td>Page table</td><td>Block table</td></tr>
              <tr><td>Page fault</td><td>Allocate KV block on new token</td></tr>
              <tr><td>Page eviction</td><td>Request preemption</td></tr>
              <tr><td>Replacement policy</td><td>Eviction policy (this project)</td></tr>
            </tbody>
          </table>
        </div>
        <p className="about-note">
          Key difference: OS page eviction preserves data on disk. LLM KV-cache eviction
          destroys computed state -- the evicted request must recompute from scratch.
        </p>
      </section>

      <section className="about-section">
        <h2>Architecture</h2>
        <pre className="arch-diagram">{`Load Generator (async Python/aiohttp)
      |
      v
Eviction Orchestrator
  |-- Polls vLLM /metrics (cache %, queue depth)
  |-- Selects victim per policy
  |-- Cancels victim's HTTP stream
      |
      v
vLLM (stock, unmodified) --> Prometheus --> Grafana`}</pre>
      </section>

      <div className="about-grid">
        <section className="about-section">
          <h2>Hardware</h2>
          <div className="table-container">
            <table className="data-table">
              <tbody>
                <tr><td>GPU</td><td>RTX 4070 Laptop (8 GB)</td></tr>
                <tr><td>Model</td><td>Qwen2.5-3B-AWQ (~2 GB)</td></tr>
                <tr><td>KV Budget</td><td>~5.5 GB</td></tr>
                <tr><td>Pressure at</td><td>10-15 concurrent requests</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="about-section">
          <h2>Stack</h2>
          <div className="table-container">
            <table className="data-table">
              <tbody>
                <tr><td>vLLM v0.8.4</td><td>LLM inference engine</td></tr>
                <tr><td>Prometheus</td><td>Metrics collection</td></tr>
                <tr><td>Grafana</td><td>Dashboards</td></tr>
                <tr><td>Python / aiohttp</td><td>Load generation</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <div className="about-footer">
        <a href="https://github.com/meet302001/kv-cache-eviction-benchmark" target="_blank" rel="noreferrer">
          View on GitHub
        </a>
      </div>
    </div>
  )
}
