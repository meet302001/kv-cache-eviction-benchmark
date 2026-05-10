# KV-Cache Eviction Policy Benchmark

Benchmarking KV-cache eviction policies for LLM inference under GPU memory pressure.

When concurrent requests exceed a GPU's KV-cache capacity, the inference server must evict a request's cached key/value tensors to make room. The evicted request loses all generation progress and restarts from scratch. This project measures how five different eviction strategies affect latency, throughput, fairness, and wasted compute.

## Architecture

```
Load Generator (async Python / aiohttp)
      |
      v
Eviction Orchestrator
  |-- Polls vLLM /metrics (cache %, queue depth)
  |-- Selects victim per policy
  |-- Cancels victim's HTTP stream
      |
      v
vLLM (stock, unmodified) --> Prometheus --> Grafana
      |
      v
React Dashboard  <--  FastAPI Backend  <-- CSV results
```

The system runs **external eviction** -- no vLLM modifications. The orchestrator monitors KV-cache pressure via Prometheus metrics, selects a victim using the chosen policy, and cancels its HTTP streaming connection. vLLM then reclaims that request's KV blocks.

## Eviction Policies

| Policy | Strategy | Evicts | Best For |
|--------|----------|--------|----------|
| FIFO | First In, First Out | Oldest request | Fairness |
| LRU | Least Recently Used | Least active request | Interactive latency |
| SJF | Shortest Job First | Most remaining work | Throughput |
| Cost-Aware | Least Work Done | Cheapest to redo | Min wasted compute |
| Random | Random selection | Any request | Baseline comparison |

These map directly to OS page replacement algorithms -- with one key difference: OS page eviction preserves data on disk, while LLM KV-cache eviction destroys computed state entirely.

## Project Structure

```
kv-cache-eviction/
  docker-compose.yml           # vLLM + Prometheus + Grafana stack
  prometheus.yml               # Scrape config for vLLM metrics
  grafana/                     # Grafana datasource provisioning

  loadtest/
    load_generator.py          # Async load generator (aiohttp, streaming)
    metrics_collector.py       # Real-time vLLM metrics poller
    orchestrator.py            # Eviction loop (monitor + select + cancel)
    policies/                  # 5 eviction policies with shared base class
    run_benchmark.py           # Full benchmark matrix runner
    analyze.py                 # Matplotlib chart generation

  dashboard/
    backend/
      server.py                # FastAPI API (metrics proxy, CSV reader)
      requirements.txt
    frontend/
      src/
        pages/
          LiveMetrics.jsx      # Real-time cache/request monitoring
          BenchmarkResults.jsx # Summary table, charts, filters
          About.jsx            # Project documentation
        components/
          Sidebar.jsx          # Navigation
          MetricCard.jsx       # Metric display card
```

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- Python 3.12+
- Node.js 20+

### 1. Start the infrastructure

```bash
docker compose up -d
```

This starts vLLM (port 8000), Prometheus (9090), and Grafana (3000).

### 2. Run a benchmark

```bash
cd loadtest
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_benchmark.py
```

This runs all 5 policies across workload profiles and saves results to `loadtest/results/`.

### 3. Launch the dashboard

```bash
# Terminal 1: Backend
cd dashboard/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py

# Terminal 2: Frontend
cd dashboard/frontend
npm install
npm run dev
```

Open **http://localhost:3001** to view the dashboard.

## Dashboard

Built with React 18, Recharts, and FastAPI.

**Live Metrics** -- Polls vLLM every 2 seconds. Displays cache utilization gauge, running/waiting/swapped request counts, and a time-series chart.

**Benchmark Results** -- Summary table with per-policy completion rates, TTFT, E2E latency, and throughput. Tabbed charts with policy/workload filters. Expandable raw request data.

**About** -- Project documentation, policy comparison tables, and architecture overview.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/metrics` | Live vLLM KV-cache and request metrics |
| `GET /api/results?policy=&workload=` | Raw per-request benchmark data |
| `GET /api/summary` | Aggregated policy x workload summary |

## Workload Profiles

| Profile | Prompt Length | Generation Length | Concurrency | Pattern |
|---------|-------------|-------------------|-------------|---------|
| Chatbot | 50-200 tokens | 100-500 tokens | 8 | Steady arrivals |
| Batch | 200-500 tokens | 500-1500 tokens | 15 | Parallel |
| Mixed | 50-500 tokens | 100-1500 tokens | 12 | Variable |
| Burst | 100-300 tokens | 200-800 tokens | 20 | Spike traffic |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4070 Laptop (8 GB VRAM) |
| Model | Qwen2.5-3B-Instruct-AWQ (~2 GB) |
| KV Budget | ~5.5 GB |
| Pressure Point | 10-15 concurrent requests |

## Stack

- **vLLM v0.8.4** -- LLM inference engine with PagedAttention
- **Prometheus** -- Metrics collection
- **Grafana** -- Infrastructure dashboards
- **Python / aiohttp** -- Async load generation and orchestration
- **React / Recharts** -- Dashboard frontend
- **FastAPI** -- Dashboard backend API
