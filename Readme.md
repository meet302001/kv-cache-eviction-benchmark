# KV-Cache Eviction Policy Comparison for LLM Inference

> _"When GPU memory is full, which request do you sacrifice — and what does that decision cost you?"_

This project benchmarks five KV-cache eviction policies under GPU memory pressure, measuring their impact on latency, throughput, fairness, and wasted compute. Built on consumer hardware (RTX 4070 Laptop, 8GB VRAM) to make the tradeoffs visible and reproducible.

---

## Table of Contents

- [The Problem](#the-problem)
- [Why KV-Cache Is the Real Bottleneck](#why-kv-cache-is-the-real-bottleneck)
- [Eviction Policies](#eviction-policies)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Hardware](#hardware)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Metrics Tracked](#metrics-tracked)
- [Workload Profiles](#workload-profiles)
- [Findings](#findings)
- [The OS Analogy](#the-os-analogy)
- [References](#references)

---

## The Problem

Traditional GPU autoscaling for LLM workloads uses GPU memory utilization as the scaling signal. This is fundamentally broken.

A GPU can report 80% memory utilization and still have plenty of KV-cache headroom — because model weights (fixed) and KV-cache (dynamic) are lumped into one number. Conversely, a GPU at 60% total memory can be completely KV-cache saturated, because the remaining 40% is occupied by model weights that can never be reclaimed.

**When KV-cache fills up, the inference server must evict a request's cached Key/Value tensors to make room.** The evicted request loses all its generation progress and must be reprocessed from scratch. The choice of _which_ request to evict determines:

- **User-facing latency** — does a nearly-finished request get killed, forcing a full restart?
- **Throughput** — are we maximizing completed requests per second?
- **Fairness** — do long-running requests get perpetually starved?
- **Wasted compute** — how many tokens of GPU work are thrown away?

This project implements five eviction policies as an external orchestrator around stock vLLM, measures each policy across four workload profiles, and publishes the results.

---

## Why KV-Cache Is the Real Bottleneck

For Qwen2.5-3B-Instruct-AWQ on an 8GB GPU:

```
Total VRAM:                    8,188 MiB
Model weights (AWQ 4-bit):    ~2,000 MiB
CUDA/framework overhead:        ~500 MiB
─────────────────────────────────────────
Available for KV-cache:       ~5,500 MiB

Per-token KV-cache memory:
  2 (K+V) × 36 (layers) × 2 (kv_heads) × 128 (head_dim) × 2 (bytes)
  = 36,864 bytes ≈ 36 KB/token

Per-request at 2048 context:   ~73.7 MiB
Max concurrent requests:       ~74 (theoretical)
At sustained load with
  long generations:            ~10-15 before pressure
```

When concurrent requests exceed KV-cache capacity, the server faces a choice: **queue new requests (increasing wait time) or evict existing requests (wasting completed work)**. The eviction policy determines which tradeoff is made.

---

## Eviction Policies

| Policy | Strategy | Eviction Target | Optimizes For | Risk |
|--------|----------|-----------------|---------------|------|
| **FIFO** | First In, First Out | Oldest request | Fairness | May evict nearly-finished requests |
| **LRU** | Least Recently Used | Least recently active request | Interactive latency | Batch jobs may starve |
| **SJF** | Shortest Job First | Request with most remaining tokens | Throughput | Long requests perpetually preempted |
| **Cost-Aware** | Least Work Done | Request with fewest tokens generated | Minimum wasted compute | May keep expensive requests too long |
| **Random** | Random selection | Any request | Baseline | No optimization (surprisingly competitive) |

### Why Random Matters

In OS page replacement research, random eviction is famously competitive with LRU in many real-world workloads. Including it as a baseline answers the question: _"Is a sophisticated eviction policy actually worth the complexity, or does random get you 90% of the way there?"_

---

## Architecture

```
┌──────────────────────────┐
│   Load Generator         │
│   (async Python)         │
│   - Ramps concurrency    │
│   - Varies prompt length │
│   - Records client TTFT  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│   Eviction Policy Orchestrator       │
│                                      │
│   Reads from Prometheus:             │
│   - vllm:gpu_cache_usage_perc       │
│   - vllm:num_requests_running       │
│   - vllm:num_requests_waiting       │
│   - vllm:num_requests_swapped       │
│                                      │
│   When cache > threshold:            │
│   → Selects victim per policy        │
│   → Cancels victim request           │
│   → Requeues for retry               │
│   → Logs eviction metadata           │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│   vLLM (stock, unmod.)   │
│   Qwen2.5-3B-AWQ         │
│   --max-model-len 2048   │
│   --gpu-mem-util 0.85    │
│   Exposes /metrics        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│   Prometheus             │────▶│   Grafana                │
│   Scrapes /metrics @5s   │     │   KV-Cache Golden        │
│                          │     │   Signals Dashboard      │
└──────────────────────────┘     └──────────────────────────┘
```

---

## Tech Stack

| Component | Role |
|-----------|------|
| [vLLM](https://github.com/vllm-project/vllm) | LLM inference engine with PagedAttention |
| [Prometheus](https://prometheus.io/) | Metrics collection and time-series storage |
| [Grafana](https://grafana.com/) | Dashboards and visualization |
| [Docker Compose](https://docs.docker.com/compose/) | Local orchestration |
| Python (aiohttp) | Load generation and eviction orchestration |
| NVIDIA Container Toolkit | GPU passthrough to containers |

---

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop |
| Architecture | Ada Lovelace |
| VRAM | 8 GB GDDR6X |
| CUDA Version | 12.6 |
| Driver Version | 561.17 |
| Model | Qwen2.5-3B-Instruct-AWQ (~2GB) |
| KV-Cache Budget | ~5.5 GB |

**Why consumer hardware?** 8GB VRAM reaches KV-cache saturation at ~10-15 concurrent requests — making eviction policy differences visible and measurable. On an 80GB H100, you'd need hundreds of concurrent users to observe the same bottleneck. Constrained hardware amplifies the tradeoffs that exist at any scale.

---

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- Docker with NVIDIA Container Toolkit
- Python 3.10+

### 1. Clone and start the stack

```bash
git clone https://github.com/YOUR_USERNAME/kv-cache-eviction-benchmark.git
cd kv-cache-eviction-benchmark

docker compose up -d

# Wait for vLLM to download and load the model (~3-5 min first run)
docker compose logs -f vllm
```

### 2. Verify the stack

```bash
# Check model is loaded
curl http://localhost:8000/v1/models

# Test a completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct-AWQ", "prompt": "The capital of France is", "max_tokens": 20}'

# Check metrics
curl -s http://localhost:8000/metrics | grep gpu_cache_usage

# Open Grafana
# http://localhost:3000 (admin/admin)
```

### 3. Run a benchmark

```bash
cd loadtest

# Install dependencies
pip install -r requirements.txt

# Run with a specific eviction policy and workload
python orchestrator.py --policy fifo --workload chatbot --duration 300

# Run full benchmark (all policies × all workloads)
python run_benchmark.py
```

### 4. View results

```bash
# Results are saved to loadtest/results/
# Open Grafana at http://localhost:3000 for real-time dashboards
# See docs/findings.md for analysis
```

---

## Project Structure

```
kv-cache-eviction-benchmark/
│
├── docker-compose.yml              # vLLM + Prometheus + Grafana stack
├── prometheus.yml                   # Prometheus scrape config
│
├── grafana/
│   └── provisioning/
│       └── datasources/
│           └── prometheus.yml       # Auto-configure Prometheus datasource
│
├── loadtest/
│   ├── requirements.txt             # Python dependencies
│   ├── orchestrator.py              # Eviction policy orchestrator
│   ├── load_generator.py            # Async request generator
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract policy interface
│   │   ├── fifo.py
│   │   ├── lru.py
│   │   ├── sjf.py
│   │   ├── cost_aware.py
│   │   └── random_policy.py
│   ├── metrics_collector.py         # Reads vLLM Prometheus metrics
│   ├── run_benchmark.py             # Runs all policies × all workloads
│   ├── analyze.py                   # Generates comparison charts
│   └── results/                     # Benchmark output (CSV + charts)
│
├── docs/
│   ├── architecture.md              # Detailed system design
│   ├── kv-cache-explainer.md        # Why KV-cache is the bottleneck
│   ├── paged-attention.md           # PagedAttention deep-dive
│   ├── eviction-policies.md         # Policy design rationale
│   └── findings.md                  # Benchmark results and analysis
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Metrics Tracked

### Server-Side (from vLLM /metrics via Prometheus)

| Metric | What It Tells You |
|--------|-------------------|
| `vllm:gpu_cache_usage_perc` | KV-cache fill level (0.0–1.0). **The real scaling signal.** |
| `vllm:num_requests_running` | Active requests on GPU |
| `vllm:num_requests_waiting` | Queued requests (users waiting) |
| `vllm:num_requests_swapped` | Requests evicted to CPU RAM (latency spike indicator) |
| `vllm:num_preemptions_total` | Total eviction count |
| `vllm:time_to_first_token_seconds` | Time from request arrival to first generated token |
| `vllm:e2e_request_latency_seconds` | Full request lifecycle duration |

### Client-Side (from orchestrator)

| Metric | What It Tells You |
|--------|-------------------|
| Request completion rate | % of requests that finish without eviction |
| Eviction count per request | How many times a request was preempted before completing |
| Wasted compute (tokens) | Total tokens generated across evicted requests that were regenerated |
| Client-measured TTFT | End-user perspective latency |
| Client-measured throughput | Completed requests per minute |
| Fairness index | Variance in completion time (Jain's fairness index) |

---

## Workload Profiles

| Profile | Description | Characteristics |
|---------|-------------|-----------------|
| **Chatbot** | Interactive conversation simulation | Short prompts (50-200 tokens), short generations (50-150 tokens), high concurrency, Poisson arrival |
| **Batch** | Document processing / summarization | Long prompts (500-1500 tokens), long generations (200-500 tokens), steady arrival rate |
| **Mixed** | Realistic production traffic | Blend of short and long requests, variable arrival rates |
| **Burst** | Sudden traffic spike | 30+ requests arriving within 5 seconds |

---

## Findings

> _Results will be published in [docs/findings.md](docs/findings.md) after benchmarking._

Expected findings based on OS page replacement research:

- **FIFO** should provide the most predictable behavior but waste the most compute on batch workloads
- **LRU** should excel at chatbot workloads where recent sessions are most valuable
- **Cost-Aware** should minimize total wasted compute across all workloads
- **SJF** should maximize raw throughput but starve long-running requests
- **Random** should be surprisingly competitive in mixed workloads (the "90% rule")

---

## The OS Analogy

This project applies operating systems fundamentals to GPU memory management. The mapping is direct:

| OS Concept | PagedAttention Equivalent |
|-----------|---------------------------|
| Virtual pages | Logical KV blocks (per request) |
| Physical page frames | Physical KV blocks (GPU VRAM) |
| Page table | Block table |
| Page fault → allocate | New token → allocate KV block |
| Page eviction → swap to disk | Request preemption → discard/swap KV blocks |
| Page replacement policy (LRU/FIFO/Clock) | **KV-cache eviction policy (this project)** |

The key difference: in OS paging, evicting a page to disk is cheap (the data is preserved). In LLM inference, evicting KV-cache means **discarding computed data** — the evicted request must recompute all its tokens from scratch, wasting GPU cycles proportional to the work already done.

This makes the eviction policy decision more consequential than OS page replacement, because the "swap" cost isn't I/O latency — it's recomputation time.

---

## References

- Kwon et al., ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) (2023) — The foundational paper for vLLM's memory management
- Remzi & Andrea Arpaci-Dusseau, ["Operating Systems: Three Easy Pieces — Beyond Physical Memory: Policies"](https://pages.cs.wisc.edu/~remzi/OSTEP/vm-beyondphys-policy.pdf) — OS page replacement fundamentals
- [vLLM Documentation](https://docs.vllm.ai/) — Inference engine used in this project
- [vLLM Metrics Reference](https://docs.vllm.ai/en/stable/usage/metrics/) — Prometheus metrics exposed by vLLM
- Xu et al., ["vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving"](https://arxiv.org/abs/2407.15309) (2024) — Alternative approach to KV-cache memory management
- Prabhu et al., ["vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention"](https://arxiv.org/abs/2405.04437) (2024) — GPU virtual memory approach

---

## License

MIT
