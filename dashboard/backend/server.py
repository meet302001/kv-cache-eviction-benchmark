"""
FastAPI backend for the KV-Cache Eviction Benchmark dashboard.

Endpoints:
    GET /api/metrics  -- live vLLM KV-cache and request metrics
    GET /api/results  -- raw per-request benchmark data from CSVs
    GET /api/summary  -- aggregated policy x workload summary
"""

import glob
import os
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KV-Cache Eviction Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://localhost:8000/metrics")
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "loadtest" / "results"

POLICY_LABELS = {
    "fifo": "FIFO",
    "lru": "LRU",
    "sjf": "SJF",
    "cost_aware": "Cost-Aware",
    "random": "Random",
}


def _parse_metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


@app.get("/api/metrics")
async def get_metrics():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(VLLM_METRICS_URL)
        text = resp.text
        return {
            "status": "ok",
            "gpu_cache_usage": _parse_metric(text, "vllm:gpu_cache_usage_perc"),
            "requests_running": int(_parse_metric(text, "vllm:num_requests_running")),
            "requests_waiting": int(_parse_metric(text, "vllm:num_requests_waiting")),
            "requests_swapped": int(_parse_metric(text, "vllm:num_requests_swapped")),
            "preemptions": int(_parse_metric(text, "vllm:num_preemptions_total")),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _load_csvs(
    policy_filter: Optional[str] = None,
    workload_filter: Optional[str] = None,
) -> pd.DataFrame:
    files = sorted(glob.glob(str(RESULTS_DIR / "*_requests.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        name = os.path.basename(f)
        # Format: {policy}_{workload}_{date}_{time}_requests.csv
        # rsplit with 3 splits separates the timestamp parts from policy_workload
        parts = name.replace("_requests.csv", "").rsplit("_", 3)
        if len(parts) == 4:
            policy, workload = parts[0], parts[1]
            run_ts = parts[2] + "_" + parts[3]
        elif len(parts) == 3:
            policy, workload, run_ts = parts[0], parts[1], parts[2]
        else:
            policy, workload, run_ts = parts[0], "unknown", ""

        if policy_filter and policy != policy_filter:
            continue
        if workload_filter and workload != workload_filter:
            continue

        df = pd.read_csv(f)
        df["policy"] = policy
        df["workload"] = workload
        df["run_ts"] = run_ts
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@app.get("/api/results")
async def get_results(
    policy: Optional[str] = Query(None),
    workload: Optional[str] = Query(None),
):
    df = _load_csvs(policy, workload)
    if df.empty:
        return {"data": [], "policies": [], "workloads": []}

    df = df.fillna("")
    return {
        "data": df.to_dict(orient="records"),
        "policies": sorted(df["policy"].unique().tolist()),
        "workloads": sorted(df["workload"].unique().tolist()),
    }


def _percentile(series: pd.Series, p: float) -> float:
    s = series.dropna().sort_values()
    if s.empty:
        return 0.0
    idx = min(int(len(s) * p), len(s) - 1)
    return round(float(s.iloc[idx]), 3)


@app.get("/api/summary")
async def get_summary():
    df = _load_csvs()
    if df.empty:
        return {"data": [], "policies": [], "workloads": []}

    rows = []
    for (policy, workload), group in df.groupby(["policy", "workload"]):
        latest_ts = group["run_ts"].max()
        run = group[group["run_ts"] == latest_ts]
        completed = run[run["completed"] == True]

        total = len(run)
        n_completed = len(completed)

        rows.append({
            "policy": policy,
            "policyLabel": POLICY_LABELS.get(policy, policy),
            "workload": workload,
            "requests": total,
            "completed": n_completed,
            "completionRate": round(n_completed / max(total, 1) * 100, 1),
            "evicted": int(run["evicted"].sum()),
            "ttftP50": _percentile(completed["ttft"], 0.50),
            "ttftP95": _percentile(completed["ttft"], 0.95),
            "e2eP50": _percentile(completed["total_latency"], 0.50),
            "e2eP95": _percentile(completed["total_latency"], 0.95),
            "totalTokens": int(completed["completion_tokens"].sum()),
            "tokensPerSec": round(completed["completion_tokens"].sum() / 60, 1),
        })

    return {
        "data": rows,
        "policies": sorted(df["policy"].unique().tolist()),
        "workloads": sorted(df["workload"].unique().tolist()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
