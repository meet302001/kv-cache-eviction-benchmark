"""
Benchmark runner — runs all eviction policies across all workload profiles.

Produces a matrix of results: 5 policies × 4 workloads = 20 runs.
Each run saves per-request CSVs and eviction logs to loadtest/results/.

Usage:
    python run_benchmark.py                          # full suite (20 runs)
    python run_benchmark.py --duration 60            # shorter runs
    python run_benchmark.py --policies fifo lru      # subset of policies
    python run_benchmark.py --workloads chatbot burst # subset of workloads
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import Orchestrator
from load_generator import WorkloadType
from policies import POLICIES

logger = logging.getLogger(__name__)


async def run_single(
    policy_name: str,
    workload: WorkloadType,
    duration: int,
    threshold: float,
    cooldown: float,
    base_url: str,
) -> dict:
    """Run a single policy × workload benchmark and return summary stats."""
    logger.info(
        "=" * 64 + "\n  RUN: policy=%s workload=%s duration=%ds\n" + "=" * 64,
        policy_name, workload.value, duration,
    )

    orch = Orchestrator(
        policy_name=policy_name,
        workload=workload,
        duration=duration,
        cache_threshold=threshold,
        cooldown=cooldown,
        base_url=base_url,
    )

    results = await orch.run()

    completed = [r for r in results if r.completed]
    evicted = [r for r in results if r.evicted]
    ttfts = sorted(r.ttft for r in completed if r.ttft is not None)
    latencies = sorted(r.total_latency for r in completed)
    total_tokens = sum(r.completion_tokens for r in completed)
    wasted = sum(e["tokens_wasted"] for e in orch.eviction_log)

    def pct(vals, p):
        if not vals:
            return 0.0
        return vals[min(int(len(vals) * p), len(vals) - 1)]

    return {
        "policy": policy_name,
        "workload": workload.value,
        "total_requests": len(results),
        "completed": len(completed),
        "completion_rate": len(completed) / max(len(results), 1),
        "evicted": len(evicted),
        "evictions_triggered": len(orch.eviction_log),
        "tokens_wasted": wasted,
        "ttft_p50": pct(ttfts, 0.50),
        "ttft_p95": pct(ttfts, 0.95),
        "ttft_p99": pct(ttfts, 0.99),
        "e2e_p50": pct(latencies, 0.50),
        "e2e_p95": pct(latencies, 0.95),
        "e2e_p99": pct(latencies, 0.99),
        "throughput_rps": len(completed) / max(duration, 1),
        "token_throughput": total_tokens / max(duration, 1),
        "total_tokens": total_tokens,
    }


async def run_benchmark(
    policies: List[str],
    workloads: List[WorkloadType],
    duration: int,
    threshold: float,
    cooldown: float,
    base_url: str,
    pause_between: int,
):
    """Run the full benchmark matrix and save a summary CSV."""
    total_runs = len(policies) * len(workloads)
    logger.info(
        "Starting benchmark: %d policies × %d workloads = %d runs, %ds each",
        len(policies), len(workloads), total_runs, duration,
    )

    summaries = []
    run_num = 0

    for workload in workloads:
        for policy_name in policies:
            run_num += 1
            logger.info(
                "\n>>> Run %d/%d: %s × %s",
                run_num, total_runs, policy_name, workload.value,
            )

            summary = await run_single(
                policy_name=policy_name,
                workload=workload,
                duration=duration,
                threshold=threshold,
                cooldown=cooldown,
                base_url=base_url,
            )
            summaries.append(summary)

            if run_num < total_runs:
                logger.info(
                    "Pausing %ds between runs for vLLM to stabilize...",
                    pause_between,
                )
                await asyncio.sleep(pause_between)

    _save_summary(summaries)
    _print_final_summary(summaries)


def _save_summary(summaries: List[dict]):
    """Write the aggregated summary CSV."""
    import csv

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"benchmark_summary_{ts}.csv"

    fieldnames = list(summaries[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    logger.info("Benchmark summary saved to %s", filepath)


def _print_final_summary(summaries: List[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("  BENCHMARK SUMMARY — ALL RUNS")
    print(f"{'='*80}")
    print(
        f"{'Policy':<12} {'Workload':<10} {'Done%':>6} {'Evict':>6} "
        f"{'Wasted':>7} {'TTFT p50':>9} {'TTFT p95':>9} "
        f"{'E2E p50':>9} {'Tput':>7}"
    )
    print("-" * 80)

    for s in summaries:
        print(
            f"{s['policy']:<12} {s['workload']:<10} "
            f"{s['completion_rate']*100:>5.1f}% "
            f"{s['evictions_triggered']:>6} "
            f"{s['tokens_wasted']:>7} "
            f"{s['ttft_p50']:>8.3f}s "
            f"{s['ttft_p95']:>8.3f}s "
            f"{s['e2e_p50']:>8.3f}s "
            f"{s['throughput_rps']:>6.2f}"
        )

    print(f"{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Run full KV-cache eviction policy benchmark",
    )
    parser.add_argument(
        "--policies", nargs="+", default=list(POLICIES.keys()),
        choices=list(POLICIES.keys()),
        help="Policies to benchmark (default: all)",
    )
    parser.add_argument(
        "--workloads", nargs="+", default=["chatbot", "batch", "mixed", "burst"],
        choices=["chatbot", "batch", "mixed", "burst"],
        help="Workload profiles to test (default: all)",
    )
    parser.add_argument("--duration", type=int, default=300, help="Duration per run in seconds")
    parser.add_argument("--threshold", type=float, default=0.90, help="Cache usage eviction threshold")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Min seconds between evictions")
    parser.add_argument("--pause", type=int, default=15, help="Seconds to pause between runs")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="vLLM base URL")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    workloads = [WorkloadType(w) for w in args.workloads]

    await run_benchmark(
        policies=args.policies,
        workloads=workloads,
        duration=args.duration,
        threshold=args.threshold,
        cooldown=args.cooldown,
        base_url=args.url,
        pause_between=args.pause,
    )


if __name__ == "__main__":
    asyncio.run(main())
