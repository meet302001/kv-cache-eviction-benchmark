"""
Eviction Policy Orchestrator.

Main loop that ties together the load generator, metrics collector, and
eviction policies. Monitors KV-cache pressure via vLLM metrics and
triggers evictions according to the selected policy.

Usage:
    python orchestrator.py --policy fifo --workload chatbot --duration 300
    python orchestrator.py --policy cost_aware --workload burst --duration 120
"""

import asyncio
import aiohttp
import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from load_generator import LoadGenerator, WorkloadType, RequestResult
from metrics_collector import MetricsCollector
from policies import POLICIES

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


class Orchestrator:
    """Monitors KV-cache pressure and evicts requests per the chosen policy.

    The eviction loop runs concurrently with the load generator:
      1. Poll vLLM metrics every `poll_interval` seconds
      2. If gpu_cache_usage_perc > threshold AND requests are waiting → evict
      3. Pick a victim using the selected policy
      4. Cancel the victim via the load generator (closes HTTP stream)
      5. Log the eviction with metadata
    """

    def __init__(
        self,
        policy_name: str,
        workload: WorkloadType,
        duration: int = 300,
        concurrency: int | None = None,
        cache_threshold: float = 0.90,
        poll_interval: float = 0.5,
        cooldown: float = 1.0,
        base_url: str = "http://localhost:8000",
    ):
        self.policy = POLICIES[policy_name]()
        self.policy_name = policy_name
        self.workload = workload
        self.cache_threshold = cache_threshold
        self.poll_interval = poll_interval
        self.cooldown = cooldown

        self.load_gen = LoadGenerator(
            base_url=base_url,
            workload=workload,
            duration=duration,
            concurrency=concurrency,
        )
        self.metrics = MetricsCollector(
            metrics_url=f"{base_url}/metrics",
            poll_interval=poll_interval,
        )

        self.eviction_log: List[dict] = []
        self._last_eviction_time = 0.0

    async def _eviction_loop(self, session: aiohttp.ClientSession):
        """Watch metrics and trigger evictions when cache pressure is high."""
        logger.info(
            "Eviction loop started: policy=%s threshold=%.0f%% cooldown=%.1fs",
            self.policy_name,
            self.cache_threshold * 100,
            self.cooldown,
        )

        while not self.load_gen._stop_event.is_set():
            snapshot = self.metrics.latest
            if snapshot is None:
                await asyncio.sleep(self.poll_interval)
                continue

            should_evict = (
                snapshot.gpu_cache_usage_perc >= self.cache_threshold
                and snapshot.num_requests_waiting > 0
                and len(self.load_gen.active_requests) > 1
                and (time.monotonic() - self._last_eviction_time) >= self.cooldown
            )

            if should_evict:
                active = self.load_gen.active_requests.copy()
                if len(active) > 1:
                    try:
                        victim_id = self.policy.select_victim(active)
                    except ValueError:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    victim = active.get(victim_id)
                    tokens_wasted = victim.tokens_generated if victim else 0

                    if self.load_gen.cancel_request(victim_id):
                        self._last_eviction_time = time.monotonic()

                        entry = {
                            "timestamp": time.time(),
                            "policy": self.policy_name,
                            "victim_id": victim_id,
                            "tokens_wasted": tokens_wasted,
                            "cache_usage": snapshot.gpu_cache_usage_perc,
                            "requests_running": snapshot.num_requests_running,
                            "requests_waiting": snapshot.num_requests_waiting,
                        }
                        self.eviction_log.append(entry)

                        logger.info(
                            "EVICTION policy=%s victim=%s wasted=%d tokens "
                            "cache=%.1f%% running=%d waiting=%d",
                            self.policy_name,
                            victim_id,
                            tokens_wasted,
                            snapshot.gpu_cache_usage_perc * 100,
                            snapshot.num_requests_running,
                            snapshot.num_requests_waiting,
                        )

            await asyncio.sleep(self.poll_interval)

    async def run(self) -> List[RequestResult]:
        """Run the full benchmark: load generation + eviction loop."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            metrics_task = asyncio.create_task(self.metrics.poll_loop(session))

            # Small delay to let metrics collector get a first reading
            await asyncio.sleep(1.0)

            eviction_task = asyncio.create_task(self._eviction_loop(session))
            results = await self.load_gen.run()

            self.metrics.stop()
            self.load_gen.stop()
            await asyncio.sleep(0.5)
            eviction_task.cancel()
            metrics_task.cancel()

            try:
                await eviction_task
            except asyncio.CancelledError:
                pass
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass

        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: List[RequestResult]):
        """Write request results and eviction log to CSV files."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.policy_name}_{self.load_gen.profile.name}_{ts}"

        results_file = RESULTS_DIR / f"{prefix}_requests.csv"
        if results:
            fieldnames = [
                "request_id", "prompt_tokens", "completion_tokens",
                "target_max_tokens", "ttft", "total_latency",
                "completed", "evicted", "eviction_count",
                "error", "start_time", "end_time",
            ]
            with open(results_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
            logger.info("Results saved to %s", results_file)

        evictions_file = RESULTS_DIR / f"{prefix}_evictions.csv"
        if self.eviction_log:
            with open(evictions_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.eviction_log[0].keys())
                writer.writeheader()
                writer.writerows(self.eviction_log)
            logger.info("Eviction log saved to %s", evictions_file)

    def _print_summary(self, results: List[RequestResult]):
        completed = [r for r in results if r.completed]
        evicted = [r for r in results if r.evicted]
        errored = [r for r in results if r.error]
        ttfts = sorted(r.ttft for r in completed if r.ttft is not None)
        latencies = sorted(r.total_latency for r in completed)
        total_tokens = sum(r.completion_tokens for r in completed)
        wasted_tokens = sum(e["tokens_wasted"] for e in self.eviction_log)
        duration = self.load_gen.profile.duration

        def pct(vals, p):
            if not vals:
                return 0.0
            idx = min(int(len(vals) * p), len(vals) - 1)
            return vals[idx]

        print(f"\n{'='*64}")
        print(f"  BENCHMARK RESULTS")
        print(f"  Policy:   {self.policy_name}")
        print(f"  Workload: {self.load_gen.profile.name}")
        print(f"  Duration: {duration}s")
        print(f"{'='*64}")
        print(f"  Requests total:      {len(results)}")
        print(f"  Completed:           {len(completed)} ({100*len(completed)/max(len(results),1):.1f}%)")
        print(f"  Evicted:             {len(evicted)}")
        print(f"  Errors:              {len(errored)}")
        print(f"  Evictions triggered: {len(self.eviction_log)}")
        print(f"  Tokens wasted:       {wasted_tokens}")

        if ttfts:
            print(f"\n  TTFT  p50:  {pct(ttfts, 0.50):.3f}s")
            print(f"  TTFT  p95:  {pct(ttfts, 0.95):.3f}s")
            print(f"  TTFT  p99:  {pct(ttfts, 0.99):.3f}s")

        if latencies:
            print(f"\n  E2E   p50:  {pct(latencies, 0.50):.3f}s")
            print(f"  E2E   p95:  {pct(latencies, 0.95):.3f}s")
            print(f"  E2E   p99:  {pct(latencies, 0.99):.3f}s")

        print(f"\n  Throughput:          {len(completed)/max(duration,1):.2f} req/s")
        print(f"  Token throughput:    {total_tokens/max(duration,1):.1f} tok/s")
        print(f"  Wasted compute:     {wasted_tokens} tokens")
        print(f"{'='*64}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Eviction Policy Orchestrator",
    )
    parser.add_argument(
        "--policy", type=str, required=True,
        choices=list(POLICIES.keys()),
        help="Eviction policy to use",
    )
    parser.add_argument(
        "--workload", type=str, default="chatbot",
        choices=["chatbot", "batch", "mixed", "burst"],
        help="Workload profile",
    )
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    parser.add_argument("--threshold", type=float, default=0.90, help="Cache usage threshold (0-1)")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Min seconds between evictions")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Metrics poll interval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    orch = Orchestrator(
        policy_name=args.policy,
        workload=WorkloadType(args.workload),
        duration=args.duration,
        concurrency=args.concurrency,
        cache_threshold=args.threshold,
        cooldown=args.cooldown,
        base_url=args.url,
        poll_interval=args.poll_interval,
    )

    await orch.run()


if __name__ == "__main__":
    asyncio.run(main())
