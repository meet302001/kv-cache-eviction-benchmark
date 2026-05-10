"""
Real-time metrics collector for vLLM.

Polls vLLM's /metrics endpoint directly (bypassing Prometheus) for
sub-second latency. The orchestrator uses these readings to decide
when KV-cache pressure requires an eviction.
"""

import asyncio
import aiohttp
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

VLLM_METRICS_URL = "http://localhost:8000/metrics"


@dataclass
class VLLMMetrics:
    """Snapshot of vLLM's KV-cache and request state."""
    gpu_cache_usage_perc: float
    num_requests_running: int
    num_requests_waiting: int
    num_requests_swapped: int
    num_preemptions_total: int
    timestamp: float

    @property
    def under_pressure(self) -> bool:
        return self.num_requests_waiting > 0 or self.num_requests_swapped > 0


def _parse_metric(text: str, name: str) -> float:
    """Extract a metric value from Prometheus text format.

    Scans for lines like:
        vllm:gpu_cache_usage_perc{engine="0",...} 0.45
    and returns the float value.
    """
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.split()[-1])
    return 0.0


class MetricsCollector:
    """Async poller that keeps a fresh snapshot of vLLM metrics.

    Usage:
        collector = MetricsCollector()
        task = asyncio.create_task(collector.poll_loop())
        ...
        snapshot = collector.latest
        if snapshot and snapshot.gpu_cache_usage_perc > 0.90:
            # trigger eviction
    """

    def __init__(
        self,
        metrics_url: str = VLLM_METRICS_URL,
        poll_interval: float = 0.5,
    ):
        self.metrics_url = metrics_url
        self.poll_interval = poll_interval
        self._latest: Optional[VLLMMetrics] = None
        self._stop_event = asyncio.Event()

    @property
    def latest(self) -> Optional[VLLMMetrics]:
        return self._latest

    async def collect_once(self, session: aiohttp.ClientSession) -> VLLMMetrics:
        """Fetch and parse metrics from vLLM."""
        async with session.get(
            self.metrics_url,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            text = await resp.text()

        metrics = VLLMMetrics(
            gpu_cache_usage_perc=_parse_metric(text, "vllm:gpu_cache_usage_perc"),
            num_requests_running=int(_parse_metric(text, "vllm:num_requests_running")),
            num_requests_waiting=int(_parse_metric(text, "vllm:num_requests_waiting")),
            num_requests_swapped=int(_parse_metric(text, "vllm:num_requests_swapped")),
            num_preemptions_total=int(_parse_metric(text, "vllm:num_preemptions_total")),
            timestamp=time.monotonic(),
        )
        self._latest = metrics
        return metrics

    async def poll_loop(self, session: aiohttp.ClientSession):
        """Continuously poll metrics until stopped."""
        logger.info(
            "Metrics collector started (interval=%.1fs, url=%s)",
            self.poll_interval, self.metrics_url,
        )
        while not self._stop_event.is_set():
            try:
                metrics = await self.collect_once(session)
                logger.debug(
                    "cache=%.1f%% running=%d waiting=%d swapped=%d preemptions=%d",
                    metrics.gpu_cache_usage_perc * 100,
                    metrics.num_requests_running,
                    metrics.num_requests_waiting,
                    metrics.num_requests_swapped,
                    metrics.num_preemptions_total,
                )
            except Exception as e:
                logger.warning("Metrics poll failed: %s", e)

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._stop_event.set()
