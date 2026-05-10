"""
Async load generator for vLLM benchmarking.

Sends streaming requests according to configurable workload profiles and
records per-request metrics (TTFT, latency, tokens generated, completion status).
Exposes active request state so the eviction orchestrator can cancel requests.
"""

import asyncio
import aiohttp
import json
import time
import random
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"

TOPICS = [
    "machine learning", "distributed systems", "database indexing",
    "network protocols", "operating system scheduling", "compiler optimization",
    "cryptographic hashing", "load balancing", "cache eviction policies",
    "memory management", "concurrent programming", "API design",
    "microservice architecture", "container orchestration", "GPU computing",
    "natural language processing", "computer vision", "reinforcement learning",
    "consensus algorithms", "stream processing", "query optimization",
    "virtual memory", "file system design", "network security",
]

PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in detail, covering its history, key principles, and modern applications.",
    "Write a comprehensive analysis of {topic}, discussing its advantages, disadvantages, and future prospects.",
    "Describe how {topic} works from first principles, including the underlying mechanisms and practical implications.",
    "Compare and contrast different approaches to {topic}, evaluating their effectiveness in various scenarios.",
    "Discuss the evolution of {topic} over the past decade and predict where it is heading next.",
]

FILLER_PHRASES = [
    "Consider the performance implications and tradeoffs involved.",
    "Analyze the scalability characteristics under increasing load.",
    "Evaluate the reliability guarantees provided by this approach.",
    "Discuss how this interacts with other system components.",
    "Examine the resource utilization patterns that emerge.",
    "Describe the failure modes and recovery mechanisms.",
    "Explain the consistency model and its practical consequences.",
    "Consider the operational complexity of deploying this in production.",
    "Analyze the latency distribution under various workload patterns.",
    "Discuss the memory overhead and how it scales with input size.",
    "Evaluate the throughput characteristics at different concurrency levels.",
    "Examine the tradeoff between simplicity and optimal performance.",
    "Describe monitoring and observability strategies for this system.",
    "Consider the impact of hardware constraints on design decisions.",
    "Analyze how this approach handles bursty traffic patterns.",
]


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt approximately target_tokens long.

    Uses ~1.3 tokens per word as a rough estimate for English text.
    Exact token count doesn't matter — we care about approximate length
    to create the right amount of KV-cache pressure.
    """
    topic = random.choice(TOPICS)
    template = random.choice(PROMPT_TEMPLATES)
    base = template.format(topic=topic)

    target_words = int(target_tokens / 1.3)
    current_words = len(base.split())

    if current_words >= target_words:
        words = base.split()[:target_words]
        return " ".join(words)

    padding_parts = []
    while current_words < target_words:
        phrase = random.choice(FILLER_PHRASES)
        padding_parts.append(phrase)
        current_words += len(phrase.split())

    return base + " " + " ".join(padding_parts)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Metrics collected for a single completed/failed request."""
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    target_max_tokens: int
    ttft: Optional[float]
    total_latency: float
    completed: bool
    evicted: bool = False
    eviction_count: int = 0
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ActiveRequest:
    """Tracks an in-flight request — read by the orchestrator to pick victims."""
    request_id: str
    arrival_time: float
    tokens_generated: int
    max_tokens: int
    prompt_tokens: int
    last_token_time: float
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


# ---------------------------------------------------------------------------
# Workload profiles
# ---------------------------------------------------------------------------

class WorkloadType(str, Enum):
    CHATBOT = "chatbot"
    BATCH = "batch"
    MIXED = "mixed"
    BURST = "burst"


@dataclass
class WorkloadProfile:
    name: str
    prompt_tokens_min: int
    prompt_tokens_max: int
    max_tokens_min: int
    max_tokens_max: int
    concurrency: int
    requests_per_second: float
    duration: int
    burst_size: int = 0

    def sample_prompt_tokens(self) -> int:
        return random.randint(self.prompt_tokens_min, self.prompt_tokens_max)

    def sample_max_tokens(self) -> int:
        return random.randint(self.max_tokens_min, self.max_tokens_max)

    def inter_arrival_time(self) -> float:
        """Poisson (exponential) inter-arrival time in seconds."""
        if self.requests_per_second <= 0:
            return 0.0
        return random.expovariate(self.requests_per_second)


WORKLOAD_PROFILES: Dict[WorkloadType, WorkloadProfile] = {
    WorkloadType.CHATBOT: WorkloadProfile(
        name="chatbot",
        prompt_tokens_min=50,
        prompt_tokens_max=200,
        max_tokens_min=50,
        max_tokens_max=150,
        concurrency=20,
        requests_per_second=2.0,
        duration=300,
    ),
    WorkloadType.BATCH: WorkloadProfile(
        name="batch",
        prompt_tokens_min=500,
        prompt_tokens_max=1500,
        max_tokens_min=200,
        max_tokens_max=500,
        concurrency=10,
        requests_per_second=0.5,
        duration=300,
    ),
    WorkloadType.MIXED: WorkloadProfile(
        name="mixed",
        prompt_tokens_min=50,
        prompt_tokens_max=1000,
        max_tokens_min=50,
        max_tokens_max=400,
        concurrency=15,
        requests_per_second=1.5,
        duration=300,
    ),
    WorkloadType.BURST: WorkloadProfile(
        name="burst",
        prompt_tokens_min=100,
        prompt_tokens_max=500,
        max_tokens_min=100,
        max_tokens_max=300,
        concurrency=35,
        requests_per_second=0.0,
        duration=120,
        burst_size=35,
    ),
}


# ---------------------------------------------------------------------------
# Load generator
# ---------------------------------------------------------------------------

class LoadGenerator:
    """Generates async HTTP load against a vLLM instance.

    The orchestrator reads `active_requests` to decide evictions and calls
    `cancel_request()` to evict a specific request by closing its stream.
    """

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model: str = MODEL_NAME,
        workload: WorkloadType = WorkloadType.CHATBOT,
        duration: int | None = None,
        concurrency: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.profile = WORKLOAD_PROFILES[workload]
        if duration is not None:
            self.profile = WorkloadProfile(
                name=self.profile.name,
                prompt_tokens_min=self.profile.prompt_tokens_min,
                prompt_tokens_max=self.profile.prompt_tokens_max,
                max_tokens_min=self.profile.max_tokens_min,
                max_tokens_max=self.profile.max_tokens_max,
                concurrency=self.profile.concurrency,
                requests_per_second=self.profile.requests_per_second,
                duration=duration,
                burst_size=self.profile.burst_size,
            )
        if concurrency is not None:
            self.profile = WorkloadProfile(
                name=self.profile.name,
                prompt_tokens_min=self.profile.prompt_tokens_min,
                prompt_tokens_max=self.profile.prompt_tokens_max,
                max_tokens_min=self.profile.max_tokens_min,
                max_tokens_max=self.profile.max_tokens_max,
                concurrency=concurrency,
                requests_per_second=self.profile.requests_per_second,
                duration=self.profile.duration,
                burst_size=self.profile.burst_size,
            )
        self.results: List[RequestResult] = []
        self.active_requests: Dict[str, ActiveRequest] = {}
        self._lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._stop_event = asyncio.Event()

    # ---- streaming request ------------------------------------------------

    async def _send_streaming_request(
        self,
        session: aiohttp.ClientSession,
        request_id: str,
        prompt_tokens: int,
        max_tokens: int,
    ) -> RequestResult:
        prompt = generate_prompt(prompt_tokens)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        }

        active = ActiveRequest(
            request_id=request_id,
            arrival_time=time.monotonic(),
            tokens_generated=0,
            max_tokens=max_tokens,
            prompt_tokens=prompt_tokens,
            last_token_time=time.monotonic(),
        )

        async with self._lock:
            self.active_requests[request_id] = active

        start_time = time.time()
        start_mono = time.monotonic()
        ttft = None
        completion_tokens = 0
        error = None
        completed = False
        evicted = False

        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    error = f"HTTP {resp.status}: {await resp.text()}"
                else:
                    async for raw_line in resp.content:
                        if active.cancel_event.is_set():
                            evicted = True
                            break

                        line = raw_line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            completed = True
                            break

                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if ttft is None:
                                    ttft = time.monotonic() - start_mono
                                completion_tokens += 1
                                active.tokens_generated = completion_tokens
                                active.last_token_time = time.monotonic()
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except asyncio.CancelledError:
            evicted = True
        except Exception as e:
            error = str(e)

        end_time = time.time()
        total_latency = time.monotonic() - start_mono

        async with self._lock:
            self.active_requests.pop(request_id, None)

        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            target_max_tokens=max_tokens,
            ttft=ttft,
            total_latency=total_latency,
            completed=completed,
            evicted=evicted,
            error=error,
            start_time=start_time,
            end_time=end_time,
        )

    # ---- single request with concurrency control --------------------------

    async def _run_request(
        self,
        session: aiohttp.ClientSession,
        prompt_tokens: int,
        max_tokens: int,
    ) -> RequestResult:
        request_id = str(uuid.uuid4())[:8]
        async with self._semaphore:
            result = await self._send_streaming_request(
                session, request_id, prompt_tokens, max_tokens,
            )
        self.results.append(result)
        logger.info(
            "req=%s completed=%s tokens=%d/%d ttft=%.3fs latency=%.3fs",
            result.request_id,
            result.completed,
            result.completion_tokens,
            result.target_max_tokens,
            result.ttft or 0,
            result.total_latency,
        )
        return result

    # ---- arrival patterns -------------------------------------------------

    async def _generate_poisson_load(self, session: aiohttp.ClientSession):
        tasks: List[asyncio.Task] = []
        deadline = time.monotonic() + self.profile.duration

        while time.monotonic() < deadline and not self._stop_event.is_set():
            pt = self.profile.sample_prompt_tokens()
            mt = self.profile.sample_max_tokens()
            task = asyncio.create_task(self._run_request(session, pt, mt))
            tasks.append(task)
            await asyncio.sleep(self.profile.inter_arrival_time())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_burst_load(self, session: aiohttp.ClientSession):
        tasks: List[asyncio.Task] = []
        for _ in range(self.profile.burst_size):
            pt = self.profile.sample_prompt_tokens()
            mt = self.profile.sample_max_tokens()
            tasks.append(asyncio.create_task(self._run_request(session, pt, mt)))

        await asyncio.gather(*tasks, return_exceptions=True)

    # ---- public API -------------------------------------------------------

    async def run(self) -> List[RequestResult]:
        """Run the workload and return collected results."""
        self._semaphore = asyncio.Semaphore(self.profile.concurrency)
        self.results = []
        self.active_requests = {}
        self._stop_event.clear()

        logger.info(
            "Starting %s workload: concurrency=%d duration=%ds rps=%.1f",
            self.profile.name,
            self.profile.concurrency,
            self.profile.duration,
            self.profile.requests_per_second,
        )

        connector = aiohttp.TCPConnector(limit=self.profile.concurrency + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            if self.profile.burst_size > 0:
                await self._generate_burst_load(session)
            else:
                await self._generate_poisson_load(session)

        completed = [r for r in self.results if r.completed]
        failed = [r for r in self.results if r.error]
        logger.info(
            "Workload complete: %d total, %d completed, %d failed",
            len(self.results), len(completed), len(failed),
        )
        return self.results

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request (called by the orchestrator to evict).

        Returns True if the request was found and cancelled.
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].cancel_event.set()
            logger.info("Evicted request %s", request_id)
            return True
        return False

    def stop(self):
        """Stop producing new requests (in-flight requests finish)."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

def percentile(sorted_vals: List[float], pct: float) -> float:
    idx = int(len(sorted_vals) * pct)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Load Generator")
    parser.add_argument(
        "--workload", type=str, default="chatbot",
        choices=["chatbot", "batch", "mixed", "burst"],
    )
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--url", type=str, default=VLLM_BASE_URL)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    gen = LoadGenerator(
        base_url=args.url,
        workload=WorkloadType(args.workload),
        duration=args.duration,
        concurrency=args.concurrency,
    )

    results = await gen.run()

    completed = [r for r in results if r.completed]
    ttfts = sorted(r.ttft for r in completed if r.ttft is not None)
    latencies = sorted(r.total_latency for r in completed)
    total_tokens = sum(r.completion_tokens for r in completed)

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.workload} workload ({args.duration}s)")
    print(f"{'='*60}")
    print(f"Total requests:    {len(results)}")
    print(f"Completed:         {len(completed)} ({100*len(completed)/max(len(results),1):.1f}%)")
    print(f"Failed/errored:    {sum(1 for r in results if r.error)}")

    if ttfts:
        print(f"\nTTFT  p50:  {percentile(ttfts, 0.50):.3f}s")
        print(f"TTFT  p95:  {percentile(ttfts, 0.95):.3f}s")
        print(f"TTFT  p99:  {percentile(ttfts, 0.99):.3f}s")

    if latencies:
        print(f"\nE2E   p50:  {percentile(latencies, 0.50):.3f}s")
        print(f"E2E   p95:  {percentile(latencies, 0.95):.3f}s")
        print(f"E2E   p99:  {percentile(latencies, 0.99):.3f}s")

    print(f"\nTotal tokens:      {total_tokens}")
    print(f"Throughput:        {len(completed)/max(args.duration,1):.2f} req/s")
    print(f"Token throughput:  {total_tokens/max(args.duration,1):.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
