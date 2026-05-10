"""SJF (Shortest Job First / Most-Remaining) eviction policy.

Evicts the request with the most tokens still remaining to generate.
This frees KV-cache space and lets shorter jobs finish first.

OS analogy: Shortest Job First CPU scheduling (inverted — evict longest).
Optimizes for: throughput (maximizes completed requests per second).
Risk: long requests get perpetually starved and never finish.
"""

from typing import Dict
from .base import EvictionPolicy


class SJFPolicy(EvictionPolicy):

    def select_victim(self, active_requests: Dict[str, object]) -> str:
        if not active_requests:
            raise ValueError("No active requests to evict")

        victim = max(
            active_requests.values(),
            key=lambda r: r.max_tokens - r.tokens_generated,
        )
        return victim.request_id
