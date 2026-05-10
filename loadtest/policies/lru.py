"""LRU (Least Recently Used) eviction policy.

Evicts the request whose last token was generated the longest ago.

OS analogy: LRU page replacement — the gold standard in OS textbooks.
Optimizes for: interactive latency (active sessions stay protected).
Risk: long-running batch jobs may starve if they pause between tokens.
"""

from typing import Dict
from .base import EvictionPolicy


class LRUPolicy(EvictionPolicy):

    def select_victim(self, active_requests: Dict[str, object]) -> str:
        if not active_requests:
            raise ValueError("No active requests to evict")

        victim = min(active_requests.values(), key=lambda r: r.last_token_time)
        return victim.request_id
