"""FIFO (First In, First Out) eviction policy.

Evicts the oldest request — the one that arrived first.

OS analogy: FIFO page replacement.
Optimizes for: fairness (requests served in arrival order).
Risk: may evict a request that's 95% done, wasting significant compute.
"""

from typing import Dict
from .base import EvictionPolicy


class FIFOPolicy(EvictionPolicy):

    def select_victim(self, active_requests: Dict[str, object]) -> str:
        if not active_requests:
            raise ValueError("No active requests to evict")

        victim = min(active_requests.values(), key=lambda r: r.arrival_time)
        return victim.request_id
