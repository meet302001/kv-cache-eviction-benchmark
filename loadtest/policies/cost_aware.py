"""Cost-Aware (Least Work Done) eviction policy.

Evicts the request that has generated the fewest tokens so far,
minimizing wasted GPU compute from reprocessing.

OS analogy: No direct equivalent — OS page eviction preserves data on
disk, but LLM KV-cache eviction destroys computed state. The cost of
eviction is proportional to tokens already generated, so evicting the
cheapest request minimizes total waste.

Optimizes for: minimum wasted compute.
Risk: may keep expensive long-running requests too long, blocking new ones.
"""

from typing import Dict
from .base import EvictionPolicy


class CostAwarePolicy(EvictionPolicy):

    def select_victim(self, active_requests: Dict[str, object]) -> str:
        if not active_requests:
            raise ValueError("No active requests to evict")

        victim = min(active_requests.values(), key=lambda r: r.tokens_generated)
        return victim.request_id
