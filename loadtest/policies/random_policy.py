"""Random eviction policy (baseline).

Evicts a random request. Serves as the baseline to answer:
"Is a sophisticated eviction policy worth the complexity?"

OS research insight: random page replacement is surprisingly competitive
with LRU in many real-world workloads — often within 10% of optimal.
"""

import random
from typing import Dict
from .base import EvictionPolicy


class RandomPolicy(EvictionPolicy):

    def select_victim(self, active_requests: Dict[str, object]) -> str:
        if not active_requests:
            raise ValueError("No active requests to evict")

        return random.choice(list(active_requests.keys()))
