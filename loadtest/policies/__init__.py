from .base import EvictionPolicy
from .fifo import FIFOPolicy
from .lru import LRUPolicy
from .sjf import SJFPolicy
from .cost_aware import CostAwarePolicy
from .random_policy import RandomPolicy

POLICIES = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "sjf": SJFPolicy,
    "cost_aware": CostAwarePolicy,
    "random": RandomPolicy,
}

__all__ = [
    "EvictionPolicy",
    "FIFOPolicy",
    "LRUPolicy",
    "SJFPolicy",
    "CostAwarePolicy",
    "RandomPolicy",
    "POLICIES",
]
