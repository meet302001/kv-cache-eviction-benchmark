"""Abstract base class for KV-cache eviction policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from load_generator import ActiveRequest


class EvictionPolicy(ABC):
    """Interface that every eviction policy must implement.

    The orchestrator calls select_victim() with a dict of all in-flight
    requests. The policy returns the request_id of the request to evict.

    This mirrors OS page replacement: given a set of pages in physical
    memory, which one do you swap out?
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def select_victim(self, active_requests: Dict[str, ActiveRequest]) -> str:
        """Choose which request to evict.

        Args:
            active_requests: Map of request_id -> ActiveRequest for all
                currently in-flight requests.

        Returns:
            The request_id of the victim to evict.

        Raises:
            ValueError: If active_requests is empty.
        """
        ...
