"""Sequential node recovery stages used by the event engine."""

from __future__ import annotations

from .events import EventQueue
from .types import RecoveryConfig


class RecoveryModel:
    def __init__(self, config: RecoveryConfig) -> None:
        config.validate()
        self.config = config

    def schedule(self, queue: EventQueue, now: float, node_id: str, generation: int) -> None:
        queue.push(
            now + self.config.detection_seconds,
            "node_detected",
            target=node_id,
            priority=0,
            generation=generation,
        )
