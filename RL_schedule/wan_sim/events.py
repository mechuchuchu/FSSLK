"""Stable priority queue for external faults and internal completion events."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True, frozen=True)
class Event:
    time_seconds: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    target: str = field(compare=False, default="")
    generation: int = field(compare=False, default=0)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._sequence = 0

    def push(
        self,
        time_seconds: float,
        kind: str,
        *,
        target: str = "",
        priority: int = 10,
        generation: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        if time_seconds < 0:
            raise ValueError("event time must be non-negative")
        event = Event(
            float(time_seconds),
            priority,
            self._sequence,
            kind,
            target,
            generation,
            {} if payload is None else dict(payload),
        )
        self._sequence += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
