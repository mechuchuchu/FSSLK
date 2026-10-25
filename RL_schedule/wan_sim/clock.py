"""Monotonic virtual clock used by the discrete-event simulator."""

from dataclasses import dataclass


@dataclass
class VirtualClock:
    now: float = 0.0

    def advance_to(self, time_seconds: float) -> float:
        if time_seconds < self.now:
            raise ValueError("virtual time cannot move backwards")
        elapsed = time_seconds - self.now
        self.now = time_seconds
        return elapsed
