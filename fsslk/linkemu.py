"""Frame-level link emulation, for running a whole cluster on one machine.

The docker-compose setup shapes each worker's network with the kernel's netem.
That needs a kernel with netem, which not every host has (a minimal VM kernel
may not). ``LinkEmulation`` gives a worker the same two effects at the protocol
layer instead: every frame it sends or receives is held for the one-way delay
plus its serialization time at the configured rate.

It does **not** emulate packet loss (TCP hides loss from an application, only
a packet-level shaper can reproduce its effect) and it treats the link as
store-and-forward per frame. Use real netem when loss matters.

Configure a worker with the environment variables
``FSSLK_LINK_DELAY_MS`` (one way, milliseconds) and ``FSSLK_LINK_RATE_MBPS``
(megabits per second, each direction). With neither set it does nothing.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Mapping

# Rough size of a frame header (length prefix plus JSON metadata).
FRAME_OVERHEAD_BYTES = 128


@dataclass(frozen=True)
class LinkEmulation:
    delay_ms: float = 0.0  # one way
    rate_mbps: float | None = None  # per direction; None means unlimited

    def __post_init__(self) -> None:
        if not math.isfinite(self.delay_ms) or self.delay_ms < 0:
            raise ValueError("link delay must be a finite, non-negative number of milliseconds")
        if self.rate_mbps is not None and (
            not math.isfinite(self.rate_mbps) or self.rate_mbps <= 0
        ):
            raise ValueError("link rate must be a finite, positive number of Mbps")

    @property
    def enabled(self) -> bool:
        return self.delay_ms > 0 or self.rate_mbps is not None

    def frame_seconds(self, payload_bytes: int) -> float:
        """Time one frame takes to cross the emulated link in one direction."""
        seconds = self.delay_ms / 1000
        if self.rate_mbps is not None:
            seconds += (payload_bytes + FRAME_OVERHEAD_BYTES) * 8 / (self.rate_mbps * 1_000_000)
        return seconds

    def wait(self, payload_bytes: int) -> None:
        if self.enabled:
            time.sleep(self.frame_seconds(payload_bytes))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LinkEmulation":
        environ = os.environ if environ is None else environ
        delay = environ.get("FSSLK_LINK_DELAY_MS")
        rate = environ.get("FSSLK_LINK_RATE_MBPS")
        return cls(
            delay_ms=float(delay) if delay else 0.0,
            rate_mbps=float(rate) if rate else None,
        )
