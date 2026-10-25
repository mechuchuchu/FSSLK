"""Training and held-out worker conditions for simulator experiments."""

from __future__ import annotations

import random

from .simulator import WorkerSpec, default_workers


def training_workers(seed: int) -> tuple[WorkerSpec, ...]:
    """Sample a training condition around the documented worker profiles."""
    rng = random.Random(seed)
    workers: list[WorkerSpec] = []
    for base in default_workers():
        workers.append(
            WorkerSpec(
                name=base.name,
                compute_units_per_second=base.compute_units_per_second
                * rng.uniform(0.65, 1.35),
                rtt_ms=base.rtt_ms * rng.uniform(0.65, 1.5),
                bandwidth_mbps=base.bandwidth_mbps * rng.uniform(0.6, 1.4),
                heartbeat_interval_seconds=rng.choice((0.5, 1.0, 2.0)),
                failure_probability=base.failure_probability,
                stall_probability=base.stall_probability,
                result_loss_probability=base.result_loss_probability,
                degradation_probability=rng.uniform(0.0, 0.35),
                degradation_factor=rng.uniform(0.2, 0.65),
            )
        )
    return tuple(workers)


def evaluation_workers(condition: str) -> tuple[WorkerSpec, ...]:
    """Return fixed seen or held-out conditions for repeatable evaluation."""
    if condition == "seen":
        return default_workers()
    if condition == "unseen":
        return (
            WorkerSpec("A", 72.0, 12.0, 300.0, degradation_probability=0.15),
            WorkerSpec("B", 92.0, 65.0, 35.0, degradation_probability=0.3),
            WorkerSpec(
                "C",
                18.0,
                150.0,
                6.0,
                failure_probability=0.08,
                stall_probability=0.12,
                result_loss_probability=0.05,
                degradation_probability=0.4,
                degradation_factor=0.25,
            ),
        )
    raise ValueError(f"unknown evaluation condition: {condition}")
