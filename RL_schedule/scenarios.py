"""Training and held-out worker conditions for simulator experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .simulator import WorkerSpec, default_workers


def _extend_profiles(
    profiles: Sequence[WorkerSpec], worker_count: int, seed: int
) -> tuple[WorkerSpec, ...]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    workers = list(profiles[:worker_count])
    rng = random.Random(seed)
    while len(workers) < worker_count:
        index = len(workers)
        base = profiles[index % len(profiles)]
        workers.append(
            WorkerSpec(
                name=f"W{index + 1}",
                compute_units_per_second=base.compute_units_per_second
                * rng.uniform(0.8, 1.2),
                rtt_ms=base.rtt_ms * rng.uniform(0.75, 1.35),
                bandwidth_mbps=base.bandwidth_mbps * rng.uniform(0.75, 1.25),
                heartbeat_interval_seconds=rng.choice((0.5, 1.0, 2.0)),
                failure_probability=base.failure_probability,
                stall_probability=base.stall_probability,
                result_loss_probability=base.result_loss_probability,
                invalid_result_probability=base.invalid_result_probability,
                degradation_probability=base.degradation_probability,
                degradation_factor=base.degradation_factor,
            )
        )
    return tuple(workers)


def training_workers(seed: int, worker_count: int = 3) -> tuple[WorkerSpec, ...]:
    """Sample a training condition around the documented worker profiles."""
    rng = random.Random(seed)
    workers: list[WorkerSpec] = []
    profiles = _extend_profiles(default_workers(), worker_count, seed + 379)
    for base in profiles:
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


def evaluation_workers(condition: str, worker_count: int = 3) -> tuple[WorkerSpec, ...]:
    """Return fixed seen or held-out conditions for repeatable evaluation."""
    if condition == "seen":
        return _extend_profiles(default_workers(), worker_count, seed=1103)
    if condition == "unseen":
        profiles = (
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
        return _extend_profiles(profiles, worker_count, seed=2207)
    raise ValueError(f"unknown evaluation condition: {condition}")
