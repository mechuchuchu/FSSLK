"""Static equal and throughput-proportional shard allocation baselines."""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

from .simulator import WorkerSpec, default_workers


def equal_split(shard_count: int, workers: Sequence[WorkerSpec]) -> dict[str, int]:
    if shard_count < 0:
        raise ValueError("shard_count must be non-negative")
    if not workers:
        raise ValueError("at least one worker is required")
    if len({worker.name for worker in workers}) != len(workers):
        raise ValueError("worker names must be unique")
    counts = {worker.name: shard_count // len(workers) for worker in workers}
    for worker in workers[: shard_count % len(workers)]:
        counts[worker.name] += 1
    return counts


def proportional_split(
    shard_count: int,
    workers: Sequence[WorkerSpec],
    work_units_per_shard: float = 1000.0,
    input_bytes: int = 196_608,
    result_bytes: int = 100_000,
) -> dict[str, int]:
    """Allocate by inverse predicted compute-plus-transfer time per shard."""
    if shard_count < 0:
        raise ValueError("shard_count must be non-negative")
    if not workers:
        raise ValueError("at least one worker is required")
    if len({worker.name for worker in workers}) != len(workers):
        raise ValueError("worker names must be unique")
    if not math.isfinite(work_units_per_shard) or work_units_per_shard <= 0:
        raise ValueError("work_units_per_shard must be positive")
    if input_bytes < 0 or result_bytes < 0:
        raise ValueError("payload sizes must be non-negative")
    for worker in workers:
        worker.validate()
    if shard_count == 0:
        return {worker.name: 0 for worker in workers}

    predicted_seconds: list[float] = []
    for worker in workers:
        transfer_seconds = 2 * worker.rtt_ms / 1000 + (
            (input_bytes + result_bytes) * 8
            / (worker.bandwidth_mbps * 1_000_000)
        )
        compute_seconds = work_units_per_shard / worker.compute_units_per_second
        predicted_seconds.append(transfer_seconds + compute_seconds)

    weights = [1.0 / seconds for seconds in predicted_seconds]
    total_weight = sum(weights)
    exact_counts = [shard_count * weight / total_weight for weight in weights]
    assigned = [int(value) for value in exact_counts]
    remainder = shard_count - sum(assigned)
    priority = sorted(
        range(len(workers)),
        key=lambda index: (-(exact_counts[index] - assigned[index]), index),
    )
    for index in priority[:remainder]:
        assigned[index] += 1
    return {worker.name: assigned[index] for index, worker in enumerate(workers)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show static shard-allocation baselines.")
    parser.add_argument("--shards", type=int, default=32)
    args = parser.parse_args(argv)
    workers = default_workers()
    report = {
        "workers": [worker.name for worker in workers],
        "equal_split": equal_split(args.shards, workers),
        "proportional": proportional_split(args.shards, workers),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
