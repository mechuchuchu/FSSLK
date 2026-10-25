"""Evaluate reference scheduling policies over seeded simulator episodes."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Sequence

from .policies import PolicyName, select_action
from .scenarios import evaluation_workers
from .simulator import SchedulingEnv, WorkerSpec


def run_episode(
    policy: PolicyName,
    seed: int,
    fixed_timeout_seconds: float = 5.0,
    history_length: int = 4,
    workers: Sequence[WorkerSpec] | None = None,
) -> dict[str, Any]:
    env = SchedulingEnv(workers=workers, history_length=history_length)
    observation, mask = env.reset(seed)
    while not env.done:
        action = select_action(
            policy,
            env,
            observation,
            mask,
            fixed_timeout_seconds=fixed_timeout_seconds,
        )
        observation, _, _, mask, _ = env.step(action)
    return env.metrics()


def evaluate_policies(
    episodes: int,
    seed: int = 0,
    fixed_timeout_seconds: float = 5.0,
    condition: str = "seen",
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if fixed_timeout_seconds <= 0:
        raise ValueError("fixed_timeout_seconds must be positive")

    reports: dict[str, Any] = {}
    workers = evaluation_workers(condition)
    policies: tuple[PolicyName, ...] = (
        "wait_only",
        "fixed_timeout",
        "adaptive_heuristic",
    )
    for policy in policies:
        runs = [
            run_episode(
                policy,
                seed + episode,
                fixed_timeout_seconds=fixed_timeout_seconds,
                workers=workers,
            )
            for episode in range(episodes)
        ]
        successful = [run for run in runs if run["success"]]
        makespans = [float(run["makespan_seconds"]) for run in successful]
        reports[policy] = {
            "episodes": episodes,
            "completion_rate": len(successful) / episodes,
            "mean_makespan_seconds": statistics.fmean(makespans) if makespans else None,
            "makespan_stddev_seconds": statistics.pstdev(makespans) if len(makespans) > 1 else 0.0,
            "mean_network_work_seconds": statistics.fmean(
                float(run["network_work_seconds"]) for run in runs
            ),
            "mean_network_bytes_transferred": statistics.fmean(
                float(run["network_bytes_transferred"]) for run in runs
            ),
            "mean_duplicate_compute_seconds": statistics.fmean(
                float(run["duplicate_compute_seconds"]) for run in runs
            ),
            "mean_duplicates_started": statistics.fmean(
                float(run["duplicates_started"]) for run in runs
            ),
            "mean_failed_copies": statistics.fmean(
                float(run["failed_copies"]) for run in runs
            ),
            "mean_lost_results": statistics.fmean(
                float(run["lost_results"]) for run in runs
            ),
            "mean_invalid_results": statistics.fmean(
                float(run["invalid_results"]) for run in runs
            ),
            "mean_late_results_ignored": statistics.fmean(
                float(run["late_results_ignored"]) for run in runs
            ),
            "mean_recovery_time_seconds": statistics.fmean(
                float(run["recovery_time_seconds"]) for run in runs
            ),
        }
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare reference policies in the worker/link simulator."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-timeout", type=float, default=5.0)
    parser.add_argument("--condition", choices=("seen", "unseen"), default="seen")
    args = parser.parse_args(argv)
    result = evaluate_policies(
        args.episodes, args.seed, args.fixed_timeout, condition=args.condition
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
