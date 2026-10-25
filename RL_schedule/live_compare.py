"""Repeat and compare live scheduling policies over the same worker workload."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from .live_train import train_live
from .mlp_baseline import BaselineConfig


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "runs": 0,
            "mean": None,
            "sample_stddev": None,
            "ci95_low": None,
            "ci95_high": None,
            "median": None,
            "p95": None,
        }
    average = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(values))
    return {
        "runs": len(values),
        "mean": average,
        "sample_stddev": deviation,
        "ci95_low": average - margin,
        "ci95_high": average + margin,
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
    }


def _write_snapshot(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({record["method"] for record in records})
    by_method: dict[str, Any] = {}
    by_repetition: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record.get("result") is None:
            continue
        method = record["method"]
        metrics = record["result"]["step_metrics"][0]
        makespan = float(metrics["makespan_seconds"])
        by_method.setdefault(method, {"makespans": [], "attempts": [], "duplicates": [], "gradient_errors": []})
        by_method[method]["makespans"].append(makespan)
        by_method[method]["attempts"].append(float(metrics["attempts"]))
        by_method[method]["duplicates"].append(float(metrics["duplicates_started"]))
        by_method[method]["gradient_errors"].append(
            float(metrics["reference_gradient_max_abs_error"])
        )
        by_repetition.setdefault(int(record["repetition"]), {})[method] = {
            "makespan": makespan,
            "metrics": metrics,
        }

    method_summary: dict[str, Any] = {}
    for method in methods:
        values = by_method.get(method, {})
        makespans = values.get("makespans", [])
        method_summary[method] = {
            "successful_runs": len(makespans),
            "failed_runs": sum(
                1 for record in records if record["method"] == method and record.get("result") is None
            ),
            "makespan_seconds": _summary(makespans),
            "mean_attempts": statistics.mean(values["attempts"]) if values.get("attempts") else None,
            "mean_duplicates_started": statistics.mean(values["duplicates"]) if values.get("duplicates") else None,
            "max_gradient_error": max(values["gradient_errors"]) if values.get("gradient_errors") else None,
        }

    comparisons = (
        ("rloo_time_success", "rloo_cost_sensitive"),
        ("rloo_time_success", "adaptive"),
        ("rloo_cost_sensitive", "adaptive"),
        ("rloo_time_success", "fixed"),
    )
    paired_differences: dict[str, Any] = {}
    for left, right in comparisons:
        differences = [
            methods_for_rep[left]["makespan"] - methods_for_rep[right]["makespan"]
            for methods_for_rep in by_repetition.values()
            if left in methods_for_rep and right in methods_for_rep
        ]
        paired_differences[f"{left}_minus_{right}_seconds"] = _summary(differences)

    return {"methods": method_summary, "paired_makespan_differences": paired_differences}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired repeated live tests for old and new RLOO rewards."
    )
    parser.add_argument("--workers", required=True, help="comma-separated worker URLs")
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-loss", type=float, default=0.01)
    parser.add_argument("--fixed-timeout", type=float, default=5.0)
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoints = [url.strip().rstrip("/") for url in args.workers.split(",") if url.strip()]
    if len(endpoints) < 1 or len(set(endpoints)) != len(endpoints):
        parser.error("worker endpoints must be non-empty and unique")
    if args.repetitions < 1:
        parser.error("repetitions must be at least 1")
    if args.old_checkpoint == args.new_checkpoint:
        parser.error("old and new checkpoints must be different")

    config = BaselineConfig(
        samples=args.samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        epochs=1,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
    )
    methods = [
        ("rloo_cost_sensitive", "ppo", args.old_checkpoint),
        ("rloo_time_success", "ppo", args.new_checkpoint),
        ("adaptive", "adaptive", None),
        ("fixed", "fixed", None),
    ]
    rng = random.Random(args.order_seed)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    snapshot: dict[str, Any] = {
        "experiment": "live_reward_objective_comparison",
        "status": "running",
        "repetitions_requested": args.repetitions,
        "worker_count": len(endpoints),
        "workers": endpoints,
        "workload": {
            "samples": args.samples,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "seed": args.seed,
            "steps": 1,
            "shards": len(endpoints),
        },
        "reward_definitions": {
            "rloo_cost_sensitive": "-elapsed - 0.02*network_work - 0.1*duplicate_compute; -120 on failure",
            "rloo_time_success": "-elapsed + 1 for success; no network/duplicate cost or failure penalty",
        },
        "method_order_seed": args.order_seed,
        "runs_completed": 0,
        "runs": records,
    }
    _write_snapshot(args.output, snapshot)

    for repetition in range(1, args.repetitions + 1):
        ordered_methods = list(methods)
        rng.shuffle(ordered_methods)
        for order, (method, scheduler, checkpoint) in enumerate(ordered_methods, start=1):
            record: dict[str, Any] = {
                "repetition": repetition,
                "order_in_repetition": order,
                "method": method,
                "scheduler": scheduler,
                "policy_checkpoint": str(checkpoint) if checkpoint is not None else None,
            }
            try:
                result = train_live(
                    endpoints=endpoints,
                    config=config,
                    steps=1,
                    shard_count=len(endpoints),
                    scheduler_name=scheduler,
                    policy_checkpoint=str(checkpoint) if checkpoint is not None else None,
                    fixed_timeout_seconds=args.fixed_timeout,
                    round_timeout_seconds=args.round_timeout,
                    request_timeout_seconds=args.request_timeout,
                    gradient_tolerance=args.gradient_tolerance,
                    validate_every_step=False,
                )
                record["result"] = result
                metric = result["step_metrics"][0]
                progress = f"makespan={metric['makespan_seconds']:.3f}s attempts={metric['attempts']}"
            except Exception as exc:  # Keep the remaining paired runs after a failed trial.
                record["result"] = None
                record["error"] = f"{type(exc).__name__}: {exc}"
                progress = f"error={record['error']}"
            records.append(record)
            snapshot["runs_completed"] = len(records)
            snapshot["elapsed_seconds"] = time.monotonic() - started
            snapshot["summary"] = _summarize(records)
            _write_snapshot(args.output, snapshot)
            print(
                f"rep={repetition}/{args.repetitions} order={order}/4 "
                f"method={method} {progress}",
                flush=True,
            )

    snapshot["status"] = "complete"
    snapshot["elapsed_seconds"] = time.monotonic() - started
    snapshot["summary"] = _summarize(records)
    _write_snapshot(args.output, snapshot)
    print(json.dumps(snapshot["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
