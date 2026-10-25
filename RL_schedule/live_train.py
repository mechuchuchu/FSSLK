"""Distributed MLP training driven by live WAIT/DUPLICATE decisions."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .distributed_train import _local_full_gradient, _request_payload, _wait_for_workers
from .live_scheduler import LiveAttempt, LiveObservationHistory, LivePolicy
from .mlp_baseline import BaselineConfig, make_dataset, make_model
from .result_registry import GradientResultRegistry
from .scenarios import evaluation_workers
from .simulator import SimulationConfig
from .tensor_wire import decode_float32


def _http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker request failed at {url}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"worker returned a non-object response at {url}")
    return result


def _task_url(endpoint: str, route: str, task_id: str) -> str:
    return f"{endpoint.rstrip('/')}/{route}/{urllib.parse.quote(task_id, safe='')}"


def _submit(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = _http_json(
        f"{endpoint.rstrip('/')}/submit",
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    if not response.get("ok"):
        raise RuntimeError(f"worker rejected task {payload.get('task_id')}")
    return response


def _poll_statuses(
    endpoints: Sequence[str], attempts: Sequence[LiveAttempt], timeout: float
) -> None:
    pending = [attempt for attempt in attempts if attempt.active]
    if not pending:
        return
    poll_timeout = min(timeout, 5.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futures = {
            pool.submit(
                _http_json,
                _task_url(endpoints[attempt.worker_index], "status", attempt.task_id),
                "GET",
                None,
                poll_timeout,
            ): attempt
            for attempt in pending
        }
        for future, attempt in futures.items():
            try:
                status = future.result()
            except (OSError, RuntimeError, ValueError) as exc:
                if "unknown task" in str(exc).lower():
                    attempt.status = "failed"
                # A dropped status request does not prove that the worker stopped.
                # Keep it reserved so the coordinator cannot overload it with a copy.
                attempt.error = str(exc)
                if attempt.active:
                    attempt.heartbeat_age_seconds = max(
                        attempt.heartbeat_age_seconds, 2.0
                    )
                continue
            if status.get("task_id") != attempt.task_id:
                attempt.error = "worker returned status for another task"
                attempt.heartbeat_age_seconds = max(attempt.heartbeat_age_seconds, 2.0)
                continue
            reported_status = str(status.get("status", ""))
            if reported_status not in {"queued", "running", "completed", "cancelled", "failed"}:
                attempt.error = "worker returned an unknown task status"
                attempt.heartbeat_age_seconds = max(attempt.heartbeat_age_seconds, 2.0)
                continue
            try:
                progress = float(status.get("progress", 0.0))
                heartbeat_age = float(status.get("heartbeat_age_seconds", 0.0))
                if not math.isfinite(progress) or not math.isfinite(heartbeat_age):
                    raise ValueError("worker returned non-finite task telemetry")
            except (TypeError, ValueError):
                attempt.error = "worker returned invalid task telemetry"
                attempt.heartbeat_age_seconds = max(attempt.heartbeat_age_seconds, 2.0)
                continue
            attempt.status = reported_status
            attempt.progress = min(1.0, max(0.0, progress))
            attempt.heartbeat_age_seconds = max(0.0, heartbeat_age)
            attempt.error = status.get("error")


def _get_result(
    endpoint: str, attempt: LiveAttempt, timeout: float
) -> dict[str, Any]:
    return _http_json(
        _task_url(endpoint, "result", attempt.task_id), timeout=timeout
    )


def _forget(endpoint: str, task_id: str, timeout: float) -> None:
    timeout = min(timeout, 5.0)
    try:
        _http_json(
            _task_url(endpoint, "forget", task_id),
            method="POST",
            payload={},
            timeout=timeout,
        )
    except (OSError, RuntimeError, ValueError):
        return


def _cancel(endpoint: str, task_id: str, timeout: float) -> None:
    timeout = min(timeout, 5.0)
    try:
        _http_json(
            _task_url(endpoint, "cancel", task_id),
            method="POST",
            payload={},
            timeout=timeout,
        )
    except (OSError, RuntimeError, ValueError):
        return


def _attempts_by_worker(
    workers_count: int, attempts: Sequence[LiveAttempt], target_shard: int
) -> tuple[list[LiveAttempt | None], dict[int, LiveAttempt]]:
    active: list[LiveAttempt | None] = [None] * workers_count
    latest_target: dict[int, LiveAttempt] = {}
    for attempt in attempts:
        if attempt.shard_id == target_shard:
            existing = latest_target.get(attempt.worker_index)
            if existing is None or attempt.attempt_id > existing.attempt_id:
                latest_target[attempt.worker_index] = attempt
        if attempt.active:
            if active[attempt.worker_index] is not None:
                raise RuntimeError("coordinator assigned two active jobs to one worker")
            active[attempt.worker_index] = attempt
    return active, latest_target


def _choose_straggler(
    pending_shards: Sequence[int], attempts_by_shard: dict[int, list[LiveAttempt]], now: float
) -> int:
    def score(shard_id: int) -> tuple[int, float, int]:
        active = [attempt for attempt in attempts_by_shard[shard_id] if attempt.active]
        if not active:
            return (1, now, -shard_id)
        least_progress = min(attempt.progress for attempt in active)
        oldest_start = min(attempt.start_time for attempt in active)
        return (0, oldest_start + (1.0 - least_progress), -shard_id)

    return max(pending_shards, key=score)


def _cancel_and_cleanup(
    endpoints: Sequence[str],
    attempts: Sequence[LiveAttempt],
    timeout: float,
    grace_seconds: float = 5.0,
) -> tuple[set[str], int]:
    late_result_ids: set[str] = set()
    pending = [attempt for attempt in attempts if attempt.active]
    for attempt in pending:
        _cancel(endpoints[attempt.worker_index], attempt.task_id, timeout)
    deadline = time.monotonic() + grace_seconds
    while pending and time.monotonic() < deadline:
        _poll_statuses(endpoints, pending, timeout)
        newly_terminal = [attempt for attempt in pending if not attempt.active]
        for attempt in newly_terminal:
            if attempt.status == "completed":
                late_result_ids.add(attempt.task_id)
            _forget(endpoints[attempt.worker_index], attempt.task_id, timeout)
        pending = [attempt for attempt in pending if attempt.active]
        if pending:
            time.sleep(0.1)
    for attempt in attempts:
        if not attempt.active:
            _forget(endpoints[attempt.worker_index], attempt.task_id, timeout)
    return late_result_ids, len(pending)


def _run_gradient_round(
    endpoints: Sequence[str],
    run_id: str,
    config: BaselineConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_version: int,
    shard_count: int,
    policy: LivePolicy,
    timeout: float,
    round_timeout: float,
    max_attempts_per_shard: int,
    validate_reference: dict[str, torch.Tensor] | None,
    reference_loss: float | None,
    gradient_tolerance: float,
) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    if shard_count < 1 or shard_count > len(endpoints):
        raise ValueError("shard_count must be between 1 and worker count")
    index_shards = torch.tensor_split(torch.arange(config.samples), shard_count)
    shard_sizes = [len(indices) for indices in index_shards]
    parameter_shapes = {
        name: parameter.shape for name, parameter in model.named_parameters()
    }
    registry = GradientResultRegistry(
        model_version=model_version,
        expected_shards=shard_count,
        parameter_shapes=parameter_shapes,
    )
    attempts_by_shard: dict[int, list[LiveAttempt]] = {
        shard_id: [] for shard_id in range(shard_count)
    }
    attempt_counters = [0] * shard_count
    shard_losses: dict[int, float] = {}
    processed_task_ids: set[str] = set()
    accepted_task_ids: set[str] = set()
    history = policy.history
    for shard_id in range(shard_count):
        policy.reset_shard(shard_id)

    round_started = time.monotonic()
    state_dict = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def dispatch(shard_id: int, worker_index: int) -> None:
        attempt_id = attempt_counters[shard_id]
        attempt_counters[shard_id] += 1
        task_id = f"{run_id}:{model_version}:{shard_id}:{attempt_id}"
        payload = _request_payload(
            config,
            state_dict,
            inputs[index_shards[shard_id]],
            targets[index_shards[shard_id]],
            model_version=model_version,
            shard_id=shard_id,
            attempt_id=attempt_id,
        )
        payload["batch_size"] = config.batch_size
        payload["task_id"] = task_id
        attempt = LiveAttempt(
            shard_id=shard_id,
            worker_index=worker_index,
            attempt_id=attempt_id,
            task_id=task_id,
            start_time=time.monotonic() - round_started,
        )
        attempts_by_shard[shard_id].append(attempt)
        _submit(endpoints[worker_index], payload, timeout)

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=shard_count
        ) as pool:
            initial_submissions = [
                pool.submit(dispatch, shard_id, shard_id)
                for shard_id in range(shard_count)
            ]
            for future in initial_submissions:
                future.result()

        invalid_responses = 0
        ignored_result_ids: set[str] = set()
        action_counts = {"wait": 0, "duplicate": 0}
        while not registry.complete():
            now = time.monotonic() - round_started
            if now >= round_timeout:
                raise TimeoutError(
                    f"model version {model_version} exceeded its {round_timeout:g}s round timeout"
                )

            all_attempts = [
                attempt
                for shard_attempts in attempts_by_shard.values()
                for attempt in shard_attempts
            ]
            _poll_statuses(endpoints, all_attempts, timeout)
            for shard_id, shard_attempts in attempts_by_shard.items():
                for attempt in shard_attempts:
                    if (
                        attempt.status != "completed"
                        or attempt.task_id in processed_task_ids
                    ):
                        continue
                    processed_task_ids.add(attempt.task_id)
                    if shard_id in registry.results:
                        ignored_result_ids.add(attempt.task_id)
                        registry.duplicate_results_ignored += 1
                        _forget(endpoints[attempt.worker_index], attempt.task_id, timeout)
                        continue
                    try:
                        response = _get_result(
                            endpoints[attempt.worker_index], attempt, timeout
                        )
                        if (
                            int(response.get("model_version", -1)) != model_version
                            or int(response.get("shard_id", -1)) != shard_id
                            or int(response.get("attempt_id", -1)) != attempt.attempt_id
                            or response.get("task_id") != attempt.task_id
                            or int(response.get("samples", -1)) != shard_sizes[shard_id]
                        ):
                            registry.stale_results_rejected += 1
                            invalid_responses += 1
                            continue
                        encoded = response.get("gradients")
                        if not isinstance(encoded, dict):
                            registry.invalid_results_rejected += 1
                            invalid_responses += 1
                            continue
                        gradients = {
                            name: decode_float32(encoded[name])
                            for name in parameter_shapes
                            if name in encoded
                        }
                        if set(encoded) != set(parameter_shapes) or set(gradients) != set(parameter_shapes):
                            registry.invalid_results_rejected += 1
                            invalid_responses += 1
                            continue
                        shard_loss = float(response["mean_squared_error"])
                        if not math.isfinite(shard_loss) or shard_loss < 0:
                            registry.invalid_results_rejected += 1
                            invalid_responses += 1
                            continue
                        accepted = registry.accept(
                            model_version=model_version,
                            shard_id=shard_id,
                            task_id=attempt.task_id,
                            gradients=gradients,
                        )
                        if accepted:
                            shard_losses[shard_id] = shard_loss
                            accepted_task_ids.add(attempt.task_id)
                        else:
                            invalid_responses += 1
                        _forget(endpoints[attempt.worker_index], attempt.task_id, timeout)
                    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                        registry.invalid_results_rejected += 1
                        invalid_responses += 1
                        attempt.status = "failed"
                        attempt.error = "worker result failed validation or retrieval"
                        _forget(endpoints[attempt.worker_index], attempt.task_id, timeout)

            for shard_id, shard_attempts in attempts_by_shard.items():
                if shard_id not in registry.results:
                    continue
                for attempt in shard_attempts:
                    if attempt.active:
                        _cancel(endpoints[attempt.worker_index], attempt.task_id, timeout)

            if registry.complete():
                break

            pending_shards = [
                shard_id for shard_id in range(shard_count) if shard_id not in registry.results
            ]
            target_shard = _choose_straggler(
                pending_shards,
                attempts_by_shard,
                time.monotonic() - round_started,
            )
            all_attempts = [
                attempt
                for shard_attempts in attempts_by_shard.values()
                for attempt in shard_attempts
            ]
            worker_current, latest_for_shard = _attempts_by_worker(
                len(endpoints), all_attempts, target_shard
            )
            shard_attempts = attempts_by_shard[target_shard]
            observation = history.observe(
                target_shard,
                time.monotonic() - round_started,
                shard_attempts,
                latest_for_shard,
            )
            busy_workers = [attempt is not None for attempt in worker_current]
            action_mask = torch.zeros(3 + len(endpoints), dtype=torch.bool)
            if any(attempt.active for attempt in all_attempts):
                action_mask[:3] = True
            if len(shard_attempts) < max_attempts_per_shard:
                for worker_index, busy in enumerate(busy_workers):
                    if not busy:
                        action_mask[3 + worker_index] = True
            now = time.monotonic() - round_started
            action = policy.act(
                observation,
                action_mask,
                now,
                shard_attempts,
                shard_id=target_shard,
            )
            action_valid = (
                0 <= action < action_mask.numel() and bool(action_mask[action])
            )
            if not action_valid:
                available = [
                    index
                    for index in range(len(endpoints))
                    if bool(action_mask[3 + index])
                ]
                if available:
                    action = 3 + available[0]
                elif bool(action_mask[:3].any()):
                    action = min(
                        range(3), key=lambda index: policy.config.wait_seconds[index]
                    )
                else:
                    raise RuntimeError("live scheduler has no valid action")
            if action < 3:
                action_counts["wait"] += 1
                delay = policy.config.wait_seconds[action]
                time.sleep(min(delay, max(0.0, round_timeout - now)))
                continue
            if action >= 3:
                worker_index = action - 3
                if (
                    worker_index >= len(endpoints)
                    or busy_workers[worker_index]
                    or not bool(action_mask[action])
                ):
                    raise RuntimeError("live scheduler selected an unavailable worker")
                action_counts["duplicate"] += 1
                dispatch(target_shard, worker_index)

        results_complete_at = time.monotonic()
        aggregation_started = time.monotonic()
        aggregated = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
        }
        weighted_loss = 0.0
        for shard_id, gradients in registry.results.items():
            weight = shard_sizes[shard_id] / config.samples
            weighted_loss += shard_losses[shard_id] * weight
            for name, gradient in gradients.items():
                aggregated[name].add_(gradient, alpha=weight)

        max_abs_error: float | None = None
        l2_error: float | None = None
        reference_loss_difference: float | None = None
        if validate_reference is not None and reference_loss is not None:
            differences = [
                aggregated[name] - validate_reference[name] for name in aggregated
            ]
            max_abs_error = max(float(value.abs().max().item()) for value in differences)
            l2_error = math.sqrt(
                sum(float(value.square().sum().item()) for value in differences)
            )
            reference_loss_difference = abs(weighted_loss - reference_loss)
            if max_abs_error > gradient_tolerance:
                raise RuntimeError(
                    f"live distributed gradient error {max_abs_error:.6g} exceeds "
                    f"tolerance {gradient_tolerance:.6g}"
                )
        gradient_aggregation_seconds = time.monotonic() - aggregation_started

    except Exception:
        active_round_attempts = [
            attempt
            for shard_attempts in attempts_by_shard.values()
            for attempt in shard_attempts
        ]
        _cancel_and_cleanup(endpoints, active_round_attempts, timeout)
        raise

    cleanup_late_result_ids, pending_cleanup = _cancel_and_cleanup(
        endpoints,
        [
            attempt
            for shard_attempts in attempts_by_shard.values()
            for attempt in shard_attempts
        ],
        timeout,
    )
    newly_ignored_ids = cleanup_late_result_ids - ignored_result_ids
    ignored_result_ids.update(cleanup_late_result_ids)
    for shard_id, shard_attempts in attempts_by_shard.items():
        if shard_id not in registry.results:
            continue
        for attempt in shard_attempts:
            if (
                attempt.status == "completed"
                and attempt.task_id not in accepted_task_ids
                and attempt.task_id not in processed_task_ids
            ):
                newly_ignored_ids.add(attempt.task_id)
                ignored_result_ids.add(attempt.task_id)
    registry.duplicate_results_ignored += len(newly_ignored_ids)
    late_results_ignored = len(ignored_result_ids)
    if pending_cleanup:
        raise RuntimeError(
            f"could not confirm cancellation of {pending_cleanup} worker task(s)"
        )
    metrics = {
        "model_version": model_version,
        "makespan_seconds": results_complete_at - round_started,
        "gradient_aggregation_seconds": gradient_aggregation_seconds,
        "shard_count": shard_count,
        "shard_sizes": shard_sizes,
        "attempts": sum(len(items) for items in attempts_by_shard.values()),
        "duplicates_started": max(
            0, sum(len(items) for items in attempts_by_shard.values()) - shard_count
        ),
        "wait_actions": action_counts["wait"],
        "duplicate_actions": action_counts["duplicate"],
        "late_results_ignored": late_results_ignored,
        "pending_worker_cancellations": pending_cleanup,
        "invalid_results_rejected": registry.invalid_results_rejected,
        "invalid_responses": invalid_responses,
        "stale_results_rejected": registry.stale_results_rejected,
        "duplicate_results_ignored": registry.duplicate_results_ignored,
        "weighted_pre_update_loss": weighted_loss,
        "reference_gradient_max_abs_error": max_abs_error,
        "reference_gradient_l2_error": l2_error,
        "reference_loss_difference": reference_loss_difference,
    }
    return aggregated, weighted_loss, metrics


def train_live(
    endpoints: Sequence[str],
    config: BaselineConfig,
    steps: int,
    shard_count: int,
    scheduler_name: str,
    policy_checkpoint: str | None,
    fixed_timeout_seconds: float,
    round_timeout_seconds: float,
    request_timeout_seconds: float,
    gradient_tolerance: float,
    validate_every_step: bool,
) -> dict[str, Any]:
    config.validate()
    if not endpoints:
        raise ValueError("at least one worker endpoint is required")
    endpoints = tuple(endpoint.rstrip("/") for endpoint in endpoints)
    if any(not endpoint for endpoint in endpoints):
        raise ValueError("worker endpoints must not be empty")
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("worker endpoints must be unique")
    if len(endpoints) > config.samples:
        raise ValueError("worker count cannot exceed the number of samples")
    if shard_count < 1 or shard_count > len(endpoints):
        raise ValueError("shard count must be between 1 and worker count")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    for name, value in (
        ("fixed_timeout_seconds", fixed_timeout_seconds),
        ("round_timeout_seconds", round_timeout_seconds),
        ("request_timeout_seconds", request_timeout_seconds),
        ("gradient_tolerance", gradient_tolerance),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    profiles = evaluation_workers("seen", len(endpoints))
    sim_config = SimulationConfig()
    policy = LivePolicy(
        scheduler_name,
        profiles,
        sim_config,
        checkpoint=policy_checkpoint,
        fixed_timeout_seconds=fixed_timeout_seconds,
    )
    _wait_for_workers(endpoints, timeout=30.0)
    torch.set_num_threads(1)
    inputs, targets = make_dataset(config)
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    round_metrics: list[dict[str, Any]] = []
    target_step: int | None = None
    total_started = time.monotonic()
    run_id = uuid.uuid4().hex

    for step in range(1, steps + 1):
        local_reference: dict[str, torch.Tensor] | None = None
        reference_loss: float | None = None
        if step == 1 or validate_every_step:
            reference_loss, local_reference = _local_full_gradient(model, inputs, targets)
        gradients, remote_loss, metrics = _run_gradient_round(
            endpoints=endpoints,
            run_id=run_id,
            config=config,
            model=model,
            inputs=inputs,
            targets=targets,
            model_version=step - 1,
            shard_count=shard_count,
            policy=policy,
            timeout=request_timeout_seconds,
            round_timeout=round_timeout_seconds,
            max_attempts_per_shard=len(endpoints) * 2,
            validate_reference=local_reference,
            reference_loss=reference_loss,
            gradient_tolerance=gradient_tolerance,
        )
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        optimizer.step()
        with torch.no_grad():
            train_loss = float(F.mse_loss(model(inputs), targets).item())
        metrics["remote_pre_update_loss"] = remote_loss
        metrics["local_loss_after_update"] = train_loss
        metrics["step"] = step
        round_metrics.append(metrics)
        if train_loss <= config.target_loss:
            target_step = step
            break

    final_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        final_digest.update(name.encode("utf-8") + b"\0")
        final_digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return {
        "scheduler": scheduler_name,
        "run_id": run_id,
        "policy_checkpoint": policy_checkpoint,
        "config": {
            "samples": config.samples,
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "batch_size": config.batch_size,
            "dataset_seed": config.seed,
            "model_seed": config.seed + 1,
            "learning_rate": config.learning_rate,
            "target_loss": config.target_loss,
        },
        "workers": list(endpoints),
        "shard_count": shard_count,
        "steps_completed": len(round_metrics),
        "final_train_loss": round_metrics[-1]["local_loss_after_update"],
        "target_reached": target_step is not None,
        "target_loss_step": target_step,
        "elapsed_seconds": time.monotonic() - total_started,
        "final_model_sha256": final_digest.hexdigest(),
        "step_metrics": round_metrics,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the fixed MLP with a live worker scheduling policy."
    )
    parser.add_argument("--workers", required=True, help="comma-separated worker URLs")
    parser.add_argument("--scheduler", choices=("adaptive", "fixed", "ppo"), default="adaptive")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="number of samples per worker mini-batch within each shard",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--shards", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-loss", type=float, default=0.01)
    parser.add_argument("--fixed-timeout", type=float, default=5.0)
    parser.add_argument("--round-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-5)
    parser.add_argument("--validate-every-step", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    endpoints = [url.strip() for url in args.workers.split(",") if url.strip()]
    config = BaselineConfig(
        samples=args.samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        epochs=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
    )
    try:
        result = train_live(
            endpoints=endpoints,
            config=config,
            steps=args.steps,
            shard_count=(args.shards if args.shards is not None else len(endpoints)),
            scheduler_name=args.scheduler,
            policy_checkpoint=(
                str(args.policy_checkpoint) if args.policy_checkpoint is not None else None
            ),
            fixed_timeout_seconds=args.fixed_timeout,
            round_timeout_seconds=args.round_timeout,
            request_timeout_seconds=args.request_timeout,
            gradient_tolerance=args.gradient_tolerance,
            validate_every_step=args.validate_every_step,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
