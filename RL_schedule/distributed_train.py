"""Synchronous MLP training with real HTTP worker processes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .mlp_baseline import BaselineConfig, make_dataset, make_model
from .result_registry import GradientResultRegistry
from .tensor_wire import decode_float32, encode_float32


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError(f"worker returned a non-object response from {url}")
    return result


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/gradient",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker {url} rejected the gradient request: {detail}") from exc
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"worker {url} returned an invalid gradient response")
    return result


def _wait_for_workers(workers: Sequence[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    pending = set(workers)
    last_errors: dict[str, str] = {}
    while pending and time.monotonic() < deadline:
        for worker in tuple(pending):
            try:
                response = _get_json(f"{worker.rstrip('/')}/health", timeout=1.0)
                if response.get("ok"):
                    pending.remove(worker)
            except (OSError, RuntimeError, ValueError) as exc:
                last_errors[worker] = str(exc)
        if pending:
            time.sleep(0.25)
    if pending:
        details = "; ".join(f"{worker}: {last_errors.get(worker, 'not ready')}" for worker in sorted(pending))
        raise TimeoutError(f"workers did not become ready: {details}")


def _request_payload(
    config: BaselineConfig,
    state_dict: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_version: int,
    shard_id: int,
    attempt_id: int = 0,
) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "shard_id": shard_id,
        "attempt_id": attempt_id,
        "task_id": f"{model_version}:{shard_id}:{attempt_id}",
        "model_config": {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
        },
        "state_dict": {
            name: encode_float32(value) for name, value in state_dict.items()
        },
        "inputs": encode_float32(inputs),
        "targets": encode_float32(targets),
    }


def _local_full_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return float(loss.detach().item()), gradients


def train_distributed(
    workers: Sequence[str],
    config: BaselineConfig,
    steps: int = 200,
    timeout_seconds: float = 120.0,
    ready_timeout_seconds: float = 30.0,
    gradient_tolerance: float = 1e-5,
    validate_every_step: bool = False,
) -> dict[str, Any]:
    config.validate()
    if not workers:
        raise ValueError("at least one worker endpoint is required")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if len(workers) > config.samples:
        raise ValueError("worker count cannot exceed the sample count")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if not math.isfinite(ready_timeout_seconds) or ready_timeout_seconds <= 0:
        raise ValueError("ready_timeout_seconds must be positive and finite")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive and finite")

    endpoints = tuple(worker.rstrip("/") for worker in workers)
    _wait_for_workers(endpoints, ready_timeout_seconds)
    torch.set_num_threads(1)
    inputs, targets = make_dataset(config)
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shards = torch.tensor_split(torch.arange(config.samples), len(endpoints))
    shard_indices = [indices for indices in shards]
    shard_sizes = [len(indices) for indices in shard_indices]
    local_loss, local_reference = _local_full_gradient(model, inputs, targets)
    step_metrics: list[dict[str, float | int]] = []
    first_gradient_max_abs_error: float | None = None
    first_gradient_l2_error: float | None = None
    target_loss_step: int | None = None
    total_started = time.perf_counter()

    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        if step == 1:
            reference_loss, reference_gradients = local_loss, local_reference
        elif validate_every_step:
            reference_loss, reference_gradients = _local_full_gradient(
                model, inputs, targets
            )
        else:
            reference_loss, reference_gradients = None, None
        state_dict = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        payloads = [
            _request_payload(
                config,
                state_dict,
                inputs[indices],
                targets[indices],
                model_version=step - 1,
                shard_id=shard_id,
            )
            for shard_id, indices in enumerate(shard_indices)
        ]
        remote_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = [
                pool.submit(_post_json, endpoint, payload, timeout_seconds)
                for endpoint, payload in zip(endpoints, payloads)
            ]
            responses = [future.result() for future in futures]
        remote_seconds = time.perf_counter() - remote_started

        parameter_shapes = {
            name: parameter.shape for name, parameter in model.named_parameters()
        }
        registry = GradientResultRegistry(
            model_version=step - 1,
            expected_shards=len(shard_indices),
            parameter_shapes=parameter_shapes,
        )
        shard_losses: dict[int, float] = {}
        remote_loss = 0.0
        sample_total = 0
        for shard_id, (expected_size, response) in enumerate(zip(shard_sizes, responses)):
            if int(response.get("samples", -1)) != expected_size:
                raise RuntimeError("worker response sample count does not match its shard")
            if (
                int(response.get("model_version", -1)) != step - 1
                or int(response.get("shard_id", -1)) != shard_id
                or int(response.get("attempt_id", -1)) != 0
                or response.get("task_id") != f"{step - 1}:{shard_id}:0"
            ):
                raise RuntimeError("worker response task identity does not match its request")
            encoded_gradients = response.get("gradients")
            if not isinstance(encoded_gradients, dict) or set(encoded_gradients) != set(parameter_shapes):
                raise RuntimeError("worker gradient names do not match the model")
            decoded_gradients: dict[str, torch.Tensor] = {}
            for name in parameter_shapes:
                gradient = decode_float32(encoded_gradients[name])
                if gradient.shape != parameter_shapes[name]:
                    raise RuntimeError(f"worker gradient shape mismatch for {name}")
                decoded_gradients[name] = gradient
            duplicate_count = registry.duplicate_results_ignored
            accepted = registry.accept(
                model_version=int(response["model_version"]),
                shard_id=int(response["shard_id"]),
                task_id=str(response["task_id"]),
                gradients=decoded_gradients,
            )
            if not accepted and registry.duplicate_results_ignored == duplicate_count + 1:
                continue
            if not accepted:
                raise RuntimeError(f"worker returned a stale or invalid shard {shard_id}")
            sample_total += expected_size
            shard_losses[shard_id] = float(response["mean_squared_error"])
        if sample_total != config.samples:
            raise RuntimeError("worker shards do not cover the fixed dataset")
        if not registry.complete():
            raise RuntimeError("not all current-version shard results were accepted")

        aggregated = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
        }
        for shard_id, shard_gradients in registry.results.items():
            weight = shard_sizes[shard_id] / config.samples
            remote_loss += shard_losses[shard_id] * weight
            for name, gradient in shard_gradients.items():
                aggregated[name].add_(gradient, alpha=weight)

        gradient_max_abs_error: float | None = None
        loss_difference: float | None = None
        if reference_gradients is not None and reference_loss is not None:
            differences = [
                aggregated[name] - reference_gradients[name] for name in aggregated
            ]
            gradient_max_abs_error = max(
                float(value.abs().max().item()) for value in differences
            )
            gradient_l2_error = math.sqrt(
                sum(float(value.square().sum().item()) for value in differences)
            )
            loss_difference = abs(remote_loss - reference_loss)
            if gradient_max_abs_error > gradient_tolerance:
                raise RuntimeError(
                    "distributed gradient differs from the local reference: "
                    f"max_abs_error={gradient_max_abs_error:.6g}, "
                    f"tolerance={gradient_tolerance:.6g}"
                )
            if step == 1:
                first_gradient_max_abs_error = gradient_max_abs_error
                first_gradient_l2_error = gradient_l2_error

        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.grad = aggregated[name]
        optimizer.step()
        with torch.no_grad():
            train_loss = float(F.mse_loss(model(inputs), targets).item())
        step_metrics.append(
            {
                "step": step,
                "remote_gradient_seconds": remote_seconds,
                "step_seconds": time.perf_counter() - step_started,
                "remote_pre_update_loss": remote_loss,
                "local_loss_after_update": train_loss,
                "gradient_max_abs_error": gradient_max_abs_error,
                "reference_loss_difference": loss_difference,
            }
        )
        if target_loss_step is None and train_loss <= config.target_loss:
            target_loss_step = step
            break

    return {
        "config": {
            "samples": config.samples,
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "dataset_seed": config.seed,
            "model_seed": config.seed + 1,
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "target_loss": config.target_loss,
        },
        "workers": list(endpoints),
        "shard_sizes": shard_sizes,
        "steps_completed": len(step_metrics),
        "final_train_loss": step_metrics[-1]["local_loss_after_update"],
        "target_reached": target_loss_step is not None,
        "target_loss_step": target_loss_step,
        "gradient_tolerance": gradient_tolerance,
        "validate_every_step": validate_every_step,
        "gradient_validation_passed": True,
        "first_gradient_max_abs_error": first_gradient_max_abs_error,
        "first_gradient_l2_error": first_gradient_l2_error,
        "elapsed_seconds": time.perf_counter() - total_started,
        "step_metrics": step_metrics,
        "final_model_sha256": _model_digest(model),
    }


def _model_digest(model: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the fixed MLP using synchronized HTTP gradient workers."
    )
    parser.add_argument(
        "--workers",
        required=True,
        help="comma-separated worker URLs, for example http://worker-a:27500,http://worker-b:27500",
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-loss", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-5)
    parser.add_argument("--validate-every-step", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    config = BaselineConfig(
        samples=args.samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        epochs=args.steps,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
    )
    endpoints = [url.strip() for url in args.workers.split(",") if url.strip()]
    try:
        result = train_distributed(
            endpoints,
            config,
            steps=args.steps,
            timeout_seconds=args.timeout,
            ready_timeout_seconds=args.ready_timeout,
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
