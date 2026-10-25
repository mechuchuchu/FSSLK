"""Reproducible single-process MLP baseline for the RL scheduling experiments.

Run from the repository root with::

    python RL_schedule/mlp_baseline.py --json results/mlp-baseline.json

The input and target vectors are generated once from a fixed seed and are kept
unchanged throughout training. This baseline is deliberately independent of
the FSSLK runtime; later stages can reuse its dataset and model for distributed
gradient experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BaselineConfig:
    samples: int = 1024
    input_dim: int = 32
    output_dim: int = 16
    hidden_dim: int = 512
    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.003
    target_loss: float = 0.01

    def validate(self) -> None:
        positive_fields = (
            "samples",
            "input_dim",
            "output_dim",
            "hidden_dim",
            "epochs",
            "batch_size",
        )
        for name in positive_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.target_loss) or self.target_loss < 0:
            raise ValueError("target_loss must be non-negative")


class MemorizationMLP(nn.Module):
    """Small fully connected network used by the fixed-vector experiment."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def make_dataset(config: BaselineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed random input/target dataset on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    inputs = torch.randn(config.samples, config.input_dim, generator=generator)
    targets = torch.randn(config.samples, config.output_dim, generator=generator)
    return inputs, targets


def make_model(config: BaselineConfig) -> MemorizationMLP:
    """Create the same seeded initial model for baseline and shard runs."""
    # Isolate model initialization from the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 1)
        return MemorizationMLP(config.input_dim, config.hidden_dim, config.output_dim)


def _tensor_fingerprint(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def _dataset_fingerprint(inputs: torch.Tensor, targets: torch.Tensor) -> str:
    return _tensor_fingerprint((("inputs", inputs), ("targets", targets)))


def _model_fingerprint(model: nn.Module) -> str:
    return _tensor_fingerprint(tuple(model.state_dict().items()))


def _initial_gradient_fingerprint(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> str:
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    gradients = tuple(
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    fingerprint = _tensor_fingerprint(gradients)
    model.zero_grad(set_to_none=True)
    return fingerprint


def run_baseline(config: BaselineConfig) -> dict[str, Any]:
    """Train the MLP and return the full reproducibility and timing record."""
    config.validate()
    inputs, targets = make_dataset(config)
    model = make_model(config)
    dataset_fingerprint = _dataset_fingerprint(inputs, targets)
    gradient_fingerprint = _initial_gradient_fingerprint(model, inputs, targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)

    step_times: list[float] = []
    epoch_metrics: list[dict[str, float | int]] = []
    started = time.perf_counter()
    target_loss_step: int | None = None
    target_loss_seconds: float | None = None
    total_steps = 0

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        order = torch.randperm(config.samples, generator=shuffle_generator)
        for start in range(0, config.samples, config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]

            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch_loss = F.mse_loss(model(batch_inputs), batch_targets)
            batch_loss.backward()
            optimizer.step()
            step_times.append(time.perf_counter() - step_started)
            total_steps += 1

        with torch.no_grad():
            train_loss = float(F.mse_loss(model(inputs), targets).item())
        epoch_metrics.append(
            {
                "epoch": epoch,
                "step": total_steps,
                "train_loss": train_loss,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        if target_loss_step is None and train_loss <= config.target_loss:
            target_loss_step = total_steps
            target_loss_seconds = time.perf_counter() - started
            break

    elapsed_seconds = time.perf_counter() - started
    final_loss = epoch_metrics[-1]["train_loss"] if epoch_metrics else float("nan")
    return {
        "config": asdict(config),
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "optimizer": "Adam",
        },
        "result": {
            "epochs_completed": len(epoch_metrics),
            "steps_completed": total_steps,
            "final_train_loss": final_loss,
            "target_reached": target_loss_step is not None,
            "target_loss_step": target_loss_step,
            "target_loss_seconds": target_loss_seconds,
            "elapsed_seconds": elapsed_seconds,
            "mean_step_seconds": (
                sum(step_times) / len(step_times) if step_times else 0.0
            ),
        },
        "reproducibility": {
            "dataset_seed": config.seed,
            "model_seed": config.seed + 1,
            "shuffle_seed": config.seed + 2,
            "dataset_sha256": dataset_fingerprint,
            "initial_full_dataset_gradient_sha256": gradient_fingerprint,
            "final_model_sha256": _model_fingerprint(model),
        },
        "epoch_metrics": epoch_metrics,
        "step_times_seconds": step_times,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fixed-random-vector single-process MLP baseline."
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-loss", type=float, default=0.01)
    parser.add_argument(
        "--json",
        type=Path,
        help="write the complete run record, including per-step timings, to this file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = BaselineConfig(
        samples=args.samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
    )
    record = run_baseline(config)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    summary = {
        "result": record["result"],
        "reproducibility": record["reproducibility"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
