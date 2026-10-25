"""Compare exact shard-gradient aggregation with a single-process gradient.

Run from the repository root with::

    python -m RL_schedule.distributed_gradient --num-shards 4

Each shard uses the same initial model version. Its mean-reduced MSE gradient
is weighted by the shard's fraction of the full dataset before aggregation.
No optimizer step occurs until every shard gradient has been collected.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .mlp_baseline import (
    BaselineConfig,
    _tensor_fingerprint,
    make_dataset,
    make_model,
)


def compare_gradients(config: BaselineConfig, num_shards: int) -> dict[str, Any]:
    """Return numeric error metrics for the full and weighted shard gradients."""
    config.validate()
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if num_shards > config.samples:
        raise ValueError("num_shards cannot exceed the number of samples")

    inputs, targets = make_dataset(config)
    model = make_model(config)
    named_parameters = tuple(model.named_parameters())

    model.zero_grad(set_to_none=True)
    full_loss = F.mse_loss(model(inputs), targets)
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in named_parameters
        if parameter.grad is not None
    }

    aggregated = {name: torch.zeros_like(grad) for name, grad in full_gradients.items()}
    shards = torch.tensor_split(torch.arange(config.samples), num_shards)
    shard_sizes: list[int] = []
    for indices in shards:
        shard_size = len(indices)
        shard_sizes.append(shard_size)
        model.zero_grad(set_to_none=True)
        shard_loss = F.mse_loss(model(inputs[indices]), targets[indices])
        shard_loss.backward()
        weight = shard_size / config.samples
        for name, parameter in named_parameters:
            if parameter.grad is not None:
                aggregated[name].add_(parameter.grad, alpha=weight)

    differences = [aggregated[name] - full_gradients[name] for name in full_gradients]
    max_abs_error = max(float(value.abs().max().item()) for value in differences)
    l2_error = sum(float(value.square().sum().item()) for value in differences) ** 0.5
    full_fingerprint = _tensor_fingerprint(tuple(full_gradients.items()))
    aggregate_fingerprint = _tensor_fingerprint(tuple(aggregated.items()))

    return {
        "config": {
            "samples": config.samples,
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
        },
        "num_shards": num_shards,
        "dataset_seed": config.seed,
        "model_seed": config.seed + 1,
        "shard_sizes": shard_sizes,
        "full_dataset_loss": float(full_loss.item()),
        "max_abs_error": max_abs_error,
        "l2_error": l2_error,
        "full_gradient_sha256": full_fingerprint,
        "weighted_shard_gradient_sha256": aggregate_fingerprint,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare single-process and weighted shard gradients."
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = BaselineConfig(
        samples=args.samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    print(json.dumps(compare_gradients(config, args.num_shards), indent=2))


if __name__ == "__main__":
    main()
