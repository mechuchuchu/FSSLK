"""Versioned, duplicate-safe collection of distributed shard gradients."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GradientResultRegistry:
    model_version: int
    expected_shards: int
    parameter_shapes: dict[str, torch.Size]
    results: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    duplicate_results_ignored: int = 0
    invalid_results_rejected: int = 0
    stale_results_rejected: int = 0

    def accept(
        self,
        *,
        model_version: int,
        shard_id: int,
        task_id: str,
        gradients: dict[str, torch.Tensor],
    ) -> bool:
        """Accept one valid result; return False for a duplicate shard result."""
        try:
            task_parts = task_id.split(":")
            if len(task_parts) not in {3, 4}:
                raise ValueError("task ID has an invalid number of fields")
            task_version, task_shard, task_attempt = (
                int(part) for part in task_parts[-3:]
            )
        except (TypeError, ValueError):
            self.stale_results_rejected += 1
            return False
        if (
            model_version != self.model_version
            or task_version != model_version
            or task_shard != shard_id
            or task_attempt < 0
        ):
            self.stale_results_rejected += 1
            return False
        if shard_id < 0 or shard_id >= self.expected_shards:
            self.invalid_results_rejected += 1
            return False
        if shard_id in self.results:
            self.duplicate_results_ignored += 1
            return False
        if set(gradients) != set(self.parameter_shapes):
            self.invalid_results_rejected += 1
            return False
        for name, value in gradients.items():
            if value.shape != self.parameter_shapes[name] or not bool(
                torch.isfinite(value).all()
            ):
                self.invalid_results_rejected += 1
                return False
        self.results[shard_id] = {
            name: value.detach().clone() for name, value in gradients.items()
        }
        return True

    def complete(self) -> bool:
        return len(self.results) == self.expected_shards
