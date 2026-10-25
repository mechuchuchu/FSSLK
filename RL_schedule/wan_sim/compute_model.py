"""Closed-form compute timing; no tensor or GPU work is executed."""

from __future__ import annotations

from .types import NodeSpec, Workload


class ComputeModel:
    @staticmethod
    def work_flops(node: NodeSpec, workload: Workload, local_samples: int) -> float:
        if local_samples < 0 or local_samples > node.max_local_batch:
            raise ValueError(f"local batch is outside {node.node_id}'s supported range")
        return float(local_samples) * workload.flops_per_sample

    @classmethod
    def duration_seconds(
        cls,
        node: NodeSpec,
        workload: Workload,
        local_samples: int,
        micro_batch_size: int,
    ) -> float:
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        micro_batches = (local_samples + micro_batch_size - 1) // micro_batch_size
        return (
            cls.work_flops(node, workload, local_samples) / node.flops_per_second
            + micro_batches * workload.batch_launch_overhead_seconds
        )

    @staticmethod
    def peak_memory_bytes(
        workload: Workload, local_samples: int, micro_batch_size: int
    ) -> int:
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        live_samples = min(local_samples, micro_batch_size)
        return (
            workload.model_state_bytes
            + live_samples * workload.activation_bytes_per_sample
        )
