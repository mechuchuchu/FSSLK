"""Typed configuration and state vocabulary for the WAN simulator."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class NodeState(str, Enum):
    ACTIVE = "active"
    NEAR_ZERO = "near_zero"
    STALLED = "stalled"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"


class LinkState(str, Enum):
    UP = "up"
    DOWN = "down"


class FaultKind(str, Enum):
    NODE_STALL = "node_stall"
    NODE_RESUME = "node_resume"
    NODE_SPEED_DROP = "node_speed_drop"
    NODE_SPEED_RESTORE = "node_speed_restore"
    NODE_LEAVE = "node_leave"
    NODE_JOIN = "node_join"
    NODE_FAIL = "node_fail"
    LINK_DOWN = "link_down"
    LINK_UP = "link_up"


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    region: str
    flops_per_second: float
    memory_bytes: int
    max_local_batch: int

    @property
    def node_uuid(self) -> str:
        """Stable identity for a topology node, independent of display name."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fsslk-wan-node:{self.node_id}"))

    def validate(self) -> None:
        if not self.node_id or not self.region:
            raise ValueError("node_id and region must be non-empty")
        if not math.isfinite(self.flops_per_second) or self.flops_per_second <= 0:
            raise ValueError("node flops_per_second must be positive")
        if self.memory_bytes < 1 or self.max_local_batch < 1:
            raise ValueError("node memory and local batch limit must be positive")


@dataclass(frozen=True)
class LinkSpec:
    link_id: str
    region_a: str
    region_b: str
    bytes_per_second: float
    latency_seconds: float

    def validate(self) -> None:
        if not self.link_id or not self.region_a or not self.region_b:
            raise ValueError("link ID and endpoint regions must be non-empty")
        if not math.isfinite(self.bytes_per_second) or self.bytes_per_second <= 0:
            raise ValueError("link bytes_per_second must be positive")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("link latency_seconds must be non-negative")


@dataclass(frozen=True)
class Topology:
    nodes: tuple[NodeSpec, ...]
    links: tuple[LinkSpec, ...]

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("topology must contain at least one node")
        node_ids = [node.node_id for node in self.nodes]
        link_ids = [link.link_id for link in self.links]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node IDs must be unique")
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("link IDs must be unique")
        regions = {node.region for node in self.nodes}
        for node in self.nodes:
            node.validate()
        for link in self.links:
            link.validate()
            if link.region_a not in regions or link.region_b not in regions:
                raise ValueError(f"link {link.link_id} references an unknown region")


@dataclass(frozen=True)
class Workload:
    global_batch_size: int
    target_samples: int
    flops_per_sample: float
    gradient_bytes: int
    default_micro_batch_size: int = 32
    model_state_bytes: int = 0
    activation_bytes_per_sample: int = 0
    batch_launch_overhead_seconds: float = 0.0
    input_bytes_per_sample: int = 4096

    def validate(self) -> None:
        if self.global_batch_size < 1 or self.target_samples < 1:
            raise ValueError("global batch and target sample counts must be positive")
        if not math.isfinite(self.flops_per_sample) or self.flops_per_sample <= 0:
            raise ValueError("flops_per_sample must be positive")
        if self.gradient_bytes < 0:
            raise ValueError("payload sizes must be non-negative")
        if self.default_micro_batch_size < 1:
            raise ValueError("default_micro_batch_size must be positive")
        if self.model_state_bytes < 0 or self.activation_bytes_per_sample < 0:
            raise ValueError("memory estimates must be non-negative")
        if self.input_bytes_per_sample < 0:
            raise ValueError("input_bytes_per_sample must be non-negative")
        if (
            not math.isfinite(self.batch_launch_overhead_seconds)
            or self.batch_launch_overhead_seconds < 0
        ):
            raise ValueError("batch_launch_overhead_seconds must be non-negative")


@dataclass(frozen=True)
class RecoveryConfig:
    detection_seconds: float = 1.0
    restart_seconds: float = 10.0
    membership_seconds: float = 2.0
    restore_seconds: float = 5.0

    def validate(self) -> None:
        for name, value in (
            ("detection_seconds", self.detection_seconds),
            ("restart_seconds", self.restart_seconds),
            ("membership_seconds", self.membership_seconds),
            ("restore_seconds", self.restore_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_seconds(self) -> float:
        return (
            self.detection_seconds
            + self.restart_seconds
            + self.membership_seconds
            + self.restore_seconds
        )


@dataclass(frozen=True)
class FaultEvent:
    time_seconds: float
    kind: FaultKind
    target: str
    duration_seconds: float | None = None

    def validate(self) -> None:
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("fault event time must be non-negative")
        if not self.target:
            raise ValueError("fault event target must be non-empty")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds < 0
        ):
            raise ValueError("fault duration must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    near_zero_speed_fraction: float = 0.01
    max_events_per_step: int = 100_000

    def validate(self) -> None:
        self.recovery.validate()
        if (
            not math.isfinite(self.near_zero_speed_fraction)
            or not 0 < self.near_zero_speed_fraction < 1
        ):
            raise ValueError("near_zero_speed_fraction must be between 0 and 1")
        if self.max_events_per_step < 1:
            raise ValueError("max_events_per_step must be positive")


@dataclass(frozen=True)
class StepResult:
    committed: bool
    start_seconds: float
    end_seconds: float
    committed_samples: int
    failed_reason: str | None
    compute_seconds: float
    collective_seconds: float
    barrier_wait_seconds: float
    recovery_seconds: float
    participant_completion_seconds: Mapping[str, float]
    micro_batch_sizes: Mapping[str, int]
    events_processed: int
    batch_uuid: str = ""
    shard_uuids: Mapping[str, str] = field(default_factory=dict)
    attempt_uuids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    winner_attempt_uuids: Mapping[str, str] = field(default_factory=dict)
    duplicates_started: int = 0
    late_results_ignored: int = 0


@dataclass(frozen=True)
class EpisodeMetrics:
    elapsed_seconds: float
    committed_samples: int
    step_count: int
    failed_steps: int
    compute_seconds: float
    collective_seconds: float
    barrier_wait_seconds: float
    recovery_seconds: float
    recovery_stage_seconds: Mapping[str, float]
    goodput_samples_per_second: float
    duplicates_started: int = 0
    duplicate_wins: int = 0
    late_results_ignored: int = 0


def validate_fault_schedule(events: Sequence[FaultEvent], topology: Topology) -> None:
    topology.validate()
    node_ids = {node.node_id for node in topology.nodes}
    link_ids = {link.link_id for link in topology.links}
    previous_time = -1.0
    for event in events:
        event.validate()
        if event.time_seconds < previous_time:
            raise ValueError("fault schedule must be sorted by time")
        previous_time = event.time_seconds
        if event.kind in {FaultKind.LINK_DOWN, FaultKind.LINK_UP}:
            if event.target not in link_ids:
                raise ValueError(f"fault references unknown link {event.target}")
        elif event.target not in node_ids:
            raise ValueError(f"fault references unknown node {event.target}")
