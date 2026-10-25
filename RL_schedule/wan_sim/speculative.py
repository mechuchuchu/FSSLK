"""Duplicate-aware shard execution over a virtual WAN clock.

This environment exposes every currently feasible start, duplicate, and wait
choice as a candidate action. It does not prescribe a scheduler or execute
tensors. Logical shard UUIDs survive retries; each execution attempt has its
own UUID, and only the first valid result can complete a shard.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .clock import VirtualClock
from .compute_model import ComputeModel
from .events import Event, EventQueue
from .network_model import CollectiveEstimate, WANNetworkModel
from .types import (
    FaultEvent,
    FaultKind,
    LinkState,
    NodeState,
    SimulationConfig,
    Topology,
    Workload,
    validate_fault_schedule,
)


class DecisionKind(str, Enum):
    START = "start"
    DUPLICATE = "duplicate"
    WAIT = "wait"
    CANCEL = "cancel"


class AttemptState(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LATE = "late"


@dataclass(frozen=True)
class DecisionAction:
    kind: DecisionKind
    shard_uuid: str | None = None
    node_uuid: str | None = None
    node_id: str | None = None
    micro_batch_size: int | None = None
    wait_seconds: float | None = None
    attempt_uuid: str | None = None


@dataclass
class _Shard:
    shard_uuid: str
    shard_index: int
    samples: int
    micro_batch_size: int | None = None
    winner_attempt_uuid: str | None = None
    attempt_uuids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.attempt_uuids is None:
            self.attempt_uuids = []


@dataclass
class _Attempt:
    attempt_uuid: str
    shard_uuid: str
    node_id: str
    micro_batch_size: int
    remaining_flops: float
    remaining_overhead_seconds: float
    started_at: float
    last_progress_at: float
    is_duplicate: bool = False
    stage: str = "compute"
    active_link_ids: frozenset[str] = frozenset()
    generation: int = 0
    state: AttemptState = AttemptState.RUNNING
    completed_at: float | None = None
    compute_seconds: float = 0.0


@dataclass
class _NodeRuntime:
    state: NodeState = NodeState.ACTIVE
    generation: int = 0


class SpeculativeWANScheduler:
    """Event-driven batch environment with unrestricted feasible dispatch choices."""

    WAIT_OPTIONS_SECONDS = tuple(round(step / 10.0, 1) for step in range(1, 101))

    def __init__(
        self,
        topology: Topology,
        workload: Workload,
        *,
        shard_count: int,
        config: SimulationConfig | None = None,
        fault_schedule: Sequence[FaultEvent] = (),
        initial_active_nodes: Sequence[str] | None = None,
        micro_batch_options: Sequence[int] = (32, 48, 64),
        batches_per_episode: int = 1,
        max_decisions: int = 2000,
        decision_overhead_seconds: float = 0.001,
    ) -> None:
        topology.validate()
        workload.validate()
        if shard_count < 1 or workload.global_batch_size % shard_count:
            raise ValueError("shard_count must evenly divide the global batch")
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        if batches_per_episode < 1:
            raise ValueError("batches_per_episode must be positive")
        if not math.isfinite(decision_overhead_seconds) or decision_overhead_seconds < 0:
            raise ValueError("decision_overhead_seconds must be non-negative")
        if not micro_batch_options or any(size < 1 for size in micro_batch_options):
            raise ValueError("micro-batch options must be positive")
        if len(set(micro_batch_options)) != len(micro_batch_options):
            raise ValueError("micro-batch options must be unique")
        validate_fault_schedule(fault_schedule, topology)

        self.topology = topology
        self.workload = workload
        self.shard_count = shard_count
        self.config = config or SimulationConfig()
        self.config.validate()
        self.fault_schedule = tuple(fault_schedule)
        self.initial_active_nodes = tuple(
            node.node_id for node in topology.nodes
        ) if initial_active_nodes is None else tuple(initial_active_nodes)
        known_nodes = {node.node_id for node in topology.nodes}
        if not set(self.initial_active_nodes) <= known_nodes:
            raise ValueError("initial membership references an unknown node")
        if not self.initial_active_nodes:
            raise ValueError("at least one node must be active initially")
        self.micro_batch_options = tuple(int(size) for size in micro_batch_options)
        self.batches_per_episode = int(batches_per_episode)
        self.max_decisions = max_decisions
        self.decision_overhead_seconds = decision_overhead_seconds
        self.nodes = {node.node_id: node for node in topology.nodes}
        self.node_ids = tuple(node.node_id for node in topology.nodes)
        self.node_uuids = {node_id: node.node_uuid for node_id, node in self.nodes.items()}
        self.node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self.network = WANNetworkModel(topology)
        self._transfer_cache: dict[tuple[Any, ...], Any] = {}
        self._fit_cache: dict[tuple[Any, ...], bool] = {}
        self.clock = VirtualClock()
        self.events = EventQueue()
        self.node_runtime: dict[str, _NodeRuntime] = {}
        self.link_runtime: dict[str, LinkState] = {}
        self.link_generation: dict[str, int] = {}
        self.shards: dict[str, _Shard] = {}
        self.attempts: dict[str, _Attempt] = {}
        self.run_uuid = ""
        self.batch_uuid = ""
        self._seed = 0
        self._attempt_sequence = 0
        self._batch_index = 0
        self._batches_completed = 0
        self._committed_samples = 0
        self._batch_history: list[dict[str, Any]] = []
        self._decision_count = 0
        self._compute_seconds = 0.0
        self._network_seconds = 0.0
        self._network_bytes = 0
        self._collective_seconds = 0.0
        self._collective_restarts = 0
        self._barrier_wait_seconds = 0.0
        self._duplicates_started = 0
        self._duplicate_wins = 0
        self._late_results_ignored = 0
        self._failed_attempts = 0
        self._committed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._allreduce_done = False
        self.reset(seed=0)

    @property
    def done(self) -> bool:
        return self._committed or self._failed or self._decision_count >= self.max_decisions

    @property
    def success(self) -> bool:
        return self._committed

    def reset(
        self,
        *,
        seed: int = 0,
        fault_schedule: Sequence[FaultEvent] | None = None,
        initial_active_nodes: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, tuple[DecisionAction, ...]]:
        if fault_schedule is not None:
            validate_fault_schedule(fault_schedule, self.topology)
            self.fault_schedule = tuple(fault_schedule)
        if initial_active_nodes is not None:
            known_nodes = set(self.node_ids)
            if not set(initial_active_nodes) <= known_nodes or not initial_active_nodes:
                raise ValueError("initial membership must contain known nodes")
            self.initial_active_nodes = tuple(initial_active_nodes)
        self._seed = int(seed)
        self.run_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"fsslk-wan-run:{self._seed}")
        )
        self.clock = VirtualClock()
        self.events = EventQueue()
        self.node_runtime = {
            node_id: _NodeRuntime(
                state=(
                    NodeState.ACTIVE
                    if node_id in self.initial_active_nodes
                    else NodeState.OFFLINE
                )
            )
            for node_id in self.node_ids
        }
        self.link_runtime = {
            link.link_id: LinkState.UP for link in self.topology.links
        }
        self._transfer_cache.clear()
        self._fit_cache.clear()
        self.link_generation = {link.link_id: 0 for link in self.topology.links}
        self.attempts = {}
        self._attempt_sequence = 0
        self._batch_index = 0
        self._batches_completed = 0
        self._committed_samples = 0
        self._batch_history = []
        self._decision_count = 0
        self._compute_seconds = 0.0
        self._network_seconds = 0.0
        self._network_bytes = 0
        self._collective_seconds = 0.0
        self._collective_restarts = 0
        self._barrier_wait_seconds = 0.0
        self._duplicates_started = 0
        self._duplicate_wins = 0
        self._late_results_ignored = 0
        self._failed_attempts = 0
        self._committed = False
        self._failed = False
        self._failure_reason = None
        self._allreduce_done = False
        self._initialize_batch_work()
        for fault in self.fault_schedule:
            self.events.push(
                fault.time_seconds,
                f"fault:{fault.kind.value}",
                target=fault.target,
                priority=0,
                payload={"duration_seconds": fault.duration_seconds},
            )
        self._process_current_time_events()
        return self.observation(), self.candidate_actions()

    def _initialize_batch_work(self) -> None:
        self.batch_uuid = str(
            uuid.uuid5(uuid.UUID(self.run_uuid), f"global-batch:{self._batch_index}")
        )
        shard_samples = self.workload.global_batch_size // self.shard_count
        self.shards = {}
        for index in range(self.shard_count):
            shard_uuid = str(
                uuid.uuid5(uuid.UUID(self.batch_uuid), f"shard:{index}")
            )
            self.shards[shard_uuid] = _Shard(shard_uuid, index, shard_samples)

    def candidate_actions(self) -> tuple[DecisionAction, ...]:
        if self.done:
            return ()
        candidates: list[DecisionAction] = []
        idle_nodes = self._idle_active_nodes()
        for shard in sorted(self.shards.values(), key=lambda item: item.shard_index):
            if shard.winner_attempt_uuid is not None:
                continue
            live_attempts = self._live_attempts(shard.shard_uuid)
            for node_id in idle_nodes:
                if any(attempt.node_id == node_id for attempt in live_attempts):
                    continue
                if live_attempts:
                    for size in self.micro_batch_options:
                        if self._can_fit(node_id, shard.samples, size):
                            candidates.append(
                                DecisionAction(
                                    DecisionKind.DUPLICATE,
                                    shard_uuid=shard.shard_uuid,
                                    node_uuid=self.node_uuids[node_id],
                                    node_id=node_id,
                                    micro_batch_size=size,
                                )
                            )
                else:
                    for size in self.micro_batch_options:
                        if self._can_fit(node_id, shard.samples, size):
                            candidates.append(
                                DecisionAction(
                                    DecisionKind.START,
                                    shard_uuid=shard.shard_uuid,
                                    node_uuid=self.node_uuids[node_id],
                                    node_id=node_id,
                                    micro_batch_size=size,
                                )
                            )
        for attempt in self.attempts.values():
            if attempt.state == AttemptState.RUNNING:
                candidates.append(
                    DecisionAction(
                        DecisionKind.CANCEL,
                        shard_uuid=attempt.shard_uuid,
                        node_uuid=self.node_uuids[attempt.node_id],
                        node_id=attempt.node_id,
                        micro_batch_size=attempt.micro_batch_size,
                        attempt_uuid=attempt.attempt_uuid,
                    )
                )
        if any(shard.winner_attempt_uuid is None for shard in self.shards.values()):
            candidates.extend(
                DecisionAction(DecisionKind.WAIT, wait_seconds=duration)
                for duration in self.WAIT_OPTIONS_SECONDS
            )
        if (
            not candidates
            and not all(shard.winner_attempt_uuid is not None for shard in self.shards.values())
        ):
            self._failed = True
            self._failure_reason = "no_feasible_action_or_future_event"
        return tuple(candidates)

    def step(
        self,
        action_index: int,
        candidates: Sequence[DecisionAction] | None = None,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode has ended")
        candidates = self.candidate_actions() if candidates is None else candidates
        if action_index < 0 or action_index >= len(candidates):
            raise ValueError("action index is outside the current candidate set")
        action = candidates[action_index]
        self._decision_count += 1
        started = self.clock.now

        if action.kind in {DecisionKind.START, DecisionKind.DUPLICATE}:
            assert action.shard_uuid is not None
            assert action.node_id is not None
            assert action.micro_batch_size is not None
            self._start_attempt(action)
            if self.decision_overhead_seconds > 0:
                self._advance_wait(self.decision_overhead_seconds)
        elif action.kind == DecisionKind.CANCEL:
            assert action.attempt_uuid is not None
            attempt = self.attempts[action.attempt_uuid]
            if attempt.state != AttemptState.RUNNING:
                raise ValueError("attempt is no longer running")
            attempt.state = AttemptState.CANCELLED
            attempt.generation += 1
            if self.decision_overhead_seconds > 0:
                self._advance_wait(self.decision_overhead_seconds)
        else:
            self._advance_wait(float(action.wait_seconds or 0.1))

        if not self._failed and all(
            shard.winner_attempt_uuid is not None for shard in self.shards.values()
        ):
            self._finish_collective()
        if not self._committed and not self._failed and not self.events:
            if (
                not any(
                    self._live_attempts(shard.shard_uuid)
                    for shard in self.shards.values()
                )
                and not self._idle_active_nodes()
            ):
                self._failed = True
                self._failure_reason = "no_active_attempt_or_future_membership_event"
        if self._decision_count >= self.max_decisions and not self._committed:
            self._failed = True
            self._failure_reason = "decision_limit_reached"

        elapsed = self.clock.now - started
        return self.observation(), -elapsed, self.done, {
            "action": action,
            "elapsed_seconds": elapsed,
            "state": self.state(),
            "metrics": self.metrics(),
            "candidate_actions": self.candidate_actions(),
        }

    def observation(self) -> np.ndarray:
        values: list[float] = [
            min(1.0, self.clock.now / 100.0),
            self._committed_samples
            / max(
                1,
                self.workload.global_batch_size * self.batches_per_episode,
            ),
            sum(shard.winner_attempt_uuid is not None for shard in self.shards.values())
            / max(1, self.shard_count),
            min(1.0, self._failed_attempts / max(1, self.shard_count * 2)),
            min(1.0, self._duplicates_started / max(1, self.shard_count)),
            sum(state == LinkState.UP for state in self.link_runtime.values())
            / max(1, len(self.link_runtime)),
        ]
        for node_id in self.node_ids:
            node = self.nodes[node_id]
            state = self.node_runtime[node_id].state
            values.extend(float(state == candidate) for candidate in NodeState)
            values.extend(
                [
                    min(1.0, node.flops_per_second / max(n.flops_per_second for n in self.nodes.values())),
                    min(1.0, node.memory_bytes / max(n.memory_bytes for n in self.nodes.values())),
                    float(any(a.node_id == node_id and a.state == AttemptState.RUNNING for a in self.attempts.values())),
                ]
            )
        for link_id, state in self.link_runtime.items():
            values.extend(
                [
                    float(state == LinkState.UP),
                    min(
                        1.0,
                        self._active_flows_on_link(link_id)
                        / max(1, len(self.node_ids)),
                    ),
                ]
            )
        for shard in sorted(self.shards.values(), key=lambda item: item.shard_index):
            live = self._live_attempts(shard.shard_uuid)
            progress = max((self._attempt_progress(a) for a in live), default=0.0)
            values.extend(
                [
                    float(shard.winner_attempt_uuid is not None),
                    len(live) / max(1, len(self.node_ids)),
                    progress,
                    shard.samples / max(1, self.workload.global_batch_size),
                    (shard.micro_batch_size or 0) / max(self.micro_batch_options),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def candidate_features(
        self, candidates: Sequence[DecisionAction] | None = None
    ) -> np.ndarray:
        actions = self.candidate_actions() if candidates is None else candidates
        node_denominator = max(1, len(self.node_ids) - 1)
        shard_denominator = max(1, self.shard_count - 1)
        rows = []
        for action in actions:
            rows.append(
                [
                    float(action.kind == DecisionKind.START),
                    float(action.kind == DecisionKind.DUPLICATE),
                    float(action.kind == DecisionKind.WAIT),
                    float(action.kind == DecisionKind.CANCEL),
                    (self.shards[action.shard_uuid].shard_index / shard_denominator)
                    if action.shard_uuid is not None
                    else 0.0,
                    (self.node_index[action.node_id] / node_denominator)
                    if action.node_id is not None
                    else 0.0,
                    (action.micro_batch_size or 0) / max(self.micro_batch_options),
                    (action.wait_seconds or 0.0) / max(self.WAIT_OPTIONS_SECONDS),
                    (len(self._live_attempts(action.shard_uuid)) / max(1, len(self.node_ids)))
                    if action.shard_uuid is not None
                    else 0.0,
                    min(1.0, self._candidate_network_seconds(action) / 10.0),
                ]
            )
        return np.asarray(rows, dtype=np.float32).reshape(-1, 10)

    def _candidate_network_seconds(self, action: DecisionAction) -> float:
        if action.shard_uuid is None or action.node_id is None:
            return 0.0
        shard = self.shards[action.shard_uuid]
        coordinator_region = self.topology.nodes[0].region
        input_transfer = self._transfer_with_contention(
            coordinator_region,
            self.nodes[action.node_id].region,
            shard.samples * self.workload.input_bytes_per_sample,
        )
        output_transfer = self._transfer_with_contention(
            self.nodes[action.node_id].region,
            coordinator_region,
            self.workload.gradient_bytes
            * shard.samples
            // self.workload.global_batch_size,
        )
        if input_transfer is None or output_transfer is None:
            return float("inf")
        return input_transfer.duration_seconds + output_transfer.duration_seconds

    def state(self) -> dict[str, Any]:
        return {
            "run_uuid": self.run_uuid,
            "batch_uuid": self.batch_uuid,
            "batch_index": self._batch_index,
            "batches_completed": self._batches_completed,
            "batches_per_episode": self.batches_per_episode,
            "committed_samples": self._committed_samples,
            "batch_history": list(self._batch_history),
            "time_seconds": self.clock.now,
            "nodes": {
                node_id: {
                    "node_uuid": self.node_uuids[node_id],
                    "state": runtime.state.value,
                }
                for node_id, runtime in self.node_runtime.items()
            },
            "shards": {
                shard.shard_uuid: {
                    "shard_index": shard.shard_index,
                    "samples": shard.samples,
                    "winner_attempt_uuid": shard.winner_attempt_uuid,
                    "attempt_uuids": list(shard.attempt_uuids or ()),
                }
                for shard in self.shards.values()
            },
            "attempts": {
                attempt.attempt_uuid: {
                    "shard_uuid": attempt.shard_uuid,
                    "node_uuid": self.node_uuids[attempt.node_id],
                    "node_id": attempt.node_id,
                    "state": attempt.state.value,
                    "started_at": attempt.started_at,
                    "completed_at": attempt.completed_at,
                    "micro_batch_size": attempt.micro_batch_size,
                    "stage": attempt.stage,
                }
                for attempt in self.attempts.values()
            },
            "committed": self._committed,
            "failed": self._failed,
            "failure_reason": self._failure_reason,
        }

    def metrics(self) -> dict[str, float | int | str | bool]:
        elapsed = self.clock.now
        return {
            "elapsed_seconds": elapsed,
            "decision_count": self._decision_count,
            "decision_overhead_seconds": self.decision_overhead_seconds,
            "batches_completed": self._batches_completed,
            "batches_per_episode": self.batches_per_episode,
            "episode_target_samples": (
                self.workload.global_batch_size * self.batches_per_episode
            ),
            "committed_samples": self._committed_samples,
            "success": self._committed,
            "failure_reason": self._failure_reason,
            "compute_seconds": self._compute_seconds,
            "network_seconds": self._network_seconds,
            "network_bytes": self._network_bytes,
            "collective_seconds": self._collective_seconds,
            "collective_restarts": self._collective_restarts,
            "barrier_wait_seconds": self._barrier_wait_seconds,
            "duplicates_started": self._duplicates_started,
            "duplicate_wins": self._duplicate_wins,
            "late_results_ignored": self._late_results_ignored,
            "failed_attempts": self._failed_attempts,
            "goodput_samples_per_second": (
                self._committed_samples / elapsed
                if self._committed_samples > 0 and elapsed > 0
                else 0.0
            ),
        }

    def _can_fit(self, node_id: str, samples: int, micro_batch_size: int) -> bool:
        key = (
            node_id,
            samples,
            micro_batch_size,
            tuple((link_id, state.value) for link_id, state in self.link_runtime.items()),
        )
        if key in self._fit_cache:
            return self._fit_cache[key]
        memory_ok = (
            ComputeModel.peak_memory_bytes(
                self.workload, samples, micro_batch_size
            )
            <= self.nodes[node_id].memory_bytes
        )
        coordinator_region = self.topology.nodes[0].region
        send = self._transfer(
            coordinator_region,
            self.nodes[node_id].region,
            samples * self.workload.input_bytes_per_sample,
        )
        receive = self._transfer(
            self.nodes[node_id].region,
            coordinator_region,
            self.workload.gradient_bytes * samples // self.workload.global_batch_size,
        )
        result = memory_ok and send is not None and receive is not None
        self._fit_cache[key] = result
        return result

    def _transfer(self, source_region: str, target_region: str, payload_bytes: int):
        link_state_key = tuple(
            (link_id, state.value) for link_id, state in self.link_runtime.items()
        )
        key = (source_region, target_region, payload_bytes, link_state_key)
        if key not in self._transfer_cache:
            self._transfer_cache[key] = self.network.transfer(
                source_region,
                target_region,
                payload_bytes,
                self.link_runtime,
            )
        return self._transfer_cache[key]

    def _active_flows_on_link(self, link_id: str) -> int:
        return sum(
            attempt.state == AttemptState.RUNNING
            and attempt.stage in {"input", "result"}
            and link_id in attempt.active_link_ids
            for attempt in self.attempts.values()
        )

    def _transfer_with_contention(
        self, source_region: str, target_region: str, payload_bytes: int
    ) -> CollectiveEstimate | None:
        estimate = self._transfer(source_region, target_region, payload_bytes)
        if estimate is None or not estimate.links_used:
            return estimate
        concurrent_flows = max(
            self._active_flows_on_link(link_id) for link_id in estimate.links_used
        )
        return CollectiveEstimate(
            duration_seconds=estimate.duration_seconds * (concurrent_flows + 1),
            links_used=estimate.links_used,
            bytes_transferred=estimate.bytes_transferred,
        )

    def _idle_active_nodes(self) -> list[str]:
        busy = {
            attempt.node_id
            for attempt in self.attempts.values()
            if attempt.state == AttemptState.RUNNING
        }
        return [
            node_id
            for node_id in self.node_ids
            if self.node_runtime[node_id].state
            in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
            and node_id not in busy
        ]

    def _live_attempts(self, shard_uuid: str) -> list[_Attempt]:
        shard = self.shards[shard_uuid]
        if shard.attempt_uuids is None:
            return []
        return [
            self.attempts[attempt_uuid]
            for attempt_uuid in shard.attempt_uuids
            if self.attempts[attempt_uuid].state == AttemptState.RUNNING
        ]

    def _attempt_progress(self, attempt: _Attempt) -> float:
        total = self.shards[attempt.shard_uuid].samples * self.workload.flops_per_sample
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - attempt.remaining_flops / total))

    def _start_attempt(self, action: DecisionAction) -> None:
        assert action.shard_uuid is not None and action.node_id is not None
        assert action.micro_batch_size is not None
        shard = self.shards[action.shard_uuid]
        if shard.winner_attempt_uuid is not None:
            raise ValueError("cannot execute a shard that already has a winner")
        if action.node_id not in self._idle_active_nodes():
            raise ValueError("target node is not idle and active")
        if not self._can_fit(action.node_id, shard.samples, action.micro_batch_size):
            raise ValueError("selected micro-batch exceeds target node memory")
        if shard.micro_batch_size is None:
            shard.micro_batch_size = action.micro_batch_size

        node = self.nodes[action.node_id]
        coordinator_region = self.topology.nodes[0].region
        input_transfer = self._transfer_with_contention(
            coordinator_region,
            node.region,
            shard.samples * self.workload.input_bytes_per_sample,
        )
        if input_transfer is None:
            raise ValueError("no WAN route to dispatch shard to target node")
        attempt_uuid = str(
            uuid.uuid5(
                uuid.UUID(shard.shard_uuid), f"attempt:{self._attempt_sequence}"
            )
        )
        self._attempt_sequence += 1
        flops = ComputeModel.work_flops(node, self.workload, shard.samples)
        duration = ComputeModel.duration_seconds(
            node, self.workload, shard.samples, action.micro_batch_size
        )
        attempt = _Attempt(
            attempt_uuid=attempt_uuid,
            shard_uuid=shard.shard_uuid,
            node_id=action.node_id,
            micro_batch_size=action.micro_batch_size,
            remaining_flops=flops,
            remaining_overhead_seconds=max(
                0.0, duration - flops / node.flops_per_second
            ),
            started_at=self.clock.now,
            last_progress_at=self.clock.now,
            is_duplicate=action.kind == DecisionKind.DUPLICATE,
            stage="input" if input_transfer.duration_seconds > 0 else "compute",
            active_link_ids=input_transfer.links_used,
        )
        self.attempts[attempt_uuid] = attempt
        assert shard.attempt_uuids is not None
        shard.attempt_uuids.append(attempt_uuid)
        if action.kind == DecisionKind.DUPLICATE:
            self._duplicates_started += 1
        self._network_seconds += input_transfer.duration_seconds
        self._network_bytes += input_transfer.bytes_transferred
        if input_transfer.duration_seconds > 0:
            attempt.generation += 1
            self.events.push(
                self.clock.now + input_transfer.duration_seconds,
                "attempt_input_complete",
                target=attempt.attempt_uuid,
                priority=10,
                generation=attempt.generation,
            )
        else:
            self._schedule_attempt(attempt)

    def _schedule_attempt(self, attempt: _Attempt) -> None:
        if attempt.state != AttemptState.RUNNING or attempt.stage != "compute":
            return
        speed = self._effective_speed(attempt.node_id)
        attempt.generation += 1
        if speed <= 0:
            return
        duration = attempt.remaining_flops / speed + attempt.remaining_overhead_seconds
        self.events.push(
            self.clock.now + duration,
            "attempt_compute_complete",
            target=attempt.attempt_uuid,
            priority=10,
            generation=attempt.generation,
        )

    def _advance_wait(self, wait_seconds: float) -> None:
        deadline = self.clock.now + wait_seconds
        while self.events:
            event = self.events.peek()
            if event is None or event.time_seconds > deadline:
                break
            event = self.events.pop()
            if event.kind.startswith("attempt_") and not self._attempt_event_is_current(event):
                continue
            self._advance_progress(event.time_seconds)
            if event.kind == "attempt_input_complete":
                attempt = self.attempts[event.target]
                attempt.stage = "compute"
                attempt.active_link_ids = frozenset()
                self._schedule_attempt(attempt)
            elif event.kind == "attempt_compute_complete":
                self._begin_result_transfer(self.attempts[event.target])
            elif event.kind == "attempt_result_arrived":
                self._accept_attempt(self.attempts[event.target])
            elif self._is_stale(event):
                continue
            else:
                self._apply_event(event)
            self._process_current_time_events()
            # Return at the first event boundary so the policy can react.
            return
        if self.clock.now < deadline:
            self._advance_progress(deadline)

    def _begin_result_transfer(self, attempt: _Attempt) -> None:
        if attempt.state != AttemptState.RUNNING:
            return
        node_state = self.node_runtime[attempt.node_id].state
        if node_state not in {NodeState.ACTIVE, NodeState.NEAR_ZERO}:
            attempt.state = AttemptState.FAILED
            self._failed_attempts += 1
            return
        attempt.remaining_flops = 0.0
        attempt.remaining_overhead_seconds = 0.0
        shard = self.shards[attempt.shard_uuid]
        coordinator_region = self.topology.nodes[0].region
        transfer = self._transfer_with_contention(
            self.nodes[attempt.node_id].region,
            coordinator_region,
            self.workload.gradient_bytes
            * shard.samples
            // self.workload.global_batch_size,
        )
        if transfer is None:
            attempt.state = AttemptState.FAILED
            self._failed_attempts += 1
            return
        self._network_seconds += transfer.duration_seconds
        self._network_bytes += transfer.bytes_transferred
        attempt.stage = "result"
        attempt.active_link_ids = transfer.links_used
        if transfer.duration_seconds <= 0:
            self._accept_attempt(attempt)
            return
        attempt.generation += 1
        self.events.push(
            self.clock.now + transfer.duration_seconds,
            "attempt_result_arrived",
            target=attempt.attempt_uuid,
            priority=10,
            generation=attempt.generation,
        )

    def _accept_attempt(self, attempt: _Attempt) -> None:
        if attempt.state != AttemptState.RUNNING:
            return
        node_state = self.node_runtime[attempt.node_id].state
        if node_state not in {NodeState.ACTIVE, NodeState.NEAR_ZERO}:
            attempt.state = AttemptState.FAILED
            self._failed_attempts += 1
            return
        attempt.completed_at = self.clock.now
        attempt.state = AttemptState.COMPLETE
        shard = self.shards[attempt.shard_uuid]
        if shard.winner_attempt_uuid is None:
            shard.winner_attempt_uuid = attempt.attempt_uuid
            if self._duplicates_started and len(shard.attempt_uuids or ()) > 1:
                if attempt.is_duplicate:
                    self._duplicate_wins += 1
            for other_id in shard.attempt_uuids or ():
                if other_id == attempt.attempt_uuid:
                    continue
                other = self.attempts[other_id]
                if other.state == AttemptState.RUNNING:
                    other.state = AttemptState.CANCELLED
                    other.generation += 1
                    self._late_results_ignored += 1
        else:
            attempt.state = AttemptState.LATE
            self._late_results_ignored += 1

    def _attempt_event_is_current(self, event: Event) -> bool:
        attempt = self.attempts.get(event.target)
        if (
            attempt is None
            or attempt.state != AttemptState.RUNNING
            or attempt.generation != event.generation
        ):
            return False
        expected_stage = {
            "attempt_input_complete": "input",
            "attempt_compute_complete": "compute",
            "attempt_result_arrived": "result",
        }.get(event.kind)
        return expected_stage == attempt.stage

    def _advance_progress(self, to_time: float) -> None:
        elapsed = self.clock.advance_to(to_time)
        if elapsed <= 0:
            return
        for attempt in self.attempts.values():
            if attempt.state != AttemptState.RUNNING or attempt.stage != "compute":
                continue
            speed = self._effective_speed(attempt.node_id)
            if speed <= 0:
                continue
            compute_left = attempt.remaining_flops / speed
            overhead_before = attempt.remaining_overhead_seconds
            compute_elapsed = min(elapsed, compute_left)
            progress = min(attempt.remaining_flops, speed * compute_elapsed)
            attempt.remaining_flops = max(0.0, attempt.remaining_flops - progress)
            attempt.remaining_overhead_seconds = max(
                0.0,
                overhead_before - max(0.0, elapsed - compute_elapsed),
            )
            busy = min(elapsed, compute_left + overhead_before)
            attempt.compute_seconds += busy
            self._compute_seconds += busy
            attempt.last_progress_at = self.clock.now

    def _process_current_time_events(self) -> bool:
        processed = False
        while self.events:
            event = self.events.peek()
            if event is None or event.time_seconds > self.clock.now:
                break
            event = self.events.pop()
            if event.kind.startswith("attempt_"):
                if not self._attempt_event_is_current(event):
                    continue
                attempt = self.attempts[event.target]
                if event.kind == "attempt_input_complete":
                    attempt.stage = "compute"
                    attempt.active_link_ids = frozenset()
                    self._schedule_attempt(attempt)
                elif event.kind == "attempt_compute_complete":
                    self._begin_result_transfer(attempt)
                elif event.kind == "attempt_result_arrived":
                    self._accept_attempt(attempt)
                continue
            if self._is_stale(event):
                continue
            self._apply_event(event)
            processed = True
        return processed

    def _is_stale(self, event: Event) -> bool:
        if event.kind.startswith("node_"):
            runtime = self.node_runtime.get(event.target)
            return runtime is None or (
                event.payload.get("automatic")
                and runtime.generation != event.generation
            )
        if event.kind == "link_up" and event.payload.get("automatic"):
            return (
                self.link_runtime.get(event.target) != LinkState.DOWN
                or self.link_generation.get(event.target) != event.generation
            )
        return False

    def _apply_event(self, event: Event) -> None:
        if event.kind.startswith("fault:"):
            self._apply_fault(FaultKind(event.kind.split(":", 1)[1]), event)
            return
        if event.kind == "node_join":
            runtime = self.node_runtime[event.target]
            if event.payload.get("automatic") and runtime.generation != event.generation:
                return
            runtime.state = NodeState.ACTIVE
        elif event.kind == "node_resume":
            runtime = self.node_runtime[event.target]
            if runtime.state == NodeState.STALLED:
                runtime.state = NodeState.ACTIVE
        elif event.kind == "node_speed_restore":
            runtime = self.node_runtime[event.target]
            if runtime.state == NodeState.NEAR_ZERO:
                runtime.state = NodeState.ACTIVE
        elif event.kind == "link_up":
            self.link_generation[event.target] += 1
            self.link_runtime[event.target] = LinkState.UP
        for attempt in self.attempts.values():
            if attempt.state == AttemptState.RUNNING and attempt.stage == "compute":
                self._schedule_attempt(attempt)

    def _apply_fault(self, kind: FaultKind, event: Event) -> None:
        duration = event.payload.get("duration_seconds")
        if kind == FaultKind.LINK_DOWN:
            self.link_generation[event.target] += 1
            self.link_runtime[event.target] = LinkState.DOWN
            for attempt in self.attempts.values():
                if (
                    attempt.state == AttemptState.RUNNING
                    and attempt.stage in {"input", "result"}
                    and event.target in attempt.active_link_ids
                ):
                    attempt.state = AttemptState.FAILED
                    attempt.generation += 1
                    self._failed_attempts += 1
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "link_up",
                    target=event.target,
                    priority=0,
                    generation=self.link_generation[event.target],
                    payload={"automatic": True},
                )
            return
        if kind == FaultKind.LINK_UP:
            self.link_generation[event.target] += 1
            self.link_runtime[event.target] = LinkState.UP
            return

        runtime = self.node_runtime[event.target]
        runtime.generation += 1
        if kind == FaultKind.NODE_LEAVE:
            runtime.state = NodeState.OFFLINE
            self._fail_node_attempts(event.target)
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_join",
                    target=event.target,
                    priority=0,
                    generation=runtime.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_JOIN:
            runtime.state = NodeState.ACTIVE
        elif kind == FaultKind.NODE_STALL:
            runtime.state = NodeState.STALLED
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_resume",
                    target=event.target,
                    priority=0,
                    generation=runtime.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_RESUME:
            runtime.state = NodeState.ACTIVE
        elif kind == FaultKind.NODE_SPEED_DROP:
            runtime.state = NodeState.NEAR_ZERO
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_speed_restore",
                    target=event.target,
                    priority=0,
                    generation=runtime.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_SPEED_RESTORE:
            runtime.state = NodeState.ACTIVE
        elif kind == FaultKind.NODE_FAIL:
            runtime.state = NodeState.FAILED
            self._fail_node_attempts(event.target)
        for attempt in self.attempts.values():
            if attempt.state == AttemptState.RUNNING and attempt.stage == "compute":
                self._schedule_attempt(attempt)

    def _fail_node_attempts(self, node_id: str) -> None:
        for attempt in self.attempts.values():
            if attempt.node_id == node_id and attempt.state == AttemptState.RUNNING:
                attempt.state = AttemptState.FAILED
                attempt.generation += 1
                self._failed_attempts += 1

    def _effective_speed(self, node_id: str) -> float:
        state = self.node_runtime[node_id].state
        base = self.nodes[node_id].flops_per_second
        if state == NodeState.ACTIVE:
            return base
        if state == NodeState.NEAR_ZERO:
            return base * self.config.near_zero_speed_fraction
        return 0.0

    def _finish_collective(self) -> None:
        if self._allreduce_done or self._failed:
            return
        winner_attempts = [
            self.attempts[shard.winner_attempt_uuid]
            for shard in self.shards.values()
            if shard.winner_attempt_uuid is not None
        ]
        winner_nodes = tuple(dict.fromkeys(attempt.node_id for attempt in winner_attempts))
        completion_times = [attempt.completed_at or self.clock.now for attempt in winner_attempts]
        if completion_times:
            last = max(completion_times)
            self._barrier_wait_seconds += sum(last - done for done in completion_times)
        while not self._failed:
            participants = tuple(
                node_id
                for node_id in winner_nodes
                if self.node_runtime[node_id].state
                in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
            )
            if not participants:
                participants = self._wait_for_allreduce_route(winner_nodes)
                if participants is None:
                    self._failed = True
                    self._failure_reason = "no_active_allreduce_participant"
                    return
            estimate = self.network.ring_allreduce(
                participants,
                self.workload.gradient_bytes,
                self.link_runtime,
            )
            if estimate is None:
                participants = self._wait_for_allreduce_route(winner_nodes)
                if participants is None:
                    self._failed = True
                    self._failure_reason = "no_wan_route_for_allreduce"
                    return
                continue

            start = self.clock.now
            finish = start + estimate.duration_seconds
            interrupted = False
            while self.events and self.events.peek() is not None:
                event = self.events.peek()
                assert event is not None
                if event.time_seconds > finish:
                    break
                event = self.events.pop()
                if event.kind.startswith("attempt_"):
                    continue
                self.clock.advance_to(event.time_seconds)
                if self._is_stale(event):
                    continue
                self._apply_event(event)
                if event.kind == "fault:link_down" and event.target in estimate.links_used:
                    self._collective_seconds += self.clock.now - start
                    self._collective_restarts += 1
                    interrupted = True
                    break
                if event.kind.startswith("fault:node_") and event.target in participants:
                    if kind_is_membership_or_failure(event.kind):
                        self._collective_seconds += self.clock.now - start
                        self._collective_restarts += 1
                        interrupted = True
                        break
            if interrupted:
                continue
            self.clock.advance_to(finish)
            self._collective_seconds += finish - start
            self._allreduce_done = True
            self._batches_completed += 1
            self._committed_samples += self.workload.global_batch_size
            self._batch_history.append(
                {
                    "batch_index": self._batch_index,
                    "batch_uuid": self.batch_uuid,
                    "committed_at": self.clock.now,
                    "winner_attempt_uuids": [
                        shard.winner_attempt_uuid for shard in self.shards.values()
                    ],
                }
            )
            if self._batches_completed >= self.batches_per_episode:
                self._committed = True
                return
            self._batch_index += 1
            self._initialize_batch_work()
            self._allreduce_done = False
            return

    def _wait_for_allreduce_route(
        self, winner_nodes: Sequence[str]
    ) -> tuple[str, ...] | None:
        participants = tuple(
            node_id
            for node_id in winner_nodes
            if self.node_runtime[node_id].state
            in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
        )
        while self.events:
            if participants and self.network.ring_allreduce(
                participants, self.workload.gradient_bytes, self.link_runtime
            ) is not None:
                return participants
            event = self.events.pop()
            if event.kind.startswith("attempt_") or self._is_stale(event):
                continue
            self.clock.advance_to(event.time_seconds)
            self._apply_event(event)
            participants = tuple(
                node_id
                for node_id in winner_nodes
                if self.node_runtime[node_id].state
                in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
            )
        return None


def kind_is_membership_or_failure(kind: str) -> bool:
    return kind in {"fault:node_leave", "fault:node_fail", "fault:node_stall"}
