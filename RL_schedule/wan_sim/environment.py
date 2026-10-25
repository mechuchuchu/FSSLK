"""Synchronous data-parallel WAN simulation with virtual time and explicit events.

The core has no Gymnasium, RL, tensor, or GPU dependency. A caller supplies a
per-node sample allocation for one global batch; the environment models the
local compute, barrier, ring AllReduce, commit, and configured fault events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .clock import VirtualClock
from .compute_model import ComputeModel
from .events import Event, EventQueue
from .network_model import CollectiveEstimate, WANNetworkModel
from .recovery import RecoveryModel
from .types import (
    EpisodeMetrics,
    FaultEvent,
    FaultKind,
    LinkState,
    NodeState,
    SimulationConfig,
    StepResult,
    Topology,
    Workload,
    validate_fault_schedule,
)

_EPSILON = 1e-7


@dataclass
class _NodeRuntime:
    state: NodeState = NodeState.ACTIVE
    generation: int = 0
    recovery_started_at: float | None = None


@dataclass
class _LinkRuntime:
    state: LinkState = LinkState.UP
    generation: int = 0


@dataclass
class _ComputeTask:
    node_id: str
    remaining_flops: float
    remaining_overhead_seconds: float
    started_at: float
    last_progress_at: float
    generation: int = 0
    completed_at: float | None = None
    compute_seconds: float = 0.0

    @property
    def complete(self) -> bool:
        return self.completed_at is not None


@dataclass
class _Counters:
    committed_samples: int = 0
    step_count: int = 0
    failed_steps: int = 0
    compute_seconds: float = 0.0
    collective_seconds: float = 0.0
    barrier_wait_seconds: float = 0.0
    recovery_seconds: float = 0.0
    recovery_stages: dict[str, float] = field(
        default_factory=lambda: {
            "detection": 0.0,
            "restart": 0.0,
            "membership": 0.0,
            "restore": 0.0,
        }
    )


class SynchronousWANScheduler:
    """Event-driven environment for synchronous data-parallel training.

    Fault schedules use absolute episode seconds. Same-time faults are handled
    before compute or collective completions, so a fault exactly at completion
    invalidates that completion deterministically.
    """

    def __init__(
        self,
        topology: Topology,
        workload: Workload,
        config: SimulationConfig | None = None,
        fault_schedule: Sequence[FaultEvent] = (),
    ) -> None:
        topology.validate()
        workload.validate()
        self.config = config or SimulationConfig()
        self.config.validate()
        validate_fault_schedule(fault_schedule, topology)

        self.topology = topology
        self.workload = workload
        self.fault_schedule = tuple(fault_schedule)
        self.nodes = {node.node_id: node for node in topology.nodes}
        self.links = {link.link_id: link for link in topology.links}
        self.network = WANNetworkModel(topology)
        self.recovery = RecoveryModel(self.config.recovery)
        self.clock = VirtualClock()
        self.events = EventQueue()
        self.node_runtime: dict[str, _NodeRuntime] = {}
        self.link_runtime: dict[str, _LinkRuntime] = {}
        self.counters = _Counters()
        self._tasks: dict[str, _ComputeTask] = {}
        self._step_generation = 0
        self._collective_generation = 0
        self.reset()

    def reset(
        self, fault_schedule: Sequence[FaultEvent] | None = None
    ) -> Mapping[str, Any]:
        if fault_schedule is not None:
            validate_fault_schedule(fault_schedule, self.topology)
            self.fault_schedule = tuple(fault_schedule)
        self.clock = VirtualClock()
        self.events = EventQueue()
        self.node_runtime = {
            node_id: _NodeRuntime() for node_id in self.nodes
        }
        self.link_runtime = {
            link_id: _LinkRuntime() for link_id in self.links
        }
        self.counters = _Counters()
        self._tasks = {}
        self._step_generation = 0
        self._collective_generation = 0
        for fault in self.fault_schedule:
            self.events.push(
                fault.time_seconds,
                f"fault:{fault.kind.value}",
                target=fault.target,
                priority=0,
                payload={"duration_seconds": fault.duration_seconds},
            )
        self._process_events_at_current_time()
        return self.state()

    def state(self) -> Mapping[str, Any]:
        return {
            "time_seconds": self.clock.now,
            "committed_samples": self.counters.committed_samples,
            "step_count": self.counters.step_count,
            "nodes": {
                node_id: runtime.state.value
                for node_id, runtime in self.node_runtime.items()
            },
            "links": {
                link_id: runtime.state.value
                for link_id, runtime in self.link_runtime.items()
            },
        }

    def advance_to_next_event(self) -> Event | None:
        """Advance an idle simulator to its next lifecycle event."""
        while self.events:
            event = self.events.pop()
            if event.kind in {"compute_complete", "collective_complete"}:
                continue
            self.clock.advance_to(event.time_seconds)
            if self._is_stale_lifecycle_event(event):
                continue
            self._apply_lifecycle_event(event)
            return event
        return None

    def step(
        self,
        local_samples: Mapping[str, int],
        micro_batch_sizes: Mapping[str, int] | None = None,
    ) -> StepResult:
        """Execute one synchronous global batch and commit it only on success."""
        self._process_events_at_current_time()
        if self.counters.committed_samples >= self.workload.target_samples:
            raise RuntimeError("target sample count has already been reached")
        allocation = self._validate_allocation(local_samples)
        participants = tuple(node_id for node_id, count in allocation.items() if count > 0)
        selected_micro_batches = self._validate_micro_batch_sizes(
            micro_batch_sizes, participants, allocation
        )
        start = self.clock.now
        recovery_before = self.counters.recovery_seconds
        self._step_generation += 1
        step_generation = self._step_generation
        tasks: dict[str, _ComputeTask] = {}
        self._tasks = tasks
        for node_id in participants:
            node = self.nodes[node_id]
            local_flops = ComputeModel.work_flops(
                node, self.workload, allocation[node_id]
            )
            local_duration = ComputeModel.duration_seconds(
                node,
                self.workload,
                allocation[node_id],
                selected_micro_batches[node_id],
            )
            tasks[node_id] = _ComputeTask(
                node_id=node_id,
                remaining_flops=local_flops,
                remaining_overhead_seconds=max(
                    0.0, local_duration - local_flops / node.flops_per_second
                ),
                started_at=self.clock.now,
                last_progress_at=self.clock.now,
            )
        self._schedule_compute_completions(step_generation)
        events_processed = 0
        failed_reason: str | None = None

        while not all(task.complete for task in tasks.values()):
            event = self.events.pop() if self.events else None
            if event is None:
                failed_reason = "compute_deadlock_no_future_event"
                break
            if event.kind == "compute_complete":
                task = tasks.get(event.target)
                if (
                    event.payload.get("step_generation") != step_generation
                    or task is None
                    or task.generation != event.generation
                    or task.complete
                ):
                    continue
            elif event.kind == "collective_complete":
                continue
            self._advance_compute_progress(event.time_seconds, tasks)
            events_processed += 1
            if events_processed > self.config.max_events_per_step:
                failed_reason = "max_events_per_step_exceeded"
                break
            if event.kind == "compute_complete":
                if (
                    event.payload.get("step_generation") == step_generation
                    and event.target in tasks
                    and tasks[event.target].generation == event.generation
                    and self.node_runtime[event.target].state
                    in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
                ):
                    task = tasks[event.target]
                    task.remaining_flops = 0.0
                    task.remaining_overhead_seconds = 0.0
                    task.completed_at = self.clock.now
                continue

            if self._is_stale_lifecycle_event(event):
                continue
            self._apply_lifecycle_event(event)
            if event.kind.startswith("fault:"):
                failed_node = self._failed_participant(event, participants)
                if failed_node is not None:
                    failed_reason = f"participant_{failed_node}_{self.node_runtime[failed_node].state.value}"
                    self._process_events_at_current_time()
                    break
            self._schedule_compute_completions(step_generation)

        compute_seconds = sum(task.compute_seconds for task in tasks.values())
        completion_times = {
            node_id: task.completed_at
            for node_id, task in tasks.items()
            if task.completed_at is not None
        }
        if failed_reason is None and not all(task.complete for task in tasks.values()):
            failed_reason = "compute_step_incomplete"

        barrier_wait = 0.0
        if completion_times:
            last_compute = max(completion_times.values())
            barrier_wait = sum(last_compute - value for value in completion_times.values())

        collective_seconds = 0.0
        if failed_reason is None:
            estimate = self.network.ring_allreduce(
                participants,
                self.workload.gradient_bytes,
                self._link_states(),
            )
            if estimate is None:
                failed_reason = "no_wan_route_for_collective"
            else:
                collective_seconds, collective_events, collective_reason = (
                    self._run_collective(estimate, participants)
                )
                events_processed += collective_events
                failed_reason = collective_reason

        committed = failed_reason is None
        self.counters.collective_seconds += collective_seconds
        self.counters.barrier_wait_seconds += barrier_wait
        if committed:
            self.counters.committed_samples += self.workload.global_batch_size
            self.counters.step_count += 1
        else:
            self.counters.failed_steps += 1
            self._collective_generation += 1
        self.counters.compute_seconds += compute_seconds
        self._tasks = {}
        return StepResult(
            committed=committed,
            start_seconds=start,
            end_seconds=self.clock.now,
            committed_samples=(
                self.workload.global_batch_size if committed else 0
            ),
            failed_reason=failed_reason,
            compute_seconds=compute_seconds,
            collective_seconds=collective_seconds,
            barrier_wait_seconds=barrier_wait,
            recovery_seconds=self.counters.recovery_seconds - recovery_before,
            participant_completion_seconds={
                node_id: value - start for node_id, value in completion_times.items()
            },
            micro_batch_sizes=selected_micro_batches,
            events_processed=events_processed,
        )

    def metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            elapsed_seconds=self.clock.now,
            committed_samples=self.counters.committed_samples,
            step_count=self.counters.step_count,
            failed_steps=self.counters.failed_steps,
            compute_seconds=self.counters.compute_seconds,
            collective_seconds=self.counters.collective_seconds,
            barrier_wait_seconds=self.counters.barrier_wait_seconds,
            recovery_seconds=self.counters.recovery_seconds,
            recovery_stage_seconds=dict(self.counters.recovery_stages),
            goodput_samples_per_second=(
                self.counters.committed_samples / self.clock.now
                if self.clock.now > 0
                else 0.0
            ),
        )

    def _validate_allocation(self, allocation: Mapping[str, int]) -> dict[str, int]:
        unknown = set(allocation) - set(self.nodes)
        if unknown:
            raise ValueError(f"allocation references unknown nodes: {sorted(unknown)}")
        normalized: dict[str, int] = {}
        for node_id in self.nodes:
            value = allocation.get(node_id, 0)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("local sample counts must be integers")
            normalized[node_id] = value
        if any(value < 0 for value in normalized.values()):
            raise ValueError("local sample counts must be non-negative")
        if sum(normalized.values()) != self.workload.global_batch_size:
            raise ValueError("local sample counts must sum to the global batch size")
        participants = [node_id for node_id, count in normalized.items() if count > 0]
        if not participants:
            raise ValueError("at least one node must participate in a global batch")
        for node_id in participants:
            if self.node_runtime[node_id].state not in {
                NodeState.ACTIVE,
                NodeState.NEAR_ZERO,
            }:
                raise ValueError(f"node {node_id} is not active")
            if normalized[node_id] > self.nodes[node_id].max_local_batch:
                raise ValueError(f"local batch exceeds {node_id}'s configured limit")
        return normalized

    def _validate_micro_batch_sizes(
        self,
        micro_batch_sizes: Mapping[str, int] | None,
        participants: Sequence[str],
        allocation: Mapping[str, int],
    ) -> dict[str, int]:
        supplied = {} if micro_batch_sizes is None else dict(micro_batch_sizes)
        unexpected = set(supplied) - set(participants)
        if unexpected:
            raise ValueError(
                f"micro-batch sizes supplied for non-participants: {sorted(unexpected)}"
            )
        selected: dict[str, int] = {}
        for node_id in participants:
            size = supplied.get(node_id, self.workload.default_micro_batch_size)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("micro-batch sizes must be positive integers")
            peak_memory = ComputeModel.peak_memory_bytes(
                self.workload, allocation[node_id], size
            )
            if peak_memory > self.nodes[node_id].memory_bytes:
                raise ValueError(
                    f"micro-batch size {size} exceeds {node_id}'s memory limit"
                )
            selected[node_id] = size
        return selected

    def _schedule_compute_completions(self, step_generation: int) -> None:
        for node_id, task in self._tasks.items():
            if task.complete:
                continue
            speed = self._effective_flops_per_second(node_id)
            if (
                task.remaining_flops <= _EPSILON
                and task.remaining_overhead_seconds <= _EPSILON
            ):
                if speed > 0:
                    task.completed_at = self.clock.now
                continue
            task.generation += 1
            if speed <= 0:
                continue
            duration = (
                task.remaining_flops / speed + task.remaining_overhead_seconds
            )
            self.events.push(
                self.clock.now + duration,
                "compute_complete",
                target=node_id,
                priority=10,
                generation=task.generation,
                payload={"step_generation": step_generation},
            )

    def _advance_compute_progress(
        self, to_time: float, tasks: Mapping[str, _ComputeTask]
    ) -> None:
        elapsed = self.clock.advance_to(to_time)
        if elapsed <= 0:
            return
        for node_id, task in tasks.items():
            speed = self._effective_flops_per_second(node_id)
            if task.complete or speed <= 0:
                continue
            compute_seconds_left = task.remaining_flops / speed
            overhead_before = task.remaining_overhead_seconds
            compute_elapsed = min(elapsed, compute_seconds_left)
            progress = min(task.remaining_flops, speed * compute_elapsed)
            task.remaining_flops -= progress
            task.remaining_overhead_seconds = max(
                0.0,
                overhead_before - max(0.0, elapsed - compute_elapsed),
            )
            task.compute_seconds += min(
                elapsed,
                compute_seconds_left + overhead_before,
            )
            task.last_progress_at = self.clock.now

    def _run_collective(
        self,
        estimate: CollectiveEstimate,
        participants: Sequence[str],
    ) -> tuple[float, int, str | None]:
        if estimate.duration_seconds <= 0:
            return 0.0, 0, None
        self._collective_generation += 1
        generation = self._collective_generation
        start = self.clock.now
        self.events.push(
            start + estimate.duration_seconds,
            "collective_complete",
            priority=10,
            generation=generation,
        )
        processed = 0
        while self.events:
            event = self.events.pop()
            if event.kind == "collective_complete" and event.generation != generation:
                continue
            if event.kind == "compute_complete":
                continue
            self.clock.advance_to(event.time_seconds)
            processed += 1
            if processed > self.config.max_events_per_step:
                return self.clock.now - start, processed, "max_events_per_step_exceeded"
            if event.kind == "collective_complete":
                return self.clock.now - start, processed, None
            if self._is_stale_lifecycle_event(event):
                continue
            self._apply_lifecycle_event(event)
            if event.kind.startswith("fault:"):
                if event.kind in {
                    "fault:link_down",
                    "fault:node_stall",
                    "fault:node_leave",
                    "fault:node_fail",
                }:
                    if event.kind == "fault:link_down" and event.target not in estimate.links_used:
                        continue
                    if event.kind.startswith("fault:node_") and event.target not in participants:
                        continue
                    self._collective_generation += 1
                    self._process_events_at_current_time()
                    return self.clock.now - start, processed, f"collective_interrupted_by_{event.kind[6:]}"
        return self.clock.now - start, processed, "collective_deadlock_no_future_event"

    def _apply_lifecycle_event(self, event: Event) -> None:
        if event.kind.startswith("fault:"):
            kind = FaultKind(event.kind.split(":", 1)[1])
            self._apply_fault(kind, event)
            return
        node = self.node_runtime.get(event.target)
        if event.kind == "node_detected" and node is not None:
            if node.generation != event.generation or node.state != NodeState.FAILED:
                return
            node.state = NodeState.RECOVERING
            self.counters.recovery_stages["detection"] += self.config.recovery.detection_seconds
            self.events.push(
                self.clock.now + self.config.recovery.restart_seconds,
                "node_restarted",
                target=event.target,
                priority=0,
                generation=event.generation,
            )
        elif event.kind == "node_restarted" and node is not None:
            if node.generation != event.generation or node.state != NodeState.RECOVERING:
                return
            self.counters.recovery_stages["restart"] += self.config.recovery.restart_seconds
            self.events.push(
                self.clock.now + self.config.recovery.membership_seconds,
                "node_membership_ready",
                target=event.target,
                priority=0,
                generation=event.generation,
            )
        elif event.kind == "node_membership_ready" and node is not None:
            if node.generation != event.generation or node.state != NodeState.RECOVERING:
                return
            self.counters.recovery_stages["membership"] += self.config.recovery.membership_seconds
            self.events.push(
                self.clock.now + self.config.recovery.restore_seconds,
                "node_restored",
                target=event.target,
                priority=0,
                generation=event.generation,
            )
        elif event.kind == "node_restored" and node is not None:
            if node.generation != event.generation or node.state != NodeState.RECOVERING:
                return
            self.counters.recovery_stages["restore"] += self.config.recovery.restore_seconds
            node.state = NodeState.ACTIVE
            if node.recovery_started_at is not None:
                self.counters.recovery_seconds += self.clock.now - node.recovery_started_at
                node.recovery_started_at = None
        elif event.kind == "node_join" and node is not None:
            if event.payload.get("automatic") and node.generation != event.generation:
                return
            node.state = NodeState.ACTIVE
            if node.recovery_started_at is not None:
                self.counters.recovery_seconds += (
                    self.clock.now - node.recovery_started_at
                )
                node.recovery_started_at = None
        elif event.kind in {"node_resume", "node_speed_restore"} and node is not None:
            if event.payload.get("automatic") and node.generation != event.generation:
                return
            if event.kind == "node_resume" and node.state == NodeState.STALLED:
                node.state = NodeState.ACTIVE
            elif event.kind == "node_speed_restore" and node.state == NodeState.NEAR_ZERO:
                node.state = NodeState.ACTIVE
        elif event.kind == "link_up" and event.target in self.link_runtime:
            link = self.link_runtime[event.target]
            if event.payload.get("automatic") and link.generation != event.generation:
                return
            link.state = LinkState.UP

    def _apply_fault(self, kind: FaultKind, event: Event) -> None:
        duration = event.payload.get("duration_seconds")
        if kind in {FaultKind.LINK_DOWN, FaultKind.LINK_UP}:
            link = self.link_runtime[event.target]
            link.generation += 1
            link.state = LinkState.DOWN if kind == FaultKind.LINK_DOWN else LinkState.UP
            if kind == FaultKind.LINK_DOWN and duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "link_up",
                    target=event.target,
                    priority=0,
                    generation=link.generation,
                    payload={"automatic": True},
                )
            return

        node = self.node_runtime[event.target]
        if kind == FaultKind.NODE_STALL:
            node.generation += 1
            node.state = NodeState.STALLED
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_resume",
                    target=event.target,
                    priority=0,
                    generation=node.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_SPEED_DROP:
            node.generation += 1
            node.state = NodeState.NEAR_ZERO
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_speed_restore",
                    target=event.target,
                    priority=0,
                    generation=node.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_RESUME:
            node.generation += 1
            if node.state == NodeState.STALLED:
                node.state = NodeState.ACTIVE
        elif kind == FaultKind.NODE_SPEED_RESTORE:
            node.generation += 1
            if node.state == NodeState.NEAR_ZERO:
                node.state = NodeState.ACTIVE
        elif kind == FaultKind.NODE_LEAVE:
            node.generation += 1
            node.state = NodeState.OFFLINE
            node.recovery_started_at = self.clock.now
            if duration is not None:
                self.events.push(
                    self.clock.now + float(duration),
                    "node_join",
                    target=event.target,
                    priority=0,
                    generation=node.generation,
                    payload={"automatic": True},
                )
        elif kind == FaultKind.NODE_JOIN:
            node.generation += 1
            node.state = NodeState.ACTIVE
            if node.recovery_started_at is not None:
                self.counters.recovery_seconds += (
                    self.clock.now - node.recovery_started_at
                )
                node.recovery_started_at = None
        elif kind == FaultKind.NODE_FAIL:
            node.generation += 1
            node.state = NodeState.FAILED
            node.recovery_started_at = self.clock.now
            self.recovery.schedule(self.events, self.clock.now, event.target, node.generation)

    def _is_stale_lifecycle_event(self, event: Event) -> bool:
        if event.kind.startswith("node_"):
            node = self.node_runtime.get(event.target)
            if node is None:
                return True
            if event.payload.get("automatic"):
                return node.generation != event.generation
            if event.kind in {"node_detected", "node_restarted", "node_membership_ready", "node_restored"}:
                return node.generation != event.generation
        if event.kind == "link_up" and event.payload.get("automatic"):
            link = self.link_runtime.get(event.target)
            return link is None or link.generation != event.generation
        return False

    def _effective_flops_per_second(self, node_id: str) -> float:
        state = self.node_runtime[node_id].state
        base = self.nodes[node_id].flops_per_second
        if state == NodeState.ACTIVE:
            return base
        if state == NodeState.NEAR_ZERO:
            return base * self.config.near_zero_speed_fraction
        return 0.0

    def _failed_participant(
        self, event: Event, participants: Sequence[str]
    ) -> str | None:
        if event.kind not in {
            "fault:node_leave",
            "fault:node_fail",
        }:
            return None
        if (
            event.target in participants
            and self.node_runtime[event.target].state
            in {NodeState.OFFLINE, NodeState.FAILED, NodeState.RECOVERING}
        ):
            return event.target
        return None

    def _link_states(self) -> Mapping[str, LinkState]:
        return {
            link_id: runtime.state for link_id, runtime in self.link_runtime.items()
        }

    def _process_events_at_current_time(self) -> None:
        while self.events and self.events.peek() is not None:
            event = self.events.peek()
            assert event is not None
            if event.time_seconds > self.clock.now:
                return
            event = self.events.pop()
            if event.kind in {"compute_complete", "collective_complete"}:
                continue
            if not self._is_stale_lifecycle_event(event):
                self._apply_lifecycle_event(event)
