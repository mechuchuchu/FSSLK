"""Optional Gymnasium adapter around the RL-independent WAN simulation core."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .compute_model import ComputeModel
from .environment import SynchronousWANScheduler
from .types import (
    FaultEvent,
    LinkState,
    NodeState,
    SimulationConfig,
    Topology,
    Workload,
)


class GymnasiumSynchronousWANEnv(gym.Env[np.ndarray, dict[str, np.ndarray]]):
    """A provisional Gym interface for batch allocation and local micro-batches.

    The underlying simulator remains independent of this action encoding.
    ``allocation_weights`` are projected to a feasible integer split of the
    fixed global batch. ``micro_batch_indices`` select one configured size per
    node; entries for nodes with no assigned samples are ignored.
    """

    metadata = {"render_modes": []}
    _NODE_STATES = tuple(NodeState)

    def __init__(
        self,
        topology: Topology,
        workload: Workload,
        config: SimulationConfig | None = None,
        fault_schedule: Sequence[FaultEvent] = (),
        micro_batch_options: Sequence[int] = (32, 48, 64),
        max_episode_steps: int = 1000,
        time_scale_seconds: float = 100.0,
    ) -> None:
        super().__init__()
        topology.validate()
        workload.validate()
        if not micro_batch_options or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in micro_batch_options
        ):
            raise ValueError("micro_batch_options must contain positive sizes")
        if len(set(micro_batch_options)) != len(micro_batch_options):
            raise ValueError("micro_batch_options must be unique")
        if (
            max_episode_steps < 1
            or not np.isfinite(time_scale_seconds)
            or time_scale_seconds <= 0
        ):
            raise ValueError("episode step limit and time scale must be positive")

        self.topology = topology
        self.workload = workload
        self.config = config or SimulationConfig()
        self.fault_schedule = tuple(fault_schedule)
        self.micro_batch_options = tuple(int(size) for size in micro_batch_options)
        self.max_episode_steps = max_episode_steps
        self.time_scale_seconds = float(time_scale_seconds)
        self.node_ids = tuple(node.node_id for node in topology.nodes)
        self.link_ids = tuple(link.link_id for link in topology.links)
        self.node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self.simulator = SynchronousWANScheduler(
            topology,
            workload,
            self.config,
            fault_schedule=self.fault_schedule,
        )
        self._max_flops = max(node.flops_per_second for node in topology.nodes)
        self._max_memory = max(node.memory_bytes for node in topology.nodes)
        self._max_link_bandwidth = max(
            (link.bytes_per_second for link in topology.links), default=1.0
        )
        self._max_link_latency = max(
            (link.latency_seconds for link in topology.links), default=1.0
        )
        self._last_allocation = {node_id: 0 for node_id in self.node_ids}
        self._last_micro_batches = {
            node_id: workload.default_micro_batch_size for node_id in self.node_ids
        }
        self._decision_count = 0

        self.action_space = spaces.Dict(
            {
                "allocation_weights": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self.node_ids),),
                    dtype=np.float32,
                ),
                "micro_batch_indices": spaces.MultiDiscrete(
                    np.full(len(self.node_ids), len(self.micro_batch_options), dtype=np.int64)
                ),
            }
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.node_ids) * 11 + len(self.link_ids) * 3 + 5,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        schedule = self.fault_schedule
        if options is not None and "fault_schedule" in options:
            schedule = tuple(options["fault_schedule"])
        state = self.simulator.reset(schedule)
        self._last_allocation = {node_id: 0 for node_id in self.node_ids}
        self._last_micro_batches = {
            node_id: self.workload.default_micro_batch_size for node_id in self.node_ids
        }
        self._decision_count = 0
        return self._observation(), {
            "state": state,
            "micro_batch_action_mask": self._micro_batch_action_mask(),
        }

    def step(
        self, action: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        weights = np.asarray(action["allocation_weights"], dtype=np.float64)
        raw_batch_indices = np.asarray(action["micro_batch_indices"])
        if not np.issubdtype(raw_batch_indices.dtype, np.integer):
            raise ValueError("micro_batch_indices must contain integers")
        batch_indices = raw_batch_indices.astype(np.int64, copy=False)
        if weights.shape != (len(self.node_ids),):
            raise ValueError("allocation_weights has the wrong shape")
        if batch_indices.shape != (len(self.node_ids),):
            raise ValueError("micro_batch_indices has the wrong shape")
        if np.any(batch_indices < 0) or np.any(batch_indices >= len(self.micro_batch_options)):
            raise ValueError("micro_batch_indices contains an invalid option")
        if not np.all(np.isfinite(weights)):
            raise ValueError("allocation_weights must be finite")

        weights = np.clip(weights, 0.0, 1.0)
        allocation = self._project_allocation(weights, batch_indices)
        if allocation is None:
            waited_seconds = self._advance_for_capacity(batch_indices)
            observation = self._observation()
            self._decision_count += 1
            terminal = waited_seconds is None
            return (
                observation,
                -(waited_seconds or 0.0),
                terminal,
                self._decision_count >= self.max_episode_steps and not terminal,
                {
                    "failed_reason": "insufficient_active_capacity",
                    "state": self.simulator.state(),
                    "metrics": asdict(self.simulator.metrics()),
                    "waited_for_membership_event": waited_seconds is not None,
                    "waited_seconds": waited_seconds or 0.0,
                    "micro_batch_action_mask": self._micro_batch_action_mask(),
                },
            )

        micro_batches = {
            node_id: self.micro_batch_options[int(batch_indices[index])]
            for index, node_id in enumerate(self.node_ids)
            if allocation[node_id] > 0
        }
        result = self.simulator.step(allocation, micro_batches)
        waited_for_link_seconds = 0.0
        link_wait_failed = False
        if result.failed_reason == "no_wan_route_for_collective" or (
            result.failed_reason is not None
            and result.failed_reason.startswith("collective_interrupted_by_link_down")
        ):
            waited_for_link_seconds = self._advance_until_allreduce_route(
                tuple(node_id for node_id, count in allocation.items() if count > 0)
            )
            link_wait_failed = waited_for_link_seconds is None
        self._last_allocation = allocation
        self._last_micro_batches.update(micro_batches)
        self._decision_count += 1
        elapsed = result.end_seconds - result.start_seconds + (waited_for_link_seconds or 0.0)
        terminated = (
            self.simulator.counters.committed_samples >= self.workload.target_samples
            or link_wait_failed
        )
        truncated = self._decision_count >= self.max_episode_steps and not terminated
        info = {
            "step_result": asdict(result),
            "state": self.simulator.state(),
            "metrics": asdict(self.simulator.metrics()),
            "allocation": allocation,
            "micro_batch_sizes": micro_batches,
            "waited_for_link_seconds": waited_for_link_seconds or 0.0,
            "micro_batch_action_mask": self._micro_batch_action_mask(allocation),
        }
        return self._observation(), -elapsed, terminated, truncated, info

    def _project_allocation(
        self, weights: np.ndarray, batch_indices: np.ndarray
    ) -> dict[str, int] | None:
        active = [
            node_id
            for node_id in self.node_ids
            if self.simulator.node_runtime[node_id].state
            in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
        ]
        capacities = {
            node_id: min(
                self.workload.global_batch_size,
                self.topology.nodes[self.node_index[node_id]].max_local_batch,
            )
            for node_id in active
        }
        for node_id in tuple(capacities):
            node_index = self.node_index[node_id]
            node = self.topology.nodes[node_index]
            size = self.micro_batch_options[int(batch_indices[node_index])]
            activation_budget = node.memory_bytes - self.workload.model_state_bytes
            if activation_budget < 0:
                capacities[node_id] = 0
            elif self.workload.activation_bytes_per_sample > 0:
                max_live_samples = (
                    activation_budget // self.workload.activation_bytes_per_sample
                )
                if max_live_samples < size:
                    capacities[node_id] = min(
                        capacities[node_id], int(max_live_samples)
                    )
        if sum(capacities.values()) < self.workload.global_batch_size:
            return None

        allocation = {node_id: 0 for node_id in self.node_ids}
        remaining = self.workload.global_batch_size
        while remaining > 0:
            eligible = [
                node_id for node_id in active if allocation[node_id] < capacities[node_id]
            ]
            if not eligible:
                return None
            eligible_weights = np.asarray(
                [weights[self.node_index[node_id]] for node_id in eligible],
                dtype=np.float64,
            )
            if eligible_weights.sum() <= 0:
                eligible_weights.fill(1.0)
            quotas = remaining * eligible_weights / eligible_weights.sum()
            bases = np.floor(quotas).astype(np.int64)
            assigned = 0
            for index, node_id in enumerate(eligible):
                room = capacities[node_id] - allocation[node_id]
                amount = min(int(bases[index]), room)
                allocation[node_id] += amount
                assigned += amount
            remaining -= assigned
            if remaining == 0:
                break
            fractions = sorted(
                range(len(eligible)),
                key=lambda index: (-(quotas[index] - bases[index]), index),
            )
            for index in fractions:
                node_id = eligible[index]
                if allocation[node_id] < capacities[node_id]:
                    allocation[node_id] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
            if assigned == 0 and not fractions:
                return None
        return allocation

    def _advance_for_capacity(self, batch_indices: np.ndarray) -> float | None:
        started = self.simulator.clock.now
        while self.simulator.events:
            event = self.simulator.advance_to_next_event()
            if event is None:
                continue
            if self._project_allocation(
                np.ones(len(self.node_ids), dtype=np.float64), batch_indices
            ) is not None:
                return self.simulator.clock.now - started
        return None

    def _micro_batch_action_mask(
        self, allocation: Mapping[str, int] | None = None
    ) -> np.ndarray:
        mask = np.zeros(
            (len(self.node_ids), len(self.micro_batch_options)), dtype=np.int8
        )
        for node_index, node_id in enumerate(self.node_ids):
            if self.simulator.node_runtime[node_id].state not in {
                NodeState.ACTIVE,
                NodeState.NEAR_ZERO,
            }:
                continue
            local_samples = (
                min(
                    self.workload.global_batch_size,
                    self.topology.nodes[node_index].max_local_batch,
                )
                if allocation is None
                else allocation.get(node_id, 0)
            )
            if local_samples == 0:
                mask[node_index, :] = 1
                continue
            node = self.topology.nodes[node_index]
            for size_index, size in enumerate(self.micro_batch_options):
                mask[node_index, size_index] = int(
                    ComputeModel.peak_memory_bytes(
                        self.workload, local_samples, size
                    )
                    <= node.memory_bytes
                )
        return mask

    def _advance_until_allreduce_route(
        self, participant_ids: Sequence[str]
    ) -> float | None:
        started = self.simulator.clock.now
        while self.simulator.events:
            event = self.simulator.advance_to_next_event()
            if event is None:
                continue
            participants_ready = all(
                self.simulator.node_runtime[node_id].state
                in {NodeState.ACTIVE, NodeState.NEAR_ZERO}
                for node_id in participant_ids
            )
            if not participants_ready:
                continue
            if self.simulator.network.ring_allreduce(
                participant_ids,
                self.workload.gradient_bytes,
                self.simulator._link_states(),
            ) is not None:
                return self.simulator.clock.now - started
        return None

    def _observation(self) -> np.ndarray:
        state = self.simulator.state()
        features: list[float] = []
        for node in self.topology.nodes:
            node_state = NodeState(state["nodes"][node.node_id])
            features.extend(float(node_state == candidate) for candidate in self._NODE_STATES)
            features.extend(
                [
                    math_log_ratio(node.flops_per_second, self._max_flops),
                    math_log_ratio(node.memory_bytes, self._max_memory),
                    min(1.0, node.max_local_batch / max(1, self.workload.global_batch_size)),
                    min(1.0, self._last_allocation[node.node_id] / max(1, self.workload.global_batch_size)),
                    min(
                        1.0,
                        self._last_micro_batches[node.node_id]
                        / max(self.micro_batch_options),
                    ),
                ]
            )
        for link in self.topology.links:
            is_up = state["links"][link.link_id] == LinkState.UP.value
            features.extend(
                [
                    float(is_up),
                    math_log_ratio(link.bytes_per_second, self._max_link_bandwidth),
                    math_log_ratio(link.latency_seconds, self._max_link_latency),
                ]
            )
        features.extend(
            [
                min(1.0, state["committed_samples"] / self.workload.target_samples),
                min(1.0, state["time_seconds"] / self.time_scale_seconds),
                min(1.0, state["step_count"] / max(1, self.workload.target_samples // self.workload.global_batch_size)),
                min(1.0, self.simulator.counters.failed_steps / 100.0),
                min(1.0, self._decision_count / self.max_episode_steps),
            ]
        )
        return np.asarray(features, dtype=np.float32)


def math_log_ratio(value: float, maximum: float) -> float:
    if maximum <= 0 or value <= 0:
        return 0.0
    return min(1.0, float(np.log1p(value) / np.log1p(maximum)))
