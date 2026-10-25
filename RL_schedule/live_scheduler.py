"""Live telemetry encoding and policy actions for asynchronous worker jobs."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .rl_policy import _load_policy
from .simulator import SimulationConfig, WorkerSpec


@dataclass
class LiveAttempt:
    shard_id: int
    worker_index: int
    attempt_id: int
    task_id: str
    start_time: float
    status: str = "queued"
    progress: float = 0.0
    heartbeat_age_seconds: float = 0.0
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status in {"queued", "running"}


class LiveObservationHistory:
    """Builds the same feature layout the trained simulator policy receives."""

    def __init__(
        self,
        workers: Sequence[WorkerSpec],
        config: SimulationConfig,
        history_length: int,
    ) -> None:
        self.workers = tuple(workers)
        self.config = config
        self.history_length = history_length
        self.global_feature_size = 5
        self.worker_feature_size = 11
        self.frame_size = self.global_feature_size + len(workers) * self.worker_feature_size
        self._histories: dict[int, deque[torch.Tensor]] = {}
        self._previous_progress: dict[tuple[int, int], float] = {}
        self._previous_rates: dict[tuple[int, int], float] = {}
        self._previous_observation_time: dict[int, float] = {}
        self._last_progress_time: dict[int, float] = {}
        self._last_global_progress: dict[int, float] = {}

    def reset_shard(self, shard_id: int) -> None:
        self._histories.pop(shard_id, None)
        self._previous_observation_time.pop(shard_id, None)
        self._last_progress_time.pop(shard_id, None)
        self._last_global_progress.pop(shard_id, None)
        for worker_index in range(len(self.workers)):
            self._previous_progress.pop((shard_id, worker_index), None)
            self._previous_rates.pop((shard_id, worker_index), None)

    def observe(
        self,
        shard_id: int,
        now: float,
        attempts: Sequence[LiveAttempt],
        worker_latest_for_shard: dict[int, LiveAttempt],
    ) -> torch.Tensor:
        previous_time = self._previous_observation_time.get(shard_id, now)
        delta_time = max(0.0, now - previous_time)
        worker_values: list[float] = []
        current_rates: list[float] = []
        target_progresses: list[float] = []
        for worker_index, worker in enumerate(self.workers):
            run = worker_latest_for_shard.get(worker_index)
            key = (shard_id, worker_index)
            if run is None:
                progress = 0.0
                active = 0.0
                rate = 0.0
                task_age = 0.0
                heartbeat_alive = 1.0
                heartbeat_age = 0.0
            else:
                progress = min(1.0, max(0.0, run.progress))
                active = 1.0 if run.active else 0.0
                task_age = max(0.0, now - run.start_time)
                old_progress = self._previous_progress.get(key, progress)
                rate = (
                    max(0.0, progress - old_progress) / delta_time
                    if delta_time > 0
                    else 0.0
                )
                heartbeat_age = max(0.0, run.heartbeat_age_seconds)
                heartbeat_alive = 1.0 if heartbeat_age <= 1.5 else 0.0

            nominal_rate = worker.compute_units_per_second / self.config.work_units
            normalized_rate = min(4.0, rate / nominal_rate) if nominal_rate else 0.0
            normalized_capacity = min(
                1.0,
                math.log1p(worker.compute_units_per_second) / math.log1p(200.0),
            )
            rate_change = normalized_rate - self._previous_rates.get(key, 0.0)
            send_seconds = worker.rtt_ms / 1000 + (
                self.config.input_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            return_seconds = worker.rtt_ms / 1000 + (
                self.config.result_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            if rate > 0:
                remaining_seconds = (1.0 - progress) / rate + return_seconds
            else:
                remaining_seconds = max(
                    0.0,
                    send_seconds
                    + self.config.work_units / worker.compute_units_per_second
                    + return_seconds
                    - task_age,
                )
                if (
                    active
                    and progress < 1.0
                    and heartbeat_age > 1.5 * worker.heartbeat_interval_seconds
                ):
                    remaining_seconds = max(
                        remaining_seconds, 2.0 * self.config.max_episode_seconds
                    )
            worker_values.extend(
                [
                    active,
                    normalized_capacity,
                    progress,
                    normalized_rate,
                    max(-4.0, min(4.0, rate_change)),
                    heartbeat_alive,
                    min(2.0, heartbeat_age / max(worker.heartbeat_interval_seconds, 1e-6)),
                    min(1.0, math.log1p(worker.rtt_ms) / math.log1p(1000.0)),
                    min(1.0, math.log1p(worker.bandwidth_mbps) / math.log1p(1000.0)),
                    min(2.0, task_age / self.config.max_episode_seconds),
                    min(2.0, remaining_seconds / self.config.max_episode_seconds),
                ]
            )
            current_rates.append(normalized_rate)
            self._previous_progress[key] = progress
            target_progresses.append(
                max(
                    (attempt.progress for attempt in attempts if attempt.worker_index == worker_index),
                    default=0.0,
                )
            )

        max_progress = max(target_progresses, default=0.0)
        last_global = self._last_global_progress.get(shard_id, 0.0)
        last_progress_time = self._last_progress_time.get(shard_id, 0.0)
        if max_progress > last_global:
            last_global = max_progress
            last_progress_time = now
        self._last_global_progress[shard_id] = last_global
        self._last_progress_time[shard_id] = last_progress_time
        time_without_progress = min(
            1.0,
            max(0.0, now - last_progress_time) / self.config.max_episode_seconds,
        )
        active_target_count = sum(attempt.active for attempt in attempts)
        global_values = [
            min(1.0, now / self.config.max_episode_seconds),
            min(1.0, len(attempts) / len(self.workers)),
            min(1.0, active_target_count / len(self.workers)),
            min(1.0, max(0, len(attempts) - 1) / len(self.workers)),
            time_without_progress,
        ]
        raw = torch.tensor(global_values + worker_values, dtype=torch.float32)
        history = self._histories.get(shard_id)
        if history is None:
            history = deque(maxlen=self.history_length)
            for _ in range(self.history_length):
                history.append(raw.clone())
            self._histories[shard_id] = history
        else:
            history.append(raw)
        for worker_index, rate in enumerate(current_rates):
            self._previous_rates[(shard_id, worker_index)] = rate
        self._previous_observation_time[shard_id] = now
        return torch.cat(tuple(history))


class LivePolicy:
    def __init__(
        self,
        scheduler: str,
        workers: Sequence[WorkerSpec],
        config: SimulationConfig,
        history_length: int | None = None,
        checkpoint: str | None = None,
        fixed_timeout_seconds: float = 5.0,
    ) -> None:
        if scheduler not in {"adaptive", "fixed", "ppo"}:
            raise ValueError(f"unsupported scheduler: {scheduler}")
        self.scheduler = scheduler
        self.workers = tuple(workers)
        self.config = config
        self.fixed_timeout_seconds = fixed_timeout_seconds
        self.policy: nn.Module | None = None
        if scheduler == "ppo":
            if checkpoint is None:
                raise ValueError("a PPO checkpoint is required for scheduler=ppo")
            self.policy, loaded_history, loaded_worker_count = _load_policy(
                Path(checkpoint)
            )
            if history_length is not None and loaded_history != history_length:
                raise ValueError("checkpoint and live scheduler history lengths differ")
            if loaded_worker_count != len(self.workers):
                raise ValueError("checkpoint and live scheduler worker counts differ")
            history_length = loaded_history
            expected_size = history_length * (
                5 + len(self.workers) * 11
            )
            if self.policy.observation_size != expected_size:
                raise ValueError("checkpoint observation size does not match live workers")
            expected_actions = len(config.wait_seconds) + len(self.workers)
            if self.policy.action_count != expected_actions:
                raise ValueError("checkpoint action count does not match live workers")
            self.policy.eval()
        if history_length is None:
            history_length = 4
        self.history_length = history_length
        self.history = LiveObservationHistory(workers, config, history_length)

    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        now: float,
        shard_attempts: Sequence[LiveAttempt],
    ) -> int:
        if self.scheduler == "ppo":
            assert self.policy is not None
            with torch.no_grad():
                logits, _ = self.policy(observation.unsqueeze(0), action_mask.unsqueeze(0))
            return int(logits.argmax(dim=-1).item())
        return self._heuristic_action(action_mask, now, shard_attempts)

    def _heuristic_action(
        self,
        action_mask: torch.Tensor,
        now: float,
        shard_attempts: Sequence[LiveAttempt],
    ) -> int:
        wait_count = len(self.config.wait_seconds)
        active = [attempt for attempt in shard_attempts if attempt.active]
        available = [
            index
            for index in range(len(self.workers))
            if bool(action_mask[wait_count + index])
        ]
        if not active and available:
            best = max(
                available,
                key=lambda index: self.workers[index].compute_units_per_second,
            )
            return wait_count + best
        if self.scheduler == "fixed":
            shard_started = min(
                (attempt.start_time for attempt in shard_attempts), default=now
            )
            shard_age = max(0.0, now - shard_started)
            if shard_age >= self.fixed_timeout_seconds and available:
                best = max(
                    available,
                    key=lambda index: self.workers[index].compute_units_per_second,
                )
                return wait_count + best
            desired = max(
                min(
                    self.fixed_timeout_seconds - shard_age,
                    max(self.config.wait_seconds),
                ),
                min(self.config.wait_seconds),
            )
            return min(
                range(wait_count),
                key=lambda index: abs(self.config.wait_seconds[index] - desired),
            )

        incumbent_remaining = min(
            self.config.max_episode_seconds
            * max(
                0.0,
                float(
                    self.history._histories[attempt.shard_id][-1][
                        self.history.global_feature_size
                        + attempt.worker_index * self.history.worker_feature_size
                        + 10
                    ].item()
                ),
            )
            for attempt in active
        ) if active else self.config.max_episode_seconds
        candidates: list[tuple[float, int, float, float]] = []
        for index in available:
            worker = self.workers[index]
            network = 2 * worker.rtt_ms / 1000 + (
                (self.config.input_bytes + self.config.result_bytes) * 8
                / (worker.bandwidth_mbps * 1_000_000)
            )
            compute = self.config.work_units / worker.compute_units_per_second
            candidates.append((network + compute, index, compute, network))
        if candidates:
            finish, worker_index, compute, network = min(candidates)
            saving = incumbent_remaining - finish
            cost = (
                self.config.network_cost_weight * network
                + self.config.duplicate_compute_cost_weight * compute
            )
            if saving > max(0.5, cost):
                return wait_count + worker_index
        return min(
            range(wait_count),
            key=lambda index: self.config.wait_seconds[index],
        )
