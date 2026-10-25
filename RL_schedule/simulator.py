"""Discrete-event simulator for a single delayed shard and its duplicate runs."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    compute_units_per_second: float
    rtt_ms: float
    bandwidth_mbps: float
    heartbeat_interval_seconds: float = 1.0
    failure_probability: float = 0.0
    stall_probability: float = 0.0
    result_loss_probability: float = 0.0
    invalid_result_probability: float = 0.0
    degradation_probability: float = 0.0
    degradation_factor: float = 0.35

    def validate(self) -> None:
        if not math.isfinite(self.compute_units_per_second) or self.compute_units_per_second <= 0:
            raise ValueError("worker compute rate must be positive")
        if not math.isfinite(self.rtt_ms) or self.rtt_ms < 0:
            raise ValueError("worker RTT must be non-negative")
        if not math.isfinite(self.bandwidth_mbps) or self.bandwidth_mbps <= 0:
            raise ValueError("worker bandwidth must be positive")
        if (
            not math.isfinite(self.heartbeat_interval_seconds)
            or self.heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat interval must be positive")
        probabilities = (
            self.failure_probability,
            self.stall_probability,
            self.result_loss_probability,
            self.invalid_result_probability,
            self.degradation_probability,
        )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("worker failure probabilities must be between 0 and 1")
        if self.failure_probability + self.stall_probability > 1:
            raise ValueError("failure and stall probabilities cannot sum above 1")
        if not math.isfinite(self.degradation_factor) or not 0 < self.degradation_factor <= 1:
            raise ValueError("degradation_factor must be in the interval (0, 1]")


@dataclass(frozen=True)
class SimulationConfig:
    work_units: float = 1000.0
    input_bytes: int = 196_608
    result_bytes: int = 100_000
    wait_seconds: tuple[float, ...] = (0.5, 2.0, 5.0)
    max_episode_seconds: float = 120.0
    network_cost_weight: float = 0.02
    duplicate_compute_cost_weight: float = 0.1
    failure_penalty: float = 120.0
    reward_mode: str = "cost_sensitive"

    def validate(self) -> None:
        if self.reward_mode not in {"cost_sensitive", "time_success"}:
            raise ValueError("reward_mode must be cost_sensitive or time_success")
        if not math.isfinite(self.work_units) or self.work_units <= 0:
            raise ValueError("work_units must be positive")
        if self.input_bytes < 0 or self.result_bytes < 0:
            raise ValueError("payload sizes must be non-negative")
        if not self.wait_seconds or any(
            not math.isfinite(value) or value <= 0 for value in self.wait_seconds
        ):
            raise ValueError("wait_seconds must contain positive values")
        if not math.isfinite(self.max_episode_seconds) or self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.network_cost_weight, self.duplicate_compute_cost_weight)
        ):
            raise ValueError("cost weights must be non-negative")
        if not math.isfinite(self.failure_penalty) or self.failure_penalty < 0:
            raise ValueError("failure_penalty must be non-negative")


@dataclass
class _CopyRun:
    worker_index: int
    start_time: float
    compute_start: float
    compute_duration: float
    result_time: float
    return_seconds: float
    duplicate: bool
    failure_time: float | None
    stall_time: float | None
    slowdown_time: float | None
    slowdown_factor: float
    result_lost: bool
    result_valid: bool

    def compute_end(self) -> float:
        if self.slowdown_time is None:
            return self.compute_start + self.compute_duration
        before_fraction = (self.slowdown_time - self.compute_start) / self.compute_duration
        remaining_duration = (1.0 - before_fraction) * self.compute_duration
        return self.slowdown_time + remaining_duration / self.slowdown_factor

    def progress(self, now: float) -> float:
        if now <= self.compute_start:
            return 0.0
        stop = self.compute_end()
        if self.failure_time is not None:
            stop = min(stop, self.failure_time)
        if self.stall_time is not None:
            stop = min(stop, self.stall_time)
        effective_end = min(now, stop)
        if self.slowdown_time is not None and effective_end > self.slowdown_time:
            before_fraction = (
                self.slowdown_time - self.compute_start
            ) / self.compute_duration
            after_fraction = (
                effective_end - self.slowdown_time
            ) * self.slowdown_factor / self.compute_duration
            return min(1.0, max(0.0, before_fraction + after_fraction))
        elapsed = effective_end - self.compute_start
        return min(1.0, max(0.0, elapsed / self.compute_duration))

    def status(self, now: float) -> str:
        if self.failure_time is not None and now >= self.failure_time:
            return "failed"
        if self.stall_time is not None and now >= self.stall_time:
            return "stalled"
        if now >= self.result_time:
            if self.result_lost:
                return "lost"
            return "completed" if self.result_valid else "invalid"
        return "running"

    def consumed_compute_seconds(self, start: float, end: float) -> float:
        compute_end = self.compute_end()
        if self.failure_time is not None:
            compute_end = min(compute_end, self.failure_time)
        if self.stall_time is not None:
            compute_end = min(compute_end, self.stall_time)
        return max(0.0, min(end, compute_end) - max(start, self.compute_start))


def default_workers() -> tuple[WorkerSpec, ...]:
    """Return the example fast, medium, and slow workers from the design doc."""
    return (
        WorkerSpec("A", 100.0, 5.0, 500.0),
        WorkerSpec("B", 60.0, 20.0, 50.0),
        WorkerSpec(
            "C",
            20.0,
            100.0,
            10.0,
            failure_probability=0.03,
            stall_probability=0.07,
            result_loss_probability=0.03,
            degradation_probability=0.25,
        ),
    )


class SchedulingEnv:
    """Wait for or duplicate one shard while simulating worker and link delays.

    Action ids ``0..len(wait_seconds)-1`` wait for one of the configured
    intervals. The remaining actions duplicate the shard on a worker, in the
    same order as ``workers``. An action mask prevents dispatching to a worker
    that already has a live copy.
    """

    def __init__(
        self,
        workers: Sequence[WorkerSpec] | None = None,
        config: SimulationConfig | None = None,
        history_length: int = 4,
    ) -> None:
        self.workers = tuple(default_workers() if workers is None else workers)
        self.config = config or SimulationConfig()
        self.config.validate()
        for worker in self.workers:
            worker.validate()
        if not self.workers:
            raise ValueError("at least one worker is required")
        if history_length < 1:
            raise ValueError("history_length must be at least 1")

        self.history_length = history_length
        self.worker_feature_size = 11
        self.global_feature_size = 5
        self.observation_size = (
            history_length
            * (self.global_feature_size + len(self.workers) * self.worker_feature_size)
        )
        self.action_count = len(self.config.wait_seconds) + len(self.workers)
        self._history: deque[torch.Tensor] = deque(maxlen=history_length)
        self._rng = random.Random()
        self.now = 0.0
        self.done = False
        self.success = False
        self.terminal_reason: str | None = None
        self.copies: list[_CopyRun] = []
        self._previous_progress: list[float] = []
        self._previous_rates: list[float] = []
        self._last_observation_time = 0.0
        self._last_progress_time = 0.0
        self._last_global_progress = 0.0
        self.network_work_seconds = 0.0
        self.network_bytes_transferred = 0
        self.total_compute_seconds = 0.0
        self.duplicate_compute_seconds = 0.0
        self.duplicates_started = 0
        self.failed_copies = 0
        self.lost_results = 0
        self.invalid_results = 0
        self.late_results_ignored = 0
        self.recovery_start_time: float | None = None
        self.recovery_time_seconds = 0.0
        self.completion_worker: str | None = None

    @property
    def wait_action_count(self) -> int:
        return len(self.config.wait_seconds)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self._rng = random.Random(seed)
        self.now = 0.0
        self.done = False
        self.success = False
        self.terminal_reason = None
        self.copies = []
        self.network_work_seconds = 0.0
        self.network_bytes_transferred = 0
        self.total_compute_seconds = 0.0
        self.duplicate_compute_seconds = 0.0
        self.duplicates_started = 0
        self.failed_copies = 0
        self.lost_results = 0
        self.invalid_results = 0
        self.late_results_ignored = 0
        self.recovery_start_time = None
        self.recovery_time_seconds = 0.0
        self.completion_worker = None
        self._last_observation_time = 0.0
        self._last_progress_time = 0.0
        self._last_global_progress = 0.0
        self._previous_progress = [0.0] * len(self.workers)
        self._previous_rates = [0.0] * len(self.workers)

        initial_worker = self._rng.randrange(len(self.workers))
        self._dispatch(initial_worker, duplicate=False)
        initial = self._raw_observation()
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(initial.clone())
        return self.observation(), self.action_mask()

    def observation(self) -> torch.Tensor:
        if not self._history:
            raise RuntimeError("reset must be called before observation")
        return torch.cat(tuple(self._history))

    def action_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.action_count, dtype=torch.bool)
        if self.done:
            return mask
        has_active_copy = any(
            copy.status(self.now) in {"running", "stalled"} for copy in self.copies
        )
        if has_active_copy:
            mask[: self.wait_action_count] = True
        for worker_index in range(len(self.workers)):
            if not any(
                copy.worker_index == worker_index
                and copy.status(self.now) in {"running", "stalled"}
                for copy in self.copies
            ):
                mask[self.wait_action_count + worker_index] = True
        return mask

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, torch.Tensor, dict[str, float | int | str | None | bool]]:
        if self.done:
            raise RuntimeError("episode is already done")
        mask = self.action_mask()
        if action < 0 or action >= self.action_count or not bool(mask[action]):
            raise ValueError(f"action {action} is invalid in the current state")

        old_time = self.now
        old_network_cost = self.network_work_seconds
        old_duplicate_compute = self.duplicate_compute_seconds
        if action < self.wait_action_count:
            requested_wait = self.config.wait_seconds[action]
            self._advance(requested_wait)
        else:
            worker_index = action - self.wait_action_count
            self._dispatch(worker_index, duplicate=True)
            # A dispatch consumes time even if it later fails or loses its result.
            self._advance(0.0)

        delta_time = self.now - old_time
        delta_network = self.network_work_seconds - old_network_cost
        delta_duplicate_compute = self.duplicate_compute_seconds - old_duplicate_compute
        if self.config.reward_mode == "time_success":
            reward = -delta_time
            if self.done and self.success:
                reward += 1.0
        else:
            reward = -delta_time
            reward -= self.config.network_cost_weight * delta_network
            reward -= self.config.duplicate_compute_cost_weight * delta_duplicate_compute
            if self.done and not self.success:
                reward -= self.config.failure_penalty

        self._history.append(self._raw_observation())
        info: dict[str, float | int | str | None | bool] = {
            "delta_seconds": delta_time,
            "makespan_seconds": self.now if self.done and self.success else 0.0,
            "network_work_seconds": self.network_work_seconds,
            "network_bytes_transferred": self.network_bytes_transferred,
            "total_compute_seconds": self.total_compute_seconds,
            "duplicate_compute_seconds": self.duplicate_compute_seconds,
            "duplicates_started": self.duplicates_started,
            "failed_copies": self.failed_copies,
            "lost_results": self.lost_results,
            "success": self.success,
            "terminal_reason": self.terminal_reason,
            "completion_worker": self.completion_worker,
        }
        return self.observation(), reward, self.done, self.action_mask(), info

    def _dispatch(self, worker_index: int, duplicate: bool) -> None:
        worker = self.workers[worker_index]
        send_seconds = worker.rtt_ms / 1000 + (
            self.config.input_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
        )
        return_seconds = worker.rtt_ms / 1000 + (
            self.config.result_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
        )
        compute_duration = self.config.work_units / worker.compute_units_per_second
        compute_duration *= self._rng.lognormvariate(0.0, 0.08)
        compute_start = self.now + send_seconds
        failure_time = None
        stall_time = None
        slowdown_time = None
        mode = self._rng.random()
        if mode < worker.failure_probability:
            failure_time = compute_start + compute_duration * self._rng.uniform(0.1, 0.9)
        elif mode < worker.failure_probability + worker.stall_probability:
            stall_time = compute_start + compute_duration * self._rng.uniform(0.2, 0.8)
        if self._rng.random() < worker.degradation_probability:
            slowdown_time = compute_start + compute_duration * self._rng.uniform(0.2, 0.7)
        result_lost = self._rng.random() < worker.result_loss_probability
        result_valid = self._rng.random() >= worker.invalid_result_probability
        run = _CopyRun(
            worker_index=worker_index,
            start_time=self.now,
            compute_start=compute_start,
            compute_duration=compute_duration,
            result_time=0.0,
            return_seconds=return_seconds,
            duplicate=duplicate,
            failure_time=failure_time,
            stall_time=stall_time,
            slowdown_time=slowdown_time,
            slowdown_factor=worker.degradation_factor,
            result_lost=result_lost,
            result_valid=result_valid,
        )
        run.result_time = run.compute_end() + return_seconds
        self.copies.append(run)
        self.network_work_seconds += send_seconds
        self.network_bytes_transferred += self.config.input_bytes
        self._previous_progress[worker_index] = 0.0
        self._previous_rates[worker_index] = 0.0
        if duplicate:
            self.duplicates_started += 1

    def _advance(self, requested_seconds: float) -> None:
        old_time = self.now
        target_time = min(
            old_time + requested_seconds, self.config.max_episode_seconds
        )
        successful_events = [
            copy.result_time
            for copy in self.copies
            if not copy.result_lost
            and copy.failure_time is None
            and copy.stall_time is None
            and old_time < copy.result_time <= target_time
        ]
        if successful_events:
            target_time = min(successful_events)

        for copy in self.copies:
            consumed = copy.consumed_compute_seconds(old_time, target_time)
            self.total_compute_seconds += consumed
            if copy.duplicate:
                self.duplicate_compute_seconds += consumed

        for copy in self.copies:
            if copy.failure_time is not None and old_time < copy.failure_time <= target_time:
                self.failed_copies += 1
                if self.recovery_start_time is None:
                    self.recovery_start_time = copy.failure_time
            if copy.stall_time is not None and old_time < copy.stall_time <= target_time:
                if self.recovery_start_time is None:
                    self.recovery_start_time = copy.stall_time
            if (
                copy.result_lost
                and copy.failure_time is None
                and copy.stall_time is None
                and old_time < copy.result_time <= target_time
            ):
                self.lost_results += 1
                if self.recovery_start_time is None:
                    self.recovery_start_time = copy.result_time
            if (
                not copy.result_lost
                and not copy.result_valid
                and copy.failure_time is None
                and copy.stall_time is None
                and old_time < copy.result_time <= target_time
            ):
                self.invalid_results += 1
                if self.recovery_start_time is None:
                    self.recovery_start_time = copy.result_time
            if (
                old_time < copy.result_time <= target_time
                and copy.failure_time is None
                and copy.stall_time is None
            ):
                self.network_work_seconds += copy.return_seconds
                self.network_bytes_transferred += self.config.result_bytes

        self.now = target_time
        completed = [
            copy
            for copy in self.copies
            if not copy.result_lost
            and copy.result_valid
            and copy.failure_time is None
            and copy.stall_time is None
            and math.isclose(copy.result_time, self.now, rel_tol=0.0, abs_tol=1e-9)
        ]
        if completed:
            self.done = True
            self.success = True
            self.terminal_reason = "result_received"
            self.completion_worker = self.workers[completed[0].worker_index].name
            if self.recovery_start_time is not None:
                self.recovery_time_seconds = max(
                    0.0, self.now - self.recovery_start_time
                )
            self.late_results_ignored = sum(
                1
                for copy in self.copies
                if copy is not completed[0]
                and not copy.result_lost
                and copy.failure_time is None
                and copy.stall_time is None
                and copy.result_time > self.now
            )
        elif self.now >= self.config.max_episode_seconds:
            self.done = True
            self.terminal_reason = "time_limit"

    def _raw_observation(self) -> torch.Tensor:
        elapsed_since_observation = self.now - self._last_observation_time
        worker_values: list[float] = []
        current_rates: list[float] = []
        progress_values: list[float] = []
        for index, worker in enumerate(self.workers):
            run = next(
                (
                    copy
                    for copy in reversed(self.copies)
                    if copy.worker_index == index
                ),
                None,
            )
            if run is None:
                progress = 0.0
                active = 0.0
                rate = 0.0
                task_age = 0.0
                heartbeat_alive = 1.0
                heartbeat_age = 0.0
            else:
                progress = run.progress(self.now)
                status = run.status(self.now)
                active = 1.0 if status in {"running", "stalled"} else 0.0
                task_age = max(0.0, self.now - run.start_time)
                delta_progress = max(0.0, progress - self._previous_progress[index])
                rate = (
                    delta_progress / elapsed_since_observation
                    if elapsed_since_observation > 0
                    else 0.0
                )
                failure_stop = run.failure_time
                heartbeat_elapsed = max(
                    0.0,
                    min(self.now, failure_stop or self.now) - run.start_time,
                )
                interval = worker.heartbeat_interval_seconds
                latest_heartbeat = run.start_time + math.floor(
                    heartbeat_elapsed / interval
                ) * interval
                heartbeat_age = max(0.0, self.now - latest_heartbeat) / interval
                heartbeat_alive = 1.0 if heartbeat_age <= 1.5 else 0.0

            nominal_rate = worker.compute_units_per_second / self.config.work_units
            normalized_rate = min(4.0, rate / nominal_rate) if nominal_rate else 0.0
            normalized_capacity = min(
                1.0,
                math.log1p(worker.compute_units_per_second) / math.log1p(200.0),
            )
            rate_change = normalized_rate - self._previous_rates[index]
            send_seconds = worker.rtt_ms / 1000 + (
                self.config.input_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            return_seconds = worker.rtt_ms / 1000 + (
                self.config.result_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            if rate > 0:
                remaining_seconds = (1.0 - progress) / rate + return_seconds
            else:
                nominal_total = (
                    send_seconds
                    + self.config.work_units / worker.compute_units_per_second
                    + return_seconds
                )
                remaining_seconds = max(0.0, nominal_total - task_age)
                if active and progress < 1.0 and task_age > send_seconds + 0.1:
                    # A live heartbeat with no progress is a likely stalled job.
                    remaining_seconds = max(
                        remaining_seconds, 2.0 * self.config.max_episode_seconds
                    )
            expected_remaining = remaining_seconds / self.config.max_episode_seconds
            rtt_feature = math.log1p(worker.rtt_ms) / math.log1p(1000.0)
            bandwidth_feature = math.log1p(worker.bandwidth_mbps) / math.log1p(1000.0)
            worker_values.extend(
                [
                    active,
                    normalized_capacity,
                    progress,
                    normalized_rate,
                    max(-4.0, min(4.0, rate_change)),
                    heartbeat_alive,
                    min(2.0, heartbeat_age),
                    min(1.0, rtt_feature),
                    min(1.0, bandwidth_feature),
                    min(2.0, task_age / self.config.max_episode_seconds),
                    min(2.0, max(0.0, expected_remaining)),
                ]
            )
            current_rates.append(normalized_rate)
            progress_values.append(progress)

        max_progress = max(progress_values, default=0.0)
        if max_progress > self._last_global_progress:
            self._last_progress_time = self.now
            self._last_global_progress = max_progress
        time_without_progress = min(
            1.0,
            max(0.0, self.now - self._last_progress_time)
            / self.config.max_episode_seconds,
        )
        active_count = sum(
            copy.status(self.now) in {"running", "stalled"} for copy in self.copies
        )
        global_values = [
            min(1.0, self.now / self.config.max_episode_seconds),
            min(1.0, len(self.copies) / len(self.workers)),
            min(1.0, active_count / len(self.workers)),
            min(1.0, self.duplicates_started / len(self.workers)),
            time_without_progress,
        ]
        raw = torch.tensor(global_values + worker_values, dtype=torch.float32)
        self._previous_progress = progress_values
        self._previous_rates = current_rates
        self._last_observation_time = self.now
        return raw

    def metrics(self) -> dict[str, float | int | str | None | bool]:
        return {
            "success": self.success,
            "terminal_reason": self.terminal_reason,
            "makespan_seconds": self.now if self.success else 0.0,
            "network_work_seconds": self.network_work_seconds,
            "network_bytes_transferred": self.network_bytes_transferred,
            "total_compute_seconds": self.total_compute_seconds,
            "duplicate_compute_seconds": self.duplicate_compute_seconds,
            "duplicates_started": self.duplicates_started,
            "failed_copies": self.failed_copies,
            "lost_results": self.lost_results,
            "invalid_results": self.invalid_results,
            "late_results_ignored": self.late_results_ignored,
            "recovery_time_seconds": self.recovery_time_seconds,
            "completion_worker": self.completion_worker,
        }
