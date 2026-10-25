"""Reference policies for the simulated wait/duplicate decision problem."""

from __future__ import annotations

from typing import Literal

import torch

from .simulator import SchedulingEnv

PolicyName = Literal["wait_only", "fixed_timeout", "adaptive_heuristic"]


def _nearest_wait_action(env: SchedulingEnv, desired_seconds: float) -> int:
    waits = env.config.wait_seconds
    return min(range(len(waits)), key=lambda index: abs(waits[index] - desired_seconds))


def _fastest_available_duplicate(env: SchedulingEnv, mask: torch.Tensor) -> int | None:
    candidates = [
        index
        for index in range(len(env.workers))
        if bool(mask[env.wait_action_count + index])
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda index: -env.workers[index].compute_units_per_second)
    return env.wait_action_count + best


def _adaptive_action(env: SchedulingEnv, observation: torch.Tensor, mask: torch.Tensor) -> int:
    frame_size = env.global_feature_size + len(env.workers) * env.worker_feature_size
    frame = observation[-frame_size:]
    active: list[tuple[int, float]] = []
    for index, worker in enumerate(env.workers):
        offset = env.global_feature_size + index * env.worker_feature_size
        is_active = bool(frame[offset].item())
        if is_active:
            estimated_remaining = max(0.0, float(frame[offset + 10].item()))
            active.append(
                (index, estimated_remaining * env.config.max_episode_seconds)
            )

    if not active:
        duplicate = _fastest_available_duplicate(env, mask)
        if duplicate is not None:
            return duplicate

    available = [
        index
        for index in range(len(env.workers))
        if bool(mask[env.wait_action_count + index])
    ]
    if active and available:
        incumbent_remaining = min(remaining for _, remaining in active)
        candidate_finish_seconds: list[tuple[float, int, float, float]] = []
        for index in available:
            worker = env.workers[index]
            send = worker.rtt_ms / 1000 + (
                env.config.input_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            receive = worker.rtt_ms / 1000 + (
                env.config.result_bytes * 8 / (worker.bandwidth_mbps * 1_000_000)
            )
            compute = env.config.work_units / worker.compute_units_per_second
            candidate_finish_seconds.append(
                (send + compute + receive, index, compute, send + receive)
            )

        finish, candidate, compute, network = min(candidate_finish_seconds)
        expected_saving = incumbent_remaining - finish
        duplicate_cost = (
            env.config.network_cost_weight * network
            + env.config.duplicate_compute_cost_weight * compute
        )
        if expected_saving > max(0.5, duplicate_cost):
            return env.wait_action_count + candidate

    return _nearest_wait_action(env, min(env.config.wait_seconds))


def select_action(
    policy: PolicyName,
    env: SchedulingEnv,
    observation: torch.Tensor,
    mask: torch.Tensor,
    fixed_timeout_seconds: float = 5.0,
) -> int:
    """Choose a valid discrete action using one of the reference policies."""
    if policy == "adaptive_heuristic":
        return _adaptive_action(env, observation, mask)

    active = any(
        bool(observation[
            -(env.global_feature_size + len(env.workers) * env.worker_feature_size)
            + env.global_feature_size
            + index * env.worker_feature_size
        ].item())
        for index in range(len(env.workers))
    )
    if not active:
        duplicate = _fastest_available_duplicate(env, mask)
        if duplicate is not None:
            return duplicate

    if policy == "fixed_timeout" and env.now >= fixed_timeout_seconds:
        duplicate = _fastest_available_duplicate(env, mask)
        if duplicate is not None:
            return duplicate
    if policy not in {"wait_only", "fixed_timeout"}:
        raise ValueError(f"unknown reference policy: {policy}")
    if policy == "wait_only":
        wait_index = max(range(env.wait_action_count), key=lambda i: env.config.wait_seconds[i])
    else:
        remaining = max(0.0, fixed_timeout_seconds - env.now)
        desired_wait = min(
            max(env.config.wait_seconds),
            max(remaining, min(env.config.wait_seconds)),
        )
        wait_index = _nearest_wait_action(env, desired_wait)
    if bool(mask[wait_index]):
        return wait_index
    duplicate = _fastest_available_duplicate(env, mask)
    if duplicate is not None:
        return duplicate
    return _nearest_wait_action(env, min(env.config.wait_seconds))
