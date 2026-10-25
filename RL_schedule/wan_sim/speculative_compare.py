"""Compare fixed and adaptive dispatchers with a candidate-action RLOO policy."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .examples import example_workload, four_region_topology
from .speculative import DecisionAction, DecisionKind, SpeculativeWANScheduler
from .types import FaultEvent, FaultKind, SimulationConfig, Topology, Workload

_CANDIDATE_FEATURES = 10
_INITIAL_NODE_COUNT = 6


class CandidateRLOOActor(nn.Module):
    """Scores every feasible action candidate instead of choosing a template."""

    def __init__(
        self,
        observation_size: int,
        candidate_count: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.candidate_count = candidate_count
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(observation_size, hidden_size, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size + _CANDIDATE_FEATURES, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.lstm.weight_ih_l0
        shape = (1, 1, self.hidden_size)
        return weight.new_zeros(shape), weight.new_zeros(shape)

    def forward_step(
        self,
        observation: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        encoded, next_state = self.lstm(observation.reshape(1, 1, -1), hidden_state)
        hidden = encoded[0, 0].expand(self.candidate_count, -1)
        logits = self.scorer(torch.cat((hidden, candidate_features), dim=-1)).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e9), next_state

    def forward_sequence(
        self,
        frames: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_masks: torch.Tensor,
    ) -> torch.Tensor:
        encoded, _ = self.lstm(frames.unsqueeze(0))
        hidden = encoded.squeeze(0)
        repeated = hidden[:, None, :].expand(-1, self.candidate_count, -1)
        logits = self.scorer(
            torch.cat((repeated, candidate_features), dim=-1)
        ).squeeze(-1)
        return logits.masked_fill(~candidate_masks, -1e9)


def _fault_scenario(
    seed: int, topology: Topology, batches_per_episode: int = 1
) -> tuple[FaultEvent, ...]:
    rng = random.Random(seed)
    nodes = [node.node_id for node in topology.nodes]
    initially_active = nodes[:_INITIAL_NODE_COUNT]
    events = [
        FaultEvent(
            rng.uniform(0.2, 1.5), FaultKind.NODE_JOIN, nodes[6]
        ),
        FaultEvent(
            rng.uniform(1.0, 3.0), FaultKind.NODE_JOIN, nodes[7]
        ),
    ]
    if rng.random() < 0.7:
        events.append(
            FaultEvent(
                rng.uniform(0.3, 2.0),
                FaultKind.NODE_SPEED_DROP,
                rng.choice(initially_active),
                duration_seconds=rng.uniform(2.0, 5.0),
            )
        )
    if rng.random() < 0.5:
        events.append(
            FaultEvent(
                rng.uniform(1.5, 4.0),
                FaultKind.NODE_LEAVE,
                rng.choice(initially_active),
                duration_seconds=rng.uniform(2.0, 5.0),
            )
        )
    wan_links = [link for link in topology.links if link.region_a != link.region_b]
    if wan_links and rng.random() < 0.5:
        link = rng.choice(wan_links)
        events.append(
            FaultEvent(
                rng.uniform(1.0, 4.0),
                FaultKind.LINK_DOWN,
                link.link_id,
                duration_seconds=rng.uniform(1.0, 3.0),
            )
        )
    extra_churn_waves = max(0, (batches_per_episode - 1) // 2)
    for wave in range(extra_churn_waves):
        events.append(
            FaultEvent(
                4.5 + wave * 4.5,
                FaultKind.NODE_LEAVE,
                rng.choice(nodes),
                duration_seconds=rng.uniform(1.5, 3.0),
            )
        )
    return tuple(sorted(events, key=lambda event: event.time_seconds))


def _new_env(
    *,
    seed: int,
    topology: Topology,
    workload: Workload,
    shard_count: int,
    batches_per_episode: int = 1,
    max_decisions: int = 256,
) -> SpeculativeWANScheduler:
    return SpeculativeWANScheduler(
        topology,
        workload,
        shard_count=shard_count,
        config=SimulationConfig(),
        fault_schedule=_fault_scenario(seed, topology, batches_per_episode),
        initial_active_nodes=tuple(
            node.node_id for node in topology.nodes[:_INITIAL_NODE_COUNT]
        ),
        micro_batch_options=(32, 48, 64),
        batches_per_episode=batches_per_episode,
        max_decisions=max_decisions,
    )


def _candidate_padding(
    env: SpeculativeWANScheduler,
    candidates: Sequence[DecisionAction],
    maximum_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.zeros((maximum_candidates, _CANDIDATE_FEATURES), dtype=np.float32)
    mask = np.zeros(maximum_candidates, dtype=np.bool_)
    if candidates:
        encoded = env.candidate_features(candidates)
        features[: len(candidates)] = encoded
        mask[: len(candidates)] = True
    return torch.from_numpy(features), torch.from_numpy(mask)


def _collect_episode(
    policy: CandidateRLOOActor,
    *,
    seed: int,
    topology: Topology,
    workload: Workload,
    shard_count: int,
    max_decisions: int,
    maximum_candidates: int,
    failure_penalty_seconds: float,
) -> dict[str, Any]:
    env = _new_env(
        seed=seed,
        topology=topology,
        workload=workload,
        shard_count=shard_count,
        max_decisions=max_decisions,
    )
    observation, candidates = env.reset(
        seed=seed,
        fault_schedule=_fault_scenario(seed, topology),
        initial_active_nodes=tuple(
            node.node_id for node in topology.nodes[:_INITIAL_NODE_COUNT]
        ),
    )
    hidden = policy.initial_state()
    frames: list[torch.Tensor] = []
    candidate_features: list[torch.Tensor] = []
    candidate_masks: list[torch.Tensor] = []
    actions: list[int] = []

    while not env.done:
        padded_features, mask = _candidate_padding(
            env, candidates, maximum_candidates
        )
        with torch.no_grad():
            logits, hidden = policy.forward_step(
                torch.from_numpy(observation), padded_features, mask, hidden
            )
            action_index = int(Categorical(logits=logits).sample().item())
        frames.append(torch.from_numpy(observation))
        candidate_features.append(padded_features)
        candidate_masks.append(mask)
        actions.append(action_index)
        observation, _, _, info = env.step(action_index, candidates)
        candidates = info["candidate_actions"]

    metrics = env.metrics()
    success = bool(metrics["success"])
    metrics["episode_return"] = -float(metrics["elapsed_seconds"]) - (
        0.0 if success else failure_penalty_seconds
    )
    metrics["decision_count"] = len(actions)
    return {
        "frames": frames,
        "candidate_features": candidate_features,
        "candidate_masks": candidate_masks,
        "actions": actions,
        "metrics": metrics,
        "state": env.state(),
    }


def _rloo_update(
    policy: CandidateRLOOActor,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[dict[str, Any]],
    *,
    entropy_weight: float,
    reward_scale: float,
) -> dict[str, float]:
    returns = torch.tensor(
        [trajectory["metrics"]["episode_return"] for trajectory in trajectories],
        dtype=torch.float32,
    ) / reward_scale
    total = returns.sum()
    advantages = torch.stack(
        [
            returns[index] - (total - returns[index]) / (len(returns) - 1)
            for index in range(len(returns))
        ]
    ).detach()
    policy_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for trajectory, advantage in zip(trajectories, advantages):
        frames = torch.stack(trajectory["frames"])
        candidate_features = torch.stack(trajectory["candidate_features"])
        candidate_masks = torch.stack(trajectory["candidate_masks"])
        actions = torch.tensor(trajectory["actions"], dtype=torch.long)
        logits = policy.forward_sequence(frames, candidate_features, candidate_masks)
        distribution = Categorical(logits=logits)
        policy_terms.append(-advantage * distribution.log_prob(actions).sum())
        entropy_terms.append(distribution.entropy().mean())
    policy_loss = torch.stack(policy_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    loss = policy_loss - entropy_weight * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach()),
        "entropy": float(entropy.detach()),
        "mean_return_seconds": float(returns.mean() * reward_scale),
        "success_rate": statistics.fmean(
            float(trajectory["metrics"]["success"]) for trajectory in trajectories
        ),
    }


def _wait_action_index(candidates: Sequence[DecisionAction], seconds: float) -> int:
    return min(
        (i for i, action in enumerate(candidates) if action.kind == DecisionKind.WAIT),
        key=lambda i: abs(float(candidates[i].wait_seconds or 0.0) - seconds),
    )


def _baseline_action(
    name: str,
    env: SpeculativeWANScheduler,
    candidates: Sequence[DecisionAction],
    *,
    fixed_duplicate_after_seconds: float = 1.0,
) -> int:
    start_candidates = [
        (index, action)
        for index, action in enumerate(candidates)
        if action.kind == DecisionKind.START
    ]
    if start_candidates:
        if name == "fixed":
            target = min(
                start_candidates,
                key=lambda pair: (
                    env.shards[pair[1].shard_uuid].shard_index,
                    env.node_index[pair[1].node_id],
                    pair[1].micro_batch_size,
                ),
            )
            wanted_size = 32
        else:
            target = min(
                start_candidates,
                key=lambda pair: (
                    env._effective_speed(pair[1].node_id) * -1,
                    env.shards[pair[1].shard_uuid].shard_index,
                ),
            )
            wanted_size = 64
        same = [
            pair
            for pair in start_candidates
            if pair[1].shard_uuid == target[1].shard_uuid
            and pair[1].node_id == target[1].node_id
            and pair[1].micro_batch_size == wanted_size
        ]
        return same[0][0] if same else target[0]

    duplicate_candidates = [
        (index, action)
        for index, action in enumerate(candidates)
        if action.kind == DecisionKind.DUPLICATE
    ]
    running = [
        attempt
        for attempt in env.attempts.values()
        if attempt.state.value == "running"
    ]
    if duplicate_candidates and running:
        if name == "fixed":
            eligible = [
                attempt
                for attempt in running
                if env.clock.now - attempt.started_at >= fixed_duplicate_after_seconds
                and len(env.shards[attempt.shard_uuid].attempt_uuids or ()) == 1
            ]
            if eligible:
                shard = min(eligible, key=lambda attempt: attempt.started_at)
                choices = [
                    pair
                    for pair in duplicate_candidates
                    if pair[1].shard_uuid == shard.shard_uuid
                ]
                if choices:
                    return min(choices, key=lambda pair: env.node_index[pair[1].node_id])[0]
        else:
            targets = {action.node_id for _, action in duplicate_candidates}
            single_copy_shards = {
                shard_uuid
                for shard_uuid, shard in env.shards.items()
                if len(shard.attempt_uuids or ()) == 1
            }
            if targets:
                target_id = max(
                    targets,
                    key=lambda node_id: env._effective_speed(node_id),
                )
                best_source = max(
                    [attempt for attempt in running if attempt.shard_uuid in single_copy_shards],
                    key=lambda attempt: (
                        attempt.remaining_flops
                        / max(1.0, env._effective_speed(attempt.node_id))
                    ),
                    default=None,
                )
                if best_source is None:
                    return _wait_action_index(candidates, 10.0)
                target_speed = env._effective_speed(target_id)
                source_speed = env._effective_speed(best_source.node_id)
                source_eta = best_source.remaining_flops / max(1.0, source_speed)
                target_eta = best_source.remaining_flops / max(1.0, target_speed)
                choices = [
                    pair
                    for pair in duplicate_candidates
                    if pair[1].shard_uuid == best_source.shard_uuid
                    and pair[1].node_id == target_id
                ]
                if choices and source_eta - target_eta > 0.1:
                    return max(choices, key=lambda pair: pair[1].micro_batch_size or 0)[0]
    return _wait_action_index(
        candidates, 0.5 if name == "fixed" else 10.0
    )


def _run_baseline_episode(
    name: str,
    *,
    seed: int,
    topology: Topology,
    workload: Workload,
    shard_count: int,
    batches_per_episode: int,
    max_decisions: int,
    fixed_duplicate_after_seconds: float,
    failure_penalty_seconds: float,
) -> dict[str, Any]:
    env = _new_env(
        seed=seed,
        topology=topology,
        workload=workload,
        shard_count=shard_count,
        batches_per_episode=batches_per_episode,
        max_decisions=max_decisions,
    )
    _, candidates = env.reset(
        seed=seed,
        fault_schedule=_fault_scenario(seed, topology, batches_per_episode),
        initial_active_nodes=tuple(node.node_id for node in topology.nodes[:_INITIAL_NODE_COUNT]),
    )
    while not env.done:
        action_index = _baseline_action(
            name,
            env,
            candidates,
            fixed_duplicate_after_seconds=fixed_duplicate_after_seconds,
        )
        _, _, _, info = env.step(action_index, candidates)
        candidates = info["candidate_actions"]
    metrics = env.metrics()
    success = bool(metrics["success"])
    metrics.update(
        {
            "policy": name,
            "seed": seed,
            "batches_per_episode": batches_per_episode,
            "penalized_time_seconds": float(metrics["elapsed_seconds"])
            + (0.0 if success else failure_penalty_seconds),
            "decision_count": env._decision_count,
            "fault_count": len(_fault_scenario(seed, topology, batches_per_episode)),
        }
    )
    return metrics


def _evaluate_rloo_episode(
    policy: CandidateRLOOActor,
    *,
    seed: int,
    topology: Topology,
    workload: Workload,
    shard_count: int,
    max_decisions: int,
    maximum_candidates: int,
    failure_penalty_seconds: float,
) -> dict[str, Any]:
    env = _new_env(
        seed=seed,
        topology=topology,
        workload=workload,
        shard_count=shard_count,
        max_decisions=max_decisions,
    )
    observation, candidates = env.reset(
        seed=seed,
        fault_schedule=_fault_scenario(seed, topology),
        initial_active_nodes=tuple(node.node_id for node in topology.nodes[:_INITIAL_NODE_COUNT]),
    )
    hidden = policy.initial_state()
    action_generator = torch.Generator(device="cpu").manual_seed(seed)
    while not env.done:
        features, mask = _candidate_padding(env, candidates, maximum_candidates)
        with torch.no_grad():
            logits, hidden = policy.forward_step(
                torch.from_numpy(observation), features, mask, hidden
            )
            action_index = int(
                torch.multinomial(
                    torch.softmax(logits, dim=-1),
                    num_samples=1,
                    generator=action_generator,
                ).item()
            )
        observation, _, _, info = env.step(action_index, candidates)
        candidates = info["candidate_actions"]
    metrics = env.metrics()
    success = bool(metrics["success"])
    metrics.update(
        {
            "policy": "rloo",
            "seed": seed,
            "penalized_time_seconds": float(metrics["elapsed_seconds"])
            + (0.0 if success else failure_penalty_seconds),
            "decision_count": env._decision_count,
            "fault_count": len(_fault_scenario(seed, topology)),
            "policy_action_mode": "seeded_stochastic",
        }
    )
    return metrics


def _aggregate(
    runs: Sequence[dict[str, Any]], *, completion_unit: str = "batch"
) -> dict[str, Any]:
    successful = [run for run in runs if run["success"]]
    times = [float(run["elapsed_seconds"]) for run in successful]
    aggregate = {
        "episodes": len(runs),
        "success_rate": statistics.fmean(float(run["success"]) for run in runs),
        "mean_penalized_time_seconds": statistics.fmean(
            float(run["penalized_time_seconds"]) for run in runs
        ),
        "mean_goodput_samples_per_second": statistics.fmean(
            float(run["goodput_samples_per_second"]) for run in runs
        ),
        "mean_duplicates_started": statistics.fmean(
            float(run["duplicates_started"]) for run in runs
        ),
        "mean_duplicate_wins": statistics.fmean(
            float(run["duplicate_wins"]) for run in runs
        ),
        "mean_late_results_ignored": statistics.fmean(
            float(run["late_results_ignored"]) for run in runs
        ),
        "mean_failed_attempts": statistics.fmean(
            float(run["failed_attempts"]) for run in runs
        ),
        "mean_network_seconds": statistics.fmean(
            float(run["network_seconds"]) for run in runs
        ),
        "mean_network_bytes": statistics.fmean(
            float(run["network_bytes"]) for run in runs
        ),
        "mean_barrier_wait_seconds": statistics.fmean(
            float(run["barrier_wait_seconds"]) for run in runs
        ),
        "mean_collective_seconds": statistics.fmean(
            float(run["collective_seconds"]) for run in runs
        ),
        "mean_collective_restarts": statistics.fmean(
            float(run["collective_restarts"]) for run in runs
        ),
        "mean_decision_count": statistics.fmean(
            float(run["decision_count"]) for run in runs
        ),
    }
    aggregate[f"mean_{completion_unit}_completion_seconds"] = (
        statistics.fmean(times) if times else None
    )
    aggregate[f"p90_{completion_unit}_completion_seconds"] = (
        float(np.percentile(times, 90)) if times else None
    )
    return aggregate


def run_baseline_scale_sweep(
    *,
    episode_scales: Sequence[int] = (1, 4, 16),
    eval_episodes: int = 64,
    seed: int = 0,
    fixed_duplicate_after_seconds: float = 1.0,
    max_decisions_per_batch: int = 256,
) -> dict[str, Any]:
    scales = tuple(dict.fromkeys(int(scale) for scale in episode_scales))
    if not scales or any(scale < 1 for scale in scales):
        raise ValueError("episode scales must be positive integers")
    if eval_episodes < 1 or max_decisions_per_batch < 1:
        raise ValueError("evaluation count and decision cap must be positive")

    topology = four_region_topology()
    workload = example_workload()
    evaluation_seeds = [seed + 1_000_003 + index * 13 for index in range(eval_episodes)]
    scale_results: dict[str, Any] = {}
    for scale in scales:
        failure_penalty = 100.0 * scale
        max_decisions = max_decisions_per_batch * scale
        runs: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ("fixed", "adaptive")
        }
        for evaluation_seed in evaluation_seeds:
            for name in runs:
                runs[name].append(
                    _run_baseline_episode(
                        name,
                        seed=evaluation_seed,
                        topology=topology,
                        workload=workload,
                        shard_count=8,
                        batches_per_episode=scale,
                        max_decisions=max_decisions,
                        fixed_duplicate_after_seconds=fixed_duplicate_after_seconds,
                        failure_penalty_seconds=failure_penalty,
                    )
                )
        scale_results[str(scale)] = {
            "episode_scale": scale,
            "samples_per_episode": workload.global_batch_size * scale,
            "max_decisions": max_decisions,
            "failure_penalty_seconds": failure_penalty,
            "policies": {
                name: _aggregate(values, completion_unit="episode")
                for name, values in runs.items()
            },
            "episode_results": runs,
        }

    return {
        "experiment": "speculative_wan_episode_scale_fixed_adaptive",
        "seed": seed,
        "rloo_training_performed": False,
        "evaluation_episodes_per_scale": eval_episodes,
        "batches_per_episode": list(scales),
        "shards_per_batch": 8,
        "samples_per_batch": workload.global_batch_size,
        "evaluation_seeds": evaluation_seeds,
            "fault_model": "same seeded fault prefix across scales; additional node leave/rejoin waves extend with episode scale",
        "policies_by_scale": scale_results,
    }


def _paired_improvement(
    reference: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> float:
    return statistics.fmean(
        100.0
        * (
            float(first["penalized_time_seconds"])
            - float(second["penalized_time_seconds"])
        )
        / max(1e-9, float(first["penalized_time_seconds"]))
        for first, second in zip(reference, candidate)
    )


def run_comparison(
    *,
    episodes: int = 512,
    group_size: int = 8,
    eval_episodes: int = 64,
    hidden_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_weight: float = 0.01,
    seed: int = 0,
    fixed_duplicate_after_seconds: float = 1.0,
    max_decisions: int = 256,
) -> tuple[dict[str, Any], CandidateRLOOActor]:
    if episodes < group_size or episodes % group_size or group_size < 2:
        raise ValueError("episodes must divide evenly by a group size >= 2")
    if eval_episodes < 1 or hidden_size < 1 or learning_rate <= 0 or max_decisions < 1:
        raise ValueError("evaluation count, hidden size, and learning rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    topology = four_region_topology()
    workload = example_workload()
    shard_count = 8
    max_candidates = (
        2 * shard_count * len(topology.nodes) * len((32, 48, 64))
        + len(topology.nodes)
        + len(SpeculativeWANScheduler.WAIT_OPTIONS_SECONDS)
    )
    prototype = _new_env(
        seed=seed,
        topology=topology,
        workload=workload,
        shard_count=shard_count,
    )
    policy = CandidateRLOOActor(
        len(prototype.observation()), max_candidates, hidden_size
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    failure_penalty = 100.0
    updates = []

    for group_index in range(episodes // group_size):
        scenario_seed = seed + 10_007 + group_index * 97
        trajectories = [
            _collect_episode(
                policy,
                seed=scenario_seed,
                topology=topology,
                workload=workload,
                shard_count=shard_count,
                max_decisions=max_decisions,
                maximum_candidates=max_candidates,
                failure_penalty_seconds=failure_penalty,
            )
            for _ in range(group_size)
        ]
        update = _rloo_update(
            policy,
            optimizer,
            trajectories,
            entropy_weight=entropy_weight,
            reward_scale=100.0,
        )
        update["episodes_completed"] = (group_index + 1) * group_size
        updates.append(update)

    evaluation_seeds = [seed + 1_000_003 + index * 13 for index in range(eval_episodes)]
    runs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("fixed", "adaptive", "rloo")
    }
    for evaluation_seed in evaluation_seeds:
        for name in ("fixed", "adaptive"):
            runs[name].append(
                _run_baseline_episode(
                    name,
                    seed=evaluation_seed,
                    topology=topology,
                    workload=workload,
                    shard_count=shard_count,
                    batches_per_episode=1,
                    max_decisions=max_decisions,
                    fixed_duplicate_after_seconds=fixed_duplicate_after_seconds,
                    failure_penalty_seconds=failure_penalty,
                )
            )
        runs["rloo"].append(
            _evaluate_rloo_episode(
                policy,
                seed=evaluation_seed,
                topology=topology,
                workload=workload,
                shard_count=shard_count,
                max_decisions=max_decisions,
                maximum_candidates=max_candidates,
                failure_penalty_seconds=failure_penalty,
            )
        )

    aggregates = {name: _aggregate(values) for name, values in runs.items()}
    report = {
        "experiment": "speculative_wan_fixed_adaptive_rloo",
        "seed": seed,
        "training": {
            "episodes": episodes,
            "group_size": group_size,
            "updates": len(updates),
            "eval_episodes": eval_episodes,
            "hidden_size": hidden_size,
            "failure_penalty_seconds": failure_penalty,
            "action_space": {
                "start": "any pending shard × any idle active node × feasible micro-batch",
                "duplicate": "any running shard × any idle active node",
                "cancel": "any running attempt",
                "wait": "0.1-second increments from 0.1 through 10 seconds",
            },
            "decision_overhead_seconds": 0.001,
            "max_decisions_per_episode": max_decisions,
            "episode_scope": "one global batch composed of logical shards",
        },
        "evaluation_action_mode": "seeded_stochastic",
        "baseline_definitions": {
            "fixed": "node order, micro-batch 32, duplicate a single-copy shard after 1 second",
            "adaptive": "fastest effective node, micro-batch 64, duplicate when estimated time savings exceed 0.1 second",
        },
        "workload": {
            **workload.__dict__,
            "shard_count": shard_count,
            "samples_per_shard": workload.global_batch_size // shard_count,
        },
        "policies": aggregates,
        "paired_improvements_percent": {
            "adaptive_vs_fixed": _paired_improvement(runs["fixed"], runs["adaptive"]),
            "rloo_vs_fixed": _paired_improvement(runs["fixed"], runs["rloo"]),
            "rloo_vs_adaptive": _paired_improvement(runs["adaptive"], runs["rloo"]),
        },
        "rloo_updates": updates,
        "evaluation_seeds": evaluation_seeds,
        "episode_results": runs,
    }
    return report, policy


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare speculative fixed, adaptive, and RLOO WAN schedulers."
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=256,
        help="episode guard; episode-scale mode applies this per global batch",
    )
    parser.add_argument(
        "--episode-scales",
        type=int,
        nargs="+",
        help="run fixed/adaptive scale sweep only; skips RLOO training",
    )
    parser.add_argument(
        "--fixed-duplicate-after-seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    if args.episode_scales:
        report = run_baseline_scale_sweep(
            episode_scales=args.episode_scales,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            fixed_duplicate_after_seconds=args.fixed_duplicate_after_seconds,
            max_decisions_per_batch=args.max_decisions,
        )
        output = args.output or Path(
            "RL_schedule/results/wan-speculative-episode-scale-fixed-adaptive.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "policies_by_scale": {
                        scale: result["policies"]
                        for scale, result in report["policies_by_scale"].items()
                    },
                    "rloo_training_performed": False,
                    "report": str(output),
                },
                indent=2,
            )
        )
        return

    report, policy = run_comparison(
        episodes=args.episodes,
        group_size=args.group_size,
        eval_episodes=args.eval_episodes,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        fixed_duplicate_after_seconds=args.fixed_duplicate_after_seconds,
        max_decisions=args.max_decisions,
    )
    output = args.output or Path(
        "RL_schedule/results/wan-speculative-fixed-adaptive-rloo.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".pt")
    torch.save(
        {
            "format": "wan-speculative-rloo-v2",
            "state_dict": policy.state_dict(),
            "hidden_size": args.hidden_size,
            "seed": args.seed,
        },
        checkpoint,
    )
    report["checkpoint"] = str(checkpoint)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "policies": report["policies"],
                "paired_improvements_percent": report[
                    "paired_improvements_percent"
                ],
                "report": str(output),
                "checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
