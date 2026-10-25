"""PPO policies for the simulated wait/duplicate scheduling decisions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .evaluate import run_episode
from .scenarios import evaluation_workers, training_workers
from .simulator import SchedulingEnv, SimulationConfig, WorkerSpec


@dataclass(frozen=True)
class PPOConfig:
    episodes: int = 1000
    rollout_episodes: int = 16
    worker_count: int = 3
    architecture: str = "mlp"
    history_length: int = 4
    hidden_size: int = 128
    seed: int = 0
    learning_rate: float = 0.0003
    gamma_per_second: float = 0.995
    gae_lambda_per_second: float = 0.97
    clip_epsilon: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128

    def validate(self) -> None:
        if self.episodes < 1 or self.rollout_episodes < 1 or self.worker_count < 1:
            raise ValueError("episode counts and worker_count must be at least 1")
        if self.history_length < 1 or self.hidden_size < 1:
            raise ValueError("history length and hidden size must be at least 1")
        if self.architecture not in {"mlp", "lstm"}:
            raise ValueError("architecture must be 'mlp' or 'lstm'")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite")
        if (
            not math.isfinite(self.entropy_weight)
            or self.entropy_weight < 0
            or not math.isfinite(self.value_weight)
            or self.value_weight < 0
        ):
            raise ValueError("loss weights must be finite and non-negative")
        if not 0 < self.gamma_per_second <= 1:
            raise ValueError("gamma_per_second must be in the interval (0, 1]")
        if not 0 < self.gae_lambda_per_second <= 1:
            raise ValueError("gae_lambda_per_second must be in the interval (0, 1]")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in the interval (0, 1)")
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("update_epochs and minibatch_size must be at least 1")


class PolicyValueNetwork(nn.Module):
    """MLP actor and critic over concatenated recent observations."""

    def __init__(self, observation_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_count = action_count
        self.trunk = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        logits = self.policy_head(hidden).masked_fill(~action_masks, -1e9)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class LSTMPolicyValueNetwork(nn.Module):
    """Actor-critic that recurrently encodes the recent observation frames."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_size: int,
        history_length: int,
    ) -> None:
        super().__init__()
        if observation_size % history_length:
            raise ValueError("observation size must be divisible by history length")
        self.observation_size = observation_size
        self.action_count = action_count
        self.history_length = history_length
        self.frame_size = observation_size // history_length
        self.lstm = nn.LSTM(
            input_size=self.frame_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = observations.reshape(
            observations.shape[0], self.history_length, self.frame_size
        )
        encoded, _ = self.lstm(sequence)
        hidden = encoded[:, -1, :]
        logits = self.policy_head(hidden).masked_fill(~action_masks, -1e9)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def _make_policy_network(
    architecture: str,
    observation_size: int,
    action_count: int,
    hidden_size: int,
    history_length: int,
) -> nn.Module:
    if architecture == "mlp":
        return PolicyValueNetwork(observation_size, action_count, hidden_size)
    if architecture == "lstm":
        return LSTMPolicyValueNetwork(
            observation_size, action_count, hidden_size, history_length
        )
    raise ValueError(f"unknown policy architecture: {architecture}")


def _gae(
    rewards: list[float],
    durations: list[float],
    values: list[float],
    gamma_per_second: float,
    gae_lambda_per_second: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_value = 0.0
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        discount = gamma_per_second**durations[index]
        trace = gae_lambda_per_second**durations[index]
        delta = rewards[index] + discount * next_value - values[index]
        gae = delta + discount * trace * gae
        advantages[index] = gae
        returns[index] = gae + values[index]
        next_value = values[index]
    return advantages, returns


def _collect_episode(
    policy: nn.Module,
    workers: Sequence[WorkerSpec],
    history_length: int,
    seed: int,
    config: PPOConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = SchedulingEnv(workers=workers, history_length=history_length)
    observation, mask = env.reset(seed)
    trajectory: list[dict[str, Any]] = []
    rewards: list[float] = []
    durations: list[float] = []
    values: list[float] = []

    while not env.done:
        with torch.no_grad():
            logits, value = policy(observation.unsqueeze(0), mask.unsqueeze(0))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
        next_observation, reward, done, next_mask, info = env.step(int(action.item()))
        trajectory.append(
            {
                "observation": observation,
                "mask": mask,
                "action": int(action.item()),
                "old_log_probability": float(log_probability.item()),
            }
        )
        rewards.append(reward)
        durations.append(float(info["delta_seconds"]))
        values.append(float(value.item()))
        observation, mask = next_observation, next_mask
        if done:
            break

    advantages, returns = _gae(
        rewards,
        durations,
        values,
        gamma_per_second=config.gamma_per_second,
        gae_lambda_per_second=config.gae_lambda_per_second,
    )
    for item, advantage, target in zip(trajectory, advantages, returns):
        item["advantage"] = advantage
        item["return"] = target
    metrics = env.metrics()
    metrics["episode_return"] = sum(rewards)
    return trajectory, metrics


def _update_policy(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectory: list[dict[str, Any]],
    config: PPOConfig,
) -> dict[str, float]:
    observations = torch.stack([item["observation"] for item in trajectory])
    masks = torch.stack([item["mask"] for item in trajectory])
    actions = torch.tensor([item["action"] for item in trajectory], dtype=torch.long)
    old_log_probabilities = torch.tensor(
        [item["old_log_probability"] for item in trajectory], dtype=torch.float32
    )
    advantages = torch.tensor(
        [item["advantage"] for item in trajectory], dtype=torch.float32
    )
    returns = torch.tensor([item["return"] for item in trajectory], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    for _ in range(config.update_epochs):
        order = torch.randperm(len(trajectory))
        for start in range(0, len(trajectory), config.minibatch_size):
            indices = order[start : start + config.minibatch_size]
            logits, values = policy(observations[indices], masks[indices])
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(actions[indices])
            ratio = torch.exp(new_log_probabilities - old_log_probabilities[indices])
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[indices])
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_weight * value_loss
                - config.entropy_weight * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            entropies.append(float(entropy.detach().item()))

    return {
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "entropy": sum(entropies) / len(entropies),
    }


def train_policy(
    config: PPOConfig,
    output: Path,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    prototype_env = SchedulingEnv(
        workers=evaluation_workers("seen", config.worker_count),
        history_length=config.history_length,
    )
    policy = _make_policy_network(
        config.architecture,
        prototype_env.observation_size,
        prototype_env.action_count,
        config.hidden_size,
        config.history_length,
    )
    starting_episode = 0
    resumed_checkpoint: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resumed_checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=True
        )
        expected = {
            "history_length": config.history_length,
            "worker_count": config.worker_count,
            "architecture": config.architecture,
            "hidden_size": config.hidden_size,
            "observation_size": prototype_env.observation_size,
            "action_count": prototype_env.action_count,
        }
        actual = {
            "history_length": int(resumed_checkpoint["history_length"]),
            "worker_count": int(
                resumed_checkpoint.get(
                    "worker_count",
                    int(resumed_checkpoint["action_count"])
                    - len(SimulationConfig().wait_seconds),
                )
            ),
            "architecture": str(resumed_checkpoint.get("architecture", "mlp")),
            "hidden_size": int(resumed_checkpoint["hidden_size"]),
            "observation_size": int(resumed_checkpoint["observation_size"]),
            "action_count": int(resumed_checkpoint["action_count"]),
        }
        if actual != expected:
            raise ValueError(
                f"resume checkpoint dimensions/configuration {actual} do not match {expected}"
            )
        policy.load_state_dict(resumed_checkpoint["state_dict"])
        previous_config = resumed_checkpoint.get("training_config", {})
        starting_episode = int(
            resumed_checkpoint.get(
                "episodes_completed", previous_config.get("episodes", 0)
            )
        )
        if starting_episode < 0:
            raise ValueError("resume checkpoint has an invalid completed episode count")

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    optimizer_state_restored = False
    if resumed_checkpoint is not None:
        if "optimizer_state_dict" in resumed_checkpoint:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
            optimizer_state_restored = True
        if "torch_rng_state" in resumed_checkpoint:
            torch.set_rng_state(resumed_checkpoint["torch_rng_state"])
        else:
            torch.manual_seed(config.seed + starting_episode)
    update_reports: list[dict[str, float | int]] = []
    episode = starting_episode
    target_episode = starting_episode + config.episodes

    while episode < target_episode:
        batch_trajectories: list[dict[str, Any]] = []
        batch_metrics: list[dict[str, Any]] = []
        batch_count = min(config.rollout_episodes, target_episode - episode)
        for _ in range(batch_count):
            workers = training_workers(
                config.seed + episode * 7919, config.worker_count
            )
            trajectory, metrics = _collect_episode(
                policy,
                workers,
                config.history_length,
                seed=config.seed + 100_003 + episode,
                config=config,
            )
            batch_trajectories.extend(trajectory)
            batch_metrics.append(metrics)
            episode += 1

        update_metrics = _update_policy(policy, optimizer, batch_trajectories, config)
        successful = [metric for metric in batch_metrics if metric["success"]]
        report: dict[str, float | int] = {
            "episodes_completed": episode,
            "episodes_in_update": batch_count,
            "mean_reward": sum(
                float(metric["episode_return"]) for metric in batch_metrics
            )
            / batch_count,
            "completion_rate": len(successful) / batch_count,
            "mean_makespan_seconds": (
                sum(float(metric["makespan_seconds"]) for metric in successful)
                / len(successful)
                if successful
                else 0.0
            ),
            **update_metrics,
        }
        update_reports.append(report)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "rl-schedule-ppo-v1",
            "history_length": config.history_length,
            "worker_count": config.worker_count,
            "architecture": config.architecture,
            "hidden_size": config.hidden_size,
            "observation_size": prototype_env.observation_size,
            "action_count": prototype_env.action_count,
            "episodes_completed": episode,
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "training_config": asdict(config),
        },
        output,
    )
    return {
        "checkpoint": str(output),
        "algorithm": "PPO",
        "policy_type": "instantaneous" if config.history_length == 1 else "history_aware",
        "architecture": config.architecture,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "training_config": asdict(config),
        "starting_episodes": starting_episode,
        "episodes_trained": config.episodes,
        "episodes_completed_total": episode,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "optimizer_state_restored": optimizer_state_restored,
        "updates": update_reports,
    }


def _load_policy(checkpoint_path: Path) -> tuple[nn.Module, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    history_length = int(checkpoint["history_length"])
    policy = _make_policy_network(
        str(checkpoint.get("architecture", "mlp")),
        int(checkpoint["observation_size"]),
        int(checkpoint["action_count"]),
        int(checkpoint["hidden_size"]),
        history_length,
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    worker_count = int(
        checkpoint.get(
            "worker_count",
            int(checkpoint["action_count"])
            - len(SimulationConfig().wait_seconds),
        )
    )
    return policy, history_length, worker_count


def _run_policy_episode(
    policy: nn.Module,
    history_length: int,
    workers: Sequence[WorkerSpec],
    seed: int,
) -> dict[str, Any]:
    env = SchedulingEnv(workers=workers, history_length=history_length)
    observation, mask = env.reset(seed)
    while not env.done:
        with torch.no_grad():
            logits, _ = policy(observation.unsqueeze(0), mask.unsqueeze(0))
        action = int(logits.argmax(dim=-1).item())
        observation, _, _, mask, _ = env.step(action)
    return env.metrics()


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run["success"]]
    makespans = [float(run["makespan_seconds"]) for run in successful]
    return {
        "episodes": len(runs),
        "completion_rate": len(successful) / len(runs),
        "mean_makespan_seconds": sum(makespans) / len(makespans) if makespans else None,
        "makespan_stddev_seconds": (
            math.sqrt(sum((value - sum(makespans) / len(makespans)) ** 2 for value in makespans) / len(makespans))
            if makespans
            else None
        ),
        "mean_network_bytes_transferred": sum(
            float(run["network_bytes_transferred"]) for run in runs
        )
        / len(runs),
        "mean_network_work_seconds": sum(
            float(run["network_work_seconds"]) for run in runs
        )
        / len(runs),
        "mean_total_compute_seconds": sum(
            float(run["total_compute_seconds"]) for run in runs
        )
        / len(runs),
        "mean_duplicate_compute_seconds": sum(
            float(run["duplicate_compute_seconds"]) for run in runs
        )
        / len(runs),
        "mean_duplicates_started": sum(float(run["duplicates_started"]) for run in runs)
        / len(runs),
        "mean_late_results_ignored": sum(
            float(run["late_results_ignored"]) for run in runs
        )
        / len(runs),
        "mean_invalid_results": sum(float(run["invalid_results"]) for run in runs)
        / len(runs),
        "mean_recovery_time_seconds": sum(
            float(run["recovery_time_seconds"]) for run in runs
        )
        / len(runs),
    }


def evaluate_checkpoint(
    checkpoint_path: Path, episodes: int = 100, seed: int = 0
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    policy, history_length, worker_count = _load_policy(checkpoint_path)
    reports: dict[str, Any] = {}
    for condition in ("seen", "unseen"):
        workers = evaluation_workers(condition, worker_count)
        policy_runs = [
            _run_policy_episode(
                policy,
                history_length,
                workers,
                seed + episode,
            )
            for episode in range(episodes)
        ]
        baseline_runs: dict[str, list[dict[str, Any]]] = {
            policy_name: [
                run_episode(
                    policy_name,
                    seed + episode,
                    workers=workers,
                )
                for episode in range(episodes)
            ]
            for policy_name in ("wait_only", "fixed_timeout", "adaptive_heuristic")
        }
        reports[condition] = {
            "trained_policy": _aggregate_runs(policy_runs),
            "baselines": {
                name: _aggregate_runs(runs) for name, runs in baseline_runs.items()
            },
        }
    return reports


def compare_checkpoints(
    instantaneous_path: Path,
    history_aware_path: Path,
    episodes: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare both policy inputs on identical seen and unseen episode seeds."""
    return {
        "instantaneous": evaluate_checkpoint(instantaneous_path, episodes, seed),
        "history_aware": evaluate_checkpoint(history_aware_path, episodes, seed),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate an RL scheduler.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train a PPO scheduler")
    train_parser.add_argument("--episodes", type=int, default=1000)
    train_parser.add_argument("--rollout-episodes", type=int, default=16)
    train_parser.add_argument("--worker-count", type=int, default=3)
    train_parser.add_argument("--architecture", choices=("mlp", "lstm"), default="mlp")
    train_parser.add_argument("--history-length", type=int, default=4)
    train_parser.add_argument("--hidden-size", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="continue training from a compatible PPO checkpoint",
    )
    train_parser.add_argument("--metrics", type=Path)
    train_parser.add_argument("--eval-episodes", type=int, default=100)

    eval_parser = commands.add_parser("evaluate", help="compare a saved PPO scheduler")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=0)
    compare_parser = commands.add_parser(
        "compare", help="compare instantaneous and history-aware checkpoints"
    )
    compare_parser.add_argument("--instantaneous", type=Path, required=True)
    compare_parser.add_argument("--history-aware", type=Path, required=True)
    compare_parser.add_argument("--episodes", type=int, default=100)
    compare_parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "train":
        config = PPOConfig(
            episodes=args.episodes,
            rollout_episodes=args.rollout_episodes,
            worker_count=args.worker_count,
            architecture=args.architecture,
            history_length=args.history_length,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        result = train_policy(config, args.output, args.resume_checkpoint)
        if args.eval_episodes > 0:
            result["evaluation"] = evaluate_checkpoint(
                args.output,
                episodes=args.eval_episodes,
                seed=args.seed + 500_000,
            )
        rendered = json.dumps(result, indent=2)
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate_checkpoint(args.checkpoint, args.episodes, args.seed),
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                compare_checkpoints(
                    args.instantaneous,
                    args.history_aware,
                    args.episodes,
                    args.seed,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
