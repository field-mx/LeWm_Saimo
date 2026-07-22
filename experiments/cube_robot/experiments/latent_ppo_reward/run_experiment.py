from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import ogbench
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    HERE,
    append_jsonl,
    build_image_transform,
    encode_image,
    image_to_tensor,
    load_action_stats,
    load_config,
    load_world_model,
    predict_latent_block,
    resolve_device,
    save_json,
    set_seed,
    synchronized_call,
)
from models import RewardNetwork

from actor import GaussianActor, ValueCritic
from actor_data import actor_observation_from_env, normalize_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a latent reward head and a PPO actor inside LeWM."
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "config.yaml"
    )
    parser.add_argument(
        "--stage",
        choices=("collect", "reward", "ppo", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny interface test in outputs/smoke.",
    )
    parser.add_argument("--force-collect", action="store_true")
    return parser.parse_args()


def apply_smoke_config(config: dict) -> None:
    config["paths"]["output_dir"] = config["paths"]["output_dir"] / "smoke"
    config["collection"].update(episodes=1, max_steps=10, reuse_existing=False)
    config["reward_training"].update(epochs=2, batch_size=8)
    config["ppo"].update(
        updates=2,
        num_envs=4,
        rollout_steps=4,
        imagination_horizon=2,
        update_epochs=1,
        minibatch_size=8,
    )
    config["evaluation"].update(episodes=1, max_steps=10)


def load_bc_actor(config: dict, device: torch.device):
    checkpoint = torch.load(
        config["paths"]["bc_actor_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    actor = GaussianActor(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    actor.load_state_dict(checkpoint["actor"])
    actor = actor.to(device).eval()
    mean = np.asarray(checkpoint["observation_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["observation_scale"], dtype=np.float32)
    return actor, mean, scale


def collect_reward_data(config: dict, *, force: bool = False) -> Path:
    output_dir = config["paths"]["output_dir"]
    data_path = output_dir / "data" / "reward_rollouts.npz"
    metadata_path = output_dir / "data" / "reward_rollouts.json"
    settings = config["collection"]
    if data_path.exists() and settings["reuse_existing"] and not force:
        print(f"Reusing reward data: {data_path}")
        return data_path

    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))
    model = load_world_model(config["paths"]["model_dir"], device)
    bc_actor, observation_mean, observation_scale = load_bc_actor(config, device)
    action_mean, action_scale = load_action_stats(
        config["paths"]["action_dataset"]
    )
    transform = build_image_transform(int(config["image_size"]))
    rng = np.random.default_rng(int(config["seed"]))
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)

    starts: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    rewards: list[float] = []
    distances: list[float] = []
    successes: list[bool] = []
    episode_ids: list[int] = []
    prediction_mses: list[float] = []
    wm_seconds = 0.0
    wm_calls = 0
    action_block = int(settings["action_block"])

    for episode in range(int(settings["episodes"])):
        state, reset_info = env.reset(
            seed=int(config["seed"]) + episode,
            options={"render_goal": True},
        )
        goal_state = np.asarray(reset_info["goal"], dtype=np.float32)
        goal_pixels = image_to_tensor(
            reset_info["goal_rendered"], transform, device
        )
        goal_latent, elapsed = synchronized_call(
            lambda: encode_image(model, goal_pixels), device
        )
        wm_seconds += elapsed
        wm_calls += 1
        current_pixels = image_to_tensor(env.render(), transform, device)
        current_latent, elapsed = synchronized_call(
            lambda: encode_image(model, current_pixels), device
        )
        wm_seconds += elapsed
        wm_calls += 1
        total_steps = 0

        while total_steps < int(settings["max_steps"]):
            raw_actions = []
            terminated = False
            truncated = False
            success = False
            next_state = state

            for _ in range(action_block):
                actor_input = actor_observation_from_env(state, goal_state)
                actor_input = normalize_observations(
                    actor_input, observation_mean, observation_scale
                )
                actor_tensor = torch.from_numpy(actor_input).to(device).unsqueeze(0)
                with torch.inference_mode():
                    action = bc_actor.deterministic_action(actor_tensor)[0]
                action = action.cpu().numpy()
                action += rng.normal(
                    0.0, float(settings["exploration_std"]), size=action.shape
                )
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                next_state, _, terminated, truncated, info = env.step(action)
                raw_actions.append(action)
                state = next_state
                total_steps += 1
                success = bool(info.get("success", False))
                if terminated or truncated or total_steps >= int(settings["max_steps"]):
                    break

            while len(raw_actions) < action_block:
                raw_actions.append(np.zeros(5, dtype=np.float32))
            raw_action_block = np.stack(raw_actions)
            normalized_block = (
                (raw_action_block - action_mean) / action_scale
            ).reshape(-1).astype(np.float32)
            block_tensor = torch.from_numpy(normalized_block).to(device).unsqueeze(0)
            predicted_latent, elapsed = synchronized_call(
                lambda: predict_latent_block(model, current_latent, block_tensor),
                device,
            )
            wm_seconds += elapsed
            wm_calls += 1

            next_pixels = image_to_tensor(env.render(), transform, device)
            actual_next_latent, elapsed = synchronized_call(
                lambda: encode_image(model, next_pixels), device
            )
            wm_seconds += elapsed
            wm_calls += 1

            distance_m = float(
                np.linalg.norm(next_state[19:22] - goal_state[19:22]) / 10.0
            )
            dense_score = 1.0 - min(
                distance_m / float(settings["reward_distance_scale_m"]), 1.0
            )
            reward_target = dense_score + float(settings["success_bonus"]) * success

            starts.append(current_latent[0].float().cpu().numpy())
            predictions.append(predicted_latent[0].float().cpu().numpy())
            goals.append(goal_latent[0].float().cpu().numpy())
            action_blocks.append(normalized_block)
            rewards.append(float(reward_target))
            distances.append(distance_m)
            successes.append(success)
            episode_ids.append(episode)
            prediction_mses.append(
                float((predicted_latent - actual_next_latent).square().mean().item())
            )
            current_latent = actual_next_latent

            if terminated or truncated:
                break

        print(
            json.dumps(
                {
                    "stage": "collect",
                    "episode": episode,
                    "steps": total_steps,
                    "success": success,
                    "samples": len(starts),
                }
            )
        )

    env.close()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        start_latents=np.asarray(starts, dtype=np.float32),
        predicted_latents=np.asarray(predictions, dtype=np.float32),
        goal_latents=np.asarray(goals, dtype=np.float32),
        normalized_action_blocks=np.asarray(action_blocks, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        distances_m=np.asarray(distances, dtype=np.float32),
        successes=np.asarray(successes, dtype=bool),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )
    metadata = {
        "samples": len(starts),
        "episodes": int(settings["episodes"]),
        "success_samples": int(np.sum(successes)),
        "mean_prediction_mse": float(np.mean(prediction_mses)),
        "world_model_inference_seconds": wm_seconds,
        "world_model_calls": wm_calls,
        "mean_world_model_call_seconds": wm_seconds / max(1, wm_calls),
        "reward_definition": "1 - clip(cube_goal_distance / scale, 0, 1) + success_bonus",
        "action_mean": action_mean.tolist(),
        "action_scale": action_scale.tolist(),
    }
    save_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    return data_path


def split_indices_by_episode(
    episode_ids: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_episodes = episodes[:validation_count]
    validation = np.flatnonzero(np.isin(episode_ids, validation_episodes))
    training = np.flatnonzero(~np.isin(episode_ids, validation_episodes))
    if len(training) == 0 or len(validation) == 0:
        indices = rng.permutation(len(episode_ids))
        split = min(max(1, int(0.8 * len(indices))), max(1, len(indices) - 1))
        training, validation = indices[:split], indices[split:]
    if len(validation) == 0:
        validation = training.copy()
    return training, validation


@torch.no_grad()
def evaluate_reward_model(
    model: RewardNetwork, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for predicted, goal, target in loader:
        predicted, goal, target = (
            predicted.to(device),
            goal.to(device),
            target.to(device),
        )
        total += (model(predicted, goal) - target).square().sum().item()
        count += len(target)
    return total / max(1, count)


def train_reward(config: dict, data_path: Path) -> Path:
    output_dir = config["paths"]["output_dir"]
    checkpoint_path = output_dir / "checkpoints" / "reward_best.pt"
    log_path = output_dir / "logs" / "reward_training.jsonl"
    log_path.unlink(missing_ok=True)
    settings = config["reward_training"]
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))

    with np.load(data_path) as data:
        predicted = np.asarray(data["predicted_latents"], dtype=np.float32)
        goals = np.asarray(data["goal_latents"], dtype=np.float32)
        targets = np.asarray(data["rewards"], dtype=np.float32)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int32)
    train_indices, validation_indices = split_indices_by_episode(
        episode_ids,
        float(settings["validation_fraction"]),
        int(config["seed"]),
    )
    features = np.concatenate([predicted, goals], axis=-1)
    feature_mean = features[train_indices].mean(axis=0).astype(np.float32)
    feature_scale = np.maximum(
        features[train_indices].std(axis=0).astype(np.float32), 1e-5
    )
    latent_dim = predicted.shape[-1]
    model = RewardNetwork(latent_dim, int(settings["hidden_dim"])).to(device)
    model.set_normalization(
        torch.from_numpy(feature_mean).to(device),
        torch.from_numpy(feature_scale).to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=1e-5
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(predicted[train_indices]),
            torch.from_numpy(goals[train_indices]),
            torch.from_numpy(targets[train_indices]),
        ),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(predicted[validation_indices]),
            torch.from_numpy(goals[validation_indices]),
            torch.from_numpy(targets[validation_indices]),
        ),
        batch_size=int(settings["batch_size"]),
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch_predicted, batch_goal, batch_target in train_loader:
            batch_predicted = batch_predicted.to(device)
            batch_goal = batch_goal.to(device)
            batch_target = batch_target.to(device)
            loss = nn.functional.mse_loss(
                model(batch_predicted, batch_goal), batch_target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_target)
            sample_count += len(batch_target)
        record = {
            "stage": "reward_training",
            "epoch": epoch,
            "train_mse": total_loss / max(1, sample_count),
            "validation_mse": evaluate_reward_model(
                model, validation_loader, device
            ),
        }
        append_jsonl(log_path, record)
        print(json.dumps(record))
        if record["validation_mse"] < best_validation:
            best_validation = record["validation_mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_dim": latent_dim,
                    "hidden_dim": int(settings["hidden_dim"]),
                    "validation_mse": best_validation,
                    "data_path": str(data_path),
                },
                checkpoint_path,
            )
    return checkpoint_path


def load_reward_model(path: Path, device: torch.device) -> RewardNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RewardNetwork(
        int(checkpoint["latent_dim"]), int(checkpoint["hidden_dim"])
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval().requires_grad_(False)


def train_ppo(config: dict, data_path: Path, reward_path: Path) -> Path:
    output_dir = config["paths"]["output_dir"]
    checkpoint_path = output_dir / "checkpoints" / "latent_ppo_actor.pt"
    log_path = output_dir / "logs" / "ppo_training.jsonl"
    log_path.unlink(missing_ok=True)
    settings = config["ppo"]
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))
    model = load_world_model(config["paths"]["model_dir"], device)
    reward_model = load_reward_model(reward_path, device)
    with np.load(data_path) as data:
        starts_np = np.asarray(data["start_latents"], dtype=np.float32)
        goals_np = np.asarray(data["goal_latents"], dtype=np.float32)
        normalized_blocks_np = np.asarray(
            data["normalized_action_blocks"], dtype=np.float32
        )
    start_bank = torch.from_numpy(starts_np).to(device)
    goal_bank = torch.from_numpy(goals_np).to(device)
    action_mean, action_scale = load_action_stats(
        config["paths"]["action_dataset"]
    )
    action_mean_tensor = torch.from_numpy(action_mean).to(device).view(1, 1, 5)
    action_scale_tensor = torch.from_numpy(action_scale).to(device).view(1, 1, 5)
    latent_dim = starts_np.shape[-1]
    observation_dim = 2 * latent_dim
    action_dim = int(config["collection"]["action_block"]) * 5
    actor = GaussianActor(
        observation_dim, action_dim, int(settings["hidden_dim"])
    ).to(device)
    critic = ValueCritic(observation_dim, int(settings["hidden_dim"])).to(device)
    raw_blocks_np = (
        normalized_blocks_np.reshape(-1, int(config["collection"]["action_block"]), 5)
        * action_scale.reshape(1, 1, 5)
        + action_mean.reshape(1, 1, 5)
    ).reshape(-1, action_dim)
    bc_observations = np.concatenate([starts_np, goals_np], axis=-1)
    bc_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(bc_observations),
            torch.from_numpy(raw_blocks_np.astype(np.float32)),
        ),
        batch_size=int(settings["bc_initialization_batch_size"]),
        shuffle=True,
    )
    bc_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(settings["bc_initialization_learning_rate"]),
        weight_decay=1e-5,
    )
    bc_log_path = output_dir / "logs" / "ppo_bc_initialization.jsonl"
    bc_log_path.unlink(missing_ok=True)
    for epoch in range(1, int(settings["bc_initialization_epochs"]) + 1):
        actor.train()
        total_bc_loss = 0.0
        bc_count = 0
        for batch_observation, batch_action in bc_loader:
            batch_observation = batch_observation.to(device)
            batch_action = batch_action.to(device)
            bc_loss = nn.functional.mse_loss(
                actor.deterministic_action(batch_observation), batch_action
            )
            bc_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            bc_optimizer.step()
            total_bc_loss += bc_loss.item() * len(batch_action)
            bc_count += len(batch_action)
        bc_record = {
            "stage": "ppo_bc_initialization",
            "epoch": epoch,
            "action_mse": total_bc_loss / max(1, bc_count),
        }
        append_jsonl(bc_log_path, bc_record)
        print(json.dumps(bc_record))
    anchor_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(settings["learning_rate"]),
        weight_decay=1e-5,
    )
    rng = np.random.default_rng(int(config["seed"]))
    num_envs = int(settings["num_envs"])

    def sample_initial(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(
            rng.integers(0, len(start_bank), size=count, dtype=np.int64)
        ).to(device)
        return start_bank[indices], goal_bank[indices]

    latent, goal = sample_initial(num_envs)
    ages = torch.zeros(num_envs, dtype=torch.long, device=device)
    cumulative_wm_seconds = 0.0
    cumulative_wm_calls = 0

    for update in range(1, int(settings["updates"]) + 1):
        observations = []
        actions = []
        old_log_probs = []
        rewards = []
        dones = []
        values = []
        update_wm_seconds = 0.0

        actor.eval()
        critic.eval()
        for _ in range(int(settings["rollout_steps"])):
            observation = torch.cat([latent, goal], dim=-1)
            with torch.no_grad():
                action, log_prob = actor.sample(observation)
                value = critic(observation)
            normalized_action = (
                action.view(num_envs, int(config["collection"]["action_block"]), 5)
                - action_mean_tensor
            ) / action_scale_tensor
            normalized_action = normalized_action.reshape(num_envs, -1)
            next_latent, elapsed = synchronized_call(
                lambda: predict_latent_block(model, latent, normalized_action), device
            )
            update_wm_seconds += elapsed
            cumulative_wm_calls += 1
            with torch.no_grad():
                predicted_reward = reward_model(next_latent, goal)
                shaped_reward = predicted_reward - float(
                    settings["action_penalty"]
                ) * action.square().mean(dim=-1)
            ages += 1
            done = (ages >= int(settings["imagination_horizon"])) | (
                predicted_reward >= float(settings["predicted_success_reward"])
            )
            observations.append(observation)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(shaped_reward)
            dones.append(done.float())
            values.append(value)
            latent = next_latent
            if done.any():
                replacement_latent, replacement_goal = sample_initial(int(done.sum()))
                latent = latent.clone()
                goal = goal.clone()
                latent[done] = replacement_latent
                goal[done] = replacement_goal
                ages[done] = 0

        cumulative_wm_seconds += update_wm_seconds
        with torch.no_grad():
            last_value = critic(torch.cat([latent, goal], dim=-1))
        observations_t = torch.stack(observations)
        actions_t = torch.stack(actions)
        old_log_probs_t = torch.stack(old_log_probs)
        rewards_t = torch.stack(rewards)
        dones_t = torch.stack(dones)
        values_t = torch.stack(values)
        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(num_envs, device=device)
        gamma = float(settings["gamma"])
        gae_lambda = float(settings["gae_lambda"])
        for step in reversed(range(len(rewards))):
            next_value = last_value if step == len(rewards) - 1 else values_t[step + 1]
            nonterminal = 1.0 - dones_t[step]
            delta = rewards_t[step] + gamma * next_value * nonterminal - values_t[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values_t

        flat_observations = observations_t.flatten(0, 1)
        flat_actions = actions_t.flatten(0, 1)
        flat_old_log_probs = old_log_probs_t.flatten()
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )
        indices = np.arange(len(flat_observations))
        losses = []
        actor.train()
        critic.train()
        for _ in range(int(settings["update_epochs"])):
            rng.shuffle(indices)
            for start in range(0, len(indices), int(settings["minibatch_size"])):
                batch_indices = torch.as_tensor(
                    indices[start : start + int(settings["minibatch_size"])],
                    device=device,
                )
                batch_observation = flat_observations[batch_indices]
                new_log_prob = actor.log_prob(
                    batch_observation, flat_actions[batch_indices]
                )
                ratio = (new_log_prob - flat_old_log_probs[batch_indices]).exp()
                unclipped = ratio * flat_advantages[batch_indices]
                clipped = ratio.clamp(
                    1.0 - float(settings["clip_ratio"]),
                    1.0 + float(settings["clip_ratio"]),
                ) * flat_advantages[batch_indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(
                    critic(batch_observation), flat_returns[batch_indices]
                )
                entropy = actor.distribution(batch_observation).entropy().sum(-1).mean()
                with torch.no_grad():
                    anchor_action = anchor_actor.deterministic_action(batch_observation)
                anchor_loss = nn.functional.mse_loss(
                    actor.deterministic_action(batch_observation), anchor_action
                )
                loss = (
                    policy_loss
                    + float(settings["value_coefficient"]) * value_loss
                    - float(settings["entropy_coefficient"]) * entropy
                    + float(settings["bc_anchor_coefficient"]) * anchor_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    float(settings["max_grad_norm"]),
                )
                optimizer.step()
                losses.append(
                    (
                        policy_loss.item(),
                        value_loss.item(),
                        entropy.item(),
                        anchor_loss.item(),
                    )
                )

        record = {
            "stage": "ppo_training",
            "update": update,
            "imagined_reward_mean": float(rewards_t.mean().item()),
            "imagined_reward_max": float(rewards_t.max().item()),
            "done_fraction": float(dones_t.mean().item()),
            "policy_loss": float(np.mean([value[0] for value in losses])),
            "value_loss": float(np.mean([value[1] for value in losses])),
            "raw_policy_entropy": float(np.mean([value[2] for value in losses])),
            "bc_anchor_loss": float(np.mean([value[3] for value in losses])),
            "world_model_inference_seconds": update_wm_seconds,
            "cumulative_world_model_inference_seconds": cumulative_wm_seconds,
            "world_model_calls": int(settings["rollout_steps"]),
        }
        append_jsonl(log_path, record)
        print(json.dumps(record))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "latent_dim": latent_dim,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": int(settings["hidden_dim"]),
            "action_mean": action_mean,
            "action_scale": action_scale,
            "actor_action_space": "raw_env_actions_in_minus_one_to_one",
            "world_model_inference_seconds": cumulative_wm_seconds,
            "world_model_calls": cumulative_wm_calls,
            "reward_checkpoint": str(reward_path),
        },
        checkpoint_path,
    )
    return checkpoint_path


def evaluate_actor(config: dict, actor_path: Path) -> dict:
    output_dir = config["paths"]["output_dir"]
    settings = config["evaluation"]
    device = resolve_device(config["device"])
    model = load_world_model(config["paths"]["model_dir"], device)
    checkpoint = torch.load(actor_path, map_location="cpu", weights_only=False)
    actor = GaussianActor(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    )
    actor.load_state_dict(checkpoint["actor"])
    actor = actor.to(device).eval()
    transform = build_image_transform(int(config["image_size"]))
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    episode_results = []
    wm_seconds = 0.0
    wm_calls = 0

    for episode in range(int(settings["episodes"])):
        _, info = env.reset(
            seed=int(config["seed"]) + episode,
            options={"render_goal": True},
        )
        goal_pixels = image_to_tensor(info["goal_rendered"], transform, device)
        goal_latent, elapsed = synchronized_call(
            lambda: encode_image(model, goal_pixels), device
        )
        wm_seconds += elapsed
        wm_calls += 1
        frames = []
        steps = 0
        success = False
        terminated = False
        truncated = False

        while steps < int(settings["max_steps"]) and not (terminated or truncated):
            current_image = np.asarray(env.render()).copy()
            frames.append(current_image)
            current_pixels = image_to_tensor(current_image, transform, device)
            current_latent, elapsed = synchronized_call(
                lambda: encode_image(model, current_pixels), device
            )
            wm_seconds += elapsed
            wm_calls += 1
            observation = torch.cat([current_latent, goal_latent], dim=-1)
            with torch.inference_mode():
                if bool(settings["deterministic"]):
                    action_block = actor.deterministic_action(observation)[0]
                else:
                    action_block, _ = actor.sample(observation)
                    action_block = action_block[0]
            action_block = action_block.cpu().numpy().reshape(-1, 5)
            for action in action_block:
                _, _, terminated, truncated, step_info = env.step(
                    np.clip(action, -1.0, 1.0).astype(np.float32)
                )
                steps += 1
                success = bool(step_info.get("success", False))
                frames.append(np.asarray(env.render()).copy())
                if terminated or truncated or steps >= int(settings["max_steps"]):
                    break

        if episode == 0:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                video_dir / "latent_ppo_episode_0.mp4",
                frames,
                fps=int(settings["fps"]),
            )
        result = {
            "episode": episode,
            "steps": steps,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        episode_results.append(result)
        print(json.dumps(result))

    env.close()
    summary = {
        "episodes": len(episode_results),
        "successes": int(sum(item["success"] for item in episode_results)),
        "success_rate": float(np.mean([item["success"] for item in episode_results])),
        "world_model_encoder_seconds": wm_seconds,
        "world_model_encoder_calls": wm_calls,
        "mean_world_model_encoder_seconds": wm_seconds / max(1, wm_calls),
        "episode_results": episode_results,
    }
    save_json(output_dir / "evaluation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.smoke:
        apply_smoke_config(config)
    output_dir = config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config_to_json(config))

    data_path = output_dir / "data" / "reward_rollouts.npz"
    reward_path = output_dir / "checkpoints" / "reward_best.pt"
    actor_path = output_dir / "checkpoints" / "latent_ppo_actor.pt"

    if args.stage in ("collect", "all"):
        data_path = collect_reward_data(config, force=args.force_collect)
    if args.stage in ("reward", "all"):
        if not data_path.exists():
            raise FileNotFoundError(f"Reward data missing: {data_path}")
        reward_path = train_reward(config, data_path)
    if args.stage in ("ppo", "all"):
        if not reward_path.exists():
            raise FileNotFoundError(f"Reward checkpoint missing: {reward_path}")
        actor_path = train_ppo(config, data_path, reward_path)
    if args.stage in ("evaluate", "all"):
        if not actor_path.exists():
            raise FileNotFoundError(f"PPO Actor checkpoint missing: {actor_path}")
        evaluate_actor(config, actor_path)


def config_to_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: config_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [config_to_json(item) for item in value]
    return value


if __name__ == "__main__":
    main()
