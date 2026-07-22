from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import ogbench
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from torch import nn
from torchvision.transforms import v2 as transforms


HERE = Path(__file__).resolve().parent
CUBE_ROOT = HERE.parents[1]
if str(CUBE_ROOT) not in sys.path:
    sys.path.insert(0, str(CUBE_ROOT))

from actor import GaussianActor
from actor_data import actor_observation_from_env, normalize_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actor-guided random-shooting MPC with LeWM scoring."
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny interface test."
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key, value in config["paths"].items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (HERE / candidate).resolve()
        config["paths"][key] = candidate
    return config


def apply_smoke_config(config: dict) -> None:
    config["paths"]["output_dir"] = config["paths"]["output_dir"] / "smoke"
    config["planner"].update(
        num_candidates=4,
        horizon_blocks=2,
        execute_steps=2,
        elite_count=2,
    )
    config["evaluation"].update(episodes=1, max_steps=4)


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")


def synchronized_call(function, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def image_tensor(image, transform, device):
    return transform(np.asarray(image)).unsqueeze(0).unsqueeze(0).to(device)


def load_actor(config: dict, device: torch.device):
    checkpoint = torch.load(
        config["paths"]["actor_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    actor = GaussianActor(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    )
    actor.load_state_dict(checkpoint["actor"])
    actor = actor.to(device).eval()
    observation_mean = np.asarray(
        checkpoint["observation_mean"], dtype=np.float32
    )
    observation_scale = np.asarray(
        checkpoint["observation_scale"], dtype=np.float32
    )
    return actor, checkpoint, observation_mean, observation_scale


def load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as dataset:
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-4)


def sample_actor_guided_plans(
    base_action: np.ndarray,
    *,
    num_candidates: int,
    total_steps: int,
    exploration_std: float,
    temporal_correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = np.zeros((num_candidates, total_steps, 5), dtype=np.float32)
    innovation_scale = exploration_std * np.sqrt(
        max(0.0, 1.0 - temporal_correlation**2)
    )
    noise[:, 0] = rng.normal(
        0.0, exploration_std, size=(num_candidates, 5)
    )
    for step in range(1, total_steps):
        noise[:, step] = temporal_correlation * noise[:, step - 1]
        noise[:, step] += rng.normal(
            0.0, innovation_scale, size=(num_candidates, 5)
        )
    plans = np.clip(base_action[None, None, :] + noise, -1.0, 1.0)
    plans[0] = np.broadcast_to(base_action, (total_steps, 5))
    return plans.astype(np.float32)


@torch.inference_mode()
def score_plans(
    model,
    current_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    normalized_candidates: torch.Tensor,
) -> torch.Tensor:
    _, samples, horizon, action_dim = normalized_candidates.shape
    current = current_pixels.unsqueeze(1).expand(
        1, samples, *current_pixels.shape[1:]
    )
    goal = goal_pixels.unsqueeze(1).expand(1, samples, *goal_pixels.shape[1:])
    dummy_action = torch.zeros(
        1,
        samples,
        1,
        action_dim,
        device=normalized_candidates.device,
        dtype=normalized_candidates.dtype,
    )
    info = {"pixels": current, "goal": goal, "action": dummy_action}
    return model.get_cost(info, normalized_candidates)


def update_actor(
    actor: GaussianActor,
    anchor_actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    actor_observation: torch.Tensor,
    target_action: torch.Tensor,
    settings: dict,
) -> float:
    actor.train()
    losses = []
    with torch.no_grad():
        anchor_action = anchor_actor.deterministic_action(actor_observation)
    for _ in range(int(settings["updates_per_plan"])):
        prediction = actor.deterministic_action(actor_observation)
        planner_loss = nn.functional.mse_loss(prediction, target_action)
        anchor_loss = nn.functional.mse_loss(prediction, anchor_action)
        loss = planner_loss + float(settings["bc_anchor_coefficient"]) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), float(settings["max_grad_norm"]))
        optimizer.step()
        losses.append(loss.item())
    actor.eval()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.smoke:
        apply_smoke_config(config)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM planning.")
    device = torch.device(config["device"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    output_dir = config["paths"]["output_dir"]
    log_path = output_dir / "logs" / "planning.jsonl"
    log_path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as output:
        json.dump(json_value(config), output, indent=2)

    actor, actor_checkpoint, observation_mean, observation_scale = load_actor(
        config, device
    )
    anchor_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    adaptation = config["adaptation"]
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(adaptation["learning_rate"]), weight_decay=1e-5
    )
    action_mean, action_scale = load_action_stats(
        config["paths"]["action_dataset"]
    )
    model = swm.wm.utils.load_pretrained(str(config["paths"]["model_dir"]))
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = build_transform(int(config["image_size"]))
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    planner = config["planner"]
    evaluation = config["evaluation"]
    total_plan_steps = int(planner["horizon_blocks"]) * int(
        planner["action_block"]
    )
    if int(planner["execute_steps"]) > total_plan_steps:
        raise ValueError("execute_steps cannot exceed the proposed plan length.")
    episode_results = []
    cumulative_wm_seconds = 0.0
    cumulative_wm_calls = 0

    for episode in range(int(evaluation["episodes"])):
        state, reset_info = env.reset(
            seed=seed + episode, options={"render_goal": True}
        )
        goal_state = np.asarray(reset_info["goal"], dtype=np.float32)
        goal_pixels = image_tensor(reset_info["goal_rendered"], transform, device)
        frames = [np.asarray(env.render()).copy()]
        steps = 0
        planning_round = 0
        success = False
        terminated = False
        truncated = False

        while steps < int(evaluation["max_steps"]) and not (terminated or truncated):
            planning_round += 1
            actor_input = actor_observation_from_env(state, goal_state)
            actor_input = normalize_observations(
                actor_input, observation_mean, observation_scale
            )
            actor_tensor = torch.from_numpy(actor_input).to(device).unsqueeze(0)
            with torch.inference_mode():
                base_action = actor.deterministic_action(actor_tensor)[0].cpu().numpy()
            raw_candidates = sample_actor_guided_plans(
                base_action,
                num_candidates=int(planner["num_candidates"]),
                total_steps=total_plan_steps,
                exploration_std=float(planner["exploration_std"]),
                temporal_correlation=float(planner["temporal_correlation"]),
                rng=rng,
            )
            normalized = (raw_candidates - action_mean) / action_scale
            normalized = normalized.reshape(
                int(planner["num_candidates"]),
                int(planner["horizon_blocks"]),
                int(planner["action_block"]) * 5,
            )
            normalized_tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
            current_pixels = image_tensor(env.render(), transform, device)
            costs, wm_elapsed = synchronized_call(
                lambda: score_plans(
                    model, current_pixels, goal_pixels, normalized_tensor
                ),
                device,
            )
            cumulative_wm_seconds += wm_elapsed
            cumulative_wm_calls += 1
            costs_np = costs[0].float().cpu().numpy()
            best_index = int(np.argmin(costs_np))
            base_cost = float(costs_np[0])
            relative_improvement = (base_cost - float(costs_np[best_index])) / max(
                abs(base_cost), 1e-6
            )
            planner_accepted = (
                best_index != 0
                and relative_improvement
                >= float(adaptation["min_relative_cost_improvement"])
            )
            selected_index = best_index if planner_accepted else 0
            elite_count = min(int(planner["elite_count"]), len(costs_np))
            elite_indices = np.argpartition(costs_np, elite_count - 1)[:elite_count]
            elite_costs = costs_np[elite_indices]
            temperature = max(float(planner["elite_temperature"]), 1e-6)
            elite_weights = np.exp(-(elite_costs - elite_costs.min()) / temperature)
            elite_weights /= elite_weights.sum()
            target_first_action = np.sum(
                raw_candidates[elite_indices, 0] * elite_weights[:, None], axis=0
            )
            max_delta = float(adaptation["max_target_action_delta"])
            target_first_action = base_action + np.clip(
                target_first_action - base_action, -max_delta, max_delta
            )
            adaptation_loss = 0.0
            if bool(adaptation["enabled"]) and planner_accepted:
                adaptation_loss = update_actor(
                    actor,
                    anchor_actor,
                    optimizer,
                    actor_tensor,
                    torch.from_numpy(target_first_action).to(device).unsqueeze(0),
                    adaptation,
                )

            executed = 0
            for action in raw_candidates[
                selected_index, : int(planner["execute_steps"])
            ]:
                state, _, terminated, truncated, info = env.step(action)
                steps += 1
                executed += 1
                success = bool(info.get("success", False))
                frames.append(np.asarray(env.render()).copy())
                if terminated or truncated or steps >= int(evaluation["max_steps"]):
                    break

            record = {
                "episode": episode,
                "planning_round": planning_round,
                "environment_steps": steps,
                "executed_steps": executed,
                "best_candidate": best_index,
                "selected_candidate": selected_index,
                "planner_accepted": planner_accepted,
                "base_cost": base_cost,
                "relative_cost_improvement": relative_improvement,
                "best_cost": float(costs_np.min()),
                "median_cost": float(np.median(costs_np)),
                "worst_cost": float(costs_np.max()),
                "adaptation_loss": adaptation_loss,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "world_model_inference_seconds": wm_elapsed,
                "cumulative_world_model_inference_seconds": cumulative_wm_seconds,
            }
            append_jsonl(log_path, record)
            print(json.dumps(record))

        if episode == 0:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                video_dir / "actor_world_model_mpc_episode_0.mp4",
                frames,
                fps=int(evaluation["fps"]),
            )
        episode_results.append(
            {
                "episode": episode,
                "steps": steps,
                "planning_rounds": planning_round,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    env.close()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapted_checkpoint = dict(actor_checkpoint)
    adapted_checkpoint["actor"] = actor.state_dict()
    adapted_checkpoint["adaptation"] = json_value(adaptation)
    adapted_checkpoint["world_model_inference_seconds"] = cumulative_wm_seconds
    torch.save(adapted_checkpoint, checkpoint_dir / "adapted_actor.pt")
    summary = {
        "episodes": len(episode_results),
        "successes": int(sum(item["success"] for item in episode_results)),
        "success_rate": float(np.mean([item["success"] for item in episode_results])),
        "world_model_inference_seconds": cumulative_wm_seconds,
        "world_model_calls": cumulative_wm_calls,
        "mean_world_model_inference_seconds": cumulative_wm_seconds
        / max(1, cumulative_wm_calls),
        "episode_results": episode_results,
    }
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
