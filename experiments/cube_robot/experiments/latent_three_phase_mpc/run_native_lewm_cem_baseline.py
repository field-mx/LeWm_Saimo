"""Run the paper-native LeWM+CEM planner on the Proprio V2 Cube task.

This is an intentionally isolated baseline: it uses only the frozen LeWM
checkpoint, the package CEMSolver, current RGB, and the fixed goal RGB.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import ogbench
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms


HERE = Path(__file__).resolve().parent


class SingleEnvironmentAdapter:
    """Expose one Gym environment with the vector-space contract of policy.py."""

    def __init__(self, env) -> None:
        self.num_envs = 1
        self.single_action_space = env.action_space
        self.action_space = Box(
            low=np.expand_dims(env.action_space.low, axis=0),
            high=np.expand_dims(env.action_space.high, axis=0),
            dtype=env.action_space.dtype,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native paper LeWM+CEM on the current Cube task."
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "native_lewm_cem_config.yaml"
    )
    parser.add_argument("--run-name", default="paper_native_seed42")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][key] = (
            candidate if candidate.is_absolute() else (HERE / candidate).resolve()
        )
    return config


def task_reset_options(config: dict[str, Any]) -> dict[str, Any]:
    task = config["task"]
    return {
        "render_goal": True,
        "task_info": {
            "task_name": str(task["name"]),
            "init_xyzs": np.asarray([task["init_xyz"]], dtype=np.float64),
            "goal_xyzs": np.asarray([task["goal_xyz"]], dtype=np.float64),
        },
    }


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def fit_action_scaler(path: Path) -> StandardScaler:
    with np.load(path) as dataset:
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    return StandardScaler().fit(actions)


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError("--run-name must be a single directory name.")
    config = load_config(args.config.resolve())
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native LeWM+CEM baseline.")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = config["paths"]["output_dir"] / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "native_lewm_cem_episode_0.mp4"
    metadata_path = output_dir / "summary.json"
    config_path = output_dir / "resolved_config.json"

    model_dir = config["paths"]["model_dir"]
    action_dataset = config["paths"]["action_dataset"]
    for required in (model_dir / "config.json", model_dir / "weights.pt", action_dataset):
        if not required.exists():
            raise FileNotFoundError(f"Required baseline asset missing: {required}")

    device = torch.device(config["device"])
    model = swm.wm.utils.load_pretrained(str(model_dir))
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan = swm.PlanConfig(**config["plan_config"])
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        device=str(device),
        seed=seed,
        **config["cem"],
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan,
        process={"action": fit_action_scaler(action_dataset)},
        transform={
            "pixels": build_transform(int(config["image_size"])),
            "goal": build_transform(int(config["image_size"])),
        },
    )

    env = ogbench.make_env_and_datasets(
        "cube-single-play-v0", env_only=True, terminate_at_goal=False
    )
    env._max_episode_steps = int(config["evaluation"]["max_env_steps"])
    policy.set_env(SingleEnvironmentAdapter(env))

    _, reset_info = env.reset(seed=seed, options=task_reset_options(config))
    goal_image = np.asarray(reset_info["goal_rendered"]).copy()
    current_image = np.asarray(env.render()).copy()
    frames = [current_image]
    imageio.imwrite(output_dir / "goal_rgb.png", goal_image)

    replan_times: list[float] = []
    steps = 0
    success = False
    terminated = False
    truncated = False
    started = time.perf_counter()

    while steps < int(config["evaluation"]["max_env_steps"]):
        # Shape: (environment=1, history=1, height, width, channels).
        # The action history is a required upstream LeWM key. CEM candidates
        # overwrite it before every world-model rollout.
        planner_input = {
            "pixels": current_image[None, None, ...],
            "goal": goal_image[None, None, ...],
            "action": np.zeros((1, 1, env.action_space.shape[0]), dtype=np.float32),
        }
        before = time.perf_counter()
        batched_action = policy.get_action(planner_input)
        elapsed = time.perf_counter() - before
        if steps % plan.plan_len == 0:
            replan_times.append(elapsed)
        action = np.asarray(batched_action[0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        current_image = np.asarray(env.render()).copy()
        frames.append(current_image)
        success = bool(info.get("success", False))
        if success or terminated or truncated:
            break

    imageio.mimsave(video_path, frames, fps=int(config["evaluation"]["fps"]))
    summary = {
        "baseline": "paper-native LeWM + package CEMSolver only",
        "task": config["task"],
        "planner": {
            **config["plan_config"],
            **config["cem"],
            "plan_env_steps": plan.plan_len,
        },
        "evaluation": config["evaluation"],
        "steps_executed": steps,
        "frames_written": len(frames),
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "planner_calls": len(replan_times),
        "planner_seconds": replan_times,
        "planner_seconds_total": float(sum(replan_times)),
        "rollout_seconds": time.perf_counter() - started,
        "video": video_path,
        "goal_rgb": output_dir / "goal_rgb.png",
    }
    with metadata_path.open("w", encoding="utf-8") as output:
        json.dump(json_value(summary), output, indent=2, ensure_ascii=True)
    with config_path.open("w", encoding="utf-8") as output:
        json.dump(json_value(config), output, indent=2, ensure_ascii=True)

    print(json.dumps(json_value(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
