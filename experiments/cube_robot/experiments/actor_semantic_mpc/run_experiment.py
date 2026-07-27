from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
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
from actor_data import (
    actor_observation_from_env,
    load_demonstrations,
    normalize_observations,
)


XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALE = 10.0
ACTION_RANGE = np.array([0.05, 0.05, 0.05, 0.30, 1.0], dtype=np.float32)
PHASE_ORDER = {
    "approach": 0,
    "descend": 1,
    "grasp": 2,
    "lift": 3,
    "transport": 4,
    "place": 5,
    "complete": 6,
}


@dataclass
class SemanticState:
    effector: np.ndarray
    cube: np.ndarray
    goal: np.ndarray
    effector_yaw: float
    cube_yaw: float
    gripper_opening: float
    contact: float

    def clone(self) -> "SemanticState":
        return SemanticState(
            effector=self.effector.copy(),
            cube=self.cube.copy(),
            goal=self.goal.copy(),
            effector_yaw=float(self.effector_yaw),
            cube_yaw=float(self.cube_yaw),
            gripper_opening=float(self.gripper_opening),
            contact=float(self.contact),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-aware LeWM MPC with real semantic-gated Actor updates."
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--stage", choices=("train", "evaluate", "all"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
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


def validate_task_config(config: dict) -> None:
    task = config.get("task")
    if not task or not bool(task.get("validate_workspace", True)):
        return
    if "init_xyz" not in task or "goal_xyz" not in task:
        return

    bounds = np.asarray(
        task.get(
            "workspace_bounds",
            [[0.30, 0.55], [-0.30, 0.30], [0.0, 0.20]],
        ),
        dtype=np.float64,
    )
    if bounds.shape != (3, 2):
        raise ValueError("task.workspace_bounds must have shape [3, 2].")

    for name in ("init_xyz", "goal_xyz"):
        position = np.asarray(task[name], dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"task.{name} must contain exactly three coordinates.")
        outside = (position < bounds[:, 0]) | (position > bounds[:, 1])
        if np.any(outside):
            raise ValueError(
                f"task.{name}={position.tolist()} is outside the configured "
                f"workspace bounds {bounds.tolist()}. Set valid coordinates or "
                "set task.validate_workspace=false to override."
            )


def apply_smoke_config(config: dict) -> None:
    config["paths"]["output_dir"] = config["paths"]["output_dir"] / "smoke"
    config["planner"].update(num_candidates=8, horizon_blocks=2)
    config["adaptation"].update(
        training_episodes=1,
        max_steps=8,
        warmup_verified_samples=1,
        anchor_pool_size=128,
        anchor_batch_size=8,
        verified_batch_size=8,
    )
    config["evaluation"].update(episodes=1, max_steps=8)


def json_value(value):
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


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(json_value(record)) + "\n")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(json_value(value), output, indent=2)


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


def load_actor(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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


def load_anchor_pool(
    dataset_path: Path,
    checkpoint: dict,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    observations, _, _ = load_demonstrations(
        dataset_path,
        max_goal_offset=int(checkpoint["max_goal_offset"]),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(observations), size=min(count, len(observations)), replace=False)
    pool = normalize_observations(
        observations[indices],
        np.asarray(checkpoint["observation_mean"], dtype=np.float32),
        np.asarray(checkpoint["observation_scale"], dtype=np.float32),
    )
    return pool


def physical_position(scaled: np.ndarray) -> np.ndarray:
    return np.asarray(scaled, dtype=np.float32) / XYZ_SCALE + XYZ_CENTER


def angle_from_cos_sin(values: np.ndarray) -> float:
    return float(np.arctan2(float(values[1]), float(values[0])))


def semantic_state(state: np.ndarray, goal_state: np.ndarray) -> SemanticState:
    state = np.asarray(state, dtype=np.float32)
    goal_state = np.asarray(goal_state, dtype=np.float32)
    return SemanticState(
        effector=physical_position(state[12:15]),
        cube=physical_position(state[19:22]),
        goal=physical_position(goal_state[19:22]),
        effector_yaw=angle_from_cos_sin(state[15:17]),
        cube_yaw=angle_from_cos_sin(state[26:28]),
        gripper_opening=float(np.clip(state[17] / 3.0, 0.0, 1.0)),
        contact=float(np.clip(state[18], 0.0, 1.0)),
    )


def shortest_angle(target: float, current: float) -> float:
    return float((target - current + np.pi) % (2.0 * np.pi) - np.pi)


def detect_phase(value: SemanticState, settings: dict) -> str:
    goal_xyz = float(np.linalg.norm(value.cube - value.goal))
    if goal_xyz <= float(settings["goal_xyz_m"]):
        return "complete"

    xy_to_cube = float(np.linalg.norm(value.effector[:2] - value.cube[:2]))
    xyz_to_cube = float(np.linalg.norm(value.effector - value.cube))
    has_contact = value.contact >= float(settings["contact_threshold"])
    goal_xy = float(np.linalg.norm(value.cube[:2] - value.goal[:2]))

    if not has_contact:
        if xy_to_cube > float(settings["xy_alignment_m"]):
            return "approach"
        if xyz_to_cube > float(settings["xyz_alignment_m"]):
            return "descend"
        return "grasp"
    # Once XY is aligned with the destination, placing takes priority over the
    # transport-height guard. Otherwise descending immediately switches back to
    # lift and the controller oscillates above the goal.
    if goal_xy <= float(settings["goal_xy_m"]):
        return "place"
    if value.cube[2] < float(settings["transport_height_m"]):
        return "lift"
    return "transport"


def phase_target(value: SemanticState, phase: str, settings: dict) -> np.ndarray:
    if phase == "approach":
        return value.cube + np.array(
            [0.0, 0.0, float(settings["above_offset_m"])], dtype=np.float32
        )
    if phase in ("descend", "grasp"):
        return value.cube.copy()
    if phase == "lift":
        return np.array(
            [
                value.cube[0],
                value.cube[1],
                max(
                    float(settings["transport_height_m"])
                    + float(settings["above_offset_m"]) * 0.25,
                    value.cube[2] + 0.08,
                ),
            ],
            dtype=np.float32,
        )
    if phase == "transport":
        return value.goal + np.array(
            [0.0, 0.0, float(settings["transport_height_m"])], dtype=np.float32
        )
    return value.goal.copy()


def controller_action(
    value: SemanticState,
    phase: str,
    base_action: np.ndarray,
    settings: dict,
) -> np.ndarray:
    if phase == "complete":
        return np.zeros(5, dtype=np.float32)

    target = phase_target(value, phase, settings)
    position_action = np.clip((target - value.effector) / ACTION_RANGE[:3], -1.0, 1.0)
    yaw_action = np.clip(
        shortest_angle(value.cube_yaw, value.effector_yaw) / ACTION_RANGE[3],
        -1.0,
        1.0,
    )
    action = np.zeros(5, dtype=np.float32)
    action[:3] = position_action
    action[3] = 0.7 * yaw_action + 0.3 * float(base_action[3])
    action[4] = -1.0 if phase in ("approach", "descend") else 1.0

    if phase == "grasp":
        action[:3] *= 0.35
    elif phase == "lift":
        action[:2] *= 0.35
    elif phase == "place":
        action[:2] *= 0.5
    return np.clip(action, -1.0, 1.0)


def approximate_step(
    value: SemanticState,
    action: np.ndarray,
    settings: dict,
) -> SemanticState:
    next_value = value.clone()
    delta = np.asarray(action[:3], dtype=np.float32) * ACTION_RANGE[:3]
    next_value.effector = next_value.effector + delta
    next_value.effector_yaw += float(action[3]) * float(ACTION_RANGE[3])
    next_value.gripper_opening = float(
        np.clip(next_value.gripper_opening + float(action[4]), 0.0, 1.0)
    )

    xyz_to_cube = float(np.linalg.norm(next_value.effector - next_value.cube))
    if (
        xyz_to_cube <= float(settings["xyz_alignment_m"]) * 1.25
        and next_value.gripper_opening >= 0.55
    ):
        next_value.contact = 1.0
    elif float(action[4]) < -0.2:
        next_value.contact = 0.0

    if next_value.contact >= float(settings["contact_threshold"]):
        next_value.cube = next_value.cube + delta
    return next_value


def build_template(
    initial: SemanticState,
    base_action: np.ndarray,
    total_steps: int,
    settings: dict,
) -> tuple[np.ndarray, list[str]]:
    value = initial.clone()
    actions = []
    phases = []
    for _ in range(total_steps):
        phase = detect_phase(value, settings)
        action = controller_action(value, phase, base_action, settings)
        actions.append(action)
        phases.append(phase)
        value = approximate_step(value, action, settings)
    return np.asarray(actions, dtype=np.float32), phases


def correlated_noise(
    count: int,
    steps: int,
    std: float,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = np.zeros((count, steps, 5), dtype=np.float32)
    scales = np.array([1.0, 1.0, 0.8, 0.7, 0.35], dtype=np.float32) * std
    innovation = scales * np.sqrt(max(0.0, 1.0 - correlation**2))
    noise[:, 0] = rng.normal(0.0, scales, size=(count, 5))
    for step in range(1, steps):
        noise[:, step] = correlation * noise[:, step - 1]
        noise[:, step] += rng.normal(0.0, innovation, size=(count, 5))
    return noise


def sample_phase_aware_plans(
    base_action: np.ndarray,
    value: SemanticState,
    *,
    num_candidates: int,
    total_steps: int,
    planner: dict,
    semantics: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    template, phases = build_template(value, base_action, total_steps, semantics)
    plans = template[None].repeat(num_candidates, axis=0)
    if num_candidates > 2:
        plans[2:] += correlated_noise(
            num_candidates - 2,
            total_steps,
            float(planner["exploration_std"]),
            float(planner["temporal_correlation"]),
            rng,
        )

    # Candidate zero is the unchanged BC baseline; candidate one is deterministic.
    plans[0] = np.broadcast_to(base_action, (total_steps, 5))
    plans[1] = template

    # Preserve open/close intent so noise cannot create impossible phase switches.
    for step, phase in enumerate(phases):
        if phase in ("approach", "descend"):
            plans[1:, step, 4] = np.minimum(plans[1:, step, 4], -0.35)
        elif phase != "complete":
            plans[1:, step, 4] = np.maximum(plans[1:, step, 4], 0.35)

    max_delta = float(planner["max_first_action_delta"])
    plans[1:, 0] = base_action + np.clip(
        plans[1:, 0] - base_action, -max_delta, max_delta
    )
    return np.clip(plans, -1.0, 1.0).astype(np.float32), template, phases


def sample_direct_goal_plans(
    base_action: np.ndarray,
    *,
    num_candidates: int,
    total_steps: int,
    planner: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    template = np.broadcast_to(base_action, (total_steps, 5)).copy()
    plans = template[None].repeat(num_candidates, axis=0)
    if num_candidates > 1:
        plans[1:] += correlated_noise(
            num_candidates - 1,
            total_steps,
            float(planner["exploration_std"]),
            float(planner["temporal_correlation"]),
            rng,
        )
    return (
        np.clip(plans, -1.0, 1.0).astype(np.float32),
        template,
        ["direct_goal"] * total_steps,
    )


@torch.inference_mode()
def score_plans(
    model,
    current_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    normalized_candidates: torch.Tensor,
) -> torch.Tensor:
    _, samples, _, action_dim = normalized_candidates.shape
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


def phase_loss(value: SemanticState, phase: str, settings: dict) -> float:
    target = phase_target(value, phase, settings)
    if phase in ("approach", "descend"):
        return float(np.linalg.norm(value.effector - target))
    if phase == "grasp":
        distance = float(np.linalg.norm(value.effector - value.cube))
        return distance + 0.04 * (1.0 - value.contact) + 0.02 * (
            1.0 - value.gripper_opening
        )
    if phase == "lift":
        target_height = float(settings["transport_height_m"])
        return max(0.0, target_height - float(value.cube[2])) + 0.05 * (
            1.0 - value.contact
        )
    if phase == "transport":
        return float(np.linalg.norm(value.cube[:2] - value.goal[:2])) + 0.05 * (
            1.0 - value.contact
        )
    if phase == "place":
        return float(np.linalg.norm(value.cube - value.goal))
    return 0.0


def real_semantic_progress(
    before: SemanticState,
    after: SemanticState,
    phase_before: str,
    settings: dict,
) -> tuple[bool, float, str]:
    phase_after = detect_phase(after, settings)
    if phase_after == "complete":
        return True, 1.0, phase_after

    before_rank = PHASE_ORDER[phase_before]
    after_rank = PHASE_ORDER[phase_after]
    if after_rank > before_rank:
        return True, float(after_rank - before_rank), phase_after
    if after_rank < before_rank:
        return False, float(after_rank - before_rank), phase_after

    before_loss = phase_loss(before, phase_before, settings)
    after_loss = phase_loss(after, phase_before, settings)
    progress = before_loss - after_loss
    threshold = float(settings["minimum_progress_m"])

    if phase_before == "grasp":
        opening_progress = after.gripper_opening - before.gripper_opening
        contact_progress = after.contact - before.contact
        accepted = (
            contact_progress > 0.02
            or (
                opening_progress >= float(settings["minimum_gripper_progress"])
                and np.linalg.norm(after.effector - after.cube)
                <= np.linalg.norm(before.effector - before.cube)
                + float(settings["maximum_regression_m"])
            )
        )
        return bool(accepted), float(max(progress, contact_progress)), phase_after

    if phase_before in ("lift", "transport"):
        if after.contact < float(settings["contact_threshold"]) * 0.6:
            return False, float(progress), phase_after
    return bool(progress >= threshold), float(progress), phase_after


def direct_goal_progress(
    before: SemanticState,
    after: SemanticState,
    settings: dict,
) -> tuple[bool, float, str]:
    before_distance = float(np.linalg.norm(before.cube - before.goal))
    after_distance = float(np.linalg.norm(after.cube - after.goal))
    progress = before_distance - after_distance
    phase_after = (
        "complete"
        if after_distance <= float(settings["goal_xyz_m"])
        else "direct_goal"
    )
    return (
        bool(progress >= float(settings["minimum_progress_m"])),
        float(progress),
        phase_after,
    )


def update_actor_from_verified(
    actor: GaussianActor,
    anchor_actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    replay: deque,
    anchor_pool: torch.Tensor,
    settings: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    actor.train()
    losses = []
    for _ in range(int(settings["updates_per_accept"])):
        verified_count = min(int(settings["verified_batch_size"]), len(replay))
        verified_indices = rng.choice(len(replay), size=verified_count, replace=False)
        verified_observations = torch.stack(
            [replay[int(index)][0] for index in verified_indices]
        ).to(device)
        verified_actions = torch.stack(
            [replay[int(index)][1] for index in verified_indices]
        ).to(device)

        anchor_count = min(int(settings["anchor_batch_size"]), len(anchor_pool))
        anchor_indices = torch.from_numpy(
            rng.choice(len(anchor_pool), size=anchor_count, replace=False)
        ).long()
        anchor_observations = anchor_pool[anchor_indices].to(device)
        with torch.no_grad():
            anchor_actions = anchor_actor.deterministic_action(anchor_observations)

        verified_loss = nn.functional.mse_loss(
            actor.deterministic_action(verified_observations), verified_actions
        )
        anchor_loss = nn.functional.mse_loss(
            actor.deterministic_action(anchor_observations), anchor_actions
        )
        loss = verified_loss + float(settings["bc_anchor_coefficient"]) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings["max_grad_norm"])
        )
        optimizer.step()
        losses.append(float(loss.item()))
    actor.eval()
    return float(np.mean(losses))


def run_episodes(
    *,
    config: dict,
    actor: GaussianActor,
    anchor_actor: GaussianActor,
    actor_checkpoint: dict,
    observation_mean: np.ndarray,
    observation_scale: np.ndarray,
    model,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    transform,
    device: torch.device,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    replay: deque | None,
    anchor_pool: torch.Tensor | None,
    log_path: Path,
    video_path: Path | None,
) -> dict:
    planner = config["planner"]
    semantics = config["semantics"]
    settings = config["adaptation"] if training else config["evaluation"]
    episodes = (
        int(config["adaptation"]["training_episodes"])
        if training
        else int(config["evaluation"]["episodes"])
    )
    seed_offset = int(settings["seed_offset"])
    max_steps = int(settings["max_steps"])
    total_steps_per_plan = int(planner["horizon_blocks"]) * int(
        planner["action_block"]
    )
    control_mode = str(planner.get("control_mode", "phase_aware"))
    if control_mode not in ("phase_aware", "direct_goal"):
        raise ValueError(
            "planner.control_mode must be 'phase_aware' or 'direct_goal'."
        )
    rng = np.random.default_rng(
        int(config["seed"]) + seed_offset + (0 if training else 100000)
    )
    reset_options = {"render_goal": True}
    task = config.get("task")
    if task:
        if "task_id" in task:
            reset_options["task_id"] = int(task["task_id"])
        elif "init_xyz" in task and "goal_xyz" in task:
            reset_options["task_info"] = {
                "task_name": str(task.get("name", "configured_cube_task")),
                "init_xyzs": np.asarray([task["init_xyz"]], dtype=np.float64),
                "goal_xyzs": np.asarray([task["goal_xyz"]], dtype=np.float64),
            }
        else:
            raise ValueError(
                "task must contain task_id or both init_xyz and goal_xyz."
            )
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    episode_results = []
    phase_counts = Counter()
    real_accept_counts = Counter()
    model_accept_count = 0
    verified_count = 0
    wm_seconds = 0.0
    wm_calls = 0
    log_path.unlink(missing_ok=True)

    for episode in range(episodes):
        state, reset_info = env.reset(
            seed=int(config["seed"]) + seed_offset + episode,
            options=reset_options,
        )
        goal_state = np.asarray(reset_info["goal"], dtype=np.float32)
        initial_semantics = semantic_state(state, goal_state)
        goal_pixels = image_tensor(reset_info["goal_rendered"], transform, device)
        frames = [np.asarray(env.render()).copy()]
        steps = 0
        success = False
        terminated = False
        truncated = False

        while steps < max_steps and not (terminated or truncated):
            before = semantic_state(state, goal_state)
            phase_before = (
                detect_phase(before, semantics)
                if control_mode == "phase_aware"
                else "direct_goal"
            )
            phase_counts[phase_before] += 1
            actor_input = actor_observation_from_env(state, goal_state)
            actor_input = normalize_observations(
                actor_input, observation_mean, observation_scale
            )
            actor_tensor = torch.from_numpy(actor_input).to(device).unsqueeze(0)
            with torch.inference_mode():
                base_action = actor.deterministic_action(actor_tensor)[0].cpu().numpy()

            if control_mode == "phase_aware":
                candidates, template, template_phases = sample_phase_aware_plans(
                    base_action,
                    before,
                    num_candidates=int(planner["num_candidates"]),
                    total_steps=total_steps_per_plan,
                    planner=planner,
                    semantics=semantics,
                    rng=rng,
                )
            else:
                candidates, template, template_phases = sample_direct_goal_plans(
                    base_action,
                    num_candidates=int(planner["num_candidates"]),
                    total_steps=total_steps_per_plan,
                    planner=planner,
                    rng=rng,
                )
            normalized = (candidates - action_mean) / action_scale
            normalized = normalized.reshape(
                int(planner["num_candidates"]),
                int(planner["horizon_blocks"]),
                int(planner["action_block"]) * 5,
            )
            normalized_tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
            current_pixels = image_tensor(env.render(), transform, device)
            costs, elapsed = synchronized_call(
                lambda: score_plans(
                    model, current_pixels, goal_pixels, normalized_tensor
                ),
                device,
            )
            wm_seconds += elapsed
            wm_calls += 1
            costs_np = costs[0].float().cpu().numpy()
            best_index = int(np.argmin(costs_np))
            base_cost = float(costs_np[0])
            best_cost = float(costs_np[best_index])
            relative_improvement = (base_cost - best_cost) / max(abs(base_cost), 1e-6)
            planner_accepted = (
                best_index != 0
                and relative_improvement >= float(planner["min_model_improvement"])
            )
            selected_index = best_index if planner_accepted else 0
            if planner_accepted:
                model_accept_count += 1

            executed_action = candidates[selected_index, 0].copy()
            next_state, _, terminated, truncated, info = env.step(executed_action)
            steps += 1
            success = bool(info.get("success", False))
            frames.append(np.asarray(env.render()).copy())
            after = semantic_state(next_state, goal_state)
            if control_mode == "phase_aware":
                real_accepted, semantic_progress, phase_after = real_semantic_progress(
                    before, after, phase_before, semantics
                )
            else:
                real_accepted, semantic_progress, phase_after = direct_goal_progress(
                    before, after, semantics
                )
            if real_accepted:
                real_accept_counts[phase_before] += 1

            adaptation_loss = 0.0
            learned = False
            if (
                training
                and bool(config["adaptation"]["enabled"])
                and planner_accepted
                and real_accepted
                and selected_index != 0
            ):
                replay.append(
                    (
                        torch.from_numpy(actor_input.copy()),
                        torch.from_numpy(executed_action.copy()),
                    )
                )
                verified_count += 1
                if len(replay) >= int(
                    config["adaptation"]["warmup_verified_samples"]
                ):
                    adaptation_loss = update_actor_from_verified(
                        actor,
                        anchor_actor,
                        optimizer,
                        replay,
                        anchor_pool,
                        config["adaptation"],
                        rng,
                        device,
                    )
                    learned = True

            record = {
                "stage": "train" if training else "evaluate",
                "episode": episode,
                "environment_step": steps,
                "phase_before": phase_before,
                "phase_after": phase_after,
                "template_first_phases": template_phases[:5],
                "planner_accepted": planner_accepted,
                "real_semantic_accepted": real_accepted,
                "learned": learned,
                "base_cost": base_cost,
                "best_cost": best_cost,
                "relative_model_improvement": relative_improvement,
                "semantic_progress": semantic_progress,
                "selected_candidate": selected_index,
                "base_action": base_action,
                "template_action": template[0],
                "executed_action": executed_action,
                "effector_cube_distance_before": float(
                    np.linalg.norm(before.effector - before.cube)
                ),
                "effector_cube_distance_after": float(
                    np.linalg.norm(after.effector - after.cube)
                ),
                "cube_goal_distance_before": float(
                    np.linalg.norm(before.cube - before.goal)
                ),
                "cube_goal_distance_after": float(
                    np.linalg.norm(after.cube - after.goal)
                ),
                "cube_height_before": float(before.cube[2]),
                "cube_height_after": float(after.cube[2]),
                "gripper_opening_before": before.gripper_opening,
                "gripper_opening_after": after.gripper_opening,
                "contact_before": before.contact,
                "contact_after": after.contact,
                "adaptation_loss": adaptation_loss,
                "replay_size": 0 if replay is None else len(replay),
                "success": success,
                "world_model_inference_seconds": elapsed,
                "cumulative_world_model_inference_seconds": wm_seconds,
            }
            append_jsonl(log_path, record)
            state = next_state

        if video_path is not None and episode == 0:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=int(config["evaluation"]["fps"]))

        result = {
            "episode": episode,
            "steps": steps,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "cube_start_m": initial_semantics.cube.tolist(),
            "cube_goal_m": initial_semantics.goal.tolist(),
            "final_phase": (
                detect_phase(semantic_state(state, goal_state), semantics)
                if control_mode == "phase_aware"
                else ("complete" if success else "direct_goal")
            ),
        }
        episode_results.append(result)
        print(json.dumps(result))

    env.close()
    return {
        "stage": "train" if training else "evaluate",
        "episodes": episodes,
        "successes": int(sum(item["success"] for item in episode_results)),
        "success_rate": float(np.mean([item["success"] for item in episode_results])),
        "model_accepted_actions": model_accept_count,
        "real_semantic_accept_counts": dict(real_accept_counts),
        "verified_training_actions": verified_count,
        "phase_counts": dict(phase_counts),
        "world_model_inference_seconds": wm_seconds,
        "world_model_calls": wm_calls,
        "mean_world_model_inference_seconds": wm_seconds / max(1, wm_calls),
        "episode_results": episode_results,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    validate_task_config(config)
    if args.smoke:
        apply_smoke_config(config)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM planning.")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    output_dir = config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)

    actor, actor_checkpoint, observation_mean, observation_scale = load_actor(
        config["paths"]["actor_checkpoint"], device
    )
    anchor_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    action_mean, action_scale = load_action_stats(config["paths"]["action_dataset"])
    model = swm.wm.utils.load_pretrained(str(config["paths"]["model_dir"]))
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = build_transform(int(config["image_size"]))

    checkpoint_path = output_dir / "checkpoints" / "semantic_actor.pt"
    if args.stage in ("train", "all"):
        anchor_pool_np = load_anchor_pool(
            config["paths"]["action_dataset"],
            actor_checkpoint,
            count=int(config["adaptation"]["anchor_pool_size"]),
            seed=seed,
        )
        anchor_pool = torch.from_numpy(anchor_pool_np)
        replay = deque(maxlen=int(config["adaptation"]["replay_capacity"]))
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=float(config["adaptation"]["learning_rate"]),
            weight_decay=1e-5,
        )
        training_summary = run_episodes(
            config=config,
            actor=actor,
            anchor_actor=anchor_actor,
            actor_checkpoint=actor_checkpoint,
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            model=model,
            action_mean=action_mean,
            action_scale=action_scale,
            transform=transform,
            device=device,
            training=True,
            optimizer=optimizer,
            replay=replay,
            anchor_pool=anchor_pool,
            log_path=output_dir / "logs" / "training.jsonl",
            video_path=output_dir
            / "videos"
            / "semantic_mpc_training_episode_0.mp4",
        )
        save_json(output_dir / "training_summary.json", training_summary)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trained_checkpoint = dict(actor_checkpoint)
        trained_checkpoint["actor"] = actor.state_dict()
        trained_checkpoint["semantic_adaptation"] = json_value(config["adaptation"])
        trained_checkpoint["training_summary"] = training_summary
        torch.save(trained_checkpoint, checkpoint_path)

    if args.stage in ("evaluate", "all"):
        if args.stage == "evaluate":
            evaluation_checkpoint = config["paths"].get(
                "evaluation_actor_checkpoint", checkpoint_path
            )
            if not evaluation_checkpoint.exists():
                raise FileNotFoundError(
                    "Evaluation Actor checkpoint not found: "
                    f"{evaluation_checkpoint}"
                )
            trained_checkpoint = torch.load(
                evaluation_checkpoint, map_location="cpu", weights_only=False
            )
            actor.load_state_dict(trained_checkpoint["actor"])
            actor.eval()

        evaluation_summary = run_episodes(
            config=config,
            actor=actor,
            anchor_actor=anchor_actor,
            actor_checkpoint=actor_checkpoint,
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            model=model,
            action_mean=action_mean,
            action_scale=action_scale,
            transform=transform,
            device=device,
            training=False,
            optimizer=None,
            replay=None,
            anchor_pool=None,
            log_path=output_dir / "logs" / "evaluation.jsonl",
            video_path=output_dir / "videos" / "semantic_mpc_episode_0.mp4",
        )
        save_json(output_dir / "evaluation.json", evaluation_summary)
        print(json.dumps(evaluation_summary, indent=2))


if __name__ == "__main__":
    main()
