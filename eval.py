import os

os.environ["MUJOCO_GL"] = "egl"

from collections import defaultdict
from copy import deepcopy
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.plot import save_panel_videos
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = np.asarray(dataset.get_col_data(col_name))
    episode_ids, first_indices = np.unique(episode_idx, return_index=True)
    order = np.argsort(first_indices)
    episode_ids = episode_ids[order]
    first_indices = first_indices[order]
    lengths = np.diff(np.append(first_indices, len(episode_idx)))
    length_by_episode = dict(zip(episode_ids.tolist(), lengths.tolist()))
    return np.asarray(
        [length_by_episode[episode_id] for episode_id in episodes],
        dtype=np.int64,
    )


def get_episode_bounds(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = np.asarray(dataset.get_col_data(col_name))
    episode_ids, first_indices = np.unique(episode_idx, return_index=True)
    order = np.argsort(first_indices)
    episode_ids = episode_ids[order]
    first_indices = first_indices[order]
    lengths = np.diff(np.append(first_indices, len(episode_idx)))
    first_by_episode = dict(zip(episode_ids.tolist(), first_indices.tolist()))
    length_by_episode = dict(zip(episode_ids.tolist(), lengths.tolist()))
    first_rows = np.asarray(
        [first_by_episode[episode_id] for episode_id in episodes],
        dtype=np.int64,
    )
    episode_lengths = np.asarray(
        [length_by_episode[episode_id] for episode_id in episodes],
        dtype=np.int64,
    )
    return first_rows, episode_lengths


def find_contact_start(states, distance_threshold):
    states = np.asarray(states)
    agent_pos = states[:, :2]
    block_pos = states[:, 2:4]
    distances = np.linalg.norm(agent_pos - block_pos, axis=1)
    contact_indices = np.nonzero(distances <= float(distance_threshold))[0]
    if len(contact_indices) == 0:
        return None
    return int(contact_indices[0])


def angle_distance(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 2 * np.pi - diff)


def expert_state_distance(
    current_state,
    expert_states,
    agent_weight=1.0,
    agent_relative_weight=1.0,
    block_weight=1.0,
    angle_weight=20.0,
):
    current_state = np.asarray(current_state)
    expert_states = np.asarray(expert_states)
    agent_dist = np.linalg.norm(expert_states[:, :2] - current_state[:2], axis=1)
    block_dist = np.linalg.norm(expert_states[:, 2:4] - current_state[2:4], axis=1)
    current_agent_relative = current_state[:2] - current_state[2:4]
    expert_agent_relative = expert_states[:, :2] - expert_states[:, 2:4]
    agent_relative_dist = np.linalg.norm(
        expert_agent_relative - current_agent_relative, axis=1
    )
    angle_dist = angle_distance(expert_states[:, 4], current_state[4])
    return (
        float(agent_weight) * agent_dist
        + float(agent_relative_weight) * agent_relative_dist
        + float(block_weight) * block_dist
        + float(angle_weight) * angle_dist
    )


def find_nearest_expert_offset(
    current_state,
    trajectory,
    search_start,
    search_end,
    agent_weight=1.0,
    agent_relative_weight=1.0,
    block_weight=1.0,
    angle_weight=20.0,
):
    states = trajectory["state"]
    search_start = int(np.clip(search_start, 0, len(states) - 1))
    search_end = int(np.clip(search_end, search_start, len(states) - 1))
    candidates = states[search_start : search_end + 1]
    distances = expert_state_distance(
        current_state,
        candidates,
        agent_weight=agent_weight,
        agent_relative_weight=agent_relative_weight,
        block_weight=block_weight,
        angle_weight=angle_weight,
    )
    return search_start + int(np.argmin(distances))


def block_goal_errors(state, goal_position, goal_angle):
    state = np.asarray(state)
    position_error = float(
        np.linalg.norm(state[2:4] - np.asarray(goal_position, dtype=np.float64))
    )
    rotation_error = float(angle_distance(state[4], float(goal_angle)))
    return position_error, rotation_error


def find_task_end(
    states,
    start_offset,
    goal_position,
    goal_angle,
    position_threshold,
    angle_threshold,
):
    states = np.asarray(states)
    if int(start_offset) >= len(states):
        return None

    candidates = states[int(start_offset) :]
    position_errors = np.linalg.norm(
        candidates[:, 2:4] - np.asarray(goal_position, dtype=np.float64), axis=1
    )
    rotation_errors = angle_distance(candidates[:, 4], float(goal_angle))
    reached = (position_errors < float(position_threshold)) & (
        rotation_errors < float(angle_threshold)
    )
    reached_indices = np.nonzero(reached)[0]
    if len(reached_indices) == 0:
        return None
    return int(start_offset) + int(reached_indices[0])


def subgoal_errors(current_state, target_state):
    current_state = np.asarray(current_state)
    target_state = np.asarray(target_state)
    block_position_error = float(
        np.linalg.norm(current_state[2:4] - target_state[2:4])
    )
    block_angle_error = float(angle_distance(current_state[4], target_state[4]))
    current_agent_relative = current_state[:2] - current_state[2:4]
    target_agent_relative = target_state[:2] - target_state[2:4]
    agent_relative_error = float(
        np.linalg.norm(current_agent_relative - target_agent_relative)
    )
    return block_position_error, block_angle_error, agent_relative_error


def normalized_subgoal_error(
    current_state,
    target_state,
    block_position_threshold,
    block_angle_threshold,
    agent_relative_threshold,
):
    block_error, angle_error, agent_relative_error = subgoal_errors(
        current_state, target_state
    )
    score = (
        block_error / float(block_position_threshold)
        + angle_error / float(block_angle_threshold)
        + agent_relative_error / float(agent_relative_threshold)
    )
    success = (
        block_error < float(block_position_threshold)
        and angle_error < float(block_angle_threshold)
        and agent_relative_error < float(agent_relative_threshold)
    )
    return success, score, (block_error, angle_error, agent_relative_error)


def normalized_final_error(
    current_state,
    goal_position,
    goal_angle,
    position_threshold,
    angle_threshold,
):
    position_error, rotation_error = block_goal_errors(
        current_state, goal_position, goal_angle
    )
    score = (
        position_error / float(position_threshold)
        + rotation_error / float(angle_threshold)
    )
    success = position_error < float(position_threshold) and rotation_error < float(
        angle_threshold
    )
    return success, score, (position_error, rotation_error)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def build_subgoal_offsets(final_offset, interval):
    final_offset = int(final_offset)
    interval = int(interval)
    if final_offset <= 0:
        raise ValueError("The selected expert trajectory must contain at least 2 frames.")
    if interval <= 0:
        raise ValueError("eval.goal_offset_steps must be greater than zero.")

    offsets = list(range(interval, final_offset + 1, interval))
    if not offsets or offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def load_eval_trajectories(dataset, episodes_idx, start_steps, final_offset):
    chunks = dataset.load_chunk(
        np.asarray(episodes_idx),
        np.asarray(start_steps),
        np.asarray(start_steps) + int(final_offset) + 1,
    )

    trajectories = []
    for chunk in chunks:
        trajectory = {}
        for col in dataset.column_names:
            if col not in chunk or col.startswith("goal"):
                continue
            value = chunk[col]
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            value = to_numpy(value)
            if col.startswith("pixels"):
                value = np.transpose(value, (0, 2, 3, 1))
            trajectory[col] = value
        trajectories.append(trajectory)
    return trajectories


def state_at_offsets(trajectories, offsets, goal=False):
    state = {}
    for col in trajectories[0]:
        values = np.stack(
            [
                trajectory[col][offset]
                for trajectory, offset in zip(trajectories, offsets)
            ]
        )
        if goal:
            key = "goal" if col == "pixels" else f"goal_{col}"
        else:
            key = col
        state[key] = values
    return state


def update_world_infos(world, values, goal=False):
    shape_prefix = world.infos["pixels"].shape[:2]
    for key, value in values.items():
        if goal or key in world.infos:
            world.infos[key] = np.broadcast_to(
                value[:, None, ...], shape_prefix + value.shape[1:]
            ).copy()


def apply_callables(env, callables, values, goal_only=False):
    for spec in callables or []:
        if goal_only and "goal" not in spec["method"]:
            continue
        if not hasattr(env, spec["method"]):
            continue

        prepared = {}
        for name, data in spec.get("args", {}).items():
            if data.get("in_dataset", True):
                key = data.get("value")
                if key in values:
                    prepared[name] = deepcopy(values[key])
            else:
                prepared[name] = data.get("value")
        getattr(env, spec["method"])(**prepared)


def evaluate_sequential_subgoals(
    world,
    dataset,
    episodes_idx,
    start_steps,
    subgoal_interval,
    final_offset,
    segment_budget,
    max_subgoal_retries,
    callables,
    video_dir,
    realign_interval,
    realign_search_window,
    agent_align_weight,
    agent_relative_align_weight,
    block_align_weight,
    angle_align_weight,
    block_goal_position,
    block_goal_angle,
    subgoal_block_position_threshold,
    subgoal_block_angle_threshold,
    subgoal_agent_relative_threshold,
    final_block_position_threshold,
    final_block_angle_threshold,
    stagnation_limit_steps,
    stagnation_min_improvement,
):
    reference_offsets = build_subgoal_offsets(final_offset, subgoal_interval)
    trajectories = load_eval_trajectories(
        dataset, episodes_idx, start_steps, final_offset
    )
    num_envs = len(trajectories)
    num_subgoals = len(reference_offsets)
    init_state = state_at_offsets(trajectories, [0] * num_envs)

    target_indices = np.zeros(num_envs, dtype=np.int64)
    current_offsets = np.full(num_envs, reference_offsets[0], dtype=np.int64)
    confirmed_offsets = np.zeros(num_envs, dtype=np.int64)
    matched_offsets = np.zeros(num_envs, dtype=np.int64)
    goal_state = state_at_offsets(trajectories, current_offsets, goal=True)

    world.reset(seed=init_state.get("seed"))
    for env_idx, env in enumerate(world.envs.envs):
        values = {
            **{key: value[env_idx] for key, value in init_state.items()},
            **{key: value[env_idx] for key, value in goal_state.items()},
        }
        apply_callables(env.unwrapped, callables, values)

    update_world_infos(world, init_state)
    update_world_infos(world, goal_state, goal=True)
    world.infos["_needs_flush"] = np.ones(num_envs, dtype=bool)

    active = np.ones(num_envs, dtype=bool)
    final_successes = np.zeros(num_envs, dtype=bool)
    subgoal_successes = np.zeros((num_envs, num_subgoals), dtype=bool)
    subgoal_attempts = np.zeros((num_envs, num_subgoals), dtype=np.int64)
    subgoal_attempts[:, 0] = 1
    segment_steps = np.zeros(num_envs, dtype=np.int64)
    retry_counts = np.zeros(num_envs, dtype=np.int64)
    rollout_steps = np.zeros(num_envs, dtype=np.int64)
    stagnation_steps = np.zeros(num_envs, dtype=np.int64)
    best_target_scores = np.full(num_envs, np.inf, dtype=np.float64)
    failure_reasons = [""] * num_envs
    last_error_details = [None] * num_envs
    agent_frames = defaultdict(list)
    goal_frames = defaultdict(list)
    target_history = [[int(current_offsets[i])] for i in range(num_envs)]
    matched_history = [[0] for _ in range(num_envs)]

    max_attempts = int(max_subgoal_retries) + 1
    if max_attempts <= 0:
        raise ValueError("eval.max_subgoal_retries must be zero or greater.")
    realign_interval = max(1, int(realign_interval))
    realign_search_window = max(1, int(realign_search_window))
    stagnation_limit_steps = int(stagnation_limit_steps)
    total_budget = num_subgoals * int(segment_budget) * max_attempts

    for env_idx in range(num_envs):
        agent_frames[env_idx].append(world.infos["pixels"][env_idx, -1].copy())
        goal_frames[env_idx].append(goal_state["goal"][env_idx].copy())

    for _ in range(total_budget):
        if not active.any():
            break

        was_active = active.copy()
        actions = world._get_actions()
        _, world.rewards, world.terminateds, world.truncateds, world.infos = (
            world.envs.step(actions, mask=was_active)
        )

        for env_idx in np.where(was_active)[0]:
            pixels = world.infos["pixels"][env_idx]
            frame = pixels[-1] if pixels.ndim > 3 else pixels
            agent_frames[env_idx].append(np.asarray(frame).copy())
            goal_frames[env_idx].append(goal_state["goal"][env_idx].copy())

        rollout_steps[was_active] += 1
        segment_steps[was_active] += 1
        native_truncated = was_active & world.truncateds
        reached = np.zeros(num_envs, dtype=bool)
        scores = np.full(num_envs, np.inf, dtype=np.float64)

        for env_idx in np.where(was_active & ~native_truncated)[0]:
            current_state = np.asarray(world.infos["state"][env_idx, -1])
            target_offset = int(current_offsets[env_idx])
            if target_offset >= int(final_offset):
                success, score, details = normalized_final_error(
                    current_state,
                    block_goal_position,
                    block_goal_angle,
                    final_block_position_threshold,
                    final_block_angle_threshold,
                )
            else:
                target_state = trajectories[env_idx]["state"][target_offset]
                success, score, details = normalized_subgoal_error(
                    current_state,
                    target_state,
                    subgoal_block_position_threshold,
                    subgoal_block_angle_threshold,
                    subgoal_agent_relative_threshold,
                )
            reached[env_idx] = success
            scores[env_idx] = score
            last_error_details[env_idx] = details

            if score < (
                best_target_scores[env_idx] - float(stagnation_min_improvement)
            ):
                best_target_scores[env_idx] = score
                stagnation_steps[env_idx] = 0
            else:
                stagnation_steps[env_idx] += 1

        should_realign = (
            was_active
            & active
            & ~reached
            & ~native_truncated
            & ((rollout_steps % realign_interval) == 0)
        )
        for env_idx in np.where(should_realign)[0]:
            current_state = np.asarray(world.infos["state"][env_idx, -1])
            search_start = int(confirmed_offsets[env_idx])
            search_end = min(
                int(current_offsets[env_idx]),
                search_start + realign_search_window,
            )
            nearest_offset = find_nearest_expert_offset(
                current_state,
                trajectories[env_idx],
                search_start,
                search_end,
                agent_weight=agent_align_weight,
                agent_relative_weight=agent_relative_align_weight,
                block_weight=block_align_weight,
                angle_weight=angle_align_weight,
            )
            if nearest_offset != matched_offsets[env_idx]:
                old_match = int(matched_offsets[env_idx])
                matched_offsets[env_idx] = nearest_offset
                matched_history[env_idx].append(int(nearest_offset))
                print(
                    f"[env {env_idx}] matched expert frame "
                    f"{old_match}->{nearest_offset}; fixed target remains "
                    f"{current_offsets[env_idx]}."
                )

        flush = np.zeros(num_envs, dtype=bool)
        for env_idx in np.where(reached)[0]:
            target_idx = int(target_indices[env_idx])
            target_offset = int(current_offsets[env_idx])
            subgoal_successes[env_idx, target_idx] = True
            confirmed_offsets[env_idx] = target_offset
            if matched_offsets[env_idx] != target_offset:
                matched_history[env_idx].append(target_offset)
            matched_offsets[env_idx] = target_offset
            details = last_error_details[env_idx]

            if target_offset >= int(final_offset):
                final_successes[env_idx] = True
                active[env_idx] = False
                print(
                    f"[env {env_idx}] completed PushT at expert frame "
                    f"{final_offset} in {rollout_steps[env_idx]} steps; "
                    f"block position error={details[0]:.2f}, "
                    f"angle error={details[1]:.3f}."
                )
                continue

            print(
                f"[env {env_idx}] reached expert frame {target_offset} "
                f"on attempt {subgoal_attempts[env_idx, target_idx]}; "
                f"block error={details[0]:.2f}, angle error={details[1]:.3f}, "
                f"agent-relative error={details[2]:.2f}."
            )
            target_indices[env_idx] += 1
            next_target_idx = int(target_indices[env_idx])
            current_offsets[env_idx] = reference_offsets[next_target_idx]
            target_history[env_idx].append(int(current_offsets[env_idx]))
            subgoal_attempts[env_idx, next_target_idx] = 1
            segment_steps[env_idx] = 0
            retry_counts[env_idx] = 0
            stagnation_steps[env_idx] = 0
            best_target_scores[env_idx] = np.inf
            flush[env_idx] = True
            print(
                f"[env {env_idx}] next fixed target is expert frame "
                f"{current_offsets[env_idx]}."
            )

        timed_out = was_active & active & ~reached & (
            segment_steps >= int(segment_budget)
        )
        for env_idx in np.where(timed_out)[0]:
            target_idx = int(target_indices[env_idx])
            if retry_counts[env_idx] < int(max_subgoal_retries):
                retry_counts[env_idx] += 1
                subgoal_attempts[env_idx, target_idx] += 1
                segment_steps[env_idx] = 0
                flush[env_idx] = True
                print(
                    f"[env {env_idx}] target {current_offsets[env_idx]} timed out; "
                    f"retry {retry_counts[env_idx]}/{max_subgoal_retries}, "
                    f"matched frame={matched_offsets[env_idx]}, "
                    f"normalized error={scores[env_idx]:.3f}."
                )
            else:
                active[env_idx] = False
                failure_reasons[env_idx] = "attempts_exhausted"
                print(
                    f"[env {env_idx}] stopped at target "
                    f"{current_offsets[env_idx]} after {max_attempts} attempts; "
                    f"matched frame={matched_offsets[env_idx]}."
                )

        if stagnation_limit_steps > 0:
            stalled = (
                was_active
                & active
                & ~reached
                & (stagnation_steps >= stagnation_limit_steps)
            )
            for env_idx in np.where(stalled)[0]:
                active[env_idx] = False
                failure_reasons[env_idx] = "stagnated"
                print(
                    f"[env {env_idx}] stopped after "
                    f"{stagnation_steps[env_idx]} stagnant steps at target "
                    f"{current_offsets[env_idx]}; matched frame="
                    f"{matched_offsets[env_idx]}."
                )

        for env_idx in np.where(native_truncated & active)[0]:
            active[env_idx] = False
            failure_reasons[env_idx] = "environment_truncated"

        continuing = was_active & active
        for env_idx in np.where(continuing)[0]:
            world.infos["terminated"][env_idx] = False
            world.infos["truncated"][env_idx] = False
            world.terminateds[env_idx] = False
            world.truncateds[env_idx] = False

        if flush.any():
            goal_state = state_at_offsets(trajectories, current_offsets, goal=True)
            update_world_infos(world, goal_state, goal=True)
            for env_idx in np.where(flush & active)[0]:
                values = {key: value[env_idx] for key, value in goal_state.items()}
                apply_callables(
                    world.envs.envs[env_idx].unwrapped,
                    callables,
                    values,
                    goal_only=True,
                )
            world.infos["_needs_flush"] = flush

    for env_idx in np.where(active)[0]:
        failure_reasons[env_idx] = "global_budget_exhausted"
        active[env_idx] = False

    dataset_videos = [
        trajectory["pixels"][: len(agent_frames[i])]
        for i, trajectory in enumerate(trajectories)
    ]
    save_panel_videos(
        video_dir,
        {
            "agent": [np.asarray(agent_frames[i]) for i in range(num_envs)],
            "dataset": dataset_videos,
            "goal": [np.asarray(goal_frames[i]) for i in range(num_envs)],
        },
    )

    attempted_subgoals = subgoal_attempts > 0
    subgoal_success_rate = (
        float(subgoal_successes[attempted_subgoals].mean() * 100.0)
        if attempted_subgoals.any()
        else 0.0
    )
    return {
        "success_rate": float(final_successes.mean() * 100.0),
        "episode_successes": final_successes,
        "subgoal_success_rate": subgoal_success_rate,
        "subgoal_successes": subgoal_successes,
        "subgoal_attempts": subgoal_attempts,
        "reference_subgoal_offsets": np.asarray(reference_offsets),
        "confirmed_offsets": confirmed_offsets,
        "matched_offsets": matched_offsets,
        "current_target_offsets": current_offsets,
        "target_history": target_history,
        "matched_history": matched_history,
        "stagnation_steps": stagnation_steps,
        "failure_reasons": np.asarray(failure_reasons, dtype=object),
        "rollout_steps": rollout_steps,
        "seeds": init_state.get("seed"),
    }

def resolve_output_dir(path, default_dir):
    if path in (None, ""):
        return default_dir
    return Path(hydra.utils.to_absolute_path(str(path)))


def rename_saved_videos(video_dir, video_name, num_videos):
    if video_name in (None, ""):
        return

    target = Path(str(video_name))
    if target.parent != Path("."):
        raise ValueError(
            "output.video_name should be a filename only; use output.video_dir "
            "to configure the save location."
        )

    suffix = target.suffix or ".mp4"
    stem = target.stem if target.suffix else target.name

    for i in range(num_videos):
        src = video_dir / f"env_{i}.mp4"
        if not src.exists():
            continue
        dst_name = f"{stem}{suffix}" if num_videos == 1 else f"{stem}_{i}{suffix}"
        dst = video_dir / dst_name
        if src != dst:
            src.replace(dst)


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    plan_steps = int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block)
    execute_steps = int(cfg.plan_config.receding_horizon) * int(
        cfg.plan_config.action_block
    )
    assert plan_steps <= cfg.eval.eval_budget, (
        "Planning horizon must be smaller than or equal to eval_budget"
    )

    # create world environment
    sequential = bool(cfg.eval.get("sequential_subgoals", False))
    max_task_steps = int(cfg.eval.get("max_task_steps", cfg.eval.goal_offset_steps))
    max_subgoal_retries = int(cfg.eval.get("max_subgoal_retries", 0))
    block_goal_position = np.asarray(
        cfg.eval.get("block_goal_position", [256.0, 256.0]), dtype=np.float64
    )
    block_goal_angle = float(cfg.eval.get("block_goal_angle", np.pi / 4))
    subgoal_block_position_threshold = float(
        cfg.eval.get("subgoal_block_position_threshold", 20.0)
    )
    subgoal_block_angle_threshold = float(
        cfg.eval.get("subgoal_block_angle_threshold", np.pi / 9)
    )
    subgoal_agent_relative_threshold = float(
        cfg.eval.get("subgoal_agent_relative_threshold", 25.0)
    )
    final_block_position_threshold = float(
        cfg.eval.get("final_block_position_threshold", 20.0)
    )
    final_block_angle_threshold = float(
        cfg.eval.get("final_block_angle_threshold", np.pi / 9)
    )
    stagnation_limit_steps = int(cfg.eval.get("stagnation_limit_steps", 100))
    stagnation_min_improvement = float(
        cfg.eval.get("stagnation_min_improvement", 0.05)
    )
    if max_subgoal_retries < 0:
        raise ValueError("eval.max_subgoal_retries must be zero or greater.")
    if block_goal_position.shape != (2,):
        raise ValueError("eval.block_goal_position must contain exactly two values.")
    if min(
        subgoal_block_position_threshold,
        subgoal_block_angle_threshold,
        subgoal_agent_relative_threshold,
        final_block_position_threshold,
        final_block_angle_threshold,
    ) <= 0:
        raise ValueError("All PushT success thresholds must be greater than zero.")
    if sequential:
        if int(cfg.eval.num_eval) != 1:
            raise ValueError(
                "Sequential full-trajectory evaluation currently requires "
                "eval.num_eval: 1 so one complete expert trajectory maps to one video."
            )
        if int(cfg.eval.goal_offset_steps) != plan_steps:
            raise ValueError(
                "For official 25-step PushT inference, eval.goal_offset_steps "
                "must equal plan_config.horizon * plan_config.action_block."
            )
        max_offsets = build_subgoal_offsets(max_task_steps, cfg.eval.goal_offset_steps)
        max_rollout_steps = (
            len(max_offsets) * int(cfg.eval.eval_budget) * (max_subgoal_retries + 1)
        )
    else:
        max_rollout_steps = int(cfg.eval.eval_budget)
    cfg.world.max_episode_steps = max_rollout_steps + 1
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        #model = /home/muxiang/work/LeWm_Saimo/.stable-wm/checkpoints/pusht/lewm_8gpu/original/weights.pt"
        model = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cfg.cache_dir)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    default_output_dir = Path(__file__).parent

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    g = np.random.default_rng(cfg.seed)
    if sequential:
        episode_first_rows, episode_len = get_episode_bounds(dataset, ep_indices)
        all_states = np.asarray(dataset.get_col_data("state"))
        contact_starts = np.full(len(ep_indices), -1, dtype=np.int64)
        task_end_steps = np.full(len(ep_indices), -1, dtype=np.int64)
        task_steps = np.full(len(ep_indices), -1, dtype=np.int64)
        contact_distance_threshold = float(
            cfg.eval.get("contact_distance_threshold", 85.0)
        )

        for i, first_row in enumerate(episode_first_rows):
            states = all_states[first_row : first_row + episode_len[i]]
            contact_start = find_contact_start(states, contact_distance_threshold)
            if contact_start is None:
                continue
            task_end = find_task_end(
                states,
                contact_start,
                block_goal_position,
                block_goal_angle,
                final_block_position_threshold,
                final_block_angle_threshold,
            )
            if task_end is None or task_end <= contact_start:
                continue
            contact_starts[i] = contact_start
            task_end_steps[i] = task_end
            task_steps[i] = task_end - contact_start

        eligible = (task_steps > 0) & (task_steps <= max_task_steps)
        if not eligible.any():
            raise ValueError(
                "No expert episode both completes PushT and has contact-to-goal "
                f"steps <= eval.max_task_steps={max_task_steps}. Try increasing "
                "eval.max_task_steps or checking the configured block goal."
            )

        final_offset = int(task_steps[eligible].max())
        candidate_mask = eligible & (task_steps == final_offset)
        candidate_positions = np.nonzero(candidate_mask)[0]
        selected_pos = int(g.choice(candidate_positions))
        selected_episode = ep_indices[selected_pos]
        contact_start = int(contact_starts[selected_pos])
        task_end = int(task_end_steps[selected_pos])
        eval_episodes = np.asarray([selected_episode])
        eval_start_idx = np.asarray([contact_start], dtype=np.int64)
        print(
            f"Selected verified expert episode {selected_episode}: contact starts "
            f"at frame {contact_start}, PushT first succeeds at frame {task_end}, "
            f"{final_offset + 1} frames / {final_offset} environment steps. "
            f"Fixed subgoals: "
            f"{build_subgoal_offsets(final_offset, cfg.eval.goal_offset_steps)}"
        )
    else:
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {
            ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
        }
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
        )

        valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), "valid starting points found for evaluation.")

        if len(valid_indices) < cfg.eval.num_eval:
            raise ValueError(
                "Not enough valid starting points for the requested num_eval."
            )

        random_episode_indices = np.sort(
            g.choice(valid_indices, size=cfg.eval.num_eval, replace=False)
        )

        print(random_episode_indices)

        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    video_dir = resolve_output_dir(cfg.output.get("video_dir"), default_output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    callables = OmegaConf.to_container(
        cfg.eval.get("callables"), resolve=True
    )
    if sequential:
        metrics = evaluate_sequential_subgoals(
            world=world,
            dataset=dataset,
            episodes_idx=eval_episodes.tolist(),
            start_steps=eval_start_idx.tolist(),
            subgoal_interval=cfg.eval.goal_offset_steps,
            final_offset=final_offset,
            segment_budget=cfg.eval.eval_budget,
            max_subgoal_retries=max_subgoal_retries,
            callables=callables,
            video_dir=video_dir,
            realign_interval=cfg.eval.get("realign_interval_steps", execute_steps),
            realign_search_window=cfg.eval.get("realign_search_window", 25),
            agent_align_weight=cfg.eval.get("agent_align_weight", 1.0),
            agent_relative_align_weight=cfg.eval.get(
                "agent_relative_align_weight", 1.0
            ),
            block_align_weight=cfg.eval.get("block_align_weight", 1.0),
            angle_align_weight=cfg.eval.get("angle_align_weight", 20.0),
            block_goal_position=block_goal_position,
            block_goal_angle=block_goal_angle,
            subgoal_block_position_threshold=subgoal_block_position_threshold,
            subgoal_block_angle_threshold=subgoal_block_angle_threshold,
            subgoal_agent_relative_threshold=subgoal_agent_relative_threshold,
            final_block_position_threshold=final_block_position_threshold,
            final_block_angle_threshold=final_block_angle_threshold,
            stagnation_limit_steps=stagnation_limit_steps,
            stagnation_min_improvement=stagnation_min_improvement,
        )
        metrics["selected_episode"] = eval_episodes.copy()
        metrics["selected_start_step"] = eval_start_idx.copy()
        metrics["selected_expert_frames"] = np.asarray([final_offset + 1])
        metrics["selected_task_steps"] = np.asarray([final_offset])
    else:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            video=video_dir,
        )
    rename_saved_videos(video_dir, cfg.output.get("video_name"), cfg.eval.num_eval)
    end_time = time.time()
    
    print(metrics)
    print(f"Videos saved to {video_dir}")

    results_path = video_dir / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
