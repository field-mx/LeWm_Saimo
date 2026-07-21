from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


STATE_DIM = 28
GOAL_DIM = 5
ACTOR_OBSERVATION_DIM = STATE_DIM + GOAL_DIM
ACTION_DIM = 5

XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALE = 10.0


def _read_first(dataset: h5py.File, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in dataset:
            return np.asarray(dataset[name], dtype=np.float32)
    raise KeyError(f"None of the HDF5 columns exist: {names}")


def goal_features_from_position_yaw(
    position: np.ndarray, yaw: np.ndarray
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32).reshape(-1, 1)
    scaled_position = (position - XYZ_CENTER) * XYZ_SCALE
    return np.concatenate([scaled_position, np.cos(yaw), np.sin(yaw)], axis=-1)


def actor_observation_from_env(
    state: np.ndarray, target_state: np.ndarray
) -> np.ndarray:
    """Build the same 33-D feature used for demonstration training.

    Cube state layout stores block xyz at 19:22 and block yaw cos/sin at 26:28.
    """
    state = np.asarray(state, dtype=np.float32)
    target_state = np.asarray(target_state, dtype=np.float32)
    goal = np.concatenate([target_state[..., 19:22], target_state[..., 26:28]], axis=-1)
    return np.concatenate([state, goal], axis=-1).astype(np.float32, copy=False)


def _episode_ids_from_terminals(terminals: np.ndarray) -> np.ndarray:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    episode_ids = np.zeros(len(terminals), dtype=np.int32)
    if len(terminals) > 1:
        episode_ids[1:] = np.cumsum(terminals[:-1], dtype=np.int32)
    return episode_ids


def _sample_future_goals(
    states: np.ndarray,
    episode_ids: np.ndarray,
    max_goal_offset: int | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    goal_indices = np.arange(len(states), dtype=np.int64)
    for episode_id in np.unique(episode_ids):
        episode_indices = np.flatnonzero(episode_ids == episode_id)
        if len(episode_indices) == 0:
            continue
        episode_end = int(episode_indices[-1])
        upper = np.full(len(episode_indices), episode_end, dtype=np.int64)
        if max_goal_offset is not None:
            upper = np.minimum(upper, episode_indices + max_goal_offset)
        span = upper - episode_indices + 1
        goal_indices[episode_indices] = episode_indices + (
            rng.random(len(episode_indices)) * span
        ).astype(np.int64)

    goal_states = states[goal_indices]
    return np.concatenate([goal_states[:, 19:22], goal_states[:, 26:28]], axis=-1)


def load_demonstrations(
    path: str | Path,
    *,
    max_goal_offset: int | None = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as dataset:
            states = np.asarray(dataset["observations"], dtype=np.float32)
            actions = np.asarray(dataset["actions"], dtype=np.float32)
            terminal_key = "terminals" if "terminals" in dataset else "dones"
            episode_ids = _episode_ids_from_terminals(dataset[terminal_key])
        goal_features = _sample_future_goals(
            states, episode_ids, max_goal_offset, seed
        )
        observations = np.concatenate([states, goal_features], axis=-1)
        return observations.astype(np.float32), actions, episode_ids

    with h5py.File(path, "r") as dataset:
        states = _read_first(dataset, ("observation",))
        actions = _read_first(dataset, ("action",))
        target_positions = _read_first(
            dataset,
            (
                "privileged_target_block_pos",
                "privileged/target_block_pos",
            ),
        )
        target_yaws = _read_first(
            dataset,
            (
                "privileged_target_block_yaw",
                "privileged/target_block_yaw",
            ),
        )

        if "ep_offset" in dataset and "ep_len" in dataset:
            offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
            lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
            episode_ids = np.empty(len(actions), dtype=np.int32)
            for episode_id, (offset, length) in enumerate(zip(offsets, lengths)):
                episode_ids[offset : offset + length] = episode_id
        else:
            episode_ids = np.asarray(
                dataset.get("episode_idx", dataset.get("ep_idx")), dtype=np.int32
            )

    if states.shape != (len(actions), STATE_DIM):
        raise ValueError(f"Expected state shape (N, {STATE_DIM}), got {states.shape}")
    if actions.shape != (len(states), ACTION_DIM):
        raise ValueError(f"Expected action shape (N, {ACTION_DIM}), got {actions.shape}")

    goal_features = goal_features_from_position_yaw(target_positions, target_yaws)
    observations = np.concatenate([states, goal_features], axis=-1)
    return observations.astype(np.float32), actions.astype(np.float32), episode_ids


def normalize_observations(
    observations: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((observations - mean) / scale).astype(np.float32, copy=False)
