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
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import ogbench
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "rgb_latent_value_mpc"):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import run_experiment as base
from privileged_teacher import PrivilegedTeacher, semantic_state
from proprio_v2_models import (
    ProprioBlockActor,
    ProprioSemanticNet,
    ProprioTargetNet,
    ProprioTransitionNet,
    ProprioValueEnsemble,
)


STATE_NAMES = ("align", "grasp", "transfer")
ALIGN, GRASP, TRANSFER = range(len(STATE_NAMES))
PREDICATE_NAMES = ("aligned", "grasped")
ALIGNED, GRASPED = range(len(PREDICATE_NAMES))


def robot_proprio(state: np.ndarray) -> np.ndarray:
    """Runtime-safe robot state: XYZ, yaw cos/sin, gripper closure."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape[-1] < 18:
        raise ValueError("Cube observation must contain at least 18 values.")
    closure = np.clip(value[..., 17:18] / 3.0, 0.0, 1.0)
    return np.concatenate(
        [value[..., 12:15], value[..., 15:17], closure], axis=-1
    ).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cube RGB-latent MPC v2 with robot proprioception in the Actor, "
            "semantic recognizer, and learned transition model."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "proprio_v2_config.yaml"
    )
    parser.add_argument(
        "--stage", choices=("train", "adapt", "all"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default="proprio_v2_round1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_smoke_config(config: dict[str, Any]) -> None:
    config["paths"]["output_dir"] /= "smoke"
    config["offline_training"].update(epochs=1, batch_size=128)
    config["task_finetuning"].update(
        episodes=4, max_steps=120, epochs=1, batch_size=64,
        reuse_dataset=False,
    )
    config["planner"].update(num_candidates=8, horizon_blocks=2)
    config["online_training"].update(
        episodes=1, max_steps=8, warmup_samples=2, batch_size=4
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    base.append_jsonl(path, value)


def save_json(path: Path, value: dict[str, Any]) -> None:
    base.save_json(path, value)


def state_ids_from_predicates(
    predicates: np.ndarray,
    proprios: np.ndarray,
    closed_threshold: float,
) -> np.ndarray:
    aligned = predicates[:, ALIGNED] >= 0.5
    grasped = (
        (predicates[:, GRASPED] >= 0.5)
        & (proprios[:, 5] >= closed_threshold)
    )
    return np.where(grasped, TRANSFER, np.where(aligned, GRASP, ALIGN)).astype(
        np.int64
    )


def _phase_targets(
    episode_ids: np.ndarray,
    state_ids: np.ndarray,
    predicate_targets: np.ndarray,
    goal_rows: np.ndarray,
) -> np.ndarray:
    del goal_rows
    targets = np.arange(len(episode_ids), dtype=np.int64)
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode)
        next_aligned = -1
        next_grasped = -1
        for row in rows[::-1]:
            if predicate_targets[row, GRASPED] >= 0.5:
                next_grasped = row
                next_aligned = row
            elif predicate_targets[row, ALIGNED] >= 0.5:
                next_aligned = row
            if state_ids[row] == ALIGN and next_aligned >= 0:
                targets[row] = next_aligned
            elif state_ids[row] == GRASP and next_grasped >= 0:
                targets[row] = next_grasped
    return targets


def build_source_dataset(
    config: dict[str, Any],
) -> tuple[TensorDataset, TensorDataset, dict[str, Any]]:
    cache_path = config["paths"]["source_latent_cache"]
    source_path = config["paths"]["action_dataset"]
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Source-derived latent cache missing: {cache_path}. "
            "Generate cube_semantic_latents.npz before v2 training."
        )

    with np.load(cache_path) as cache, np.load(source_path) as source:
        latent_bank = np.asarray(cache["latents"], dtype=np.float32)
        unique_indices = np.asarray(cache["unique_source_indices"], dtype=np.int64)
        previous_rows = np.asarray(cache["previous_rows"], dtype=np.int64)
        current_rows = np.asarray(cache["current_rows"], dtype=np.int64)
        goal_rows = np.asarray(cache["goal_rows"], dtype=np.int64)
        source_indices = np.asarray(cache["source_indices"], dtype=np.int64)
        episode_ids = np.asarray(cache["episode_ids"], dtype=np.int64)
        predicates = np.asarray(cache["predicate_targets"][:, :2], dtype=np.float32)
        observations = np.asarray(source["observations"], dtype=np.float32)
        actions = np.asarray(source["actions"], dtype=np.float32)
        terminals = np.asarray(source["terminals"], dtype=bool)

    previous = latent_bank[previous_rows]
    current = latent_bank[current_rows]
    goal = latent_bank[goal_rows]
    previous_proprio = robot_proprio(observations[unique_indices[previous_rows]])
    proprio = robot_proprio(observations[source_indices])
    state_ids = state_ids_from_predicates(
        predicates,
        proprio,
        float(config["labels"]["closed_gripper_threshold"]),
    )

    order = np.lexsort((source_indices, episode_ids))
    next_sample = np.arange(len(source_indices), dtype=np.int64)
    for episode in np.unique(episode_ids):
        rows = order[episode_ids[order] == episode]
        if len(rows) > 1:
            next_sample[rows[:-1]] = rows[1:]
    next_latent = current[next_sample]
    next_proprio = proprio[next_sample]
    next_state_ids = state_ids[next_sample]
    done = (next_sample == np.arange(len(next_sample))).astype(np.float32)

    phase_target_rows = _phase_targets(
        episode_ids, state_ids, predicates, goal_rows
    )
    keyframe_target = np.empty_like(current)
    transfer_mask = state_ids == TRANSFER
    keyframe_target[transfer_mask] = goal[transfer_mask]
    non_transfer = ~transfer_mask
    keyframe_target[non_transfer] = current[phase_target_rows[non_transfer]]
    next_keyframe_target = keyframe_target[next_sample]

    block = int(config["model"]["action_block"])
    action_blocks = np.empty(
        (len(source_indices), block * actions.shape[1]), dtype=np.float32
    )
    terminal_indices = np.flatnonzero(terminals)
    for row, source_index in enumerate(source_indices):
        terminal_pos = np.searchsorted(terminal_indices, source_index)
        episode_end = (
            int(terminal_indices[terminal_pos])
            if terminal_pos < len(terminal_indices)
            else len(actions) - 1
        )
        indices = np.minimum(
            source_index + np.arange(block, dtype=np.int64), episode_end
        )
        action_blocks[row] = actions[indices].reshape(-1)

    reward = np.full(len(source_indices), -float(
        config["offline_training"]["step_penalty"]
    ), dtype=np.float32)
    advanced = next_state_ids > state_ids
    recovered = next_state_ids < state_ids
    reward[advanced] += float(
        config["offline_training"]["state_transition_reward"]
    )
    reward[recovered] -= float(
        config["offline_training"]["recovery_penalty"]
    )
    transfer_progress = (
        np.mean((current - goal) ** 2, axis=1)
        - np.mean((next_latent - goal) ** 2, axis=1)
    )
    reward[transfer_mask] += np.clip(
        10.0 * transfer_progress[transfer_mask], -0.25, 0.25
    )

    transition_inputs = state_ids.copy()
    transition_inputs[order[1:]] = state_ids[order[:-1]]
    episode_change = episode_ids[order[1:]] != episode_ids[order[:-1]]
    transition_inputs[order[1:][episode_change]] = ALIGN

    arrays = (
        previous,
        current,
        next_latent,
        goal,
        keyframe_target,
        next_keyframe_target,
        previous_proprio,
        proprio,
        next_proprio,
        action_blocks,
        transition_inputs,
        state_ids,
        next_state_ids,
        predicates,
        reward,
        done,
    )
    validation_fraction = float(
        config["offline_training"]["validation_fraction"]
    )
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(int(config["seed"]))
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_mask = np.isin(episode_ids, episodes[:validation_count])

    def make_dataset(indices: np.ndarray) -> TensorDataset:
        tensors = []
        for index, values in enumerate(arrays):
            if index in (10, 11, 12):
                tensors.append(torch.from_numpy(values[indices].astype(np.int64)))
            else:
                tensors.append(torch.from_numpy(values[indices].astype(np.float32)))
        return TensorDataset(*tensors)

    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    metadata = {
        "source_episodes": int(len(episodes)),
        "samples": int(len(source_indices)),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "state_counts": {
            STATE_NAMES[index]: int(np.sum(state_ids == index))
            for index in range(len(STATE_NAMES))
        },
        "predicate_counts": {
            PREDICATE_NAMES[index]: int(predicates[:, index].sum())
            for index in range(len(PREDICATE_NAMES))
        },
        "runtime_proprio_fields": [
            "effector_x",
            "effector_y",
            "effector_z",
            "effector_yaw_cos",
            "effector_yaw_sin",
            "gripper_closure",
        ],
        "privileged_runtime_fields": [],
    }
    return (
        make_dataset(train_indices),
        make_dataset(validation_indices),
        metadata,
    )


def build_models(config: dict[str, Any], device: torch.device):
    settings = config["model"]
    latent = int(settings["latent_dim"])
    proprio = int(settings["proprio_dim"])
    hidden = int(settings["hidden_dim"])
    predicates = int(settings["predicate_count"])
    states = int(settings["state_count"])
    semantic = ProprioSemanticNet(latent, proprio, hidden, predicates).to(device)
    transition = ProprioTransitionNet(
        latent, proprio, hidden, predicates, states
    ).to(device)
    target = ProprioTargetNet(latent, proprio, hidden, states).to(device)
    actor = ProprioBlockActor(
        latent,
        proprio,
        int(settings["action_dim"]),
        int(settings["action_block"]),
        hidden,
        states,
    ).to(device)
    critic = ProprioValueEnsemble(
        latent,
        proprio,
        hidden,
        states,
        int(settings["value_ensemble_size"]),
    ).to(device)
    return semantic, transition, target, actor, critic

def model_losses(
    models,
    target_critic: ProprioValueEnsemble,
    batch: tuple[torch.Tensor, ...],
    config: dict[str, Any],
    positive_weights: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic, transition, target, actor, critic = models
    settings = config["offline_training"]
    (
        previous,
        current,
        next_latent,
        goal,
        keyframe_target,
        next_keyframe_target,
        previous_proprio,
        proprio,
        next_proprio,
        action_blocks,
        transition_input,
        state_id,
        next_state_id,
        predicates,
        reward,
        done,
    ) = [item.to(device) for item in batch]

    semantic_logits = semantic(
        previous, current, previous_proprio, proprio
    )
    semantic_probabilities = torch.sigmoid(semantic_logits).detach()
    transition_logits = transition(
        previous,
        current,
        previous_proprio,
        proprio,
        semantic_probabilities,
        transition_input,
    )
    predicted_target = target(
        previous, current, goal, state_id, proprio
    )
    predicted_actions = actor.mean(
        previous, current, keyframe_target, goal, state_id, proprio
    )
    values = critic(
        previous, current, keyframe_target, goal, state_id, proprio
    )
    with torch.no_grad():
        next_values = target_critic(
            current,
            next_latent,
            next_keyframe_target,
            goal,
            next_state_id,
            next_proprio,
        ).mean(dim=0)
        td_target = reward + float(settings["gamma"]) * (1.0 - done) * next_values

    semantic_loss = nn.functional.binary_cross_entropy_with_logits(
        semantic_logits, predicates, pos_weight=positive_weights
    )
    transition_loss = nn.functional.cross_entropy(
        transition_logits, state_id
    )
    target_loss = nn.functional.mse_loss(predicted_target, keyframe_target)
    actor_loss = nn.functional.mse_loss(predicted_actions, action_blocks)
    critic_loss = nn.functional.mse_loss(
        values, td_target.unsqueeze(0).expand_as(values)
    )
    total = (
        float(settings["semantic_coefficient"]) * semantic_loss
        + float(settings["transition_coefficient"]) * transition_loss
        + float(settings["target_coefficient"]) * target_loss
        + float(settings["actor_coefficient"]) * actor_loss
        + float(settings["critic_coefficient"]) * critic_loss
    )
    with torch.no_grad():
        semantic_accuracy = (
            (torch.sigmoid(semantic_logits) >= 0.5) == predicates.bool()
        ).float().mean()
        transition_accuracy = (
            transition_logits.argmax(dim=-1) == state_id
        ).float().mean()
    return total, {
        "loss": float(total.detach().item()),
        "semantic_loss": float(semantic_loss.detach().item()),
        "transition_loss": float(transition_loss.detach().item()),
        "target_loss": float(target_loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "critic_loss": float(critic_loss.detach().item()),
        "semantic_accuracy": float(semantic_accuracy.detach().item()),
        "transition_accuracy": float(transition_accuracy.detach().item()),
    }


def _mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def save_checkpoint(
    path: Path,
    config: dict[str, Any],
    models,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    extra: dict[str, Any],
) -> None:
    semantic, transition, target, actor, critic = models
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cube_proprio_v2",
            "semantic_net": semantic.state_dict(),
            "transition_net": transition.state_dict(),
            "target_net": target.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "action_mean": action_mean,
            "action_scale": action_scale,
            "config": base.json_value(config),
            "extra": base.json_value(extra),
        },
        path,
    )


def load_checkpoint(
    path: Path, config: dict[str, Any], device: torch.device
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "cube_proprio_v2":
        raise ValueError(
            f"{path} is not a proprio-v2 checkpoint; retraining is required."
        )
    models = build_models(config, device)
    keys = (
        "semantic_net",
        "transition_net",
        "target_net",
        "actor",
        "critic",
    )
    for model, key in zip(models, keys):
        model.load_state_dict(checkpoint[key])
        model.eval()
    return (
        models,
        np.asarray(checkpoint["action_mean"], dtype=np.float32),
        np.asarray(checkpoint["action_scale"], dtype=np.float32),
    )


def train_offline(
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
):
    train_dataset, validation_dataset, metadata = build_source_dataset(config)
    settings = config["offline_training"]
    models = build_models(config, device)
    target_critic = copy.deepcopy(models[-1]).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for model in models for parameter in model.parameters()],
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    predicate_targets = train_dataset.tensors[13].numpy()
    positive = predicate_targets.sum(axis=0)
    negative = len(predicate_targets) - positive
    positive_weights = torch.from_numpy(
        np.clip(
            negative / np.maximum(positive, 1.0),
            1.0,
            float(settings["max_positive_weight"]),
        ).astype(np.float32)
    ).to(device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    log_path.unlink(missing_ok=True)
    best_validation = float("inf")
    best_states = None
    started = time.perf_counter()

    for epoch in range(int(settings["epochs"])):
        for model in models:
            model.train()
        train_records = []
        for batch in train_loader:
            loss, metrics = model_losses(
                models,
                target_critic,
                batch,
                config,
                positive_weights,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for model in models for parameter in model.parameters()],
                float(settings["max_grad_norm"]),
            )
            optimizer.step()
            with torch.no_grad():
                tau = float(settings["target_tau"])
                for target_parameter, parameter in zip(
                    target_critic.parameters(), models[-1].parameters()
                ):
                    target_parameter.lerp_(parameter, tau)
            train_records.append(metrics)

        for model in models:
            model.eval()
        validation_records = []
        with torch.no_grad():
            for batch in validation_loader:
                _, metrics = model_losses(
                    models,
                    target_critic,
                    batch,
                    config,
                    positive_weights,
                    device,
                )
                validation_records.append(metrics)
        record = {
            "epoch": epoch + 1,
            "train": _mean_metrics(train_records),
            "validation": _mean_metrics(validation_records),
        }
        append_jsonl(log_path, record)
        print(json.dumps(record))
        if record["validation"]["loss"] < best_validation:
            best_validation = record["validation"]["loss"]
            best_states = [
                copy.deepcopy(model.state_dict()) for model in models
            ]

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)
    for model in models:
        model.eval()
    training_summary = {
        **metadata,
        "epochs": int(settings["epochs"]),
        "best_validation_loss": best_validation,
        "training_seconds": time.perf_counter() - started,
    }
    save_checkpoint(
        checkpoint_path,
        config,
        models,
        action_mean,
        action_scale,
        {"offline_training": training_summary},
    )
    return models, training_summary


@torch.inference_mode()
def semantic_probabilities(
    semantic: ProprioSemanticNet,
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_proprio: torch.Tensor,
    proprio: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(
        semantic(previous, current, previous_proprio, proprio)
    )[0]


@torch.inference_mode()
def transition_probabilities(
    transition: ProprioTransitionNet,
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_proprio: torch.Tensor,
    proprio: torch.Tensor,
    predicates: torch.Tensor,
    state_id: int,
) -> torch.Tensor:
    state = torch.full(
        (1,), state_id, dtype=torch.long, device=current.device
    )
    return torch.softmax(
        transition(
            previous,
            current,
            previous_proprio,
            proprio,
            predicates.unsqueeze(0),
            state,
        ),
        dim=-1,
    )[0]


@dataclass
class StateTransition:
    previous: int
    current: int
    changed: bool
    rollback: bool
    reason: str
    complete: bool


class LearnedRecoveryStateMachine:
    """Learned state proposal with topology, hysteresis, and temporal guards."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.state = ALIGN
        self.steps_in_state = 0
        self.cooldown = 0
        self.aligned_streak = 0
        self.grasp_streak = 0
        self.lost_alignment_streak = 0
        self.lost_grasp_streak = 0
        self.complete_streak = 0
        self.recoveries = 0
        self.complete = False

    def _change(self, desired: int, reason: str) -> StateTransition:
        previous = self.state
        self.state = desired
        self.steps_in_state = 0
        self.cooldown = int(self.settings["state_cooldown_steps"])
        self.aligned_streak = 0
        self.grasp_streak = 0
        self.lost_alignment_streak = 0
        self.lost_grasp_streak = 0
        rollback = desired < previous
        if rollback:
            self.recoveries += 1
        return StateTransition(
            previous, desired, True, rollback, reason, self.complete
        )

    def observe(
        self,
        predicates: np.ndarray,
        transition_proposal: np.ndarray,
        proprio: np.ndarray,
        *,
        transfer_goal_reached: bool,
    ) -> StateTransition:
        previous = self.state
        self.steps_in_state += 1
        self.cooldown = max(0, self.cooldown - 1)
        aligned = float(predicates[ALIGNED])
        grasped = float(predicates[GRASPED])
        closed = float(proprio[5])
        proposal_state = int(np.argmax(transition_proposal))
        proposal_confidence = float(transition_proposal[proposal_state])
        learned_ok = proposal_confidence >= float(
            self.settings["transition_model_confidence"]
        )

        if self.state == ALIGN:
            ready = (
                aligned >= float(self.settings["aligned_enter_threshold"])
                and learned_ok
                and proposal_state in (GRASP, TRANSFER)
            )
            self.aligned_streak = self.aligned_streak + 1 if ready else 0
            if (
                self.cooldown == 0
                and self.aligned_streak
                >= int(self.settings["aligned_confirm_frames"])
            ):
                return self._change(GRASP, "aligned_confirmed")
            if self.steps_in_state >= int(self.settings["max_align_steps"]):
                self.steps_in_state = 0

        elif self.state == GRASP:
            grasp_ready = (
                grasped >= float(self.settings["grasp_enter_threshold"])
                and closed >= float(self.settings["gripper_closed_threshold"])
            )
            self.grasp_streak = self.grasp_streak + 1 if grasp_ready else 0
            if self.grasp_streak >= int(
                self.settings["grasp_confirm_frames"]
            ):
                return self._change(TRANSFER, "grasp_confirmed")
            if self.steps_in_state >= int(self.settings["max_grasp_steps"]):
                return self._change(ALIGN, "grasp_timeout")

        else:
            # TRANSFER is latched after a confirmed grasp. Visual grasp scores
            # can fluctuate while the arm moves; rolling back would switch the
            # planner to ALIGN and actively reopen the gripper.
            self.lost_grasp_streak = 0
            self.complete_streak = (
                self.complete_streak + 1 if transfer_goal_reached else 0
            )
            if self.complete_streak >= int(
                self.settings["complete_consecutive_frames"]
            ):
                self.complete = True
                return StateTransition(
                    previous,
                    self.state,
                    False,
                    False,
                    "transfer_goal_confirmed",
                    True,
                )
        return StateTransition(
            previous, self.state, False, False, "stay", self.complete
        )


def critic_statistics(
    critic: ProprioValueEnsemble,
    previous: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    goal: torch.Tensor,
    state: torch.Tensor,
    proprio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = critic(previous, current, target, goal, state, proprio)
    return values.mean(dim=0), values.std(dim=0, unbiased=False)


def imagine_proprio(
    proprio: torch.Tensor,
    action_block: torch.Tensor,
    settings: dict[str, Any],
) -> torch.Tensor:
    result = proprio.clone()
    xyz_scale = torch.as_tensor(
        settings["xyz_action_scale"],
        dtype=result.dtype,
        device=result.device,
    )
    result[:, :3] += (
        action_block[..., :3] * xyz_scale
    ).sum(dim=1)
    yaw = torch.atan2(result[:, 4], result[:, 3])
    yaw += 0.30 * action_block[..., 3].sum(dim=1)
    result[:, 3] = torch.cos(yaw)
    result[:, 4] = torch.sin(yaw)
    result[:, 5] += 0.10 * action_block[..., 4].sum(dim=1)
    lower = torch.as_tensor(
        settings["proprio_min"], dtype=result.dtype, device=result.device
    )
    upper = torch.as_tensor(
        settings["proprio_max"], dtype=result.dtype, device=result.device
    )
    return torch.minimum(torch.maximum(result, lower), upper)


@torch.inference_mode()
def propose_and_score(
    *,
    actor: ProprioBlockActor,
    target_net: ProprioTargetNet,
    critic: ProprioValueEnsemble,
    world_model,
    previous: torch.Tensor,
    current: torch.Tensor,
    goal: torch.Tensor,
    transfer_goal: torch.Tensor,
    proprio: torch.Tensor,
    state_id: int,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    planner: dict[str, Any],
    model_settings: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    device = current.device
    count = int(planner["num_candidates"])
    horizon = int(planner["horizon_blocks"])
    block = actor.action_block
    action_dim = actor.action_dim
    state = torch.full((1,), state_id, dtype=torch.long, device=device)
    keyframe_target = (
        transfer_goal
        if state_id == TRANSFER
        else target_net(previous, current, goal, state, proprio)
    )
    candidate_previous = previous.expand(count, -1).clone()
    candidate_current = current.expand(count, -1).clone()
    candidate_goal = goal.expand(count, -1)
    candidate_target = keyframe_target.expand(count, -1)
    candidate_proprio = proprio.expand(count, -1).clone()
    candidate_state = state.expand(count)
    mean_tensor = torch.as_tensor(
        action_mean, dtype=current.dtype, device=device
    )
    scale_tensor = torch.as_tensor(
        action_scale, dtype=current.dtype, device=device
    )
    exploration = float(planner["exploration_std"])
    blocks = []
    values_along_path = []
    predicted_latents = []

    for _ in range(horizon):
        mean_block = actor.mean(
            candidate_previous,
            candidate_current,
            candidate_target,
            candidate_goal,
            candidate_state,
            candidate_proprio,
        ).reshape(count, block, action_dim)
        noise = torch.from_numpy(
            rng.normal(
                0.0,
                exploration,
                size=(count, block, action_dim),
            ).astype(np.float32)
        ).to(device)
        noise[0].zero_()
        raw_block = (mean_block + noise).clamp(-1.0, 1.0)
        if state_id == ALIGN:
            raw_block[..., 4].clamp_(max=-0.35)
        else:
            raw_block[..., 4].clamp_(min=0.35)

        normalized = ((raw_block - mean_tensor) / scale_tensor).reshape(
            count, -1
        )
        next_latent = base.predict_latent_block(
            world_model, candidate_current, normalized
        )
        next_proprio = imagine_proprio(
            candidate_proprio, raw_block, model_settings
        )
        value, _ = critic_statistics(
            critic,
            candidate_current,
            next_latent,
            candidate_target,
            candidate_goal,
            candidate_state,
            next_proprio,
        )
        blocks.append(raw_block)
        values_along_path.append(value)
        predicted_latents.append(next_latent)
        candidate_previous, candidate_current = candidate_current, next_latent
        candidate_proprio = next_proprio
        exploration *= float(planner["exploration_decay"])

    plans = torch.stack(blocks, dim=1)
    flattened = plans.reshape(count, horizon * block, action_dim)
    terminal = predicted_latents[-1]
    terminal_value, uncertainty = critic_statistics(
        critic,
        candidate_previous,
        terminal,
        candidate_target,
        candidate_goal,
        candidate_state,
        candidate_proprio,
    )
    trajectory_value = torch.stack(values_along_path, dim=1).mean(dim=1)
    short_cost = (terminal - candidate_target).square().mean(dim=-1)
    behavior_deviation = (
        plans - plans[0:1]
    ).square().mean(dim=(1, 2, 3))
    smoothness = (
        flattened[:, 1:] - flattened[:, :-1]
    ).square().mean(dim=(1, 2))
    score = (
        -float(planner["short_target_coefficient"]) * short_cost
        + float(planner["terminal_value_coefficient"]) * terminal_value
        + float(planner["trajectory_value_coefficient"]) * trajectory_value
        - float(planner["uncertainty_coefficient"]) * uncertainty
        - float(planner["action_prior_coefficient"]) * behavior_deviation
        - float(planner["action_smoothness_coefficient"]) * smoothness
    )
    best_index = int(torch.argmax(score).item())
    elite_count = min(int(planner["elite_count"]), count)
    elite_scores, elite_indices = torch.topk(score, elite_count)
    weights = torch.softmax(
        (elite_scores - elite_scores.max())
        / max(float(planner["elite_temperature"]), 1e-6),
        dim=0,
    )
    elite_plan = (
        flattened[elite_indices] * weights[:, None, None]
    ).sum(dim=0)
    return {
        "flattened": flattened,
        "score": score,
        "best_index": best_index,
        "keyframe_target": keyframe_target,
        "elite_plan": elite_plan,
        "terminal_value": terminal_value,
        "uncertainty": uncertainty,
        "short_cost": short_cost,
    }


def render_transfer_goal(
    config: dict[str, Any], env
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config["transfer_goal"]
    labels = config["labels"]
    controller = PrivilegedTeacher()
    state, reset_info = env.reset(
        seed=int(settings["reference_seed"]),
        options=base.task_reset_options(config),
    )
    goal_state = np.asarray(reset_info["goal"], dtype=np.float32)
    teacher_settings = {
        **labels,
        "goal_xyz_m": float(settings["goal_xyz_m"]),
        "goal_xy_m": float(settings["goal_xyz_m"]),
    }
    for step in range(int(settings["max_steps"]) + 1):
        value = semantic_state(state, goal_state)
        goal_distance = float(np.linalg.norm(value.cube - value.goal))
        grasped = (
            value.contact >= float(labels["contact_threshold"])
            and value.gripper_opening
            >= float(labels["closed_gripper_threshold"])
        )
        if grasped and goal_distance <= float(settings["goal_xyz_m"]):
            return np.asarray(env.render()).copy(), {
                "generation_steps": step,
                "cube_goal_distance_m": goal_distance,
                "task_init_xyz": config["task"]["init_xyz"],
                "task_goal_xyz": config["task"]["goal_xyz"],
            }
        action, _ = controller.action(state, goal_state, teacher_settings)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    raise RuntimeError(
        "Failed to render the task-specific grasped goal RGB within "
        f"{settings['max_steps']} simulator steps."
    )


def update_actor_online(
    actor: ProprioBlockActor,
    anchor: ProprioBlockActor,
    optimizer: torch.optim.Optimizer,
    replay: deque,
    settings: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    actor.train()
    losses = []
    for _ in range(int(settings["updates_per_step"])):
        count = min(int(settings["batch_size"]), len(replay))
        indices = rng.choice(len(replay), size=count, replace=False)
        columns = [
            torch.stack([replay[int(index)][column] for index in indices]).to(device)
            for column in range(7)
        ]
        previous, current, target, goal, state, proprio, selected_action = columns
        prediction = actor.mean(
            previous, current, target, goal, state, proprio
        ).reshape(count, actor.action_block, actor.action_dim)
        with torch.no_grad():
            anchor_prediction = anchor.mean(
                previous, current, target, goal, state, proprio
            ).reshape(count, actor.action_block, actor.action_dim)
        selected_loss = nn.functional.mse_loss(
            prediction[:, 0], selected_action
        )
        anchor_loss = nn.functional.mse_loss(
            prediction, anchor_prediction
        )
        loss = selected_loss + float(
            settings["bc_anchor_coefficient"]
        ) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings["max_grad_norm"])
        )
        optimizer.step()
        losses.append(float(loss.item()))
    actor.eval()
    return float(np.mean(losses))


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def run_online(
    *,
    config: dict[str, Any],
    models,
    world_model,
    transform,
    device: torch.device,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    log_path: Path,
    video_dir: Path,
) -> dict[str, Any]:
    semantic, transition_net, target_net, actor, critic = models
    settings = config["online_training"]
    planner = config["planner"]
    transfer_settings = config["transfer_goal"]
    machine_settings = {
        **config["state_machine"],
        "complete_consecutive_frames": int(
            transfer_settings["complete_consecutive_frames"]
        ),
    }
    rng = np.random.default_rng(
        int(config["seed"]) + int(settings["seed_offset"])
    )
    env = ogbench.make_env_and_datasets(
        "cube-single-play-v0",
        env_only=True,
        terminate_at_goal=False,
    )
    transfer_goal_image, transfer_metadata = render_transfer_goal(config, env)
    transfer_goal_latent = base.encode_image(
        world_model, transfer_goal_image, transform, device
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    goal_rgb_path = video_dir / "transfer_goal_rgb.png"
    imageio.imwrite(goal_rgb_path, transfer_goal_image)
    anchor_actor = copy.deepcopy(actor).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=1e-5,
    )
    replay = deque(maxlen=int(settings["replay_capacity"]))
    log_path.unlink(missing_ok=True)
    episode_results = []
    total_world_model_seconds = 0.0
    total_world_model_calls = 0

    for episode in range(int(settings["episodes"])):
        observation, reset_info = env.reset(
            seed=int(config["seed"]) + int(settings["seed_offset"]) + episode,
            options=base.task_reset_options(config),
        )
        goal_image = np.asarray(reset_info["goal_rendered"]).copy()
        goal_latent = base.encode_image(
            world_model, goal_image, transform, device
        )
        current_image = np.asarray(env.render()).copy()
        current_latent = base.encode_image(
            world_model, current_image, transform, device
        )
        previous_latent = current_latent.clone()
        current_proprio = torch.from_numpy(
            robot_proprio(observation)
        ).to(device).unsqueeze(0)
        previous_proprio = current_proprio.clone()
        predicate_tensor = semantic_probabilities(
            semantic,
            previous_latent,
            current_latent,
            previous_proprio,
            current_proprio,
        )
        predicates = predicate_tensor.cpu().numpy()
        machine = LearnedRecoveryStateMachine(machine_settings)
        frames = [current_image]
        state_history = [machine.state]
        replay_accepts = 0
        actor_updates = 0
        environment_step = 0
        planning_cycle = 0
        terminated = False
        truncated = False
        oracle_success = False

        while environment_step < int(settings["max_steps"]):
            state_before_plan = machine.state
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            proposal = propose_and_score(
                actor=actor,
                target_net=target_net,
                critic=critic,
                world_model=world_model,
                previous=previous_latent,
                current=current_latent,
                goal=goal_latent,
                transfer_goal=transfer_goal_latent,
                proprio=current_proprio,
                state_id=state_before_plan,
                action_mean=action_mean,
                action_scale=action_scale,
                planner=planner,
                model_settings=config["model"],
                rng=rng,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            planning_seconds = time.perf_counter() - started
            total_world_model_seconds += planning_seconds
            total_world_model_calls += 1
            planning_cycle += 1
            best_index = int(proposal["best_index"])
            scores = proposal["score"]
            planner_advantage = float(
                (scores[best_index] - scores[0]).cpu().item()
            )
            base_plan = proposal["flattened"][0]
            selected_plan = proposal["flattened"][best_index]
            keyframe_target = proposal["keyframe_target"]
            execute_steps = min(
                int(planner["execute_steps_by_state"][state_before_plan]),
                int(selected_plan.shape[0]),
                int(settings["max_steps"]) - environment_step,
            )
            blend = torch.as_tensor(
                planner["action_blend"],
                dtype=selected_plan.dtype,
                device=device,
            )

            for action_index in range(execute_steps):
                state_before = machine.state
                current_target_mse = float(
                    (current_latent - keyframe_target)
                    .square()
                    .mean()
                    .cpu()
                    .item()
                )
                base_action = base_plan[action_index]
                planned_action = selected_plan[action_index]
                executed_action = torch.lerp(
                    base_action, planned_action, blend
                ).clamp(-1.0, 1.0)
                action = executed_action.cpu().numpy()
                next_observation, _, terminated, truncated, info = env.step(action)
                environment_step += 1
                oracle_success = bool(info.get("success", False))
                next_image = np.asarray(env.render()).copy()
                next_latent = base.encode_image(
                    world_model, next_image, transform, device
                )
                next_proprio = torch.from_numpy(
                    robot_proprio(next_observation)
                ).to(device).unsqueeze(0)
                next_predicate_tensor = semantic_probabilities(
                    semantic,
                    current_latent,
                    next_latent,
                    current_proprio,
                    next_proprio,
                )
                next_predicates = next_predicate_tensor.cpu().numpy()
                transition_tensor = transition_probabilities(
                    transition_net,
                    current_latent,
                    next_latent,
                    current_proprio,
                    next_proprio,
                    next_predicate_tensor,
                    state_before,
                )
                transition_probabilities_np = transition_tensor.cpu().numpy()
                transfer_mse = float(
                    (next_latent - transfer_goal_latent)
                    .square()
                    .mean()
                    .cpu()
                    .item()
                )
                transfer_reached = (
                    state_before == TRANSFER
                    and transfer_mse
                    <= float(transfer_settings["latent_mse_threshold"])
                )
                state_transition = machine.observe(
                    next_predicates,
                    transition_probabilities_np,
                    next_proprio[0].cpu().numpy(),
                    transfer_goal_reached=transfer_reached,
                )
                if state_transition.changed:
                    state_history.append(state_transition.current)

                if state_before == TRANSFER:
                    progress = current_target_mse - transfer_mse
                    safe = (
                        next_predicates[GRASPED]
                        >= float(config["state_machine"]["grasp_exit_threshold"])
                    )
                else:
                    predicate_index = (
                        ALIGNED if state_before == ALIGN else GRASPED
                    )
                    progress = float(
                        next_predicates[predicate_index]
                        - predicates[predicate_index]
                    )
                    safe = True
                accepted = (
                    planner_advantage
                    >= float(settings["minimum_planner_advantage"])
                    and progress
                    >= float(settings["minimum_semantic_progress"])
                    and safe
                )
                adaptation_loss = 0.0
                if accepted:
                    replay.append(
                        (
                            previous_latent[0].detach().cpu(),
                            current_latent[0].detach().cpu(),
                            keyframe_target[0].detach().cpu(),
                            goal_latent[0].detach().cpu(),
                            torch.tensor(state_before, dtype=torch.long),
                            current_proprio[0].detach().cpu(),
                            proposal["elite_plan"][action_index].detach().cpu(),
                        )
                    )
                    replay_accepts += 1
                    if len(replay) >= int(settings["warmup_samples"]):
                        adaptation_loss = update_actor_online(
                            actor,
                            anchor_actor,
                            optimizer,
                            replay,
                            settings,
                            rng,
                            device,
                        )
                        actor_updates += 1

                record = {
                    "mode": "proprio_v2_online",
                    "episode": episode,
                    "environment_step": environment_step,
                    "planning_cycle": planning_cycle,
                    "state_before": STATE_NAMES[state_before],
                    "state_after": STATE_NAMES[machine.state],
                    "transition_reason": state_transition.reason,
                    "state_changed": state_transition.changed,
                    "rollback": state_transition.rollback,
                    "predicate_probabilities": {
                        name: float(next_predicates[index])
                        for index, name in enumerate(PREDICATE_NAMES)
                    },
                    "transition_probabilities": {
                        name: float(transition_probabilities_np[index])
                        for index, name in enumerate(STATE_NAMES)
                    },
                    "robot_proprio": next_proprio,
                    "transfer_goal_mse": transfer_mse,
                    "transfer_goal_reached": transfer_reached,
                    "complete": state_transition.complete,
                    "planner_advantage": planner_advantage,
                    "best_score": float(scores[best_index].cpu().item()),
                    "short_target_cost": float(
                        proposal["short_cost"][best_index].cpu().item()
                    ),
                    "progress": progress,
                    "sample_accepted": accepted,
                    "adaptation_loss": adaptation_loss,
                    "executed_action": action,
                    "oracle_success_for_metrics_only": oracle_success,
                    "world_model_inference_seconds": (
                        planning_seconds if action_index == 0 else 0.0
                    ),
                }
                append_jsonl(log_path, record)
                if (
                    environment_step == 1
                    or environment_step % 10 == 0
                    or state_transition.changed
                    or state_transition.complete
                ):
                    print(json.dumps(base.json_value(record)))

                frames.append(next_image)
                previous_latent, current_latent = current_latent, next_latent
                previous_proprio, current_proprio = current_proprio, next_proprio
                predicates = next_predicates
                observation = next_observation
                if (
                    state_transition.changed
                    or state_transition.complete
                    or terminated
                    or truncated
                ):
                    break

            if machine.complete or terminated or truncated:
                break

        success = bool(machine.complete)
        status = "success" if success else "failed"
        video_path = video_dir / f"proprio_v2_episode_{episode}_{status}.mp4"
        save_video(video_path, frames, int(settings["fps"]))
        result = {
            "episode": episode,
            "steps": environment_step,
            "success": success,
            "oracle_success_for_metrics_only": oracle_success,
            "state_history": [STATE_NAMES[index] for index in state_history],
            "recoveries": machine.recoveries,
            "accepted_online_samples": replay_accepts,
            "actor_updates": actor_updates,
            "video": str(video_path),
        }
        episode_results.append(result)
        append_jsonl(log_path, {"episode_summary": result})
        print(json.dumps(result))

    env.close()
    return {
        "mode": "proprio_v2_online",
        "episodes": len(episode_results),
        "successes": int(sum(result["success"] for result in episode_results)),
        "grasp_confirmed_episodes": int(
            sum(TRANSFER in [
                STATE_NAMES.index(name) for name in result["state_history"]
            ] for result in episode_results)
        ),
        "transfer_goal_rgb": str(goal_rgb_path),
        "transfer_goal_metadata": transfer_metadata,
        "world_model_inference_seconds": total_world_model_seconds,
        "world_model_calls": total_world_model_calls,
        "mean_world_model_inference_seconds": (
            total_world_model_seconds / max(1, total_world_model_calls)
        ),
        "episode_results": episode_results,
    }



def task_predicates(
    observation: np.ndarray,
    goal_observation: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    value = semantic_state(observation, goal_observation)
    labels = config["labels"]
    aligned = (
        value.contact < float(labels["contact_threshold"])
        and np.linalg.norm(value.effector[:2] - value.cube[:2])
        <= float(labels["xy_alignment_m"])
        and np.linalg.norm(value.effector - value.cube)
        <= float(labels["xyz_alignment_m"])
    )
    grasped = (
        value.contact >= float(labels["contact_threshold"])
        and value.gripper_opening
        >= float(labels["closed_gripper_threshold"])
    )
    return np.asarray([aligned, grasped], dtype=np.float32)


def collect_task_finetuning_dataset(
    config: dict[str, Any],
    world_model,
    transform,
    device: torch.device,
    path: Path,
    log_path: Path,
) -> dict[str, Any]:
    settings = config["task_finetuning"]
    if path.exists() and bool(settings["reuse_dataset"]):
        with np.load(path) as dataset:
            return {
                "reused": True,
                "samples": int(len(dataset["latents"])),
                "episodes": int(len(np.unique(dataset["episode_ids"]))),
                "successful_episodes": int(
                    len(np.unique(dataset["episode_ids"][dataset["successes"] > 0]))
                ),
            }

    env = ogbench.make_env_and_datasets(
        "cube-single-play-v0",
        env_only=True,
        terminate_at_goal=False,
    )
    env._max_episode_steps = int(settings["max_steps"])
    teacher = PrivilegedTeacher()
    teacher_settings = {
        **config["labels"],
        "goal_xyz_m": float(config["transfer_goal"]["goal_xyz_m"]),
        "goal_xy_m": float(config["transfer_goal"]["goal_xyz_m"]),
    }
    rng = np.random.default_rng(
        int(config["seed"]) + int(settings["seed_offset"])
    )
    block = int(config["model"]["action_block"])
    names = (
        "previous_latents",
        "latents",
        "next_latents",
        "goal_latents",
        "keyframe_target_latents",
        "next_keyframe_target_latents",
        "previous_proprios",
        "proprios",
        "next_proprios",
        "action_blocks",
        "transition_inputs",
        "state_ids",
        "next_state_ids",
        "predicate_targets",
        "rewards",
        "dones",
        "episode_ids",
        "successes",
    )
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in names}
    log_path.unlink(missing_ok=True)
    successful_episodes = 0
    attempt = 0
    requested_episodes = int(settings["episodes"])
    max_attempts = int(
        settings.get("max_collection_attempts", requested_episodes * 2)
    )

    while successful_episodes < requested_episodes:
        if attempt >= max_attempts:
            raise RuntimeError(
                "Could not collect the requested number of successful expert "
                f"trajectories: {successful_episodes}/{requested_episodes} "
                f"after {attempt} attempts."
            )
        attempt_index = attempt
        attempt += 1
        observation, reset_info = env.reset(
            seed=(
                int(config["seed"])
                + int(settings["seed_offset"])
                + attempt_index
            ),
            options=base.task_reset_options(config),
        )
        goal_observation = np.asarray(reset_info["goal"], dtype=np.float32)
        goal_image = np.asarray(reset_info["goal_rendered"]).copy()
        frames = [np.asarray(env.render()).copy()]
        observations = [np.asarray(observation, dtype=np.float32).copy()]
        actions: list[np.ndarray] = []
        stable_success = 0

        for _ in range(int(settings["max_steps"])):
            value = semantic_state(observation, goal_observation)
            predicates = task_predicates(
                observation, goal_observation, config
            )
            near_goal = (
                np.linalg.norm(value.cube - value.goal)
                <= float(config["transfer_goal"]["goal_xyz_m"])
            )
            stable_success = (
                stable_success + 1
                if predicates[GRASPED] >= 0.5 and near_goal
                else 0
            )
            if stable_success >= 1:
                break
            ideal_action, _ = teacher.action(
                observation, goal_observation, teacher_settings
            )
            noise = rng.normal(
                0.0,
                float(settings["action_noise_std"]),
                size=ideal_action.shape,
            ).astype(np.float32)
            noise[4] = 0.0
            executed_action = np.clip(
                ideal_action + noise, -1.0, 1.0
            ).astype(np.float32)
            actions.append(np.asarray(ideal_action, dtype=np.float32))
            observation, _, terminated, truncated, _ = env.step(
                executed_action
            )
            frames.append(np.asarray(env.render()).copy())
            observations.append(
                np.asarray(observation, dtype=np.float32).copy()
            )
            if terminated or truncated:
                break

        predicate_targets = np.asarray(
            [
                task_predicates(value, goal_observation, config)
                for value in observations
            ],
            dtype=np.float32,
        )
        proprios = robot_proprio(np.asarray(observations))
        state_ids = state_ids_from_predicates(
            predicate_targets,
            proprios,
            float(config["labels"]["closed_gripper_threshold"]),
        )
        final_value = semantic_state(observations[-1], goal_observation)
        success = bool(
            predicate_targets[-1, GRASPED] >= 0.5
            and np.linalg.norm(final_value.cube - final_value.goal)
            <= float(config["transfer_goal"]["goal_xyz_m"])
        )
        if not success:
            record = {
                "attempt": attempt_index,
                "episode": None,
                "frames": len(frames),
                "success": False,
                "retained": False,
                "state_counts": {
                    STATE_NAMES[state]: int(np.sum(state_ids == state))
                    for state in range(len(STATE_NAMES))
                },
            }
            append_jsonl(log_path, record)
            print(json.dumps(record))
            continue

        episode = successful_episodes
        successful_episodes += 1
        latents = base.encode_images(
            world_model, frames, transform, device
        )
        goal_latent = base.encode_images(
            world_model, [goal_image], transform, device
        )[0]
        terminal = len(frames) - 1

        align_endpoint = np.full(len(frames), terminal, dtype=np.int64)
        grasp_endpoint = np.full(len(frames), terminal, dtype=np.int64)
        next_align = terminal
        next_grasp = terminal
        for index in range(terminal, -1, -1):
            if predicate_targets[index, GRASPED] >= 0.5:
                next_grasp = index
                next_align = index
            elif predicate_targets[index, ALIGNED] >= 0.5:
                next_align = index
            align_endpoint[index] = next_align
            grasp_endpoint[index] = next_grasp

        targets = np.empty_like(latents)
        for index, state_id in enumerate(state_ids):
            if state_id == ALIGN:
                targets[index] = latents[align_endpoint[index]]
            elif state_id == GRASP:
                targets[index] = latents[grasp_endpoint[index]]
            else:
                targets[index] = latents[terminal]

        for index in range(len(frames)):
            next_index = min(index + 1, terminal)
            state_id = int(state_ids[index])
            next_state_id = int(state_ids[next_index])
            reward = -float(config["offline_training"]["step_penalty"])
            if next_state_id > state_id:
                reward += float(
                    config["offline_training"]["state_transition_reward"]
                )
            elif next_state_id < state_id:
                reward -= float(
                    config["offline_training"]["recovery_penalty"]
                )
            arrays["previous_latents"].append(latents[max(0, index - 1)])
            arrays["latents"].append(latents[index])
            arrays["next_latents"].append(latents[next_index])
            arrays["goal_latents"].append(goal_latent)
            arrays["keyframe_target_latents"].append(targets[index])
            arrays["next_keyframe_target_latents"].append(targets[next_index])
            arrays["previous_proprios"].append(
                proprios[max(0, index - 1)]
            )
            arrays["proprios"].append(proprios[index])
            arrays["next_proprios"].append(proprios[next_index])
            action_block = np.zeros((block, 5), dtype=np.float32)
            available = actions[index : index + block]
            if available:
                action_block[: len(available)] = np.asarray(available)
                if len(available) < block:
                    action_block[len(available) :] = available[-1]
            arrays["action_blocks"].append(action_block.reshape(-1))
            arrays["transition_inputs"].append(
                np.asarray(
                    state_ids[max(0, index - 1)], dtype=np.int64
                )
            )
            arrays["state_ids"].append(
                np.asarray(state_id, dtype=np.int64)
            )
            arrays["next_state_ids"].append(
                np.asarray(next_state_id, dtype=np.int64)
            )
            arrays["predicate_targets"].append(predicate_targets[index])
            arrays["rewards"].append(np.asarray(reward, dtype=np.float32))
            arrays["dones"].append(
                np.asarray(index == terminal, dtype=np.float32)
            )
            arrays["episode_ids"].append(
                np.asarray(episode, dtype=np.int64)
            )
            arrays["successes"].append(
                np.asarray(success, dtype=np.int64)
            )

        record = {
            "attempt": attempt_index,
            "episode": episode,
            "frames": len(frames),
            "success": success,
            "retained": True,
            "state_counts": {
                STATE_NAMES[state]: int(np.sum(state_ids == state))
                for state in range(len(STATE_NAMES))
            },
        }
        append_jsonl(log_path, record)
        print(json.dumps(record))

    env.close()
    if successful_episodes != requested_episodes:
        raise RuntimeError(
            "Expert collection count mismatch: "
            f"{successful_episodes}/{requested_episodes}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{name: np.asarray(values) for name, values in arrays.items()}
    )
    return {
        "reused": False,
        "samples": len(arrays["latents"]),
        "episodes": requested_episodes,
        "successful_episodes": successful_episodes,
        "collection_attempts": attempt,
    }


def task_dataset_splits(
    config: dict[str, Any], path: Path
) -> tuple[TensorDataset, TensorDataset]:
    names = (
        "previous_latents",
        "latents",
        "next_latents",
        "goal_latents",
        "keyframe_target_latents",
        "next_keyframe_target_latents",
        "previous_proprios",
        "proprios",
        "next_proprios",
        "action_blocks",
        "transition_inputs",
        "state_ids",
        "next_state_ids",
        "predicate_targets",
        "rewards",
        "dones",
    )
    with np.load(path) as dataset:
        arrays = [np.asarray(dataset[name]) for name in names]
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(int(config["seed"]) + 91)
    rng.shuffle(episodes)
    count = max(
        1,
        int(
            round(
                len(episodes)
                * float(config["task_finetuning"]["validation_fraction"])
            )
        ),
    )
    validation_mask = np.isin(episode_ids, episodes[:count])

    def make(indices: np.ndarray) -> TensorDataset:
        tensors = []
        for column, values in enumerate(arrays):
            dtype = np.int64 if column in (10, 11, 12) else np.float32
            tensors.append(torch.from_numpy(values[indices].astype(dtype)))
        return TensorDataset(*tensors)

    return make(np.flatnonzero(~validation_mask)), make(
        np.flatnonzero(validation_mask)
    )


def fine_tune_task_models(
    config: dict[str, Any],
    models,
    dataset_path: Path,
    device: torch.device,
    log_path: Path,
) -> dict[str, Any]:
    settings = config["task_finetuning"]
    train_dataset, validation_dataset = task_dataset_splits(
        config, dataset_path
    )
    target_critic = copy.deepcopy(models[-1]).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for model in models for parameter in model.parameters()],
        lr=float(settings["learning_rate"]),
        weight_decay=float(config["offline_training"]["weight_decay"]),
    )
    predicates = train_dataset.tensors[13].numpy()
    positive = predicates.sum(axis=0)
    negative = len(predicates) - positive
    positive_weights = torch.from_numpy(
        np.clip(
            negative / np.maximum(positive, 1.0),
            1.0,
            float(config["offline_training"]["max_positive_weight"]),
        ).astype(np.float32)
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 92),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )
    log_path.unlink(missing_ok=True)
    best_loss = float("inf")
    best_states = None
    started = time.perf_counter()

    for epoch in range(int(settings["epochs"])):
        for model in models:
            model.train()
        train_records = []
        for batch in train_loader:
            loss, metrics = model_losses(
                models,
                target_critic,
                batch,
                config,
                positive_weights,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for model in models for parameter in model.parameters()],
                float(config["offline_training"]["max_grad_norm"]),
            )
            optimizer.step()
            with torch.no_grad():
                tau = float(config["offline_training"]["target_tau"])
                for target_parameter, parameter in zip(
                    target_critic.parameters(), models[-1].parameters()
                ):
                    target_parameter.lerp_(parameter, tau)
            train_records.append(metrics)
        for model in models:
            model.eval()
        validation_records = []
        with torch.no_grad():
            for batch in validation_loader:
                _, metrics = model_losses(
                    models,
                    target_critic,
                    batch,
                    config,
                    positive_weights,
                    device,
                )
                validation_records.append(metrics)
        record = {
            "epoch": epoch + 1,
            "train": _mean_metrics(train_records),
            "validation": _mean_metrics(validation_records),
        }
        append_jsonl(log_path, record)
        print(json.dumps({"task_finetuning": record}))
        if record["validation"]["loss"] < best_loss:
            best_loss = record["validation"]["loss"]
            best_states = [
                copy.deepcopy(model.state_dict()) for model in models
            ]

    if best_states is not None:
        for model, state in zip(models, best_states):
            model.load_state_dict(state)
    for model in models:
        model.eval()
    return {
        "epochs": int(settings["epochs"]),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_validation_loss": best_loss,
        "training_seconds": time.perf_counter() - started,
    }



def main() -> None:
    args = parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError("--run-name must be a single directory name.")
    config = base.load_config(args.config.resolve())
    if args.smoke:
        apply_smoke_config(config)
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM planning.")

    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    output_dir = config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs" / args.run_name
    video_dir = output_dir / "videos" / args.run_name
    offline_checkpoint = output_dir / "checkpoints" / "proprio_v2_offline.pt"
    final_checkpoint = (
        output_dir / "checkpoints" / f"{args.run_name}_actor_final.pt"
    )
    summary_path = output_dir / f"{args.run_name}_summary.json"
    save_json(output_dir / f"{args.run_name}_resolved_config.json", config)

    transform = base.build_transform(int(config["image_size"]))
    world_model = base.load_world_model(config["paths"]["model_dir"], device)
    action_mean, action_scale = base.load_action_stats(
        config["paths"]["action_dataset"]
    )
    summary: dict[str, Any] = {
        "architecture": {
            "runtime_inputs": [
                "current RGB latent",
                "previous RGB latent",
                "task goal RGB latent",
                "end-effector XYZ",
                "end-effector yaw cos/sin",
                "gripper closure",
            ],
            "offline_label_only": [
                "cube XYZ",
                "goal XYZ",
                "simulator contact",
            ],
            "control_modes": list(STATE_NAMES),
            "transfer_target": (
                "task-specific MuJoCo goal RGB latent; TargetNet is bypassed "
                "after grasp confirmation"
            ),
        }
    }

    task_dataset_path = (
        output_dir / "data" / "proprio_v2_task_continuous.npz"
    )
    trained_models = None
    if args.stage in ("train", "all"):
        collection_summary = collect_task_finetuning_dataset(
            config,
            world_model,
            transform,
            device,
            task_dataset_path,
            log_dir / "task_collection.jsonl",
        )
        trained_models = build_models(config, device)
        training_summary = fine_tune_task_models(
            config,
            trained_models,
            task_dataset_path,
            device,
            log_dir / "complete_trajectory_training.jsonl",
        )
        save_checkpoint(
            offline_checkpoint,
            config,
            trained_models,
            action_mean,
            action_scale,
            {
                "complete_trajectory_collection": collection_summary,
                "complete_trajectory_training": training_summary,
            },
        )
        summary["offline_training"] = {
            "sampling": "none",
            "collection": collection_summary,
            "training": training_summary,
            "checkpoint": str(offline_checkpoint),
        }

    if args.stage in ("adapt", "all"):
        if trained_models is None:
            source_checkpoint = (
                args.resume_checkpoint.expanduser().resolve()
                if args.resume_checkpoint is not None
                else offline_checkpoint
            )
            if not source_checkpoint.exists():
                raise FileNotFoundError(
                    f"v2 checkpoint missing: {source_checkpoint}. "
                    "Run --stage train first."
                )
            trained_models, action_mean, action_scale = load_checkpoint(
                source_checkpoint, config, device
            )
        else:
            source_checkpoint = offline_checkpoint
        for model in trained_models[:-2]:
            model.eval().requires_grad_(False)
        trained_models[-1].eval().requires_grad_(False)
        trained_models[-2].eval()
        summary["online_training"] = run_online(
            config=config,
            models=trained_models,
            world_model=world_model,
            transform=transform,
            device=device,
            action_mean=action_mean,
            action_scale=action_scale,
            log_path=log_dir / "online_training.jsonl",
            video_dir=video_dir,
        )
        save_checkpoint(
            final_checkpoint,
            config,
            trained_models,
            action_mean,
            action_scale,
            {"online_training": summary["online_training"]},
        )
        summary["online_training"]["source_checkpoint"] = str(
            source_checkpoint
        )
        summary["online_training"]["checkpoint"] = str(final_checkpoint)

    save_json(summary_path, summary)
    print(json.dumps(base.json_value(summary), indent=2))


if __name__ == "__main__":
    main()

