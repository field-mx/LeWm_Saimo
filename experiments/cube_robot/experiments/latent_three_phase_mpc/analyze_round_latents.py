from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ogbench
import torch

import run_recovery_experiment as recovery


HERE = Path(__file__).resolve().parent
STATE_NAMES = recovery.STATE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RGB latent distances for one recovery MPC run."
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "recovery_config.yaml"
    )
    parser.add_argument(
        "--run-name",
        default="recovery_state_machine_continue_round2",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def read_online_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("mode") != "recovery_online_train":
                continue
            episodes[int(row["episode"])].append(row)
    for rows in episodes.values():
        rows.sort(key=lambda row: int(row["environment_step"]))
    return dict(episodes)


def read_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame).copy() for frame in reader]
    finally:
        reader.close()


@torch.inference_mode()
def phase_targets_and_mse(
    target_net,
    latents: np.ndarray,
    goal_latent: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    current = torch.from_numpy(latents).to(device)
    previous_np = np.concatenate([latents[:1], latents[:-1]], axis=0)
    previous = torch.from_numpy(previous_np).to(device)
    goal = torch.from_numpy(
        np.repeat(goal_latent[None], len(latents), axis=0)
    ).to(device)
    targets = []
    distances = []
    for state_id in range(len(STATE_NAMES)):
        state = torch.full(
            (len(latents),), state_id, dtype=torch.long, device=device
        )
        target = target_net(previous, current, goal, state)
        targets.append(target.cpu().numpy().astype(np.float32))
        distances.append(
            (current - target).square().mean(dim=-1).cpu().numpy()
        )
    return (
        np.stack(targets, axis=1).astype(np.float32),
        np.stack(distances, axis=1).astype(np.float32),
    )


@torch.inference_mode()
def world_model_prediction_errors(
    world_model,
    latents: np.ndarray,
    actions: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    block: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(actions) - block + 1, len(latents) - block)
    if count <= 0:
        return (
            np.empty((0, latents.shape[-1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    action_blocks = np.stack(
        [actions[start : start + block] for start in range(count)], axis=0
    )
    normalized = (
        (action_blocks - action_mean[None, None])
        / action_scale[None, None]
    ).reshape(count, -1)
    predicted = recovery.base.predict_latent_block(
        world_model,
        torch.from_numpy(latents[:count]).to(device),
        torch.from_numpy(normalized.astype(np.float32)).to(device),
    )
    predicted_np = predicted.cpu().numpy().astype(np.float32)
    target_indices = np.arange(block, block + count, dtype=np.int64)
    mse = np.mean(
        (predicted_np - latents[target_indices]) ** 2, axis=-1
    ).astype(np.float32)
    return predicted_np, target_indices, mse


def frame_states(rows: list[dict[str, Any]], frame_count: int) -> list[str]:
    if not rows:
        return ["unknown"] * frame_count
    states = [str(rows[0]["state_before"])]
    states.extend(str(row["state_after"]) for row in rows)
    if len(states) < frame_count:
        states.extend([states[-1]] * (frame_count - len(states)))
    return states[:frame_count]


def write_episode_csv(
    path: Path,
    states: list[str],
    phase_mse: np.ndarray,
    goal_mse: np.ndarray,
    wm_target_indices: np.ndarray,
    wm_mse: np.ndarray,
) -> None:
    wm_by_frame = {
        int(frame): float(error)
        for frame, error in zip(wm_target_indices, wm_mse)
    }
    fields = ["frame_index", "state", "goal_latent_mse"]
    fields.extend(f"target_{name}_mse" for name in STATE_NAMES)
    fields.extend(["wm_predicted_from_frame", "wm_to_encoder_mse"])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        block = int(wm_target_indices[0]) if len(wm_target_indices) else 0
        for frame_index in range(len(states)):
            row: dict[str, Any] = {
                "frame_index": frame_index,
                "state": states[frame_index],
                "goal_latent_mse": float(goal_mse[frame_index]),
                "wm_predicted_from_frame": (
                    frame_index - block if frame_index in wm_by_frame else ""
                ),
                "wm_to_encoder_mse": wm_by_frame.get(frame_index, ""),
            }
            for state_id, name in enumerate(STATE_NAMES):
                row[f"target_{name}_mse"] = float(
                    phase_mse[frame_index, state_id]
                )
            writer.writerow(row)


def plot_episode(
    path: Path,
    episode: int,
    success: bool,
    states: list[str],
    phase_mse: np.ndarray,
    wm_target_indices: np.ndarray,
    wm_mse: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    frames = np.arange(len(states))
    state_array = np.asarray(states)
    for state_id, name in enumerate(STATE_NAMES):
        current_target_mse = np.where(
            state_array == name,
            phase_mse[:, state_id],
            np.nan,
        )
        axes[0].plot(
            frames,
            current_target_mse,
            linewidth=1.5,
            label=name,
        )
    axes[0].set_ylabel("MSE to current phase target latent")
    axes[0].set_title(
        f"Episode {episode}: RGB latent to current phase target "
        f"({'success' if success else 'failed'})"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=len(STATE_NAMES), fontsize=8)

    axes[1].plot(
        wm_target_indices,
        wm_mse,
        color="black",
        linewidth=1.4,
        label="LeWM predicted vs encoded RGB latent",
    )
    axes[1].set_xlabel("Target video frame")
    axes[1].set_ylabel("5-action prediction MSE")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    for frame_index in range(1, len(states)):
        if states[frame_index] == states[frame_index - 1]:
            continue
        for axis in axes:
            axis.axvline(frame_index, color="gray", alpha=0.25, linewidth=0.8)
        axes[0].text(
            frame_index,
            axes[0].get_ylim()[1],
            states[frame_index],
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(
    path: Path,
    phase_curves: list[np.ndarray],
    state_curves: list[list[str]],
    wm_curves: list[np.ndarray],
) -> None:
    progress = np.linspace(0.0, 1.0, 101)

    def interpolate(values: np.ndarray) -> np.ndarray:
        source = np.linspace(0.0, 1.0, len(values))
        return np.interp(progress, source, values)

    active_stack = np.stack(
        [
            interpolate(
                np.asarray(
                    [
                        curve[frame, STATE_NAMES.index(state)]
                        for frame, state in enumerate(states)
                    ],
                    dtype=np.float32,
                )
            )
            for curve, states in zip(phase_curves, state_curves)
        ]
    )
    wm_stack = np.stack([interpolate(curve) for curve in wm_curves], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    active_mean = active_stack.mean(axis=0)
    active_std = active_stack.std(axis=0)
    axes[0].plot(progress, active_mean, color="tab:blue")
    axes[0].fill_between(
        progress,
        active_mean - active_std,
        active_mean + active_std,
        color="tab:blue",
        alpha=0.12,
    )
    axes[0].set_ylabel("Mean current-target MSE")
    axes[0].grid(alpha=0.25)

    wm_mean = wm_stack.mean(axis=0)
    wm_std = wm_stack.std(axis=0)
    axes[1].plot(progress, wm_mean, color="black")
    axes[1].fill_between(
        progress, wm_mean - wm_std, wm_mean + wm_std, color="black", alpha=0.12
    )
    axes[1].set_xlabel("Normalized episode progress")
    axes[1].set_ylabel("Mean 5-action prediction MSE")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Ten-episode latent analysis (mean +/- std)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = recovery.base.load_config(args.config.resolve())
    output_root = config["paths"]["output_dir"]
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else output_root / f"{args.run_name}_summary.json"
    )
    log_path = (
        args.log.expanduser().resolve()
        if args.log is not None
        else output_root / "logs" / args.run_name / "online_training.jsonl"
    )
    summary = read_json(summary_path)["online_training"]
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else Path(summary["checkpoint"]).expanduser().resolve()
    )
    analysis_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else output_root / "analysis" / f"{args.run_name}_latent_analysis"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM latent analysis.")
    transform = recovery.base.build_transform(int(config["image_size"]))
    world_model = recovery.base.load_world_model(
        config["paths"]["model_dir"], device
    )
    (
        _,
        target_net,
        _,
        _,
        action_mean,
        action_scale,
    ) = recovery.load_checkpoint(checkpoint_path, config, device)
    target_net.eval().requires_grad_(False)

    rows_by_episode = read_online_rows(log_path)
    results = {
        int(result["episode"]): result for result in summary["episode_results"]
    }
    env = ogbench.make_env_and_datasets(
        "cube-single-play-v0", env_only=True, terminate_at_goal=False
    )
    summary_rows = []
    phase_curves = []
    state_curves = []
    wm_curves = []
    block = int(config["model"]["action_block"])

    for episode in sorted(results):
        result = results[episode]
        video_path = Path(result["video"])
        rows = rows_by_episode[episode]
        frames = read_video(video_path)
        expected_frames = len(rows) + 1
        if len(frames) != expected_frames:
            raise ValueError(
                f"Episode {episode}: video has {len(frames)} frames, "
                f"but log implies {expected_frames}."
            )
        _, reset_info = env.reset(
            seed=int(config["seed"])
            + int(config["online_training"]["seed_offset"])
            + episode,
            options=recovery.base.task_reset_options(config),
        )
        goal_image = np.asarray(reset_info["goal_rendered"]).copy()
        latents = recovery.base.encode_images(
            world_model,
            frames,
            transform,
            device,
            batch_size=int(args.batch_size),
        )
        goal_latent = recovery.base.encode_images(
            world_model, [goal_image], transform, device
        )[0]
        targets, phase_mse = phase_targets_and_mse(
            target_net, latents, goal_latent, device
        )
        goal_mse = np.mean((latents - goal_latent[None]) ** 2, axis=-1)
        actions = np.asarray(
            [row["executed_action"] for row in rows], dtype=np.float32
        )
        predicted, target_indices, wm_mse = world_model_prediction_errors(
            world_model,
            latents,
            actions,
            action_mean,
            action_scale,
            block,
            device,
        )
        states = frame_states(rows, len(frames))
        prefix = analysis_dir / f"episode_{episode:02d}"
        write_episode_csv(
            prefix.with_suffix(".csv"),
            states,
            phase_mse,
            goal_mse,
            target_indices,
            wm_mse,
        )
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            frame_latents=latents,
            goal_latent=goal_latent,
            phase_target_latents=targets,
            phase_target_mse=phase_mse,
            actions=actions,
            wm_predicted_latents=predicted,
            wm_target_frame_indices=target_indices,
            wm_prediction_mse=wm_mse,
            state_names=np.asarray(STATE_NAMES),
            frame_states=np.asarray(states),
        )
        plot_episode(
            prefix.with_suffix(".png"),
            episode,
            bool(result["success"]),
            states,
            phase_mse,
            target_indices,
            wm_mse,
        )
        row: dict[str, Any] = {
            "episode": episode,
            "success": bool(result["success"]),
            "frames": len(frames),
            "wm_prediction_count": len(wm_mse),
            "wm_mse_mean": float(wm_mse.mean()),
            "wm_mse_median": float(np.median(wm_mse)),
            "wm_mse_p95": float(np.quantile(wm_mse, 0.95)),
            "wm_mse_max": float(wm_mse.max()),
        }
        for state_id, name in enumerate(STATE_NAMES):
            row[f"{name}_target_mse_mean"] = float(
                phase_mse[:, state_id].mean()
            )
            row[f"{name}_target_mse_final"] = float(
                phase_mse[-1, state_id]
            )
        summary_rows.append(row)
        phase_curves.append(phase_mse)
        state_curves.append(states)
        wm_curves.append(wm_mse)
        print(json.dumps(row))

    env.close()
    write_summary_csv(analysis_dir / "episode_summary.csv", summary_rows)
    plot_aggregate(
        analysis_dir / "all_episodes_mean.png",
        phase_curves,
        state_curves,
        wm_curves,
    )
    metadata = {
        "run_name": args.run_name,
        "summary": str(summary_path),
        "log": str(log_path),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(analysis_dir),
        "episodes": len(summary_rows),
        "prediction_action_block": block,
        "phase_targets": list(STATE_NAMES),
        "note": (
            "Frame latents come from decoded MP4 frames. World-model errors "
            "compare a t+5 prediction from five logged actions with the "
            "encoder latent of decoded video frame t+5."
        ),
    }
    with (analysis_dir / "analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(metadata, output, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
