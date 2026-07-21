from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from actor import GaussianActor
from actor_data import ACTION_DIM, ACTOR_OBSERVATION_DIM, load_demonstrations


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "ogbench_state" / "cube-single-play-v0.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "actor_bc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Cube actor by behavior cloning.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-goal-offset",
        type=int,
        default=100,
        help="Largest future-state offset used as a goal for NPZ trajectories.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def split_by_episode(
    episode_ids: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        indices = np.arange(len(episode_ids))
        split = max(1, int(round(len(indices) * (1.0 - validation_fraction))))
        return indices[:split], indices[split:]

    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_episodes = episodes[:validation_count]
    validation_mask = np.isin(episode_ids, validation_episodes)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


@torch.no_grad()
def evaluate(
    actor: GaussianActor, loader: DataLoader, device: torch.device
) -> float:
    actor.eval()
    squared_error = 0.0
    element_count = 0
    for observations, actions in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predictions = actor.deterministic_action(observations)
        squared_error += (predictions - actions).square().sum().item()
        element_count += actions.numel()
    return squared_error / max(1, element_count)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}. Download cube-single-play-v0 first."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    observations, actions, episode_ids = load_demonstrations(
        args.dataset,
        max_goal_offset=args.max_goal_offset,
        seed=args.seed,
    )
    train_indices, validation_indices = split_by_episode(
        episode_ids, args.validation_fraction, args.seed
    )
    if len(validation_indices) == 0:
        validation_indices = train_indices[-max(1, len(train_indices) // 10) :]
        train_indices = train_indices[: -len(validation_indices)]

    observation_mean = observations[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    observation_scale = observations[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    observation_scale = np.maximum(observation_scale, 1e-4)
    observations = ((observations - observation_mean) / observation_scale).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(observations[train_indices]),
        torch.from_numpy(actions[train_indices]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(observations[validation_indices]),
        torch.from_numpy(actions[validation_indices]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    actor = GaussianActor(
        observation_dim=ACTOR_OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    criterion = nn.MSELoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        actor.train()
        total_loss = 0.0
        sample_count = 0
        for batch_observations, batch_actions in train_loader:
            batch_observations = batch_observations.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            predictions = actor.deterministic_action(batch_observations)
            loss = criterion(predictions, batch_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_actions)
            sample_count += len(batch_actions)

        train_mse = total_loss / max(1, sample_count)
        validation_mse = evaluate(actor, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "actor": actor.state_dict(),
            "observation_mean": observation_mean,
            "observation_scale": observation_scale,
            "observation_dim": ACTOR_OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "hidden_dim": args.hidden_dim,
            "max_goal_offset": args.max_goal_offset,
            "dataset": str(args.dataset.resolve()),
            "epoch": epoch,
            "validation_mse": validation_mse,
        }
        torch.save(checkpoint, args.output_dir / "actor_last.pt")
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            torch.save(checkpoint, args.output_dir / "actor_best.pt")

    summary = {
        "dataset": str(args.dataset.resolve()),
        "num_samples": len(observations),
        "num_episodes": int(np.unique(episode_ids).size),
        "num_train_samples": len(train_indices),
        "num_validation_samples": len(validation_indices),
        "best_validation_mse": best_validation_mse,
        "device": str(device),
        "history": history,
    }
    with (args.output_dir / "training_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
