from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import ogbench
import torch

from actor import GaussianActor
from actor_data import actor_observation_from_env, normalize_observations


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "outputs" / "actor_bc" / "actor_best.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "actor_bc" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the behavior-cloned Cube actor.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(
        observation_dim=int(checkpoint["observation_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    actor.load_state_dict(checkpoint["actor"])
    actor = actor.to(args.device).eval()
    observation_mean = np.asarray(checkpoint["observation_mean"], dtype=np.float32)
    observation_scale = np.asarray(checkpoint["observation_scale"], dtype=np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = ogbench.make_env_and_datasets("cube-single-play-v0", env_only=True)
    episode_results = []

    for episode in range(args.episodes):
        state, info = env.reset(seed=args.seed + episode)
        goal = info["goal"]
        frames = []
        success = False

        for step in range(args.max_steps):
            if episode == 0:
                frames.append(np.asarray(env.render()).copy())

            actor_observation = actor_observation_from_env(state, goal)
            actor_observation = normalize_observations(
                actor_observation, observation_mean, observation_scale
            )
            tensor = torch.from_numpy(actor_observation).to(args.device).unsqueeze(0)
            with torch.no_grad():
                action = actor.deterministic_action(tensor)[0].cpu().numpy()

            state, reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("success", False))
            if terminated or truncated:
                break

        episode_results.append(
            {
                "episode": episode,
                "steps": step + 1,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        print(json.dumps(episode_results[-1]))

        if episode == 0 and frames:
            imageio.mimsave(args.output_dir / "actor_episode_0.mp4", frames, fps=20)

    env.close()
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": args.episodes,
        "successes": sum(result["success"] for result in episode_results),
        "success_rate": float(np.mean([result["success"] for result in episode_results])),
        "episode_results": episode_results,
    }
    with (args.output_dir / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
