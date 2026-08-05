from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from run_experiment import (
    HERE,
    build_transform,
    json_value,
    load_checkpoint,
    load_config,
    load_world_model,
    run_closed_loop,
    save_json,
    set_seed,
)


class ZeroCritic(nn.Module):
    """Preserve the planner interface while removing learned value estimates."""

    def forward(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        target: torch.Tensor,
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        del previous, target, goal, phase
        return current.new_zeros((2, current.shape[0]))


def main() -> None:
    config = copy.deepcopy(load_config(HERE / "config.yaml"))
    config["evaluation"]["episodes"] = 4

    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LeWM planning.")

    set_seed(int(config["seed"]))
    device = torch.device(config["device"])
    output_dir: Path = config["paths"]["output_dir"]
    checkpoint = output_dir / "checkpoints" / "three_phase_actor_final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Successful actor checkpoint not found: {checkpoint}"
        )

    world_model = load_world_model(config["paths"]["model_dir"], device)
    transform = build_transform(int(config["image_size"]))
    (
        phase_net,
        target_net,
        actor,
        _trained_critic,
        action_mean,
        action_scale,
    ) = load_checkpoint(checkpoint, config, device)
    zero_critic = ZeroCritic().to(device).eval().requires_grad_(False)
    for model in (phase_net, target_net, actor):
        model.eval().requires_grad_(False)

    video_dir = output_dir / "videos" / "no_critic"
    result = run_closed_loop(
        config=config,
        phase_net=phase_net,
        target_net=target_net,
        actor=actor,
        critic=zero_critic,
        world_model=world_model,
        transform=transform,
        device=device,
        action_mean=action_mean,
        action_scale=action_scale,
        training=False,
        log_path=output_dir / "logs" / "no_critic_evaluation.jsonl",
        selected_video_path=video_dir / "no_critic_frozen_selected.mp4",
    )
    summary = {
        "experiment": "successful_actor_without_learned_critic",
        "checkpoint": str(checkpoint),
        "critic": "ZeroCritic: value=0 and uncertainty=0 for every state",
        "evaluation": result,
    }
    save_json(output_dir / "no_critic_summary.json", summary)
    print(json.dumps(json_value(summary), indent=2))


if __name__ == "__main__":
    main()
