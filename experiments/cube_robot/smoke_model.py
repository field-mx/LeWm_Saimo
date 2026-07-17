import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box


ROOT = Path(__file__).resolve().parent


def preprocess(frame: np.ndarray) -> torch.Tensor:
    stats = spt.data.dataset_stats.ImageNet
    chw = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
    mean = torch.as_tensor(stats["mean"]).view(3, 1, 1)
    std = torch.as_tensor(stats["std"]).view(3, 1, 1)
    chw = (chw - mean) / std
    return chw.unsqueeze(0).unsqueeze(0)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model smoke test.")

    env = gym.make(
        "swm/OGBCube-v0",
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
    )
    env.reset(seed=42)
    current = np.asarray(env.render()).copy()
    for _ in range(5):
        env.step(env.action_space.sample())
    goal = np.asarray(env.render()).copy()

    model = swm.wm.utils.load_pretrained(str(ROOT / "model" / "lewm-cube-mapped"))
    model = model.to("cuda").eval()
    model.requires_grad_(False)

    config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=8,
        n_steps=2,
        topk=2,
        device="cuda",
        seed=42,
    )
    vector_action_space = Box(
        low=-1.0, high=1.0, shape=(1, env.action_space.shape[-1])
    )
    solver.configure(action_space=vector_action_space, n_envs=1, config=config)

    outputs = solver.solve(
        {
            "pixels": preprocess(current),
            "goal": preprocess(goal),
            "action": torch.zeros(1, 1, 25),
        }
    )
    actions = outputs["actions"]
    print(f"actions_shape={tuple(actions.shape)}")
    print(f"elite_cost={outputs['costs'][0]:.6f}")
    print(f"finite_actions={bool(torch.isfinite(actions).all())}")
    env.close()


if __name__ == "__main__":
    main()
