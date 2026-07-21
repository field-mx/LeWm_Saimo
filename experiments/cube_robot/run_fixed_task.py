import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.envs.ogbench import ExpertPolicy
from stable_worldmodel.plot import save_video
from torchvision.transforms import v2 as transforms


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "datasets" / "fixed_task" / "cube_horizontal_expert.h5"
MODEL_DIR = ROOT / "model" / "lewm-cube-mapped"
OUTPUT_DIR = ROOT / "outputs" / "fixed_task"

SEED = 42
EXPERT_MAX_STEPS = 300 #专家轨迹最多300条
GOAL_OFFSET = 25
EVAL_BUDGET = 50
IMAGE_SIZE = 224

TASK_START = np.array([[0.425, 0.1]], dtype=np.float64)
TASK_GOAL = np.array([[0.425, -0.1]], dtype=np.float64)
TASK_YAW = np.array([0.0], dtype=np.float64)
AGENT_START = np.array([0.425, 0.0, 0.275], dtype=np.float64)


def fixed_task_options():
    return {
        "variation": [],
        "variation_values": {
            "cube.start_position": TASK_START,
            "cube.start_yaw": TASK_YAW,
            "cube.goal_position": TASK_GOAL,
            "cube.goal_yaw": TASK_YAW,
            "agent.ee_start_position": AGENT_START,
        },
    }


def make_world(*, mode="task", max_episode_steps=100):
    return swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=max_episode_steps,
        env_type="single",
        ob_type="states",
        mode=mode,
        multiview=False,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
    )


def flatten_hdf5_groups(path):
    """Convert collector groups to the flat columns used by evaluation."""
    with h5py.File(path, "r+") as dataset:
        for group_name in ("privileged", "proprio"):
            group = dataset.get(group_name)
            if not isinstance(group, h5py.Group):
                continue
            for name, values in group.items():
                flat_name = f"{group_name}_{name}"
                if flat_name not in dataset:
                    dataset.create_dataset(flat_name, data=values[:])
            del dataset[group_name]


def collect_expert(regenerate=False):
    if regenerate and DATASET_PATH.exists():
        DATASET_PATH.unlink()

    if not DATASET_PATH.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.random.seed(SEED)
        world = make_world(mode="data_collection", max_episode_steps=EXPERT_MAX_STEPS)
        world.set_policy(
            ExpertPolicy(
                policy_type="markov_oracle",
                action_noise=0.0,
                p_random_action=0.0,
                seed=SEED,
            )
        )
        world.collect(
            path=DATASET_PATH,
            episodes=1,
            seed=SEED,
            options=fixed_task_options(),
            format="hdf5",
            progress=True,
        )
        world.close()

    flatten_hdf5_groups(DATASET_PATH)

    with h5py.File(DATASET_PATH, "r") as dataset:
        expert_steps = int(dataset["ep_len"][0])
        expert_success = bool(np.asarray(dataset["success"][-1]).item())
        frames = [frame.copy() for frame in dataset["pixels"][:expert_steps]]

    if expert_steps <= GOAL_OFFSET:
        raise RuntimeError(
            f"Expert task has only {expert_steps} steps; at least {GOAL_OFFSET + 1} are required."
        )
    if not expert_success:
        raise RuntimeError("The generated oracle trajectory did not complete the Cube task.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_video(OUTPUT_DIR / "expert_full.mp4", frames)
    return expert_steps


def image_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=IMAGE_SIZE),
        ]
    )


def crop_agent_video(panel_path, output_path):
    reader = imageio.get_reader(panel_path)
    pad = max(12, IMAGE_SIZE // 14)
    frames = [
        np.asarray(frame)[pad : pad + IMAGE_SIZE, pad : pad + IMAGE_SIZE].copy()
        for frame in reader
    ]
    reader.close()
    save_video(output_path, frames)


def run_agent(expert_steps):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper-configuration CEM evaluation.")

    dataset = swm.data.HDF5Dataset(
        path=DATASET_PATH,
        keys_to_cache=["action"],
    )
    action_processor = StandardScaler().fit(dataset.get_col_data("action"))

    model = swm.wm.utils.load_pretrained(str(MODEL_DIR))
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(
        horizon=5,
        receding_horizon=5,
        action_block=5,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=300,  # 每轮采样300条候选序列
        var_scale=1.0,    # 初始采样方差
        n_steps=30,       # CEM优化30轮
        topk=50,          # 每轮保留最优30条
        device="cuda",
        seed=SEED,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process={"action": action_processor},
        transform={"pixels": image_transform(), "goal": image_transform()},
    )

    world = make_world(max_episode_steps=2 * EVAL_BUDGET)
    world.set_policy(policy)
    panel_dir = OUTPUT_DIR / "agent_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=[0],
        start_steps=[0],
        goal_offset=GOAL_OFFSET,
        eval_budget=EVAL_BUDGET,
        callables=[
            {
                "method": "set_state",
                "args": {
                    "qpos": {"value": "qpos"},
                    "qvel": {"value": "qvel"},
                },
            },
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged_block_0_pos"},
                    "target_quat": {"value": "goal_privileged_block_0_quat"},
                },
            },
        ],
        video=panel_dir,
    )
    world.close()

    generated_panel = panel_dir / "env_0.mp4"
    final_panel = OUTPUT_DIR / "agent_short_panel.mp4"
    generated_panel.replace(final_panel)
    crop_agent_video(final_panel, OUTPUT_DIR / "agent_short.mp4")

    result = {
        "task": "single-cube horizontal pick-and-place",
        "seed": SEED,
        "cube_start_xy": TASK_START[0].tolist(),
        "cube_final_goal_xy": TASK_GOAL[0].tolist(),
        "expert_steps": expert_steps,
        "agent_start_step": 0,
        "agent_goal_expert_step": GOAL_OFFSET,
        "agent_eval_budget": EVAL_BUDGET,
        "plan_config": {
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
        },
        "cem": {
            "num_samples": 300,
            "n_steps": 30,
            "topk": 30,
            "var_scale": 1.0,
        },
        "agent_success": bool(metrics["episode_successes"][0]),
        "success_rate": float(metrics["success_rate"]),
        "action_mean": action_processor.mean_.tolist(),
        "action_scale": action_processor.scale_.tolist(),
    }
    with (OUTPUT_DIR / "fixed_task_results.json").open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-expert",
        action="store_true",
        help="Regenerate the deterministic oracle trajectory before inference.",
    )
    args = parser.parse_args()

    expert_steps = collect_expert(regenerate=args.regenerate_expert)
    result = run_agent(expert_steps)
    print(json.dumps(result, indent=2))
    print(f"Expert video: {OUTPUT_DIR / 'expert_full.mp4'}")
    print(f"Agent video:  {OUTPUT_DIR / 'agent_short.mp4'}")
    print(f"Panel video:  {OUTPUT_DIR / 'agent_short_panel.mp4'}")


if __name__ == "__main__":
    main()
