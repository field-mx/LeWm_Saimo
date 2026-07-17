import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import stable_worldmodel as swm


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "datasets" / "smoke" / "cube_random.h5"


def flatten_collected_groups():
    """Match the flat privileged-field layout used by the official dataset."""
    with h5py.File(OUTPUT, "r+") as dataset:
        privileged = dataset.get("privileged")
        if isinstance(privileged, h5py.Group):
            for name, values in privileged.items():
                flat_name = f"privileged_{name}"
                if flat_name not in dataset:
                    dataset.create_dataset(flat_name, data=values[:])

        for name in list(dataset.keys()):
            if isinstance(dataset[name], h5py.Group):
                del dataset[name]


def main():
    if OUTPUT.exists():
        flatten_collected_groups()
        print(f"Synthetic smoke dataset already exists: {OUTPUT}")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=40,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        image_shape=(224, 224),
    )
    world.set_policy(swm.policy.RandomPolicy(seed=42))
    world.collect(
        path=OUTPUT,
        episodes=2,
        seed=42,
        format="hdf5",
        progress=False,
    )
    flatten_collected_groups()
    print(f"Synthetic smoke dataset created: {OUTPUT}")


if __name__ == "__main__":
    main()
