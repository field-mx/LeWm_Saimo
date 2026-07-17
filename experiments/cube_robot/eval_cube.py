import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def image_transform(cfg: DictConfig):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_dataset(cfg: DictConfig):
    data_root = Path(cfg.paths.data_root)
    expected = data_root / "datasets" / f"{cfg.eval.dataset_name}.h5"
    if not expected.exists():
        raise FileNotFoundError(
            f"Cube dataset not found at {expected}. Run ./prepare_assets.sh first."
        )
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=data_root,
    )




def fit_processors(dataset, keys):
    processors = {}
    for key in keys:
        values = dataset.get_col_data(key)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(values)
        processors[key] = scaler
        if key != "action":
            processors[f"goal_{key}"] = scaler
    return processors


def select_evaluations(dataset, cfg: DictConfig):
    episode_key = next(
        (key for key in ("episode_idx", "ep_idx") if key in dataset.column_names),
        None,
    )
    episodes = np.arange(len(dataset.lengths))
    lengths = np.asarray(dataset.lengths)
    episode_ids = (
        dataset.get_col_data(episode_key)
        if episode_key is not None
        else np.repeat(episodes, lengths)
    )
    max_start = lengths - int(cfg.eval.goal_offset_steps) - 1
    max_start_by_episode = {
        episode: max_start[i] for i, episode in enumerate(episodes)
    }
    row_max_start = np.asarray(
        [max_start_by_episode[episode] for episode in episode_ids]
    )
    valid_rows = np.flatnonzero(
        dataset.get_col_data("step_idx") <= row_max_start
    )

    num_eval = int(cfg.eval.num_eval)
    if len(valid_rows) < num_eval:
        raise ValueError(
            f"Only {len(valid_rows)} valid starts found, but num_eval={num_eval}."
        )

    rng = np.random.default_rng(int(cfg.seed))
    rows = np.sort(rng.choice(valid_rows, size=num_eval, replace=False))
    selected_episodes = (
        dataset.get_col_data(episode_key)[rows]
        if episode_key is not None
        else np.searchsorted(dataset.offsets[1:], rows, side="right")
    )
    print(f"Selected {num_eval} starts from {len(valid_rows)} valid rows.")
    print(f"Dataset row indices: {rows.tolist()}")
    selected_steps = dataset.get_col_data("step_idx")[rows]
    return selected_episodes.tolist(), selected_steps.tolist()


@hydra.main(version_base=None, config_path="./config", config_name="cube")
def run(cfg: DictConfig):
    plan_steps = int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block)
    if plan_steps > int(cfg.eval.eval_budget):
        raise ValueError(
            f"Planning length {plan_steps} exceeds eval budget "
            f"{cfg.eval.eval_budget}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper-scale Cube evaluation.")

    model_dir = Path(cfg.paths.model_dir)
    for filename in ("config.json", "weights.pt"):
        if not (model_dir / filename).exists():
            raise FileNotFoundError(
                f"Cube checkpoint file missing: {model_dir / filename}. "
                "Run ./prepare_assets.sh first."
            )

    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg)
    processors = fit_processors(dataset, cfg.dataset.keys_to_cache)

    model = swm.wm.utils.load_pretrained(
        str(model_dir), cache_dir=cfg.paths.data_root
    )
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=processors,
        transform={
            "pixels": image_transform(cfg),
            "goal": image_transform(cfg),
        },
    )
    world.set_policy(policy)

    episodes, starts = select_evaluations(dataset, cfg)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos" if cfg.eval.save_video else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=video_dir,
    )
    elapsed = time.time() - start_time

    print(metrics)
    print(f"Evaluation time: {elapsed:.2f} seconds")
    if video_dir is not None:
        print(f"Videos saved to {video_dir}")

    results_path = output_dir / cfg.output.filename
    with results_path.open("a", encoding="utf-8") as output:
        output.write("\n==== CONFIG ====\n")
        output.write(OmegaConf.to_yaml(cfg, resolve=True))
        output.write("\n==== RESULTS ====\n")
        output.write(f"metrics: {metrics}\n")
        output.write(f"evaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    run()
