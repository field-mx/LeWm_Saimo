from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from torchvision.transforms import v2 as transforms


HERE = Path(__file__).resolve().parent
CUBE_ROOT = HERE.parents[1]
if str(CUBE_ROOT) not in sys.path:
    sys.path.insert(0, str(CUBE_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key, value in config["paths"].items():
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (HERE / candidate).resolve()
        config["paths"][key] = candidate
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the LeWM Cube experiments.")
    return torch.device(name)


def build_image_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def image_to_tensor(
    image: np.ndarray, transform: Callable, device: torch.device
) -> torch.Tensor:
    return transform(np.asarray(image)).unsqueeze(0).unsqueeze(0).to(device)


def load_world_model(model_dir: Path, device: torch.device):
    model = swm.wm.utils.load_pretrained(str(model_dir))
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def synchronized_call(
    function: Callable[[], torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


@torch.inference_mode()
def encode_image(model, pixels: torch.Tensor) -> torch.Tensor:
    return model.encode({"pixels": pixels})["emb"][:, -1]


@torch.inference_mode()
def predict_latent_block(
    model, latent: torch.Tensor, normalized_action_block: torch.Tensor
) -> torch.Tensor:
    action_embedding = model.action_encoder(normalized_action_block.unsqueeze(1))
    return model.predict(latent.unsqueeze(1), action_embedding)[:, -1]


def load_action_stats(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset_path) as dataset:
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(scale, 1e-4)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=True) + "\n")


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=True)
