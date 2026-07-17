import json
import shutil
from pathlib import Path

import hydra
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model" / "lewm-cube"
DESTINATION = ROOT / "model" / "lewm-cube-mapped"


def rename_encoder_key(key: str) -> str:
    if not key.startswith("encoder.encoder.layer."):
        return key
    return (
        key.replace("encoder.encoder.layer.", "encoder.layers.")
        .replace(".attention.attention.query.", ".attention.q_proj.")
        .replace(".attention.attention.key.", ".attention.k_proj.")
        .replace(".attention.attention.value.", ".attention.v_proj.")
        .replace(".attention.output.dense.", ".attention.o_proj.")
        .replace(".intermediate.dense.", ".mlp.fc1.")
        .replace(".output.dense.", ".mlp.fc2.")
    )


def main():
    source_weights = SOURCE / "weights.pt"
    source_config = SOURCE / "config.json"
    if not source_weights.exists() or not source_config.exists():
        raise FileNotFoundError("Official Cube checkpoint files are incomplete.")

    state_dict = torch.load(source_weights, map_location="cpu", weights_only=False)
    mapped = {rename_encoder_key(key): value for key, value in state_dict.items()}

    DESTINATION.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, DESTINATION / "config.json")
    torch.save(mapped, DESTINATION / "weights.pt")

    config = json.loads(source_config.read_text(encoding="utf-8"))
    model = hydra.utils.instantiate(config)
    model.load_state_dict(mapped, strict=True)
    print(f"Mapped and validated {len(mapped)} tensors at {DESTINATION}")


if __name__ == "__main__":
    main()
