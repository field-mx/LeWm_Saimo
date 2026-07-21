#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_BIN="/publicworkspace/envs/le-wm-py310/bin/hf"
MODEL_DIR="$ROOT/model/lewm-cube"
DOWNLOAD_DIR="$ROOT/downloads"
DATA_ROOT="$ROOT/data"
ARCHIVE="$DOWNLOAD_DIR/cube_single_expert.tar.zst"
EXPECTED_DATASET="$DATA_ROOT/datasets/ogbench/cube_single_expert.h5"

export HF_HOME="$ROOT/.hf"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
mkdir -p "$MODEL_DIR" "$DOWNLOAD_DIR" "$DATA_ROOT/datasets/ogbench"

"$HF_BIN" download quentinll/lewm-cube config.json weights.pt \
  --local-dir "$MODEL_DIR"
"$ROOT/.venv/bin/python" "$ROOT/map_checkpoint.py"

if [[ ! -f "$EXPECTED_DATASET" ]]; then
  "$HF_BIN" download quentinll/lewm-cube cube_single_expert.tar.zst \
    --repo-type dataset \
    --local-dir "$DOWNLOAD_DIR"

  tar --zstd -xf "$ARCHIVE" -C "$DATA_ROOT/datasets"

  if [[ ! -f "$EXPECTED_DATASET" ]]; then
    DATASET_PATH="$(find "$DATA_ROOT/datasets" -type f -name 'cube_single_expert.h5' -print -quit)"
    if [[ -z "$DATASET_PATH" ]]; then
      echo "cube_single_expert.h5 was not found after extraction." >&2
      exit 1
    fi
    ln -s "$DATASET_PATH" "$EXPECTED_DATASET"
  fi
fi

echo "Model:   $MODEL_DIR"
echo "Dataset: $EXPECTED_DATASET"
