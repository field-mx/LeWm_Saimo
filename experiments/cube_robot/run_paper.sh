#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$ROOT/data/datasets/ogbench/cube_single_expert.h5"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

if [[ ! -f "$DATASET" ]]; then
  echo "Official paper dataset is missing: $DATASET" >&2
  echo "Download the 46.2 GB OGBench-Cube archive with: $ROOT/prepare_assets.sh" >&2
  exit 2
fi

cd "$ROOT"
exec "$ROOT/.venv/bin/python" eval_cube.py
