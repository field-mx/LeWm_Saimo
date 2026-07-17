#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

cd "$ROOT"
exec "$ROOT/.venv/bin/python" eval_cube.py
