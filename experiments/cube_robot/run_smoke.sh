#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

cd "$ROOT"
"$ROOT/.venv/bin/python" make_synthetic_smoke_data.py
exec "$ROOT/.venv/bin/python" eval_cube.py \
  eval.num_eval=1 \
  eval.dataset_name=smoke/cube_random \
  eval.eval_budget=25 \
  solver.num_samples=8 \
  solver.n_steps=2 \
  solver.topk=2 \
  output.filename=cube_smoke_results.txt
