#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL=egl

cd "$ROOT"
exec "$ROOT/.venv/bin/python" run_fixed_task.py "$@"
