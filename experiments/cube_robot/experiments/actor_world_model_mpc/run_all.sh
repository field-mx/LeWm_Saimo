#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/publicworkspace/envs/le-wm-py310/bin/python}"

exec "$PYTHON" "$SCRIPT_DIR/run_experiment.py" "$@"
