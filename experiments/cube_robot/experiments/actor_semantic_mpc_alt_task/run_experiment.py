from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
IMPLEMENTATION_DIR = HERE.parent / "actor_semantic_mpc"
sys.path.insert(0, str(IMPLEMENTATION_DIR))

from run_experiment import main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(arg == "--config" or arg.startswith("--config=") for arg in arguments):
        sys.argv.extend(["--config", str(HERE / "config.yaml")])
    if not any(arg == "--stage" or arg.startswith("--stage=") for arg in arguments):
        sys.argv.extend(["--stage", "train"])
    main()
