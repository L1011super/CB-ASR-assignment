"""CLI entry for stage-2 contextual-biasing training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_stage2 import main


if __name__ == "__main__":
    main(sys.argv[1:])
