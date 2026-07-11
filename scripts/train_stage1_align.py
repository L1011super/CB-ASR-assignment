"""CLI entrypoint for stage-1 speech-to-Qwen alignment training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_stage1 import main  # noqa: E402


if __name__ == "__main__":
    main()
