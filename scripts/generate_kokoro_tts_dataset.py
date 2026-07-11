"""CLI entrypoint for Kokoro TTS dataset generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tts.generate_kokoro_tts import generate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/kokoro_tts.yaml")
    parser.add_argument("--limit_stations", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--null_only",
        action="store_true",
        help="Only generate missing NULL samples and append them to existing manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        config_path=args.config,
        limit_stations=args.limit_stations,
        overwrite=args.overwrite,
        null_only=args.null_only,
    )


if __name__ == "__main__":
    main()
