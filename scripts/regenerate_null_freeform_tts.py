"""Regenerate a small free-form subset of NULL Kokoro TTS samples.

The main dataset generator now focuses on canonical station commands. This
utility edits only existing NULL rows: it samples a low ratio of them, replaces
their text with ticket-machine scene chatter, and regenerates the referenced
audio files in place.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tts.audio_postprocess import process_audio  # noqa: E402
from src.tts.kokoro_voice import KokoroSynthesizer, resolve_kokoro_model_dir, resolve_voice_dir  # noqa: E402


FREEFORM_NULL_TEXTS: tuple[str, ...] = (
    "我的钱包呢",
    "这里能退票吗",
    "附近有没有工作人员",
    "我应该怎么买票",
    "这个机器怎么没有反应",
    "能不能帮我看一下",
    "我要坐几号线",
    "票价是多少钱",
    "我手机没电了怎么办",
    "附近洗手间在哪里",
    "我是不是走错入口了",
    "这张票还能用吗",
    "可以用现金买票吗",
    "怎么切换目的地",
    "我要找客服",
    "请问这里能开发票吗",
    "我刚才点错了",
    "这个屏幕太暗了",
    "我需要一张单程票",
    "有没有无障碍通道",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kokoro_tts.yaml")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config)
    rng = random.Random(int(config["seed"]) + 503)
    np_rng = np.random.default_rng(int(config["seed"]) + 503)
    manifest_dir = Path(config["paths"]["manifest_dir"])
    model_dir = resolve_kokoro_model_dir(config["kokoro"]["model_path"])
    voice_dir = resolve_voice_dir(config["paths"]["voice_dir"], model_dir)
    synthesizer = KokoroSynthesizer(model_dir=model_dir, voice_dir=voice_dir, lang_code=config["kokoro"]["lang_code"])

    all_rows: list[tuple[str, int, dict[str, Any]]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = manifest_dir / f"{split}_tts.jsonl"
        rows = _read_jsonl(path)
        rows_by_split[split] = rows
        for index, row in enumerate(rows):
            if row.get("station_id") == "NULL":
                all_rows.append((split, index, row))

    sample_count = max(1, round(len(all_rows) * float(args.ratio))) if all_rows else 0
    selected = rng.sample(all_rows, sample_count)
    print(f"Selected {len(selected)} of {len(all_rows)} NULL rows for free-form regeneration.")
    if args.dry_run:
        return

    for split, index, row in tqdm(selected, desc="Regenerating NULL free-form TTS"):
        new_text = rng.choice(FREEFORM_NULL_TEXTS)
        row["text"] = new_text
        row["tts_text"] = new_text
        row["has_station"] = False
        row["target_station"] = None
        row["freeform_null"] = True

        raw_path = Path(str(row["raw_audio_path"]))
        audio_path = Path(str(row["audio_path"]))
        synthesizer.synthesize_one(
            text=new_text,
            voice=str(row["voice"]),
            speed=float(row["speed"]),
            output_path=raw_path,
        )
        sample_rate, duration_sec = process_audio(
            raw_audio_path=raw_path,
            output_path=audio_path,
            target_sample_rate=int(config["audio"]["target_sample_rate"]),
            leading_silence_ms=int(row.get("leading_silence_ms") or 0),
            trailing_silence_ms=int(row.get("trailing_silence_ms") or 0),
            noise_type=row.get("noise_type"),
            snr_db=row.get("snr_db"),
            rng=np_rng,
        )
        row["sample_rate"] = sample_rate
        row["duration_sec"] = round(duration_sec, 4)
        rows_by_split[split][index] = row

    for split, rows in rows_by_split.items():
        _write_jsonl(manifest_dir / f"{split}_tts.jsonl", rows)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
