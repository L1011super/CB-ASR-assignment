"""Generate 14 TTS utterances for 渌水道 using Microsoft Edge neural voices.
Run on Windows in your conda env:
  pip install edge-tts
  winget install Gyan.FFmpeg
  python generate_lushuidao_edge_tts.py
Outputs 16kHz mono wav files under out_lushuidao_edge_wav/.
"""
from __future__ import annotations

import asyncio
import csv
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest_lushuidao_14.csv"
TMP_DIR = ROOT / "out_lushuidao_edge_mp3"
OUT_DIR = ROOT / "out_lushuidao_edge_wav"


def run_ffmpeg(mp3_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def synth_one(row: dict[str, str]) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mp3_path = TMP_DIR / row["output_wav"].replace(".wav", ".mp3")
    wav_path = OUT_DIR / row["output_wav"]

    communicate = edge_tts.Communicate(
        text=row["text"],
        voice=row["voice"],
        rate=row["rate"],
        volume=row["volume"],
    )
    await communicate.save(str(mp3_path))
    run_ffmpeg(mp3_path, wav_path)
    print(f"saved {wav_path}")


async def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig")))
    for row in rows:
        await synth_one(row)


if __name__ == "__main__":
    asyncio.run(main())
