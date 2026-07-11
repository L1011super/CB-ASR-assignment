"""Generate a small station-pause Kokoro TTS sample set under temp/.

This script is intentionally separate from the full dataset generator. It lets
us listen-check the no-jieba station synthesis path before overwriting data.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.station_lexicon import load_station_lexicon  # noqa: E402
from src.data.station_text import positive_text, sample_thinking_filler, station_destination_text  # noqa: E402
from src.tts.audio_postprocess import process_audio  # noqa: E402
from src.tts.kokoro_voice import (  # noqa: E402
    KokoroSynthesizer,
    build_station_jieba_terms,
    discover_chinese_voices,
    resolve_kokoro_model_dir,
    resolve_voice_dir,
)


def main() -> None:
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    output_dir = Path("temp/station_pause_tts_test")
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stations = load_station_lexicon("data/stations/tianjin_metro_stations.csv")
    wanted_names = ["中心渔港", "团泊健康城", "团泊医学园", "远洋国际中心"]
    by_name = {station.name: station for station in stations}

    model_dir = resolve_kokoro_model_dir("pretrained/Kokoro-82M-v1.1-zh")
    voice_dir = resolve_voice_dir("pretrained/Kokoro-82M-v1.1-zh/voices", model_dir)
    voices = discover_chinese_voices(voice_dir)
    voice = voices[0]
    synthesizer = KokoroSynthesizer(
        model_dir=model_dir,
        voice_dir=voice_dir,
        lang_code="z",
        station_terms=build_station_jieba_terms(station.name for station in stations),
    )

    rows: list[dict[str, object]] = []
    for index, name in enumerate(wanted_names, start=1):
        station = by_name[name]
        filler = "嗯" if index == 1 else sample_thinking_filler(rng, 0.05)
        prefix_text = f"{filler}我要去"
        station_text = station_destination_text(station.name)
        pause_ms = int(round(rng.uniform(0.15, 0.85) * 1000))
        raw_path = raw_dir / f"{station.station_id}_test.wav"
        processed_path = processed_dir / f"{station.station_id}_test.wav"

        synthesizer.synthesize_station_command(
            command_text="我要去",
            station_text=station_text,
            voice=voice,
            speed=1.0,
            pause_ms=pause_ms,
            output_path=raw_path,
            thinking_filler=filler,
            thinking_pause_ms=300,
        )
        sample_rate, duration_sec = process_audio(
            raw_audio_path=raw_path,
            output_path=processed_path,
            target_sample_rate=16000,
            rng=np_rng,
        )

        rows.append(
            {
                "station_id": station.station_id,
                "target_station": station.name,
                "text": positive_text(station.name, filler=filler),
                "tts_text": f"{prefix_text}{station_text.replace('泊', '博')}",
                "voice": voice,
                "command_pause_ms": pause_ms,
                "thinking_filler": filler,
                "raw_audio_path": raw_path.as_posix(),
                "audio_path": processed_path.as_posix(),
                "sample_rate": sample_rate,
                "duration_sec": round(duration_sec, 4),
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
