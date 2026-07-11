"""Quality checks for generated Kokoro TTS manifests and audio files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.station_lexicon import load_station_lexicon  # noqa: E402
from src.tts.audio_postprocess import is_all_silence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config)
    report, bad_cases = check_dataset(config)

    output_dir = Path("outputs/data_checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "kokoro_tts_check_report.json"
    bad_path = output_dir / "kokoro_tts_bad_cases.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_bad_cases(bad_path, bad_cases)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["bad_case_count"] > 0:
        raise SystemExit(1)


def check_dataset(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_dir = Path(config["paths"]["manifest_dir"])
    station_csv = Path(config["paths"]["station_csv"])
    target_sample_rate = int(config["audio"]["target_sample_rate"])
    manifests = {
        "train": manifest_dir / "train_tts.jsonl",
        "valid": manifest_dir / "valid_tts.jsonl",
        "test": manifest_dir / "test_tts.jsonl",
    }

    bad_cases: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, path in manifests.items():
        if not path.exists():
            bad_cases.append(_bad(split, "", "missing_manifest", str(path)))
            rows_by_split[split] = []
        else:
            rows_by_split[split] = _read_jsonl(path)

    stations = load_station_lexicon(station_csv)
    legal_names = {station.name for station in stations}
    station_id_to_name = {station.station_id: station.name for station in stations}
    meta = _load_meta(manifest_dir)
    expected_station_ids = meta.get("station_ids") or list(station_id_to_name)
    expected_per_station = meta.get("expected_per_station") or {
        "train": int(config["generation"]["train_per_station"]),
        "valid": int(config["generation"]["valid_per_station"]),
        "test": int(config["generation"]["test_per_station"]),
    }
    expected_null_counts = meta.get("expected_null_counts") or {
        "train": int(config["generation"]["null_train"]),
        "valid": int(config["generation"]["null_valid"]),
        "test": int(config["generation"]["null_test"]),
    }

    per_station_counts: dict[str, Counter[str]] = defaultdict(Counter)
    null_counts: Counter[str] = Counter()
    total_duration = 0.0

    for split, rows in rows_by_split.items():
        for row in rows:
            utt_id = str(row.get("utt_id", ""))
            station_id = str(row.get("station_id", ""))
            audio_path = Path(str(row.get("audio_path", "")))
            target_station = row.get("target_station")
            has_station = bool(row.get("has_station"))

            if split != row.get("split"):
                bad_cases.append(_bad(split, utt_id, "split_mismatch", str(row.get("split"))))

            if station_id == "NULL":
                null_counts[split] += 1
                if target_station is not None or has_station:
                    bad_cases.append(_bad(split, utt_id, "invalid_null_label", str(target_station)))
            else:
                per_station_counts[station_id][split] += 1
                if station_id not in station_id_to_name:
                    bad_cases.append(_bad(split, utt_id, "unknown_station_id", station_id))
                if target_station not in legal_names:
                    bad_cases.append(_bad(split, utt_id, "illegal_target_station", str(target_station)))
                if station_id in station_id_to_name and target_station != station_id_to_name[station_id]:
                    bad_cases.append(_bad(split, utt_id, "station_target_mismatch", str(target_station)))
                if not has_station:
                    bad_cases.append(_bad(split, utt_id, "positive_has_station_false", station_id))

            if not audio_path.exists():
                bad_cases.append(_bad(split, utt_id, "missing_audio", str(audio_path)))
                continue
            try:
                info = sf.info(str(audio_path))
                if info.samplerate != target_sample_rate:
                    bad_cases.append(_bad(split, utt_id, "bad_sample_rate", str(info.samplerate)))
                if info.channels != 1:
                    bad_cases.append(_bad(split, utt_id, "not_mono", str(info.channels)))
                duration = float(info.duration)
                total_duration += duration
                if duration < 0.2 or duration > 8.0:
                    bad_cases.append(_bad(split, utt_id, "duration_out_of_range", f"{duration:.4f}"))
                audio, _ = sf.read(str(audio_path), always_2d=False)
                if is_all_silence(np.asarray(audio)):
                    bad_cases.append(_bad(split, utt_id, "all_silence", str(audio_path)))
            except Exception as exc:
                bad_cases.append(_bad(split, utt_id, "audio_read_error", repr(exc)))

    for station_id in expected_station_ids:
        for split, expected in expected_per_station.items():
            actual = per_station_counts[station_id][split]
            if actual != int(expected):
                bad_cases.append(
                    _bad(split, station_id, "station_count_mismatch", f"expected={expected}, actual={actual}")
                )

    for split, expected in expected_null_counts.items():
        actual = null_counts[split]
        if actual != int(expected):
            bad_cases.append(_bad(split, "NULL", "null_count_mismatch", f"expected={expected}, actual={actual}"))

    split_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    report = {
        "ok": len(bad_cases) == 0,
        "bad_case_count": len(bad_cases),
        "split_counts": split_counts,
        "station_count_checked": len(expected_station_ids),
        "null_counts": dict(null_counts),
        "total_duration_sec": round(total_duration, 3),
        "manifest_dir": str(manifest_dir),
        "target_sample_rate": target_sample_rate,
    }
    return report, bad_cases


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_meta(manifest_dir: Path) -> dict[str, Any]:
    path = manifest_dir / "kokoro_generation_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bad(split: str, utt_id: str, issue: str, detail: str) -> dict[str, Any]:
    return {"split": split, "utt_id": utt_id, "issue": issue, "detail": detail}


def _write_bad_cases(path: Path, bad_cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "utt_id", "issue", "detail"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_cases)


if __name__ == "__main__":
    main()
