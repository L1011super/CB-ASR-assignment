from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from rapidfuzz import fuzz, process
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.run_whisper_raw import (  # noqa: E402
    WhisperRawConfig,
    load_whisper,
    read_audio,
    resolve_path,
    select_audio_path,
    transcribe_one,
    write_jsonl_line,
)
from src.data.station_lexicon import (  # noqa: E402
    StationEntry,
    load_station_lexicon,
    normalize_station_name,
)

try:
    from pypinyin import Style, lazy_pinyin
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install pypinyin for pinyin fuzzy baselines: pip install pypinyin") from exc


METHODS = {
    0: "whisper_raw",
    1: "whisper_exact_match",
    2: "whisper_edit_fuzzy",
    3: "whisper_pinyin_fuzzy",
    4: "whisper_lexicon_normalization",
}


@dataclass
class BaselineConfig:
    model_path: str = "pretrained/whisper-small"
    station_csv: str = "data/stations/tianjin_metro_stations.csv"
    manifest: str = "data/manifests/test_tts.jsonl"
    output_dir: str = "outputs/baseline/whisper_station_baselines"
    error_output: str = "outputs/baseline/whisper_station_baselines/errors.jsonl"
    language: str = "zh"
    task: str = "transcribe"
    device: str = "cuda:0"
    max_new_tokens: int = 64
    num_beams: int = 1
    do_sample: bool = False
    fp16: bool = True
    resume: bool = True
    limit: int | None = None
    edit_threshold: float = 75.0
    pinyin_threshold: float = 82.0
    lexicon_threshold: float = 82.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper station baselines 0-4.")
    parser.add_argument("--config", default="configs/whisper_station_baselines.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--station_csv", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--error_output", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--edit_threshold", type=float, default=None)
    parser.add_argument("--pinyin_threshold", type=float, default=None)
    parser.add_argument("--lexicon_threshold", type=float, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> BaselineConfig:
    defaults = BaselineConfig().__dict__
    config_data: dict[str, Any] = {}
    if args.config:
        config_path = resolve_path(args.config, PROJECT_ROOT)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
    merged = {**defaults, **config_data}
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return BaselineConfig(**merged)


def iter_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                utt_id = json.loads(line).get("utt_id")
            except json.JSONDecodeError:
                continue
            if utt_id:
                ids.add(str(utt_id))
    return ids


def force_device(requested: str, fp16: bool) -> tuple[torch.device, torch.dtype]:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested}, but CUDA is not available.")
    device = torch.device(requested)
    dtype = torch.float16 if device.type == "cuda" and fp16 else torch.float32
    return device, dtype


def load_whisper_on_device(model_path: Path, config: BaselineConfig) -> tuple[Any, Any, torch.device, torch.dtype]:
    raw_config = WhisperRawConfig(
        model_path=config.model_path,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        do_sample=config.do_sample,
        fp16=config.fp16,
    )
    processor, model, _, _ = load_whisper(model_path, fp16=False)
    device, dtype = force_device(config.device, config.fp16)
    model.to(device)
    if dtype == torch.float16:
        model.half()
    model.eval()
    return processor, model, device, dtype


def clean_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def strip_station_suffix(text: str) -> str:
    value = clean_text(text)
    for suffix in ("地铁站", "站"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def extract_slot(asr_text: str) -> str:
    text = clean_text(asr_text)
    if not text:
        return ""
    match = re.search(r"我要去(.+?)站", text)
    if match:
        return strip_station_suffix(match.group(1))
    match = re.search(r"去(.+?)站", text)
    if match:
        return strip_station_suffix(match.group(1))
    match = re.search(r"我要去(.+)$", text)
    if match:
        return strip_station_suffix(match.group(1))
    return ""


def station_for_eval(row: dict[str, Any]) -> str | None:
    station = row.get("target_station")
    if station in ("", "null", "NULL"):
        return None
    return station


def station_id_for_eval(row: dict[str, Any]) -> str | None:
    station_id = row.get("target_station_id") or row.get("station_id")
    if station_id in ("", "null", "NULL"):
        return None
    return station_id


def pinyin_compact(text: str) -> str:
    syllables = lazy_pinyin(text, style=Style.NORMAL, errors="ignore", strict=False)
    return "".join(s.lower() for s in syllables if s)


def candidate_rows(matches: Iterable[tuple[str, float, Any]]) -> list[dict[str, Any]]:
    return [{"station": str(name), "score": round(float(score), 4)} for name, score, _ in matches]


def exact_match(asr_text: str, lexicon: list[StationEntry]) -> str | None:
    text = clean_text(asr_text)
    for entry in sorted(lexicon, key=lambda item: len(item.name), reverse=True):
        spoken_name = entry.name if entry.name.endswith("站") else f"{entry.name}站"
        if entry.name in text or spoken_name in text:
            return entry.name
    return None


def edit_fuzzy_match(slot_text: str, lexicon: list[StationEntry], threshold: float) -> tuple[str | None, float, list[dict[str, Any]]]:
    slot = strip_station_suffix(slot_text)
    if not slot:
        return None, 0.0, []
    names = [entry.name for entry in lexicon]
    matches = process.extract(slot, names, scorer=fuzz.ratio, limit=3)
    top3 = candidate_rows(matches)
    if not matches:
        return None, 0.0, top3
    best_name, score, _ = matches[0]
    pred = str(best_name) if float(score) >= threshold else None
    return pred, float(score), top3


def pinyin_fuzzy_match(
    slot_text: str,
    lexicon: list[StationEntry],
    threshold: float,
) -> tuple[str | None, float, list[dict[str, Any]], str]:
    slot = strip_station_suffix(slot_text)
    if not slot:
        return None, 0.0, [], ""
    query_pinyin = pinyin_compact(slot)
    choices = {entry.pinyin_compact: entry.name for entry in lexicon}
    matches = process.extract(query_pinyin, list(choices), scorer=fuzz.ratio, limit=3)
    top3 = [
        {"station": choices[str(pinyin)], "pinyin": str(pinyin), "score": round(float(score), 4)}
        for pinyin, score, _ in matches
    ]
    if not matches:
        return None, 0.0, top3, query_pinyin
    best_pinyin, score, _ = matches[0]
    pred = choices[str(best_pinyin)] if float(score) >= threshold else None
    return pred, float(score), top3, query_pinyin


def alias_confusion_match(slot_text: str, lexicon: list[StationEntry]) -> str | None:
    slot = strip_station_suffix(slot_text)
    if not slot:
        return None
    alias_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    for entry in lexicon:
        for alias in (*entry.aliases, *entry.confusions):
            key = strip_station_suffix(alias)
            if not key:
                continue
            previous = alias_map.get(key)
            if previous is not None and previous != entry.name:
                ambiguous.add(key)
            alias_map[key] = entry.name
    for key in ambiguous:
        alias_map.pop(key, None)
    return alias_map.get(slot)


def lexicon_normalization(
    slot_text: str,
    lexicon: list[StationEntry],
    edit_threshold: float,
    pinyin_threshold: float,
) -> tuple[str | None, float, str]:
    slot = strip_station_suffix(slot_text)
    if not slot:
        return None, 0.0, "no_slot"

    exact = exact_match(slot, lexicon)
    if exact is not None:
        return exact, 100.0, "exact"

    alias = alias_confusion_match(slot, lexicon)
    if alias is not None:
        return alias, 100.0, "alias_or_confusion"

    normalized = normalize_station_name(slot, lexicon)
    if normalized is not None:
        edit_score = fuzz.ratio(slot, normalized)
        pinyin_score = fuzz.ratio(pinyin_compact(slot), pinyin_compact(normalized))
        score = max(float(edit_score), float(pinyin_score))
        return normalized, score, "lexicon_builtin"

    edit_pred, edit_score, _ = edit_fuzzy_match(slot, lexicon, edit_threshold)
    if edit_pred is not None:
        return edit_pred, edit_score, "edit"

    pinyin_pred, pinyin_score, _, _ = pinyin_fuzzy_match(slot, lexicon, pinyin_threshold)
    if pinyin_pred is not None:
        return pinyin_pred, pinyin_score, "pinyin"

    return None, max(float(edit_score), float(pinyin_score)), "none"


def base_row(row: dict[str, Any], audio_path: Path, asr_text: str, decode_time: float, duration: float) -> dict[str, Any]:
    return {
        "utt_id": row.get("utt_id"),
        "audio_path": audio_path.as_posix(),
        "target_text": row.get("text"),
        "target_station": station_for_eval(row),
        "target_station_id": station_id_for_eval(row),
        "has_station": bool(row.get("has_station")),
        "source": row.get("source"),
        "split": row.get("split"),
        "voice": row.get("voice"),
        "asr_text": asr_text,
        "duration_sec": round(duration, 4),
        "decode_time_sec": round(decode_time, 4),
    }


def with_eval(result: dict[str, Any], pred_station: str | None) -> dict[str, Any]:
    target = result.get("target_station")
    result["pred_station"] = pred_station
    result["correct"] = pred_station == target
    return result


def update_stats(stats: dict[int, dict[str, int]], baseline_id: int, row: dict[str, Any]) -> None:
    if baseline_id == 0:
        return
    stats[baseline_id]["total"] += 1
    if row.get("correct"):
        stats[baseline_id]["correct"] += 1
    target = row.get("target_station")
    pred = row.get("pred_station")
    if target is None:
        stats[baseline_id]["null_total"] += 1
        if pred is None:
            stats[baseline_id]["null_correct"] += 1
    else:
        stats[baseline_id]["station_total"] += 1
        if pred == target:
            stats[baseline_id]["station_correct"] += 1


def make_summary(stats: dict[int, dict[str, int]], total: int, success: int, errors: int, decode_time: float) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for baseline_id, method in METHODS.items():
        if baseline_id == 0:
            baselines[method] = {"note": "raw ASR text only; no station prediction"}
            continue
        item = stats[baseline_id]
        total_count = item["total"]
        station_total = item["station_total"]
        null_total = item["null_total"]
        baselines[method] = {
            "total": total_count,
            "correct": item["correct"],
            "accuracy": round(item["correct"] / total_count, 6) if total_count else 0.0,
            "station_total": station_total,
            "station_correct": item["station_correct"],
            "station_accuracy": round(item["station_correct"] / station_total, 6) if station_total else 0.0,
            "null_total": null_total,
            "null_correct": item["null_correct"],
            "null_accuracy": round(item["null_correct"] / null_total, 6) if null_total else 0.0,
        }
    return {
        "total_manifest_rows": total,
        "asr_success": success,
        "asr_errors": errors,
        "total_decode_time_sec": round(decode_time, 4),
        "avg_decode_time_sec": round(decode_time / success, 4) if success else 0.0,
        "baselines": baselines,
    }


def main() -> None:
    config = load_config(parse_args())
    manifest_path = resolve_path(config.manifest, PROJECT_ROOT)
    station_csv = resolve_path(config.station_csv, PROJECT_ROOT)
    model_path = resolve_path(config.model_path, PROJECT_ROOT)
    output_dir = resolve_path(config.output_dir, PROJECT_ROOT)
    error_output = resolve_path(config.error_output, PROJECT_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    error_output.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_manifest(manifest_path, config.limit)
    lexicon = load_station_lexicon(station_csv)
    raw_config = WhisperRawConfig(
        model_path=config.model_path,
        language=config.language,
        task=config.task,
        max_new_tokens=config.max_new_tokens,
        num_beams=config.num_beams,
        do_sample=config.do_sample,
        fp16=config.fp16,
    )
    processor, model, device, dtype = load_whisper_on_device(model_path, config)

    output_paths = {
        0: output_dir / "baseline0_whisper_raw.jsonl",
        1: output_dir / "baseline1_exact_match.jsonl",
        2: output_dir / "baseline2_edit_fuzzy.jsonl",
        3: output_dir / "baseline3_pinyin_fuzzy.jsonl",
        4: output_dir / "baseline4_lexicon_normalization.jsonl",
    }
    done = completed_ids(output_paths[0]) if config.resume else set()
    stats = {
        baseline_id: {
            "total": 0,
            "correct": 0,
            "station_total": 0,
            "station_correct": 0,
            "null_total": 0,
            "null_correct": 0,
        }
        for baseline_id in METHODS
    }
    errors = 0
    success = 0
    skipped = 0
    total_decode_time = 0.0

    write_mode = "a" if config.resume else "w"
    handles = {baseline_id: path.open(write_mode, encoding="utf-8") for baseline_id, path in output_paths.items()}
    with error_output.open(write_mode, encoding="utf-8") as err_f:
        try:
            for row in tqdm(rows, desc="Whisper baselines 0-4"):
                utt_id = str(row.get("utt_id") or "")
                if config.resume and utt_id in done:
                    skipped += 1
                    continue

                audio_path, path_error = select_audio_path(row, PROJECT_ROOT)
                if path_error or audio_path is None or not audio_path.exists():
                    errors += 1
                    write_jsonl_line(
                        err_f,
                        {
                            "utt_id": row.get("utt_id"),
                            "audio_path": str(audio_path) if audio_path else row.get("audio_path"),
                            "error": path_error or "audio file not found",
                        },
                    )
                    continue

                try:
                    audio, _, duration = read_audio(audio_path)
                    start = time.perf_counter()
                    asr_text = transcribe_one(audio, processor, model, device, dtype, raw_config)
                    decode_time = time.perf_counter() - start
                except Exception as exc:
                    errors += 1
                    write_jsonl_line(
                        err_f,
                        {"utt_id": row.get("utt_id"), "audio_path": audio_path.as_posix(), "error": repr(exc)},
                    )
                    continue

                total_decode_time += decode_time
                success += 1
                common = base_row(row, audio_path, asr_text, decode_time, duration)
                slot = extract_slot(asr_text)

                raw_row = {**common, "method": METHODS[0]}
                write_jsonl_line(handles[0], raw_row)

                b1_pred = exact_match(asr_text, lexicon)
                b1 = with_eval({**common, "method": METHODS[1], "slot_text": slot}, b1_pred)
                write_jsonl_line(handles[1], b1)
                update_stats(stats, 1, b1)

                b2_pred, b2_score, b2_top3 = edit_fuzzy_match(slot, lexicon, config.edit_threshold)
                b2 = with_eval(
                    {
                        **common,
                        "method": METHODS[2],
                        "slot_text": slot,
                        "match_score": round(b2_score, 4),
                        "top3": b2_top3,
                    },
                    b2_pred,
                )
                write_jsonl_line(handles[2], b2)
                update_stats(stats, 2, b2)

                b3_pred, b3_score, b3_top3, slot_pinyin = pinyin_fuzzy_match(slot, lexicon, config.pinyin_threshold)
                b3 = with_eval(
                    {
                        **common,
                        "method": METHODS[3],
                        "slot_text": slot,
                        "slot_pinyin": slot_pinyin,
                        "match_score": round(b3_score, 4),
                        "top3": b3_top3,
                    },
                    b3_pred,
                )
                write_jsonl_line(handles[3], b3)
                update_stats(stats, 3, b3)

                b4_pred, b4_score, b4_reason = lexicon_normalization(
                    slot,
                    lexicon,
                    edit_threshold=max(config.edit_threshold, config.lexicon_threshold),
                    pinyin_threshold=max(config.pinyin_threshold, config.lexicon_threshold),
                )
                b4 = with_eval(
                    {
                        **common,
                        "method": METHODS[4],
                        "slot_text": slot,
                        "match_score": round(b4_score, 4),
                        "match_reason": b4_reason,
                    },
                    b4_pred,
                )
                write_jsonl_line(handles[4], b4)
                update_stats(stats, 4, b4)
        finally:
            for handle in handles.values():
                handle.close()

    summary = make_summary(stats, len(rows), success, errors, total_decode_time)
    summary["skipped_by_resume"] = skipped
    summary["device"] = str(device)
    summary["dtype"] = str(dtype)
    summary["output_dir"] = output_dir.as_posix()
    summary["error_output"] = error_output.as_posix()
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
