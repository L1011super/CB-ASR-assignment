from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm


TARGET_SAMPLE_RATE = 16000
METHOD_NAME = "whisper_raw"
MIN_DURATION_SEC = 0.1


@dataclass
class WhisperRawConfig:
    model_path: str = "pretrained/whisper-small"
    manifest: str = "data/manifests/test.jsonl"
    output: str = "outputs/baseline/whisper_raw_test_predictions.jsonl"
    error_output: str = "outputs/baseline/whisper_raw_errors.jsonl"
    language: str = "zh"
    task: str = "transcribe"
    max_new_tokens: int = 64
    num_beams: int = 1
    do_sample: bool = False
    fp16: bool = True
    resume: bool = True
    limit: int | None = None
    batch_size: int = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = resolve_path(path, repo_root())
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def build_config(args: argparse.Namespace) -> WhisperRawConfig:
    data = load_yaml_config(args.config)
    defaults = WhisperRawConfig().__dict__
    merged = {**defaults, **data}

    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value

    return WhisperRawConfig(**merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw Whisper ASR baseline on a JSONL manifest.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--error_output", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_completed_utt_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    done: set[str] = set()
    with output_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                utt_id = json.loads(line).get("utt_id")
            except json.JSONDecodeError:
                continue
            if utt_id:
                done.add(str(utt_id))
    return done


def select_audio_path(row: dict[str, Any], root: Path) -> tuple[Path | None, str | None]:
    raw_path = row.get("processed_audio_path") or row.get("audio_path")
    if not raw_path:
        return None, "missing audio_path and processed_audio_path"
    return resolve_path(str(raw_path), root), None


def read_audio(path: Path) -> tuple[np.ndarray, int, float]:
    try:
        audio, sample_rate = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"failed to read audio: {exc}") from exc

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)

    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("librosa is required for resampling audio to 16000 Hz") from exc
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    duration_sec = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
    if duration_sec < MIN_DURATION_SEC:
        raise RuntimeError(f"audio too short: {duration_sec:.3f}s")
    if not np.isfinite(audio).all():
        raise RuntimeError("audio contains NaN or Inf")

    return audio, sample_rate, duration_sec


def load_whisper(model_path: Path, fp16: bool) -> tuple[Any, Any, torch.device, torch.dtype]:
    try:
        from transformers import AutoProcessor, WhisperForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency transformers. Install with: "
            "pip install transformers librosa soundfile tqdm pyyaml accelerate"
        ) from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"Whisper model path does not exist: {model_path}. "
            "Place the model locally or pass --model_path. Online download is disabled."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and fp16 else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local Whisper model from {model_path}. "
            "Check that the Hugging Face model files are complete. Online download is disabled."
        ) from exc

    model.to(device)
    if dtype == torch.float16:
        model.half()
    model.eval()
    return processor, model, device, dtype


def transcribe_one(
    audio: np.ndarray,
    processor: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    config: WhisperRawConfig,
) -> str:
    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=config.language, task=config.task)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            do_sample=config.do_sample,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def build_prediction_row(
    row: dict[str, Any],
    audio_path: Path,
    asr_text: str,
    config: WhisperRawConfig,
    duration_sec: float,
    decode_time_sec: float,
) -> dict[str, Any]:
    return {
        "utt_id": row.get("utt_id"),
        "audio_path": str(audio_path),
        "target_text": row.get("text"),
        "target_station": row.get("target_station"),
        "target_station_id": row.get("target_station_id") or row.get("station_id"),
        "has_station": row.get("has_station"),
        "source": row.get("source"),
        "split": row.get("split"),
        "voice": row.get("voice"),
        "asr_text": asr_text,
        "method": METHOD_NAME,
        "model_path": config.model_path,
        "duration_sec": round(duration_sec, 4),
        "decode_time_sec": round(decode_time_sec, 4),
    }


def build_error_row(row: dict[str, Any], error: str, audio_path: Path | None = None) -> dict[str, Any]:
    return {
        "utt_id": row.get("utt_id"),
        "audio_path": str(audio_path) if audio_path else row.get("processed_audio_path") or row.get("audio_path"),
        "target_text": row.get("text"),
        "target_station": row.get("target_station"),
        "target_station_id": row.get("target_station_id") or row.get("station_id"),
        "has_station": row.get("has_station"),
        "source": row.get("source"),
        "split": row.get("split"),
        "voice": row.get("voice"),
        "method": METHOD_NAME,
        "error": error,
    }


def main() -> None:
    root = repo_root()
    config = build_config(parse_args())

    manifest_path = resolve_path(config.manifest, root)
    output_path = resolve_path(config.output, root)
    error_path = resolve_path(config.error_output, root)
    model_path = resolve_path(config.model_path, root)

    if config.batch_size != 1:
        print("batch_size > 1 is accepted as a parameter, but this first baseline runs batch_size=1.")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_jsonl(manifest_path, limit=config.limit)
    completed = load_completed_utt_ids(output_path) if config.resume else set()
    processor, model, device, dtype = load_whisper(model_path, fp16=config.fp16)

    success_count = 0
    error_count = 0
    skipped_count = 0
    total_decode_time = 0.0

    with output_path.open("a", encoding="utf-8") as out_f, error_path.open("a", encoding="utf-8") as err_f:
        for row in tqdm(rows, desc="Whisper raw ASR"):
            utt_id = row.get("utt_id")
            if config.resume and utt_id and str(utt_id) in completed:
                skipped_count += 1
                continue

            audio_path, audio_error = select_audio_path(row, root)
            if audio_error or audio_path is None:
                write_jsonl_line(err_f, build_error_row(row, audio_error or "missing audio path"))
                error_count += 1
                continue
            if not audio_path.exists():
                write_jsonl_line(err_f, build_error_row(row, f"audio file not found: {audio_path}", audio_path))
                error_count += 1
                continue

            try:
                audio, _, duration_sec = read_audio(audio_path)
                start = time.perf_counter()
                asr_text = transcribe_one(audio, processor, model, device, dtype, config)
                decode_time_sec = time.perf_counter() - start
            except Exception as exc:
                write_jsonl_line(err_f, build_error_row(row, str(exc), audio_path))
                error_count += 1
                continue

            total_decode_time += decode_time_sec
            success_count += 1
            write_jsonl_line(
                out_f,
                build_prediction_row(row, audio_path, asr_text, config, duration_sec, decode_time_sec),
            )

    avg_decode_time = total_decode_time / success_count if success_count else 0.0
    print(json.dumps(
        {
            "total_samples": len(rows),
            "skipped_by_resume": skipped_count,
            "success_count": success_count,
            "error_count": error_count,
            "total_decode_time": round(total_decode_time, 4),
            "average_decode_time": round(avg_decode_time, 4),
            "output_path": str(output_path),
            "error_path": str(error_path),
            "device": str(device),
            "dtype": str(dtype),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
