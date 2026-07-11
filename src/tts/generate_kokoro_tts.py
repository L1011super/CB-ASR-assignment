"""Generate Kokoro TTS data and manifests for fixed station-name commands."""

from __future__ import annotations

import json
import os
import random
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from src.data.station_text import (
    StationTextSample,
    build_positive_samples,
    load_stations,
    null_text,
    positive_text,
    sample_null_destinations,
    sample_thinking_filler,
    station_destination_text,
)
from src.tts.audio_postprocess import process_audio
from src.tts.kokoro_voice import (
    KokoroSynthesizer,
    VoiceSplit,
    build_station_jieba_terms,
    discover_chinese_voices,
    resolve_kokoro_model_dir,
    resolve_voice_dir,
    save_voice_split,
    split_voices,
)


@dataclass(frozen=True)
class GenerationJob:
    station_id: str
    target_station: str | None
    text: str
    has_station: bool
    split: str
    index: int
    augmentation_type: str
    speed: float
    leading_silence_ms: int
    trailing_silence_ms: int
    noise_type: str | None
    snr_db: float | None
    voice: str
    voice_pool: str
    command_text: str
    station_spoken_text: str | None
    command_pause_ms: int
    thinking_filler: str
    thinking_pause_ms: int
    tts_text: str

    @property
    def filename(self) -> str:
        return f"{self.station_id}_tts_{self.split}_{self.index:04d}.wav"

    @property
    def utt_id(self) -> str:
        return self.filename[:-4]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset(
    config_path: str | Path,
    limit_stations: int | None = None,
    overwrite: bool = False,
    null_only: bool = False,
) -> None:
    config = load_config(config_path)
    seed = int(config["seed"])
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    paths = config["paths"]
    output_root = Path(paths["output_root"])
    processed_root = Path(paths["processed_root"])
    manifest_dir = Path(paths["manifest_dir"])
    failed_log = Path("logs/kokoro_tts_failed.jsonl")

    if overwrite and null_only:
        raise ValueError("--overwrite and --null_only should not be used together.")

    if overwrite:
        _remove_dir(output_root)
        _remove_dir(processed_root)
        for name in ("train_tts.jsonl", "valid_tts.jsonl", "test_tts.jsonl"):
            path = manifest_dir / name
            if path.exists():
                path.unlink()
        meta_path = manifest_dir / "kokoro_generation_meta.json"
        if meta_path.exists():
            meta_path.unlink()
        if failed_log.exists():
            failed_log.unlink()

    output_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    failed_log.parent.mkdir(parents=True, exist_ok=True)

    all_stations = load_stations(paths["station_csv"])
    stations = list(all_stations)
    if limit_stations is not None:
        stations = stations[:limit_stations]
    if null_only:
        stations = []
    if not stations and not null_only:
        raise ValueError("No station rows available for TTS generation.")

    model_dir = resolve_kokoro_model_dir(config["kokoro"]["model_path"])
    voice_dir = resolve_voice_dir(paths["voice_dir"], model_dir)
    voices = discover_chinese_voices(voice_dir)
    voice_split = split_voices(
        voices=voices,
        seed=seed,
        train_ratio=float(config["voice_split"]["train_ratio"]),
        valid_ratio=float(config["voice_split"]["valid_ratio"]),
        test_ratio=float(config["voice_split"]["test_ratio"]),
    )
    save_voice_split(manifest_dir / "kokoro_voice_split.json", voice_split, seed, voice_dir)

    null_counts = _null_counts(config, limit_stations)
    existing_null_counts = _existing_null_counts(manifest_dir) if null_only else {"train": 0, "valid": 0, "test": 0}
    if null_only:
        null_counts = {
            split: max(0, int(null_counts[split]) - int(existing_null_counts.get(split, 0)))
            for split in ("train", "valid", "test")
        }
    elif not bool(config["generation"].get("generate_null", True)):
        null_counts = {"train": 0, "valid": 0, "test": 0}
    jobs = _build_jobs(
        config,
        stations,
        all_stations,
        null_counts,
        voice_split,
        rng,
        null_start_indices={
            split: int(existing_null_counts.get(split, 0)) + 1
            for split in ("train", "valid", "test")
        },
    )
    _write_generation_meta(
        manifest_dir / "kokoro_generation_meta.json",
        config=config,
        stations=stations,
        voice_split=voice_split,
        null_counts=null_counts,
        limit_stations=limit_stations,
        model_dir=model_dir,
        voice_dir=voice_dir,
    )

    if null_only and not jobs:
        print("NULL samples are already complete; no generation needed.")
        return

    synthesizer = KokoroSynthesizer(
        model_dir=model_dir,
        voice_dir=voice_dir,
        lang_code=str(config["kokoro"]["lang_code"]),
        station_terms=build_station_jieba_terms(station.name for station in all_stations),
    )

    manifest_handles = {
        split: (manifest_dir / f"{split}_tts.jsonl").open("a", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    try:
        for job in tqdm(jobs, desc="Generating Kokoro TTS"):
            try:
                row = _run_job(
                    job=job,
                    config=config,
                    output_root=output_root,
                    processed_root=processed_root,
                    synthesizer=synthesizer,
                    np_rng=np_rng,
                )
                manifest_handles[job.split].write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest_handles[job.split].flush()
            except Exception as exc:  # Keep batch generation moving.
                failure = {
                    "utt_id": job.utt_id,
                    "station_id": job.station_id,
                    "split": job.split,
                    "text": job.text,
                    "voice": job.voice,
                    "error": repr(exc),
                }
                with failed_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    finally:
        for handle in manifest_handles.values():
            handle.close()


def _build_jobs(
    config: dict[str, Any],
    stations: list[Any],
    all_stations: list[Any],
    null_counts: dict[str, int],
    voice_split: VoiceSplit,
    rng: random.Random,
    null_start_indices: dict[str, int] | None = None,
) -> list[GenerationJob]:
    samples = build_positive_samples(stations)
    jobs: list[GenerationJob] = []
    split_counts = {
        "train": int(config["generation"]["train_per_station"]),
        "valid": int(config["generation"]["valid_per_station"]),
        "test": int(config["generation"]["test_per_station"]),
    }

    for sample in samples:
        for split, count in split_counts.items():
            plans = _augmentation_plan(config, split, count, rng)
            for index, plan in enumerate(plans, start=1):
                jobs.append(_make_job(sample, split, index, plan, voice_split, rng, config))

    if bool(config["generation"].get("generate_null", True)):
        station_names = [station.name for station in all_stations]
        for split, count in null_counts.items():
            destinations = sample_null_destinations(count, rng, station_names)
            start_index = 1 if null_start_indices is None else int(null_start_indices.get(split, 1))
            for index, destination in enumerate(destinations, start=start_index):
                sample = StationTextSample(
                    station_id="NULL",
                    target_station=None,
                    text=null_text(destination),
                    has_station=False,
                )
                plan = _one_null_plan(config, split, rng)
                jobs.append(_make_job(sample, split, index, plan, voice_split, rng, config))

    return jobs


def _make_job(
    sample: StationTextSample,
    split: str,
    index: int,
    plan: dict[str, Any],
    voice_split: VoiceSplit,
    rng: random.Random,
    config: dict[str, Any],
) -> GenerationJob:
    pool = voice_split.pool_for(split)
    if not pool:
        raise ValueError(f"Voice pool for split {split} is empty.")
    voice = rng.choice(pool)
    command = _station_command_fields(sample, config, rng)
    return GenerationJob(
        station_id=sample.station_id,
        target_station=sample.target_station,
        text=command["text"],
        has_station=sample.has_station,
        split=split,
        index=index,
        augmentation_type=plan["augmentation_type"],
        speed=float(plan["speed"]),
        leading_silence_ms=int(plan.get("leading_silence_ms", 0)),
        trailing_silence_ms=int(plan.get("trailing_silence_ms", 0)),
        noise_type=plan.get("noise_type"),
        snr_db=plan.get("snr_db"),
        voice=voice,
        voice_pool=split,
        command_text=command["command_text"],
        station_spoken_text=command["station_spoken_text"],
        command_pause_ms=command["command_pause_ms"],
        thinking_filler=command["thinking_filler"],
        thinking_pause_ms=command["thinking_pause_ms"],
        tts_text=command["tts_text"],
    )


def _run_job(
    job: GenerationJob,
    config: dict[str, Any],
    output_root: Path,
    processed_root: Path,
    synthesizer: KokoroSynthesizer,
    np_rng: np.random.Generator,
) -> dict[str, Any]:
    station_dir_name = "NULL" if job.station_id == "NULL" else f"{job.station_id}_{job.target_station}"
    raw_path = output_root / station_dir_name / job.filename
    processed_path = processed_root / job.filename

    if job.has_station and job.station_spoken_text:
        synthesizer.synthesize_station_command(
            command_text=job.command_text,
            station_text=job.station_spoken_text,
            voice=job.voice,
            speed=job.speed,
            pause_ms=job.command_pause_ms,
            output_path=raw_path,
            thinking_filler=job.thinking_filler,
            thinking_pause_ms=job.thinking_pause_ms,
        )
    else:
        synthesizer.synthesize_one(
            text=job.tts_text,
            voice=job.voice,
            speed=job.speed,
            output_path=raw_path,
        )
    sample_rate, duration_sec = process_audio(
        raw_audio_path=raw_path,
        output_path=processed_path,
        target_sample_rate=int(config["audio"]["target_sample_rate"]),
        leading_silence_ms=job.leading_silence_ms,
        trailing_silence_ms=job.trailing_silence_ms,
        noise_type=job.noise_type,
        snr_db=job.snr_db,
        rng=np_rng,
    )

    return {
        "utt_id": job.utt_id,
        "station_id": job.station_id,
        "target_station": job.target_station,
        "text": job.text,
        "has_station": job.has_station,
        "split": job.split,
        "source": "kokoro_tts",
        "version": config["generation"]["version"],
        "voice": job.voice,
        "voice_pool": job.voice_pool,
        "speed": job.speed,
        "augmentation_type": job.augmentation_type,
        "leading_silence_ms": job.leading_silence_ms,
        "trailing_silence_ms": job.trailing_silence_ms,
        "noise_type": job.noise_type,
        "snr_db": job.snr_db,
        "command_pause_ms": job.command_pause_ms,
        "thinking_filler": job.thinking_filler,
        "thinking_pause_ms": job.thinking_pause_ms,
        "tts_text": job.tts_text,
        "raw_audio_path": _as_posix(raw_path),
        "audio_path": _as_posix(processed_path),
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 4),
    }


def _augmentation_plan(
    config: dict[str, Any],
    split: str,
    expected_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if split == "train":
        train_plan = config["augmentation"]["train_plan"]
        kinds = (
            ["clean"] * int(train_plan["clean"])
            + ["speed"] * int(train_plan["speed"])
            + ["leading_silence"] * int(train_plan["leading_silence"])
            + ["light_noise"] * int(train_plan["light_noise"])
        )
    elif split == "valid":
        valid_plan = config["augmentation"]["valid_plan"]
        kinds = ["clean"] * int(valid_plan["clean"]) + ["speed"] * int(valid_plan["speed"])
    elif split == "test":
        test_plan = config["augmentation"]["test_plan"]
        kinds = ["clean"] * int(test_plan["clean"]) + ["light_aug"] * int(test_plan["light_aug"])
    else:
        raise ValueError(f"Unknown split: {split}")

    if len(kinds) != expected_count:
        raise ValueError(f"{split} augmentation plan has {len(kinds)} items, expected {expected_count}")
    return [_materialize_augmentation(config, kind, rng) for kind in kinds]


def _one_null_plan(config: dict[str, Any], split: str, rng: random.Random) -> dict[str, Any]:
    plans = _augmentation_plan(config, split, _split_count(config, split), rng)
    return rng.choice(plans)


def _split_count(config: dict[str, Any], split: str) -> int:
    if split == "train":
        return int(config["generation"]["train_per_station"])
    if split == "valid":
        return int(config["generation"]["valid_per_station"])
    if split == "test":
        return int(config["generation"]["test_per_station"])
    raise ValueError(split)


def _station_command_fields(
    sample: StationTextSample,
    config: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if not sample.has_station or not sample.target_station:
        return {
            "text": sample.text,
            "command_text": "",
            "station_spoken_text": None,
            "command_pause_ms": 0,
            "thinking_filler": "",
            "thinking_pause_ms": 0,
            "tts_text": sample.text.replace("\u6cca", "\u535a"),
        }

    command_config = config.get("station_command", {})
    filler_probability = float(command_config.get("thinking_filler_probability", 0.0))
    fillers = command_config.get("thinking_fillers") or ["\u55ef", "\u90a3\u4e2a", "\u5443"]
    pause_min = float(command_config.get("pause_sec_min", 0.15))
    pause_max = float(command_config.get("pause_sec_max", 0.5))
    thinking_pause_ms = int(round(float(command_config.get("thinking_filler_pause_sec", 0.3)) * 1000))
    if pause_max < pause_min:
        raise ValueError("station_command.pause_sec_max must be >= pause_sec_min")

    filler = sample_thinking_filler(rng, filler_probability, fillers)
    command_text = "\u6211\u8981\u53bb"
    station_text = station_destination_text(sample.target_station)
    display_text = positive_text(sample.target_station, filler=filler)
    pause_ms = int(round(rng.uniform(pause_min, pause_max) * 1000))
    tts_station_text = station_text.replace("\u56e2\u6cca", "\u56e2\u535a")
    tts_prefix = f"{filler}{command_text}" if filler else command_text
    return {
        "text": display_text,
        "command_text": command_text,
        "station_spoken_text": station_text,
        "command_pause_ms": pause_ms,
        "thinking_filler": filler,
        "thinking_pause_ms": thinking_pause_ms if filler else 0,
        "tts_text": f"{tts_prefix}{tts_station_text}",
    }


def _materialize_augmentation(
    config: dict[str, Any],
    kind: str,
    rng: random.Random,
) -> dict[str, Any]:
    speed_values = [float(v) for v in config["augmentation"]["speed_values"]]
    non_unit_speeds = [v for v in speed_values if abs(v - 1.0) > 1e-6]
    leading_options = [int(v) for v in config["augmentation"]["leading_silence_ms"]]
    snr_options = [float(v) for v in config["augmentation"]["noise_snr_db"]]

    if kind == "clean":
        return {"augmentation_type": "clean", "speed": 1.0}
    if kind == "speed":
        return {"augmentation_type": "speed", "speed": rng.choice(non_unit_speeds)}
    if kind == "leading_silence":
        return {
            "augmentation_type": "leading_silence",
            "speed": 1.0,
            "leading_silence_ms": rng.choice(leading_options),
        }
    if kind == "light_noise":
        return {
            "augmentation_type": "light_noise",
            "speed": 1.0,
            "noise_type": "white",
            "snr_db": rng.choice(snr_options),
        }
    if kind == "light_aug":
        return _materialize_augmentation(
            config,
            rng.choice(["speed", "leading_silence", "light_noise"]),
            rng,
        )
    raise ValueError(f"Unknown augmentation type: {kind}")


def _null_counts(config: dict[str, Any], limit_stations: int | None) -> dict[str, int]:
    if limit_stations is None:
        return {
            "train": int(config["generation"]["null_train"]),
            "valid": int(config["generation"]["null_valid"]),
            "test": int(config["generation"]["null_test"]),
        }

    # Smoke-test mode: keep generation fast while preserving split proportions.
    return {
        "train": min(int(config["generation"]["null_train"]), limit_stations * 10),
        "valid": min(int(config["generation"]["null_valid"]), limit_stations * 2),
        "test": min(int(config["generation"]["null_test"]), limit_stations * 2),
    }


def _existing_null_counts(manifest_dir: Path) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    for split in counts:
        path = manifest_dir / f"{split}_tts.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("station_id") == "NULL" or row.get("has_station") is False:
                    counts[split] += 1
    return counts


def _write_generation_meta(
    path: Path,
    config: dict[str, Any],
    stations: list[Any],
    voice_split: VoiceSplit,
    null_counts: dict[str, int],
    limit_stations: int | None,
    model_dir: Path,
    voice_dir: Path,
) -> None:
    meta = {
        "version": config["generation"]["version"],
        "limit_stations": limit_stations,
        "station_ids": [station.station_id for station in stations],
        "station_count": len(stations),
        "expected_per_station": {
            "train": int(config["generation"]["train_per_station"]),
            "valid": int(config["generation"]["valid_per_station"]),
            "test": int(config["generation"]["test_per_station"]),
        },
        "expected_null_counts": null_counts,
        "voice_counts": {
            "train": len(voice_split.train),
            "valid": len(voice_split.valid),
            "test": len(voice_split.test),
        },
        "model_dir": str(model_dir),
        "voice_dir": str(voice_dir),
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _remove_dir(path: Path) -> None:
    if path.exists():
        def _onerror(func: Any, target: str, exc_info: Any) -> None:
            try:
                os.chmod(target, stat.S_IWRITE)
                func(target)
            except PermissionError:
                time.sleep(0.2)
                os.chmod(target, stat.S_IWRITE)
                func(target)

        shutil.rmtree(path, onerror=_onerror)


def _as_posix(path: Path) -> str:
    return path.as_posix()
