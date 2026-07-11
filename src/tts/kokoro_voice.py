"""Local Kokoro model and voice handling."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOCAL_KOKORO_FALLBACK = Path("ckpts/kokoro-v1.1")
REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
RAW_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class VoiceSplit:
    train: list[str]
    valid: list[str]
    test: list[str]

    def pool_for(self, split: str) -> list[str]:
        if split == "train":
            return self.train
        if split == "valid":
            return self.valid
        if split == "test":
            return self.test
        raise ValueError(f"Unknown split: {split}")


def resolve_kokoro_model_dir(configured_model_path: str | Path) -> Path:
    model_dir = Path(configured_model_path)
    if model_dir.exists():
        return model_dir
    if LOCAL_KOKORO_FALLBACK.exists():
        warnings.warn(
            f"Configured Kokoro path {model_dir} does not exist; using {LOCAL_KOKORO_FALLBACK}.",
            RuntimeWarning,
        )
        return LOCAL_KOKORO_FALLBACK
    raise FileNotFoundError(
        f"Kokoro model directory not found: {model_dir}. "
        f"Expected local config.json, kokoro-v1_1-zh.pth, and voices/*.pt. "
        f"No network download is attempted."
    )


def resolve_voice_dir(configured_voice_dir: str | Path, model_dir: str | Path) -> Path:
    voice_dir = Path(configured_voice_dir)
    if voice_dir.exists():
        return voice_dir
    fallback = Path(model_dir) / "voices"
    if fallback.exists():
        warnings.warn(
            f"Configured voice_dir {voice_dir} does not exist; using {fallback}.",
            RuntimeWarning,
        )
        return fallback
    raise FileNotFoundError(f"Kokoro voice directory not found: {voice_dir}")


def discover_chinese_voices(voice_dir: str | Path) -> list[str]:
    """Discover Chinese Kokoro voices from local zf_/zm_ voice packs."""

    directory = Path(voice_dir)
    voices = sorted(path.stem for path in directory.glob("*.pt") if path.stem.startswith(("zf_", "zm_")))
    if not voices:
        raise FileNotFoundError(f"No Chinese Kokoro voices found in {directory}")
    return voices


def split_voices(
    voices: list[str],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> VoiceSplit:
    rng = random.Random(seed)
    shuffled = list(voices)
    rng.shuffle(shuffled)

    if len(shuffled) < 10:
        warnings.warn("Fewer than 10 Chinese voices found; train/valid/test will reuse voices.")
        return VoiceSplit(train=shuffled, valid=shuffled, test=shuffled)

    total = len(shuffled)
    train_count = max(1, int(total * train_ratio))
    valid_count = max(1, int(total * valid_ratio))
    if train_count + valid_count >= total:
        valid_count = max(1, total - train_count - 1)
    test_count = total - train_count - valid_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, total - valid_count - test_count)

    return VoiceSplit(
        train=shuffled[:train_count],
        valid=shuffled[train_count : train_count + valid_count],
        test=shuffled[train_count + valid_count :],
    )


def save_voice_split(path: str | Path, split: VoiceSplit, seed: int, voice_dir: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "voice_dir": str(voice_dir),
        "train": split.train,
        "valid": split.valid,
        "test": split.test,
        "counts": {
            "train": len(split.train),
            "valid": len(split.valid),
            "test": len(split.test),
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class KokoroSynthesizer:
    """Small wrapper around Kokoro that always loads local model artifacts."""

    def __init__(
        self,
        model_dir: str | Path,
        voice_dir: str | Path,
        lang_code: str = "z",
        station_terms: Iterable[str] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.voice_dir = Path(voice_dir)
        self.lang_code = lang_code
        self.station_terms = tuple(station_terms or ())
        self.sample_rate = RAW_SAMPLE_RATE
        self._pipeline: Any | None = None

    def synthesize_one(self, text: str, voice: str, speed: float, output_path: str | Path) -> Path:
        """Synthesize one utterance to a local wav file."""

        pipeline = self._get_pipeline()
        voice_path = self.voice_dir / f"{voice}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

        result = next(pipeline(text, voice=str(voice_path), speed=float(speed), split_pattern=None))
        audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else result.audio
        wav = np.asarray(audio, dtype=np.float32)

        from src.tts.audio_postprocess import write_wav

        out_path = Path(output_path)
        write_wav(out_path, wav, self.sample_rate)
        return out_path

    def synthesize_station_command(
        self,
        command_text: str,
        station_text: str,
        voice: str,
        speed: float,
        pause_ms: int,
        output_path: str | Path,
        thinking_filler: str = "",
        thinking_pause_ms: int = 300,
    ) -> Path:
        """Synthesize prefix and station separately, then join with explicit silence.

        Station text still goes through Kokoro's Chinese frontend, but all
        station names are registered as jieba words before synthesis.
        """

        pipeline = self._get_pipeline()
        voice_path = self.voice_dir / f"{voice}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Kokoro voice file not found: {voice_path}")

        segments: list[np.ndarray] = []
        if thinking_filler:
            segments.append(self._synthesize_text_array(pipeline, thinking_filler, voice_path, speed))
            segments.append(_silence(self.sample_rate, thinking_pause_ms))

        tts_station_text = _tts_pronunciation_text(station_text)
        if pause_ms <= 0:
            segments.append(self._synthesize_text_array(pipeline, f"{command_text}{tts_station_text}", voice_path, speed))
        else:
            command_audio = self._synthesize_text_array(pipeline, command_text, voice_path, speed)
            station_audio = self._synthesize_text_array(pipeline, tts_station_text, voice_path, speed)
            segments.extend([command_audio, _silence(self.sample_rate, pause_ms), station_audio])
        wav = np.concatenate(segments).astype(np.float32)

        from src.tts.audio_postprocess import write_wav

        out_path = Path(output_path)
        write_wav(out_path, wav, self.sample_rate)
        return out_path

    def _synthesize_text_array(self, pipeline: Any, text: str, voice_path: Path, speed: float) -> np.ndarray:
        result = next(pipeline(text, voice=str(voice_path), speed=float(speed), split_pattern=None))
        audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else result.audio
        return np.asarray(audio, dtype=np.float32)

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise ImportError(
                "Kokoro is not importable in the current Python environment. "
                "Run with conda environment CB-ASR and install kokoro/misaki[zh]."
            ) from exc

        config_path = self.model_dir / "config.json"
        model_path = self.model_dir / "kokoro-v1_1-zh.pth"
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files missing under {self.model_dir}. "
                "Required: config.json and kokoro-v1_1-zh.pth. No network download is attempted."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KModel(
            repo_id=REPO_ID,
            config=str(config_path),
            model=str(model_path),
        ).to(device).eval()
        self._pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id=REPO_ID,
            model=model,
        )
        register_station_terms_for_jieba(self.station_terms)
        return self._pipeline


def synthesize_one(text: str, voice: str, speed: float, output_path: str | Path) -> Path:
    """Convenience function using the default local Kokoro directory."""

    model_dir = resolve_kokoro_model_dir("pretrained/Kokoro-82M-v1.1-zh")
    voice_dir = resolve_voice_dir("pretrained/Kokoro-82M-v1.1-zh/voices", model_dir)
    return KokoroSynthesizer(model_dir=model_dir, voice_dir=voice_dir).synthesize_one(
        text=text,
        voice=voice,
        speed=speed,
        output_path=output_path,
    )


def build_station_jieba_terms(station_names: Iterable[str]) -> list[str]:
    """Build station terms and station+站 terms for jieba."""

    terms: list[str] = []
    seen: set[str] = set()
    for name in station_names:
        for term in (name, name if name.endswith("\u7ad9") else f"{name}\u7ad9"):
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def register_station_terms_for_jieba(station_terms: Iterable[str], freq: int = 2_000_000) -> None:
    """Register all station terms as high-frequency jieba words."""

    terms = tuple(station_terms)
    if not terms:
        return
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("Kokoro Chinese G2P requires jieba for station lexicon control.") from exc

    for term in terms:
        jieba.add_word(term, freq=freq)


def _tts_pronunciation_text(text: str) -> str:
    """Apply TTS-only pronunciation bypasses while keeping manifest labels canonical."""

    return text.replace("\u56e2\u6cca", "\u56e2\u535a")


def _silence(sample_rate: int, silence_ms: int) -> np.ndarray:
    samples = int(sample_rate * max(0, silence_ms) / 1000)
    return np.zeros(samples, dtype=np.float32)
