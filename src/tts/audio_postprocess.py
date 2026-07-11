"""Audio post-processing for generated TTS samples."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, _to_mono_float32(audio), sample_rate)


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    return _to_mono_float32(audio), int(sample_rate)


def process_audio(
    raw_audio_path: str | Path,
    output_path: str | Path,
    target_sample_rate: int,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
    noise_type: str | None = None,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, float]:
    """Convert raw TTS audio to mono target-rate wav with light augmentations."""

    audio, sample_rate = load_wav(raw_audio_path)
    if sample_rate != target_sample_rate:
        audio = resample_audio(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    if leading_silence_ms > 0:
        audio = prepend_silence(audio, sample_rate, leading_silence_ms)
    if trailing_silence_ms > 0:
        audio = append_silence(audio, sample_rate, trailing_silence_ms)
    if noise_type == "white" and snr_db is not None:
        audio = add_white_noise(audio, snr_db, rng=rng)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    write_wav(output_path, audio, sample_rate)
    duration_sec = float(len(audio) / sample_rate) if sample_rate else 0.0
    return sample_rate, duration_sec


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def prepend_silence(audio: np.ndarray, sample_rate: int, silence_ms: int) -> np.ndarray:
    n_samples = int(sample_rate * silence_ms / 1000)
    if n_samples <= 0:
        return audio
    return np.concatenate([np.zeros(n_samples, dtype=np.float32), audio])


def append_silence(audio: np.ndarray, sample_rate: int, silence_ms: int) -> np.ndarray:
    n_samples = int(sample_rate * silence_ms / 1000)
    if n_samples <= 0:
        return audio
    return np.concatenate([audio, np.zeros(n_samples, dtype=np.float32)])


def add_white_noise(
    audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=audio.shape)
    return (audio + noise.astype(np.float32)).astype(np.float32)


def is_all_silence(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    return bool(np.max(np.abs(audio)) < threshold) if audio.size else True


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32, copy=False)
