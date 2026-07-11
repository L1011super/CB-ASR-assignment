"""Batch collation for audio-to-Qwen station recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.data.audio_dataset import AudioSample
from src.data.prompts import (
    STAGE_SYSTEM_PROMPT,
    StationCandidateBuilder,
    build_stage1_answer,
    build_stage1_prompt,
    build_stage2_answer,
    build_stage2_prompt,
)


class Stage1AudioQwenCollator:
    """Read wav files, build Whisper features, and tokenize prompt/answer."""

    def __init__(
        self,
        whisper_processor: Any,
        qwen_tokenizer: Any,
        target_sample_rate: int = 16000,
        max_prompt_length: int = 256,
        max_answer_length: int = 64,
    ) -> None:
        self.whisper_processor = whisper_processor
        self.qwen_tokenizer = qwen_tokenizer
        self.target_sample_rate = target_sample_rate
        self.max_prompt_length = max_prompt_length
        self.max_answer_length = max_answer_length

    def __call__(self, samples: list[AudioSample]) -> dict[str, Any]:
        audios = [self._read_audio(sample.audio_path) for sample in samples]
        whisper_inputs = self.whisper_processor(
            audios,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )

        prompts = [self._format_chat_prompt(build_stage1_prompt()) for _ in samples]
        answers = [self._append_eos(build_stage1_answer(sample.target_station)) for sample in samples]
        prompt_tokens = self.qwen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        answer_tokens = self.qwen_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return {
            "input_features": whisper_inputs.input_features,
            "prompt_input_ids": prompt_tokens.input_ids,
            "prompt_attention_mask": prompt_tokens.attention_mask,
            "answer_input_ids": answer_tokens.input_ids,
            "answer_attention_mask": answer_tokens.attention_mask,
            "utt_ids": [sample.utt_id for sample in samples],
            "target_stations": [sample.target_station for sample in samples],
            "audio_paths": [sample.audio_path.as_posix() for sample in samples],
        }

    def _format_chat_prompt(self, user_prompt: str) -> str:
        """Format prompts for Qwen-Instruct while keeping a plain fallback."""

        if hasattr(self.qwen_tokenizer, "apply_chat_template"):
            return self.qwen_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": STAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{STAGE_SYSTEM_PROMPT}\n{user_prompt}\n"

    def _append_eos(self, answer: str) -> str:
        """Teach the LM to stop immediately after the JSON answer."""

        eos_token = getattr(self.qwen_tokenizer, "eos_token", None)
        return answer + (eos_token or "")

    def _read_audio(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sample_rate = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sample_rate != self.target_sample_rate:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError("librosa is required for audio resampling") from exc
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sample_rate)
        if not np.isfinite(audio).all():
            raise RuntimeError(f"Audio contains NaN or Inf: {path}")
        return audio


class Stage2AudioQwenCollator(Stage1AudioQwenCollator):
    """Collator for contextual-biasing training with optional hotwords."""

    def __init__(
        self,
        whisper_processor: Any,
        qwen_tokenizer: Any,
        candidate_builder: StationCandidateBuilder,
        with_hotwords: bool = True,
        target_sample_rate: int = 16000,
        max_prompt_length: int = 2048,
        max_answer_length: int = 64,
    ) -> None:
        super().__init__(
            whisper_processor=whisper_processor,
            qwen_tokenizer=qwen_tokenizer,
            target_sample_rate=target_sample_rate,
            max_prompt_length=max_prompt_length,
            max_answer_length=max_answer_length,
        )
        self.candidate_builder = candidate_builder
        self.with_hotwords = with_hotwords

    def __call__(self, samples: list[AudioSample]) -> dict[str, Any]:
        audios = [self._read_audio(sample.audio_path) for sample in samples]
        whisper_inputs = self.whisper_processor(
            audios,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )

        candidate_lists = [self.candidate_builder.build(sample.target_station) for sample in samples]
        prompts = [
            self._format_chat_prompt(build_stage2_prompt(candidates=candidates, with_hotwords=self.with_hotwords))
            for candidates in candidate_lists
        ]
        answers = [self._append_eos(build_stage2_answer(sample.target_station)) for sample in samples]
        prompt_tokens = self.qwen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        answer_tokens = self.qwen_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return {
            "input_features": whisper_inputs.input_features,
            "prompt_input_ids": prompt_tokens.input_ids,
            "prompt_attention_mask": prompt_tokens.attention_mask,
            "answer_input_ids": answer_tokens.input_ids,
            "answer_attention_mask": answer_tokens.attention_mask,
            "utt_ids": [sample.utt_id for sample in samples],
            "target_stations": [sample.target_station for sample in samples],
            "has_stations": [sample.has_station for sample in samples],
            "audio_paths": [sample.audio_path.as_posix() for sample in samples],
            "prompts": prompts,
            "candidate_lists": candidate_lists,
        }
