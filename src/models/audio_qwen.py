"""Audio-conditioned Qwen model for stage-1 alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.lora_utils import apply_lora, load_lora_for_inference
from src.models.speech_adapter import build_speech_adapter
from src.models.whisper_encoder import FrozenWhisperEncoder


class AudioQwenForStage1(nn.Module):
    """Frozen Whisper + trainable adapter/projector + frozen Qwen LM."""

    def __init__(
        self,
        whisper_model_path: str | Path,
        qwen_model_path: str | Path,
        adapter_type: str = "mean_pool_mlp",
        num_query_tokens: int = 16,
        adapter_hidden_size: int = 1024,
        adapter_dropout: float = 0.1,
        freeze_whisper: bool = True,
        freeze_qwen: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.whisper = FrozenWhisperEncoder(whisper_model_path, freeze=freeze_whisper)
        qwen_path = Path(qwen_model_path)
        if not qwen_path.exists():
            raise FileNotFoundError(f"Qwen model path not found: {qwen_path}")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            str(qwen_path),
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        if freeze_qwen:
            self.freeze_qwen()

        self.adapter = build_speech_adapter(
            adapter_type=adapter_type,
            input_size=self.whisper.hidden_size,
            num_query_tokens=num_query_tokens,
            hidden_size=adapter_hidden_size,
            dropout=adapter_dropout,
        )
        qwen_hidden = int(self.qwen.config.hidden_size)
        self.projector = nn.Linear(self.whisper.hidden_size, qwen_hidden)
        self.num_query_tokens = num_query_tokens

    def freeze_qwen(self) -> None:
        self.qwen.eval()
        for parameter in self.qwen.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> "AudioQwenForStage1":
        super().train(mode)
        self.whisper.encoder.eval()
        self.qwen.eval()
        return self

    def forward(
        self,
        input_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        input_features = input_features.to(device)
        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)
        answer_input_ids = answer_input_ids.to(device)
        answer_attention_mask = answer_attention_mask.to(device)

        with torch.no_grad():
            whisper_states = self.whisper(input_features)

        speech_tokens = self.adapter(whisper_states.to(dtype=self.projector.weight.dtype))
        speech_embeds = self.projector(speech_tokens)

        embedding = self.qwen.get_input_embeddings()
        prompt_embeds = embedding(prompt_input_ids)
        answer_embeds = embedding(answer_input_ids)
        inputs_embeds = torch.cat([prompt_embeds, speech_embeds, answer_embeds], dim=1)

        batch_size = input_features.shape[0]
        speech_attention = torch.ones(
            batch_size,
            speech_embeds.shape[1],
            dtype=prompt_attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([prompt_attention_mask, speech_attention, answer_attention_mask], dim=1)

        ignore_prompt = torch.full_like(prompt_input_ids, -100)
        ignore_speech = torch.full(
            (batch_size, speech_embeds.shape[1]),
            -100,
            dtype=answer_input_ids.dtype,
            device=device,
        )
        answer_labels = answer_input_ids.masked_fill(answer_attention_mask == 0, -100)
        labels = torch.cat([ignore_prompt, ignore_speech, answer_labels], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "whisper_hidden_shape": tuple(whisper_states.shape),
            "speech_embeds_shape": tuple(speech_embeds.shape),
            "inputs_embeds_shape": tuple(inputs_embeds.shape),
        }

    def trainable_state_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.state_dict(),
            "projector": self.projector.state_dict(),
        }


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def trainable_parameter_report(model: nn.Module) -> list[tuple[str, int]]:
    return [
        (name, parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


class AudioQwenForStage2(nn.Module):
    """Frozen Whisper + stage-1 adapter/projector + Qwen LoRA fine-tuning."""

    def __init__(
        self,
        whisper_model_path: str | Path,
        qwen_model_path: str | Path,
        adapter_type: str = "mean_pool_mlp",
        num_query_tokens: int = 16,
        adapter_hidden_size: int = 1024,
        adapter_dropout: float = 0.1,
        freeze_whisper: bool = True,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.whisper = FrozenWhisperEncoder(whisper_model_path, freeze=freeze_whisper)
        qwen_path = Path(qwen_model_path)
        if not qwen_path.exists():
            raise FileNotFoundError(f"Qwen model path not found: {qwen_path}")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            str(qwen_path),
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        self.freeze_qwen_base()

        self.adapter = build_speech_adapter(
            adapter_type=adapter_type,
            input_size=self.whisper.hidden_size,
            num_query_tokens=num_query_tokens,
            hidden_size=adapter_hidden_size,
            dropout=adapter_dropout,
        )
        qwen_hidden = int(self.qwen.config.hidden_size)
        self.projector = nn.Linear(self.whisper.hidden_size, qwen_hidden)
        self.num_query_tokens = num_query_tokens

    def freeze_qwen_base(self) -> None:
        for parameter in self.qwen.parameters():
            parameter.requires_grad = False

    def enable_lora(
        self,
        target_modules: list[str],
        r: int,
        alpha: int,
        dropout: float,
        bias: str = "none",
    ) -> None:
        self.qwen = apply_lora(
            self.qwen,
            target_modules=target_modules,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=bias,
        )

    def load_lora(self, lora_dir: str | Path) -> None:
        self.qwen = load_lora_for_inference(self.qwen, lora_dir)

    def load_stage1_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        if "adapter" not in state or "projector" not in state:
            raise ValueError(f"Invalid stage-1 checkpoint, missing adapter/projector: {path}")
        adapter_result = self.adapter.load_state_dict(state["adapter"], strict=False)
        if adapter_result.missing_keys or adapter_result.unexpected_keys:
            print(
                "warning: stage-1 adapter checkpoint is partially incompatible; "
                f"missing={adapter_result.missing_keys} unexpected={adapter_result.unexpected_keys}. "
                "The current adapter will train from its initialized parameters."
            )
        self.projector.load_state_dict(state["projector"])

    def load_adapter_projector(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter/projector checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        self.adapter.load_state_dict(state["adapter"])
        self.projector.load_state_dict(state["projector"])

    def train(self, mode: bool = True) -> "AudioQwenForStage2":
        super().train(mode)
        self.whisper.encoder.eval()
        self.qwen.train(mode)
        return self

    def build_conditioned_embeds(
        self,
        input_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        input_features = input_features.to(device)
        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        with torch.no_grad():
            whisper_states = self.whisper(input_features)

        speech_tokens = self.adapter(whisper_states.to(dtype=self.projector.weight.dtype))
        speech_embeds = self.projector(speech_tokens)
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_input_ids)
        inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)

        speech_attention = torch.ones(
            input_features.shape[0],
            speech_embeds.shape[1],
            dtype=prompt_attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([prompt_attention_mask, speech_attention], dim=1)
        return inputs_embeds, attention_mask, whisper_states, speech_embeds

    def forward(
        self,
        input_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        answer_input_ids = answer_input_ids.to(device)
        answer_attention_mask = answer_attention_mask.to(device)

        prefix_embeds, prefix_attention, whisper_states, speech_embeds = self.build_conditioned_embeds(
            input_features=input_features,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        answer_embeds = self.qwen.get_input_embeddings()(answer_input_ids)
        inputs_embeds = torch.cat([prefix_embeds, answer_embeds], dim=1)
        attention_mask = torch.cat([prefix_attention, answer_attention_mask], dim=1)

        ignore_prefix = torch.full(
            (answer_input_ids.shape[0], prefix_embeds.shape[1]),
            -100,
            dtype=answer_input_ids.dtype,
            device=device,
        )
        answer_labels = answer_input_ids.masked_fill(answer_attention_mask == 0, -100)
        labels = torch.cat([ignore_prefix, answer_labels], dim=1)

        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "whisper_hidden_shape": tuple(whisper_states.shape),
            "speech_embeds_shape": tuple(speech_embeds.shape),
            "inputs_embeds_shape": tuple(inputs_embeds.shape),
        }

    def generate(
        self,
        input_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        inputs_embeds, attention_mask, _, _ = self.build_conditioned_embeds(
            input_features=input_features,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        generation_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if temperature is not None and do_sample:
            generation_kwargs["temperature"] = temperature
        return self.qwen.generate(**generation_kwargs)

    def adapter_projector_state_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter.state_dict(),
            "projector": self.projector.state_dict(),
        }
