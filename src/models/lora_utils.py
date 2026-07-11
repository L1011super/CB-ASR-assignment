"""PEFT LoRA helpers for Qwen fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch import nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:  # pragma: no cover - only exercised without deps.
    raise ImportError("Stage-2 training requires peft. Install it with `pip install peft`.") from exc


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def filter_lora_targets(model: nn.Module, targets: Sequence[str]) -> list[str]:
    """Keep only LoRA target module suffixes that exist in the model."""

    existing = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    return [target for target in targets if target in existing]


def apply_lora(
    model: nn.Module,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    bias: str = "none",
) -> nn.Module:
    """Wrap a causal LM with LoRA after filtering unavailable target modules."""

    filtered = filter_lora_targets(model, target_modules)
    if not filtered:
        raise ValueError(f"No LoRA target modules found. Requested: {list(target_modules)}")
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=filtered,
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_lora_adapter(model: nn.Module, lora_dir: str | Path, is_trainable: bool = False) -> nn.Module:
    """Load a saved LoRA adapter onto a base causal LM."""

    return PeftModel.from_pretrained(model, str(lora_dir), is_trainable=is_trainable)


def load_lora_for_inference(model: nn.Module, lora_dir: str | Path) -> nn.Module:
    """Load a saved LoRA adapter onto a base causal LM for inference."""

    return load_lora_adapter(model, lora_dir, is_trainable=False)


def lora_target_report(model: nn.Module, targets: Sequence[str]) -> dict[str, Any]:
    """Return target filtering details for logs/checkpoints."""

    return {
        "requested": list(targets),
        "available": filter_lora_targets(model, targets),
    }
