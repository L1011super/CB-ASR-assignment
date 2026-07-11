"""Frozen Whisper encoder wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import WhisperModel


class FrozenWhisperEncoder(nn.Module):
    """Load local Whisper and expose encoder hidden states."""

    def __init__(self, model_path: str | Path, freeze: bool = True) -> None:
        super().__init__()
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Whisper model path not found: {path}")
        self.model = WhisperModel.from_pretrained(str(path), local_files_only=True)
        self.encoder = self.model.encoder
        self.config = self.model.config
        if freeze:
            self.freeze()

    @property
    def hidden_size(self) -> int:
        return int(self.config.d_model)

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        outputs: Any = self.encoder(input_features=input_features, return_dict=True)
        return outputs.last_hidden_state
