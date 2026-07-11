"""Speech adapter modules for mapping Whisper frames to query tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanPoolMLPAdapter(nn.Module):
    """Compress [B, T, D] Whisper states into fixed [B, Q, D] speech tokens."""

    def __init__(
        self,
        input_size: int,
        num_query_tokens: int = 16,
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected Whisper hidden states [B,T,D], got {tuple(hidden_states.shape)}")
        pooled = F.adaptive_avg_pool1d(
            hidden_states.transpose(1, 2),
            output_size=self.num_query_tokens,
        ).transpose(1, 2)
        return self.mlp(pooled)


class AttentionPoolAdapter(nn.Module):
    """Use learnable queries to attend over Whisper frames.

    This keeps the existing speech-token interface but avoids collapsing station
    details with pure temporal averaging.
    """

    def __init__(
        self,
        input_size: int,
        num_query_tokens: int = 32,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, input_size) * 0.02)
        self.input_norm = nn.LayerNorm(input_size)
        self.query_norm = nn.LayerNorm(input_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected Whisper hidden states [B,T,D], got {tuple(hidden_states.shape)}")
        batch_size = hidden_states.shape[0]
        keys = self.input_norm(hidden_states)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_norm(queries)
        attended, _ = self.cross_attn(query=queries, key=keys, value=keys, need_weights=False)
        queries = queries + attended
        refined, _ = self.self_attn(query=queries, key=queries, value=queries, need_weights=False)
        queries = queries + refined
        return queries + self.ffn(queries)


def build_speech_adapter(
    adapter_type: str,
    input_size: int,
    num_query_tokens: int,
    hidden_size: int,
    dropout: float,
) -> nn.Module:
    if adapter_type == "mean_pool_mlp":
        return MeanPoolMLPAdapter(
            input_size=input_size,
            num_query_tokens=num_query_tokens,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    if adapter_type == "attention_pool_mlp":
        return AttentionPoolAdapter(
            input_size=input_size,
            num_query_tokens=num_query_tokens,
            hidden_size=hidden_size,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported adapter_type: {adapter_type}")
