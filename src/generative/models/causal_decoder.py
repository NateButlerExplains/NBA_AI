"""Causal decoder: transformer decoder with KV cache support.

Pre-norm architecture with sinusoidal positional encoding.
Context tokens (2) get type embeddings; state tokens get positional encoding offset by 2.
Supports both full-sequence training and incremental inference with KV cache.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig  # noqa: F401


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: -inf above diagonal, 0 on/below."""
    return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)


class CausalDecoderLayer(nn.Module):
    """Single decoder layer with KV cache support.

    Pre-norm architecture:
      x = x + self_attn(LN(x))
      x = x + ff(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV cache.

        Training: x is the full sequence, attn_mask is causal, kv_cache=None.
        Inference: x is a single token, attn_mask=None, kv_cache provided.

        Args:
            x: (B, T, D) input — full sequence (training) or single token (inference).
            attn_mask: optional (T, T) causal mask for training.
            kv_cache: optional (cached_keys, cached_values), each (B, S, D).

        Returns:
            (output, updated_kv_cache) — cache is None during training.
        """
        normed = self.norm1(x)

        new_kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        if kv_cache is not None:
            # Inference mode: single query token attending to cache + self
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, normed], dim=1)
            v = torch.cat([cached_v, normed], dim=1)

            attn_out, _ = self.self_attn(query=normed, key=k, value=v)
            new_kv_cache = (k, v)
        else:
            # Training mode: full sequence with explicit causal mask
            attn_out, _ = self.self_attn(
                query=normed, key=normed, value=normed,
                attn_mask=attn_mask,
            )

        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_kv_cache


class CausalDecoder(nn.Module):
    """Stack of CausalDecoderLayers with positional encoding.

    Sinusoidal positional encoding (max_len=800).
    Context type embedding: 2 types (home_ctx=0, away_ctx=1) added to context positions.
    State positions get positional encoding offset by 2 (for the 2 context tokens).
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim

        self.context_type_embed = nn.Embedding(2, config.hidden_dim)

        self.layers = nn.ModuleList([
            CausalDecoderLayer(
                d_model=config.hidden_dim,
                n_heads=config.decoder_heads,
                d_ff=config.decoder_ff_dim,
                dropout=config.decoder_dropout,
            )
            for _ in range(config.decoder_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)

        pe = self._build_sinusoidal_pe(config.decoder_max_seq_len, config.hidden_dim)
        self.register_buffer("pos_encoding", pe)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        context_tokens: torch.Tensor,
        state_embeds: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training mode: full sequence with causal masking.

        Args:
            context_tokens: (B, 2, 512) from context encoder.
            state_embeds: (B, T, 512) from state embedder.
            state_mask: reserved for future use.

        Returns:
            (B, 2+T, 512) decoder output for all positions.
        """
        B, T, D = state_embeds.shape

        # Add context type embeddings
        type_ids = torch.arange(2, device=context_tokens.device)
        context_with_type = context_tokens + self.context_type_embed(type_ids).unsqueeze(0)

        # Add positional encoding
        context_with_type = context_with_type + self.pos_encoding[:2].unsqueeze(0)
        state_embeds = state_embeds + self.pos_encoding[2:2 + T].unsqueeze(0)

        # Concatenate: (B, 2+T, D)
        x = torch.cat([context_with_type, state_embeds], dim=1)

        # Generate explicit causal mask
        total_len = 2 + T
        mask = _causal_mask(total_len, device=x.device)

        for layer in self.layers:
            x, _ = layer(x, attn_mask=mask, kv_cache=None)

        return self.norm(x)

    def init_kv_cache(
        self, context_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Initialize KV cache from context tokens.

        Args:
            context_tokens: (B, 2, 512) context tokens.

        Returns:
            (context_output, kv_caches)
        """
        type_ids = torch.arange(2, device=context_tokens.device)
        x = context_tokens + self.context_type_embed(type_ids).unsqueeze(0)
        x = x + self.pos_encoding[:2].unsqueeze(0)

        mask = _causal_mask(2, device=x.device)

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            normed = layer.norm1(x)
            attn_out, _ = layer.self_attn(
                query=normed, key=normed, value=normed, attn_mask=mask,
            )
            x = x + attn_out
            x = x + layer.ff(layer.norm2(x))
            kv_caches.append((normed.clone(), normed.clone()))

        return self.norm(x), kv_caches

    def decode_step(
        self,
        token_embed: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        step: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Inference mode: single token with KV cache.

        Args:
            token_embed: (B, 1, 512) current token embedding.
            kv_caches: list of (cached_k, cached_v) per layer.
            step: current position in sequence (0-indexed, including context).

        Returns:
            (output, updated_kv_caches)
        """
        x = token_embed + self.pos_encoding[step:step + 1].unsqueeze(0)

        updated_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, cache in zip(self.layers, kv_caches):
            x, new_cache = layer(x, attn_mask=None, kv_cache=cache)
            assert new_cache is not None
            updated_caches.append(new_cache)

        return self.norm(x), updated_caches
