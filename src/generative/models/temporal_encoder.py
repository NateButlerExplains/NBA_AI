"""Temporal encoder: transformer over historical game sequence → team context token.

3-layer transformer with sinusoidal days-before positional encoding.
Attention pooling with 4 queries → concat → Linear(2048, 512) → LN.
"""

import math

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig


def sinusoidal_encoding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute sinusoidal positional encoding from continuous positions.

    Args:
        positions: (...) arbitrary shape of position values (e.g., days before).
        dim: embedding dimension (must be even).

    Returns:
        (..., dim) sinusoidal encoding.
    """
    pe = torch.zeros(
        *positions.shape, dim, device=positions.device, dtype=positions.dtype
    )
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=positions.dtype)
        * -(math.log(10000.0) / dim)
    )
    pe[..., 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
    pe[..., 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
    return pe


class TemporalEncoder(nn.Module):
    """3-layer transformer over per-game representations → team context.

    Input: (B, G, 512) per-game representations + days_before positional encoding.
    Output: (B, 512) team context token.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Transformer encoder: pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.temporal_heads,
            dim_feedforward=config.temporal_ff_dim,
            dropout=config.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.temporal_layers,
        )

        # Attention pooling: 4 queries → concat → project
        n_queries = config.temporal_n_pool_queries
        self.pool_queries = nn.Parameter(
            torch.randn(1, n_queries, config.hidden_dim)
            * (1.0 / config.hidden_dim**0.5)
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.temporal_heads,
            batch_first=True,
        )

        # Output projection: concat of 4 pooled queries → 512
        self.output_proj = nn.Sequential(
            nn.Linear(n_queries * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        game_reprs: torch.Tensor,
        days_before: torch.Tensor,
        game_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sequence of historical game representations.

        Args:
            game_reprs: (B, G, 512) per-game encoded representations.
            days_before: (B, G) float32 — days before target game.
            game_mask: (B, G) bool — True for valid games.

        Returns:
            (B, 512) team context representation.
        """
        B, G, D = game_reprs.shape

        # Add sinusoidal positional encoding based on days_before
        pos_enc = sinusoidal_encoding(days_before, D)  # (B, G, 512)
        x = game_reprs + pos_enc

        # src_key_padding_mask: True means IGNORE in PyTorch
        src_key_padding_mask = ~game_mask  # (B, G)

        # Transformer encoding
        x = self.transformer(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, G, 512)

        # Attention pooling with 4 queries
        queries = self.pool_queries.expand(B, -1, -1)  # (B, 4, 512)
        pooled, _ = self.pool_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=src_key_padding_mask,
        )  # (B, 4, 512)

        # Concat queries and project: (B, 2048) → (B, 512)
        pooled_flat = pooled.reshape(B, -1)
        return self.output_proj(pooled_flat)
