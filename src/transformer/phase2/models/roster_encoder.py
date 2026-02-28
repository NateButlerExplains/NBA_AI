"""
Phase 2 Roster Encoder.

Project-up-first design: Linear(128, 512) per player before self-attention.
Optionally concatenates player form vectors (64-d) from PlayerFormEncoder.
"""

import torch
import torch.nn as nn

from typing import Optional


class Phase2RosterEncoder(nn.Module):
    """
    Roster encoder with project-up-first architecture.

    1. player_embed(roster_ids) -> (B, R, 128)
    2. Optionally concat form vectors: (B, R, 128+64) = (B, R, 192)
    3. Linear(input_dim, 512) + LN + GELU -> (B, R, 512)
    4. 2-layer TransformerEncoder at 512-d, 8 heads, FF=2048, pre-norm
    5. Attention pool: learned query + MultiheadAttention

    Input: roster_ids (B, max_roster) int64, 0=padding
           form_vectors (B, max_roster, form_dim) float32 [optional]
    Output: (B, 512)
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.2,
        form_dim: int = 0,
    ):
        super().__init__()

        self.player_embed = player_embed  # Shared reference, not owned
        embed_dim = player_embed.embedding_dim  # 128
        self.form_dim = form_dim

        # Project up before self-attention (wider input if form vectors present)
        project_input_dim = embed_dim + form_dim
        self.project_up = nn.Sequential(
            nn.Linear(project_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Self-attention at hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        roster_ids: torch.Tensor,
        form_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            roster_ids: (B, max_roster) int64, 0=padding
            form_vectors: (B, max_roster, form_dim) float32, optional

        Returns:
            (B, hidden_dim)
        """
        batch_size = roster_ids.shape[0]
        padding_mask = roster_ids == 0  # (B, max_roster)

        # Embed and project up
        x = self.player_embed(roster_ids)  # (B, R, 128)

        if form_vectors is not None and self.form_dim > 0:
            x = torch.cat([x, form_vectors], dim=-1)  # (B, R, 128+64)

        x = self.project_up(x)  # (B, R, 512)

        # Self-attention
        x = self.self_attention(x, src_key_padding_mask=padding_mask)

        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(
            query=query, key=x, value=x,
            key_padding_mask=padding_mask, need_weights=False
        )

        return pooled.squeeze(1)  # (B, hidden_dim)
