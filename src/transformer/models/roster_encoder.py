"""
Roster Encoder for NBA Game Prediction (Phase 1b).

Encodes a variable-length roster (set of player IDs) into a fixed-size vector.
This tells the model WHO is playing in the target game — the strongest remaining
signal source after Phase 1a/1c experiments showed PBP and schedule features
hit a ceiling at ~12.2 Spread MAE.

Architecture:
    player_ids → shared player_emb (64-d) → Self-Attention (2 layers, 4 heads)
                                                    ↓
                                            Attention Pooling (learned query)
                                                    ↓
                                            Linear(64, 256) → roster_repr

Key design choices:
    - Shared embedding: Reuses EventEncoder's player_emb (64-d). PBP events
      provide hundreds of per-player signals per game; sharing lets the roster
      encoder leverage these richer learned representations.
    - Self-attention: Learns player interactions/synergies. Permutation-invariant
      over roster order (roster is a SET, not a sequence).
    - Attention pooling: Learned query attends to all players, producing a
      fixed-size vector regardless of roster size. More expressive than mean
      pooling — can learn to weight star players vs role players.
    - Single encoder shared between home and away (rosters are symmetric).

Usage:
    # Receives shared embedding from EventEncoder
    player_emb = event_encoder.event_embedding.player_emb
    encoder = RosterEncoder(player_embed=player_emb, hidden_dim=256)
    roster_repr = encoder(roster_ids)  # (batch, hidden_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RosterEncoder(nn.Module):
    """
    Encodes a set of player IDs into a fixed-size roster representation.

    Uses self-attention to model player interactions, then attention pooling
    to aggregate into a single vector.
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        n_heads: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            player_embed: Shared nn.Embedding from EventEncoder (not copied — same object)
            n_heads: Number of attention heads in self-attention layers
            n_layers: Number of transformer encoder layers
            hidden_dim: Output dimension (must match model's hidden_dim for fusion)
            dropout: Dropout probability
        """
        super().__init__()

        self.player_embed = player_embed  # Shared reference, NOT a copy
        embed_dim = player_embed.embedding_dim  # 64-d from EventEncoder

        # Self-attention layers: learn player interactions/synergies.
        # Using batch_first=True for consistency with the rest of the codebase.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Attention pooling: learned query vector attends to all player embeddings.
        # Produces a single fixed-size vector regardless of roster size.
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Project from embed_dim (64) to hidden_dim (256) for fusion concatenation.
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, roster_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of rosters into fixed-size representations.

        Args:
            roster_ids: (batch, max_roster_size) — player token IDs, 0 = padding

        Returns:
            roster_repr: (batch, hidden_dim)
        """
        batch_size = roster_ids.shape[0]

        # Create padding mask: True where player_id == 0 (padding)
        padding_mask = roster_ids == 0  # (batch, max_roster)

        # Embed player IDs using the shared embedding
        x = self.player_embed(roster_ids)  # (batch, max_roster, embed_dim)

        # Self-attention over roster players (learns interactions)
        # src_key_padding_mask: positions with True are ignored by attention
        x = self.self_attention(x, src_key_padding_mask=padding_mask)

        # Attention pooling: learned query attends to all players
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        pooled, _ = self.pool_attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )  # (batch, 1, embed_dim)

        pooled = pooled.squeeze(1)  # (batch, embed_dim)

        # Project to hidden_dim for fusion
        return self.projection(pooled)  # (batch, hidden_dim)
