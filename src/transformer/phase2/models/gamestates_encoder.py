"""
Phase 2 GameStates Encoder.

512-d internal dimension, attention pooling, gradient checkpointing.
Processes score trajectories for recent N games.
"""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint


class GameStateEmbedding512(nn.Module):
    """Embedding layer for GameStates rows, projecting to 512-d."""

    def __init__(self, hidden_dim: int = 512, embed_dim: int = 16):
        super().__init__()

        self.period_emb = nn.Embedding(5, embed_dim, padding_idx=0)
        self.clock_emb = nn.Embedding(721, embed_dim, padding_idx=0)
        self.home_score_emb = nn.Embedding(51, embed_dim)
        self.away_score_emb = nn.Embedding(51, embed_dim)
        self.margin_emb = nn.Embedding(121, embed_dim)

        self.embed_dim = embed_dim * 5  # 80
        self.projection = nn.Linear(self.embed_dim, hidden_dim)

    def forward(self, periods, clock_buckets, home_score_buckets,
                away_score_buckets, margin_buckets):
        p = self.period_emb(periods)
        c = self.clock_emb(clock_buckets)
        hs = self.home_score_emb(home_score_buckets)
        as_ = self.away_score_emb(away_score_buckets)
        m = self.margin_emb(margin_buckets)

        combined = torch.cat([p, c, hs, as_, m], dim=-1)
        return self.projection(combined)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for GameStates sequences."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Phase2GameStatesEncoder(nn.Module):
    """
    4-layer transformer encoder for GameStates with attention pooling
    and gradient checkpointing.

    Input: gs_data dict (5 features, each (B, N_recent, max_rows)) + gs_lengths (B, N_recent)
    Output: (B, N_recent, 512)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = GameStateEmbedding512(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # ModuleList for per-layer gradient checkpointing
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, gs_data: dict, gs_lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode GameStates for recent games.

        Args:
            gs_data: Dict with keys periods, clock_buckets, home_score_buckets,
                     away_score_buckets, margin_buckets — each (B, N_recent, max_rows)
            gs_lengths: (B, N_recent) — number of valid rows per game

        Returns:
            Game embeddings (B, N_recent, hidden_dim)
        """
        batch_size = gs_data["periods"].shape[0]
        n_games = gs_data["periods"].shape[1]
        max_rows = gs_data["periods"].shape[2]

        # Flatten batch and games: (B * N, max_rows)
        def reshape(t):
            return t.reshape(batch_size * n_games, max_rows)

        periods = reshape(gs_data["periods"])
        clock_buckets = reshape(gs_data["clock_buckets"])
        home_score_buckets = reshape(gs_data["home_score_buckets"])
        away_score_buckets = reshape(gs_data["away_score_buckets"])
        margin_buckets = reshape(gs_data["margin_buckets"])

        # Embed: (B * N, max_rows, hidden_dim)
        x = self.embedding(periods, clock_buckets, home_score_buckets,
                           away_score_buckets, margin_buckets)
        x = self.pos_encoder(x)

        # Create padding mask
        lengths_flat = gs_lengths.reshape(-1)
        mask = torch.arange(max_rows, device=x.device).unsqueeze(0) >= lengths_flat.unsqueeze(1)

        # Apply transformer layers with gradient checkpointing
        for layer in self.layers:
            if self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, None, mask, use_reentrant=False
                )
            else:
                x = layer(x, src_key_padding_mask=mask)

        x = self.output_norm(x)

        # Attention pooling (replaces mean pooling)
        query = self.pool_query.expand(batch_size * n_games, -1, -1)
        pooled, _ = self.pool_attention(
            query, x, x, key_padding_mask=mask, need_weights=False
        )
        game_emb = pooled.squeeze(1)  # (B * N, hidden_dim)

        return game_emb.reshape(batch_size, n_games, self.hidden_dim)
