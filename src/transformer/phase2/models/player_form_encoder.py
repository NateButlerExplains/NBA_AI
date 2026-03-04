"""
Player Form Encoder.

Learns per-player scoring form from raw (points, days_before) appearances
across context games. Replaces hand-engineered stats with learned representations.
"""

import torch
import torch.nn as nn


class PlayerFormEncoder(nn.Module):
    """
    Encode per-player scoring history into form vectors.

    For each of R roster players, takes up to A raw appearances
    (points, days_before) and produces a 64-d form vector.

    Pipeline:
        1. Per-appearance: points_feat(1) || days_embed(32) → Linear(33, 64) + LN + GELU
        2. 1-layer TransformerEncoder(d=64, heads=4, ff=256)
        3. Attention pool (learned query) → (B, R, 64)
        4. Zero-appearance players → learned no_history_embed(64)

    Input:
        points:  (B, R, A) float32 — points / 30.0 per appearance
        days:    (B, R, A) int64   — days_before per appearance
        mask:    (B, R, A) bool    — True = padding

    Output:
        (B, R, form_dim) float32
    """

    def __init__(
        self,
        form_dim: int = 64,
        days_embed_dim: int = 32,
        max_days: int = 180,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.2,
        n_player_stats: int = 0,
        stat_hidden_dim: int = 64,
    ):
        super().__init__()
        self.form_dim = form_dim
        self.n_player_stats = n_player_stats

        # Days embedding (0..179 + padding)
        self.days_embed = nn.Embedding(max_days, days_embed_dim)

        if n_player_stats > 0:
            # Full stats path: stat MLP + days embedding
            self.stat_mlp = nn.Sequential(
                nn.Linear(n_player_stats + 1, stat_hidden_dim),
                nn.LayerNorm(stat_hidden_dim),
                nn.GELU(),
                nn.Linear(stat_hidden_dim, stat_hidden_dim),
                nn.LayerNorm(stat_hidden_dim),
                nn.GELU(),
            )
            input_dim = stat_hidden_dim + days_embed_dim
        else:
            # Legacy path: points(1) + days_embed
            input_dim = 1 + days_embed_dim  # 33

        # Per-appearance projection
        self.appearance_proj = nn.Sequential(
            nn.Linear(input_dim, form_dim),
            nn.LayerNorm(form_dim),
            nn.GELU(),
        )

        # Self-attention over appearances
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=form_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Attention pool: learned query
        self.pool_query = nn.Parameter(torch.randn(1, 1, form_dim))
        self.pool_attn = nn.MultiheadAttention(
            form_dim, n_heads, dropout=dropout, batch_first=True
        )

        # Fallback for players with zero appearances
        self.no_history_embed = nn.Parameter(torch.randn(form_dim))

    def forward(
        self,
        points: torch.Tensor,
        days: torch.Tensor,
        mask: torch.Tensor,
        stats: torch.Tensor = None,
        pm_available: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, R, A) float32 — normalized points per appearance
            days:   (B, R, A) int64   — days_before per appearance
            mask:   (B, R, A) bool    — True = padding
            stats:  (B, R, A, 16) float32, optional — normalized box score stats
            pm_available: (B, R, A) float32, optional — binary indicator

        Returns:
            (B, R, form_dim) float32
        """
        B, R, A = points.shape

        # Reshape to (B*R, A, ...) for processing
        days_flat = days.reshape(B * R, A)         # (B*R, A)
        mask_flat = mask.reshape(B * R, A)         # (B*R, A)

        # Per-appearance features
        days_emb = self.days_embed(days_flat)                    # (B*R, A, 32)

        if self.n_player_stats > 0 and stats is not None:
            stats_flat = stats.reshape(B * R, A, -1)             # (B*R, A, 16)
            pm_flat = pm_available.reshape(B * R, A)             # (B*R, A)
            stat_input = torch.cat([stats_flat, pm_flat.unsqueeze(-1)], dim=-1)  # (B*R, A, 17)
            stat_repr = self.stat_mlp(stat_input)                # (B*R, A, stat_hidden_dim)
            app_input = torch.cat([stat_repr, days_emb], dim=-1) # (B*R, A, stat_hidden_dim+32)
        else:
            points_flat = points.reshape(B * R, A)               # (B*R, A)
            pts_feat = points_flat.unsqueeze(-1)                  # (B*R, A, 1)
            app_input = torch.cat([pts_feat, days_emb], dim=-1)  # (B*R, A, 33)

        app_repr = self.appearance_proj(app_input)                # (B*R, A, form_dim)

        # Self-attention over appearances (masked)
        app_repr = self.transformer(app_repr, src_key_padding_mask=mask_flat)

        # Attention pool per player
        query = self.pool_query.expand(B * R, -1, -1)            # (B*R, 1, 64)
        pooled, _ = self.pool_attn(
            query=query, key=app_repr, value=app_repr,
            key_padding_mask=mask_flat, need_weights=False,
        )                                                         # (B*R, 1, 64)
        pooled = pooled.squeeze(1)                                # (B*R, 64)

        # Reshape back to (B, R, 64)
        form = pooled.reshape(B, R, self.form_dim)

        # Replace zero-appearance players with learned no_history_embed
        all_padding = mask.all(dim=-1)                            # (B, R)
        form = form.masked_fill(all_padding.unsqueeze(-1), 0.0)
        form = form + all_padding.unsqueeze(-1).float() * self.no_history_embed

        return form
