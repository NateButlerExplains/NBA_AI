"""Player encoder: variable-count players → fixed-size team representation.

Per player:
  player_embed(player_id) → (P, 128)
  Linear(16_stats, 64) → GELU → (P, 64)
  concat → Linear(192, 256) → LN → (P, 256)

Attention pool (1 query, 4 heads) → (256,)
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig


class PlayerEncoder(nn.Module):
    """Encode variable-count players into a fixed-size representation per game.

    Uses a shared player embedding (passed in from ContextEncoder) combined with
    projected per-game stats, then attention-pools over the roster to produce a
    single 256-d vector.
    """

    def __init__(self, config: GenerativeModelConfig, player_embed: nn.Embedding) -> None:
        super().__init__()
        self.player_embed = player_embed  # shared, (n_players, player_embed_dim=128)

        # Project raw stats (16-d) → stat_dim (64-d)
        self.stat_proj = nn.Sequential(
            nn.Linear(config.n_player_stats if hasattr(config, "n_player_stats") else 16, config.player_stat_dim),
            nn.GELU(),
        )

        # Combine embed + stat projection → player_hidden_dim
        combine_in = config.player_embed_dim + config.player_stat_dim  # 128 + 64 = 192
        self.combine = nn.Sequential(
            nn.Linear(combine_in, config.player_hidden_dim),
            nn.LayerNorm(config.player_hidden_dim),
        )

        # Attention pooling: 1 learned query, 4 heads
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.player_hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=config.player_hidden_dim,
            num_heads=config.player_pool_heads,
            batch_first=True,
        )

        self.hidden_dim = config.player_hidden_dim

    def forward(
        self,
        player_ids: torch.Tensor,
        player_stats: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of player rosters into fixed-size representations.

        Args:
            player_ids: (B, P) int64 — contiguous player indices.
            player_stats: (B, P, 16) float32 — normalized per-game stats.
            player_mask: (B, P) bool — True for valid players.

        Returns:
            (B, 256) pooled player representation.
        """
        B, P = player_ids.shape

        # Embed players: (B, P, 128)
        embeds = self.player_embed(player_ids)

        # Project stats: (B, P, 64)
        stat_feats = self.stat_proj(player_stats)

        # Concat and combine: (B, P, 256)
        combined = self.combine(torch.cat([embeds, stat_feats], dim=-1))

        # Build key_padding_mask for attention: True means IGNORE in PyTorch MHA
        key_padding_mask = ~player_mask  # (B, P) — True for invalid/padded players

        # Expand query for batch: (B, 1, 256)
        query = self.pool_query.expand(B, -1, -1)

        # Attention pool: query attends to player representations
        pooled, _ = self.pool_attn(
            query=query,
            key=combined,
            value=combined,
            key_padding_mask=key_padding_mask,
        )

        # (B, 1, 256) → (B, 256)
        return pooled.squeeze(1)
