"""Simplified context encoder: rolling aggregate stats + small MLP.

Replaces the complex ContextEncoder (13.1M params) with ~0.2M params.
Input: pre-computed rolling stats (24 features per team) + team indices.
Output: (B, 2, 512) context tokens [home, away].

The rolling stats capture team strength, pace, shooting efficiency, momentum,
and schedule context — all the signals the decoder needs without the capacity
to memorize individual game outcomes.
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig


class SimpleContextEncoder(nn.Module):
    """Encode pre-computed rolling stats for both teams → (B, 2, 512)."""

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        n_features = config.n_rolling_features  # 24
        hidden = config.simple_context_hidden  # 256
        out_dim = config.hidden_dim  # 512
        team_embed_dim = config.simple_team_embed_dim  # 64

        # Team identity embedding
        self.team_embed = nn.Embedding(config.n_teams, team_embed_dim)

        # Shared MLP: rolling stats + team embed → context token
        self.mlp = nn.Sequential(
            nn.Linear(n_features + team_embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, context_data: dict) -> torch.Tensor:
        """Encode rolling stats for both teams.

        Args:
            context_data: dict with:
                home_rolling_stats: (B, 24)
                away_rolling_stats: (B, 24)
                home_team_idx: (B,)
                away_team_idx: (B,)

        Returns:
            (B, 2, 512) [home_context, away_context]
        """
        contexts = []
        for side in ["home", "away"]:
            stats = context_data[f"{side}_rolling_stats"]  # (B, 24)
            team_idx = context_data[f"{side}_team_idx"]  # (B,)
            team_emb = self.team_embed(team_idx)  # (B, 64)
            x = torch.cat([stats, team_emb], dim=-1)  # (B, 88)
            ctx = self.mlp(x)  # (B, 512)
            contexts.append(ctx)
        return torch.stack(contexts, dim=1)  # (B, 2, 512)
