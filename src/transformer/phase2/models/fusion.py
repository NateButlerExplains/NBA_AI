"""
Phase 2 Fusion with Interaction Features.

Two modes:
1. MLP (default): Concat([home, away, home-away, home*away]) -> MLP -> + diff residual.
2. Cross-attention: home/away cross-attend to each other -> combine with diff residual.

The residual connection ensures the raw team-difference signal always reaches
the prediction heads, even if the MLP compresses other dimensions. Without it,
the MLP collapses the 512-dim representation to ~3 effective dimensions,
causing spread prediction collapse (mean |pred| 3.68 vs actual 12.43).
"""

import torch
import torch.nn as nn


class Phase2Fusion(nn.Module):
    """
    Fusion with explicit interaction features and diff residual.

    Input: home_repr (B, 512), away_repr (B, 512)
    Output: matchup_repr (B, 512)

    MLP mode: learns non-linear matchup interactions (style clashes, etc.)
    Cross-attention mode: dynamic bilinear interaction between home/away.
    Both preserve the linear team-difference signal via residual.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_cross_attention: bool = False,
        n_heads: int = 8,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True,
            )
            # home_attn + away_attn + diff -> hidden_dim
            self.combine = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        else:
            input_dim = hidden_dim * 4  # home, away, diff, product

            # Interaction MLP — no final LN/GELU/Dropout since the output is
            # summed with the raw diff residual; the prediction heads have their
            # own normalization layers.
            self.interaction = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

    def forward(self, home_repr: torch.Tensor, away_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            home_repr: (B, hidden_dim)
            away_repr: (B, hidden_dim)

        Returns:
            matchup_repr: (B, hidden_dim)
        """
        diff = home_repr - away_repr

        if self.use_cross_attention:
            # Unsqueeze to (B, 1, h) for attention (sequence length 1)
            home_q = home_repr.unsqueeze(1)
            away_q = away_repr.unsqueeze(1)

            # Cross-attend: home queries against away, and vice versa
            home_attn, _ = self.cross_attn(home_q, away_q, away_q)
            away_attn, _ = self.cross_attn(away_q, home_q, home_q)

            home_attn = home_attn.squeeze(1)  # (B, h)
            away_attn = away_attn.squeeze(1)  # (B, h)

            combined = torch.cat([home_attn, away_attn, diff], dim=-1)
            return self.combine(combined) + diff
        else:
            product = home_repr * away_repr
            combined = torch.cat([home_repr, away_repr, diff, product], dim=-1)
            return self.interaction(combined) + diff
