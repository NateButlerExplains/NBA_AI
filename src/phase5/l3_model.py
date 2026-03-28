"""
L3 Team Model.

Architecture:
  1. Coach embedding with experience-based shrinkage gate
  2. Continuity gate blending team history vs player composition
  3. Input projection + residual MLP producing 128-d team representation

Input: L2 team vector (134-d) + team features (34-d) + roster summary (12-d)
       + coach embedding (16-d) = 196-d total
Output: 128-d team representation capturing coaching/system/organizational effects

Key design: L3 does NOT see individual player vectors — only the 134-d L2 aggregate.
The residual formulation means L3 enriches the representation with information that
player talent (L1) and synergy (L2) alone do not explain.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .l3_config import L3Config


class CoachEmbedding(nn.Module):
    """
    Learned per-coach embedding with experience-based shrinkage gate.

    New coaches with few games get an embedding shrunk toward zero (generic),
    while coaches with substantial data get their full learned embedding.
    Gate: sigmoid(log(games / midpoint)), so at midpoint games the gate = 0.5.
    """

    def __init__(self, cfg: L3Config):
        super().__init__()
        self.embedding = nn.Embedding(cfg.n_coaches, cfg.d_coach_embed)
        self.midpoint = cfg.coach_shrinkage_midpoint

        # Initialize embeddings small — coaches start near generic
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self, coach_idx: torch.Tensor, coach_games: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coach_idx: (B,) integer index per coach
            coach_games: (B,) number of games this coach has coached
        Returns:
            coach_repr: (B, d_coach_embed) shrinkage-gated embedding
        """
        raw = self.embedding(coach_idx)  # (B, d_coach_embed)

        # Shrinkage gate: sigmoid(log(games / midpoint))
        # At midpoint games, gate = sigmoid(0) = 0.5
        # At 0 games, gate -> 0 (fully shrunk)
        # At many games, gate -> 1 (full embedding)
        ratio = coach_games.float().clamp(min=1.0) / self.midpoint
        gate = torch.sigmoid(torch.log(ratio))  # (B,)
        gate = gate.unsqueeze(-1)  # (B, 1)

        return raw * gate  # (B, d_coach_embed)


class ContinuityGate(nn.Module):
    """
    Learned gate that blends team-level history features vs L2 player composition.

    Following FiveThirtyEight's insight: optimal split is ~35% team / ~65% player
    on average, but varies with roster continuity.

    High continuity (>0.7): trust team-level trends (coaching system, culture)
    Low continuity (<0.3): rely mostly on player composition from L2
    """

    def __init__(self):
        super().__init__()
        # Learned affine transform on roster_continuity
        self.w = nn.Parameter(
            torch.tensor(2.0)
        )  # positive: higher continuity -> more team weight
        self.b = nn.Parameter(torch.tensor(-0.5))  # bias: sets average gate value

    def forward(self, roster_continuity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roster_continuity: (B, 1) fraction of minutes returning from last season
        Returns:
            gate: (B, 1) in [0, 1] — how much to weight team history vs L2
        """
        return torch.sigmoid(self.w * roster_continuity + self.b)  # (B, 1)


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> add.

    Pre-norm (LayerNorm before transform) is more stable than post-norm for
    small models. GELU provides smooth non-linearity.
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model)
        Returns:
            (B, d_model) with residual connection
        """
        return x + self.net(self.norm(x))


class TeamModel(nn.Module):
    """
    Level 3: Team Model.

    Produces a 128-d team representation capturing coaching, system, and
    organizational effects beyond player talent and synergy.

    Architecture:
      1. CoachEmbedding: 16-d per coach with shrinkage gate
      2. ContinuityGate: blends team history vs L2 player composition
      3. Input projection: Linear(196, 128)
      4. Residual MLP: 2 ResidualBlocks(128, 256)
      5. Output LayerNorm for stable downstream consumption

    The continuity gate operates on the L2 team vector and team features
    separately, then combines them. High roster continuity increases the
    weight on team-level features (system/coaching persist), while low
    continuity increases reliance on the L2 player-derived representation.
    """

    def __init__(self, cfg: L3Config):
        super().__init__()
        self.cfg = cfg

        # --- Component modules ---
        self.coach_embed = CoachEmbedding(cfg)
        self.continuity_gate = ContinuityGate()

        # --- Input projection ---
        self.input_proj = nn.Linear(cfg.d_input, cfg.d_team)

        # --- Residual MLP ---
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(cfg.d_team, cfg.mlp_hidden, cfg.dropout)
                for _ in range(cfg.n_residual_blocks)
            ]
        )

        # --- Output normalization ---
        self.output_norm = nn.LayerNorm(cfg.d_team)

    def forward(
        self,
        l2_team: torch.Tensor,
        team_features: torch.Tensor,
        roster_summary: torch.Tensor,
        coach_idx: torch.Tensor,
        roster_continuity: torch.Tensor,
        coach_games: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            l2_team: (B, 134) aggregated L2 team vector (frozen)
            team_features: (B, 34) dual-scale rolling stats + coaching + org + defense
            roster_summary: (B, 12) roster composition summary from L1 vectors
            coach_idx: (B,) integer coach index for embedding lookup
            roster_continuity: (B, 1) fraction of minutes returning from prior season
            coach_games: (B,) number of games coached (for shrinkage gate).
                         If None, defaults to midpoint (gate=0.5 for all).
        Returns:
            team_repr: (B, 128) team representation
        """
        B = l2_team.shape[0]
        device = l2_team.device

        # --- Coach embedding with shrinkage ---
        if coach_games is None:
            coach_games = torch.full(
                (B,), self.cfg.coach_shrinkage_midpoint, device=device
            )
        coach_repr = self.coach_embed(coach_idx, coach_games)  # (B, 16)

        # --- Continuity-gated blending ---
        # gate near 1.0 = trust team features; near 0.0 = trust L2 player composition
        gate = self.continuity_gate(roster_continuity)  # (B, 1)

        # Scale team_features by gate, scale l2_team by (1 - gate)
        # This implements the FiveThirtyEight insight: high continuity -> team trends,
        # low continuity -> player composition
        l2_gated = l2_team * (1.0 - gate)  # (B, 134)
        team_feat_gated = team_features * gate  # (B, 34)

        # --- Concatenate all inputs ---
        combined = torch.cat(
            [l2_gated, team_feat_gated, roster_summary, coach_repr], dim=-1
        )  # (B, 196)

        # --- Input projection ---
        h = self.input_proj(combined)  # (B, 128)

        # --- Residual MLP blocks ---
        for block in self.blocks:
            h = block(h)

        # --- Output normalization ---
        team_repr = self.output_norm(h)  # (B, 128)

        return team_repr


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
