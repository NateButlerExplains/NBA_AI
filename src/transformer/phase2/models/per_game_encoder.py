"""
Phase 2 Per-Game Encoder.

Encodes each historical game as a holistic embedding containing:
scores + opponent + location + player contributions + (optional) dynamics.
"""

import torch
import torch.nn as nn


class PlayerContributionEncoder(nn.Module):
    """
    Attention-pooled player contributions within a single game.

    Per player: concat([player_emb(128), norm_points(1)]) -> Linear(129, 256) -> LN + GELU
    Attention pool over players: learned query + MultiheadAttention.

    Input: player_ids (B, G, P), player_points (B, G, P), player_mask (B, G, P)
    Output: (B, G, 256)
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        contribution_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.player_embed = player_embed  # Shared reference
        embed_dim = player_embed.embedding_dim  # 128

        self.player_projection = nn.Sequential(
            nn.Linear(embed_dim + 1, contribution_dim),
            nn.LayerNorm(contribution_dim),
            nn.GELU(),
        )

        # Attention pool over players
        self.pool_query = nn.Parameter(torch.randn(1, 1, contribution_dim))
        self.pool_attention = nn.MultiheadAttention(
            contribution_dim, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        player_ids: torch.Tensor,
        player_points: torch.Tensor,
        player_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            player_ids: (B, G, P) int64
            player_points: (B, G, P) float32 (normalized by /30.0)
            player_mask: (B, G, P) bool, True=padding

        Returns:
            (B, G, contribution_dim)
        """
        B, G, P = player_ids.shape

        # Flatten to (B*G, P)
        ids_flat = player_ids.reshape(B * G, P)
        points_flat = player_points.reshape(B * G, P)
        mask_flat = player_mask.reshape(B * G, P)

        # Embed players
        emb = self.player_embed(ids_flat)  # (B*G, P, 128)
        points_feat = points_flat.unsqueeze(-1)  # (B*G, P, 1)

        # Concat and project
        combined = torch.cat([emb, points_feat], dim=-1)  # (B*G, P, 129)
        x = self.player_projection(combined)  # (B*G, P, contribution_dim)

        # Attention pool
        query = self.pool_query.expand(B * G, -1, -1)
        pooled, _ = self.pool_attention(
            query, x, x, key_padding_mask=mask_flat, need_weights=False
        )
        pooled = pooled.squeeze(1)  # (B*G, contribution_dim)

        return pooled.reshape(B, G, -1)


class PerGameEncoder(nn.Module):
    """
    Encodes each historical game into a 512-d embedding.

    Components:
    - Score: Linear(4, 128) + LN + GELU
    - Opponent: Embedding(30, 64)
    - Location: Embedding(2, 32)
    - Player contributions: PlayerContributionEncoder -> (B, G, 256)
    - Context combine: concat([128, 64, 32, 256]) = 480 -> Linear(480, 512) -> LN
    - Dynamics merge: concat([context(512), dynamics(512)]) -> Linear(1024, 512) -> LN

    Input: scores, opponent_ids, location, player_ids/points/mask, dynamics, is_recent
    Output: game_reprs (B, G, 512)
    """

    def __init__(
        self,
        player_embed: nn.Embedding,
        hidden_dim: int = 512,
        score_dim: int = 128,
        opponent_dim: int = 64,
        location_dim: int = 32,
        contribution_dim: int = 256,
        contribution_heads: int = 4,
        contribution_dropout: float = 0.2,
        n_teams: int = 30,
        use_ple: bool = False,
        n_ple_bins: int = 16,
    ):
        super().__init__()

        # Score projection
        if use_ple:
            from src.transformer.phase2.models.piecewise_linear import PiecewiseLinearEncoding
            self.score_proj = PiecewiseLinearEncoding(
                n_features=4, n_bins=n_ple_bins, output_dim=score_dim,
            )
        else:
            self.score_proj = nn.Sequential(
                nn.Linear(4, score_dim),
                nn.LayerNorm(score_dim),
                nn.GELU(),
            )

        # Opponent embedding
        self.opponent_emb = nn.Embedding(n_teams, opponent_dim)

        # Location embedding
        self.location_emb = nn.Embedding(2, location_dim)

        # Player contribution encoder
        self.player_encoder = PlayerContributionEncoder(
            player_embed=player_embed,
            contribution_dim=contribution_dim,
            n_heads=contribution_heads,
            dropout=contribution_dropout,
        )

        # Context combine
        context_dim = score_dim + opponent_dim + location_dim + contribution_dim
        self.context_combine = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Gated dynamics merge (for recent games with GameStates)
        # Instead of concat-with-zeros (which poisons 94% of games with zero input),
        # use an additive gate: context + gate * proj(dynamics) for recent games only
        self.dynamics_gate = nn.Linear(hidden_dim, hidden_dim)
        self.dynamics_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        scores: torch.Tensor,
        opponent_ids: torch.Tensor,
        location: torch.Tensor,
        player_ids: torch.Tensor,
        player_points: torch.Tensor,
        player_mask: torch.Tensor,
        dynamics: torch.Tensor = None,
        is_recent: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            scores: (B, G, 4) float32 — normalized [home, away, margin, total]
            opponent_ids: (B, G) int64 — team index 0-29
            location: (B, G) int64 — 0=away, 1=home
            player_ids: (B, G, P) int64
            player_points: (B, G, P) float32
            player_mask: (B, G, P) bool, True=padding
            dynamics: (B, G, hidden_dim) or None — GameStates dynamics for recent games
            is_recent: (B, G) bool — True for games with dynamics data

        Returns:
            game_reprs: (B, G, hidden_dim)
        """
        B, G = scores.shape[:2]

        # Score features
        score_repr = self.score_proj(scores)  # (B, G, 128)

        # Opponent features
        opp_repr = self.opponent_emb(opponent_ids)  # (B, G, 64)

        # Location features
        loc_repr = self.location_emb(location)  # (B, G, 32)

        # Player contribution features
        player_repr = self.player_encoder(player_ids, player_points, player_mask)  # (B, G, 256)

        # Combine context features
        context = torch.cat([score_repr, opp_repr, loc_repr, player_repr], dim=-1)
        context = self.context_combine(context)  # (B, G, hidden_dim)

        # Gated dynamics merge for recent games only
        # Non-recent games (94%) pass through as pure context — no zero-concat pollution
        game_reprs = context

        if dynamics is not None and is_recent is not None:
            context = context.to(dynamics.dtype)
            game_reprs = context.clone()

            # Compute gate and projection over all positions
            gate = torch.sigmoid(self.dynamics_gate(dynamics))  # (B, G, h)
            projected = self.dynamics_proj(dynamics)  # (B, G, h)

            # Apply only at recent positions (where dynamics is non-zero)
            mask = is_recent.unsqueeze(-1).float()  # (B, G, 1)
            game_reprs = game_reprs + mask * gate * projected

        return game_reprs
