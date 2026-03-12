"""Context encoder: season history for both teams → (B, 2, 512) context tokens.

Pipeline per team:
  For each historical game:
    score_features [4] → Linear(4, 128)
    opponent_id → Embedding(30, 64)
    location (home/away) → Embedding(2, 32)
    players → PlayerEncoder → (256,)
    concat [128+64+32+256=480] → Linear(480, 512) → LN → (512,)

  Stack games → TemporalEncoder (with days_before pos encoding) → (512,)

  Concat with rest_days_embed → Linear(512+64, 512) → LN → team_context
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig
from src.generative.models.player_encoder import PlayerEncoder
from src.generative.models.temporal_encoder import TemporalEncoder


class ContextEncoder(nn.Module):
    """Encode season history for both teams → (B, 2, 512) context tokens."""

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config

        # Shared player embedding (used by PlayerEncoder)
        self.player_embed = nn.Embedding(config.n_players, config.player_embed_dim)

        # Player encoder (shares the embedding)
        self.player_encoder = PlayerEncoder(config, self.player_embed)

        # Per-game feature projections
        self.score_proj = nn.Sequential(
            nn.Linear(config.score_input_dim, config.score_dim),  # 4 → 128
            nn.GELU(),
        )
        self.opponent_embed = nn.Embedding(config.n_teams, config.opponent_dim)  # 30 → 64
        self.location_embed = nn.Embedding(2, config.location_dim)  # 2 → 32

        # Combine per-game features: 128 + 64 + 32 + 256 = 480 → 512
        game_combine_in = config.score_dim + config.opponent_dim + config.location_dim + config.player_hidden_dim
        self.game_combine = nn.Sequential(
            nn.Linear(game_combine_in, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(config)

        # Rest days embedding and final projection
        self.rest_embed = nn.Embedding(config.max_rest_days + 1, config.rest_dim)  # 31 → 64
        self.context_proj = nn.Sequential(
            nn.Linear(config.hidden_dim + config.rest_dim, config.hidden_dim),  # 512+64 → 512
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, context_data: dict) -> torch.Tensor:
        """Encode context for both teams.

        Args:
            context_data: dict with keys per team (home/away):
                {side}_scores: (B, G, 4) score features
                {side}_opponents: (B, G) opponent team indices
                {side}_locations: (B, G) 0=away, 1=home
                {side}_player_ids: (B, G, P) player indices
                {side}_player_stats: (B, G, P, 16) player stats
                {side}_player_mask: (B, G, P) valid player mask
                {side}_days_before: (B, G) days before target
                {side}_game_mask: (B, G) valid game mask
                {side}_rest_days: (B,) rest days before target

        Returns:
            (B, 2, 512) [home_context, away_context]
        """
        contexts = []
        for side in ["home", "away"]:
            ctx = self._encode_team(context_data, side)
            contexts.append(ctx)
        return torch.stack(contexts, dim=1)  # (B, 2, 512)

    def _encode_team(self, data: dict, side: str) -> torch.Tensor:
        """Encode a single team's context history.

        Args:
            data: full context_data dict.
            side: "home" or "away".

        Returns:
            (B, 512) team context representation.
        """
        scores = data[f"{side}_scores"]          # (B, G, 4)
        opponents = data[f"{side}_opponents"]    # (B, G)
        locations = data[f"{side}_locations"]    # (B, G)
        player_ids = data[f"{side}_player_ids"]  # (B, G, P)
        player_stats = data[f"{side}_player_stats"]  # (B, G, P, 16)
        player_mask = data[f"{side}_player_mask"]    # (B, G, P)
        days_before = data[f"{side}_days_before"]    # (B, G)
        game_mask = data[f"{side}_game_mask"]        # (B, G)
        rest_days = data[f"{side}_rest_days"]        # (B,)

        B, G, P = player_ids.shape

        # --- Per-game features ---

        # Score features: (B, G, 4) → (B, G, 128)
        score_feats = self.score_proj(scores)

        # Opponent embedding: (B, G) → (B, G, 64)
        opp_feats = self.opponent_embed(opponents)

        # Location embedding: (B, G) → (B, G, 32)
        loc_feats = self.location_embed(locations)

        # Player encoding: reshape (B, G, P, ...) → (B*G, P, ...) for PlayerEncoder
        player_ids_flat = player_ids.reshape(B * G, P)
        player_stats_flat = player_stats.reshape(B * G, P, -1)
        player_mask_flat = player_mask.reshape(B * G, P)

        player_feats_flat = self.player_encoder(
            player_ids_flat, player_stats_flat, player_mask_flat
        )  # (B*G, 256)
        player_feats = player_feats_flat.reshape(B, G, -1)  # (B, G, 256)

        # Combine per-game: concat → (B, G, 480) → (B, G, 512)
        game_feats = torch.cat([score_feats, opp_feats, loc_feats, player_feats], dim=-1)
        game_reprs = self.game_combine(game_feats)  # (B, G, 512)

        # --- Temporal encoding ---
        team_ctx = self.temporal_encoder(game_reprs, days_before, game_mask)  # (B, 512)

        # --- Rest days ---
        # Clamp rest_days to valid embedding range
        rest_clamped = rest_days.clamp(0, self.config.max_rest_days)
        rest_feats = self.rest_embed(rest_clamped)  # (B, 64)

        # Final projection: (B, 576) → (B, 512)
        ctx = self.context_proj(torch.cat([team_ctx, rest_feats], dim=-1))
        return ctx
