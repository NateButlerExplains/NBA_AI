"""Full context encoder with Phase 3 player-aware signals.

Imports PlayerContributionEncoder from Phase 3 and builds a lighter temporal
encoder (2 layers vs 3, 4 heads vs 8). Supplements with rolling stats as a
stable baseline signal.

Architecture per team:
    PlayerContributionEncoder(16 stats, 5 positions, interaction) -> 256-d per game
    + score_proj(4 -> 128) + opp_embed(30 -> 64) + loc_embed(2 -> 32)
    -> game_combine(480 -> 512) -> LayerNorm
    -> TemporalEncoder (2L, 4H, ff=1024, sinusoidal PE)
    -> 2-query AttentionPool -> 512-d
    + rest_embed(31 -> 64) + rolling_stats_proj(28 -> 128) + team_id_embed(30 -> 64)
    -> team_combine(768 -> 512) -> LayerNorm
    -> team_repr (B, 512)

Forward: context_data dict -> (B, 2, 512) [home, away]

The player_embed is shared and intended to be frozen during training (handled
by the trainer, not this class).
"""

import math

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig
from src.transformer.phase2.models.per_game_encoder import PlayerContributionEncoder


class SinusoidalDaysEncoding(nn.Module):
    """Sinusoidal positional encoding based on days_before value.

    Uses the standard sin/cos encoding formula from "Attention Is All You Need"
    but applied to calendar distance (days before target game) rather than
    sequential position.
    """

    def __init__(self, d_model: int, max_days: int = 180, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Pre-compute sinusoidal encoding table: (max_days, d_model)
        pe = torch.zeros(max_days, d_model)
        position = torch.arange(0, max_days, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_days, d_model)

    def forward(self, x: torch.Tensor, days_before: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal days-before positional encoding.

        Args:
            x: (B, G, d_model) game embeddings.
            days_before: (B, G) int64, days before target game.

        Returns:
            (B, G, d_model) with positional encoding added.
        """
        days_clamped = days_before.clamp(0, self.pe.shape[0] - 1).long()  # (B, G)
        pos_emb = self.pe[days_clamped]  # (B, G, d_model)
        return self.dropout(x + pos_emb)


class FullContextEncoder(nn.Module):
    """Phase 3-style player-aware context encoder for the generative model.

    Imports PlayerContributionEncoder from Phase 3 directly. Builds a simpler
    temporal encoder (2 layers, 4 heads) and supplements with rolling stats.

    Forward signature matches SimpleContextEncoder: (context_data: dict) -> (B, 2, 512).
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim  # 512
        contribution_dim = 256

        # --- Shared player embedding (frozen during training by trainer) ---
        self.player_embed = nn.Embedding(
            config.n_players, config.player_embed_dim, padding_idx=0
        )

        # --- Player contribution encoder (imported from Phase 3) ---
        self.player_encoder = PlayerContributionEncoder(
            player_embed=self.player_embed,
            contribution_dim=contribution_dim,
            n_heads=config.player_pool_heads,
            dropout=config.full_context_player_dropout,
            n_player_stats=16,
            stat_hidden_dim=config.player_stat_dim,
            n_positions=config.n_positions,
            position_dim=config.position_dim,
            interaction_layers=config.player_interaction_layers,
            interaction_heads=config.player_interaction_heads,
            interaction_ff_dim=config.player_interaction_ff_dim,
            interaction_dropout=config.player_interaction_dropout,
            n_pool_queries=1,
        )

        # --- Per-game encoding ---
        self.score_proj = nn.Sequential(
            nn.Linear(4, config.score_dim),
            nn.LayerNorm(config.score_dim),
            nn.GELU(),
        )
        self.opp_embed = nn.Embedding(config.n_teams, config.opponent_dim)
        self.loc_embed = nn.Embedding(2, config.location_dim)

        # Combine: score(128) + opp(64) + loc(32) + player(256) = 480 -> 512
        game_combine_dim = (
            config.score_dim
            + config.opponent_dim
            + config.location_dim
            + contribution_dim
        )
        self.game_combine = nn.Sequential(
            nn.Linear(game_combine_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # --- Temporal encoder (lighter than Phase 3: 2 layers, 4 heads) ---
        self.temporal_pos = SinusoidalDaysEncoding(
            d_model=hidden_dim,
            max_days=config.temporal_max_days,
            dropout=config.full_context_temporal_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.full_context_temporal_heads,
            dim_feedforward=config.full_context_temporal_ff_dim,
            dropout=config.full_context_temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.full_context_temporal_layers,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 2-query attention pooling -> 512-d
        n_pool = config.full_context_temporal_pool_queries
        self.pool_queries = nn.Parameter(torch.randn(1, n_pool, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim,
            config.full_context_temporal_heads,
            dropout=config.full_context_temporal_dropout,
            batch_first=True,
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(n_pool * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # --- Rolling stats projection ---
        self.rolling_proj = nn.Sequential(
            nn.Linear(config.n_rolling_features, config.rolling_stats_proj_dim),
            nn.LayerNorm(config.rolling_stats_proj_dim),
            nn.GELU(),
        )

        # --- Team identity embedding ---
        self.team_id_embed = nn.Embedding(config.n_teams, config.opponent_dim)

        # --- Rest days embedding (0-30) ---
        self.rest_embed = nn.Embedding(config.max_rest_days + 1, config.rest_dim)

        # --- Team combine ---
        # temporal(512) + rest(64) + rolling(128) + team_id(64) = 768 -> 512
        team_combine_dim = (
            hidden_dim
            + config.rest_dim
            + config.rolling_stats_proj_dim
            + config.opponent_dim
        )
        self.team_combine = nn.Sequential(
            nn.Linear(team_combine_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _encode_side(
        self,
        scores: torch.Tensor,
        opponents: torch.Tensor,
        locations: torch.Tensor,
        player_ids: torch.Tensor,
        player_stats: torch.Tensor,
        player_mask: torch.Tensor,
        player_positions: torch.Tensor,
        player_pm_available: torch.Tensor,
        days_before: torch.Tensor,
        game_mask: torch.Tensor,
        rest_days: torch.Tensor,
        rolling_stats: torch.Tensor,
        team_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one side (home or away) into a 512-d team representation.

        Args:
            scores: (B, G, 4) normalized [team, opp, margin, total].
            opponents: (B, G) opponent team indices 0-29.
            locations: (B, G) 0=away, 1=home.
            player_ids: (B, G, P) player IDs.
            player_stats: (B, G, P, 16) normalized box score stats.
            player_mask: (B, G, P) bool, True=padding.
            player_positions: (B, G, P) position indices 0-4.
            player_pm_available: (B, G, P) float, 1.0 for all valid players.
            days_before: (B, G) days before target game.
            game_mask: (B, G) bool, True=padding.
            rest_days: (B,) rest days before target game.
            rolling_stats: (B, n_rolling) rolling aggregate stats.
            team_idx: (B,) team identity index 0-29.

        Returns:
            (B, 512) team representation.
        """
        B, G, P = player_ids.shape

        # --- Convert mask convention ---
        # Collate convention: True=valid, False=padding
        # PyTorch convention: True=padding/ignore, False=valid
        padding_mask = ~game_mask

        # --- Feature dropout: randomly mask out entire games during training ---
        if self.training and self.config.context_feature_dropout > 0:
            drop_prob = self.config.context_feature_dropout
            # Per-game Bernoulli: True = drop this game
            game_drop = torch.rand(B, G, device=padding_mask.device) < drop_prob
            # Only drop valid (non-padded) games
            padding_mask = padding_mask | game_drop

        # --- Construct player_points from stats (pts is index 1) ---
        player_points = player_stats[:, :, :, 1]  # (B, G, P) — already normalized

        # --- Player contribution encoding ---
        # PlayerContributionEncoder expects True=padding convention
        player_padding_mask = ~player_mask
        player_repr = self.player_encoder(
            player_ids=player_ids,
            player_points=player_points,
            player_mask=player_padding_mask,
            player_stats=player_stats,
            player_positions=player_positions,
            player_pm_available=player_pm_available,
        )  # (B, G, 256)

        # Replace NaN with 0 — games with all-padding players produce NaN
        # from empty attention pooling (softmax over all -inf → NaN weights)
        player_repr = torch.nan_to_num(player_repr, nan=0.0)

        # --- Per-game encoding ---
        score_repr = self.score_proj(scores)  # (B, G, 128)
        opp_repr = self.opp_embed(opponents)  # (B, G, 64)
        loc_repr = self.loc_embed(locations)  # (B, G, 32)

        game_repr = torch.cat([score_repr, opp_repr, loc_repr, player_repr], dim=-1)
        game_repr = self.game_combine(game_repr)  # (B, G, 512)

        # Zero out padding game positions to prevent NaN propagation.
        # Padding games have all-masked players → NaN from empty attention pooling.
        # Even with correct key_padding_mask, NaN in V contaminates via 0*NaN=NaN.
        game_repr = game_repr.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # --- Temporal encoder ---
        game_repr = self.temporal_pos(game_repr, days_before)

        game_repr = self.temporal_encoder(game_repr, src_key_padding_mask=padding_mask)
        game_repr = self.temporal_norm(game_repr)

        # --- Attention pooling ---
        queries = self.pool_queries.expand(B, -1, -1)  # (B, n_pool, 512)
        pooled, _ = self.pool_attention(
            queries,
            game_repr,
            game_repr,
            key_padding_mask=padding_mask,
            need_weights=False,
        )  # (B, n_pool, 512)

        n_pool = self.config.full_context_temporal_pool_queries
        pooled_flat = pooled.reshape(B, n_pool * self.config.hidden_dim)
        temporal_repr = self.pool_projection(pooled_flat)  # (B, 512)

        # --- Rolling stats ---
        rolling_repr = self.rolling_proj(rolling_stats)  # (B, 128)

        # --- Team identity ---
        team_repr = self.team_id_embed(team_idx)  # (B, 64)

        # --- Rest days ---
        rest_clamped = rest_days.clamp(0, self.config.max_rest_days)
        rest_repr = self.rest_embed(rest_clamped)  # (B, 64)

        # --- Team combine ---
        combined = torch.cat(
            [temporal_repr, rest_repr, rolling_repr, team_repr], dim=-1
        )
        team_output = self.team_combine(combined)  # (B, 512)

        return team_output

    def forward(self, context_data: dict) -> torch.Tensor:
        """Encode context for both teams.

        Args:
            context_data: dict with per-side keys:
                {side}_scores: (B, G, 4) normalized scores per game
                {side}_opponents: (B, G) opponent team indices
                {side}_locations: (B, G) 0=away, 1=home
                {side}_player_ids: (B, G, P) player IDs
                {side}_player_stats: (B, G, P, 16) normalized box score stats
                {side}_player_mask: (B, G, P) True for padding
                {side}_player_positions: (B, G, P) position indices 0-4
                {side}_days_before: (B, G) days before target game
                {side}_game_mask: (B, G) True for padding
                {side}_rest_days: (B,) rest days before target
                {side}_rolling_stats: (B, n_rolling) rolling aggregate stats
                {side}_team_idx: (B,) team identity index

        Returns:
            (B, 2, 512) context tokens [home, away].
        """
        contexts = []
        for side in ["home", "away"]:
            # Construct player_pm_available as all-ones (required by PlayerContributionEncoder)
            player_mask = context_data[f"{side}_player_mask"]
            player_pm_available = torch.ones_like(player_mask, dtype=torch.float32)

            ctx = self._encode_side(
                scores=context_data[f"{side}_scores"],
                opponents=context_data[f"{side}_opponents"],
                locations=context_data[f"{side}_locations"],
                player_ids=context_data[f"{side}_player_ids"],
                player_stats=context_data[f"{side}_player_stats"],
                player_mask=player_mask,
                player_positions=context_data[f"{side}_player_positions"],
                player_pm_available=player_pm_available,
                days_before=context_data[f"{side}_days_before"],
                game_mask=context_data[f"{side}_game_mask"],
                rest_days=context_data[f"{side}_rest_days"],
                rolling_stats=context_data[f"{side}_rolling_stats"],
                team_idx=context_data[f"{side}_team_idx"],
            )
            contexts.append(ctx)

        return torch.stack(contexts, dim=1)  # (B, 2, 512)
