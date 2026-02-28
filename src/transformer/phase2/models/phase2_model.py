"""
Phase 2 Model — Top-level wiring.

Combines all Phase 2 components into a single forward pass:
GameStates encoder → Per-game encoder → Temporal attention → Roster → Fusion → Prediction.
"""

import torch
import torch.nn as nn

from src.transformer.models.prediction_heads import PredictionHeads, GamePrediction
from src.transformer.phase2.config import Phase2ModelConfig
from src.transformer.phase2.models.gamestates_encoder import Phase2GameStatesEncoder
from src.transformer.phase2.models.temporal_attention import Phase2TemporalAttention
from src.transformer.phase2.models.per_game_encoder import PerGameEncoder
from src.transformer.phase2.models.roster_encoder import Phase2RosterEncoder
from src.transformer.phase2.models.fusion import Phase2Fusion


class Phase2Model(nn.Module):
    """
    Full Phase 2 NBA prediction model.

    Forward pass per team:
    1. gs_encoder(gs_data, gs_lengths) -> (B, N_recent, 512) dynamics
    2. Scatter dynamics into full game sequence at is_recent positions
    3. per_game_encoder(scores, opp, loc, players, dynamics, is_recent) -> (B, G, 512)
    4. temporal_attention(game_reprs, days_before, game_mask) -> (B, 512)
    5. roster_encoder(roster_ids, form_vectors) -> (B, 512)
    6. rest_embed(rest_days) -> (B, 512)
    7. team_combine: concat([season, roster, rest]) -> Linear(1536, 512) -> LN
    8. fusion(home_repr, away_repr) -> (B, 512)
    9. prediction_heads(matchup) -> GamePrediction
    """

    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Shared player embedding
        self.player_embed = nn.Embedding(
            config.n_players, config.player_embed_dim, padding_idx=0
        )

        # GameStates encoder
        self.gs_encoder = Phase2GameStatesEncoder(
            hidden_dim=h,
            num_layers=config.gs_encoder_layers,
            num_heads=config.gs_encoder_heads,
            ff_dim=config.gs_encoder_ff_dim,
            dropout=config.gs_encoder_dropout,
            max_seq_len=config.gs_max_seq_len,
        )

        # Per-game encoder (uses shared player_embed)
        self.per_game_encoder = PerGameEncoder(
            player_embed=self.player_embed,
            hidden_dim=h,
            score_dim=config.score_dim,
            opponent_dim=config.opponent_dim,
            location_dim=config.location_dim,
            contribution_dim=config.player_contribution_dim,
            contribution_heads=config.player_contribution_heads,
            contribution_dropout=config.player_contribution_dropout,
            n_teams=config.n_teams,
            use_ple=config.use_ple,
            n_ple_bins=config.n_ple_bins,
        )

        # Temporal module (transformer or GRU)
        if config.temporal_type == "gru":
            from src.transformer.phase2.models.temporal_gru import Phase2TemporalGRU
            self.temporal_attention = Phase2TemporalGRU(
                hidden_dim=h,
                gru_hidden=config.temporal_gru_hidden,
                num_layers=config.temporal_gru_layers,
                dropout=config.temporal_gru_dropout,
                max_days=config.temporal_max_days,
                time_dim=config.temporal_gru_time_dim,
                n_pool_queries=config.temporal_n_pool_queries,
                pool_heads=config.temporal_heads,
            )
        else:
            self.temporal_attention = Phase2TemporalAttention(
                hidden_dim=h,
                num_layers=config.temporal_layers,
                num_heads=config.temporal_heads,
                ff_dim=config.temporal_ff_dim,
                dropout=config.temporal_dropout,
                max_days=config.temporal_max_days,
                n_pool_queries=config.temporal_n_pool_queries,
                pos_encoding=config.temporal_pos_encoding,
            )

        # Player form encoder (optional)
        self.form_encoder = None
        form_dim = 0
        if config.enable_player_form:
            from src.transformer.phase2.models.player_form_encoder import PlayerFormEncoder
            form_dim = config.player_form_dim
            self.form_encoder = PlayerFormEncoder(
                form_dim=form_dim,
                days_embed_dim=config.player_form_days_dim,
                max_days=config.temporal_max_days,
                n_heads=config.player_form_heads,
                ff_dim=config.player_form_ff_dim,
                dropout=config.player_form_dropout,
            )

        # Roster encoder (uses shared player_embed, optionally wider with form)
        self.roster_encoder = Phase2RosterEncoder(
            player_embed=self.player_embed,
            hidden_dim=h,
            n_heads=config.roster_heads,
            n_layers=config.roster_layers,
            ff_dim=config.roster_ff_dim,
            dropout=config.roster_dropout,
            form_dim=form_dim,
        )

        # Rest days embedding (kept at rest_embed_dim, not wastefully projected to h)
        self.rest_embed = nn.Sequential(
            nn.Embedding(config.max_rest_days, config.rest_embed_dim),
        )

        # Per-team combine: concat([season(h), roster(h), rest(rest_embed_dim)]) -> h
        team_combine_input = h * 2 + config.rest_embed_dim
        self.team_combine = nn.Sequential(
            nn.Linear(team_combine_input, h),
            nn.LayerNorm(h),
        )

        # Fusion
        self.fusion = Phase2Fusion(
            hidden_dim=h,
            dropout=config.fusion_dropout,
            use_cross_attention=config.use_cross_attention_fusion,
            n_heads=config.fusion_heads,
        )

        # Prediction heads (reuse Phase 1)
        self.prediction_heads = PredictionHeads(
            input_dim=h,
            hidden_dim=config.prediction_hidden_dim,
            spread_min_std=config.spread_min_std,
            spread_max_std=config.spread_max_std,
            score_min_std=config.score_min_std,
            dropout=config.prediction_dropout,
            derive_spread=config.derive_spread_from_scores,
        )

    def _encode_team(self, batch: dict, prefix: str) -> torch.Tensor:
        """Encode one team's full context into a single representation."""
        p = prefix  # "home_" or "away_"

        scores = batch[p + "scores"]
        opponent_ids = batch[p + "opponent_ids"]
        location = batch[p + "location"]
        player_ids = batch[p + "player_ids"]
        player_points = batch[p + "player_points"]
        player_mask = batch[p + "player_mask"]
        days_before = batch[p + "days_before"]
        is_recent = batch[p + "is_recent"]
        game_mask = batch[p + "game_mask"]

        B, G = scores.shape[:2]
        h = self.config.hidden_dim

        # 1. Encode GameStates dynamics for recent games
        gs_data = {
            "periods": batch[p + "gs_periods"],
            "clock_buckets": batch[p + "gs_clock_buckets"],
            "home_score_buckets": batch[p + "gs_home_score_buckets"],
            "away_score_buckets": batch[p + "gs_away_score_buckets"],
            "margin_buckets": batch[p + "gs_margin_buckets"],
        }
        gs_lengths = batch[p + "gs_lengths"]

        dynamics_recent = self.gs_encoder(gs_data, gs_lengths)  # (B, N_recent, h)

        # 2. Scatter dynamics into full game sequence at is_recent positions
        dynamics = torch.zeros(B, G, h, device=scores.device, dtype=dynamics_recent.dtype)
        for b in range(B):
            recent_positions = is_recent[b].nonzero(as_tuple=True)[0]
            n_avail = min(len(recent_positions), dynamics_recent.shape[1])
            if n_avail > 0:
                dynamics[b, recent_positions[:n_avail]] = dynamics_recent[b, :n_avail]

        # 3. Per-game encoder
        game_reprs = self.per_game_encoder(
            scores=scores,
            opponent_ids=opponent_ids,
            location=location,
            player_ids=player_ids,
            player_points=player_points,
            player_mask=player_mask,
            dynamics=dynamics,
            is_recent=is_recent,
        )  # (B, G, h)

        # 4. Temporal attention
        season_repr = self.temporal_attention(
            game_reprs, days_before, game_mask
        )  # (B, h)

        # 5. Roster encoder (with optional form vectors)
        roster_ids = batch[p[:-1] + "_roster"]
        form_vectors = None

        if self.form_encoder is not None:
            form_key = p + "roster_form_points"
            if form_key in batch:
                form_vectors = self.form_encoder(
                    points=batch[p + "roster_form_points"],
                    days=batch[p + "roster_form_days"],
                    mask=batch[p + "roster_form_mask"],
                )  # (B, R, form_dim)

        roster_repr = self.roster_encoder(roster_ids, form_vectors)  # (B, h)

        # 6. Rest days (kept at rest_embed_dim, no wasteful projection to h)
        rest_days = batch[p[:-1] + "_rest_days"]
        rest_repr = self.rest_embed[0](rest_days)  # (B, rest_embed_dim)

        # 7. Per-team combine
        team_repr = self.team_combine(
            torch.cat([season_repr, roster_repr, rest_repr], dim=-1)
        )  # (B, h)

        return team_repr

    def forward(self, batch: dict) -> GamePrediction:
        """
        Forward pass through the complete Phase 2 model.

        Args:
            batch: Dict with home_ and away_ prefixed tensors

        Returns:
            GamePrediction with all prediction outputs
        """
        home_repr = self._encode_team(batch, "home_")
        away_repr = self._encode_team(batch, "away_")

        matchup_repr = self.fusion(home_repr, away_repr)

        return self.prediction_heads(matchup_repr)

    def get_num_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        components = {
            "player_embed": self.player_embed,
            "gs_encoder": self.gs_encoder,
            "per_game_encoder": self.per_game_encoder,
            "temporal_attention": self.temporal_attention,
            "roster_encoder": self.roster_encoder,
            "rest_embed": self.rest_embed[0],
            "team_combine": self.team_combine,
            "fusion": self.fusion,
            "prediction_heads": self.prediction_heads,
        }

        if self.form_encoder is not None:
            components["form_encoder"] = self.form_encoder

        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n

        # Shared params are counted under player_embed and not double-counted
        # in per_game_encoder and roster_encoder. Compute actual total.
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
