"""
Per-Match Outcome Pre-training Model (Phase 3 Experiment 6).

Predicts game outcome (win/loss + margin) from player stats with score features masked.
Teaches the encoder "what a winning game looks like" from player performance alone.

Pre-trains: player_embed, per_game_encoder (excluding score_proj).
"""

import torch
import torch.nn as nn

from src.transformer.phase2.config import Phase2ModelConfig
from src.transformer.phase2.models.per_game_encoder import PerGameEncoder


class OutcomePretrainModel(nn.Module):
    """
    Per-match outcome pre-training model.

    Each sample is a single game from one team's perspective.
    Score features (4-d) are zeroed and replaced with a learned mask token,
    forcing the encoder to infer outcome from player performance + context.

    Outputs: win_logit (BCE target) + margin_pred (MSE target).
    """

    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        # Shared player embedding
        self.player_embed = nn.Embedding(
            config.n_players, config.player_embed_dim, padding_idx=0
        )

        # Per-game encoder
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
            n_player_stats=config.n_player_stats,
            stat_hidden_dim=config.stat_hidden_dim,
            n_positions=config.n_positions,
            position_dim=config.position_dim,
            interaction_layers=config.player_interaction_layers,
            interaction_heads=config.player_interaction_heads,
            interaction_ff_dim=config.player_interaction_ff_dim,
            interaction_dropout=config.player_interaction_dropout,
            n_pool_queries=config.player_contribution_n_pool_queries,
        )

        # Learned mask token for score features (replaces zeroed scores)
        self.score_mask_token = nn.Parameter(torch.randn(4))

        # Outcome prediction head: hidden_dim -> 2 (win_logit, margin_pred)
        self.outcome_head = nn.Sequential(
            nn.Linear(h, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, batch: dict) -> dict:
        """
        Forward pass.

        Args:
            batch: dict with scores(B,1,4), opponent_ids(B,1), location(B,1),
                   player_ids(B,1,P), player_stats(B,1,P,16), etc.

        Returns:
            dict with win_logit(B,), margin_pred(B,)
        """
        scores = batch["scores"]  # (B, 1, 4)
        B = scores.shape[0]

        # Zero out score features and replace with learned mask token
        masked_scores = self.score_mask_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        # Encode game with masked scores
        game_repr = self.per_game_encoder(
            scores=masked_scores,
            opponent_ids=batch["opponent_ids"],
            location=batch["location"],
            player_ids=batch["player_ids"],
            player_points=batch["player_points"],
            player_mask=batch["player_mask"],
            player_stats=batch.get("player_stats"),
            player_positions=batch.get("player_positions"),
            player_pm_available=batch.get("player_pm_available"),
        )  # (B, 1, h)

        game_repr = game_repr.squeeze(1)  # (B, h)

        # Predict outcome
        output = self.outcome_head(game_repr)  # (B, 2)
        win_logit = output[:, 0]
        margin_pred = output[:, 1]

        return {
            "win_logit": win_logit,
            "margin_pred": margin_pred,
        }

    def get_transferable_state_dict(self) -> dict:
        """Extract state dict for components that transfer to Phase2Model."""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith(("player_embed.", "per_game_encoder.")):
                state[name] = param.data.clone()
        for name, buf in self.named_buffers():
            if name.startswith(("player_embed.", "per_game_encoder.")):
                state[name] = buf.clone()
        return state
