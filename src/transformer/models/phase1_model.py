"""
Phase 1 Complete Model for NBA Game Prediction.

Combines all components into a single end-to-end model:
1. Event Encoder - Transforms PBP sequences to game embeddings
2. Temporal Attention - Attends over historical games
3. Fusion Layer - Combines home and away history streams
4. Prediction Heads - Probabilistic outputs

Usage:
    model = Phase1Model(config)
    predictions = model(home_history, away_history)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.transformer.training.config import ModelConfig
from src.transformer.models.event_encoder import EventEncoder
from src.transformer.models.temporal_attention import TemporalAttention
from src.transformer.models.fusion import SimpleFusion, CrossAttentionFusion
from src.transformer.models.roster_encoder import RosterEncoder
from src.transformer.models.prediction_heads import (
    PredictionHeads,
    GamePrediction,
)


class Phase1Model(nn.Module):
    """
    Complete Phase 1 model for NBA game prediction.

    THIS IS THE TOP-LEVEL MODEL that wires all the components together.
    It orchestrates the entire prediction pipeline from raw tokenized data
    to probabilistic game predictions.

    FORWARD PASS DATA FLOW:
    Raw PBP tokens -> EventEncoder (per-game embeddings) -> TemporalAttention (team history)
                                                                      |
                                                              SimpleFusion (MLP)
                                                                      |
                                                              PredictionHeads
                                                                      |
                                                    GamePrediction (spread, scores, win prob)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Phase 1 Model.

        Args:
            config: ModelConfig dataclass with all hyperparameters
        """
        super().__init__()
        self.config = config

        # ========== Stream 1: PBP Sequence Processing ==========
        self._use_history = getattr(config, "use_history", True)

        if self._use_history:
            # EventEncoder: tokenized play-by-play -> per-game embeddings
            # Takes each game's sequence of plays and produces a single 256-d vector per game.
            self.event_encoder = EventEncoder(
                vocab_sizes=config.vocab_sizes,
                hidden_dim=config.hidden_dim,
                num_layers=config.event_encoder_layers,
                num_heads=config.event_encoder_heads,
                dropout=config.dropout,
                max_seq_len=config.max_pbp_length + 1,  # +1 to accommodate CLS token if used
                pooling="mean",  # Average all play embeddings to get game embedding
            )

            # TemporalAttention: sequence of game embeddings -> single team history vector
            # Processes the last N games to capture recent form, trends, and patterns.
            self.temporal_attention = TemporalAttention(
                hidden_dim=config.hidden_dim,
                num_layers=config.temporal_attention_layers,
                num_heads=config.temporal_attention_heads,
                dropout=config.dropout,
                max_games=config.max_history_games,
                positional_encoding="learned",   # Learn position embeddings for each game slot
                pooling="attention",             # Learn which games matter most
            )
        else:
            self.event_encoder = None
            self.temporal_attention = None

        # ========== Roster Encoder (Phase 1b) ==========
        # Encodes target game roster (who is playing) into a fixed-size vector.
        if getattr(config, "use_roster", False):
            if self._use_history:
                # Share player embedding with EventEncoder — PBP events provide
                # rich per-player signal that the roster encoder leverages.
                shared_player_emb = self.event_encoder.embedding.player_emb
            else:
                # Standalone player embedding when history is disabled.
                # Same size as EventEncoder's player_emb (64-d).
                shared_player_emb = nn.Embedding(
                    config.vocab_sizes["player"], 64, padding_idx=0
                )
                self.standalone_player_emb = shared_player_emb

            self.roster_encoder = RosterEncoder(
                player_embed=shared_player_emb,
                n_heads=config.roster_attention_heads,
                n_layers=config.roster_attention_layers,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        else:
            self.roster_encoder = None

        # ========== Rest Days Embedding (Combined Phase 1) ==========
        # Embeds the number of rest days before the target game as a per-team feature.
        # This is the strongest schedule-related signal from XGBoost (rest_days, B2B).
        if getattr(config, "use_rest_days", False):
            self.rest_days_embed = nn.Sequential(
                nn.Embedding(30, 32),        # 0-29 rest days → 32-d
                nn.Linear(32, config.hidden_dim),  # → 256-d to match hidden_dim
                nn.LayerNorm(config.hidden_dim),
            )
        else:
            self.rest_days_embed = None

        # ========== Per-Team Combination ==========
        # When roster and/or rest_days are enabled, we combine all per-team
        # signals (history + roster + rest) via element-wise addition into a
        # single team representation before fusion. This is cleaner than feeding
        # 6 separate streams: fusion operates on 2 team representations.
        self._has_per_team_combine = (
            self.roster_encoder is not None or self.rest_days_embed is not None
        )
        if self._has_per_team_combine:
            self.team_combine_norm = nn.LayerNorm(config.hidden_dim)

        # ========== Fusion: Combine Home and Away ==========
        # After per-team combination, fusion always receives exactly 2 streams
        # (home_team, away_team). Select fusion type based on config.
        fusion_type = getattr(config, "fusion_type", "simple")
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                hidden_dim=config.hidden_dim,
                n_heads=getattr(config, "cross_attention_heads", 4),
                dropout=config.dropout,
            )
        else:
            self.fusion = SimpleFusion(
                hidden_dim=config.hidden_dim,
                n_input_streams=2,  # Always 2 after per-team combine
                dropout=config.dropout,
            )

        # ========== Prediction Heads ==========
        # Convert the matchup representation into actual game predictions.
        # Outputs: spread (mean + std), home score (mean + std),
        #          away score (mean + std), win probability.
        self.prediction_heads = PredictionHeads(
            input_dim=config.hidden_dim,
            hidden_dim=128,
            spread_min_std=1.0,   # Spread uncertainty >= 1 point
            score_min_std=5.0,    # Score uncertainty >= 5 points
            dropout=config.dropout,
        )

        # ========== Schedule Embeddings (Phase 1c) ==========
        # Temporal embeddings added to per-game embeddings before TemporalAttention.
        # Lets the model learn rest effects, B2B fatigue, tanking, playoff push, etc.
        # Shared across home and away streams (temporal encoding is team-agnostic).
        if getattr(config, "n_schedule_features", 0) > 0:
            self.days_before_embed = nn.Embedding(180, config.hidden_dim)
            self.season_game_embed = nn.Embedding(110, config.hidden_dim)
        else:
            self.days_before_embed = None
            self.season_game_embed = None

    def forward(
        self,
        home_history: dict[str, torch.Tensor],
        away_history: dict[str, torch.Tensor],
        home_roster: Optional[torch.Tensor] = None,
        away_roster: Optional[torch.Tensor] = None,
        home_rest_days: Optional[torch.Tensor] = None,
        away_rest_days: Optional[torch.Tensor] = None,
    ) -> GamePrediction:
        """
        Forward pass through the complete model.

        Args:
            home_history: Dict of tensors for home team's historical games
                - action_type_ids: (batch, n_games, max_plays)
                - sub_type_ids: (batch, n_games, max_plays)
                - periods: (batch, n_games, max_plays)
                - clock_buckets: (batch, n_games, max_plays)
                - team_indicators: (batch, n_games, max_plays)
                - score_diff_buckets: (batch, n_games, max_plays)
                - player_ids: (batch, n_games, max_plays)
                - shot_results: (batch, n_games, max_plays)
                - game_lengths: (batch, n_games)
            away_history: Same structure as home_history
            home_roster: (batch, max_roster_size) player token IDs for home team
            away_roster: (batch, max_roster_size) player token IDs for away team
            home_rest_days: (batch,) rest days before target game for home team
            away_rest_days: (batch,) rest days before target game for away team

        Returns:
            GamePrediction with all prediction outputs
        """
        # ===== STEP 1-3: History stream (PBP → game embeddings → temporal attention) =====
        if self._use_history:
            # Each team's last N games of play-by-play data are converted
            # into per-game embeddings, then compressed into a single team history vector.
            home_game_embeddings = self.event_encoder(
                home_history, home_history.get("game_lengths")
            )
            away_game_embeddings = self.event_encoder(
                away_history, away_history.get("game_lengths")
            )

            # Add schedule embeddings (Phase 1c) if enabled
            if self.days_before_embed is not None and "days_before_target" in home_history:
                home_game_embeddings = home_game_embeddings + self.days_before_embed(
                    home_history["days_before_target"]
                ) + self.season_game_embed(home_history["season_game_number"])

                away_game_embeddings = away_game_embeddings + self.days_before_embed(
                    away_history["days_before_target"]
                ) + self.season_game_embed(away_history["season_game_number"])

            home_game_mask = self._create_game_mask(home_history)
            away_game_mask = self._create_game_mask(away_history)

            home_history_repr = self.temporal_attention(
                home_game_embeddings, home_game_mask
            )
            away_history_repr = self.temporal_attention(
                away_game_embeddings, away_game_mask
            )

            home_team_repr = home_history_repr
            away_team_repr = away_history_repr
        else:
            # No history stream — initialize team representations from zero.
            # Roster and/or rest_days will provide all signal.
            batch_size = home_roster.shape[0] if home_roster is not None else home_rest_days.shape[0]
            device = home_roster.device if home_roster is not None else home_rest_days.device
            home_team_repr = torch.zeros(batch_size, self.config.hidden_dim, device=device)
            away_team_repr = torch.zeros(batch_size, self.config.hidden_dim, device=device)

        # ===== STEP 4: Per-team combine =====
        # Combine all per-team signals (history + roster + rest_days) into a
        # single team representation via element-wise addition. All components
        # are already projected to hidden_dim (256-d). LayerNorm stabilizes.

        if self.roster_encoder is not None and home_roster is not None:
            home_team_repr = home_team_repr + self.roster_encoder(home_roster)
            away_team_repr = away_team_repr + self.roster_encoder(away_roster)

        if self.rest_days_embed is not None and home_rest_days is not None:
            home_team_repr = home_team_repr + self.rest_days_embed(home_rest_days)
            away_team_repr = away_team_repr + self.rest_days_embed(away_rest_days)

        if self._has_per_team_combine:
            home_team_repr = self.team_combine_norm(home_team_repr)
            away_team_repr = self.team_combine_norm(away_team_repr)

        # ===== STEP 5: Fuse into matchup representation =====
        # Fusion always receives exactly 2 team representations.
        # SimpleFusion: concat + MLP. CrossAttentionFusion: cross-attend + MLP.
        matchup_repr = self.fusion(home_team_repr, away_team_repr)

        # ===== STEP 6: Generate probabilistic predictions =====
        # The prediction heads convert the matchup representation into
        # spread, score, and win probability distributions.
        predictions = self.prediction_heads(matchup_repr)

        return predictions

    def _create_game_mask(self, history: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Create mask for padded games in history.

        A game is considered "padding" if ALL its plays are zero (no real data).
        This happens when a team has fewer historical games than max_history_games.
        For example, if max_history_games=10 but we only have 7 real games,
        games 8-10 are padding and should be masked out (ignored by attention).
        """
        # Look at action_type_ids -- if every play in a game is 0, that game is padding
        action_ids = history["action_type_ids"]  # (batch, n_games, max_plays)

        # Sum all play token IDs per game. A sum of 0 means no real plays.
        game_has_plays = action_ids.sum(dim=-1) > 0  # (batch, n_games) -- True if real

        # Invert: mask convention is True = IGNORE this position (padding)
        return ~game_has_plays

    def get_num_parameters(self) -> dict[str, int]:
        """Get parameter counts by component."""
        counts = {
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "prediction_heads": sum(p.numel() for p in self.prediction_heads.parameters()),
        }
        if self.event_encoder is not None:
            counts["event_encoder"] = sum(p.numel() for p in self.event_encoder.parameters())
        if self.temporal_attention is not None:
            counts["temporal_attention"] = sum(p.numel() for p in self.temporal_attention.parameters())
        if hasattr(self, "standalone_player_emb"):
            counts["standalone_player_emb"] = sum(
                p.numel() for p in self.standalone_player_emb.parameters()
            )
        if self.days_before_embed is not None:
            counts["schedule_embeddings"] = (
                sum(p.numel() for p in self.days_before_embed.parameters())
                + sum(p.numel() for p in self.season_game_embed.parameters())
            )
        if self.roster_encoder is not None:
            # Exclude shared player_embed (already counted in event_encoder)
            roster_params = sum(
                p.numel() for n, p in self.roster_encoder.named_parameters()
                if not n.startswith("player_embed.")
            )
            counts["roster_encoder"] = roster_params
        if self.rest_days_embed is not None:
            counts["rest_days_embed"] = sum(
                p.numel() for p in self.rest_days_embed.parameters()
            )
        if self._has_per_team_combine:
            counts["team_combine_norm"] = sum(
                p.numel() for p in self.team_combine_norm.parameters()
            )
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


def create_model_from_tokenizer(
    tokenizer,
    hidden_dim: int = 256,
    **config_kwargs,
) -> Phase1Model:
    """
    Convenience factory function that creates a Phase1Model using vocabulary
    sizes extracted from a tokenizer.

    This avoids manually specifying vocab_sizes -- the tokenizer already knows
    how many unique action types, players, etc. exist in the dataset.

    Args:
        tokenizer: PBPTokenizer instance (has a .vocab_sizes dict)
        hidden_dim: Model hidden dimension (default 256)
        **config_kwargs: Any additional ModelConfig parameters to override

    Returns:
        Initialized Phase1Model ready for training or inference
    """
    config = ModelConfig(
        vocab_sizes=tokenizer.vocab_sizes,
        hidden_dim=hidden_dim,
        **config_kwargs,
    )
    return Phase1Model(config)


def test_phase1_model():
    """Test complete Phase 1 model."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Sample vocabulary sizes
    vocab_sizes = {
        "action_type": 40,
        "sub_type": 170,
        "player": 1500,
        "period": 11,
        "clock_bucket": 12,
        "team_indicator": 3,
        "score_diff_bucket": 13,
        "shot_result": 3,
        "shot_distance_bucket": 11,
        "shot_modifier": 30,
    }

    # Create model config
    config = ModelConfig(
        vocab_sizes=vocab_sizes,
        hidden_dim=256,
        event_encoder_layers=4,
        event_encoder_heads=8,
        temporal_attention_layers=2,
        temporal_attention_heads=4,
        dropout=0.1,
    )

    print("Creating Phase 1 Model...")
    model = Phase1Model(config)
    model.eval()

    # Print parameter counts
    param_counts = model.get_num_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Create sample batch
    batch_size = 4
    n_games = 5
    max_plays = 100

    def create_history(batch_size, n_games, max_plays):
        return {
            "action_type_ids": torch.randint(0, 40, (batch_size, n_games, max_plays)),
            "sub_type_ids": torch.randint(0, 170, (batch_size, n_games, max_plays)),
            "periods": torch.randint(1, 5, (batch_size, n_games, max_plays)),
            "clock_buckets": torch.randint(0, 12, (batch_size, n_games, max_plays)),
            "team_indicators": torch.randint(0, 3, (batch_size, n_games, max_plays)),
            "score_diff_buckets": torch.randint(0, 13, (batch_size, n_games, max_plays)),
            "player_ids": torch.randint(0, 1500, (batch_size, n_games, max_plays)),
            "shot_results": torch.randint(0, 3, (batch_size, n_games, max_plays)),
            "shot_distance_buckets": torch.randint(0, 11, (batch_size, n_games, max_plays)),
            "shot_modifier_ids": torch.randint(0, 30, (batch_size, n_games, max_plays)),
            "game_lengths": torch.randint(50, 100, (batch_size, n_games)),
        }

    home_history = create_history(batch_size, n_games, max_plays)
    away_history = create_history(batch_size, n_games, max_plays)

    # Forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        predictions = model(home_history, away_history)

    print(f"  Spread: mean={predictions.spread_mean.mean():.2f}, "
          f"std={predictions.spread_std.mean():.2f}")
    print(f"  Home: mean={predictions.home_score_mean.mean():.2f}, "
          f"std={predictions.home_score_std.mean():.2f}")
    print(f"  Away: mean={predictions.away_score_mean.mean():.2f}, "
          f"std={predictions.away_score_std.mean():.2f}")
    print(f"  Win prob: {predictions.home_win_prob.mean():.3f}")

    # Test loss computation
    print("\nTesting loss computation...")
    from src.transformer.training.loss import CombinedLoss

    target_home = torch.tensor([110.0, 105.0, 115.0, 100.0])
    target_away = torch.tensor([108.0, 112.0, 102.0, 98.0])

    loss_fn = CombinedLoss()
    losses = loss_fn(predictions, target_home, target_away)
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Spread total: {losses['spread_total'].item():.4f}")
    print(f"  Score total: {losses['score_total'].item():.4f}")
    print(f"  Win prob: {losses['win_prob_bce'].item():.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    model.train()
    predictions = model(home_history, away_history)
    losses = loss_fn(predictions, target_home, target_away)
    losses["total"].backward()

    # Check gradients exist
    has_grad = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"  All parameters have gradients: {has_grad}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_phase1_model())
