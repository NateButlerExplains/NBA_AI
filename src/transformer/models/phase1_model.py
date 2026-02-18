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
from src.transformer.models.fusion import SimpleFusion
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

        # ========== Fusion: Combine Team Histories ==========
        # Concatenates home and away history vectors and passes through MLP
        # to learn interactions between the two teams.
        self.fusion = SimpleFusion(
            hidden_dim=config.hidden_dim,
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

    def forward(
        self,
        home_history: dict[str, torch.Tensor],
        away_history: dict[str, torch.Tensor],
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

        Returns:
            GamePrediction with all prediction outputs
        """
        # ===== STEP 1: Encode historical PBP sequences =====
        # Each team's last N games of play-by-play data are converted
        # into per-game embeddings. The EventEncoder processes all games
        # in parallel. Output: (batch, n_games, hidden_dim)
        home_game_embeddings = self.event_encoder(
            home_history, home_history.get("game_lengths")
        )
        away_game_embeddings = self.event_encoder(
            away_history, away_history.get("game_lengths")
        )

        # ===== STEP 2: Create game-level masks =====
        # Detect which "games" in the history are actually just padding
        # (all-zero plays = no real game data). This tells TemporalAttention
        # which game slots to ignore.
        home_game_mask = self._create_game_mask(home_history)
        away_game_mask = self._create_game_mask(away_history)

        # ===== STEP 3: Temporal attention over game sequences =====
        # Compress the sequence of per-game embeddings into a single
        # "team history" vector that captures the team's recent trajectory.
        # Output: (batch, hidden_dim) per team
        home_history_repr = self.temporal_attention(
            home_game_embeddings, home_game_mask
        )
        away_history_repr = self.temporal_attention(
            away_game_embeddings, away_game_mask
        )

        # ===== STEP 4: Fuse into matchup representation =====
        # The output is a single (batch, hidden_dim) vector per matchup that
        # encodes everything the model knows about this particular game.
        matchup_repr = self.fusion(home_history_repr, away_history_repr)

        # ===== STEP 5: Generate probabilistic predictions =====
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
        return {
            "event_encoder": sum(p.numel() for p in self.event_encoder.parameters()),
            "temporal_attention": sum(p.numel() for p in self.temporal_attention.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "prediction_heads": sum(p.numel() for p in self.prediction_heads.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


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
