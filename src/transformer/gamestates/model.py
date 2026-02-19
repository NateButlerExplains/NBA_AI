"""
GameStates Model for NBA Game Prediction.

Uses score trajectory embeddings instead of PBP event embeddings.
Reuses TemporalAttention, SimpleFusion, and PredictionHeads from the PBP model.

Architecture:
    GameStateEmbedding (5 embeds, 80-d → 256-d)
    → TransformerEncoder (4 layers, 8 heads)
    → Mean pooling → per-game embedding (256-d)
    → TemporalAttention (2 layers, 4 heads)
    → SimpleFusion
    → PredictionHeads

Usage:
    model = GameStateModel(config)
    predictions = model(home_history, away_history)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from src.transformer.training.config import ModelConfig
from src.transformer.models.temporal_attention import TemporalAttention
from src.transformer.models.fusion import SimpleFusion
from src.transformer.models.prediction_heads import PredictionHeads, GamePrediction


class GameStateEmbedding(nn.Module):
    """
    Embedding layer for GameStates rows.

    5 embedding tables → concatenate to 80-d → project to hidden_dim.
    """

    def __init__(self, hidden_dim: int = 256, embed_dim: int = 16):
        super().__init__()

        self.period_emb = nn.Embedding(5, embed_dim, padding_idx=0)
        self.clock_emb = nn.Embedding(721, embed_dim, padding_idx=0)
        self.home_score_emb = nn.Embedding(51, embed_dim)
        self.away_score_emb = nn.Embedding(51, embed_dim)
        self.margin_emb = nn.Embedding(121, embed_dim)

        self.embed_dim = embed_dim * 5  # 80
        self.projection = nn.Linear(self.embed_dim, hidden_dim)

    def forward(
        self,
        periods: torch.Tensor,
        clock_buckets: torch.Tensor,
        home_score_buckets: torch.Tensor,
        away_score_buckets: torch.Tensor,
        margin_buckets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed all GameState fields and combine.

        Args:
            All inputs: (..., seq_len) integer tensors

        Returns:
            Combined embeddings of shape (..., hidden_dim)
        """
        p = self.period_emb(periods)
        c = self.clock_emb(clock_buckets)
        hs = self.home_score_emb(home_score_buckets)
        as_ = self.away_score_emb(away_score_buckets)
        m = self.margin_emb(margin_buckets)

        combined = torch.cat([p, c, hs, as_, m], dim=-1)
        return self.projection(combined)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for GameStates sequences."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GameStateEncoder(nn.Module):
    """
    Transformer encoder for GameStates sequences.

    Processes each game's score trajectory into a single game embedding.
    Same architecture depth as EventEncoder (4 layers, 8 heads).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = GameStateEmbedding(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        ff_dim = ff_dim or 4 * hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        history: dict[str, torch.Tensor],
        game_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode historical GameStates sequences.

        Args:
            history: Dict with keys: periods, clock_buckets, home_score_buckets,
                     away_score_buckets, margin_buckets — all shape (batch, n_games, max_rows)
                     game_lengths: (batch, n_games)
            game_lengths: If not in history dict, provide separately

        Returns:
            Game embeddings of shape (batch, n_games, hidden_dim)
        """
        batch_size = history["periods"].shape[0]
        n_games = history["periods"].shape[1]
        max_rows = history["periods"].shape[2]

        if game_lengths is None:
            game_lengths = history.get("game_lengths")

        # Flatten batch and games: (batch * n_games, max_rows)
        def reshape(t):
            return t.view(batch_size * n_games, max_rows)

        periods = reshape(history["periods"])
        clock_buckets = reshape(history["clock_buckets"])
        home_score_buckets = reshape(history["home_score_buckets"])
        away_score_buckets = reshape(history["away_score_buckets"])
        margin_buckets = reshape(history["margin_buckets"])

        # Embed: (batch * n_games, max_rows, hidden_dim)
        x = self.embedding(periods, clock_buckets, home_score_buckets,
                           away_score_buckets, margin_buckets)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask
        if game_lengths is not None:
            game_lengths_flat = game_lengths.view(-1)
            mask = torch.arange(max_rows, device=x.device).unsqueeze(0)
            mask = mask >= game_lengths_flat.unsqueeze(1)
        else:
            mask = None

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean pooling over non-padding positions
        if game_lengths is not None:
            game_lengths_flat = game_lengths.view(-1).unsqueeze(1).float()
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            game_emb = x.sum(dim=1) / game_lengths_flat.clamp(min=1)
        else:
            game_emb = x.mean(dim=1)

        game_emb = self.output_norm(game_emb)

        # Reshape back: (batch, n_games, hidden_dim)
        return game_emb.view(batch_size, n_games, self.hidden_dim)


class GameStateModel(nn.Module):
    """
    Complete GameStates model for NBA game prediction.

    Replaces EventEncoder with GameStateEncoder, reuses everything else
    from the PBP model (TemporalAttention, SimpleFusion, PredictionHeads).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.game_state_encoder = GameStateEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.event_encoder_layers,
            num_heads=config.event_encoder_heads,
            dropout=config.dropout,
            max_seq_len=config.max_pbp_length + 1,
        )

        self.temporal_attention = TemporalAttention(
            hidden_dim=config.hidden_dim,
            num_layers=config.temporal_attention_layers,
            num_heads=config.temporal_attention_heads,
            dropout=config.dropout,
            max_games=config.max_history_games,
            positional_encoding="learned",
            pooling="attention",
        )

        self.fusion = SimpleFusion(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.prediction_heads = PredictionHeads(
            input_dim=config.hidden_dim,
            hidden_dim=128,
            spread_min_std=1.0,
            score_min_std=5.0,
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
            home_history: Dict of tensors for home team's historical GameStates
                - periods: (batch, n_games, max_rows)
                - clock_buckets: (batch, n_games, max_rows)
                - home_score_buckets: (batch, n_games, max_rows)
                - away_score_buckets: (batch, n_games, max_rows)
                - margin_buckets: (batch, n_games, max_rows)
                - game_lengths: (batch, n_games)
            away_history: Same structure

        Returns:
            GamePrediction with all prediction outputs
        """
        home_game_embeddings = self.game_state_encoder(
            home_history, home_history.get("game_lengths")
        )
        away_game_embeddings = self.game_state_encoder(
            away_history, away_history.get("game_lengths")
        )

        home_game_mask = self._create_game_mask(home_history)
        away_game_mask = self._create_game_mask(away_history)

        home_history_repr = self.temporal_attention(
            home_game_embeddings, home_game_mask
        )
        away_history_repr = self.temporal_attention(
            away_game_embeddings, away_game_mask
        )

        matchup_repr = self.fusion(home_history_repr, away_history_repr)
        return self.prediction_heads(matchup_repr)

    def _create_game_mask(self, history: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Create mask for padded games. A game is padding if all its rows are zero."""
        periods = history["periods"]  # (batch, n_games, max_rows)
        game_has_data = periods.sum(dim=-1) > 0  # (batch, n_games)
        return ~game_has_data

    def get_num_parameters(self) -> dict[str, int]:
        return {
            "game_state_encoder": sum(p.numel() for p in self.game_state_encoder.parameters()),
            "temporal_attention": sum(p.numel() for p in self.temporal_attention.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "prediction_heads": sum(p.numel() for p in self.prediction_heads.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
