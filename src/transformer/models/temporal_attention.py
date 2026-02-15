"""
Temporal Attention for NBA Game History.

Attends over historical game embeddings to produce a team history representation.
Uses recency-aware positional encoding (most recent games are position 0).

Usage:
    temporal_attn = TemporalAttention(hidden_dim=256, num_layers=2)
    history_embedding = temporal_attn(game_embeddings)  # Shape: (batch, hidden_dim)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for game sequences.

    Position 0 = most recent game, Position N-1 = oldest game.

    Unlike the sinusoidal encoding in EventEncoder (which uses fixed sine/cosine
    formulas), this encoding LEARNS a different embedding vector for each
    position via training. This lets the model discover its own representation
    for "how recent" each game is.

    Advantage: more flexible than fixed formulas.
    Disadvantage: cannot generalize to positions longer than max_len (unseen positions).
    """

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # An embedding table with one learned vector per position index.
        # Position 0 = most recent game, so the model can learn that
        # recent games get different treatment than older ones.
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to game embeddings.

        Args:
            x: (batch, n_games, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)  # Number of games in the history
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # where 0 = most recent game, 1 = second most recent, etc.
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Look up the learned embedding for each position index
        pos_emb = self.position_embedding(positions)
        # Add position info to the game embeddings (element-wise)
        return self.dropout(x + pos_emb)


class SinusoidalRecencyEncoding(nn.Module):
    """
    Sinusoidal encoding that emphasizes recency.

    Uses higher frequency components for recent positions to provide
    better discrimination between recent games. Same math as the
    PositionalEncoding in event_encoder.py, but applied to the game-level
    sequence rather than the play-level sequence.

    Unlike LearnedPositionalEncoding, this uses fixed sine/cosine formulas
    (no learnable parameters), which means it can generalize to sequence
    lengths not seen during training.
    """

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the fixed sinusoidal encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term creates a spectrum of frequencies.
        # Low-indexed dimensions oscillate quickly (distinguish nearby games),
        # high-indexed dimensions oscillate slowly (capture broad temporal patterns).
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cosine
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for batch broadcasting

        # register_buffer: saved with the model but NOT trained (fixed values)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: (batch, n_games, d_model)

        Returns:
            Tensor with positional encoding added
        """
        # Slice pe to match actual sequence length and add to input
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TemporalAttentionLayer(nn.Module):
    """
    Single layer of temporal attention with pre-norm.

    This is similar to a standard transformer encoder layer, but applied to
    the SEQUENCE OF GAMES (not plays within a game). Each game embedding
    can attend to every other game embedding, learning temporal patterns
    like winning/losing streaks or performance trends.

    Uses "pre-norm" style: normalize BEFORE attention and FFN (not after).
    This is the same approach used in GPT-2 and later models, and tends
    to train more stably than post-norm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Feed-forward dimension defaults to 4x the model dimension
        ff_dim = ff_dim or d_model * 4

        # Self-attention: each game can attend to every other game
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network: applied independently to each game position
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),    # Expand
            nn.GELU(),                      # Non-linearity
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),    # Contract
            nn.Dropout(dropout),
        )

        # Two LayerNorms for pre-norm style (one before attention, one before FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal attention layer.

        Args:
            x: (batch, n_games, d_model)
            mask: (batch, n_games) - True for padding positions

        Returns:
            Output tensor of shape (batch, n_games, d_model)
        """
        # Pre-norm self-attention:
        # 1. Normalize first (pre-norm), then apply attention, then add residual.
        # The residual connection (x + ...) lets gradients bypass the attention
        # layer during backpropagation, making training easier.
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,   # Query=Key=Value (self-attention)
            key_padding_mask=mask,     # Ignore padded games
            need_weights=False,
        )
        x = x + self.dropout(attn_out)  # Residual connection

        # Pre-norm FFN: normalize, apply FFN, add residual.
        # The FFN processes each game position independently (no cross-game interaction),
        # giving the model additional capacity to transform the representations.
        x = x + self.ffn(self.norm2(x))

        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention over game history.

    Takes the per-game embeddings from EventEncoder (one vector per historical
    game) and processes the SEQUENCE of games to produce a single "team history"
    representation that captures the team's recent performance trajectory.

    WHY THIS IS NEEDED:
    The EventEncoder produces one embedding per game, but each game is encoded
    independently. TemporalAttention adds CROSS-GAME reasoning: how a team
    performed across games, whether they are on a winning streak, recovering
    from a loss, etc. Attention lets the model decide which historical games
    are most relevant for predicting the next game.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_games: int = 32,
        positional_encoding: str = "learned",  # "learned" or "sinusoidal"
        pooling: str = "attention",  # "attention", "mean", or "first"
    ):
        """
        Initialize Temporal Attention.

        Args:
            hidden_dim: Dimension of game embeddings
            num_layers: Number of attention layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
            dropout: Dropout probability
            max_games: Maximum number of historical games
            positional_encoding: Type of positional encoding
            pooling: Pooling strategy for final output
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Positional encoding: tells the model which game is most recent,
        # second most recent, etc. Position 0 = most recent game.
        if positional_encoding == "learned":
            # Learned: each position gets a trainable embedding vector
            self.pos_encoder = LearnedPositionalEncoding(hidden_dim, max_games, dropout)
        else:
            # Sinusoidal: fixed sine/cosine patterns (no trainable parameters)
            self.pos_encoder = SinusoidalRecencyEncoding(hidden_dim, max_games, dropout)

        # Stack of temporal attention layers.
        # Each layer lets games attend to each other, building up richer
        # representations that capture cross-game patterns (streaks, trends, etc.)
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization before pooling
        self.norm = nn.LayerNorm(hidden_dim)

        # Attention pooling: compresses all game embeddings into a single
        # "team history" vector. A learnable query attends over all games,
        # learning which games are most important for predicting the next one.
        # For example, the model might learn to weight recent games more heavily,
        # or focus on games against similar opponents.
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pool_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

    def forward(
        self,
        game_embeddings: torch.Tensor,
        game_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode historical game sequence to team history embedding.

        Args:
            game_embeddings: (batch, n_games, hidden_dim) from Event Encoder
            game_mask: (batch, n_games) - True for padding positions

        Returns:
            Team history embedding of shape (batch, hidden_dim)
        """
        # Step 1: Add positional encoding so the model knows which game is
        # most recent (position 0), second most recent (position 1), etc.
        x = self.pos_encoder(game_embeddings)

        # Step 2: Apply temporal attention layers.
        # Each layer lets every game attend to every other game, building up
        # context-aware representations (e.g., a game after a back-to-back
        # might be weighted differently than a game after 3 days rest).
        for layer in self.layers:
            x = layer(x, mask=game_mask)

        # Step 3: Normalize the outputs
        x = self.norm(x)

        # Step 4: Pool all game embeddings into a SINGLE team history vector.
        # Three strategies available:
        if self.pooling == "attention":
            # "attention" pooling: a learned query attends over all games.
            # The model learns WHICH games matter most for prediction.
            # This is the most expressive strategy and the default.
            batch_size = x.shape[0]
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(
                query, x, x,
                key_padding_mask=game_mask,  # Ignore padded games
                need_weights=False,
            )
            return pooled.squeeze(1)  # (batch, 1, dim) -> (batch, dim)
        elif self.pooling == "first":
            # "first" pooling: just use the most recent game embedding.
            # Simple but ignores all historical context beyond the last game.
            return x[:, 0, :]
        else:  # mean pooling
            # "mean" pooling: average all game embeddings equally.
            # Ignores recency -- every game contributes the same amount.
            if game_mask is not None:
                # Zero out padding before averaging, divide by real game count
                x = x.masked_fill(game_mask.unsqueeze(-1), 0.0)
                valid_counts = (~game_mask).sum(dim=1, keepdim=True).float()
                return x.sum(dim=1) / valid_counts.clamp(min=1)
            else:
                return x.mean(dim=1)

    def forward_with_weights(
        self,
        game_embeddings: torch.Tensor,
        game_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for INTERPRETABILITY.

        This is useful for understanding the model's behavior: the attention
        weights show which historical games the model is focusing on when
        making its prediction. For example, you might see the model paying
        extra attention to a recent blowout loss, or to a previous matchup
        against the same opponent.

        Returns:
            (history_embedding, attention_weights)
            - history_embedding: (batch, hidden_dim) team history vector
            - attention_weights: (batch, n_games) how much each game contributed
        """
        # Same as forward() but we capture the attention weights at the pooling step
        x = self.pos_encoder(game_embeddings)

        # Collect attention weights from each layer (placeholder for future use)
        all_weights = []
        for layer in self.layers:
            x = layer(x, mask=game_mask)

        x = self.norm(x)

        # Pool and capture attention weights
        if self.pooling == "attention":
            batch_size = x.shape[0]
            query = self.pool_query.expand(batch_size, -1, -1)
            # need_weights=True returns the attention weight matrix
            pooled, weights = self.pool_attention(
                query, x, x,
                key_padding_mask=game_mask,
                need_weights=True,  # This time we DO want the weights
            )
            # weights shape: (batch, 1, n_games) -> squeeze to (batch, n_games)
            # Each value shows how much the model "paid attention" to that game
            return pooled.squeeze(1), weights.squeeze(1)
        else:
            # For non-attention pooling, return uniform weights since there's
            # no attention mechanism to extract weights from.
            n_games = game_embeddings.shape[1]
            weights = torch.ones(game_embeddings.shape[0], n_games, device=x.device)
            if game_mask is not None:
                weights = weights.masked_fill(game_mask, 0.0)
            # Normalize weights to sum to 1 (like a probability distribution)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

            if self.pooling == "first":
                return x[:, 0, :], weights
            else:
                return x.mean(dim=1) if game_mask is None else (
                    x.masked_fill(game_mask.unsqueeze(-1), 0.0).sum(dim=1) /
                    (~game_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
                ), weights


def test_temporal_attention():
    """Test Temporal Attention with sample data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    hidden_dim = 256

    # Create Temporal Attention
    model = TemporalAttention(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        positional_encoding="learned",
        pooling="attention",
    )
    model.eval()

    print(f"Temporal Attention created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create sample game embeddings (as if from Event Encoder)
    batch_size = 4
    n_games = 5

    game_embeddings = torch.randn(batch_size, n_games, hidden_dim)

    print(f"\nInput game_embeddings shape: {game_embeddings.shape}")

    # Forward pass without mask
    print("\nTesting forward pass (no mask)...")
    with torch.no_grad():
        output = model(game_embeddings)

    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {hidden_dim})")
    assert output.shape == (batch_size, hidden_dim), "Output shape mismatch!"

    # Test with variable-length histories (some teams have fewer games)
    print("\nTesting with variable-length histories...")
    game_mask = torch.zeros(batch_size, n_games, dtype=torch.bool)
    game_mask[0, 4] = True  # First sample has only 4 games
    game_mask[1, 3:] = True  # Second sample has only 3 games

    with torch.no_grad():
        output_masked = model(game_embeddings, game_mask)

    print(f"  Output shape: {output_masked.shape}")

    # Test attention weights
    print("\nTesting attention weight extraction...")
    with torch.no_grad():
        output_with_weights, weights = model.forward_with_weights(
            game_embeddings, game_mask
        )

    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Sample weights (sample 0): {weights[0].tolist()}")

    # Test different pooling strategies
    print("\nTesting pooling strategies...")
    for pooling in ["attention", "mean", "first"]:
        m = TemporalAttention(hidden_dim=hidden_dim, num_layers=1, pooling=pooling)
        m.eval()
        with torch.no_grad():
            out = m(game_embeddings, game_mask)
        print(f"  {pooling} pooling: {out.shape}")

    # Test different positional encodings
    print("\nTesting positional encodings...")
    for pe_type in ["learned", "sinusoidal"]:
        m = TemporalAttention(hidden_dim=hidden_dim, num_layers=1, positional_encoding=pe_type)
        m.eval()
        with torch.no_grad():
            out = m(game_embeddings)
        print(f"  {pe_type}: {out.shape}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_temporal_attention())
