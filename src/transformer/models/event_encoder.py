"""
Event Encoder for NBA Transformer Model.

Transforms tokenized PBP sequences into game-level embeddings using
a Transformer encoder architecture.

Architecture:
    1. Embedding layers for each token component
    2. Linear projection to hidden dimension
    3. Transformer encoder with positional encoding
    4. Pooling to single game embedding

Usage:
    encoder = EventEncoder(vocab_sizes, hidden_dim=256, num_layers=4)
    game_embedding = encoder(batch)  # Shape: (batch, n_games, hidden_dim)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.

    WHY THIS EXISTS:
    Transformers process all tokens in parallel (unlike RNNs which go one-by-one).
    This means a transformer has NO built-in notion of order -- it doesn't know
    if a play happened at the start or end of a game. Positional encoding solves
    this by adding a unique "fingerprint" to each position using sine and cosine
    waves of different frequencies.

    The key insight: each position gets a unique pattern of sine/cosine values,
    so the model can learn to distinguish "play #1" from "play #500."
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        # Dropout randomly zeros some values during training to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the positional encoding matrix once (never changes during training).
        # Shape will be (max_len, d_model) -- one row per possible position, one col per dimension.
        pe = torch.zeros(max_len, d_model)

        # position = [[0], [1], [2], ...] -- column vector of position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term controls the wavelength of each sine/cosine pair.
        # Lower dimensions get shorter wavelengths (higher frequency),
        # higher dimensions get longer wavelengths (lower frequency).
        # This creates a spectrum from fine-grained to coarse-grained position info.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even-indexed dimensions use sine, odd-indexed use cosine.
        # Together, they create a unique "fingerprint" for each position.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model) so it broadcasts across the batch
        pe = pe.unsqueeze(0)

        # register_buffer: stores this tensor as part of the module's state
        # (saved/loaded with the model) but it is NOT a learnable parameter --
        # the positional encoding is fixed, not trained.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        # Add the positional encoding to the input embeddings (element-wise addition).
        # We slice self.pe to match the actual sequence length (x.size(1)).
        # The model now knows BOTH what a token is (from the embedding) and
        # WHERE it is in the sequence (from the positional encoding).
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EventEmbedding(nn.Module):
    """
    Embedding layer for PBP (play-by-play) events.

    HOW THIS WORKS:
    Each play in a basketball game is described by multiple fields (action type,
    player, period, etc.). Each field is a discrete integer ID (like a category).
    Neural networks can't work with raw integers, so we use nn.Embedding to
    convert each integer ID into a learned vector of continuous numbers.

    Think of nn.Embedding as a lookup table:
      - Input: integer ID (e.g., player_id=42)
      - Output: a learned vector (e.g., [0.3, -0.1, 0.7, ...])
      - The vector values are learned during training to capture meaningful
        properties (similar players get similar vectors).

    Each field gets its OWN embedding table with a dimension chosen to reflect
    the field's complexity:
      - action_type (32-d) -- "made shot", "turnover", "foul", etc.
      - sub_type (32-d) -- "3-pointer", "layup", "offensive foul", etc.
      - period (16-d) -- quarter 1-4 or overtime periods
      - clock_bucket (16-d) -- discretized game clock
      - team_indicator (8-d) -- home vs away
      - score_diff_bucket (16-d) -- binned score difference
      - player_id (64-d) -- who performed the action (larger because ~1500 players)
      - shot_result (8-d) -- made, missed, or N/A

    These are concatenated (joined end-to-end) into a 192-d vector, then
    projected (linear transform) down to hidden_dim (256) for the transformer.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        hidden_dim: int = 256,
        action_dim: int = 32,
        subtype_dim: int = 32,
        period_dim: int = 16,
        clock_dim: int = 16,
        team_dim: int = 8,
        score_diff_dim: int = 16,
        player_dim: int = 64,
        shot_result_dim: int = 8,
        shot_distance_dim: int = 16,
        shot_modifier_dim: int = 16,
    ):
        super().__init__()

        self.vocab_sizes = vocab_sizes

        # --- Component Embedding Tables ---
        # nn.Embedding(num_entries, vector_size, padding_idx=0):
        #   - num_entries: how many unique IDs exist (vocabulary size)
        #   - vector_size: how many dimensions each embedding vector has
        #   - padding_idx=0: the embedding for index 0 is forced to stay all-zeros
        #     during training. Index 0 represents "padding" (no real data here),
        #     so we want it to contribute nothing to the model's computation.

        self.action_type_emb = nn.Embedding(
            vocab_sizes["action_type"], action_dim, padding_idx=0
        )
        self.sub_type_emb = nn.Embedding(
            vocab_sizes["sub_type"], subtype_dim, padding_idx=0
        )
        self.period_emb = nn.Embedding(vocab_sizes["period"], period_dim, padding_idx=0)
        self.clock_emb = nn.Embedding(
            vocab_sizes["clock_bucket"], clock_dim, padding_idx=0
        )
        # team_indicator and score_diff_bucket don't use padding_idx because
        # index 0 is a valid category for these fields (not padding).
        self.team_emb = nn.Embedding(vocab_sizes["team_indicator"], team_dim)
        self.score_diff_emb = nn.Embedding(
            vocab_sizes["score_diff_bucket"], score_diff_dim
        )
        self.player_emb = nn.Embedding(
            vocab_sizes["player"], player_dim, padding_idx=0
        )
        self.shot_result_emb = nn.Embedding(vocab_sizes["shot_result"], shot_result_dim)

        # --- New Shot Feature Embeddings ---
        # shot_distance_bucket: 0=not a shot, 1-10=distance bins (padding_idx=0)
        self.shot_distance_emb = nn.Embedding(
            vocab_sizes["shot_distance_bucket"], shot_distance_dim, padding_idx=0
        )
        # shot_modifier: descriptor vocab (pullup, driving, step back, etc.)
        self.shot_modifier_emb = nn.Embedding(
            vocab_sizes["shot_modifier"], shot_modifier_dim, padding_idx=0
        )

        # Total embedding dimension = sum of all component dimensions
        # 32 + 32 + 16 + 16 + 8 + 16 + 64 + 8 + 16 + 16 = 224
        self.embed_dim = (
            action_dim
            + subtype_dim
            + period_dim
            + clock_dim
            + team_dim
            + score_diff_dim
            + player_dim
            + shot_result_dim
            + shot_distance_dim
            + shot_modifier_dim
        )

        # Linear projection: a single matrix multiply that transforms the
        # 224-d concatenated embedding down to hidden_dim (256).
        # This lets the transformer work with a consistent dimension size.
        self.projection = nn.Linear(self.embed_dim, hidden_dim)

    def forward(
        self,
        action_type_ids: torch.Tensor,
        sub_type_ids: torch.Tensor,
        periods: torch.Tensor,
        clock_buckets: torch.Tensor,
        team_indicators: torch.Tensor,
        score_diff_buckets: torch.Tensor,
        player_ids: torch.Tensor,
        shot_results: torch.Tensor,
        shot_distance_buckets: torch.Tensor,
        shot_modifier_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed all token components and combine.

        Args:
            action_type_ids: (batch, seq_len) or (batch, n_games, seq_len)
            sub_type_ids: (batch, seq_len) or (batch, n_games, seq_len)
            ... (all same shape)

        Returns:
            Combined embeddings of shape (..., hidden_dim)
        """
        # Look up each integer ID in its embedding table.
        # Each call converts shape (..., seq_len) of integer IDs
        # into (..., seq_len, embed_dim) of continuous vectors.
        action_emb = self.action_type_emb(action_type_ids)    # (..., seq_len, 32)
        subtype_emb = self.sub_type_emb(sub_type_ids)          # (..., seq_len, 32)
        period_emb = self.period_emb(periods)                  # (..., seq_len, 16)
        clock_emb = self.clock_emb(clock_buckets)              # (..., seq_len, 16)
        team_emb = self.team_emb(team_indicators)              # (..., seq_len, 8)
        score_diff_emb = self.score_diff_emb(score_diff_buckets)  # (..., seq_len, 16)
        player_emb = self.player_emb(player_ids)               # (..., seq_len, 64)
        shot_emb = self.shot_result_emb(shot_results)          # (..., seq_len, 8)
        shot_dist_emb = self.shot_distance_emb(shot_distance_buckets)  # (..., seq_len, 16)
        shot_mod_emb = self.shot_modifier_emb(shot_modifier_ids)       # (..., seq_len, 16)

        # Concatenate all embeddings along the last dimension (dim=-1).
        # This joins them end-to-end: [action|subtype|period|...|shot_dist|shot_mod]
        # Result shape: (..., seq_len, 224)  -- one 224-d vector per play
        combined = torch.cat(
            [
                action_emb,
                subtype_emb,
                period_emb,
                clock_emb,
                team_emb,
                score_diff_emb,
                player_emb,
                shot_emb,
                shot_dist_emb,
                shot_mod_emb,
            ],
            dim=-1,
        )

        # Project 224-d concatenated vector down to hidden_dim (256).
        # This learned linear transform lets the model combine information
        # from all fields into a unified representation for each play.
        return self.projection(combined)


class EventEncoder(nn.Module):
    """
    Transformer encoder for PBP event sequences.

    Takes tokenized PBP sequences and produces game-level embeddings.
    Processes each historical game independently, then returns embeddings
    for all games in the sequence.

    HIGH-LEVEL FLOW:
    1. Each play's fields (action, player, period, etc.) are embedded and concatenated
    2. Positional encoding is added so the model knows play ordering
    3. A Transformer encoder processes the sequence -- this is where self-attention
       lets each play "look at" every other play to learn relationships
       (e.g., a foul call right after a contested shot)
    4. Pooling collapses the variable-length sequence of play embeddings into
       a single fixed-size vector representing the entire game
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        pooling: str = "mean",  # "mean", "cls", or "max"
    ):
        """
        Initialize Event Encoder.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from tokenizer
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
            pooling: Pooling strategy for game embedding ("mean", "cls", "max")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Event embedding layer -- converts raw token IDs into continuous vectors
        self.embedding = EventEmbedding(vocab_sizes, hidden_dim)

        # Positional encoding -- adds sequence position information
        # (play 1 vs play 500) so the transformer knows the ORDER of plays
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # [CLS] token for "cls" pooling strategy.
        # CLS (classification) token is a special learned vector prepended to the
        # sequence. After transformer processing, the CLS output position acts as
        # a summary of the whole sequence. This approach comes from BERT.
        # nn.Parameter makes it a trainable weight (updated via gradient descent).
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # ---- Transformer Encoder: the core of the architecture ----
        # ff_dim (feed-forward dimension) defaults to 4x hidden_dim.
        # Each transformer layer has two sub-layers:
        #   1. Self-attention: every play can "look at" every other play to learn
        #      relationships (e.g., a steal followed by a fast-break dunk)
        #   2. Feed-forward network (FFN): a small MLP that processes each play
        #      independently after the attention step
        ff_dim = ff_dim or 4 * hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,       # Input/output dimension of the layer
            nhead=num_heads,          # Number of parallel attention heads (8)
                                      # Each head attends to different aspects of the data
            dim_feedforward=ff_dim,   # Width of the FFN inside each layer (1024)
            dropout=dropout,          # Randomly drops connections during training
            activation="gelu",        # Activation function (smoother version of ReLU)
            batch_first=True,         # Input shape is (batch, seq, features) not (seq, batch, features)
            norm_first=True,          # Pre-norm: normalize BEFORE attention/FFN (not after).
                                      # This improves training stability, especially for
                                      # deeper networks, by keeping gradient magnitudes in check.
        )
        # Stack multiple encoder layers. Each layer refines the representations
        # by applying attention and feed-forward transformations again.
        # More layers = more capacity to learn complex patterns (but slower).
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final LayerNorm stabilizes the output magnitudes before pooling.
        # LayerNorm normalizes each sample independently (mean=0, std=1 across features).
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        history: dict[str, torch.Tensor],
        game_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode historical game sequences.

        Args:
            history: Dictionary containing batched tensors:
                - action_type_ids: (batch, n_games, max_plays)
                - sub_type_ids: (batch, n_games, max_plays)
                - periods: (batch, n_games, max_plays)
                - clock_buckets: (batch, n_games, max_plays)
                - team_indicators: (batch, n_games, max_plays)
                - score_diff_buckets: (batch, n_games, max_plays)
                - player_ids: (batch, n_games, max_plays)
                - shot_results: (batch, n_games, max_plays)
                - game_lengths: (batch, n_games) - actual play counts per game
            game_lengths: If not in history dict, provide separately

        Returns:
            Game embeddings of shape (batch, n_games, hidden_dim)
        """
        # --- Step 0: Extract tensor dimensions ---
        # Input shape: (batch_size, n_games, max_plays)
        # batch_size = number of matchups in this training batch
        # n_games = number of historical games per team (e.g., last 10 games)
        # max_plays = maximum number of plays per game (padded to same length)
        batch_size = history["action_type_ids"].shape[0]
        n_games = history["action_type_ids"].shape[1]
        max_plays = history["action_type_ids"].shape[2]

        # game_lengths tells us how many REAL plays each game has (the rest is padding).
        # Example: game_lengths[0][2] = 87 means batch item 0, game 2 has 87 real plays.
        if game_lengths is None:
            game_lengths = history.get("game_lengths")

        # --- Step 1: Reshape to process all games at once ---
        # We flatten batch and games dimensions together:
        # (batch, n_games, max_plays) -> (batch * n_games, max_plays)
        # This lets us run the transformer on ALL games in one pass (more efficient
        # than looping over games one-by-one). We'll reshape back later.
        def reshape_for_encoding(t):
            return t.view(batch_size * n_games, max_plays)

        action_type_ids = reshape_for_encoding(history["action_type_ids"])
        sub_type_ids = reshape_for_encoding(history["sub_type_ids"])
        periods = reshape_for_encoding(history["periods"])
        clock_buckets = reshape_for_encoding(history["clock_buckets"])
        team_indicators = reshape_for_encoding(history["team_indicators"])
        score_diff_buckets = reshape_for_encoding(history["score_diff_buckets"])
        player_ids = reshape_for_encoding(history["player_ids"])
        shot_results = reshape_for_encoding(history["shot_results"])
        shot_distance_buckets = reshape_for_encoding(history["shot_distance_buckets"])
        shot_modifier_ids = reshape_for_encoding(history["shot_modifier_ids"])

        # --- Step 2: Embed all token fields into continuous vectors ---
        # Each play's 10 integer fields are embedded and concatenated into a
        # 224-d vector, then projected to hidden_dim (256).
        # Output shape: (batch * n_games, max_plays, hidden_dim)
        x = self.embedding(
            action_type_ids,
            sub_type_ids,
            periods,
            clock_buckets,
            team_indicators,
            score_diff_buckets,
            player_ids,
            shot_results,
            shot_distance_buckets,
            shot_modifier_ids,
        )

        # --- Step 3: Optionally prepend [CLS] token ---
        # If using "cls" pooling, we prepend a special learned token to the sequence.
        # After the transformer processes it, this token's output serves as a
        # summary of the entire game. We expand it to match the batch dimension.
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size * n_games, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # Prepend CLS to each sequence
            max_plays_with_cls = max_plays + 1      # Sequence is now 1 longer
        else:
            max_plays_with_cls = max_plays

        # --- Step 4: Add positional encoding ---
        # Injects information about WHERE each play is in the sequence.
        # Without this, the transformer treats the sequence as an unordered bag.
        x = self.pos_encoder(x)

        # --- Step 5: Create attention mask for padding positions ---
        # Games have different numbers of plays, so shorter games are padded with
        # zeros to reach max_plays. We need to tell the transformer to IGNORE
        # these padding positions during attention (otherwise, it would treat
        # padding as real data and produce wrong results).
        # The mask is True where positions are padding (should be ignored).
        if game_lengths is not None:
            game_lengths_flat = game_lengths.view(-1)  # (batch * n_games,)

            # Build a boolean mask: position indices >= game_length means padding.
            # Example: if game_length=87 and max_plays=100, positions 87-99 are masked.
            mask = torch.arange(max_plays_with_cls, device=x.device).unsqueeze(0)
            if self.pooling == "cls":
                # Account for the extra CLS token at position 0
                mask = mask >= (game_lengths_flat.unsqueeze(1) + 1)
            else:
                mask = mask >= game_lengths_flat.unsqueeze(1)
        else:
            mask = None

        # --- Step 6: Run the Transformer Encoder ---
        # This is where the magic happens: self-attention lets EVERY play look at
        # EVERY other play. The model learns which plays are related to each other
        # (e.g., a missed shot followed by an offensive rebound, or a timeout
        # called during a scoring run). The padding mask ensures the transformer
        # ignores the fake padding positions.
        x = self.transformer(x, src_key_padding_mask=mask)

        # --- Step 7: Pool the sequence to a single game embedding ---
        # The transformer outputs one vector per play position. We need to
        # collapse this variable-length sequence into a SINGLE fixed-size vector
        # that represents the entire game. Three strategies:
        #
        # "cls": Take only the CLS token's output (position 0). The CLS token
        #   has been attending to all other plays, so its output summarizes them.
        # "max": Take the element-wise maximum across all play positions.
        #   Each dimension keeps the highest activation -- captures the most
        #   salient features across the game.
        # "mean": Average all play embeddings. This gives equal weight to every
        #   play and is the default strategy.
        if self.pooling == "cls":
            # CLS token is at position 0 after prepending
            game_emb = x[:, 0, :]
        elif self.pooling == "max":
            # Set padding positions to -inf so they never win the max
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            game_emb = x.max(dim=1)[0]  # [0] gets values, [1] would get indices
        else:  # mean pooling
            # Average over only the REAL (non-padding) plays.
            # We zero out padding, sum everything, then divide by the actual count.
            if game_lengths is not None:
                game_lengths_flat = game_lengths.view(-1).unsqueeze(1).float()
                mask_expanded = mask.unsqueeze(-1) if mask is not None else None
                if mask_expanded is not None:
                    x = x.masked_fill(mask_expanded, 0.0)  # Zero out padding
                # .clamp(min=1) avoids division by zero for empty games
                game_emb = x.sum(dim=1) / game_lengths_flat.clamp(min=1)
            else:
                game_emb = x.mean(dim=1)

        # Normalize the game embedding to stabilize magnitudes
        game_emb = self.output_norm(game_emb)

        # --- Step 8: Reshape back to separate batch and game dimensions ---
        # (batch * n_games, hidden_dim) -> (batch, n_games, hidden_dim)
        # Now we have one 256-d embedding per game, per sample in the batch.
        game_emb = game_emb.view(batch_size, n_games, self.hidden_dim)

        return game_emb

    def encode_single_game(
        self,
        action_type_ids: torch.Tensor,
        sub_type_ids: torch.Tensor,
        periods: torch.Tensor,
        clock_buckets: torch.Tensor,
        team_indicators: torch.Tensor,
        score_diff_buckets: torch.Tensor,
        player_ids: torch.Tensor,
        shot_results: torch.Tensor,
        shot_distance_buckets: torch.Tensor,
        shot_modifier_ids: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode a single game's PBP sequence (convenience method).

        Args:
            All tensors have shape (batch, seq_len)
            seq_len: Actual sequence length (if padding present)

        Returns:
            Game embedding of shape (batch, hidden_dim)
        """
        # unsqueeze(1) adds a dummy n_games=1 dimension so we can reuse
        # the main forward() method, which expects (batch, n_games, max_plays)
        history = {
            "action_type_ids": action_type_ids.unsqueeze(1),
            "sub_type_ids": sub_type_ids.unsqueeze(1),
            "periods": periods.unsqueeze(1),
            "clock_buckets": clock_buckets.unsqueeze(1),
            "team_indicators": team_indicators.unsqueeze(1),
            "score_diff_buckets": score_diff_buckets.unsqueeze(1),
            "player_ids": player_ids.unsqueeze(1),
            "shot_results": shot_results.unsqueeze(1),
            "shot_distance_buckets": shot_distance_buckets.unsqueeze(1),
            "shot_modifier_ids": shot_modifier_ids.unsqueeze(1),
        }

        if seq_len is not None:
            batch_size = action_type_ids.shape[0]
            game_lengths = torch.full(
                (batch_size, 1), seq_len, device=action_type_ids.device
            )
        else:
            game_lengths = None

        # Forward pass returns (batch, 1, hidden_dim); squeeze(1) removes the
        # game dimension to give (batch, hidden_dim) for a single game.
        return self.forward(history, game_lengths).squeeze(1)


def test_event_encoder():
    """Test Event Encoder with sample data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Sample vocab sizes (from typical tokenizer)
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

    # Create encoder
    encoder = EventEncoder(
        vocab_sizes=vocab_sizes,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        pooling="mean",
    )

    print(f"Event Encoder created")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Create sample batch: (batch=2, n_games=5, max_plays=100)
    batch_size = 2
    n_games = 5
    max_plays = 100

    history = {
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

    # Forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = encoder(history)

    print(f"  Input shape: ({batch_size}, {n_games}, {max_plays})")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {n_games}, 256)")

    assert output.shape == (batch_size, n_games, 256), "Output shape mismatch!"

    # Test single game encoding
    print("\nTesting single game encoding...")
    single_game_tensors = {
        k: v[:, 0, :] for k, v in history.items() if k != "game_lengths"
    }
    with torch.no_grad():
        single_output = encoder.encode_single_game(
            **single_game_tensors, seq_len=history["game_lengths"][:, 0].max().item()
        )
    print(f"  Single game output shape: {single_output.shape}")
    assert single_output.shape == (batch_size, 256), "Single game output shape mismatch!"

    # Test different pooling strategies
    print("\nTesting pooling strategies...")
    for pooling in ["mean", "cls", "max"]:
        enc = EventEncoder(vocab_sizes=vocab_sizes, hidden_dim=128, num_layers=2, pooling=pooling)
        with torch.no_grad():
            out = enc(history)
        print(f"  {pooling} pooling: {out.shape}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_event_encoder())
