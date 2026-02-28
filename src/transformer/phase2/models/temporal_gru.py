"""
Phase 2/3 Temporal GRU.

Drop-in replacement for Phase2TemporalAttention. Uses a bidirectional GRU
with time-gap embedding and multi-query attention pooling.

GRU gates provide exponential decay natively — exactly the inductive bias
needed for form/streaks that the transformer must learn from positional encoding.
Time embedding lets the GRU distinguish "1 day between games" from "14 days"
(All-Star break, rest periods), modulating its gating accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.transformer.phase2.models.temporal_attention import MultiQueryAttentionPool


class Phase2TemporalGRU(nn.Module):
    """
    Time-aware bidirectional GRU with multi-query attention pooling.

    Replaces Phase2TemporalAttention with same interface:
        forward(game_embeddings, days_before, game_mask) -> (B, hidden_dim)

    Architecture:
    1. Embed days_before (calendar distance) -> (B, G, time_dim)
    2. Concat [game_embeddings, time_embed] -> (B, G, hidden_dim + time_dim)
    3. LayerNorm on input
    4. Bidirectional GRU -> (B, G, hidden_dim)
    5. MultiQueryAttentionPool -> (B, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        gru_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_days: int = 180,
        time_dim: int = 64,
        n_pool_queries: int = 4,
        pool_heads: int = 8,
    ):
        super().__init__()

        assert gru_hidden * 2 == hidden_dim, (
            f"Bidirectional GRU output (2 × {gru_hidden} = {gru_hidden * 2}) "
            f"must match hidden_dim ({hidden_dim})"
        )

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Time embedding: calendar distance -> dense vector
        self.time_embed = nn.Embedding(max_days, time_dim)

        # Input normalization (stabilizes GRU input scale)
        self.input_norm = nn.LayerNorm(hidden_dim + time_dim)

        # Bidirectional GRU
        # Input: hidden_dim + time_dim (game features + temporal context)
        # Output: gru_hidden * 2 = hidden_dim (forward + backward)
        self.gru = nn.GRU(
            input_size=hidden_dim + time_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output normalization (matches Phase 2 transformer's final LayerNorm)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Reuse Phase 2's multi-query attention pool
        self.pool = MultiQueryAttentionPool(
            hidden_dim, n_pool_queries, pool_heads, dropout
        )

    def forward(
        self,
        game_embeddings: torch.Tensor,
        days_before: torch.Tensor,
        game_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            game_embeddings: (B, n_games, hidden_dim) float32
            days_before: (B, n_games) int64, values [0, 179]
            game_mask: (B, n_games) bool, True=padding

        Returns:
            (B, hidden_dim)
        """
        B, G, _ = game_embeddings.shape

        # 1. Time embedding
        time_emb = self.time_embed(days_before.clamp(0, 179))  # (B, G, time_dim)

        # 2. Concatenate game features with time context
        x = torch.cat([game_embeddings, time_emb], dim=-1)  # (B, G, hidden_dim + time_dim)

        # 3. Input normalization
        x = self.input_norm(x)

        # 4. Pack sequences for variable-length GRU processing
        if game_mask is not None:
            # Compute actual lengths: count of non-padded games per sample
            lengths = (~game_mask).sum(dim=1).cpu()  # (B,)
            # Clamp to at least 1 to avoid pack_padded_sequence errors
            lengths = lengths.clamp(min=1)

            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out_packed, _ = self.gru(packed)
            gru_out, _ = pad_packed_sequence(
                gru_out_packed, batch_first=True, total_length=G
            )
        else:
            gru_out, _ = self.gru(x)

        # gru_out: (B, G, hidden_dim) — forward + backward concatenated

        # 5. Output normalization
        gru_out = self.output_norm(gru_out)

        # 6. Multi-query attention pool over GRU hidden states
        return self.pool(gru_out, mask=game_mask)
