"""
Phase 2 Temporal Attention.

3-layer pre-norm transformer with days-before-target positional encoding
and multi-query attention pooling.
"""

import torch
import torch.nn as nn


class DaysBeforePositionalEncoding(nn.Module):
    """Calendar-distance positional encoding: Embedding(180, hidden_dim)."""

    def __init__(self, hidden_dim: int = 512, max_days: int = 180, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_days, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, days_before: torch.Tensor) -> torch.Tensor:
        """
        Add calendar-distance positional encoding.

        Args:
            x: (B, n_games, hidden_dim)
            days_before: (B, n_games) int64, clamped 0-179

        Returns:
            (B, n_games, hidden_dim)
        """
        pos_emb = self.position_embedding(days_before.clamp(0, 179))
        return self.dropout(x + pos_emb)


class OrdinalPositionalEncoding(nn.Module):
    """Ordinal positional encoding: Embedding(max_len, hidden_dim) indexed by sequence position."""

    def __init__(self, hidden_dim: int = 512, max_len: int = 82, dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, days_before: torch.Tensor = None) -> torch.Tensor:
        """
        Add ordinal positional encoding (ignores days_before).

        Args:
            x: (B, n_games, hidden_dim)
            days_before: ignored, accepted for interface compatibility

        Returns:
            (B, n_games, hidden_dim)
        """
        B, n_games, _ = x.shape
        positions = torch.arange(n_games, device=x.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)
        return self.dropout(x + pos_emb)


class TemporalAttentionLayer(nn.Module):
    """Pre-norm self-attention + FFN layer."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=mask, need_weights=False
        )
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MultiQueryAttentionPool(nn.Module):
    """
    4 learned queries attend to the full sequence, concatenate, project to hidden_dim.

    Output: (B, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 512, n_queries: int = 4, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.n_queries = n_queries
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim))
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.projection = nn.Sequential(
            nn.Linear(n_queries * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, n_games, hidden_dim)
            mask: (B, n_games) bool, True=padding

        Returns:
            (B, hidden_dim)
        """
        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        pooled, _ = self.attention(
            queries, x, x, key_padding_mask=mask, need_weights=False
        )
        # pooled: (B, n_queries, hidden_dim)

        concat = pooled.reshape(batch_size, self.n_queries * self.hidden_dim)
        return self.projection(concat)


class Phase2TemporalAttention(nn.Module):
    """
    3-layer pre-norm transformer with days-before-target PE and multi-query pooling.

    Input: game_embeddings (B, n_games, 512), days_before (B, n_games), game_mask (B, n_games)
    Output: (B, 512)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_days: int = 180,
        n_pool_queries: int = 4,
        pos_encoding: str = "days_before",
    ):
        super().__init__()

        if pos_encoding == "ordinal":
            self.pos_encoder = OrdinalPositionalEncoding(hidden_dim, max_len=82, dropout=dropout)
        else:
            self.pos_encoder = DaysBeforePositionalEncoding(hidden_dim, max_days, dropout)

        self.layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.pool = MultiQueryAttentionPool(
            hidden_dim, n_pool_queries, num_heads, dropout
        )

    def forward_positions(
        self,
        game_embeddings: torch.Tensor,
        days_before: torch.Tensor,
        game_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return per-position contextualized embeddings (B, G, h) before pooling.

        Args:
            game_embeddings: (B, n_games, hidden_dim)
            days_before: (B, n_games) int64
            game_mask: (B, n_games) bool, True=padding

        Returns:
            (B, n_games, hidden_dim)
        """
        x = self.pos_encoder(game_embeddings, days_before)

        for layer in self.layers:
            x = layer(x, mask=game_mask)

        return self.norm(x)

    def forward(
        self,
        game_embeddings: torch.Tensor,
        days_before: torch.Tensor,
        game_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            game_embeddings: (B, n_games, hidden_dim)
            days_before: (B, n_games) int64
            game_mask: (B, n_games) bool, True=padding

        Returns:
            (B, hidden_dim)
        """
        x = self.forward_positions(game_embeddings, days_before, game_mask)
        return self.pool(x, mask=game_mask)
