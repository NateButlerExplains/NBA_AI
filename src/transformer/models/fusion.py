"""
Fusion Layers for NBA Game Prediction.

Combines team representations into a unified matchup representation.

Available fusion types:
- SimpleFusion: Concatenation + MLP (baseline)
- CrossAttentionFusion: Home/away cross-attend before MLP (explicit matchup modeling)

Usage:
    fusion = SimpleFusion(hidden_dim=256)
    matchup_repr = fusion(home_repr, away_repr)

    fusion = CrossAttentionFusion(hidden_dim=256)
    matchup_repr = fusion(home_repr, away_repr)
"""

import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    """
    Fusion layer using concatenation and MLP.

    Concatenates home and away team history representations and passes
    them through an MLP to learn interactions between the two teams.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_input_streams: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input: concatenation of all streams.
        # Default (2): home_history + away_history = 512-d
        # With roster (4): home_hist + away_hist + home_roster + away_roster = 1024-d
        input_dim = hidden_dim * n_input_streams

        # Fusion MLP: compresses the concatenated streams.
        # The MLP must learn ALL interactions between teams, so it's
        # deep enough to model non-trivial relationships.
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        """
        Fuse input streams into a matchup representation.

        Args:
            *streams: Variable number of (batch, hidden_dim) tensors.
                Default (2 streams): home_history, away_history
                With roster (4 streams): home_hist, away_hist, home_roster, away_roster

        Returns:
            Matchup representation of shape (batch, hidden_dim)
        """
        combined = torch.cat(streams, dim=-1)
        return self.fusion(combined)  # (batch, hidden_dim)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: home and away team representations explicitly
    attend to each other before final MLP projection.

    Unlike SimpleFusion (which just concatenates and projects), this lets
    the model directly compare teams: "how does home's strength relate to
    away's defense?" The cross-attention creates matchup-aware representations
    before the final projection.

    Input: two team representations, each (batch, hidden_dim).
    Output: single matchup representation (batch, hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Home queries away (learns what about the away team matters for home)
        self.home_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        # Away queries home (learns what about the home team matters for away)
        self.away_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # Post-attention layer norms (pre-norm style for stable training)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.dropout_h = nn.Dropout(dropout)
        self.dropout_a = nn.Dropout(dropout)

        # Final MLP: concat cross-attended representations → matchup vector
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, home_repr: torch.Tensor, away_repr: torch.Tensor) -> torch.Tensor:
        """
        Cross-attend home and away team representations.

        Args:
            home_repr: (batch, hidden_dim) — combined home team representation
            away_repr: (batch, hidden_dim) — combined away team representation

        Returns:
            Matchup representation of shape (batch, hidden_dim)
        """
        # Unsqueeze to (batch, 1, hidden_dim) for attention interface
        h = home_repr.unsqueeze(1)
        a = away_repr.unsqueeze(1)

        # Home attends to away (what about the opponent matters for home?)
        h_cross, _ = self.home_cross_attn(query=h, key=a, value=a)
        # Away attends to home (what about the opponent matters for away?)
        a_cross, _ = self.away_cross_attn(query=a, key=h, value=h)

        # Residual connection + LayerNorm (stabilizes training)
        h_out = self.norm_h(home_repr + self.dropout_h(h_cross.squeeze(1)))
        a_out = self.norm_a(away_repr + self.dropout_a(a_cross.squeeze(1)))

        # Concatenate matchup-aware representations and project to final matchup vector
        return self.mlp(torch.cat([h_out, a_out], dim=-1))


def test_fusion():
    """Test SimpleFusion and CrossAttentionFusion with sample data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    hidden_dim = 256
    batch_size = 4

    # Create sample inputs (as if from TemporalAttention)
    home_history = torch.randn(batch_size, hidden_dim)
    away_history = torch.randn(batch_size, hidden_dim)

    # Test SimpleFusion
    print("Testing SimpleFusion...")
    fusion = SimpleFusion(hidden_dim=hidden_dim)
    fusion.eval()

    print(f"  Parameters: {sum(p.numel() for p in fusion.parameters()):,}")

    with torch.no_grad():
        output = fusion(home_history, away_history)

    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {hidden_dim})")
    assert output.shape == (batch_size, hidden_dim), "Shape mismatch!"

    # Test gradient flow
    print("\nTesting gradient flow...")
    fusion.train()
    home = torch.randn(batch_size, hidden_dim, requires_grad=True)
    away = torch.randn(batch_size, hidden_dim, requires_grad=True)
    out = fusion(home, away)
    out.sum().backward()
    assert home.grad is not None, "No gradient for home_history"
    assert away.grad is not None, "No gradient for away_history"
    print("  Gradients flow correctly")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_fusion())
