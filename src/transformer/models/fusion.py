"""
Fusion Layer for NBA Game Prediction.

Combines team history streams into a unified matchup representation:
- Home team history embedding (from TemporalAttention)
- Away team history embedding (from TemporalAttention)

Uses concatenation + MLP to learn interactions between home and away teams.

Usage:
    fusion = SimpleFusion(hidden_dim=256)
    matchup_repr = fusion(home_history, away_history)
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
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input: home_history + away_history = 2 streams
        input_dim = hidden_dim * 2

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

    def forward(
        self,
        home_history: torch.Tensor,
        away_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse home and away history streams.

        Args:
            home_history: (batch, hidden_dim) from Temporal Attention
            away_history: (batch, hidden_dim) from Temporal Attention

        Returns:
            Matchup representation of shape (batch, hidden_dim)
        """
        combined = torch.cat([home_history, away_history], dim=-1)
        return self.fusion(combined)  # (batch, hidden_dim)


def test_fusion():
    """Test SimpleFusion with sample data."""
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
