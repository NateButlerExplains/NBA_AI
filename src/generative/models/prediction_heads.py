"""Prediction heads for the generative model.

ScoreHead: 7-class score event classification.
ClockHead: next state normalized clock prediction.
ContextMarginHead: expected final margin from context tokens.
"""

import torch
import torch.nn as nn

from src.generative.config import GenerativeModelConfig


class ScoreHead(nn.Module):
    """Predict 7-class score event logits.

    Classes: {no_score, home+1, home+2, home+3, away+1, away+2, away+3}
    """

    def __init__(self, hidden_dim: int, head_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce score event logits.

        Args:
            x: (..., hidden_dim) decoder output.

        Returns:
            (..., 7) score event logits.
        """
        return self.net(x)


class ClockHead(nn.Module):
    """Predict next state's normalized clock value."""

    def __init__(self, hidden_dim: int, head_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict clock value.

        Args:
            x: (..., hidden_dim) decoder output.

        Returns:
            (..., 1) clock prediction.
        """
        return self.net(x)


class ContextMarginHead(nn.Module):
    """Predict expected final margin from context tokens alone."""

    def __init__(self, hidden_dim: int, head_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict final margin.

        Args:
            x: (..., hidden_dim) decoder output.

        Returns:
            (..., 1) margin prediction.
        """
        return self.net(x)
