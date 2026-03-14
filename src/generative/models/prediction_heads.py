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


class PreDecoderHead(nn.Module):
    """Pre-decoder prediction from matchup features.

    Takes matchup = cat(home_ctx, away_ctx, home_ctx - away_ctx) and produces
    a scalar prediction. Used for both margin (MSE) and win (BCE) targets —
    the loss function handles the difference.

    Provides direct gradient to context encoder without going through decoder.
    """

    def __init__(self, hidden_dim: int, head_hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, head_hidden_dim),  # 1536 → 128
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, matchup: torch.Tensor) -> torch.Tensor:
        """Predict scalar from matchup features.

        Args:
            matchup: (B, hidden_dim * 3) = cat(home, away, home - away).

        Returns:
            (B, 1) scalar prediction.
        """
        return self.net(matchup)


# Aliases for clarity in model wiring
PreDecoderMarginHead = PreDecoderHead
PreDecoderWinHead = PreDecoderHead


class ContextScoreBias(nn.Module):
    """Compute score event bias from context tokens.

    Creates a direct gradient path from context encoder to score predictions,
    bypassing the decoder's self-attention. Takes concatenated home+away context
    and produces a 7-dim bias added to score logits at every state position.
    """

    def __init__(
        self, hidden_dim: int, head_hidden_dim: int, n_classes: int = 7
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_hidden_dim),  # concat home+away
            nn.GELU(),
            nn.Linear(head_hidden_dim, n_classes),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Compute score bias from context.

        Args:
            context_tokens: (B, 2, hidden_dim) [home, away] context.

        Returns:
            (B, n_classes) score logit bias.
        """
        ctx_flat = context_tokens.reshape(context_tokens.shape[0], -1)
        return self.net(ctx_flat)
