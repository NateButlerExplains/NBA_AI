"""Outcome head for per-position final spread prediction (endpoint guidance).

Produces a Gaussian distribution (mu, sigma) for the final game spread at
every sequence position. During training, the loss is weighted by game_progress
so late-game positions (most informative) get stronger signal.

Architecture:
    Linear(512, 128) -> GELU -> Linear(128, 2)
    -> split into mu (unbounded) and raw_sigma
    -> sigma = softplus(raw_sigma).clamp(min=min_std, max=max_std)

Input:  (B, T, 512) decoder output at every position
Output: (spread_mu, spread_sigma) each (B, T)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.generative.config import GenerativeModelConfig


class OutcomeHead(nn.Module):
    """Gaussian head for per-position final spread prediction.

    Predicts the final game spread (home_score - away_score) from every
    decoder position. The sigma output is constrained to [min_std, max_std]
    via softplus + clamping.
    """

    def __init__(self, config: GenerativeModelConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim  # 512
        head_hidden = config.head_hidden_dim  # 128
        self.min_std = config.outcome_min_std  # 1.0
        self.max_std = config.outcome_max_std  # 20.0

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),  # [mu, raw_sigma]
        )

        # Initialize bias so initial sigma ~ 10 (reasonable for NBA spread uncertainty)
        # softplus(x) = log(1 + exp(x)), so we want softplus(bias) ~ 9
        # (then clamp ensures min_std=1.0 is the floor, so effective sigma ~ 10)
        with torch.no_grad():
            self.net[-1].bias[1] = math.log(math.exp(9.0) - 1)  # softplus^-1(9)

    def forward(self, decoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict final spread distribution at every sequence position.

        Args:
            decoder_out: (B, T, 512) decoder hidden states.

        Returns:
            spread_mu: (B, T) predicted final spread (unbounded).
            spread_sigma: (B, T) predicted spread uncertainty, in [min_std, max_std].
        """
        out = self.net(decoder_out)  # (B, T, 2)
        mu = out[..., 0]  # (B, T)
        raw_sigma = out[..., 1]  # (B, T)

        sigma = F.softplus(raw_sigma).clamp(
            min=self.min_std, max=self.max_std
        )  # (B, T)

        return mu, sigma
