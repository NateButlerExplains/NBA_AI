"""
Phase 2 Loss Function.

Wraps Phase 1's CombinedLoss and adds consistency loss:
MSE(spread_mean, home_score_mean - away_score_mean)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer.models.prediction_heads import GamePrediction
from src.transformer.training.loss import CombinedLoss


class ConsistencyLoss(nn.Module):
    """
    Encourages coherent predictions: spread should match score difference.

    MSE(μ_spread, μ_home - μ_away)
    """

    def forward(self, predictions: GamePrediction) -> torch.Tensor:
        score_diff = predictions.home_score_mean - predictions.away_score_mean
        return F.mse_loss(predictions.spread_mean, score_diff)


class Phase2CombinedLoss(nn.Module):
    """
    Phase 2 combined loss = CombinedLoss + weighted ConsistencyLoss.

    Total = spread_weight × SpreadLoss + score_weight × ScoreLoss
          + win_prob_weight × WinProbLoss + consistency_weight × ConsistencyLoss
    """

    def __init__(
        self,
        spread_weight: float = 1.0,
        score_weight: float = 0.5,
        win_prob_weight: float = 0.3,
        consistency_weight: float = 0.1,
        nll_weight: float = 1.0,
        mse_weight: float = 0.1,
        spread_huber_delta: Optional[float] = None,
        score_huber_delta: Optional[float] = None,
    ):
        super().__init__()

        self.combined_loss = CombinedLoss(
            spread_weight=spread_weight,
            score_weight=score_weight,
            win_prob_weight=win_prob_weight,
            nll_weight=nll_weight,
            mse_weight=mse_weight,
            spread_huber_delta=spread_huber_delta,
            score_huber_delta=score_huber_delta,
        )

        self.consistency_loss = ConsistencyLoss()
        self.consistency_weight = consistency_weight

    def forward(
        self,
        predictions: GamePrediction,
        target_home_scores: torch.Tensor,
        target_away_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Base combined loss
        result = self.combined_loss(predictions, target_home_scores, target_away_scores)

        # Add consistency loss
        consistency = self.consistency_loss(predictions)
        result["consistency"] = consistency
        result["total"] = result["total"] + self.consistency_weight * consistency

        return result
