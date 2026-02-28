"""
Loss Functions for NBA Transformer Training.

Implements Gaussian NLL loss for probabilistic predictions,
along with auxiliary losses for training stability.

KEY CONCEPT — Why Gaussian NLL?
    Unlike a simple model that predicts a single number (e.g., "the spread will be 5"),
    our model predicts a *distribution*: a mean (mu) and standard deviation (sigma).
    This lets us say "the spread will be around 5, give or take 3 points."

    The Gaussian NLL (Negative Log-Likelihood) loss measures how well that predicted
    distribution fits the actual outcome. A lower NLL means the model assigned higher
    probability to what actually happened.

    Example:
        Model predicts mu=5, sigma=3. Actual spread is 5.  -> NLL is LOW (good!)
        Model predicts mu=5, sigma=3. Actual spread is 20. -> NLL is HIGH (bad!)
        Model predicts mu=5, sigma=20. Actual spread is 5. -> NLL is MEDIUM
            (the prediction "covers" the outcome, but the log(sigma) term penalizes
             overly wide/uncertain predictions — encouraging tight, confident predictions)

Usage:
    criterion = CombinedLoss(spread_weight=1.0, score_weight=0.5)
    loss_dict = criterion(predictions, targets)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer.models.prediction_heads import GamePrediction


def gaussian_nll_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.

    NLL = 0.5 * log(2π) + log(σ) + 0.5 * ((y - μ) / σ)²

    Args:
        target: Ground truth values (batch,)
        mean: Predicted means (batch,)
        std: Predicted standard deviations (batch,)
        reduction: "mean", "sum", or "none"
        eps: Small constant for numerical stability

    Returns:
        Loss value
    """
    # Variance = sigma^2. We add eps (a tiny number like 0.000001) to prevent
    # division by zero if the model ever predicts sigma=0.
    var = std.pow(2) + eps

    # log(sigma) is used in the NLL formula. eps prevents log(0) = -infinity.
    log_std = torch.log(std + eps)

    # The Gaussian NLL formula has three terms:
    #   1. 0.5 * log(2*pi) — a constant (doesn't affect optimization, but keeps the math correct)
    #   2. log(sigma) — penalizes overly wide predictions (forces the model to be confident)
    #   3. (y - mu)^2 / (2 * sigma^2) — penalizes predictions far from the actual value,
    #      scaled by the predicted uncertainty. If sigma is large, this term is forgiving;
    #      if sigma is small, even small errors get heavily penalized.
    nll = 0.5 * (math.log(2 * math.pi) + 2 * log_std + (target - mean).pow(2) / var)

    # Reduction controls how we aggregate the per-sample losses:
    #   "mean" = average over all samples (most common for training)
    #   "sum"  = sum over all samples
    #   "none" = return per-sample losses (useful for debugging or weighting)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


class SpreadLoss(nn.Module):
    """
    Loss for spread (point differential) prediction.

    Combines NLL for probabilistic prediction with optional
    auxiliary MSE (or Huber) for gradient stability.

    WHY COMBINE NLL + MSE?
        NLL alone trains the full distribution (mean AND uncertainty), but early
        in training the gradients from NLL can be noisy because the model is also
        learning sigma. The auxiliary MSE loss provides a direct, clean gradient
        signal on just the mean prediction: "move the mean closer to the target."
        This helps the model learn reasonable mean predictions faster, stabilizing
        early training. The MSE weight is kept small (0.1) so it does not
        dominate the NLL objective long-term.
    """

    def __init__(
        self,
        nll_weight: float = 1.0,
        mse_weight: float = 0.1,
        huber_delta: Optional[float] = None,
    ):
        super().__init__()
        self.nll_weight = nll_weight  # Weight for the probabilistic NLL component
        self.mse_weight = mse_weight  # Weight for the auxiliary MSE/Huber component
        self.huber_delta = huber_delta  # If set, use Huber instead of MSE

    def forward(
        self,
        spread_mean: torch.Tensor,
        spread_std: torch.Tensor,
        target_spread: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute spread loss.

        Args:
            spread_mean: Predicted spread mean
            spread_std: Predicted spread std
            target_spread: Actual spread (home - away)

        Returns:
            Dict with loss components
        """
        # NLL trains the full predicted distribution (mean + uncertainty)
        nll = gaussian_nll_loss(target_spread, spread_mean, spread_std)

        # Auxiliary loss on mean prediction. Huber is robust to blowout outliers
        # (transitions from L2 to L1 at delta), MSE is the simpler default.
        if self.huber_delta is not None:
            mse = F.huber_loss(spread_mean, target_spread, delta=self.huber_delta)
        else:
            mse = F.mse_loss(spread_mean, target_spread)

        # Weighted combination: NLL dominates (1.0) while MSE/Huber is auxiliary
        total = self.nll_weight * nll + self.mse_weight * mse

        # Return all components separately so we can log and monitor each one
        return {
            "spread_total": total,
            "spread_nll": nll,
            "spread_mse": mse,
        }


class ScoreLoss(nn.Module):
    """
    Loss for absolute score prediction (e.g., home=110, away=105).

    Handles both home and away score predictions separately, then averages them.
    Same NLL + MSE/Huber approach as SpreadLoss, applied to each team's score.
    """

    def __init__(
        self,
        nll_weight: float = 1.0,
        mse_weight: float = 0.1,
        huber_delta: Optional[float] = None,
    ):
        super().__init__()
        self.nll_weight = nll_weight
        self.mse_weight = mse_weight
        self.huber_delta = huber_delta  # If set, use Huber instead of MSE

    def forward(
        self,
        home_mean: torch.Tensor,
        home_std: torch.Tensor,
        away_mean: torch.Tensor,
        away_std: torch.Tensor,
        target_home: torch.Tensor,
        target_away: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute score loss.

        Returns:
            Dict with loss components
        """
        # Compute NLL for each team's score prediction independently
        home_nll = gaussian_nll_loss(target_home, home_mean, home_std)
        away_nll = gaussian_nll_loss(target_away, away_mean, away_std)

        # Auxiliary loss for each team's score (Huber or MSE)
        if self.huber_delta is not None:
            home_mse = F.huber_loss(home_mean, target_home, delta=self.huber_delta)
            away_mse = F.huber_loss(away_mean, target_away, delta=self.huber_delta)
        else:
            home_mse = F.mse_loss(home_mean, target_home)
            away_mse = F.mse_loss(away_mean, target_away)

        # Average across home and away so both contribute equally
        nll = (home_nll + away_nll) / 2
        mse = (home_mse + away_mse) / 2

        total = self.nll_weight * nll + self.mse_weight * mse

        return {
            "score_total": total,
            "score_nll": nll,
            "score_mse": mse,
            "home_nll": home_nll,
            "away_nll": away_nll,
            "home_mse": home_mse,
            "away_mse": away_mse,
        }


class WinProbLoss(nn.Module):
    """
    Loss for win probability prediction.

    Uses Binary Cross-Entropy (BCE) — the standard loss function for probability
    predictions. BCE measures how far the predicted probability is from the actual
    binary outcome (win=1, loss=0).

    Example:
        Model predicts 90% home win probability, home team wins  -> BCE is low (good)
        Model predicts 90% home win probability, home team loses -> BCE is high (bad)
        Model predicts 50% (no idea), either outcome             -> BCE is moderate
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        # Label smoothing converts hard targets (0 or 1) toward 0.5 slightly.
        # This prevents the model from becoming overconfident (predicting 99.9%).
        self.label_smoothing = label_smoothing

    def forward(
        self,
        win_prob: torch.Tensor,
        target_home: torch.Tensor,
        target_away: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute win probability loss.

        Args:
            win_prob: Predicted home win probability
            target_home: Actual home score
            target_away: Actual away score

        Returns:
            Dict with loss components
        """
        # Convert scores to a binary outcome: 1.0 if home won, 0.0 if away won.
        # Ties are extremely rare in the NBA (overtime rules), but we handle them
        # as 0.5 to be safe.
        home_won = (target_home > target_away).float()
        tied = (target_home == target_away).float()
        target = home_won + 0.5 * tied

        # Label smoothing: instead of training on hard 0/1 targets, we soften
        # them slightly toward 0.5. E.g., with smoothing=0.1, a "1" becomes 0.95
        # and a "0" becomes 0.05. This regularizes the model against overconfidence.
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clamp predicted probabilities away from exactly 0 or 1.
        # BCE computes log(p) and log(1-p), which go to -infinity at the extremes.
        # Clamping to [1e-7, 1-1e-7] prevents NaN/Inf in the loss.
        win_prob_clamped = win_prob.clamp(1e-7, 1 - 1e-7)

        # IMPORTANT CUDA WORKAROUND: Binary cross-entropy is numerically unstable
        # when run in float16 (half precision) during mixed-precision training.
        # The log operations can overflow/underflow in float16's limited range.
        # We disable autocast here to force this computation in float32 (full precision).
        with torch.amp.autocast(device_type="cuda", enabled=False):
            bce = F.binary_cross_entropy(
                win_prob_clamped.float(), target.float()  # .float() ensures float32
            )

        return {
            "win_prob_bce": bce,
        }


class CombinedLoss(nn.Module):
    """
    Combined loss function that our model actually trains on.

    Our model makes three types of predictions simultaneously, and each has its
    own loss function. CombinedLoss computes all three and combines them into
    a single scalar that PyTorch can backpropagate through.

    The weights control how much each task matters during training:
        - spread_weight = 1.0  (PRIMARY objective — predicting the point differential)
        - score_weight  = 0.5  (secondary — predicting individual team scores)
        - win_prob_weight = 0.1 (auxiliary — predicting who wins)

    Spread prediction is our main goal, so it gets the highest weight. Score
    prediction supports spread learning (if you predict scores well, the spread
    follows). Win probability is cheap to compute from the spread, so it gets
    a small weight — just enough to learn calibrated probabilities.
    """

    def __init__(
        self,
        spread_weight: float = 1.0,
        score_weight: float = 0.5,
        win_prob_weight: float = 0.1,
        nll_weight: float = 1.0,
        mse_weight: float = 0.1,
        label_smoothing: float = 0.0,
        spread_huber_delta: Optional[float] = None,
        score_huber_delta: Optional[float] = None,
    ):
        """
        Initialize combined loss.

        Args:
            spread_weight: Weight for spread loss (default 1.0 — primary objective)
            score_weight: Weight for score loss (default 0.5 — secondary)
            win_prob_weight: Weight for win probability loss (default 0.1 — auxiliary)
            nll_weight: Weight for NLL component within each loss
            mse_weight: Weight for MSE component within each loss
            label_smoothing: Label smoothing for win probability
            spread_huber_delta: If set, use Huber loss (with this delta) instead of MSE for spread
            score_huber_delta: If set, use Huber loss (with this delta) instead of MSE for scores
        """
        super().__init__()

        # These weights control the relative importance of each prediction task
        self.spread_weight = spread_weight
        self.score_weight = score_weight
        self.win_prob_weight = win_prob_weight

        # Each sub-loss handles one prediction task
        self.spread_loss = SpreadLoss(nll_weight, mse_weight, spread_huber_delta)
        self.score_loss = ScoreLoss(nll_weight, mse_weight, score_huber_delta)
        self.win_prob_loss = WinProbLoss(label_smoothing)

    def forward(
        self,
        predictions: GamePrediction,
        target_home_scores: torch.Tensor,
        target_away_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: GamePrediction from model
            target_home_scores: Actual home scores
            target_away_scores: Actual away scores

        Returns:
            Dict with all loss components and total
        """
        # The spread is simply home score minus away score (e.g., 110-105 = +5 home)
        target_spread = target_home_scores - target_away_scores

        # Compute each loss independently — they operate on different prediction heads
        spread_losses = self.spread_loss(
            predictions.spread_mean,
            predictions.spread_std,
            target_spread,
        )

        score_losses = self.score_loss(
            predictions.home_score_mean,
            predictions.home_score_std,
            predictions.away_score_mean,
            predictions.away_score_std,
            target_home_scores,
            target_away_scores,
        )

        win_prob_losses = self.win_prob_loss(
            predictions.home_win_prob,
            target_home_scores,
            target_away_scores,
        )

        # Weighted sum of all three losses into one scalar value.
        # This is the single number that PyTorch will backpropagate through to
        # update ALL model weights. The weights determine how much each task
        # influences the gradient updates:
        #   spread (1.0) contributes most, score (0.5) half as much, win prob (0.1) least
        total = (
            self.spread_weight * spread_losses["spread_total"]
            + self.score_weight * score_losses["score_total"]
            + self.win_prob_weight * win_prob_losses["win_prob_bce"]
        )

        # Return all individual components alongside the total — this lets us
        # log each loss separately to monitor which tasks are improving
        result = {"total": total}
        result.update(spread_losses)
        result.update(score_losses)
        result.update(win_prob_losses)

        return result


def test_losses():
    """Test loss functions."""
    import logging

    logging.basicConfig(level=logging.INFO)

    batch_size = 8

    # Create sample predictions
    predictions = GamePrediction(
        spread_mean=torch.randn(batch_size),
        spread_std=torch.ones(batch_size) * 5,
        home_score_mean=torch.ones(batch_size) * 110,
        home_score_std=torch.ones(batch_size) * 10,
        away_score_mean=torch.ones(batch_size) * 108,
        away_score_std=torch.ones(batch_size) * 10,
        home_win_prob=torch.sigmoid(torch.randn(batch_size)),
    )

    # Sample targets
    target_home = torch.tensor([110.0, 105.0, 115.0, 100.0, 120.0, 95.0, 108.0, 112.0])
    target_away = torch.tensor([108.0, 112.0, 102.0, 98.0, 118.0, 100.0, 110.0, 105.0])

    # Test individual losses
    print("Testing SpreadLoss...")
    spread_loss = SpreadLoss()
    spread_result = spread_loss(
        predictions.spread_mean,
        predictions.spread_std,
        target_home - target_away,
    )
    print(f"  Total: {spread_result['spread_total'].item():.4f}")
    print(f"  NLL: {spread_result['spread_nll'].item():.4f}")
    print(f"  MSE: {spread_result['spread_mse'].item():.4f}")

    print("\nTesting ScoreLoss...")
    score_loss = ScoreLoss()
    score_result = score_loss(
        predictions.home_score_mean,
        predictions.home_score_std,
        predictions.away_score_mean,
        predictions.away_score_std,
        target_home,
        target_away,
    )
    print(f"  Total: {score_result['score_total'].item():.4f}")
    print(f"  NLL: {score_result['score_nll'].item():.4f}")

    print("\nTesting WinProbLoss...")
    win_loss = WinProbLoss()
    win_result = win_loss(
        predictions.home_win_prob,
        target_home,
        target_away,
    )
    print(f"  BCE: {win_result['win_prob_bce'].item():.4f}")

    # Test combined loss
    print("\nTesting CombinedLoss...")
    combined = CombinedLoss(
        spread_weight=1.0,
        score_weight=0.5,
        win_prob_weight=0.1,
    )

    result = combined(predictions, target_home, target_away)
    print(f"  Total: {result['total'].item():.4f}")
    print(f"  Components: {list(result.keys())}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    predictions_grad = GamePrediction(
        spread_mean=torch.randn(batch_size, requires_grad=True),
        spread_std=F.softplus(torch.randn(batch_size, requires_grad=True)) + 1,
        home_score_mean=torch.randn(batch_size, requires_grad=True) + 110,
        home_score_std=F.softplus(torch.randn(batch_size, requires_grad=True)) + 5,
        away_score_mean=torch.randn(batch_size, requires_grad=True) + 108,
        away_score_std=F.softplus(torch.randn(batch_size, requires_grad=True)) + 5,
        home_win_prob=torch.sigmoid(torch.randn(batch_size, requires_grad=True)),
    )

    result = combined(predictions_grad, target_home, target_away)
    result["total"].backward()

    print(f"  spread_mean grad: {predictions_grad.spread_mean.grad is not None}")
    print(f"  home_score_mean grad: {predictions_grad.home_score_mean.grad is not None}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_losses())
