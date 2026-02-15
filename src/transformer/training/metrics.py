"""
Evaluation Metrics for NBA Transformer Model.

This module answers the question: "How good are our predictions?"
There are many different ways to measure prediction quality, and each
metric captures a different aspect:

- Point prediction accuracy (MAE, RMSE):
    How close is our predicted spread/score to the actual value?
    Our goal: beat the XGBoost baseline of ~10.1 MAE on spread.

- Probability calibration (Brier, ECE):
    When we say "70% chance home wins," does home actually win ~70% of the time?

- Classification performance (AUC, Accuracy):
    How well can our win probabilities separate actual wins from losses?

- Uncertainty quality (Coverage, Sharpness):
    Are our confidence intervals (error bars) reliable and tight?

Usage:
    metrics = MetricsCalculator()
    metrics.update(predictions, targets)   # call once per batch
    results = metrics.compute()            # compute everything at the end
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from src.transformer.models.prediction_heads import GamePrediction


@dataclass
class MetricResults:
    """Container for all computed metrics — one number per metric."""

    # --- Point prediction metrics ---
    # MAE (Mean Absolute Error): average of |predicted - actual|.
    # Lower is better. Our XGBoost baseline is ~10.1 for spread.
    spread_mae: float = 0.0
    # RMSE (Root Mean Squared Error): sqrt(mean((predicted - actual)^2)).
    # Like MAE, but penalizes large errors MORE heavily because it squares them
    # before averaging. Useful for catching models that are "usually okay but
    # occasionally wildly wrong."
    spread_rmse: float = 0.0
    home_mae: float = 0.0
    away_mae: float = 0.0
    total_mae: float = 0.0  # MAE for total points (home + away combined)

    # --- Win probability metrics ---
    win_accuracy: float = 0.0  # Simple accuracy: % of games where favored team won
    # AUC (Area Under the ROC Curve): measures how well win probabilities separate
    # actual wins from losses. 1.0 = perfect separation, 0.5 = random guessing.
    win_auc: float = 0.0
    # Brier Score: mean((predicted_prob - actual_outcome)^2).
    # 0.0 = perfect, 0.25 = random guessing (always predicting 50%).
    # Rewards both accuracy AND calibration (confident correct predictions score best).
    brier_score: float = 0.0

    # --- Calibration metrics ---
    # ECE (Expected Calibration Error): bins predictions by confidence level and
    # checks if predicted probabilities match observed frequencies.
    # E.g., do games we predict with 70% confidence actually result in wins ~70%?
    # 0.0 = perfectly calibrated.
    ece: float = 0.0
    # MCE (Maximum Calibration Error): the worst single bin's calibration gap.
    mce: float = 0.0

    # --- Uncertainty metrics ---
    # Coverage: what fraction of actual outcomes fall within our confidence interval?
    # A well-calibrated 90% CI should contain the true value ~90% of the time.
    # If coverage >> 90%, our intervals are too wide (underconfident).
    # If coverage << 90%, our intervals are too narrow (overconfident).
    spread_coverage_90: float = 0.0  # 90% prediction interval coverage
    spread_coverage_50: float = 0.0  # 50% prediction interval coverage
    # Sharpness: average width of the 90% confidence interval.
    # We want this to be as NARROW as possible while maintaining correct coverage.
    spread_sharpness: float = 0.0
    home_coverage_90: float = 0.0
    away_coverage_90: float = 0.0

    # Sample count
    n_samples: int = 0


class MetricsCalculator:
    """
    Accumulates predictions across batches, then computes all metrics at once.

    WHY ACCUMULATE INSTEAD OF COMPUTING PER-BATCH?
        Some metrics (like AUC and calibration) need to see the FULL dataset
        to be computed correctly. You cannot compute AUC on a single batch of
        8 samples and average it — the result would be meaningless. So we store
        all predictions during evaluation, then compute everything at the end.

    Usage pattern:
        calculator = MetricsCalculator()
        for batch in test_loader:
            predictions = model(batch)
            calculator.update(predictions, targets)   # accumulates
        results = calculator.compute()                 # computes all metrics
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reset()

    def reset(self):
        """Clear accumulated predictions — call before starting a new evaluation."""
        # Each list will hold one numpy array per batch, concatenated later
        self.spread_preds = []
        self.spread_stds = []
        self.home_preds = []
        self.home_stds = []
        self.away_preds = []
        self.away_stds = []
        self.win_probs = []
        self.target_spreads = []
        self.target_homes = []
        self.target_aways = []

    def update(
        self,
        predictions: GamePrediction,
        target_home_scores: torch.Tensor,
        target_away_scores: torch.Tensor,
    ):
        """
        Add batch of predictions to accumulator.

        Args:
            predictions: Model predictions
            target_home_scores: Actual home scores
            target_away_scores: Actual away scores
        """
        # Move tensors from GPU to CPU and convert to numpy arrays for metric
        # computation. We do this per-batch to avoid accumulating large GPU tensors.
        self.spread_preds.append(predictions.spread_mean.cpu().numpy())
        self.spread_stds.append(predictions.spread_std.cpu().numpy())
        self.home_preds.append(predictions.home_score_mean.cpu().numpy())
        self.home_stds.append(predictions.home_score_std.cpu().numpy())
        self.away_preds.append(predictions.away_score_mean.cpu().numpy())
        self.away_stds.append(predictions.away_score_std.cpu().numpy())
        self.win_probs.append(predictions.home_win_prob.cpu().numpy())

        # Compute target spread (home - away) from the raw scores
        target_spread = (target_home_scores - target_away_scores).cpu().numpy()
        self.target_spreads.append(target_spread)
        self.target_homes.append(target_home_scores.cpu().numpy())
        self.target_aways.append(target_away_scores.cpu().numpy())

    def compute(self) -> MetricResults:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            MetricResults with all computed values
        """
        if not self.spread_preds:
            return MetricResults()

        # Concatenate all per-batch arrays into single arrays spanning the full dataset.
        # E.g., if we had 100 batches of 32 samples, we now have arrays of 3200 samples.
        spread_pred = np.concatenate(self.spread_preds)
        spread_std = np.concatenate(self.spread_stds)
        home_pred = np.concatenate(self.home_preds)
        home_std = np.concatenate(self.home_stds)
        away_pred = np.concatenate(self.away_preds)
        away_std = np.concatenate(self.away_stds)
        win_prob = np.concatenate(self.win_probs)

        target_spread = np.concatenate(self.target_spreads)
        target_home = np.concatenate(self.target_homes)
        target_away = np.concatenate(self.target_aways)

        n_samples = len(spread_pred)

        # ---- Point prediction metrics ----
        # MAE = mean(|predicted - actual|). Simple, interpretable: "on average, we're
        # off by X points." Our target is to beat the XGBoost baseline of ~10.1.
        spread_mae = np.mean(np.abs(spread_pred - target_spread))
        # RMSE = sqrt(mean((predicted - actual)^2)). Squaring means a single
        # prediction that's off by 20 points hurts more than two off by 10.
        spread_rmse = np.sqrt(np.mean((spread_pred - target_spread) ** 2))
        home_mae = np.mean(np.abs(home_pred - target_home))
        away_mae = np.mean(np.abs(away_pred - target_away))

        # Total points MAE (home + away combined score)
        total_pred = home_pred + away_pred
        total_actual = target_home + target_away
        total_mae = np.mean(np.abs(total_pred - total_actual))

        # ---- Win probability metrics ----
        # Binary outcome: 1.0 if home team won, 0.0 if away team won
        home_won = (target_home > target_away).astype(float)
        # Simple accuracy: what % of games did we correctly pick the winner?
        # (favored = win_prob > 0.5)
        win_accuracy = np.mean((win_prob > 0.5) == home_won)

        # AUC: how well do our probabilities rank-order wins vs losses?
        # Perfect AUC (1.0) means every win gets a higher probability than every loss.
        win_auc = compute_auc(home_won, win_prob)

        # Brier score: mean squared error between probabilities and outcomes.
        # 0.0 = perfect, 0.25 = always predicting 50% (coin flip).
        brier = np.mean((win_prob - home_won) ** 2)

        # ---- Calibration metrics ----
        # ECE bins predictions (e.g., 0-10%, 10-20%, ..., 90-100%) and checks
        # whether the average predicted probability in each bin matches the
        # actual win rate. Low ECE = well-calibrated.
        ece, mce = compute_calibration_error(win_prob, home_won, n_bins=10)

        # ---- Coverage metrics ----
        # Do our confidence intervals actually contain the true outcome?
        # A well-calibrated model's 90% CI should cover ~90% of actual outcomes.
        spread_cov_90 = compute_coverage(target_spread, spread_pred, spread_std, 0.90)
        spread_cov_50 = compute_coverage(target_spread, spread_pred, spread_std, 0.50)
        home_cov_90 = compute_coverage(target_home, home_pred, home_std, 0.90)
        away_cov_90 = compute_coverage(target_away, away_pred, away_std, 0.90)

        # Sharpness: how wide are our confidence intervals on average?
        # z_90 = 1.645 is the z-score for a 90% CI (from the standard normal table).
        # The full interval width = 2 * z * sigma (extending z*sigma in each direction).
        # We want this to be as SMALL as possible while maintaining correct coverage.
        z_90 = 1.645
        spread_sharpness = np.mean(2 * z_90 * spread_std)

        return MetricResults(
            spread_mae=float(spread_mae),
            spread_rmse=float(spread_rmse),
            home_mae=float(home_mae),
            away_mae=float(away_mae),
            total_mae=float(total_mae),
            win_accuracy=float(win_accuracy),
            win_auc=float(win_auc),
            brier_score=float(brier),
            ece=float(ece),
            mce=float(mce),
            spread_coverage_90=float(spread_cov_90),
            spread_coverage_50=float(spread_cov_50),
            spread_sharpness=float(spread_sharpness),
            home_coverage_90=float(home_cov_90),
            away_coverage_90=float(away_cov_90),
            n_samples=n_samples,
        )


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC).

    HOW AUC WORKS:
        Imagine sorting all games by predicted win probability (highest first).
        If the model is good, actual wins should cluster at the top of this list.
        AUC measures this: it is the probability that a randomly chosen win
        is ranked higher than a randomly chosen loss.

        - AUC = 1.0: perfect ranking (all wins ranked above all losses)
        - AUC = 0.5: random (no better than coin flip)
        - AUC < 0.5: worse than random (model has the relationship backwards)

    Args:
        labels: Binary labels (1 = home win, 0 = home loss)
        scores: Predicted home win probabilities

    Returns:
        AUC score (0.0 to 1.0)
    """
    # Edge case: if all games had the same outcome (all wins or all losses),
    # AUC is undefined. Return 0.5 (random baseline) as a safe default.
    if len(np.unique(labels)) < 2:
        return 0.5

    # Sort all predictions from highest probability to lowest
    sorted_indices = np.argsort(-scores)
    labels_sorted = labels[sorted_indices]

    # Count total positives (wins) and negatives (losses)
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Build the ROC curve by walking through the sorted predictions:
    # TPR (True Positive Rate) = fraction of actual wins seen so far
    # FPR (False Positive Rate) = fraction of actual losses seen so far
    tpr = np.cumsum(labels_sorted) / n_pos
    fpr = np.cumsum(1 - labels_sorted) / n_neg

    # The ROC curve starts at (0, 0) — before we have seen any predictions
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # AUC = area under the ROC curve, computed via trapezoidal integration
    auc = np.trapezoid(tpr, fpr)

    return auc


def compute_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    HOW CALIBRATION WORKS:
        We group predictions into bins by confidence level:
            Bin 1: predictions between 0% and 10%
            Bin 2: predictions between 10% and 20%
            ...
            Bin 10: predictions between 90% and 100%

        For each bin, we compare:
            - Average predicted probability (confidence): e.g., 72%
            - Actual win rate in that bin (accuracy): e.g., 65%
            - Gap = |72% - 65%| = 7% (this bin is overconfident by 7%)

        ECE = weighted average of all bin gaps (lower is better, 0 = perfect)
        MCE = the largest single bin gap (catches the worst miscalibration)

    Args:
        probs: Predicted probabilities
        labels: Binary labels (1 = win, 0 = loss)
        n_bins: Number of bins for calibration (default 10)

    Returns:
        (ECE, MCE) tuple — both in [0, 1], lower is better
    """
    # Create 10 equally-spaced bins from 0 to 1
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    # Assign each prediction to a bin based on its probability value
    bin_indices = np.digitize(probs, bin_boundaries[1:-1])

    ece = 0.0
    mce = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) == 0:
            continue  # Skip empty bins (no predictions fell in this range)

        bin_probs = probs[mask]
        bin_labels = labels[mask]

        # Average predicted probability in this bin (the model's "confidence")
        avg_confidence = np.mean(bin_probs)
        # Actual win rate in this bin (what really happened)
        avg_accuracy = np.mean(bin_labels)

        # Calibration error for this bin: how far off is confidence from reality?
        bin_error = np.abs(avg_confidence - avg_accuracy)
        # Weight by the fraction of samples in this bin (larger bins matter more)
        bin_weight = np.sum(mask) / len(probs)

        ece += bin_weight * bin_error  # Weighted sum for ECE
        mce = max(mce, bin_error)      # Track the worst single bin

    return ece, mce


def compute_coverage(
    targets: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    confidence: float = 0.90,
) -> float:
    """
    Compute coverage of prediction intervals.

    HOW COVERAGE WORKS:
        The model predicts a mean and standard deviation for each game.
        From these, we can construct a confidence interval (CI):
            [mean - z*std, mean + z*std]

        For a 90% CI, z = 1.645 (from the standard normal distribution).
        If the model is well-calibrated, 90% of actual outcomes should fall
        within the 90% CI.

        Example:
            Model predicts spread = 5 +/- 8 (at 90% confidence).
            The 90% CI is [5 - 1.645*8, 5 + 1.645*8] = [-8.2, 18.2].
            If the actual spread is 10, it falls inside -> covered.
            If the actual spread is 25, it falls outside -> not covered.

        Ideal: coverage_90 ~ 0.90. If it is much higher (like 0.99), the model
        is underconfident (intervals are too wide). If much lower (like 0.70),
        the model is overconfident (intervals are too narrow).

    Args:
        targets: Actual values
        means: Predicted means
        stds: Predicted standard deviations
        confidence: Target confidence level (e.g., 0.90 for 90%)

    Returns:
        Actual coverage fraction (should be close to `confidence` if calibrated)
    """
    from scipy.stats import norm

    # Look up the z-score for the requested confidence level.
    # For 90% CI: z = 1.645. For 50% CI: z = 0.674. For 95% CI: z = 1.96.
    z = norm.ppf((1 + confidence) / 2)

    # Construct the confidence interval for each prediction
    lower = means - z * stds
    upper = means + z * stds

    # Check which actual outcomes fall within the interval
    in_interval = (targets >= lower) & (targets <= upper)
    coverage = np.mean(in_interval)

    return coverage


def compute_crps(
    targets: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS is a "proper scoring rule" that measures both calibration and sharpness
    in a single number. Unlike Brier score (which only works for binary outcomes),
    CRPS works for continuous predictions (like point spreads).

    Lower CRPS is better. It can be thought of as the "average distance" between
    the predicted distribution and the actual outcome, integrated across all
    possible thresholds. A sharp, well-calibrated model will have low CRPS.

    Args:
        targets: Actual values
        means: Predicted means
        stds: Predicted standard deviations

    Returns:
        Average CRPS (lower is better)
    """
    from scipy.stats import norm

    # Standardize: how many standard deviations is the target from the mean?
    z = (targets - means) / stds
    # Closed-form CRPS for Gaussian distributions (no need for numerical integration)
    crps = stds * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    return np.mean(crps)


def test_metrics():
    """Test metrics computation."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    n_samples = 100
    np.random.seed(42)

    # Simulated predictions (somewhat calibrated)
    spread_pred = np.random.randn(n_samples) * 5
    spread_std = np.abs(np.random.randn(n_samples)) * 3 + 2
    home_pred = np.random.randn(n_samples) * 10 + 110
    home_std = np.abs(np.random.randn(n_samples)) * 5 + 5
    away_pred = np.random.randn(n_samples) * 10 + 108
    away_std = np.abs(np.random.randn(n_samples)) * 5 + 5

    # Generate targets with noise
    target_spread = spread_pred + np.random.randn(n_samples) * 8
    target_home = home_pred + np.random.randn(n_samples) * 12
    target_away = away_pred + np.random.randn(n_samples) * 12

    # Win probability (correlated with spread)
    win_prob = 1 / (1 + np.exp(-spread_pred / 5))
    win_prob = np.clip(win_prob + np.random.randn(n_samples) * 0.1, 0.01, 0.99)

    # Convert to tensors
    predictions = GamePrediction(
        spread_mean=torch.tensor(spread_pred, dtype=torch.float32),
        spread_std=torch.tensor(spread_std, dtype=torch.float32),
        home_score_mean=torch.tensor(home_pred, dtype=torch.float32),
        home_score_std=torch.tensor(home_std, dtype=torch.float32),
        away_score_mean=torch.tensor(away_pred, dtype=torch.float32),
        away_score_std=torch.tensor(away_std, dtype=torch.float32),
        home_win_prob=torch.tensor(win_prob, dtype=torch.float32),
    )

    target_home_t = torch.tensor(target_home, dtype=torch.float32)
    target_away_t = torch.tensor(target_away, dtype=torch.float32)

    # Compute metrics
    print("Testing MetricsCalculator...")
    calculator = MetricsCalculator()

    # Simulate batched updates
    batch_size = 20
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_pred = GamePrediction(
            spread_mean=predictions.spread_mean[i:end],
            spread_std=predictions.spread_std[i:end],
            home_score_mean=predictions.home_score_mean[i:end],
            home_score_std=predictions.home_score_std[i:end],
            away_score_mean=predictions.away_score_mean[i:end],
            away_score_std=predictions.away_score_std[i:end],
            home_win_prob=predictions.home_win_prob[i:end],
        )
        calculator.update(batch_pred, target_home_t[i:end], target_away_t[i:end])

    results = calculator.compute()

    print(f"\nResults ({results.n_samples} samples):")
    print(f"  Spread MAE: {results.spread_mae:.2f}")
    print(f"  Spread RMSE: {results.spread_rmse:.2f}")
    print(f"  Home MAE: {results.home_mae:.2f}")
    print(f"  Away MAE: {results.away_mae:.2f}")
    print(f"  Total MAE: {results.total_mae:.2f}")
    print(f"  Win Accuracy: {results.win_accuracy:.3f}")
    print(f"  Win AUC: {results.win_auc:.3f}")
    print(f"  Brier Score: {results.brier_score:.4f}")
    print(f"  ECE: {results.ece:.4f}")
    print(f"  MCE: {results.mce:.4f}")
    print(f"  Spread Coverage 90%: {results.spread_coverage_90:.3f}")
    print(f"  Spread Coverage 50%: {results.spread_coverage_50:.3f}")
    print(f"  Spread Sharpness: {results.spread_sharpness:.2f}")

    # Test individual functions
    print("\nTesting individual functions...")

    # AUC
    labels = (target_spread > 0).astype(float)
    auc = compute_auc(labels, win_prob)
    print(f"  AUC: {auc:.3f}")

    # Calibration
    home_won = (target_home > target_away).astype(float)
    ece, mce = compute_calibration_error(win_prob, home_won)
    print(f"  ECE: {ece:.4f}, MCE: {mce:.4f}")

    # Coverage
    cov = compute_coverage(target_spread, spread_pred, spread_std, 0.90)
    print(f"  90% Coverage: {cov:.3f}")

    # CRPS
    crps = compute_crps(target_spread, spread_pred, spread_std)
    print(f"  CRPS: {crps:.3f}")

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_metrics())
