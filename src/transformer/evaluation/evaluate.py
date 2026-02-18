"""
Test Set Evaluation for NBA Transformer Model.

=============================================================================
OVERVIEW: WHAT IS TEST SET EVALUATION?
=============================================================================

In machine learning, we split our data into three sets:
1. TRAINING SET: Data the model learns from (like studying textbook examples)
2. VALIDATION SET: Data used to tune hyperparameters and prevent overfitting
   (like practice quizzes to check understanding)
3. TEST SET: Data the model has NEVER seen during training or tuning
   (like the final exam - used only once at the very end)

WHY IS THE TEST SET SPECIAL?
- The test set is our "hold-out" data - completely untouched during development
- It gives us an unbiased estimate of how the model will perform on new games
- If we peeked at the test set during development, we'd be "cheating" and our
  performance estimates would be overly optimistic
- Think of it like this: if a student sees the final exam questions beforehand,
  their score won't reflect their true understanding

FOR NBA PREDICTIONS SPECIFICALLY:
- Training set: Games from 2019-2022 seasons
- Validation set: Games from early 2023-2024
- Test set: Games from late 2023-2024 (simulating future predictions)

This module provides comprehensive evaluation capabilities:
- Point prediction accuracy (how close are our score predictions?)
- Probability calibration (when we say 70% chance, does home team win 70% of time?)
- Uncertainty quality (are our confidence intervals reliable?)
- Comparison to baseline models (are we actually better than simpler approaches?)

Usage:
    from src.transformer.evaluation import Evaluator

    # Create evaluator with trained model and test data
    evaluator = Evaluator(model, test_loader, device="cuda")

    # Run evaluation on all test games
    results = evaluator.evaluate()

    # Print human-readable summary with all metrics
    evaluator.print_summary()
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transformer.models import Phase1Model, GamePrediction
from src.transformer.training.metrics import MetricsCalculator, MetricResults


logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION RESULTS DATA CLASS
# =============================================================================

@dataclass
class EvaluationResults:
    """
    Complete evaluation results container with all metrics and metadata.

    This dataclass bundles together everything we want to record about a
    model's performance on the test set. Using a dataclass gives us:
    - Clean, organized storage of related values
    - Automatic __init__, __repr__, and comparison methods
    - Easy serialization to JSON for saving results

    Attributes:
        metrics: The MetricResults object containing all computed metrics
            (MAE, RMSE, calibration scores, coverage, etc.)

        model_name: Human-readable name for this model (e.g., "transformer_v2")

        checkpoint_path: File path to the saved model weights that were evaluated

        test_seasons: List of NBA seasons in the test set (e.g., ["2023-2024"])

        n_games: Total number of games evaluated

        baseline_mae: MAE of the simple baseline (historical average prediction).
            This is our "sanity check" - any reasonable model should beat this.

        xgboost_mae: MAE of our best traditional ML model (XGBoost).
            Value of 10.1 comes from our previous best model.
            This is the bar we're trying to beat with the transformer.

        improvement_vs_baseline: Percentage improvement over simple baseline.
            Positive = our model is better. Formula: (baseline - ours) / baseline * 100

        improvement_vs_xgboost: Percentage improvement over XGBoost.
            This is the key metric - positive means transformer beats XGBoost!
    """

    # Core metrics - all the numbers we compute
    metrics: MetricResults

    # Metadata - information about this evaluation run
    model_name: str = ""
    checkpoint_path: str = ""
    test_seasons: list[str] = None  # Will be initialized in __post_init__
    n_games: int = 0

    # Comparison to baselines - how we stack up against known alternatives
    # These are Optional because we might run evaluation without baselines
    baseline_mae: Optional[float] = None
    xgboost_mae: Optional[float] = None
    improvement_vs_baseline: Optional[float] = None
    improvement_vs_xgboost: Optional[float] = None

    # Final score metrics (including OT) — populated when OT games exist in the
    # test set. Lets us compare regulation-based predictions against actual final
    # outcomes for real-world evaluation (since bets settle on final scores).
    final_score_metrics: Optional[MetricResults] = None

    def __post_init__(self):
        """
        Called automatically after __init__ to handle mutable defaults.

        Python quirk: You can't use mutable defaults (like []) in dataclass
        fields because they'd be shared across all instances. We use None
        as the default and convert to an empty list here.
        """
        if self.test_seasons is None:
            self.test_seasons = []

    def to_dict(self) -> dict:
        """
        Convert results to a plain dictionary for JSON serialization.

        JSON can only store basic Python types (str, int, float, list, dict),
        not custom objects. This method flattens our dataclass into a dict
        that can be saved to disk and loaded later.

        Returns:
            Dictionary containing all evaluation data in JSON-compatible format
        """
        result = {
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "test_seasons": self.test_seasons,
            "n_games": self.n_games,
            # asdict() recursively converts nested dataclasses to dicts
            "metrics": asdict(self.metrics),
        }
        # Only include baseline comparisons if they were computed
        if self.baseline_mae is not None:
            result["baseline_mae"] = self.baseline_mae
            result["improvement_vs_baseline"] = self.improvement_vs_baseline
        if self.xgboost_mae is not None:
            result["xgboost_mae"] = self.xgboost_mae
            result["improvement_vs_xgboost"] = self.improvement_vs_xgboost
        if self.final_score_metrics is not None:
            result["final_score_metrics"] = asdict(self.final_score_metrics)
        return result

    def save(self, path: str):
        """
        Save evaluation results to a JSON file.

        This is useful for:
        - Keeping a record of how each model performed
        - Comparing results across experiments
        - Loading results later for analysis without re-running evaluation

        Args:
            path: File path where results will be saved (e.g., "results/eval_v2.json")
        """
        with open(path, "w") as f:
            # indent=2 makes the JSON human-readable with nice formatting
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {path}")


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================

class Evaluator:
    """
    Comprehensive model evaluator for test set evaluation.

    This class orchestrates the entire evaluation process:
    1. Takes a trained model and test data loader
    2. Runs the model on all test games (without gradient computation)
    3. Collects predictions and computes comprehensive metrics
    4. Compares performance against known baselines
    5. Provides formatted output for analysis

    WORKFLOW OVERVIEW:
    ==================

    Step 1: Initialization
        - Load the model and move to GPU/CPU
        - Set model to evaluation mode (disables dropout, etc.)
        - Store references to test data

    Step 2: Evaluation (evaluate() method)
        - Loop through all test games in batches
        - For each batch: run forward pass, collect predictions
        - Accumulate metrics across all batches
        - Compare against baseline MAE values

    Step 3: Analysis (print_summary() method)
        - Display formatted results for human review
        - Highlight whether we beat baseline models

    WHY COMPARE TO BASELINES?
    =========================

    Baselines give us context for our model's performance:

    1. Simple Baseline (MAE = 11.2):
       - Just predicts historical team averages
       - Any reasonable model should beat this
       - If we don't, something is very wrong

    2. XGBoost Baseline (MAE = 10.1):
       - Our previous best traditional ML model
       - Uses hand-crafted features (team stats, player stats, etc.)
       - This is what we're trying to beat with the transformer
       - Beating XGBoost validates that the added complexity is worth it

    MAE (Mean Absolute Error) Interpretation:
    - MAE = 10.1 means on average, our score predictions are off by 10.1 points
    - For NBA games (where scores are ~100-120), this is reasonable but not amazing
    - Lower is better

    Example Usage:
        >>> evaluator = Evaluator(model, test_loader, device="cuda")
        >>> results = evaluator.evaluate()
        >>> print(f"Our MAE: {results.metrics.spread_mae:.2f}")
        >>> evaluator.print_summary()
    """

    # ==========================================================================
    # BASELINE VALUES - These are our comparison targets
    # ==========================================================================

    # Simple historical average baseline: predicts each team will score their
    # season average. MAE of 11.2 points per game. This is our "floor" - we
    # should easily beat this with any reasonable model.
    BASELINE_MAE = 11.2

    # XGBoost gradient boosting model: Our previous best using traditional
    # machine learning with hand-crafted features. MAE of 10.1 points.
    # This is our "target" - beating this justifies the transformer approach.
    # The 10.1 value comes from our production XGBoost model's test performance.
    XGBOOST_MAE = 10.1

    def __init__(
        self,
        model: Phase1Model,
        test_loader: DataLoader,
        device: str = "cpu",
        model_name: str = "transformer",
        checkpoint_path: str = "",
    ):
        """
        Initialize the evaluator with a model and test data.

        Args:
            model: The trained Phase1Model transformer to evaluate.
                Should already have weights loaded from a checkpoint.

            test_loader: PyTorch DataLoader containing test set games.
                This should be created from games the model has NEVER seen.
                Typically uses shuffle=False for reproducibility.

            device: Where to run evaluation ("cpu" or "cuda").
                GPU ("cuda") is much faster for large test sets.
                CPU works fine for small-scale testing.

            model_name: Human-readable name for this model version.
                Used in output and saved results for identification.

            checkpoint_path: Path to the model checkpoint file.
                Stored for reference but not used directly here.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        # Move model to the specified device (GPU or CPU)
        # This copies all model weights to the device memory
        self.model.to(self.device)

        # Set model to evaluation mode
        # This is CRITICAL for correct evaluation:
        # - Disables dropout (which randomly zeroes neurons during training)
        # - LayerNorm is unaffected by eval mode (no running statistics)
        # Forgetting this is a common bug that causes inconsistent results!
        self.model.eval()

        # Will store results after evaluate() is called
        self.results: Optional[EvaluationResults] = None

    @torch.no_grad()  # Decorator that disables gradient computation for efficiency
    def evaluate(self, show_progress: bool = True) -> EvaluationResults:
        """
        Run full evaluation on the test set.

        This method performs the core evaluation loop:
        1. Iterates through all batches in the test set
        2. Runs model inference on each batch
        3. Accumulates metrics using MetricsCalculator
        4. Computes final aggregate metrics
        5. Calculates improvement vs baselines

        The @torch.no_grad() decorator is important:
        - During training, PyTorch tracks gradients for backpropagation
        - Gradient tracking uses significant memory and compute
        - For evaluation, we only need forward passes (no learning)
        - Disabling gradients makes evaluation ~2x faster and uses less memory

        Args:
            show_progress: If True, displays a progress bar showing evaluation
                progress. Set to False for quiet operation (e.g., in scripts).

        Returns:
            EvaluationResults containing all computed metrics, comparisons,
            and metadata about this evaluation run.

        Example:
            >>> results = evaluator.evaluate(show_progress=True)
            >>> print(f"Spread MAE: {results.metrics.spread_mae:.2f}")
            >>> print(f"Win Accuracy: {results.metrics.win_accuracy:.3f}")
        """
        logger.info("Starting test set evaluation...")

        # MetricsCalculator accumulates predictions across batches and computes
        # aggregate metrics at the end. This is more numerically stable than
        # computing per-batch and averaging.
        metrics_calc = MetricsCalculator(device=self.device)

        # Second calculator for final scores (including OT) — only used when
        # OT filtering is active and the batch contains final_home_scores.
        final_metrics_calc = None
        has_ot_games = False

        # Optionally wrap the data loader with tqdm for a progress bar
        iterator = self.test_loader
        if show_progress:
            # leave=False means the progress bar disappears after completion
            iterator = tqdm(iterator, desc="Evaluating", leave=False)

        # =======================================================================
        # MAIN EVALUATION LOOP
        # =======================================================================
        for batch in iterator:
            # Skip empty/corrupted batches (shouldn't happen in normal use)
            if batch is None:
                continue

            # Move all batch tensors to the correct device (GPU/CPU)
            # This is necessary because data loaders produce CPU tensors by default
            batch = self._move_to_device(batch)

            # Forward pass: run the model on this batch of games
            # The model returns a GamePrediction object containing:
            # - home_score_mean, home_score_std: Predicted home team score distribution
            # - away_score_mean, away_score_std: Predicted away team score distribution
            # - spread_mean, spread_std: Predicted point spread (home - away)
            # - home_win_prob: Probability that home team wins
            predictions = self.model(
                home_history=batch["home_history"],      # Recent games for home team
                away_history=batch["away_history"],      # Recent games for away team
            )

            # Update running metrics with this batch's predictions and targets
            # MetricsCalculator internally tracks sums and counts for later aggregation
            metrics_calc.update(
                predictions,
                batch["target_home_scores"],  # Actual home team scores
                batch["target_away_scores"],  # Actual away team scores
            )

            # If the batch contains final scores (from OT filtering), track
            # metrics against those too. This lets us see how regulation-trained
            # predictions perform against actual final outcomes.
            if "final_home_scores" in batch and "final_away_scores" in batch:
                # Check if any OT games exist (otherwise final == regulation)
                if any(batch.get("is_overtime", [])):
                    has_ot_games = True
                if final_metrics_calc is None:
                    final_metrics_calc = MetricsCalculator(device=self.device)
                final_metrics_calc.update(
                    predictions,
                    batch["final_home_scores"],
                    batch["final_away_scores"],
                )

        # =======================================================================
        # COMPUTE FINAL METRICS
        # =======================================================================

        # Convert accumulated statistics into final metric values
        metrics = metrics_calc.compute()

        # Get test seasons from dataset if available (for metadata)
        test_seasons = []
        if hasattr(self.test_loader.dataset, "seasons"):
            test_seasons = self.test_loader.dataset.seasons

        # Compute final score metrics if OT games were present
        final_score_metrics = None
        if final_metrics_calc is not None and has_ot_games:
            final_score_metrics = final_metrics_calc.compute()

        # Create results object with all computed metrics
        self.results = EvaluationResults(
            metrics=metrics,
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            test_seasons=test_seasons,
            n_games=metrics.n_samples,
            baseline_mae=self.BASELINE_MAE,
            xgboost_mae=self.XGBOOST_MAE,
            final_score_metrics=final_score_metrics,
        )

        # =======================================================================
        # COMPUTE IMPROVEMENT VS BASELINES
        # =======================================================================

        # Average the home and away MAE for a single comparison number
        # (Some prefer to use spread MAE directly; either approach is valid)
        avg_mae = (metrics.home_mae + metrics.away_mae) / 2

        # Calculate percentage improvement over simple baseline
        # Formula: (baseline - ours) / baseline * 100
        # Positive value = we're better, negative = we're worse
        # Example: baseline=11.2, ours=9.5 -> (11.2-9.5)/11.2*100 = 15.2% improvement
        self.results.improvement_vs_baseline = (
            (self.BASELINE_MAE - avg_mae) / self.BASELINE_MAE * 100
        )

        # Calculate percentage improvement over XGBoost
        # This is the key metric: positive means we beat traditional ML!
        self.results.improvement_vs_xgboost = (
            (self.XGBOOST_MAE - avg_mae) / self.XGBOOST_MAE * 100
        )

        logger.info(f"Evaluation complete: {metrics.n_samples} games")
        return self.results

    def _move_to_device(self, batch: dict) -> dict:
        """
        Move all tensors in a batch to the evaluation device (GPU/CPU).

        This method handles the complexity of batch structures:
        - Simple tensors: directly move to device
        - Nested dicts: recursively process inner dicts
        - Lists of tensors: move each tensor in the list
        - Non-tensor values: pass through unchanged

        Why is this necessary?
        - PyTorch DataLoaders produce CPU tensors by default
        - Our model lives on a specific device (e.g., GPU)
        - Tensors on different devices can't interact directly
        - We must move data to the model's device before processing

        Args:
            batch: Dictionary containing batch data with potentially nested
                tensors, lists, and sub-dictionaries.

        Returns:
            New dictionary with all tensors moved to self.device
        """
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Simple case: move tensor to device
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Recursive case: handle nested dictionaries
                moved[key] = self._move_to_device(value)
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    # List of tensors: move each one
                    moved[key] = [v.to(self.device) for v in value]
                else:
                    # List of non-tensors: pass through
                    moved[key] = value
            else:
                # Non-tensor values (strings, ints, etc.): pass through
                moved[key] = value
        return moved

    def print_summary(self, include_comparison: bool = True):
        """
        Print a formatted summary of evaluation results.

        This method outputs a human-readable report including:
        - Basic info (model name, number of test games, seasons)
        - Point prediction metrics (MAE, RMSE for scores and spread)
        - Win probability metrics (accuracy, AUC, Brier score)
        - Calibration metrics (ECE, MCE)
        - Uncertainty/coverage metrics
        - Comparison to baseline models

        UNDERSTANDING THE OUTPUT:
        =========================

        POINT PREDICTIONS Section:
        - Spread MAE: Average absolute error in point spread prediction
          (e.g., 8.5 means we're off by 8.5 points on average for the margin)
        - Spread RMSE: Root mean squared error (penalizes large errors more)
        - Home/Away MAE: Error in predicting individual team scores
        - Total MAE: Error in predicting combined game total

        WIN PROBABILITY Section:
        - Accuracy: Fraction of games where we correctly predicted the winner
          (e.g., 0.65 means we got 65% of games right)
        - AUC: Area under ROC curve, measures ranking quality (1.0 is perfect)
        - Brier Score: Average squared probability error (lower is better)
          (e.g., 0.20 is good, 0.25 is mediocre, 0.30+ is poor)

        CALIBRATION Section:
        - ECE (Expected Calibration Error): Average gap between predicted
          probability and actual win rate across probability bins
          (e.g., 0.02 means when we say 70%, teams actually win 68-72%)
        - MCE (Maximum Calibration Error): Worst calibration in any bin

        UNCERTAINTY Section:
        - Coverage 90%: Fraction of actual spreads within 90% prediction interval
          (Target: 0.900 - if model is well-calibrated, this should match)
        - Coverage 50%: Same for 50% interval (target: 0.500)
        - Sharpness: Width of prediction intervals (narrower = more confident)

        Args:
            include_comparison: If True, includes baseline comparison at the end
        """
        if self.results is None:
            logger.warning("No results to print. Run evaluate() first.")
            return

        m = self.results.metrics  # Shorthand for cleaner code below

        # =====================================================================
        # HEADER
        # =====================================================================
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS: {self.model_name}")
        print("=" * 60)
        print(f"Test Games: {m.n_samples}")
        print(f"Seasons: {', '.join(self.results.test_seasons)}")
        print("-" * 60)

        # =====================================================================
        # POINT PREDICTION METRICS
        # These measure how close our score predictions are to actual scores
        # =====================================================================
        print("\nPOINT PREDICTIONS:")
        print(f"  Spread MAE:  {m.spread_mae:.2f}")   # Key metric for betting
        print(f"  Spread RMSE: {m.spread_rmse:.2f}")  # Penalizes outliers more
        print(f"  Home MAE:    {m.home_mae:.2f}")     # Home team score accuracy
        print(f"  Away MAE:    {m.away_mae:.2f}")     # Away team score accuracy
        print(f"  Total MAE:   {m.total_mae:.2f}")    # Combined total accuracy

        # =====================================================================
        # WIN PROBABILITY METRICS
        # These measure how well our probability predictions work
        # =====================================================================
        print("\nWIN PROBABILITY:")
        print(f"  Accuracy:    {m.win_accuracy:.3f}")  # Simple % correct
        print(f"  AUC:         {m.win_auc:.3f}")       # Ranking quality (0.5 = random)
        print(f"  Brier Score: {m.brier_score:.4f}")   # Probability accuracy

        # =====================================================================
        # CALIBRATION METRICS
        # These measure if our probabilities mean what they say
        # Example: When we predict 70% home win, home should win ~70% of time
        # =====================================================================
        print("\nCALIBRATION:")
        print(f"  ECE:         {m.ece:.4f}")  # Average calibration error
        print(f"  MCE:         {m.mce:.4f}")  # Worst-case calibration error

        # =====================================================================
        # UNCERTAINTY METRICS
        # These measure if our confidence intervals are reliable
        # "Coverage" = fraction of true values within predicted intervals
        # =====================================================================
        print("\nUNCERTAINTY (Coverage):")
        print(f"  Spread 90%:  {m.spread_coverage_90:.3f} (target: 0.900)")
        print(f"  Spread 50%:  {m.spread_coverage_50:.3f} (target: 0.500)")
        print(f"  Home 90%:    {m.home_coverage_90:.3f}")
        print(f"  Away 90%:    {m.away_coverage_90:.3f}")
        print(f"  Sharpness:   {m.spread_sharpness:.2f} points")

        # =====================================================================
        # BASELINE COMPARISON
        # The most important question: is our model actually better?
        # =====================================================================
        if include_comparison:
            avg_mae = (m.home_mae + m.away_mae) / 2
            print("\nCOMPARISON TO BASELINES:")
            print(f"  Avg Score MAE:   {avg_mae:.2f}")
            print(f"  vs Baseline ({self.BASELINE_MAE:.1f}): "
                  f"{self.results.improvement_vs_baseline:+.1f}%")
            print(f"  vs XGBoost ({self.XGBOOST_MAE:.1f}):  "
                  f"{self.results.improvement_vs_xgboost:+.1f}%")

            # Highlight the key result
            if avg_mae < self.XGBOOST_MAE:
                print("\n  *** BEATS XGBOOST BASELINE! ***")
            elif avg_mae < self.BASELINE_MAE:
                print("\n  Beats simple baseline, but not XGBoost yet.")

        # =====================================================================
        # FINAL SCORE METRICS (when OT filtering is active)
        # These show how regulation-trained predictions compare to actual final
        # outcomes (including OT). Since bets settle on final scores, this is
        # the real-world performance measure.
        # =====================================================================
        if self.results.final_score_metrics is not None:
            fm = self.results.final_score_metrics
            print("-" * 60)
            print("\nFINAL SCORE METRICS (including OT):")
            print("  (Model trained on regulation scores, evaluated vs final scores)")
            print(f"  Spread MAE:  {fm.spread_mae:.2f}")
            print(f"  Spread RMSE: {fm.spread_rmse:.2f}")
            print(f"  Home MAE:    {fm.home_mae:.2f}")
            print(f"  Away MAE:    {fm.away_mae:.2f}")
            print(f"  Total MAE:   {fm.total_mae:.2f}")
            print(f"  Win Accuracy: {fm.win_accuracy:.3f}")
            print(f"  Win AUC:     {fm.win_auc:.3f}")
            print(f"  Brier Score: {fm.brier_score:.4f}")

            # Show the delta between regulation and final score metrics
            reg_spread_mae = m.spread_mae
            final_spread_mae = fm.spread_mae
            delta = final_spread_mae - reg_spread_mae
            print(f"\n  Spread MAE delta (final - reg): {delta:+.2f}")

        print("=" * 60 + "\n")

    def get_predictions_df(self) -> "pd.DataFrame":
        """
        Get all predictions as a pandas DataFrame for detailed analysis.

        This method is useful when you want to:
        - Create custom visualizations beyond what Visualizer provides
        - Analyze specific subsets of games (e.g., playoffs, close games)
        - Export predictions for use in other tools
        - Debug model behavior on specific examples

        The returned DataFrame includes:
        - Actual scores and spread
        - Predicted scores and spread (with uncertainties)
        - Win probability predictions
        - Derived columns like spread_error, home_error, away_error

        Note: This re-runs the model on all test data to collect predictions.
        The evaluate() method doesn't store individual predictions (only metrics)
        to save memory for large test sets.

        Returns:
            pandas DataFrame with one row per game and columns for predictions,
            actuals, and computed errors.

        Raises:
            ValueError: If evaluate() hasn't been run yet.
        """
        import pandas as pd

        if self.results is None:
            raise ValueError("No results. Run evaluate() first.")

        # We need to re-run inference to collect individual predictions
        # (evaluate() only computed aggregate metrics)
        predictions_list = []

        # Ensure model is in eval mode
        self.model.eval()

        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(self.test_loader, desc="Collecting predictions", leave=False):
                if batch is None:
                    continue

                batch = self._move_to_device(batch)
                predictions = self.model(
                    home_history=batch["home_history"],
                    away_history=batch["away_history"],
                )

                # Extract predictions for each game in the batch
                batch_size = len(batch["target_home_scores"])
                for i in range(batch_size):
                    predictions_list.append({
                        # Actual game outcomes
                        "home_score_actual": batch["target_home_scores"][i].item(),
                        "away_score_actual": batch["target_away_scores"][i].item(),
                        # Model predictions (mean and uncertainty)
                        "home_score_pred": predictions.home_score_mean[i].item(),
                        "home_score_std": predictions.home_score_std[i].item(),
                        "away_score_pred": predictions.away_score_mean[i].item(),
                        "away_score_std": predictions.away_score_std[i].item(),
                        "spread_pred": predictions.spread_mean[i].item(),
                        "spread_std": predictions.spread_std[i].item(),
                        "home_win_prob": predictions.home_win_prob[i].item(),
                    })

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(predictions_list)

        # Add derived columns for convenience
        df["spread_actual"] = df["home_score_actual"] - df["away_score_actual"]
        df["home_won"] = (df["home_score_actual"] > df["away_score_actual"]).astype(int)
        df["spread_error"] = df["spread_pred"] - df["spread_actual"]
        df["home_error"] = df["home_score_pred"] - df["home_score_actual"]
        df["away_error"] = df["away_score_pred"] - df["away_score_actual"]

        return df


# =============================================================================
# CONVENIENCE FUNCTION FOR LOADING AND EVALUATING
# =============================================================================

def load_and_evaluate(
    checkpoint_path: str,
    test_loader: DataLoader,
    tokenizer: "PBPTokenizer",
    device: str = "cuda",
    save_results: Optional[str] = None,
) -> EvaluationResults:
    """
    Convenience function to load a checkpoint and run evaluation in one step.

    This combines the common workflow of:
    1. Loading a saved model checkpoint
    2. Reconstructing the model architecture from saved config
    3. Loading the trained weights
    4. Creating an Evaluator and running evaluation
    5. Optionally saving results to disk

    Args:
        checkpoint_path: Path to saved model checkpoint (.pt file).
            The checkpoint should contain:
            - "model_state_dict": The trained model weights
            - "config" or "model_config": Model architecture parameters

        test_loader: DataLoader for the test set.
            Should contain games not seen during training.

        tokenizer: PBPTokenizer instance with vocabulary.
            Needed to set vocab_sizes in the model config.

        device: Device to use for evaluation ("cuda" or "cpu").
            GPU recommended for faster evaluation.

        save_results: Optional path to save results JSON.
            If provided, results will be saved after evaluation.

    Returns:
        EvaluationResults with all metrics and comparisons.

    Example:
        >>> from src.transformer.tokenizer import PBPTokenizer
        >>> tokenizer = PBPTokenizer.load("tokenizer.json")
        >>> results = load_and_evaluate(
        ...     "checkpoints/best.pt",
        ...     test_loader,
        ...     tokenizer,
        ...     save_results="results/eval.json"
        ... )
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint - map_location handles GPU->CPU loading if needed
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint
    # We support two checkpoint formats for backwards compatibility:
    from src.transformer.training.config import ModelConfig
    from dataclasses import fields as dc_fields

    # Try new format first (full config with nested structure)
    config = checkpoint.get("config")
    if config is not None and isinstance(config, dict):
        # New format: config.model contains model params
        model_config_dict = config.get("model", {})
        training_config = config.get("training", {})
        model_name = training_config.get("experiment_name", "transformer")
    else:
        # Old format: model_config stored directly
        model_config_dict = checkpoint.get("model_config", {})
        model_name = checkpoint.get("experiment_name", "transformer")

    if not model_config_dict:
        raise ValueError("Checkpoint missing model configuration")

    # Add vocabulary sizes from tokenizer (not stored in checkpoint)
    model_config_dict["vocab_sizes"] = tokenizer.vocab_sizes

    # Filter out any unexpected keys (e.g., from older checkpoint formats)
    valid_fields = {f.name for f in dc_fields(ModelConfig)}
    model_config_dict = {k: v for k, v in model_config_dict.items() if k in valid_fields}

    model_config = ModelConfig(**model_config_dict)

    # Create model and load trained weights
    model = Phase1Model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create evaluator and run evaluation
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )

    results = evaluator.evaluate()
    evaluator.print_summary()

    # Optionally save results to disk
    if save_results:
        results.save(save_results)

    return results


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_evaluator():
    """
    Test the Evaluator with mock data.

    This function creates synthetic data and runs the full evaluation pipeline
    to verify everything works correctly. It's useful for:
    - Quick sanity checks during development
    - Continuous integration testing
    - Understanding how the Evaluator works

    The mock data simulates a simple model that predicts random scores.
    We don't expect good metrics - we just want to verify the code runs.

    Returns:
        0 on success (standard Unix convention for exit codes)
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Evaluator...")

    # Create mock model and data
    from src.transformer.training.config import ModelConfig
    from src.transformer.models import Phase1Model

    # Minimal vocabulary configuration for testing
    # Keys must match what EventEmbedding expects
    vocab_sizes = {
        "action_type": 50,
        "sub_type": 30,
        "period": 10,
        "clock_bucket": 50,
        "team_indicator": 35,
        "score_diff_bucket": 100,
        "player": 1000,
        "shot_result": 5,
    }

    # Small model config for fast testing
    config = ModelConfig(
        vocab_sizes=vocab_sizes,
        hidden_dim=64,  # Small for speed
        event_encoder_layers=1,
        temporal_attention_layers=1,
    )

    model = Phase1Model(config)

    # Create mock dataset that mirrors the structure produced by
    # NBAGameDataset + collate_games. Phase1Model.forward() expects:
    #   home_history/away_history: dict of tensors (batch, n_games, n_plays)
    #   target_home_scores/target_away_scores: (batch,)
    n_games = 3
    n_plays = 50
    batch_size = 4
    n_batches = 5
    history_keys = [
        "action_type_ids", "sub_type_ids", "periods", "clock_buckets",
        "team_indicators", "score_diff_buckets", "player_ids", "shot_results",
        "shot_distance_buckets", "shot_modifier_ids",
    ]

    class MockDataset:
        def __init__(self, n_samples=20):
            self.n_samples = n_samples
            self.seasons = ["2024-25"]

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            # Return a pre-collated batch-shaped sample (collate is identity)
            return idx

    def mock_collate(indices):
        """Build a batch matching the real collate_games output structure."""
        bs = len(indices)
        history = {k: torch.randint(0, 30, (bs, n_games, n_plays)) for k in history_keys}
        history["game_lengths"] = torch.full((bs, n_games), n_plays, dtype=torch.long)
        return {
            "home_history": history,
            "away_history": {k: v.clone() for k, v in history.items()},
            "home_roster": torch.randint(0, 1000, (bs, 10)),
            "away_roster": torch.randint(0, 1000, (bs, 10)),
            "target_home_scores": torch.normal(110.0, 10.0, (bs,)),
            "target_away_scores": torch.normal(105.0, 10.0, (bs,)),
        }

    test_loader = DataLoader(
        MockDataset(n_samples=20),
        batch_size=4,
        collate_fn=mock_collate,
    )

    # Run evaluation
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device="cpu",
        model_name="test_model",
    )

    results = evaluator.evaluate()
    evaluator.print_summary()

    print("\nTest passed!")
    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(test_evaluator())
