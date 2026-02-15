"""
Visualization Tools for NBA Transformer Model.

=============================================================================
OVERVIEW: WHY VISUALIZATION MATTERS
=============================================================================

Numbers alone don't tell the whole story. A model might report:
    "Spread MAE: 8.5, Win Accuracy: 65%"

But visualization reveals:
- Are errors concentrated in certain situations?
- Are probability predictions actually calibrated?
- Is uncertainty meaningful (high when model should be unsure)?
- Are there systematic biases (always predicting home team too high)?

This module provides publication-quality diagnostic plots:

1. CALIBRATION DIAGRAM (Reliability Curve)
   - Shows if probability predictions mean what they claim
   - "When we say 70% home win, do they actually win 70%?"

2. COVERAGE PLOTS
   - Shows if prediction intervals are reliable
   - "Does our 90% confidence interval contain the true value 90% of the time?"

3. ERROR DISTRIBUTIONS
   - Shows the shape and bias of prediction errors
   - "Are we systematically over/under-predicting?"

4. UNCERTAINTY ANALYSIS
   - Shows if model uncertainty correlates with actual error
   - "Does the model know when it doesn't know?"

Usage:
    from src.transformer.evaluation import Visualizer

    # Create visualizer with predictions DataFrame
    viz = Visualizer(predictions_df, output_dir="plots/")

    # Generate individual plots
    viz.plot_calibration()
    viz.plot_error_distribution()
    viz.plot_coverage()
    viz.plot_uncertainty_analysis()

    # Or generate all plots at once
    viz.plot_all()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# MATPLOTLIB AVAILABILITY CHECK
# =============================================================================

# matplotlib is a large dependency that might not be installed in all environments
# (e.g., minimal Docker containers, production servers). We check for availability
# and gracefully disable visualization if it's missing.
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Visualization disabled.")


# =============================================================================
# MAIN VISUALIZER CLASS
# =============================================================================

class Visualizer:
    """
    Visualization toolkit for analyzing transformer model predictions.

    This class takes a DataFrame of predictions (from Evaluator.get_predictions_df())
    and generates various diagnostic plots to understand model behavior.

    WHY THESE SPECIFIC VISUALIZATIONS?
    ===================================

    1. CALIBRATION: Critical for decision-making. If you bet on games where the
       model says 70% home win, you expect to win ~70% of those bets. Poor
       calibration means the probabilities can't be trusted at face value.

    2. COVERAGE: Validates uncertainty estimates. A model that says "+/- 8 points
       90% of the time" should actually be right 90% of the time. Overconfident
       models (narrow intervals, low coverage) are dangerous.

    3. ERROR DISTRIBUTION: Reveals bias and outliers. A symmetric distribution
       centered on 0 is ideal. Skew or heavy tails indicate problems.

    4. UNCERTAINTY ANALYSIS: Tests if the model "knows what it doesn't know".
       High-uncertainty predictions should have larger errors. If not, the
       uncertainty is meaningless.

    PLOT DESIGN PRINCIPLES:
    =======================

    - Clear labels and titles (you should understand the plot without reading code)
    - Consistent color scheme across plots
    - Reference lines where appropriate (e.g., perfect calibration diagonal)
    - Saved at high resolution (150 DPI) for publication/presentation

    Attributes:
        df: DataFrame with predictions and actual values
        output_dir: Where to save generated plots
        figsize: Default figure size (width, height in inches)
        dpi: Dots per inch for saved figures
        COLORS: Color scheme dictionary for consistent styling
    """

    # ==========================================================================
    # COLOR SCHEME
    # ==========================================================================
    # Using a consistent color palette makes plots look professional and
    # helps viewers quickly identify what different elements represent.

    COLORS = {
        "primary": "#1f77b4",     # Blue - main data/predictions
        "secondary": "#ff7f0e",   # Orange - comparison/secondary data
        "tertiary": "#2ca02c",    # Green - third category
        "error": "#d62728",       # Red - errors/problems/mean lines
        "neutral": "#7f7f7f",     # Gray - background elements
        "diagonal": "#333333",    # Dark gray - reference lines (perfect calibration, etc.)
    }

    def __init__(
        self,
        predictions_df: pd.DataFrame,
        output_dir: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 150,
    ):
        """
        Initialize visualizer with a predictions DataFrame.

        The DataFrame should come from Evaluator.get_predictions_df() and contain:
        - home_score_actual, away_score_actual: True game outcomes
        - home_score_pred, away_score_pred: Model's point predictions
        - home_score_std, away_score_std: Model's uncertainty (standard deviation)
        - spread_pred, spread_std: Predicted point spread with uncertainty
        - home_win_prob: Model's probability that home team wins (0-1)

        Derived columns (added automatically if missing):
        - spread_actual: Actual point spread (home - away)
        - home_won: Binary indicator (1 if home won, 0 otherwise)

        Args:
            predictions_df: DataFrame with required prediction columns.
                Must have at least: home_score_actual, away_score_actual,
                home_score_pred, away_score_pred, home_win_prob, and std columns.

            output_dir: Directory to save plots. If None, plots are displayed
                but not saved. Will be created if it doesn't exist.

            figsize: Default figure size as (width, height) in inches.
                10x8 works well for most screens and presentations.

            dpi: Dots per inch for saved figures. Higher = larger file, better quality.
                150 is a good balance; use 300 for publication-quality prints.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib required for visualization. "
                "Install with: pip install matplotlib"
            )

        self.df = predictions_df
        self.output_dir = Path(output_dir) if output_dir else None
        self.figsize = figsize
        self.dpi = dpi

        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure derived columns exist (user might not have created them)
        if "spread_actual" not in self.df.columns:
            self.df["spread_actual"] = (
                self.df["home_score_actual"] - self.df["away_score_actual"]
            )
        if "home_won" not in self.df.columns:
            self.df["home_won"] = (
                self.df["home_score_actual"] > self.df["away_score_actual"]
            ).astype(int)

    # =========================================================================
    # PLOT 1: CALIBRATION DIAGRAM (RELIABILITY CURVE)
    # =========================================================================

    def plot_calibration(
        self,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot reliability diagram for win probability calibration.

        WHAT IS CALIBRATION?
        ====================

        Calibration measures whether predicted probabilities match actual outcomes.
        A well-calibrated model satisfies:

            "Of all games where we predicted P(home win) = 0.7,
             the home team should actually win about 70% of them."

        If the model says 70% but home teams only win 50%, the model is
        OVERCONFIDENT in home teams. If they win 90%, it's UNDERCONFIDENT.

        HOW TO READ THIS PLOT:
        ======================

        - X-axis: Predicted probability (binned, e.g., 0.6-0.7)
        - Y-axis: Actual win rate for games in that bin
        - Diagonal line: Perfect calibration
        - Bars: Actual win rate per bin
        - Orange line (secondary y-axis): Number of samples per bin

        INTERPRETATION:
        ===============

        BARS ON THE DIAGONAL: Perfect! Probabilities match reality.

        BARS ABOVE DIAGONAL: Model is underconfident. When it says 60%,
            the actual rate is higher. This is usually less problematic.

        BARS BELOW DIAGONAL: Model is overconfident. When it says 80%,
            the actual rate is lower. This is dangerous for betting!

        ECE (Expected Calibration Error):
            Weighted average gap between bars and diagonal.
            Lower is better. ECE < 0.05 is generally good.

        WHY BINNING?
        ============

        We can't check "all games with exactly P=0.7234" (too few).
        Instead, we group predictions into bins (0.6-0.7, 0.7-0.8, etc.)
        and check the average outcome for each bin.

        Args:
            n_bins: Number of probability bins (default 10 gives bins like
                0.0-0.1, 0.1-0.2, ..., 0.9-1.0). More bins = more detail
                but noisier estimates. 10 is standard in ML papers.

            save_path: Override path for saving. If None, uses output_dir.

        Returns:
            matplotlib Figure object (can be further customized or displayed).
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Extract probability predictions and actual outcomes
        probs = self.df["home_win_prob"].values    # Model's predictions (0-1)
        outcomes = self.df["home_won"].values       # Actual results (0 or 1)

        # Create bin edges from 0 to 1
        # e.g., [0.0, 0.1, 0.2, ..., 1.0] for n_bins=10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        # Assign each prediction to a bin
        # np.digitize returns bin index for each value
        bin_indices = np.digitize(probs, bin_boundaries[1:-1])

        # Compute statistics for each bin
        bin_accuracies = []   # Actual win rate in each bin
        bin_confidences = []  # Average predicted probability in each bin
        bin_counts = []       # Number of samples in each bin

        for i in range(n_bins):
            mask = bin_indices == i  # Boolean mask for this bin
            if np.sum(mask) > 0:
                # Bin has samples: compute actual win rate and average prediction
                bin_accuracies.append(np.mean(outcomes[mask]))
                bin_confidences.append(np.mean(probs[mask]))
                bin_counts.append(np.sum(mask))
            else:
                # Empty bin: use NaN for accuracy, use bin center as placeholder
                bin_accuracies.append(np.nan)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)

        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)

        # =====================================================================
        # PLOT BARS (actual win rate per bin)
        # =====================================================================
        bar_widths = np.diff(bin_boundaries)  # Width of each bin
        ax.bar(
            bin_centers,
            bin_accuracies,
            width=bar_widths * 0.8,  # Slightly narrower than bin for visual clarity
            alpha=0.7,
            color=self.COLORS["primary"],
            label="Model",
            edgecolor="white",
        )

        # =====================================================================
        # PLOT DIAGONAL (perfect calibration reference)
        # =====================================================================
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            color=self.COLORS["diagonal"],
            linewidth=2,
            label="Perfect",
        )

        # =====================================================================
        # PLOT SAMPLE COUNTS (secondary y-axis)
        # =====================================================================
        # This shows how many games are in each bin - important because
        # bins with few samples have noisy estimates
        ax2 = ax.twinx()  # Create second y-axis sharing the same x-axis
        ax2.plot(
            bin_centers,
            bin_counts,
            color=self.COLORS["secondary"],
            marker="o",
            linestyle="-",
            alpha=0.5,
            label="Samples",
        )
        ax2.set_ylabel("Sample Count", color=self.COLORS["secondary"])
        ax2.tick_params(axis="y", labelcolor=self.COLORS["secondary"])

        # =====================================================================
        # COMPUTE AND ANNOTATE ECE (Expected Calibration Error)
        # =====================================================================
        # ECE = weighted average of |accuracy - confidence| across bins
        # Weights are the fraction of samples in each bin
        valid_mask = ~np.isnan(bin_accuracies)
        ece = np.sum(
            bin_counts[valid_mask] / np.sum(bin_counts[valid_mask]) *
            np.abs(bin_accuracies[valid_mask] - bin_confidences[valid_mask])
        )

        # Add ECE annotation to plot
        ax.annotate(
            f"ECE = {ece:.4f}",
            xy=(0.05, 0.95),  # Position in axes coordinates (0-1)
            xycoords="axes fraction",
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # =====================================================================
        # LABELS AND FORMATTING
        # =====================================================================
        ax.set_xlabel("Predicted Win Probability", fontsize=12)
        ax.set_ylabel("Actual Win Rate", fontsize=12)
        ax.set_title("Win Probability Calibration", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save if path provided or output_dir is set
        if save_path or self.output_dir:
            path = save_path or str(self.output_dir / "calibration.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved calibration plot to {path}")

        return fig

    # =========================================================================
    # PLOT 2: ERROR DISTRIBUTION
    # =========================================================================

    def plot_error_distribution(
        self,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot distribution of prediction errors (histograms).

        WHAT IS ERROR DISTRIBUTION?
        ===========================

        Error = Predicted - Actual

        Positive error: We predicted too high (overestimate)
        Negative error: We predicted too low (underestimate)
        Error of 0: Perfect prediction

        The distribution shows:
        - Mean error (bias): Is the model systematically off?
        - Spread: How variable are errors?
        - Shape: Are there outliers? Is it symmetric?

        HOW TO READ THIS PLOT:
        ======================

        Three histograms are shown:
        1. Spread Error: Error in point spread prediction (home-away margin)
        2. Home Score Error: Error in home team score prediction
        3. Away Score Error: Error in away team score prediction

        For each histogram:
        - Black dashed line at 0: Perfect prediction
        - Red solid line: Mean error (bias)

        INTERPRETATION:
        ===============

        IDEAL: Bell curve centered at 0 (Gaussian with zero mean)

        MEAN FAR FROM 0: Systematic bias. E.g., mean = +3 means model
            consistently predicts scores 3 points too high.

        WIDE DISTRIBUTION: High variance/uncertainty in predictions.

        HEAVY TAILS: Many large errors (outliers). Might indicate model
            struggles with certain game types.

        SKEWED: Asymmetric errors. E.g., model might underpredict blowouts.

        Args:
            save_path: Override path for saving the figure.

        Returns:
            matplotlib Figure object with three histograms.
        """
        # Create figure with 3 side-by-side subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        # =====================================================================
        # SUBPLOT 1: Spread Error (most important for betting)
        # =====================================================================
        spread_error = self.df["spread_pred"] - self.df["spread_actual"]

        axes[0].hist(
            spread_error,
            bins=50,
            alpha=0.7,
            color=self.COLORS["primary"],
            edgecolor="white",
        )
        # Reference line at 0 (perfect prediction)
        axes[0].axvline(0, color=self.COLORS["diagonal"], linestyle="--", linewidth=2)
        # Mean error line (shows bias)
        axes[0].axvline(
            spread_error.mean(),
            color=self.COLORS["error"],
            linestyle="-",
            linewidth=2,
            label=f"Mean: {spread_error.mean():.2f}",
        )
        axes[0].set_xlabel("Spread Prediction Error", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Spread Error Distribution", fontsize=12)
        axes[0].legend()

        # =====================================================================
        # SUBPLOT 2: Home Score Error
        # =====================================================================
        home_error = self.df["home_score_pred"] - self.df["home_score_actual"]

        axes[1].hist(
            home_error,
            bins=50,
            alpha=0.7,
            color=self.COLORS["secondary"],
            edgecolor="white",
        )
        axes[1].axvline(0, color=self.COLORS["diagonal"], linestyle="--", linewidth=2)
        axes[1].axvline(
            home_error.mean(),
            color=self.COLORS["error"],
            linestyle="-",
            linewidth=2,
            label=f"Mean: {home_error.mean():.2f}",
        )
        axes[1].set_xlabel("Home Score Prediction Error", fontsize=11)
        axes[1].set_title("Home Score Error Distribution", fontsize=12)
        axes[1].legend()

        # =====================================================================
        # SUBPLOT 3: Away Score Error
        # =====================================================================
        away_error = self.df["away_score_pred"] - self.df["away_score_actual"]

        axes[2].hist(
            away_error,
            bins=50,
            alpha=0.7,
            color=self.COLORS["tertiary"],
            edgecolor="white",
        )
        axes[2].axvline(0, color=self.COLORS["diagonal"], linestyle="--", linewidth=2)
        axes[2].axvline(
            away_error.mean(),
            color=self.COLORS["error"],
            linestyle="-",
            linewidth=2,
            label=f"Mean: {away_error.mean():.2f}",
        )
        axes[2].set_xlabel("Away Score Prediction Error", fontsize=11)
        axes[2].set_title("Away Score Error Distribution", fontsize=12)
        axes[2].legend()

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or str(self.output_dir / "error_distribution.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved error distribution plot to {path}")

        return fig

    # =========================================================================
    # PLOT 3: COVERAGE ANALYSIS (PREDICTION INTERVALS)
    # =========================================================================

    def plot_coverage(
        self,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot prediction interval coverage analysis.

        WHAT ARE PREDICTION INTERVALS?
        ==============================

        A prediction interval captures uncertainty:
            "We predict spread = +5, with 90% confidence interval [-3, +13]"

        This means: "We're 90% confident the actual spread will be between -3 and +13"

        The model outputs a mean (center of interval) and standard deviation.
        For a Gaussian distribution:
        - 50% interval: mean +/- 0.67 * std
        - 90% interval: mean +/- 1.645 * std
        - 95% interval: mean +/- 1.96 * std

        WHAT IS COVERAGE?
        =================

        Coverage = Fraction of actual values that fall within the predicted interval

        For a 90% interval, we expect coverage of 0.90 (90% of actuals inside).
        For a 50% interval, we expect coverage of 0.50.

        HOW TO READ THIS PLOT:
        ======================

        Two plots are shown:
        1. Spread coverage: How well spread intervals capture true spread
        2. Score coverage: How well score intervals capture true scores

        - X-axis: Target coverage (0.1, 0.2, ..., 0.9)
        - Y-axis: Actual coverage (fraction of actuals in interval)
        - Diagonal: Perfect coverage

        INTERPRETATION:
        ===============

        ON DIAGONAL: Perfect! 90% intervals contain 90% of values.

        ABOVE DIAGONAL: Under-confident. Intervals are too wide.
            The 90% interval actually covers 95%. This is conservative but wasteful.

        BELOW DIAGONAL: Over-confident. Intervals are too narrow.
            The 90% interval only covers 75%. DANGEROUS for risk management!

        SHADED AREA: Gap between actual and target coverage.
            Larger area = worse calibration of uncertainty.

        WHY THIS MATTERS:
        =================

        If you're making decisions based on uncertainty (e.g., "don't bet when
        uncertainty is high"), you need to trust the uncertainty estimates.
        Poor coverage means the uncertainty is misleading.

        Args:
            save_path: Override path for saving the figure.

        Returns:
            matplotlib Figure object with coverage analysis.
        """
        # scipy.stats for computing z-scores of confidence levels
        from scipy.stats import norm

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        # Confidence levels to check (10%, 20%, ..., 90%)
        confidence_levels = np.arange(0.1, 1.0, 0.1)

        # =====================================================================
        # SUBPLOT 1: Spread Coverage
        # =====================================================================
        spread_coverages = []
        for conf in confidence_levels:
            # z-score for this confidence level
            # For 90% confidence, we need 1.645 standard deviations on each side
            z = norm.ppf((1 + conf) / 2)

            # Compute interval bounds
            lower = self.df["spread_pred"] - z * self.df["spread_std"]
            upper = self.df["spread_pred"] + z * self.df["spread_std"]

            # Check which actuals fall within the interval
            in_interval = (self.df["spread_actual"] >= lower) & (
                self.df["spread_actual"] <= upper
            )
            spread_coverages.append(in_interval.mean())  # Fraction inside

        # Plot actual coverage vs target
        axes[0].plot(
            confidence_levels,
            spread_coverages,
            "o-",
            color=self.COLORS["primary"],
            linewidth=2,
            markersize=8,
            label="Actual",
        )
        # Perfect coverage diagonal
        axes[0].plot(
            [0, 1], [0, 1],
            linestyle="--",
            color=self.COLORS["diagonal"],
            linewidth=2,
            label="Perfect",
        )
        # Shaded area showing the gap (miscalibration)
        axes[0].fill_between(
            confidence_levels,
            confidence_levels,  # Target (diagonal)
            spread_coverages,   # Actual
            alpha=0.2,
            color=self.COLORS["error"],
        )
        axes[0].set_xlabel("Target Coverage", fontsize=11)
        axes[0].set_ylabel("Actual Coverage", fontsize=11)
        axes[0].set_title("Spread Prediction Interval Coverage", fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        # =====================================================================
        # SUBPLOT 2: Score Coverage (Home and Away)
        # =====================================================================

        # Home score coverage
        home_coverages = []
        for conf in confidence_levels:
            z = norm.ppf((1 + conf) / 2)
            lower = self.df["home_score_pred"] - z * self.df["home_score_std"]
            upper = self.df["home_score_pred"] + z * self.df["home_score_std"]
            in_interval = (self.df["home_score_actual"] >= lower) & (
                self.df["home_score_actual"] <= upper
            )
            home_coverages.append(in_interval.mean())

        axes[1].plot(
            confidence_levels,
            home_coverages,
            "o-",
            color=self.COLORS["secondary"],
            linewidth=2,
            markersize=8,
            label="Home Score",
        )

        # Away score coverage
        away_coverages = []
        for conf in confidence_levels:
            z = norm.ppf((1 + conf) / 2)
            lower = self.df["away_score_pred"] - z * self.df["away_score_std"]
            upper = self.df["away_score_pred"] + z * self.df["away_score_std"]
            in_interval = (self.df["away_score_actual"] >= lower) & (
                self.df["away_score_actual"] <= upper
            )
            away_coverages.append(in_interval.mean())

        axes[1].plot(
            confidence_levels,
            away_coverages,
            "s-",  # Square markers to distinguish from home
            color=self.COLORS["tertiary"],
            linewidth=2,
            markersize=8,
            label="Away Score",
        )

        # Perfect coverage diagonal
        axes[1].plot(
            [0, 1], [0, 1],
            linestyle="--",
            color=self.COLORS["diagonal"],
            linewidth=2,
            label="Perfect",
        )
        axes[1].set_xlabel("Target Coverage", fontsize=11)
        axes[1].set_ylabel("Actual Coverage", fontsize=11)
        axes[1].set_title("Score Prediction Interval Coverage", fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or str(self.output_dir / "coverage.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved coverage plot to {path}")

        return fig

    # =========================================================================
    # PLOT 4: PREDICTIONS VS ACTUAL (SCATTER PLOTS)
    # =========================================================================

    def plot_predictions_vs_actual(
        self,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Create scatter plots of predicted vs actual values.

        WHAT THIS SHOWS:
        ================

        Each point represents one game:
        - X-axis: Actual value (true outcome)
        - Y-axis: Predicted value (model's guess)

        Points on the diagonal: Perfect predictions
        Points above diagonal: Overprediction
        Points below diagonal: Underprediction

        HOW TO READ THIS PLOT:
        ======================

        Three scatter plots show different prediction types:
        1. Spread: Point spread (home - away)
        2. Home Score: Home team final score
        3. Away Score: Away team final score

        INTERPRETATION:
        ===============

        TIGHT AROUND DIAGONAL: Strong predictions. Points cluster near y=x.

        WIDE SCATTER: High variance in predictions. Model struggles.

        SYSTEMATIC TILT: If the cloud of points has a different slope than the
            diagonal, the model has a regression-to-the-mean problem (common).

        CLUSTERS OR PATTERNS: Might indicate specific game types the model
            handles differently.

        EDGE EFFECTS: If predictions pile up near certain values (e.g., always
            predicting ~110 points), the model might be too conservative.

        Args:
            save_path: Override path for saving the figure.

        Returns:
            matplotlib Figure object with three scatter plots.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        # =====================================================================
        # SUBPLOT 1: Spread
        # =====================================================================
        axes[0].scatter(
            self.df["spread_actual"],
            self.df["spread_pred"],
            alpha=0.3,  # Transparency to show density
            s=10,       # Small points for many games
            color=self.COLORS["primary"],
        )
        # Compute axis limits with padding
        lims = [
            min(self.df["spread_actual"].min(), self.df["spread_pred"].min()) - 5,
            max(self.df["spread_actual"].max(), self.df["spread_pred"].max()) + 5,
        ]
        # Perfect prediction diagonal
        axes[0].plot(lims, lims, "--", color=self.COLORS["diagonal"], linewidth=2)
        axes[0].set_xlabel("Actual Spread", fontsize=11)
        axes[0].set_ylabel("Predicted Spread", fontsize=11)
        axes[0].set_title("Spread: Predicted vs Actual", fontsize=12)
        axes[0].grid(alpha=0.3)

        # =====================================================================
        # SUBPLOT 2: Home Score
        # =====================================================================
        axes[1].scatter(
            self.df["home_score_actual"],
            self.df["home_score_pred"],
            alpha=0.3,
            s=10,
            color=self.COLORS["secondary"],
        )
        lims = [
            min(self.df["home_score_actual"].min(), self.df["home_score_pred"].min()) - 10,
            max(self.df["home_score_actual"].max(), self.df["home_score_pred"].max()) + 10,
        ]
        axes[1].plot(lims, lims, "--", color=self.COLORS["diagonal"], linewidth=2)
        axes[1].set_xlabel("Actual Home Score", fontsize=11)
        axes[1].set_ylabel("Predicted Home Score", fontsize=11)
        axes[1].set_title("Home Score: Predicted vs Actual", fontsize=12)
        axes[1].grid(alpha=0.3)

        # =====================================================================
        # SUBPLOT 3: Away Score
        # =====================================================================
        axes[2].scatter(
            self.df["away_score_actual"],
            self.df["away_score_pred"],
            alpha=0.3,
            s=10,
            color=self.COLORS["tertiary"],
        )
        lims = [
            min(self.df["away_score_actual"].min(), self.df["away_score_pred"].min()) - 10,
            max(self.df["away_score_actual"].max(), self.df["away_score_pred"].max()) + 10,
        ]
        axes[2].plot(lims, lims, "--", color=self.COLORS["diagonal"], linewidth=2)
        axes[2].set_xlabel("Actual Away Score", fontsize=11)
        axes[2].set_ylabel("Predicted Away Score", fontsize=11)
        axes[2].set_title("Away Score: Predicted vs Actual", fontsize=12)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or str(self.output_dir / "predictions_vs_actual.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved predictions vs actual plot to {path}")

        return fig

    # =========================================================================
    # PLOT 5: UNCERTAINTY ANALYSIS
    # =========================================================================

    def plot_uncertainty_analysis(
        self,
        save_path: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Analyze relationship between predicted uncertainty and actual error.

        WHAT THIS TESTS:
        ================

        Does the model "know when it doesn't know"?

        A well-calibrated model should:
        - Predict HIGH uncertainty for games that end up having LARGE errors
        - Predict LOW uncertainty for games that end up having SMALL errors

        If uncertainty doesn't correlate with error, the uncertainty is meaningless
        and can't be trusted for decision-making.

        HOW TO READ THIS PLOT:
        ======================

        Two views are provided:
        1. Scatter: Each point is a game, x=predicted std, y=actual absolute error
        2. Binned: Games grouped by uncertainty decile, showing avg error per group

        SCATTER PLOT (Left):
        - Should show positive correlation (upward trend)
        - Trend line shows the relationship
        - Slope should be positive (higher uncertainty -> higher error)

        BINNED PLOT (Right):
        - X-axis: Uncertainty decile (1=lowest uncertainty, 10=highest)
        - Blue bars: Average absolute error in each decile
        - Orange line: Average predicted std in each decile

        INTERPRETATION:
        ===============

        IDEAL: Bars increase left-to-right. Model is most wrong when most uncertain.

        FLAT BARS: Uncertainty is meaningless. Low-uncertainty predictions have
            the same error as high-uncertainty ones.

        DECREASING BARS: VERY BAD! Model is most wrong when most confident.
            This indicates a serious problem with uncertainty estimation.

        WHY THIS MATTERS:
        =================

        If uncertainty is meaningful, you can:
        - Skip bets when uncertainty is high
        - Increase bet size when uncertainty is low
        - Build confidence-weighted portfolios

        If uncertainty is meaningless, these strategies will fail.

        Args:
            save_path: Override path for saving the figure.

        Returns:
            matplotlib Figure object with uncertainty analysis.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        # =====================================================================
        # SUBPLOT 1: Scatter of std vs absolute error
        # =====================================================================
        spread_abs_error = np.abs(
            self.df["spread_pred"] - self.df["spread_actual"]
        )

        axes[0].scatter(
            self.df["spread_std"],
            spread_abs_error,
            alpha=0.3,
            s=10,
            color=self.COLORS["primary"],
        )

        # Fit and plot trend line (linear regression)
        # z[0] is slope, z[1] is intercept
        z = np.polyfit(self.df["spread_std"], spread_abs_error, 1)
        p = np.poly1d(z)  # Create polynomial function

        # Generate points for trend line
        x_line = np.linspace(
            self.df["spread_std"].min(),
            self.df["spread_std"].max(),
            100,
        )
        axes[0].plot(
            x_line,
            p(x_line),
            color=self.COLORS["error"],
            linewidth=2,
            label=f"Trend (slope={z[0]:.2f})",
        )

        axes[0].set_xlabel("Predicted Std Dev", fontsize=11)
        axes[0].set_ylabel("Absolute Error", fontsize=11)
        axes[0].set_title("Spread: Uncertainty vs Error", fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # =====================================================================
        # SUBPLOT 2: Binned analysis by uncertainty decile
        # =====================================================================

        # Create decile bins based on uncertainty
        # pd.qcut creates bins with equal number of samples
        std_bins = pd.qcut(self.df["spread_std"], q=10, duplicates="drop")

        # Compute average std per bin
        binned = self.df.groupby(std_bins).agg({
            "spread_std": "mean",
        }).reset_index(drop=True)

        # Compute average absolute error per bin
        binned["abs_error"] = self.df.groupby(std_bins).apply(
            lambda x: np.abs(x["spread_pred"] - x["spread_actual"]).mean()
        ).values

        # Bar chart of average error per decile
        axes[1].bar(
            range(len(binned)),
            binned["abs_error"],
            alpha=0.7,
            color=self.COLORS["primary"],
            label="Avg Abs Error",
        )

        # Overlay line chart of average std
        ax2 = axes[1].twinx()
        ax2.plot(
            range(len(binned)),
            binned["spread_std"],
            "o-",
            color=self.COLORS["secondary"],
            linewidth=2,
            markersize=8,
            label="Avg Std Dev",
        )

        axes[1].set_xlabel("Uncertainty Decile (low -> high)", fontsize=11)
        axes[1].set_ylabel("Average Absolute Error", fontsize=11)
        ax2.set_ylabel("Average Predicted Std", color=self.COLORS["secondary"])
        axes[1].set_title("Binned Uncertainty Analysis", fontsize=12)
        ax2.tick_params(axis="y", labelcolor=self.COLORS["secondary"])

        # Combine legends from both y-axes
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or str(self.output_dir / "uncertainty_analysis.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved uncertainty analysis plot to {path}")

        return fig

    # =========================================================================
    # CONVENIENCE METHOD: GENERATE ALL PLOTS
    # =========================================================================

    def plot_all(self):
        """
        Generate all diagnostic plots at once.

        This is a convenience method that calls all individual plot methods.
        Useful for generating a complete analysis with one function call.

        Plots generated:
        1. Calibration diagram (calibration.png)
        2. Error distribution (error_distribution.png)
        3. Coverage analysis (coverage.png)
        4. Predictions vs actual (predictions_vs_actual.png)
        5. Uncertainty analysis (uncertainty_analysis.png)

        All plots are saved to self.output_dir if set.
        """
        if self.output_dir is None:
            logger.warning("No output directory set. Plots will not be saved.")

        self.plot_calibration()
        self.plot_error_distribution()
        self.plot_coverage()
        self.plot_predictions_vs_actual()
        self.plot_uncertainty_analysis()

        logger.info("All plots generated.")


# =============================================================================
# STANDALONE TRAINING CURVES FUNCTION
# =============================================================================

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    metrics_history: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """
    Plot training and validation loss curves over epochs.

    This is a standalone function (not part of Visualizer) because it's used
    during training, before we have predictions to analyze.

    WHAT THIS SHOWS:
    ================

    Loss curves reveal training dynamics:
    - Is the model learning? (losses should decrease)
    - Is it overfitting? (train loss drops, val loss rises)
    - Has it converged? (losses plateau)
    - Was training stopped too early/late?

    HOW TO READ THIS PLOT:
    ======================

    - X-axis: Training epoch (iteration over full dataset)
    - Y-axis: Loss value (lower is better)
    - Blue line: Training loss
    - Orange line: Validation loss

    INTERPRETATION:
    ===============

    BOTH DECREASING: Good! Model is learning.

    TRAIN DROPS, VAL RISES: Classic overfitting. Model is memorizing
        training data instead of learning general patterns.

    GAP BETWEEN CURVES: Normal to have some gap. Large gap = overfit.

    VAL LOWER THAN TRAIN: Unusual. Check for data leakage or wrong data splits.

    CURVES PLATEAU: Model has converged. More training won't help.

    CURVES NOISY: Learning rate might be too high, or batch size too small.

    Args:
        train_losses: List of training loss values, one per epoch.

        val_losses: List of validation loss values, one per epoch.
            Should be same length as train_losses.

        metrics_history: Optional dictionary mapping metric names to lists of
            values. If provided, a second subplot shows metric evolution.
            Example: {"spread_mae": [10.5, 9.8, 9.2, ...], "win_acc": [...]}

        save_path: Path to save the figure. If None, figure is displayed but
            not saved.

    Returns:
        matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")

    # Determine number of subplots
    n_plots = 1 + (1 if metrics_history else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), dpi=150)

    # Handle single subplot case (axes not a list)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # =========================================================================
    # SUBPLOT 1: Loss Curves
    # =========================================================================
    axes[0].plot(epochs, train_losses, label="Train", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # =========================================================================
    # SUBPLOT 2: Metrics Curves (optional)
    # =========================================================================
    if metrics_history:
        for name, values in metrics_history.items():
            if len(values) == len(epochs):
                axes[1].plot(epochs, values, label=name, linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("Metrics Over Training")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")

    return fig


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_visualizer():
    """
    Test visualization with synthetic mock data.

    This function creates fake predictions and runs all visualization methods
    to verify they work correctly. The mock data simulates realistic NBA
    predictions with some noise.

    Useful for:
    - Quick testing during development
    - Verifying matplotlib installation
    - Understanding what the plots look like

    Returns:
        0 on success (standard Unix exit code)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Skipping visualization test.")
        return 0

    print("Testing Visualizer...")

    # =========================================================================
    # CREATE MOCK PREDICTIONS DATAFRAME
    # =========================================================================

    np.random.seed(42)  # For reproducibility
    n_samples = 500

    # Generate realistic NBA-like scores
    df = pd.DataFrame({
        "home_score_actual": np.random.normal(110, 12, n_samples),  # ~110 +/- 12
        "away_score_actual": np.random.normal(108, 12, n_samples),  # ~108 +/- 12
        "home_score_pred": np.random.normal(110, 8, n_samples),     # Predictions with lower std
        "away_score_pred": np.random.normal(108, 8, n_samples),
        "home_score_std": np.abs(np.random.normal(10, 3, n_samples)),  # Uncertainty
        "away_score_std": np.abs(np.random.normal(10, 3, n_samples)),
    })

    # Add spread predictions
    df["spread_pred"] = df["home_score_pred"] - df["away_score_pred"]
    # Combine uncertainties (variance adds, so std = sqrt(sum of variances))
    df["spread_std"] = np.sqrt(df["home_score_std"]**2 + df["away_score_std"]**2)

    # Add win probability (logistic function of spread)
    # Larger positive spread -> higher home win probability
    df["home_win_prob"] = 1 / (1 + np.exp(-df["spread_pred"] / 5))
    # Add some noise to probabilities
    df["home_win_prob"] = np.clip(
        df["home_win_prob"] + np.random.normal(0, 0.1, n_samples),
        0.01,
        0.99,
    )

    # =========================================================================
    # RUN VISUALIZER
    # =========================================================================

    viz = Visualizer(df, output_dir="/tmp/viz_test")
    viz.plot_all()

    # =========================================================================
    # TEST TRAINING CURVES
    # =========================================================================

    # Generate mock training curves (loss decreasing with noise)
    train_losses = [10 - 0.1 * i + np.random.normal(0, 0.3) for i in range(50)]
    val_losses = [10 - 0.08 * i + np.random.normal(0, 0.5) for i in range(50)]
    plot_training_curves(
        train_losses,
        val_losses,
        save_path="/tmp/viz_test/training_curves.png",
    )

    print("\nAll visualization tests passed!")
    print("Plots saved to /tmp/viz_test/")
    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(test_visualizer())
