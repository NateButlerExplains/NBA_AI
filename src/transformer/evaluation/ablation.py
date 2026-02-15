"""
Ablation Study Runner for NBA Transformer Model.

=============================================================================
WHAT IS AN ABLATION STUDY?
=============================================================================

An ablation study is a systematic experiment to understand which parts of
a model actually contribute to its performance. The term comes from medicine,
where "ablation" means removing tissue to study its function.

Think of it like debugging a car that runs well:
- Remove the turbocharger -> car is slower -> turbo helps performance
- Remove the spoiler -> same speed -> spoiler is cosmetic only
- Remove fuel injector -> car won't start -> fuel injector is essential

In machine learning, we "remove" or "reduce" components and measure impact:
- Which features actually help predictions?
- Does a bigger model really perform better?
- Is that complex attention mechanism worth the computational cost?

WHY ARE ABLATION STUDIES IMPORTANT?
===================================

1. UNDERSTANDING: They reveal what's actually driving your model's performance.
   Without ablations, you might have a working model but not understand WHY
   it works. This makes debugging and improvement nearly impossible.

2. SIMPLIFICATION: Often, simpler models are better in practice:
   - Faster inference (important for real-time predictions)
   - Less memory usage
   - Easier to deploy and maintain
   - Less prone to overfitting
   If an ablation shows a component doesn't help, you can safely remove it.

3. PUBLICATION/DOCUMENTATION: If you claim "feature X improves accuracy by 5%",
   you need evidence. Ablation studies provide rigorous proof of contributions.

4. COST-BENEFIT ANALYSIS: Some components are expensive (GPU time, data,
   complexity). Ablations help decide if the cost is justified by the benefit.

ABLATION TYPES IN THIS MODULE:
==============================

1. SEQUENCE LENGTH ABLATION:
   - Question: How much history does the model need?
   - Tests: 3, 5, 10, 20 previous games
   - Insight: Reveals the "memory" requirements of NBA prediction

2. MODEL SIZE ABLATION:
   - Question: Is our model too small, too big, or just right?
   - Tests: Small (128d), Medium (256d), Large (512d) hidden dimensions
   - Insight: Balances accuracy vs. computational cost

Usage:
    from src.transformer.evaluation import AblationRunner

    # Set up with base configuration
    runner = AblationRunner(base_config)

    # Run all ablation experiments
    results = runner.run_all()

    # Generate summary report
    runner.generate_report()
"""

import copy
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.transformer.training.config import (
    ExperimentConfig,
    get_small_config,
    get_medium_config,
    get_large_config,
    get_ablation_configs,
)
from src.transformer.training.trainer import Trainer
from src.transformer.training.metrics import MetricResults
from src.transformer.models import Phase1Model
from src.transformer.tokenizer import PBPTokenizer
from src.transformer.dataset import NBAGameDataset, collate_games


logger = logging.getLogger(__name__)


# =============================================================================
# ABLATION RESULT DATA CLASS
# =============================================================================

@dataclass
class AblationResult:
    """
    Container for results from a single ablation experiment.

    Each ablation experiment produces one of these objects, capturing:
    - What was changed from baseline (config_changes)
    - How well it performed (metrics)
    - How long it took (training_time_seconds)
    - Where the best model was saved (checkpoint_path)

    This standardized format makes it easy to compare experiments and
    generate summary reports.

    Attributes:
        name: Short identifier for this experiment (e.g., "seq_len_10").
            Used in filenames and reports.

        config_changes: Dictionary describing what changed from baseline.
            Example: {"n_history_games": 10} or {"hidden_dim": 128}
            This is crucial for understanding what made this experiment different.

        metrics: Final MetricResults from training (validation metrics at best epoch).
            Includes MAE, AUC, calibration, etc.

        training_time_seconds: Wall-clock time for the full training run.
            Useful for understanding computational costs of different configs.

        best_epoch: Which epoch achieved the best validation loss.
            Low values might indicate quick convergence or too-simple model.
            High values close to max_epochs might indicate need for more training.

        total_epochs: How many epochs were actually run (might be less than
            max_epochs if early stopping triggered).

        checkpoint_path: Path to saved model weights for this experiment.
            Allows re-loading the best model for further analysis.
    """

    name: str                      # Experiment identifier
    config_changes: dict           # What changed from baseline
    metrics: MetricResults         # Performance metrics
    training_time_seconds: float   # Training duration
    best_epoch: int                # Epoch with best validation loss
    total_epochs: int              # Total epochs run
    checkpoint_path: str = ""      # Path to saved model

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary for saving.

        Returns:
            Dictionary with all fields converted to basic Python types.
            MetricResults is converted using dataclasses.asdict().
        """
        return {
            "name": self.name,
            "config_changes": self.config_changes,
            "metrics": asdict(self.metrics),
            "training_time_seconds": self.training_time_seconds,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
            "checkpoint_path": self.checkpoint_path,
        }


# =============================================================================
# MAIN ABLATION RUNNER CLASS
# =============================================================================

class AblationRunner:
    """
    Systematic ablation study runner.

    This class automates the process of running multiple related experiments
    with controlled variations. It handles:
    - Creating model configurations for each experiment
    - Setting up data loaders
    - Running training
    - Collecting and saving results
    - Generating summary reports

    HOW IT WORKS (CONCEPTUALLY):
    ============================

    1. START WITH BASELINE: Define a "base" configuration that represents your
       best current model. All experiments will be compared against this.

    2. SYSTEMATIC VARIATION: For each ablation type, create variations that
       change ONE thing at a time. This isolation is key - if you change
       multiple things, you can't tell which caused the performance change.

    3. TRAIN EACH VARIANT: Run the full training pipeline for each configuration.
       This can be time-consuming but is necessary for fair comparisons.

    4. COLLECT RESULTS: Store all metrics and metadata in a standardized format.

    5. ANALYZE: Compare variants to baseline to understand component contributions.

    ISOLATION PRINCIPLE:
    ====================

    The most important rule in ablation studies is to change ONE thing at a time.

    BAD EXPERIMENT:
        Variant: Bigger model + more data + new feature
        Result: 5% improvement
        Problem: What caused the improvement? You can't tell!

    GOOD EXPERIMENT:
        Variant A: Bigger model only -> 2% improvement
        Variant B: More data only -> 4% improvement
        Variant C: New feature only -> 1% improvement
        Now you know: Data helped most, model size helped some, feature barely helped.

    Example Usage:
        >>> base_config = get_medium_config()
        >>> runner = AblationRunner(base_config, output_dir="experiments/ablation")
        >>>
        >>> # Run all ablation studies
        >>> results = runner.run_all()
        >>>
        >>> # Or run specific ablation types
        >>> seq_results = runner.run_sequence_length_ablation()
        >>> size_results = runner.run_model_size_ablation()
        >>>
        >>> # Generate markdown report
        >>> report = runner.generate_report()
        >>> print(report)
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        output_dir: str = "experiments/ablation",
        device: str = "cuda",
    ):
        """
        Initialize the ablation runner.

        Args:
            base_config: The baseline configuration to compare against.
                This should be your "default" or "best current" setup.
                All ablation experiments will modify this config.

            output_dir: Directory to save results, checkpoints, and reports.
                Each experiment creates its own subdirectory here.
                Will be created if it doesn't exist.

            device: Device for training ("cuda" for GPU, "cpu" for CPU).
                GPU is strongly recommended - ablation studies involve
                training many models, which is slow on CPU.
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate results as experiments complete
        self.results: list[AblationResult] = []

        # Tokenizer is cached after first load (shared across experiments)
        # This saves time since tokenizer building is slow
        self.tokenizer: Optional[PBPTokenizer] = None

    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================

    def _load_tokenizer(self, config: ExperimentConfig) -> PBPTokenizer:
        """
        Load or build the tokenizer (cached for reuse).

        The tokenizer converts play-by-play text into numeric tokens.
        Building it requires scanning the database, which is slow, so we
        cache it after the first build and reuse across experiments.

        Args:
            config: Experiment config containing tokenizer path and vocab settings.

        Returns:
            PBPTokenizer ready to tokenize play-by-play data.
        """
        # Return cached tokenizer if available
        if self.tokenizer is not None:
            return self.tokenizer

        tokenizer = PBPTokenizer()
        tokenizer_path = Path(config.data.tokenizer_path)

        if tokenizer_path.exists():
            # Load pre-built tokenizer from disk
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            tokenizer.load(str(tokenizer_path))
        else:
            # Build tokenizer from database (slow, but only done once)
            logger.info("Building tokenizer from database...")
            tokenizer.build_vocab_from_db(
                seasons=config.data.train_seasons,
                min_count=config.data.min_token_count,
            )
            # Save for future use
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tokenizer_path))

        # Cache for subsequent experiments
        self.tokenizer = tokenizer
        return tokenizer

    def _create_dataloaders(
        self,
        config: ExperimentConfig,
        tokenizer: PBPTokenizer,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders from config.

        Each experiment needs its own data loaders because some config
        parameters affect data loading (e.g., n_history_games changes
        how much historical data is fetched per game).

        Args:
            config: Experiment config with data parameters.
            tokenizer: Tokenizer for converting text to tokens.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets for each split
        # Note: n_history_games is a key parameter that varies in ablations
        train_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=config.data.train_seasons,
            n_history_games=config.data.n_history_games,
            min_history_games=config.data.min_history_games,
            cache_size=config.data.cache_size,
        )

        val_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=config.data.val_seasons,
            n_history_games=config.data.n_history_games,
            min_history_games=config.data.min_history_games,
            cache_size=config.data.cache_size,
        )

        test_dataset = NBAGameDataset(
            tokenizer=tokenizer,
            seasons=config.data.test_seasons,
            n_history_games=config.data.n_history_games,
            min_history_games=config.data.min_history_games,
            cache_size=config.data.cache_size,
        )

        # Wrap datasets in DataLoaders for batching
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,  # Shuffle training data each epoch
            collate_fn=collate_games,  # Custom function to batch variable-length games
            num_workers=config.data.num_workers,  # Parallel data loading
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  # Don't shuffle validation (reproducibility)
            collate_fn=collate_games,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  # Don't shuffle test (reproducibility)
            collate_fn=collate_games,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )

        return train_loader, val_loader, test_loader

    def _create_model(
        self,
        config: ExperimentConfig,
        tokenizer: PBPTokenizer,
    ) -> Phase1Model:
        """
        Create a model instance from experiment configuration.

        Different ablation experiments test different model architectures,
        so we need to create a fresh model for each experiment.

        Args:
            config: Experiment config with model architecture parameters.
            tokenizer: Tokenizer (provides vocabulary sizes for embeddings).

        Returns:
            Newly initialized Phase1Model (untrained, random weights).
        """
        # Inject vocabulary sizes from the tokenizer into the model config
        config.model.vocab_sizes = tokenizer.vocab_sizes
        return Phase1Model(config.model)

    # =========================================================================
    # CORE EXPERIMENT RUNNER
    # =========================================================================

    def run_experiment(
        self,
        name: str,
        config: ExperimentConfig,
        config_changes: dict,
    ) -> AblationResult:
        """
        Run a single ablation experiment (train model, evaluate, record).

        This is the core method that all ablation types use. It handles
        the full lifecycle of one experiment:
        1. Set up data and model from config
        2. Train until convergence (with early stopping)
        3. Record metrics and save checkpoint
        4. Return standardized result object

        Args:
            name: Identifier for this experiment (e.g., "seq_len_10").
                Used for logging, filenames, and report generation.

            config: Complete experiment configuration.
                This is typically a modified copy of base_config.

            config_changes: Dictionary documenting what changed from baseline.
                Example: {"n_history_games": 10}
                This is stored in results for later analysis.

        Returns:
            AblationResult with metrics, timing, and metadata.

        Example:
            >>> config = copy.deepcopy(self.base_config)
            >>> config.data.n_history_games = 10
            >>> result = runner.run_experiment(
            ...     name="seq_len_10",
            ...     config=config,
            ...     config_changes={"n_history_games": 10}
            ... )
            >>> print(f"MAE: {result.metrics.spread_mae:.2f}")
        """
        # Log experiment start with clear visual separation
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation: {name}")
        logger.info(f"Changes: {config_changes}")
        logger.info("=" * 60)

        # Update config with experiment name and device
        config.training.experiment_name = f"ablation_{name}"
        config.training.device = self.device

        # =====================================================================
        # SETUP: Create data loaders and model
        # =====================================================================
        tokenizer = self._load_tokenizer(config)
        train_loader, val_loader, test_loader = self._create_dataloaders(
            config, tokenizer
        )
        model = self._create_model(config, tokenizer)

        # =====================================================================
        # TRAINING: Run full training loop with timing
        # =====================================================================
        import time
        start_time = time.time()

        # Trainer handles the full training loop:
        # - Forward/backward passes
        # - Gradient updates
        # - Validation at each epoch
        # - Early stopping when validation loss plateaus
        # - Checkpointing best model
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        # train() runs until early stopping or max_epochs
        # Returns final validation metrics
        final_metrics = trainer.train()
        training_time = time.time() - start_time

        # =====================================================================
        # RECORD RESULTS
        # =====================================================================
        result = AblationResult(
            name=name,
            config_changes=config_changes,
            metrics=final_metrics,
            training_time_seconds=training_time,
            best_epoch=trainer.state.epoch - trainer.state.patience_counter,
            total_epochs=trainer.state.epoch,
            checkpoint_path=str(trainer.checkpoint_dir / "best.pt"),
        )

        # Add to running list of results
        self.results.append(result)

        # Save individual experiment result as JSON
        # Useful if the full ablation is interrupted - partial results are preserved
        result_path = self.output_dir / f"{name}.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Ablation {name} complete. Spread MAE: {final_metrics.spread_mae:.2f}")

        return result

    # =========================================================================
    # ABLATION TYPE: SEQUENCE LENGTH
    # =========================================================================

    def run_sequence_length_ablation(self) -> list[AblationResult]:
        """
        Run ablation on historical sequence length (number of past games).

        WHAT THIS TESTS:
        ================
        How much historical context does the model need to make good predictions?
        We vary the number of previous games (N) provided as input:
        - N=3: Very recent history only (last ~1 week)
        - N=5: About 2 weeks of games
        - N=10: About a month of games
        - N=20: About 2 months of games

        WHY THIS MATTERS:
        =================
        - More history might capture trends, streaks, matchup patterns
        - But more history also means:
          * More computation (longer sequences = slower training)
          * Risk of including outdated/irrelevant information
          * Harder for model to learn what to focus on

        EXPECTED OUTCOMES:
        ==================
        - Too few games (N=3): Might miss important patterns
        - Sweet spot (N=5-10): Usually optimal
        - Too many games (N=20): Might include noise, no improvement
        - Diminishing returns: Improvements shrink as N increases

        FOR NBA SPECIFICALLY:
        =====================
        NBA teams play ~3-4 games per week. The optimal N depends on:
        - How quickly team performance changes (injuries, trades)
        - How important recent form vs. long-term ability is
        - How consistent individual game outcomes are

        Returns:
            List of AblationResult objects, one per sequence length tested.
        """
        results = []

        # Test different sequence lengths
        for n_games in [3, 5, 10, 20]:
            # Create a modified copy of base config
            # copy.deepcopy() ensures we don't accidentally modify the original
            config = copy.deepcopy(self.base_config)
            config.data.n_history_games = n_games

            # Run experiment
            result = self.run_experiment(
                name=f"seq_len_{n_games}",  # e.g., "seq_len_10"
                config=config,
                config_changes={"n_history_games": n_games},
            )
            results.append(result)

        return results

    # =========================================================================
    # ABLATION TYPE: MODEL SIZE
    # =========================================================================

    def run_model_size_ablation(self) -> list[AblationResult]:
        """
        Run ablation on model size (hidden dimensions, layer counts).

        WHAT THIS TESTS:
        ================
        Is our model too small (underfitting) or too big (overfitting/wasteful)?

        We test three predefined configurations:
        - Small: hidden_dim=256, fewer layers
        - Medium: hidden_dim=512, moderate layers
        - Large: hidden_dim=768, more layers

        WHY THIS MATTERS:
        =================
        Model size affects:
        - Capacity: Bigger models can learn more complex patterns
        - Speed: Bigger models are slower to train and run
        - Overfitting: Bigger models can memorize training data
        - Memory: Bigger models need more GPU RAM

        EXPECTED OUTCOMES:
        ==================
        - If Small beats Medium: Model is overfitting, reduce complexity
        - If Medium beats Large: Diminishing returns, stick with Medium
        - If Large beats Medium: More capacity helps, maybe try XLarge

        THE BIAS-VARIANCE TRADEOFF:
        ===========================
        (This is a fundamental ML concept)

        - Small models have HIGH BIAS: They're too simple to capture patterns.
          They underfit - same errors on training AND test data.

        - Large models have HIGH VARIANCE: They memorize training data.
          They overfit - low training error but high test error.

        - The sweet spot balances both: Complex enough to learn patterns,
          but not so complex that it memorizes noise.

        Returns:
            List of AblationResult objects, one per model size.
        """
        # Get predefined configurations for each size
        configs = {
            "small": get_small_config(),   # ~2M parameters
            "medium": get_medium_config(), # ~8M parameters
            "large": get_large_config(),   # ~20M parameters
        }

        results = []
        for name, config in configs.items():
            result = self.run_experiment(
                name=f"size_{name}",  # e.g., "size_medium"
                config=config,
                config_changes={"model_size": name},
            )
            results.append(result)

        return results

    # =========================================================================
    # FULL ABLATION SUITE
    # =========================================================================

    def run_all(self) -> list[AblationResult]:
        """
        Run complete ablation study (all ablation types).

        This method runs the full suite of ablation experiments:
        1. Baseline (unchanged config)
        2. Sequence length ablation (4 experiments)
        3. Model size ablation (3 experiments)
        4. Context ablation (3 experiments)
        5. Architecture ablation (3 experiments)

        Total: ~14 experiments, each requiring full training.

        WARNING: This takes a long time! On a modern GPU, expect:
        - Quick ablation (small models, few epochs): ~2-4 hours
        - Full ablation (medium models, full training): ~12-24 hours

        Results are saved incrementally, so partial runs can be resumed.

        Returns:
            List of all AblationResult objects.
        """
        logger.info("Starting full ablation study suite...")

        # =====================================================================
        # BASELINE: Train the unchanged base configuration first
        # =====================================================================
        # This is the reference point for all comparisons
        baseline_config = copy.deepcopy(self.base_config)
        self.run_experiment(
            name="baseline",
            config=baseline_config,
            config_changes={},  # No changes from base
        )

        # =====================================================================
        # RUN ALL ABLATION TYPES
        # =====================================================================
        self.run_sequence_length_ablation()
        self.run_model_size_ablation()

        # =====================================================================
        # SAVE COMBINED RESULTS
        # =====================================================================
        self.save_all_results()

        return self.results

    def save_all_results(self):
        """
        Save all results to a combined JSON file.

        This creates a single file with all experiment results, useful for:
        - Loading and analyzing results in other tools (e.g., Jupyter notebook)
        - Archiving experiment results
        - Comparing across ablation studies
        """
        results_path = self.output_dir / "all_results.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_experiments": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.results)} results to {results_path}")

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self) -> str:
        """
        Generate a markdown report summarizing ablation results.

        The report includes:
        - Summary table with all experiments ranked by performance
        - Best configuration identification
        - Detailed tables for each ablation type

        The report is saved to disk and returned as a string.

        Returns:
            Markdown-formatted report as a string.
        """
        if not self.results:
            return "No results to report."

        lines = [
            "# Ablation Study Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            "",
            # Markdown table header
            "| Experiment | Spread MAE | Win Acc | Win AUC | ECE | Training Time |",
            "|------------|------------|---------|---------|-----|---------------|",
        ]

        # Sort by spread MAE (lower is better)
        sorted_results = sorted(self.results, key=lambda r: r.metrics.spread_mae)

        # Add row for each experiment
        for r in sorted_results:
            time_str = f"{r.training_time_seconds / 60:.1f} min"
            lines.append(
                f"| {r.name} | {r.metrics.spread_mae:.2f} | "
                f"{r.metrics.win_accuracy:.3f} | {r.metrics.win_auc:.3f} | "
                f"{r.metrics.ece:.4f} | {time_str} |"
            )

        # Highlight best configuration
        lines.extend([
            "",
            "## Best Configuration",
            "",
            f"**{sorted_results[0].name}** achieved the lowest Spread MAE "
            f"of {sorted_results[0].metrics.spread_mae:.2f}",
            "",
        ])

        # =====================================================================
        # DETAILED TABLES BY ABLATION TYPE
        # =====================================================================
        lines.extend([
            "## Ablation Details",
            "",
        ])

        # Sequence length results
        seq_results = [r for r in self.results if r.name.startswith("seq_len_")]
        if seq_results:
            lines.extend([
                "### Sequence Length",
                "",
                "| N Games | Spread MAE | Win Acc |",
                "|---------|------------|---------|",
            ])
            for r in sorted(seq_results, key=lambda x: x.config_changes.get("n_history_games", 0)):
                lines.append(
                    f"| {r.config_changes.get('n_history_games', '?')} | "
                    f"{r.metrics.spread_mae:.2f} | {r.metrics.win_accuracy:.3f} |"
                )
            lines.append("")

        # Model size results
        size_results = [r for r in self.results if r.name.startswith("size_")]
        if size_results:
            lines.extend([
                "### Model Size",
                "",
                "| Size | Spread MAE | Win Acc | Training Time |",
                "|------|------------|---------|---------------|",
            ])
            for r in size_results:
                time_str = f"{r.training_time_seconds / 60:.1f} min"
                lines.append(
                    f"| {r.config_changes.get('model_size', '?')} | "
                    f"{r.metrics.spread_mae:.2f} | {r.metrics.win_accuracy:.3f} | {time_str} |"
                )
            lines.append("")

        # Combine lines and save
        report = "\n".join(lines)

        report_path = self.output_dir / "ablation_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved report to {report_path}")
        return report


# =============================================================================
# CONVENIENCE FUNCTION FOR QUICK TESTING
# =============================================================================

def run_quick_ablation(device: str = "cuda"):
    """
    Run a quick ablation study with small models for testing.

    This is useful for:
    - Verifying the ablation pipeline works correctly
    - Quick experiments during development
    - Testing on machines with limited resources

    Uses small models and few epochs to complete quickly (~30-60 min on GPU).

    Args:
        device: Device for training ("cuda" or "cpu").

    Returns:
        List of AblationResult objects.
    """
    logger.info("Running quick ablation study...")

    # Use small configuration for speed
    base_config = get_small_config()

    # Reduce training for faster iteration
    base_config.training.max_epochs = 5
    base_config.training.patience = 3  # Early stop after 3 epochs without improvement

    # Use smaller data subset
    base_config.data.train_seasons = ["2022-2023"]
    base_config.data.val_seasons = ["2023-2024"]
    base_config.data.test_seasons = ["2023-2024"]

    # Create runner and execute
    runner = AblationRunner(
        base_config=base_config,
        output_dir="experiments/quick_ablation",
        device=device,
    )

    results = runner.run_all()
    report = runner.generate_report()
    print(report)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_quick_ablation(device)
