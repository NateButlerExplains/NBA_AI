#!/usr/bin/env python
"""
Training Script for NBA Transformer Model.

This is the ENTRY POINT for training — you run this from the command line.
It handles:
    1. Loading configuration (from YAML file, preset, or quick-test defaults)
    2. Building the tokenizer (vocabulary for play-by-play text)
    3. Creating train/val/test data loaders (with temporal split to prevent leakage)
    4. Constructing the model
    5. Running the Trainer (which handles the actual training loop)
    6. Printing final evaluation results

Usage:
    # Train with a config file (required for real training)
    python scripts/train_transformer.py --config configs/transformer/full_baseline.yaml

    # Quick test run (small model, 1 season, 2 epochs — for fast iteration/debugging)
    python scripts/train_transformer.py --quick-test

    # Resume from checkpoint (continues where training left off)
    python scripts/train_transformer.py --resume checkpoints/transformer/baseline/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to Python's import search path so we can import src.transformer.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.transformer.tokenizer import PBPTokenizer
from src.transformer.dataset import NBAGameDataset, collate_games
from src.transformer.models import Phase1Model
from src.transformer.training.config import (
    ExperimentConfig,
    get_small_config,
)
from src.transformer.training.trainer import Trainer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_or_build_tokenizer(config: ExperimentConfig) -> PBPTokenizer:
    """
    Load existing tokenizer or build a new one from the training data.

    The tokenizer converts play-by-play text into numerical token IDs that the
    model can process. It is built ONLY from training data to prevent data leakage
    (test season vocabulary does not influence the tokenizer).

    The tokenizer is saved to disk after building, so subsequent training runs
    skip the expensive build step.
    """
    tokenizer = PBPTokenizer()
    tokenizer_path = Path(config.data.tokenizer_path)

    if tokenizer_path.exists():
        # Tokenizer already built — just load it (fast)
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer.load(str(tokenizer_path))
    else:
        # First time: scan all training data to build vocabulary
        logging.info("Building tokenizer from database...")
        # IMPORTANT: only use TRAINING seasons for vocabulary to prevent data leakage
        tokenizer.build_vocab_from_db(
            seasons=config.data.train_seasons,
            min_count=config.data.min_token_count,
        )
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        logging.info(f"Saved tokenizer to {tokenizer_path}")

    logging.info(f"Vocabulary sizes: {tokenizer.vocab_sizes}")
    return tokenizer


def create_dataloaders(
    config: ExperimentConfig,
    tokenizer: PBPTokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    TEMPORAL SPLIT EXPLAINED:
        We split by season (not randomly!) to prevent data leakage:
        - Train: 2000-2023 (older seasons — the model learns from these)
        - Validation: 2023-2024 (recent — used to tune hyperparameters and early stopping)
        - Test: 2024-2026 (newest — only evaluated ONCE at the end for final results)

        If we randomly split, the model could see a team's December game in training
        and their November game in testing, which leaks future information.
    """

    logging.info("Creating training dataset...")
    train_dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=config.data.train_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Train dataset: {len(train_dataset)} games")

    logging.info("Creating validation dataset...")
    val_dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=config.data.val_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Validation dataset: {len(val_dataset)} games")

    logging.info("Creating test dataset...")
    test_dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=config.data.test_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Test dataset: {len(test_dataset)} games")

    # Create data loaders — these handle batching, shuffling, and parallel data loading.
    # A DataLoader wraps a Dataset and yields batches of data during training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,             # Shuffle training data each epoch (prevents ordering bias)
        collate_fn=collate_games, # Custom function to pad variable-length sequences into batches
        num_workers=config.data.num_workers,  # Parallel data loading threads
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),  # Faster CPU->GPU transfer
    )

    # Validation and test loaders do NOT shuffle — we want deterministic evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_games,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_games,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def create_model(config: ExperimentConfig, tokenizer: PBPTokenizer) -> Phase1Model:
    """
    Create the transformer model from configuration and tokenizer.

    The tokenizer provides vocab_sizes (how many unique tokens exist for each
    field type), which the model needs to create its embedding layers.
    """
    # Inject vocabulary sizes from the tokenizer into the model config.
    # vocab_sizes is the only field not specified in YAML — it comes from the data.
    config.model.vocab_sizes = tokenizer.vocab_sizes

    model = Phase1Model(config.model)

    # Log parameter count — gives a sense of model size.
    # "parameters" = the individual numbers (weights) the model learns.
    # A 5M-parameter model has 5 million learnable numbers.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train NBA Transformer Model")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with small model and limited data",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # --- Load configuration ---
    # Two modes: (1) custom YAML file, (2) quick test for debugging
    if args.config:
        # Load a fully custom configuration from a YAML file
        logger.info(f"Loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    elif args.quick_test:
        # Quick test mode: tiny model, minimal data, 2 epochs.
        # Use this for fast iteration — verify code works before long training runs.
        # Takes ~1-2 minutes instead of hours.
        logger.info("Using quick test configuration")
        config = get_small_config()
        # Use a single season for all splits (not a valid ML setup, but fast for testing)
        config.data.train_seasons = ["2023-2024"]
        config.data.val_seasons = ["2023-2024"]
        config.data.test_seasons = ["2023-2024"]
        config.data.tokenizer_path = "data/tokenized/test_tokenizer.json"
        config.data.batch_size = 4       # Very small batches for minimal memory
        config.data.num_workers = 0      # No parallel loading (simpler debugging)
        config.training.max_epochs = 2   # Just 2 epochs to verify training loop works
        config.training.patience = 5
        config.training.log_every_n_steps = 10
        config.training.eval_every_n_steps = 50
        config.training.experiment_name = "quick_test"
    else:
        parser.error("--config is required. Use --quick-test for debugging.\n"
                      "  Example: python scripts/train_transformer.py --config configs/transformer/full_baseline.yaml")

    # Apply command-line overrides (these take priority over config file values)
    if args.experiment_name:
        logger.info(f"CLI override: experiment_name = {args.experiment_name}")
        config.training.experiment_name = args.experiment_name
    if args.device:
        logger.info(f"CLI override: device = {args.device}")
        config.training.device = args.device
    if args.seed is not None:
        logger.info(f"CLI override: seed = {args.seed}")
        config.training.seed = args.seed

    # Set random seed for reproducibility — ensures the same random initialization,
    # data shuffling, and dropout masks every time you run with the same seed.
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)  # Seed all GPUs

    logger.info(f"Experiment: {config.training.experiment_name}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Seed: {config.training.seed}")

    # Load tokenizer
    tokenizer = load_or_build_tokenizer(config)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

    # Create model
    model = create_model(config, tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Resume from a previously saved checkpoint if specified.
    # This restores model weights, optimizer state, scheduler position, and
    # training progress (epoch number, best loss, etc.) — training continues
    # exactly where it left off.
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Run the full training loop (see Trainer.train() in trainer.py)
    logger.info("Starting training...")
    final_metrics = trainer.train()

    # Print final evaluation results (on test set if available, otherwise validation)
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Spread MAE: {final_metrics.spread_mae:.2f}")
    print(f"Spread RMSE: {final_metrics.spread_rmse:.2f}")
    print(f"Home MAE: {final_metrics.home_mae:.2f}")
    print(f"Away MAE: {final_metrics.away_mae:.2f}")
    print(f"Win Accuracy: {final_metrics.win_accuracy:.3f}")
    print(f"Win AUC: {final_metrics.win_auc:.3f}")
    print(f"Brier Score: {final_metrics.brier_score:.4f}")
    print(f"ECE: {final_metrics.ece:.4f}")
    print(f"90% Coverage: {final_metrics.spread_coverage_90:.3f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
