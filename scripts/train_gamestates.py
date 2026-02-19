#!/usr/bin/env python
"""
Training Script for GameStates Model.

Same structure as train_transformer.py but uses GameStates score trajectories
instead of play-by-play data.

Usage:
    python scripts/train_gamestates.py --config configs/transformer/gamestates_2season.yaml
    python scripts/train_gamestates.py --quick-test
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.transformer.gamestates.tokenizer import GameStateTokenizer
from src.transformer.gamestates.dataset import GameStatesDataset, collate_gamestates
from src.transformer.gamestates.model import GameStateModel
from src.transformer.training.config import ExperimentConfig, ModelConfig


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_dataloaders(
    config: ExperimentConfig,
    tokenizer: GameStateTokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    logging.info("Creating training dataset...")
    train_dataset = GameStatesDataset(
        tokenizer=tokenizer,
        seasons=config.data.train_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Train dataset: {len(train_dataset)} games")

    logging.info("Creating validation dataset...")
    val_dataset = GameStatesDataset(
        tokenizer=tokenizer,
        seasons=config.data.val_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Validation dataset: {len(val_dataset)} games")

    logging.info("Creating test dataset...")
    test_dataset = GameStatesDataset(
        tokenizer=tokenizer,
        seasons=config.data.test_seasons,
        n_history_games=config.data.n_history_games,
        min_history_games=config.data.min_history_games,
        cache_size=config.data.cache_size,
    )
    logging.info(f"Test dataset: {len(test_dataset)} games")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_gamestates,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_gamestates,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_gamestates,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def create_model(config: ExperimentConfig) -> GameStateModel:
    """Create the GameStates model from configuration."""
    # GameStates model uses fixed vocab sizes — no tokenizer injection needed.
    # But ModelConfig validation expects PBP vocab keys, so we inject dummy values
    # to pass validation. The GameStateModel ignores vocab_sizes entirely.
    config.model.vocab_sizes = {
        "action_type": 1, "sub_type": 1, "player": 1, "period": 5,
        "clock_bucket": 721, "team_indicator": 3, "score_diff_bucket": 121,
        "shot_result": 3, "shot_distance_bucket": 11, "shot_modifier": 1,
    }

    model = GameStateModel(config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    param_counts = model.get_num_parameters()
    for name, count in param_counts.items():
        logging.info(f"  {name}: {count:,}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GameStates NBA Model")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with small model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    elif args.quick_test:
        logger.info("Using quick test configuration")
        config = ExperimentConfig()
        config.model.hidden_dim = 128
        config.model.event_encoder_layers = 2
        config.model.event_encoder_heads = 4
        config.model.temporal_attention_layers = 1
        config.data.train_seasons = ["2023-2024"]
        config.data.val_seasons = ["2023-2024"]
        config.data.test_seasons = ["2023-2024"]
        config.data.batch_size = 4
        config.data.num_workers = 0
        config.training.max_epochs = 2
        config.training.patience = 5
        config.training.log_every_n_steps = 10
        config.training.eval_every_n_steps = 50
        config.training.experiment_name = "gamestates_quick_test"
    else:
        parser.error("--config is required. Use --quick-test for debugging.\n"
                      "  Example: python scripts/train_gamestates.py --config configs/transformer/gamestates_2season.yaml")

    if args.experiment_name:
        config.training.experiment_name = args.experiment_name
    if args.device:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    logger.info(f"Experiment: {config.training.experiment_name}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Seed: {config.training.seed}")

    # GameStates tokenizer has fixed vocab — no build/load step
    tokenizer = GameStateTokenizer()

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer)

    # Create model
    model = create_model(config)

    # Import Trainer — it works with any model that accepts (home_history, away_history)
    from src.transformer.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    logger.info("Starting training...")
    final_metrics = trainer.train()

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
