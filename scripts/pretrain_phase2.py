#!/usr/bin/env python
"""
Pre-training Script for Phase 3 Experiment 2.

BERT-style masked reconstruction on all available seasons.

Usage:
    # Build pre-training cache first:
    python -m src.transformer.phase2.cache_builder \
        --seasons 2001-2002 2002-2003 ... 2025-2026 \
        --cache-dir data/pretrain_cache --skip-gamestates

    # Run pre-training:
    python scripts/pretrain_phase2.py --config configs/transformer/phase3_exp2_pretrain.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import build_cache, load_cache
from src.transformer.phase2.pretrain.dataset import PretrainDataset, collate_pretrain
from src.transformer.phase2.pretrain.model import PretrainModel
from src.transformer.phase2.pretrain.trainer import PretrainTrainer


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_cache(config: Phase2ExperimentConfig) -> dict:
    """Load cache if it exists, otherwise build it (no GameStates)."""
    cache_dir = config.data.cache_dir
    cache_path = Path(cache_dir)

    if (cache_path / "game_features.pt").exists():
        logging.info(f"Loading existing cache from {cache_dir}")
        return load_cache(cache_dir)
    else:
        all_seasons = sorted(set(
            config.data.train_seasons + config.data.val_seasons
        ))
        logging.info(f"Building pre-training cache for {len(all_seasons)} seasons...")
        build_cache(all_seasons, cache_dir, skip_gamestates=True)
        return load_cache(cache_dir)


def main():
    parser = argparse.ArgumentParser(description="Pre-train Phase 2 NBA Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    # Load config
    log.info(f"Loading config from {args.config}")
    config = Phase2ExperimentConfig.from_yaml(args.config)

    if args.device:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Load cache
    cache = ensure_cache(config)

    # The cache may not have gs_cache when built with --skip-gamestates
    # load_cache will try to load it; handle missing gracefully
    game_features = cache["game_features"]
    season_index = cache["season_index"]
    player_id_map = cache.get("player_id_map", {})

    log.info(f"Cache loaded: {len(game_features)} games, {len(player_id_map)} players, "
             f"{len(season_index)} team-season pairs")

    # Create datasets
    train_dataset = PretrainDataset(
        game_features=game_features,
        season_index=season_index,
        player_id_map=player_id_map,
        seasons=config.data.train_seasons,
        mask_ratio=config.data.mask_ratio,
        max_players_per_game=config.data.max_players_per_game,
    )

    val_dataset = PretrainDataset(
        game_features=game_features,
        season_index=season_index,
        player_id_map=player_id_map,
        seasons=config.data.val_seasons,
        mask_ratio=config.data.mask_ratio,
        max_players_per_game=config.data.max_players_per_game,
    )

    pin_memory = config.data.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_pretrain,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_pretrain,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    # Create model
    model = PretrainModel(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"PretrainModel: {total_params:,} parameters")

    # Create trainer and train
    trainer = PretrainTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    best_mse = trainer.train()

    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best val MSE: {best_mse:.6f}")
    print(f"Transferable weights saved to: {trainer.checkpoint_dir / 'transferable_weights.pt'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
