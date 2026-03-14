"""Train Phase 4 generative model.

Usage::

    python scripts/train_generative.py --config configs/generative/exp1_baseline.yaml
    python scripts/train_generative.py --config configs/generative/exp1_baseline.yaml --resume checkpoints/generative/gen_exp1_baseline/best.pt
"""

import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.models.generative_model import GenerativeModel
from src.generative.training.trainer import GenerativeTrainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 4 generative model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume", default=None, help="Resume training from checkpoint path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ---- Config ------------------------------------------------------------
    config = GenerativeExperimentConfig.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")

    # ---- Reproducibility ---------------------------------------------------
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Datasets ----------------------------------------------------------
    use_simple = config.model.use_simplified_context
    use_compressed = config.model.use_scoring_events_only
    max_se = config.model.max_scoring_events
    ds_kwargs = dict(
        use_simplified_context=use_simple,
        use_scoring_events_only=use_compressed,
        max_scoring_events=max_se,
    )
    train_ds = GenerativeDataset(config.data, split="train", **ds_kwargs)
    val_ds = GenerativeDataset(config.data, split="val", **ds_kwargs)
    test_ds = GenerativeDataset(config.data, split="test", **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=generative_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=generative_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=generative_collate,
    )

    logger.info(
        f"Train: {len(train_ds)} games, Val: {len(val_ds)} games, Test: {len(test_ds)} games"
    )

    # ---- Model -------------------------------------------------------------
    model = GenerativeModel(config.model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ---- Trainer -----------------------------------------------------------
    trainer = GenerativeTrainer(model, config, train_loader, val_loader, test_loader)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ---- Train -------------------------------------------------------------
    final_metrics = trainer.train()
    logger.info(f"Training complete. Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
