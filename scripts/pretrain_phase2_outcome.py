#!/usr/bin/env python
"""
Per-Match Outcome Pre-training Script (Phase 3 Experiment 6).

Trains encoder to predict win/loss + margin from player stats (scores masked).

Usage:
    python scripts/pretrain_phase2_outcome.py \
        --config configs/transformer/phase3_exp6_pretrain.yaml
"""

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import load_cache
from src.transformer.phase2.pretrain.outcome_model import OutcomePretrainModel
from src.transformer.phase2.pretrain.outcome_dataset import OutcomePretrainDataset, collate_outcome


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Per-match outcome pre-training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    config = Phase2ExperimentConfig.from_yaml(args.config)
    if args.device:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Load cache
    cache = load_cache(config.data.cache_dir)
    game_features = cache["game_features"]
    player_id_map = cache.get("player_id_map", {})
    log.info(f"Cache loaded: {len(game_features)} games, {len(player_id_map)} players")

    # Datasets
    train_ds = OutcomePretrainDataset(
        game_features=game_features,
        player_id_map=player_id_map,
        seasons=config.data.train_seasons,
        max_players_per_game=config.data.max_players_per_game,
        n_player_stats=config.data.n_player_stats,
    )
    val_ds = OutcomePretrainDataset(
        game_features=game_features,
        player_id_map=player_id_map,
        seasons=config.data.val_seasons,
        max_players_per_game=config.data.max_players_per_game,
        n_player_stats=config.data.n_player_stats,
    )

    pin_memory = config.data.pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True,
        collate_fn=collate_outcome, num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size, shuffle=False,
        collate_fn=collate_outcome, num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    # Model
    model = OutcomePretrainModel(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"OutcomePretrainModel: {total_params:,} parameters")

    # Device
    device_str = config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model = model.to(device)

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embedding" in name or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW([
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.optimizer.learning_rate, betas=tuple(config.optimizer.betas),
       eps=config.optimizer.eps)

    # Scheduler
    accum = config.training.gradient_accumulation_steps
    steps_per_epoch = max(len(train_loader) // accum, 1)
    total_steps = steps_per_epoch * config.training.max_epochs
    warmup_steps = int(total_steps * config.optimizer.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup_steps, 1))
    min_lr = config.optimizer.learning_rate * config.optimizer.min_lr_ratio
    main_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[warmup_steps])

    # AMP
    use_amp = config.training.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    amp_dtype = getattr(torch, config.training.amp_dtype) if use_amp else None

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    margin_weight = 0.3

    # Checkpointing
    checkpoint_dir = Path(config.training.checkpoint_dir) / config.training.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(checkpoint_dir / "config.yaml")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    smoothing_window = collections.deque(maxlen=config.training.smoothing_window)

    for epoch in range(config.training.max_epochs):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if use_amp:
                with autocast("cuda", dtype=amp_dtype):
                    out = model(batch)
                    loss_bce = bce_loss(out["win_logit"], batch["target_win"])
                    loss_mse = mse_loss(out["margin_pred"], batch["target_margin"])
                    loss = loss_bce + margin_weight * loss_mse
                    scaled = loss / accum
                scaler.scale(scaled).backward()
            else:
                out = model(batch)
                loss_bce = bce_loss(out["win_logit"], batch["target_win"])
                loss_mse = mse_loss(out["margin_pred"], batch["target_margin"])
                loss = loss_bce + margin_weight * loss_mse
                (loss / accum).backward()

            train_loss_sum += loss.item()
            n_train += 1

            if (batch_idx + 1) % accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Final accumulation step
        if n_train % accum != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss_sum / max(n_train, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if use_amp:
                    with autocast("cuda", dtype=amp_dtype):
                        out = model(batch)
                        loss_bce = bce_loss(out["win_logit"], batch["target_win"])
                        loss_mse = mse_loss(out["margin_pred"], batch["target_margin"])
                        loss = loss_bce + margin_weight * loss_mse
                else:
                    out = model(batch)
                    loss_bce = bce_loss(out["win_logit"], batch["target_win"])
                    loss_mse = mse_loss(out["margin_pred"], batch["target_margin"])
                    loss = loss_bce + margin_weight * loss_mse

                val_loss_sum += loss.item()
                n_val += 1

                # Accuracy
                preds = (out["win_logit"] > 0).float()
                val_correct += (preds == batch["target_win"]).sum().item()
                val_total += batch["target_win"].shape[0]

        val_loss = val_loss_sum / max(n_val, 1)
        val_acc = val_correct / max(val_total, 1)
        epoch_time = time.time() - epoch_start

        log.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.3f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"time={epoch_time:.0f}s"
        )

        # Early stopping
        smoothing_window.append(val_loss)
        smoothed = sum(smoothing_window) / len(smoothing_window)

        if smoothed < best_val_loss - config.training.min_delta:
            best_val_loss = smoothed
            patience_counter = 0

            # Save transferable weights
            state = model.get_transferable_state_dict()
            torch.save(state, checkpoint_dir / "transferable_weights.pt")
            log.info(f"  New best! Smoothed val_loss: {smoothed:.4f}, saved {len(state)} tensors")
        else:
            patience_counter += 1

        if patience_counter >= config.training.patience:
            log.info(f"Early stopping at epoch {epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("OUTCOME PRE-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Transferable weights: {checkpoint_dir / 'transferable_weights.pt'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
