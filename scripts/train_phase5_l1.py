#!/usr/bin/env python3
"""
Phase 5 L1: NKE-H Training Script.

Two-phase training:
  Phase 1: Hierarchy pre-training on single games (2001-2017)
  Phase 2: Sequential Kalman training on career sequences (2001-2017)

Usage:
    python scripts/train_phase5_l1.py --phase 1          # Phase 1 only
    python scripts/train_phase5_l1.py --phase 2          # Phase 2 (requires Phase 1 checkpoint)
    python scripts/train_phase5_l1.py --phase both       # Full training
    python scripts/train_phase5_l1.py --phase 1 --epochs 5 --debug  # Quick test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.config import NKEHConfig
from src.phase5.model import NKEH, count_parameters
from src.phase5.dataset import (
    SingleGameDataset,
    CareerSequenceDataset,
    load_archetypes,
    load_metadata,
    load_profiles,
)
from src.phase5.losses import phase1_loss, phase2_loss

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"


def get_player_ids(metadata: dict) -> list[int]:
    """Get all player IDs from the cache."""
    profiles = load_profiles()
    return [int(pid) for pid in profiles["person_ids"]]


# ---------------------------------------------------------------------------
# Phase 1: Hierarchy pre-training
# ---------------------------------------------------------------------------


def train_phase1(
    cfg: NKEHConfig,
    epochs: int = 50,
    device: str = "cuda",
    debug: bool = False,
):
    """Phase 1: Train on single games to learn hierarchy + decoder heads."""
    logging.info("=== Phase 1: Hierarchy Pre-Training ===")

    metadata = load_metadata()
    player_ids = get_player_ids(metadata)

    # Datasets
    train_ds = SingleGameDataset(player_ids, metadata, split="pretrain")
    val_ds = SingleGameDataset(player_ids, metadata, split="pretrain_val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logging.info(f"Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")

    # Validate config dimensions against cache metadata
    assert cfg.n_box_stats == len(
        metadata["box_stat_columns"]
    ), f"Config n_box_stats={cfg.n_box_stats} != cache {len(metadata['box_stat_columns'])}"
    assert cfg.n_pbp_stats == len(
        metadata["pbp_stat_columns"]
    ), f"Config n_pbp_stats={cfg.n_pbp_stats} != cache {len(metadata['pbp_stat_columns'])}"
    assert cfg.n_context == len(
        metadata["context_columns"]
    ), f"Config n_context={cfg.n_context} != cache {len(metadata['context_columns'])}"
    assert cfg.n_profile == len(
        metadata["profile_columns"]
    ), f"Config n_profile={cfg.n_profile} != cache {len(metadata['profile_columns'])}"

    # Model — initialize archetypes before moving to device
    model = NKEH(cfg)

    # Initialize archetype prototypes from pre-computed k-means centroids
    try:
        centroids = load_archetypes()
        model.archetype_network.initialize_from_centroids(centroids)
        logging.info(
            f"Initialized archetype prototypes from centroids "
            f"({centroids.shape[0]}x{centroids.shape[1]} -> {cfg.n_archetypes}x{cfg.d_ability})"
        )
    except FileNotFoundError:
        logging.warning("Archetype centroids not found, using random initialization")

    model = model.to(device)

    logging.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = {}
        n_batches = 0

        for batch in train_loader:
            outputs = model.forward_single_game(
                box_stats=batch["box_stats"].to(device),
                pbp_stats=batch["pbp_stats"].to(device),
                context=batch["context"].to(device),
                profile=batch["profile"].to(device),
            )

            losses = phase1_loss(outputs, batch, cfg)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()

            for k, v in losses.items():
                train_losses[k] = train_losses.get(k, 0) + v.item()
            n_batches += 1

            if debug and n_batches >= 5:
                break

        scheduler.step()

        # Average train losses
        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_losses = {}
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model.forward_single_game(
                    box_stats=batch["box_stats"].to(device),
                    pbp_stats=batch["pbp_stats"].to(device),
                    context=batch["context"].to(device),
                    profile=batch["profile"].to(device),
                )
                losses = phase1_loss(outputs, batch, cfg)
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v.item()
                n_val += 1

                if debug and n_val >= 3:
                    break

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        # --- Logging ---
        lr = optimizer.param_groups[0]["lr"]
        arch_ent_str = ""
        if "archetype_entropy" in train_losses:
            arch_ent_str = f" arch_ent={train_losses['archetype_entropy']:.4f}"
        logging.info(
            f"Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | "
            f"Train: total={train_losses['total']:.4f} recon={train_losses['recon']:.4f} "
            f"next={train_losses['next_game']:.4f} dpm={train_losses['dpm']:.4f}"
            f"{arch_ent_str} | "
            f"Val: total={val_losses['total']:.4f} recon={val_losses['recon']:.4f}"
        )

        # --- Checkpointing ---
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_losses,
                CHECKPOINT_DIR / "phase1_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    logging.info(f"Phase 1 complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Phase 2: Sequential Kalman training
# ---------------------------------------------------------------------------


def train_phase2(
    cfg: NKEHConfig,
    epochs: int = 30,
    device: str = "cuda",
    debug: bool = False,
    checkpoint_path: Path | None = None,
):
    """Phase 2: Train on career sequences to learn Kalman dynamics."""
    logging.info("=== Phase 2: Sequential Kalman Training ===")

    metadata = load_metadata()
    player_ids = get_player_ids(metadata)

    # Datasets
    train_ds = CareerSequenceDataset(
        player_ids,
        metadata,
        max_len=cfg.max_career_len,
        split="pretrain",
    )
    val_ds = CareerSequenceDataset(
        player_ids,
        metadata,
        max_len=cfg.max_career_len,
        split="pretrain_val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.seq_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.seq_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logging.info(
        f"Train: {len(train_ds)} players, Val: {len(val_ds)} players, "
        f"max_len={cfg.max_career_len}"
    )

    # Model — load from Phase 1 checkpoint
    model = NKEH(cfg).to(device)

    if checkpoint_path and checkpoint_path.exists():
        logging.info(f"Loading Phase 1 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logging.warning("No Phase 1 checkpoint found. Training from scratch.")

    logging.info(f"Model parameters: {count_parameters(model):,}")

    # Lower learning rate for Phase 2
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate * 0.3,  # 3e-4 → ~1e-4
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_losses = {}
        n_batches = 0

        for batch in train_loader:
            outputs = model.forward_sequence(
                box_stats_seq=batch["box_stats"].to(device),
                pbp_stats_seq=batch["pbp_stats"].to(device),
                context_seq=batch["context"].to(device),
                profile=batch["profile"].to(device),
                age_seq=batch["age"].to(device),
                seq_mask=batch["mask"].to(device),
            )

            losses = phase2_loss(outputs, batch, cfg)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()

            for k, v in losses.items():
                train_losses[k] = train_losses.get(k, 0) + v.item()
            n_batches += 1

            if debug and n_batches >= 3:
                break

        scheduler.step()

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_losses = {}
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model.forward_sequence(
                    box_stats_seq=batch["box_stats"].to(device),
                    pbp_stats_seq=batch["pbp_stats"].to(device),
                    context_seq=batch["context"].to(device),
                    profile=batch["profile"].to(device),
                    age_seq=batch["age"].to(device),
                    seq_mask=batch["mask"].to(device),
                )
                losses = phase2_loss(outputs, batch, cfg)
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v.item()
                n_val += 1

                if debug and n_val >= 2:
                    break

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | "
            f"Train: total={train_losses['total']:.4f} recon={train_losses['recon']:.4f} "
            f"dpm={train_losses['dpm']:.4f} | "
            f"Val: total={val_losses['total']:.4f}"
        )

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_losses,
                CHECKPOINT_DIR / "phase2_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    logging.info(f"Phase 2 complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_losses: dict,
    path: Path,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_losses": val_losses,
        },
        path,
    )
    logging.info(f"Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train Phase 5 L1 NKE-H model")
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Training phase: 1, 2, or both",
    )
    parser.add_argument("--epochs1", type=int, default=100, help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=60, help="Phase 2 epochs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="Quick test (few batches)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    cfg = NKEHConfig()

    if args.phase in ("1", "both"):
        train_phase1(cfg, epochs=args.epochs1, device=device, debug=args.debug)

    if args.phase in ("2", "both"):
        train_phase2(
            cfg,
            epochs=args.epochs2,
            device=device,
            debug=args.debug,
            checkpoint_path=CHECKPOINT_DIR / "phase1_best.pt",
        )


if __name__ == "__main__":
    main()
