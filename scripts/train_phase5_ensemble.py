#!/usr/bin/env python3
"""
Phase C Ensemble: Train 3 Phase C models with different seeds, ensemble predictions.

Reuses the same Phase B cache and L1 vectors (data loading is done once).
For each seed: loads Phase B pretrained weights, adds random weight perturbation,
trains Phase C for 20 epochs with gradual unfreezing.

Ensemble aggregation:
  - spread_mu: simple average
  - win_prob: logit averaging
  - ATS: pick from ensemble spread_mu vs Vegas line

Usage:
    python scripts/train_phase5_ensemble.py
    python scripts/train_phase5_ensemble.py --epochs 20 --seeds 42,137,256
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.l2_config import L2Config
from src.phase5.l2_model import (
    PlayerSynergyNetwork,
    count_parameters as l2_count_params,
)
from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel, count_parameters as l3_count_params
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor, count_parameters as l4_count_params
from src.phase5.l4_losses import game_prediction_loss

# Reuse Phase C data-building infrastructure
from scripts.train_phase5_c import (
    PhaseCDataset,
    build_phase_c_data,
    make_train_val_datasets,
    season_start_year,
    forward_l2_l3_l4,
    apply_freeze_schedule,
    DriftMonitor,
    load_wowy_lookup,
    build_years_together_lookup,
    load_birth_years,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"

# ATS constants
WIN_PAYOUT = 100 / 1.10  # ~$90.91 profit on a $100 bet
LOSS_COST = 100.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vegas spread loading
# ---------------------------------------------------------------------------


def load_vegas_spreads() -> dict[str, float]:
    """Load Vegas closing spreads from Betting table."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute("""
        SELECT game_id,
               COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread)
        FROM Betting
        WHERE COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread)
              IS NOT NULL
    """).fetchall()
    conn.close()
    spreads = {gid: float(spread) for gid, spread in rows}
    logger.info(f"Loaded Vegas spreads for {len(spreads)} games")
    return spreads


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    all_preds: dict[str, list],
    all_targets: dict[str, list],
) -> dict[str, float]:
    """Compute evaluation metrics from accumulated predictions/targets."""
    spread_mu = np.concatenate(all_preds["spread_mu"])
    spread_sigma = np.concatenate(all_preds["spread_sigma"])
    win_prob = np.concatenate(all_preds["win_prob"])
    total_mu = np.concatenate(all_preds["total_mu"])

    actual_margin = np.concatenate(all_targets["margin"])
    actual_win = np.concatenate(all_targets["win"])
    actual_total = np.concatenate(all_targets["total"])

    n = len(actual_margin)

    spread_mae = np.abs(spread_mu - actual_margin).mean()
    total_mae = np.abs(total_mu - actual_total).mean()

    win_pred = (win_prob > 0.5).astype(float)
    win_acc = (win_pred == actual_win).mean()

    # AUC
    order = np.argsort(win_prob)
    sorted_labels = actual_win[order]
    n_pos = int(actual_win.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(sorted_labels)
        auc = 1.0 - cum_pos[sorted_labels == 0].sum() / (n_pos * n_neg)
    else:
        auc = 0.5

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        mask = (win_prob >= bin_boundaries[j]) & (win_prob < bin_boundaries[j + 1])
        if j == n_bins - 1:
            mask = mask | (win_prob == bin_boundaries[j + 1])
        if mask.sum() > 0:
            avg_confidence = win_prob[mask].mean()
            avg_accuracy = actual_win[mask].mean()
            ece += mask.sum() / n * abs(avg_confidence - avg_accuracy)

    mean_sigma = spread_sigma.mean()

    return {
        "spread_mae": float(spread_mae),
        "total_mae": float(total_mae),
        "win_acc": float(win_acc),
        "win_auc": float(auc),
        "ece": float(ece),
        "mean_sigma": float(mean_sigma),
        "n_games": n,
    }


def compute_ats_metrics(
    spread_mu: np.ndarray,
    actual_margin: np.ndarray,
    vegas_spread: np.ndarray,
    has_vegas: np.ndarray,
) -> dict[str, float]:
    """Compute ATS metrics from spread-based picks."""
    valid = has_vegas.astype(bool)
    if not valid.any():
        return {"ats_pct": 0.0, "ats_total": 0, "ats_correct": 0, "roi": 0.0}

    vm = actual_margin[valid]
    vs = vegas_spread[valid]
    pm = spread_mu[valid]

    # Did home actually cover?
    ats_result = vm - (-vs)  # = vm + vs
    home_covered = ats_result > 0

    # Exclude pushes
    not_push = np.abs(ats_result) > 0.25
    vm = vm[not_push]
    vs = vs[not_push]
    pm = pm[not_push]
    home_covered = home_covered[not_push]

    # Our pick: home covers when pred_margin > -vegas_spread
    our_pick_home = pm > (-vs)

    correct = (our_pick_home == home_covered).sum()
    total = len(vm)
    ats_pct = correct / max(total, 1)

    # ROI at -110
    wins = correct
    losses = total - correct
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = profit / (total * LOSS_COST) * 100

    return {
        "ats_pct": float(ats_pct),
        "ats_correct": int(correct),
        "ats_total": int(total),
        "roi": float(roi),
    }


# ---------------------------------------------------------------------------
# Train/eval helpers
# ---------------------------------------------------------------------------


def train_epoch(
    l2_model, l3_model, l4_model, loader, optimizer, l4_cfg, device, grad_clip=1.0
):
    """Train one epoch."""
    l2_model.train()
    l3_model.train()
    l4_model.train()

    total_losses = {}
    n_batches = 0

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)

        predictions = forward_l2_l3_l4(batch, l2_model, l3_model, l4_model, device)

        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            actual_total=actual_total,
            cfg=l4_cfg,
        )

        optimizer.zero_grad()
        losses["total_loss"].backward()

        all_params = (
            list(l2_model.parameters())
            + list(l3_model.parameters())
            + list(l4_model.parameters())
        )
        trainable_params = [p for p in all_params if p.requires_grad]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(l2_model, l3_model, l4_model, loader, l4_cfg, device):
    """Evaluate on a dataset. Returns (loss_dict, metrics_dict, raw_preds)."""
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()

    total_losses = {}
    n_batches = 0

    all_preds = {"spread_mu": [], "spread_sigma": [], "win_prob": [], "total_mu": []}
    all_targets = {"margin": [], "win": [], "total": []}

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)

        predictions = forward_l2_l3_l4(batch, l2_model, l3_model, l4_model, device)

        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            actual_total=actual_total,
            cfg=l4_cfg,
        )

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

        all_preds["spread_mu"].append(predictions["spread_mu"].cpu().numpy())
        all_preds["spread_sigma"].append(predictions["spread_sigma"].cpu().numpy())
        all_preds["win_prob"].append(predictions["win_prob"].cpu().numpy())
        all_preds["total_mu"].append(predictions["total_mu"].cpu().numpy())

        all_targets["margin"].append(actual_margin.cpu().numpy())
        all_targets["win"].append(actual_win.cpu().numpy())
        all_targets["total"].append(actual_total.cpu().numpy())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    metrics = compute_metrics(all_preds, all_targets)

    # Return raw concatenated arrays for ensemble use
    raw = {
        "spread_mu": np.concatenate(all_preds["spread_mu"]),
        "spread_sigma": np.concatenate(all_preds["spread_sigma"]),
        "win_prob": np.concatenate(all_preds["win_prob"]),
        "total_mu": np.concatenate(all_preds["total_mu"]),
        "actual_margin": np.concatenate(all_targets["margin"]),
        "actual_win": np.concatenate(all_targets["win"]),
        "actual_total": np.concatenate(all_targets["total"]),
    }

    return avg_losses, metrics, raw


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------


def ensemble_predictions(all_raw: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Combine predictions from multiple models.

    - spread_mu, total_mu: simple average
    - spread_sigma: mixture-of-Gaussians
    - win_prob: logit averaging
    """
    spread_means = np.stack([r["spread_mu"] for r in all_raw])
    spread_sigmas = np.stack([r["spread_sigma"] for r in all_raw])
    total_means = np.stack([r["total_mu"] for r in all_raw])
    win_probs = np.stack([r["win_prob"] for r in all_raw])

    # Simple average of means
    ens_spread_mu = spread_means.mean(axis=0)
    ens_total_mu = total_means.mean(axis=0)

    # Mixture-of-Gaussians sigma
    ens_spread_var = (
        np.mean(spread_sigmas**2, axis=0)
        + np.mean(spread_means**2, axis=0)
        - ens_spread_mu**2
    )
    ens_spread_sigma = np.sqrt(np.maximum(ens_spread_var, 1e-6))

    # Win probability: logit averaging
    eps = 1e-7
    wp_clipped = np.clip(win_probs, eps, 1 - eps)
    logits = np.log(wp_clipped / (1 - wp_clipped))
    avg_logit = logits.mean(axis=0)
    ens_win_prob = 1.0 / (1.0 + np.exp(-avg_logit))

    return {
        "spread_mu": ens_spread_mu,
        "spread_sigma": ens_spread_sigma,
        "win_prob": ens_win_prob,
        "total_mu": ens_total_mu,
        "actual_margin": all_raw[0]["actual_margin"],
        "actual_win": all_raw[0]["actual_win"],
        "actual_total": all_raw[0]["actual_total"],
    }


def metrics_from_raw(raw: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute metrics from raw prediction arrays."""
    all_preds = {
        "spread_mu": [raw["spread_mu"]],
        "spread_sigma": [raw["spread_sigma"]],
        "win_prob": [raw["win_prob"]],
        "total_mu": [raw["total_mu"]],
    }
    all_targets = {
        "margin": [raw["actual_margin"]],
        "win": [raw["actual_win"]],
        "total": [raw["actual_total"]],
    }
    return compute_metrics(all_preds, all_targets)


# ---------------------------------------------------------------------------
# Weight perturbation
# ---------------------------------------------------------------------------


def perturb_weights(model: nn.Module, scale: float = 0.01, seed: int = 42):
    """Add small random perturbation to model weights for diversity.

    Uses multiplicative noise: w' = w * (1 + eps), eps ~ N(0, scale).
    This preserves the relative magnitude of weights while adding diversity.
    """
    # Generate noise on CPU (Generator doesn't support CUDA), then move to device
    rng = torch.Generator()
    rng.manual_seed(seed)

    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=rng) * scale
            param.data.mul_(1.0 + noise.to(param.device))


# ---------------------------------------------------------------------------
# Single seed training
# ---------------------------------------------------------------------------


def train_single_seed(
    seed: int,
    data: dict,
    train_ds: PhaseCDataset,
    val_ds: PhaseCDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> Path:
    """Train Phase C with a specific seed. Returns path to best checkpoint."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING SEED {seed}")
    logger.info(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create data loaders with seed-specific shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create models
    l2_cfg = L2Config()
    l3_cfg = L3Config()
    l4_cfg = L4Config()

    l2_cfg.n_players = max(l2_cfg.n_players, data["n_players_for_embed"])

    l2_model = PlayerSynergyNetwork(l2_cfg).to(device)
    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    # Load pretrained weights
    l2_ckpt = torch.load(
        str(CHECKPOINT_DIR / "l2_best.pt"),
        map_location=device,
        weights_only=False,
    )
    missing_l2, unexpected_l2 = l2_model.load_state_dict(
        l2_ckpt["model_state_dict"], strict=False
    )
    if unexpected_l2:
        logger.warning(f"L2 unexpected keys: {unexpected_l2}")

    b_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_b_best.pt"),
        map_location=device,
        weights_only=False,
    )
    missing_l3, _ = l3_model.load_state_dict(b_ckpt["l3_state_dict"], strict=False)
    missing_l4, _ = l4_model.load_state_dict(b_ckpt["l4_state_dict"], strict=False)
    if missing_l3:
        logger.info(f"L3 newly initialized keys: {missing_l3}")
    if missing_l4:
        logger.info(f"L4 newly initialized keys: {missing_l4}")

    logger.info(
        f"Loaded pretrained weights (L2 epoch {l2_ckpt.get('epoch', '?')}, "
        f"Phase B epoch {b_ckpt.get('epoch', '?')})"
    )

    # Add random perturbation for diversity (different seed -> different perturbation)
    perturb_scale = 0.005  # small perturbation to preserve pretrained quality
    perturb_weights(l2_model, scale=perturb_scale, seed=seed)
    perturb_weights(l3_model, scale=perturb_scale, seed=seed + 1000)
    perturb_weights(l4_model, scale=perturb_scale, seed=seed + 2000)
    logger.info(f"Applied weight perturbation (scale={perturb_scale}, seed={seed})")

    # Drift monitor
    drift_monitor = DriftMonitor(l2_model, name="L2")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": l2_model.parameters(), "lr": args.l2_lr},
            {"params": l3_model.parameters(), "lr": args.l3_lr},
            {"params": l4_model.parameters(), "lr": args.l4_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Checkpoint path for this seed
    ckpt_name = f"phase_c_seed{seed}_best.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Apply gradual unfreezing
        schedule_name = apply_freeze_schedule(
            epoch,
            l2_model,
            l3_model,
            l4_model,
            l4_only_epochs=args.l4_only_epochs,
            l3l4_epochs=args.l3l4_epochs,
        )

        # Train
        train_losses = train_epoch(
            l2_model,
            l3_model,
            l4_model,
            train_loader,
            optimizer,
            l4_cfg,
            device,
            args.grad_clip,
        )

        scheduler.step()

        # Validate
        val_losses, val_metrics, _ = evaluate(
            l2_model, l3_model, l4_model, val_loader, l4_cfg, device
        )

        # Drift
        drift = drift_monitor.compute_drift(l2_model)
        elapsed = time.time() - t0

        logger.info(
            f"  [seed={seed}] Epoch {epoch:2d}/{args.epochs} ({elapsed:.1f}s) "
            f"[{schedule_name}] | "
            f"Train loss={train_losses['total_loss']:.4f} | "
            f"Val MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"Acc={val_metrics['win_acc']:.3f} | "
            f"drift={drift['overall']:.4f}"
        )

        # Checkpoint (best val spread MAE)
        val_mae = val_metrics["spread_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "l2_state_dict": l2_model.state_dict(),
                    "l3_state_dict": l3_model.state_dict(),
                    "l4_state_dict": l4_model.state_dict(),
                    "l2_config": l2_cfg.__dict__,
                    "l3_config": l3_cfg.__dict__,
                    "l4_config": l4_cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                str(ckpt_path),
            )
            logger.info(
                f"  [seed={seed}] ** New best: MAE={best_val_mae:.3f} at epoch {epoch}"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"  [seed={seed}] Early stopping at epoch {epoch} "
                    f"(patience={args.patience})"
                )
                break

    logger.info(
        f"  [seed={seed}] Done. Best MAE={best_val_mae:.3f} at epoch {best_epoch}"
    )

    # Clean up GPU memory
    del l2_model, l3_model, l4_model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ckpt_path


# ---------------------------------------------------------------------------
# Inference with a checkpoint
# ---------------------------------------------------------------------------


def load_and_infer(
    ckpt_path: Path,
    data: dict,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Load a checkpoint and run inference on a DataLoader."""
    l2_cfg = L2Config()
    l3_cfg = L3Config()
    l4_cfg = L4Config()

    l2_cfg.n_players = max(l2_cfg.n_players, data["n_players_for_embed"])

    l2_model = PlayerSynergyNetwork(l2_cfg).to(device)
    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    l2_model.load_state_dict(ckpt["l2_state_dict"])
    l3_model.load_state_dict(ckpt["l3_state_dict"])
    l4_model.load_state_dict(ckpt["l4_state_dict"])

    _, metrics, raw = evaluate(l2_model, l3_model, l4_model, loader, l4_cfg, device)

    del l2_model, l3_model, l4_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics, raw


# ---------------------------------------------------------------------------
# Make test dataset
# ---------------------------------------------------------------------------


def make_test_dataset(
    data: dict,
    test_season: str = "2024-2025",
    # Reuse normalization from train set
    l3_mean=None,
    l3_std=None,
    l4_mean=None,
    l4_std=None,
    rs_mean=None,
    rs_std=None,
    wowy_lookup=None,
    years_together_lookup=None,
    birth_years=None,
) -> PhaseCDataset:
    """Create test dataset."""
    seasons = data["seasons"]
    test_year = season_start_year(test_season)

    test_indices = []
    for i, s in enumerate(seasons):
        if not s:
            continue
        try:
            year = season_start_year(s)
        except (ValueError, IndexError):
            continue
        if year == test_year:
            test_indices.append(i)

    test_indices = np.array(test_indices, dtype=np.int64)
    logger.info(f"Test: {len(test_indices)} games ({test_season})")

    return PhaseCDataset(
        indices=test_indices,
        player_abilities=data["player_abilities"],
        player_uncertainties=data["player_uncertainties"],
        player_archetypes=data["player_archetypes"],
        player_masks=data["player_masks"],
        player_indices=data["player_indices"],
        team_features=data["team_features"],
        game_context=data["game_context"],
        roster_summaries=data["roster_summaries"],
        targets_margin=data["targets_margin"],
        targets_win=data["targets_win"],
        targets_total=data["targets_total"],
        coach_indices=data.get("coach_indices"),
        coach_games=data.get("coach_games"),
        player_ids=data.get("player_ids"),
        seasons=data.get("seasons"),
        wowy_lookup=wowy_lookup,
        years_together_lookup=years_together_lookup,
        birth_years=birth_years,
        l3_mean=l3_mean,
        l3_std=l3_std,
        l4_mean=l4_mean,
        l4_std=l4_std,
        rs_mean=rs_mean,
        rs_std=rs_std,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase C Ensemble: 3 seeds")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--l2-lr", type=float, default=3e-5)
    parser.add_argument("--l3-lr", type=float, default=1e-4)
    parser.add_argument("--l4-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--l4-only-epochs", type=int, default=3)
    parser.add_argument("--l3l4-epochs", type=int, default=6)
    parser.add_argument("--seeds", type=str, default="42,137,256")
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2017-2018,2022-2023",
    )
    parser.add_argument("--val-season", type=str, default="2023-2024")
    parser.add_argument("--test-season", type=str, default="2024-2025")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, just ensemble existing checkpoints",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Seeds: {seeds}")

    # --- Load data once (shared across all seeds) ---
    logger.info("\n--- Loading Phase C data (shared across seeds) ---")
    data = build_phase_c_data()

    train_start, train_end = args.train_seasons.split(",")
    train_ds, val_ds = make_train_val_datasets(
        data,
        train_seasons=(train_start, train_end),
        val_season=args.val_season,
    )

    logger.info(f"Train dataset: {len(train_ds)} games")
    logger.info(f"Val dataset: {len(val_ds)} games")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset! Check season ranges.")
        sys.exit(1)

    # Build test dataset reusing normalization from train_ds
    # We need the edge feature lookups from train_ds
    wowy_lookup = train_ds.wowy_lookup
    years_together_lookup = train_ds.years_together_lookup
    birth_years_dict = train_ds.birth_years

    test_ds = make_test_dataset(
        data,
        test_season=args.test_season,
        l3_mean=train_ds.l3_mean,
        l3_std=train_ds.l3_std,
        l4_mean=train_ds.l4_mean,
        l4_std=train_ds.l4_std,
        rs_mean=train_ds.rs_mean,
        rs_std=train_ds.rs_std,
        wowy_lookup=wowy_lookup,
        years_together_lookup=years_together_lookup,
        birth_years=birth_years_dict,
    )
    logger.info(f"Test dataset: {len(test_ds)} games")

    # --- Train each seed ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_paths = []
    if not args.skip_train:
        for seed in seeds:
            ckpt_path = train_single_seed(
                seed=seed,
                data=data,
                train_ds=train_ds,
                val_ds=val_ds,
                device=device,
                args=args,
            )
            ckpt_paths.append(ckpt_path)
    else:
        for seed in seeds:
            ckpt_path = CHECKPOINT_DIR / f"phase_c_seed{seed}_best.pt"
            if not ckpt_path.exists():
                logger.error(f"Checkpoint not found: {ckpt_path}")
                sys.exit(1)
            ckpt_paths.append(ckpt_path)
        logger.info(f"Skipping training, using existing checkpoints: {ckpt_paths}")

    # --- Inference on val and test ---
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE INFERENCE")
    logger.info(f"{'='*70}")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Vegas spreads for ATS
    vegas_spreads_dict = load_vegas_spreads()
    game_ids = data["game_ids"]

    # Build Vegas arrays for val and test
    def build_vegas_arrays(ds: PhaseCDataset) -> tuple[np.ndarray, np.ndarray]:
        n = len(ds)
        vs = np.zeros(n, dtype=np.float32)
        hv = np.zeros(n, dtype=bool)
        for j in range(n):
            idx = ds.indices[j]
            gid = game_ids[idx]
            if gid in vegas_spreads_dict:
                vs[j] = vegas_spreads_dict[gid]
                hv[j] = True
        return vs, hv

    val_vegas, val_has_vegas = build_vegas_arrays(val_ds)
    test_vegas, test_has_vegas = build_vegas_arrays(test_ds)
    logger.info(
        f"Vegas coverage: val={val_has_vegas.sum()}/{len(val_ds)}, "
        f"test={test_has_vegas.sum()}/{len(test_ds)}"
    )

    # Collect predictions from each model
    val_individual_metrics = []
    test_individual_metrics = []
    val_individual_ats = []
    test_individual_ats = []
    val_all_raw = []
    test_all_raw = []

    for i, ckpt_path in enumerate(ckpt_paths):
        seed = seeds[i]
        logger.info(f"\nInference: seed={seed} ({ckpt_path.name})")

        val_metrics, val_raw = load_and_infer(ckpt_path, data, val_loader, device)
        test_metrics, test_raw = load_and_infer(ckpt_path, data, test_loader, device)

        val_individual_metrics.append(val_metrics)
        test_individual_metrics.append(test_metrics)
        val_all_raw.append(val_raw)
        test_all_raw.append(test_raw)

        # ATS for individual models
        val_ats = compute_ats_metrics(
            val_raw["spread_mu"], val_raw["actual_margin"], val_vegas, val_has_vegas
        )
        test_ats = compute_ats_metrics(
            test_raw["spread_mu"], test_raw["actual_margin"], test_vegas, test_has_vegas
        )
        val_individual_ats.append(val_ats)
        test_individual_ats.append(test_ats)

        logger.info(
            f"  Val:  MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"ATS={val_ats['ats_pct']:.1%} ({val_ats['ats_correct']}/{val_ats['ats_total']})"
        )
        logger.info(
            f"  Test: MAE={test_metrics['spread_mae']:.2f}, "
            f"AUC={test_metrics['win_auc']:.3f}, "
            f"ATS={test_ats['ats_pct']:.1%} ({test_ats['ats_correct']}/{test_ats['ats_total']})"
        )

    # --- Ensemble predictions ---
    val_ens_raw = ensemble_predictions(val_all_raw)
    test_ens_raw = ensemble_predictions(test_all_raw)

    val_ens_metrics = metrics_from_raw(val_ens_raw)
    test_ens_metrics = metrics_from_raw(test_ens_raw)

    val_ens_ats = compute_ats_metrics(
        val_ens_raw["spread_mu"], val_ens_raw["actual_margin"], val_vegas, val_has_vegas
    )
    test_ens_ats = compute_ats_metrics(
        test_ens_raw["spread_mu"],
        test_ens_raw["actual_margin"],
        test_vegas,
        test_has_vegas,
    )

    # --- Print comparison table ---
    print(f"\n{'='*90}")
    print("PHASE C ENSEMBLE RESULTS")
    print(f"{'='*90}")

    for split_name, indiv_metrics, indiv_ats, ens_m, ens_a in [
        (
            "VAL",
            val_individual_metrics,
            val_individual_ats,
            val_ens_metrics,
            val_ens_ats,
        ),
        (
            "TEST",
            test_individual_metrics,
            test_individual_ats,
            test_ens_metrics,
            test_ens_ats,
        ),
    ]:
        header = (
            f"| {'Model':<18s} | {'Spread MAE':>10s} | {'Win AUC':>8s} | "
            f"{'Win Acc':>8s} | {'ECE':>6s} | {'ATS%':>7s} | {'ATS Record':>11s} | "
            f"{'ROI%':>6s} |"
        )
        sep = (
            f"|{'-'*20}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*8}|"
            f"{'-'*9}|{'-'*13}|{'-'*8}|"
        )

        print(f"\n  {split_name} SET ({indiv_metrics[0]['n_games']} games)")
        print(header)
        print(sep)

        for i, (m, a) in enumerate(zip(indiv_metrics, indiv_ats)):
            seed = seeds[i]
            record = f"{a['ats_correct']}/{a['ats_total']}"
            print(
                f"| {'Seed ' + str(seed):<18s} | {m['spread_mae']:>10.2f} | "
                f"{m['win_auc']:>8.3f} | {m['win_acc']*100:>7.1f}% | "
                f"{m['ece']:>6.4f} | {a['ats_pct']*100:>6.1f}% | {record:>11s} | "
                f"{a['roi']:>+5.1f}% |"
            )

        record = f"{ens_a['ats_correct']}/{ens_a['ats_total']}"
        print(
            f"| {'** ENSEMBLE **':<18s} | {ens_m['spread_mae']:>10.2f} | "
            f"{ens_m['win_auc']:>8.3f} | {ens_m['win_acc']*100:>7.1f}% | "
            f"{ens_m['ece']:>6.4f} | {ens_a['ats_pct']*100:>6.1f}% | {record:>11s} | "
            f"{ens_a['roi']:>+5.1f}% |"
        )
        print(sep)

    print(f"\n{'='*90}")

    # Model agreement analysis
    print("\n  MODEL AGREEMENT ANALYSIS")
    for split_name, all_raw in [("VAL", val_all_raw), ("TEST", test_all_raw)]:
        spread_preds = np.stack([r["spread_mu"] for r in all_raw])
        # Pairwise correlation
        n_models = len(seeds)
        corrs = []
        for a in range(n_models):
            for b in range(a + 1, n_models):
                c = np.corrcoef(spread_preds[a], spread_preds[b])[0, 1]
                corrs.append(c)

        # Mean std across games
        mean_std = spread_preds.std(axis=0).mean()

        print(
            f"  {split_name}: Mean pairwise corr={np.mean(corrs):.4f}, "
            f"Mean inter-model std={mean_std:.2f}"
        )

    print(f"\n{'='*90}")

    # --- Save results ---
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    results = {
        "seeds": seeds,
        "args": vars(args),
        "checkpoints": [str(p) for p in ckpt_paths],
        "val_individual_metrics": make_serializable(val_individual_metrics),
        "val_individual_ats": make_serializable(val_individual_ats),
        "val_ensemble_metrics": make_serializable(val_ens_metrics),
        "val_ensemble_ats": make_serializable(val_ens_ats),
        "test_individual_metrics": make_serializable(test_individual_metrics),
        "test_individual_ats": make_serializable(test_individual_ats),
        "test_ensemble_metrics": make_serializable(test_ens_metrics),
        "test_ensemble_ats": make_serializable(test_ens_ats),
    }

    results_path = RESULTS_DIR / "phase_c_ensemble_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save ensemble checkpoint (just metadata + pointers)
    torch.save(
        {
            "seeds": seeds,
            "checkpoint_paths": [str(p) for p in ckpt_paths],
            "val_metrics": val_ens_metrics,
            "test_metrics": test_ens_metrics,
            "val_ats": val_ens_ats,
            "test_ats": test_ens_ats,
        },
        str(CHECKPOINT_DIR / "phase_c_ensemble.pt"),
    )
    logger.info(
        f"Ensemble checkpoint saved to {CHECKPOINT_DIR / 'phase_c_ensemble.pt'}"
    )


if __name__ == "__main__":
    main()
