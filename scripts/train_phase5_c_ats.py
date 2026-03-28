#!/usr/bin/env python3
"""
Phase C ATS Fine-tuning: Add ATS classification head + spread decompression.

Loads the Phase C best checkpoint and fine-tunes for additional epochs with:
  1. ATS classification head (256-d hidden + Vegas spread -> P(home covers))
  2. ATS BCE loss ramped in after initial stability epochs
  3. Post-hoc spread decompression for evaluation

The ATS head is auxiliary -- it should not hurt spread MAE.

Usage:
    python scripts/train_phase5_c_ats.py
    python scripts/train_phase5_c_ats.py --epochs 15 --patience 8
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
from src.phase5.l4_losses import game_prediction_loss, ats_loss

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

# Data paths
PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"
L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
WOWY_DB_PATH = PROJECT_ROOT / "data" / "l2_cache" / "wowy.sqlite"
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import shared helpers from the Phase C script
# ---------------------------------------------------------------------------

# We reuse the data-building functions from train_phase5_c.py to avoid
# massive duplication. Import the PhaseCDataset and build_phase_c_data.
from train_phase5_c import (
    PhaseCDataset,
    build_phase_c_data,
    load_wowy_lookup,
    build_years_together_lookup,
    load_birth_years,
    season_start_year,
    forward_l2_l3_l4,
    DriftMonitor,
    apply_freeze_schedule,
)

# ---------------------------------------------------------------------------
# Vegas spread loading
# ---------------------------------------------------------------------------


def load_vegas_spreads() -> dict[str, float]:
    """Load Vegas closing spreads from Betting table.

    Uses COALESCE priority: espn_closing > covers_closing > espn_current.
    Returns dict mapping game_id -> spread (negative = home favored).
    """
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
# Extended Phase C Dataset with Vegas spreads
# ---------------------------------------------------------------------------


class PhaseCATSDataset(PhaseCDataset):
    """Phase C dataset extended with Vegas spreads for ATS training."""

    def __init__(self, vegas_spreads: np.ndarray, has_vegas: np.ndarray, **kwargs):
        """
        Additional args beyond PhaseCDataset:
            vegas_spreads: (N,) float32 spread values (0 where unavailable)
            has_vegas: (N,) bool mask indicating which games have spreads
        """
        super().__init__(**kwargs)
        self.vegas_spreads = vegas_spreads
        self.has_vegas = has_vegas

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        i = self.indices[idx]

        item["vegas_spread"] = torch.tensor(
            float(self.vegas_spreads[i]), dtype=torch.float32
        )
        item["has_vegas"] = torch.tensor(bool(self.has_vegas[i]), dtype=torch.bool)

        return item


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def make_train_val_test_datasets(
    data: dict,
    vegas_spreads_dict: dict[str, float],
    train_seasons: tuple[str, str] = ("2017-2018", "2022-2023"),
    val_season: str = "2023-2024",
    test_season: str = "2024-2025",
) -> tuple[PhaseCATSDataset, PhaseCATSDataset, PhaseCATSDataset]:
    """Split data by season and create train/val/test datasets with Vegas spreads."""
    seasons = data["seasons"]
    game_ids = data["game_ids"]
    train_start = season_start_year(train_seasons[0])
    train_end = season_start_year(train_seasons[1])
    val_year = season_start_year(val_season)
    test_year = season_start_year(test_season)

    train_indices = []
    val_indices = []
    test_indices = []

    for i, s in enumerate(seasons):
        if not s:
            continue
        try:
            year = season_start_year(s)
        except (ValueError, IndexError):
            continue
        if train_start <= year <= train_end:
            train_indices.append(i)
        elif year == val_year:
            val_indices.append(i)
        elif year == test_year:
            test_indices.append(i)

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    logger.info(
        f"Train: {len(train_indices)} games "
        f"({train_seasons[0]} - {train_seasons[1]})"
    )
    logger.info(f"Val: {len(val_indices)} games ({val_season})")
    logger.info(f"Test: {len(test_indices)} games ({test_season})")

    # Build Vegas spread arrays
    n_games = len(game_ids)
    vegas_spreads_arr = np.zeros(n_games, dtype=np.float32)
    has_vegas_arr = np.zeros(n_games, dtype=bool)

    n_found = 0
    for i in range(n_games):
        gid = game_ids[i]
        if gid in vegas_spreads_dict:
            vegas_spreads_arr[i] = vegas_spreads_dict[gid]
            has_vegas_arr[i] = True
            n_found += 1

    logger.info(
        f"Vegas spread coverage: {n_found}/{n_games} ({n_found/n_games*100:.1f}%)"
    )

    # Train set coverage
    train_has = has_vegas_arr[train_indices].sum()
    val_has = has_vegas_arr[val_indices].sum()
    test_has = has_vegas_arr[test_indices].sum()
    logger.info(
        f"  Train: {train_has}/{len(train_indices)}, "
        f"Val: {val_has}/{len(val_indices)}, "
        f"Test: {test_has}/{len(test_indices)}"
    )

    # Compute normalization from training set only
    train_tf = data["team_features"][train_indices]
    train_gc = data["game_context"][train_indices]
    train_rs = data["roster_summaries"][train_indices]

    tf_flat = train_tf.reshape(-1, 34)
    l3_mean = tf_flat.mean(axis=0).astype(np.float32)
    l3_std = tf_flat.std(axis=0).astype(np.float32)
    l3_std[l3_std < 1e-8] = 1e-8

    l4_mean = train_gc.mean(axis=0).astype(np.float32)
    l4_std = train_gc.std(axis=0).astype(np.float32)
    l4_std[l4_std < 1e-8] = 1e-8

    rs_flat = train_rs.reshape(-1, 12)
    rs_mean = rs_flat.mean(axis=0).astype(np.float32)
    rs_std = rs_flat.std(axis=0).astype(np.float32)
    rs_std[rs_std < 1e-8] = 1e-8

    # Load edge feature lookups
    logger.info("Loading edge feature lookups...")
    wowy_lookup = load_wowy_lookup()
    years_together_lookup = build_years_together_lookup(wowy_lookup)
    birth_years = load_birth_years()

    common_kwargs = dict(
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

    train_ds = PhaseCATSDataset(
        vegas_spreads=vegas_spreads_arr,
        has_vegas=has_vegas_arr,
        indices=train_indices,
        **common_kwargs,
    )
    val_ds = PhaseCATSDataset(
        vegas_spreads=vegas_spreads_arr,
        has_vegas=has_vegas_arr,
        indices=val_indices,
        **common_kwargs,
    )
    test_ds = PhaseCATSDataset(
        vegas_spreads=vegas_spreads_arr,
        has_vegas=has_vegas_arr,
        indices=test_indices,
        **common_kwargs,
    )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Forward pass with Vegas spread
# ---------------------------------------------------------------------------


def forward_l2_l3_l4_ats(
    batch: dict[str, torch.Tensor],
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    device: torch.device,
    use_vegas: bool = True,
) -> dict[str, torch.Tensor]:
    """Run L2 -> L3 -> L4 forward pass, passing Vegas spread to L4."""
    # Reuse the existing forward pass for L2->L3
    home_ab = batch["home_abilities"].to(device)
    away_ab = batch["away_abilities"].to(device)
    home_unc = batch["home_uncertainties"].to(device)
    away_unc = batch["away_uncertainties"].to(device)
    home_arch = batch["home_archetypes"].to(device)
    away_arch = batch["away_archetypes"].to(device)
    home_mask = batch["home_player_mask"].to(device)
    away_mask = batch["away_player_mask"].to(device)
    home_pidx = batch["home_player_idx"].to(device)
    away_pidx = batch["away_player_idx"].to(device)
    home_ef = batch["home_edge_features"].to(device)
    away_ef = batch["away_edge_features"].to(device)
    home_tf = batch["home_team_features"].to(device)
    away_tf = batch["away_team_features"].to(device)
    home_rs = batch["home_roster_summary"].to(device)
    away_rs = batch["away_roster_summary"].to(device)
    home_coach = batch["home_coach_idx"].to(device)
    away_coach = batch["away_coach_idx"].to(device)
    home_coach_games = batch["home_coach_games"].to(device)
    away_coach_games = batch["away_coach_games"].to(device)
    home_cont = batch["home_continuity"].to(device)
    away_cont = batch["away_continuity"].to(device)
    game_ctx = batch["game_context"].to(device)

    # L2 forward
    home_l2_out = l2_model(
        ability=home_ab,
        uncertainty=home_unc,
        archetypes=home_arch,
        mask=home_mask,
        player_idx=home_pidx,
        edge_features=home_ef,
    )
    away_l2_out = l2_model(
        ability=away_ab,
        uncertainty=away_unc,
        archetypes=away_arch,
        mask=away_mask,
        player_idx=away_pidx,
        edge_features=away_ef,
    )

    home_l2 = home_l2_out["team_vector"]
    away_l2 = away_l2_out["team_vector"]

    # L3 forward
    home_repr = l3_model(
        l2_team=home_l2,
        team_features=home_tf,
        roster_summary=home_rs,
        coach_idx=home_coach,
        roster_continuity=home_cont,
        coach_games=home_coach_games,
    )
    away_repr = l3_model(
        l2_team=away_l2,
        team_features=away_tf,
        roster_summary=away_rs,
        coach_idx=away_coach,
        roster_continuity=away_cont,
        coach_games=away_coach_games,
    )

    # L4 forward (with Vegas spread if available)
    vegas_spread = None
    if use_vegas and "vegas_spread" in batch:
        vegas_spread = batch["vegas_spread"].to(device)

    predictions = l4_model(
        team_home=home_repr,
        team_away=away_repr,
        l2_home=home_l2,
        l2_away=away_l2,
        context=game_ctx,
        vegas_spread=vegas_spread,
    )

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Standard -110 odds payout
WIN_PAYOUT = 100 / 1.10  # ~$90.91 profit
LOSS_COST = 100.0


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
    ats_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute ATS metrics.

    We pick home to cover when our predicted_margin > -vegas_spread.
    If ats_prob is provided, also evaluate the ATS head's predictions.

    Returns dict with ats_pct, ats_correct, ats_total, roi, etc.
    """
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
    ats_result = ats_result[not_push]

    # Our spread-based pick: pick home to cover when pred_margin > -vegas_spread
    our_pick_home = pm > (-vs)

    # ATS accuracy: we were right when our_pick_home == home_covered
    correct = (our_pick_home == home_covered).sum()
    total = len(vm)
    ats_pct = correct / max(total, 1)

    # ROI at -110
    wins = correct
    losses = total - correct
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = profit / (total * LOSS_COST) * 100  # percentage

    result = {
        "ats_pct": float(ats_pct),
        "ats_correct": int(correct),
        "ats_total": int(total),
        "roi": float(roi),
    }

    # ATS head evaluation (if provided)
    if ats_prob is not None:
        ap = ats_prob[valid][not_push]
        # Pick home when ats_prob > 0.5
        head_pick_home = ap > 0.5
        head_correct = (head_pick_home == home_covered).sum()
        head_pct = head_correct / max(total, 1)
        head_profit = head_correct * WIN_PAYOUT - (total - head_correct) * LOSS_COST
        head_roi = head_profit / (total * LOSS_COST) * 100

        result["ats_head_pct"] = float(head_pct)
        result["ats_head_correct"] = int(head_correct)
        result["ats_head_roi"] = float(head_roi)

    return result


# ---------------------------------------------------------------------------
# ATS weight ramping
# ---------------------------------------------------------------------------


def get_ats_weight(epoch: int, cfg: L4Config) -> float:
    """Compute ATS loss weight with linear ramp."""
    if epoch < cfg.ats_ramp_start:
        return 0.0
    ramp_progress = min(
        1.0, (epoch - cfg.ats_ramp_start + 1) / max(cfg.ats_ramp_epochs, 1)
    )
    return cfg.w_ats * ramp_progress


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l4_cfg: L4Config,
    device: torch.device,
    grad_clip: float = 1.0,
    w_ats: float = 0.0,
) -> dict[str, float]:
    """Train one epoch with optional ATS loss."""
    l2_model.train()
    l3_model.train()
    l4_model.train()

    total_losses = {}
    n_batches = 0

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)
        vegas_spread_batch = batch["vegas_spread"].to(device)
        has_vegas_batch = batch["has_vegas"].to(device)

        # Forward (always pass Vegas spread so the head gets trained)
        predictions = forward_l2_l3_l4_ats(
            batch, l2_model, l3_model, l4_model, device, use_vegas=True
        )

        # Base losses
        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            actual_total=actual_total,
            cfg=l4_cfg,
        )

        # ATS loss (only for games with Vegas lines)
        L_ats = torch.tensor(0.0, device=device)
        if w_ats > 0 and "ats_logit" in predictions and has_vegas_batch.any():
            # Filter to games with Vegas spreads
            valid_mask = has_vegas_batch
            if valid_mask.any():
                L_ats = ats_loss(
                    predictions["ats_logit"][valid_mask],
                    actual_margin[valid_mask],
                    vegas_spread_batch[valid_mask],
                )

        total_loss = losses["total_loss"] + w_ats * L_ats

        # Backward
        optimizer.zero_grad()
        total_loss.backward()

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
        total_losses.setdefault("ats", 0.0)
        total_losses["ats"] += L_ats.item()
        total_losses.setdefault("combined", 0.0)
        total_losses["combined"] += total_loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    l4_cfg: L4Config,
    device: torch.device,
    use_vegas: bool = True,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Evaluate on a dataset. Returns (loss_dict, metrics_dict, ats_dict)."""
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()

    total_losses = {}
    n_batches = 0

    all_preds = {"spread_mu": [], "spread_sigma": [], "win_prob": [], "total_mu": []}
    all_targets = {"margin": [], "win": [], "total": []}
    all_vegas = []
    all_has_vegas = []
    all_ats_prob = []

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)

        predictions = forward_l2_l3_l4_ats(
            batch, l2_model, l3_model, l4_model, device, use_vegas=use_vegas
        )

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

        all_vegas.append(batch["vegas_spread"].numpy())
        all_has_vegas.append(batch["has_vegas"].numpy())

        if "ats_prob" in predictions:
            all_ats_prob.append(predictions["ats_prob"].cpu().numpy())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    metrics = compute_metrics(all_preds, all_targets)

    # ATS metrics
    spread_mu = np.concatenate(all_preds["spread_mu"])
    actual_margin_all = np.concatenate(all_targets["margin"])
    vegas_all = np.concatenate(all_vegas)
    has_vegas_all = np.concatenate(all_has_vegas)
    ats_prob_all = np.concatenate(all_ats_prob) if all_ats_prob else None

    ats_metrics = compute_ats_metrics(
        spread_mu=spread_mu,
        actual_margin=actual_margin_all,
        vegas_spread=vegas_all,
        has_vegas=has_vegas_all,
        ats_prob=ats_prob_all,
    )

    return avg_losses, metrics, ats_metrics


# ---------------------------------------------------------------------------
# Spread Decompression
# ---------------------------------------------------------------------------


def compute_decompression(
    train_preds: np.ndarray,
    train_targets: np.ndarray,
) -> tuple[float, float, float]:
    """Compute decompression scale factor from training set.

    Returns (pred_mean, scale, target_std).
    """
    pred_mean = train_preds.mean()
    pred_std = train_preds.std()
    target_std = train_targets.std()
    scale = target_std / max(pred_std, 1e-8)

    logger.info(
        f"Decompression: pred_std={pred_std:.2f}, target_std={target_std:.2f}, "
        f"scale={scale:.3f}, pred_mean={pred_mean:.2f}"
    )
    return float(pred_mean), float(scale), float(target_std)


def apply_decompression(
    preds: np.ndarray,
    pred_mean: float,
    scale: float,
) -> np.ndarray:
    """Apply spread decompression: push predictions away from mean."""
    return pred_mean + (preds - pred_mean) * scale


# ---------------------------------------------------------------------------
# Collect predictions for decompression
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    device: torch.device,
    use_vegas: bool = True,
) -> dict[str, np.ndarray]:
    """Collect all predictions and targets from a loader."""
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()

    all_spread_mu = []
    all_margin = []
    all_win_prob = []
    all_win = []
    all_total_mu = []
    all_total = []
    all_spread_sigma = []
    all_vegas = []
    all_has_vegas = []
    all_ats_prob = []

    for batch in loader:
        predictions = forward_l2_l3_l4_ats(
            batch, l2_model, l3_model, l4_model, device, use_vegas=use_vegas
        )

        all_spread_mu.append(predictions["spread_mu"].cpu().numpy())
        all_spread_sigma.append(predictions["spread_sigma"].cpu().numpy())
        all_win_prob.append(predictions["win_prob"].cpu().numpy())
        all_total_mu.append(predictions["total_mu"].cpu().numpy())
        all_margin.append(batch["actual_margin"].numpy())
        all_win.append(batch["actual_home_win"].numpy())
        all_total.append(batch["actual_total"].numpy())
        all_vegas.append(batch["vegas_spread"].numpy())
        all_has_vegas.append(batch["has_vegas"].numpy())

        if "ats_prob" in predictions:
            all_ats_prob.append(predictions["ats_prob"].cpu().numpy())

    return {
        "spread_mu": np.concatenate(all_spread_mu),
        "spread_sigma": np.concatenate(all_spread_sigma),
        "win_prob": np.concatenate(all_win_prob),
        "total_mu": np.concatenate(all_total_mu),
        "actual_margin": np.concatenate(all_margin),
        "actual_win": np.concatenate(all_win),
        "actual_total": np.concatenate(all_total),
        "vegas_spread": np.concatenate(all_vegas),
        "has_vegas": np.concatenate(all_has_vegas),
        "ats_prob": np.concatenate(all_ats_prob) if all_ats_prob else None,
    }


# ---------------------------------------------------------------------------
# Full evaluation with decompression
# ---------------------------------------------------------------------------


def full_evaluation(
    label: str,
    spread_mu: np.ndarray,
    spread_sigma: np.ndarray,
    win_prob: np.ndarray,
    actual_margin: np.ndarray,
    actual_win: np.ndarray,
    vegas_spread: np.ndarray,
    has_vegas: np.ndarray,
    ats_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Run complete evaluation suite for a prediction set."""
    n = len(actual_margin)
    mae = np.abs(spread_mu - actual_margin).mean()

    # Win metrics
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

    # ATS
    ats = compute_ats_metrics(
        spread_mu, actual_margin, vegas_spread, has_vegas, ats_prob
    )

    logger.info(
        f"  [{label}] MAE={mae:.3f}, AUC={auc:.3f}, WinAcc={win_acc:.3f}, "
        f"ATS={ats['ats_pct']:.3f} ({ats['ats_correct']}/{ats['ats_total']}), "
        f"ROI={ats['roi']:.1f}%"
    )

    if ats_prob is not None and "ats_head_pct" in ats:
        logger.info(
            f"           ATS-Head={ats['ats_head_pct']:.3f} "
            f"({ats['ats_head_correct']}/{ats['ats_total']}), "
            f"ROI={ats['ats_head_roi']:.1f}%"
        )

    return {
        "label": label,
        "mae": float(mae),
        "auc": float(auc),
        "win_acc": float(win_acc),
        "ats_pct": ats["ats_pct"],
        "ats_correct": ats["ats_correct"],
        "ats_total": ats["ats_total"],
        "roi": ats["roi"],
        "ats_head_pct": ats.get("ats_head_pct"),
        "ats_head_roi": ats.get("ats_head_roi"),
        "n_games": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase C ATS: Fine-tune with ATS head + decompression"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--l2-lr", type=float, default=1e-5)
    parser.add_argument("--l3-lr", type=float, default=3e-5)
    parser.add_argument("--l4-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2017-2018,2022-2023",
    )
    parser.add_argument("--val-season", type=str, default="2023-2024")
    parser.add_argument("--test-season", type=str, default="2024-2025")
    parser.add_argument(
        "--ats-ramp-start",
        type=int,
        default=1,
        help="Epoch to start ramping ATS loss (1-indexed, relative to this fine-tuning run)",
    )
    parser.add_argument(
        "--ats-ramp-epochs",
        type=int,
        default=5,
        help="Epochs over which to ramp ATS loss from 0 to w_ats",
    )
    parser.add_argument("--w-ats", type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load data ---
    data = build_phase_c_data()
    vegas_spreads_dict = load_vegas_spreads()

    train_start, train_end = args.train_seasons.split(",")
    train_ds, val_ds, test_ds = make_train_val_test_datasets(
        data,
        vegas_spreads_dict,
        train_seasons=(train_start, train_end),
        val_season=args.val_season,
        test_season=args.test_season,
    )

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
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

    # --- Create models ---
    l2_cfg = L2Config()
    l3_cfg = L3Config()
    l4_cfg = L4Config()
    l4_cfg.w_ats = args.w_ats
    l4_cfg.ats_ramp_start = args.ats_ramp_start
    l4_cfg.ats_ramp_epochs = args.ats_ramp_epochs

    l2_cfg.n_players = max(l2_cfg.n_players, data["n_players_for_embed"])

    l2_model = PlayerSynergyNetwork(l2_cfg).to(device)
    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    # --- Load Phase C best checkpoint ---
    phase_c_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_c_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l2_model.load_state_dict(phase_c_ckpt["l2_state_dict"])
    l3_model.load_state_dict(phase_c_ckpt["l3_state_dict"])

    # Load L4 with strict=False since we added the ats_head
    l4_missing, l4_unexpected = l4_model.load_state_dict(
        phase_c_ckpt["l4_state_dict"], strict=False
    )
    logger.info(
        f"Loaded Phase C checkpoint (epoch {phase_c_ckpt.get('epoch', '?')}). "
        f"L4 missing keys (new ATS head): {l4_missing}"
    )
    if l4_unexpected:
        logger.warning(f"Unexpected L4 keys: {l4_unexpected}")

    logger.info(
        f"Phase C baseline: "
        f"MAE={phase_c_ckpt['val_metrics']['spread_mae']:.3f}, "
        f"AUC={phase_c_ckpt['val_metrics']['win_auc']:.3f}"
    )

    l4_params = l4_count_params(l4_model)
    logger.info(f"L4 GamePredictor: {l4_params:,} params (including ATS head)")

    # --- Evaluate Phase C baseline on val and test ---
    logger.info("\n--- Phase C Baseline Evaluation (no ATS) ---")
    _, base_val_metrics, base_val_ats = evaluate(
        l2_model, l3_model, l4_model, val_loader, l4_cfg, device, use_vegas=True
    )
    logger.info(
        f"Val: MAE={base_val_metrics['spread_mae']:.3f}, "
        f"AUC={base_val_metrics['win_auc']:.3f}, "
        f"ATS={base_val_ats['ats_pct']:.3f} ({base_val_ats['ats_correct']}/{base_val_ats['ats_total']})"
    )

    _, base_test_metrics, base_test_ats = evaluate(
        l2_model, l3_model, l4_model, test_loader, l4_cfg, device, use_vegas=True
    )
    logger.info(
        f"Test: MAE={base_test_metrics['spread_mae']:.3f}, "
        f"AUC={base_test_metrics['win_auc']:.3f}, "
        f"ATS={base_test_ats['ats_pct']:.3f} ({base_test_ats['ats_correct']}/{base_test_ats['ats_total']})"
    )

    # --- Optimizer ---
    # Lower LR since we're fine-tuning from a good checkpoint
    optimizer = torch.optim.AdamW(
        [
            {"params": l2_model.parameters(), "lr": args.l2_lr},
            {"params": l3_model.parameters(), "lr": args.l3_lr},
            {"params": l4_model.parameters(), "lr": args.l4_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    drift_monitor = DriftMonitor(l2_model, name="L2")

    # --- Training loop ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info(f"\nStarting ATS fine-tuning for {args.epochs} epochs...")
    logger.info(f"  LR: L2={args.l2_lr}, L3={args.l3_lr}, L4={args.l4_lr}")
    logger.info(
        f"  ATS: w_ats={args.w_ats}, ramp_start={args.ats_ramp_start}, "
        f"ramp_epochs={args.ats_ramp_epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # All levels trainable from start (fine-tuning from Phase C)
        for p in l2_model.parameters():
            p.requires_grad = True
        for p in l3_model.parameters():
            p.requires_grad = True
        for p in l4_model.parameters():
            p.requires_grad = True

        # ATS weight ramping
        w_ats = get_ats_weight(epoch, l4_cfg)

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
            w_ats=w_ats,
        )

        scheduler.step()

        # Validate
        val_losses, val_metrics, val_ats = evaluate(
            l2_model, l3_model, l4_model, val_loader, l4_cfg, device
        )

        drift = drift_monitor.compute_drift(l2_model)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss={train_losses['combined']:.4f} "
            f"(spr={train_losses['spread']:.4f}, ats={train_losses['ats']:.4f}, "
            f"w_ats={w_ats:.3f}) | "
            f"Val MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"ATS={val_ats['ats_pct']:.3f} "
            f"({val_ats['ats_correct']}/{val_ats['ats_total']}), "
            f"ROI={val_ats['roi']:.1f}% | "
            f"drift={drift['overall']:.4f}"
        )

        epoch_data = {
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "val_ats": val_ats,
            "w_ats": w_ats,
            "l2_drift": drift["overall"],
            "time_s": elapsed,
        }
        history.append(epoch_data)

        # Checkpoint (best val spread MAE)
        val_mae = val_metrics["spread_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "l2_state_dict": l2_model.state_dict(),
                    "l3_state_dict": l3_model.state_dict(),
                    "l4_state_dict": l4_model.state_dict(),
                    "l2_config": l2_cfg.__dict__,
                    "l3_config": l3_cfg.__dict__,
                    "l4_config": l4_cfg.__dict__,
                    "val_metrics": val_metrics,
                    "val_ats": val_ats,
                },
                str(CHECKPOINT_DIR / "phase_c_ats_best.pt"),
            )
            logger.info(f"  ** New best: MAE={best_val_mae:.3f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # --- Load best checkpoint ---
    logger.info(f"\nLoading best checkpoint (epoch {best_epoch})...")
    best_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_c_ats_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l2_model.load_state_dict(best_ckpt["l2_state_dict"])
    l3_model.load_state_dict(best_ckpt["l3_state_dict"])
    l4_model.load_state_dict(best_ckpt["l4_state_dict"])

    # --- Evaluate: (a) baseline, (b) ATS head, (c) ATS + decompression ---
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    # Collect predictions on train (for decompression) and test
    logger.info("\nCollecting predictions...")
    train_data = collect_predictions(
        l2_model, l3_model, l4_model, train_loader, device, use_vegas=True
    )
    test_data = collect_predictions(
        l2_model, l3_model, l4_model, test_loader, device, use_vegas=True
    )
    val_data_collected = collect_predictions(
        l2_model, l3_model, l4_model, val_loader, device, use_vegas=True
    )

    # (a) Phase C baseline (reload original checkpoint, no ATS)
    logger.info("\n--- (a) Phase C Baseline (original checkpoint, no ATS head) ---")
    l4_model_baseline = GamePredictor(l4_cfg).to(device)
    l4_model_baseline.load_state_dict(phase_c_ckpt["l4_state_dict"], strict=False)
    l2_model.load_state_dict(phase_c_ckpt["l2_state_dict"])
    l3_model.load_state_dict(phase_c_ckpt["l3_state_dict"])

    base_test = collect_predictions(
        l2_model, l3_model, l4_model_baseline, test_loader, device, use_vegas=False
    )
    result_a = full_evaluation(
        "Baseline",
        base_test["spread_mu"],
        base_test["spread_sigma"],
        base_test["win_prob"],
        base_test["actual_margin"],
        base_test["actual_win"],
        base_test["vegas_spread"],
        base_test["has_vegas"],
    )

    # (b) ATS-finetuned model
    logger.info("\n--- (b) + ATS Head (fine-tuned) ---")
    l2_model.load_state_dict(best_ckpt["l2_state_dict"])
    l3_model.load_state_dict(best_ckpt["l3_state_dict"])
    l4_model.load_state_dict(best_ckpt["l4_state_dict"])

    test_data = collect_predictions(
        l2_model, l3_model, l4_model, test_loader, device, use_vegas=True
    )
    result_b = full_evaluation(
        "ATS-finetuned",
        test_data["spread_mu"],
        test_data["spread_sigma"],
        test_data["win_prob"],
        test_data["actual_margin"],
        test_data["actual_win"],
        test_data["vegas_spread"],
        test_data["has_vegas"],
        ats_prob=test_data["ats_prob"],
    )

    # (c) ATS + Decompression
    logger.info("\n--- (c) + ATS Head + Decompression ---")

    # Compute decompression from TRAINING set
    train_data_fresh = collect_predictions(
        l2_model, l3_model, l4_model, train_loader, device, use_vegas=True
    )
    pred_mean, scale, target_std = compute_decompression(
        train_data_fresh["spread_mu"],
        train_data_fresh["actual_margin"],
    )

    decompressed_test = apply_decompression(test_data["spread_mu"], pred_mean, scale)
    result_c = full_evaluation(
        "ATS+Decompress",
        decompressed_test,
        test_data["spread_sigma"],
        test_data["win_prob"],
        test_data["actual_margin"],
        test_data["actual_win"],
        test_data["vegas_spread"],
        test_data["has_vegas"],
        ats_prob=test_data["ats_prob"],
    )

    # Also do decompression on val
    logger.info("\n--- Val set results ---")
    val_data_fresh = collect_predictions(
        l2_model, l3_model, l4_model, val_loader, device, use_vegas=True
    )
    decompressed_val = apply_decompression(
        val_data_fresh["spread_mu"], pred_mean, scale
    )

    val_result_raw = full_evaluation(
        "Val ATS-finetuned",
        val_data_fresh["spread_mu"],
        val_data_fresh["spread_sigma"],
        val_data_fresh["win_prob"],
        val_data_fresh["actual_margin"],
        val_data_fresh["actual_win"],
        val_data_fresh["vegas_spread"],
        val_data_fresh["has_vegas"],
        ats_prob=val_data_fresh["ats_prob"],
    )
    val_result_decomp = full_evaluation(
        "Val ATS+Decompress",
        decompressed_val,
        val_data_fresh["spread_sigma"],
        val_data_fresh["win_prob"],
        val_data_fresh["actual_margin"],
        val_data_fresh["actual_win"],
        val_data_fresh["vegas_spread"],
        val_data_fresh["has_vegas"],
        ats_prob=val_data_fresh["ats_prob"],
    )

    # --- Summary table ---
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (2024-2025 Unseen Test Set)")
    logger.info("=" * 70)
    logger.info(
        f"{'Variant':<25} {'MAE':>7} {'AUC':>7} {'WinAcc':>7} "
        f"{'ATS%':>7} {'ATS':>10} {'ROI':>7} {'ATSHead%':>9} {'HeadROI':>8}"
    )
    logger.info("-" * 90)
    for r in [result_a, result_b, result_c]:
        ats_head_str = (
            f"{r['ats_head_pct']:.3f}" if r.get("ats_head_pct") is not None else "N/A"
        )
        head_roi_str = (
            f"{r['ats_head_roi']:.1f}%" if r.get("ats_head_roi") is not None else "N/A"
        )
        logger.info(
            f"{r['label']:<25} {r['mae']:>7.3f} {r['auc']:>7.3f} {r['win_acc']:>7.3f} "
            f"{r['ats_pct']:>7.3f} {r['ats_correct']:>4}/{r['ats_total']:<5} "
            f"{r['roi']:>6.1f}% {ats_head_str:>9} {head_roi_str:>8}"
        )

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "args": vars(args),
        "baseline_val": base_val_metrics,
        "baseline_val_ats": base_val_ats,
        "baseline_test": base_test_metrics,
        "baseline_test_ats": base_test_ats,
        "best_epoch": best_epoch,
        "decompression": {
            "pred_mean": pred_mean,
            "scale": scale,
            "target_std": target_std,
        },
        "test_results": {
            "a_baseline": make_serializable(result_a),
            "b_ats_finetuned": make_serializable(result_b),
            "c_ats_decompressed": make_serializable(result_c),
        },
        "val_results": {
            "ats_finetuned": make_serializable(val_result_raw),
            "ats_decompressed": make_serializable(val_result_decomp),
        },
        "history": make_serializable(history),
    }

    results_path = RESULTS_DIR / "phase_c_ats_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Checkpoint saved to {CHECKPOINT_DIR / 'phase_c_ats_best.pt'}")


if __name__ == "__main__":
    main()
