#!/usr/bin/env python
"""
Diagnostic Script for Phase 2 Experiments.

Loads a checkpoint and produces detailed analysis:
1. Spread collapse check
2. Sigma distribution analysis
3. Val vs test comparison (overfitting check)
4. Error by spread bucket
5. Loss component breakdown

Usage:
    python scripts/diagnose_phase2.py --config configs/transformer/phase2_exp4_fixes.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.models.prediction_heads import GamePrediction
from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import build_cache, load_cache
from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2
from src.transformer.phase2.models.phase2_model import Phase2Model
from src.transformer.phase2.training.loss import Phase2CombinedLoss
from src.transformer.training.metrics import MetricsCalculator

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def ensure_cache(config: Phase2ExperimentConfig) -> dict:
    cache_dir = config.data.cache_dir
    cache_path = Path(cache_dir)
    if (cache_path / "game_features.pt").exists():
        logging.info(f"Loading existing cache from {cache_dir}")
        return load_cache(cache_dir)
    else:
        all_seasons = sorted(set(
            config.data.train_seasons +
            config.data.val_seasons +
            config.data.test_seasons
        ))
        logging.info(f"Building cache for seasons: {all_seasons}")
        build_cache(all_seasons, cache_dir)
        return load_cache(cache_dir)


def create_loader(config: Phase2ExperimentConfig, cache: dict, seasons: list[str]) -> DataLoader:
    common_kwargs = dict(
        game_features=cache["game_features"],
        season_index=cache["season_index"],
        gs_cache=cache["gs_cache"],
        player_id_map=cache.get("player_id_map", {}),
        n_recent_games=config.data.n_recent_games,
        max_context_games=config.data.max_context_games,
        min_history_games=config.data.min_history_games,
        max_players_per_game=config.data.max_players_per_game,
        max_roster_size=config.data.max_roster_size,
        enable_player_form=config.model.enable_player_form,
        max_player_appearances=config.model.max_player_appearances,
    )
    dataset = Phase2Dataset(
        seasons=seasons,
        enable_augmentation=False,
        **common_kwargs,
    )
    pin_memory = config.data.pin_memory and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_phase2,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )


def load_model_from_checkpoint(
    config: Phase2ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> Phase2Model:
    """Create model and load checkpoint weights (with EMA if available)."""
    model = Phase2Model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    # If EMA weights exist, swap them in (this is what the trainer uses for eval)
    if "ema_state_dict" in checkpoint:
        ema_shadow = checkpoint["ema_state_dict"]["shadow"]
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])
        logger.info("Loaded EMA weights for evaluation")

    model = model.to(device)
    model.eval()

    state = checkpoint.get("state", {})
    logger.info(
        f"Checkpoint loaded: epoch {state.get('epoch', '?')}, "
        f"best_metric {state.get('best_metric', '?')}"
    )
    return model


def collect_predictions(
    model: Phase2Model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    desc: str = "Collecting predictions",
) -> tuple[list[GamePrediction], list[torch.Tensor], list[torch.Tensor]]:
    """Run model on loader, collect all predictions and targets."""
    all_preds = []
    all_home = []
    all_away = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            if batch is None:
                continue

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    preds = model(batch_dev)
            else:
                preds = model(batch_dev)

            # Detach to CPU
            all_preds.append(GamePrediction(
                spread_mean=preds.spread_mean.cpu(),
                spread_std=preds.spread_std.cpu(),
                home_score_mean=preds.home_score_mean.cpu(),
                home_score_std=preds.home_score_std.cpu(),
                away_score_mean=preds.away_score_mean.cpu(),
                away_score_std=preds.away_score_std.cpu(),
                home_win_prob=preds.home_win_prob.cpu(),
            ))
            all_home.append(batch["target_home_scores"].cpu())
            all_away.append(batch["target_away_scores"].cpu())

    return all_preds, all_home, all_away


def concat_predictions(
    preds_list: list[GamePrediction],
    home_list: list[torch.Tensor],
    away_list: list[torch.Tensor],
) -> tuple[GamePrediction, torch.Tensor, torch.Tensor]:
    """Concatenate batched predictions into single tensors."""
    combined = GamePrediction(
        spread_mean=torch.cat([p.spread_mean for p in preds_list]),
        spread_std=torch.cat([p.spread_std for p in preds_list]),
        home_score_mean=torch.cat([p.home_score_mean for p in preds_list]),
        home_score_std=torch.cat([p.home_score_std for p in preds_list]),
        away_score_mean=torch.cat([p.away_score_mean for p in preds_list]),
        away_score_std=torch.cat([p.away_score_std for p in preds_list]),
        home_win_prob=torch.cat([p.home_win_prob for p in preds_list]),
    )
    home = torch.cat(home_list)
    away = torch.cat(away_list)
    return combined, home, away


def compute_metrics(preds: GamePrediction, home: torch.Tensor, away: torch.Tensor) -> "MetricResults":
    calc = MetricsCalculator()
    calc.update(preds, home, away)
    return calc.compute()


def section_spread_collapse(preds: GamePrediction, home: torch.Tensor, away: torch.Tensor):
    """Section 1: Did the spread collapse fixes work?"""
    actual_spread = (home - away).numpy()
    pred_spread = preds.spread_mean.numpy()

    print("\n" + "=" * 60)
    print("1. SPREAD COLLAPSE CHECK")
    print("=" * 60)

    print(f"\n  Mean |predicted spread|:  {np.mean(np.abs(pred_spread)):.2f}")
    print(f"  Mean |actual spread|:     {np.mean(np.abs(actual_spread)):.2f}")
    print(f"  Ratio (pred/actual):      {np.mean(np.abs(pred_spread)) / np.mean(np.abs(actual_spread)):.3f}")

    print(f"\n  Std of predicted spreads: {np.std(pred_spread):.2f}")
    print(f"  Std of actual spreads:    {np.std(actual_spread):.2f}")

    corr = np.corrcoef(pred_spread, actual_spread)[0, 1]
    print(f"\n  Pearson correlation:      {corr:.4f}")

    # Verdict
    ratio = np.mean(np.abs(pred_spread)) / np.mean(np.abs(actual_spread))
    if ratio < 0.5:
        print("\n  VERDICT: Spread is STILL COLLAPSED (ratio < 0.5)")
    elif ratio < 0.8:
        print("\n  VERDICT: Spread partially recovered but still compressed")
    else:
        print("\n  VERDICT: Spread collapse is FIXED (ratio >= 0.8)")


def section_sigma_distribution(preds: GamePrediction, home: torch.Tensor, away: torch.Tensor):
    """Section 2: Is the sigma cap too tight?"""
    spread_std = preds.spread_std.numpy()
    home_std = preds.home_score_std.numpy()
    away_std = preds.away_score_std.numpy()
    actual_spread = (home - away).numpy()

    print("\n" + "=" * 60)
    print("2. SIGMA DISTRIBUTION")
    print("=" * 60)

    print("\n  Spread sigma:")
    print(f"    Mean:   {np.mean(spread_std):.3f}")
    print(f"    Std:    {np.std(spread_std):.3f}")
    print(f"    Min:    {np.min(spread_std):.3f}")
    print(f"    Max:    {np.max(spread_std):.3f}")
    print(f"    Median: {np.median(spread_std):.3f}")
    print(f"    P10:    {np.percentile(spread_std, 10):.3f}")
    print(f"    P25:    {np.percentile(spread_std, 25):.3f}")
    print(f"    P75:    {np.percentile(spread_std, 75):.3f}")
    print(f"    P90:    {np.percentile(spread_std, 90):.3f}")

    # Check how many predictions are hitting the sigma cap
    cap = 8.0  # from config
    at_cap = np.mean(spread_std >= cap - 0.01)
    print(f"\n    % at cap ({cap}):  {at_cap * 100:.1f}%")

    print("\n  Score sigma (home):")
    print(f"    Mean:   {np.mean(home_std):.3f}")
    print(f"    Std:    {np.std(home_std):.3f}")
    print(f"    Min:    {np.min(home_std):.3f}")
    print(f"    Max:    {np.max(home_std):.3f}")

    print("\n  Score sigma (away):")
    print(f"    Mean:   {np.mean(away_std):.3f}")
    print(f"    Std:    {np.std(away_std):.3f}")
    print(f"    Min:    {np.min(away_std):.3f}")
    print(f"    Max:    {np.max(away_std):.3f}")

    # Sharpness = avg 90% CI width
    z_90 = 1.645
    sharpness = np.mean(2 * z_90 * spread_std)
    ideal_std = np.std(actual_spread - preds.spread_mean.numpy())
    ideal_sharpness = 2 * z_90 * ideal_std
    print(f"\n  Sharpness (avg 90% CI width): {sharpness:.2f}")
    print(f"  Ideal sharpness (from residuals): {ideal_sharpness:.2f}")

    if at_cap > 0.5:
        print(f"\n  WARNING: {at_cap*100:.0f}% of predictions hitting sigma cap!")
        print("  The cap is too tight — model cannot express uncertainty.")


def section_val_vs_test(
    val_metrics, test_metrics,
):
    """Section 3: Val vs test comparison — overfitting check."""
    print("\n" + "=" * 60)
    print("3. VAL vs TEST COMPARISON (Overfitting Check)")
    print("=" * 60)

    rows = [
        ("Spread MAE", val_metrics.spread_mae, test_metrics.spread_mae),
        ("Spread RMSE", val_metrics.spread_rmse, test_metrics.spread_rmse),
        ("Home MAE", val_metrics.home_mae, test_metrics.home_mae),
        ("Away MAE", val_metrics.away_mae, test_metrics.away_mae),
        ("Win Accuracy", val_metrics.win_accuracy, test_metrics.win_accuracy),
        ("Win AUC", val_metrics.win_auc, test_metrics.win_auc),
        ("Brier Score", val_metrics.brier_score, test_metrics.brier_score),
        ("ECE", val_metrics.ece, test_metrics.ece),
        ("90% Coverage", val_metrics.spread_coverage_90, test_metrics.spread_coverage_90),
        ("Sharpness", val_metrics.spread_sharpness, test_metrics.spread_sharpness),
    ]

    print(f"\n  {'Metric':<18} {'Val':>10} {'Test':>10} {'Gap':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for name, val, test in rows:
        gap = test - val
        print(f"  {name:<18} {val:>10.4f} {test:>10.4f} {gap:>+10.4f}")

    print(f"\n  Val samples:  {val_metrics.n_samples}")
    print(f"  Test samples: {test_metrics.n_samples}")

    mae_gap = test_metrics.spread_mae - val_metrics.spread_mae
    if mae_gap > 0.3:
        print(f"\n  WARNING: Val-test MAE gap of {mae_gap:.2f} suggests overfitting")
    elif mae_gap > 0.15:
        print(f"\n  NOTE: Val-test MAE gap of {mae_gap:.2f} is moderate")
    else:
        print(f"\n  Val-test gap of {mae_gap:.2f} looks reasonable")


def section_error_by_bucket(preds: GamePrediction, home: torch.Tensor, away: torch.Tensor):
    """Section 4: Error analysis by actual spread magnitude."""
    actual_spread = (home - away).numpy()
    pred_spread = preds.spread_mean.numpy()
    pred_std = preds.spread_std.numpy()

    print("\n" + "=" * 60)
    print("4. ERROR BY SPREAD BUCKET")
    print("=" * 60)

    abs_actual = np.abs(actual_spread)
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]

    print(f"\n  {'Bucket':<12} {'N':>6} {'MAE':>8} {'Mean|Pred|':>12} {'Mean Sigma':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*12} {'-'*12}")

    for lo, hi in buckets:
        mask = (abs_actual >= lo) & (abs_actual < hi)
        n = np.sum(mask)
        if n == 0:
            continue
        mae = np.mean(np.abs(pred_spread[mask] - actual_spread[mask]))
        mean_pred_abs = np.mean(np.abs(pred_spread[mask]))
        mean_sigma = np.mean(pred_std[mask])
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        print(f"  {label:<12} {n:>6} {mae:>8.2f} {mean_pred_abs:>12.2f} {mean_sigma:>12.3f}")

    print(f"\n  Overall:     {len(actual_spread):>6} {np.mean(np.abs(pred_spread - actual_spread)):>8.2f} "
          f"{np.mean(np.abs(pred_spread)):>12.2f} {np.mean(pred_std):>12.3f}")


def section_loss_breakdown(
    preds: GamePrediction,
    home: torch.Tensor,
    away: torch.Tensor,
    config: Phase2ExperimentConfig,
):
    """Section 5: Loss component breakdown on test set."""
    print("\n" + "=" * 60)
    print("5. LOSS COMPONENT BREAKDOWN (Test Set)")
    print("=" * 60)

    criterion = Phase2CombinedLoss(
        spread_weight=config.training.spread_loss_weight,
        score_weight=config.training.score_loss_weight,
        win_prob_weight=config.training.win_prob_loss_weight,
        consistency_weight=config.training.consistency_loss_weight,
        nll_weight=config.training.nll_weight,
        mse_weight=config.training.mse_weight,
    )

    # Compute loss on full set (need gradients disabled, float32)
    with torch.no_grad():
        preds_f32 = GamePrediction(
            spread_mean=preds.spread_mean.float(),
            spread_std=preds.spread_std.float(),
            home_score_mean=preds.home_score_mean.float(),
            home_score_std=preds.home_score_std.float(),
            away_score_mean=preds.away_score_mean.float(),
            away_score_std=preds.away_score_std.float(),
            home_win_prob=preds.home_win_prob.float(),
        )
        losses = criterion(preds_f32, home.float(), away.float())

    # Extract the individual components
    components = [
        ("spread_nll", "Spread NLL"),
        ("spread_mse", "Spread MSE"),
        ("score_nll", "Score NLL"),
        ("score_mse", "Score MSE"),
        ("consistency", "Consistency"),
        ("win_prob_bce", "Win Prob BCE"),
        ("total", "TOTAL"),
    ]

    weights = {
        "spread_nll": config.training.spread_loss_weight * config.training.nll_weight,
        "spread_mse": config.training.spread_loss_weight * config.training.mse_weight,
        "score_nll": config.training.score_loss_weight * config.training.nll_weight,
        "score_mse": config.training.score_loss_weight * config.training.mse_weight,
        "consistency": config.training.consistency_loss_weight,
        "win_prob_bce": config.training.win_prob_loss_weight,
    }

    print(f"\n  {'Component':<18} {'Raw Value':>12} {'Weight':>8} {'Weighted':>12}")
    print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*12}")

    for key, name in components:
        val = losses[key].item()
        if key == "total":
            print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*12}")
            print(f"  {name:<18} {'':<12} {'':>8} {val:>12.4f}")
        else:
            w = weights.get(key, 0)
            print(f"  {name:<18} {val:>12.4f} {w:>8.2f} {val * w:>12.4f}")


def section_representation_analysis(
    model: Phase2Model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    """Section 6: Fusion representation and spread head internals."""
    print("\n" + "=" * 60)
    print("6. REPRESENTATION ANALYSIS")
    print("=" * 60)

    # Capture activations via hooks
    fusion_outputs = []
    spread_hidden = []
    spread_mean_raw = []
    spread_std_raw = []
    home_reprs = []
    away_reprs = []

    def fusion_hook(module, inp, out):
        fusion_outputs.append(out.detach().float().cpu())

    def spread_shared_hook(module, inp, out):
        spread_hidden.append(out.detach().float().cpu())

    def spread_mean_hook(module, inp, out):
        spread_mean_raw.append(out.detach().float().cpu())

    def spread_std_hook(module, inp, out):
        spread_std_raw.append(out.detach().float().cpu())

    # Hook into fusion to capture home/away inputs too
    def fusion_input_hook(module, inp, out):
        home_reprs.append(inp[0].detach().float().cpu())
        away_reprs.append(inp[1].detach().float().cpu())

    handles = [
        model.fusion.register_forward_hook(fusion_input_hook),
        model.fusion.mlp.register_forward_hook(fusion_hook),
        model.prediction_heads.spread_head.shared_layers.register_forward_hook(spread_shared_hook),
        model.prediction_heads.spread_head.mean_head.register_forward_hook(spread_mean_hook),
        model.prediction_heads.spread_head.std_head.register_forward_hook(spread_std_hook),
    ]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Repr analysis", leave=False):
            if batch is None:
                continue
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    model(batch_dev)
            else:
                model(batch_dev)

    for h in handles:
        h.remove()

    # Concatenate
    fusion_all = torch.cat(fusion_outputs).numpy()
    spread_h = torch.cat(spread_hidden).numpy()
    mean_raw = torch.cat(spread_mean_raw).numpy().squeeze(-1)
    std_raw = torch.cat(spread_std_raw).numpy().squeeze(-1)
    home_all = torch.cat(home_reprs).numpy()
    away_all = torch.cat(away_reprs).numpy()

    # --- Team representations ---
    print("\n  Team Representations (input to fusion):")
    home_var = np.var(home_all, axis=0).mean()
    away_var = np.var(away_all, axis=0).mean()
    diff_repr = home_all - away_all
    diff_var = np.var(diff_repr, axis=0).mean()
    print(f"    Home repr per-dim variance (mean): {home_var:.6f}")
    print(f"    Away repr per-dim variance (mean): {away_var:.6f}")
    print(f"    Diff (home-away) per-dim variance:  {diff_var:.6f}")

    # How many effective dimensions?
    from numpy.linalg import svd
    diff_centered = diff_repr - diff_repr.mean(axis=0)
    _, s, _ = svd(diff_centered, full_matrices=False)
    s_squared = s ** 2
    s_cumvar = np.cumsum(s_squared) / np.sum(s_squared)
    n_90 = np.searchsorted(s_cumvar, 0.90) + 1
    n_95 = np.searchsorted(s_cumvar, 0.95) + 1
    n_99 = np.searchsorted(s_cumvar, 0.99) + 1
    print(f"    Diff effective dims (90% var): {n_90}")
    print(f"    Diff effective dims (95% var): {n_95}")
    print(f"    Diff effective dims (99% var): {n_99}")
    print(f"    Top-1 singular value % var:    {s_squared[0] / s_squared.sum() * 100:.1f}%")

    # --- Fusion output ---
    print("\n  Fusion Output (matchup_repr):")
    fusion_var = np.var(fusion_all, axis=0).mean()
    print(f"    Per-dim variance (mean): {fusion_var:.6f}")
    print(f"    Sample std (mean):       {np.std(fusion_all, axis=0).mean():.4f}")
    print(f"    Mean magnitude:          {np.mean(np.abs(fusion_all)):.4f}")

    fusion_centered = fusion_all - fusion_all.mean(axis=0)
    _, s_f, _ = svd(fusion_centered, full_matrices=False)
    sf_sq = s_f ** 2
    sf_cum = np.cumsum(sf_sq) / np.sum(sf_sq)
    n_90f = np.searchsorted(sf_cum, 0.90) + 1
    n_95f = np.searchsorted(sf_cum, 0.95) + 1
    print(f"    Effective dims (90% var): {n_90f}")
    print(f"    Effective dims (95% var): {n_95f}")
    print(f"    Top-1 singular value % var: {sf_sq[0] / sf_sq.sum() * 100:.1f}%")

    # --- Spread head internals ---
    print("\n  Spread Head (internal activations):")
    print(f"    Hidden layer output:")
    print(f"      Mean magnitude: {np.mean(np.abs(spread_h)):.4f}")
    print(f"      Per-dim std:    {np.std(spread_h, axis=0).mean():.4f}")
    print(f"      % dead (always 0): {np.mean(np.all(spread_h == 0, axis=0)) * 100:.1f}%")

    print(f"\n    Mean head raw output (pre-squeeze):")
    print(f"      Mean:    {np.mean(mean_raw):.4f}")
    print(f"      Std:     {np.std(mean_raw):.4f}")
    print(f"      Min:     {np.min(mean_raw):.4f}")
    print(f"      Max:     {np.max(mean_raw):.4f}")
    print(f"      |Mean|:  {np.mean(np.abs(mean_raw)):.4f}")

    print(f"\n    Std head raw output (pre-softplus):")
    print(f"      Mean:    {np.mean(std_raw):.4f}")
    print(f"      Std:     {np.std(std_raw):.4f}")
    print(f"      Min:     {np.min(std_raw):.4f}")
    print(f"      Max:     {np.max(std_raw):.4f}")
    softplus_vals = np.log(1 + np.exp(std_raw))
    print(f"      After softplus + min_std: mean={np.mean(softplus_vals) + 1.0:.3f}")

    # Is the spread head getting discriminative signal?
    print(f"\n  Signal Check:")
    print(f"    Fusion output CoV (std/|mean|): {np.std(fusion_all, axis=0).mean() / (np.mean(np.abs(fusion_all)) + 1e-8):.4f}")
    print(f"    Spread mean range / actual spread range: {(np.max(mean_raw) - np.min(mean_raw)) / (np.max(home_all.sum(1) - away_all.sum(1)) - np.min(home_all.sum(1) - away_all.sum(1)) + 1e-8):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 2 checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (default: auto-discover best.pt)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = Phase2ExperimentConfig.from_yaml(args.config)
    if args.device:
        config.training.device = args.device

    device_str = config.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    use_amp = config.training.use_amp and device.type == "cuda"

    # Discover checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = (
            Path(config.training.checkpoint_dir)
            / config.training.experiment_name
            / "best.pt"
        )
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Device: {device}")

    # Load cache and create data loaders
    cache = ensure_cache(config)

    logger.info("Creating val loader...")
    val_loader = create_loader(config, cache, config.data.val_seasons)
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    logger.info("Creating test loader...")
    test_loader = create_loader(config, cache, config.data.test_seasons)
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    model = load_model_from_checkpoint(config, ckpt_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Collect predictions on val and test
    logger.info("Running inference on val set...")
    val_preds_list, val_home_list, val_away_list = collect_predictions(
        model, val_loader, device, use_amp, desc="Val inference"
    )
    val_preds, val_home, val_away = concat_predictions(val_preds_list, val_home_list, val_away_list)

    logger.info("Running inference on test set...")
    test_preds_list, test_home_list, test_away_list = collect_predictions(
        model, test_loader, device, use_amp, desc="Test inference"
    )
    test_preds, test_home, test_away = concat_predictions(test_preds_list, test_home_list, test_away_list)

    # Compute standard metrics
    val_metrics = compute_metrics(val_preds, val_home, val_away)
    test_metrics = compute_metrics(test_preds, test_home, test_away)

    # Print header
    print("\n" + "#" * 60)
    print(f"# PHASE 2 DIAGNOSTIC REPORT")
    print(f"# Experiment: {config.training.experiment_name}")
    print(f"# Checkpoint: {ckpt_path.name}")
    print(f"# Val samples: {val_metrics.n_samples}, Test samples: {test_metrics.n_samples}")
    print("#" * 60)

    # Quick sanity: print test metrics to verify against known results
    print("\n  Test metrics (verify against known results):")
    print(f"    Spread MAE:  {test_metrics.spread_mae:.2f}")
    print(f"    Spread RMSE: {test_metrics.spread_rmse:.2f}")
    print(f"    Win Accuracy: {test_metrics.win_accuracy:.3f}")
    print(f"    Win AUC:     {test_metrics.win_auc:.3f}")
    print(f"    Brier Score: {test_metrics.brier_score:.4f}")
    print(f"    ECE:         {test_metrics.ece:.4f}")
    print(f"    90% Coverage: {test_metrics.spread_coverage_90:.3f}")

    # Run all 6 analysis sections
    section_spread_collapse(test_preds, test_home, test_away)
    section_sigma_distribution(test_preds, test_home, test_away)
    section_val_vs_test(val_metrics, test_metrics)
    section_error_by_bucket(test_preds, test_home, test_away)
    section_loss_breakdown(test_preds, test_home, test_away, config)
    section_representation_analysis(model, test_loader, device, use_amp)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
