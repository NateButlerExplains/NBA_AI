#!/usr/bin/env python
"""
Ensemble evaluation for Phase 3 Exp 9: Deep Ensemble (3 seeds).

Loads 3 trained Phase2Model checkpoints, runs inference on val/test sets,
and combines predictions using:
  - Spread/score means: simple average of mu values
  - Sigma: mixture-of-Gaussians (captures model disagreement)
  - Win probability: logit averaging (better calibrated than prob averaging)

Usage:
    # Train seeds 137 and 256 first (seed 42 = existing Exp 4):
    python scripts/train_phase2.py --config configs/transformer/phase3_exp4_interaction.yaml \
        --seed 137 --experiment-name phase3_exp9_seed137
    python scripts/train_phase2.py --config configs/transformer/phase3_exp4_interaction.yaml \
        --seed 256 --experiment-name phase3_exp9_seed256

    # Then evaluate ensemble:
    python scripts/ensemble_evaluate.py \
        --config configs/transformer/phase3_exp4_interaction.yaml \
        --checkpoints checkpoints/phase3/phase3_exp4_interaction/best.pt \
                      checkpoints/phase3/phase3_exp9_seed137/best.pt \
                      checkpoints/phase3/phase3_exp9_seed256/best.pt
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import load_cache
from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2
from src.transformer.phase2.models.phase2_model import Phase2Model
from src.transformer.models.prediction_heads import GamePrediction
from src.transformer.training.metrics import MetricsCalculator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_with_ema(config, checkpoint_path: str, device: str) -> Phase2Model:
    """Create model from config and load EMA weights from checkpoint."""
    model = Phase2Model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema_state_dict" in checkpoint:
        shadow = checkpoint["ema_state_dict"]["shadow"]
        missing, unexpected = model.load_state_dict(shadow, strict=False)
    else:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )

    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def run_inference(
    model: Phase2Model,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, np.ndarray]:
    """Run model inference, collecting all prediction components."""
    spread_means = []
    spread_stds = []
    home_means = []
    home_stds = []
    away_means = []
    away_stds = []
    win_probs = []
    target_homes = []
    target_aways = []

    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.autocast(
                "cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"
            ):
                pred = model(batch_device)

            spread_means.append(pred.spread_mean.float().cpu().numpy())
            spread_stds.append(pred.spread_std.float().cpu().numpy())
            home_means.append(pred.home_score_mean.float().cpu().numpy())
            home_stds.append(pred.home_score_std.float().cpu().numpy())
            away_means.append(pred.away_score_mean.float().cpu().numpy())
            away_stds.append(pred.away_score_std.float().cpu().numpy())
            win_probs.append(pred.home_win_prob.float().cpu().numpy())
            target_homes.append(batch["target_home_scores"].numpy())
            target_aways.append(batch["target_away_scores"].numpy())

    return {
        "spread_mean": np.concatenate(spread_means),
        "spread_std": np.concatenate(spread_stds),
        "home_mean": np.concatenate(home_means),
        "home_std": np.concatenate(home_stds),
        "away_mean": np.concatenate(away_means),
        "away_std": np.concatenate(away_stds),
        "win_prob": np.concatenate(win_probs),
        "target_home": np.concatenate(target_homes),
        "target_away": np.concatenate(target_aways),
    }


def ensemble_predictions(
    all_preds: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Combine predictions from multiple models.

    Aggregation strategy:
      - Means (spread, home, away): simple average
      - Sigma: mixture-of-Gaussians variance
          sigma_ens^2 = mean(sigma_i^2) + var(mu_i)
        This captures BOTH per-model uncertainty AND inter-model disagreement.
      - Win probability: logit averaging (better calibrated than prob averaging)
    """
    n_models = len(all_preds)

    # Stack predictions: (n_models, n_games)
    spread_means = np.stack([p["spread_mean"] for p in all_preds])
    spread_stds = np.stack([p["spread_std"] for p in all_preds])
    home_means = np.stack([p["home_mean"] for p in all_preds])
    home_stds = np.stack([p["home_std"] for p in all_preds])
    away_means = np.stack([p["away_mean"] for p in all_preds])
    away_stds = np.stack([p["away_std"] for p in all_preds])
    win_probs = np.stack([p["win_prob"] for p in all_preds])

    # Average means
    ens_spread_mean = spread_means.mean(axis=0)
    ens_home_mean = home_means.mean(axis=0)
    ens_away_mean = away_means.mean(axis=0)

    # Mixture-of-Gaussians sigma
    # Var_mixture = E[sigma^2] + Var[mu]
    # = mean(sigma_i^2) + mean(mu_i^2) - mean(mu_i)^2
    ens_spread_var = (
        np.mean(spread_stds**2, axis=0)
        + np.mean(spread_means**2, axis=0)
        - ens_spread_mean**2
    )
    ens_spread_std = np.sqrt(np.maximum(ens_spread_var, 1e-6))

    ens_home_var = (
        np.mean(home_stds**2, axis=0)
        + np.mean(home_means**2, axis=0)
        - ens_home_mean**2
    )
    ens_home_std = np.sqrt(np.maximum(ens_home_var, 1e-6))

    ens_away_var = (
        np.mean(away_stds**2, axis=0)
        + np.mean(away_means**2, axis=0)
        - ens_away_mean**2
    )
    ens_away_std = np.sqrt(np.maximum(ens_away_var, 1e-6))

    # Win probability: logit averaging
    # Clip to avoid log(0) or log(inf)
    eps = 1e-7
    win_probs_clipped = np.clip(win_probs, eps, 1 - eps)
    logits = np.log(win_probs_clipped / (1 - win_probs_clipped))
    avg_logit = logits.mean(axis=0)
    ens_win_prob = 1.0 / (1.0 + np.exp(-avg_logit))

    return {
        "spread_mean": ens_spread_mean,
        "spread_std": ens_spread_std,
        "home_mean": ens_home_mean,
        "home_std": ens_home_std,
        "away_mean": ens_away_mean,
        "away_std": ens_away_std,
        "win_prob": ens_win_prob,
        "target_home": all_preds[0]["target_home"],
        "target_away": all_preds[0]["target_away"],
    }


def compute_metrics_from_arrays(preds: dict[str, np.ndarray]):
    """Compute full MetricResults from numpy prediction arrays."""
    calc = MetricsCalculator()

    # Feed all at once as a single "batch"
    game_pred = GamePrediction(
        spread_mean=torch.tensor(preds["spread_mean"], dtype=torch.float32),
        spread_std=torch.tensor(preds["spread_std"], dtype=torch.float32),
        home_score_mean=torch.tensor(preds["home_mean"], dtype=torch.float32),
        home_score_std=torch.tensor(preds["home_std"], dtype=torch.float32),
        away_score_mean=torch.tensor(preds["away_mean"], dtype=torch.float32),
        away_score_std=torch.tensor(preds["away_std"], dtype=torch.float32),
        home_win_prob=torch.tensor(preds["win_prob"], dtype=torch.float32),
    )
    calc.update(
        game_pred,
        torch.tensor(preds["target_home"], dtype=torch.float32),
        torch.tensor(preds["target_away"], dtype=torch.float32),
    )
    return calc.compute()


def print_results(label: str, metrics):
    """Print formatted metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Spread MAE:      {metrics.spread_mae:.2f}")
    print(f"  Spread RMSE:     {metrics.spread_rmse:.2f}")
    print(f"  Home MAE:        {metrics.home_mae:.2f}")
    print(f"  Away MAE:        {metrics.away_mae:.2f}")
    print(f"  Total MAE:       {metrics.total_mae:.2f}")
    print(f"  Win Accuracy:    {metrics.win_accuracy:.3f}")
    print(f"  Win AUC:         {metrics.win_auc:.3f}")
    print(f"  Brier Score:     {metrics.brier_score:.4f}")
    print(f"  ECE:             {metrics.ece:.4f}")
    print(f"  90% Coverage:    {metrics.spread_coverage_90:.3f}")
    print(f"  Spread Sharpness:{metrics.spread_sharpness:.1f}")
    print(f"  Samples:         {metrics.n_samples}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation (Exp 9)")
    parser.add_argument("--config", type=str, required=True,
                        help="Base model config (Exp 4 yaml)")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to 2+ checkpoint files")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/phase3/phase3_exp9_ensemble")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    n_models = len(args.checkpoints)
    logger.info(f"Ensemble of {n_models} models")
    for i, cp in enumerate(args.checkpoints):
        logger.info(f"  Model {i + 1}: {cp}")

    # Load config
    config = Phase2ExperimentConfig.from_yaml(args.config)
    device = args.device or config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    use_amp = config.training.use_amp
    amp_dtype_str = getattr(config.training, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Load cache and create datasets
    logger.info(f"Loading cache from {config.data.cache_dir}")
    cache = load_cache(config.data.cache_dir)

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
        n_player_stats=config.data.n_player_stats,
        enable_team_gat=config.model.enable_team_gat,
        n_efficiency_features=config.data.n_efficiency_features,
        player_experience=cache.get("player_experience", {}),
    )

    splits = {
        "val": config.data.val_seasons,
        "test": config.data.test_seasons,
    }

    # Pre-build datasets and loaders (shared across all models)
    loaders = {}
    for split_name, seasons in splits.items():
        dataset = Phase2Dataset(
            seasons=seasons,
            enable_augmentation=False,
            **common_kwargs,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_phase2,
            num_workers=0,
            pin_memory=device == "cuda",
        )
        loaders[split_name] = loader
        logger.info(f"{split_name}: {len(dataset)} games")

    # Run inference for each model on each split
    all_split_preds = {split: [] for split in splits}
    individual_metrics = {split: [] for split in splits}

    for i, checkpoint_path in enumerate(args.checkpoints):
        logger.info(f"\nLoading model {i + 1}/{n_models}: {checkpoint_path}")
        model = load_model_with_ema(config, checkpoint_path, device)

        for split_name in splits:
            logger.info(f"  Running inference on {split_name}...")
            preds = run_inference(
                model, loaders[split_name], device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            all_split_preds[split_name].append(preds)

            # Individual model metrics
            metrics = compute_metrics_from_arrays(preds)
            individual_metrics[split_name].append(metrics)
            logger.info(
                f"  Model {i + 1} {split_name}: "
                f"MAE={metrics.spread_mae:.2f}, "
                f"AUC={metrics.win_auc:.3f}, "
                f"Acc={metrics.win_accuracy:.3f}"
            )

        # Free GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Ensemble predictions and evaluate
    ensemble_metrics = {}
    for split_name in splits:
        ens_preds = ensemble_predictions(all_split_preds[split_name])
        metrics = compute_metrics_from_arrays(ens_preds)
        ensemble_metrics[split_name] = metrics

    # Print comparison table
    print("\n" + "=" * 80)
    print("PHASE 3 EXP 9: DEEP ENSEMBLE RESULTS")
    print("=" * 80)

    for split_name in splits:
        header = (
            f"| {'Model':<20s} | {'Spread MAE':>10s} | {'RMSE':>6s} | "
            f"{'Win AUC':>8s} | {'Win Acc':>8s} | {'ECE':>6s} | {'90% Cov':>8s} |"
        )
        sep = f"|{'-' * 22}|{'-' * 12}|{'-' * 8}|{'-' * 10}|{'-' * 10}|{'-' * 8}|{'-' * 10}|"

        print(f"\n  {split_name.upper()} SET")
        print(header)
        print(sep)

        for i, m in enumerate(individual_metrics[split_name]):
            print(
                f"| {'Seed ' + str(i + 1):<20s} | {m.spread_mae:>10.2f} | {m.spread_rmse:>6.2f} | "
                f"{m.win_auc:>8.3f} | {m.win_accuracy * 100:>7.1f}% | {m.ece:>6.4f} | {m.spread_coverage_90:>7.3f}  |"
            )

        m = ensemble_metrics[split_name]
        print(
            f"| {'**ENSEMBLE**':<20s} | {m.spread_mae:>10.2f} | {m.spread_rmse:>6.2f} | "
            f"{m.win_auc:>8.3f} | {m.win_accuracy * 100:>7.1f}% | {m.ece:>6.4f} | {m.spread_coverage_90:>7.3f}  |"
        )
        print(sep)

    print("=" * 80)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_models": n_models,
        "checkpoints": args.checkpoints,
    }
    for split_name in splits:
        results[f"{split_name}_individual"] = [
            asdict(m) for m in individual_metrics[split_name]
        ]
        results[f"{split_name}_ensemble"] = asdict(ensemble_metrics[split_name])

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
