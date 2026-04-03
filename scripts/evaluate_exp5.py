"""Evaluate Phase 4 Exp 5: outcome head + rollout, with prefix-length ablation.

Two prediction modes:
  1. Outcome head direct: (mu, sigma) from last valid position -> spread prediction
  2. Rollout: N trajectories via AutoregressiveRollout -> aggregate spread/win

Prefix-length ablation: N=0 (pre-game), end-Q1 (~25%), halftime (~50%), end-Q3 (~75%)

Critical diagnostic: Outcome Head MAE vs Rollout MAE side by side.
Comparison table with Phase 3 ensemble and Poisson baseline.

Usage::

    python scripts/evaluate_exp5.py \\
        --config configs/generative/exp5_full_context.yaml \\
        --checkpoint checkpoints/generative/gen_exp5_full_context/best.pt \\
        --split test --n-rollouts 100

    # Quick test with fewer games
    python scripts/evaluate_exp5.py \\
        --config configs/generative/exp5_full_context.yaml \\
        --checkpoint checkpoints/generative/gen_exp5_full_context/best.pt \\
        --max-games 50 --skip-prefix-ablation
"""

import argparse
import logging
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.inference.rollout import AutoregressiveRollout
from src.generative.models.generative_model import GenerativeModel

logger = logging.getLogger(__name__)

# Prefix-length ablation: target game_progress values
PREFIX_CONFIGS = {
    "pre_game": 0.0,  # N=0: no observed states
    "end_Q1": 0.25,  # ~end of first quarter
    "halftime": 0.50,  # ~end of second quarter
    "end_Q3": 0.75,  # ~end of third quarter
}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (10-bin)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return float(ece)


def find_prefix_end(states: torch.Tensor, target_progress: float) -> int:
    """Find the scoring event index closest to target game_progress.

    Args:
        states: (T, D) state tensor, game_progress at index 2.
        target_progress: target game_progress value (0.25, 0.50, 0.75).

    Returns:
        Number of states to include in prefix. 0 means pre-game.
    """
    if target_progress <= 0:
        return 0
    progress = states[:, 2]  # (T,)
    # Find closest index to target_progress
    diffs = (progress - target_progress).abs()
    closest_idx = int(diffs.argmin().item())
    # Include up to and including that state
    return closest_idx + 1


def compute_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    win_probs: np.ndarray,
    win_true: np.ndarray,
) -> dict:
    """Compute spread MAE, RMSE, Win AUC, Win Accuracy, ECE."""
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(win_true, win_probs))
    except (ValueError, ImportError):
        auc = float("nan")

    win_acc = float(np.mean((win_probs > 0.5) == win_true))
    ece = compute_ece(win_probs, win_true)

    return {
        "spread_mae": mae,
        "spread_rmse": rmse,
        "win_auc": auc,
        "win_acc": win_acc,
        "ece": ece,
    }


def evaluate_outcome_head(
    model: GenerativeModel,
    loader: DataLoader,
    device: torch.device,
    max_games: int,
    prefix_progress: float = 1.0,
) -> dict:
    """Evaluate outcome head at a given prefix length.

    Forward pass through the model, read (mu, sigma) at the last valid
    position of the given prefix to produce a spread prediction.

    Args:
        model: trained model in eval mode.
        loader: data loader (batch_size=1).
        device: torch device.
        max_games: max number of games to evaluate.
        prefix_progress: target game_progress for prefix cutoff (1.0 = full game).

    Returns:
        dict with spread_mae, spread_rmse, win_auc, win_acc, ece.
    """
    model.eval()
    has_outcome_head = hasattr(model, "outcome_head") and model.outcome_head is not None
    if not has_outcome_head:
        return {"spread_mae": float("nan"), "spread_rmse": float("nan")}

    all_pred = []
    all_true = []
    all_win_prob = []
    all_win_true = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_games:
                break
            if batch is None:
                continue

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward through model to get outcome head predictions
            context_tokens = model.encode_context(batch)
            cond = model.pool_context(context_tokens)

            states = batch["states"]  # (1, T, D)
            T = states.shape[1]

            # Determine prefix end
            if prefix_progress < 1.0:
                prefix_end = find_prefix_end(states[0], prefix_progress)
                if prefix_end == 0:
                    # Pre-game: use just the initial state
                    prefix_end = 1
            else:
                prefix_end = T - 1  # All states except last

            input_states = states[:, :prefix_end, :]
            state_embed = model.state_embedder(input_states)
            decoder_out = model.decoder(state_embed, cond)

            outcome_mu, outcome_sigma = model.outcome_head(decoder_out)
            # Use last position for prediction
            pred_margin = outcome_mu[0, -1].item() * 50.0  # denormalize
            pred_sigma = outcome_sigma[0, -1].item() * 50.0

            true_margin = batch["final_margin"].item()
            true_win = 1 if true_margin > 0 else 0

            # Win probability from Gaussian CDF: P(margin > 0)
            if pred_sigma > 0:
                z = pred_margin / pred_sigma
                win_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            else:
                win_prob = 1.0 if pred_margin > 0 else 0.0

            all_pred.append(pred_margin)
            all_true.append(true_margin)
            all_win_prob.append(win_prob)
            all_win_true.append(true_win)

    if not all_pred:
        return {"spread_mae": float("nan"), "spread_rmse": float("nan")}

    return compute_metrics(
        np.array(all_pred),
        np.array(all_true),
        np.array(all_win_prob),
        np.array(all_win_true),
    )


def evaluate_rollout(
    rollout_engine: AutoregressiveRollout,
    loader: DataLoader,
    device: torch.device,
    n_rollouts: int,
    temperature: float,
    max_games: int,
    prefix_progress: float = 0.0,
) -> dict:
    """Evaluate via rollout, optionally from a prefix.

    For pre-game (prefix_progress=0), uses standard rollout().
    For in-game prefixes, uses rollout_from_prefix() with observed states
    up to the target game_progress.

    Args:
        rollout_engine: AutoregressiveRollout instance.
        loader: data loader (batch_size=1).
        device: torch device.
        n_rollouts: rollouts per game.
        temperature: sampling temperature.
        max_games: max games to evaluate.
        prefix_progress: target game_progress for prefix cutoff (0 = pre-game).

    Returns:
        dict with spread_mae, spread_rmse, win_auc, win_acc, ece, outcome_mae.
    """
    all_pred = []
    all_true = []
    all_win_prob = []
    all_win_true = []
    all_outcome_pred = []

    for i, batch in enumerate(loader):
        if i >= max_games:
            break
        if batch is None:
            continue

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        true_margin = batch["final_margin"].item()
        true_win = 1 if true_margin > 0 else 0

        if prefix_progress > 0:
            # Rollout from prefix: use observed states up to target progress
            states = batch["states"]  # (1, T, D)
            prefix_end = find_prefix_end(states[0], prefix_progress)
            if prefix_end == 0:
                prefix_end = 1
            observed = states[:, :prefix_end, :]

            result = rollout_engine.rollout_from_prefix(
                batch,
                observed,
                n_rollouts=n_rollouts,
                temperature=temperature,
            )
        else:
            # Pre-game: standard rollout (no observed states)
            result = rollout_engine.rollout(
                batch,
                n_rollouts=n_rollouts,
                temperature=temperature,
            )

        pred_spread = result.get("adjusted_spread", result["spread_mean"])
        all_pred.append(pred_spread)
        all_true.append(true_margin)
        all_win_prob.append(result["win_prob"])
        all_win_true.append(true_win)

        if result.get("outcome_spread") is not None:
            all_outcome_pred.append(result["outcome_spread"])

        if (i + 1) % 50 == 0:
            running_mae = np.mean(np.abs(np.array(all_pred) - np.array(all_true)))
            logger.info(f"  Game {i + 1}/{max_games}: running MAE = {running_mae:.2f}")

    if not all_pred:
        return {"spread_mae": float("nan"), "spread_rmse": float("nan")}

    pred = np.array(all_pred)
    true = np.array(all_true)
    win_probs = np.array(all_win_prob)
    win_true = np.array(all_win_true)

    metrics = compute_metrics(pred, true, win_probs, win_true)

    # Outcome head MAE from rollout (pre-game position 0 estimate)
    if all_outcome_pred:
        outcome_pred = np.array(all_outcome_pred)
        outcome_mae = float(np.mean(np.abs(outcome_pred - true[: len(outcome_pred)])))
        metrics["outcome_mae"] = outcome_mae

    return metrics


def format_metrics(m: dict, label: str, width: int = 14) -> str:
    """Format a single metrics row for the comparison table."""
    mae = m.get("spread_mae", float("nan"))
    rmse = m.get("spread_rmse", float("nan"))
    auc = m.get("win_auc", float("nan"))
    win_acc = m.get("win_acc", float("nan"))
    ece = m.get("ece", float("nan"))
    return (
        f"  {label:{width}s}  "
        f"MAE {mae:6.2f}  "
        f"RMSE {rmse:6.2f}  "
        f"AUC {auc:.3f}  "
        f"Win% {win_acc:.1%}  "
        f"ECE {ece:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 4 Exp 5 generative model"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-rollouts", type=int, default=100, help="Rollouts per game")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-games", type=int, default=None, help="Limit games to evaluate"
    )
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="Skip rollout evaluation (outcome head only)",
    )
    parser.add_argument(
        "--skip-prefix-ablation",
        action="store_true",
        help="Skip prefix-length ablation (pre-game only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ---- Config & model ----------------------------------------------------
    config = GenerativeExperimentConfig.from_yaml(args.config)

    use_full = config.model.use_full_context
    use_simple = config.model.use_simplified_context
    use_compressed = config.model.use_scoring_events_only
    max_se = config.model.max_scoring_events

    ds = GenerativeDataset(
        config.data,
        split=args.split,
        use_simplified_context=use_simple,
        use_full_context=use_full,
        use_scoring_events_only=use_compressed,
        max_scoring_events=max_se,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=generative_collate,
    )

    model = GenerativeModel(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded checkpoint from {args.checkpoint} ({n_params:,} params)")
    logger.info(f"Evaluating {args.split} split: {len(ds)} games")

    max_games = args.max_games or len(ds)

    # ---- Header ------------------------------------------------------------
    print(f"\n{'=' * 76}")
    print("Phase 4 Exp 5 Evaluation")
    print(f"{'=' * 76}")
    print(f"  Split:           {args.split} ({max_games} games)")
    print(f"  Rollouts/game:   {args.n_rollouts}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Full context:    {use_full}")
    print()

    # Collect all results for final comparison table
    all_results: dict[str, dict] = {}

    # ---- Mode 1: Outcome Head Direct ---------------------------------------
    print(f"{'=' * 76}")
    print("Mode 1: Outcome Head Direct (forward pass, read mu/sigma at last position)")
    print(f"{'=' * 76}")

    oh_full = evaluate_outcome_head(
        model, loader, device, max_games, prefix_progress=1.0
    )
    if not math.isnan(oh_full["spread_mae"]):
        print(format_metrics(oh_full, "Full-game"))
        all_results["OH full-game"] = oh_full
    else:
        print("  Outcome head not available in this model.")

    # Prefix ablation for outcome head
    oh_prefix_results: dict[str, dict] = {}
    if not args.skip_prefix_ablation and not math.isnan(
        oh_full.get("spread_mae", float("nan"))
    ):
        print()
        for name, progress in PREFIX_CONFIGS.items():
            oh_prefix = evaluate_outcome_head(
                model,
                loader,
                device,
                max_games,
                prefix_progress=progress,
            )
            if not math.isnan(oh_prefix["spread_mae"]):
                print(format_metrics(oh_prefix, f"OH {name}"))
                oh_prefix_results[name] = oh_prefix
                all_results[f"OH {name}"] = oh_prefix

    # ---- Mode 2: Rollout ---------------------------------------------------
    rollout_results_by_prefix: dict[str, dict] = {}
    rollout_pregame: dict = {}

    if not args.skip_rollout:
        print(f"\n{'=' * 76}")
        print("Mode 2: Rollout (autoregressive generation)")
        print(f"{'=' * 76}")

        rollout_engine = AutoregressiveRollout(model, config, device=device)

        # Pre-game rollout
        logger.info("Running pre-game rollout evaluation...")
        rollout_pregame = evaluate_rollout(
            rollout_engine,
            loader,
            device,
            args.n_rollouts,
            args.temperature,
            max_games,
            prefix_progress=0.0,
        )
        print(format_metrics(rollout_pregame, "pre_game"))
        if "outcome_mae" in rollout_pregame:
            print(
                f"                 Outcome Head MAE (pos 0): {rollout_pregame['outcome_mae']:.2f}"
            )
        all_results["Rollout pre_game"] = rollout_pregame
        rollout_results_by_prefix["pre_game"] = rollout_pregame

        # Prefix ablation for rollout
        if not args.skip_prefix_ablation:
            print()
            for name, progress in PREFIX_CONFIGS.items():
                if progress == 0.0:
                    continue  # Already did pre-game
                logger.info(f"Running {name} prefix rollout evaluation...")
                prefix_result = evaluate_rollout(
                    rollout_engine,
                    loader,
                    device,
                    args.n_rollouts,
                    args.temperature,
                    max_games,
                    prefix_progress=progress,
                )
                print(format_metrics(prefix_result, name))
                rollout_results_by_prefix[name] = prefix_result
                all_results[f"Rollout {name}"] = prefix_result

    # ---- Critical Diagnostic: Outcome Head MAE vs Rollout MAE --------------
    if not args.skip_rollout and not math.isnan(
        oh_full.get("spread_mae", float("nan"))
    ):
        print(f"\n{'=' * 76}")
        print("CRITICAL DIAGNOSTIC: Outcome Head vs Rollout")
        print(f"{'=' * 76}")
        oh_mae = oh_full["spread_mae"]
        rollout_mae = rollout_pregame.get("spread_mae", float("nan"))
        outcome_at_rollout = rollout_pregame.get("outcome_mae", float("nan"))

        print(f"  Outcome Head MAE (full game, teacher-forced):  {oh_mae:.2f}")
        print(f"  Rollout MAE (pre-game, 100 trajectories):      {rollout_mae:.2f}")
        if not math.isnan(outcome_at_rollout):
            print(
                f"  Outcome Head MAE (pre-game, pos 0):            {outcome_at_rollout:.2f}"
            )
        print()
        if not math.isnan(rollout_mae):
            if oh_mae < rollout_mae:
                print("  >> Outcome Head WINS -- rollouts may add noise, not signal")
            else:
                print("  >> Rollout WINS -- generation adds distributional value")
            diff = abs(oh_mae - rollout_mae)
            print(f"     Difference: {diff:.2f} pts")

    # ---- Comparison Table --------------------------------------------------
    print(f"\n{'=' * 76}")
    print("COMPARISON TABLE")
    print(f"{'=' * 76}")
    header = f"  {'Model':<24s}  {'MAE':>6s}  {'RMSE':>6s}  {'AUC':>5s}  {'Win%':>6s}  {'ECE':>6s}"
    print(header)
    print(f"  {'-' * 24}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 6}  {'-' * 6}")

    # Reference baselines (fixed numbers from prior experiments)
    print(
        f"  {'Phase 3 Ensemble':<24s}  {10.66:6.2f}  {'--':>6s}  {0.718:5.3f}  {66.5:5.1f}%  {'--':>6s}"
    )
    print(
        f"  {'Poisson Baseline':<24s}  {11.98:6.2f}  {'--':>6s}  {0.700:5.3f}  {64.8:5.1f}%  {'--':>6s}"
    )
    print(
        f"  {'Phase 4 Exp 4':<24s}  {11.76:6.2f}  {'--':>6s}  {0.662:5.3f}  {'--':>6s}  {'--':>6s}"
    )

    # Exp 5 results
    for label, m in all_results.items():
        mae = m.get("spread_mae", float("nan"))
        rmse_val = m.get("spread_rmse", float("nan"))
        auc = m.get("win_auc", float("nan"))
        win_acc = m.get("win_acc", float("nan"))
        ece = m.get("ece", float("nan"))
        rmse_str = f"{rmse_val:6.2f}" if not math.isnan(rmse_val) else "--".rjust(6)
        auc_str = f"{auc:5.3f}" if not math.isnan(auc) else "--".rjust(5)
        ece_str = f"{ece:6.4f}" if not math.isnan(ece) else "--".rjust(6)
        win_str = f"{win_acc * 100:5.1f}%" if not math.isnan(win_acc) else "--".rjust(6)
        print(
            f"  {label:<24s}  {mae:6.2f}  {rmse_str}  {auc_str}  {win_str}  {ece_str}"
        )

    print(f"\n{'=' * 76}")


if __name__ == "__main__":
    main()
