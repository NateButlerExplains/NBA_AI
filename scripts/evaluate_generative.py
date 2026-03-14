"""Evaluate Phase 4 generative model via autoregressive rollout.

Usage::

    python scripts/evaluate_generative.py \\
        --config configs/generative/exp1_baseline.yaml \\
        --checkpoint checkpoints/generative/gen_exp1_baseline/best.pt \\
        --split test --n-rollouts 100 --temperature 1.0
"""

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.generative.config import GenerativeExperimentConfig
from src.generative.dataset import GenerativeDataset, generative_collate
from src.generative.models.generative_model import GenerativeModel
from src.generative.inference.rollout import AutoregressiveRollout

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generative model via rollout"
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ---- Config & dataset --------------------------------------------------
    config = GenerativeExperimentConfig.from_yaml(args.config)

    use_simple = config.model.use_simplified_context
    use_compressed = config.model.use_scoring_events_only
    max_se = config.model.max_scoring_events
    ds = GenerativeDataset(
        config.data,
        split=args.split,
        use_simplified_context=use_simple,
        use_scoring_events_only=use_compressed,
        max_scoring_events=max_se,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=generative_collate,
    )

    logger.info(f"Evaluating {args.split} split: {len(ds)} games")

    # ---- Model -------------------------------------------------------------
    model = GenerativeModel(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Support both raw state_dict and full checkpoint format
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint from {args.checkpoint} "
        f"({sum(p.numel() for p in model.parameters()):,} params)"
    )

    rollout_engine = AutoregressiveRollout(model, config, device=device)

    # ---- Evaluate ----------------------------------------------------------
    all_spreads_pred: list[float] = []
    all_spreads_true: list[float] = []
    all_win_probs: list[float] = []
    all_win_true: list[int] = []

    n_games = min(len(ds), args.max_games) if args.max_games else len(ds)

    for i, batch in enumerate(loader):
        if i >= n_games:
            break

        # Move tensors to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Run rollouts
        result = rollout_engine.rollout(
            batch,
            n_rollouts=args.n_rollouts,
            temperature=args.temperature,
        )

        # Ground truth
        true_margin = batch["final_margin"].item()
        true_win = 1 if true_margin > 0 else 0

        all_spreads_pred.append(result["spread_mean"])
        all_spreads_true.append(true_margin)
        all_win_probs.append(result["win_prob"])
        all_win_true.append(true_win)

        if (i + 1) % 50 == 0:
            running_mae = np.mean(
                np.abs(np.array(all_spreads_pred) - np.array(all_spreads_true))
            )
            logger.info(f"Game {i + 1}/{n_games}: running MAE = {running_mae:.2f}")

    # ---- Final metrics -----------------------------------------------------
    spreads_pred = np.array(all_spreads_pred)
    spreads_true = np.array(all_spreads_true)
    win_probs = np.array(all_win_probs)
    win_true = np.array(all_win_true)

    mae = float(np.mean(np.abs(spreads_pred - spreads_true)))
    rmse = float(np.sqrt(np.mean((spreads_pred - spreads_true) ** 2)))

    # Win AUC
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(win_true, win_probs))
    except (ValueError, ImportError):
        auc = float("nan")

    win_acc = float(np.mean((win_probs > 0.5) == win_true))

    print(f"\n{'=' * 60}")
    print(f"Phase 4 Generative Model Evaluation ({args.split})")
    print(f"{'=' * 60}")
    print(f"Games evaluated:    {len(spreads_pred)}")
    print(f"Rollouts per game:  {args.n_rollouts}")
    print(f"Temperature:        {args.temperature}")
    print(f"{'=' * 60}")
    print(f"Spread MAE:         {mae:.2f}")
    print(f"Spread RMSE:        {rmse:.2f}")
    print(f"Win AUC:            {auc:.3f}")
    print(f"Win Accuracy:       {win_acc:.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
