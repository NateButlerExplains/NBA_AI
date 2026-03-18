#!/usr/bin/env python
"""
Phase 6 Experiment 1: Add Vegas betting data as features to the Phase 3 transformer.

Tests whether combining model predictions with Vegas closing lines in a meta-model
can beat both the standalone model and Vegas alone. This is a wrapper approach that
does NOT modify any src/ code.

Two-stage pipeline:
  Stage 1 (--extract): Run Phase 3 Exp 4 inference, save per-game predictions + game_ids.
  Stage 2 (--train):   Load predictions + betting data, train meta-models, evaluate.

Usage:
    # Stage 1: Extract model predictions (needs GPU, ~5 min)
    python scripts/phase6_exp1_betting_features.py --extract \
        --config configs/transformer/phase3_exp4_interaction.yaml \
        --checkpoint checkpoints/phase3/phase3_exp4_interaction/best.pt

    # Stage 2: Train meta-models (CPU, fast)
    python scripts/phase6_exp1_betting_features.py --train

    # Both stages in one go:
    python scripts/phase6_exp1_betting_features.py --extract --train \
        --config configs/transformer/phase3_exp4_interaction.yaml \
        --checkpoint checkpoints/phase3/phase3_exp4_interaction/best.pt

    # Use LLM predictions instead of transformer:
    python scripts/phase6_exp1_betting_features.py --train --model-source llm

    # Vegas-only baseline (no model predictions):
    python scripts/phase6_exp1_betting_features.py --train --model-source none
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# Optional: XGBoost
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Optional: PyTorch (only needed for Stage 1)
try:
    import torch
    from torch.utils.data import DataLoader, Subset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase6_exp1"
PREDICTIONS_FILE = OUTPUT_DIR / "model_predictions.npz"
LLM_PREDICTIONS_FILE = (
    PROJECT_ROOT / "data" / "exp7" / "test_gpt-5.4-mini_results.jsonl"
)

# ── Season splits (matching Phase 3 Exp 4 config) ───────────────────────────

TRAIN_SEASONS = [f"{y}-{y + 1}" for y in range(2008, 2023)]
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: Extract Phase 3 model predictions
# =============================================================================


def extract_model_predictions(
    config_path: str, checkpoint_path: str, device: str = None
):
    """Run Phase 3 Exp 4 inference and save per-game predictions with game_ids."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for --extract. Install torch first.")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.transformer.phase2.config import Phase2ExperimentConfig
    from src.transformer.phase2.cache_builder import load_cache
    from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2
    from src.transformer.phase2.models.phase2_model import Phase2Model

    # Load config
    config = Phase2ExperimentConfig.from_yaml(config_path)
    if device is None:
        device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.info(f"Device: {device}")

    use_amp = config.training.use_amp
    amp_dtype_str = getattr(config.training, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Load cache
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

    # Load model
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
        logger.warning(f"Missing keys: {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}...")
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # Run inference on each split
    all_results = {}
    for split_name, seasons in [
        ("train", TRAIN_SEASONS),
        ("val", VAL_SEASONS),
        ("test", TEST_SEASONS),
    ]:
        logger.info(f"Running inference on {split_name}: {seasons}")
        dataset = Phase2Dataset(
            seasons=seasons,
            enable_augmentation=False,
            **common_kwargs,
        )

        # Pre-validate to keep game_id alignment
        valid_indices = []
        valid_game_ids = []
        for i in range(len(dataset)):
            game_id = dataset.game_ids[i]
            sample = dataset.builder.build_sample(game_id)
            if sample is not None:
                valid_indices.append(i)
                valid_game_ids.append(game_id)

        n_skipped = len(dataset) - len(valid_indices)
        if n_skipped > 0:
            logger.warning(
                f"  {split_name}: skipped {n_skipped}/{len(dataset)} invalid games"
            )

        valid_dataset = Subset(dataset, valid_indices)
        loader = DataLoader(
            valid_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate_phase2,
            num_workers=0,
            pin_memory=device == "cuda",
        )

        # Collect predictions
        spread_means, spread_stds = [], []
        home_means, home_stds = [], []
        away_means, away_stds = [], []
        win_probs = []
        target_homes, target_aways = [], []

        with torch.inference_mode():
            for batch in loader:
                if batch is None:
                    continue
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.autocast(
                    "cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"
                ):
                    pred = model(batch_dev)

                spread_means.append(pred.spread_mean.float().cpu().numpy())
                spread_stds.append(pred.spread_std.float().cpu().numpy())
                home_means.append(pred.home_score_mean.float().cpu().numpy())
                home_stds.append(pred.home_score_std.float().cpu().numpy())
                away_means.append(pred.away_score_mean.float().cpu().numpy())
                away_stds.append(pred.away_score_std.float().cpu().numpy())
                win_probs.append(pred.home_win_prob.float().cpu().numpy())
                target_homes.append(batch["target_home_scores"].numpy())
                target_aways.append(batch["target_away_scores"].numpy())

        all_results[split_name] = {
            "game_ids": np.array(valid_game_ids),
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
        logger.info(f"  {split_name}: {len(valid_game_ids)} games extracted")

    # Combine and save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = {}
    for key in all_results["train"].keys():
        combined[key] = np.concatenate(
            [all_results[s][key] for s in ["train", "val", "test"]], axis=0
        )
    n_train = len(all_results["train"]["game_ids"])
    n_val = len(all_results["val"]["game_ids"])
    n_test = len(all_results["test"]["game_ids"])
    combined["split_sizes"] = np.array([n_train, n_val, n_test])

    np.savez_compressed(PREDICTIONS_FILE, **combined)
    logger.info(f"Saved predictions to {PREDICTIONS_FILE}")
    logger.info(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Free GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Betting data loading
# =============================================================================


def load_betting_data() -> pd.DataFrame:
    """Load all games with scores and betting data from the database.

    Returns DataFrame with columns:
        game_id, home_team, away_team, season, date,
        home_pts, away_pts, actual_spread, actual_total,
        vegas_spread, vegas_total, vegas_ml_home, vegas_ml_away,
        spread_result, ou_result
    """
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,
            g.season,
            g.date_time_utc,
            t_home.pts AS home_pts,
            t_away.pts AS away_pts,
            -- Best available spread (COALESCE priority)
            COALESCE(b.espn_closing_spread, b.covers_closing_spread, b.espn_current_spread) AS vegas_spread,
            -- Best available total
            COALESCE(b.espn_closing_total, b.covers_closing_total, b.espn_current_total) AS vegas_total,
            -- Best available moneyline (closing preferred, fall back to current)
            COALESCE(b.espn_closing_ml_home, b.espn_current_ml_home) AS vegas_ml_home,
            COALESCE(b.espn_closing_ml_away, b.espn_current_ml_away) AS vegas_ml_away,
            -- ATS result
            b.spread_result,
            b.ou_result
        FROM Games g
        -- Join TeamBox for scores: home team
        JOIN TeamBox t_home ON g.game_id = t_home.game_id
        JOIN Teams tm_home ON t_home.team_id = tm_home.team_id
            AND tm_home.abbreviation = g.home_team
        -- Join TeamBox for scores: away team
        JOIN TeamBox t_away ON g.game_id = t_away.game_id
        JOIN Teams tm_away ON t_away.team_id = tm_away.team_id
            AND tm_away.abbreviation = g.away_team
        -- Betting data (LEFT JOIN: not all games have betting)
        LEFT JOIN Betting b ON g.game_id = b.game_id
        WHERE g.status = 3
          AND g.season_type = 'Regular Season'
        ORDER BY g.date_time_utc
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Compute derived columns
    df["actual_spread"] = df["home_pts"] - df["away_pts"]  # positive = home won by X
    df["actual_total"] = df["home_pts"] + df["away_pts"]
    df["home_win"] = (df["actual_spread"] > 0).astype(int)

    logger.info(f"Loaded {len(df)} games from database")
    logger.info(f"  With spread: {df['vegas_spread'].notna().sum()}")
    logger.info(f"  With total: {df['vegas_total'].notna().sum()}")
    logger.info(f"  With ML: {df['vegas_ml_home'].notna().sum()}")

    return df


def ml_to_implied_prob(ml) -> float:
    """Convert American moneyline odds to implied probability.

    Positive ML (underdog): prob = 100 / (ML + 100)
    Negative ML (favorite): prob = |ML| / (|ML| + 100)
    """
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


# =============================================================================
# Feature engineering
# =============================================================================


def build_feature_matrix(
    df: pd.DataFrame,
    model_preds: dict = None,
    model_source: str = "transformer",
) -> pd.DataFrame:
    """Build feature matrix combining model predictions + betting features.

    Features:
      Betting (always available):
        1. vegas_spread_norm           - Vegas spread / 15
        2. vegas_total_norm            - (Vegas total - 224) / 20
        3. vegas_ml_home_prob          - Implied prob from home ML
        4. vegas_ml_away_prob          - Implied prob from away ML
        5. vegas_implied_spread        - Derived from ML probs (logit diff * scaling)
        6. vegas_spread_abs            - |spread| / 15 (game closeness)
        7. big_favorite                - 1 if |spread| >= 10

      Model (when available):
        8.  model_spread               - Model predicted spread
        9.  model_total                 - Model predicted total
        10. model_win_prob              - Model home win probability
        11. model_spread_std            - Model uncertainty on spread
        12. spread_disagreement         - model_spread - vegas_spread (KEY FEATURE)
        13. total_disagreement          - model_total - vegas_total
        14. prob_disagreement           - model_win_prob - vegas_ml_home_prob
        15. model_confident_disagree    - |disagree| > 3 AND model_std < median
    """
    features = pd.DataFrame(index=df.index)

    # ── Betting features (always) ──────────────────────────────────────────
    features["vegas_spread_norm"] = df["vegas_spread"] / 15.0
    features["vegas_total_norm"] = (df["vegas_total"] - 224.0) / 20.0
    features["vegas_ml_home_prob"] = df["vegas_ml_home"].apply(ml_to_implied_prob)
    features["vegas_ml_away_prob"] = df["vegas_ml_away"].apply(ml_to_implied_prob)

    # Normalize ML probs to sum to 1 (remove vig)
    prob_sum = features["vegas_ml_home_prob"] + features["vegas_ml_away_prob"]
    features["vegas_ml_home_prob_novig"] = features["vegas_ml_home_prob"] / prob_sum
    features["vegas_ml_away_prob_novig"] = features["vegas_ml_away_prob"] / prob_sum

    # Spread-derived features
    features["vegas_spread_abs"] = df["vegas_spread"].abs() / 15.0
    features["big_favorite"] = (df["vegas_spread"].abs() >= 10).astype(float)

    # ── Model features ─────────────────────────────────────────────────────
    if model_preds is not None and model_source != "none":
        # Map model predictions to df rows by game_id
        game_id_to_idx = {gid: i for i, gid in enumerate(model_preds["game_ids"])}

        model_spread = np.full(len(df), np.nan)
        model_total = np.full(len(df), np.nan)
        model_win_prob = np.full(len(df), np.nan)
        model_spread_std = np.full(len(df), np.nan)

        for i, game_id in enumerate(df["game_id"]):
            if game_id in game_id_to_idx:
                j = game_id_to_idx[game_id]
                # Model spread = -(home_mean - away_mean) convention check:
                # Phase 3 spread_mean is home - away (positive = home favored)
                # Vegas spread: negative = home favored
                # So model_spread should be NEGATED to match Vegas convention
                model_spread[i] = -model_preds["spread_mean"][j]
                model_total[i] = (
                    model_preds["home_mean"][j] + model_preds["away_mean"][j]
                )
                model_win_prob[i] = model_preds["win_prob"][j]
                model_spread_std[i] = model_preds["spread_std"][j]

        features["model_spread"] = model_spread / 15.0
        features["model_total"] = (model_total - 224.0) / 20.0
        features["model_win_prob"] = model_win_prob
        features["model_spread_std"] = model_spread_std / 15.0

        # Disagreement features (KEY — these are the most likely to add value)
        features["spread_disagreement"] = (model_spread - df["vegas_spread"]) / 15.0
        features["total_disagreement"] = (model_total - df["vegas_total"]) / 20.0
        features["prob_disagreement"] = (
            model_win_prob - features["vegas_ml_home_prob_novig"]
        )

        # Confident disagreement: model disagrees AND is confident
        median_std = np.nanmedian(model_spread_std)
        features["model_confident_disagree"] = (
            (np.abs(model_spread - df["vegas_spread"]) > 3.0)
            & (model_spread_std < median_std)
        ).astype(float)

    return features


def load_llm_predictions() -> dict:
    """Load LLM predictions from exp7 JSONL file into a dict matching model_preds format."""
    if not LLM_PREDICTIONS_FILE.exists():
        logger.warning(f"LLM predictions not found: {LLM_PREDICTIONS_FILE}")
        return None

    records = []
    with open(LLM_PREDICTIONS_FILE) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("success") and rec.get("prediction"):
                records.append(rec)

    game_ids = []
    spread_means = []
    home_means = []
    away_means = []
    win_probs = []
    spread_stds = []

    for rec in records:
        pred = rec["prediction"]
        game_ids.append(rec["game_id"])
        h = pred["home_score"]
        a = pred["away_score"]
        home_means.append(h)
        away_means.append(a)
        # LLM spread: home - away (positive = home won)
        # We'll negate in build_feature_matrix to match Vegas sign convention
        spread_means.append(h - a)
        win_probs.append(pred.get("home_win_probability", 0.5))
        spread_stds.append(12.0)  # LLM has no uncertainty estimate; use constant

    logger.info(
        f"Loaded {len(game_ids)} LLM predictions from {LLM_PREDICTIONS_FILE.name}"
    )
    return {
        "game_ids": np.array(game_ids),
        "spread_mean": np.array(spread_means, dtype=np.float32),
        "home_mean": np.array(home_means, dtype=np.float32),
        "away_mean": np.array(away_means, dtype=np.float32),
        "win_prob": np.array(win_probs, dtype=np.float32),
        "spread_std": np.array(spread_stds, dtype=np.float32),
    }


# =============================================================================
# Evaluation metrics
# =============================================================================


def compute_ats_metrics(
    pred_spread: np.ndarray,
    actual_spread: np.ndarray,
    vegas_spread: np.ndarray,
    pred_win_prob: np.ndarray = None,
    actual_home_win: np.ndarray = None,
    label: str = "",
) -> dict:
    """Compute comprehensive ATS + prediction quality metrics.

    Args:
        pred_spread: Predicted spread in Vegas convention (negative = home favored)
        actual_spread: Actual home - away score margin
        vegas_spread: Vegas closing spread (negative = home favored)
        pred_win_prob: Predicted home win probability (optional)
        actual_home_win: Binary home win indicator (optional)
        label: Name for display

    Returns:
        Dictionary of metrics.
    """
    results = {"label": label, "n_games": len(pred_spread)}

    # ── Spread MAE ─────────────────────────────────────────────────────────
    # Predicted margin = -pred_spread (convert from Vegas convention)
    pred_margin = -pred_spread
    results["spread_mae"] = mean_absolute_error(actual_spread, pred_margin)

    # ── Vegas baseline MAE ─────────────────────────────────────────────────
    vegas_margin = -vegas_spread
    results["vegas_mae"] = mean_absolute_error(actual_spread, vegas_margin)

    # ── ATS record ─────────────────────────────────────────────────────────
    # ATS: did our predicted margin beat the spread?
    # If we predict home wins by 5 and Vegas says home -3, we're backing home.
    # Home covers if actual_spread > -vegas_spread (actual margin > spread)
    # But we only bet when we disagree with Vegas.

    # Strategy: bet on home to cover when our model says home will beat the spread
    # model_margin = -pred_spread, vegas_line = -vegas_spread
    # Bet home if model_margin > vegas_margin (i.e. we think home is better than Vegas thinks)
    # Bet away if model_margin < vegas_margin

    model_margin = pred_margin
    vegas_line = vegas_margin

    # Which side to bet
    bet_home = model_margin > vegas_line  # True = bet home covers
    bet_away = model_margin < vegas_line  # True = bet away covers
    no_bet = ~bet_home & ~bet_away  # Exactly on the line

    # Home covers when actual margin > vegas line (accounting for push)
    home_covers = actual_spread > (-vegas_spread)
    away_covers = actual_spread < (-vegas_spread)
    push = actual_spread == (-vegas_spread)

    # ATS results for our bets
    wins = (bet_home & home_covers) | (bet_away & away_covers)
    losses = (bet_home & away_covers) | (bet_away & home_covers)
    pushes = (bet_home | bet_away) & push
    total_bets = wins.sum() + losses.sum() + pushes.sum()

    results["ats_wins"] = int(wins.sum())
    results["ats_losses"] = int(losses.sum())
    results["ats_pushes"] = int(pushes.sum())
    results["ats_total_bets"] = int(total_bets)
    if wins.sum() + losses.sum() > 0:
        results["ats_win_pct"] = wins.sum() / (wins.sum() + losses.sum())
    else:
        results["ats_win_pct"] = 0.0

    # Profit at -110 (risk $110 to win $100 on each bet)
    # Win: +100, Loss: -110, Push: 0
    profit = wins.sum() * 100 - losses.sum() * 110
    results["ats_profit_per_bet"] = profit / max(total_bets, 1)
    results["ats_total_profit"] = int(profit)
    results["ats_roi"] = profit / (max(total_bets, 1) * 110) * 100  # ROI %

    # ── Selective betting: only bet when disagreement exceeds threshold ─────
    disagreement = np.abs(model_margin - vegas_line)
    for threshold in [1.0, 2.0, 3.0, 5.0]:
        mask = disagreement >= threshold
        if mask.sum() < 10:
            continue
        t_wins = (mask & wins).sum()
        t_losses = (mask & losses).sum()
        t_pushes = (mask & pushes).sum()
        t_total = t_wins + t_losses + t_pushes
        t_pct = t_wins / max(t_wins + t_losses, 1)
        t_profit = t_wins * 100 - t_losses * 110
        results[f"ats_{threshold:.0f}pt_bets"] = int(t_total)
        results[f"ats_{threshold:.0f}pt_pct"] = t_pct
        results[f"ats_{threshold:.0f}pt_profit"] = int(t_profit)
        results[f"ats_{threshold:.0f}pt_roi"] = t_profit / (max(t_total, 1) * 110) * 100

    # ── Win prediction metrics ─────────────────────────────────────────────
    if pred_win_prob is not None and actual_home_win is not None:
        valid = ~np.isnan(pred_win_prob)
        if valid.sum() > 0:
            results["win_auc"] = roc_auc_score(
                actual_home_win[valid], pred_win_prob[valid]
            )
            results["win_accuracy"] = (
                (pred_win_prob[valid] > 0.5) == actual_home_win[valid].astype(bool)
            ).mean()
            results["brier_score"] = brier_score_loss(
                actual_home_win[valid], pred_win_prob[valid]
            )

    return results


def print_ats_results(results: dict):
    """Pretty-print ATS evaluation results."""
    label = results.get("label", "Model")
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Games:             {results['n_games']}")
    print(f"  Spread MAE:        {results['spread_mae']:.2f}")
    print(f"  Vegas MAE:         {results['vegas_mae']:.2f}")
    print(f"  MAE improvement:   {results['vegas_mae'] - results['spread_mae']:+.2f}")
    print()
    print(
        f"  ATS Record:        {results['ats_wins']}-{results['ats_losses']}"
        f"-{results['ats_pushes']} ({results['ats_total_bets']} bets)"
    )
    print(f"  ATS Win %:         {results['ats_win_pct']:.1%}")
    print(f"  Breakeven:         52.4%")
    print(f"  Profit ($110/bet): ${results['ats_total_profit']:+,}")
    print(f"  Profit/bet:        ${results['ats_profit_per_bet']:+.2f}")
    print(f"  ROI:               {results['ats_roi']:+.1f}%")

    # Selective betting
    for threshold in [1.0, 2.0, 3.0, 5.0]:
        key = f"ats_{threshold:.0f}pt_bets"
        if key in results:
            t = threshold
            print(
                f"\n  Selective (>={t:.0f}pt disagree): "
                f"{results[f'ats_{t:.0f}pt_bets']} bets, "
                f"{results[f'ats_{t:.0f}pt_pct']:.1%} win, "
                f"${results[f'ats_{t:.0f}pt_profit']:+,}, "
                f"ROI {results[f'ats_{t:.0f}pt_roi']:+.1f}%"
            )

    if "win_auc" in results:
        print(f"\n  Win AUC:           {results['win_auc']:.3f}")
        print(f"  Win Accuracy:      {results['win_accuracy']:.1%}")
        print(f"  Brier Score:       {results['brier_score']:.4f}")
    print(f"{'=' * 70}")


# =============================================================================
# Direct evaluation (for LLM mode where we only have test predictions)
# =============================================================================


def _evaluate_direct_only(
    features,
    test_mask,
    valid,
    feature_cols,
    args,
    df,
    y_test_spread,
    y_test_win,
    vegas_test,
):
    """Evaluate model+betting directly when no training data has model predictions.

    Used for LLM mode where predictions only exist for the test set.
    Runs: standalone model eval, Vegas baseline, and blend sweep.
    """
    print("\n" + "#" * 70)
    print(
        f"#  PHASE 6 EXP 1: DIRECT EVALUATION ({args.model_source.upper()} predictions)"
    )
    print(f"#  No meta-model training (model predictions only on test set)")
    print("#" * 70)

    # Vegas baseline on test
    vegas_test_results = compute_ats_metrics(
        pred_spread=vegas_test,
        actual_spread=y_test_spread,
        vegas_spread=vegas_test,
        label="Vegas Baseline (TEST)",
    )
    print_ats_results(vegas_test_results)

    # Standalone model
    if "model_spread" in features.columns:
        model_test_spread_raw = (
            features.loc[test_mask & valid, "model_spread"].values * 15.0
        )
        model_test_winprob = features.loc[test_mask & valid, "model_win_prob"].values

        standalone_test = compute_ats_metrics(
            pred_spread=model_test_spread_raw,
            actual_spread=y_test_spread,
            vegas_spread=vegas_test,
            pred_win_prob=model_test_winprob,
            actual_home_win=y_test_win,
            label=f"Standalone {args.model_source.upper()} Model (TEST)",
        )
        print_ats_results(standalone_test)

        # Blend sweep
        print("\n" + "#" * 70)
        print("#  BLEND SWEEP: Model weight from 0% to 100%")
        print("#" * 70)
        vegas_test_margin = -vegas_test
        model_test_margin = -model_test_spread_raw

        best_w, best_mae = 0.0, float("inf")
        print(
            f"\n  {'Weight':>7s} | {'MAE':>6s} | {'ATS%':>5s} | {'Record':>12s} | {'Profit':>8s}"
        )
        print(f"  {'-' * 55}")
        for w in np.arange(0.0, 1.05, 0.05):
            blended = w * model_test_margin + (1 - w) * vegas_test_margin
            r = compute_ats_metrics(-blended, y_test_spread, vegas_test)
            if r["spread_mae"] < best_mae:
                best_mae = r["spread_mae"]
                best_w = w
            record = f"{r['ats_wins']}-{r['ats_losses']}-{r['ats_pushes']}"
            print(
                f"  {w:6.0%}  | {r['spread_mae']:5.2f} | {r['ats_win_pct']:5.1%} | "
                f"{record:>12s} | ${r['ats_total_profit']:>+7,}"
            )
        print(
            f"\n  Best blend: model={best_w:.0%}, vegas={1-best_w:.0%} -> MAE {best_mae:.2f}"
        )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"results_{args.model_source}.json"

    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    save_data = {
        "model_source": args.model_source,
        "mode": "direct_evaluation",
        "vegas_test": vegas_test_results,
    }
    if "model_spread" in features.columns:
        save_data["standalone_test"] = standalone_test
        save_data["best_blend_weight"] = float(best_w)

    with open(results_path, "w") as f:
        json.dump(convert_numpy(save_data), f, indent=2)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Meta-model training
# =============================================================================


def train_and_evaluate(args):
    """Stage 2: Load data, build features, train meta-models, evaluate."""

    # ── Load betting data ──────────────────────────────────────────────────
    logger.info("Loading betting data from database...")
    df = load_betting_data()

    # ── Load model predictions ─────────────────────────────────────────────
    model_preds = None
    if args.model_source == "transformer":
        if PREDICTIONS_FILE.exists():
            logger.info(f"Loading transformer predictions from {PREDICTIONS_FILE}")
            data = np.load(PREDICTIONS_FILE, allow_pickle=True)
            model_preds = {k: data[k] for k in data.files}
            split_sizes = model_preds.pop("split_sizes")
            logger.info(
                f"  Loaded {len(model_preds['game_ids'])} predictions "
                f"(train={split_sizes[0]}, val={split_sizes[1]}, test={split_sizes[2]})"
            )
        else:
            logger.warning(
                f"Transformer predictions not found at {PREDICTIONS_FILE}. "
                f"Run with --extract first, or use --model-source llm/none."
            )
            return
    elif args.model_source == "llm":
        model_preds = load_llm_predictions()
        if model_preds is None:
            return
    elif args.model_source == "none":
        logger.info("No model predictions — Vegas-only baseline.")
    else:
        raise ValueError(f"Unknown model source: {args.model_source}")

    # ── Build features ─────────────────────────────────────────────────────
    logger.info("Building feature matrix...")
    features = build_feature_matrix(df, model_preds, args.model_source)

    # Add targets to the feature df for convenience
    features["actual_spread"] = df["actual_spread"].values
    features["actual_total"] = df["actual_total"].values
    features["home_win"] = df["home_win"].values
    features["game_id"] = df["game_id"].values
    features["season"] = df["season"].values
    features["vegas_spread_raw"] = df["vegas_spread"].values
    features["vegas_total_raw"] = df["vegas_total"].values

    # ── Split data ─────────────────────────────────────────────────────────
    train_mask = df["season"].isin(TRAIN_SEASONS).values
    val_mask = df["season"].isin(VAL_SEASONS).values
    test_mask = df["season"].isin(TEST_SEASONS).values

    logger.info(
        f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}"
    )

    # ── Select feature columns ─────────────────────────────────────────────
    # Separate feature groups for ablation
    betting_cols = [
        "vegas_spread_norm",
        "vegas_total_norm",
        "vegas_ml_home_prob_novig",
        "vegas_ml_away_prob_novig",
        "vegas_spread_abs",
        "big_favorite",
    ]
    model_cols = [
        "model_spread",
        "model_total",
        "model_win_prob",
        "model_spread_std",
        "spread_disagreement",
        "total_disagreement",
        "prob_disagreement",
        "model_confident_disagree",
    ]

    if args.model_source == "none":
        feature_cols = betting_cols
    else:
        # Only include model columns that exist (LLM may lack some)
        available_model_cols = [c for c in model_cols if c in features.columns]
        feature_cols = betting_cols + available_model_cols

    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── Filter to games with betting data ──────────────────────────────────
    has_spread = features["vegas_spread_raw"].notna()
    logger.info(f"Games with spread data: {has_spread.sum()} / {len(features)}")

    # Also require model predictions to be present (non-NaN model_spread)
    if args.model_source != "none" and "model_spread" in features.columns:
        has_model = features["model_spread"].notna()
        logger.info(
            f"Games with model predictions: {has_model.sum()} / {len(features)}"
        )
        valid = has_spread & has_model
    else:
        valid = has_spread

    # Also need at least spread for the features to work
    for col in feature_cols:
        if col in features.columns:
            col_valid = features[col].notna()
            n_missing = (~col_valid & valid).sum()
            if n_missing > 0:
                logger.info(f"  {col}: {n_missing} missing values in valid set")

    # For ML features that may be NaN (moneyline), fill with 0.5
    for col in [
        "vegas_ml_home_prob",
        "vegas_ml_away_prob",
        "vegas_ml_home_prob_novig",
        "vegas_ml_away_prob_novig",
    ]:
        if col in features.columns:
            features[col] = features[col].fillna(0.5)

    # Fill remaining NaN model features with 0
    for col in feature_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)

    # ── Prepare train / val / test arrays ──────────────────────────────────
    X_train = features.loc[train_mask & valid, feature_cols].values
    y_train_spread = features.loc[train_mask & valid, "actual_spread"].values
    y_train_win = features.loc[train_mask & valid, "home_win"].values
    vegas_train = features.loc[train_mask & valid, "vegas_spread_raw"].values

    X_val = features.loc[val_mask & valid, feature_cols].values
    y_val_spread = features.loc[val_mask & valid, "actual_spread"].values
    y_val_win = features.loc[val_mask & valid, "home_win"].values
    vegas_val = features.loc[val_mask & valid, "vegas_spread_raw"].values

    X_test = features.loc[test_mask & valid, feature_cols].values
    y_test_spread = features.loc[test_mask & valid, "actual_spread"].values
    y_test_win = features.loc[test_mask & valid, "home_win"].values
    vegas_test = features.loc[test_mask & valid, "vegas_spread_raw"].values

    logger.info(
        f"After filtering — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    # ── Handle case where model predictions only exist for test set (e.g. LLM) ──
    if len(X_train) == 0:
        logger.warning(
            "No training data with model predictions. "
            "Skipping meta-model training; running direct evaluation only."
        )
        _evaluate_direct_only(
            features,
            test_mask,
            valid,
            feature_cols,
            args,
            df,
            y_test_spread,
            y_test_win,
            vegas_test,
        )
        return

    # ── Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # =================================================================
    # Baseline: Vegas alone
    # =================================================================
    print("\n" + "#" * 70)
    print("#  PHASE 6 EXPERIMENT 1: BETTING FEATURES META-MODEL")
    print(f"#  Model source: {args.model_source}")
    print(f"#  Features: {len(feature_cols)}")
    print("#" * 70)

    # Vegas baseline
    vegas_val_results = compute_ats_metrics(
        pred_spread=vegas_val,
        actual_spread=y_val_spread,
        vegas_spread=vegas_val,
        label="Vegas Baseline (VAL)",
    )
    print_ats_results(vegas_val_results)

    vegas_test_results = compute_ats_metrics(
        pred_spread=vegas_test,
        actual_spread=y_test_spread,
        vegas_spread=vegas_test,
        label="Vegas Baseline (TEST)",
    )
    print_ats_results(vegas_test_results)

    # =================================================================
    # If we have model predictions, show standalone model performance
    # =================================================================
    if args.model_source != "none" and "model_spread" in features.columns:
        # Model spread is already normalized; recover raw
        model_val_spread_raw = (
            features.loc[val_mask & valid, "model_spread"].values * 15.0
        )
        model_test_spread_raw = (
            features.loc[test_mask & valid, "model_spread"].values * 15.0
        )

        model_val_winprob = features.loc[val_mask & valid, "model_win_prob"].values
        model_test_winprob = features.loc[test_mask & valid, "model_win_prob"].values

        standalone_val = compute_ats_metrics(
            pred_spread=model_val_spread_raw,
            actual_spread=y_val_spread,
            vegas_spread=vegas_val,
            pred_win_prob=model_val_winprob,
            actual_home_win=y_val_win,
            label=f"Standalone {args.model_source.upper()} Model (VAL)",
        )
        print_ats_results(standalone_val)

        standalone_test = compute_ats_metrics(
            pred_spread=model_test_spread_raw,
            actual_spread=y_test_spread,
            vegas_spread=vegas_test,
            pred_win_prob=model_test_winprob,
            actual_home_win=y_test_win,
            label=f"Standalone {args.model_source.upper()} Model (TEST)",
        )
        print_ats_results(standalone_test)

    # =================================================================
    # Train meta-models
    # =================================================================
    all_results = []

    # ── 1. Linear Regression ───────────────────────────────────────────
    logger.info("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_spread)

    lr_val_pred = lr.predict(X_val_scaled)
    lr_test_pred = lr.predict(X_test_scaled)
    # Convert predicted margin to spread convention (negate)
    lr_val_spread = -lr_val_pred
    lr_test_spread = -lr_test_pred

    lr_val_results = compute_ats_metrics(
        lr_val_spread,
        y_val_spread,
        vegas_val,
        label="Linear Regression (VAL)",
    )
    lr_test_results = compute_ats_metrics(
        lr_test_spread,
        y_test_spread,
        vegas_test,
        label="Linear Regression (TEST)",
    )
    print_ats_results(lr_val_results)
    print_ats_results(lr_test_results)
    all_results.extend([lr_val_results, lr_test_results])

    # Print coefficients
    print("\n  Linear Regression Coefficients:")
    for name, coef in zip(feature_cols, lr.coef_):
        print(f"    {name:<35s}: {coef:+.4f}")
    print(f"    {'intercept':<35s}: {lr.intercept_:+.4f}")

    # ── 2. Ridge Regression ────────────────────────────────────────────
    logger.info("Training Ridge Regression (alpha search)...")
    best_ridge = None
    best_ridge_mae = float("inf")
    best_alpha = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train_spread)
        val_pred = ridge.predict(X_val_scaled)
        mae = mean_absolute_error(y_val_spread, val_pred)
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_ridge = ridge
            best_alpha = alpha

    logger.info(f"Best Ridge alpha: {best_alpha} (val MAE: {best_ridge_mae:.2f})")
    ridge_val_pred = best_ridge.predict(X_val_scaled)
    ridge_test_pred = best_ridge.predict(X_test_scaled)

    ridge_val_results = compute_ats_metrics(
        -ridge_val_pred,
        y_val_spread,
        vegas_val,
        label=f"Ridge (alpha={best_alpha}) (VAL)",
    )
    ridge_test_results = compute_ats_metrics(
        -ridge_test_pred,
        y_test_spread,
        vegas_test,
        label=f"Ridge (alpha={best_alpha}) (TEST)",
    )
    print_ats_results(ridge_val_results)
    print_ats_results(ridge_test_results)
    all_results.extend([ridge_val_results, ridge_test_results])

    # ── 3. MLP Regressor ──────────────────────────────────────────────
    logger.info("Training MLP Regressor...")
    best_mlp = None
    best_mlp_mae = float("inf")
    best_mlp_config = None

    mlp_configs = [
        (32,),
        (64, 32),
        (128, 64, 32),
        (64, 32, 16),
    ]

    for hidden in mlp_configs:
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            learning_rate_init=0.001,
            alpha=0.001,  # L2 regularization
        )
        mlp.fit(X_train_scaled, y_train_spread)
        val_pred = mlp.predict(X_val_scaled)
        mae = mean_absolute_error(y_val_spread, val_pred)
        if mae < best_mlp_mae:
            best_mlp_mae = mae
            best_mlp = mlp
            best_mlp_config = hidden

    logger.info(f"Best MLP config: {best_mlp_config} (val MAE: {best_mlp_mae:.2f})")
    mlp_val_pred = best_mlp.predict(X_val_scaled)
    mlp_test_pred = best_mlp.predict(X_test_scaled)

    mlp_val_results = compute_ats_metrics(
        -mlp_val_pred,
        y_val_spread,
        vegas_val,
        label=f"MLP {best_mlp_config} (VAL)",
    )
    mlp_test_results = compute_ats_metrics(
        -mlp_test_pred,
        y_test_spread,
        vegas_test,
        label=f"MLP {best_mlp_config} (TEST)",
    )
    print_ats_results(mlp_val_results)
    print_ats_results(mlp_test_results)
    all_results.extend([mlp_val_results, mlp_test_results])

    # ── 4. XGBoost ─────────────────────────────────────────────────────
    if HAS_XGB:
        logger.info("Training XGBoost spread model...")
        best_xgb = None
        best_xgb_mae = float("inf")
        best_xgb_params = None

        xgb_param_grid = [
            {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05},
            {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.03},
            {"max_depth": 5, "n_estimators": 500, "learning_rate": 0.01},
            {"max_depth": 3, "n_estimators": 500, "learning_rate": 0.01},
            {"max_depth": 6, "n_estimators": 200, "learning_rate": 0.05},
        ]

        for params in xgb_param_grid:
            model_xgb = xgb.XGBRegressor(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                early_stopping_rounds=30,
                random_state=42,
                verbosity=0,
            )
            model_xgb.fit(
                X_train,
                y_train_spread,
                eval_set=[(X_val, y_val_spread)],
                verbose=False,
            )
            val_pred = model_xgb.predict(X_val)
            mae = mean_absolute_error(y_val_spread, val_pred)
            if mae < best_xgb_mae:
                best_xgb_mae = mae
                best_xgb = model_xgb
                best_xgb_params = params

        logger.info(
            f"Best XGBoost params: {best_xgb_params} (val MAE: {best_xgb_mae:.2f})"
        )
        xgb_val_pred = best_xgb.predict(X_val)
        xgb_test_pred = best_xgb.predict(X_test)

        xgb_val_results = compute_ats_metrics(
            -xgb_val_pred,
            y_val_spread,
            vegas_val,
            label=f"XGBoost (VAL)",
        )
        xgb_test_results = compute_ats_metrics(
            -xgb_test_pred,
            y_test_spread,
            vegas_test,
            label=f"XGBoost (TEST)",
        )
        print_ats_results(xgb_val_results)
        print_ats_results(xgb_test_results)
        all_results.extend([xgb_val_results, xgb_test_results])

        # Feature importance
        print("\n  XGBoost Feature Importance (gain):")
        importance = best_xgb.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        for idx in sorted_idx:
            print(f"    {feature_cols[idx]:<35s}: {importance[idx]:.4f}")

    # ── 5. Win Classification (Logistic Regression + XGBoost) ──────────
    logger.info("Training win probability models...")

    # Logistic Regression for win prob
    log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    log_reg.fit(X_train_scaled, y_train_win)
    lr_val_winprob = log_reg.predict_proba(X_val_scaled)[:, 1]
    lr_test_winprob = log_reg.predict_proba(X_test_scaled)[:, 1]

    lr_win_val = compute_ats_metrics(
        -lr_val_pred,
        y_val_spread,
        vegas_val,
        pred_win_prob=lr_val_winprob,
        actual_home_win=y_val_win,
        label="Logistic Regression Win Prob (VAL)",
    )
    lr_win_test = compute_ats_metrics(
        -lr_test_pred,
        y_test_spread,
        vegas_test,
        pred_win_prob=lr_test_winprob,
        actual_home_win=y_test_win,
        label="Logistic Regression Win Prob (TEST)",
    )
    print_ats_results(lr_win_val)
    print_ats_results(lr_win_test)

    if HAS_XGB:
        # XGBoost classifier for win prob
        logger.info("Training XGBoost win classifier...")
        xgb_clf = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        )
        xgb_clf.fit(
            X_train,
            y_train_win,
            eval_set=[(X_val, y_val_win)],
            verbose=False,
        )
        xgb_val_winprob = xgb_clf.predict_proba(X_val)[:, 1]
        xgb_test_winprob = xgb_clf.predict_proba(X_test)[:, 1]

        xgb_win_val = compute_ats_metrics(
            -xgb_val_pred,
            y_val_spread,
            vegas_val,
            pred_win_prob=xgb_val_winprob,
            actual_home_win=y_val_win,
            label="XGBoost Win Classifier (VAL)",
        )
        xgb_win_test = compute_ats_metrics(
            -xgb_test_pred,
            y_test_spread,
            vegas_test,
            pred_win_prob=xgb_test_winprob,
            actual_home_win=y_test_win,
            label="XGBoost Win Classifier (TEST)",
        )
        print_ats_results(xgb_win_val)
        print_ats_results(xgb_win_test)

    # ── 6. Blended model: weighted average of Vegas + model ────────────
    if args.model_source != "none" and "model_spread" in features.columns:
        print("\n" + "#" * 70)
        print("#  BLENDED MODELS: Weighted average of Vegas + Model spread")
        print("#" * 70)

        model_val_spread_raw = (
            features.loc[val_mask & valid, "model_spread"].values * 15.0
        )
        model_test_spread_raw = (
            features.loc[test_mask & valid, "model_spread"].values * 15.0
        )

        # Vegas margin = -vegas_spread, Model margin = -model_spread * 15
        vegas_val_margin = -vegas_val
        model_val_margin = -model_val_spread_raw

        vegas_test_margin = -vegas_test
        model_test_margin = -model_test_spread_raw

        best_blend_weight = None
        best_blend_mae = float("inf")

        for w in np.arange(0.0, 1.05, 0.05):
            blended_margin = w * model_val_margin + (1 - w) * vegas_val_margin
            mae = mean_absolute_error(y_val_spread, blended_margin)
            if mae < best_blend_mae:
                best_blend_mae = mae
                best_blend_weight = w

        logger.info(
            f"Best blend weight (model): {best_blend_weight:.2f} (val MAE: {best_blend_mae:.2f})"
        )

        # Evaluate best blend on test
        blend_test_margin = (
            best_blend_weight * model_test_margin
            + (1 - best_blend_weight) * vegas_test_margin
        )
        blend_test_spread = -blend_test_margin

        blend_val_margin = (
            best_blend_weight * model_val_margin
            + (1 - best_blend_weight) * vegas_val_margin
        )
        blend_val_spread = -blend_val_margin

        blend_val_results = compute_ats_metrics(
            blend_val_spread,
            y_val_spread,
            vegas_val,
            label=f"Blended (model={best_blend_weight:.0%}, vegas={1-best_blend_weight:.0%}) (VAL)",
        )
        blend_test_results = compute_ats_metrics(
            blend_test_spread,
            y_test_spread,
            vegas_test,
            label=f"Blended (model={best_blend_weight:.0%}, vegas={1-best_blend_weight:.0%}) (TEST)",
        )
        print_ats_results(blend_val_results)
        print_ats_results(blend_test_results)

        # Also try a few fixed blends for comparison
        for w in [0.1, 0.2, 0.3]:
            test_margin = w * model_test_margin + (1 - w) * vegas_test_margin
            blend_results = compute_ats_metrics(
                -test_margin,
                y_test_spread,
                vegas_test,
                label=f"Blended (model={w:.0%}, vegas={1-w:.0%}) (TEST)",
            )
            print(
                f"  model={w:.0%}: Spread MAE={blend_results['spread_mae']:.2f}, "
                f"ATS={blend_results['ats_win_pct']:.1%} "
                f"({blend_results['ats_wins']}-{blend_results['ats_losses']})"
            )

    # =================================================================
    # Summary comparison table
    # =================================================================
    print("\n" + "=" * 90)
    print("  SUMMARY COMPARISON")
    print("=" * 90)
    header = (
        f"{'Model':<45s} | {'MAE':>5s} | {'ATS%':>5s} | "
        f"{'Record':>10s} | {'Profit':>8s} | {'ROI':>6s}"
    )
    print(header)
    print("-" * 90)

    # Collect all test results for summary
    summary_results = [
        vegas_test_results,
    ]
    if args.model_source != "none" and "model_spread" in features.columns:
        summary_results.append(standalone_test)
    summary_results.extend([lr_test_results, ridge_test_results, mlp_test_results])
    if HAS_XGB:
        summary_results.append(xgb_test_results)
    if args.model_source != "none" and "model_spread" in features.columns:
        summary_results.append(blend_test_results)

    for r in summary_results:
        label = r["label"][:44]
        record = f"{r['ats_wins']}-{r['ats_losses']}-{r['ats_pushes']}"
        print(
            f"{label:<45s} | {r['spread_mae']:5.2f} | {r['ats_win_pct']:5.1%} | "
            f"{record:>10s} | ${r['ats_total_profit']:>+7,} | {r['ats_roi']:>+5.1f}%"
        )
    print("=" * 90)

    # =================================================================
    # Save results
    # =================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"results_{args.model_source}.json"

    save_data = {
        "model_source": args.model_source,
        "feature_cols": feature_cols,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "vegas_test": vegas_test_results,
        "linear_regression_test": lr_test_results,
        "ridge_test": ridge_test_results,
        "mlp_test": mlp_test_results,
    }
    if HAS_XGB:
        save_data["xgboost_test"] = xgb_test_results
        # Save feature importance
        save_data["xgb_feature_importance"] = {
            feature_cols[i]: float(importance[i]) for i in range(len(feature_cols))
        }
    if args.model_source != "none" and "model_spread" in features.columns:
        save_data["standalone_model_test"] = standalone_test
        save_data["best_blend_weight"] = float(best_blend_weight)
        save_data["blend_test"] = blend_test_results

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_numpy(save_data), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ── Linear Regression coefficients insight ─────────────────────────
    print("\n" + "=" * 70)
    print("  KEY INSIGHT: Linear Regression Coefficients")
    print("  (What the meta-model learned about each feature)")
    print("=" * 70)
    coef_importance = sorted(
        zip(feature_cols, lr.coef_), key=lambda x: abs(x[1]), reverse=True
    )
    for name, coef in coef_importance:
        bar = (
            "+" * int(min(abs(coef) * 10, 40))
            if coef > 0
            else "-" * int(min(abs(coef) * 10, 40))
        )
        print(f"  {name:<35s}: {coef:+8.4f}  {bar}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Exp 1: Betting features meta-model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Stage 1: Extract Phase 3 model predictions",
    )
    parser.add_argument(
        "--train", action="store_true", help="Stage 2: Train meta-models and evaluate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer/phase3_exp4_interaction.yaml",
        help="Phase 3 config yaml (for --extract)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phase3/phase3_exp4_interaction/best.pt",
        help="Phase 3 checkpoint path (for --extract)",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="transformer",
        choices=["transformer", "llm", "none"],
        help="Source of model predictions: transformer (Phase 3 Exp 4), llm (GPT-5.4-mini), none (Vegas only)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device for model inference (cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.extract and not args.train:
        parser.error("Must specify at least one of --extract or --train")

    if args.extract:
        logger.info("=" * 60)
        logger.info("Stage 1: Extracting model predictions")
        logger.info("=" * 60)
        extract_model_predictions(args.config, args.checkpoint, args.device)

    if args.train:
        logger.info("=" * 60)
        logger.info("Stage 2: Training meta-models")
        logger.info("=" * 60)
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
