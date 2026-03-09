#!/usr/bin/env python
"""
Train hybrid XGBoost model on transformer embeddings + engineered features (Phase 3 Exp 8).

Concatenates 1536-d transformer embeddings with ~47 rolling features,
tunes XGBoost with Optuna, and evaluates on the same test set as the transformer.

Usage:
    python scripts/train_hybrid.py \
        --embeddings data/hybrid/embeddings.npz \
        --features data/hybrid/features.npz \
        --n-trials 100

    # Run all 3 ablation modes and print comparison table:
    python scripts/train_hybrid.py --run-ablations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
)
from xgboost import XGBRegressor, XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(embeddings_path: str, features_path: str):
    """Load and align embeddings and features by game_id."""
    logger = logging.getLogger(__name__)

    emb = np.load(embeddings_path, allow_pickle=True)
    feat = np.load(features_path, allow_pickle=True)

    emb_ids = emb["game_ids"]
    feat_ids = feat["game_ids"]
    feature_names = list(feat["feature_names"])

    # Build lookup from feature game_ids
    feat_lookup = {gid: i for i, gid in enumerate(feat_ids)}

    # Align: for each embedding game_id, find matching feature row
    aligned_emb_indices = []
    aligned_feat_indices = []

    for i, gid in enumerate(emb_ids):
        gid_str = str(gid)
        if gid_str in feat_lookup:
            aligned_emb_indices.append(i)
            aligned_feat_indices.append(feat_lookup[gid_str])

    logger.info(
        f"Aligned {len(aligned_emb_indices)} / {len(emb_ids)} embedding games "
        f"with engineered features"
    )

    # Build feature matrix: [home_repr(512), away_repr(512), matchup_repr(512), features(~47)]
    home_repr = emb["home_repr"][aligned_emb_indices]
    away_repr = emb["away_repr"][aligned_emb_indices]
    matchup_repr = emb["matchup_repr"][aligned_emb_indices]
    eng_features = feat["features"][aligned_feat_indices]

    X = np.hstack([home_repr, away_repr, matchup_repr, eng_features])

    # Targets
    home_scores = emb["target_home_score"][aligned_emb_indices]
    away_scores = emb["target_away_score"][aligned_emb_indices]
    spreads = home_scores - away_scores
    wins = (spreads > 0).astype(int)

    # Split indices based on embedding split_sizes
    split_sizes = emb["split_sizes"]
    n_train, n_val, n_test = split_sizes

    # Map aligned indices back to original split membership
    train_mask = np.array(aligned_emb_indices) < n_train
    val_mask = (np.array(aligned_emb_indices) >= n_train) & (
        np.array(aligned_emb_indices) < n_train + n_val
    )
    test_mask = np.array(aligned_emb_indices) >= n_train + n_val

    logger.info(
        f"Split: {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test"
    )
    logger.info(f"Feature dims: {X.shape[1]} (1536 embedding + {eng_features.shape[1]} engineered)")

    all_feature_names = (
        [f"home_emb_{i}" for i in range(512)]
        + [f"away_emb_{i}" for i in range(512)]
        + [f"matchup_emb_{i}" for i in range(512)]
        + feature_names
    )

    return {
        "X": X,
        "spreads": spreads,
        "wins": wins,
        "home_scores": home_scores,
        "away_scores": away_scores,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "feature_names": all_feature_names,
        "game_ids": np.array(emb_ids)[aligned_emb_indices],
    }


def optimize_spread_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune XGBoost regressor for spread prediction with Optuna."""
    logger = logging.getLogger(__name__)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best spread MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def optimize_total_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune XGBoost regressor for total score prediction with Optuna."""
    logger = logging.getLogger(__name__)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best total MAE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def optimize_win_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
) -> dict:
    """Tune XGBoost classifier for win prediction with Optuna."""
    logger = logging.getLogger(__name__)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best win AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return ece


def evaluate(
    spread_model: XGBRegressor,
    win_model: XGBClassifier,
    total_model: XGBRegressor,
    X: np.ndarray,
    spreads: np.ndarray,
    wins: np.ndarray,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    label: str = "Test",
) -> dict:
    """Evaluate all three models on a dataset."""
    logger = logging.getLogger(__name__)

    # Spread predictions
    pred_spread = spread_model.predict(X)
    spread_mae = mean_absolute_error(spreads, pred_spread)
    spread_rmse = np.sqrt(mean_squared_error(spreads, pred_spread))

    # Total predictions and derived individual scores
    pred_total = total_model.predict(X)
    actual_total = home_scores + away_scores
    total_mae = mean_absolute_error(actual_total, pred_total)
    pred_home = (pred_total + pred_spread) / 2
    pred_away = (pred_total - pred_spread) / 2
    home_mae = mean_absolute_error(home_scores, pred_home)
    away_mae = mean_absolute_error(away_scores, pred_away)

    # Win predictions
    pred_win_prob = win_model.predict_proba(X)[:, 1]
    pred_win = (pred_win_prob >= 0.5).astype(int)
    win_acc = accuracy_score(wins, pred_win)
    win_auc = roc_auc_score(wins, pred_win_prob)
    brier = brier_score_loss(wins, pred_win_prob)
    ece = compute_ece(pred_win_prob, wins)

    metrics = {
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "total_mae": total_mae,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "win_accuracy": win_acc,
        "win_auc": win_auc,
        "brier_score": brier,
        "ece": ece,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{label} Results:")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Spread MAE:   {spread_mae:.2f}")
    logger.info(f"  Spread RMSE:  {spread_rmse:.2f}")
    logger.info(f"  Total MAE:    {total_mae:.2f}")
    logger.info(f"  Home MAE:     {home_mae:.2f}")
    logger.info(f"  Away MAE:     {away_mae:.2f}")
    logger.info(f"  Win Accuracy: {win_acc:.3f}")
    logger.info(f"  Win AUC:      {win_auc:.3f}")
    logger.info(f"  Brier Score:  {brier:.4f}")
    logger.info(f"  ECE:          {ece:.4f}")
    logger.info(f"  90% Coverage: N/A (point estimate only)")
    logger.info(f"{'=' * 60}")

    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_spread_train: np.ndarray,
    y_spread_val: np.ndarray,
    y_spread_test: np.ndarray,
    y_total_train: np.ndarray,
    y_total_val: np.ndarray,
    y_total_test: np.ndarray,
    y_win_train: np.ndarray,
    y_win_val: np.ndarray,
    y_win_test: np.ndarray,
    home_scores_val: np.ndarray,
    away_scores_val: np.ndarray,
    home_scores_test: np.ndarray,
    away_scores_test: np.ndarray,
    n_trials: int,
    mode_label: str = "Combined",
) -> dict:
    """Run full Optuna optimization, training, and evaluation for one feature set.

    Returns dict with val_metrics, test_metrics, and trained models.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'#' * 60}")
    logger.info(f"  MODE: {mode_label}")
    logger.info(f"  Feature dim: {X_train.shape[1]}")
    logger.info(f"{'#' * 60}")

    # Optimize spread model
    logger.info("\n--- Optimizing spread model ---")
    spread_params = optimize_spread_model(
        X_train, y_spread_train, X_val, y_spread_val, n_trials=n_trials
    )

    # Optimize total model
    logger.info("\n--- Optimizing total model ---")
    total_params = optimize_total_model(
        X_train, y_total_train, X_val, y_total_val, n_trials=n_trials
    )

    # Optimize win model
    logger.info("\n--- Optimizing win model ---")
    win_params = optimize_win_model(
        X_train, y_win_train, X_val, y_win_val, n_trials=n_trials
    )

    # Train final models with best params + early stopping
    logger.info("\n--- Training final models ---")
    spread_model = XGBRegressor(
        **spread_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )
    spread_model.fit(
        X_train, y_spread_train,
        eval_set=[(X_val, y_spread_val)],
        verbose=False,
    )

    total_model = XGBRegressor(
        **total_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )
    total_model.fit(
        X_train, y_total_train,
        eval_set=[(X_val, y_total_val)],
        verbose=False,
    )

    win_model = XGBClassifier(
        **win_params,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )
    win_model.fit(
        X_train, y_win_train,
        eval_set=[(X_val, y_win_val)],
        verbose=False,
    )

    # Evaluate
    val_metrics = evaluate(
        spread_model, win_model, total_model,
        X_val, y_spread_val, y_win_val,
        home_scores_val, away_scores_val,
        label=f"{mode_label} — Validation",
    )

    test_metrics = evaluate(
        spread_model, win_model, total_model,
        X_test, y_spread_test, y_win_test,
        home_scores_test, away_scores_test,
        label=f"{mode_label} — Test",
    )

    return {
        "spread_params": spread_params,
        "total_params": total_params,
        "win_params": win_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "spread_model": spread_model,
        "total_model": total_model,
        "win_model": win_model,
    }


def print_comparison_table(ablation_results: dict):
    """Print a comparison table across ablation modes."""
    header = (
        f"| {'Mode':<20s} | {'Spread MAE':>10s} | {'Total MAE':>10s} | "
        f"{'Win AUC':>8s} | {'Win Acc':>8s} |"
    )
    sep = f"|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 10}|{'-' * 10}|"

    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(header)
    print(sep)

    for mode, result in ablation_results.items():
        m = result["test_metrics"]
        print(
            f"| {mode:<20s} | {m['spread_mae']:>10.2f} | {m['total_mae']:>10.2f} | "
            f"{m['win_auc']:>8.3f} | {m['win_accuracy'] * 100:>7.1f}% |"
        )

    # Reference row for transformer baseline
    print(
        f"| {'Transformer (Exp4)':<20s} | {'10.83':>10s} | {'---':>10s} | "
        f"{'0.705':>8s} | {'65.1%':>8s} |"
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train hybrid XGBoost")
    parser.add_argument("--embeddings", type=str, default="data/hybrid/embeddings.npz")
    parser.add_argument("--features", type=str, default="data/hybrid/features.npz")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase3/phase3_exp8_hybrid")
    parser.add_argument("--embeddings-only", action="store_true",
                        help="Use only transformer embeddings (no engineered features)")
    parser.add_argument("--features-only", action="store_true",
                        help="Use only engineered features (no embeddings)")
    parser.add_argument("--run-ablations", action="store_true",
                        help="Run all 3 modes (combined, embeddings-only, features-only) "
                             "and print comparison table")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load data
    data = load_data(args.embeddings, args.features)

    X_full = data["X"]
    spreads = data["spreads"]
    wins = data["wins"]
    home_scores = data["home_scores"]
    away_scores = data["away_scores"]
    totals = home_scores + away_scores

    n_emb = 1536

    # Prepare split targets (shared across modes)
    y_spread_train = spreads[data["train_mask"]]
    y_spread_val = spreads[data["val_mask"]]
    y_spread_test = spreads[data["test_mask"]]

    y_total_train = totals[data["train_mask"]]
    y_total_val = totals[data["val_mask"]]
    y_total_test = totals[data["test_mask"]]

    y_win_train = wins[data["train_mask"]]
    y_win_val = wins[data["val_mask"]]
    y_win_test = wins[data["test_mask"]]

    home_scores_val = home_scores[data["val_mask"]]
    away_scores_val = away_scores[data["val_mask"]]
    home_scores_test = home_scores[data["test_mask"]]
    away_scores_test = away_scores[data["test_mask"]]

    # ----------------------------------------------------------------
    # Ablation mode: run all 3 experiments and compare
    # ----------------------------------------------------------------
    if args.run_ablations:
        ablation_results = {}

        modes = [
            ("Combined", X_full),
            ("Embeddings only", X_full[:, :n_emb]),
            ("Features only", X_full[:, n_emb:]),
        ]

        for mode_name, X in modes:
            X_train = X[data["train_mask"]]
            X_val = X[data["val_mask"]]
            X_test = X[data["test_mask"]]

            result = train_and_evaluate(
                X_train, X_val, X_test,
                y_spread_train, y_spread_val, y_spread_test,
                y_total_train, y_total_val, y_total_test,
                y_win_train, y_win_val, y_win_test,
                home_scores_val, away_scores_val,
                home_scores_test, away_scores_test,
                n_trials=args.n_trials,
                mode_label=mode_name,
            )
            ablation_results[mode_name] = result

        print_comparison_table(ablation_results)

        # Save combined mode results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        combined = ablation_results["Combined"]
        combined["spread_model"].save_model(str(output_dir / "xgb_spread.json"))
        combined["total_model"].save_model(str(output_dir / "xgb_total.json"))
        combined["win_model"].save_model(str(output_dir / "xgb_win.json"))

        results = {
            mode: {
                "spread_params": r["spread_params"],
                "total_params": r["total_params"],
                "win_params": r["win_params"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
            }
            for mode, r in ablation_results.items()
        }
        results["n_trials"] = args.n_trials
        results["train_size"] = int(data["train_mask"].sum())
        results["val_size"] = int(data["val_mask"].sum())
        results["test_size"] = int(data["test_mask"].sum())

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nModels saved to {output_dir}")
        return 0

    # ----------------------------------------------------------------
    # Single mode
    # ----------------------------------------------------------------
    if args.embeddings_only:
        logger.info("Using ONLY transformer embeddings (no engineered features)")
        X = X_full[:, :n_emb]
    elif args.features_only:
        logger.info("Using ONLY engineered features (no embeddings)")
        X = X_full[:, n_emb:]
    else:
        X = X_full

    X_train = X[data["train_mask"]]
    X_val = X[data["val_mask"]]
    X_test = X[data["test_mask"]]

    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Validation data: {X_val.shape}")
    logger.info(f"Test data: {X_test.shape}")

    result = train_and_evaluate(
        X_train, X_val, X_test,
        y_spread_train, y_spread_val, y_spread_test,
        y_total_train, y_total_val, y_total_test,
        y_win_train, y_win_val, y_win_test,
        home_scores_val, away_scores_val,
        home_scores_test, away_scores_test,
        n_trials=args.n_trials,
        mode_label="Hybrid",
    )

    spread_model = result["spread_model"]
    total_model = result["total_model"]
    win_model = result["win_model"]
    test_metrics = result["test_metrics"]

    # Save models and results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spread_model.save_model(str(output_dir / "xgb_spread.json"))
    total_model.save_model(str(output_dir / "xgb_total.json"))
    win_model.save_model(str(output_dir / "xgb_win.json"))

    results = {
        "spread_params": result["spread_params"],
        "total_params": result["total_params"],
        "win_params": result["win_params"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "n_trials": args.n_trials,
        "feature_dim": X.shape[1],
        "train_size": int(data["train_mask"].sum()),
        "val_size": int(data["val_mask"].sum()),
        "test_size": int(data["test_mask"].sum()),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nModels saved to {output_dir}")

    # Feature importance (top 20 for engineered features)
    if not args.embeddings_only:
        importance = spread_model.feature_importances_
        if not args.features_only:
            eng_importance = importance[n_emb:]
            eng_names = data["feature_names"][n_emb:]
        else:
            eng_importance = importance
            eng_names = data["feature_names"]

        top_idx = np.argsort(eng_importance)[::-1][:20]
        logger.info("\nTop 20 engineered feature importances (spread model):")
        for i in top_idx:
            logger.info(f"  {eng_names[i]}: {eng_importance[i]:.4f}")

    # Print final summary
    print("\n" + "=" * 60)
    print("PHASE 3 EXP 8: HYBRID TRANSFORMER + XGBOOST")
    print("=" * 60)
    print(f"Spread MAE:   {test_metrics['spread_mae']:.2f}")
    print(f"Spread RMSE:  {test_metrics['spread_rmse']:.2f}")
    print(f"Total MAE:    {test_metrics['total_mae']:.2f}")
    print(f"Home MAE:     {test_metrics['home_mae']:.2f}")
    print(f"Away MAE:     {test_metrics['away_mae']:.2f}")
    print(f"Win Accuracy: {test_metrics['win_accuracy']:.3f}")
    print(f"Win AUC:      {test_metrics['win_auc']:.3f}")
    print(f"Brier Score:  {test_metrics['brier_score']:.4f}")
    print(f"ECE:          {test_metrics['ece']:.4f}")
    print(f"90% Coverage: N/A (point estimate only)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
