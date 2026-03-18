#!/usr/bin/env python
"""
Phase 6: Extract per-game spread predictions from Phase 3 + Phase 4 models.

Loads trained model checkpoints, runs inference on the test (or val) set, and
saves per-game predictions as JSONL files that the ATS evaluator can consume.

Supports:
  - All Phase 3 transformer models (Exp 1-7)
  - Phase 3 Exp 9 deep ensemble (average of 3 seeds)
  - Phase 4 generative models (Exp 4, Exp 5, Exp 5b) via outcome head
  - Phase 3 Exp 8 hybrid (XGBoost — requires re-running inference)

Output format (JSONL, one line per game):
    {"game_id": "0022400123", "model": "phase3_exp4", "home_team": "LAL",
     "away_team": "BOS", "season": "2024-2025", "actual_home_score": 112,
     "actual_away_score": 108, "prediction": {"home_score": 110.3,
     "away_score": 106.1, "home_win_probability": 0.58}, "success": true}

Usage:
    # Single Phase 3 model
    python scripts/phase6_extract_predictions.py --model phase3_exp4_interaction --split test

    # Phase 3 ensemble
    python scripts/phase6_extract_predictions.py --model phase3_exp9_ensemble --split test

    # Phase 4 generative model
    python scripts/phase6_extract_predictions.py --model gen_exp5_full_context --split test

    # All available models
    python scripts/phase6_extract_predictions.py --all --split test

    # Val set
    python scripts/phase6_extract_predictions.py --all --split val
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import load_cache
from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_PHASE3 = PROJECT_ROOT / "checkpoints" / "phase3"
CHECKPOINTS_GEN = PROJECT_ROOT / "checkpoints" / "generative"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase6"

# ---------------------------------------------------------------------------
# Model registry: name -> config for loading
# ---------------------------------------------------------------------------
# Phase 3 transformer models (all share the same inference pattern)
PHASE3_MODELS = {
    "phase3_exp1_gru": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp1_gru" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp1_gru" / "config.yaml",
    },
    "phase3_exp2_finetune": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp2_finetune" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp2_finetune" / "config.yaml",
    },
    "phase3_exp3a_boxscore": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp3a_boxscore" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp3a_boxscore" / "config.yaml",
    },
    "phase3_exp3b_extended": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp3b_extended" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp3b_extended" / "config.yaml",
    },
    "phase3_exp4_interaction": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp4_interaction" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp4_interaction" / "config.yaml",
    },
    "phase3_exp4b_multiquery": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp4b_multiquery" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp4b_multiquery" / "config.yaml",
    },
    "phase3_exp5_roster_temporal": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp5_roster_temporal" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp5_roster_temporal" / "config.yaml",
    },
    "phase3_exp6_higformer": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp6_higformer" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp6_higformer" / "config.yaml",
    },
    "phase3_exp7_kitchen_sink": {
        "checkpoint": CHECKPOINTS_PHASE3 / "phase3_exp7_kitchen_sink" / "best.pt",
        "config": CHECKPOINTS_PHASE3 / "phase3_exp7_kitchen_sink" / "config.yaml",
    },
}

# Phase 3 Exp 9 ensemble: 3 seed checkpoints averaged
ENSEMBLE_CHECKPOINTS = [
    CHECKPOINTS_PHASE3 / "phase3_exp4_interaction" / "best.pt",  # seed 42
    CHECKPOINTS_PHASE3 / "phase3_exp9_seed137" / "best.pt",
    CHECKPOINTS_PHASE3 / "phase3_exp9_seed256" / "best.pt",
]
ENSEMBLE_CONFIG = CHECKPOINTS_PHASE3 / "phase3_exp4_interaction" / "config.yaml"

# Phase 4 generative models with outcome head
GENERATIVE_MODELS = {
    "gen_exp4_compressed": {
        "checkpoint": CHECKPOINTS_GEN / "gen_exp4_compressed" / "best.pt",
        "config": CHECKPOINTS_GEN / "gen_exp4_compressed" / "config.yaml",
    },
    "gen_exp5_full_context": {
        "checkpoint": CHECKPOINTS_GEN / "gen_exp5_full_context" / "best.pt",
        "config": CHECKPOINTS_GEN / "gen_exp5_full_context" / "config.yaml",
    },
    "gen_exp5b_clock_delta": {
        "checkpoint": CHECKPOINTS_GEN / "gen_exp5b_clock_delta" / "best.pt",
        "config": CHECKPOINTS_GEN / "gen_exp5b_clock_delta" / "config.yaml",
    },
}


# ---------------------------------------------------------------------------
# Phase 3 transformer model loading + inference
# ---------------------------------------------------------------------------
def load_phase3_model(config, checkpoint_path: str, device: str):
    """Create Phase2Model from config and load EMA weights from checkpoint."""
    from src.transformer.phase2.models.phase2_model import Phase2Model

    model = Phase2Model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Prefer EMA weights (used at eval time during training)
    if "ema_state_dict" in checkpoint:
        shadow = checkpoint["ema_state_dict"]["shadow"]
        missing, unexpected = model.load_state_dict(shadow, strict=False)
    else:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )

    if missing:
        logger.debug(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.debug(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.to(device)
    model.eval()
    return model


def build_phase3_dataset(config, cache, split: str) -> Phase2Dataset:
    """Build Phase2Dataset for the given split using the loaded cache."""
    if split == "test":
        seasons = config.data.test_seasons
    elif split == "val":
        seasons = config.data.val_seasons
    else:
        raise ValueError(f"Unknown split: {split}")

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

    dataset = Phase2Dataset(
        seasons=seasons,
        enable_augmentation=False,
        **common_kwargs,
    )
    return dataset


def run_phase3_inference(
    model,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, np.ndarray]:
    """Run inference collecting per-game predictions."""
    home_means, away_means = [], []
    win_probs = []
    target_homes, target_aways = [], []

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

            home_means.append(pred.home_score_mean.float().cpu().numpy())
            away_means.append(pred.away_score_mean.float().cpu().numpy())
            win_probs.append(pred.home_win_prob.float().cpu().numpy())
            target_homes.append(batch["target_home_scores"].numpy())
            target_aways.append(batch["target_away_scores"].numpy())

    return {
        "home_mean": np.concatenate(home_means),
        "away_mean": np.concatenate(away_means),
        "win_prob": np.concatenate(win_probs),
        "target_home": np.concatenate(target_homes),
        "target_away": np.concatenate(target_aways),
    }


def extract_phase3_single(
    model_name: str,
    model_info: dict,
    split: str,
    device: str,
    batch_size: int = 4,
) -> list[dict]:
    """Extract predictions for a single Phase 3 transformer model."""
    config_path = model_info["config"]
    checkpoint_path = model_info["checkpoint"]

    if not config_path.exists() or not checkpoint_path.exists():
        logger.warning(f"Skipping {model_name}: missing checkpoint or config")
        return []

    logger.info(f"Loading config from {config_path}")
    config = Phase2ExperimentConfig.from_yaml(str(config_path))

    # Determine device
    use_amp = config.training.use_amp
    amp_dtype_str = getattr(config.training, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Load cache
    logger.info(f"Loading cache from {config.data.cache_dir}")
    cache = load_cache(config.data.cache_dir)

    # Build dataset
    dataset = build_phase3_dataset(config, cache, split)
    logger.info(f"Dataset: {len(dataset)} games for {split}")

    if len(dataset) == 0:
        logger.warning(f"No games found for {model_name} {split}")
        return []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_phase2,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_phase3_model(config, str(checkpoint_path), device)

    # Run inference
    logger.info("Running inference...")
    preds = run_phase3_inference(
        model, loader, device, use_amp=use_amp, amp_dtype=amp_dtype
    )

    # Build JSONL records by matching predictions to game_ids
    game_features = cache["game_features"]
    game_ids = dataset.game_ids
    records = build_records(game_ids, game_features, preds, model_name)

    # Free GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return records


def extract_phase3_ensemble(
    split: str,
    device: str,
    batch_size: int = 4,
) -> list[dict]:
    """Extract predictions for Phase 3 Exp 9 deep ensemble (3 seeds)."""
    model_name = "phase3_exp9_ensemble"

    # Check all checkpoints exist
    for cp in ENSEMBLE_CHECKPOINTS:
        if not cp.exists():
            logger.warning(f"Skipping ensemble: missing checkpoint {cp}")
            return []
    if not ENSEMBLE_CONFIG.exists():
        logger.warning(f"Skipping ensemble: missing config {ENSEMBLE_CONFIG}")
        return []

    logger.info(f"Ensemble of {len(ENSEMBLE_CHECKPOINTS)} models")
    config = Phase2ExperimentConfig.from_yaml(str(ENSEMBLE_CONFIG))

    use_amp = config.training.use_amp
    amp_dtype_str = getattr(config.training, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Load cache (shared across all seeds)
    logger.info(f"Loading cache from {config.data.cache_dir}")
    cache = load_cache(config.data.cache_dir)

    # Build dataset (shared)
    dataset = build_phase3_dataset(config, cache, split)
    logger.info(f"Dataset: {len(dataset)} games for {split}")

    if len(dataset) == 0:
        return []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_phase2,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    # Run inference for each seed
    all_preds = []
    for i, cp in enumerate(ENSEMBLE_CHECKPOINTS):
        logger.info(f"Loading seed model {i + 1}/{len(ENSEMBLE_CHECKPOINTS)}: {cp}")
        model = load_phase3_model(config, str(cp), device)

        preds = run_phase3_inference(
            model, loader, device, use_amp=use_amp, amp_dtype=amp_dtype
        )
        all_preds.append(preds)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Ensemble: average means, logit-average win probabilities
    home_means = np.stack([p["home_mean"] for p in all_preds])
    away_means = np.stack([p["away_mean"] for p in all_preds])
    win_probs_stack = np.stack([p["win_prob"] for p in all_preds])

    ens_home_mean = home_means.mean(axis=0)
    ens_away_mean = away_means.mean(axis=0)

    # Logit averaging for win probability
    eps = 1e-7
    win_probs_clipped = np.clip(win_probs_stack, eps, 1 - eps)
    logits = np.log(win_probs_clipped / (1 - win_probs_clipped))
    avg_logit = logits.mean(axis=0)
    ens_win_prob = 1.0 / (1.0 + np.exp(-avg_logit))

    ens_preds = {
        "home_mean": ens_home_mean,
        "away_mean": ens_away_mean,
        "win_prob": ens_win_prob,
        "target_home": all_preds[0]["target_home"],
        "target_away": all_preds[0]["target_away"],
    }

    game_features = cache["game_features"]
    game_ids = dataset.game_ids
    records = build_records(game_ids, game_features, ens_preds, model_name)

    return records


# ---------------------------------------------------------------------------
# Phase 4 generative model loading + inference (outcome head only)
# ---------------------------------------------------------------------------
def extract_generative(
    model_name: str,
    model_info: dict,
    split: str,
    device: str,
) -> list[dict]:
    """Extract predictions from a Phase 4 generative model via outcome head.

    Uses the outcome head's (mu, sigma) at the last valid position to produce
    a spread prediction, then converts to home/away scores.
    """
    from src.generative.config import GenerativeExperimentConfig
    from src.generative.dataset import GenerativeDataset, generative_collate
    from src.generative.models.generative_model import GenerativeModel

    config_path = model_info["config"]
    checkpoint_path = model_info["checkpoint"]

    if not config_path.exists() or not checkpoint_path.exists():
        logger.warning(f"Skipping {model_name}: missing checkpoint or config")
        return []

    config = GenerativeExperimentConfig.from_yaml(str(config_path))

    use_full = config.model.use_full_context
    use_simple = config.model.use_simplified_context
    use_compressed = config.model.use_scoring_events_only
    max_se = config.model.max_scoring_events

    ds = GenerativeDataset(
        config.data,
        split=split,
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
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "ema_state_dict" in checkpoint and "shadow" in checkpoint["ema_state_dict"]:
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    has_outcome_head = hasattr(model, "outcome_head") and model.outcome_head is not None
    if not has_outcome_head:
        logger.warning(
            f"Skipping {model_name}: no outcome head available. "
            f"Only Exp 5+ generative models support direct spread prediction."
        )
        del model
        return []

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {model_name} ({n_params:,} params), {len(ds)} games")

    # Get game metadata from the generative dataset's game_features
    gen_game_features = ds.game_features
    # game_id is dropped by generative_collate, so we track by dataset index
    all_game_ids = ds.game_ids

    records = []
    game_idx = 0  # Tracks which game_id we're on (batch_size=1, no shuffle)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                game_idx += 1
                continue

            game_id = all_game_ids[game_idx] if game_idx < len(all_game_ids) else None
            game_idx += 1

            batch_dev = {
                k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward through model to get outcome head predictions
            context_tokens = model.encode_context(batch_dev)
            cond = model.pool_context(context_tokens)

            states = batch_dev["states"]  # (1, T, D)
            T = states.shape[1]

            # Use all states except last for full-game prediction
            input_states = states[:, : T - 1, :]
            if input_states.shape[1] == 0:
                input_states = states[:, :1, :]

            state_embed = model.state_embedder(input_states)
            decoder_out = model.decoder(state_embed, cond)

            outcome_mu, outcome_sigma = model.outcome_head(decoder_out)
            # Use last position for prediction
            pred_margin = outcome_mu[0, -1].item() * 50.0  # denormalize
            pred_sigma = outcome_sigma[0, -1].item() * 50.0

            true_margin = batch_dev["final_margin"].item()
            true_home = None
            true_away = None

            # Win probability from Gaussian CDF: P(margin > 0)
            if pred_sigma > 0:
                z = pred_margin / pred_sigma
                win_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            else:
                win_prob = 1.0 if pred_margin > 0 else 0.0

            # Convert spread to approximate home/away scores
            # Use ~108 as league average score
            pred_home = 108.0 + pred_margin / 2
            pred_away = 108.0 - pred_margin / 2

            # Get metadata from generative cache
            gf = gen_game_features.get(game_id, {}) if game_id else {}
            home_team = gf.get("home_team", "UNK")
            away_team = gf.get("away_team", "UNK")
            season = ds._get_season(game_id) if game_id else ""

            # Derive actual scores from margin if possible
            # (generative cache only stores final_margin, not individual scores)
            actual_home = None
            actual_away = None

            record = {
                "game_id": game_id or f"unknown_{i}",
                "model": model_name,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "actual_home_score": actual_home,
                "actual_away_score": actual_away,
                "actual_margin": float(true_margin),
                "prediction": {
                    "home_score": round(float(pred_home), 2),
                    "away_score": round(float(pred_away), 2),
                    "home_win_probability": round(float(win_prob), 4),
                    "predicted_margin": round(float(pred_margin), 2),
                },
                "success": True,
            }
            records.append(record)

            if (i + 1) % 200 == 0:
                running_mae = np.mean(
                    np.abs(
                        np.array([r["prediction"]["predicted_margin"] for r in records])
                        - np.array([r["actual_margin"] for r in records])
                    )
                )
                logger.info(
                    f"  Game {i + 1}/{len(ds)}: running spread MAE = {running_mae:.2f}"
                )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if records:
        pred_margins = np.array([r["prediction"]["predicted_margin"] for r in records])
        true_margins = np.array([r["actual_margin"] for r in records])
        mae = np.mean(np.abs(pred_margins - true_margins))
        logger.info(f"  {model_name}: {len(records)} games, spread MAE = {mae:.2f}")

    return records


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def build_records(
    game_ids: list[str],
    game_features: dict,
    preds: dict[str, np.ndarray],
    model_name: str,
) -> list[dict]:
    """Build JSONL records from game_ids + prediction arrays.

    Used for Phase 3 transformer models where game_ids align 1:1 with
    prediction arrays from the DataLoader (shuffle=False, deterministic).
    """
    records = []
    n = len(game_ids)

    for i in range(n):
        gid = game_ids[i]
        gf = game_features.get(gid)
        if gf is None:
            continue

        # PerGameFeatures is a dataclass with named attributes
        home_team = gf.home_team
        away_team = gf.away_team
        season = gf.season
        actual_home = float(gf.home_score)
        actual_away = float(gf.away_score)

        pred_home = float(preds["home_mean"][i])
        pred_away = float(preds["away_mean"][i])
        win_prob = float(preds["win_prob"][i])

        record = {
            "game_id": gid,
            "model": model_name,
            "home_team": home_team,
            "away_team": away_team,
            "season": season,
            "actual_home_score": actual_home,
            "actual_away_score": actual_away,
            "prediction": {
                "home_score": round(pred_home, 2),
                "away_score": round(pred_away, 2),
                "home_win_probability": round(win_prob, 4),
            },
            "success": True,
        }
        records.append(record)

    return records


def save_predictions(records: list[dict], output_path: Path):
    """Save prediction records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(records)} predictions to {output_path}")


def compute_quick_metrics(records: list[dict]) -> dict:
    """Compute basic metrics from prediction records."""
    if not records:
        return {}

    pred_home = np.array([r["prediction"]["home_score"] for r in records])
    pred_away = np.array([r["prediction"]["away_score"] for r in records])
    win_probs = np.array([r["prediction"]["home_win_probability"] for r in records])

    # Filter records with actual scores
    has_actual = [r for r in records if r.get("actual_home_score") is not None]
    if not has_actual:
        # For generative models, use actual_margin
        has_margin = [r for r in records if r.get("actual_margin") is not None]
        if has_margin:
            pred_margins = np.array(
                [
                    r["prediction"].get(
                        "predicted_margin",
                        r["prediction"]["home_score"] - r["prediction"]["away_score"],
                    )
                    for r in has_margin
                ]
            )
            true_margins = np.array([r["actual_margin"] for r in has_margin])
            spread_mae = float(np.mean(np.abs(pred_margins - true_margins)))
            true_wins = (true_margins > 0).astype(float)
            win_acc = float(np.mean((win_probs[: len(has_margin)] > 0.5) == true_wins))
            return {
                "spread_mae": spread_mae,
                "n_games": len(has_margin),
                "win_acc": win_acc,
            }
        return {"n_games": len(records)}

    actual_home = np.array([r["actual_home_score"] for r in has_actual])
    actual_away = np.array([r["actual_away_score"] for r in has_actual])

    pred_h = np.array([r["prediction"]["home_score"] for r in has_actual])
    pred_a = np.array([r["prediction"]["away_score"] for r in has_actual])
    wp = np.array([r["prediction"]["home_win_probability"] for r in has_actual])

    spread_mae = float(np.mean(np.abs((pred_h - pred_a) - (actual_home - actual_away))))
    home_mae = float(np.mean(np.abs(pred_h - actual_home)))
    away_mae = float(np.mean(np.abs(pred_a - actual_away)))

    true_wins = (actual_home > actual_away).astype(float)
    win_acc = float(np.mean((wp > 0.5) == true_wins))

    try:
        from sklearn.metrics import roc_auc_score

        win_auc = float(roc_auc_score(true_wins, wp))
    except (ValueError, ImportError):
        win_auc = float("nan")

    return {
        "spread_mae": spread_mae,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "win_acc": win_acc,
        "win_auc": win_auc,
        "n_games": len(has_actual),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def get_all_model_names() -> list[str]:
    """Return all available model names."""
    names = list(PHASE3_MODELS.keys())
    names.append("phase3_exp9_ensemble")
    names.extend(GENERATIVE_MODELS.keys())
    return names


def extract_model(
    model_name: str,
    split: str,
    device: str,
    batch_size: int = 4,
) -> list[dict]:
    """Extract predictions for a single model by name."""
    if model_name == "phase3_exp9_ensemble":
        return extract_phase3_ensemble(split, device, batch_size)
    elif model_name in PHASE3_MODELS:
        return extract_phase3_single(
            model_name, PHASE3_MODELS[model_name], split, device, batch_size
        )
    elif model_name in GENERATIVE_MODELS:
        return extract_generative(
            model_name, GENERATIVE_MODELS[model_name], split, device
        )
    else:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {get_all_model_names()}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-game predictions from Phase 3 + Phase 4 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to evaluate (e.g., phase3_exp4_interaction, phase3_exp9_ensemble, gen_exp5_full_context)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract predictions for ALL available models",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for Phase 3 models (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for JSONL files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have output files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available model names and exit",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.list:
        print("Available models:")
        for name in get_all_model_names():
            print(f"  {name}")
        return 0

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Determine which models to run
    if args.all:
        model_names = get_all_model_names()
    elif args.model:
        model_names = [args.model]
    else:
        parser.error(
            "Either --model or --all must be specified (use --list to see models)"
        )
        return 1

    # Run extraction
    output_dir = args.output_dir
    all_results = {}

    for model_name in model_names:
        output_path = output_dir / f"{args.split}_{model_name}_predictions.jsonl"

        if args.skip_existing and output_path.exists():
            logger.info(f"Skipping {model_name}: {output_path} already exists")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Extracting: {model_name} ({args.split})")
        logger.info(f"{'=' * 60}")

        try:
            records = extract_model(model_name, args.split, device, args.batch_size)

            if records:
                save_predictions(records, output_path)
                metrics = compute_quick_metrics(records)
                all_results[model_name] = metrics

                # Print quick summary
                if "spread_mae" in metrics:
                    logger.info(
                        f"  {model_name}: "
                        f"MAE={metrics['spread_mae']:.2f}, "
                        f"Win%={metrics.get('win_acc', 0):.1%}, "
                        f"AUC={metrics.get('win_auc', float('nan')):.3f}, "
                        f"N={metrics['n_games']}"
                    )
            else:
                logger.warning(f"  No predictions extracted for {model_name}")

        except Exception as e:
            logger.error(f"Failed to extract {model_name}: {e}", exc_info=True)
            continue

    # Print summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print(f"  PREDICTION EXTRACTION SUMMARY ({args.split} set)")
        print(f"{'=' * 80}")
        print(
            f"  {'Model':<35s} {'Spread MAE':>10s} {'Win Acc':>8s} {'Win AUC':>8s} {'Games':>7s}"
        )
        print(f"  {'-' * 72}")

        for name, m in sorted(
            all_results.items(), key=lambda x: x[1].get("spread_mae", 999)
        ):
            mae_str = f"{m['spread_mae']:.2f}" if "spread_mae" in m else "N/A"
            acc_str = f"{m.get('win_acc', 0):.1%}" if "win_acc" in m else "N/A"
            auc_str = (
                f"{m.get('win_auc', float('nan')):.3f}" if "win_auc" in m else "N/A"
            )
            print(
                f"  {name:<35s} {mae_str:>10s} {acc_str:>8s} {auc_str:>8s} {m['n_games']:>7d}"
            )

        print(f"  {'-' * 72}")
        print(f"\n  Output files saved to: {output_dir}/")

    # Print ATS eval hint
    if all_results:
        print(f"\n  To run ATS evaluation:")
        for name in all_results:
            jsonl_path = output_dir / f"{args.split}_{name}_predictions.jsonl"
            print(f"    python scripts/phase6_ats_evaluate.py --results {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
