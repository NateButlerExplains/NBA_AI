#!/usr/bin/env python
"""
Extract transformer embeddings for hybrid XGBoost model (Phase 3 Exp 8).

Loads a trained Phase2Model checkpoint and extracts 512-d team representations
and 512-d matchup representations for every game in train/val/test splits.

Usage:
    python scripts/extract_embeddings.py \
        --config configs/transformer/phase3_exp4_interaction.yaml \
        --checkpoint checkpoints/phase3/phase3_exp4_interaction/best.pt \
        --output data/hybrid/embeddings.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.phase2.config import Phase2ExperimentConfig
from src.transformer.phase2.cache_builder import load_cache
from src.transformer.phase2.dataset import Phase2Dataset, collate_phase2
from src.transformer.phase2.models.phase2_model import Phase2Model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_with_ema(model: Phase2Model, checkpoint_path: str, device: str):
    """Load model weights, preferring EMA shadow weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema_state_dict" in checkpoint:
        # EMA shadow contains the averaged weights
        ema_state = checkpoint["ema_state_dict"]
        shadow = ema_state["shadow"]
        missing, unexpected = model.load_state_dict(shadow, strict=False)
        logging.info(f"Loaded EMA weights from {checkpoint_path}")
    else:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        logging.info(f"Loaded model weights from {checkpoint_path}")

    if missing:
        logging.warning(f"Missing keys (new modules, will use random init): {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys (old modules, ignored): {unexpected}")

    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model: Phase2Model,
    dataloader: DataLoader,
    valid_game_ids: list[str],
    device: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, np.ndarray]:
    """Extract home_repr, away_repr, matchup_repr for all games.

    Args:
        valid_game_ids: Pre-validated game_ids corresponding 1:1 with the
            dataset samples (all guaranteed to produce valid samples).
    """
    all_home = []
    all_away = []
    all_matchup = []
    all_home_scores = []
    all_away_scores = []

    idx = 0
    with torch.inference_mode():
        for batch in dataloader:
            if batch is None:
                continue

            # Move batch to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v

            # Extract intermediate representations with matching AMP context
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device == "cuda"):
                home_repr = model._encode_team(batch_device, "home_")  # (B, 512)
                away_repr = model._encode_team(batch_device, "away_")  # (B, 512)
                matchup_repr = model.fusion(home_repr, away_repr)      # (B, 512)

            # Cast to float32 before saving to numpy
            all_home.append(home_repr.float().cpu().numpy())
            all_away.append(away_repr.float().cpu().numpy())
            all_matchup.append(matchup_repr.float().cpu().numpy())
            all_home_scores.append(batch["target_home_scores"].numpy())
            all_away_scores.append(batch["target_away_scores"].numpy())

            idx += home_repr.shape[0]

    # valid_game_ids is 1:1 with the dataset; idx should match
    assert idx == len(valid_game_ids), (
        f"Embedding count ({idx}) != valid_game_ids ({len(valid_game_ids)}). "
        "Dataset returned unexpected Nones."
    )

    return {
        "game_ids": np.array(valid_game_ids),
        "home_repr": np.concatenate(all_home, axis=0),
        "away_repr": np.concatenate(all_away, axis=0),
        "matchup_repr": np.concatenate(all_matchup, axis=0),
        "target_home_score": np.concatenate(all_home_scores, axis=0),
        "target_away_score": np.concatenate(all_away_scores, axis=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract transformer embeddings")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/hybrid/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config = Phase2ExperimentConfig.from_yaml(args.config)
    device = args.device or config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load cache
    logger.info(f"Loading cache from {config.data.cache_dir}")
    cache = load_cache(config.data.cache_dir)

    # Common dataset kwargs
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

    # Create model and load weights
    model = Phase2Model(config.model)
    model = load_model_with_ema(model, args.checkpoint, device)

    # Resolve AMP settings from training config
    use_amp = config.training.use_amp
    amp_dtype_str = getattr(config.training, "amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Process each split
    all_results = {}
    for split_name, seasons in [
        ("train", config.data.train_seasons),
        ("val", config.data.val_seasons),
        ("test", config.data.test_seasons),
    ]:
        logger.info(f"Processing {split_name} split: {seasons}")
        dataset = Phase2Dataset(
            seasons=seasons,
            enable_augmentation=False,  # No augmentation for embeddings
            **common_kwargs,
        )

        # Pre-validate: build each sample to find which game_ids succeed.
        # This ensures the Subset dataset never triggers the recursive
        # fallback in __getitem__, so game_id tracking stays aligned.
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
                f"  {split_name}: skipped {n_skipped}/{len(dataset)} games "
                f"that failed build_sample"
            )

        valid_dataset = Subset(dataset, valid_indices)

        loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_phase2,
            num_workers=0,  # Avoid multiprocessing to keep deterministic order
            pin_memory=device == "cuda",
        )

        result = extract_embeddings(
            model, loader, valid_game_ids, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        logger.info(
            f"  {split_name}: {len(result['game_ids'])} games, "
            f"home_repr shape: {result['home_repr'].shape}"
        )
        all_results[split_name] = result

    # Combine all splits
    combined = {}
    for key in ["game_ids", "home_repr", "away_repr", "matchup_repr",
                "target_home_score", "target_away_score"]:
        combined[key] = np.concatenate(
            [all_results[s][key] for s in ["train", "val", "test"]], axis=0
        )

    # Save split boundaries
    n_train = len(all_results["train"]["game_ids"])
    n_val = len(all_results["val"]["game_ids"])
    n_test = len(all_results["test"]["game_ids"])
    combined["split_sizes"] = np.array([n_train, n_val, n_test])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    logger.info(
        f"Saved embeddings to {output_path}: "
        f"{n_train} train + {n_val} val + {n_test} test = "
        f"{n_train + n_val + n_test} total games"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
