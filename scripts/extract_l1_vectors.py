#!/usr/bin/env python3
"""
Extract per-player-per-game L1 ability vectors from the trained NKE-H model.

Runs the Kalman filter over each player's full career to produce:
  - ability[t]: (32,) ability vector at each game t
  - uncertainty[t]: (32,) P diagonal at each game t
  - archetype_weights: (10,) soft archetype assignment

Output: data/l2_cache/l1_vectors/{person_id}.npz

Resume-capable: skips players whose output file already exists.

Usage:
    python scripts/extract_l1_vectors.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phase5.config import NKEHConfig
from src.phase5.dataset import CACHE_DIR, Normalizer, load_metadata, load_profiles
from src.phase5.model import NKEH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "phase5" / "phase2_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Maximum sequence length per forward pass (must match training)
MAX_LEN = 300


def load_model(metadata: dict) -> NKEH:
    """Load NKE-H model from phase2 checkpoint."""
    cfg = NKEHConfig(
        n_box_stats=len(metadata["box_stat_columns"]),
        n_pbp_stats=len(metadata["pbp_stat_columns"]),
        n_context=len(metadata["context_columns"]),
        n_profile=len(metadata["profile_columns"]),
    )
    model = NKEH(cfg)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded (epoch {epoch}, {n_params:,} params) on {DEVICE}")
    return model


def prepare_player_inputs(
    data: dict,
    normalizer: Normalizer,
    profile_vec: torch.Tensor,
    start: int,
    end: int,
    init_context_raw: np.ndarray,
) -> dict[str, torch.Tensor]:
    """
    Prepare model inputs for a player's game subsequence [start:end].

    Returns a dict of tensors ready for model.forward_sequence(), each with
    batch dim = 1.
    """
    T = end - start

    box = torch.tensor(data["box_stats"][start:end], dtype=torch.float32)
    pbp = torch.tensor(data["pbp_stats"][start:end], dtype=torch.float32)
    ctx = torch.tensor(data["context"][start:end], dtype=torch.float32)

    # Age: context col 0 is career tenure -> normalize to ~[-1, 1.5]
    age = (ctx[:, 0:1] - 8.0) / 8.0  # (T, 1)

    # Days gap: context col 8 is raw days_since_last_game (extract BEFORE normalization)
    days_gap = ctx[:, 8:9].clone()  # (T, 1)

    # Normalize inputs
    box_norm = normalizer.normalize_box(box)

    # Zero out PBP for games without PBP data
    pbp_norm = normalizer.normalize_pbp(pbp)
    has_pbp = data.get("has_pbp")
    if has_pbp is not None:
        no_pbp = ~torch.tensor(has_pbp[start:end], dtype=torch.bool)
        pbp_norm[no_pbp] = 0.0

    ctx_norm = normalizer.normalize_context(ctx)
    profile_norm = normalizer.normalize_profile(profile_vec)

    # Career-start context for prior initialization
    init_ctx = torch.tensor(init_context_raw, dtype=torch.float32)
    init_ctx_norm = normalizer.normalize_context(init_ctx)

    # Pad to MAX_LEN if needed
    pad_len = MAX_LEN - T
    mask = torch.zeros(MAX_LEN, dtype=torch.bool)
    mask[:T] = True

    if pad_len > 0:
        box_norm = torch.cat([box_norm, torch.zeros(pad_len, box_norm.shape[1])], dim=0)
        pbp_norm = torch.cat([pbp_norm, torch.zeros(pad_len, pbp_norm.shape[1])], dim=0)
        ctx_norm = torch.cat([ctx_norm, torch.zeros(pad_len, ctx_norm.shape[1])], dim=0)
        age = torch.cat([age, torch.zeros(pad_len, 1)], dim=0)
        days_gap = torch.cat([days_gap, torch.zeros(pad_len, 1)], dim=0)

    # Add batch dimension and move to device
    return {
        "box_stats_seq": box_norm.unsqueeze(0).to(DEVICE),
        "pbp_stats_seq": pbp_norm.unsqueeze(0).to(DEVICE),
        "context_seq": ctx_norm.unsqueeze(0).to(DEVICE),
        "profile": profile_norm.unsqueeze(0).to(DEVICE),
        "age_seq": age.unsqueeze(0).to(DEVICE),
        "seq_mask": mask.unsqueeze(0).to(DEVICE),
        "days_gap_seq": days_gap.unsqueeze(0).to(DEVICE),
        "init_context": init_ctx_norm.unsqueeze(0).to(DEVICE),
    }


def extract_player(
    pid: int,
    data: dict,
    model: NKEH,
    normalizer: Normalizer,
    profile_vec: torch.Tensor,
) -> dict[str, np.ndarray]:
    """
    Run the Kalman filter over a player's full career using a sliding window.

    For players with <= MAX_LEN games, one forward pass suffices.
    For longer careers, we use overlapping windows: process the first MAX_LEN
    games, then slide by 1 game at a time? No -- that's too slow. Instead:
      - Window 1: games [0, MAX_LEN) -> extract vectors for games [0, MAX_LEN)
      - Window 2: games [1, MAX_LEN+1) -> extract vector for game MAX_LEN
      - Window k: games [k-1, MAX_LEN+k-1) -> extract vector for game MAX_LEN+k-2
    This gives each game the maximum context. But for simplicity and speed,
    we use a step-based approach: slide the window so that each game is covered
    at least once, always taking the vector from the last position of its window
    (maximizing context).

    Actually, the simplest correct approach for long careers:
      - Use non-overlapping windows of MAX_LEN, EXCEPT the last window
        which overlaps to cover remaining games.
      - For each window, the Kalman state builds from position 0, so earlier
        games in each window have less context.
      - To maximize quality: use a single sliding window that steps by 1,
        always extracting the LAST valid position. But this is O(N*MAX_LEN).

    Compromise approach (good quality, efficient):
      - Process games [0, MAX_LEN) -> extract all MAX_LEN vectors
      - For remaining games, use windows that step by (MAX_LEN // 2),
        always extracting only the second half of each window.
      - This gives every game at least MAX_LEN//2 context games.
    """
    N = len(data["box_stats"])
    init_context_raw = data["context"][0].copy()

    if N <= MAX_LEN:
        # Simple case: single forward pass
        inputs = prepare_player_inputs(
            data, normalizer, profile_vec, 0, N, init_context_raw
        )
        with torch.no_grad():
            out = model.forward_sequence(**inputs)

        ability = out["ability"][0, :N].cpu().numpy()  # (N, 32)
        P = out["P"][0, :N].cpu().numpy()  # (N, 32)
        arch_weights = out["archetype_weights"][0].cpu().numpy()  # (10,)

        return {
            "ability": ability,
            "uncertainty": P,
            "archetype_weights": arch_weights,
        }

    # Long career: sliding window with half-step overlap
    step = MAX_LEN // 2  # 150 games
    ability_all = np.zeros((N, 32), dtype=np.float32)
    P_all = np.zeros((N, 32), dtype=np.float32)
    arch_weights_final = None
    extracted = np.zeros(N, dtype=bool)

    # First window: extract all positions
    inputs = prepare_player_inputs(
        data, normalizer, profile_vec, 0, MAX_LEN, init_context_raw
    )
    with torch.no_grad():
        out = model.forward_sequence(**inputs)

    ability_all[:MAX_LEN] = out["ability"][0, :MAX_LEN].cpu().numpy()
    P_all[:MAX_LEN] = out["P"][0, :MAX_LEN].cpu().numpy()
    arch_weights_final = out["archetype_weights"][0].cpu().numpy()
    extracted[:MAX_LEN] = True

    # Subsequent windows: step by `step`, extract only the new (second-half) positions
    window_start = step
    while window_start < N:
        window_end = min(window_start + MAX_LEN, N)
        T_window = window_end - window_start

        inputs = prepare_player_inputs(
            data, normalizer, profile_vec, window_start, window_end, init_context_raw
        )
        with torch.no_grad():
            out = model.forward_sequence(**inputs)

        # Extract positions that haven't been extracted yet, or that now have
        # more context (the second half of this window)
        # For positions [window_start + step, window_end), this window provides
        # at least `step` games of context, which is better than no context.
        # For the first window's second half, we already have good vectors,
        # so only extract positions beyond what we've already covered.
        extract_from = max(step, 0)  # local index within this window
        # But we only want to extract positions not yet covered
        global_extract_start = window_start + extract_from
        if global_extract_start >= N:
            break

        # Extract from `extract_from` to end of valid data in this window
        local_end = T_window
        global_end = window_start + local_end

        # Only overwrite positions not yet extracted (or with less context)
        for local_idx in range(extract_from, local_end):
            global_idx = window_start + local_idx
            if global_idx < N and not extracted[global_idx]:
                ability_all[global_idx] = out["ability"][0, local_idx].cpu().numpy()
                P_all[global_idx] = out["P"][0, local_idx].cpu().numpy()
                extracted[global_idx] = True

        # Update archetype weights from the latest window
        arch_weights_final = out["archetype_weights"][0].cpu().numpy()

        window_start += step

    # Verify all positions were extracted
    if not extracted.all():
        missing = (~extracted).sum()
        logger.warning(
            f"Player {pid}: {missing}/{N} games not extracted (should not happen)"
        )

    return {
        "ability": ability_all,
        "uncertainty": P_all,
        "archetype_weights": arch_weights_final,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract L1 ability vectors for L2")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    normalizer = Normalizer(metadata)
    profiles_data = load_profiles()

    # Build profile lookup
    pids_array = profiles_data["person_ids"]
    profile_idx = {int(pid): i for i, pid in enumerate(pids_array)}
    profile_columns = metadata["profile_columns"]

    # Load model
    model = load_model(metadata)

    # Discover all player cache files
    player_dir = CACHE_DIR / "players"
    player_files = sorted(player_dir.glob("*.npz"))
    logger.info(f"Found {len(player_files)} player cache files")

    # Filter to players with profiles
    valid_players = []
    for pf in player_files:
        pid = int(pf.stem)
        if pid in profile_idx:
            valid_players.append((pid, pf))
        else:
            pass  # No profile, skip silently

    logger.info(f"{len(valid_players)} players have profiles")

    # Process players
    processed = 0
    skipped = 0
    errors = 0
    total_games = 0
    t0 = time.time()

    for i, (pid, cache_path) in enumerate(valid_players):
        out_path = OUTPUT_DIR / f"{pid}.npz"

        # Resume: skip if already processed
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            data = np.load(cache_path, allow_pickle=True)
            player_data = {
                "box_stats": data["box_stats"],
                "pbp_stats": data["pbp_stats"],
                "context": data["context"],
                "has_pbp": data.get(
                    "has_pbp", np.ones(len(data["game_ids"]), dtype=bool)
                ),
            }
            game_ids = data["game_ids"]
            N = len(game_ids)

            if N < 2:
                continue

            # Get profile vector
            idx = profile_idx[pid]
            vals = [float(profiles_data[col][idx]) for col in profile_columns]
            profile_vec = torch.tensor(vals, dtype=torch.float32)

            # Extract vectors
            result = extract_player(pid, player_data, model, normalizer, profile_vec)

            # Save
            np.savez_compressed(
                out_path,
                game_ids=game_ids,
                ability=result["ability"],
                uncertainty=result["uncertainty"],
                archetype_weights=result["archetype_weights"],
            )

            processed += 1
            total_games += N

        except Exception as e:
            logger.error(f"Error processing player {pid}: {e}")
            errors += 1

        # Progress logging
        done = processed + skipped + errors
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {done}/{len(valid_players)} "
                f"({processed} processed, {skipped} skipped, {errors} errors) "
                f"[{rate:.1f} players/sec, {total_games:,} total games]"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Done: {processed} processed, {skipped} skipped, {errors} errors "
        f"in {elapsed:.1f}s ({total_games:,} total games)"
    )

    # Save summary metadata
    summary = {
        "n_players": processed + skipped,
        "n_processed": processed,
        "n_skipped": skipped,
        "n_errors": errors,
        "total_games": total_games,
        "d_ability": 32,
        "n_archetypes": 10,
        "max_len": MAX_LEN,
        "checkpoint": str(CKPT_PATH),
        "device": str(DEVICE),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
