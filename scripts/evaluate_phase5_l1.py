#!/usr/bin/env python3
"""
Comprehensive evaluation of Phase 5 NKE-H L1 player ability model.

Evaluates whether the 32-d ability vectors are meaningful via:
  1. Archetype Cluster Analysis
  2. Trade Stability Test (cosine sim before/after team change)
  3. DPM Correlation (linear regression from ability → DPM)
  4. Uncertainty Reduction Curve
  5. Player Similarity Search (nearest neighbors)
  6. Curry vs Jokic Dimension Analysis
  7. Decoder Quality Check (predicted vs actual box stats)

Usage:
    python scripts/evaluate_phase5_l1.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import get_db
from src.phase5.config import NKEHConfig
from src.phase5.dataset import (
    CACHE_DIR,
    CareerSequenceDataset,
    Normalizer,
    load_metadata,
    load_profiles,
)
from src.phase5.model import NKEH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reference players for similarity search
REFERENCE_PLAYERS = {
    203999: "Nikola Jokic",
    201939: "Stephen Curry",
    2544: "LeBron James",
    203507: "Giannis Antetokounmpo",
    203497: "Rudy Gobert",
}


# ============================================================================
# Helpers
# ============================================================================


def load_model(metadata: dict) -> NKEH:
    """Load NKE-H model from best available checkpoint."""
    cfg = NKEHConfig(
        n_box_stats=len(metadata["box_stat_columns"]),
        n_pbp_stats=len(metadata["pbp_stat_columns"]),
        n_context=len(metadata["context_columns"]),
        n_profile=len(metadata["profile_columns"]),
    )
    model = NKEH(cfg)

    # Try phase2 checkpoint first, fall back to phase1
    ckpt_path = CKPT_DIR / "phase2_best.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "phase1_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {CKPT_DIR}. "
            "Expected phase2_best.pt or phase1_best.pt."
        )

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    logger.info(f"Model loaded (epoch {epoch}), device={DEVICE}")
    return model


def get_player_name(person_id: int) -> str:
    """Look up a player's name from the database."""
    try:
        with get_db(str(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT full_name FROM Players WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            if row:
                return row[0]
    except Exception:
        pass
    return f"Player#{person_id}"


def build_player_name_cache(player_ids: list[int]) -> dict[int, str]:
    """Bulk-load player names from the database."""
    names = {}
    try:
        with get_db(str(DB_PATH)) as conn:
            placeholders = ",".join("?" for _ in player_ids)
            rows = conn.execute(
                f"SELECT person_id, full_name FROM Players "
                f"WHERE person_id IN ({placeholders})",
                player_ids,
            ).fetchall()
            for pid, name in rows:
                names[pid] = name
    except Exception:
        pass
    return names


def extract_all_ability_vectors(
    model: NKEH,
    dataset: CareerSequenceDataset,
    batch_size: int = 32,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict]]:
    """
    Run the model on all players and extract final ability vectors,
    uncertainties, and full sequential outputs.

    Returns:
        abilities: {player_id: (32,) ndarray}
        uncertainties: {player_id: (32,) ndarray}
        seq_data: {player_id: {"ability": (T, 32), "P": (T, 32), "seq_len": int}}
    """
    abilities = {}
    uncertainties = {}
    seq_data = {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            box = batch["box_stats"].to(DEVICE)
            pbp = batch["pbp_stats"].to(DEVICE)
            ctx = batch["context"].to(DEVICE)
            profile = batch["profile"].to(DEVICE)
            age = batch["age"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            days_gap = batch["days_gap"].to(DEVICE)
            seq_lens = batch["seq_len"].cpu().numpy()

            init_ctx = batch.get("init_context", ctx[:, 0]).to(DEVICE)
            out = model.forward_sequence(
                box, pbp, ctx, profile, age, mask, days_gap, init_context=init_ctx
            )

            ability_seq = out["ability"].cpu().numpy()  # (B, T, 32)
            P_seq = out["P"].cpu().numpy()  # (B, T, 32)

            B = box.shape[0]
            for i in range(B):
                global_idx = batch_idx * batch_size + i
                if global_idx >= len(dataset.players):
                    break
                pid = dataset.players[global_idx]
                T = int(seq_lens[i])

                abilities[pid] = ability_seq[i, T - 1]
                uncertainties[pid] = P_seq[i, T - 1]
                seq_data[pid] = {
                    "ability": ability_seq[i, :T],
                    "P": P_seq[i, :T],
                    "seq_len": T,
                }

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} players"
                )

    logger.info(f"Extracted ability vectors for {len(abilities)} players")
    return abilities, uncertainties, seq_data


# ============================================================================
# Evaluation 1: Archetype Cluster Analysis
# ============================================================================


def eval_archetype_clusters(
    model: NKEH,
    dataset: CareerSequenceDataset,
    abilities: dict[int, np.ndarray],
    names: dict[int, str],
) -> None:
    """Analyze archetype prototype structure and top players per cluster."""
    print("\n" + "=" * 80)
    print("EVALUATION 1: Archetype Cluster Analysis")
    print("=" * 80)

    # Get archetype prototypes from the model
    proto_mu = model.archetype_network.prototype_mu.detach().cpu().numpy()  # (K, 32)
    K, d = proto_mu.shape
    print(f"\nArchetype prototypes: K={K}, d_ability={d}")

    # Compute soft assignment for all players by running initialize_state
    # We need profile + context for each player — use the dataset
    archetype_weights_all = {}
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            profile = batch["profile"].to(DEVICE)
            ctx_first = batch["context"][:, 0].to(DEVICE)  # first game context

            _, _, arch_w = model.initialize_state(profile, ctx_first)
            arch_w = arch_w.cpu().numpy()

            B = profile.shape[0]
            for i in range(B):
                global_idx = batch_idx * 64 + i
                if global_idx < len(dataset.players):
                    pid = dataset.players[global_idx]
                    archetype_weights_all[pid] = arch_w[i]

    # For each archetype, find top 5 players by weight
    print("\n--- Top 5 Players per Archetype ---")
    for k in range(K):
        scores = []
        for pid, w in archetype_weights_all.items():
            scores.append((pid, w[k]))
        scores.sort(key=lambda x: x[1], reverse=True)
        top5 = scores[:5]

        print(f"\nArchetype {k}:")
        # Report centroid stats summary (norm, top 3 dims)
        centroid = proto_mu[k]
        top_dims = np.argsort(np.abs(centroid))[::-1][:3]
        dim_str = ", ".join(f"d{d_i}={centroid[d_i]:+.3f}" for d_i in top_dims)
        print(f"  Centroid: norm={np.linalg.norm(centroid):.3f}, top dims: {dim_str}")

        for pid, weight in top5:
            name = names.get(pid, f"#{pid}")
            print(f"  {name:35s}  weight={weight:.4f}")

    # Archetype usage distribution
    print("\n--- Archetype Usage Distribution ---")
    # For each player, find dominant archetype
    dominant_counts = np.zeros(K, dtype=int)
    for w in archetype_weights_all.values():
        dominant_counts[np.argmax(w)] += 1
    for k in range(K):
        bar = "#" * int(dominant_counts[k] / max(dominant_counts) * 40)
        print(f"  Archetype {k}: {dominant_counts[k]:4d} players  {bar}")


# ============================================================================
# Evaluation 2: Trade Stability Test
# ============================================================================


def eval_trade_stability(
    model: NKEH,
    metadata: dict,
    abilities: dict[int, np.ndarray],
    names: dict[int, str],
) -> None:
    """
    Find mid-season team changes and compare ability vectors before/after.
    Target: cosine similarity > 0.85.
    """
    print("\n" + "=" * 80)
    print("EVALUATION 2: Trade Stability Test")
    print("=" * 80)

    # Query DB for players who changed teams mid-season
    trades = []
    with get_db(str(DB_PATH)) as conn:
        # Find players with multiple team_ids in a single season
        # Focus on seasons 2015-2024 for relevance
        rows = conn.execute("""
            SELECT pb.player_id, pb.team_id, g.game_id, g.date_time_utc, g.season
            FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            WHERE pb.min > 0
              AND g.season_type = 'Regular Season'
              AND g.season >= '2015-2016'
              AND g.season <= '2024-2025'
            ORDER BY pb.player_id, g.date_time_utc
            """).fetchall()

    # Group by player_id, detect team changes within a season
    player_games = defaultdict(list)
    for player_id, team_id, game_id, dt, season in rows:
        player_games[player_id].append(
            {
                "team_id": team_id,
                "game_id": game_id,
                "date": dt,
                "season": season,
            }
        )

    trade_events = []
    for pid, games in player_games.items():
        if pid not in abilities:
            continue

        for i in range(1, len(games)):
            if (
                games[i]["team_id"] != games[i - 1]["team_id"]
                and games[i]["season"] == games[i - 1]["season"]
            ):
                # Found a mid-season trade
                trade_events.append(
                    {
                        "player_id": pid,
                        "trade_idx": i,
                        "from_team": games[i - 1]["team_id"],
                        "to_team": games[i]["team_id"],
                        "season": games[i]["season"],
                        "date": games[i]["date"],
                        "n_games": len(games),
                    }
                )

    logger.info(f"Found {len(trade_events)} mid-season team changes")

    # Filter: need at least 20 games before AND 20 games after
    valid_trades = []
    for te in trade_events:
        idx = te["trade_idx"]
        if idx >= 20 and (te["n_games"] - idx) >= 20:
            valid_trades.append(te)

    # Deduplicate: keep only the first trade per player per season
    seen = set()
    unique_trades = []
    for te in valid_trades:
        key = (te["player_id"], te["season"])
        if key not in seen:
            seen.add(key)
            unique_trades.append(te)

    logger.info(f"Trades with 20+ games before/after: {len(unique_trades)}")

    if not unique_trades:
        print("\n  No trade events found with sufficient game data. Skipping.")
        return

    # For each trade, run the model on the 20-game windows and compare
    normalizer = Normalizer(metadata)
    profiles_data = load_profiles()
    profile_pids = profiles_data["person_ids"]
    profile_idx_map = {int(pid): i for i, pid in enumerate(profile_pids)}
    profile_cols = metadata["profile_columns"]

    sims = []
    results = []

    for te in unique_trades[:50]:  # Cap at 50 for performance
        pid = te["player_id"]
        if pid not in profile_idx_map:
            continue

        cache_path = CACHE_DIR / "players" / f"{pid}.npz"
        if not cache_path.exists():
            continue

        data = np.load(cache_path, allow_pickle=True)
        game_ids = list(data["game_ids"])
        n_total = len(game_ids)

        # We need to find the trade point in the cache (which is chronological)
        # Use the trade_idx from the DB query as an approximation: match by
        # finding where team changes happen. Instead, run the full career and
        # extract the pre/post windows from the sequential output.
        # Actually, to be rigorous, we need to align DB ordering with cache ordering.
        # The simplest approach: run the full career through the model and
        # pick timesteps near the trade.

        # Load full career
        box = torch.tensor(data["box_stats"], dtype=torch.float32)
        pbp = torch.tensor(data["pbp_stats"], dtype=torch.float32)
        ctx = torch.tensor(data["context"], dtype=torch.float32)

        has_pbp = data.get("has_pbp", np.ones(n_total, dtype=bool))

        # Get profile
        pidx = profile_idx_map[pid]
        profile_vals = [float(profiles_data[col][pidx]) for col in profile_cols]
        profile_t = torch.tensor(profile_vals, dtype=torch.float32)

        # Normalize
        box_norm = normalizer.normalize_box(box)
        pbp_norm = normalizer.normalize_pbp(pbp)
        no_pbp = ~torch.tensor(has_pbp, dtype=torch.bool)
        pbp_norm[no_pbp] = 0.0
        ctx_norm = normalizer.normalize_context(ctx)
        profile_norm = normalizer.normalize_profile(profile_t)

        age_t = ctx[:, 0:1]  # (T, 1) unnormalized age for Kalman predict
        # Use normalized context for the model
        age_input = ctx_norm[:, 0:1]  # normalized age
        # Raw days_since_last_game for time-scaled Kalman predict (col 8)
        days_gap_raw = ctx[:, 8:9]  # (T, 1) — raw calendar days

        # Truncate if too long
        max_len = min(n_total, 600)
        if n_total > max_len:
            offset = n_total - max_len
        else:
            offset = 0

        T = min(n_total, max_len)

        # Run forward pass
        with torch.no_grad():
            ctx_slice = ctx_norm[offset : offset + T].unsqueeze(0).to(DEVICE)
            out = model.forward_sequence(
                box_norm[offset : offset + T].unsqueeze(0).to(DEVICE),
                pbp_norm[offset : offset + T].unsqueeze(0).to(DEVICE),
                ctx_slice,
                profile_norm.unsqueeze(0).to(DEVICE),
                age_input[offset : offset + T].unsqueeze(0).to(DEVICE),
                seq_mask=None,
                days_gap_seq=days_gap_raw[offset : offset + T].unsqueeze(0).to(DEVICE),
                init_context=ctx_norm[0:1].to(DEVICE),  # career start context
            )

        ability_seq = out["ability"][0].cpu().numpy()  # (T, 32)

        # The trade_idx is in the DB's game ordering; approximate position
        # in cache using relative position
        approx_trade_t = te["trade_idx"] - offset
        if approx_trade_t < 20 or approx_trade_t >= T - 20:
            continue

        # Pre-trade: ability at timestep (trade - 1), post-trade: at (trade + 19)
        vec_pre = ability_seq[approx_trade_t - 1]
        vec_post = ability_seq[min(approx_trade_t + 19, T - 1)]

        cos_sim = float(
            np.dot(vec_pre, vec_post)
            / (np.linalg.norm(vec_pre) * np.linalg.norm(vec_post) + 1e-8)
        )
        sims.append(cos_sim)
        results.append(
            {
                "player": names.get(pid, f"#{pid}"),
                "season": te["season"],
                "cos_sim": cos_sim,
            }
        )

    if not sims:
        print("\n  Could not compute any trade stability comparisons. Skipping.")
        return

    sims_arr = np.array(sims)
    print(f"\n  Trades analyzed: {len(sims)}")
    print(f"  Mean cosine similarity:   {sims_arr.mean():.4f}")
    print(f"  Median cosine similarity: {np.median(sims_arr):.4f}")
    print(f"  Std:                      {sims_arr.std():.4f}")
    print(f"  Min:                      {sims_arr.min():.4f}")
    print(f"  Max:                      {sims_arr.max():.4f}")
    print(f"  Fraction >= 0.85:         {(sims_arr >= 0.85).mean():.1%}")
    target_met = "PASS" if sims_arr.mean() >= 0.85 else "FAIL"
    print(f"  Target (mean >= 0.85):    {target_met}")

    # Show a few examples
    results.sort(key=lambda x: x["cos_sim"])
    print("\n  Lowest stability (potential concern):")
    for r in results[:5]:
        print(f"    {r['player']:35s} {r['season']}  cos_sim={r['cos_sim']:.4f}")
    print("\n  Highest stability:")
    for r in results[-5:]:
        print(f"    {r['player']:35s} {r['season']}  cos_sim={r['cos_sim']:.4f}")


# ============================================================================
# Evaluation 3: DPM Correlation
# ============================================================================


def eval_dpm_correlation(
    abilities: dict[int, np.ndarray],
    names: dict[int, str],
) -> None:
    """
    Train linear regression from 32-d ability vector to DPM.
    Target: Pearson r > 0.70.
    """
    print("\n" + "=" * 80)
    print("EVALUATION 3: DPM Correlation")
    print("=" * 80)

    # Get average DPM per player from the database
    dpm_data = {}
    with get_db(str(DB_PATH)) as conn:
        rows = conn.execute("""
            SELECT person_id,
                   AVG(dpm) as avg_dpm,
                   AVG(o_dpm) as avg_o_dpm,
                   AVG(d_dpm) as avg_d_dpm,
                   COUNT(*) as n_games
            FROM DarkoDPM
            WHERE dpm IS NOT NULL
            GROUP BY person_id
            HAVING n_games >= 50
            """).fetchall()
        for pid, avg_dpm, avg_o_dpm, avg_d_dpm, n_games in rows:
            dpm_data[pid] = {
                "dpm": avg_dpm,
                "o_dpm": avg_o_dpm,
                "d_dpm": avg_d_dpm,
                "n_games": n_games,
            }

    # Intersect players with both ability vectors and DPM data
    common_pids = sorted(set(abilities.keys()) & set(dpm_data.keys()))
    logger.info(f"DPM correlation: {len(common_pids)} players with both ability + DPM")

    if len(common_pids) < 30:
        print("\n  Too few players with DPM data. Skipping.")
        return

    X = np.array([abilities[pid] for pid in common_pids])  # (N, 32)
    y_dpm = np.array([dpm_data[pid]["dpm"] for pid in common_pids])
    y_odpm = np.array([dpm_data[pid]["o_dpm"] for pid in common_pids])
    y_ddpm = np.array([dpm_data[pid]["d_dpm"] for pid in common_pids])

    for target_name, y in [("DPM", y_dpm), ("O-DPM", y_odpm), ("D-DPM", y_ddpm)]:
        # Filter NaNs
        valid = ~np.isnan(y)
        X_valid = X[valid]
        y_valid = y[valid]

        if len(y_valid) < 30:
            print(f"\n  {target_name}: Too few valid samples. Skipping.")
            continue

        reg = LinearRegression()
        reg.fit(X_valid, y_valid)
        y_pred = reg.predict(X_valid)

        r_val, p_val = scipy_stats.pearsonr(y_valid, y_pred)
        r2 = reg.score(X_valid, y_valid)
        mae = np.mean(np.abs(y_valid - y_pred))
        rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))

        target_met = "PASS" if abs(r_val) >= 0.70 else "FAIL"
        print(f"\n  {target_name} (N={len(y_valid)}):")
        print(f"    Pearson r:  {r_val:.4f}  (p={p_val:.2e})  [{target_met}]")
        print(f"    R-squared:  {r2:.4f}")
        print(f"    MAE:        {mae:.3f}")
        print(f"    RMSE:       {rmse:.3f}")

        # Top dimensions by regression weight
        top_dims = np.argsort(np.abs(reg.coef_))[::-1][:5]
        weights_str = ", ".join(f"d{d_i}={reg.coef_[d_i]:+.4f}" for d_i in top_dims)
        print(f"    Top 5 dims: {weights_str}")


# ============================================================================
# Evaluation 4: Uncertainty Reduction Curve
# ============================================================================


def eval_uncertainty_reduction(
    seq_data: dict[int, dict],
    names: dict[int, str],
) -> None:
    """
    Track mean uncertainty P over game count. Should decrease monotonically.
    """
    print("\n" + "=" * 80)
    print("EVALUATION 4: Uncertainty Reduction Curve")
    print("=" * 80)

    # Collect uncertainty at various game counts
    checkpoints = [1, 5, 10, 20, 50, 82, 200, 400]
    uncertainty_at = {t: [] for t in checkpoints}

    for pid, sd in seq_data.items():
        T = sd["seq_len"]
        P_seq = sd["P"]  # (T, 32)

        for cp in checkpoints:
            if cp <= T:
                # Mean uncertainty (mean of diagonal P) at this game
                mean_unc = P_seq[cp - 1].mean()
                uncertainty_at[cp].append(mean_unc)

    print(
        f"\n  {'Games':>8s}  {'N Players':>10s}  {'Mean Unc':>10s}  {'Std':>10s}  {'Median':>10s}"
    )
    print("  " + "-" * 58)

    prev_mean = None
    monotonic = True
    for cp in checkpoints:
        vals = uncertainty_at[cp]
        if not vals:
            continue
        arr = np.array(vals)
        mean_val = arr.mean()
        std_val = arr.std()
        median_val = np.median(arr)

        marker = ""
        if prev_mean is not None and mean_val > prev_mean:
            marker = " <-- NOT monotonic"
            monotonic = False
        prev_mean = mean_val

        print(
            f"  {cp:>8d}  {len(vals):>10d}  {mean_val:>10.5f}  "
            f"{std_val:>10.5f}  {median_val:>10.5f}{marker}"
        )

    result = "PASS" if monotonic else "FAIL"
    print(f"\n  Monotonically decreasing: {result}")

    # Show a specific example: a long-career player
    long_careers = [
        (pid, sd["seq_len"]) for pid, sd in seq_data.items() if sd["seq_len"] >= 200
    ]
    long_careers.sort(key=lambda x: x[1], reverse=True)

    if long_careers:
        ex_pid, ex_len = long_careers[0]
        ex_name = names.get(ex_pid, f"#{ex_pid}")
        P_ex = seq_data[ex_pid]["P"]  # (T, 32)
        print(f"\n  Example player: {ex_name} ({ex_len} games)")
        print(f"  {'Game':>8s}  {'Mean Unc':>10s}")
        for cp in checkpoints:
            if cp <= ex_len:
                mu = P_ex[cp - 1].mean()
                print(f"  {cp:>8d}  {mu:>10.5f}")


# ============================================================================
# Evaluation 5: Player Similarity Search
# ============================================================================


def eval_player_similarity(
    abilities: dict[int, np.ndarray],
    names: dict[int, str],
) -> None:
    """Find 5 nearest neighbors by cosine similarity for reference players."""
    print("\n" + "=" * 80)
    print("EVALUATION 5: Player Similarity Search")
    print("=" * 80)

    # Build ability matrix
    all_pids = sorted(abilities.keys())
    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
    ability_mat = np.array([abilities[pid] for pid in all_pids])  # (N, 32)

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(ability_mat, axis=1, keepdims=True) + 1e-8
    ability_normed = ability_mat / norms

    for ref_pid, ref_name in REFERENCE_PLAYERS.items():
        if ref_pid not in pid_to_idx:
            print(f"\n  {ref_name}: Not found in dataset. Skipping.")
            continue

        ref_idx = pid_to_idx[ref_pid]
        ref_vec = ability_normed[ref_idx]

        # Cosine similarity to all players
        sims = ability_normed @ ref_vec  # (N,)
        # Exclude self
        sims[ref_idx] = -1.0

        top_k = np.argsort(sims)[::-1][:5]
        print(f"\n  {ref_name} — 5 nearest neighbors:")
        for rank, idx in enumerate(top_k, 1):
            pid = all_pids[idx]
            name = names.get(pid, f"#{pid}")
            print(f"    {rank}. {name:35s}  cos_sim={sims[idx]:.4f}")


# ============================================================================
# Evaluation 6: Curry vs Jokic Dimension Analysis
# ============================================================================


def eval_curry_vs_jokic(
    abilities: dict[int, np.ndarray],
    uncertainties: dict[int, np.ndarray],
) -> None:
    """Per-dimension comparison of two top players with very different styles."""
    print("\n" + "=" * 80)
    print("EVALUATION 6: Curry vs Jokic Dimension Analysis")
    print("=" * 80)

    curry_pid = 201939
    jokic_pid = 203999

    if curry_pid not in abilities or jokic_pid not in abilities:
        print("\n  One or both players not found. Skipping.")
        return

    curry_vec = abilities[curry_pid]
    jokic_vec = abilities[jokic_pid]
    curry_unc = uncertainties[curry_pid]
    jokic_unc = uncertainties[jokic_pid]

    cos_sim = float(
        np.dot(curry_vec, jokic_vec)
        / (np.linalg.norm(curry_vec) * np.linalg.norm(jokic_vec) + 1e-8)
    )
    l2_dist = float(np.linalg.norm(curry_vec - jokic_vec))

    print(f"\n  Cosine similarity: {cos_sim:.4f}")
    print(f"  L2 distance:       {l2_dist:.4f}")
    print(f"  Curry vector norm: {np.linalg.norm(curry_vec):.4f}")
    print(f"  Jokic vector norm: {np.linalg.norm(jokic_vec):.4f}")
    print(f"  Curry mean uncert: {curry_unc.mean():.5f}")
    print(f"  Jokic mean uncert: {jokic_unc.mean():.5f}")

    print(
        f"\n  {'Dim':>5s}  {'Curry':>10s}  {'Jokic':>10s}  {'Diff':>10s}  {'|Diff|':>8s}"
    )
    print("  " + "-" * 50)

    diffs = curry_vec - jokic_vec
    sorted_dims = np.argsort(np.abs(diffs))[::-1]

    for d_i in sorted_dims:
        marker = " ***" if np.abs(diffs[d_i]) > 0.5 else ""
        print(
            f"  {d_i:>5d}  {curry_vec[d_i]:>+10.4f}  {jokic_vec[d_i]:>+10.4f}  "
            f"{diffs[d_i]:>+10.4f}  {np.abs(diffs[d_i]):>8.4f}{marker}"
        )


# ============================================================================
# Evaluation 7: Decoder Quality Check
# ============================================================================


def eval_decoder_quality(
    model: NKEH,
    dataset: CareerSequenceDataset,
    metadata: dict,
) -> None:
    """
    Predict box stats from ability vectors and compare to actuals.
    Reports per-stat Pearson correlation.
    """
    print("\n" + "=" * 80)
    print("EVALUATION 7: Decoder Quality Check")
    print("=" * 80)

    stat_cols = metadata["box_stat_columns"]
    n_stats = len(stat_cols)

    # Collect predictions and actuals for up to 100 random players
    np.random.seed(42)
    n_eval = min(100, len(dataset))
    indices = np.random.choice(len(dataset), size=n_eval, replace=False)

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[int(idx)]
            seq_len = sample["seq_len"].item()
            if seq_len < 5:
                continue

            box = sample["box_stats"].unsqueeze(0).to(DEVICE)
            pbp = sample["pbp_stats"].unsqueeze(0).to(DEVICE)
            ctx = sample["context"].unsqueeze(0).to(DEVICE)
            profile = sample["profile"].unsqueeze(0).to(DEVICE)
            age = sample["age"].unsqueeze(0).to(DEVICE)
            mask = sample["mask"].unsqueeze(0).to(DEVICE)
            days_gap = sample["days_gap"].unsqueeze(0).to(DEVICE)

            init_ctx = sample.get("init_context", ctx[0, 0]).unsqueeze(0).to(DEVICE)
            out = model.forward_sequence(
                box, pbp, ctx, profile, age, mask, days_gap, init_context=init_ctx
            )

            # stat_recon and stat_target are both normalized — un-normalize for eval
            box_mean = dataset.normalizer.box_mean.numpy()
            box_std = dataset.normalizer.box_std.numpy()
            pred = out["stat_recon"][0, :seq_len].cpu().numpy() * box_std + box_mean
            actual = sample["stat_target"][:seq_len].numpy() * box_std + box_mean

            all_preds.append(pred)
            all_actuals.append(actual)

    if not all_preds:
        print("\n  No data collected. Skipping.")
        return

    all_preds = np.concatenate(all_preds, axis=0)  # (N_total, n_stats)
    all_actuals = np.concatenate(all_actuals, axis=0)

    print(f"\n  Total game-predictions evaluated: {len(all_preds)}")
    print(
        f"\n  {'Stat':>16s}  {'Pearson r':>10s}  {'MAE':>8s}  {'RMSE':>8s}  {'Actual Mean':>12s}  {'Pred Mean':>12s}"
    )
    print("  " + "-" * 76)

    correlations = []
    for i, col in enumerate(stat_cols):
        if i >= all_preds.shape[1]:
            break
        pred_col = all_preds[:, i]
        actual_col = all_actuals[:, i]

        valid = ~(np.isnan(pred_col) | np.isnan(actual_col))
        if valid.sum() < 10:
            continue

        r_val, _ = scipy_stats.pearsonr(actual_col[valid], pred_col[valid])
        mae = np.mean(np.abs(actual_col[valid] - pred_col[valid]))
        rmse = np.sqrt(np.mean((actual_col[valid] - pred_col[valid]) ** 2))
        actual_mean = actual_col[valid].mean()
        pred_mean = pred_col[valid].mean()
        correlations.append(r_val)

        print(
            f"  {col:>16s}  {r_val:>10.4f}  {mae:>8.2f}  {rmse:>8.2f}  "
            f"{actual_mean:>12.2f}  {pred_mean:>12.2f}"
        )

    if correlations:
        mean_r = np.mean(correlations)
        print(f"\n  Mean correlation across stats: {mean_r:.4f}")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 80)
    print("Phase 5 NKE-H L1 Player Model — Comprehensive Evaluation")
    print("=" * 80)

    # Load metadata and model
    metadata = load_metadata()
    model = load_model(metadata)

    # Build dataset with all players (use "test" split for broadest coverage,
    # or combine all splits — use test to avoid any data leakage concerns)
    profiles = load_profiles()
    all_player_ids = [int(pid) for pid in profiles["person_ids"]]
    logger.info(f"Total players in cache: {len(all_player_ids)}")

    # Use a large max_len to capture full careers; use "test" split bounds
    # that include everything from 2024+, but for evaluation we want ALL data,
    # so we create a dataset manually with broadened bounds.
    dataset = CareerSequenceDataset(
        player_ids=all_player_ids,
        metadata=metadata,
        max_len=300,
        split="test",  # 2024+ for clean evaluation
    )

    # If test split is too small, fall back to "val" or "pretrain"
    if len(dataset) < 100:
        logger.info(f"Test split has only {len(dataset)} players, trying 'train' split")
        dataset = CareerSequenceDataset(
            player_ids=all_player_ids,
            metadata=metadata,
            max_len=300,
            split="train",
        )

    if len(dataset) < 50:
        logger.info(
            f"Train split has only {len(dataset)} players, trying 'pretrain' split"
        )
        dataset = CareerSequenceDataset(
            player_ids=all_player_ids,
            metadata=metadata,
            max_len=300,
            split="pretrain",
        )

    logger.info(f"Evaluation dataset: {len(dataset)} players")

    # Build player name cache
    player_names = build_player_name_cache(
        [pid for pid in all_player_ids if pid in set(dataset.players)]
    )
    logger.info(f"Loaded names for {len(player_names)} players")

    # Extract ability vectors for all players
    logger.info("Extracting ability vectors for all players...")
    abilities, uncertainties, seq_data = extract_all_ability_vectors(
        model, dataset, batch_size=32
    )

    # Run evaluations
    eval_archetype_clusters(model, dataset, abilities, player_names)
    eval_trade_stability(model, metadata, abilities, player_names)
    eval_dpm_correlation(abilities, player_names)
    eval_uncertainty_reduction(seq_data, player_names)
    eval_player_similarity(abilities, player_names)
    eval_curry_vs_jokic(abilities, uncertainties)
    eval_decoder_quality(model, dataset, metadata)

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
