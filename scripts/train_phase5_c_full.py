#!/usr/bin/env python3
"""
Phase C Full Training: L1 UNFROZEN end-to-end with L2+L3+L4.

This is the most expensive training run. For each game in the batch, we run the
L1 Kalman filter live (forward_sequence) over the last T=20 games for every
player, producing ability vectors with gradients attached. These flow through
L2->L3->L4, and backward propagates all the way to L1 weights.

Approach:
  - Pre-load all player L1 cache data (box_stats, pbp_stats, context) into memory
  - For each game, look up the last 20 games for each player
  - Run L1.forward_sequence over the 20-game window (warm-started from pre-computed
    state at T-21 when available, otherwise cold-started)
  - Extract final ability vector (with grad_fn) -> feed to L2->L3->L4
  - Backward through entire stack including L1

Gradual unfreezing schedule:
  - Epochs 1-3:   L4 only (L1/L2/L3 frozen)
  - Epochs 4-6:   L3+L4
  - Epochs 7-10:  L2+L3+L4
  - Epochs 11+:   L1+L2+L3+L4 (full stack)

Discriminative learning rates:
  - L1: 1e-5  (lowest -- preserve Kalman dynamics)
  - L2: 3e-5
  - L3: 1e-4
  - L4: 3e-4  (highest -- prediction head adapts fastest)

Usage:
    python scripts/train_phase5_c_full.py
    python scripts/train_phase5_c_full.py --test --epochs 5
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.config import NKEHConfig
from src.phase5.model import NKEH, count_parameters as l1_count_params
from src.phase5.dataset import Normalizer, load_metadata, load_profiles
from src.phase5.l2_config import L2Config
from src.phase5.l2_model import (
    PlayerSynergyNetwork,
    count_parameters as l2_count_params,
)
from src.phase5.l3_config import L3Config
from src.phase5.l3_model import TeamModel, count_parameters as l3_count_params
from src.phase5.l4_config import L4Config
from src.phase5.l4_model import GamePredictor, count_parameters as l4_count_params
from src.phase5.l4_losses import game_prediction_loss

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase5"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

# Data paths
PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"
L1_CACHE_DIR = PROJECT_ROOT / "data" / "phase5_cache"
L1_VECTORS_DIR = PROJECT_ROOT / "data" / "l2_cache" / "l1_vectors"
WOWY_DB_PATH = PROJECT_ROOT / "data" / "l2_cache" / "wowy.sqlite"
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import shared helpers from Phase C script
# ---------------------------------------------------------------------------

from train_phase5_c import (
    load_wowy_lookup,
    build_years_together_lookup,
    load_birth_years,
    season_start_year,
    DriftMonitor,
)

# ---------------------------------------------------------------------------
# L1 raw data cache: load all player game data into memory
# ---------------------------------------------------------------------------


class L1RawCache:
    """Loads all player L1 cache data into memory for fast access.

    For each player, stores normalized box_stats, pbp_stats, context,
    age, days_gap, profile, and game_id index for temporal lookups.
    """

    def __init__(self, normalizer: Normalizer, metadata: dict):
        self.normalizer = normalizer
        self.metadata = metadata
        self.profiles_data = load_profiles()

        pids = self.profiles_data["person_ids"]
        self.profile_idx = {int(pid): i for i, pid in enumerate(pids)}

        profile_cols = metadata["profile_columns"]

        self.player_data: dict[int, dict] = {}
        player_dir = L1_CACHE_DIR / "players"
        npz_files = sorted(player_dir.glob("*.npz"))

        logger.info(f"Loading {len(npz_files)} player L1 caches into memory...")
        t0 = time.time()
        skipped = 0

        for path in npz_files:
            pid = int(path.stem)
            if pid not in self.profile_idx:
                skipped += 1
                continue

            data = np.load(str(path), allow_pickle=True)
            game_ids = list(data["game_ids"])
            n_games = len(game_ids)
            if n_games < 2:
                skipped += 1
                continue

            box = torch.tensor(data["box_stats"], dtype=torch.float32)
            pbp = torch.tensor(data["pbp_stats"], dtype=torch.float32)
            ctx_raw = torch.tensor(data["context"], dtype=torch.float32)
            has_pbp = data.get("has_pbp", np.ones(n_games, dtype=bool))

            # Normalize
            box_norm = normalizer.normalize_box(box)
            pbp_norm = normalizer.normalize_pbp(pbp)
            # Zero out PBP for games without PBP data
            no_pbp_mask = ~torch.tensor(has_pbp, dtype=torch.bool)
            pbp_norm[no_pbp_mask] = 0.0

            # Extract age and days_gap BEFORE normalizing context
            age = (ctx_raw[:, 0:1] - 8.0) / 8.0  # career tenure -> normalized age
            days_gap = ctx_raw[:, 8:9].clone()  # raw days since last game

            ctx_norm = normalizer.normalize_context(ctx_raw)

            # Profile (static)
            pidx = self.profile_idx[pid]
            vals = [float(self.profiles_data[col][pidx]) for col in profile_cols]
            profile = torch.tensor(vals, dtype=torch.float32)
            profile_norm = normalizer.normalize_profile(profile)

            # Init context: first game's context (for prior initialization)
            init_ctx = normalizer.normalize_context(ctx_raw[0:1]).squeeze(0)

            # Store as numpy for pickling (num_workers>0 in DataLoader)
            self.player_data[pid] = {
                "game_ids": game_ids,
                "box_stats": box_norm.numpy(),  # (T, 16)
                "pbp_stats": pbp_norm.numpy(),  # (T, 56)
                "context": ctx_norm.numpy(),  # (T, 12)
                "age": age.numpy(),  # (T, 1)
                "days_gap": days_gap.numpy(),  # (T, 1)
                "profile": profile_norm.numpy(),  # (9,)
                "init_context": init_ctx.numpy(),  # (12,)
            }

        elapsed = time.time() - t0
        logger.info(
            f"Loaded L1 raw cache: {len(self.player_data)} players in {elapsed:.1f}s "
            f"(skipped {skipped})"
        )

    def get_window(
        self, pid: int, game_id: str, window_size: int = 20
    ) -> dict[str, torch.Tensor] | None:
        """Get the last `window_size` games before `game_id` for a player.

        Returns normalized tensors ready for L1.forward_sequence, or None if
        the player has insufficient history.
        """
        if pid not in self.player_data:
            return None

        data = self.player_data[pid]
        game_ids = data["game_ids"]

        # Find the index of game_id (we want games BEFORE this one)
        if game_id in game_ids:
            target_idx = game_ids.index(game_id)
        else:
            # Binary search for the insertion point
            target_idx = 0
            for i, gid in enumerate(game_ids):
                if gid < game_id:
                    target_idx = i + 1
                else:
                    break

        if target_idx < 1:
            return None  # No history before this game

        # Take last window_size games ending before target_idx
        start = max(0, target_idx - window_size)
        end = target_idx

        T = end - start

        return {
            "box_stats": torch.from_numpy(data["box_stats"][start:end].copy()),
            "pbp_stats": torch.from_numpy(data["pbp_stats"][start:end].copy()),
            "context": torch.from_numpy(data["context"][start:end].copy()),
            "age": torch.from_numpy(data["age"][start:end].copy()),
            "days_gap": torch.from_numpy(data["days_gap"][start:end].copy()),
            "profile": torch.from_numpy(data["profile"].copy()),
            "init_context": torch.from_numpy(data["init_context"].copy()),
            "seq_len": T,
        }


# ---------------------------------------------------------------------------
# Pre-computed L1 vectors (for warm start + fallback)
# ---------------------------------------------------------------------------


def load_l1_vectors() -> dict[int, dict]:
    """Load all pre-computed L1 ability vectors."""
    vectors = {}
    npz_files = sorted(L1_VECTORS_DIR.glob("*.npz"))
    logger.info(f"Loading L1 pre-computed vectors from {len(npz_files)} files...")

    for path in npz_files:
        player_id = int(path.stem)
        data = np.load(str(path), allow_pickle=True)
        vectors[player_id] = {
            "ability": data["ability"],
            "uncertainty": data["uncertainty"],
            "game_ids": list(data["game_ids"]),
            "archetype_weights": data["archetype_weights"],
        }

    logger.info(f"Loaded L1 vectors for {len(vectors)} players")
    return vectors


def get_l1_at_game(
    l1_data: dict, game_id: str, use_latest: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Get pre-computed L1 vectors BEFORE a specific game (no leakage)."""
    game_ids = l1_data["game_ids"]
    archetype = l1_data["archetype_weights"]

    if game_id in game_ids:
        idx = game_ids.index(game_id)
        if idx > 0:
            idx = idx - 1
        return l1_data["ability"][idx], l1_data["uncertainty"][idx], archetype

    best_idx = -1
    for i, gid in enumerate(game_ids):
        if gid < game_id:
            best_idx = i
        else:
            break

    if best_idx >= 0:
        return l1_data["ability"][best_idx], l1_data["uncertainty"][best_idx], archetype

    if use_latest and len(game_ids) > 0:
        return l1_data["ability"][0], l1_data["uncertainty"][0], archetype

    return None, None, None


def build_player_to_idx(l1_vectors: dict[int, dict]) -> dict[int, int]:
    """Build player_id -> embedding index mapping. Index 0 = padding."""
    player_ids = sorted(l1_vectors.keys())
    return {pid: i + 1 for i, pid in enumerate(player_ids)}


# ---------------------------------------------------------------------------
# Dataset: carries per-player L1 window data + L2/L3/L4 features
# ---------------------------------------------------------------------------


class FullPhaseCDataset(Dataset):
    """
    Each sample provides:
    - Per-player L1 raw windows (for live L1 forward, gradient-attached)
    - Per-player pre-computed L1 vectors (fallback when window unavailable)
    - L2 edge features, L3 team features, L4 game context, targets

    The collate function handles variable-length L1 windows via padding.
    """

    N_EDGE_FEATURES = 9

    def __init__(
        self,
        indices: np.ndarray,
        # Per-game player data
        player_ids_arr: np.ndarray,  # (N, 2, A) int64 - actual player IDs
        player_masks: np.ndarray,  # (N, 2, A) bool
        player_indices: np.ndarray,  # (N, 2, A) int64 - embedding indices
        # Pre-computed fallback
        player_abilities: np.ndarray,  # (N, 2, A, 32)
        player_uncertainties: np.ndarray,  # (N, 2, A, 32)
        player_archetypes: np.ndarray,  # (N, 2, A, 10)
        # L1 raw cache for live forward
        l1_raw_cache: L1RawCache,
        l1_window_size: int,
        # Game identifiers
        game_ids_arr: np.ndarray,  # (N,) str - for temporal lookup
        seasons: np.ndarray,  # (N,) str
        # Phase B cache data
        team_features: np.ndarray,  # (N, 2, 34)
        game_context: np.ndarray,  # (N, 14)
        roster_summaries: np.ndarray,  # (N, 2, 12)
        targets_margin: np.ndarray,  # (N,)
        targets_win: np.ndarray,  # (N,)
        targets_total: np.ndarray,  # (N,)
        # Coach data
        coach_indices: np.ndarray | None = None,
        coach_games: np.ndarray | None = None,
        # Edge feature lookups
        wowy_lookup: dict | None = None,
        years_together_lookup: dict | None = None,
        birth_years: dict | None = None,
        # Normalization
        l3_mean: np.ndarray | None = None,
        l3_std: np.ndarray | None = None,
        l4_mean: np.ndarray | None = None,
        l4_std: np.ndarray | None = None,
        rs_mean: np.ndarray | None = None,
        rs_std: np.ndarray | None = None,
    ):
        self.indices = indices
        self.player_ids_arr = player_ids_arr
        self.player_masks = player_masks
        self.player_indices = player_indices
        self.player_abilities = player_abilities
        self.player_uncertainties = player_uncertainties
        self.player_archetypes = player_archetypes
        self.l1_raw_cache = l1_raw_cache
        self.l1_window_size = l1_window_size
        self.game_ids_arr = game_ids_arr
        self.seasons = seasons
        self.team_features = team_features
        self.game_context = game_context
        self.roster_summaries = roster_summaries
        self.targets_margin = targets_margin
        self.targets_win = targets_win
        self.targets_total = targets_total
        self.coach_indices = coach_indices
        self.coach_games = coach_games
        self.wowy_lookup = wowy_lookup or {}
        self.years_together_lookup = years_together_lookup or {}
        self.birth_years = birth_years or {}
        self.has_edge_features = True
        self.l3_mean = l3_mean
        self.l3_std = l3_std
        self.l4_mean = l4_mean
        self.l4_std = l4_std
        self.rs_mean = rs_mean
        self.rs_std = rs_std

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize(
        self, x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None
    ) -> np.ndarray:
        if mean is not None and std is not None:
            return (x - mean) / std
        return x

    def _build_edge_features(
        self, pids, mask, abilities, archetypes, season
    ) -> np.ndarray:
        """Build (A, A, 9) edge feature tensor for one team."""
        A = len(pids)
        edge_feat = np.zeros((A, A, self.N_EDGE_FEATURES), dtype=np.float32)
        valid_indices = np.where(mask)[0]
        n_valid = len(valid_indices)
        if n_valid < 2:
            return edge_feat

        for ii in range(n_valid):
            for jj in range(n_valid):
                if ii == jj:
                    continue
                vi = valid_indices[ii]
                vj = valid_indices[jj]
                pid_i = int(pids[vi])
                pid_j = int(pids[vj])
                ab_i = abilities[vi]
                ab_j = abilities[vj]
                arch_i = archetypes[vi]
                arch_j = archetypes[vj]

                pair_key = (min(pid_i, pid_j), max(pid_i, pid_j))
                wowy_key = (pair_key[0], pair_key[1], season)
                pf = self.wowy_lookup.get(wowy_key, {})

                edge_feat[vi, vj, 2] = np.log1p(pf.get("shared_minutes", 0.0))
                yt_data = self.years_together_lookup.get(pair_key, {})
                years = yt_data.get(season, 0)
                edge_feat[vi, vj, 3] = min(years, 5) / 5.0
                edge_feat[vi, vj, 4] = pf.get("net_rtg_together", 0.0) / 10.0

                norm_ai = np.linalg.norm(arch_i)
                norm_aj = np.linalg.norm(arch_j)
                if norm_ai > 1e-8 and norm_aj > 1e-8:
                    edge_feat[vi, vj, 5] = float(
                        np.dot(arch_i, arch_j) / (norm_ai * norm_aj)
                    )

                min_tog = pf.get("shared_minutes", 0.0)
                if pid_i < pid_j:
                    min_solo = pf.get("minutes_a_only", 0.0)
                else:
                    min_solo = pf.get("minutes_b_only", 0.0)
                total_min_i = min_tog + min_solo
                if total_min_i > 0:
                    edge_feat[vi, vj, 6] = min_tog / total_min_i

                norm_i = np.linalg.norm(ab_i)
                norm_j = np.linalg.norm(ab_j)
                if norm_i > 1e-8 and norm_j > 1e-8:
                    edge_feat[vi, vj, 7] = float(np.dot(ab_i, ab_j) / (norm_i * norm_j))

                by_i = self.birth_years.get(pid_i)
                by_j = self.birth_years.get(pid_j)
                if by_i is not None and by_j is not None:
                    edge_feat[vi, vj, 8] = abs(by_i - by_j) / 10.0

        return edge_feat

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        game_id = str(self.game_ids_arr[i])
        season = str(self.seasons[i])
        max_roster = self.player_abilities.shape[2]

        # Collect L1 windows for each player
        l1_windows = []  # List of per-player window dicts (for both teams)

        # For L2 edge features, use pre-computed abilities as fallback
        home_abilities_for_edges = self.player_abilities[i, 0].copy()
        away_abilities_for_edges = self.player_abilities[i, 1].copy()

        for side in range(2):
            for slot in range(max_roster):
                if not self.player_masks[i, side, slot]:
                    l1_windows.append(None)
                    continue

                pid = int(self.player_ids_arr[i, side, slot])
                window = self.l1_raw_cache.get_window(pid, game_id, self.l1_window_size)
                l1_windows.append(window)

        # Team features / context / targets
        home_tf = self._normalize(self.team_features[i, 0], self.l3_mean, self.l3_std)
        away_tf = self._normalize(self.team_features[i, 1], self.l3_mean, self.l3_std)
        home_rs = self._normalize(
            self.roster_summaries[i, 0], self.rs_mean, self.rs_std
        )
        away_rs = self._normalize(
            self.roster_summaries[i, 1], self.rs_mean, self.rs_std
        )
        gc = self._normalize(self.game_context[i], self.l4_mean, self.l4_std)

        if self.coach_indices is not None:
            home_coach_idx = int(self.coach_indices[i, 0])
            away_coach_idx = int(self.coach_indices[i, 1])
        else:
            home_coach_idx = 0
            away_coach_idx = 0

        if self.coach_games is not None:
            home_coach_games = float(self.coach_games[i, 0])
            away_coach_games = float(self.coach_games[i, 1])
        else:
            home_coach_games = 0.0
            away_coach_games = 0.0

        home_continuity = self.team_features[i, 0, 32]
        away_continuity = self.team_features[i, 1, 32]

        # Edge features (use pre-computed abilities for edges)
        home_pids = self.player_ids_arr[i, 0]
        away_pids = self.player_ids_arr[i, 1]
        home_mask = self.player_masks[i, 0]
        away_mask = self.player_masks[i, 1]
        home_arch = self.player_archetypes[i, 0]
        away_arch = self.player_archetypes[i, 1]

        home_ef = self._build_edge_features(
            home_pids, home_mask, home_abilities_for_edges, home_arch, season
        )
        away_ef = self._build_edge_features(
            away_pids, away_mask, away_abilities_for_edges, away_arch, season
        )

        return {
            # L1 windows (list of dicts, handled in custom collate)
            "l1_windows": l1_windows,
            # Pre-computed fallback L1 vectors
            "home_abilities": torch.tensor(
                self.player_abilities[i, 0], dtype=torch.float32
            ),
            "away_abilities": torch.tensor(
                self.player_abilities[i, 1], dtype=torch.float32
            ),
            "home_uncertainties": torch.tensor(
                self.player_uncertainties[i, 0], dtype=torch.float32
            ),
            "away_uncertainties": torch.tensor(
                self.player_uncertainties[i, 1], dtype=torch.float32
            ),
            "home_archetypes": torch.tensor(
                self.player_archetypes[i, 0], dtype=torch.float32
            ),
            "away_archetypes": torch.tensor(
                self.player_archetypes[i, 1], dtype=torch.float32
            ),
            "home_player_mask": torch.tensor(home_mask, dtype=torch.bool),
            "away_player_mask": torch.tensor(away_mask, dtype=torch.bool),
            "home_player_idx": torch.tensor(
                self.player_indices[i, 0], dtype=torch.long
            ),
            "away_player_idx": torch.tensor(
                self.player_indices[i, 1], dtype=torch.long
            ),
            "home_edge_features": torch.tensor(home_ef, dtype=torch.float32),
            "away_edge_features": torch.tensor(away_ef, dtype=torch.float32),
            # L3 inputs
            "home_team_features": torch.tensor(home_tf, dtype=torch.float32),
            "away_team_features": torch.tensor(away_tf, dtype=torch.float32),
            "home_roster_summary": torch.tensor(home_rs, dtype=torch.float32),
            "away_roster_summary": torch.tensor(away_rs, dtype=torch.float32),
            "home_coach_idx": torch.tensor(home_coach_idx, dtype=torch.long),
            "away_coach_idx": torch.tensor(away_coach_idx, dtype=torch.long),
            "home_coach_games": torch.tensor(home_coach_games, dtype=torch.float32),
            "away_coach_games": torch.tensor(away_coach_games, dtype=torch.float32),
            "home_continuity": torch.tensor([home_continuity], dtype=torch.float32),
            "away_continuity": torch.tensor([away_continuity], dtype=torch.float32),
            # L4 inputs
            "game_context": torch.tensor(gc, dtype=torch.float32),
            # Targets
            "actual_margin": torch.tensor(self.targets_margin[i], dtype=torch.float32),
            "actual_home_win": torch.tensor(self.targets_win[i], dtype=torch.float32),
            "actual_total": torch.tensor(self.targets_total[i], dtype=torch.float32),
        }


def custom_collate(batch: list[dict]) -> dict:
    """Custom collate that handles the l1_windows list-of-dicts.

    For tensor fields, uses default stacking. For l1_windows, keeps as
    list-of-lists for per-player L1 forward in the training loop.
    """
    result = {}
    # Extract l1_windows separately
    l1_windows_batch = [sample.pop("l1_windows") for sample in batch]
    result["l1_windows"] = l1_windows_batch  # list of list[dict|None]

    # Stack all tensor fields
    keys = batch[0].keys()
    for key in keys:
        vals = [sample[key] for sample in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        else:
            result[key] = vals

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_phase_c_full_data(max_roster: int = 15) -> dict:
    """Build Phase C data including per-player L1 raw cache.

    This extends the Phase C data builder to also load all L1 raw data.
    """
    # Load Phase B cache
    logger.info("Loading Phase B cache...")
    game_ids = np.load(str(PHASE_B_CACHE / "game_ids.npy"), allow_pickle=True)
    seasons = np.load(str(PHASE_B_CACHE / "seasons.npy"), allow_pickle=True)
    tf_data = np.load(str(PHASE_B_CACHE / "team_features.npz"))
    gc_data = np.load(str(PHASE_B_CACHE / "game_context.npz"))
    rs_data = np.load(str(PHASE_B_CACHE / "roster_summaries.npz"))
    tgt_data = np.load(str(PHASE_B_CACHE / "targets.npz"))

    coach_indices_path = PHASE_B_CACHE / "coach_indices.npy"
    coach_games_path = PHASE_B_CACHE / "coach_games.npy"
    if coach_indices_path.exists() and coach_games_path.exists():
        coach_indices_all = np.load(str(coach_indices_path))
        coach_games_all = np.load(str(coach_games_path))
    else:
        coach_indices_all = None
        coach_games_all = None

    with open(str(PHASE_B_CACHE / "metadata.json")) as f:
        metadata = json.load(f)

    n_games = len(game_ids)
    logger.info(f"Phase B cache: {n_games} games")

    # Load L1 pre-computed vectors
    l1_vectors = load_l1_vectors()
    player_to_idx = build_player_to_idx(l1_vectors)

    # Load L1 raw cache
    l1_metadata = load_metadata()
    normalizer = Normalizer(l1_metadata)
    l1_raw_cache = L1RawCache(normalizer, l1_metadata)

    # Load rosters from database
    logger.info("Loading rosters from database...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    games_info = {}
    rows = conn.execute("""
        SELECT g.game_id, g.home_team, g.away_team
        FROM Games g
        WHERE g.status = 3
    """).fetchall()
    for gid, home, away in rows:
        games_info[gid] = {"home_team": home, "away_team": away}

    roster_query = conn.execute("""
        SELECT pb.game_id, pb.player_id, pb.team_id, pb.min
        FROM PlayerBox pb
        WHERE pb.min > 0
        ORDER BY pb.game_id, pb.team_id, pb.min DESC
    """).fetchall()

    game_rosters: dict[str, dict] = {}
    for gid, pid, tid, minutes in roster_query:
        if gid not in game_rosters:
            game_rosters[gid] = {}
        if tid not in game_rosters[gid]:
            game_rosters[gid][tid] = []
        game_rosters[gid][tid].append(pid)

    team_abbr_to_id = {}
    for tid, abbr in conn.execute("SELECT team_id, abbreviation FROM Teams").fetchall():
        team_abbr_to_id[abbr] = tid

    conn.close()
    logger.info(f"Loaded rosters for {len(game_rosters)} games")

    # Pre-allocate per-player arrays
    n_arch = 10
    d_ability = 32

    player_abilities = np.zeros((n_games, 2, max_roster, d_ability), dtype=np.float32)
    player_uncertainties = np.zeros(
        (n_games, 2, max_roster, d_ability), dtype=np.float32
    )
    player_archetypes = np.zeros((n_games, 2, max_roster, n_arch), dtype=np.float32)
    player_masks = np.zeros((n_games, 2, max_roster), dtype=bool)
    player_indices = np.zeros((n_games, 2, max_roster), dtype=np.int64)
    player_id_arr = np.zeros((n_games, 2, max_roster), dtype=np.int64)
    valid_mask = np.ones(n_games, dtype=bool)

    logger.info("Building per-player arrays...")
    t0 = time.time()
    skipped = 0

    for gi in range(n_games):
        if gi % 5000 == 0 and gi > 0:
            elapsed = time.time() - t0
            rate = gi / elapsed
            eta = (n_games - gi) / max(rate, 1e-6)
            logger.info(
                f"  [{gi}/{n_games}] {rate:.0f} games/s, "
                f"ETA {eta/60:.1f} min, skipped {skipped}"
            )

        game_id = game_ids[gi]
        if game_id not in games_info:
            valid_mask[gi] = False
            skipped += 1
            continue

        info = games_info[game_id]
        home_team = info["home_team"]
        away_team = info["away_team"]

        if game_id not in game_rosters:
            valid_mask[gi] = False
            skipped += 1
            continue

        roster = game_rosters[game_id]
        home_tid = team_abbr_to_id.get(home_team)
        away_tid = team_abbr_to_id.get(away_team)

        if home_tid is None or away_tid is None:
            valid_mask[gi] = False
            skipped += 1
            continue

        teams = [(0, roster.get(home_tid, [])), (1, roster.get(away_tid, []))]
        game_valid = True

        for side, pids in teams:
            pids = pids[:max_roster]
            filled = 0
            for pid in pids:
                if pid not in l1_vectors:
                    continue
                ab, unc, arch = get_l1_at_game(l1_vectors[pid], game_id)
                if ab is None:
                    ab, unc, arch = get_l1_at_game(
                        l1_vectors[pid], game_id, use_latest=True
                    )
                    if ab is None:
                        continue

                player_abilities[gi, side, filled] = ab
                player_uncertainties[gi, side, filled] = unc
                player_archetypes[gi, side, filled] = arch
                player_masks[gi, side, filled] = True
                player_indices[gi, side, filled] = player_to_idx.get(pid, 0)
                player_id_arr[gi, side, filled] = pid
                filled += 1

            if filled == 0:
                game_valid = False
                break

        if not game_valid:
            valid_mask[gi] = False
            skipped += 1

    elapsed = time.time() - t0
    n_valid = valid_mask.sum()
    logger.info(
        f"Built player arrays in {elapsed:.1f}s: "
        f"{n_valid}/{n_games} valid, {skipped} skipped"
    )

    valid_idx = np.where(valid_mask)[0]
    norm = metadata["normalization"]

    return {
        "player_abilities": player_abilities[valid_idx],
        "player_uncertainties": player_uncertainties[valid_idx],
        "player_archetypes": player_archetypes[valid_idx],
        "player_masks": player_masks[valid_idx],
        "player_indices": player_indices[valid_idx],
        "player_ids": player_id_arr[valid_idx],
        "team_features": tf_data["features"][valid_idx],
        "game_context": gc_data["features"][valid_idx],
        "roster_summaries": rs_data["summaries"][valid_idx],
        "targets_margin": tgt_data["margin"][valid_idx],
        "targets_win": tgt_data["home_win"][valid_idx],
        "targets_total": tgt_data["total"][valid_idx],
        "coach_indices": (
            coach_indices_all[valid_idx] if coach_indices_all is not None else None
        ),
        "coach_games": (
            coach_games_all[valid_idx] if coach_games_all is not None else None
        ),
        "seasons": seasons[valid_idx],
        "game_ids": game_ids[valid_idx],
        "n_players_for_embed": len(player_to_idx) + 1,
        "l1_raw_cache": l1_raw_cache,
        # Normalization
        "l3_mean": np.array(norm["l3_mean"], dtype=np.float32),
        "l3_std": np.array(norm["l3_std"], dtype=np.float32),
        "l4_mean": np.array(norm["l4_mean"], dtype=np.float32),
        "l4_std": np.array(norm["l4_std"], dtype=np.float32),
        "rs_mean": np.array(norm["roster_summary_mean"], dtype=np.float32),
        "rs_std": np.array(norm["roster_summary_std"], dtype=np.float32),
    }


def make_train_val_datasets(
    data: dict,
    l1_window_size: int = 20,
    train_seasons: tuple[str, str] = ("2017-2018", "2022-2023"),
    val_season: str = "2023-2024",
) -> tuple[FullPhaseCDataset, FullPhaseCDataset]:
    """Split data by season and create train/val datasets."""
    seasons = data["seasons"]
    train_start = season_start_year(train_seasons[0])
    train_end = season_start_year(train_seasons[1])
    val_year = season_start_year(val_season)

    train_indices = []
    val_indices = []

    for i, s in enumerate(seasons):
        if not s:
            continue
        try:
            year = season_start_year(s)
        except (ValueError, IndexError):
            continue
        if train_start <= year <= train_end:
            train_indices.append(i)
        elif year == val_year:
            val_indices.append(i)

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)

    logger.info(
        f"Train: {len(train_indices)} games "
        f"({train_seasons[0]} - {train_seasons[1]})"
    )
    logger.info(f"Val: {len(val_indices)} games ({val_season})")

    # Compute normalization from training set
    train_tf = data["team_features"][train_indices]
    train_gc = data["game_context"][train_indices]
    train_rs = data["roster_summaries"][train_indices]

    tf_flat = train_tf.reshape(-1, 34)
    l3_mean = tf_flat.mean(axis=0).astype(np.float32)
    l3_std = tf_flat.std(axis=0).astype(np.float32)
    l3_std[l3_std < 1e-8] = 1e-8

    l4_mean = train_gc.mean(axis=0).astype(np.float32)
    l4_std = train_gc.std(axis=0).astype(np.float32)
    l4_std[l4_std < 1e-8] = 1e-8

    rs_flat = train_rs.reshape(-1, 12)
    rs_mean = rs_flat.mean(axis=0).astype(np.float32)
    rs_std = rs_flat.std(axis=0).astype(np.float32)
    rs_std[rs_std < 1e-8] = 1e-8

    # Edge feature lookups
    logger.info("Loading edge feature lookups...")
    wowy_lookup = load_wowy_lookup()
    years_together_lookup = build_years_together_lookup(wowy_lookup)
    birth_years = load_birth_years()

    common_kwargs = dict(
        player_ids_arr=data["player_ids"],
        player_masks=data["player_masks"],
        player_indices=data["player_indices"],
        player_abilities=data["player_abilities"],
        player_uncertainties=data["player_uncertainties"],
        player_archetypes=data["player_archetypes"],
        l1_raw_cache=data["l1_raw_cache"],
        l1_window_size=l1_window_size,
        game_ids_arr=data["game_ids"],
        seasons=data["seasons"],
        team_features=data["team_features"],
        game_context=data["game_context"],
        roster_summaries=data["roster_summaries"],
        targets_margin=data["targets_margin"],
        targets_win=data["targets_win"],
        targets_total=data["targets_total"],
        coach_indices=data.get("coach_indices"),
        coach_games=data.get("coach_games"),
        wowy_lookup=wowy_lookup,
        years_together_lookup=years_together_lookup,
        birth_years=birth_years,
        l3_mean=l3_mean,
        l3_std=l3_std,
        l4_mean=l4_mean,
        l4_std=l4_std,
        rs_mean=rs_mean,
        rs_std=rs_std,
    )

    train_ds = FullPhaseCDataset(indices=train_indices, **common_kwargs)
    val_ds = FullPhaseCDataset(indices=val_indices, **common_kwargs)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# L1 forward pass: run Kalman filter over player windows
# ---------------------------------------------------------------------------


def run_l1_forward_for_batch(
    l1_model: NKEH,
    batch: dict,
    device: torch.device,
    l1_trainable: bool = False,
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run L1 forward_sequence for all players in the batch, producing
    (B, 2, max_roster, 32) ability tensors with gradients when l1_trainable=True.

    For players without L1 windows (insufficient history), falls back to the
    pre-computed ability vectors (detached, no gradient).

    Strategy: flatten all player windows across the batch into groups by
    sequence length, batch them together for efficient GPU forward passes,
    then scatter results back.

    Args:
        l1_model: NKEH model (may or may not have requires_grad)
        batch: collated batch dict
        device: torch device
        l1_trainable: if True, L1 output has grad_fn; if False, detach
        use_amp: whether to use automatic mixed precision

    Returns:
        abilities: (B, 2, A, 32) with grad_fn when l1_trainable
        archetypes: (B, 2, A, 10) archetype weights (always detached)
    """
    B = batch["home_abilities"].shape[0]
    A = batch["home_abilities"].shape[1]  # max_roster

    # Start with fallback pre-computed vectors
    home_ab = batch["home_abilities"].to(device)  # (B, A, 32)
    away_ab = batch["away_abilities"].to(device)
    # Archetype weights (static, from pre-computed)
    home_arch = batch["home_archetypes"].to(device)
    away_arch = batch["away_archetypes"].to(device)

    if not l1_trainable:
        # No L1 forward needed -- use pre-computed vectors
        abilities = torch.stack([home_ab, away_ab], dim=1)  # (B, 2, A, 32)
        archetypes = torch.stack([home_arch, away_arch], dim=1)  # (B, 2, A, 10)
        return abilities, archetypes

    # Collect all valid L1 windows and their positions
    l1_windows_batch = batch["l1_windows"]  # list of B lists of 2*A items

    # Group windows by sequence length for batched forward
    # Key: seq_len -> list of (batch_idx, side, slot, window_dict)
    length_groups: dict[int, list] = {}

    for bi in range(B):
        windows = l1_windows_batch[bi]
        for idx, window in enumerate(windows):
            if window is None:
                continue
            side = idx // A
            slot = idx % A
            T = window["seq_len"]
            if T not in length_groups:
                length_groups[T] = []
            length_groups[T].append((bi, side, slot, window))

    # Build output tensor with pre-computed as default (detached)
    # We need to make this differentiable where L1 windows exist
    abilities = torch.stack([home_ab, away_ab], dim=1).clone()  # (B, 2, A, 32)
    archetypes = torch.stack([home_arch, away_arch], dim=1)

    # Process each length group as a batched L1 forward
    for seq_len, items in length_groups.items():
        n_items = len(items)
        if n_items == 0:
            continue

        # Stack into batched tensors
        box_batch = torch.zeros(n_items, seq_len, 16, device=device)
        pbp_batch = torch.zeros(n_items, seq_len, 56, device=device)
        ctx_batch = torch.zeros(n_items, seq_len, 12, device=device)
        age_batch = torch.zeros(n_items, seq_len, 1, device=device)
        gap_batch = torch.zeros(n_items, seq_len, 1, device=device)
        prof_batch = torch.zeros(n_items, 9, device=device)
        init_ctx_batch = torch.zeros(n_items, 12, device=device)
        mask_batch = torch.ones(n_items, seq_len, dtype=torch.bool, device=device)

        for j, (bi, side, slot, window) in enumerate(items):
            T = window["seq_len"]
            box_batch[j, :T] = window["box_stats"].to(device)
            pbp_batch[j, :T] = window["pbp_stats"].to(device)
            ctx_batch[j, :T] = window["context"].to(device)
            age_batch[j, :T] = window["age"].to(device)
            gap_batch[j, :T] = window["days_gap"].to(device)
            prof_batch[j] = window["profile"].to(device)
            init_ctx_batch[j] = window["init_context"].to(device)

        # Run L1 forward_sequence
        if use_amp:
            with torch.amp.autocast("cuda"):
                l1_out = l1_model.forward_sequence(
                    box_stats_seq=box_batch,
                    pbp_stats_seq=pbp_batch,
                    context_seq=ctx_batch,
                    profile=prof_batch,
                    age_seq=age_batch,
                    seq_mask=mask_batch,
                    days_gap_seq=gap_batch,
                    init_context=init_ctx_batch,
                )
        else:
            l1_out = l1_model.forward_sequence(
                box_stats_seq=box_batch,
                pbp_stats_seq=pbp_batch,
                context_seq=ctx_batch,
                profile=prof_batch,
                age_seq=age_batch,
                seq_mask=mask_batch,
                days_gap_seq=gap_batch,
                init_context=init_ctx_batch,
            )

        # Extract final ability vector (last valid timestep)
        final_abilities = l1_out["ability"][:, -1, :]  # (n_items, 32)

        # Scatter back into abilities tensor
        for j, (bi, side, slot, _) in enumerate(items):
            abilities[bi, side, slot] = final_abilities[j]

    return abilities, archetypes


# ---------------------------------------------------------------------------
# Forward pass: L1 (live) -> L2 -> L3 -> L4
# ---------------------------------------------------------------------------


def forward_l1_l2_l3_l4(
    batch: dict,
    l1_model: NKEH,
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    device: torch.device,
    l1_trainable: bool = False,
    use_amp: bool = False,
) -> dict[str, torch.Tensor]:
    """Full L1->L2->L3->L4 forward pass.

    When l1_trainable=True, L1 forward_sequence runs live and gradients flow
    back through the Kalman filter to L1 parameters.
    """
    # Run L1 forward (live when trainable)
    abilities, archetypes = run_l1_forward_for_batch(
        l1_model, batch, device, l1_trainable=l1_trainable, use_amp=use_amp
    )

    # Unpack abilities for L2
    home_ab = abilities[:, 0]  # (B, A, 32)
    away_ab = abilities[:, 1]
    home_arch = archetypes[:, 0]
    away_arch = archetypes[:, 1]

    home_unc = batch["home_uncertainties"].to(device)
    away_unc = batch["away_uncertainties"].to(device)
    home_mask = batch["home_player_mask"].to(device)
    away_mask = batch["away_player_mask"].to(device)
    home_pidx = batch["home_player_idx"].to(device)
    away_pidx = batch["away_player_idx"].to(device)
    home_ef = batch["home_edge_features"].to(device)
    away_ef = batch["away_edge_features"].to(device)

    home_tf = batch["home_team_features"].to(device)
    away_tf = batch["away_team_features"].to(device)
    home_rs = batch["home_roster_summary"].to(device)
    away_rs = batch["away_roster_summary"].to(device)
    home_coach = batch["home_coach_idx"].to(device)
    away_coach = batch["away_coach_idx"].to(device)
    home_coach_games = batch["home_coach_games"].to(device)
    away_coach_games = batch["away_coach_games"].to(device)
    home_cont = batch["home_continuity"].to(device)
    away_cont = batch["away_continuity"].to(device)
    game_ctx = batch["game_context"].to(device)

    # --- L2 forward ---
    home_l2_out = l2_model(
        ability=home_ab,
        uncertainty=home_unc,
        archetypes=home_arch,
        mask=home_mask,
        player_idx=home_pidx,
        edge_features=home_ef,
    )
    away_l2_out = l2_model(
        ability=away_ab,
        uncertainty=away_unc,
        archetypes=away_arch,
        mask=away_mask,
        player_idx=away_pidx,
        edge_features=away_ef,
    )

    home_l2 = home_l2_out["team_vector"]
    away_l2 = away_l2_out["team_vector"]

    # --- L3 forward ---
    home_repr = l3_model(
        l2_team=home_l2,
        team_features=home_tf,
        roster_summary=home_rs,
        coach_idx=home_coach,
        roster_continuity=home_cont,
        coach_games=home_coach_games,
    )
    away_repr = l3_model(
        l2_team=away_l2,
        team_features=away_tf,
        roster_summary=away_rs,
        coach_idx=away_coach,
        roster_continuity=away_cont,
        coach_games=away_coach_games,
    )

    # --- L4 forward ---
    predictions = l4_model(
        team_home=home_repr,
        team_away=away_repr,
        l2_home=home_l2,
        l2_away=away_l2,
        context=game_ctx,
    )

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    all_preds: dict[str, list],
    all_targets: dict[str, list],
) -> dict[str, float]:
    """Compute evaluation metrics from accumulated predictions/targets."""
    spread_mu = np.concatenate(all_preds["spread_mu"])
    spread_sigma = np.concatenate(all_preds["spread_sigma"])
    win_prob = np.concatenate(all_preds["win_prob"])
    total_mu = np.concatenate(all_preds["total_mu"])

    actual_margin = np.concatenate(all_targets["margin"])
    actual_win = np.concatenate(all_targets["win"])
    actual_total = np.concatenate(all_targets["total"])

    n = len(actual_margin)

    spread_mae = np.abs(spread_mu - actual_margin).mean()
    total_mae = np.abs(total_mu - actual_total).mean()

    win_pred = (win_prob > 0.5).astype(float)
    win_acc = (win_pred == actual_win).mean()

    # AUC
    order = np.argsort(win_prob)
    sorted_labels = actual_win[order]
    n_pos = int(actual_win.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        cum_pos = np.cumsum(sorted_labels)
        auc = 1.0 - cum_pos[sorted_labels == 0].sum() / (n_pos * n_neg)
    else:
        auc = 0.5

    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for j in range(n_bins):
        mask = (win_prob >= bin_boundaries[j]) & (win_prob < bin_boundaries[j + 1])
        if j == n_bins - 1:
            mask = mask | (win_prob == bin_boundaries[j + 1])
        if mask.sum() > 0:
            avg_confidence = win_prob[mask].mean()
            avg_accuracy = actual_win[mask].mean()
            ece += mask.sum() / n * abs(avg_confidence - avg_accuracy)

    mean_sigma = spread_sigma.mean()

    return {
        "spread_mae": float(spread_mae),
        "total_mae": float(total_mae),
        "win_acc": float(win_acc),
        "win_auc": float(auc),
        "ece": float(ece),
        "mean_sigma": float(mean_sigma),
        "n_games": n,
    }


def compute_ats(
    spread_mu: np.ndarray,
    actual_margin: np.ndarray,
    vegas_spreads: dict[str, float],
    game_ids: np.ndarray,
) -> dict[str, float]:
    """Compute ATS (against the spread) metrics.

    ATS win: model predicts home margin > actual Vegas spread
    (i.e., model thinks the line is too generous to the away team).
    """
    correct = 0
    total = 0

    for i, gid in enumerate(game_ids):
        gid_str = str(gid)
        if gid_str not in vegas_spreads:
            continue

        vegas_spread = vegas_spreads[gid_str]
        pred_margin = float(spread_mu[i])
        actual = float(actual_margin[i])

        # Model's implied line: pred_margin. Vegas line: vegas_spread (neg=home fav)
        # Home covers if actual_margin > -vegas_spread (since spread is neg for fav)
        # Model predicts cover if pred_margin > -vegas_spread
        home_covers_actual = actual > -vegas_spread
        model_pred_covers = pred_margin > -vegas_spread

        if model_pred_covers == home_covers_actual:
            correct += 1
        total += 1

    if total == 0:
        return {"ats_pct": 0.0, "ats_correct": 0, "ats_total": 0}

    return {
        "ats_pct": correct / total,
        "ats_correct": correct,
        "ats_total": total,
    }


# ---------------------------------------------------------------------------
# Gradual unfreezing (extended to include L1)
# ---------------------------------------------------------------------------


def apply_freeze_schedule(
    epoch: int,
    l1_model: NKEH,
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    l4_only_epochs: int = 3,
    l3l4_epochs: int = 6,
    l2l3l4_epochs: int = 10,
) -> tuple[str, bool]:
    """Apply gradual unfreezing schedule including L1.

    Returns (schedule_name, l1_trainable).
    l1_trainable indicates whether we need live L1 forward (expensive).
    """
    if epoch <= l4_only_epochs:
        for p in l1_model.parameters():
            p.requires_grad = False
        for p in l2_model.parameters():
            p.requires_grad = False
        for p in l3_model.parameters():
            p.requires_grad = False
        for p in l4_model.parameters():
            p.requires_grad = True
        return "L4 only", False

    elif epoch <= l3l4_epochs:
        for p in l1_model.parameters():
            p.requires_grad = False
        for p in l2_model.parameters():
            p.requires_grad = False
        for p in l3_model.parameters():
            p.requires_grad = True
        for p in l4_model.parameters():
            p.requires_grad = True
        return "L3+L4", False

    elif epoch <= l2l3l4_epochs:
        for p in l1_model.parameters():
            p.requires_grad = False
        for p in l2_model.parameters():
            p.requires_grad = True
        for p in l3_model.parameters():
            p.requires_grad = True
        for p in l4_model.parameters():
            p.requires_grad = True
        return "L2+L3+L4", False

    else:
        # Full stack including L1
        for p in l1_model.parameters():
            p.requires_grad = True
        for p in l2_model.parameters():
            p.requires_grad = True
        for p in l3_model.parameters():
            p.requires_grad = True
        for p in l4_model.parameters():
            p.requires_grad = True
        return "L1+L2+L3+L4", True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    l1_model: NKEH,
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l4_cfg: L4Config,
    device: torch.device,
    l1_trainable: bool = False,
    grad_clip: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """Train one epoch. Returns dict of mean loss components."""
    l1_model.train() if l1_trainable else l1_model.eval()
    l2_model.train()
    l3_model.train()
    l4_model.train()

    total_losses = {}
    n_batches = 0

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                predictions = forward_l1_l2_l3_l4(
                    batch,
                    l1_model,
                    l2_model,
                    l3_model,
                    l4_model,
                    device,
                    l1_trainable=l1_trainable,
                    use_amp=use_amp,
                )
                losses = game_prediction_loss(
                    predictions=predictions,
                    actual_margin=actual_margin,
                    actual_home_win=actual_win,
                    actual_total=actual_total,
                    cfg=l4_cfg,
                )

            scaler.scale(losses["total_loss"]).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            all_params = (
                list(l1_model.parameters())
                + list(l2_model.parameters())
                + list(l3_model.parameters())
                + list(l4_model.parameters())
            )
            trainable_params = [p for p in all_params if p.requires_grad]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = forward_l1_l2_l3_l4(
                batch,
                l1_model,
                l2_model,
                l3_model,
                l4_model,
                device,
                l1_trainable=l1_trainable,
                use_amp=False,
            )
            losses = game_prediction_loss(
                predictions=predictions,
                actual_margin=actual_margin,
                actual_home_win=actual_win,
                actual_total=actual_total,
                cfg=l4_cfg,
            )

            losses["total_loss"].backward()

            all_params = (
                list(l1_model.parameters())
                + list(l2_model.parameters())
                + list(l3_model.parameters())
                + list(l4_model.parameters())
            )
            trainable_params = [p for p in all_params if p.requires_grad]
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(
    l1_model: NKEH,
    l2_model: PlayerSynergyNetwork,
    l3_model: TeamModel,
    l4_model: GamePredictor,
    loader: DataLoader,
    l4_cfg: L4Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate on a dataset. Returns (loss_dict, metrics_dict)."""
    l1_model.eval()
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()

    total_losses = {}
    n_batches = 0

    all_preds = {"spread_mu": [], "spread_sigma": [], "win_prob": [], "total_mu": []}
    all_targets = {"margin": [], "win": [], "total": []}

    for batch in loader:
        actual_margin = batch["actual_margin"].to(device)
        actual_win = batch["actual_home_win"].to(device)
        actual_total = batch["actual_total"].to(device)

        # Evaluation always uses pre-computed vectors (no L1 forward needed)
        predictions = forward_l1_l2_l3_l4(
            batch,
            l1_model,
            l2_model,
            l3_model,
            l4_model,
            device,
            l1_trainable=False,
            use_amp=False,
        )

        losses = game_prediction_loss(
            predictions=predictions,
            actual_margin=actual_margin,
            actual_home_win=actual_win,
            actual_total=actual_total,
            cfg=l4_cfg,
        )

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        n_batches += 1

        all_preds["spread_mu"].append(predictions["spread_mu"].cpu().numpy())
        all_preds["spread_sigma"].append(predictions["spread_sigma"].cpu().numpy())
        all_preds["win_prob"].append(predictions["win_prob"].cpu().numpy())
        all_preds["total_mu"].append(predictions["total_mu"].cpu().numpy())

        all_targets["margin"].append(actual_margin.cpu().numpy())
        all_targets["win"].append(actual_win.cpu().numpy())
        all_targets["total"].append(actual_total.cpu().numpy())

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    metrics = compute_metrics(all_preds, all_targets)

    return avg_losses, metrics


# ---------------------------------------------------------------------------
# Vegas spread loading
# ---------------------------------------------------------------------------


def load_vegas_spreads() -> dict[str, float]:
    """Load Vegas closing spreads from Betting table."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute("""
        SELECT game_id,
               COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread)
        FROM Betting
        WHERE COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread)
              IS NOT NULL
    """).fetchall()
    conn.close()
    return {str(gid): float(spread) for gid, spread in rows}


# ---------------------------------------------------------------------------
# L1 drift monitoring (cosine similarity based)
# ---------------------------------------------------------------------------


class L1DriftMonitor:
    """Tracks drift in L1 ability vectors for a sample of test players.

    Measures cosine similarity between original pre-computed vectors and
    vectors produced by the current (potentially modified) L1 model.
    """

    def __init__(self, n_sample: int = 100):
        self.n_sample = n_sample
        self.reference_vectors = None
        self.sample_items = None

    def setup(self, l1_raw_cache: L1RawCache, l1_vectors: dict[int, dict]):
        """Select sample players and store their reference ability vectors."""
        # Pick players with sufficient history
        candidates = []
        for pid, data in l1_raw_cache.player_data.items():
            if len(data["game_ids"]) >= 40 and pid in l1_vectors:
                candidates.append(pid)

        np.random.seed(42)
        sample_pids = np.random.choice(
            candidates, size=min(self.n_sample, len(candidates)), replace=False
        )

        self.sample_items = []
        refs = []
        for pid in sample_pids:
            l1_data = l1_vectors[pid]
            # Use the last game as reference point
            game_id = l1_data["game_ids"][-1]
            ability = l1_data["ability"][-1]  # (32,) pre-computed reference
            refs.append(ability)
            self.sample_items.append((int(pid), game_id))

        self.reference_vectors = torch.tensor(np.array(refs), dtype=torch.float32)

    @torch.no_grad()
    def compute_drift(
        self,
        l1_model: NKEH,
        l1_raw_cache: L1RawCache,
        device: torch.device,
        window_size: int = 20,
    ) -> float:
        """Compute mean cosine similarity between reference and current L1 vectors.

        Returns 1.0 - cosine_sim (so 0 = no drift, 1 = maximal drift).
        """
        if self.sample_items is None:
            return 0.0

        l1_model.eval()
        current_vectors = []

        for pid, game_id in self.sample_items:
            window = l1_raw_cache.get_window(pid, game_id, window_size)
            if window is None:
                current_vectors.append(torch.zeros(32))
                continue

            T = window["seq_len"]
            box = window["box_stats"].unsqueeze(0).to(device)  # (1, T, 16)
            pbp = window["pbp_stats"].unsqueeze(0).to(device)
            ctx = window["context"].unsqueeze(0).to(device)
            age = window["age"].unsqueeze(0).to(device)
            gap = window["days_gap"].unsqueeze(0).to(device)
            prof = window["profile"].unsqueeze(0).to(device)
            init_ctx = window["init_context"].unsqueeze(0).to(device)
            mask = torch.ones(1, T, dtype=torch.bool, device=device)

            out = l1_model.forward_sequence(
                box_stats_seq=box,
                pbp_stats_seq=pbp,
                context_seq=ctx,
                profile=prof,
                age_seq=age,
                seq_mask=mask,
                days_gap_seq=gap,
                init_context=init_ctx,
            )
            current_vectors.append(out["ability"][0, -1].cpu())

        current = torch.stack(current_vectors)
        ref = self.reference_vectors

        # Cosine similarity
        cos_sim = F.cosine_similarity(current, ref, dim=-1).mean().item()
        return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase C Full: End-to-end L1+L2+L3+L4 fine-tuning"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--l1-lr", type=float, default=1e-5)
    parser.add_argument("--l2-lr", type=float, default=3e-5)
    parser.add_argument("--l3-lr", type=float, default=1e-4)
    parser.add_argument("--l4-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--l1-window", type=int, default=20)
    parser.add_argument(
        "--l4-only-epochs",
        type=int,
        default=2,
        help="Epochs with only L4 trainable",
    )
    parser.add_argument(
        "--l3l4-epochs",
        type=int,
        default=4,
        help="Epochs with L3+L4 trainable",
    )
    parser.add_argument(
        "--l2l3l4-epochs",
        type=int,
        default=7,
        help="Epochs with L2+L3+L4 trainable (after this, full L1+L2+L3+L4)",
    )
    parser.add_argument("--test", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2017-2018,2022-2023",
        help="Start,end seasons for training",
    )
    parser.add_argument("--val-season", type=str, default="2023-2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--l1-drift-threshold",
        type=float,
        default=0.05,
        help="L1 drift threshold for LR halving (cosine distance)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    logger.info(f"Device: {device}, AMP: {use_amp}")

    # --- Load and build data ---
    data = build_phase_c_full_data()

    train_start, train_end = args.train_seasons.split(",")
    train_ds, val_ds = make_train_val_datasets(
        data,
        l1_window_size=args.l1_window,
        train_seasons=(train_start, train_end),
        val_season=args.val_season,
    )

    logger.info(f"Train dataset: {len(train_ds)} games")
    logger.info(f"Val dataset: {len(val_ds)} games")

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset! Check season ranges.")
        sys.exit(1)

    # Custom collate for L1 windows
    # num_workers=0: L1 raw cache is ~500MB, too large to copy per worker
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate,
    )

    # --- Create models ---
    l1_cfg = NKEHConfig()
    l2_cfg = L2Config()
    l3_cfg = L3Config()
    l4_cfg = L4Config()

    l2_cfg.n_players = max(l2_cfg.n_players, data["n_players_for_embed"])

    l1_model = NKEH(l1_cfg).to(device)
    l2_model = PlayerSynergyNetwork(l2_cfg).to(device)
    l3_model = TeamModel(l3_cfg).to(device)
    l4_model = GamePredictor(l4_cfg).to(device)

    # --- Load pretrained weights ---
    # L1: from Phase 2 Kalman training
    l1_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase2_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l1_model.load_state_dict(l1_ckpt["model_state_dict"], strict=False)
    logger.info(
        f"Loaded L1 weights from phase2_best.pt " f"(epoch {l1_ckpt.get('epoch', '?')})"
    )

    # L2+L3+L4: from Phase C best checkpoint
    c_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_c_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l2_model.load_state_dict(c_ckpt["l2_state_dict"])
    l3_model.load_state_dict(c_ckpt["l3_state_dict"])
    l4_model.load_state_dict(c_ckpt["l4_state_dict"], strict=False)
    logger.info(
        f"Loaded L2+L3+L4 weights from phase_c_best.pt "
        f"(epoch {c_ckpt.get('epoch', '?')}, "
        f"MAE={c_ckpt['val_metrics']['spread_mae']:.3f}, "
        f"AUC={c_ckpt['val_metrics']['win_auc']:.3f})"
    )

    l1_params = l1_count_params(l1_model)
    l2_params = l2_count_params(l2_model)
    l3_params = l3_count_params(l3_model)
    l4_params = l4_count_params(l4_model)
    total = l1_params + l2_params + l3_params + l4_params
    logger.info(f"L1 NKEH: {l1_params:,} params")
    logger.info(f"L2 PlayerSynergyNetwork: {l2_params:,} params")
    logger.info(f"L3 TeamModel: {l3_params:,} params")
    logger.info(f"L4 GamePredictor: {l4_params:,} params")
    logger.info(f"Total: {total:,} params")

    # --- Drift monitors ---
    l1_drift_monitor = L1DriftMonitor(n_sample=100)
    l1_vectors = load_l1_vectors()
    l1_drift_monitor.setup(data["l1_raw_cache"], l1_vectors)
    del l1_vectors  # Free memory

    l2_drift_monitor = DriftMonitor(l2_model, name="L2")

    # --- Optimizer with discriminative learning rates ---
    optimizer = torch.optim.AdamW(
        [
            {"params": l1_model.parameters(), "lr": args.l1_lr},
            {"params": l2_model.parameters(), "lr": args.l2_lr},
            {"params": l3_model.parameters(), "lr": args.l3_lr},
            {"params": l4_model.parameters(), "lr": args.l4_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Load Vegas spreads for ATS ---
    vegas_spreads = load_vegas_spreads()
    logger.info(f"Loaded Vegas spreads for {len(vegas_spreads)} games")

    # Get val game IDs for ATS computation
    val_game_ids = data["game_ids"][val_ds.indices]

    # --- Evaluate Phase C baseline ---
    logger.info("\nEvaluating Phase C baseline (before full fine-tuning)...")
    baseline_losses, baseline_metrics = evaluate(
        l1_model, l2_model, l3_model, l4_model, val_loader, l4_cfg, device
    )

    # Compute baseline ATS
    val_preds_for_ats = {"spread_mu": []}
    val_targets_for_ats = {"margin": []}
    l1_model.eval()
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            preds = forward_l1_l2_l3_l4(
                batch,
                l1_model,
                l2_model,
                l3_model,
                l4_model,
                device,
                l1_trainable=False,
            )
            val_preds_for_ats["spread_mu"].append(preds["spread_mu"].cpu().numpy())
            val_targets_for_ats["margin"].append(batch["actual_margin"].numpy())

    baseline_spread_mu = np.concatenate(val_preds_for_ats["spread_mu"])
    baseline_actual_margin = np.concatenate(val_targets_for_ats["margin"])
    baseline_ats = compute_ats(
        baseline_spread_mu, baseline_actual_margin, vegas_spreads, val_game_ids
    )

    logger.info(
        f"Phase C baseline: MAE={baseline_metrics['spread_mae']:.3f}, "
        f"AUC={baseline_metrics['win_auc']:.3f}, "
        f"WinAcc={baseline_metrics['win_acc']:.3f}, "
        f"ECE={baseline_metrics['ece']:.4f}, "
        f"ATS={baseline_ats['ats_pct']:.1%} "
        f"({baseline_ats['ats_correct']}/{baseline_ats['ats_total']})"
    )

    # --- Training loop ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    l1_lr_halved = False

    logger.info(f"\nStarting Phase C Full training for {args.epochs} epochs...")
    logger.info(
        f"  L1 lr={args.l1_lr}, L2 lr={args.l2_lr}, "
        f"L3 lr={args.l3_lr}, L4 lr={args.l4_lr}, "
        f"wd={args.weight_decay}, batch={args.batch_size}, "
        f"L1 window={args.l1_window}, patience={args.patience}"
    )
    logger.info(
        f"  Gradual unfreezing: "
        f"L4 only (ep 1-{args.l4_only_epochs}), "
        f"L3+L4 (ep {args.l4_only_epochs+1}-{args.l3l4_epochs}), "
        f"L2+L3+L4 (ep {args.l3l4_epochs+1}-{args.l2l3l4_epochs}), "
        f"L1+L2+L3+L4 (ep {args.l2l3l4_epochs+1}+)"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Apply freeze schedule
        schedule_name, l1_trainable = apply_freeze_schedule(
            epoch,
            l1_model,
            l2_model,
            l3_model,
            l4_model,
            l4_only_epochs=args.l4_only_epochs,
            l3l4_epochs=args.l3l4_epochs,
            l2l3l4_epochs=args.l2l3l4_epochs,
        )

        n_trainable = sum(
            p.numel()
            for m in [l1_model, l2_model, l3_model, l4_model]
            for p in m.parameters()
            if p.requires_grad
        )

        # Train
        train_losses = train_epoch(
            l1_model,
            l2_model,
            l3_model,
            l4_model,
            train_loader,
            optimizer,
            l4_cfg,
            device,
            l1_trainable=l1_trainable,
            grad_clip=args.grad_clip,
            scaler=scaler,
            use_amp=use_amp,
        )

        scheduler.step()

        # Validate
        val_losses, val_metrics = evaluate(
            l1_model, l2_model, l3_model, l4_model, val_loader, l4_cfg, device
        )

        # Drift monitoring
        l1_drift = l1_drift_monitor.compute_drift(
            l1_model, data["l1_raw_cache"], device, args.l1_window
        )
        l2_drift = l2_drift_monitor.compute_drift(l2_model)

        # L1 drift-based LR halving
        if l1_trainable and l1_drift > args.l1_drift_threshold and not l1_lr_halved:
            old_lr = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[0]["lr"] = old_lr / 2
            l1_lr_halved = True
            logger.info(
                f"  L1 drift={l1_drift:.4f} > threshold={args.l1_drift_threshold}, "
                f"halving L1 LR: {old_lr:.1e} -> {old_lr/2:.1e}"
            )

        elapsed = time.time() - t0

        lr_l1 = optimizer.param_groups[0]["lr"]
        lr_l2 = optimizer.param_groups[1]["lr"]
        lr_l3 = optimizer.param_groups[2]["lr"]
        lr_l4 = optimizer.param_groups[3]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) [{schedule_name}] "
            f"({n_trainable:,} params) | "
            f"Train loss={train_losses['total_loss']:.4f} "
            f"(spr={train_losses.get('spread', 0):.4f}, "
            f"win={train_losses.get('win', 0):.4f}) | "
            f"Val MAE={val_metrics['spread_mae']:.2f}, "
            f"AUC={val_metrics['win_auc']:.3f}, "
            f"WinAcc={val_metrics['win_acc']:.3f}, "
            f"ECE={val_metrics['ece']:.4f}, "
            f"sigma={val_metrics['mean_sigma']:.2f} | "
            f"lr={lr_l1:.1e}/{lr_l2:.1e}/{lr_l3:.1e}/{lr_l4:.1e} | "
            f"L1 drift={l1_drift:.4f}, L2 drift={l2_drift['overall']:.4f}"
        )

        # Track history
        epoch_data = {
            "epoch": epoch,
            "schedule": schedule_name,
            "n_trainable": n_trainable,
            "train_losses": {k: float(v) for k, v in train_losses.items()},
            "val_losses": {k: float(v) for k, v in val_losses.items()},
            "val_metrics": val_metrics,
            "lr_l1": lr_l1,
            "lr_l2": lr_l2,
            "lr_l3": lr_l3,
            "lr_l4": lr_l4,
            "l1_drift": l1_drift,
            "l2_drift": l2_drift["overall"],
            "time_s": elapsed,
        }
        history.append(epoch_data)

        # Checkpoint
        val_mae = val_metrics["spread_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "l1_state_dict": l1_model.state_dict(),
                    "l2_state_dict": l2_model.state_dict(),
                    "l3_state_dict": l3_model.state_dict(),
                    "l4_state_dict": l4_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "l1_config": l1_cfg.__dict__,
                    "l2_config": l2_cfg.__dict__,
                    "l3_config": l3_cfg.__dict__,
                    "l4_config": l4_cfg.__dict__,
                    "val_metrics": val_metrics,
                    "val_losses": val_losses,
                    "l1_drift": l1_drift,
                    "l2_drift": l2_drift["overall"],
                    "baseline_metrics": baseline_metrics,
                    "baseline_ats": baseline_ats,
                },
                str(CHECKPOINT_DIR / "phase_c_full_best.pt"),
            )
            logger.info(f"  ** New best: MAE={best_val_mae:.3f} at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (patience={args.patience})"
                )
                break

        if args.test and epoch >= 5:
            logger.info("Test mode: stopping after 5 epochs.")
            break

    # --- Final evaluation ---
    logger.info(f"\n{'='*70}")
    logger.info("Phase C Full Training complete!")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val spread MAE: {best_val_mae:.3f}")

    # Load best checkpoint
    best_ckpt = torch.load(
        str(CHECKPOINT_DIR / "phase_c_full_best.pt"),
        map_location=device,
        weights_only=False,
    )
    l1_model.load_state_dict(best_ckpt["l1_state_dict"])
    l2_model.load_state_dict(best_ckpt["l2_state_dict"])
    l3_model.load_state_dict(best_ckpt["l3_state_dict"])
    l4_model.load_state_dict(best_ckpt["l4_state_dict"])

    _, final_metrics = evaluate(
        l1_model, l2_model, l3_model, l4_model, val_loader, l4_cfg, device
    )

    # Compute ATS for final model
    l1_model.eval()
    l2_model.eval()
    l3_model.eval()
    l4_model.eval()
    final_preds = {"spread_mu": []}
    final_targets = {"margin": []}
    with torch.no_grad():
        for batch in val_loader:
            preds = forward_l1_l2_l3_l4(
                batch,
                l1_model,
                l2_model,
                l3_model,
                l4_model,
                device,
                l1_trainable=False,
            )
            final_preds["spread_mu"].append(preds["spread_mu"].cpu().numpy())
            final_targets["margin"].append(batch["actual_margin"].numpy())

    final_spread_mu = np.concatenate(final_preds["spread_mu"])
    final_actual_margin = np.concatenate(final_targets["margin"])
    final_ats = compute_ats(
        final_spread_mu, final_actual_margin, vegas_spreads, val_game_ids
    )

    logger.info(
        f"\nFinal validation metrics (best checkpoint, epoch {best_ckpt['epoch']}):"
    )
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info(
        f"  ATS: {final_ats['ats_pct']:.1%} "
        f"({final_ats['ats_correct']}/{final_ats['ats_total']})"
    )

    # Compare to Phase C baseline
    logger.info("\nComparison to Phase C baseline (L1 frozen):")
    for key in ["spread_mae", "win_auc", "win_acc", "ece"]:
        b_val = baseline_metrics[key]
        c_val = final_metrics[key]
        delta = c_val - b_val
        direction = (
            "better"
            if (
                (key == "spread_mae" and delta < 0)
                or (key in ("win_auc", "win_acc") and delta > 0)
                or (key == "ece" and delta < 0)
            )
            else "worse"
        )
        logger.info(f"  {key}: {b_val:.4f} -> {c_val:.4f} ({delta:+.4f}, {direction})")

    ats_delta = final_ats["ats_pct"] - baseline_ats["ats_pct"]
    logger.info(
        f"  ATS: {baseline_ats['ats_pct']:.1%} -> {final_ats['ats_pct']:.1%} "
        f"({ats_delta:+.1%})"
    )
    logger.info(f"  L1 drift at best: {best_ckpt.get('l1_drift', 'N/A')}")
    logger.info(f"  L2 drift at best: {best_ckpt.get('l2_drift', 'N/A')}")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results = {
        "args": vars(args),
        "l1_params": l1_params,
        "l2_params": l2_params,
        "l3_params": l3_params,
        "l4_params": l4_params,
        "total_params": total,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "baseline_metrics": baseline_metrics,
        "baseline_ats": baseline_ats,
        "final_metrics": final_metrics,
        "final_ats": final_ats,
        "history": make_serializable(history),
    }
    results_path = RESULTS_DIR / "phase_c_full_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "l1_state_dict": l1_model.state_dict(),
            "l2_state_dict": l2_model.state_dict(),
            "l3_state_dict": l3_model.state_dict(),
            "l4_state_dict": l4_model.state_dict(),
            "l1_config": l1_cfg.__dict__,
            "l2_config": l2_cfg.__dict__,
            "l3_config": l3_cfg.__dict__,
            "l4_config": l4_cfg.__dict__,
            "val_metrics": final_metrics,
            "final_ats": final_ats,
        },
        str(CHECKPOINT_DIR / "phase_c_full_final.pt"),
    )
    logger.info(f"Final checkpoint saved to {CHECKPOINT_DIR / 'phase_c_full_final.pt'}")


if __name__ == "__main__":
    main()
