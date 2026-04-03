#!/usr/bin/env python3
"""
Build coach embedding indices and experience counts for Phase B cache.

Reads the Coaches table and Games table to produce:
  - coach_indices.npy: shape (N, 2) -- home and away coach embedding indices
  - coach_games.npy: shape (N, 2) -- cumulative games coached (career-wide) for home/away

Coach embedding index 0 = unknown/missing coach. Real coaches get indices 1..N_coaches.

For seasons with 2 head coaches (mid-season firing), we pick the one with the
higher coach_id (typically the replacement).

Usage:
    python scripts/build_coach_cache.py
"""

import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
PHASE_B_CACHE = PROJECT_ROOT / "data" / "phase_b_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def games_season_to_coaches_season(games_season: str) -> str:
    """Convert Games season format to Coaches season format.

    Games: '2017-2018' -> Coaches: '2017-18'
    """
    parts = games_season.split("-")
    if len(parts) != 2:
        return ""
    return f"{parts[0]}-{parts[1][2:]}"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # --- Load Phase B cache game_ids and seasons ---
    game_ids = np.load(str(PHASE_B_CACHE / "game_ids.npy"), allow_pickle=True)
    seasons = np.load(str(PHASE_B_CACHE / "seasons.npy"), allow_pickle=True)
    n_games = len(game_ids)
    logger.info(f"Phase B cache: {n_games} games")

    # --- Build coach_id -> embedding index mapping (1-based, 0 = unknown) ---
    coaches_rows = conn.execute(
        "SELECT DISTINCT coach_id, coach_name FROM Coaches WHERE is_head_coach = 1 ORDER BY coach_id"
    ).fetchall()
    coach_to_idx = {str(cid): i + 1 for i, (cid, _) in enumerate(coaches_rows)}
    n_coaches = len(coach_to_idx)
    logger.info(f"Unique head coaches: {n_coaches}")

    # --- Build (team_id, coaches_season) -> coach_id mapping ---
    # For seasons with 2 head coaches, pick the one with the higher coach_id
    coach_rows = conn.execute(
        "SELECT team_id, season, coach_id, coach_name FROM Coaches WHERE is_head_coach = 1 ORDER BY team_id, season, coach_id"
    ).fetchall()

    team_season_coach: dict[tuple[str, str], str] = {}
    for team_id, c_season, coach_id, coach_name in coach_rows:
        key = (str(team_id), c_season)
        # Later entries (higher coach_id) overwrite earlier ones
        team_season_coach[key] = str(coach_id)

    logger.info(f"Team-season-coach mappings: {len(team_season_coach)}")

    # --- Build team abbreviation -> team_id mapping ---
    team_abbr_to_id = {}
    rows = conn.execute("SELECT team_id, abbreviation FROM Teams").fetchall()
    for tid, abbr in rows:
        team_abbr_to_id[abbr] = str(tid)

    # --- Load game info: game_id -> (home_team, away_team, season, date_time_utc) ---
    logger.info("Loading game info...")
    games_info = {}
    rows = conn.execute(
        "SELECT game_id, home_team, away_team, season, date_time_utc FROM Games WHERE status = 3"
    ).fetchall()
    for gid, home, away, season, dt in rows:
        games_info[gid] = {
            "home_team": home,
            "away_team": away,
            "season": season,
            "date_time_utc": dt,
        }
    logger.info(f"Loaded info for {len(games_info)} games")

    # --- Compute cumulative games coached (career-wide, chronological) ---
    # First, get all games chronologically with their coaches
    logger.info("Computing cumulative coaching experience...")

    # For each game, determine coach for each team
    # Then sort by date and compute running count per coach
    game_coach_assignments = []  # (date_time_utc, game_id, side, coach_id)

    for game_id in game_ids:
        game_id_str = str(game_id)
        if game_id_str not in games_info:
            continue

        info = games_info[game_id_str]
        c_season = games_season_to_coaches_season(info["season"])

        for side, team_abbr in enumerate([info["home_team"], info["away_team"]]):
            team_id = team_abbr_to_id.get(team_abbr)
            if team_id is None:
                continue
            coach_id = team_season_coach.get((team_id, c_season))
            if coach_id is not None:
                game_coach_assignments.append(
                    (info["date_time_utc"], game_id_str, side, coach_id)
                )

    # Sort by date
    game_coach_assignments.sort(key=lambda x: x[0])

    # Build running count per coach
    coach_game_count: dict[str, int] = defaultdict(int)
    # Map (game_id, side) -> cumulative games at that point (BEFORE this game)
    game_side_to_coach_games: dict[tuple[str, int], int] = {}
    game_side_to_coach_id: dict[tuple[str, int], str] = {}

    for dt, game_id, side, coach_id in game_coach_assignments:
        game_side_to_coach_games[(game_id, side)] = coach_game_count[coach_id]
        game_side_to_coach_id[(game_id, side)] = coach_id
        coach_game_count[coach_id] += 1

    # --- Fill output arrays ---
    coach_indices = np.zeros((n_games, 2), dtype=np.int64)
    coach_games = np.zeros((n_games, 2), dtype=np.float32)

    matched = 0
    unmatched = 0

    for i, game_id in enumerate(game_ids):
        game_id_str = str(game_id)
        for side in [0, 1]:  # 0=home, 1=away
            key = (game_id_str, side)
            c_id = game_side_to_coach_id.get(key)
            if c_id is not None:
                coach_indices[i, side] = coach_to_idx.get(c_id, 0)
                coach_games[i, side] = game_side_to_coach_games.get(key, 0)
                matched += 1
            else:
                # Unknown coach -> index 0, games = 0
                unmatched += 1

    logger.info(
        f"Coach assignments: {matched} matched, {unmatched} unmatched "
        f"({matched / (matched + unmatched) * 100:.1f}% coverage)"
    )

    # --- Save ---
    np.save(str(PHASE_B_CACHE / "coach_indices.npy"), coach_indices)
    np.save(str(PHASE_B_CACHE / "coach_games.npy"), coach_games)

    # Save coach mapping for reference
    coach_mapping = {
        "coach_to_idx": coach_to_idx,
        "n_coaches": n_coaches,
        "n_coaches_with_padding": n_coaches + 1,  # +1 for index 0 (unknown)
    }
    with open(str(PHASE_B_CACHE / "coach_mapping.json"), "w") as f:
        json.dump(coach_mapping, f, indent=2)

    # --- Update metadata.json ---
    with open(str(PHASE_B_CACHE / "metadata.json")) as f:
        metadata = json.load(f)
    metadata["n_coaches"] = n_coaches + 1  # including unknown at index 0
    metadata["coach_coverage"] = matched / (matched + unmatched)
    with open(str(PHASE_B_CACHE / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    conn.close()

    logger.info(f"\nSaved to {PHASE_B_CACHE}:")
    logger.info(
        f"  coach_indices.npy: {coach_indices.shape} (min={coach_indices.min()}, max={coach_indices.max()})"
    )
    logger.info(f"  coach_games.npy: {coach_games.shape} (max={coach_games.max():.0f})")
    logger.info(f"  coach_mapping.json: {n_coaches} coaches + 1 unknown")

    # Sanity check: distribution of coach indices
    nonzero = coach_indices[coach_indices > 0]
    logger.info(f"\nSanity check:")
    logger.info(
        f"  Games with at least one known coach: {(coach_indices.max(axis=1) > 0).sum()} / {n_games}"
    )
    logger.info(f"  Unique coaches seen: {len(np.unique(nonzero))}")
    logger.info(
        f"  Mean coach games (when known): {coach_games[coach_indices > 0].mean():.1f}"
    )
    logger.info(f"  Max coach games: {coach_games.max():.0f}")


if __name__ == "__main__":
    main()
