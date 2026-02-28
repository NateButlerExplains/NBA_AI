"""
Cache Builder for Phase 2.

Pre-extracts per-game features from the database to disk.
Run once before training to avoid per-sample DB queries.

Usage:
    python -m src.transformer.phase2.cache_builder --seasons 2018-2019 2019-2020 ...
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.database import get_db
from src.transformer.gamestates.tokenizer import GameStateTokenizer

logger = logging.getLogger(__name__)

# 30 NBA team tricodes -> integer indices
TEAM_TO_IDX = {
    "ATL": 0, "BOS": 1, "BKN": 2, "CHA": 3, "CHI": 4,
    "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9,
    "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 14,
    "MIA": 15, "MIL": 16, "MIN": 17, "NOP": 18, "NYK": 19,
    "OKC": 20, "ORL": 21, "PHI": 22, "PHX": 23, "POR": 24,
    "SAC": 25, "SAS": 26, "TOR": 27, "UTA": 28, "WAS": 29,
}


@dataclass
class PerGameFeatures:
    """Pre-extracted features for a single game."""

    game_id: str
    game_date: str
    season: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    margin: int
    total: int
    is_overtime: bool
    reg_home_score: int
    reg_away_score: int
    home_player_points: list[tuple[int, int]] = field(default_factory=list)
    away_player_points: list[tuple[int, int]] = field(default_factory=list)


def _parse_players_data(players_data_str: Optional[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Parse players_data JSON into per-side [(player_id, points), ...] lists."""
    if not players_data_str:
        return [], []

    try:
        data = json.loads(players_data_str)
    except (json.JSONDecodeError, TypeError):
        return [], []

    home_players = []
    away_players = []

    for side_key, player_list in [("home", home_players), ("away", away_players)]:
        side_data = data.get(side_key, {})
        for pid_str, pdata in side_data.items():
            try:
                pid = int(pid_str)
                if isinstance(pdata, dict):
                    points = int(pdata.get("points", 0))
                else:
                    points = 0
                player_list.append((pid, points))
            except (ValueError, TypeError):
                continue

    return home_players, away_players


def _get_regulation_scores(game_id: str, home_score: int, away_score: int,
                           db_path: Optional[str] = None) -> tuple[int, int, bool]:
    """Get regulation-end scores. Returns (reg_home, reg_away, is_overtime)."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT home_score, away_score
            FROM GameStates
            WHERE game_id = ? AND period = 4
            ORDER BY play_id DESC
            LIMIT 1
            """,
            (game_id,),
        ).fetchone()

    if row is None:
        return home_score, away_score, False

    reg_home, reg_away = row
    is_overtime = (home_score != reg_home) or (away_score != reg_away)
    return reg_home, reg_away, is_overtime


def build_cache(
    seasons: list[str],
    cache_dir: str = "data/phase2_cache",
    db_path: Optional[str] = None,
) -> dict:
    """
    Build the Phase 2 feature cache from the database.

    Outputs:
        game_features.pt - Dict[game_id -> PerGameFeatures dict]
        season_index.pt  - Dict[(team, season) -> list[game_id]] chronologically ordered
        team_mapping.json - TEAM_TO_IDX dict
        gamestates_cache.pt - Dict[game_id -> dict of numpy arrays]
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building Phase 2 cache for seasons: {seasons}")

    # Query all final-state game data in batch
    season_placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season,
               gs.home_score, gs.away_score, gs.players_data
        FROM Games g
        JOIN GameStates gs ON gs.game_id = g.game_id AND gs.is_final_state = 1
        WHERE g.season IN ({season_placeholders})
          AND g.status = 3
          AND g.game_data_finalized = 1
          AND g.season_type IN ('Regular Season', 'Post Season')
        ORDER BY g.date_time_utc
    """

    with get_db(db_path) as conn:
        rows = conn.execute(query, seasons).fetchall()

    logger.info(f"Fetched {len(rows)} games from database")

    game_features = {}
    season_index = defaultdict(list)

    for game_id, home_team, away_team, date_utc, season, home_score, away_score, players_data in rows:
        home_players, away_players = _parse_players_data(players_data)

        game_date = str(date_utc)[:10] if date_utc else ""

        margin = home_score - away_score
        total = home_score + away_score

        # Get regulation scores
        reg_home, reg_away, is_overtime = _get_regulation_scores(
            game_id, home_score, away_score, db_path
        )

        features = PerGameFeatures(
            game_id=game_id,
            game_date=game_date,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            margin=margin,
            total=total,
            is_overtime=is_overtime,
            reg_home_score=reg_home,
            reg_away_score=reg_away,
            home_player_points=home_players,
            away_player_points=away_players,
        )

        game_features[game_id] = features

        # Build season index: both teams reference this game
        season_index[(home_team, season)].append(game_id)
        season_index[(away_team, season)].append(game_id)

    # Convert season_index defaultdict to regular dict for serialization
    season_index = dict(season_index)

    logger.info(f"Processed {len(game_features)} games, {len(season_index)} team-season entries")

    # Log per-season counts
    season_counts = defaultdict(int)
    for gf in game_features.values():
        season_counts[gf.season] += 1
    for s, c in sorted(season_counts.items()):
        logger.info(f"  {s}: {c} games")

    # Save game features as plain dicts (avoids pickle class-path issues)
    game_features_dicts = {gid: asdict(gf) for gid, gf in game_features.items()}
    game_features_path = cache_path / "game_features.pt"
    torch.save(game_features_dicts, game_features_path)
    logger.info(f"Saved game features to {game_features_path}")

    # Save season index
    season_index_path = cache_path / "season_index.pt"
    torch.save(season_index, season_index_path)
    logger.info(f"Saved season index to {season_index_path}")

    # Save team mapping
    mapping_path = cache_path / "team_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(TEAM_TO_IDX, f, indent=2)

    # Build player ID mapping: raw NBA API IDs -> sequential indices (0=padding)
    all_player_ids = set()
    for gf in game_features.values():
        for pid, _ in gf.home_player_points:
            all_player_ids.add(pid)
        for pid, _ in gf.away_player_points:
            all_player_ids.add(pid)

    # Sort for reproducibility, 0 is reserved for padding
    player_id_map = {pid: idx + 1 for idx, pid in enumerate(sorted(all_player_ids))}
    player_mapping_path = cache_path / "player_mapping.json"
    with open(player_mapping_path, "w") as f:
        json.dump({str(k): v for k, v in player_id_map.items()}, f)
    logger.info(f"Saved player mapping: {len(player_id_map)} unique players "
                f"(indices 1-{len(player_id_map)}, 0=padding)")

    # Build GameStates cache
    _build_gamestates_cache(list(game_features.keys()), cache_path, db_path)

    return {
        "game_features": game_features,
        "season_index": season_index,
        "player_id_map": player_id_map,
        "n_games": len(game_features),
    }


def _build_gamestates_cache(
    game_ids: list[str],
    cache_path: Path,
    db_path: Optional[str] = None,
):
    """Pre-tokenize GameStates for all games using GameStateTokenizer."""
    logger.info(f"Building GameStates cache for {len(game_ids)} games...")

    tokenizer = GameStateTokenizer(db_path)
    gs_cache = {}
    n_success = 0
    n_fail = 0

    for i, game_id in enumerate(game_ids):
        result = tokenizer.tokenize_game(game_id)
        if result is not None:
            gs_cache[game_id] = result
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"  Tokenized {i + 1}/{len(game_ids)} games ({n_success} success, {n_fail} fail)")

    gs_path = cache_path / "gamestates_cache.pt"
    torch.save(gs_cache, gs_path)
    logger.info(f"Saved GameStates cache: {n_success} games to {gs_path}")


def _dict_to_per_game_features(d: dict) -> PerGameFeatures:
    """Reconstruct PerGameFeatures from a plain dict."""
    # Convert player points lists of lists back to lists of tuples
    d = dict(d)
    d["home_player_points"] = [tuple(p) for p in d.get("home_player_points", [])]
    d["away_player_points"] = [tuple(p) for p in d.get("away_player_points", [])]
    return PerGameFeatures(**d)


def load_cache(cache_dir: str = "data/phase2_cache") -> dict:
    """Load the pre-built cache from disk."""
    cache_path = Path(cache_dir)

    raw_features = torch.load(cache_path / "game_features.pt", weights_only=False)
    game_features = {
        gid: _dict_to_per_game_features(d) for gid, d in raw_features.items()
    }
    season_index = torch.load(cache_path / "season_index.pt", weights_only=False)
    gs_cache = torch.load(cache_path / "gamestates_cache.pt", weights_only=False)

    with open(cache_path / "team_mapping.json") as f:
        team_mapping = json.load(f)

    player_mapping_path = cache_path / "player_mapping.json"
    if player_mapping_path.exists():
        with open(player_mapping_path) as f:
            player_id_map = {int(k): v for k, v in json.load(f).items()}
    else:
        player_id_map = {}

    return {
        "game_features": game_features,
        "season_index": season_index,
        "gs_cache": gs_cache,
        "team_mapping": team_mapping,
        "player_id_map": player_id_map,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build Phase 2 cache")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[
            "2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023",
            "2023-2024", "2024-2025", "2025-2026",
        ],
        help="Seasons to cache",
    )
    parser.add_argument("--cache-dir", default="data/phase2_cache")

    args = parser.parse_args()
    result = build_cache(args.seasons, args.cache_dir)
    print(f"\nCache built: {result['n_games']} games")
