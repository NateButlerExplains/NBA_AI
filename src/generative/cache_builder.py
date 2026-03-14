"""
Cache Builder for Phase 4 Generative Model.

Pre-processes all data from the database into tensor caches for training.
Two cache types:
  1. Game State Cache — per-game state tensors, score events, and clock targets
  2. Context Cache — season-level game features, player stats, and mappings

Usage:
    python scripts/build_generative_cache.py --seasons 2008-2009 2009-2010 ...
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from src.database import get_db

logger = logging.getLogger(__name__)

# 30 NBA team tricodes -> integer indices
TEAM_TO_IDX = {
    "ATL": 0,
    "BOS": 1,
    "BKN": 2,
    "CHA": 3,
    "CHI": 4,
    "CLE": 5,
    "DAL": 6,
    "DEN": 7,
    "DET": 8,
    "GSW": 9,
    "HOU": 10,
    "IND": 11,
    "LAC": 12,
    "LAL": 13,
    "MEM": 14,
    "MIA": 15,
    "MIL": 16,
    "MIN": 17,
    "NOP": 18,
    "NYK": 19,
    "OKC": 20,
    "ORL": 21,
    "PHI": 22,
    "PHX": 23,
    "POR": 24,
    "SAC": 25,
    "SAS": 26,
    "TOR": 27,
    "UTA": 28,
    "WAS": 29,
}

# Historical team codes -> current franchise codes
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn Nets
    "SEA": "OKC",  # Seattle SuperSonics -> Oklahoma City Thunder
    "NOH": "NOP",  # New Orleans Hornets -> New Orleans Pelicans
    "NOK": "NOP",  # New Orleans/OKC Hornets -> New Orleans Pelicans
    "VAN": "MEM",  # Vancouver Grizzlies -> Memphis Grizzlies
    "CHH": "CHA",  # Charlotte Hornets (old) -> Charlotte Hornets (current)
}

# 16 PlayerBox stats in order
STAT_COLUMNS = [
    "min",
    "pts",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "fga",
    "fgm",
    "fg3a",
    "fg3m",
    "fta",
    "ftm",
    "plus_minus",
]


def parse_clock(clock_str: str) -> float:
    """Parse 'MM:SS' or 'M:SS' clock string to seconds remaining.

    Returns seconds as a float. Handles missing/malformed clocks gracefully.
    """
    if not clock_str or not isinstance(clock_str, str):
        return 0.0
    try:
        parts = clock_str.strip().split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60.0 + seconds
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def normalize_team(abbrev: str) -> str:
    """Map historical team abbreviations to current."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


class GenerativeCacheBuilder:
    """Builds tensor caches from the database for the generative model."""

    def __init__(self, db_path: str, cache_dir: str, seasons: list[str]):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.seasons = seasons

        # Sub-directories
        self.states_dir = self.cache_dir / "states"
        self.context_dir = self.cache_dir / "context"

        # Tracking
        self.eligible_game_ids: list[str] = []
        self.all_player_ids: set[int] = set()

        # Statistics
        self.stats = {
            "total_games": 0,
            "games_with_states": 0,
            "games_skipped": 0,
            "total_states": 0,
            "class_counts": np.zeros(7, dtype=np.int64),
        }

    def build(self):
        """Build all caches."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir.mkdir(parents=True, exist_ok=True)

        self._fetch_eligible_games()
        self._build_game_state_cache()
        self._build_context_cache()
        self._save_metadata()

    def _fetch_eligible_games(self):
        """Query all eligible game_ids: status=3, not All-Star, in requested seasons."""
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT g.game_id
            FROM Games g
            WHERE g.status = 3
              AND g.season IN ({season_placeholders})
              AND g.game_id NOT LIKE '003%'
            ORDER BY g.date_time_utc
        """
        with get_db(self.db_path) as conn:
            rows = conn.execute(query, self.seasons).fetchall()

        self.eligible_game_ids = [row[0] for row in rows]
        self.stats["total_games"] = len(self.eligible_game_ids)
        logger.info(
            f"Found {len(self.eligible_game_ids)} eligible games across {len(self.seasons)} seasons"
        )

    def _build_game_state_cache(self):
        """Parse GameStates -> state tensors + score events + clock targets.

        Saves one .pt file per game in states/ directory.
        """
        logger.info("Building game state cache...")

        chunk_size = 500
        game_ids = self.eligible_game_ids

        # Fetch and process in chunks
        processed = 0
        skipped = 0

        for chunk_start in tqdm(
            range(0, len(game_ids), chunk_size), desc="Game states", unit="chunk"
        ):
            chunk = game_ids[chunk_start : chunk_start + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))

            with get_db(self.db_path) as conn:
                rows = conn.execute(
                    f"""
                    SELECT game_id, play_id, period, clock,
                           home_score, away_score
                    FROM GameStates
                    WHERE game_id IN ({placeholders})
                    ORDER BY game_id, play_id
                    """,
                    chunk,
                ).fetchall()

            # Group rows by game_id
            game_rows = defaultdict(list)
            for row in rows:
                game_rows[row[0]].append(
                    row[1:]
                )  # (play_id, period, clock, home_score, away_score)

            for game_id in chunk:
                if game_id not in game_rows:
                    skipped += 1
                    continue

                result = self._process_game_states(game_id, game_rows[game_id])
                if result is None:
                    skipped += 1
                    continue

                # Save individual game cache
                save_path = self.states_dir / f"{game_id}.pt"
                torch.save(result, save_path)
                processed += 1

                # Update statistics
                n = result["n_states"]
                self.stats["total_states"] += n
                if n > 1:
                    events = result["score_events"]
                    for cls in range(7):
                        self.stats["class_counts"][cls] += int((events == cls).sum())

        self.stats["games_with_states"] = processed
        self.stats["games_skipped"] = skipped

        logger.info(f"Game state cache: {processed} games saved, {skipped} skipped")
        if processed > 0:
            avg_states = self.stats["total_states"] / processed
            logger.info(f"Average states per game: {avg_states:.1f}")
            logger.info(f"Score event class distribution:")
            class_names = [
                "no_score",
                "home+1",
                "home+2",
                "home+3",
                "away+1",
                "away+2",
                "away+3",
            ]
            total_events = self.stats["class_counts"].sum()
            for i, name in enumerate(class_names):
                count = self.stats["class_counts"][i]
                pct = 100.0 * count / total_events if total_events > 0 else 0.0
                logger.info(f"  {name}: {count:,} ({pct:.1f}%)")

    def _process_game_states(self, game_id: str, rows: list) -> Optional[dict]:
        """Process raw GameStates rows for a single game into tensors.

        Args:
            game_id: The game identifier.
            rows: List of (play_id, period, clock, home_score, away_score) tuples,
                  ordered by play_id.

        Returns:
            Dict with states, score_events, clock_targets, n_states,
            final_home_score, final_away_score, final_margin.
            None if game has insufficient data.
        """
        # Truncate at regulation end (period <= 4)
        reg_rows = [
            (play_id, period, clock, h_score, a_score)
            for play_id, period, clock, h_score, a_score in rows
            if period is not None and period <= 4
        ]

        if len(reg_rows) < 2:
            return None

        # Build state vectors
        state_list = []
        for play_id, period, clock, h_score, a_score in reg_rows:
            seconds_remaining = parse_clock(clock)
            h_score = h_score or 0
            a_score = a_score or 0

            period_norm = period / 4.0
            clock_norm = seconds_remaining / 720.0
            elapsed = (period - 1) * 720.0 + (720.0 - seconds_remaining)
            game_progress = elapsed / 2880.0
            home_score_norm = h_score / 150.0
            away_score_norm = a_score / 150.0
            margin_norm = (h_score - a_score) / 50.0
            total_norm = (h_score + a_score) / 300.0

            state_list.append(
                [
                    period_norm,
                    clock_norm,
                    game_progress,
                    home_score_norm,
                    away_score_norm,
                    margin_norm,
                    total_norm,
                ]
            )

        states = torch.tensor(state_list, dtype=torch.float32)
        n_states = states.shape[0]

        # Build score events: classify delta between consecutive states
        score_events = []
        for i in range(n_states - 1):
            # Current scores from raw rows (before normalization)
            h_curr = reg_rows[i][3] or 0
            a_curr = reg_rows[i][4] or 0
            h_next = reg_rows[i + 1][3] or 0
            a_next = reg_rows[i + 1][4] or 0

            h_delta = h_next - h_curr
            a_delta = a_next - a_curr

            if h_delta > 0 and a_delta > 0:
                # Both teams scored simultaneously — prioritize home
                h_delta = min(h_delta, 3)
                event = h_delta  # 1, 2, or 3
            elif h_delta > 0:
                h_delta = min(h_delta, 3)
                event = h_delta  # 1=home+1, 2=home+2, 3=home+3
            elif a_delta > 0:
                a_delta = min(a_delta, 3)
                event = a_delta + 3  # 4=away+1, 5=away+2, 6=away+3
            else:
                event = 0  # no_score

            score_events.append(event)

        score_events = torch.tensor(score_events, dtype=torch.int64)

        # Clock targets: next state's clock_norm
        clock_targets = states[1:, 1].clone()  # column 1 = clock_norm

        # Final scores at regulation end
        final_h = reg_rows[-1][3] or 0
        final_a = reg_rows[-1][4] or 0

        return {
            "states": states,
            "score_events": score_events,
            "clock_targets": clock_targets,
            "n_states": n_states,
            "final_home_score": final_h,
            "final_away_score": final_a,
            "final_margin": final_h - final_a,
        }

    def _build_context_cache(self):
        """Build season-level context from Games + PlayerBox.

        Produces:
          - season_games: {(team_abbrev, season) -> [game_id, ...]}
          - game_features: {game_id -> {home_team, away_team, home_score, away_score,
                            date, home_player_stats, away_player_stats}}
          - player_id_map: {person_id -> contiguous index}
          - team_id_map: {team_abbrev -> index}
        """
        logger.info("Building context cache...")

        # Fetch game metadata (scores come from TeamBox, not Games table)
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT g.game_id, g.home_team, g.away_team,
                   g.date_time_utc, g.season
            FROM Games g
            WHERE g.status = 3
              AND g.season IN ({season_placeholders})
              AND g.game_id NOT LIKE '003%'
            ORDER BY g.date_time_utc
        """
        with get_db(self.db_path) as conn:
            game_rows = conn.execute(query, self.seasons).fetchall()

        logger.info(f"Fetched {len(game_rows)} games for context cache")

        # Batch-fetch scores from TeamBox
        game_scores = self._batch_query_scores([r[0] for r in game_rows])
        n_missing_scores = len(game_rows) - len(game_scores)
        logger.info(f"Fetched scores for {len(game_scores)} games from TeamBox")
        if n_missing_scores > 0:
            logger.warning(
                f"Missing TeamBox scores for {n_missing_scores} games — will default to 0"
            )

        # Build season_games index and collect game metadata
        season_games = defaultdict(list)
        game_meta = {}

        for game_id, home, away, date_utc, season in game_rows:
            home = normalize_team(home)
            away = normalize_team(away)
            game_date = str(date_utc)[:10] if date_utc else ""

            scores = game_scores.get(game_id, {})
            h_score = scores.get(home, 0)
            a_score = scores.get(away, 0)

            game_meta[game_id] = {
                "home_team": home,
                "away_team": away,
                "home_score": h_score,
                "away_score": a_score,
                "date": game_date,
                "season": season,
            }

            season_games[(home, season)].append(game_id)
            season_games[(away, season)].append(game_id)

        # Batch fetch PlayerBox data
        game_id_list = list(game_meta.keys())
        playerbox = self._batch_query_playerbox(game_id_list)

        n_with_pb = sum(1 for gid in game_id_list if gid in playerbox)
        logger.info(f"PlayerBox data: {n_with_pb}/{len(game_id_list)} games")

        # Build game_features with player stats
        game_features = {}
        for game_id, meta in tqdm(
            game_meta.items(), desc="Context features", unit="game"
        ):
            home = meta["home_team"]
            away = meta["away_team"]

            pb = playerbox.get(game_id, {})
            home_stats = pb.get(home, [])
            away_stats = pb.get(away, [])

            # Collect player IDs
            for pid, _ in home_stats:
                self.all_player_ids.add(pid)
            for pid, _ in away_stats:
                self.all_player_ids.add(pid)

            game_features[game_id] = {
                "home_team": home,
                "away_team": away,
                "home_score": meta["home_score"],
                "away_score": meta["away_score"],
                "date": meta["date"],
                "home_player_stats": home_stats,
                "away_player_stats": away_stats,
            }

        # Build player_id_map: NBA person_id -> contiguous index (0 to N-1)
        sorted_pids = sorted(self.all_player_ids)
        player_id_map = {pid: idx for idx, pid in enumerate(sorted_pids)}

        # team_id_map is just TEAM_TO_IDX
        team_id_map = dict(TEAM_TO_IDX)

        # Save context caches
        season_games_dict = {f"{k[0]}|{k[1]}": v for k, v in season_games.items()}
        torch.save(season_games_dict, self.context_dir / "season_games.pt")
        logger.info(f"Saved season_games: {len(season_games_dict)} team-season entries")

        torch.save(game_features, self.context_dir / "game_features.pt")
        logger.info(f"Saved game_features: {len(game_features)} games")

        torch.save(player_id_map, self.context_dir / "player_id_map.pt")
        logger.info(
            f"Saved player_id_map: {len(player_id_map)} players (indices 0-{len(player_id_map) - 1})"
        )

        torch.save(team_id_map, self.context_dir / "team_id_map.pt")
        logger.info(f"Saved team_id_map: {len(team_id_map)} teams")

        # Log per-season counts
        season_counts = defaultdict(int)
        for meta in game_meta.values():
            season_counts[meta["season"]] += 1
        for s, c in sorted(season_counts.items()):
            logger.info(f"  {s}: {c} games")

    def _batch_query_scores(self, game_ids: list[str]) -> dict:
        """Batch query TeamBox for game scores.

        Returns {game_id: {team_abbrev: pts}}.
        """
        if not game_ids:
            return {}

        result: dict[str, dict[str, int]] = {}
        chunk_size = 500

        with get_db(self.db_path) as conn:
            for i in range(0, len(game_ids), chunk_size):
                chunk = game_ids[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))

                rows = conn.execute(
                    f"""
                    SELECT tb.game_id, t.abbreviation, tb.pts
                    FROM TeamBox tb
                    JOIN Teams t ON tb.team_id = t.team_id
                    WHERE tb.game_id IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()

                for game_id, abbrev, pts in rows:
                    abbrev = normalize_team(abbrev)
                    if game_id not in result:
                        result[game_id] = {}
                    result[game_id][abbrev] = pts or 0

        return result

    def _batch_query_playerbox(self, game_ids: list[str]) -> dict:
        """Batch query PlayerBox for multiple games.

        Returns {game_id: {team_abbrev: [(player_id, [16 floats]), ...]}}.
        Limited to 13 players per team per game, sorted by minutes DESC.
        """
        if not game_ids:
            return {}

        result = {}
        chunk_size = 500
        max_players = 13

        with get_db(self.db_path) as conn:
            for i in tqdm(
                range(0, len(game_ids), chunk_size),
                desc="PlayerBox queries",
                unit="chunk",
            ):
                chunk = game_ids[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))

                rows = conn.execute(
                    f"""
                    SELECT pb.game_id, pb.player_id, t.abbreviation,
                           pb.min, pb.pts, pb.oreb, pb.dreb, pb.ast, pb.stl, pb.blk,
                           pb.tov, pb.pf, pb.fga, pb.fgm, pb.fg3a, pb.fg3m, pb.fta,
                           pb.ftm, pb.plus_minus
                    FROM PlayerBox pb
                    JOIN Teams t ON pb.team_id = t.team_id
                    WHERE pb.game_id IN ({placeholders})
                      AND pb.min > 0
                    ORDER BY pb.game_id, t.abbreviation, pb.min DESC
                    """,
                    chunk,
                ).fetchall()

                for row in rows:
                    game_id = row[0]
                    player_id = row[1]
                    team_abbrev = normalize_team(row[2])

                    # Extract 16 stats, coalesce NULL to 0.0
                    stats = [
                        float(row[3 + j]) if row[3 + j] is not None else 0.0
                        for j in range(16)
                    ]

                    if game_id not in result:
                        result[game_id] = {}
                    if team_abbrev not in result[game_id]:
                        result[game_id][team_abbrev] = []

                    result[game_id][team_abbrev].append((player_id, stats))

        # Trim to max_players per team (already sorted by minutes DESC)
        for game_id in result:
            for team in result[game_id]:
                result[game_id][team] = result[game_id][team][:max_players]

        return result

    def _save_metadata(self):
        """Save dataset statistics and build metadata."""
        metadata = {
            "seasons": self.seasons,
            "total_games": self.stats["total_games"],
            "games_with_states": self.stats["games_with_states"],
            "games_skipped": self.stats["games_skipped"],
            "total_states": int(self.stats["total_states"]),
            "avg_states_per_game": (
                self.stats["total_states"] / self.stats["games_with_states"]
                if self.stats["games_with_states"] > 0
                else 0.0
            ),
            "n_players": len(self.all_player_ids),
            "class_distribution": {
                "no_score": int(self.stats["class_counts"][0]),
                "home+1": int(self.stats["class_counts"][1]),
                "home+2": int(self.stats["class_counts"][2]),
                "home+3": int(self.stats["class_counts"][3]),
                "away+1": int(self.stats["class_counts"][4]),
                "away+2": int(self.stats["class_counts"][5]),
                "away+3": int(self.stats["class_counts"][6]),
            },
        }

        meta_path = self.cache_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")


def build_cache_cli():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build generative model cache")
    parser.add_argument(
        "--db-path", default="data/NBA_AI_full.sqlite", help="Path to SQLite database"
    )
    parser.add_argument(
        "--cache-dir", default="data/generative_cache", help="Output cache directory"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to cache (default: 2008-2009 through 2025-2026)",
    )
    args = parser.parse_args()

    seasons = args.seasons or [f"{y}-{y + 1}" for y in range(2008, 2026)]

    builder = GenerativeCacheBuilder(args.db_path, args.cache_dir, seasons)
    builder.build()


if __name__ == "__main__":
    build_cache_cli()
