#!/usr/bin/env python
"""Phase 4 Experiment 7: LLM-Based NBA Game Prediction via OpenAI GPT-5.4.

Sends structured game context (team stats, player stats, records, recent form)
to OpenAI's GPT-5.4 family and gets score/spread/win predictions back.
Evaluates on the same test set and metrics as all Phase 4 experiments.

Models (3 tiers):
    gpt-5.4-nano  (budget)
    gpt-5.4-mini  (mid)
    gpt-5.4       (premium)

Data split (chronological, same as all Phase 4):
    Train: seasons 2008-2022  (not used — LLM has no training step)
    Val:   season 2023-2024
    Test:  seasons 2024-2025, 2025-2026

Usage::

    # Build prompts and inspect a sample
    python scripts/exp7_llm_predict.py build-prompts --split test

    # Preview a few prompts (no API call)
    python scripts/exp7_llm_predict.py preview --split test --n 3

    # Run via batch API (50% cost savings)
    python scripts/exp7_llm_predict.py run-batch --split test --model gpt-5.4-nano
    python scripts/exp7_llm_predict.py run-batch --split test --model gpt-5.4-mini
    python scripts/exp7_llm_predict.py run-batch --split test --model gpt-5.4

    # Check batch status
    python scripts/exp7_llm_predict.py check-batch --batch-id batch_abc123

    # Download completed batch results
    python scripts/exp7_llm_predict.py download-batch --batch-id batch_abc123 \\
        --output data/exp7/test_gpt-5.4-nano_results.jsonl

    # Run via standard API (small runs, debugging)
    python scripts/exp7_llm_predict.py run-standard --split val --model gpt-5.4-nano --limit 10

    # Evaluate results
    python scripts/exp7_llm_predict.py evaluate --results data/exp7/test_gpt-5.4-nano_results.jsonl

    # Compare all models
    python scripts/exp7_llm_predict.py compare --results-dir data/exp7/

Install dependencies (if not already installed)::

    pip install openai tqdm numpy scikit-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not required if env vars are set directly

logger = logging.getLogger(__name__)

# =============================================================================
# Paths and constants
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
EXP7_DIR = PROJECT_ROOT / "data" / "exp7"

TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2023)]
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]

SUPPORTED_MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"]

# Historical team abbreviation -> current franchise abbreviation
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "VAN": "MEM",
    "CHH": "CHA",
}

# OpenAI structured output schema for predictions
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "game_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "home_score": {"type": "integer"},
                "away_score": {"type": "integer"},
                "home_win_probability": {"type": "number"},
            },
            "required": [
                "analysis",
                "home_score",
                "away_score",
                "home_win_probability",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """\
You are an expert NBA game prediction model. Your task is to predict the \
outcome of an NBA game using ONLY the statistical data provided below.

IMPORTANT RULES:
- Use ONLY the provided statistics to make your prediction
- Do NOT use knowledge of actual game outcomes
- Do NOT search the web or access external information
- Analyze the data systematically before making your prediction

CALIBRATION CONTEXT:
- Average NBA game total: ~224 points (~112 per team)
- Home court advantage: worth approximately 2-3 points
- Typical point spread range: -15 to +15
- Home teams win approximately 57% of games
- Standard deviation of NBA point spreads: ~12 points

PREDICTION APPROACH:
Analyze the matchup systematically:
1. Compare offensive and defensive efficiency
2. Evaluate key player matchups and availability
3. Consider rest, schedule, and home/away factors
4. Account for recent form and trends
5. Assess pace and style compatibility

Provide your prediction as a JSON object with these exact fields:
{
  "analysis": "2-4 sentence analysis of the key factors",
  "home_score": <integer>,
  "away_score": <integer>,
  "home_win_probability": <float between 0 and 1>
}"""

# Token pricing per million tokens (as of 2026-03, USD, standard API)
# Source: https://developers.openai.com/api/docs/pricing
PRICING = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
}


# =============================================================================
# Database helpers
# =============================================================================
def normalize_team(abbrev: str) -> str:
    """Map historical team abbreviations to current franchise codes."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


@contextmanager
def get_db(db_path: str | Path | None = None):
    """Context manager for SQLite database connections with WAL mode."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
    finally:
        conn.close()


# =============================================================================
# Data loading
# =============================================================================
def load_team_info(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load team_id -> abbreviation and full_name mapping from Teams table."""
    rows = conn.execute("SELECT team_id, abbreviation, full_name FROM Teams").fetchall()
    by_id = {}
    by_abbrev = {}
    for team_id, abbrev, full_name in rows:
        by_id[team_id] = {"abbreviation": abbrev, "full_name": full_name}
        by_abbrev[abbrev] = {"team_id": team_id, "full_name": full_name}
    return {"by_id": by_id, "by_abbrev": by_abbrev}


def load_games(conn: sqlite3.Connection, seasons: list[str]) -> list[dict]:
    """Load completed regular-season games for the given seasons, chronologically."""
    placeholders = ",".join(["?"] * len(seasons))
    rows = conn.execute(
        f"""
        SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.season
        FROM Games g
        WHERE g.status = 3
          AND g.season_type = 'Regular Season'
          AND g.season IN ({placeholders})
          AND g.game_id NOT LIKE '003%'
        ORDER BY g.date_time_utc
        """,
        seasons,
    ).fetchall()

    games = []
    for game_id, date_utc, home, away, season in rows:
        games.append(
            {
                "game_id": game_id,
                "date_utc": date_utc,
                "date": str(date_utc)[:10],
                "home_team": home,
                "away_team": away,
                "home_norm": normalize_team(home),
                "away_norm": normalize_team(away),
                "season": season,
            }
        )
    return games


def load_scores(
    conn: sqlite3.Connection, game_ids: list[str], team_info: dict
) -> dict[str, dict]:
    """Batch fetch scores from TeamBox for a list of game_ids.

    Returns {game_id: {"home_score": int, "away_score": int}} after matching
    team_ids against the game's home/away teams.
    """
    result = {}
    chunk_size = 500
    # Pre-build team_id -> current abbreviation map
    id_to_abbrev = {
        tid: normalize_team(info["abbreviation"])
        for tid, info in team_info["by_id"].items()
    }

    for i in range(0, len(game_ids), chunk_size):
        chunk = game_ids[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT tb.game_id, tb.team_id, tb.pts
            FROM TeamBox tb
            WHERE tb.game_id IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for gid, team_id, pts in rows:
            abbrev = id_to_abbrev.get(team_id, "???")
            if gid not in result:
                result[gid] = {}
            result[gid][abbrev] = pts or 0

    return result


# =============================================================================
# Context building: per-game prompt construction
# =============================================================================
class GameContextBuilder:
    """Builds the user prompt for a single game from database queries.

    Pre-loads all data needed for the evaluation split into memory, then
    constructs prompts game-by-game using only data available before each game.
    """

    def __init__(self, db_path: str | Path, seasons: list[str]):
        self.db_path = str(db_path)
        # We need all seasons from 2008 onward for rolling history
        all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
        with get_db(self.db_path) as conn:
            self.team_info = load_team_info(conn)
            self.all_games = load_games(conn, all_seasons)
            all_game_ids = [g["game_id"] for g in self.all_games]
            self.scores = load_scores(conn, all_game_ids, self.team_info)
            self._load_teambox(conn, all_game_ids)
            self._load_playerbox(conn, all_game_ids)

        # Attach scores to games
        for g in self.all_games:
            s = self.scores.get(g["game_id"], {})
            g["home_score"] = s.get(g["home_norm"], 0)
            g["away_score"] = s.get(g["away_norm"], 0)

        # Build index: team -> chronological list of game dicts
        self.team_games: dict[str, list[dict]] = defaultdict(list)
        for g in self.all_games:
            self.team_games[g["home_norm"]].append(g)
            self.team_games[g["away_norm"]].append(g)

        # Build index: game_id -> game dict for quick lookup
        self.game_by_id: dict[str, dict] = {g["game_id"]: g for g in self.all_games}

        logger.info(
            f"Loaded {len(self.all_games)} games, "
            f"{len(self.teambox)} TeamBox rows, "
            f"{len(self.playerbox)} PlayerBox game entries"
        )

    def _load_teambox(self, conn: sqlite3.Connection, game_ids: list[str]) -> None:
        """Load all TeamBox data into memory: {(game_id, team_abbrev): stats_dict}."""
        self.teambox: dict[tuple[str, str], dict] = {}
        id_to_abbrev = {
            tid: normalize_team(info["abbreviation"])
            for tid, info in self.team_info["by_id"].items()
        }
        chunk_size = 500
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT tb.game_id, tb.team_id, tb.pts, tb.pts_allowed,
                       tb.reb, tb.ast, tb.stl, tb.blk, tb.tov,
                       tb.fga, tb.fgm, tb.fg3a, tb.fg3m, tb.fta, tb.ftm
                FROM TeamBox tb
                WHERE tb.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                gid = row[0]
                abbrev = id_to_abbrev.get(row[1], "???")
                self.teambox[(gid, abbrev)] = {
                    "pts": row[2] or 0,
                    "pts_allowed": row[3] or 0,
                    "reb": row[4] or 0,
                    "ast": row[5] or 0,
                    "stl": row[6] or 0,
                    "blk": row[7] or 0,
                    "tov": row[8] or 0,
                    "fga": row[9] or 0,
                    "fgm": row[10] or 0,
                    "fg3a": row[11] or 0,
                    "fg3m": row[12] or 0,
                    "fta": row[13] or 0,
                    "ftm": row[14] or 0,
                }

    def _load_playerbox(self, conn: sqlite3.Connection, game_ids: list[str]) -> None:
        """Load all PlayerBox data into memory.

        Structure: {game_id: {team_abbrev: [player_dict, ...]}}
        """
        self.playerbox: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        id_to_abbrev = {
            tid: normalize_team(info["abbreviation"])
            for tid, info in self.team_info["by_id"].items()
        }
        chunk_size = 500
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT pb.game_id, pb.team_id, pb.player_id, pb.player_name,
                       pb.position, pb.min, pb.pts, pb.oreb, pb.dreb, pb.reb,
                       pb.ast, pb.stl, pb.blk, pb.tov,
                       pb.fga, pb.fgm, pb.fg3a, pb.fg3m, pb.fta, pb.ftm,
                       pb.plus_minus
                FROM PlayerBox pb
                WHERE pb.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                gid = row[0]
                abbrev = id_to_abbrev.get(row[1], "???")
                self.playerbox[gid][abbrev].append(
                    {
                        "player_id": row[2],
                        "player_name": row[3] or "Unknown",
                        "position": row[4] or "",
                        "min": row[5] or 0.0,
                        "pts": row[6] or 0,
                        "oreb": row[7] or 0,
                        "dreb": row[8] or 0,
                        "reb": row[9] or 0,
                        "ast": row[10] or 0,
                        "stl": row[11] or 0,
                        "blk": row[12] or 0,
                        "tov": row[13] or 0,
                        "fga": row[14] or 0,
                        "fgm": row[15] or 0,
                        "fg3a": row[16] or 0,
                        "fg3m": row[17] or 0,
                        "fta": row[18] or 0,
                        "ftm": row[19] or 0,
                        "plus_minus": row[20] or 0,
                    }
                )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _prior_team_games(
        self,
        team: str,
        before_date: str,
        same_season: bool = False,
        season: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Get a team's games before a given date, most recent first."""
        games = []
        for g in reversed(self.team_games.get(team, [])):
            if g["date"] >= before_date:
                continue
            if same_season and season and g["season"] != season:
                continue
            games.append(g)
            if limit and len(games) >= limit:
                break
        return games

    def _team_record(self, team: str, before_date: str, season: str) -> dict:
        """Compute W-L record, home/away splits, last-10, and streak."""
        games = self._prior_team_games(
            team, before_date, same_season=True, season=season
        )
        total_w, total_l = 0, 0
        home_w, home_l = 0, 0
        away_w, away_l = 0, 0
        last_10_w, last_10_l = 0, 0
        streak_type = None
        streak_count = 0

        for i, g in enumerate(games):
            is_home = g["home_norm"] == team
            score = g["home_score"] if is_home else g["away_score"]
            opp_score = g["away_score"] if is_home else g["home_score"]
            won = score > opp_score

            if won:
                total_w += 1
                if is_home:
                    home_w += 1
                else:
                    away_w += 1
            else:
                total_l += 1
                if is_home:
                    home_l += 1
                else:
                    away_l += 1

            if i < 10:
                if won:
                    last_10_w += 1
                else:
                    last_10_l += 1

            # Streak: count from most recent
            if i == 0:
                streak_type = "W" if won else "L"
                streak_count = 1
            elif (won and streak_type == "W") or (not won and streak_type == "L"):
                if streak_count == i:  # still unbroken
                    streak_count += 1

        streak_str = f"{streak_type}{streak_count}" if streak_type else "—"
        return {
            "W": total_w,
            "L": total_l,
            "home_W": home_w,
            "home_L": home_l,
            "away_W": away_w,
            "away_L": away_l,
            "last_10_W": last_10_w,
            "last_10_L": last_10_l,
            "streak": streak_str,
        }

    def _rolling_team_stats(
        self, team: str, before_date: str, windows: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """Compute rolling team stat averages for Last 5, Last 10, Season.

        Returns {"Last 5": {"PPG": x, ...}, "Last 10": {...}, "Season": {...}}
        """
        if windows is None:
            windows = [5, 10]

        # Get all prior games for this team (any season, for rolling windows)
        all_prior = self._prior_team_games(team, before_date, same_season=False)

        # Get season games for "Season" window
        if all_prior:
            # Find the season of the target game by looking at what season we're in
            # We need the season of the game being predicted, passed externally
            # For now, get it from the most recent game's season as approximation
            pass

        result = {}
        for window_name, games in [
            ("Last 5", all_prior[:5]),
            ("Last 10", all_prior[:10]),
        ]:
            if not games:
                result[window_name] = self._empty_team_stats()
                continue
            result[window_name] = self._avg_team_stats(team, games)

        # Season average: use all prior same-season games
        # We need to know the season -- derive from the first prior game in the list
        # Actually, let's accept season as param
        return result

    def _rolling_team_stats_with_season(
        self, team: str, before_date: str, season: str
    ) -> dict[str, dict[str, str]]:
        """Compute rolling team stat averages for Last 5, Last 10, Season.

        Returns formatted strings for table display.
        """
        all_prior = self._prior_team_games(team, before_date, same_season=False)
        season_prior = self._prior_team_games(
            team, before_date, same_season=True, season=season
        )

        result = {}
        for window_name, games in [
            ("Last 5", all_prior[:5]),
            ("Last 10", all_prior[:10]),
            ("Season", season_prior),
        ]:
            if not games:
                result[window_name] = self._empty_team_stats()
                continue
            result[window_name] = self._avg_team_stats(team, games)

        return result

    def _avg_team_stats(self, team: str, games: list[dict]) -> dict[str, float]:
        """Compute average team stats across a list of games."""
        totals = defaultdict(float)
        n = 0
        for g in games:
            key = (g["game_id"], team)
            tb = self.teambox.get(key)
            if tb is None:
                continue
            n += 1
            totals["pts"] += tb["pts"]
            totals["pts_allowed"] += tb["pts_allowed"]
            totals["reb"] += tb["reb"]
            totals["ast"] += tb["ast"]
            totals["stl"] += tb["stl"]
            totals["blk"] += tb["blk"]
            totals["tov"] += tb["tov"]
            totals["fga"] += tb["fga"]
            totals["fgm"] += tb["fgm"]
            totals["fg3a"] += tb["fg3a"]
            totals["fg3m"] += tb["fg3m"]
            totals["fta"] += tb["fta"]
            totals["ftm"] += tb["ftm"]

        if n == 0:
            return self._empty_team_stats()

        fg_pct = (totals["fgm"] / totals["fga"] * 100) if totals["fga"] > 0 else 0.0
        fg3_pct = (totals["fg3m"] / totals["fg3a"] * 100) if totals["fg3a"] > 0 else 0.0
        ft_pct = (totals["ftm"] / totals["fta"] * 100) if totals["fta"] > 0 else 0.0

        return {
            "PPG": totals["pts"] / n,
            "Opp PPG": totals["pts_allowed"] / n,
            "FG%": fg_pct,
            "3P%": fg3_pct,
            "FT%": ft_pct,
            "REB": totals["reb"] / n,
            "AST": totals["ast"] / n,
            "TOV": totals["tov"] / n,
            "STL": totals["stl"] / n,
            "BLK": totals["blk"] / n,
        }

    def _empty_team_stats(self) -> dict[str, float]:
        return {
            "PPG": 0.0,
            "Opp PPG": 0.0,
            "FG%": 0.0,
            "3P%": 0.0,
            "FT%": 0.0,
            "REB": 0.0,
            "AST": 0.0,
            "TOV": 0.0,
            "STL": 0.0,
            "BLK": 0.0,
        }

    def _rest_days(self, team: str, before_date: str) -> int:
        """Compute days of rest before the target game."""
        prior = self._prior_team_games(team, before_date, limit=1)
        if not prior:
            return 3  # default for season openers
        try:
            target = datetime.strptime(before_date, "%Y-%m-%d")
            last = datetime.strptime(prior[0]["date"], "%Y-%m-%d")
            return max((target - last).days - 1, 0)
        except (ValueError, TypeError):
            return 1

    def _recent_games_table(
        self, team: str, before_date: str, n: int = 10
    ) -> list[dict]:
        """Get recent game results for the table display."""
        prior = self._prior_team_games(team, before_date, limit=n)
        rows = []
        for g in prior:
            is_home = g["home_norm"] == team
            opp = g["away_norm"] if is_home else g["home_norm"]
            ha = "H" if is_home else "A"
            score = g["home_score"] if is_home else g["away_score"]
            opp_score = g["away_score"] if is_home else g["home_score"]
            margin = score - opp_score
            result = "W" if margin > 0 else "L"
            rows.append(
                {
                    "date": g["date"],
                    "opp": opp,
                    "ha": ha,
                    "score": f"{result} {score}-{opp_score}",
                    "margin": f"{'+' if margin > 0 else ''}{margin}",
                }
            )
        return rows

    def _key_players_table(
        self, team: str, before_date: str, n_games: int = 10, top_n: int = 10
    ) -> list[dict]:
        """Get top players by average minutes over last N games.

        Returns list of dicts with averaged stats per player.
        """
        prior = self._prior_team_games(team, before_date, limit=n_games)
        if not prior:
            return []

        # Accumulate stats per player
        player_stats: dict[int, dict] = {}
        for g in prior:
            gid = g["game_id"]
            players = self.playerbox.get(gid, {}).get(team, [])
            for p in players:
                if p["min"] <= 0:
                    continue
                pid = p["player_id"]
                if pid not in player_stats:
                    player_stats[pid] = {
                        "player_name": p["player_name"],
                        "position": p["position"],
                        "games": 0,
                        "min": 0.0,
                        "pts": 0,
                        "reb": 0,
                        "ast": 0,
                        "stl": 0,
                        "blk": 0,
                        "tov": 0,
                        "fga": 0,
                        "fgm": 0,
                        "fg3a": 0,
                        "fg3m": 0,
                        "fta": 0,
                        "ftm": 0,
                        "plus_minus": 0,
                    }
                ps = player_stats[pid]
                ps["games"] += 1
                ps["min"] += p["min"]
                ps["pts"] += p["pts"]
                ps["reb"] += p["reb"]
                ps["ast"] += p["ast"]
                ps["stl"] += p["stl"]
                ps["blk"] += p["blk"]
                ps["tov"] += p["tov"]
                ps["fga"] += p["fga"]
                ps["fgm"] += p["fgm"]
                ps["fg3a"] += p["fg3a"]
                ps["fg3m"] += p["fg3m"]
                ps["fta"] += p["fta"]
                ps["ftm"] += p["ftm"]
                ps["plus_minus"] += p["plus_minus"]

        # Sort by average minutes descending, take top_n
        sorted_players = sorted(
            player_stats.values(),
            key=lambda x: x["min"] / max(x["games"], 1),
            reverse=True,
        )[:top_n]

        result = []
        for ps in sorted_players:
            n = ps["games"]
            fg_pct = (ps["fgm"] / ps["fga"] * 100) if ps["fga"] > 0 else 0.0
            fg3_pct = (ps["fg3m"] / ps["fg3a"] * 100) if ps["fg3a"] > 0 else 0.0
            ft_pct = (ps["ftm"] / ps["fta"] * 100) if ps["fta"] > 0 else 0.0
            result.append(
                {
                    "player_name": ps["player_name"],
                    "position": ps["position"],
                    "gp": n,
                    "min": ps["min"] / n,
                    "pts": ps["pts"] / n,
                    "reb": ps["reb"] / n,
                    "ast": ps["ast"] / n,
                    "stl": ps["stl"] / n,
                    "blk": ps["blk"] / n,
                    "tov": ps["tov"] / n,
                    "fg_pct": fg_pct,
                    "fg3_pct": fg3_pct,
                    "ft_pct": ft_pct,
                    "plus_minus": ps["plus_minus"] / n,
                }
            )
        return result

    def _tonight_roster(self, game_id: str, team: str) -> list[dict]:
        """Get players who actually played in the target game (min > 0)."""
        players = self.playerbox.get(game_id, {}).get(team, [])
        return [
            {"player_name": p["player_name"], "position": p["position"]}
            for p in players
            if p["min"] > 0
        ]

    def _notable_absences(
        self,
        game_id: str,
        team: str,
        before_date: str,
        season: str,
        min_mpg: float = 15.0,
    ) -> list[str]:
        """Find players who averaged >min_mpg MPG this season but didn't play tonight."""
        # Get season games before this date
        season_games = self._prior_team_games(
            team, before_date, same_season=True, season=season
        )
        if not season_games:
            return []

        # Compute season MPG per player
        player_minutes: dict[int, dict] = {}
        for g in season_games:
            gid = g["game_id"]
            players = self.playerbox.get(gid, {}).get(team, [])
            for p in players:
                if p["min"] <= 0:
                    continue
                pid = p["player_id"]
                if pid not in player_minutes:
                    player_minutes[pid] = {
                        "player_name": p["player_name"],
                        "total_min": 0.0,
                        "games": 0,
                    }
                player_minutes[pid]["total_min"] += p["min"]
                player_minutes[pid]["games"] += 1

        # Players who qualify (>min_mpg)
        qualified = {}
        for pid, pm in player_minutes.items():
            mpg = pm["total_min"] / pm["games"]
            if mpg >= min_mpg:
                qualified[pid] = pm["player_name"]

        # Players who played in tonight's game
        tonight_pids = set()
        players = self.playerbox.get(game_id, {}).get(team, [])
        for p in players:
            if p["min"] > 0:
                tonight_pids.add(p["player_id"])

        # Absences: qualified players who didn't play tonight
        absences = [
            name for pid, name in sorted(qualified.items()) if pid not in tonight_pids
        ]
        return absences

    def _h2h_this_season(
        self, home_team: str, away_team: str, before_date: str, season: str
    ) -> list[str]:
        """Find previous meetings between these teams this season."""
        results = []
        for g in self.all_games:
            if g["season"] != season or g["date"] >= before_date:
                continue
            h = g["home_norm"]
            a = g["away_norm"]
            if (h == home_team and a == away_team) or (
                h == away_team and a == home_team
            ):
                winner = h if g["home_score"] > g["away_score"] else a
                results.append(
                    f"{g['date']}: {a} @ {h} — "
                    f"{g['away_score']}-{g['home_score']} "
                    f"({winner} win)"
                )
        return results

    # ── Prompt assembly ──────────────────────────────────────────────────

    def _format_team_stats_table(self, stats: dict[str, dict[str, float]]) -> str:
        """Format rolling team stats as a markdown table."""
        stat_names = [
            "PPG",
            "Opp PPG",
            "FG%",
            "3P%",
            "FT%",
            "REB",
            "AST",
            "TOV",
            "STL",
            "BLK",
        ]
        lines = []
        lines.append("|           | Last 5  | Last 10 | Season  |")
        lines.append("|-----------|---------|---------|---------|")
        for stat in stat_names:
            vals = []
            for window in ["Last 5", "Last 10", "Season"]:
                v = stats.get(window, {}).get(stat, 0.0)
                if stat in ("FG%", "3P%", "FT%"):
                    vals.append(f"{v:.1f}%")
                else:
                    vals.append(f"{v:.1f}")
            lines.append(f"| {stat:<9} | {vals[0]:<7} | {vals[1]:<7} | {vals[2]:<7} |")
        return "\n".join(lines)

    def _format_recent_games_table(self, recent: list[dict]) -> str:
        """Format recent games as a markdown table."""
        if not recent:
            return "No recent games available."
        lines = []
        lines.append("| Date       | Opp | H/A | Result        | Margin |")
        lines.append("|------------|-----|-----|---------------|--------|")
        for r in recent:
            lines.append(
                f"| {r['date']} | {r['opp']:<3} | {r['ha']}   "
                f"| {r['score']:<13} | {r['margin']:<6} |"
            )
        return "\n".join(lines)

    def _format_players_table(self, players: list[dict]) -> str:
        """Format key players as a markdown table."""
        if not players:
            return "No player data available."
        lines = []
        lines.append(
            "| Player               | Pos | GP | MIN  | PTS  | REB  | AST  "
            "| STL | BLK | TOV | FG%  | 3P%  | FT%  | +/-  |"
        )
        lines.append(
            "|----------------------|-----|----|------|------|------|------"
            "|-----|-----|-----|------|------|------|------|"
        )
        for p in players:
            name = p["player_name"][:20]
            lines.append(
                f"| {name:<20} | {p['position']:<3} | {p['gp']:<2} "
                f"| {p['min']:<4.1f} | {p['pts']:<4.1f} | {p['reb']:<4.1f} "
                f"| {p['ast']:<4.1f} | {p['stl']:<3.1f} | {p['blk']:<3.1f} "
                f"| {p['tov']:<3.1f} | {p['fg_pct']:<4.1f} | {p['fg3_pct']:<4.1f} "
                f"| {p['ft_pct']:<4.1f} | {p['plus_minus']:+.1f} |"
            )
        return "\n".join(lines)

    def _format_roster_list(self, roster: list[dict]) -> str:
        """Format a list of players with positions."""
        if not roster:
            return "  (no data)"
        parts = []
        for p in roster:
            pos = p["position"] if p["position"] else "?"
            parts.append(f"{p['player_name']} ({pos})")
        return "  " + ", ".join(parts)

    def _team_full_name(self, abbrev: str) -> str:
        """Get full team name from abbreviation."""
        info = self.team_info["by_abbrev"].get(abbrev)
        if info:
            return info["full_name"]
        return abbrev

    def build_prompt(self, game: dict) -> str:
        """Build the full user prompt for a single game."""
        gid = game["game_id"]
        home = game["home_norm"]
        away = game["away_norm"]
        date = game["date"]
        season = game["season"]

        parts = []

        # Header
        parts.append(
            f"## Game: {away} @ {home}\n" f"Date: {date}\n" f"Season: {season}"
        )
        parts.append("---")

        # Build both teams' sections
        for side, team, opp in [
            ("HOME", home, away),
            ("AWAY", away, home),
        ]:
            record = self._team_record(team, date, season)
            rest = self._rest_days(team, date)
            stats = self._rolling_team_stats_with_season(team, date, season)
            recent = self._recent_games_table(team, date, n=10)
            players = self._key_players_table(team, date, n_games=10, top_n=10)
            full_name = self._team_full_name(team)

            parts.append(
                f"### {side}: {full_name} ({record['W']}-{record['L']})\n"
                f"Home Record: {record['home_W']}-{record['home_L']} | "
                f"Away Record: {record['away_W']}-{record['away_L']}\n"
                f"Last 10: {record['last_10_W']}-{record['last_10_L']}\n"
                f"Rest Days: {rest} | Streak: {record['streak']}"
            )

            parts.append("#### Team Stats (Rolling Averages)")
            parts.append(self._format_team_stats_table(stats))

            parts.append("#### Recent Games (Last 10)")
            parts.append(self._format_recent_games_table(recent))

            parts.append(f"#### Key Players (Last 10 Game Averages)")
            parts.append(self._format_players_table(players))

            parts.append("---")

        # Tonight's available players
        home_roster = self._tonight_roster(gid, home)
        away_roster = self._tonight_roster(gid, away)
        home_absences = self._notable_absences(gid, home, date, season)
        away_absences = self._notable_absences(gid, away, date, season)

        parts.append("### Tonight's Available Players")
        parts.append(f"HOME ({home}):")
        parts.append(self._format_roster_list(home_roster))
        parts.append(f"\nAWAY ({away}):")
        parts.append(self._format_roster_list(away_roster))

        parts.append(
            f"\nNotable Absences (averaged >15 MPG this season but NOT in tonight's game):"
        )
        parts.append(f"  HOME: {', '.join(home_absences) if home_absences else 'None'}")
        parts.append(f"  AWAY: {', '.join(away_absences) if away_absences else 'None'}")

        # Head-to-head this season
        h2h = self._h2h_this_season(home, away, date, season)
        parts.append("\n### Head-to-Head This Season")
        if h2h:
            for line in h2h:
                parts.append(f"  {line}")
        else:
            parts.append("  No previous meetings this season")

        return "\n\n".join(parts)


# =============================================================================
# OpenAI API helpers
# =============================================================================
def get_openai_client():
    """Create OpenAI client, checking for the package and API key."""
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai package not installed.\n" "Install with: pip install openai",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable not set.\n"
            "Export it with: export OPENAI_API_KEY='sk-...'",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenAI()


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token for English text)."""
    return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD based on token counts and model pricing."""
    pricing = PRICING.get(model, PRICING["gpt-5.4-nano"])
    input_cost = input_tokens / 1_000_000 * pricing["input"]
    output_cost = output_tokens / 1_000_000 * pricing["output"]
    return input_cost + output_cost


# =============================================================================
# Subcommands
# =============================================================================
def cmd_build_prompts(args: argparse.Namespace) -> None:
    """Build prompts for all games in the split and save to JSONL."""
    EXP7_DIR.mkdir(parents=True, exist_ok=True)

    seasons = VAL_SEASONS if args.split == "val" else TEST_SEASONS
    logger.info(f"Building prompts for {args.split} split: seasons {seasons}")

    builder = GameContextBuilder(DB_PATH, seasons)

    # Filter to evaluation games
    eval_games = [g for g in builder.all_games if g["season"] in seasons]
    logger.info(f"Found {len(eval_games)} games in {args.split} split")

    output_path = EXP7_DIR / f"{args.split}_prompts.jsonl"
    total_tokens = 0

    with open(output_path, "w") as f:
        for game in tqdm(eval_games, desc="Building prompts"):
            prompt = builder.build_prompt(game)
            tokens = estimate_tokens(SYSTEM_PROMPT + prompt)
            total_tokens += tokens
            record = {
                "game_id": game["game_id"],
                "date": game["date"],
                "season": game["season"],
                "home_team": game["home_norm"],
                "away_team": game["away_norm"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "prompt": prompt,
                "estimated_tokens": tokens,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(eval_games)} prompts to {output_path}")
    logger.info(f"Estimated total input tokens: {total_tokens:,}")
    for model in SUPPORTED_MODELS:
        # Assume ~200 output tokens per prediction
        cost = estimate_cost(total_tokens, len(eval_games) * 200, model)
        # Batch API is 50% off
        batch_cost = cost * 0.5
        logger.info(f"  {model}: ~${cost:.2f} standard, ~${batch_cost:.2f} batch")


def cmd_preview(args: argparse.Namespace) -> None:
    """Preview a few prompts without making API calls."""
    seasons = VAL_SEASONS if args.split == "val" else TEST_SEASONS
    builder = GameContextBuilder(DB_PATH, seasons)

    eval_games = [g for g in builder.all_games if g["season"] in seasons]

    if args.game_id:
        # Find specific game
        target = [g for g in eval_games if g["game_id"] == args.game_id]
        if not target:
            print(f"Game {args.game_id} not found in {args.split} split.")
            return
        eval_games = target
    else:
        # Take first N games, or random sample
        if args.random:
            import random

            random.seed(42)
            eval_games = random.sample(eval_games, min(args.n, len(eval_games)))
        else:
            eval_games = eval_games[: args.n]

    for i, game in enumerate(eval_games):
        prompt = builder.build_prompt(game)
        tokens = estimate_tokens(SYSTEM_PROMPT + prompt)
        print(f"\n{'=' * 80}")
        print(f"Game {i + 1}/{len(eval_games)}: {game['game_id']}")
        print(f"  {game['away_norm']} @ {game['home_norm']} on {game['date']}")
        print(f"  Actual: {game['home_score']}-{game['away_score']}")
        print(f"  Estimated tokens: {tokens:,}")
        print(f"{'=' * 80}")
        print()
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print()
        print("=== USER PROMPT ===")
        print(prompt)
        print()


def cmd_run_batch(args: argparse.Namespace) -> None:
    """Submit predictions to OpenAI Batch API for all games in the split."""
    client = get_openai_client()
    EXP7_DIR.mkdir(parents=True, exist_ok=True)

    # Load or build prompts
    prompts_path = EXP7_DIR / f"{args.split}_prompts.jsonl"
    if not prompts_path.exists():
        logger.info("Prompts not found, building them first...")
        # Build in place
        build_args = argparse.Namespace(split=args.split)
        cmd_build_prompts(build_args)

    # Load prompts
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # Build batch JSONL
    model = args.model
    batch_jsonl_path = EXP7_DIR / f"{args.split}_{model}_batch_input.jsonl"
    with open(batch_jsonl_path, "w") as f:
        for p in prompts:
            request = {
                "custom_id": p["game_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    "response_format": RESPONSE_FORMAT,
                    "temperature": 0.3,
                },
            }
            f.write(json.dumps(request) + "\n")

    logger.info(f"Wrote batch input to {batch_jsonl_path}")

    # Upload file
    logger.info("Uploading batch input file to OpenAI...")
    with open(batch_jsonl_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    logger.info(f"Uploaded file: {batch_input_file.id}")

    # Create batch
    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment": "exp7_llm_predict",
            "split": args.split,
            "model": model,
        },
    )
    logger.info(f"Batch created: {batch.id}")
    logger.info(f"Status: {batch.status}")

    # Save batch ID for later retrieval
    batch_info_path = EXP7_DIR / f"{args.split}_{model}_batch_info.json"
    with open(batch_info_path, "w") as f:
        json.dump(
            {
                "batch_id": batch.id,
                "input_file_id": batch_input_file.id,
                "model": model,
                "split": args.split,
                "n_games": len(prompts),
                "created_at": datetime.utcnow().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"Batch info saved to {batch_info_path}")

    if args.wait:
        logger.info("Waiting for batch completion (polling every 60s)...")
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(60)
            batch = client.batches.retrieve(batch.id)
            counts = batch.request_counts
            logger.info(
                f"  Status: {batch.status}, "
                f"completed: {counts.completed}/{counts.total}, "
                f"failed: {counts.failed}"
            )

        if batch.status == "completed":
            logger.info("Batch completed successfully!")
            _download_batch_results(client, batch, args.split, model)
        else:
            logger.error(f"Batch ended with status: {batch.status}")
            if batch.errors:
                for err in batch.errors.data:
                    logger.error(f"  Error: {err.code} — {err.message}")
    else:
        print(f"\nBatch submitted: {batch.id}")
        print(f"Check status with:")
        print(f"  python scripts/exp7_llm_predict.py check-batch --batch-id {batch.id}")
        print(f"Or wait and download with:")
        print(
            f"  python scripts/exp7_llm_predict.py download-batch "
            f"--batch-id {batch.id} "
            f"--output data/exp7/{args.split}_{model}_results.jsonl"
        )


def cmd_check_batch(args: argparse.Namespace) -> None:
    """Check the status of a batch job."""
    client = get_openai_client()
    batch = client.batches.retrieve(args.batch_id)
    counts = batch.request_counts

    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Total requests: {counts.total}")
    print(f"Completed: {counts.completed}")
    print(f"Failed: {counts.failed}")
    if batch.metadata:
        print(f"Metadata: {json.dumps(batch.metadata, indent=2)}")
    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file: {batch.error_file_id}")
    if batch.errors and batch.errors.data:
        print("Errors:")
        for err in batch.errors.data:
            print(f"  {err.code}: {err.message}")


def cmd_download_batch(args: argparse.Namespace) -> None:
    """Download completed batch results."""
    client = get_openai_client()
    batch = client.batches.retrieve(args.batch_id)

    if batch.status != "completed":
        print(f"Batch status is '{batch.status}', not 'completed'. Cannot download.")
        return

    model = (batch.metadata or {}).get("model", "unknown")
    split = (batch.metadata or {}).get("split", "unknown")
    _download_batch_results(client, batch, split, model, output_path=args.output)


def _download_batch_results(
    client,
    batch,
    split: str,
    model: str,
    output_path: str | None = None,
) -> None:
    """Download and parse batch results into our results format."""
    EXP7_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = str(EXP7_DIR / f"{split}_{model}_results.jsonl")

    # Load prompts to get ground truth
    prompts_path = EXP7_DIR / f"{split}_prompts.jsonl"
    ground_truth = {}
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                p = json.loads(line)
                ground_truth[p["game_id"]] = p

    # Download results
    logger.info(f"Downloading results from {batch.output_file_id}...")
    result_content = client.files.content(batch.output_file_id)
    raw_results = result_content.text.strip().split("\n")
    logger.info(f"Downloaded {len(raw_results)} results")

    # Parse and save
    total_input_tokens = 0
    total_output_tokens = 0
    parsed_count = 0
    error_count = 0

    with open(output_path, "w") as f:
        for line in raw_results:
            raw = json.loads(line)
            game_id = raw["custom_id"]
            gt = ground_truth.get(game_id, {})

            result = {
                "game_id": game_id,
                "model": model,
                "date": gt.get("date", ""),
                "season": gt.get("season", ""),
                "home_team": gt.get("home_team", ""),
                "away_team": gt.get("away_team", ""),
                "actual_home_score": gt.get("home_score", 0),
                "actual_away_score": gt.get("away_score", 0),
            }

            response = raw.get("response", {})
            if response.get("status_code") == 200:
                body = response["body"]
                usage = body.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

                content = body["choices"][0]["message"]["content"]
                try:
                    prediction = json.loads(content)
                    result["prediction"] = prediction
                    result["success"] = True
                    parsed_count += 1
                except json.JSONDecodeError:
                    result["prediction"] = None
                    result["success"] = False
                    result["error"] = f"JSON parse error: {content[:200]}"
                    error_count += 1
            else:
                result["prediction"] = None
                result["success"] = False
                result["error"] = f"HTTP {response.get('status_code')}"
                error_count += 1

            f.write(json.dumps(result) + "\n")

    cost = estimate_cost(total_input_tokens, total_output_tokens, model)
    # Batch API is 50% off standard pricing
    batch_cost = cost * 0.5

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Parsed: {parsed_count}, Errors: {error_count}")
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Estimated cost (batch 50% discount): ${batch_cost:.2f}")

    # Download error file if any
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = str(Path(output_path).with_suffix(".errors.jsonl"))
        with open(error_path, "w") as f:
            f.write(error_content.text)
        logger.info(f"Error details saved to {error_path}")


def cmd_run_standard(args: argparse.Namespace) -> None:
    """Run predictions via standard API with concurrency and live cost tracking."""
    client = get_openai_client()
    EXP7_DIR.mkdir(parents=True, exist_ok=True)

    # Load or build prompts
    prompts_path = EXP7_DIR / f"{args.split}_prompts.jsonl"
    if not prompts_path.exists():
        logger.info("Prompts not found, building them first...")
        build_args = argparse.Namespace(split=args.split)
        cmd_build_prompts(build_args)

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    if args.every_nth:
        prompts = prompts[:: args.every_nth]
    if args.limit:
        prompts = prompts[: args.limit]

    # Load existing results for resume support
    output_path = EXP7_DIR / f"{args.split}_{args.model}_results.jsonl"
    existing_ids = set()
    if output_path.exists() and args.resume:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["game_id"])
        logger.info(f"Resuming: found {len(existing_ids)} existing results")

    remaining = [p for p in prompts if p["game_id"] not in existing_ids]
    if not remaining:
        logger.info("All games already completed. Nothing to do.")
        return

    concurrency = args.concurrency
    logger.info(
        f"Running {len(remaining)} predictions via standard API "
        f"with model={args.model}, concurrency={concurrency}"
    )

    # Thread-safe counters and file writer
    lock = threading.Lock()
    stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "parsed": 0,
        "errors": 0,
        "cost": 0.0,
        "start_time": time.time(),
    }
    pricing = PRICING.get(args.model, PRICING["gpt-5.4-nano"])

    def _predict_one(p: dict) -> dict:
        """Make a single API call. Thread-safe."""
        game_id = p["game_id"]
        result = {
            "game_id": game_id,
            "model": args.model,
            "date": p["date"],
            "season": p["season"],
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "actual_home_score": p["home_score"],
            "actual_away_score": p["away_score"],
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    response_format=RESPONSE_FORMAT,
                    temperature=0.3,
                )

                in_tok = response.usage.prompt_tokens if response.usage else 0
                out_tok = response.usage.completion_tokens if response.usage else 0
                call_cost = (
                    in_tok / 1_000_000 * pricing["input"]
                    + out_tok / 1_000_000 * pricing["output"]
                )

                content = response.choices[0].message.content
                prediction = json.loads(content)
                result["prediction"] = prediction
                result["success"] = True
                result["input_tokens"] = in_tok
                result["output_tokens"] = out_tok
                result["cost"] = call_cost

                with lock:
                    stats["input_tokens"] += in_tok
                    stats["output_tokens"] += out_tok
                    stats["cost"] += call_cost
                    stats["parsed"] += 1

                return result

            except json.JSONDecodeError as e:
                result["prediction"] = None
                result["success"] = False
                result["error"] = f"JSON parse error: {e}"
                with lock:
                    stats["errors"] += 1
                return result

            except Exception as e:
                error_str = str(e)
                if (
                    "rate_limit" in error_str.lower() or "429" in error_str
                ) and attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        f"Rate limited on {game_id}. Retry {attempt+1}/{max_retries} in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                elif "timeout" in error_str.lower() and attempt < max_retries - 1:
                    logger.warning(
                        f"Timeout on {game_id}. Retry {attempt+1}/{max_retries}..."
                    )
                    time.sleep(5)
                    continue
                else:
                    result["prediction"] = None
                    result["success"] = False
                    result["error"] = error_str
                    with lock:
                        stats["errors"] += 1
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed after {max_retries} attempts for {game_id}: {error_str}"
                        )
                    return result

        return result  # shouldn't reach here

    # Run with thread pool
    results_buffer = []
    pbar = tqdm(total=len(remaining), desc=f"Predicting ({args.model})")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_predict_one, p): p for p in remaining}

        for future in as_completed(futures):
            result = future.result()
            results_buffer.append(result)
            pbar.update(1)

            # Live cost update in progress bar
            with lock:
                elapsed = time.time() - stats["start_time"]
                rate = stats["parsed"] / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    {
                        "cost": f"${stats['cost']:.3f}",
                        "ok": stats["parsed"],
                        "err": stats["errors"],
                        "tok": f"{(stats['input_tokens'] + stats['output_tokens']):,}",
                        "game/s": f"{rate:.1f}",
                    }
                )

    pbar.close()

    # Sort by game_id for deterministic output, then write
    results_buffer.sort(key=lambda r: r["game_id"])

    with open(output_path, "a" if args.resume else "w") as f:
        for result in results_buffer:
            f.write(json.dumps(result) + "\n")

    elapsed = time.time() - stats["start_time"]

    print(f"\n{'=' * 60}")
    print(f"  Experiment 7 — Standard API Run Complete")
    print(f"  Model: {args.model}")
    print(f"  Games: {stats['parsed']} succeeded, {stats['errors']} failed")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Throughput: {stats['parsed']/elapsed:.1f} games/sec")
    print(f"{'=' * 60}")
    print(f"  Input tokens:  {stats['input_tokens']:>12,}")
    print(f"  Output tokens: {stats['output_tokens']:>12,}")
    print(f"  Total tokens:  {stats['input_tokens'] + stats['output_tokens']:>12,}")
    print(f"  ACTUAL COST:   ${stats['cost']:>11.4f}")
    print(f"{'=' * 60}")
    print(f"  Results: {output_path}")
    print()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate predictions from a results file."""
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Filter to successful predictions
    successful = [r for r in results if r.get("success") and r.get("prediction")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        print("No successful predictions found.")
        return

    model = successful[0].get("model", "unknown")
    print(f"\n{'=' * 70}")
    print(f"  Phase 4 Experiment 7: LLM Prediction Evaluation")
    print(f"  Model: {model}")
    print(f"  Results file: {results_path}")
    print(
        f"  Total: {len(results)}, Successful: {len(successful)}, Failed: {len(failed)}"
    )
    print(f"{'=' * 70}")

    # Extract arrays
    pred_home = np.array([r["prediction"]["home_score"] for r in successful])
    pred_away = np.array([r["prediction"]["away_score"] for r in successful])
    pred_spreads = pred_home - pred_away
    pred_win_probs = np.array(
        [
            np.clip(r["prediction"]["home_win_probability"], 0.001, 0.999)
            for r in successful
        ]
    )

    true_home = np.array([r["actual_home_score"] for r in successful])
    true_away = np.array([r["actual_away_score"] for r in successful])
    true_spreads = true_home - true_away
    true_wins = (true_spreads > 0).astype(int)

    # --- Core Metrics ---

    # Spread MAE
    spread_mae = float(np.mean(np.abs(pred_spreads - true_spreads)))

    # Spread RMSE
    spread_rmse = float(np.sqrt(np.mean((pred_spreads - true_spreads) ** 2)))

    # Score MAE (individual team scores)
    home_score_mae = float(np.mean(np.abs(pred_home - true_home)))
    away_score_mae = float(np.mean(np.abs(pred_away - true_away)))

    # Total score MAE
    pred_totals = pred_home + pred_away
    true_totals = true_home + true_away
    total_mae = float(np.mean(np.abs(pred_totals - true_totals)))

    # Win AUC
    try:
        from sklearn.metrics import roc_auc_score

        win_auc = float(roc_auc_score(true_wins, pred_win_probs))
    except (ValueError, ImportError):
        win_auc = float("nan")

    # Win accuracy (from predicted spread sign)
    pred_wins_spread = (pred_spreads > 0).astype(int)
    win_acc_spread = float(np.mean(pred_wins_spread == true_wins))

    # Win accuracy (from win probability > 0.5)
    pred_wins_prob = (pred_win_probs > 0.5).astype(int)
    win_acc_prob = float(np.mean(pred_wins_prob == true_wins))

    # ECE (10-bin)
    ece = _compute_ece(pred_win_probs, true_wins, n_bins=10)

    # Spread bias (mean predicted - true spread)
    spread_bias = float(np.mean(pred_spreads - true_spreads))

    # Score biases
    home_bias = float(np.mean(pred_home - true_home))
    away_bias = float(np.mean(pred_away - true_away))

    print(f"\n--- Core Metrics ---")
    print(f"  Spread MAE:          {spread_mae:.2f}")
    print(f"  Spread RMSE:         {spread_rmse:.2f}")
    print(f"  Spread Bias:         {spread_bias:+.2f}")
    print(f"  Home Score MAE:      {home_score_mae:.2f}")
    print(f"  Away Score MAE:      {away_score_mae:.2f}")
    print(f"  Total Score MAE:     {total_mae:.2f}")
    print(f"  Win AUC:             {win_auc:.3f}")
    print(f"  Win Acc (spread):    {win_acc_spread:.1%}")
    print(f"  Win Acc (prob):      {win_acc_prob:.1%}")
    print(f"  ECE (10-bin):        {ece:.4f}")
    print(f"  Home Score Bias:     {home_bias:+.2f}")
    print(f"  Away Score Bias:     {away_bias:+.2f}")

    # --- Distribution Statistics ---
    print(f"\n--- Prediction Distribution ---")
    print(
        f"  Predicted home score: {np.mean(pred_home):.1f} +/- {np.std(pred_home):.1f} "
        f"(actual: {np.mean(true_home):.1f} +/- {np.std(true_home):.1f})"
    )
    print(
        f"  Predicted away score: {np.mean(pred_away):.1f} +/- {np.std(pred_away):.1f} "
        f"(actual: {np.mean(true_away):.1f} +/- {np.std(true_away):.1f})"
    )
    print(
        f"  Predicted spread:     {np.mean(pred_spreads):+.1f} +/- {np.std(pred_spreads):.1f} "
        f"(actual: {np.mean(true_spreads):+.1f} +/- {np.std(true_spreads):.1f})"
    )
    print(
        f"  Predicted win prob:   {np.mean(pred_win_probs):.3f} +/- {np.std(pred_win_probs):.3f} "
        f"(actual home win rate: {np.mean(true_wins):.3f})"
    )

    # --- Season-Split Analysis (Leakage Detection) ---
    seasons = sorted(set(r["season"] for r in successful))
    if len(seasons) > 1:
        print(f"\n--- Season-Split Analysis (Leakage Detection) ---")
        print(
            f"  {'Season':<12} {'N':>5} {'MAE':>7} {'RMSE':>7} "
            f"{'AUC':>7} {'WinAcc':>7} {'Bias':>7}"
        )
        print(f"  {'-' * 58}")
        for season in seasons:
            mask = np.array([r["season"] == season for r in successful])
            n = int(mask.sum())
            if n < 10:
                continue
            s_mae = float(np.mean(np.abs(pred_spreads[mask] - true_spreads[mask])))
            s_rmse = float(
                np.sqrt(np.mean((pred_spreads[mask] - true_spreads[mask]) ** 2))
            )
            s_bias = float(np.mean(pred_spreads[mask] - true_spreads[mask]))
            s_wins = true_wins[mask]
            s_probs = pred_win_probs[mask]
            try:
                s_auc = float(roc_auc_score(s_wins, s_probs))
            except (ValueError, NameError):
                s_auc = float("nan")
            s_pred_w = (pred_spreads[mask] > 0).astype(int)
            s_win_acc = float(np.mean(s_pred_w == s_wins))
            print(
                f"  {season:<12} {n:>5} {s_mae:>7.2f} {s_rmse:>7.2f} "
                f"{s_auc:>7.3f} {s_win_acc:>6.1%} {s_bias:>+7.2f}"
            )

        # Leakage warning
        if len(seasons) == 2:
            mask_0 = np.array([r["season"] == seasons[0] for r in successful])
            mask_1 = np.array([r["season"] == seasons[1] for r in successful])
            mae_0 = float(np.mean(np.abs(pred_spreads[mask_0] - true_spreads[mask_0])))
            mae_1 = float(np.mean(np.abs(pred_spreads[mask_1] - true_spreads[mask_1])))
            diff = mae_0 - mae_1
            if abs(diff) > 1.0:
                earlier = seasons[0]
                later = seasons[1]
                if diff < 0:
                    print(
                        f"\n  WARNING: {earlier} MAE ({mae_0:.2f}) is notably lower than "
                        f"{later} MAE ({mae_1:.2f})."
                    )
                    print(
                        f"  This MAY indicate data leakage — the LLM may have memorized "
                        f"{earlier} outcomes from its training data."
                    )
                else:
                    print(
                        f"\n  NOTE: {later} MAE ({mae_1:.2f}) is lower than "
                        f"{earlier} MAE ({mae_0:.2f})."
                    )
                    print(
                        f"  No leakage signal detected (older season is harder, not easier)."
                    )

    # --- Calibration Breakdown ---
    print(f"\n--- Calibration (10-bin) ---")
    print(f"  {'Bin':>12} {'Count':>6} {'Avg Pred':>9} {'Avg Actual':>11} {'Gap':>7}")
    print(f"  {'-' * 47}")
    bin_boundaries = np.linspace(0, 1, 11)
    for i in range(10):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (pred_win_probs >= lo) & (pred_win_probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_pred = float(pred_win_probs[mask].mean())
        avg_true = float(true_wins[mask].mean())
        gap = avg_pred - avg_true
        print(
            f"  [{lo:.1f}, {hi:.1f}) {n:>6} {avg_pred:>9.3f} {avg_true:>11.3f} {gap:>+7.3f}"
        )

    # --- Comparison ---
    print(f"\n--- Comparison with Other Models ---")
    print(f"  {'Model':<35} {'MAE':>7} {'AUC':>7} {'WinAcc':>7}")
    print(f"  {'-' * 58}")
    print(f"  {'Phase 3 Exp 9 (ensemble)':<35} {'10.66':>7} {'0.718':>7} {'66.5%':>7}")
    print(
        f"  {'Phase 3 Exp 4 (transformer)':<35} {'10.83':>7} {'0.705':>7} {'65.1%':>7}"
    )
    print(
        f"  {'Phase 4 Exp 4 (generative)':<35} {'11.76':>7} {'0.662':>7} {'61.4%':>7}"
    )
    print(
        f"  {f'Exp 7: {model}':<35} {spread_mae:>7.2f} "
        f"{win_auc:>7.3f} {win_acc_spread:>6.1%}"
    )
    print()


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare results across all models in the results directory."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Find all result files
    result_files = sorted(results_dir.glob("*_results.jsonl"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    print(f"\n{'=' * 80}")
    print(f"  Phase 4 Experiment 7: LLM Model Comparison")
    print(f"{'=' * 80}")

    summaries = []
    for rpath in result_files:
        results = []
        with open(rpath) as f:
            for line in f:
                results.append(json.loads(line))

        successful = [r for r in results if r.get("success") and r.get("prediction")]
        if not successful:
            continue

        model = successful[0].get("model", rpath.stem)
        split = rpath.stem.split("_")[
            0
        ]  # e.g., "test" from "test_gpt-5.4-nano_results"

        pred_spreads = np.array(
            [
                r["prediction"]["home_score"] - r["prediction"]["away_score"]
                for r in successful
            ]
        )
        true_spreads = np.array(
            [r["actual_home_score"] - r["actual_away_score"] for r in successful]
        )
        true_wins = (true_spreads > 0).astype(int)
        pred_win_probs = np.array(
            [
                np.clip(r["prediction"]["home_win_probability"], 0.001, 0.999)
                for r in successful
            ]
        )

        spread_mae = float(np.mean(np.abs(pred_spreads - true_spreads)))
        spread_rmse = float(np.sqrt(np.mean((pred_spreads - true_spreads) ** 2)))
        pred_wins = (pred_spreads > 0).astype(int)
        win_acc = float(np.mean(pred_wins == true_wins))
        ece = _compute_ece(pred_win_probs, true_wins, n_bins=10)

        try:
            from sklearn.metrics import roc_auc_score

            win_auc = float(roc_auc_score(true_wins, pred_win_probs))
        except (ValueError, ImportError):
            win_auc = float("nan")

        summaries.append(
            {
                "file": rpath.name,
                "split": split,
                "model": model,
                "n": len(successful),
                "mae": spread_mae,
                "rmse": spread_rmse,
                "auc": win_auc,
                "acc": win_acc,
                "ece": ece,
            }
        )

    if not summaries:
        print("No valid results found.")
        return

    print(
        f"\n  {'Model':<20} {'Split':>6} {'N':>6} {'MAE':>7} {'RMSE':>7} "
        f"{'AUC':>7} {'WinAcc':>7} {'ECE':>7}"
    )
    print(f"  {'-' * 73}")

    # Sort by MAE
    for s in sorted(summaries, key=lambda x: x["mae"]):
        print(
            f"  {s['model']:<20} {s['split']:>6} {s['n']:>6} "
            f"{s['mae']:>7.2f} {s['rmse']:>7.2f} "
            f"{s['auc']:>7.3f} {s['acc']:>6.1%} {s['ece']:>7.4f}"
        )

    # Reference models
    print(f"\n  --- Reference Models ---")
    print(
        f"  {'Phase 3 Exp 9':<20} {'test':>6} {'~2100':>6} "
        f"{'10.66':>7} {'—':>7} {'0.718':>7} {'66.5%':>7} {'0.0378':>7}"
    )
    print(
        f"  {'Phase 4 Exp 4':<20} {'test':>6} {'~2100':>6} "
        f"{'11.76':>7} {'—':>7} {'0.662':>7} {'61.4%':>7} {'—':>7}"
    )
    print()


# =============================================================================
# Utilities
# =============================================================================
def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_pred = probs[mask].mean()
            avg_true = labels[mask].mean()
            ece += (n_bin / n_total) * abs(avg_pred - avg_true)
    return float(ece)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Exp 7: LLM-based NBA game prediction via OpenAI GPT-5.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build-prompts --split test
  %(prog)s preview --split test --n 3
  %(prog)s run-batch --split test --model gpt-5.4-nano
  %(prog)s run-batch --split test --model gpt-5.4-nano --wait
  %(prog)s check-batch --batch-id batch_abc123
  %(prog)s download-batch --batch-id batch_abc123
  %(prog)s run-standard --split val --model gpt-5.4-nano --limit 10
  %(prog)s evaluate --results data/exp7/test_gpt-5.4-nano_results.jsonl
  %(prog)s compare --results-dir data/exp7/
        """,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # -- build-prompts --
    sp = subparsers.add_parser(
        "build-prompts", help="Build prompts for all games in a split"
    )
    sp.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Data split (default: test)",
    )

    # -- preview --
    sp = subparsers.add_parser("preview", help="Preview prompts without API calls")
    sp.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
    )
    sp.add_argument("--n", type=int, default=3, help="Number of prompts to preview")
    sp.add_argument("--game-id", type=str, help="Preview a specific game by ID")
    sp.add_argument(
        "--random", action="store_true", help="Sample randomly instead of first N"
    )

    # -- run-batch --
    sp = subparsers.add_parser(
        "run-batch", help="Submit predictions to OpenAI Batch API"
    )
    sp.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
    )
    sp.add_argument(
        "--model",
        default="gpt-5.4-nano",
        choices=SUPPORTED_MODELS,
        help="Model to use (default: gpt-5.4-nano)",
    )
    sp.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch completion (polls every 60s)",
    )

    # -- check-batch --
    sp = subparsers.add_parser("check-batch", help="Check status of a batch job")
    sp.add_argument("--batch-id", required=True, help="Batch ID to check")

    # -- download-batch --
    sp = subparsers.add_parser(
        "download-batch", help="Download completed batch results"
    )
    sp.add_argument("--batch-id", required=True, help="Batch ID to download")
    sp.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/exp7/{split}_{model}_results.jsonl)",
    )

    # -- run-standard --
    sp = subparsers.add_parser("run-standard", help="Run predictions via standard API")
    sp.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
    )
    sp.add_argument(
        "--model",
        default="gpt-5.4-nano",
        choices=SUPPORTED_MODELS,
    )
    sp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of games (for debugging)",
    )
    sp.add_argument(
        "--every-nth",
        type=int,
        default=None,
        dest="every_nth",
        help="Take every Nth game for even sampling (e.g., --every-nth 3)",
    )
    sp.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API calls (default: 10)",
    )
    sp.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )

    # -- evaluate --
    sp = subparsers.add_parser(
        "evaluate", help="Evaluate predictions from a results file"
    )
    sp.add_argument(
        "--results",
        required=True,
        help="Path to results JSONL file",
    )

    # -- compare --
    sp = subparsers.add_parser("compare", help="Compare results across all models")
    sp.add_argument(
        "--results-dir",
        default=str(EXP7_DIR),
        help="Directory containing result files (default: data/exp7/)",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    command_map = {
        "build-prompts": cmd_build_prompts,
        "preview": cmd_preview,
        "run-batch": cmd_run_batch,
        "check-batch": cmd_check_batch,
        "download-batch": cmd_download_batch,
        "run-standard": cmd_run_standard,
        "evaluate": cmd_evaluate,
        "compare": cmd_compare,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
