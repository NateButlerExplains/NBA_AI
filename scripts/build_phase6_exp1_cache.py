#!/usr/bin/env python
"""
Build auxiliary matchup-level features for Phase 6 Exp 1.

Computes 12 features per target game:
  Engineered (8, available for all games):
    0. home_win_pct_20    — home team rolling 20-game win%
    1. away_win_pct_20    — away team rolling 20-game win%
    2. home_form_5        — home team rolling 5-game win%
    3. away_form_5        — away team rolling 5-game win%
    4. home_home_wp       — home team's home win% this season
    5. away_away_wp       — away team's away win% this season
    6. season_progress    — fraction of season played (0-1)
    7. rest_diff          — (home_rest - away_rest) / 7.0

  Betting (4, available for 2021+ via covers/ESPN):
    8.  vegas_spread      — closing spread / 13.0
    9.  vegas_total       — (closing total - 216) / 20.0
    10. vegas_ml_home_prob — implied ML probability - 0.5
    11. has_betting        — binary indicator

Output: data/phase6_exp1_aux/aux_features.json
  {game_id: [12 floats]}

Usage:
    python scripts/build_phase6_exp1_cache.py
"""

import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "NBA_AI_full.sqlite")
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase6_exp1_aux"

HISTORICAL_MAP = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
}

# Feature indices
N_FEATURES = 12

logger = logging.getLogger(__name__)


def ml_to_implied_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml is None:
        return 0.5
    try:
        ml = float(ml)
    except (ValueError, TypeError):
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    elif ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    return 0.5


def load_games(db_path: str) -> list[dict]:
    """Load all completed regular season games ordered by date."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT game_id, date_time_utc, home_team, away_team, season, season_type
        FROM Games
        WHERE status = 3
          AND season_type IN ('Regular Season', 'Post Season')
        ORDER BY date_time_utc
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Normalize historical teams
    for r in rows:
        r["home_team"] = HISTORICAL_MAP.get(r["home_team"], r["home_team"])
        r["away_team"] = HISTORICAL_MAP.get(r["away_team"], r["away_team"])

    return rows


def load_game_results(db_path: str) -> dict:
    """Load game_id -> (home_score, away_score) for win/loss determination."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT g.game_id, g.home_team, g.away_team,
               MAX(CASE WHEN tb.team_id = t_home.team_id THEN tb.pts END) as home_pts,
               MAX(CASE WHEN tb.team_id = t_away.team_id THEN tb.pts END) as away_pts
        FROM Games g
        JOIN Teams t_home ON g.home_team = t_home.abbreviation
        JOIN Teams t_away ON g.away_team = t_away.abbreviation
        JOIN TeamBox tb ON tb.game_id = g.game_id
            AND tb.team_id IN (t_home.team_id, t_away.team_id)
        WHERE g.status = 3
        GROUP BY g.game_id
    """)
    results = {}
    for row in cur.fetchall():
        game_id, home, away, home_pts, away_pts = row
        if home_pts is not None and away_pts is not None:
            results[game_id] = (float(home_pts), float(away_pts))
    conn.close()
    return results


def load_betting(db_path: str) -> dict:
    """Load betting data keyed by game_id."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT game_id,
               COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread) as spread,
               COALESCE(espn_closing_total, covers_closing_total, espn_current_total) as total,
               COALESCE(espn_closing_ml_home, espn_current_ml_home) as ml_home,
               COALESCE(espn_closing_ml_away, espn_current_ml_away) as ml_away
        FROM Betting
        WHERE COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread) IS NOT NULL
    """)
    betting = {}
    for row in cur.fetchall():
        game_id, spread, total, ml_home, ml_away = row
        betting[game_id] = {
            "spread": spread,
            "total": total,
            "ml_home_prob": ml_to_implied_prob(ml_home),
        }
    conn.close()
    return betting


def compute_rest_days(games: list[dict]) -> dict:
    """Compute rest days for each team before each game."""
    # Track last game date per team
    last_game = {}  # team -> date string
    rest = {}  # game_id -> {"home": int, "away": int}

    for g in games:
        game_id = g["game_id"]
        date = g["date_time_utc"][:10]  # YYYY-MM-DD
        home = g["home_team"]
        away = g["away_team"]

        home_rest = 3  # default
        away_rest = 3
        if home in last_game:
            delta = _date_diff(last_game[home], date)
            home_rest = min(max(delta - 1, 0), 14)  # rest = days between - 1, capped
        if away in last_game:
            delta = _date_diff(last_game[away], date)
            away_rest = min(max(delta - 1, 0), 14)

        rest[game_id] = {"home": home_rest, "away": away_rest}
        last_game[home] = date
        last_game[away] = date

    return rest


def _date_diff(d1: str, d2: str) -> int:
    """Days between two YYYY-MM-DD date strings."""
    from datetime import datetime

    dt1 = datetime.strptime(d1, "%Y-%m-%d")
    dt2 = datetime.strptime(d2, "%Y-%m-%d")
    return (dt2 - dt1).days


def build_aux_features(games: list[dict], results: dict, betting: dict) -> dict:
    """Build 12 auxiliary features for each game."""
    # Per-team rolling state trackers
    # team -> list of (is_home, won) tuples in chronological order
    team_history = defaultdict(list)  # team -> [(won, is_home)]
    season_records = defaultdict(
        lambda: {
            "wins": 0,
            "losses": 0,
            "home_wins": 0,
            "home_losses": 0,
            "away_wins": 0,
            "away_losses": 0,
        }
    )
    season_game_counts = defaultdict(int)  # season -> count of games played

    # Count total games per season for season_progress
    games_per_season = defaultdict(int)
    for g in games:
        games_per_season[g["season"]] += 1

    rest_data = compute_rest_days(games)

    current_season = {}  # team -> current season string
    aux = {}

    for g in games:
        game_id = g["game_id"]
        home = g["home_team"]
        away = g["away_team"]
        season = g["season"]

        # Reset season records if new season
        for team in [home, away]:
            if current_season.get(team) != season:
                current_season[team] = season
                season_records[team] = {
                    "wins": 0,
                    "losses": 0,
                    "home_wins": 0,
                    "home_losses": 0,
                    "away_wins": 0,
                    "away_losses": 0,
                }

        # Count this game for season progress
        season_game_counts[season] += 1

        features = [0.0] * N_FEATURES

        # --- Engineered features (using data BEFORE this game) ---

        # 0-1: Rolling 20-game win%
        home_hist = team_history[home]
        away_hist = team_history[away]
        features[0] = _rolling_win_pct(home_hist, 20) - 0.5  # center at 0
        features[1] = _rolling_win_pct(away_hist, 20) - 0.5

        # 2-3: Rolling 5-game form
        features[2] = _rolling_win_pct(home_hist, 5) - 0.5
        features[3] = _rolling_win_pct(away_hist, 5) - 0.5

        # 4-5: Home/away splits
        hr = season_records[home]
        ar = season_records[away]
        features[4] = (
            _safe_ratio(hr["home_wins"], hr["home_wins"] + hr["home_losses"]) - 0.5
        )
        features[5] = (
            _safe_ratio(ar["away_wins"], ar["away_wins"] + ar["away_losses"]) - 0.5
        )

        # 6: Season progress
        total = games_per_season.get(season, 1230)
        features[6] = min(season_game_counts[season] / total, 1.0) - 0.5  # center at 0

        # 7: Rest differential
        r = rest_data.get(game_id, {"home": 1, "away": 1})
        features[7] = (r["home"] - r["away"]) / 7.0

        # --- Betting features ---
        bet = betting.get(game_id)
        if bet is not None:
            features[8] = bet["spread"] / 13.0 if bet["spread"] is not None else 0.0
            features[9] = (
                (bet["total"] - 216.0) / 20.0 if bet["total"] is not None else 0.0
            )
            features[10] = bet["ml_home_prob"] - 0.5
            features[11] = 1.0
        # else: features 8-11 remain 0.0

        aux[game_id] = features

        # --- Update trackers AFTER computing features (no leakage) ---
        if game_id in results:
            home_pts, away_pts = results[game_id]
            home_won = home_pts > away_pts

            team_history[home].append(1.0 if home_won else 0.0)
            team_history[away].append(0.0 if home_won else 1.0)

            if home_won:
                season_records[home]["wins"] += 1
                season_records[home]["home_wins"] += 1
                season_records[away]["losses"] += 1
                season_records[away]["away_losses"] += 1
            else:
                season_records[away]["wins"] += 1
                season_records[away]["away_wins"] += 1
                season_records[home]["losses"] += 1
                season_records[home]["home_losses"] += 1

    return aux


def _rolling_win_pct(history: list[float], window: int) -> float:
    """Compute win% over last `window` games. Returns 0.5 if no history."""
    if not history:
        return 0.5
    recent = history[-window:]
    return sum(recent) / len(recent)


def _safe_ratio(num: int, denom: int) -> float:
    """Safe division, returns 0.5 if denom is 0."""
    return num / denom if denom > 0 else 0.5


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Loading games...")
    games = load_games(DB_PATH)
    logger.info(f"  {len(games)} completed games")

    logger.info("Loading game results...")
    results = load_game_results(DB_PATH)
    logger.info(f"  {len(results)} games with scores")

    logger.info("Loading betting data...")
    betting = load_betting(DB_PATH)
    logger.info(f"  {len(betting)} games with betting lines")

    logger.info("Computing auxiliary features...")
    aux = build_aux_features(games, results, betting)
    logger.info(f"  {len(aux)} games with features")

    # Stats
    n_betting = sum(1 for v in aux.values() if v[11] > 0)
    logger.info(
        f"  {n_betting} games with betting features ({n_betting/len(aux)*100:.1f}%)"
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "aux_features.json"
    with open(output_path, "w") as f:
        json.dump(aux, f)
    logger.info(f"Saved to {output_path}")

    # Quick sanity check
    sample_ids = list(aux.keys())[-5:]
    for gid in sample_ids:
        feats = aux[gid]
        bet_flag = "BET" if feats[11] > 0 else "   "
        logger.info(
            f"  {gid}: wp20=[{feats[0]:+.2f},{feats[1]:+.2f}] "
            f"form5=[{feats[2]:+.2f},{feats[3]:+.2f}] "
            f"splits=[{feats[4]:+.2f},{feats[5]:+.2f}] "
            f"prog={feats[6]+0.5:.2f} rest={feats[7]:+.2f} "
            f"{bet_flag} spread={feats[8]*13:.1f}"
        )


if __name__ == "__main__":
    main()
