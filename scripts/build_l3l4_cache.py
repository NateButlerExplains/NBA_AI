"""
Build L3 (Team Model) + L4 (Game Context) feature cache.

Pre-computes per-game team features (34-d × 2 teams) and game context (14-d)
from Games, TeamBox, and PlayerBox tables.

Output:
    data/l3l4_cache/
        team_features.npz    — {game_id: (2, 34)} home [0] and away [1]
        game_context.npz     — {game_id: (14,)}
        metadata.json        — feature names, normalization stats, date ranges
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "l3l4_cache"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Historical team abbreviation → modern abbreviation
HISTORICAL_TO_MODERN = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHH": "CHA",
    "VAN": "MEM",
    "WSB": "WAS",
    "SDC": "LAC",
    "KCK": "SAC",
    "BUF": "LAC",
    "NOJ": "NOP",
    "CLT": "CHA",
}

ALTITUDE_MAP = {
    "DEN": 5280,
    "UTA": 4226,
    "SLC": 4226,
    "PHX": 1086,
    "OKC": 1201,
    "ATL": 1050,
    "SAS": 650,
    "MEM": 337,
    "DAL": 430,
    "HOU": 80,
    "MIN": 830,
    "MIL": 617,
    "IND": 715,
    "CLE": 653,
    "DET": 600,
    "CHI": 594,
    "CHA": 751,
    "CLT": 751,
    "NOP": 7,
    "NOH": 7,
    "POR": 50,
    "SAC": 30,
    "LAL": 233,
    "LAC": 233,
    "GSW": 0,
    "BOS": 141,
    "NYK": 33,
    "BKN": 33,
    "NJN": 33,
    "PHI": 39,
    "TOR": 249,
    "WAS": 0,
    "ORL": 82,
    "MIA": 6,
    "SEA": 0,
    "VAN": 0,
    "NOK": 7,
    "CHH": 751,
}

# L3 Team feature names (34-d)
L3_ROLLING_FEATURES_PER_WINDOW = [
    "eFG_pct_off",
    "TOV_pct_off",
    "ORB_pct_off",
    "FTR_off",
    "eFG_pct_def",
    "TOV_pct_def",
    "ORB_pct_def",
    "FTR_def",
    "ORtg",
    "DRtg",
    "NetRtg",
    "Pace",
]

L3_FEATURE_NAMES = (
    [f"{feat}_5g" for feat in L3_ROLLING_FEATURES_PER_WINDOW]
    + [f"{feat}_15g" for feat in L3_ROLLING_FEATURES_PER_WINDOW]
    + [
        "coach_tenure_games",
        "coach_career_winpct",
        "is_new_coach_flag",
        "games_since_coach_change",
    ]
    + [
        "opp_3PA_rate",
        "opp_paint_pts_rate",
        "steal_rate",
        "block_rate",
    ]
    + [
        "roster_continuity",
        "multi_year_trend_3yr",
    ]
)
assert len(L3_FEATURE_NAMES) == 34, f"Expected 34, got {len(L3_FEATURE_NAMES)}"

# L4 Game context feature names (14-d)
L4_FEATURE_NAMES = [
    "home_flag",
    "arena_altitude",
    "rest_days_home",
    "rest_days_away",
    "is_b2b_home",
    "is_b2b_away",
    "games_7d_home",
    "games_7d_away",
    "travel_dist_home",
    "travel_dist_away",
    "tz_crossings_home",
    "tz_crossings_away",
    "season_progress",
    "is_playoffs",
]
assert len(L4_FEATURE_NAMES) == 14, f"Expected 14, got {len(L4_FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_team_id_map(conn) -> dict[str, str]:
    """Load team_id (str) → abbreviation mapping from Teams table."""
    cursor = conn.execute("SELECT team_id, abbreviation FROM Teams")
    return {str(row[0]): row[1] for row in cursor}


def load_game_data(conn, limit: int | None = None) -> pd.DataFrame:
    """
    Load all completed games with TeamBox stats joined, plus PlayerBox OREB/DREB aggregation.

    Returns a DataFrame with one row per team per game (2 rows per game).
    """
    logger.info("Loading games and TeamBox data...")
    t0 = time.time()

    # Main query: TeamBox joined with Games
    query = """
        SELECT
            tb.game_id,
            tb.team_id,
            tb.pts,
            tb.pts_allowed,
            tb.fga,
            tb.fgm,
            tb.fg3a,
            tb.fg3m,
            tb.fta,
            tb.ftm,
            tb.tov,
            tb.stl,
            tb.blk,
            tb.reb,
            g.home_team,
            g.away_team,
            g.date_time_utc,
            g.season,
            g.season_type
        FROM TeamBox tb
        JOIN Games g ON tb.game_id = g.game_id
        WHERE g.status = 3
          AND g.season_type IN ('Regular Season', 'Post Season')
        ORDER BY g.date_time_utc
    """
    if limit:
        query += f" LIMIT {limit * 2}"  # 2 rows per game

    rows = conn.execute(query).fetchall()
    cols = [
        "game_id",
        "team_id",
        "pts",
        "pts_allowed",
        "fga",
        "fgm",
        "fg3a",
        "fg3m",
        "fta",
        "ftm",
        "tov",
        "stl",
        "blk",
        "reb",
        "home_team",
        "away_team",
        "date_time_utc",
        "season",
        "season_type",
    ]
    df = pd.DataFrame(rows, columns=cols)
    logger.info(
        f"  Loaded {len(df)} TeamBox rows ({len(df)//2} games) in {time.time()-t0:.1f}s"
    )

    return df


def load_playerbox_rebounds(conn, limit: int | None = None) -> pd.DataFrame:
    """
    Aggregate OREB and DREB from PlayerBox per team per game.
    TeamBox lacks these columns so we must aggregate from PlayerBox.
    """
    logger.info("Loading PlayerBox OREB/DREB aggregation...")
    t0 = time.time()

    query = """
        SELECT pb.game_id, pb.team_id, SUM(pb.oreb) as team_oreb, SUM(pb.dreb) as team_dreb
        FROM PlayerBox pb
        JOIN Games g ON pb.game_id = g.game_id
        WHERE pb.min > 0
          AND g.status = 3
          AND g.season_type IN ('Regular Season', 'Post Season')
        GROUP BY pb.game_id, pb.team_id
    """
    rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["game_id", "team_id", "team_oreb", "team_dreb"])
    logger.info(f"  Loaded {len(df)} OREB/DREB rows in {time.time()-t0:.1f}s")
    return df


def load_roster_minutes(conn) -> pd.DataFrame:
    """
    Load per-player per-team per-season minutes for roster continuity calculation.
    """
    logger.info("Loading roster minutes by season...")
    t0 = time.time()

    query = """
        SELECT pb.player_id, pb.team_id, g.season, SUM(pb.min) as total_min
        FROM PlayerBox pb
        JOIN Games g ON pb.game_id = g.game_id
        WHERE pb.min > 0
          AND g.status = 3
          AND g.season_type = 'Regular Season'
        GROUP BY pb.player_id, pb.team_id, g.season
    """
    rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["player_id", "team_id", "season", "total_min"])
    logger.info(
        f"  Loaded {len(df)} player-team-season records in {time.time()-t0:.1f}s"
    )
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_per_game_raw_stats(
    df: pd.DataFrame, reb_df: pd.DataFrame, team_id_map: dict[str, str]
) -> pd.DataFrame:
    """
    Prepare per-team-per-game raw stats needed for rolling features.

    Adds columns: team_abbr, is_home, opponent stats, OREB/DREB, possessions, etc.
    """
    logger.info("Computing per-game raw stats...")
    t0 = time.time()

    # Map team_id to abbreviation
    df = df.copy()
    df["team_id_str"] = df["team_id"].astype(str)
    df["team_abbr"] = df["team_id_str"].map(team_id_map)

    mask_no_match = df["team_abbr"].isna()
    if mask_no_match.any():
        logger.warning(f"  {mask_no_match.sum()} rows with unmapped team_id")

    # Determine if home or away. Need to handle historical abbreviations:
    # Games.home_team may be "NJN" but Teams table has "BKN"
    # Build reverse map: modern → set of historical abbreviations
    modern_to_historical = defaultdict(set)
    for hist, mod in HISTORICAL_TO_MODERN.items():
        modern_to_historical[mod].add(hist)

    def _is_home(row):
        abbr = row["team_abbr"]
        home = row["home_team"]
        if abbr == home:
            return True
        # Check if home_team is a historical name for this team
        if home in modern_to_historical.get(abbr, set()):
            return True
        return False

    df["is_home"] = df.apply(_is_home, axis=1)

    # Merge OREB/DREB from PlayerBox
    reb_df = reb_df.copy()
    reb_df["team_id"] = reb_df["team_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df = df.merge(reb_df, on=["game_id", "team_id"], how="left")
    df["team_oreb"] = df["team_oreb"].fillna(0)
    df["team_dreb"] = df["team_dreb"].fillna(0)

    # Self-join to get opponent stats within the same game
    # For each row, find the other team in the same game
    df_opp = df[
        [
            "game_id",
            "team_id",
            "pts",
            "fga",
            "fgm",
            "fg3a",
            "fg3m",
            "fta",
            "ftm",
            "tov",
            "stl",
            "blk",
            "team_oreb",
            "team_dreb",
        ]
    ].copy()
    df_opp.columns = [
        "game_id",
        "opp_team_id",
        "opp_pts",
        "opp_fga",
        "opp_fgm",
        "opp_fg3a",
        "opp_fg3m",
        "opp_fta",
        "opp_ftm",
        "opp_tov",
        "opp_stl",
        "opp_blk",
        "opp_oreb",
        "opp_dreb",
    ]

    # Merge: for each (game_id, team_id), get opponent where opp_team_id != team_id
    df = df.merge(df_opp, on="game_id", how="inner")
    df = df[df["team_id"] != df["opp_team_id"]].copy()

    # Parse date for sorting
    df["date"] = pd.to_datetime(df["date_time_utc"].str[:10], format="%Y-%m-%d")

    # Compute raw metrics needed for rolling features
    # Possessions estimate: FGA - OREB + TOV + 0.44*FTA
    df["poss"] = df["fga"] - df["team_oreb"] + df["tov"] + 0.44 * df["fta"]
    df["opp_poss"] = (
        df["opp_fga"] - df["opp_oreb"] + df["opp_tov"] + 0.44 * df["opp_fta"]
    )
    # Floor possessions at 1 to avoid division by zero
    df["poss"] = df["poss"].clip(lower=1)
    df["opp_poss"] = df["opp_poss"].clip(lower=1)

    # Per-game raw metrics (these will be rolling-averaged)
    # Offense
    df["eFG_off"] = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].clip(lower=1)
    df["TOV_off"] = df["tov"] / (df["fga"] + 0.44 * df["fta"] + df["tov"]).clip(lower=1)
    df["ORB_off"] = df["team_oreb"] / (df["team_oreb"] + df["opp_dreb"]).clip(lower=1)
    df["FTR_off"] = df["fta"] / df["fga"].clip(lower=1)

    # Defense (opponent's offensive stats from this game)
    df["eFG_def"] = (df["opp_fgm"] + 0.5 * df["opp_fg3m"]) / df["opp_fga"].clip(lower=1)
    df["TOV_def"] = df["opp_tov"] / (
        df["opp_fga"] + 0.44 * df["opp_fta"] + df["opp_tov"]
    ).clip(lower=1)
    df["ORB_def"] = df["opp_oreb"] / (df["opp_oreb"] + df["team_dreb"]).clip(lower=1)
    df["FTR_def"] = df["opp_fta"] / df["opp_fga"].clip(lower=1)

    # Ratings
    df["ORtg"] = 100.0 * df["pts"] / df["poss"]
    df["DRtg"] = 100.0 * df["opp_pts"] / df["opp_poss"]
    df["NetRtg"] = df["ORtg"] - df["DRtg"]
    df["Pace"] = df["poss"]  # possessions per game (not per 48 min, keep simple)

    # Defensive scheme proxies
    df["opp_3PA_rate"] = df["opp_fg3a"] / df["opp_fga"].clip(lower=1)
    # Approximate: 2pt made = opp_fgm - opp_fg3m, paint pts ≈ 2 * 2pt_made
    opp_2pt_made = (df["opp_fgm"] - df["opp_fg3m"]).clip(lower=0)
    df["opp_paint_pts_rate"] = (2 * opp_2pt_made) / df["opp_pts"].clip(lower=1)
    df["steal_rate"] = df["stl"] / df["opp_poss"]
    df["block_rate"] = df["blk"] / df["opp_fga"].clip(lower=1)

    # Win indicator for multi-year trend
    df["won"] = (df["pts"] > df["opp_pts"]).astype(float)

    logger.info(
        f"  Computed raw stats for {len(df)} team-game rows in {time.time()-t0:.1f}s"
    )
    return df


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dual-scale rolling window features (5-game and 15-game) per team.

    CRITICAL: Uses only past data (shift(1) before rolling) to prevent leakage.
    """
    logger.info("Computing rolling features...")
    t0 = time.time()

    # Columns to roll
    raw_cols = [
        "eFG_off",
        "TOV_off",
        "ORB_off",
        "FTR_off",
        "eFG_def",
        "TOV_def",
        "ORB_def",
        "FTR_def",
        "ORtg",
        "DRtg",
        "NetRtg",
        "Pace",
    ]

    defense_cols = ["opp_3PA_rate", "opp_paint_pts_rate", "steal_rate", "block_rate"]

    # Sort by team and date
    df = df.sort_values(["team_abbr", "date"]).copy()

    # Group by team and compute rolling means with shift to avoid leakage
    results = []

    for team, group in df.groupby("team_abbr"):
        group = group.sort_values("date").copy()

        # Shift by 1 so game N uses games 0..N-1
        for col in raw_cols:
            shifted = group[col].shift(1)
            group[f"{col}_5g"] = shifted.rolling(window=5, min_periods=1).mean()
            group[f"{col}_15g"] = shifted.rolling(window=15, min_periods=1).mean()

        # Defensive scheme proxies (15-game only, shifted)
        for col in defense_cols:
            shifted = group[col].shift(1)
            group[f"{col}_15g"] = shifted.rolling(window=15, min_periods=1).mean()

        results.append(group)

    df = pd.concat(results, ignore_index=True)

    # Fill NaN from shift (first game per team has no history) with league-avg defaults
    rolling_defaults = {
        "eFG_off": 0.50,
        "TOV_off": 0.13,
        "ORB_off": 0.25,
        "FTR_off": 0.27,
        "eFG_def": 0.50,
        "TOV_def": 0.13,
        "ORB_def": 0.25,
        "FTR_def": 0.27,
        "ORtg": 110.0,
        "DRtg": 110.0,
        "NetRtg": 0.0,
        "Pace": 95.0,
    }
    for col, default in rolling_defaults.items():
        df[f"{col}_5g"] = df[f"{col}_5g"].fillna(default)
        df[f"{col}_15g"] = df[f"{col}_15g"].fillna(default)

    defense_defaults = {
        "opp_3PA_rate": 0.35,
        "opp_paint_pts_rate": 0.40,
        "steal_rate": 0.08,
        "block_rate": 0.05,
    }
    for col, default in defense_defaults.items():
        df[f"{col}_15g"] = df[f"{col}_15g"].fillna(default)

    logger.info(f"  Rolling features computed in {time.time()-t0:.1f}s")
    return df


def compute_roster_continuity(
    roster_df: pd.DataFrame, team_id_map: dict[str, str]
) -> dict[tuple[str, str], float]:
    """
    Compute roster continuity per team per season.

    Roster continuity = fraction of current season's minutes played by players
    who were also on the team in the previous season.

    Returns: {(team_abbr, season): continuity_fraction}
    """
    logger.info("Computing roster continuity...")
    t0 = time.time()

    roster_df = roster_df.copy()
    roster_df["team_id"] = roster_df["team_id"].astype(str)
    roster_df["team_abbr"] = roster_df["team_id"].map(team_id_map)
    roster_df = roster_df.dropna(subset=["team_abbr"])

    # Get sorted unique seasons
    seasons = sorted(roster_df["season"].unique())

    # Build lookup: (team_abbr, season) → set of player_ids
    team_season_players = {}
    for (team, season), grp in roster_df.groupby(["team_abbr", "season"]):
        team_season_players[(team, season)] = set(grp["player_id"].values)

    # Build lookup: (team_abbr, season) → {player_id: total_min}
    team_season_minutes = {}
    for (team, season), grp in roster_df.groupby(["team_abbr", "season"]):
        team_season_minutes[(team, season)] = dict(
            zip(grp["player_id"], grp["total_min"])
        )

    result = {}
    for i, season in enumerate(seasons):
        if i == 0:
            # First season: no previous data → continuity = 0.5 (neutral)
            for team in roster_df[roster_df["season"] == season]["team_abbr"].unique():
                result[(team, season)] = 0.5
            continue

        prev_season = seasons[i - 1]
        for team in roster_df[roster_df["season"] == season]["team_abbr"].unique():
            curr_minutes = team_season_minutes.get((team, season), {})
            prev_players = team_season_players.get((team, prev_season), set())

            # Also check historical abbreviations
            modern = HISTORICAL_TO_MODERN.get(team, team)
            if team != modern:
                prev_players |= team_season_players.get((modern, prev_season), set())

            total_min = sum(curr_minutes.values())
            if total_min == 0:
                result[(team, season)] = 0.5
                continue

            returning_min = sum(
                mins for pid, mins in curr_minutes.items() if pid in prev_players
            )
            result[(team, season)] = returning_min / total_min

    logger.info(
        f"  Computed continuity for {len(result)} team-seasons in {time.time()-t0:.1f}s"
    )
    return result


def compute_multi_year_trend(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """
    Compute 3-year rolling win percentage per team per season.

    Returns: {(team_abbr, season): win_pct_3yr}
    """
    logger.info("Computing multi-year win trends...")
    t0 = time.time()

    # Only use regular season
    reg = df[df["season_type"] == "Regular Season"].copy()

    # Compute per-team per-season win count
    season_records = (
        reg.groupby(["team_abbr", "season"])
        .agg(
            wins=("won", "sum"),
            games=("won", "count"),
        )
        .reset_index()
    )
    season_records["win_pct"] = season_records["wins"] / season_records["games"].clip(
        lower=1
    )

    # Sort seasons and compute 3-year rolling average
    result = {}
    for team, grp in season_records.groupby("team_abbr"):
        grp = grp.sort_values("season").copy()
        # Use up to 3 previous seasons (shifted, no leakage)
        grp["trend_3yr"] = (
            grp["win_pct"].shift(1).rolling(window=3, min_periods=1).mean()
        )
        # Fill first season with 0.5
        grp["trend_3yr"] = grp["trend_3yr"].fillna(0.5)
        for _, row in grp.iterrows():
            result[(row["team_abbr"], row["season"])] = row["trend_3yr"]

    logger.info(
        f"  Computed trends for {len(result)} team-seasons in {time.time()-t0:.1f}s"
    )
    return result


def compute_season_game_number(df: pd.DataFrame) -> pd.DataFrame:
    """Add season_game_number column: the team's Nth game of the season."""
    df = df.sort_values(["team_abbr", "season", "date"]).copy()
    df["season_game_num"] = df.groupby(["team_abbr", "season"]).cumcount() + 1
    return df


def compute_rest_and_schedule(df: pd.DataFrame) -> dict[str, dict]:
    """
    Compute rest days, back-to-back flag, and games-in-7-days per team per game.

    Returns: {game_id: {"rest_home": float, "rest_away": float, "b2b_home": float,
                         "b2b_away": float, "g7d_home": float, "g7d_away": float}}
    """
    logger.info("Computing rest and schedule features...")
    t0 = time.time()

    # Sort chronologically per team
    df = df.sort_values(["team_abbr", "date"]).copy()

    # Build per-team date list for efficient lookups using groupby
    team_dates = {}  # team_abbr → [(date, game_id, is_home), ...]
    for team, grp in df.groupby("team_abbr"):
        team_dates[team] = list(zip(grp["date"], grp["game_id"], grp["is_home"]))

    # Compute per team-game
    # {game_id: {"rest_home": ..., "rest_away": ..., ...}}
    result = defaultdict(
        lambda: {
            "rest_home": 3.0,
            "rest_away": 3.0,
            "b2b_home": 0.0,
            "b2b_away": 0.0,
            "g7d_home": 1.0,
            "g7d_away": 1.0,
        }
    )

    for team, dates in team_dates.items():
        dates_sorted = sorted(dates, key=lambda x: x[0])
        date_list = [d[0] for d in dates_sorted]

        for i, (date, game_id, is_home) in enumerate(dates_sorted):
            # Rest days
            if i > 0:
                rest = (date - date_list[i - 1]).days
            else:
                rest = 3  # first game of career/season

            rest = min(rest, 7)  # cap at 7

            # Back-to-back
            is_b2b = 1.0 if rest <= 1 else 0.0

            # Games in last 7 days (excluding current)
            count_7d = 0
            for j in range(i - 1, max(i - 8, -1), -1):
                if j < 0:
                    break
                if (date - date_list[j]).days <= 7:
                    count_7d += 1
                else:
                    break

            suffix = "home" if is_home else "away"
            result[game_id][f"rest_{suffix}"] = float(rest)
            result[game_id][f"b2b_{suffix}"] = is_b2b
            result[game_id][f"g7d_{suffix}"] = float(count_7d)

    logger.info(
        f"  Computed rest/schedule for {len(result)} games in {time.time()-t0:.1f}s"
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def build_l3_features(
    df: pd.DataFrame, continuity: dict, trends: dict
) -> dict[str, np.ndarray]:
    """
    Assemble L3 team features (34-d) per game.

    Returns: {game_id: np.ndarray of shape (2, 34)} where [0]=home, [1]=away
    """
    logger.info("Assembling L3 team features...")
    t0 = time.time()

    # All feature columns in order (indices 0-23, then 28-31)
    feature_cols = [
        f"{c}_5g"
        for c in [
            "eFG_off",
            "TOV_off",
            "ORB_off",
            "FTR_off",
            "eFG_def",
            "TOV_def",
            "ORB_def",
            "FTR_def",
            "ORtg",
            "DRtg",
            "NetRtg",
            "Pace",
        ]
    ] + [
        f"{c}_15g"
        for c in [
            "eFG_off",
            "TOV_off",
            "ORB_off",
            "FTR_off",
            "eFG_def",
            "TOV_def",
            "ORB_def",
            "FTR_def",
            "ORtg",
            "DRtg",
            "NetRtg",
            "Pace",
        ]
    ]
    defense_cols = [
        "opp_3PA_rate_15g",
        "opp_paint_pts_rate_15g",
        "steal_rate_15g",
        "block_rate_15g",
    ]

    # Pre-compute continuity and trend arrays for vectorized access
    df = df.copy()
    df["_continuity"] = df.apply(
        lambda r: continuity.get((r["team_abbr"], r["season"]), 0.5), axis=1
    ).astype(np.float32)
    df["_trend"] = df.apply(
        lambda r: trends.get((r["team_abbr"], r["season"]), 0.5), axis=1
    ).astype(np.float32)

    # Extract arrays for fast slicing
    all_feature_vals = df[feature_cols].values.astype(np.float32)  # (N, 24)
    all_defense_vals = df[defense_cols].values.astype(np.float32)  # (N, 4)
    all_continuity = df["_continuity"].values  # (N,)
    all_trend = df["_trend"].values  # (N,)
    all_game_ids = df["game_id"].values
    all_is_home = df["is_home"].values

    result = {}

    # Process in game_id groups using index tracking
    game_id_groups = df.groupby("game_id").indices
    for game_id, indices in game_id_groups.items():
        if len(indices) != 2:
            continue

        features = np.zeros((2, 34), dtype=np.float32)

        for i in indices:
            idx = 0 if all_is_home[i] else 1
            features[idx, 0:24] = all_feature_vals[i]
            # indices 24-27 stay zero (coaching placeholders)
            features[idx, 28:32] = all_defense_vals[i]
            features[idx, 32] = all_continuity[i]
            features[idx, 33] = all_trend[i]

        result[game_id] = features

    logger.info(
        f"  Assembled L3 features for {len(result)} games in {time.time()-t0:.1f}s"
    )
    return result


def build_l4_features(df: pd.DataFrame, rest_schedule: dict) -> dict[str, np.ndarray]:
    """
    Assemble L4 game context features (14-d) per game.

    Returns: {game_id: np.ndarray of shape (14,)}
    """
    logger.info("Assembling L4 game context features...")
    t0 = time.time()

    result = {}
    # De-duplicate to one row per game (take the home team row)
    home_rows = df[df["is_home"]].drop_duplicates(subset="game_id").copy()
    # For any games where no row was identified as home, use fallback
    all_game_ids = set(df["game_id"].unique())
    covered_game_ids = set(home_rows["game_id"].values)
    missing = all_game_ids - covered_game_ids
    if missing:
        logger.warning(f"  {len(missing)} games had no home-team row, using fallback")
        fallback = df[df["game_id"].isin(missing)].drop_duplicates(subset="game_id")
        home_rows = pd.concat([home_rows, fallback], ignore_index=True)

    for _, row in home_rows.iterrows():
        game_id = row["game_id"]
        features = np.zeros(14, dtype=np.float32)

        # home_flag: always 1.0 (from home perspective)
        features[0] = 1.0

        # arena_altitude (home team's arena)
        home_team = row["home_team"]
        features[1] = float(ALTITUDE_MAP.get(home_team, 0))

        # Rest and schedule from precomputed dict
        sched = rest_schedule.get(game_id, {})
        features[2] = sched.get("rest_home", 3.0)
        features[3] = sched.get("rest_away", 3.0)
        features[4] = sched.get("b2b_home", 0.0)
        features[5] = sched.get("b2b_away", 0.0)
        features[6] = sched.get("g7d_home", 1.0)
        features[7] = sched.get("g7d_away", 1.0)

        # Travel distance (placeholder)
        features[8] = 0.0  # travel_dist_home
        features[9] = 0.0  # travel_dist_away

        # Timezone crossings (placeholder)
        features[10] = 0.0  # tz_crossings_home
        features[11] = 0.0  # tz_crossings_away

        # Season progress
        features[12] = min(row.get("season_game_num", 41) / 82.0, 1.0)

        # Is playoffs
        features[13] = 1.0 if "Post" in str(row.get("season_type", "")) else 0.0

        result[game_id] = features

    logger.info(
        f"  Assembled L4 features for {len(result)} games in {time.time()-t0:.1f}s"
    )
    return result


def compute_normalization_stats(
    l3_features: dict[str, np.ndarray],
    l4_features: dict[str, np.ndarray],
) -> dict:
    """Compute mean and std for normalization."""
    logger.info("Computing normalization statistics...")

    # Stack all L3 features: shape (N_games * 2, 34)
    all_l3 = np.concatenate(list(l3_features.values()), axis=0)  # (N*2, 34)
    l3_mean = np.nanmean(all_l3, axis=0)
    l3_std = np.nanstd(all_l3, axis=0) + 1e-8

    # Stack all L4 features: shape (N_games, 14)
    all_l4 = np.stack(list(l4_features.values()), axis=0)  # (N, 14)
    l4_mean = np.nanmean(all_l4, axis=0)
    l4_std = np.nanstd(all_l4, axis=0) + 1e-8

    return {
        "l3_mean": l3_mean.tolist(),
        "l3_std": l3_std.tolist(),
        "l4_mean": l4_mean.tolist(),
        "l4_std": l4_std.tolist(),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_cache(
    l3_features: dict[str, np.ndarray],
    l4_features: dict[str, np.ndarray],
    norm_stats: dict,
    date_range: tuple[str, str],
):
    """Save cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving cache to {CACHE_DIR}...")
    t0 = time.time()

    # Save team features as npz with game_ids as keys
    # np.savez doesn't support arbitrary string keys well, so we store as parallel arrays
    game_ids = sorted(l3_features.keys())
    l3_array = np.stack([l3_features[gid] for gid in game_ids], axis=0)  # (N, 2, 34)
    np.savez_compressed(
        CACHE_DIR / "team_features.npz",
        game_ids=np.array(game_ids, dtype=object),
        features=l3_array,
    )
    logger.info(f"  team_features.npz: {l3_array.shape}")

    # Save game context
    l4_game_ids = sorted(l4_features.keys())
    l4_array = np.stack([l4_features[gid] for gid in l4_game_ids], axis=0)  # (N, 14)
    np.savez_compressed(
        CACHE_DIR / "game_context.npz",
        game_ids=np.array(l4_game_ids, dtype=object),
        features=l4_array,
    )
    logger.info(f"  game_context.npz: {l4_array.shape}")

    # Save metadata
    metadata = {
        "l3_feature_names": L3_FEATURE_NAMES,
        "l4_feature_names": L4_FEATURE_NAMES,
        "l3_dim": 34,
        "l4_dim": 14,
        "n_games": len(game_ids),
        "date_range": list(date_range),
        "normalization": norm_stats,
        "notes": {
            "coaching_features": "Placeholder zeros (indices 24-27). Fill when Coaches table is available.",
            "travel_features": "Placeholder zeros (L4 indices 8-11). Fill when arena_data.py is ready.",
            "rolling_windows": "5-game and 15-game, shifted by 1 game to prevent leakage.",
            "roster_continuity": "Fraction of minutes from players on team in previous season.",
            "multi_year_trend": "3-year rolling win percentage (shifted, no leakage).",
        },
    }
    with open(CACHE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  metadata.json saved")

    logger.info(f"  Cache saved in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_sample(l3_features: dict, l4_features: dict, n: int = 5):
    """Print a few samples to sanity-check."""
    logger.info(f"Validating {n} sample games...")

    game_ids = sorted(l3_features.keys())[:n]
    for gid in game_ids:
        l3 = l3_features[gid]
        l4 = l4_features.get(gid)
        logger.info(f"\n  Game {gid}:")
        logger.info(f"    L3 home shape={l3[0].shape}, values: {l3[0][:6].round(3)}")
        logger.info(f"    L3 away shape={l3[1].shape}, values: {l3[1][:6].round(3)}")
        if l4 is not None:
            logger.info(f"    L4 shape={l4.shape}, values: {l4.round(3)}")

    # Check for NaN/Inf
    all_l3 = np.concatenate(list(l3_features.values()), axis=0)
    all_l4 = np.stack(list(l4_features.values()), axis=0)
    n_nan_l3 = np.isnan(all_l3).sum()
    n_inf_l3 = np.isinf(all_l3).sum()
    n_nan_l4 = np.isnan(all_l4).sum()
    n_inf_l4 = np.isinf(all_l4).sum()
    logger.info(f"\n  L3: NaN={n_nan_l3}, Inf={n_inf_l3}")
    logger.info(f"  L4: NaN={n_nan_l4}, Inf={n_inf_l4}")

    # Value range checks
    logger.info(f"\n  L3 feature ranges:")
    for i, name in enumerate(L3_FEATURE_NAMES):
        vals = all_l3[:, i]
        logger.info(
            f"    {name:25s}: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f}"
        )

    logger.info(f"\n  L4 feature ranges:")
    for i, name in enumerate(L4_FEATURE_NAMES):
        vals = all_l4[:, i]
        logger.info(
            f"    {name:25s}: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} std={vals.std():.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build L3/L4 feature cache")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to N games (for testing)"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing cache"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.validate_only:
        logger.info("Loading existing cache for validation...")
        data_l3 = np.load(CACHE_DIR / "team_features.npz", allow_pickle=True)
        data_l4 = np.load(CACHE_DIR / "game_context.npz", allow_pickle=True)
        l3_gids = data_l3["game_ids"]
        l3_feats = data_l3["features"]
        l4_gids = data_l4["game_ids"]
        l4_feats = data_l4["features"]
        l3_features = {gid: l3_feats[i] for i, gid in enumerate(l3_gids)}
        l4_features = {gid: l4_feats[i] for i, gid in enumerate(l4_gids)}
        validate_sample(l3_features, l4_features)
        return

    total_start = time.time()

    with get_db() as conn:
        # 1. Load data
        team_id_map = load_team_id_map(conn)
        df = load_game_data(conn, limit=args.limit)
        reb_df = load_playerbox_rebounds(conn, limit=args.limit)
        roster_df = load_roster_minutes(conn)

    # 2. Compute per-game raw stats (joins opponent stats)
    df = compute_per_game_raw_stats(df, reb_df, team_id_map)

    # 3. Compute rolling features
    df = compute_rolling_features(df)

    # 4. Compute season game numbers (for L4 season_progress)
    df = compute_season_game_number(df)

    # 5. Compute rest/schedule features
    rest_schedule = compute_rest_and_schedule(df)

    # 6. Compute roster continuity
    continuity = compute_roster_continuity(roster_df, team_id_map)

    # 7. Compute multi-year win trends
    trends = compute_multi_year_trend(df)

    # 8. Assemble L3 features
    l3_features = build_l3_features(df, continuity, trends)

    # 9. Assemble L4 features
    l4_features = build_l4_features(df, rest_schedule)

    # 10. Validate
    validate_sample(l3_features, l4_features)

    # 11. Normalization stats
    norm_stats = compute_normalization_stats(l3_features, l4_features)

    # 12. Date range
    dates = df["date"].dropna()
    date_range = (str(dates.min().date()), str(dates.max().date()))

    # 13. Save
    save_cache(l3_features, l4_features, norm_stats, date_range)

    elapsed = time.time() - total_start
    logger.info(
        f"\nDone! Built L3/L4 cache for {len(l3_features)} games in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
