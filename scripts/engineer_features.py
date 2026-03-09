#!/usr/bin/env python
"""
Engineer rolling features from TeamBox/Games for hybrid XGBoost (Phase 3 Exp 8).

Computes ~63 features per game: rolling efficiency metrics (3 windows),
EWM time-decay features (2 spans), venue-specific win%, record features,
rest/scheduling, and head-to-head stats.

Usage:
    python scripts/engineer_features.py --output data/hybrid/features.npz
"""

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = "data/NBA_AI_full.sqlite"

FEATURE_NAMES = []  # Populated during feature engineering


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_teambox_data(db_path: str, seasons: list[str]) -> pd.DataFrame:
    """Load TeamBox + Games data for specified seasons."""
    conn = sqlite3.connect(db_path)

    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT
        g.game_id, g.date_time_utc, g.home_team, g.away_team,
        g.season, g.season_type,
        tb.team_id, t.abbreviation as team_abbrev,
        tb.pts, tb.pts_allowed, tb.reb, tb.ast, tb.stl, tb.blk,
        tb.tov, tb.pf, tb.fga, tb.fgm, tb.fg3a, tb.fg3m, tb.fta, tb.ftm,
        tb.plus_minus
    FROM TeamBox tb
    JOIN Games g ON tb.game_id = g.game_id
    JOIN Teams t ON tb.team_id = t.team_id
    WHERE g.season IN ({placeholders})
      AND g.status = 3
      AND g.season_type IN ('Regular Season', 'Post Season')
    ORDER BY g.date_time_utc
    """
    df = pd.read_sql_query(query, conn, params=seasons)
    conn.close()

    df["date"] = pd.to_datetime(df["date_time_utc"])

    # Map historical abbreviations in Games to current (Teams table uses current)
    historical_map = {"NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "SEA": "OKC", "VAN": "MEM"}
    df["home_team"] = df["home_team"].replace(historical_map)
    df["away_team"] = df["away_team"].replace(historical_map)
    df["is_home"] = df["team_abbrev"] == df["home_team"]

    return df


def load_oreb_data(db_path: str, seasons: list[str]) -> pd.DataFrame:
    """Sum oreb from PlayerBox per team per game."""
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in seasons)
    query = f"""
    SELECT pb.game_id, pb.team_id, SUM(pb.oreb) as team_oreb
    FROM PlayerBox pb
    JOIN Games g ON pb.game_id = g.game_id
    WHERE g.season IN ({placeholders})
      AND pb.min > 0
    GROUP BY pb.game_id, pb.team_id
    """
    df = pd.read_sql_query(query, conn, params=seasons)
    conn.close()
    return df


def compute_efficiency_stats(group: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game efficiency stats for a team.

    Expects 'possessions' column to already be computed (with oreb correction).
    """
    g = group.copy()

    if "possessions" not in g.columns:
        raise ValueError(
            "possessions column must be pre-computed before calling "
            "compute_efficiency_stats (needs oreb correction)"
        )

    # Efficiency metrics
    g["efg_pct"] = np.where(
        g["fga"] > 0,
        (g["fgm"] + 0.5 * g["fg3m"]) / g["fga"],
        0.0,
    )
    g["ts_pct"] = np.where(
        g["fga"] + 0.44 * g["fta"] > 0,
        g["pts"] / (2 * (g["fga"] + 0.44 * g["fta"])),
        0.0,
    )
    g["tov_pct"] = np.where(
        g["possessions"] > 0,
        g["tov"] / g["possessions"],
        0.0,
    )
    g["ft_rate"] = np.where(g["fga"] > 0, g["fta"] / g["fga"], 0.0)
    g["fg3a_rate"] = np.where(g["fga"] > 0, g["fg3a"] / g["fga"], 0.0)
    g["ast_ratio"] = np.where(g["fgm"] > 0, g["ast"] / g["fgm"], 0.0)

    # Ratings (per 100 possessions)
    g["ortg"] = np.where(g["possessions"] > 0, g["pts"] / g["possessions"] * 100, 0.0)
    g["drtg"] = np.where(
        g["possessions"] > 0, g["pts_allowed"] / g["possessions"] * 100, 0.0
    )
    g["net_rtg"] = g["ortg"] - g["drtg"]

    # Win indicator
    g["win"] = (g["plus_minus"] > 0).astype(float)

    return g


def compute_rolling_features(team_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling window features for a single team-season."""
    metrics = [
        "efg_pct", "ts_pct", "tov_pct", "ft_rate", "fg3a_rate",
        "ast_ratio", "ortg", "drtg", "net_rtg",
    ]

    result = team_df[["game_id", "team_id", "date", "season"]].copy()

    for w in windows:
        for m in metrics:
            # Shift by 1 to exclude current game (no leakage)
            col = f"{m}_roll{w}"
            result[col] = (
                team_df[m].shift(1).rolling(window=w, min_periods=1).mean()
            )

    # Win percentage (rolling)
    for w in windows:
        result[f"win_pct_roll{w}"] = (
            team_df["win"].shift(1).rolling(window=w, min_periods=1).mean()
        )

    # Exponentially weighted features (captures momentum/recency better)
    ewm_metrics = ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "win"]
    ewm_spans = [10, 20]
    for span in ewm_spans:
        for m in ewm_metrics:
            col = f"{m}_ewm{span}"
            result[col] = team_df[m].shift(1).ewm(span=span, min_periods=1).mean()

    # Venue-specific win% (rolling within home-only or away-only games)
    home_wins = team_df["win"].where(team_df["is_home"]).shift(1)
    result["home_win_pct"] = home_wins.expanding(min_periods=1).mean()

    away_wins = team_df["win"].where(~team_df["is_home"]).shift(1)
    result["away_win_pct"] = away_wins.expanding(min_periods=1).mean()

    result["is_home"] = team_df["is_home"].values

    # Streak (current W/L streak)
    wins_shifted = team_df["win"].shift(1)
    streak = []
    current_streak = 0
    for w_val in wins_shifted:
        if pd.isna(w_val):
            streak.append(0)
            continue
        if w_val == 1:
            current_streak = max(current_streak, 0) + 1
        else:
            current_streak = min(current_streak, 0) - 1
        streak.append(np.clip(current_streak, -10, 10) / 10.0)
    result["streak"] = streak

    # Pace (possessions per game, rolling 20)
    result["pace_roll20"] = (
        team_df["possessions"].shift(1).rolling(window=20, min_periods=1).mean()
    )

    # PPG, OPP PPG (rolling 20)
    result["ppg_roll20"] = (
        team_df["pts"].shift(1).rolling(window=20, min_periods=1).mean()
    )
    result["opp_ppg_roll20"] = (
        team_df["pts_allowed"].shift(1).rolling(window=20, min_periods=1).mean()
    )

    # Game number in season
    result["game_number"] = range(1, len(result) + 1)
    result["season_progress"] = result["game_number"] / 82.0

    return result


def compute_rest_days(team_df: pd.DataFrame) -> pd.Series:
    """Compute days since last game for a team."""
    dates = team_df["date"]
    rest = dates.diff().dt.days.fillna(7).clip(upper=7) / 7.0
    return rest


def build_h2h_lookup(games_df: pd.DataFrame) -> dict:
    """Pre-compute H2H lookup: sorted team pair -> list of (date, home_team, margin)."""
    h2h_lookup = defaultdict(list)
    for _, row in games_df.iterrows():
        key = tuple(sorted([row["home_team"], row["away_team"]]))
        h2h_lookup[key].append((row["date"], row["home_team"], row["home_margin"]))
    return h2h_lookup


def compute_h2h_features(
    h2h_lookup: dict, home_team: str, away_team: str, date
) -> np.ndarray:
    """Compute H2H features from last 5 meetings between home and away team."""
    key = tuple(sorted([home_team, away_team]))
    meetings = h2h_lookup.get(key, [])

    # Filter to prior meetings only
    prior = [(d, ht, m) for d, ht, m in meetings if d < date]
    prior = prior[-5:]  # Last 5

    if len(prior) == 0:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)

    # Home team wins in meetings
    wins = 0
    margins = []
    for _, ht, m in prior:
        margin = m if ht == home_team else -m
        if margin > 0:
            wins += 1
        margins.append(margin)

    h2h_win_rate = wins / len(prior)
    h2h_avg_margin = np.mean(margins) / 13.0  # Normalize
    h2h_count = min(len(prior), 10) / 10.0

    return np.array([h2h_win_rate, h2h_avg_margin, h2h_count], dtype=np.float32)


def build_game_features(
    all_seasons: list[str], db_path: str = DB_PATH
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix for all games across all seasons."""
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading TeamBox data for {len(all_seasons)} seasons...")
    df = load_teambox_data(db_path, all_seasons)
    logger.info(f"  {len(df)} team-game rows")

    # Load oreb for pace correction
    logger.info("Loading oreb from PlayerBox...")
    oreb_df = load_oreb_data(db_path, all_seasons)
    df = df.merge(oreb_df, on=["game_id", "team_id"], how="left")
    df["team_oreb"] = df["team_oreb"].fillna(0)
    df["possessions"] = df["fga"] + 0.44 * df["fta"] + df["tov"] - df["team_oreb"]
    df["possessions"] = df["possessions"].clip(lower=50)  # Safety floor

    # Compute efficiency stats
    logger.info("Computing efficiency stats...")
    df = compute_efficiency_stats(df)

    # Build per-team rolling features
    logger.info("Computing rolling features...")
    windows = [5, 10, 20]
    team_features = {}
    for (team_abbrev, season), group in df.groupby(["team_abbrev", "season"]):
        group = group.sort_values("date").reset_index(drop=True)
        rolling = compute_rolling_features(group, windows)
        rolling["rest_days"] = compute_rest_days(group)
        rolling["b2b"] = (rolling["rest_days"] * 7 <= 1).astype(float)
        for _, row in rolling.iterrows():
            team_features[(row["game_id"], team_abbrev)] = row

    # Build game-level score margins for H2H (pre-compute lookup)
    games_df = df[df["is_home"]].copy()
    games_df["home_margin"] = games_df["plus_minus"]
    h2h_lookup = build_h2h_lookup(games_df)

    # Build game-level feature matrix
    logger.info("Building game-level features...")
    unique_games = df.drop_duplicates("game_id")[
        ["game_id", "date", "home_team", "away_team", "season"]
    ].sort_values("date")

    feature_rows = []
    game_ids = []
    feature_names = None

    for _, game in unique_games.iterrows():
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]
        date = game["date"]

        home_key = (gid, home)
        away_key = (gid, away)

        if home_key not in team_features or away_key not in team_features:
            continue

        hf = team_features[home_key]
        af = team_features[away_key]

        row = {}

        # A. Rolling efficiency (differenced: home - away)
        for w in windows:
            for m in [
                "efg_pct", "ts_pct", "tov_pct", "ft_rate", "fg3a_rate",
                "ast_ratio", "ortg", "drtg", "net_rtg",
            ]:
                col = f"{m}_roll{w}"
                h_val = hf.get(col, 0) if not pd.isna(hf.get(col, 0)) else 0
                a_val = af.get(col, 0) if not pd.isna(af.get(col, 0)) else 0
                row[f"diff_{col}"] = h_val - a_val

        # A2. EWM efficiency (differenced: home - away)
        for span in [10, 20]:
            for m in ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "win"]:
                col = f"{m}_ewm{span}"
                h_val = hf.get(col, 0) if not pd.isna(hf.get(col, 0)) else 0
                a_val = af.get(col, 0) if not pd.isna(af.get(col, 0)) else 0
                row[f"diff_{col}"] = h_val - a_val

        # B. Record features (differenced)
        for w in windows:
            h_val = hf.get(f"win_pct_roll{w}", 0.5)
            a_val = af.get(f"win_pct_roll{w}", 0.5)
            if pd.isna(h_val): h_val = 0.5
            if pd.isna(a_val): a_val = 0.5
            row[f"diff_win_pct_roll{w}"] = h_val - a_val

        # Streak
        row["diff_streak"] = float(hf.get("streak", 0)) - float(af.get("streak", 0))

        # C. Rest/scheduling (separate, not differenced)
        row["home_rest_days"] = float(hf.get("rest_days", 0.5))
        row["away_rest_days"] = float(af.get("rest_days", 0.5))
        row["home_b2b"] = float(hf.get("b2b", 0))
        row["away_b2b"] = float(af.get("b2b", 0))
        row["season_progress"] = float(hf.get("season_progress", 0.5))

        # D. H2H features
        h2h = compute_h2h_features(h2h_lookup, home, away, date)
        row["h2h_win_rate"] = h2h[0]
        row["h2h_avg_margin"] = h2h[1]
        row["h2h_count"] = h2h[2]

        # E. Venue-specific win% (home team's home record, away team's away record)
        h_home_wp = hf.get("home_win_pct", 0.5)
        a_away_wp = af.get("away_win_pct", 0.5)
        row["home_team_home_win_pct"] = 0.5 if pd.isna(h_home_wp) else float(h_home_wp)
        row["away_team_away_win_pct"] = 0.5 if pd.isna(a_away_wp) else float(a_away_wp)

        # F. Absolute team features (not differenced)
        for metric in ["ortg", "drtg", "pace", "ppg", "opp_ppg"]:
            col = f"{metric}_roll20"
            h_val = hf.get(col, 0)
            a_val = af.get(col, 0)
            if pd.isna(h_val): h_val = 0
            if pd.isna(a_val): a_val = 0
            row[f"home_{col}"] = h_val
            row[f"away_{col}"] = a_val

        if feature_names is None:
            feature_names = list(row.keys())

        feature_rows.append(list(row.values()))
        game_ids.append(gid)

    features = np.array(feature_rows, dtype=np.float32)
    # Replace any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    logger.info(f"Built {len(game_ids)} games × {features.shape[1]} features")
    logger.info(f"Feature names: {feature_names}")

    return features, np.array(game_ids), feature_names


def main():
    parser = argparse.ArgumentParser(description="Engineer rolling features")
    parser.add_argument("--output", type=str, default="data/hybrid/features.npz")
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    all_seasons = [f"{y}-{y+1}" for y in range(2008, 2026)]
    logger.info(f"Engineering features for seasons: {all_seasons[0]} to {all_seasons[-1]}")

    features, game_ids, feature_names = build_game_features(all_seasons, args.db_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        game_ids=game_ids,
        feature_names=np.array(feature_names),
    )
    logger.info(f"Saved features to {output_path}: {features.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
