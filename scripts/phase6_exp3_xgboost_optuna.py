#!/usr/bin/env python
"""
Phase 6 Experiment 3: Maximize XGBoost with Optuna + full feature stack.

Combines ALL available feature sources:
  1. Rolling team stats (PPG, Opp PPG, FG%, 3P%, FT%, eFG%, REB, AST, TOV, STL, BLK,
     pace, net rating) at windows 5/10/20 (~40 per team)
  2. EWM stats (net_rtg, ortg, drtg, efg%, ts%, win) at spans 10/20 (~12 per team)
  3. Record features (win%, home win%, away win%, L10 win%, streak)
  4. Context (rest days, B2B flag, season progress)
  5. Matchup (H2H record, H2H margin avg, H2H count)
  6. Betting data (Vegas spread, total, ML implied probs)
  7. LLM predictions (predicted spread, win prob from gpt-5.4-mini)
  8. Differentials (home - away for all team stats)
  9. Absolute team stats (both home and away raw values)

Runs Optuna hyperparameter search (200 trials) for:
  - Regression: predict margin (home - away)
  - Classification: predict cover/not-cover (ATS)

Usage:
    python scripts/phase6_exp3_xgboost_optuna.py
    python scripts/phase6_exp3_xgboost_optuna.py --n-trials 50     # Quick test
    python scripts/phase6_exp3_xgboost_optuna.py --no-betting      # Ablation without betting
    python scripts/phase6_exp3_xgboost_optuna.py --no-llm          # Ablation without LLM preds
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lazy / optional imports
# ---------------------------------------------------------------------------
try:
    import optuna
except ImportError:
    print("optuna not installed. Run: pip install optuna")
    sys.exit(1)

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    print("xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
)

try:
    import joblib
except ImportError:
    joblib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "NBA_AI_full.sqlite")
LLM_RESULTS_PATH = str(
    PROJECT_ROOT / "data" / "exp7" / "test_gpt-5.4-mini_results.jsonl"
)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase6" / "exp3_xgboost_optuna"

HISTORICAL_MAP = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
}

# Data splits (season string → split)
TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2022)]  # 2008-2009 .. 2021-2022
VAL_SEASONS = ["2022-2023", "2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]

# Seasons to load for building rolling features (need 1 prior season for warm-up)
ALL_SEASONS = [f"{y}-{y+1}" for y in range(2007, 2026)]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# 1. Feature Engineering from Database
# ============================================================================


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
    df["home_team"] = df["home_team"].replace(HISTORICAL_MAP)
    df["away_team"] = df["away_team"].replace(HISTORICAL_MAP)
    df["is_home"] = df["team_abbrev"] == df["home_team"]

    return df


def load_oreb_data(db_path: str, seasons: list[str]) -> pd.DataFrame:
    """Sum oreb from PlayerBox per team per game for possession estimation."""
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


def load_betting_data(db_path: str) -> pd.DataFrame:
    """Load betting lines from the Betting table."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        game_id,
        COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread) as vegas_spread,
        COALESCE(espn_closing_total, covers_closing_total, espn_current_total) as vegas_total,
        COALESCE(espn_closing_ml_home, espn_current_ml_home) as ml_home,
        COALESCE(espn_closing_ml_away, espn_current_ml_away) as ml_away,
        spread_result
    FROM Betting
    WHERE COALESCE(espn_closing_spread, covers_closing_spread, espn_current_spread) IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_llm_predictions(path: str) -> dict[str, dict]:
    """Load LLM predictions from JSONL file, keyed by game_id."""
    if not os.path.exists(path):
        return {}

    preds = {}
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            if not row.get("success", False):
                continue
            pred = row.get("prediction", {})
            if isinstance(pred, dict) and "home_score" in pred:
                game_id = row["game_id"]
                preds[game_id] = {
                    "llm_home_score": pred["home_score"],
                    "llm_away_score": pred["away_score"],
                    "llm_spread": pred["home_score"] - pred["away_score"],
                    "llm_win_prob": pred.get("home_win_probability", 0.5),
                }
    return preds


def ml_to_implied_prob(ml: float) -> float:
    """Convert American moneyline odds to implied probability."""
    if ml is None or np.isnan(ml):
        return 0.5
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def compute_efficiency_stats(group: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game efficiency stats for a team."""
    g = group.copy()

    g["efg_pct"] = np.where(g["fga"] > 0, (g["fgm"] + 0.5 * g["fg3m"]) / g["fga"], 0.0)
    g["ts_pct"] = np.where(
        g["fga"] + 0.44 * g["fta"] > 0,
        g["pts"] / (2 * (g["fga"] + 0.44 * g["fta"])),
        0.0,
    )
    g["fg_pct"] = np.where(g["fga"] > 0, g["fgm"] / g["fga"], 0.0)
    g["fg3_pct"] = np.where(g["fg3a"] > 0, g["fg3m"] / g["fg3a"], 0.0)
    g["ft_pct"] = np.where(g["fta"] > 0, g["ftm"] / g["fta"], 0.0)
    g["tov_pct"] = np.where(g["possessions"] > 0, g["tov"] / g["possessions"], 0.0)
    g["ft_rate"] = np.where(g["fga"] > 0, g["fta"] / g["fga"], 0.0)
    g["fg3a_rate"] = np.where(g["fga"] > 0, g["fg3a"] / g["fga"], 0.0)
    g["ast_ratio"] = np.where(g["fgm"] > 0, g["ast"] / g["fgm"], 0.0)

    # Ratings (per 100 possessions)
    g["ortg"] = np.where(g["possessions"] > 0, g["pts"] / g["possessions"] * 100, 0.0)
    g["drtg"] = np.where(
        g["possessions"] > 0, g["pts_allowed"] / g["possessions"] * 100, 0.0
    )
    g["net_rtg"] = g["ortg"] - g["drtg"]

    # Margin
    g["margin"] = g["pts"] - g["pts_allowed"]

    # Win indicator
    g["win"] = (g["plus_minus"] > 0).astype(float)

    return g


def compute_rolling_features(team_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling window features for a single team within a season.

    All rolling stats use shift(1) to avoid leaking current game data.
    """
    result = team_df[["game_id", "team_id", "date", "season", "team_abbrev"]].copy()

    # Core efficiency metrics for windowed rolling
    rolling_metrics = [
        "efg_pct",
        "ts_pct",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "tov_pct",
        "ft_rate",
        "fg3a_rate",
        "ast_ratio",
        "ortg",
        "drtg",
        "net_rtg",
    ]

    # Box score aggregates for windowed rolling
    box_metrics = ["pts", "pts_allowed", "reb", "ast", "tov", "stl", "blk"]

    for w in windows:
        for m in rolling_metrics + box_metrics:
            col = f"{m}_roll{w}"
            result[col] = team_df[m].shift(1).rolling(window=w, min_periods=1).mean()
        # Win percentage rolling
        result[f"win_pct_roll{w}"] = (
            team_df["win"].shift(1).rolling(window=w, min_periods=1).mean()
        )

    # EWM features (captures momentum/recency)
    ewm_metrics = ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "win", "margin"]
    ewm_spans = [5, 15]
    for span in ewm_spans:
        for m in ewm_metrics:
            col = f"{m}_ewm{span}"
            result[col] = team_df[m].shift(1).ewm(span=span, min_periods=1).mean()

    # Venue-specific win%
    home_wins = team_df["win"].where(team_df["is_home"]).shift(1)
    result["home_win_pct"] = home_wins.expanding(min_periods=1).mean()

    away_wins = team_df["win"].where(~team_df["is_home"]).shift(1)
    result["away_win_pct"] = away_wins.expanding(min_periods=1).mean()

    result["is_home"] = team_df["is_home"].values

    # Streak (positive = winning streak, negative = losing streak, clipped to [-10, 10])
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

    # Pace
    result["pace_roll20"] = (
        team_df["possessions"].shift(1).rolling(window=20, min_periods=1).mean()
    )

    # Season progress
    result["game_number"] = range(1, len(result) + 1)
    result["season_progress"] = result["game_number"] / 82.0

    return result


def compute_rest_days(team_df: pd.DataFrame) -> pd.Series:
    """Compute normalized rest days since last game."""
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
) -> dict:
    """Compute H2H features from last N meetings."""
    key = tuple(sorted([home_team, away_team]))
    meetings = h2h_lookup.get(key, [])
    prior = [(d, ht, m) for d, ht, m in meetings if d < date]
    prior = prior[-10:]  # Last 10 meetings

    if len(prior) == 0:
        return {
            "h2h_win_rate": 0.5,
            "h2h_avg_margin": 0.0,
            "h2h_count": 0.0,
            "h2h_recent_win_rate": 0.5,
        }

    wins = 0
    margins = []
    for _, ht, m in prior:
        margin = m if ht == home_team else -m
        if margin > 0:
            wins += 1
        margins.append(margin)

    recent = prior[-3:]
    recent_wins = sum(1 for _, ht, m in recent if (m if ht == home_team else -m) > 0)

    return {
        "h2h_win_rate": wins / len(prior),
        "h2h_avg_margin": np.mean(margins) / 13.0,  # Normalize
        "h2h_count": min(len(prior), 10) / 10.0,
        "h2h_recent_win_rate": recent_wins / len(recent) if recent else 0.5,
    }


def build_all_features(
    db_path: str,
    include_betting: bool = True,
    include_llm: bool = True,
    llm_path: str = LLM_RESULTS_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full feature matrix from database + external sources.

    Returns:
        (DataFrame with game_id, season, date, features, targets), feature_names
    """
    logger = logging.getLogger(__name__)

    # --- Load raw data ---
    logger.info(f"Loading TeamBox data for {len(ALL_SEASONS)} seasons...")
    df = load_teambox_data(db_path, ALL_SEASONS)
    logger.info(f"  {len(df)} team-game rows")

    logger.info("Loading oreb from PlayerBox...")
    oreb_df = load_oreb_data(db_path, ALL_SEASONS)
    df = df.merge(oreb_df, on=["game_id", "team_id"], how="left")
    df["team_oreb"] = df["team_oreb"].fillna(0)
    df["possessions"] = df["fga"] + 0.44 * df["fta"] + df["tov"] - df["team_oreb"]
    df["possessions"] = df["possessions"].clip(lower=50)

    # --- Compute efficiency stats ---
    logger.info("Computing efficiency stats...")
    df = compute_efficiency_stats(df)

    # --- Build per-team rolling features ---
    logger.info("Computing rolling features per team-season...")
    windows = [5, 10, 20]
    team_features = {}
    for (team_abbrev, season), group in df.groupby(["team_abbrev", "season"]):
        group = group.sort_values("date").reset_index(drop=True)
        rolling = compute_rolling_features(group, windows)
        rolling["rest_days"] = compute_rest_days(group)
        rolling["b2b"] = (rolling["rest_days"] * 7 <= 1).astype(float)
        for _, row in rolling.iterrows():
            team_features[(row["game_id"], team_abbrev)] = row

    # --- H2H lookup ---
    games_df = df[df["is_home"]].copy()
    games_df["home_margin"] = games_df["plus_minus"]
    h2h_lookup = build_h2h_lookup(games_df)

    # --- Betting data ---
    betting_lookup = {}
    if include_betting:
        logger.info("Loading betting data...")
        betting_df = load_betting_data(db_path)
        for _, row in betting_df.iterrows():
            betting_lookup[row["game_id"]] = row
        logger.info(f"  {len(betting_lookup)} games with betting lines")

    # --- LLM predictions ---
    llm_lookup = {}
    if include_llm:
        logger.info(f"Loading LLM predictions from {llm_path}...")
        llm_lookup = load_llm_predictions(llm_path)
        logger.info(f"  {len(llm_lookup)} LLM predictions loaded")

    # --- Build game-level feature matrix ---
    logger.info("Building game-level feature matrix...")

    # Unique games (use home team rows)
    unique_games = df.drop_duplicates("game_id")[
        ["game_id", "date", "home_team", "away_team", "season"]
    ].sort_values("date")

    # Pre-build targets: home_pts and away_pts from TeamBox
    home_pts_lookup = {}
    away_pts_lookup = {}
    for _, row in df.iterrows():
        gid = row["game_id"]
        if row["is_home"]:
            home_pts_lookup[gid] = row["pts"]
        else:
            away_pts_lookup[gid] = row["pts"]

    records = []

    for _, game in unique_games.iterrows():
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]
        date = game["date"]
        season = game["season"]

        home_key = (gid, home)
        away_key = (gid, away)

        if home_key not in team_features or away_key not in team_features:
            continue

        hf = team_features[home_key]
        af = team_features[away_key]

        row = {"game_id": gid, "date": date, "season": season}

        # --- Targets ---
        h_pts = home_pts_lookup.get(gid)
        a_pts = away_pts_lookup.get(gid)
        if h_pts is None or a_pts is None:
            continue
        row["target_margin"] = float(h_pts - a_pts)
        row["target_home_win"] = 1 if h_pts > a_pts else 0
        row["target_home_pts"] = float(h_pts)
        row["target_away_pts"] = float(a_pts)

        # ==================================================================
        # FEATURE GROUP A: Differential rolling efficiency (home - away)
        # ==================================================================
        rolling_metrics = [
            "efg_pct",
            "ts_pct",
            "fg_pct",
            "fg3_pct",
            "ft_pct",
            "tov_pct",
            "ft_rate",
            "fg3a_rate",
            "ast_ratio",
            "ortg",
            "drtg",
            "net_rtg",
        ]
        box_metrics = ["pts", "pts_allowed", "reb", "ast", "tov", "stl", "blk"]

        for w in windows:
            for m in rolling_metrics + box_metrics:
                col = f"{m}_roll{w}"
                h_val = _safe_get(hf, col, 0.0)
                a_val = _safe_get(af, col, 0.0)
                row[f"diff_{col}"] = h_val - a_val

            # Win pct differential
            h_wp = _safe_get(hf, f"win_pct_roll{w}", 0.5)
            a_wp = _safe_get(af, f"win_pct_roll{w}", 0.5)
            row[f"diff_win_pct_roll{w}"] = h_wp - a_wp

        # ==================================================================
        # FEATURE GROUP B: Differential EWM features
        # ==================================================================
        ewm_metrics = ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "win", "margin"]
        for span in [5, 15]:
            for m in ewm_metrics:
                col = f"{m}_ewm{span}"
                h_val = _safe_get(hf, col, 0.0)
                a_val = _safe_get(af, col, 0.0)
                row[f"diff_{col}"] = h_val - a_val

        # ==================================================================
        # FEATURE GROUP C: Record features (differential)
        # ==================================================================
        row["diff_streak"] = _safe_get(hf, "streak", 0.0) - _safe_get(af, "streak", 0.0)
        row["diff_home_away_wp"] = _safe_get(hf, "home_win_pct", 0.5) - _safe_get(
            af, "away_win_pct", 0.5
        )

        # ==================================================================
        # FEATURE GROUP D: Context / scheduling
        # ==================================================================
        row["home_rest_days"] = _safe_get(hf, "rest_days", 0.5)
        row["away_rest_days"] = _safe_get(af, "rest_days", 0.5)
        row["diff_rest_days"] = row["home_rest_days"] - row["away_rest_days"]
        row["home_b2b"] = _safe_get(hf, "b2b", 0.0)
        row["away_b2b"] = _safe_get(af, "b2b", 0.0)
        row["season_progress"] = _safe_get(hf, "season_progress", 0.5)

        # ==================================================================
        # FEATURE GROUP E: Head-to-head
        # ==================================================================
        h2h = compute_h2h_features(h2h_lookup, home, away, date)
        for k, v in h2h.items():
            row[k] = v

        # ==================================================================
        # FEATURE GROUP F: Absolute team stats (both teams, for tree splits)
        # ==================================================================
        abs_metrics_20 = [
            "ortg",
            "drtg",
            "net_rtg",
            "pts",
            "pts_allowed",
            "efg_pct",
            "ts_pct",
            "reb",
            "ast",
            "tov",
            "stl",
            "blk",
        ]
        for m in abs_metrics_20:
            col = f"{m}_roll20"
            row[f"home_{col}"] = _safe_get(hf, col, 0.0)
            row[f"away_{col}"] = _safe_get(af, col, 0.0)

        row["home_pace_roll20"] = _safe_get(hf, "pace_roll20", 0.0)
        row["away_pace_roll20"] = _safe_get(af, "pace_roll20", 0.0)

        # Venue-specific win rates
        row["home_team_home_wp"] = _safe_get(hf, "home_win_pct", 0.5)
        row["away_team_away_wp"] = _safe_get(af, "away_win_pct", 0.5)

        # Season win rates
        row["home_season_wp"] = _safe_get(hf, "win_pct_roll20", 0.5)
        row["away_season_wp"] = _safe_get(af, "win_pct_roll20", 0.5)

        # Streaks (separate)
        row["home_streak"] = _safe_get(hf, "streak", 0.0)
        row["away_streak"] = _safe_get(af, "streak", 0.0)

        # ==================================================================
        # FEATURE GROUP G: Betting data
        # ==================================================================
        if include_betting and gid in betting_lookup:
            b = betting_lookup[gid]
            row["vegas_spread"] = (
                float(b["vegas_spread"]) if pd.notna(b["vegas_spread"]) else np.nan
            )
            row["vegas_total"] = (
                float(b["vegas_total"]) if pd.notna(b["vegas_total"]) else np.nan
            )

            ml_h = b["ml_home"]
            ml_a = b["ml_away"]
            row["vegas_ml_home_prob"] = ml_to_implied_prob(
                float(ml_h) if pd.notna(ml_h) else np.nan
            )
            row["vegas_ml_away_prob"] = ml_to_implied_prob(
                float(ml_a) if pd.notna(ml_a) else np.nan
            )
            # Normalize ML implied probs to sum to 1 (remove vig)
            total_prob = row["vegas_ml_home_prob"] + row["vegas_ml_away_prob"]
            if total_prob > 0:
                row["vegas_home_win_prob"] = row["vegas_ml_home_prob"] / total_prob
            else:
                row["vegas_home_win_prob"] = 0.5
            # Implied home margin from ML probs (logit transform)
            row["vegas_ml_margin_signal"] = row["vegas_home_win_prob"] - 0.5
        elif include_betting:
            row["vegas_spread"] = np.nan
            row["vegas_total"] = np.nan
            row["vegas_ml_home_prob"] = 0.5
            row["vegas_ml_away_prob"] = 0.5
            row["vegas_home_win_prob"] = 0.5
            row["vegas_ml_margin_signal"] = 0.0

        # ==================================================================
        # FEATURE GROUP H: LLM predictions
        # ==================================================================
        if include_llm and gid in llm_lookup:
            llm = llm_lookup[gid]
            row["llm_spread"] = float(llm["llm_spread"])
            row["llm_win_prob"] = float(llm["llm_win_prob"])
            row["llm_total"] = float(llm["llm_home_score"] + llm["llm_away_score"])
        elif include_llm:
            row["llm_spread"] = np.nan
            row["llm_win_prob"] = np.nan
            row["llm_total"] = np.nan

        # ==================================================================
        # FEATURE GROUP I: Interaction / derived features
        # ==================================================================
        # Spread disagreement (model vs Vegas) -- only if both available
        if include_betting and include_llm:
            if not np.isnan(row.get("vegas_spread", np.nan)) and not np.isnan(
                row.get("llm_spread", np.nan)
            ):
                row["spread_disagreement"] = row["llm_spread"] - (-row["vegas_spread"])
            else:
                row["spread_disagreement"] = np.nan

        # Pace interaction
        row["pace_matchup"] = (
            _safe_get(hf, "pace_roll20", 95.0) + _safe_get(af, "pace_roll20", 95.0)
        ) / 2.0

        # Defensive vs offensive mismatch
        row["home_off_vs_away_def"] = _safe_get(hf, "ortg_roll20", 0.0) - _safe_get(
            af, "drtg_roll20", 0.0
        )
        row["away_off_vs_home_def"] = _safe_get(af, "ortg_roll20", 0.0) - _safe_get(
            hf, "drtg_roll20", 0.0
        )

        records.append(row)

    result_df = pd.DataFrame(records)

    # Identify feature columns (everything except metadata and targets)
    meta_cols = {
        "game_id",
        "date",
        "season",
        "target_margin",
        "target_home_win",
        "target_home_pts",
        "target_away_pts",
    }
    feature_names = [c for c in result_df.columns if c not in meta_cols]

    logger.info(f"Built {len(result_df)} games x {len(feature_names)} features")

    # Replace NaN with 0 in feature columns (XGBoost handles NaN but
    # we want consistency; betting/LLM NaN is genuinely missing data)
    # Actually, XGBoost handles NaN natively, so leave them for tree splits
    # Just make sure numeric types are correct
    for col in feature_names:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    return result_df, feature_names


def _safe_get(series, key, default=0.0):
    """Get value from a Series, returning default if missing or NaN."""
    val = series.get(key, default)
    if pd.isna(val):
        return float(default)
    return float(val)


# ============================================================================
# 2. Data Split
# ============================================================================


def split_data(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Split into train/val/test by season."""
    # Filter out warm-up season (2007-2008 is only used for rolling features)
    df = df[df["season"] >= "2008-2009"].copy()

    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask = df["season"].isin(VAL_SEASONS)
    test_mask = df["season"].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, feature_names].values.astype(np.float32)
    X_val = df.loc[val_mask, feature_names].values.astype(np.float32)
    X_test = df.loc[test_mask, feature_names].values.astype(np.float32)

    y_margin_train = df.loc[train_mask, "target_margin"].values.astype(np.float32)
    y_margin_val = df.loc[val_mask, "target_margin"].values.astype(np.float32)
    y_margin_test = df.loc[test_mask, "target_margin"].values.astype(np.float32)

    y_win_train = df.loc[train_mask, "target_home_win"].values.astype(np.int32)
    y_win_val = df.loc[val_mask, "target_home_win"].values.astype(np.int32)
    y_win_test = df.loc[test_mask, "target_home_win"].values.astype(np.int32)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_margin_train": y_margin_train,
        "y_margin_val": y_margin_val,
        "y_margin_test": y_margin_test,
        "y_win_train": y_win_train,
        "y_win_val": y_win_val,
        "y_win_test": y_win_test,
        "df_train": df[train_mask].reset_index(drop=True),
        "df_val": df[val_mask].reset_index(drop=True),
        "df_test": df[test_mask].reset_index(drop=True),
        "feature_names": feature_names,
    }


# ============================================================================
# 3. Optuna Optimization
# ============================================================================


def optimize_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 200,
    study_name: str = "spread_regression",
) -> tuple[dict, "optuna.Study"]:
    """Tune XGBoost regressor with Optuna, optimizing validation MAE."""
    logger = logging.getLogger(__name__)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"[{study_name}] Best val MAE: {study.best_value:.4f}")
    logger.info(f"[{study_name}] Best params: {study.best_params}")

    return study.best_params, study


def optimize_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 200,
    study_name: str = "ats_classifier",
) -> tuple[dict, "optuna.Study"]:
    """Tune XGBoost classifier with Optuna, optimizing validation AUC."""
    logger = logging.getLogger(__name__)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
        }

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric="logloss",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        prob = model.predict_proba(X_val)[:, 1]
        # Optimize accuracy (ATS rate) rather than AUC for the cover classifier
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred)
        return acc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"[{study_name}] Best val accuracy: {study.best_value:.4f}")
    logger.info(f"[{study_name}] Best params: {study.best_params}")

    return study.best_params, study


# ============================================================================
# 4. Training & Evaluation
# ============================================================================


def train_final_model(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "regressor",
) -> object:
    """Train final model with best params on train, early-stopping on val."""
    if model_type == "regressor":
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric="logloss",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    return model


def train_final_model_on_trainval(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "regressor",
) -> object:
    """Train final model on train+val combined, using the best_iteration from
    the train-only model as the fixed n_estimators (no early stopping needed)."""
    # First, train with early stopping to find the best iteration
    if model_type == "regressor":
        probe = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        probe.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = probe.best_iteration
    else:
        probe = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
            eval_metric="logloss",
        )
        probe.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = probe.best_iteration

    # Now train on train+val with fixed n_estimators
    params_fixed = {k: v for k, v in params.items() if k != "n_estimators"}
    params_fixed["n_estimators"] = best_iter + 1  # best_iteration is 0-indexed

    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    if model_type == "regressor":
        model = XGBRegressor(
            **params_fixed,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    else:
        model = XGBClassifier(
            **params_fixed,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
    model.fit(X_combined, y_combined, verbose=False)
    return model


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(avg_conf - avg_acc)
    return ece


def evaluate_spread_model(
    model,
    X: np.ndarray,
    y_margin: np.ndarray,
    y_win: np.ndarray,
    df_split: pd.DataFrame,
    label: str = "Test",
) -> dict:
    """Evaluate spread regression model."""
    logger = logging.getLogger(__name__)

    pred_margin = model.predict(X)

    # Spread metrics (note: model predicts margin, Vegas spread is negative = home favored)
    spread_mae = mean_absolute_error(y_margin, pred_margin)
    spread_rmse = np.sqrt(mean_squared_error(y_margin, pred_margin))

    # Win prediction from margin sign
    pred_win = (pred_margin > 0).astype(int)
    win_acc = accuracy_score(y_win, pred_win)

    # Win AUC using margin as confidence
    # Clip and scale margin to [0, 1] for AUC
    margin_scaled = 1.0 / (1.0 + np.exp(-pred_margin / 5.0))  # sigmoid
    win_auc = roc_auc_score(y_win, margin_scaled)

    # ATS evaluation (requires Vegas spread)
    ats_rate, ats_n, ats_profit, ats_roi = _compute_ats(pred_margin, y_margin, df_split)

    metrics = {
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "win_accuracy": win_acc,
        "win_auc": win_auc,
        "ats_rate": ats_rate,
        "ats_n_games": ats_n,
        "ats_profit": ats_profit,
        "ats_roi": ats_roi,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {label} — Spread Regression Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Spread MAE:   {spread_mae:.2f}")
    logger.info(f"  Spread RMSE:  {spread_rmse:.2f}")
    logger.info(f"  Win AUC:      {win_auc:.3f}")
    logger.info(f"  Win Accuracy: {win_acc * 100:.1f}%")
    logger.info(f"  ATS Rate:     {ats_rate * 100:.1f}% ({ats_n} games with spreads)")
    logger.info(f"  ATS Profit:   ${ats_profit:.0f} at -110")
    logger.info(f"  ATS ROI:      {ats_roi * 100:.1f}%")
    logger.info(f"{'=' * 60}")

    return metrics


def evaluate_ats_classifier(
    model,
    X: np.ndarray,
    y_cover: np.ndarray,
    df_split: pd.DataFrame,
    cover_mask: np.ndarray,
    label: str = "Test",
) -> dict:
    """Evaluate ATS (cover/not-cover) classifier."""
    logger = logging.getLogger(__name__)

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_cover, pred)
    auc = roc_auc_score(y_cover, prob) if len(np.unique(y_cover)) > 1 else 0.0
    brier = brier_score_loss(y_cover, prob)
    ece = compute_ece(prob, y_cover.astype(float))

    # Profit calculation: bet on every game at -110
    n_bet = len(y_cover)
    wins_bet = (pred == y_cover).sum()
    # At -110 odds: win = +100, lose = -110
    profit = wins_bet * 100 - (n_bet - wins_bet) * 110
    roi = profit / (n_bet * 110) if n_bet > 0 else 0.0

    # Selective betting: only bet when model confidence > threshold
    selective_results = {}
    for threshold in [0.52, 0.55, 0.58, 0.60]:
        confident = (prob >= threshold) | (prob <= 1 - threshold)
        if confident.sum() > 0:
            sel_pred = pred[confident]
            sel_actual = y_cover[confident]
            sel_n = confident.sum()
            sel_wins = (sel_pred == sel_actual).sum()
            sel_rate = sel_wins / sel_n
            sel_profit = sel_wins * 100 - (sel_n - sel_wins) * 110
            sel_roi = sel_profit / (sel_n * 110) if sel_n > 0 else 0.0
            selective_results[threshold] = {
                "n_bets": int(sel_n),
                "ats_rate": sel_rate,
                "profit": sel_profit,
                "roi": sel_roi,
            }

    metrics = {
        "ats_accuracy": acc,
        "ats_auc": auc,
        "ats_brier": brier,
        "ats_ece": ece,
        "ats_profit_flat": profit,
        "ats_roi_flat": roi,
        "ats_n_games": n_bet,
        "selective_results": selective_results,
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {label} — ATS Classifier Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"  ATS Accuracy: {acc * 100:.1f}% ({n_bet} games)")
    logger.info(f"  ATS AUC:      {auc:.3f}")
    logger.info(f"  ATS Brier:    {brier:.4f}")
    logger.info(f"  ATS ECE:      {ece:.4f}")
    logger.info(f"  Flat Profit:  ${profit:.0f} at -110")
    logger.info(f"  Flat ROI:     {roi * 100:.1f}%")
    logger.info(f"  --- Selective Betting ---")
    for thresh, res in selective_results.items():
        logger.info(
            f"  Conf >= {thresh:.0%}: {res['ats_rate'] * 100:.1f}% on {res['n_bets']} bets, "
            f"profit ${res['profit']:.0f}, ROI {res['roi'] * 100:.1f}%"
        )
    logger.info(f"{'=' * 60}")

    return metrics


def _compute_ats(
    pred_margin: np.ndarray,
    actual_margin: np.ndarray,
    df_split: pd.DataFrame,
) -> tuple[float, int, float, float]:
    """Compute ATS rate from regression predictions vs Vegas spread.

    Convention: vegas_spread negative = home favored.
    Our prediction: pred_margin = predicted (home - away).
    Predicted spread (in Vegas convention) = -pred_margin.
    If our predicted margin > -vegas_spread, bet home to cover.
    Home covers when actual_margin > -vegas_spread.
    """
    if "vegas_spread" not in df_split.columns:
        return 0.0, 0, 0.0, 0.0

    vegas = df_split["vegas_spread"].values
    has_spread = ~np.isnan(vegas)

    if has_spread.sum() == 0:
        return 0.0, 0, 0.0, 0.0

    v = vegas[has_spread]
    pred_m = pred_margin[has_spread]
    actual_m = actual_margin[has_spread]

    # Vegas line: home covers if actual_margin + vegas_spread > 0
    # (vegas_spread is negative when home is favored, so this means
    #  home beats the spread)
    actual_cover = actual_m + v > 0  # Home covers

    # Our bet: we bet home covers if our predicted margin > -vegas_spread
    # i.e., we think home will do better than the spread implies
    our_bet_home = pred_m > -v  # We think home covers

    # For push (actual == -spread), it's a push — no win or loss
    push = np.abs(actual_m + v) < 0.5
    non_push = ~push

    if non_push.sum() == 0:
        return 0.5, int(has_spread.sum()), 0.0, 0.0

    # We win when our bet matches the actual outcome (excluding pushes)
    correct = (our_bet_home == actual_cover) & non_push
    n_bets = non_push.sum()
    ats_rate = correct.sum() / n_bets if n_bets > 0 else 0.5

    # Profit at -110 odds
    n_wins = correct.sum()
    n_losses = n_bets - n_wins
    profit = n_wins * 100 - n_losses * 110
    roi = profit / (n_bets * 110) if n_bets > 0 else 0.0

    return float(ats_rate), int(n_bets), float(profit), float(roi)


# ============================================================================
# 5. Feature Importance & SHAP
# ============================================================================


def print_feature_importance(
    model, feature_names: list[str], top_n: int = 30, label: str = "Spread"
):
    """Print top features by XGBoost importance (gain)."""
    logger = logging.getLogger(__name__)
    importance = model.feature_importances_

    idx = np.argsort(importance)[::-1][:top_n]

    logger.info(f"\nTop {top_n} Features ({label} model):")
    logger.info("-" * 50)
    for rank, i in enumerate(idx, 1):
        logger.info(f"  {rank:>2d}. {feature_names[i]:<40s} {importance[i]:.4f}")


def run_shap_analysis(
    model, X: np.ndarray, feature_names: list[str], label: str = "Spread"
):
    """Run SHAP analysis if the shap package is available."""
    logger = logging.getLogger(__name__)
    try:
        import shap
    except ImportError:
        logger.info(
            "shap not installed — skipping SHAP analysis. Run: pip install shap"
        )
        return None

    logger.info(f"Computing SHAP values for {label} model...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs_shap)[::-1][:30]

    logger.info(f"\nTop 30 SHAP Features ({label} model):")
    logger.info("-" * 50)
    for rank, i in enumerate(idx, 1):
        logger.info(f"  {rank:>2d}. {feature_names[i]:<40s} {mean_abs_shap[i]:.4f}")

    # Save SHAP summary
    shap_dir = CHECKPOINT_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=30,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            shap_dir / f"shap_summary_{label.lower()}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info(
            f"  Saved SHAP summary plot to {shap_dir / f'shap_summary_{label.lower()}.png'}"
        )
    except Exception as e:
        logger.warning(f"  Could not save SHAP plot: {e}")

    return mean_abs_shap


# ============================================================================
# 6. Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Exp 3: XGBoost + Optuna full feature optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials per model (default: 200)",
    )
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument("--llm-path", type=str, default=LLM_RESULTS_PATH)
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--no-betting", action="store_true", help="Ablation: exclude betting features"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Ablation: exclude LLM prediction features",
    )
    parser.add_argument(
        "--skip-ats-classifier",
        action="store_true",
        help="Skip the ATS cover classifier",
    )
    parser.add_argument(
        "--train-on-trainval",
        action="store_true",
        help="Retrain final model on train+val (default: train only with val early stop)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Load dotenv if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    include_betting = not args.no_betting
    include_llm = not args.no_llm

    ablation_tag = ""
    if not include_betting:
        ablation_tag += "_no_betting"
    if not include_llm:
        ablation_tag += "_no_llm"

    logger.info("=" * 70)
    logger.info("Phase 6 Experiment 3: XGBoost + Optuna + Full Feature Stack")
    logger.info("=" * 70)
    logger.info(f"  Optuna trials:   {args.n_trials}")
    logger.info(f"  Betting data:    {'YES' if include_betting else 'NO (ablation)'}")
    logger.info(f"  LLM predictions: {'YES' if include_llm else 'NO (ablation)'}")
    logger.info(f"  Train seasons:   {TRAIN_SEASONS[0]} — {TRAIN_SEASONS[-1]}")
    logger.info(f"  Val seasons:     {VAL_SEASONS}")
    logger.info(f"  Test seasons:    {TEST_SEASONS}")
    logger.info(f"  Retrain on T+V:  {'YES' if args.train_on_trainval else 'NO'}")

    # ---------------------------------------------------------------
    # Step 1: Build features
    # ---------------------------------------------------------------
    logger.info("\n--- Step 1: Building features ---")
    df, feature_names = build_all_features(
        db_path=args.db_path,
        include_betting=include_betting,
        include_llm=include_llm,
        llm_path=args.llm_path,
    )

    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Feature list: {feature_names}")

    # ---------------------------------------------------------------
    # Step 2: Split data
    # ---------------------------------------------------------------
    logger.info("\n--- Step 2: Splitting data ---")
    data = split_data(df, feature_names)

    logger.info(f"  Train: {data['X_train'].shape}")
    logger.info(f"  Val:   {data['X_val'].shape}")
    logger.info(f"  Test:  {data['X_test'].shape}")

    # Quick sanity: NaN counts per split
    for name, X in [
        ("Train", data["X_train"]),
        ("Val", data["X_val"]),
        ("Test", data["X_test"]),
    ]:
        nan_count = np.isnan(X).sum()
        nan_pct = nan_count / X.size * 100
        logger.info(f"  {name} NaN cells: {nan_count} ({nan_pct:.1f}%)")

    # ---------------------------------------------------------------
    # Step 3: Optuna — Spread Regression
    # ---------------------------------------------------------------
    logger.info("\n--- Step 3: Optimizing spread regression model ---")
    spread_params, spread_study = optimize_regression(
        data["X_train"],
        data["y_margin_train"],
        data["X_val"],
        data["y_margin_val"],
        n_trials=args.n_trials,
        study_name="spread_regression",
    )

    # Train final spread model
    logger.info("\n--- Training final spread model ---")
    if args.train_on_trainval:
        spread_model = train_final_model_on_trainval(
            spread_params,
            data["X_train"],
            data["y_margin_train"],
            data["X_val"],
            data["y_margin_val"],
            model_type="regressor",
        )
    else:
        spread_model = train_final_model(
            spread_params,
            data["X_train"],
            data["y_margin_train"],
            data["X_val"],
            data["y_margin_val"],
            model_type="regressor",
        )

    # Evaluate spread model
    val_spread_metrics = evaluate_spread_model(
        spread_model,
        data["X_val"],
        data["y_margin_val"],
        data["y_win_val"],
        data["df_val"],
        label="Validation",
    )
    test_spread_metrics = evaluate_spread_model(
        spread_model,
        data["X_test"],
        data["y_margin_test"],
        data["y_win_test"],
        data["df_test"],
        label="Test",
    )

    # ---------------------------------------------------------------
    # Step 4: Optuna — Win Classifier (optional, for pure win prediction)
    # ---------------------------------------------------------------
    logger.info("\n--- Step 4: Optimizing win classifier ---")
    win_params, win_study = optimize_classification(
        data["X_train"],
        data["y_win_train"],
        data["X_val"],
        data["y_win_val"],
        n_trials=args.n_trials,
        study_name="win_classifier",
    )

    if args.train_on_trainval:
        win_model = train_final_model_on_trainval(
            win_params,
            data["X_train"],
            data["y_win_train"],
            data["X_val"],
            data["y_win_val"],
            model_type="classifier",
        )
    else:
        win_model = train_final_model(
            win_params,
            data["X_train"],
            data["y_win_train"],
            data["X_val"],
            data["y_win_val"],
            model_type="classifier",
        )

    # Evaluate win classifier
    win_prob_test = win_model.predict_proba(data["X_test"])[:, 1]
    win_pred_test = (win_prob_test >= 0.5).astype(int)
    win_acc_test = accuracy_score(data["y_win_test"], win_pred_test)
    win_auc_test = roc_auc_score(data["y_win_test"], win_prob_test)
    win_brier_test = brier_score_loss(data["y_win_test"], win_prob_test)
    win_ece_test = compute_ece(win_prob_test, data["y_win_test"].astype(float))

    logger.info(f"\n  Win Classifier Test Results:")
    logger.info(f"  Win Accuracy: {win_acc_test * 100:.1f}%")
    logger.info(f"  Win AUC:      {win_auc_test:.3f}")
    logger.info(f"  Brier Score:  {win_brier_test:.4f}")
    logger.info(f"  ECE:          {win_ece_test:.4f}")

    # ---------------------------------------------------------------
    # Step 5: ATS Classifier (cover/not-cover)
    # ---------------------------------------------------------------
    ats_test_metrics = None
    ats_model = None
    ats_params = None
    ats_study = None

    if not args.skip_ats_classifier and include_betting:
        logger.info("\n--- Step 5: Building ATS cover classifier ---")

        # Build cover labels: home covers if actual_margin + vegas_spread > 0
        # Only for games that have Vegas spread
        def _build_cover_data(df_split, X, feature_names):
            if "vegas_spread" not in df_split.columns:
                return None, None, None
            vegas = df_split["vegas_spread"].values
            margin = df_split["target_margin"].values
            has_spread = ~np.isnan(vegas)

            # Exclude pushes
            cover_margin = margin + vegas
            non_push = np.abs(cover_margin) >= 0.5
            valid = has_spread & non_push

            if valid.sum() < 100:
                return None, None, None

            y_cover = (cover_margin[valid] > 0).astype(np.int32)
            X_cover = X[valid]
            return X_cover, y_cover, valid

        X_train_ats, y_train_ats, mask_train = _build_cover_data(
            data["df_train"], data["X_train"], feature_names
        )
        X_val_ats, y_val_ats, mask_val = _build_cover_data(
            data["df_val"], data["X_val"], feature_names
        )
        X_test_ats, y_test_ats, mask_test = _build_cover_data(
            data["df_test"], data["X_test"], feature_names
        )

        if X_train_ats is not None and X_val_ats is not None and X_test_ats is not None:
            logger.info(
                f"  ATS Train: {X_train_ats.shape}, cover rate: {y_train_ats.mean():.3f}"
            )
            logger.info(
                f"  ATS Val:   {X_val_ats.shape}, cover rate: {y_val_ats.mean():.3f}"
            )
            logger.info(
                f"  ATS Test:  {X_test_ats.shape}, cover rate: {y_test_ats.mean():.3f}"
            )

            ats_params, ats_study = optimize_classification(
                X_train_ats,
                y_train_ats,
                X_val_ats,
                y_val_ats,
                n_trials=args.n_trials,
                study_name="ats_classifier",
            )

            if args.train_on_trainval:
                ats_model = train_final_model_on_trainval(
                    ats_params,
                    X_train_ats,
                    y_train_ats,
                    X_val_ats,
                    y_val_ats,
                    model_type="classifier",
                )
            else:
                ats_model = train_final_model(
                    ats_params,
                    X_train_ats,
                    y_train_ats,
                    X_val_ats,
                    y_val_ats,
                    model_type="classifier",
                )

            ats_test_metrics = evaluate_ats_classifier(
                ats_model,
                X_test_ats,
                y_test_ats,
                data["df_test"][mask_test].reset_index(drop=True),
                mask_test,
                label="Test",
            )
        else:
            logger.warning("Insufficient betting data for ATS classifier")

    # ---------------------------------------------------------------
    # Step 6: Feature importance
    # ---------------------------------------------------------------
    logger.info("\n--- Step 6: Feature importance ---")
    print_feature_importance(spread_model, feature_names, top_n=30, label="Spread")
    print_feature_importance(win_model, feature_names, top_n=20, label="Win")
    if ats_model is not None:
        print_feature_importance(ats_model, feature_names, top_n=20, label="ATS")

    # ---------------------------------------------------------------
    # Step 7: SHAP analysis
    # ---------------------------------------------------------------
    logger.info("\n--- Step 7: SHAP analysis ---")
    # Use a sample of test data for SHAP (faster)
    n_shap_samples = min(500, data["X_test"].shape[0])
    shap_X = data["X_test"][:n_shap_samples]
    run_shap_analysis(spread_model, shap_X, feature_names, label="Spread")
    run_shap_analysis(win_model, shap_X, feature_names, label="Win")

    # ---------------------------------------------------------------
    # Step 8: Save everything
    # ---------------------------------------------------------------
    logger.info("\n--- Step 8: Saving models and results ---")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spread_model.save_model(str(output_dir / "xgb_spread.json"))
    win_model.save_model(str(output_dir / "xgb_win.json"))
    if ats_model is not None:
        ats_model.save_model(str(output_dir / "xgb_ats.json"))

    # Save Optuna studies
    if joblib is not None:
        joblib.dump(spread_study, str(output_dir / "spread_study.pkl"))
        joblib.dump(win_study, str(output_dir / "win_study.pkl"))
        if ats_study is not None:
            joblib.dump(ats_study, str(output_dir / "ats_study.pkl"))
        logger.info(f"  Saved Optuna studies to {output_dir}")

    # Save feature names
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save comprehensive results
    elapsed = time.time() - start_time
    results = {
        "experiment": "Phase 6 Exp 3: XGBoost + Optuna",
        "n_trials": args.n_trials,
        "include_betting": include_betting,
        "include_llm": include_llm,
        "train_on_trainval": args.train_on_trainval,
        "n_features": len(feature_names),
        "train_size": int(data["X_train"].shape[0]),
        "val_size": int(data["X_val"].shape[0]),
        "test_size": int(data["X_test"].shape[0]),
        "train_seasons": TRAIN_SEASONS,
        "val_seasons": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
        "spread_model": {
            "best_params": spread_params,
            "best_val_mae": float(spread_study.best_value),
            "val_metrics": val_spread_metrics,
            "test_metrics": test_spread_metrics,
        },
        "win_model": {
            "best_params": win_params,
            "best_val_accuracy": float(win_study.best_value),
            "test_accuracy": float(win_acc_test),
            "test_auc": float(win_auc_test),
            "test_brier": float(win_brier_test),
            "test_ece": float(win_ece_test),
        },
        "elapsed_seconds": elapsed,
    }

    if ats_test_metrics is not None:
        # Convert selective_results values for JSON
        selective_json = {}
        for k, v in ats_test_metrics.get("selective_results", {}).items():
            selective_json[str(k)] = v
        ats_json = {
            k: v for k, v in ats_test_metrics.items() if k != "selective_results"
        }
        ats_json["selective_results"] = selective_json
        results["ats_model"] = {
            "best_params": ats_params,
            "best_val_accuracy": float(ats_study.best_value) if ats_study else None,
            "test_metrics": ats_json,
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Saved results to {output_dir / 'results.json'}")

    # ---------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("Phase 6 Exp 3: XGBoost + Optuna Results")
    print("=" * 70)
    print(f"Features:           {len(feature_names)}")
    print(f"Optuna trials:      {args.n_trials} per model")
    print(f"Best val MAE:       {spread_study.best_value:.2f}")
    print()
    print(f"Best hyperparameters (spread):")
    for k, v in spread_params.items():
        print(f"  {k}: {v}")
    print()
    print(f"Test Results (Spread Regression):")
    print(f"  Spread MAE:   {test_spread_metrics['spread_mae']:.2f}")
    print(f"  Spread RMSE:  {test_spread_metrics['spread_rmse']:.2f}")
    print(f"  Win AUC:      {test_spread_metrics['win_auc']:.3f}")
    print(f"  Win Accuracy: {test_spread_metrics['win_accuracy'] * 100:.1f}%")
    print(
        f"  ATS Rate:     {test_spread_metrics['ats_rate'] * 100:.1f}% ({test_spread_metrics['ats_n_games']} games)"
    )
    print(f"  ATS Profit:   ${test_spread_metrics['ats_profit']:.0f} at -110")
    print(f"  ATS ROI:      {test_spread_metrics['ats_roi'] * 100:.1f}%")
    print()
    print(f"Test Results (Win Classifier):")
    print(f"  Win Accuracy: {win_acc_test * 100:.1f}%")
    print(f"  Win AUC:      {win_auc_test:.3f}")
    print(f"  Brier Score:  {win_brier_test:.4f}")
    print(f"  ECE:          {win_ece_test:.4f}")

    if ats_test_metrics is not None:
        print()
        print(f"Test Results (ATS Cover Classifier):")
        print(
            f"  ATS Accuracy: {ats_test_metrics['ats_accuracy'] * 100:.1f}% ({ats_test_metrics['ats_n_games']} games)"
        )
        print(f"  ATS AUC:      {ats_test_metrics['ats_auc']:.3f}")
        print(f"  Flat Profit:  ${ats_test_metrics['ats_profit_flat']:.0f} at -110")
        print(f"  Flat ROI:     {ats_test_metrics['ats_roi_flat'] * 100:.1f}%")
        if ats_test_metrics.get("selective_results"):
            print(f"  --- Selective Betting ---")
            for thresh, res in ats_test_metrics["selective_results"].items():
                print(
                    f"  Conf >= {thresh:.0%}: {res['ats_rate'] * 100:.1f}% "
                    f"on {res['n_bets']} bets, "
                    f"profit ${res['profit']:.0f}, ROI {res['roi'] * 100:.1f}%"
                )

    print()
    print(f"Top 20 Features (spread model):")
    importance = spread_model.feature_importances_
    idx = np.argsort(importance)[::-1][:20]
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:>2d}. {feature_names[i]:<40s} {importance[i]:.4f}")

    print()
    print(f"Reference Baselines:")
    print(f"  Vegas:                  MAE ~9.45")
    print(f"  Phase 3 Exp 9 ensemble: MAE 10.66, AUC 0.718")
    print(f"  Phase 3 Exp 4 single:   MAE 10.83, AUC 0.705")
    print(f"  Phase 3 Exp 8 hybrid:   MAE 10.85")
    print("=" * 70)
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print(f"Models saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
