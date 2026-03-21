#!/usr/bin/env python
"""
Phase 6 Experiment 2: Engineered Features Meta-Model

Combines ~130 rolling-efficiency features (expanded from Phase 3 Exp 8's 63)
with optional Vegas betting lines and LLM predictions in XGBoost / Ridge / MLP
meta-models. Evaluates on Spread MAE, Win AUC, and ATS%.

Feature groups:
  A. Rolling efficiency (3 windows x 11 metrics = 33 per team, differenced)
  B. EWM time-decay (2 spans x 6 metrics = 12 per team, differenced)
  C. Record/streak features (3 windows + streak, differenced)
  D. Rest/scheduling (home_rest, away_rest, home_b2b, away_b2b, season_progress)
  E. H2H features (win_rate, avg_margin, count)
  F. Venue win% (home team's home record, away team's away record)
  G. Absolute team stats (pace, ppg, opp_ppg, ortg, drtg x2 = 10)
  H. Vegas betting features (spread, total, moneyline, opening spread, line movement)
  I. LLM predictions (predicted spread, total, win_prob from GPT models)

Data splits: Train 2008-2022, Val 2023-2024, Test 2024-2026

Usage:
    python scripts/phase6_exp2_engineered_features.py
    python scripts/phase6_exp2_engineered_features.py --n-trials 200
    python scripts/phase6_exp2_engineered_features.py --skip-betting --skip-llm
    python scripts/phase6_exp2_engineered_features.py --run-ablations
"""

import argparse
import json
import logging
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = "data/NBA_AI_full.sqlite"

# Historical team abbreviation mapping
HISTORICAL_MAP = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
}

# Data split boundaries (season strings)
TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2023)]  # 2008-2009 .. 2022-2023
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]
ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Data loading
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


def load_betting_data(db_path: str) -> pd.DataFrame:
    """Load betting data from Betting table."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        b.game_id,
        b.espn_opening_spread,
        b.espn_current_spread,
        b.espn_closing_spread,
        b.covers_closing_spread,
        b.espn_opening_total,
        b.espn_current_total,
        b.espn_closing_total,
        b.covers_closing_total,
        b.espn_current_ml_home,
        b.espn_current_ml_away,
        b.espn_closing_ml_home,
        b.espn_closing_ml_away
    FROM Betting b
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Compute best-available spread/total (cascading fallback)
    df["vegas_spread"] = (
        df["espn_closing_spread"]
        .fillna(df["covers_closing_spread"])
        .fillna(df["espn_current_spread"])
    )

    df["vegas_total"] = (
        df["espn_closing_total"]
        .fillna(df["covers_closing_total"])
        .fillna(df["espn_current_total"])
    )

    # Opening spread (for line movement feature)
    df["vegas_opening_spread"] = df["espn_opening_spread"].fillna(
        df["espn_current_spread"]
    )
    df["vegas_opening_total"] = df["espn_opening_total"].fillna(
        df["espn_current_total"]
    )

    # Line movement: closing - opening
    df["spread_movement"] = df["vegas_spread"] - df["vegas_opening_spread"]
    df["total_movement"] = df["vegas_total"] - df["vegas_opening_total"]

    # Moneyline: best available
    df["ml_home"] = df["espn_closing_ml_home"].fillna(df["espn_current_ml_home"])
    df["ml_away"] = df["espn_closing_ml_away"].fillna(df["espn_current_ml_away"])

    # Convert moneyline to implied probability
    def ml_to_prob(ml):
        """Convert American moneyline to implied probability."""
        try:
            ml = float(ml)
        except (ValueError, TypeError):
            return np.nan
        if pd.isna(ml) or ml == 0:
            return np.nan
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return abs(ml) / (abs(ml) + 100.0)

    df["ml_home_prob"] = df["ml_home"].apply(ml_to_prob)
    df["ml_away_prob"] = df["ml_away"].apply(ml_to_prob)

    return df


def load_llm_predictions(data_dir: str = "data/exp7") -> pd.DataFrame:
    """Load LLM predictions from JSONL files."""
    logger = logging.getLogger(__name__)
    data_path = Path(data_dir)
    all_preds = []

    for fpath in sorted(data_path.glob("*_results.jsonl")):
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("success"):
                    continue
                pred = rec.get("prediction", {})
                all_preds.append(
                    {
                        "game_id": rec["game_id"],
                        "model": rec["model"],
                        "llm_home_score": pred.get("home_score"),
                        "llm_away_score": pred.get("away_score"),
                        "llm_home_win_prob": pred.get("home_win_probability"),
                    }
                )

    if not all_preds:
        logger.warning("No LLM predictions found")
        return pd.DataFrame()

    df = pd.DataFrame(all_preds)
    df["llm_spread"] = df["llm_home_score"] - df["llm_away_score"]
    df["llm_total"] = df["llm_home_score"] + df["llm_away_score"]

    # Pivot so each model becomes separate columns
    models = df["model"].unique()
    logger.info(f"Found LLM predictions from {len(models)} models: {list(models)}")

    pivot_dfs = []
    for model in models:
        mdf = df[df["model"] == model].copy()
        model_tag = model.replace("-", "_").replace(".", "_")
        mdf = mdf.rename(
            columns={
                "llm_spread": f"llm_{model_tag}_spread",
                "llm_total": f"llm_{model_tag}_total",
                "llm_home_win_prob": f"llm_{model_tag}_win_prob",
                "llm_home_score": f"llm_{model_tag}_home_score",
                "llm_away_score": f"llm_{model_tag}_away_score",
            }
        )
        mdf = mdf.drop(columns=["model"])
        pivot_dfs.append(mdf)

    # Merge all model predictions on game_id
    result = pivot_dfs[0]
    for mdf in pivot_dfs[1:]:
        result = result.merge(mdf, on="game_id", how="outer")

    # Also add ensemble average across models
    spread_cols = [c for c in result.columns if c.endswith("_spread") and "llm_" in c]
    total_cols = [c for c in result.columns if c.endswith("_total") and "llm_" in c]
    prob_cols = [c for c in result.columns if c.endswith("_win_prob") and "llm_" in c]

    if len(spread_cols) > 1:
        result["llm_ensemble_spread"] = result[spread_cols].mean(axis=1)
        result["llm_ensemble_total"] = result[total_cols].mean(axis=1)
        result["llm_ensemble_win_prob"] = result[prob_cols].mean(axis=1)

    logger.info(
        f"Loaded {len(result)} games with LLM predictions, "
        f"{len(result.columns) - 1} feature columns"
    )

    return result


# ============================================================================
# Feature engineering
# ============================================================================


def compute_efficiency_stats(group: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game efficiency stats for a team."""
    g = group.copy()

    if "possessions" not in g.columns:
        raise ValueError("possessions column must be pre-computed")

    # Efficiency metrics
    g["fg_pct"] = np.where(g["fga"] > 0, g["fgm"] / g["fga"], 0.0)
    g["fg3_pct"] = np.where(g["fg3a"] > 0, g["fg3m"] / g["fg3a"], 0.0)
    g["ft_pct"] = np.where(g["fta"] > 0, g["ftm"] / g["fta"], 0.0)
    g["efg_pct"] = np.where(g["fga"] > 0, (g["fgm"] + 0.5 * g["fg3m"]) / g["fga"], 0.0)
    g["ts_pct"] = np.where(
        g["fga"] + 0.44 * g["fta"] > 0,
        g["pts"] / (2 * (g["fga"] + 0.44 * g["fta"])),
        0.0,
    )
    g["tov_pct"] = np.where(g["possessions"] > 0, g["tov"] / g["possessions"], 0.0)
    g["ft_rate"] = np.where(g["fga"] > 0, g["fta"] / g["fga"], 0.0)
    g["fg3a_rate"] = np.where(g["fga"] > 0, g["fg3a"] / g["fga"], 0.0)
    g["ast_ratio"] = np.where(g["fgm"] > 0, g["ast"] / g["fgm"], 0.0)
    g["oreb_rate"] = np.where(g["reb"] > 0, g["team_oreb"] / g["reb"], 0.0)

    # Ratings (per 100 possessions)
    g["ortg"] = np.where(g["possessions"] > 0, g["pts"] / g["possessions"] * 100, 0.0)
    g["drtg"] = np.where(
        g["possessions"] > 0, g["pts_allowed"] / g["possessions"] * 100, 0.0
    )
    g["net_rtg"] = g["ortg"] - g["drtg"]
    g["margin"] = g["pts"] - g["pts_allowed"]

    # Win indicator
    g["win"] = (g["plus_minus"] > 0).astype(float)

    return g


def compute_rolling_features(team_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling window features for a single team's games (within a season)."""
    result = team_df[["game_id", "team_id", "date", "season"]].copy()

    # -- Rolling efficiency metrics --
    rolling_metrics = [
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "efg_pct",
        "ts_pct",
        "tov_pct",
        "ft_rate",
        "fg3a_rate",
        "ast_ratio",
        "oreb_rate",
        "ortg",
        "drtg",
        "net_rtg",
    ]

    for w in windows:
        for m in rolling_metrics:
            col = f"{m}_roll{w}"
            result[col] = team_df[m].shift(1).rolling(window=w, min_periods=1).mean()

    # -- Rolling counting stats (per game) --
    counting_stats = ["pts", "pts_allowed", "reb", "ast", "tov", "stl", "blk"]
    for w in windows:
        for s in counting_stats:
            col = f"{s}_roll{w}"
            result[col] = team_df[s].shift(1).rolling(window=w, min_periods=1).mean()

    # -- Pace (rolling 20) --
    result["pace_roll20"] = (
        team_df["possessions"].shift(1).rolling(window=20, min_periods=1).mean()
    )

    # -- Win percentage (rolling) --
    for w in windows:
        result[f"win_pct_roll{w}"] = (
            team_df["win"].shift(1).rolling(window=w, min_periods=1).mean()
        )

    # -- EWM features (captures momentum/recency) --
    ewm_metrics = ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "margin"]
    ewm_spans = [5, 15]
    for span in ewm_spans:
        for m in ewm_metrics:
            col = f"{m}_ewm{span}"
            result[col] = team_df[m].shift(1).ewm(span=span, min_periods=1).mean()

    # -- Win streak --
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
        streak.append(np.clip(current_streak, -15, 15) / 15.0)
    result["streak"] = streak

    # -- Venue-specific win% --
    home_wins = team_df["win"].where(team_df["is_home"]).shift(1)
    result["home_win_pct"] = home_wins.expanding(min_periods=1).mean()

    away_wins = team_df["win"].where(~team_df["is_home"]).shift(1)
    result["away_win_pct"] = away_wins.expanding(min_periods=1).mean()

    # -- Schedule features --
    result["is_home"] = team_df["is_home"].values
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
    h2h_lookup: dict, home_team: str, away_team: str, date, season: str
) -> dict:
    """Compute H2H features from recent meetings between home and away team."""
    key = tuple(sorted([home_team, away_team]))
    meetings = h2h_lookup.get(key, [])

    # Filter to prior meetings only
    prior = [(d, ht, m) for d, ht, m in meetings if d < date]
    prior_recent = prior[-5:]  # Last 5
    prior_season = [
        (d, ht, m) for d, ht, m in prior if d.year >= int(season.split("-")[0])
    ]

    result = {}

    if len(prior_recent) == 0:
        result["h2h_win_rate"] = 0.5
        result["h2h_avg_margin"] = 0.0
        result["h2h_count"] = 0.0
        result["h2h_season_win_rate"] = 0.5
    else:
        wins = 0
        margins = []
        for _, ht, m in prior_recent:
            margin = m if ht == home_team else -m
            if margin > 0:
                wins += 1
            margins.append(margin)

        result["h2h_win_rate"] = wins / len(prior_recent)
        result["h2h_avg_margin"] = np.mean(margins) / 13.0  # Normalize
        result["h2h_count"] = min(len(prior_recent), 10) / 10.0

        # Season-only H2H
        if prior_season:
            s_wins = sum(
                1 for _, ht, m in prior_season if (m if ht == home_team else -m) > 0
            )
            result["h2h_season_win_rate"] = s_wins / len(prior_season)
        else:
            result["h2h_season_win_rate"] = 0.5

    return result


# ============================================================================
# Game-level feature assembly
# ============================================================================


def build_game_features(db_path: str = DB_PATH) -> pd.DataFrame:
    """Build feature DataFrame for all games: one row per game."""
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading TeamBox data for {len(ALL_SEASONS)} seasons...")
    df = load_teambox_data(db_path, ALL_SEASONS)
    logger.info(f"  {len(df)} team-game rows")

    # Load oreb for pace correction
    logger.info("Loading oreb from PlayerBox...")
    oreb_df = load_oreb_data(db_path, ALL_SEASONS)
    df = df.merge(oreb_df, on=["game_id", "team_id"], how="left")
    df["team_oreb"] = df["team_oreb"].fillna(0)
    df["possessions"] = df["fga"] + 0.44 * df["fta"] + df["tov"] - df["team_oreb"]
    df["possessions"] = df["possessions"].clip(lower=50)

    # Compute efficiency stats
    logger.info("Computing efficiency stats...")
    df = compute_efficiency_stats(df)

    # Build per-team rolling features
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

    # Build game-level score margins for H2H
    games_df = df[df["is_home"]].copy()
    games_df["home_margin"] = games_df["plus_minus"]
    h2h_lookup = build_h2h_lookup(games_df)

    # Pre-build target lookup to avoid O(n^2) per-game DataFrame filtering
    target_lookup = {}
    for _, trow in df.iterrows():
        gid_t = trow["game_id"]
        if trow["is_home"]:
            target_lookup.setdefault(gid_t, {})["home_pts"] = float(trow["pts"])
            target_lookup.setdefault(gid_t, {})["home_pm"] = float(trow["plus_minus"])
        else:
            target_lookup.setdefault(gid_t, {})["away_pts"] = float(trow["pts"])

    # Build game-level feature matrix
    logger.info("Assembling game-level features...")
    unique_games = df.drop_duplicates("game_id")[
        ["game_id", "date", "home_team", "away_team", "season"]
    ].sort_values("date")

    rows = []

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

        row = {
            "game_id": gid,
            "season": season,
            "date": date,
            "home_team": home,
            "away_team": away,
        }

        # -----------------------------------------------------------
        # A. Rolling efficiency — differenced (home - away)
        # -----------------------------------------------------------
        rolling_metrics = [
            "fg_pct",
            "fg3_pct",
            "ft_pct",
            "efg_pct",
            "ts_pct",
            "tov_pct",
            "ft_rate",
            "fg3a_rate",
            "ast_ratio",
            "oreb_rate",
            "ortg",
            "drtg",
            "net_rtg",
        ]
        for w in windows:
            for m in rolling_metrics:
                col = f"{m}_roll{w}"
                h_val = _safe_val(hf, col)
                a_val = _safe_val(af, col)
                row[f"diff_{col}"] = h_val - a_val

        # Rolling counting stats — differenced
        counting_stats = ["pts", "pts_allowed", "reb", "ast", "tov", "stl", "blk"]
        for w in windows:
            for s in counting_stats:
                col = f"{s}_roll{w}"
                h_val = _safe_val(hf, col)
                a_val = _safe_val(af, col)
                row[f"diff_{col}"] = h_val - a_val

        # -----------------------------------------------------------
        # B. EWM features — differenced
        # -----------------------------------------------------------
        ewm_metrics = ["net_rtg", "ortg", "drtg", "efg_pct", "ts_pct", "margin"]
        ewm_spans = [5, 15]
        for span in ewm_spans:
            for m in ewm_metrics:
                col = f"{m}_ewm{span}"
                h_val = _safe_val(hf, col)
                a_val = _safe_val(af, col)
                row[f"diff_{col}"] = h_val - a_val

        # -----------------------------------------------------------
        # C. Record/streak features — differenced
        # -----------------------------------------------------------
        for w in windows:
            h_val = _safe_val(hf, f"win_pct_roll{w}", default=0.5)
            a_val = _safe_val(af, f"win_pct_roll{w}", default=0.5)
            row[f"diff_win_pct_roll{w}"] = h_val - a_val

        row["diff_streak"] = _safe_val(hf, "streak") - _safe_val(af, "streak")

        # -----------------------------------------------------------
        # D. Rest/scheduling
        # -----------------------------------------------------------
        row["home_rest_days"] = _safe_val(hf, "rest_days", default=0.5)
        row["away_rest_days"] = _safe_val(af, "rest_days", default=0.5)
        row["rest_advantage"] = row["home_rest_days"] - row["away_rest_days"]
        row["home_b2b"] = _safe_val(hf, "b2b")
        row["away_b2b"] = _safe_val(af, "b2b")
        row["season_progress"] = _safe_val(hf, "season_progress", default=0.5)

        # -----------------------------------------------------------
        # E. H2H features
        # -----------------------------------------------------------
        h2h = compute_h2h_features(h2h_lookup, home, away, date, season)
        row.update(h2h)

        # -----------------------------------------------------------
        # F. Venue win%
        # -----------------------------------------------------------
        row["home_team_home_win_pct"] = _safe_val(hf, "home_win_pct", default=0.5)
        row["away_team_away_win_pct"] = _safe_val(af, "away_win_pct", default=0.5)
        row["venue_win_pct_diff"] = (
            row["home_team_home_win_pct"] - row["away_team_away_win_pct"]
        )

        # -----------------------------------------------------------
        # G. Absolute team stats (home + away, not differenced)
        # -----------------------------------------------------------
        for metric in [
            "ortg_roll20",
            "drtg_roll20",
            "pace_roll20",
            "pts_roll20",
            "pts_allowed_roll20",
        ]:
            row[f"home_{metric}"] = _safe_val(hf, metric)
            row[f"away_{metric}"] = _safe_val(af, metric)

        # -----------------------------------------------------------
        # Targets (not features — included for convenience)
        # -----------------------------------------------------------
        tgt = target_lookup.get(gid, {})
        if "home_pm" not in tgt or "home_pts" not in tgt or "away_pts" not in tgt:
            continue

        row["target_spread"] = tgt["home_pm"]  # home margin
        row["target_home_score"] = tgt["home_pts"]
        row["target_away_score"] = tgt["away_pts"]
        row["target_total"] = row["target_home_score"] + row["target_away_score"]
        row["target_home_win"] = int(row["target_spread"] > 0)

        rows.append(row)

    game_df = pd.DataFrame(rows)
    logger.info(f"Built {len(game_df)} games with {len(game_df.columns)} columns")

    return game_df


def _safe_val(series, key, default=0.0):
    """Safely get a value, returning default if missing or NaN."""
    val = series.get(key, default)
    if pd.isna(val):
        return default
    return float(val)


# ============================================================================
# Betting + LLM feature merging
# ============================================================================


def merge_betting_features(
    game_df: pd.DataFrame, db_path: str
) -> tuple[pd.DataFrame, list[str]]:
    """Merge betting features into game DataFrame."""
    logger = logging.getLogger(__name__)

    betting_df = load_betting_data(db_path)
    n_before = len(game_df)
    game_df = game_df.merge(betting_df, on="game_id", how="left")
    assert len(game_df) == n_before, "Betting merge changed row count"

    betting_cols = [
        "vegas_spread",
        "vegas_total",
        "vegas_opening_spread",
        "vegas_opening_total",
        "spread_movement",
        "total_movement",
        "ml_home_prob",
        "ml_away_prob",
    ]

    # Count how many games have betting data
    has_spread = game_df["vegas_spread"].notna().sum()
    logger.info(
        f"Betting data available: {has_spread}/{len(game_df)} games have spreads"
    )

    return game_df, betting_cols


def merge_llm_features(game_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge LLM prediction features into game DataFrame."""
    logger = logging.getLogger(__name__)

    llm_df = load_llm_predictions()
    if llm_df.empty:
        logger.info("No LLM predictions to merge")
        return game_df, []

    n_before = len(game_df)
    game_df = game_df.merge(llm_df, on="game_id", how="left")
    assert len(game_df) == n_before, "LLM merge changed row count"

    llm_cols = [c for c in llm_df.columns if c != "game_id"]
    has_llm = game_df[llm_cols[0]].notna().sum() if llm_cols else 0
    logger.info(
        f"LLM predictions available: {has_llm}/{len(game_df)} games, "
        f"{len(llm_cols)} columns"
    )

    return game_df, llm_cols


# ============================================================================
# Model training and evaluation
# ============================================================================


def get_feature_columns(
    game_df: pd.DataFrame,
    betting_cols: list[str],
    llm_cols: list[str],
    include_betting: bool,
    include_llm: bool,
) -> list[str]:
    """Get list of feature column names (excluding targets and metadata)."""
    exclude = {
        "game_id",
        "season",
        "date",
        "home_team",
        "away_team",
        "target_spread",
        "target_home_score",
        "target_away_score",
        "target_total",
        "target_home_win",
    }

    # Also exclude raw betting columns we derived features from
    raw_betting = {
        "espn_opening_spread",
        "espn_current_spread",
        "espn_closing_spread",
        "covers_closing_spread",
        "espn_opening_total",
        "espn_current_total",
        "espn_closing_total",
        "covers_closing_total",
        "espn_current_ml_home",
        "espn_current_ml_away",
        "espn_closing_ml_home",
        "espn_closing_ml_away",
        "ml_home",
        "ml_away",
    }
    exclude.update(raw_betting)

    if not include_betting:
        exclude.update(betting_cols)
    if not include_llm:
        exclude.update(llm_cols)

    feat_cols = [c for c in game_df.columns if c not in exclude]
    return feat_cols


def split_data(game_df: pd.DataFrame, feat_cols: list[str]):
    """Split into train/val/test based on season."""
    train_mask = game_df["season"].isin(TRAIN_SEASONS)
    val_mask = game_df["season"].isin(VAL_SEASONS)
    test_mask = game_df["season"].isin(TEST_SEASONS)

    X_train = game_df.loc[train_mask, feat_cols].values.astype(np.float32)
    X_val = game_df.loc[val_mask, feat_cols].values.astype(np.float32)
    X_test = game_df.loc[test_mask, feat_cols].values.astype(np.float32)

    # Replace NaN with 0
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    targets = {}
    for split_name, mask in [
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ]:
        targets[f"{split_name}_spread"] = game_df.loc[mask, "target_spread"].values
        targets[f"{split_name}_total"] = game_df.loc[mask, "target_total"].values
        targets[f"{split_name}_home_score"] = game_df.loc[
            mask, "target_home_score"
        ].values
        targets[f"{split_name}_away_score"] = game_df.loc[
            mask, "target_away_score"
        ].values
        targets[f"{split_name}_win"] = game_df.loc[mask, "target_home_win"].values

    # Vegas spreads for ATS evaluation
    if "vegas_spread" in game_df.columns:
        for split_name, mask in [
            ("train", train_mask),
            ("val", val_mask),
            ("test", test_mask),
        ]:
            targets[f"{split_name}_vegas_spread"] = game_df.loc[
                mask, "vegas_spread"
            ].values
    else:
        for split_name in ["train", "val", "test"]:
            targets[f"{split_name}_vegas_spread"] = None

    return X_train, X_val, X_test, targets


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


def compute_ats_metrics(
    pred_spread: np.ndarray, vegas_spread: np.ndarray, actual_spread: np.ndarray
) -> dict:
    """
    Compute against-the-spread (ATS) metrics.

    Sign convention: negative spread = home favored.
    Predicted margin = -spread (positive = home wins by more).
    We bet home ATS when our predicted margin > -vegas_spread (i.e., we think
    home will beat the spread).

    pred_spread: our predicted home margin (positive = home wins)
    vegas_spread: Vegas spread (negative = home favored)
    actual_spread: actual home margin
    """
    if vegas_spread is None:
        return {}

    # Only evaluate games with Vegas spread available
    valid = ~np.isnan(vegas_spread)
    if valid.sum() == 0:
        return {}

    pred = pred_spread[valid]
    vegas = vegas_spread[valid]
    actual = actual_spread[valid]

    # Vegas implied margin for home team: -vegas_spread
    # (e.g., if vegas_spread = -7, home is favored by 7, expected margin = 7)
    vegas_implied_margin = -vegas

    # We pick home ATS when our predicted margin > vegas implied margin
    # (we think home will outperform the spread)
    our_pick_home = pred > vegas_implied_margin

    # Actual: did home beat the spread?
    # Home covers when actual_margin > vegas_implied_margin
    home_covered = actual > vegas_implied_margin

    # Pushes (actual == spread line)
    push = actual == vegas_implied_margin

    # ATS record (excluding pushes)
    not_push = ~push
    if not_push.sum() == 0:
        return {"ats_games": 0}

    ats_correct = (our_pick_home == home_covered)[not_push]
    ats_wins = ats_correct.sum()
    ats_total = not_push.sum()
    ats_pct = ats_wins / ats_total

    # Profit/loss at -110 (risk 110 to win 100)
    # Win: +100, Loss: -110
    profit = np.where(ats_correct, 100.0, -110.0).sum()
    roi = profit / (ats_total * 110.0)

    # Confidence-tiered ATS (based on magnitude of disagreement with Vegas)
    disagreement = np.abs(pred - vegas_implied_margin)[not_push]
    ats_correct_np = ats_correct

    result = {
        "ats_wins": int(ats_wins),
        "ats_total": int(ats_total),
        "ats_pushes": int(push.sum()),
        "ats_pct": float(ats_pct),
        "ats_profit": float(profit),
        "ats_roi": float(roi),
    }

    # Tiered analysis: only bet when disagreement exceeds threshold
    for threshold in [0.0, 1.0, 2.0, 3.0, 5.0]:
        tier_mask = disagreement > threshold
        if tier_mask.sum() > 0:
            tier_correct = ats_correct_np[tier_mask]
            tier_wins = tier_correct.sum()
            tier_total = tier_mask.sum()
            tier_pct = tier_wins / tier_total
            tier_profit = np.where(tier_correct, 100.0, -110.0).sum()
            tier_roi = tier_profit / (tier_total * 110.0)
            result[f"ats_pct_thresh{threshold}"] = float(tier_pct)
            result[f"ats_games_thresh{threshold}"] = int(tier_total)
            result[f"ats_roi_thresh{threshold}"] = float(tier_roi)

    return result


def evaluate_model(
    pred_spread: np.ndarray,
    pred_win_prob: np.ndarray,
    actual_spread: np.ndarray,
    actual_win: np.ndarray,
    home_scores: np.ndarray,
    away_scores: np.ndarray,
    vegas_spread: np.ndarray = None,
    label: str = "Test",
) -> dict:
    """Evaluate predictions: Spread MAE, Win AUC, ATS%."""
    from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score

    spread_mae = mean_absolute_error(actual_spread, pred_spread)
    spread_rmse = np.sqrt(np.mean((actual_spread - pred_spread) ** 2))

    pred_win = (pred_win_prob >= 0.5).astype(int)
    win_acc = accuracy_score(actual_win, pred_win)
    win_auc = roc_auc_score(actual_win, pred_win_prob)
    ece = compute_ece(pred_win_prob, actual_win)

    metrics = {
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "win_accuracy": win_acc,
        "win_auc": win_auc,
        "ece": ece,
    }

    # ATS metrics
    ats = compute_ats_metrics(pred_spread, vegas_spread, actual_spread)
    metrics.update(ats)

    # Vegas spread MAE (baseline)
    if vegas_spread is not None:
        valid = ~np.isnan(vegas_spread)
        if valid.sum() > 0:
            # Vegas predicted margin = -vegas_spread
            vegas_pred_margin = -vegas_spread[valid]
            vegas_mae = mean_absolute_error(actual_spread[valid], vegas_pred_margin)
            metrics["vegas_spread_mae"] = vegas_mae

    return metrics


def print_metrics(metrics: dict, label: str):
    """Pretty-print evaluation metrics."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  {label}")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Spread MAE:    {metrics['spread_mae']:.3f}")
    logger.info(f"  Spread RMSE:   {metrics['spread_rmse']:.3f}")
    if "vegas_spread_mae" in metrics:
        logger.info(f"  Vegas MAE:     {metrics['vegas_spread_mae']:.3f}  (baseline)")
    logger.info(f"  Win Accuracy:  {metrics['win_accuracy']:.3f}")
    logger.info(f"  Win AUC:       {metrics['win_auc']:.3f}")
    logger.info(f"  ECE:           {metrics['ece']:.4f}")
    if "ats_pct" in metrics:
        logger.info(f"  ---")
        logger.info(
            f"  ATS Record:    {metrics['ats_wins']}/{metrics['ats_total']} "
            f"({metrics['ats_pct']:.3f})  [pushes: {metrics.get('ats_pushes', 0)}]"
        )
        logger.info(
            f"  ATS Profit:    ${metrics['ats_profit']:.0f}  "
            f"(ROI: {metrics['ats_roi']:.3f})"
        )
        logger.info(f"  Breakeven:     52.4% at -110")
        # Tiered
        for t in [1.0, 2.0, 3.0, 5.0]:
            key = f"ats_pct_thresh{t}"
            if key in metrics:
                n = metrics[f"ats_games_thresh{t}"]
                pct = metrics[key]
                roi = metrics[f"ats_roi_thresh{t}"]
                logger.info(f"  ATS (>{t}pt):   {pct:.3f} ({n} games, ROI {roi:.3f})")
    logger.info(f"{'=' * 70}")


# ============================================================================
# Model definitions
# ============================================================================


def train_xgboost(
    X_train, y_train, X_val, y_val, n_trials: int, task: str = "regression"
):
    """Train XGBoost with Optuna hyperparameter tuning."""
    import optuna

    logger = logging.getLogger(__name__)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if task == "regression":

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            from xgboost import XGBRegressor

            model = XGBRegressor(
                **params,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=50,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            from sklearn.metrics import mean_absolute_error

            pred = model.predict(X_val)
            return mean_absolute_error(y_val, pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"  Best val MAE: {study.best_value:.4f}")

        from xgboost import XGBRegressor

        best_model = XGBRegressor(
            **study.best_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return best_model, study.best_params

    elif task == "classification":

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            from xgboost import XGBClassifier

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                tree_method="hist",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=50,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            from sklearn.metrics import roc_auc_score

            prob = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"  Best val AUC: {study.best_value:.4f}")

        from xgboost import XGBClassifier

        best_model = XGBClassifier(
            **study.best_params,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return best_model, study.best_params


def train_ridge(X_train, y_train, X_val, y_val, task: str = "regression"):
    """Train Ridge regression/classification with alpha search."""
    from sklearn.linear_model import RidgeCV, RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    alphas = np.logspace(-3, 5, 50)

    if task == "regression":
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_train_s, y_train)
    else:
        model = RidgeClassifierCV(alphas=alphas, cv=5)
        model.fit(X_train_s, y_train)

    return model, scaler


def train_mlp(
    X_train, y_train, X_val, y_val, task: str = "regression", n_trials: int = 50
):
    """Train MLP with Optuna hyperparameter tuning."""
    import optuna
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    logger = logging.getLogger(__name__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"layer_{i}", 64, 512))

        params = {
            "hidden_layer_sizes": tuple(layers),
            "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": 42,
        }

        if task == "regression":
            model = MLPRegressor(**params)
            model.fit(X_train_s, y_train)
            pred = model.predict(X_val_s)
            return mean_absolute_error(y_val, pred)
        else:
            model = MLPClassifier(**params)
            model.fit(X_train_s, y_train)
            prob = model.predict_proba(X_val_s)[:, 1]
            return roc_auc_score(y_val, prob)

    direction = "minimize" if task == "regression" else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    n_layers = best.pop("n_layers")
    layers = tuple(best.pop(f"layer_{i}") for i in range(n_layers))
    lr = best.pop("lr")
    alpha = best.pop("alpha")
    batch_size = best.pop("batch_size")

    Cls = MLPRegressor if task == "regression" else MLPClassifier
    final_model = Cls(
        hidden_layer_sizes=layers,
        learning_rate_init=lr,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    final_model.fit(X_train_s, y_train)

    metric_name = "MAE" if task == "regression" else "AUC"
    logger.info(f"  Best MLP val {metric_name}: {study.best_value:.4f}")

    return final_model, scaler


# ============================================================================
# Feature importance analysis
# ============================================================================


def analyze_feature_importance(model, feat_cols: list[str], top_n: int = 30):
    """Print XGBoost feature importance analysis."""
    logger = logging.getLogger(__name__)

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Top {top_n} Feature Importances")
    logger.info(f"{'=' * 60}")
    for rank, i in enumerate(indices, 1):
        logger.info(f"  {rank:3d}. {feat_cols[i]:<40s}  {importance[i]:.4f}")
    logger.info(f"{'=' * 60}")

    # Group importances by category
    categories = defaultdict(float)
    for i, name in enumerate(feat_cols):
        if name.startswith("diff_") and "_roll" in name:
            categories["Rolling Efficiency (diff)"] += importance[i]
        elif name.startswith("diff_") and "_ewm" in name:
            categories["EWM Momentum (diff)"] += importance[i]
        elif "win_pct" in name or "streak" in name:
            categories["Record/Streak"] += importance[i]
        elif "rest" in name or "b2b" in name or "season_progress" in name:
            categories["Rest/Schedule"] += importance[i]
        elif "h2h" in name:
            categories["Head-to-Head"] += importance[i]
        elif "venue" in name or "home_team_home" in name or "away_team_away" in name:
            categories["Venue Win%"] += importance[i]
        elif name.startswith("home_") or name.startswith("away_"):
            if "vegas" not in name and "llm" not in name:
                categories["Absolute Team Stats"] += importance[i]
        elif (
            "vegas" in name
            or "spread_movement" in name
            or "total_movement" in name
            or "ml_" in name
        ):
            categories["Vegas Betting"] += importance[i]
        elif "llm" in name:
            categories["LLM Predictions"] += importance[i]
        else:
            categories["Other"] += importance[i]

    logger.info(f"\n  Feature Group Importance:")
    for cat, imp in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"    {cat:<30s}  {imp:.4f}")

    return importance


# ============================================================================
# Main pipeline
# ============================================================================


def run_experiment(
    game_df: pd.DataFrame,
    feat_cols: list[str],
    targets: dict,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_trials: int,
    model_type: str = "xgboost",
    label: str = "XGBoost",
) -> dict:
    """Run a single experiment with given model type."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'#' * 70}")
    logger.info(f"  Model: {label}  |  Features: {X_train.shape[1]}")
    logger.info(
        f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}"
    )
    logger.info(f"{'#' * 70}")

    results = {}

    if model_type == "xgboost":
        # Spread model
        logger.info("\n--- Training spread model (XGBoost) ---")
        spread_model, spread_params = train_xgboost(
            X_train,
            targets["train_spread"],
            X_val,
            targets["val_spread"],
            n_trials=n_trials,
            task="regression",
        )
        pred_spread_val = spread_model.predict(X_val)
        pred_spread_test = spread_model.predict(X_test)
        results["spread_model"] = spread_model
        results["spread_params"] = spread_params

        # Win model
        logger.info("\n--- Training win model (XGBoost) ---")
        win_model, win_params = train_xgboost(
            X_train,
            targets["train_win"],
            X_val,
            targets["val_win"],
            n_trials=n_trials,
            task="classification",
        )
        pred_win_val = win_model.predict_proba(X_val)[:, 1]
        pred_win_test = win_model.predict_proba(X_test)[:, 1]
        results["win_model"] = win_model
        results["win_params"] = win_params

    elif model_type == "ridge":
        from sklearn.linear_model import RidgeCV, LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        logger.info("\n--- Training spread model (Ridge) ---")
        alphas = np.logspace(-3, 5, 50)
        spread_model = RidgeCV(alphas=alphas, cv=5)
        spread_model.fit(X_train_s, targets["train_spread"])
        pred_spread_val = spread_model.predict(X_val_s)
        pred_spread_test = spread_model.predict(X_test_s)
        logger.info(f"  Best alpha: {spread_model.alpha_:.4f}")
        results["spread_model"] = spread_model

        logger.info("\n--- Training win model (Logistic Ridge) ---")
        win_model = LogisticRegressionCV(
            Cs=np.logspace(-4, 4, 50),
            cv=5,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        win_model.fit(X_train_s, targets["train_win"])
        pred_win_val = win_model.predict_proba(X_val_s)[:, 1]
        pred_win_test = win_model.predict_proba(X_test_s)[:, 1]
        logger.info(f"  Best C: {win_model.C_[0]:.4f}")
        results["win_model"] = win_model
        results["scaler"] = scaler

    elif model_type == "mlp":
        logger.info("\n--- Training spread model (MLP) ---")
        spread_model, spread_scaler = train_mlp(
            X_train,
            targets["train_spread"],
            X_val,
            targets["val_spread"],
            task="regression",
            n_trials=max(n_trials // 2, 20),
        )
        X_val_s = spread_scaler.transform(X_val)
        X_test_s = spread_scaler.transform(X_test)
        pred_spread_val = spread_model.predict(X_val_s)
        pred_spread_test = spread_model.predict(X_test_s)
        results["spread_model"] = spread_model

        logger.info("\n--- Training win model (MLP) ---")
        win_model, win_scaler = train_mlp(
            X_train,
            targets["train_win"],
            X_val,
            targets["val_win"],
            task="classification",
            n_trials=max(n_trials // 2, 20),
        )
        X_val_s_w = win_scaler.transform(X_val)
        X_test_s_w = win_scaler.transform(X_test)
        pred_win_val = win_model.predict_proba(X_val_s_w)[:, 1]
        pred_win_test = win_model.predict_proba(X_test_s_w)[:, 1]
        results["win_model"] = win_model

    # Evaluate
    val_metrics = evaluate_model(
        pred_spread_val,
        pred_win_val,
        targets["val_spread"],
        targets["val_win"],
        targets["val_home_score"],
        targets["val_away_score"],
        vegas_spread=targets.get("val_vegas_spread"),
        label=f"{label} - Val",
    )
    print_metrics(val_metrics, f"{label} - Validation")

    test_metrics = evaluate_model(
        pred_spread_test,
        pred_win_test,
        targets["test_spread"],
        targets["test_win"],
        targets["test_home_score"],
        targets["test_away_score"],
        vegas_spread=targets.get("test_vegas_spread"),
        label=f"{label} - Test",
    )
    print_metrics(test_metrics, f"{label} - Test")

    results["val_metrics"] = val_metrics
    results["test_metrics"] = test_metrics

    return results


def print_comparison_table(all_results: dict):
    """Print comparison table across all experiments."""
    print("\n")
    print("=" * 110)
    print("PHASE 6 EXP 2 — COMPARISON TABLE")
    print("=" * 110)

    header = (
        f"| {'Experiment':<35s} | {'Spread MAE':>10s} | {'Win AUC':>8s} | "
        f"{'Win Acc':>8s} | {'ATS %':>8s} | {'ATS ROI':>8s} | {'ECE':>8s} |"
    )
    sep = f"|{'-' * 37}|{'-' * 12}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|"

    print(header)
    print(sep)

    for name, result in all_results.items():
        m = result["test_metrics"]
        ats = f"{m['ats_pct']:.3f}" if "ats_pct" in m else "N/A"
        roi = f"{m['ats_roi']:.3f}" if "ats_roi" in m else "N/A"
        print(
            f"| {name:<35s} | {m['spread_mae']:>10.3f} | {m['win_auc']:>8.3f} | "
            f"{m['win_accuracy'] * 100:>7.1f}% | {ats:>8s} | {roi:>8s} | "
            f"{m['ece']:>8.4f} |"
        )

    # Reference baselines
    print(sep)
    print(
        f"| {'Phase 3 Exp 4 (transformer)':<35s} | {'10.83':>10s} | {'0.705':>8s} | "
        f"{'65.1%':>8s} | {'---':>8s} | {'---':>8s} | {'0.0142':>8s} |"
    )
    print(
        f"| {'Phase 3 Exp 9 (ensemble)':<35s} | {'10.66':>10s} | {'0.718':>8s} | "
        f"{'66.5%':>8s} | {'---':>8s} | {'---':>8s} | {'0.0378':>8s} |"
    )
    print(
        f"| {'Phase 3 Exp 8 (hybrid)':<35s} | {'10.85':>10s} | {'---':>8s} | "
        f"{'---':>8s} | {'---':>8s} | {'---':>8s} | {'---':>8s} |"
    )
    print("=" * 110)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Exp 2: Engineered Features Meta-Model"
    )
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Optuna trials per model (default: 100)",
    )
    parser.add_argument(
        "--skip-betting", action="store_true", help="Skip betting features"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM prediction features"
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run all feature subset ablations + all model types",
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/phase6/exp2_engineered"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="xgboost",
        help="Comma-separated model types: xgboost,ridge,mlp",
    )
    parser.add_argument(
        "--save-features", action="store_true", help="Save feature DataFrame to disk"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Phase 6 Experiment 2: Engineered Features Meta-Model")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Train: {TRAIN_SEASONS[0]} to {TRAIN_SEASONS[-1]}")
    logger.info(f"Val: {VAL_SEASONS}")
    logger.info(f"Test: {TEST_SEASONS}")

    # ---------------------------------------------------------------
    # Step 1: Build rolling features from database
    # ---------------------------------------------------------------
    logger.info("\n[Step 1] Building rolling features from TeamBox...")
    game_df = build_game_features(args.db_path)

    # ---------------------------------------------------------------
    # Step 2: Merge betting features
    # ---------------------------------------------------------------
    betting_cols = []
    if not args.skip_betting:
        logger.info("\n[Step 2] Merging betting features...")
        game_df, betting_cols = merge_betting_features(game_df, args.db_path)
    else:
        logger.info("\n[Step 2] Skipping betting features (--skip-betting)")

    # ---------------------------------------------------------------
    # Step 3: Merge LLM prediction features
    # ---------------------------------------------------------------
    llm_cols = []
    if not args.skip_llm:
        logger.info("\n[Step 3] Merging LLM predictions...")
        game_df, llm_cols = merge_llm_features(game_df)
    else:
        logger.info("\n[Step 3] Skipping LLM predictions (--skip-llm)")

    # ---------------------------------------------------------------
    # Step 4: Save features if requested
    # ---------------------------------------------------------------
    if args.save_features:
        out_path = Path(args.output_dir) / "features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        game_df.to_parquet(out_path, index=False)
        logger.info(f"Saved features to {out_path}")

    # ---------------------------------------------------------------
    # Step 5: Run experiments
    # ---------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_types = [m.strip() for m in args.models.split(",")]
    all_results = {}

    if args.run_ablations:
        # Run ablation study across feature subsets and model types
        ablation_configs = []

        # (a) Engineered features only (no betting, no LLM)
        ablation_configs.append(("Eng only", False, False))

        # (b) Engineered + Betting
        if not args.skip_betting and betting_cols:
            ablation_configs.append(("Eng + Betting", True, False))

        # (c) Engineered + LLM
        if not args.skip_llm and llm_cols:
            ablation_configs.append(("Eng + LLM", False, True))

        # (d) Engineered + Betting + LLM (full)
        if not args.skip_betting and betting_cols and not args.skip_llm and llm_cols:
            ablation_configs.append(("Full (Eng+Bet+LLM)", True, True))

        # (e) Betting only
        if not args.skip_betting and betting_cols:
            ablation_configs.append(("Betting only", True, False))

        for feat_label, use_bet, use_llm in ablation_configs:
            for mtype in model_types:
                exp_name = f"{feat_label} / {mtype}"
                logger.info(f"\n\n{'*' * 70}")
                logger.info(f"  ABLATION: {exp_name}")
                logger.info(f"{'*' * 70}")

                if feat_label == "Betting only":
                    # Only use betting columns
                    feat_cols = [c for c in betting_cols if c in game_df.columns]
                else:
                    feat_cols = get_feature_columns(
                        game_df,
                        betting_cols,
                        llm_cols,
                        include_betting=use_bet,
                        include_llm=use_llm,
                    )

                X_train, X_val, X_test, targets = split_data(game_df, feat_cols)
                logger.info(f"  Feature columns: {len(feat_cols)}")

                result = run_experiment(
                    game_df,
                    feat_cols,
                    targets,
                    X_train,
                    X_val,
                    X_test,
                    n_trials=args.n_trials,
                    model_type=mtype,
                    label=exp_name,
                )
                all_results[exp_name] = result

                # Feature importance for XGBoost
                if mtype == "xgboost":
                    analyze_feature_importance(
                        result["spread_model"], feat_cols, top_n=30
                    )

        print_comparison_table(all_results)

    else:
        # Single run with specified models
        feat_cols = get_feature_columns(
            game_df,
            betting_cols,
            llm_cols,
            include_betting=(not args.skip_betting and bool(betting_cols)),
            include_llm=(not args.skip_llm and bool(llm_cols)),
        )
        logger.info(f"\nFeature columns ({len(feat_cols)}):")
        for i, c in enumerate(feat_cols):
            logger.info(f"  {i+1:3d}. {c}")

        X_train, X_val, X_test, targets = split_data(game_df, feat_cols)

        logger.info(f"\nData shapes:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val:   {X_val.shape}")
        logger.info(f"  Test:  {X_test.shape}")

        for mtype in model_types:
            label = f"Phase6 Exp2 ({mtype})"
            result = run_experiment(
                game_df,
                feat_cols,
                targets,
                X_train,
                X_val,
                X_test,
                n_trials=args.n_trials,
                model_type=mtype,
                label=label,
            )
            all_results[label] = result

            # Feature importance for XGBoost spread model
            if mtype == "xgboost":
                logger.info("\n--- Feature Importance (Spread Model) ---")
                importance = analyze_feature_importance(
                    result["spread_model"], feat_cols, top_n=30
                )

                logger.info("\n--- Feature Importance (Win Model) ---")
                analyze_feature_importance(result["win_model"], feat_cols, top_n=30)

        if len(all_results) > 1:
            print_comparison_table(all_results)

    # ---------------------------------------------------------------
    # Step 6: Save results
    # ---------------------------------------------------------------
    save_results = {}
    for name, result in all_results.items():
        entry = {
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
        }
        if "spread_params" in result:
            entry["spread_params"] = result["spread_params"]
        if "win_params" in result:
            entry["win_params"] = result["win_params"]
        save_results[name] = entry

    save_results["_meta"] = {
        "n_trials": args.n_trials,
        "train_seasons": TRAIN_SEASONS,
        "val_seasons": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
        "skip_betting": args.skip_betting,
        "skip_llm": args.skip_llm,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # Save XGBoost models
    for name, result in all_results.items():
        if "spread_model" in result and hasattr(result["spread_model"], "save_model"):
            safe_name = (
                name.replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
            )
            result["spread_model"].save_model(
                str(output_dir / f"xgb_spread_{safe_name}.json")
            )
            result["win_model"].save_model(
                str(output_dir / f"xgb_win_{safe_name}.json")
            )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 6 EXPERIMENT 2 COMPLETE")
    print("=" * 70)
    for name, result in all_results.items():
        m = result["test_metrics"]
        ats_str = f"ATS {m['ats_pct']:.3f}" if "ats_pct" in m else "ATS N/A"
        print(f"  {name}:")
        print(
            f"    Spread MAE: {m['spread_mae']:.3f}  |  Win AUC: {m['win_auc']:.3f}  |  "
            f"Win Acc: {m['win_accuracy']:.3f}  |  {ats_str}"
        )
    print("=" * 70)
    print(f"Output: {output_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
