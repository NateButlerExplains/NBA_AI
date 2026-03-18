#!/usr/bin/env python
"""
Phase 6 Experiment 5: ATS (Against the Spread) Binary Classifier.

Trains a model that directly predicts whether the home team will cover the
Vegas spread. Instead of predicting the spread and comparing post-hoc, the
model receives the Vegas line as an INPUT and learns to identify systematic
biases — situations where Vegas is wrong.

Target: home_covered = (actual_margin + vegas_spread) > 0
  - Excludes pushes (actual_margin + vegas_spread == 0)

Key insight: the Vegas spread IS a feature. The model tries to find patterns
like "when team X is a 7-point home fav after a back-to-back, Vegas
overestimates them."

Models:
  1. Logistic Regression (interpretable baseline)
  2. XGBoost (primary, Optuna-tuned)
  3. LightGBM (comparison)
  4. Calibrated ensemble (average of best models)

Success criteria: ATS > 52.4% (breakeven at -110 juice).

Usage:
    # Full run with Optuna tuning
    python scripts/phase6_exp5_ats_classifier.py --optuna-trials 100

    # Quick run with defaults (no Optuna)
    python scripts/phase6_exp5_ats_classifier.py

    # Evaluate saved model only
    python scripts/phase6_exp5_ats_classifier.py --evaluate-only \\
        --model checkpoints/phase6/ats_xgboost.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase6"
EXP7_DIR = PROJECT_ROOT / "data" / "exp7"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIN_PAYOUT = 100 / 1.10  # ~$90.91 profit on a $100 bet at -110
LOSS_COST = 100.0

# Historical team abbreviation mapping (Games table -> current franchise)
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "VAN": "MEM",
    "CHH": "CHA",
}

# Abbreviation -> franchise team_id (for TeamBox joins)
ABBREV_TO_TEAM_ID = {
    "ATL": "1610612737",
    "BOS": "1610612738",
    "CLE": "1610612739",
    "NOP": "1610612740",
    "CHI": "1610612741",
    "DAL": "1610612742",
    "DEN": "1610612743",
    "GSW": "1610612744",
    "HOU": "1610612745",
    "LAC": "1610612746",
    "LAL": "1610612747",
    "MIA": "1610612748",
    "MIL": "1610612749",
    "MIN": "1610612750",
    "BKN": "1610612751",
    "NYK": "1610612752",
    "ORL": "1610612753",
    "IND": "1610612754",
    "PHI": "1610612755",
    "PHX": "1610612756",
    "POR": "1610612757",
    "SAC": "1610612758",
    "SAS": "1610612759",
    "OKC": "1610612760",
    "TOR": "1610612761",
    "UTA": "1610612762",
    "MEM": "1610612763",
    "WAS": "1610612764",
    "DET": "1610612765",
    "CHA": "1610612766",
}

# Rolling windows for team stats
ROLLING_WINDOWS = [5, 10, 20]

# Season splits for train/val/test
TRAIN_SEASONS_END = "2022-2023"  # inclusive
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_abbrev(abbrev: str) -> str:
    """Map historical abbreviations to current franchise abbreviation."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


def get_team_id(abbrev: str) -> str:
    """Convert abbreviation (current or historical) to franchise team_id."""
    current = normalize_abbrev(abbrev)
    return ABBREV_TO_TEAM_ID.get(current, "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_game_data(db_path: Path | None = None) -> pd.DataFrame:
    """
    Load all completed regular-season games with Vegas spread data and
    TeamBox stats for both teams.

    Returns a DataFrame with one row per game, columns prefixed with
    home_ / away_ for team stats.
    """
    conn = get_connection(db_path)

    logger.info("Loading games with Vegas spreads...")

    # Step 1: Get games with betting data
    query = """
        SELECT
            g.game_id,
            g.date_time_utc,
            g.home_team,
            g.away_team,
            g.season,
            COALESCE(b.espn_closing_spread, b.covers_closing_spread,
                     b.espn_current_spread) AS vegas_spread,
            COALESCE(b.espn_closing_total, b.covers_closing_total,
                     b.espn_current_total) AS vegas_total,
            b.espn_closing_ml_home,
            b.espn_closing_ml_away,
            b.espn_current_ml_home,
            b.espn_current_ml_away
        FROM Games g
        JOIN Betting b ON g.game_id = b.game_id
        WHERE g.status = 3
          AND g.season_type = 'Regular Season'
          AND COALESCE(b.espn_closing_spread, b.covers_closing_spread,
                       b.espn_current_spread) IS NOT NULL
        ORDER BY g.date_time_utc
    """
    games_df = pd.read_sql_query(query, conn)
    logger.info(f"  {len(games_df)} games with Vegas spreads")

    # Step 2: Load all TeamBox data
    teambox_query = """
        SELECT
            tb.team_id, tb.game_id,
            tb.pts, tb.pts_allowed,
            tb.reb, tb.ast, tb.stl, tb.blk, tb.tov, tb.pf,
            tb.fga, tb.fgm, tb.fg_pct, tb.fg3a, tb.fg3m, tb.fg3_pct,
            tb.fta, tb.ftm, tb.ft_pct, tb.plus_minus
        FROM TeamBox tb
    """
    teambox_df = pd.read_sql_query(teambox_query, conn)
    conn.close()

    # Index TeamBox by (team_id, game_id) for fast lookup
    teambox_df = teambox_df.set_index(["team_id", "game_id"])

    # Step 3: Map abbreviations to team_ids and join
    games_df["home_abbrev_norm"] = games_df["home_team"].apply(normalize_abbrev)
    games_df["away_abbrev_norm"] = games_df["away_team"].apply(normalize_abbrev)
    games_df["home_team_id"] = games_df["home_team"].apply(get_team_id)
    games_df["away_team_id"] = games_df["away_team"].apply(get_team_id)

    # Parse date
    games_df["game_date"] = pd.to_datetime(games_df["date_time_utc"]).dt.tz_localize(
        None
    )

    # Join home TeamBox
    home_stats = []
    away_stats = []
    valid_mask = []

    stat_cols = [
        "pts",
        "pts_allowed",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "fga",
        "fgm",
        "fg_pct",
        "fg3a",
        "fg3m",
        "fg3_pct",
        "fta",
        "ftm",
        "ft_pct",
        "plus_minus",
    ]

    for _, row in games_df.iterrows():
        h_key = (row["home_team_id"], row["game_id"])
        a_key = (row["away_team_id"], row["game_id"])

        if h_key in teambox_df.index and a_key in teambox_df.index:
            h = teambox_df.loc[h_key]
            a = teambox_df.loc[a_key]
            home_stats.append({f"home_{c}": h[c] for c in stat_cols})
            away_stats.append({f"away_{c}": a[c] for c in stat_cols})
            valid_mask.append(True)
        else:
            home_stats.append({f"home_{c}": np.nan for c in stat_cols})
            away_stats.append({f"away_{c}": np.nan for c in stat_cols})
            valid_mask.append(False)

    home_df = pd.DataFrame(home_stats, index=games_df.index)
    away_df = pd.DataFrame(away_stats, index=games_df.index)

    games_df = pd.concat([games_df, home_df, away_df], axis=1)
    n_before = len(games_df)
    games_df = games_df[valid_mask].reset_index(drop=True)
    n_dropped = n_before - len(games_df)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped} games missing TeamBox data")

    logger.info(f"  Final dataset: {len(games_df)} games")
    return games_df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def american_to_implied_prob(odds) -> float:
    """Convert American moneyline odds to implied probability."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return np.nan
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)


def compute_rolling_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling team stats (PPG, Opp PPG, FG%, 3P%, net rating)
    for each game, using only data available BEFORE that game.

    Also computes rest days, back-to-back flags, win%, home/away win%.
    """
    logger.info("Computing rolling features...")

    # Sort by date (should already be, but ensure)
    df = games_df.sort_values("game_date").reset_index(drop=True)

    # We need per-team timeseries. Build a list of all (team_id, game)
    # entries in chronological order.
    team_games = {}  # team_id -> list of (idx, game_date, is_home, stats_dict)

    for idx, row in df.iterrows():
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        h_stats = {
            "pts": row["home_pts"],
            "pts_allowed": row["away_pts"],
            "fg_pct": row["home_fg_pct"],
            "fg3_pct": row["home_fg3_pct"],
            "reb": row["home_reb"],
            "ast": row["home_ast"],
            "tov": row["home_tov"],
            "stl": row["home_stl"],
            "blk": row["home_blk"],
            "fga": row["home_fga"],
            "fgm": row["home_fgm"],
            "fg3a": row["home_fg3a"],
            "fg3m": row["home_fg3m"],
            "fta": row["home_fta"],
            "ftm": row["home_ftm"],
            "won": 1 if row["home_pts"] > row["away_pts"] else 0,
        }
        a_stats = {
            "pts": row["away_pts"],
            "pts_allowed": row["home_pts"],
            "fg_pct": row["away_fg_pct"],
            "fg3_pct": row["away_fg3_pct"],
            "reb": row["away_reb"],
            "ast": row["away_ast"],
            "tov": row["away_tov"],
            "stl": row["away_stl"],
            "blk": row["away_blk"],
            "fga": row["away_fga"],
            "fgm": row["away_fgm"],
            "fg3a": row["away_fg3a"],
            "fg3m": row["away_fg3m"],
            "fta": row["away_fta"],
            "ftm": row["away_ftm"],
            "won": 1 if row["away_pts"] > row["home_pts"] else 0,
        }

        team_games.setdefault(h_id, []).append((idx, row["game_date"], True, h_stats))
        team_games.setdefault(a_id, []).append((idx, row["game_date"], False, a_stats))

    # Pre-allocate feature columns
    n = len(df)
    feature_names = []
    feature_data = {}

    for prefix in ["home", "away"]:
        for w in ROLLING_WINDOWS:
            for stat in [
                "ppg",
                "opp_ppg",
                "fg_pct",
                "fg3_pct",
                "net_rating",
                "reb",
                "ast",
                "tov",
            ]:
                col = f"{prefix}_{stat}_L{w}"
                feature_names.append(col)
                feature_data[col] = np.full(n, np.nan)

        # Overall season stats
        for stat in ["win_pct", "home_win_pct", "away_win_pct", "games_played"]:
            col = f"{prefix}_{stat}"
            feature_names.append(col)
            feature_data[col] = np.full(n, np.nan)

        # Rest days
        for col_name in [f"{prefix}_rest_days", f"{prefix}_is_b2b"]:
            feature_data[col_name] = np.full(n, np.nan)
            feature_names.append(col_name)

    # Compute per-team rolling stats
    for team_id, entries in team_games.items():
        # entries are already sorted chronologically (df is sorted by date)
        pts_hist = []
        opp_pts_hist = []
        fg_pct_hist = []
        fg3_pct_hist = []
        reb_hist = []
        ast_hist = []
        tov_hist = []
        win_hist = []
        home_win_hist = []  # (is_home, won)
        dates = []

        for i, (idx, game_date, is_home, stats) in enumerate(entries):
            prefix = "home" if is_home else "away"

            # Rolling stats (using ONLY previous games)
            for w in ROLLING_WINDOWS:
                if len(pts_hist) >= w:
                    recent_pts = pts_hist[-w:]
                    recent_opp = opp_pts_hist[-w:]
                    recent_fg = fg_pct_hist[-w:]
                    recent_fg3 = fg3_pct_hist[-w:]
                    recent_reb = reb_hist[-w:]
                    recent_ast = ast_hist[-w:]
                    recent_tov = tov_hist[-w:]

                    ppg = np.mean(recent_pts)
                    opp_ppg = np.mean(recent_opp)

                    feature_data[f"{prefix}_ppg_L{w}"][idx] = ppg
                    feature_data[f"{prefix}_opp_ppg_L{w}"][idx] = opp_ppg
                    feature_data[f"{prefix}_fg_pct_L{w}"][idx] = np.mean(recent_fg)
                    feature_data[f"{prefix}_fg3_pct_L{w}"][idx] = np.mean(recent_fg3)
                    feature_data[f"{prefix}_net_rating_L{w}"][idx] = ppg - opp_ppg
                    feature_data[f"{prefix}_reb_L{w}"][idx] = np.mean(recent_reb)
                    feature_data[f"{prefix}_ast_L{w}"][idx] = np.mean(recent_ast)
                    feature_data[f"{prefix}_tov_L{w}"][idx] = np.mean(recent_tov)

            # Season win%
            if len(win_hist) >= 5:
                feature_data[f"{prefix}_win_pct"][idx] = np.mean(win_hist)
                feature_data[f"{prefix}_games_played"][idx] = len(win_hist)

                # Home/away splits
                home_wins = [w_ for (h_, w_) in home_win_hist if h_]
                away_wins = [w_ for (h_, w_) in home_win_hist if not h_]
                if len(home_wins) >= 3:
                    feature_data[f"{prefix}_home_win_pct"][idx] = np.mean(home_wins)
                if len(away_wins) >= 3:
                    feature_data[f"{prefix}_away_win_pct"][idx] = np.mean(away_wins)

            # Rest days
            if len(dates) > 0:
                rest = (game_date - dates[-1]).days
                feature_data[f"{prefix}_rest_days"][idx] = min(rest, 10)
                feature_data[f"{prefix}_is_b2b"][idx] = 1.0 if rest <= 1 else 0.0

            # Append current game data for future rolling calcs
            pts_hist.append(stats["pts"])
            opp_pts_hist.append(stats["pts_allowed"])
            fg_pct_hist.append(stats["fg_pct"] if stats["fg_pct"] else 0.0)
            fg3_pct_hist.append(stats["fg3_pct"] if stats["fg3_pct"] else 0.0)
            reb_hist.append(stats["reb"] if stats["reb"] else 0.0)
            ast_hist.append(stats["ast"] if stats["ast"] else 0.0)
            tov_hist.append(stats["tov"] if stats["tov"] else 0.0)
            win_hist.append(stats["won"])
            home_win_hist.append((is_home, stats["won"]))
            dates.append(game_date)

    # Assign features to DataFrame
    for col in feature_names:
        df[col] = feature_data[col]

    logger.info(f"  Computed {len(feature_names)} rolling features")
    return df


def load_llm_predictions() -> dict[str, dict]:
    """
    Load LLM predictions from exp7 JSONL files.
    Returns dict: game_id -> {model_spread, model_win_prob} (best model).
    """
    llm_preds = {}

    # Try loading in priority order: gpt-5.4 > gpt-5.4-mini > gpt-5.4-nano
    for model_file in [
        "test_gpt-5.4_results.jsonl",
        "test_gpt-5.4-mini_results.jsonl",
        "test_gpt-5.4-nano_results.jsonl",
    ]:
        path = EXP7_DIR / model_file
        if not path.exists():
            continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not obj.get("success", True):
                    continue
                pred = obj.get("prediction", {})
                if pred.get("home_score") is None or pred.get("away_score") is None:
                    continue

                gid = obj["game_id"]
                if gid in llm_preds:
                    continue  # Already have from a better model

                llm_preds[gid] = {
                    "llm_spread": pred["home_score"] - pred["away_score"],
                    "llm_win_prob": pred.get("home_win_probability", 0.5),
                }

    logger.info(f"  Loaded LLM predictions for {len(llm_preds)} games")
    return llm_preds


def build_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from games with rolling stats.

    Features:
      1. Vegas spread (normalized)
      2. Vegas total (normalized)
      3. Moneyline implied probabilities
      4. Rolling team stats (PPG, Opp PPG, FG%, 3P%, net rating x 3 windows)
      5. Rest days, back-to-back
      6. Win%, home/away win%
      7. Spread interaction features
      8. Differentials (home - away)
      9. LLM predictions (if available)
    """
    logger.info("Building feature matrix...")
    df = games_df.copy()

    # --- Target variable ---
    df["actual_margin"] = df["home_pts"] - df["away_pts"]
    df["ats_margin"] = df["actual_margin"] + df["vegas_spread"]
    # Exclude pushes
    df["is_push"] = df["ats_margin"] == 0.0
    df["home_covered"] = (df["ats_margin"] > 0).astype(int)

    # --- Vegas features ---
    features = []

    # Spread features
    df["f_vegas_spread"] = df["vegas_spread"]
    df["f_abs_spread"] = df["vegas_spread"].abs()
    df["f_spread_squared"] = df["vegas_spread"] ** 2
    features.extend(["f_vegas_spread", "f_abs_spread", "f_spread_squared"])

    # Total
    df["f_vegas_total"] = df["vegas_total"]
    has_total = df["f_vegas_total"].notna()
    if has_total.sum() > 0:
        features.append("f_vegas_total")

    # Moneyline implied probabilities
    df["ml_home"] = df["espn_closing_ml_home"].combine_first(df["espn_current_ml_home"])
    df["ml_away"] = df["espn_closing_ml_away"].combine_first(df["espn_current_ml_away"])
    df["f_ml_home_prob"] = df["ml_home"].apply(american_to_implied_prob)
    df["f_ml_away_prob"] = df["ml_away"].apply(american_to_implied_prob)
    # Vig-adjusted
    df["f_ml_vig"] = df["f_ml_home_prob"] + df["f_ml_away_prob"]
    df["f_ml_home_fair"] = df["f_ml_home_prob"] / df["f_ml_vig"]
    for col in ["f_ml_home_prob", "f_ml_away_prob", "f_ml_vig", "f_ml_home_fair"]:
        if df[col].notna().sum() > len(df) * 0.3:
            features.append(col)

    # --- Rolling team stats ---
    rolling_stats = [
        "ppg",
        "opp_ppg",
        "fg_pct",
        "fg3_pct",
        "net_rating",
        "reb",
        "ast",
        "tov",
    ]

    for w in ROLLING_WINDOWS:
        for stat in rolling_stats:
            h_col = f"home_{stat}_L{w}"
            a_col = f"away_{stat}_L{w}"
            d_col = f"f_diff_{stat}_L{w}"

            if h_col in df.columns and a_col in df.columns:
                df[f"f_{h_col}"] = df[h_col]
                df[f"f_{a_col}"] = df[a_col]
                df[d_col] = df[h_col] - df[a_col]
                features.extend([f"f_{h_col}", f"f_{a_col}", d_col])

    # --- Win % features ---
    for stat in ["win_pct", "home_win_pct", "away_win_pct", "games_played"]:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        if h_col in df.columns and a_col in df.columns:
            df[f"f_{h_col}"] = df[h_col]
            df[f"f_{a_col}"] = df[a_col]
            if stat != "games_played":
                df[f"f_diff_{stat}"] = df[h_col] - df[a_col]
                features.extend([f"f_{h_col}", f"f_{a_col}", f"f_diff_{stat}"])
            else:
                features.extend([f"f_{h_col}", f"f_{a_col}"])

    # --- Rest / B2B ---
    for prefix in ["home", "away"]:
        for col in ["rest_days", "is_b2b"]:
            src = f"{prefix}_{col}"
            dst = f"f_{prefix}_{col}"
            if src in df.columns:
                df[dst] = df[src]
                features.append(dst)

    df["f_rest_diff"] = df.get("home_rest_days", 0) - df.get("away_rest_days", 0)
    if "home_rest_days" in df.columns:
        features.append("f_rest_diff")

    # Both teams B2B
    if "home_is_b2b" in df.columns and "away_is_b2b" in df.columns:
        df["f_both_b2b"] = ((df["home_is_b2b"] == 1) & (df["away_is_b2b"] == 1)).astype(
            float
        )
        df["f_home_only_b2b"] = (
            (df["home_is_b2b"] == 1) & (df["away_is_b2b"] != 1)
        ).astype(float)
        df["f_away_only_b2b"] = (
            (df["home_is_b2b"] != 1) & (df["away_is_b2b"] == 1)
        ).astype(float)
        features.extend(["f_both_b2b", "f_home_only_b2b", "f_away_only_b2b"])

    # --- Spread interaction features ---
    if "home_rest_days" in df.columns:
        df["f_spread_x_rest_diff"] = df["vegas_spread"] * df["f_rest_diff"]
        features.append("f_spread_x_rest_diff")

    if "home_win_pct" in df.columns:
        df["f_spread_x_home_wpct"] = df["vegas_spread"] * df["home_win_pct"].fillna(0.5)
        features.append("f_spread_x_home_wpct")

    for w in [10]:
        nrc = f"f_diff_net_rating_L{w}"
        if nrc in df.columns:
            df["f_spread_x_net_diff"] = df["vegas_spread"] * df[nrc]
            features.append("f_spread_x_net_diff")

    # Spread bucket (categorical via one-hot isn't needed for tree models,
    # but include a binned version for linear)
    df["f_spread_bucket"] = pd.cut(
        df["vegas_spread"],
        bins=[-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf],
        labels=False,
    ).astype(float)
    features.append("f_spread_bucket")

    # Total x spread interaction
    if "f_vegas_total" in features:
        df["f_total_x_spread"] = df["vegas_total"] * df["vegas_spread"]
        df["f_total_x_abs_spread"] = df["vegas_total"] * df["vegas_spread"].abs()
        features.extend(["f_total_x_spread", "f_total_x_abs_spread"])

    # --- LLM predictions ---
    llm_preds = load_llm_predictions()
    if llm_preds:
        df["f_llm_spread"] = df["game_id"].map(
            lambda gid: llm_preds.get(gid, {}).get("llm_spread", np.nan)
        )
        df["f_llm_win_prob"] = df["game_id"].map(
            lambda gid: llm_preds.get(gid, {}).get("llm_win_prob", np.nan)
        )
        df["f_llm_vs_vegas"] = df["f_llm_spread"] - (-df["vegas_spread"])

        for col in ["f_llm_spread", "f_llm_win_prob", "f_llm_vs_vegas"]:
            if df[col].notna().sum() > 100:
                features.append(col)

    # --- Season progress ---
    # Approximate: games_played / 82
    if "f_home_games_played" in features:
        df["f_season_progress"] = df["home_games_played"].clip(upper=82) / 82.0
        features.append("f_season_progress")

    # Deduplicate features
    features = list(dict.fromkeys(features))

    logger.info(f"  Total features: {len(features)}")
    for f in features:
        pct_valid = df[f].notna().mean() * 100
        if pct_valid < 80:
            logger.debug(f"    {f}: {pct_valid:.0f}% non-null")

    return df, features


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, features: list[str]):
    """
    Split into train/val/test by season.
    Drops pushes and rows with insufficient feature coverage.
    """
    # Drop pushes
    n_before = len(df)
    df = df[~df["is_push"]].copy()
    n_pushes = n_before - len(df)
    logger.info(f"  Excluded {n_pushes} pushes ({n_pushes/n_before*100:.1f}%)")

    # Parse season start year for splitting
    df["season_start_year"] = df["season"].str[:4].astype(int)
    train_end_year = int(TRAIN_SEASONS_END[:4])

    train_mask = df["season_start_year"] <= train_end_year
    val_mask = df["season"].isin(VAL_SEASONS)
    test_mask = df["season"].isin(TEST_SEASONS)

    # Require minimum feature coverage (at least 70% non-null)
    feature_coverage = df[features].notna().mean(axis=1)
    coverage_mask = feature_coverage >= 0.7
    logger.info(
        f"  Feature coverage >= 70%: {coverage_mask.sum()}/{len(df)} games "
        f"({coverage_mask.mean()*100:.1f}%)"
    )

    train = df[train_mask & coverage_mask].copy()
    val = df[val_mask & coverage_mask].copy()
    test = df[test_mask & coverage_mask].copy()

    logger.info(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    logger.info(f"  Train seasons: {sorted(train['season'].unique())}")
    logger.info(f"  Val seasons:   {sorted(val['season'].unique())}")
    logger.info(f"  Test seasons:  {sorted(test['season'].unique())}")

    # Target distribution
    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        if len(subset) > 0:
            cover_rate = subset["home_covered"].mean()
            logger.info(f"  {name} home cover rate: {cover_rate:.3f}")

    return train, val, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple:
    """Train logistic regression baseline."""
    logger.info("Training Logistic Regression...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    X_val_s = scaler.transform(np.nan_to_num(X_val, nan=0.0))

    model = LogisticRegression(
        C=0.1,
        max_iter=1000,
        penalty="l2",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    train_proba = model.predict_proba(X_train_s)[:, 1]
    val_proba = model.predict_proba(X_val_s)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

    logger.info(f"  Train AUC: {train_auc:.4f}")
    logger.info(f"  Val AUC:   {val_auc:.4f}")
    logger.info(f"  Val Acc:   {val_acc:.4f}")

    # Feature importances
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top_positive = coefs.nlargest(5)
    top_negative = coefs.nsmallest(5)
    logger.info("  Top positive coefficients:")
    for feat, coef in top_positive.items():
        logger.info(f"    {feat}: {coef:+.4f}")
    logger.info("  Top negative coefficients:")
    for feat, coef in top_negative.items():
        logger.info(f"    {feat}: {coef:+.4f}")

    return model, scaler, val_proba


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_optuna_trials: int = 0,
) -> tuple:
    """Train XGBoost classifier, optionally with Optuna tuning."""
    import xgboost as xgb

    logger.info("Training XGBoost...")

    if n_optuna_trials > 0:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": 42,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 1.2),
                "early_stopping_rounds": 50,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_proba)

        logger.info(f"  Running Optuna with {n_optuna_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=True)

        logger.info(f"  Best trial AUC: {study.best_value:.4f}")
        logger.info(f"  Best params: {study.best_params}")
        best_params = study.best_params
    else:
        # Sensible defaults
        best_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "gamma": 0.5,
            "scale_pos_weight": 1.0,
        }

    # Train final model with best params
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "early_stopping_rounds": 50,
        **best_params,
    }

    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

    logger.info(f"  Train AUC: {train_auc:.4f}")
    logger.info(f"  Val AUC:   {val_auc:.4f}")
    logger.info(f"  Val Acc:   {val_acc:.4f}")

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    logger.info("  Top 15 features by importance:")
    for feat, imp in importances.head(15).items():
        logger.info(f"    {feat}: {imp:.4f}")

    return model, best_params, val_proba


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple:
    """Train LightGBM classifier."""
    import lightgbm as lgb

    logger.info("Training LightGBM...")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

    logger.info(f"  Train AUC: {train_auc:.4f}")
    logger.info(f"  Val AUC:   {val_auc:.4f}")
    logger.info(f"  Val Acc:   {val_acc:.4f}")

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    logger.info("  Top 15 features by importance:")
    for feat, imp in importances.head(15).items():
        logger.info(f"    {feat}: {imp:.4f}")

    return model, val_proba


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def kelly_fraction(p: float, decimal_odds: float = 1.909) -> float:
    """
    Kelly criterion for bet sizing.
    -110 odds = decimal 1.909.
    f* = (decimal_odds * p - 1) / (decimal_odds - 1)
    """
    f = (decimal_odds * p - 1) / (decimal_odds - 1)
    return max(0.0, f)


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    df_subset: pd.DataFrame,
    model_name: str,
    detailed: bool = True,
) -> dict:
    """
    Comprehensive ATS evaluation.

    Returns dict of metrics.
    """
    results = {"model": model_name}

    # --- Basic classification metrics ---
    y_pred = (y_proba > 0.5).astype(int)
    results["auc"] = roc_auc_score(y_true, y_proba)
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["brier"] = brier_score_loss(y_true, y_proba)
    results["log_loss"] = log_loss(y_true, y_proba)

    # --- ATS metrics ---
    n_total = len(y_true)
    wins = int((y_pred == y_true).sum())
    losses = n_total - wins
    ats_pct = wins / n_total if n_total > 0 else 0.0
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = (profit / (n_total * LOSS_COST) * 100) if n_total > 0 else 0.0

    results["n_games"] = n_total
    results["ats_wins"] = wins
    results["ats_losses"] = losses
    results["ats_pct"] = ats_pct
    results["profit"] = profit
    results["roi_pct"] = roi

    # --- Kelly criterion betting ---
    kelly_profit = 0.0
    kelly_bets = 0
    kelly_wagered = 0.0
    for p, actual in zip(y_proba, y_true):
        # Pick the side with higher probability
        if p > 0.5:
            # Betting on home to cover
            kf = kelly_fraction(p)
            if kf > 0:
                bet_size = kf * 1000  # $1000 bankroll, fractional Kelly
                kelly_wagered += bet_size
                kelly_bets += 1
                if actual == 1:
                    kelly_profit += bet_size * (1.909 - 1)
                else:
                    kelly_profit -= bet_size
        elif p < 0.5:
            # Betting on away to cover
            kf = kelly_fraction(1 - p)
            if kf > 0:
                bet_size = kf * 1000
                kelly_wagered += bet_size
                kelly_bets += 1
                if actual == 0:
                    kelly_profit += bet_size * (1.909 - 1)
                else:
                    kelly_profit -= bet_size

    results["kelly_bets"] = kelly_bets
    results["kelly_profit"] = kelly_profit
    results["kelly_wagered"] = kelly_wagered
    results["kelly_roi"] = (
        (kelly_profit / kelly_wagered * 100) if kelly_wagered > 0 else 0.0
    )

    # --- Confidence tiers ---
    # Use |p - 0.5| as confidence measure
    confidence = np.abs(y_proba - 0.5)
    tier_results = {}

    # Probability thresholds: model says > X% for the side it picks
    for tier_name, threshold in [
        ("All", 0.50),
        (">52%", 0.52),
        (">55%", 0.55),
        (">57%", 0.57),
        (">60%", 0.60),
        (">65%", 0.65),
    ]:
        mask = np.maximum(y_proba, 1 - y_proba) > threshold
        if mask.sum() == 0:
            tier_results[tier_name] = {
                "n": 0,
                "ats_pct": 0.0,
                "profit": 0.0,
                "roi": 0.0,
            }
            continue

        tier_y = y_true[mask]
        tier_pred = y_pred[mask]
        tier_n = int(mask.sum())
        tier_wins = int((tier_pred == tier_y).sum())
        tier_losses = tier_n - tier_wins
        tier_ats = tier_wins / tier_n
        tier_profit = tier_wins * WIN_PAYOUT - tier_losses * LOSS_COST
        tier_roi = tier_profit / (tier_n * LOSS_COST) * 100

        tier_results[tier_name] = {
            "n": tier_n,
            "wins": tier_wins,
            "losses": tier_losses,
            "ats_pct": tier_ats,
            "profit": tier_profit,
            "roi": tier_roi,
        }

    results["tiers"] = tier_results

    # --- Season breakdown ---
    if "season" in df_subset.columns:
        season_results = {}
        for season in sorted(df_subset["season"].unique()):
            s_mask = df_subset["season"].values == season
            if s_mask.sum() == 0:
                continue
            s_y = y_true[s_mask]
            s_pred = y_pred[s_mask]
            s_n = int(s_mask.sum())
            s_wins = int((s_pred == s_y).sum())
            s_losses = s_n - s_wins
            s_ats = s_wins / s_n
            s_profit = s_wins * WIN_PAYOUT - s_losses * LOSS_COST
            s_roi = s_profit / (s_n * LOSS_COST) * 100

            season_results[season] = {
                "n": s_n,
                "record": f"{s_wins}-{s_losses}",
                "ats_pct": s_ats,
                "profit": s_profit,
                "roi": s_roi,
            }
        results["seasons"] = season_results

    # --- Calibration ---
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=10, strategy="uniform"
        )
        ece = np.mean(np.abs(prob_true - prob_pred))
        results["ece"] = ece
        results["calibration"] = list(zip(prob_pred.tolist(), prob_true.tolist()))
    except Exception:
        results["ece"] = np.nan

    return results


def print_evaluation(results: dict, detailed: bool = True):
    """Pretty-print evaluation results."""
    name = results["model"]
    w = 80

    print(f"\n{'=' * w}")
    print(f"  {name}")
    print(f"{'=' * w}")

    # ATS Summary
    print(f"\n  ATS Performance")
    print(f"  {'-' * 55}")
    print(f"  Record:          {results['ats_wins']}-{results['ats_losses']}")
    print(f"  ATS Win Rate:    {results['ats_pct']:.1%}")
    print(f"  Games:           {results['n_games']}")
    breakeven = 1 / 1.909 * 100
    gap = results["ats_pct"] * 100 - breakeven
    print(f"  vs Breakeven:    {gap:+.2f} pp  (breakeven = {breakeven:.1f}%)")
    print()
    print(f"  Flat Betting ($100 per game at -110)")
    print(f"  {'-' * 55}")
    print(f"  Total Wagered:   ${results['n_games'] * 100:,.0f}")
    print(f"  Net P/L:         ${results['profit']:+,.2f}")
    print(f"  ROI:             {results['roi_pct']:+.2f}%")

    # Classification metrics
    print(f"\n  Classification Metrics")
    print(f"  {'-' * 55}")
    print(f"  ROC-AUC:         {results['auc']:.4f}")
    print(f"  Accuracy:        {results['accuracy']:.4f}")
    print(f"  Brier Score:     {results['brier']:.4f}")
    print(f"  Log Loss:        {results['log_loss']:.4f}")
    if "ece" in results and not np.isnan(results.get("ece", np.nan)):
        print(f"  ECE:             {results['ece']:.4f}")

    # Kelly
    print(f"\n  Kelly Criterion ($1000 bankroll)")
    print(f"  {'-' * 55}")
    print(f"  Bets Placed:     {results['kelly_bets']}")
    print(f"  Total Wagered:   ${results['kelly_wagered']:,.2f}")
    print(f"  Net P/L:         ${results['kelly_profit']:+,.2f}")
    print(f"  ROI:             {results['kelly_roi']:+.2f}%")

    # Confidence tiers
    if "tiers" in results and detailed:
        print(f"\n  Confidence Tiers")
        print(f"  {'-' * 65}")
        print(
            f"  {'Threshold':<12s} {'Record':>10s} {'ATS%':>8s} "
            f"{'Profit':>10s} {'ROI':>8s} {'Games':>7s}"
        )
        print(f"  {'-' * 65}")
        for tier_name, tier in results["tiers"].items():
            if tier["n"] == 0:
                print(
                    f"  {tier_name:<12s} {'---':>10s} {'---':>8s} "
                    f"{'---':>10s} {'---':>8s} {0:>7d}"
                )
            else:
                rec = f"{tier.get('wins', 0)}-{tier.get('losses', 0)}"
                print(
                    f"  {tier_name:<12s} {rec:>10s} "
                    f"{tier['ats_pct']:>7.1%} "
                    f"${tier['profit']:>+9.0f} "
                    f"{tier['roi']:>+7.1f}% "
                    f"{tier['n']:>7d}"
                )
        print(f"  {'-' * 65}")

    # Season breakdown
    if "seasons" in results and detailed:
        print(f"\n  Season Breakdown")
        print(f"  {'-' * 60}")
        print(
            f"  {'Season':<15s} {'Record':>10s} {'ATS%':>8s} "
            f"{'Profit':>10s} {'ROI':>8s}"
        )
        print(f"  {'-' * 60}")
        for season, data in results["seasons"].items():
            print(
                f"  {season:<15s} {data['record']:>10s} "
                f"{data['ats_pct']:>7.1%} "
                f"${data['profit']:>+9.0f} "
                f"{data['roi']:>+7.1f}%"
            )
        print(f"  {'-' * 60}")

    # Calibration
    if "calibration" in results and detailed:
        print(f"\n  Calibration (predicted vs actual)")
        print(f"  {'-' * 40}")
        print(f"  {'Predicted':>12s} {'Actual':>12s}")
        print(f"  {'-' * 40}")
        for pred, actual in results["calibration"]:
            bar = "#" * int(actual * 30)
            print(f"  {pred:>11.3f}  {actual:>11.3f}  {bar}")
        print(f"  {'-' * 40}")


def print_comparison(all_results: list[dict]):
    """Print comparison table of all models."""
    w = 110

    print(f"\n{'=' * w}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * w}")

    # Sort by ATS%
    sorted_results = sorted(all_results, key=lambda r: r["ats_pct"], reverse=True)

    print()
    print(
        f"  {'Model':<25s} {'Record':>12s} {'ATS%':>7s} {'AUC':>7s} "
        f"{'Profit':>10s} {'ROI':>8s} {'Kelly P/L':>10s} {'Brier':>7s}"
    )
    print(f"  {'-' * 100}")

    for r in sorted_results:
        rec = f"{r['ats_wins']}-{r['ats_losses']}"
        print(
            f"  {r['model']:<25s} {rec:>12s} "
            f"{r['ats_pct']:>6.1%} {r['auc']:>7.4f} "
            f"${r['profit']:>+9.0f} {r['roi_pct']:>+7.1f}% "
            f"${r['kelly_profit']:>+9.0f} {r['brier']:>7.4f}"
        )

    print(f"  {'-' * 100}")
    print(f"\n  Breakeven at -110: {1/1.909*100:.1f}%")

    # Confidence tier comparison for >55%
    print(f"\n  HIGH-CONFIDENCE PICKS (model says >55% one side)")
    print(f"  {'-' * 80}")
    print(
        f"  {'Model':<25s} {'Record':>12s} {'ATS%':>7s} "
        f"{'Profit':>10s} {'ROI':>8s} {'Games':>7s}"
    )
    print(f"  {'-' * 80}")

    for r in sorted_results:
        tier = r.get("tiers", {}).get(">55%", {})
        if tier.get("n", 0) == 0:
            print(f"  {r['model']:<25s} {'--- no picks at this threshold ---':>45s}")
        else:
            rec = f"{tier['wins']}-{tier['losses']}"
            print(
                f"  {r['model']:<25s} {rec:>12s} "
                f"{tier['ats_pct']:>6.1%} "
                f"${tier['profit']:>+9.0f} {tier['roi']:>+7.1f}% "
                f"{tier['n']:>7d}"
            )

    print(f"  {'-' * 80}")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_model(model, model_type: str, features: list[str], params: dict, scaler=None):
    """Save model and metadata."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if model_type == "xgboost":
        model_path = CHECKPOINT_DIR / "ats_xgboost.json"
        model.save_model(str(model_path))
        logger.info(f"  Saved XGBoost model to {model_path}")
    elif model_type == "lightgbm":
        model_path = CHECKPOINT_DIR / "ats_lightgbm.txt"
        model.booster_.save_model(str(model_path))
        logger.info(f"  Saved LightGBM model to {model_path}")
    elif model_type == "logistic":
        import pickle

        model_path = CHECKPOINT_DIR / "ats_logistic.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        logger.info(f"  Saved Logistic Regression to {model_path}")

    # Save metadata
    meta = {
        "model_type": model_type,
        "features": features,
        "params": params,
        "train_seasons_end": TRAIN_SEASONS_END,
        "val_seasons": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
    }
    meta_path = CHECKPOINT_DIR / f"ats_{model_type}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"  Saved metadata to {meta_path}")


def load_saved_model(model_path: Path) -> tuple:
    """Load a saved model from checkpoint."""
    meta_path = model_path.parent / (
        model_path.stem.replace("ats_", "ats_") + "_meta.json"
    )

    # Infer model type from filename
    if "xgboost" in model_path.name:
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        model_type = "xgboost"
    elif "lightgbm" in model_path.name:
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(model_path))
        model_type = "lightgbm"
    elif "logistic" in model_path.name:
        import pickle

        with open(model_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        model_type = "logistic"
    else:
        raise ValueError(f"Unknown model type for {model_path}")

    # Load metadata
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return model, model_type, meta


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------
def always_home_baseline(y_true: np.ndarray) -> dict:
    """Baseline: always pick home to cover."""
    n = len(y_true)
    wins = int(y_true.sum())
    losses = n - wins
    ats_pct = wins / n if n else 0
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = profit / (n * LOSS_COST) * 100 if n else 0

    return {
        "model": "Always Home",
        "n_games": n,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pct": ats_pct,
        "profit": profit,
        "roi_pct": roi,
        "auc": 0.5,
        "accuracy": ats_pct,
        "brier": np.mean((1.0 - y_true) ** 2),
        "log_loss": log_loss(y_true, np.full(n, 0.5 + 1e-7)),
        "kelly_bets": 0,
        "kelly_profit": 0,
        "kelly_wagered": 0,
        "kelly_roi": 0,
        "tiers": {
            "All": {
                "n": n,
                "wins": wins,
                "losses": losses,
                "ats_pct": ats_pct,
                "profit": profit,
                "roi": roi,
            }
        },
    }


def always_away_baseline(y_true: np.ndarray) -> dict:
    """Baseline: always pick away to cover."""
    n = len(y_true)
    wins = n - int(y_true.sum())
    losses = n - wins
    ats_pct = wins / n if n else 0
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = profit / (n * LOSS_COST) * 100 if n else 0

    return {
        "model": "Always Away",
        "n_games": n,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pct": ats_pct,
        "profit": profit,
        "roi_pct": roi,
        "auc": 0.5,
        "accuracy": 1 - ats_pct if ats_pct else 0,
        "brier": np.mean(y_true**2),
        "log_loss": log_loss(y_true, np.full(n, 0.5 + 1e-7)),
        "kelly_bets": 0,
        "kelly_profit": 0,
        "kelly_wagered": 0,
        "kelly_roi": 0,
        "tiers": {
            "All": {
                "n": n,
                "wins": wins,
                "losses": losses,
                "ats_pct": ats_pct,
                "profit": profit,
                "roi": roi,
            }
        },
    }


def random_baseline(y_true: np.ndarray, seed: int = 42) -> dict:
    """Baseline: random picks."""
    rng = np.random.default_rng(seed)
    y_pred = rng.integers(0, 2, size=len(y_true))
    n = len(y_true)
    wins = int((y_pred == y_true).sum())
    losses = n - wins
    ats_pct = wins / n if n else 0
    profit = wins * WIN_PAYOUT - losses * LOSS_COST
    roi = profit / (n * LOSS_COST) * 100 if n else 0

    return {
        "model": "Random",
        "n_games": n,
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pct": ats_pct,
        "profit": profit,
        "roi_pct": roi,
        "auc": 0.5,
        "accuracy": ats_pct,
        "brier": 0.25,
        "log_loss": log_loss(y_true, np.full(n, 0.5 + 1e-7)),
        "kelly_bets": 0,
        "kelly_profit": 0,
        "kelly_wagered": 0,
        "kelly_roi": 0,
        "tiers": {
            "All": {
                "n": n,
                "wins": wins,
                "losses": losses,
                "ats_pct": ats_pct,
                "profit": profit,
                "roi": roi,
            }
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6 Exp 5: ATS Binary Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full run with Optuna tuning
    python scripts/phase6_exp5_ats_classifier.py --optuna-trials 100

    # Quick run with defaults
    python scripts/phase6_exp5_ats_classifier.py

    # Evaluate saved model only
    python scripts/phase6_exp5_ats_classifier.py --evaluate-only \\
        --model checkpoints/phase6/ats_xgboost.json
        """,
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for XGBoost tuning (0 = use defaults)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training, only evaluate a saved model",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to saved model for --evaluate-only",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--no-lgbm",
        action="store_true",
        help="Skip LightGBM training",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        return 1

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"  Phase 6 Experiment 5: ATS Binary Classifier")
    print(f"{'#' * 80}\n")

    games_df = load_game_data(args.db)

    # ---------------------------------------------------------------
    # Step 2: Compute rolling features
    # ---------------------------------------------------------------
    games_df = compute_rolling_features(games_df)

    # ---------------------------------------------------------------
    # Step 3: Build feature matrix
    # ---------------------------------------------------------------
    games_df, feature_names = build_features(games_df)

    # ---------------------------------------------------------------
    # Step 4: Split
    # ---------------------------------------------------------------
    train_df, val_df, test_df = split_data(games_df, feature_names)

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["home_covered"].values.astype(np.int32)
    X_val = val_df[feature_names].values.astype(np.float32)
    y_val = val_df["home_covered"].values.astype(np.int32)
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["home_covered"].values.astype(np.int32)

    logger.info(
        f"Feature matrix shapes: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )

    # ---------------------------------------------------------------
    # Step 5: Evaluate-only mode
    # ---------------------------------------------------------------
    if args.evaluate_only:
        if args.model is None:
            logger.error("--evaluate-only requires --model path")
            return 1
        if not args.model.exists():
            logger.error(f"Model not found: {args.model}")
            return 1

        model, model_type, meta = load_saved_model(args.model)
        logger.info(f"Loaded {model_type} model from {args.model}")

        # Predict
        if model_type == "logistic":
            scaler = StandardScaler()
            scaler.fit(np.nan_to_num(X_train, nan=0.0))
            X_test_s = scaler.transform(np.nan_to_num(X_test, nan=0.0))
            test_proba = model.predict_proba(X_test_s)[:, 1]
        elif model_type == "lightgbm":
            test_proba = model.predict(np.nan_to_num(X_test, nan=np.nan))
        else:
            test_proba = model.predict_proba(X_test)[:, 1]

        results = evaluate_model(y_test, test_proba, test_df, f"{model_type} (loaded)")
        print_evaluation(results)
        return 0

    # ---------------------------------------------------------------
    # Step 6: Train models
    # ---------------------------------------------------------------
    all_results = []
    ensemble_probas = []

    # --- 6a: Logistic Regression ---
    lr_model, lr_scaler, lr_val_proba = train_logistic_regression(
        X_train, y_train, X_val, y_val, feature_names
    )
    X_test_s = lr_scaler.transform(np.nan_to_num(X_test, nan=0.0))
    lr_test_proba = lr_model.predict_proba(X_test_s)[:, 1]
    lr_val_results = evaluate_model(y_val, lr_val_proba, val_df, "LogReg (val)")
    lr_test_results = evaluate_model(y_test, lr_test_proba, test_df, "LogReg (test)")
    all_results.append(lr_test_results)
    save_model(lr_model, "logistic", feature_names, {"C": 0.1}, scaler=lr_scaler)

    # --- 6b: XGBoost ---
    xgb_model, xgb_params, xgb_val_proba = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        n_optuna_trials=args.optuna_trials,
    )
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_val_results = evaluate_model(y_val, xgb_val_proba, val_df, "XGBoost (val)")
    xgb_test_results = evaluate_model(y_test, xgb_test_proba, test_df, "XGBoost (test)")
    all_results.append(xgb_test_results)
    ensemble_probas.append(xgb_test_proba)
    save_model(xgb_model, "xgboost", feature_names, xgb_params)

    # --- 6c: LightGBM ---
    if not args.no_lgbm:
        try:
            lgb_model, lgb_val_proba = train_lightgbm(
                X_train, y_train, X_val, y_val, feature_names
            )
            lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]
            lgb_val_results = evaluate_model(
                y_val, lgb_val_proba, val_df, "LightGBM (val)"
            )
            lgb_test_results = evaluate_model(
                y_test, lgb_test_proba, test_df, "LightGBM (test)"
            )
            all_results.append(lgb_test_results)
            ensemble_probas.append(lgb_test_proba)
            save_model(lgb_model, "lightgbm", feature_names, {})
        except Exception as e:
            logger.error(f"LightGBM failed: {e}")

    # --- 6d: Calibrated XGBoost ---
    logger.info("Training Calibrated XGBoost (Platt scaling on val set)...")
    try:
        cal_model = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
        cal_model.fit(X_val, y_val)
        cal_test_proba = cal_model.predict_proba(X_test)[:, 1]
        cal_test_results = evaluate_model(
            y_test, cal_test_proba, test_df, "XGBoost-Calibrated (test)"
        )
        all_results.append(cal_test_results)
        ensemble_probas.append(cal_test_proba)
    except Exception as e:
        logger.error(f"Calibrated XGBoost failed: {e}")

    # --- 6e: Ensemble ---
    if len(ensemble_probas) >= 2:
        logger.info(f"Building ensemble from {len(ensemble_probas)} models...")
        ens_proba = np.mean(ensemble_probas, axis=0)
        ens_results = evaluate_model(y_test, ens_proba, test_df, "Ensemble (test)")
        all_results.append(ens_results)

    # --- Baselines ---
    all_results.append(always_home_baseline(y_test))
    all_results.append(always_away_baseline(y_test))
    all_results.append(random_baseline(y_test))

    # ---------------------------------------------------------------
    # Step 7: Print results
    # ---------------------------------------------------------------
    # Val set results
    print(f"\n{'#' * 80}")
    print(f"  VALIDATION SET RESULTS")
    print(f"{'#' * 80}")
    print_evaluation(lr_val_results)
    print_evaluation(xgb_val_results)
    if not args.no_lgbm and "lgb_val_results" in dir():
        print_evaluation(lgb_val_results)

    # Test set results
    print(f"\n{'#' * 80}")
    print(f"  TEST SET RESULTS")
    print(f"{'#' * 80}")
    for r in all_results:
        print_evaluation(r, detailed=True)

    # Comparison
    print_comparison(all_results)

    # ---------------------------------------------------------------
    # Step 8: Betting simulation
    # ---------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"  BETTING SIMULATION (Test Set)")
    print(f"{'#' * 80}")

    # Find the best model by val AUC
    best_model_name = "XGBoost"
    best_proba = xgb_test_proba

    print(f"\n  Using: {best_model_name}")
    print(f"  Strategy: Bet when model confidence > threshold\n")

    print(
        f"  {'Threshold':>10s} {'Bets':>6s} {'Record':>12s} {'ATS%':>7s} "
        f"{'Profit':>10s} {'ROI':>8s} {'Kelly f* avg':>12s}"
    )
    print(f"  {'-' * 75}")

    for threshold in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 0.65]:
        # For each game, pick the side we're more confident in
        picks_mask = np.maximum(best_proba, 1 - best_proba) > threshold
        if picks_mask.sum() == 0:
            print(f"  {threshold:>10.2f} {0:>6d} {'---':>12s}")
            continue

        pick_y = y_test[picks_mask]
        pick_p = best_proba[picks_mask]
        # Our predicted label: 1 if proba > 0.5 (home covers), else 0
        pick_pred = (pick_p > 0.5).astype(int)

        n = len(pick_y)
        wins = int((pick_pred == pick_y).sum())
        losses = n - wins
        ats_pct = wins / n
        profit = wins * WIN_PAYOUT - losses * LOSS_COST
        roi = profit / (n * LOSS_COST) * 100

        # Average Kelly fraction for these bets
        kelly_fracs = []
        for p in pick_p:
            side_prob = max(p, 1 - p)
            kelly_fracs.append(kelly_fraction(side_prob))

        avg_kelly = np.mean(kelly_fracs) if kelly_fracs else 0

        rec = f"{wins}-{losses}"
        print(
            f"  {threshold:>10.2f} {n:>6d} {rec:>12s} {ats_pct:>6.1%} "
            f"${profit:>+9.0f} {roi:>+7.1f}% {avg_kelly:>11.4f}"
        )

    print(f"  {'-' * 75}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"  SUMMARY")
    print(f"{'#' * 80}")

    best_test = max(
        [
            r
            for r in all_results
            if "Baseline" not in r["model"]
            and "Always" not in r["model"]
            and "Random" not in r["model"]
        ],
        key=lambda r: r["ats_pct"],
        default=None,
    )
    if best_test:
        print(f"\n  Best model: {best_test['model']}")
        print(
            f"  ATS: {best_test['ats_pct']:.1%} "
            f"({best_test['ats_wins']}-{best_test['ats_losses']})"
        )
        print(f"  ROI: {best_test['roi_pct']:+.2f}%")
        print(f"  AUC: {best_test['auc']:.4f}")

        breakeven = 1 / 1.909
        if best_test["ats_pct"] > breakeven:
            print(f"\n  ** PROFITABLE at -110 odds! **")
            print(
                f"  Edge: {(best_test['ats_pct'] - breakeven) * 100:+.2f} pp "
                f"over breakeven ({breakeven*100:.1f}%)"
            )
        else:
            print(
                f"\n  Below breakeven ({breakeven*100:.1f}%). "
                f"Gap: {(best_test['ats_pct'] - breakeven) * 100:.2f} pp"
            )

    print(f"\n  Models saved to: {CHECKPOINT_DIR}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
