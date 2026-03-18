"""Phase 4 Experiment 6a: TabPFN tabular foundation model for NBA game prediction.

Uses TabPFN 2.5 (a pre-trained tabular foundation model that performs
in-context learning) to predict NBA game spreads, scores, and win probabilities.

Three experiment variants:
  - 6a-zero:     Zero-shot TabPFN on full feature set
  - 6a-tuned:    Fine-tuned TabPFN (if supported by installed version)
  - 6a-ensemble: Ensemble of TabPFN models on different feature subsets

Data sources:
  - Pre-computed rolling stats from generative cache (data/generative_cache/context/)
  - Game features (teams, scores, dates) from generative cache
  - Additional features computed from SQLite database

Chronological split (matches Phase 4):
  - Train: 2008-2022 seasons
  - Val:   2023-2024
  - Test:  2024-2025, 2025-2026

Usage::

    python scripts/exp6a_tabpfn.py
    python scripts/exp6a_tabpfn.py --variant zero --split test
    python scripts/exp6a_tabpfn.py --variant ensemble --split val
    python scripts/exp6a_tabpfn.py --variant all --split test --save-predictions

"""

import argparse
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
CACHE_DIR = PROJECT_ROOT / "data" / "generative_cache" / "context"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Season splits (same as all Phase 4 experiments) ────────────────────────
TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2023)]
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]
ALL_SEASONS = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

# Historical team code mapping (same as cache_builder.py)
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "VAN": "MEM",
    "CHH": "CHA",
    "CHA": "CHA",
}

# ── Rolling stats feature names (24 features from cache_builder.py) ────────
ROLLING_STAT_NAMES = [
    "ppg",  # 0: avg pts scored / 150
    "opp_ppg",  # 1: avg pts allowed / 150
    "avg_margin",  # 2: avg margin / 50
    "avg_total",  # 3: avg total / 300
    "pts_std",  # 4: pts std / 30
    "margin_std",  # 5: margin std / 30
    "win_pct",  # 6: win %
    "streak",  # 7: streak / 10
    "efg_pct",  # 8: effective FG%
    "fg3a_rate",  # 9: 3PT attempt rate
    "ft_rate",  # 10: FT rate
    "ts_pct",  # 11: true shooting %
    "pace",  # 12: pace / 100
    "ast_ratio",  # 13: assist ratio / 3
    "def_rating_proxy",  # 14: defensive rating proxy (same as opp_ppg)
    "tov_rate",  # 15: turnover rate
    "rest_days",  # 16: rest days / 7
    "is_home",  # 17: is home (1.0/0.0)
    "season_progress",  # 18: games played / 82
    "ewm_margin_10",  # 19: EWM margin span 10 / 50
    "ewm_margin_20",  # 20: EWM margin span 20 / 50
    "ewm_win_10",  # 21: EWM win% span 10
    "ewm_ortg",  # 22: EWM offensive rating / 120
    "ewm_drtg",  # 23: EWM defensive rating / 120
]


def normalize_team(abbrev: str) -> str:
    """Map historical team abbreviations to current franchise codes."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_cache_data() -> tuple[dict, dict, dict]:
    """Load pre-built caches from the generative pipeline.

    Returns
    -------
    game_features : dict
        {game_id: {home_team, away_team, home_score, away_score, date, ...}}
    rolling_stats : dict
        {game_id: {home_stats: [24 floats], away_stats: [24 floats],
                    home_team_idx: int, away_team_idx: int}}
    season_games : dict
        {(team, season): [game_ids]}
    """
    import torch

    logger.info("Loading generative cache data...")

    game_features = torch.load(CACHE_DIR / "game_features.pt", weights_only=False)
    rolling_stats = torch.load(CACHE_DIR / "rolling_stats.pt", weights_only=False)

    # Convert season_games keys from "team|season" strings to (team, season) tuples
    raw_sg = torch.load(CACHE_DIR / "season_games.pt", weights_only=False)
    season_games = {}
    for key_str, game_ids in raw_sg.items():
        team, season = key_str.split("|", 1)
        season_games[(team, season)] = game_ids

    logger.info(
        f"  game_features: {len(game_features)} games, "
        f"rolling_stats: {len(rolling_stats)} games"
    )
    return game_features, rolling_stats, season_games


def get_season_from_date(date_str: str) -> str:
    """Derive NBA season string from a date (e.g., '2023-11-01' -> '2023-2024')."""
    if not date_str or len(date_str) < 10:
        return ""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        year, month = dt.year, dt.month
        if month >= 9:
            return f"{year}-{year + 1}"
        else:
            return f"{year - 1}-{year}"
    except ValueError:
        return ""


def compute_additional_features_from_db() -> dict[str, dict]:
    """Compute features from the database that are NOT in rolling_stats cache.

    Adds per-game:
      - home_h2h_wins: home team's H2H win% vs away team (last 10 meetings)
      - home_venue_win_pct: home team's home win% this season (prior games)
      - away_road_win_pct: away team's away win% this season (prior games)

    Returns {game_id: {feature_name: value}}
    """
    logger.info("Computing additional features from database...")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Fetch all completed regular season + playoff games chronologically
    season_placeholders = ",".join(["?"] * len(ALL_SEASONS))
    rows = conn.execute(
        f"""
        SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.season,
               home_tb.pts as home_pts, away_tb.pts as away_pts
        FROM Games g
        JOIN TeamBox home_tb ON g.game_id = home_tb.game_id
        JOIN Teams home_t ON home_tb.team_id = home_t.team_id
            AND home_t.abbreviation = g.home_team
        JOIN TeamBox away_tb ON g.game_id = away_tb.game_id
        JOIN Teams away_t ON away_tb.team_id = away_t.team_id
            AND away_t.abbreviation = g.away_team
        WHERE g.status = 3
          AND g.season IN ({season_placeholders})
          AND g.game_id NOT LIKE '003%'
        ORDER BY g.date_time_utc
        """,
        ALL_SEASONS,
    ).fetchall()
    conn.close()

    logger.info(f"  Loaded {len(rows)} games from database")

    # Build game list
    games = []
    for game_id, date_utc, home, away, season, h_pts, a_pts in rows:
        home = normalize_team(home)
        away = normalize_team(away)
        if not h_pts or not a_pts:
            continue
        games.append(
            {
                "game_id": game_id,
                "date": str(date_utc)[:10],
                "home": home,
                "away": away,
                "season": season,
                "home_pts": h_pts,
                "away_pts": a_pts,
                "home_win": 1 if h_pts > a_pts else 0,
            }
        )

    # Track running stats
    h2h_history: dict[tuple[str, str], list[int]] = defaultdict(
        list
    )  # (t1, t2) -> [1/0 wins for t1]
    home_record: dict[tuple[str, str], list[int]] = defaultdict(
        list
    )  # (team, season) -> [1/0 home wins]
    away_record: dict[tuple[str, str], list[int]] = defaultdict(
        list
    )  # (team, season) -> [1/0 away wins]

    extra_features = {}

    for game in games:
        gid = game["game_id"]
        home = game["home"]
        away = game["away"]
        season = game["season"]

        # H2H win% (last 10 meetings between these two teams, from home team's perspective)
        matchup_key = tuple(sorted([home, away]))
        h2h_list = h2h_history[matchup_key]
        # Compute from home team perspective
        h2h_wins_home = []
        for past_game_win in h2h_list[-10:]:
            h2h_wins_home.append(past_game_win)
        h2h_win_pct = np.mean(h2h_wins_home) if h2h_wins_home else 0.5

        # Home team's home win% this season (prior games only)
        home_rec = home_record[(home, season)]
        home_venue_wp = np.mean(home_rec) if home_rec else 0.5

        # Away team's road win% this season (prior games only)
        away_rec = away_record[(away, season)]
        away_road_wp = np.mean(away_rec) if away_rec else 0.5

        extra_features[gid] = {
            "h2h_win_pct": float(h2h_win_pct),
            "home_venue_win_pct": float(home_venue_wp),
            "away_road_win_pct": float(away_road_wp),
        }

        # Update histories (after computing features to prevent leakage)
        home_won = game["home_win"]
        # For H2H: store whether home team won (if home is first in sorted key)
        if matchup_key[0] == home:
            h2h_history[matchup_key].append(home_won)
        else:
            h2h_history[matchup_key].append(1 - home_won)

        home_record[(home, season)].append(home_won)
        away_record[(away, season)].append(1 - home_won)  # away win = home loss

    logger.info(f"  Computed extra features for {len(extra_features)} games")
    return extra_features


def build_dataframe(
    game_features: dict,
    rolling_stats: dict,
    extra_features: dict,
) -> pd.DataFrame:
    """Build a flat DataFrame with one row per game.

    Columns:
      - home_* / away_*: 24 rolling stats per side (renamed from indices to names)
      - h2h_win_pct, home_venue_win_pct, away_road_win_pct: extra DB features
      - diff_*: home - away differential features (for key stats)
      - target_spread, target_home_score, target_away_score, target_home_win: targets
      - game_id, season, date: metadata (not used as features)
    """
    logger.info("Building feature DataFrame...")
    rows = []

    for game_id, rs in rolling_stats.items():
        gf = game_features.get(game_id)
        if gf is None:
            continue

        # Need valid scores
        home_score = gf.get("home_score", 0)
        away_score = gf.get("away_score", 0)
        if home_score == 0 and away_score == 0:
            continue

        date_str = gf.get("date", "")
        season = get_season_from_date(date_str)
        if not season or season not in ALL_SEASONS:
            continue

        home_stats = rs["home_stats"]
        away_stats = rs["away_stats"]

        # Ensure we have the expected 24 features
        if len(home_stats) < 24 or len(away_stats) < 24:
            continue

        row = {
            "game_id": game_id,
            "season": season,
            "date": date_str,
        }

        # Add rolling stats as named features
        for i, name in enumerate(ROLLING_STAT_NAMES):
            h_val = float(home_stats[i])
            a_val = float(away_stats[i])
            row[f"home_{name}"] = h_val
            row[f"away_{name}"] = a_val

        # Differential features (key matchup indicators)
        # These capture the matchup directly, which is useful for tabular models
        row["diff_ppg"] = float(home_stats[0]) - float(away_stats[0])
        row["diff_margin"] = float(home_stats[2]) - float(away_stats[2])
        row["diff_win_pct"] = float(home_stats[6]) - float(away_stats[6])
        row["diff_efg"] = float(home_stats[8]) - float(away_stats[8])
        row["diff_ts_pct"] = float(home_stats[11]) - float(away_stats[11])
        row["diff_pace"] = float(home_stats[12]) - float(away_stats[12])
        row["diff_tov_rate"] = float(home_stats[15]) - float(away_stats[15])
        row["diff_ewm_margin"] = float(home_stats[19]) - float(away_stats[19])
        row["diff_ewm_win"] = float(home_stats[21]) - float(away_stats[21])
        row["diff_ewm_ortg"] = float(home_stats[22]) - float(away_stats[22])
        row["diff_ewm_drtg"] = float(home_stats[23]) - float(away_stats[23])

        # Extra DB features
        ef = extra_features.get(game_id, {})
        row["h2h_win_pct"] = ef.get("h2h_win_pct", 0.5)
        row["home_venue_win_pct"] = ef.get("home_venue_win_pct", 0.5)
        row["away_road_win_pct"] = ef.get("away_road_win_pct", 0.5)

        # Targets
        row["target_spread"] = home_score - away_score
        row["target_home_score"] = home_score
        row["target_away_score"] = away_score
        row["target_home_win"] = 1 if home_score > away_score else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"  DataFrame: {len(df)} games, {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes targets and metadata)."""
    exclude = {
        "game_id",
        "season",
        "date",
        "target_spread",
        "target_home_score",
        "target_away_score",
        "target_home_win",
    }
    return [c for c in df.columns if c not in exclude]


def get_feature_subsets() -> dict[str, list[str]]:
    """Define feature subsets for ensemble variant.

    Returns dict mapping subset name -> list of feature column name patterns.
    """
    return {
        "rolling_home_away": [
            # All 48 rolling stats (home + away) but no diffs or extras
            "home_",
            "away_",
        ],
        "differentials": [
            # Only differential features + rest/home/progress
            "diff_",
            "home_rest_days",
            "away_rest_days",
            "home_is_home",
            "home_season_progress",
        ],
        "matchup_focused": [
            # Differentials + H2H + venue
            "diff_",
            "h2h_",
            "home_venue_",
            "away_road_",
            "home_rest_days",
            "away_rest_days",
            "home_is_home",
            "home_season_progress",
        ],
        "offensive": [
            # Offensive-focused features
            "home_ppg",
            "away_ppg",
            "home_efg",
            "away_efg",
            "home_ts_pct",
            "away_ts_pct",
            "home_fg3a_rate",
            "away_fg3a_rate",
            "home_pace",
            "away_pace",
            "home_ast_ratio",
            "away_ast_ratio",
            "home_ewm_ortg",
            "away_ewm_ortg",
            "diff_ppg",
            "diff_efg",
            "diff_ts_pct",
            "diff_pace",
            "home_is_home",
            "home_season_progress",
        ],
        "momentum": [
            # Momentum/form features
            "home_win_pct",
            "away_win_pct",
            "home_streak",
            "away_streak",
            "home_ewm_margin_10",
            "away_ewm_margin_10",
            "home_ewm_margin_20",
            "away_ewm_margin_20",
            "home_ewm_win_10",
            "away_ewm_win_10",
            "home_margin_std",
            "away_margin_std",
            "diff_margin",
            "diff_win_pct",
            "diff_ewm_margin",
            "diff_ewm_win",
            "home_rest_days",
            "away_rest_days",
            "home_is_home",
            "home_season_progress",
            "h2h_win_pct",
        ],
    }


def select_features_by_patterns(
    all_features: list[str],
    patterns: list[str],
) -> list[str]:
    """Select features matching any of the given patterns (prefix match)."""
    selected = []
    for feat in all_features:
        for pattern in patterns:
            if feat.startswith(pattern) or feat == pattern:
                selected.append(feat)
                break
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(
    y_true_spread: np.ndarray,
    y_pred_spread: np.ndarray,
    y_pred_std: np.ndarray | None,
    y_true_win: np.ndarray,
    y_pred_win_prob: np.ndarray,
) -> dict:
    """Compute all evaluation metrics matching Phase 4 conventions."""
    from sklearn.metrics import roc_auc_score

    metrics = {}

    # Spread metrics
    metrics["spread_mae"] = float(np.mean(np.abs(y_pred_spread - y_true_spread)))
    metrics["spread_rmse"] = float(
        np.sqrt(np.mean((y_pred_spread - y_true_spread) ** 2))
    )

    # Win metrics
    try:
        metrics["win_auc"] = float(roc_auc_score(y_true_win, y_pred_win_prob))
    except ValueError:
        metrics["win_auc"] = float("nan")

    win_pred = (y_pred_win_prob > 0.5).astype(int)
    metrics["win_accuracy"] = float(np.mean(win_pred == y_true_win))

    # ECE (10-bin expected calibration error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred_win_prob >= bin_boundaries[i]) & (
            y_pred_win_prob < bin_boundaries[i + 1]
        )
        if mask.sum() > 0:
            avg_pred = y_pred_win_prob[mask].mean()
            avg_true = y_true_win[mask].mean()
            ece += mask.sum() / len(y_pred_win_prob) * abs(avg_pred - avg_true)
    metrics["win_ece"] = float(ece)

    # 90% coverage (if uncertainty estimates available)
    if y_pred_std is not None and np.all(np.isfinite(y_pred_std)):
        z90 = 1.645  # 90% confidence interval
        lower = y_pred_spread - z90 * y_pred_std
        upper = y_pred_spread + z90 * y_pred_std
        coverage = np.mean((y_true_spread >= lower) & (y_true_spread <= upper))
        metrics["coverage_90"] = float(coverage)
        metrics["avg_interval_width"] = float(np.mean(2 * z90 * y_pred_std))
    else:
        metrics["coverage_90"] = float("nan")
        metrics["avg_interval_width"] = float("nan")

    metrics["n_games"] = len(y_true_spread)

    return metrics


def print_metrics(metrics: dict, variant_name: str, split: str) -> None:
    """Print metrics in a clean table format."""
    print(f"\n{'=' * 65}")
    print(f"  Exp 6a-{variant_name} | {split} split | {metrics['n_games']} games")
    print(f"{'=' * 65}")
    print(f"  Spread MAE:          {metrics['spread_mae']:>8.2f}")
    print(f"  Spread RMSE:         {metrics['spread_rmse']:>8.2f}")
    print(f"  Win AUC:             {metrics['win_auc']:>8.3f}")
    print(f"  Win Accuracy:        {metrics['win_accuracy']:>8.1%}")
    print(f"  Win ECE:             {metrics['win_ece']:>8.4f}")
    if not np.isnan(metrics["coverage_90"]):
        print(f"  90% Coverage:        {metrics['coverage_90']:>8.1%}")
        print(f"  Avg Interval Width:  {metrics['avg_interval_width']:>8.2f}")
    print(f"{'=' * 65}")


# ═══════════════════════════════════════════════════════════════════════════
# TABPFN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════


def check_tabpfn_installed() -> bool:
    """Check if TabPFN is installed and print install instructions if not."""
    try:
        import tabpfn  # noqa: F401

        logger.info(f"TabPFN version: {tabpfn.__version__}")
        return True
    except ImportError:
        print(
            "\n"
            "TabPFN is not installed. Install it with:\n"
            "\n"
            "  pip install tabpfn\n"
            "\n"
            "For GPU support (recommended):\n"
            "  pip install tabpfn[cuda]\n"
            "\n"
            "Requires Python 3.10+ and PyTorch.\n"
            "See: https://github.com/PriorLabs/TabPFN\n"
        )
        return False


def spread_to_win_prob(spread_pred: np.ndarray, spread_std: np.ndarray) -> np.ndarray:
    """Convert spread predictions + uncertainty to win probability.

    P(home win) = P(spread > 0) = 1 - Phi(-spread / std)
    where Phi is the standard normal CDF.
    """
    from scipy.stats import norm

    # Avoid division by zero
    std_safe = np.maximum(spread_std, 1e-6)
    # P(home win) = P(spread > 0) = 1 - Phi(-spread/std) = Phi(spread/std)
    win_prob = norm.cdf(spread_pred / std_safe)
    return np.clip(win_prob, 0.01, 0.99)


def run_zero_shot(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    device: str = "cpu",
) -> dict:
    """Run zero-shot TabPFN (6a-zero): fit on train, predict on eval.

    TabPFN performs in-context learning — the "fit" call loads training data
    into the model's context, and "predict" runs a single forward pass.

    Note: TabPFN has a practical limit of ~10,000 training samples and ~500
    features. We subsample if needed.
    """
    from tabpfn import TabPFNRegressor

    X_train = df_train[feature_cols].values
    y_train = df_train["target_spread"].values
    X_eval = df_eval[feature_cols].values

    logger.info(
        f"  6a-zero: X_train={X_train.shape}, X_eval={X_eval.shape}, device={device}"
    )

    # TabPFN has practical limits — subsample training data if too large
    max_train = 10000
    if len(X_train) > max_train:
        logger.info(
            f"  Subsampling training data: {len(X_train)} -> {max_train} "
            f"(TabPFN context limit)"
        )
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Feature count limit — TabPFN works best with <100 features
    max_features = 100
    if X_train.shape[1] > max_features:
        logger.info(
            f"  Too many features ({X_train.shape[1]}), selecting top {max_features} "
            f"by variance"
        )
        variances = np.var(X_train, axis=0)
        top_idx = np.argsort(variances)[-max_features:]
        X_train = X_train[:, top_idx]
        X_eval = X_eval[:, top_idx]

    # Replace any NaN/inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.time()

    model = TabPFNRegressor(device=device, n_estimators=8)
    model.fit(X_train, y_train)

    # Predict with uncertainty
    try:
        y_pred, y_std = model.predict(X_eval, return_std=True)
    except TypeError:
        # Older versions may not support return_std
        y_pred = model.predict(X_eval)
        y_std = np.full_like(y_pred, 12.0)  # fallback: league-average std

    elapsed = time.time() - t0
    logger.info(f"  6a-zero completed in {elapsed:.1f}s")

    # Derive win probability from spread distribution
    win_prob = spread_to_win_prob(y_pred, y_std)

    return {
        "spread_pred": y_pred,
        "spread_std": y_std,
        "win_prob": win_prob,
    }


def run_tuned(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    device: str = "cpu",
) -> dict | None:
    """Run fine-tuned TabPFN (6a-tuned) if the API supports it.

    TabPFN 2.x may support post-hoc adaptation. Falls back gracefully
    if not available.
    """
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        return None

    X_train = df_train[feature_cols].values
    y_train = df_train["target_spread"].values
    X_eval = df_eval[feature_cols].values

    # Subsample if needed
    max_train = 10000
    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.time()

    # Try different adaptation strategies
    # Strategy 1: More estimators for better ensembling within TabPFN
    try:
        model = TabPFNRegressor(
            device=device,
            n_estimators=32,  # More internal ensembles
        )
        model.fit(X_train, y_train)

        try:
            y_pred, y_std = model.predict(X_eval, return_std=True)
        except TypeError:
            y_pred = model.predict(X_eval)
            y_std = np.full_like(y_pred, 12.0)

        elapsed = time.time() - t0
        logger.info(f"  6a-tuned (n_estimators=32) completed in {elapsed:.1f}s")

        win_prob = spread_to_win_prob(y_pred, y_std)

        return {
            "spread_pred": y_pred,
            "spread_std": y_std,
            "win_prob": win_prob,
        }

    except Exception as e:
        logger.warning(f"  6a-tuned failed: {e}")
        return None


def run_ensemble(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    device: str = "cpu",
) -> dict:
    """Run TabPFN ensemble (6a-ensemble): multiple models on feature subsets.

    Trains 5 TabPFN models on different feature subsets and combines
    predictions using inverse-variance weighting.
    """
    from tabpfn import TabPFNRegressor

    subsets = get_feature_subsets()

    all_preds = []
    all_stds = []
    subset_names = []

    for subset_name, patterns in subsets.items():
        subset_features = select_features_by_patterns(feature_cols, patterns)
        if len(subset_features) < 3:
            logger.warning(
                f"  Skipping subset '{subset_name}': only {len(subset_features)} features"
            )
            continue

        logger.info(
            f"  Ensemble subset '{subset_name}': {len(subset_features)} features"
        )

        X_train = df_train[subset_features].values
        y_train = df_train["target_spread"].values
        X_eval = df_eval[subset_features].values

        # Subsample training data
        max_train = 10000
        if len(X_train) > max_train:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X_train), max_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            model = TabPFNRegressor(device=device, n_estimators=8)
            model.fit(X_train, y_train)

            try:
                y_pred, y_std = model.predict(X_eval, return_std=True)
            except TypeError:
                y_pred = model.predict(X_eval)
                y_std = np.full_like(y_pred, 12.0)

            all_preds.append(y_pred)
            all_stds.append(y_std)
            subset_names.append(subset_name)

        except Exception as e:
            logger.warning(f"  Subset '{subset_name}' failed: {e}")
            continue

    if not all_preds:
        raise RuntimeError("All ensemble subsets failed")

    logger.info(f"  Combining {len(all_preds)} ensemble members...")

    # Combine using inverse-variance weighting
    preds = np.array(all_preds)  # (K, N)
    stds = np.array(all_stds)  # (K, N)

    # Clip stds to avoid division by zero
    stds_safe = np.maximum(stds, 1e-6)
    weights = 1.0 / (stds_safe**2)  # (K, N)
    weight_sum = weights.sum(axis=0)  # (N,)

    # Weighted mean prediction
    y_pred_ensemble = (weights * preds).sum(axis=0) / weight_sum

    # Combined uncertainty: mixture of Gaussians approximation
    y_std_ensemble = np.sqrt(1.0 / weight_sum)

    win_prob = spread_to_win_prob(y_pred_ensemble, y_std_ensemble)

    return {
        "spread_pred": y_pred_ensemble,
        "spread_std": y_std_ensemble,
        "win_prob": win_prob,
        "n_members": len(all_preds),
        "member_names": subset_names,
    }


def run_classifier(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    device: str = "cpu",
) -> dict | None:
    """Run TabPFNClassifier for direct win/loss prediction.

    Returns win probabilities from classification (not spread-derived).
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.warning("  TabPFNClassifier not available")
        return None

    X_train = df_train[feature_cols].values
    y_train = df_train["target_home_win"].values
    X_eval = df_eval[feature_cols].values

    max_train = 10000
    if len(X_train) > max_train:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.time()

    try:
        model = TabPFNClassifier(device=device, n_estimators=8)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_eval)

        # proba is (N, 2) with [P(loss), P(win)]
        if proba.shape[1] == 2:
            win_prob = proba[:, 1]
        else:
            win_prob = proba[:, 0]

        elapsed = time.time() - t0
        logger.info(f"  Classifier completed in {elapsed:.1f}s")

        return {"win_prob": win_prob}

    except Exception as e:
        logger.warning(f"  Classifier failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SAVE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════


def save_predictions(
    df_eval: pd.DataFrame,
    results: dict[str, dict],
    split: str,
) -> Path:
    """Save predictions to CSV for further analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = df_eval[
        [
            "game_id",
            "date",
            "season",
            "target_spread",
            "target_home_score",
            "target_away_score",
            "target_home_win",
        ]
    ].copy()

    for variant_name, result in results.items():
        prefix = variant_name.replace("-", "_")
        if "spread_pred" in result:
            out[f"{prefix}_spread_pred"] = result["spread_pred"]
        if "spread_std" in result:
            out[f"{prefix}_spread_std"] = result["spread_std"]
        if "win_prob" in result:
            out[f"{prefix}_win_prob"] = result["win_prob"]

    out_path = RESULTS_DIR / f"exp6a_tabpfn_predictions_{split}.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    logger.info(f"Saved predictions to {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Exp 6a: TabPFN for NBA game prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/exp6a_tabpfn.py\n"
            "  python scripts/exp6a_tabpfn.py --variant zero --split test\n"
            "  python scripts/exp6a_tabpfn.py --variant all --split val --save-predictions\n"
        ),
    )
    parser.add_argument(
        "--variant",
        default="all",
        choices=["zero", "tuned", "ensemble", "all"],
        help="Which experiment variant to run (default: all)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Evaluation split (default: test)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for TabPFN inference (default: auto-detect)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to results/exp6a_tabpfn_predictions_{split}.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Check TabPFN is installed
    if not check_tabpfn_installed():
        sys.exit(1)

    # Auto-detect device
    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    game_features, rolling_stats, season_games = load_cache_data()
    extra_features = compute_additional_features_from_db()

    df = build_dataframe(game_features, rolling_stats, extra_features)

    # ── Split data ─────────────────────────────────────────────────────────
    eval_seasons = VAL_SEASONS if args.split == "val" else TEST_SEASONS

    df_train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    df_eval = df[df["season"].isin(eval_seasons)].copy()

    logger.info(
        f"Split sizes — train: {len(df_train)}, " f"{args.split}: {len(df_eval)}"
    )

    if len(df_eval) == 0:
        print(f"No games found for {args.split} split")
        sys.exit(1)

    feature_cols = get_feature_columns(df)
    logger.info(f"Total features: {len(feature_cols)}")

    # Ground truth
    y_true_spread = df_eval["target_spread"].values
    y_true_win = df_eval["target_home_win"].values

    # ── Run experiments ────────────────────────────────────────────────────
    results = {}
    variants_to_run = (
        ["zero", "tuned", "ensemble"] if args.variant == "all" else [args.variant]
    )

    for variant in variants_to_run:
        print(f"\nRunning 6a-{variant}...")

        try:
            if variant == "zero":
                result = run_zero_shot(df_train, df_eval, feature_cols, device)
            elif variant == "tuned":
                result = run_tuned(df_train, df_eval, feature_cols, device)
                if result is None:
                    print(
                        "  6a-tuned: not supported by installed TabPFN version, skipping"
                    )
                    continue
            elif variant == "ensemble":
                result = run_ensemble(df_train, df_eval, feature_cols, device)
                if "n_members" in result:
                    print(
                        f"  Ensemble members: {result['n_members']} ({', '.join(result['member_names'])})"
                    )
            else:
                continue

            results[variant] = result

            # Compute and display metrics
            metrics = compute_metrics(
                y_true_spread=y_true_spread,
                y_pred_spread=result["spread_pred"],
                y_pred_std=result.get("spread_std"),
                y_true_win=y_true_win,
                y_pred_win_prob=result["win_prob"],
            )
            print_metrics(metrics, variant, args.split)

        except Exception as e:
            logger.error(f"  6a-{variant} failed: {e}", exc_info=True)
            continue

    # ── Also run classifier for comparison ─────────────────────────────────
    if args.variant == "all":
        print("\nRunning TabPFNClassifier (win/loss only)...")
        cls_result = run_classifier(df_train, df_eval, feature_cols, device)
        if cls_result is not None:
            from sklearn.metrics import roc_auc_score

            try:
                cls_auc = roc_auc_score(y_true_win, cls_result["win_prob"])
            except ValueError:
                cls_auc = float("nan")
            cls_acc = np.mean((cls_result["win_prob"] > 0.5).astype(int) == y_true_win)
            print(f"\n  TabPFNClassifier:  AUC={cls_auc:.3f}, Acc={cls_acc:.1%}")

    # ── Save predictions ───────────────────────────────────────────────────
    if args.save_predictions and results:
        out_path = save_predictions(df_eval, results, args.split)
        print(f"\nPredictions saved to: {out_path}")

    # ── Summary comparison ─────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Phase 4 Baselines for Comparison")
    print(f"{'=' * 65}")
    print("  Phase 3 Exp 9 (ensemble):   MAE 10.66, AUC 0.718, Acc 66.5%")
    print("  Phase 4 Exp 4 (generative):  MAE 11.76, AUC 0.662, Acc 61.4%")
    print("  Poisson baseline:            MAE ~13,   AUC ~0.60,  Acc ~58%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
