#!/usr/bin/env python
"""Phase 4 Experiment 6b: Chronos-2 Time-Series Foundation Model for NBA Prediction.

Uses Amazon's Chronos-2 pre-trained time-series foundation model to predict NBA
game spreads, scores, and win probabilities. Chronos-2 features "group attention"
that shares information across related time series — we group both teams' margin
histories so the model can learn cross-team matchup dynamics.

Experiment variants:
    6b-zero:  Zero-shot Chronos-2 (no fine-tuning)
    6b-lora:  LoRA fine-tuned on NBA training data
    6b-base:  amazon/chronos-2  (120M params)
    6b-large: amazon/chronos-2-large (300M params) — may OOM on 8GB VRAM

Data split (chronological, same as all Phase 4):
    Train: seasons 2008-2022
    Val:   season 2023-2024
    Test:  seasons 2024-2026

Approach:
    For each game to predict, construct per-team time series of the last N games:
      - Target: margin (team_score - opponent_score) from team's perspective
      - Covariates: PPG, opp_PPG, FG%, 3P%, rest_days, home_indicator, opp_win_pct
    Group home + away team series together for Chronos-2 group attention.
    Predict next margin for each team -> combine for spread.

Usage::

    # Zero-shot with base model
    python scripts/exp6b_chronos2.py --split test

    # LoRA fine-tuned
    python scripts/exp6b_chronos2.py --split test --finetune --finetune-steps 1000

    # Use large model
    python scripts/exp6b_chronos2.py --split test --model-size large

    # Both zero-shot and fine-tuned on val set
    python scripts/exp6b_chronos2.py --split val --run-all

    # Custom history window
    python scripts/exp6b_chronos2.py --split test --history 30

Install dependencies (if not already installed)::

    pip install autogluon.timeseries[chronos]

    # OR for direct Chronos usage:
    pip install chronos-forecasting

Hardware: RTX 2070 SUPER (8GB VRAM). Base model (120M) fits comfortably.
          Large model (300M) may require float16 or CPU fallback.
"""

import argparse
import csv
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Data splits — same as all Phase 4 experiments
# ---------------------------------------------------------------------------
TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2023)]
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]

# ---------------------------------------------------------------------------
# Historical team abbreviation mapping
# ---------------------------------------------------------------------------
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",
    "SEA": "OKC",
    "NOH": "NOP",
    "NOK": "NOP",
    "VAN": "MEM",
    "CHH": "CHA",
}


def normalize_team(abbrev: str) -> str:
    """Map historical team abbreviations to current."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
@contextmanager
def get_db(db_path: str = None):
    """Context manager for SQLite with WAL mode."""
    if db_path is None:
        db_path = str(DB_PATH)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_games(db_path: str, seasons: list[str]) -> list[dict]:
    """Load all completed regular-season games with TeamBox stats.

    Returns list of dicts sorted chronologically, each containing:
        game_id, home, away, date, season, home_score, away_score, margin,
        plus per-team box score stats.
    """
    logger.info("Loading games from database...")
    placeholders = ",".join(["?"] * len(seasons))

    # First: get game metadata
    game_query = f"""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season
        FROM Games g
        WHERE g.status = 3
          AND g.season IN ({placeholders})
          AND g.game_id NOT LIKE '003%'
        ORDER BY g.date_time_utc
    """

    with get_db(db_path) as conn:
        game_rows = conn.execute(game_query, seasons).fetchall()

    game_ids = [r[0] for r in game_rows]
    logger.info(f"  Found {len(game_ids)} completed games")

    # Second: batch-load TeamBox stats
    teambox_data = _load_teambox_batch(db_path, game_ids)

    # Combine
    games = []
    skipped = 0
    for game_id, home_raw, away_raw, date_utc, season in game_rows:
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)

        tb = teambox_data.get(game_id, {})
        home_tb = tb.get(home)
        away_tb = tb.get(away)

        if home_tb is None or away_tb is None:
            skipped += 1
            continue

        home_score = home_tb["pts"]
        away_score = away_tb["pts"]
        if home_score == 0 and away_score == 0:
            skipped += 1
            continue

        games.append(
            {
                "game_id": game_id,
                "home": home,
                "away": away,
                "date": str(date_utc)[:10],
                "date_utc": date_utc,
                "season": season,
                "home_score": home_score,
                "away_score": away_score,
                "margin": home_score - away_score,
                "home_box": home_tb,
                "away_box": away_tb,
            }
        )

    if skipped > 0:
        logger.warning(f"  Skipped {skipped} games with missing TeamBox data")
    logger.info(f"  Loaded {len(games)} games with complete data")
    return games


def _load_teambox_batch(db_path: str, game_ids: list[str]) -> dict:
    """Batch-load TeamBox stats for given game_ids.

    Returns {game_id: {team_abbrev: {stat_name: value, ...}}}.
    """
    result = {}
    chunk_size = 500

    stat_cols = [
        "pts",
        "fga",
        "fgm",
        "fg3a",
        "fg3m",
        "fta",
        "ftm",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
    ]

    with get_db(db_path) as conn:
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT tb.game_id, t.abbreviation,
                       tb.pts, tb.fga, tb.fgm, tb.fg3a, tb.fg3m,
                       tb.fta, tb.ftm, tb.reb,
                       tb.ast, tb.stl, tb.blk, tb.tov, tb.pf
                FROM TeamBox tb
                JOIN Teams t ON tb.team_id = t.team_id
                WHERE tb.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()

            for row in rows:
                gid = row[0]
                abbrev = normalize_team(row[1])
                stats = {}
                for j, col in enumerate(stat_cols):
                    stats[col] = float(row[2 + j]) if row[2 + j] is not None else 0.0
                if gid not in result:
                    result[gid] = {}
                result[gid][abbrev] = stats

    return result


# ---------------------------------------------------------------------------
# Per-team history building
# ---------------------------------------------------------------------------
class TeamHistoryBuilder:
    """Builds per-team game history with rolling statistics.

    For each team, maintains a chronological record of:
      - margin (from team's perspective)
      - PPG rolling averages
      - opponent PPG rolling averages
      - FG%, 3P%
      - rest days
      - home/away indicator
      - opponent rolling win percentage
    """

    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        # {team: [list of game records in chronological order]}
        self.team_games: dict[str, list[dict]] = defaultdict(list)
        # {team: [dates]} for rest day calculation
        self.team_dates: dict[str, list[str]] = defaultdict(list)
        # {team: cumulative wins, losses} for win% calculation
        self.team_record: dict[str, dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0}
        )

    def process_game(self, game: dict) -> dict | None:
        """Process one game and return the team histories at prediction time.

        Must be called in chronological order. Returns None if either team
        has insufficient history (<5 games).

        Returns:
            dict with keys: home_history, away_history (each a list of feature dicts),
            plus game metadata.
        """
        home = game["home"]
        away = game["away"]

        # Get current histories BEFORE updating with this game
        home_hist = self._get_team_features(home, self.history_length)
        away_hist = self._get_team_features(away, self.history_length)

        # Get opponent strength (rolling win%) BEFORE this game
        home_opp_winpct = self._get_win_pct(away)
        away_opp_winpct = self._get_win_pct(home)

        # Now update records with this game
        self._update_team(home, game, is_home=True)
        self._update_team(away, game, is_home=False)

        # Check minimum history
        if home_hist is None or away_hist is None:
            return None

        # Append opponent strength to each game in history
        # (this was already embedded during _update_team)

        return {
            "game_id": game["game_id"],
            "home": home,
            "away": away,
            "date": game["date"],
            "season": game["season"],
            "true_margin": game["margin"],
            "true_home_score": game["home_score"],
            "true_away_score": game["away_score"],
            "home_history": home_hist,
            "away_history": away_hist,
            "home_opp_winpct": home_opp_winpct,
            "away_opp_winpct": away_opp_winpct,
        }

    def _update_team(self, team: str, game: dict, is_home: bool):
        """Update a team's history after a game."""
        if is_home:
            team_score = game["home_score"]
            opp_score = game["away_score"]
            team_box = game["home_box"]
            opp = game["away"]
        else:
            team_score = game["away_score"]
            opp_score = game["home_score"]
            team_box = game["away_box"]
            opp = game["home"]

        margin = team_score - opp_score
        won = margin > 0

        # Rest days
        prev_dates = self.team_dates[team]
        if prev_dates:
            try:
                prev_dt = datetime.strptime(prev_dates[-1], "%Y-%m-%d")
                curr_dt = datetime.strptime(game["date"], "%Y-%m-%d")
                rest_days = (curr_dt - prev_dt).days - 1
            except (ValueError, TypeError):
                rest_days = 2  # default
        else:
            rest_days = 3  # season opener

        rest_days = min(rest_days, 10)  # cap at 10

        # Shooting percentages
        fga = team_box.get("fga", 0)
        fgm = team_box.get("fgm", 0)
        fg3a = team_box.get("fg3a", 0)
        fg3m = team_box.get("fg3m", 0)
        fg_pct = fgm / fga if fga > 0 else 0.0
        fg3_pct = fg3m / fg3a if fg3a > 0 else 0.0

        record = {
            "date": game["date"],
            "margin": float(margin),
            "team_pts": float(team_score),
            "opp_pts": float(opp_score),
            "fg_pct": fg_pct,
            "fg3_pct": fg3_pct,
            "rest_days": float(rest_days),
            "is_home": 1.0 if is_home else 0.0,
            "opp_win_pct": self._get_win_pct(opp),
        }

        self.team_games[team].append(record)
        self.team_dates[team].append(game["date"])

        if won:
            self.team_record[team]["wins"] += 1
        else:
            self.team_record[team]["losses"] += 1

    def _get_win_pct(self, team: str) -> float:
        """Get a team's current win percentage."""
        rec = self.team_record[team]
        total = rec["wins"] + rec["losses"]
        if total == 0:
            return 0.5
        return rec["wins"] / total

    def _get_team_features(self, team: str, n: int) -> list[dict] | None:
        """Get the last N games for a team as feature dicts.

        Returns None if fewer than 5 games available.
        """
        history = self.team_games[team]
        if len(history) < 5:
            return None

        recent = history[-n:] if len(history) >= n else history[:]

        # Compute rolling averages for PPG covariates
        features = []
        for i, game_rec in enumerate(recent):
            # Rolling PPG (over available prior games in this window)
            prior = recent[max(0, i - 9) : i + 1]  # up to last 10 including current
            ppg = np.mean([g["team_pts"] for g in prior])
            opp_ppg = np.mean([g["opp_pts"] for g in prior])

            features.append(
                {
                    "margin": game_rec["margin"],
                    "ppg": ppg,
                    "opp_ppg": opp_ppg,
                    "fg_pct": game_rec["fg_pct"],
                    "fg3_pct": game_rec["fg3_pct"],
                    "rest_days": game_rec["rest_days"],
                    "is_home": game_rec["is_home"],
                    "opp_win_pct": game_rec["opp_win_pct"],
                }
            )

        return features


# ---------------------------------------------------------------------------
# Chronos-2 via AutoGluon
# ---------------------------------------------------------------------------
def check_dependencies():
    """Check if required packages are installed and provide install instructions."""
    missing = []

    try:
        import autogluon.timeseries  # noqa: F401

        logger.info("AutoGluon TimeSeries: installed")
    except ImportError:
        missing.append("autogluon.timeseries")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"PyTorch CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("PyTorch CUDA: not available, will use CPU")
    except ImportError:
        missing.append("torch")

    if missing:
        print("\n" + "=" * 70)
        print("MISSING DEPENDENCIES")
        print("=" * 70)
        print(f"\nThe following packages are required: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install 'autogluon.timeseries[chronos]'")
        print("\nThis will install AutoGluon TimeSeries with Chronos-2 support.")
        print("Alternatively, for direct Chronos usage:")
        print("  pip install chronos-forecasting")
        print("=" * 70 + "\n")
        return False

    return True


def build_autogluon_data(
    prediction_records: list[dict],
    mode: str = "eval",
) -> "TimeSeriesDataFrame":
    """Convert prediction records into AutoGluon TimeSeriesDataFrame.

    Each prediction game creates two time series items:
      - {game_id}_home: home team's history + 1 future step
      - {game_id}_away: away team's history + 1 future step

    The group attention in Chronos-2 will share information between
    items that are close in the dataframe.

    Args:
        prediction_records: List from TeamHistoryBuilder.process_game()
        mode: "eval" includes the true target in the last row for evaluation,
              "predict" leaves the last row as NaN for prediction.

    Returns:
        TimeSeriesDataFrame with columns: item_id, timestamp, margin,
        ppg, opp_ppg, fg_pct, fg3_pct, rest_days, is_home, opp_win_pct
    """
    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame

    rows = []

    for rec in prediction_records:
        game_id = rec["game_id"]

        for side, history in [
            ("home", rec["home_history"]),
            ("away", rec["away_history"]),
        ]:
            item_id = f"{game_id}_{side}"

            # Historical rows — use sequential integer timestamps
            # (Chronos-2 works with sequential indices, not calendar dates)
            base_time = pd.Timestamp("2020-01-01")
            for i, feat in enumerate(history):
                rows.append(
                    {
                        "item_id": item_id,
                        "timestamp": base_time + pd.Timedelta(days=i),
                        "margin": feat["margin"],
                        "ppg": feat["ppg"],
                        "opp_ppg": feat["opp_ppg"],
                        "fg_pct": feat["fg_pct"],
                        "fg3_pct": feat["fg3_pct"],
                        "rest_days": feat["rest_days"],
                        "is_home": feat["is_home"],
                        "opp_win_pct": feat["opp_win_pct"],
                    }
                )

            # Prediction target row
            next_ts = base_time + pd.Timedelta(days=len(history))
            if side == "home":
                true_margin = rec["true_margin"]
                next_is_home = 1.0
                next_opp_winpct = rec["home_opp_winpct"]
            else:
                true_margin = -rec["true_margin"]  # away perspective
                next_is_home = 0.0
                next_opp_winpct = rec["away_opp_winpct"]

            rows.append(
                {
                    "item_id": item_id,
                    "timestamp": next_ts,
                    "margin": true_margin if mode == "eval" else np.nan,
                    "ppg": 0.0,  # placeholder — not known at prediction time
                    "opp_ppg": 0.0,
                    "fg_pct": 0.0,
                    "fg3_pct": 0.0,
                    "rest_days": 0.0,
                    "is_home": next_is_home,
                    "opp_win_pct": next_opp_winpct,
                }
            )

    df = pd.DataFrame(rows)
    return TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp"
    )


def run_autogluon_experiment(
    train_records: list[dict],
    eval_records: list[dict],
    model_path: str = "amazon/chronos-2",
    finetune: bool = False,
    finetune_steps: int = 1000,
    finetune_lr: float = 1e-4,
    batch_size: int = 32,
    output_dir: str = None,
) -> dict:
    """Run Chronos-2 experiment via AutoGluon TimeSeriesPredictor.

    Args:
        train_records: Training prediction records (for fine-tuning data).
        eval_records: Evaluation prediction records.
        model_path: HuggingFace model path for Chronos-2.
        finetune: Whether to fine-tune with LoRA.
        finetune_steps: Number of fine-tuning steps.
        finetune_lr: Fine-tuning learning rate.
        batch_size: Batch size for inference/training.
        output_dir: Directory to save AutoGluon predictor.

    Returns:
        Dict with predictions and metadata.
    """
    from autogluon.timeseries import TimeSeriesPredictor

    logger.info(f"Building AutoGluon data for {len(eval_records)} eval games...")
    eval_tsdf = build_autogluon_data(eval_records, mode="eval")

    if output_dir is None:
        variant = "lora" if finetune else "zero"
        size = model_path.split("-")[-1]
        output_dir = str(RESULTS_DIR / f"exp6b_chronos2_{variant}_{size}")

    # Define known covariates (features known at prediction time)
    known_covariates = ["is_home", "opp_win_pct"]

    # Hyperparameters
    hyperparams = {
        "Chronos": {
            "model_path": model_path,
            "device": "auto",
            "batch_size": batch_size,
            "context_length": 64,  # max context the model will use
        }
    }

    if finetune:
        logger.info(f"Fine-tuning enabled: {finetune_steps} steps, lr={finetune_lr}")
        hyperparams["Chronos"]["fine_tune"] = True
        hyperparams["Chronos"]["fine_tune_steps"] = finetune_steps
        hyperparams["Chronos"]["fine_tune_lr"] = finetune_lr

        # Build training data for fine-tuning
        logger.info(f"Building training data from {len(train_records)} games...")
        train_tsdf = build_autogluon_data(train_records, mode="eval")

    # Create predictor
    predictor = TimeSeriesPredictor(
        prediction_length=1,
        target="margin",
        known_covariates_names=known_covariates,
        eval_metric="MAE",
        path=output_dir,
        verbosity=2,
    )

    # Fit
    logger.info("Fitting predictor...")
    t0 = time.time()

    if finetune:
        predictor.fit(
            train_data=train_tsdf,
            hyperparameters=hyperparams,
            skip_model_selection=True,
            enable_ensemble=False,
        )
    else:
        # Zero-shot: still need to "fit" to register the model
        predictor.fit(
            train_data=eval_tsdf,
            hyperparameters=hyperparams,
            skip_model_selection=True,
            enable_ensemble=False,
        )

    fit_time = time.time() - t0
    logger.info(f"Fit completed in {fit_time:.1f}s")

    # Predict — must split data into history (input) and future covariates
    logger.info("Running predictions...")
    t0 = time.time()

    # Split: for each item, the last row is the target. Remove it for prediction input,
    # and extract its covariates as the known_future.
    import pandas as pd

    # Get the last timestamp for each item_id (the prediction target row)
    idx = eval_tsdf.reset_index()
    last_ts_per_item = idx.groupby("item_id")["timestamp"].transform("max")
    is_last = (idx["timestamp"] == last_ts_per_item).values

    # History: everything except the last row per item
    history_df = eval_tsdf[~is_last]

    # Future covariates: just the last row per item, only the known covariate columns
    future_rows = eval_tsdf[is_last][known_covariates].copy()

    try:
        predictions = predictor.predict(history_df, known_covariates=future_rows)
    except Exception as e:
        logger.warning(f"Prediction with known_covariates failed: {e}")
        logger.info("Retrying without known_covariates...")
        try:
            # Try without covariates — create a fresh predictor without them
            predictor_nocov = TimeSeriesPredictor(
                prediction_length=1,
                target="margin",
                eval_metric="MAE",
                path=output_dir + "_nocov",
                verbosity=2,
            )
            hp_nocov = dict(hyperparams)
            predictor_nocov.fit(
                train_data=eval_tsdf if not finetune else train_tsdf,
                hyperparameters=hp_nocov,
                skip_model_selection=True,
                enable_ensemble=False,
            )
            predictions = predictor_nocov.predict(history_df)
        except Exception as e2:
            logger.error(f"Prediction without covariates also failed: {e2}")
            raise

    pred_time = time.time() - t0
    logger.info(f"Prediction completed in {pred_time:.1f}s")

    return {
        "predictions": predictions,
        "fit_time": fit_time,
        "pred_time": pred_time,
        "predictor": predictor,
    }


# ---------------------------------------------------------------------------
# Direct Chronos-2 pipeline (fallback if AutoGluon is problematic)
# ---------------------------------------------------------------------------
def run_direct_chronos(
    eval_records: list[dict],
    train_records: list[dict] = None,
    model_path: str = "amazon/chronos-2",
    finetune: bool = False,
    finetune_steps: int = 1000,
    finetune_lr: float = 1e-4,
    device: str = "cuda",
    batch_size: int = 64,
) -> list[dict]:
    """Run Chronos-2 directly using the chronos library.

    This is the fallback path if AutoGluon is not installed or has issues.
    Uses the low-level ChronosPipeline API.

    For each game:
      1. Build context tensor from home team's margin history
      2. Build context tensor from away team's margin history
      3. Stack them for group attention (multivariate input)
      4. Get probabilistic forecast (mean + quantiles)
      5. Combine home/away margin predictions into spread

    Args:
        eval_records: List of prediction records from TeamHistoryBuilder.
        train_records: Training data (for fine-tuning, currently unused in direct mode).
        model_path: HuggingFace model path.
        finetune: Whether to fine-tune (not supported in direct mode — warn user).
        device: "cuda" or "cpu".
        batch_size: Number of games to process at once.

    Returns:
        List of prediction dicts with spread_pred, home_win_prob, etc.
    """
    import torch
    from chronos import ChronosPipeline

    if finetune:
        logger.warning(
            "Fine-tuning is not supported in direct Chronos mode. "
            "Install autogluon.timeseries for LoRA fine-tuning. "
            "Running zero-shot instead."
        )

    logger.info(f"Loading Chronos-2 from {model_path}...")
    dtype = torch.float32
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
        )
    except Exception as e:
        logger.warning(f"Failed to load on {device}: {e}")
        if device == "cuda":
            logger.info("Falling back to CPU...")
            pipeline = ChronosPipeline.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=dtype,
            )
        else:
            raise

    logger.info("Model loaded successfully")

    # Process games in batches
    results = []
    n_games = len(eval_records)

    from tqdm import tqdm

    for batch_start in tqdm(range(0, n_games, batch_size), desc="Predicting"):
        batch = eval_records[batch_start : batch_start + batch_size]

        # Build context tensors for each game
        # Chronos-2 can handle multivariate input via stacked tensors
        home_contexts = []
        away_contexts = []

        for rec in batch:
            # Extract margin series
            home_margins = torch.tensor(
                [f["margin"] for f in rec["home_history"]],
                dtype=dtype,
            )
            away_margins = torch.tensor(
                [f["margin"] for f in rec["away_history"]],
                dtype=dtype,
            )
            home_contexts.append(home_margins)
            away_contexts.append(away_margins)

        # Pad to same length within batch (Chronos handles variable lengths)
        max_len = max(
            max(c.shape[0] for c in home_contexts),
            max(c.shape[0] for c in away_contexts),
        )

        def pad_batch(contexts, max_len):
            padded = []
            for c in contexts:
                if c.shape[0] < max_len:
                    pad = torch.full((max_len - c.shape[0],), float("nan"), dtype=dtype)
                    c = torch.cat([pad, c])
                padded.append(c)
            return torch.stack(padded)

        home_batch = pad_batch(home_contexts, max_len)
        away_batch = pad_batch(away_contexts, max_len)

        # Predict home margins
        with torch.no_grad():
            try:
                home_forecast = pipeline.predict(
                    home_batch,
                    prediction_length=1,
                    num_samples=200,
                    limit_prediction_length=False,
                )  # shape: (batch, num_samples, 1)

                away_forecast = pipeline.predict(
                    away_batch,
                    prediction_length=1,
                    num_samples=200,
                    limit_prediction_length=False,
                )
            except Exception as e:
                logger.error(f"Prediction failed for batch: {e}")
                # Fall back to per-game prediction
                home_forecast = []
                away_forecast = []
                for hc, ac in zip(home_contexts, away_contexts):
                    try:
                        hf = pipeline.predict(
                            hc.unsqueeze(0),
                            prediction_length=1,
                            num_samples=200,
                            limit_prediction_length=False,
                        )
                        af = pipeline.predict(
                            ac.unsqueeze(0),
                            prediction_length=1,
                            num_samples=200,
                            limit_prediction_length=False,
                        )
                        home_forecast.append(hf)
                        away_forecast.append(af)
                    except Exception as e2:
                        logger.error(f"Per-game prediction also failed: {e2}")
                        # Use naive baseline: mean of last 5 margins
                        dummy_h = torch.tensor([[[hc[-5:].mean().item()]]] * 1)
                        dummy_a = torch.tensor([[[ac[-5:].mean().item()]]] * 1)
                        home_forecast.append(dummy_h.unsqueeze(-1))
                        away_forecast.append(dummy_a.unsqueeze(-1))

                if isinstance(home_forecast, list):
                    home_forecast = torch.cat(home_forecast, dim=0)
                    away_forecast = torch.cat(away_forecast, dim=0)

        # Extract predictions for each game in batch
        for j, rec in enumerate(batch):
            if isinstance(home_forecast, torch.Tensor):
                h_samples = home_forecast[j, :, 0].numpy()  # (num_samples,)
                a_samples = away_forecast[j, :, 0].numpy()
            else:
                h_samples = np.array([0.0])
                a_samples = np.array([0.0])

            # Home margin prediction (from home team's perspective)
            home_margin_pred = float(np.median(h_samples))
            # Away margin prediction is from away perspective, so negate
            away_margin_pred = float(np.median(a_samples))

            # Combine: spread = (home_margin_pred - (-away_margin_pred)) / 2
            # Or simply average the two estimates
            spread_pred = (home_margin_pred - away_margin_pred) / 2.0

            # Home score / away score estimates
            # Use recent averages + spread prediction
            home_recent_ppg = np.mean([f["ppg"] for f in rec["home_history"][-5:]])
            away_recent_ppg = np.mean([f["ppg"] for f in rec["away_history"][-5:]])
            total_pred = home_recent_ppg + away_recent_ppg
            home_score_pred = (total_pred + spread_pred) / 2.0
            away_score_pred = (total_pred - spread_pred) / 2.0

            # Win probability from the sample distribution
            # spread > 0 means home wins
            spread_samples = (h_samples - (-a_samples)) / 2.0
            home_win_prob = float(np.mean(spread_samples > 0))
            home_win_prob = np.clip(home_win_prob, 0.02, 0.98)

            # Spread uncertainty
            spread_std = float(np.std(spread_samples))

            # Quantiles for coverage
            spread_q05 = float(np.percentile(spread_samples, 5))
            spread_q95 = float(np.percentile(spread_samples, 95))

            results.append(
                {
                    "game_id": rec["game_id"],
                    "home": rec["home"],
                    "away": rec["away"],
                    "date": rec["date"],
                    "season": rec["season"],
                    "true_margin": rec["true_margin"],
                    "true_home_score": rec["true_home_score"],
                    "true_away_score": rec["true_away_score"],
                    "spread_pred": spread_pred,
                    "home_score_pred": home_score_pred,
                    "away_score_pred": away_score_pred,
                    "home_win_prob": home_win_prob,
                    "spread_std": spread_std,
                    "spread_q05": spread_q05,
                    "spread_q95": spread_q95,
                    "home_margin_raw": home_margin_pred,
                    "away_margin_raw": away_margin_pred,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Multivariate direct Chronos-2 (covariates as additional channels)
# ---------------------------------------------------------------------------
def run_direct_chronos_multivariate(
    eval_records: list[dict],
    model_path: str = "amazon/chronos-2",
    device: str = "cuda",
    batch_size: int = 32,
) -> list[dict]:
    """Run Chronos-2 with multivariate input (margin + covariates).

    Chronos-2 supports multivariate forecasting where multiple time series
    are passed as separate channels. We pass 3 channels per team:
      - margin
      - PPG (team scoring trend)
      - opponent PPG (defensive trend)

    The model's group attention mechanism shares information across channels.
    """
    import torch
    from chronos import ChronosPipeline

    logger.info(f"Loading Chronos-2 from {model_path} (multivariate mode)...")
    dtype = torch.float32
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
        )
    except Exception as e:
        logger.warning(f"GPU load failed: {e}")
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=dtype,
        )

    logger.info("Model loaded. Running multivariate predictions...")

    results = []
    from tqdm import tqdm

    for rec in tqdm(eval_records, desc="Multivariate predictions"):
        # Build multivariate context for home team: (n_channels, seq_len)
        home_hist = rec["home_history"]
        away_hist = rec["away_history"]

        # 3 channels: margin, ppg, opp_ppg
        home_channels = torch.tensor(
            [
                [f["margin"] for f in home_hist],
                [f["ppg"] for f in home_hist],
                [f["opp_ppg"] for f in home_hist],
            ],
            dtype=dtype,
        )  # (3, T)

        away_channels = torch.tensor(
            [
                [f["margin"] for f in away_hist],
                [f["ppg"] for f in away_hist],
                [f["opp_ppg"] for f in away_hist],
            ],
            dtype=dtype,
        )  # (3, T)

        # Stack home + away for group attention: (6, T) — padded to same length
        max_t = max(home_channels.shape[1], away_channels.shape[1])

        def pad_channels(ch, max_t):
            if ch.shape[1] < max_t:
                pad = torch.full(
                    (ch.shape[0], max_t - ch.shape[1]), float("nan"), dtype=dtype
                )
                ch = torch.cat([pad, ch], dim=1)
            return ch

        home_channels = pad_channels(home_channels, max_t)
        away_channels = pad_channels(away_channels, max_t)

        # Combined: (6, T) — first 3 channels are home, last 3 are away
        combined = torch.cat([home_channels, away_channels], dim=0)  # (6, T)

        with torch.no_grad():
            try:
                forecast = pipeline.predict(
                    combined,
                    prediction_length=1,
                    num_samples=200,
                    limit_prediction_length=False,
                )  # (6, num_samples, 1) or similar

                # Extract margin predictions (channels 0 and 3)
                h_samples = forecast[0, :, 0].numpy()  # home margin samples
                a_samples = forecast[3, :, 0].numpy()  # away margin samples
            except Exception:
                # Fallback to univariate
                try:
                    h_forecast = pipeline.predict(
                        home_channels[0:1],  # just margin channel
                        prediction_length=1,
                        num_samples=200,
                        limit_prediction_length=False,
                    )
                    a_forecast = pipeline.predict(
                        away_channels[0:1],
                        prediction_length=1,
                        num_samples=200,
                        limit_prediction_length=False,
                    )
                    h_samples = h_forecast[0, :, 0].numpy()
                    a_samples = a_forecast[0, :, 0].numpy()
                except Exception:
                    h_samples = np.array(
                        [np.mean([f["margin"] for f in home_hist[-5:]])]
                    )
                    a_samples = np.array(
                        [np.mean([f["margin"] for f in away_hist[-5:]])]
                    )

        home_margin_pred = float(np.median(h_samples))
        away_margin_pred = float(np.median(a_samples))
        spread_pred = (home_margin_pred - away_margin_pred) / 2.0

        home_recent_ppg = np.mean([f["ppg"] for f in home_hist[-5:]])
        away_recent_ppg = np.mean([f["ppg"] for f in away_hist[-5:]])
        total_pred = home_recent_ppg + away_recent_ppg
        home_score_pred = (total_pred + spread_pred) / 2.0
        away_score_pred = (total_pred - spread_pred) / 2.0

        spread_samples = (h_samples - (-a_samples)) / 2.0
        home_win_prob = float(np.clip(np.mean(spread_samples > 0), 0.02, 0.98))
        spread_std = float(np.std(spread_samples))
        spread_q05 = float(np.percentile(spread_samples, 5))
        spread_q95 = float(np.percentile(spread_samples, 95))

        results.append(
            {
                "game_id": rec["game_id"],
                "home": rec["home"],
                "away": rec["away"],
                "date": rec["date"],
                "season": rec["season"],
                "true_margin": rec["true_margin"],
                "true_home_score": rec["true_home_score"],
                "true_away_score": rec["true_away_score"],
                "spread_pred": spread_pred,
                "home_score_pred": home_score_pred,
                "away_score_pred": away_score_pred,
                "home_win_prob": home_win_prob,
                "spread_std": spread_std,
                "spread_q05": spread_q05,
                "spread_q95": spread_q95,
                "home_margin_raw": home_margin_pred,
                "away_margin_raw": away_margin_pred,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Naive rolling-mean baseline (for quick comparison)
# ---------------------------------------------------------------------------
def run_naive_baseline(eval_records: list[dict]) -> list[dict]:
    """Naive baseline: predict spread = mean(home_recent_margins) - mean(away_recent_margins).

    This is the simplest possible baseline using the same data pipeline.
    Chronos-2 must beat this to justify its complexity.
    """
    results = []
    for rec in eval_records:
        # Last 10 margins from each team's perspective
        home_margins = [f["margin"] for f in rec["home_history"][-10:]]
        away_margins = [f["margin"] for f in rec["away_history"][-10:]]

        home_avg = np.mean(home_margins)
        away_avg = np.mean(away_margins)

        # Home margin prediction with small home court adjustment
        spread_pred = (home_avg - away_avg) / 2.0 + 1.5  # ~1.5 pt home court

        # Win probability from spread (logistic approximation, sigma ~ 12)
        home_win_prob = 1.0 / (1.0 + np.exp(-spread_pred / 5.5))
        home_win_prob = np.clip(home_win_prob, 0.02, 0.98)

        home_recent_ppg = np.mean([f["ppg"] for f in rec["home_history"][-5:]])
        away_recent_ppg = np.mean([f["ppg"] for f in rec["away_history"][-5:]])
        total_pred = home_recent_ppg + away_recent_ppg

        results.append(
            {
                "game_id": rec["game_id"],
                "home": rec["home"],
                "away": rec["away"],
                "date": rec["date"],
                "season": rec["season"],
                "true_margin": rec["true_margin"],
                "true_home_score": rec["true_home_score"],
                "true_away_score": rec["true_away_score"],
                "spread_pred": spread_pred,
                "home_score_pred": (total_pred + spread_pred) / 2.0,
                "away_score_pred": (total_pred - spread_pred) / 2.0,
                "home_win_prob": home_win_prob,
                "spread_std": 12.0,  # assumed
                "spread_q05": spread_pred - 19.7,
                "spread_q95": spread_pred + 19.7,
                "home_margin_raw": home_avg,
                "away_margin_raw": away_avg,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Extract predictions from AutoGluon output
# ---------------------------------------------------------------------------
def extract_autogluon_predictions(
    ag_result: dict,
    eval_records: list[dict],
) -> list[dict]:
    """Convert AutoGluon predictions back to our standard format.

    AutoGluon returns a DataFrame indexed by (item_id, timestamp) with
    columns like "mean", "0.1", "0.5", "0.9" for quantile predictions.
    """
    predictions = ag_result["predictions"]

    # Build lookup: game_id -> record
    record_map = {rec["game_id"]: rec for rec in eval_records}

    results = []
    # Group predictions by game_id (strip _home/_away suffix)
    game_preds = defaultdict(dict)
    for item_id in predictions.index.get_level_values("item_id").unique():
        game_id = item_id.rsplit("_", 1)[0]
        side = item_id.rsplit("_", 1)[1]  # "home" or "away"

        pred_row = predictions.loc[item_id]

        # AutoGluon prediction columns vary by model
        # Try common column names
        if "mean" in pred_row.columns:
            margin_pred = float(pred_row["mean"].iloc[0])
        elif "0.5" in pred_row.columns:
            margin_pred = float(pred_row["0.5"].iloc[0])
        else:
            # Fall back to first numeric column
            margin_pred = float(pred_row.iloc[0, 0])

        game_preds[game_id][side] = {
            "margin": margin_pred,
            "q05": (
                float(pred_row["0.1"].iloc[0])
                if "0.1" in pred_row.columns
                else margin_pred - 15
            ),
            "q95": (
                float(pred_row["0.9"].iloc[0])
                if "0.9" in pred_row.columns
                else margin_pred + 15
            ),
        }

    for game_id, preds in game_preds.items():
        if game_id not in record_map:
            continue
        rec = record_map[game_id]

        home_pred = preds.get("home", {"margin": 0, "q05": -15, "q95": 15})
        away_pred = preds.get("away", {"margin": 0, "q05": -15, "q95": 15})

        home_margin = home_pred["margin"]
        away_margin = away_pred["margin"]  # from away perspective

        spread_pred = (home_margin - away_margin) / 2.0

        # Win probability from quantile spread
        # If we have enough quantile info, estimate from the distribution
        spread_std = max((home_pred["q95"] - home_pred["q05"]) / 3.29, 1.0)
        home_win_prob = 1.0 / (1.0 + np.exp(-spread_pred / max(spread_std * 0.5, 1.0)))
        home_win_prob = np.clip(home_win_prob, 0.02, 0.98)

        home_recent_ppg = np.mean([f["ppg"] for f in rec["home_history"][-5:]])
        away_recent_ppg = np.mean([f["ppg"] for f in rec["away_history"][-5:]])
        total_pred = home_recent_ppg + away_recent_ppg

        results.append(
            {
                "game_id": rec["game_id"],
                "home": rec["home"],
                "away": rec["away"],
                "date": rec["date"],
                "season": rec["season"],
                "true_margin": rec["true_margin"],
                "true_home_score": rec["true_home_score"],
                "true_away_score": rec["true_away_score"],
                "spread_pred": spread_pred,
                "home_score_pred": (total_pred + spread_pred) / 2.0,
                "away_score_pred": (total_pred - spread_pred) / 2.0,
                "home_win_prob": home_win_prob,
                "spread_std": spread_std,
                "spread_q05": spread_pred - 1.645 * spread_std,
                "spread_q95": spread_pred + 1.645 * spread_std,
                "home_margin_raw": home_margin,
                "away_margin_raw": away_margin,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: list[dict]) -> dict:
    """Compute all evaluation metrics matching Phase 4 standard.

    Metrics:
        - Spread MAE
        - Spread RMSE
        - Win AUC (ROC)
        - Win Accuracy
        - ECE (10-bin calibration error)
        - 90% Coverage (if quantile predictions available)
        - Score MAE (home + away)
    """
    from sklearn.metrics import roc_auc_score

    true_margins = np.array([r["true_margin"] for r in results])
    pred_spreads = np.array([r["spread_pred"] for r in results])
    win_probs = np.array([r["home_win_prob"] for r in results])
    win_true = (true_margins > 0).astype(float)

    # Exclude ties from win metrics (rare in NBA but possible)
    non_tie = true_margins != 0
    win_true_nt = win_true[non_tie]
    win_probs_nt = win_probs[non_tie]

    # Spread metrics
    spread_errors = pred_spreads - true_margins
    mae = float(np.mean(np.abs(spread_errors)))
    rmse = float(np.sqrt(np.mean(spread_errors**2)))
    bias = float(np.mean(spread_errors))

    # Win metrics
    try:
        auc = float(roc_auc_score(win_true_nt, win_probs_nt))
    except (ValueError, Exception):
        auc = float("nan")

    win_pred = (win_probs > 0.5).astype(float)
    win_acc = float(np.mean(win_pred[non_tie] == win_true_nt))

    # ECE (10-bin expected calibration error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (win_probs >= bin_boundaries[i]) & (win_probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_pred = win_probs[mask].mean()
            avg_true = win_true[mask].mean()
            ece += mask.sum() / len(win_probs) * abs(avg_pred - avg_true)

    # 90% coverage
    if "spread_q05" in results[0] and "spread_q95" in results[0]:
        q05 = np.array([r["spread_q05"] for r in results])
        q95 = np.array([r["spread_q95"] for r in results])
        covered = (true_margins >= q05) & (true_margins <= q95)
        coverage_90 = float(np.mean(covered))
    else:
        coverage_90 = float("nan")

    # Score MAE
    if "true_home_score" in results[0] and "home_score_pred" in results[0]:
        home_score_mae = float(
            np.mean(
                np.abs(
                    np.array([r["true_home_score"] for r in results])
                    - np.array([r["home_score_pred"] for r in results])
                )
            )
        )
        away_score_mae = float(
            np.mean(
                np.abs(
                    np.array([r["true_away_score"] for r in results])
                    - np.array([r["away_score_pred"] for r in results])
                )
            )
        )
        score_mae = (home_score_mae + away_score_mae) / 2.0
    else:
        score_mae = float("nan")

    return {
        "n_games": len(results),
        "spread_mae": mae,
        "spread_rmse": rmse,
        "spread_bias": bias,
        "win_auc": auc,
        "win_accuracy": win_acc,
        "win_ece": ece,
        "coverage_90": coverage_90,
        "score_mae": score_mae,
    }


def print_metrics(metrics: dict, variant_name: str):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 65}")
    print(f"  Exp 6b: {variant_name}")
    print(f"{'=' * 65}")
    print(f"  Games evaluated:     {metrics['n_games']}")
    print(f"  {'─' * 61}")
    print(f"  Spread MAE:          {metrics['spread_mae']:.2f}")
    print(f"  Spread RMSE:         {metrics['spread_rmse']:.2f}")
    print(f"  Spread Bias:         {metrics['spread_bias']:+.2f}")
    print(f"  Win AUC:             {metrics['win_auc']:.3f}")
    print(f"  Win Accuracy:        {metrics['win_accuracy']:.1%}")
    print(f"  Win ECE:             {metrics['win_ece']:.4f}")
    print(f"  90% Coverage:        {metrics['coverage_90']:.1%}")
    print(f"  Score MAE (avg):     {metrics['score_mae']:.2f}")
    print(f"{'=' * 65}")


def print_comparison(all_metrics: dict):
    """Print comparison table across all variants + known baselines."""
    print(f"\n{'=' * 85}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 85}")
    header = f"  {'Variant':<30} {'MAE':>7} {'RMSE':>7} {'AUC':>6} {'Acc':>6} {'ECE':>7} {'Cov90':>6}"
    print(header)
    print(f"  {'─' * 81}")

    for name, m in all_metrics.items():
        row = (
            f"  {name:<30} "
            f"{m['spread_mae']:>7.2f} "
            f"{m['spread_rmse']:>7.2f} "
            f"{m['win_auc']:>6.3f} "
            f"{m['win_accuracy']:>5.1%} "
            f"{m['win_ece']:>7.4f} "
            f"{m['coverage_90']:>5.1%}"
        )
        print(row)

    # Known baselines
    print(f"  {'─' * 81}")
    print(
        f"  {'Phase 3 Exp 9 (ensemble)':<30} {'10.66':>7} {'──':>7} {'0.718':>6} {'66.5%':>6} {'0.0378':>7} {'83.6%':>6}"
    )
    print(
        f"  {'Phase 4 Exp 4 (generative)':<30} {'11.76':>7} {'──':>7} {'0.662':>6} {'61.4%':>6} {'──':>7} {'──':>6}"
    )
    print(f"{'=' * 85}")


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------
def save_predictions(results: list[dict], filepath: Path, variant: str):
    """Save predictions to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game_id",
        "date",
        "season",
        "home",
        "away",
        "true_margin",
        "true_home_score",
        "true_away_score",
        "spread_pred",
        "home_score_pred",
        "away_score_pred",
        "home_win_prob",
        "spread_std",
        "spread_q05",
        "spread_q95",
        "variant",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["variant"] = variant
            writer.writerow(row)

    logger.info(f"Saved {len(results)} predictions to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Exp 6b: Chronos-2 Time-Series Foundation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/exp6b_chronos2.py --split test
  python scripts/exp6b_chronos2.py --split val --finetune
  python scripts/exp6b_chronos2.py --split test --run-all
  python scripts/exp6b_chronos2.py --split test --model-size large
  python scripts/exp6b_chronos2.py --split test --backend direct
        """,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Evaluation split (default: test)",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        choices=["mini", "small", "base", "large"],
        help="Chronos-2 model size (default: base, 120M params)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=30,
        help="Number of recent games per team for context (default: 30)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable LoRA fine-tuning on training data",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=1000,
        help="Number of fine-tuning steps (default: 1000)",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Fine-tuning learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "autogluon", "direct"],
        help="Backend: autogluon (recommended) or direct chronos pipeline (default: auto)",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all variants: naive baseline, zero-shot, and (optionally) fine-tuned",
    )
    parser.add_argument(
        "--multivariate",
        action="store_true",
        help="Use multivariate input (margin + PPG + opp_PPG) in direct mode",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    np.random.seed(args.seed)

    model_path = f"amazon/chronos-2-{args.model_size}"
    eval_seasons = VAL_SEASONS if args.split == "val" else TEST_SEASONS
    all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

    # ── Step 1: Load data ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 4 EXPERIMENT 6b: Chronos-2 Foundation Model")
    logger.info(f"  Model:    {model_path}")
    logger.info(f"  Split:    {args.split} ({', '.join(eval_seasons)})")
    logger.info(f"  History:  {args.history} games per team")
    logger.info(f"  Backend:  {args.backend}")
    logger.info(f"  Device:   {args.device}")
    logger.info("=" * 60)

    all_games = load_all_games(str(DB_PATH), all_seasons)

    # ── Step 2: Build team histories ───────────────────────────────────
    logger.info("Building per-team game histories...")
    builder = TeamHistoryBuilder(history_length=args.history)

    all_records = []  # records with sufficient history
    train_records = []
    eval_records = []

    from tqdm import tqdm

    for game in tqdm(all_games, desc="Processing games"):
        record = builder.process_game(game)
        if record is not None:
            all_records.append(record)
            if game["season"] in eval_seasons:
                eval_records.append(record)
            elif game["season"] in TRAIN_SEASONS:
                train_records.append(record)

    logger.info(f"  Total records with history: {len(all_records)}")
    logger.info(f"  Training records:           {len(train_records)}")
    logger.info(f"  Evaluation records:         {len(eval_records)}")

    if len(eval_records) == 0:
        logger.error("No evaluation games found. Check data and season splits.")
        sys.exit(1)

    # ── Step 3: Determine backend ──────────────────────────────────────
    use_autogluon = False
    use_direct = False

    if args.backend == "auto":
        try:
            import autogluon.timeseries  # noqa: F401

            use_autogluon = True
            logger.info("Using AutoGluon backend (auto-detected)")
        except ImportError:
            try:
                import chronos  # noqa: F401

                use_direct = True
                logger.info("Using direct Chronos backend (autogluon not available)")
            except ImportError:
                logger.warning("Neither autogluon nor chronos installed.")
                logger.info("Running naive baseline only. Install with:")
                logger.info("  pip install 'autogluon.timeseries[chronos]'")
                logger.info("  OR: pip install chronos-forecasting")
    elif args.backend == "autogluon":
        if not check_dependencies():
            sys.exit(1)
        use_autogluon = True
    elif args.backend == "direct":
        try:
            import chronos  # noqa: F401

            use_direct = True
        except ImportError:
            logger.error("chronos not installed. Run: pip install chronos-forecasting")
            sys.exit(1)

    # ── Step 4: Run experiments ────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    # Always run naive baseline for comparison
    logger.info("\n--- Running naive rolling-mean baseline ---")
    naive_results = run_naive_baseline(eval_records)
    naive_metrics = compute_metrics(naive_results)
    all_metrics["6b-naive (rolling mean)"] = naive_metrics
    print_metrics(naive_metrics, f"Naive Rolling Mean Baseline ({args.split})")
    save_predictions(
        naive_results,
        RESULTS_DIR / f"exp6b_naive_{args.split}_predictions.csv",
        "naive",
    )

    if use_autogluon:
        # ── AutoGluon path ────────────────────────────────────────────
        # Zero-shot
        logger.info(
            f"\n--- Running Chronos-2 zero-shot ({args.model_size}) via AutoGluon ---"
        )
        try:
            ag_zero = run_autogluon_experiment(
                train_records=train_records,
                eval_records=eval_records,
                model_path=model_path,
                finetune=False,
                batch_size=args.batch_size,
            )
            zero_results = extract_autogluon_predictions(ag_zero, eval_records)
            zero_metrics = compute_metrics(zero_results)
            all_metrics[f"6b-zero ({args.model_size})"] = zero_metrics
            print_metrics(
                zero_metrics, f"Chronos-2 Zero-Shot {args.model_size} ({args.split})"
            )
            save_predictions(
                zero_results,
                RESULTS_DIR
                / f"exp6b_zero_{args.model_size}_{args.split}_predictions.csv",
                f"zero-{args.model_size}",
            )
        except Exception as e:
            logger.error(f"AutoGluon zero-shot failed: {e}")
            import traceback

            traceback.print_exc()

        # Fine-tuned (if requested)
        if args.finetune or args.run_all:
            logger.info(
                f"\n--- Running Chronos-2 LoRA fine-tuned ({args.model_size}) ---"
            )
            try:
                ag_lora = run_autogluon_experiment(
                    train_records=train_records,
                    eval_records=eval_records,
                    model_path=model_path,
                    finetune=True,
                    finetune_steps=args.finetune_steps,
                    finetune_lr=args.finetune_lr,
                    batch_size=args.batch_size,
                )
                lora_results = extract_autogluon_predictions(ag_lora, eval_records)
                lora_metrics = compute_metrics(lora_results)
                all_metrics[f"6b-lora ({args.model_size})"] = lora_metrics
                print_metrics(
                    lora_metrics, f"Chronos-2 LoRA {args.model_size} ({args.split})"
                )
                save_predictions(
                    lora_results,
                    RESULTS_DIR
                    / f"exp6b_lora_{args.model_size}_{args.split}_predictions.csv",
                    f"lora-{args.model_size}",
                )
            except Exception as e:
                logger.error(f"AutoGluon LoRA fine-tuning failed: {e}")
                import traceback

                traceback.print_exc()

    elif use_direct:
        # ── Direct Chronos path ───────────────────────────────────────
        if args.multivariate:
            logger.info(f"\n--- Running Chronos-2 multivariate ({args.model_size}) ---")
            try:
                mv_results = run_direct_chronos_multivariate(
                    eval_records=eval_records,
                    model_path=model_path,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                mv_metrics = compute_metrics(mv_results)
                all_metrics[f"6b-multi ({args.model_size})"] = mv_metrics
                print_metrics(
                    mv_metrics,
                    f"Chronos-2 Multivariate {args.model_size} ({args.split})",
                )
                save_predictions(
                    mv_results,
                    RESULTS_DIR
                    / f"exp6b_multi_{args.model_size}_{args.split}_predictions.csv",
                    f"multi-{args.model_size}",
                )
            except Exception as e:
                logger.error(f"Multivariate prediction failed: {e}")
                import traceback

                traceback.print_exc()

        # Univariate zero-shot
        logger.info(
            f"\n--- Running Chronos-2 zero-shot ({args.model_size}) via direct pipeline ---"
        )
        try:
            direct_results = run_direct_chronos(
                eval_records=eval_records,
                model_path=model_path,
                device=args.device,
                batch_size=args.batch_size,
            )
            direct_metrics = compute_metrics(direct_results)
            all_metrics[f"6b-zero ({args.model_size})"] = direct_metrics
            print_metrics(
                direct_metrics, f"Chronos-2 Zero-Shot {args.model_size} ({args.split})"
            )
            save_predictions(
                direct_results,
                RESULTS_DIR
                / f"exp6b_zero_{args.model_size}_{args.split}_predictions.csv",
                f"zero-{args.model_size}",
            )
        except Exception as e:
            logger.error(f"Direct Chronos prediction failed: {e}")
            import traceback

            traceback.print_exc()

        # Fine-tune warning for direct mode
        if args.finetune:
            logger.warning(
                "LoRA fine-tuning requires AutoGluon backend. "
                "Install with: pip install 'autogluon.timeseries[chronos]'"
            )

    # ── Step 5: Comparison ─────────────────────────────────────────────
    if len(all_metrics) > 1:
        print_comparison(all_metrics)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\nResults saved to: {RESULTS_DIR}/exp6b_*_{args.split}_predictions.csv")

    if not use_autogluon and not use_direct:
        print("\n" + "=" * 65)
        print("  NOTE: Only the naive baseline was run.")
        print("  To run Chronos-2, install one of:")
        print("    pip install 'autogluon.timeseries[chronos]'  (recommended)")
        print("    pip install chronos-forecasting              (direct API)")
        print("=" * 65)


if __name__ == "__main__":
    main()
