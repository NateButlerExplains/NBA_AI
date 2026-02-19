"""
GameState Tokenizer for NBA Score Trajectory Data.

Converts raw GameStates rows (period, clock, scores) into discrete tokens.
Unlike the PBP tokenizer, this has fixed vocabularies — no build step needed.

Features per row (5 total):
    1. period          - Which quarter (1-4, capped at regulation)
    2. clock_bucket    - Seconds remaining in period (0-720)
    3. home_score_bucket - Home score in 5-point bins (0-50)
    4. away_score_bucket - Away score in 5-point bins (0-50)
    5. margin_bucket     - home_margin clamped to [-60, +60] → bucket 0-120

Usage:
    tokenizer = GameStateTokenizer()
    tensors = tokenizer.tokenize_game(game_id)  # dict of 5 numpy arrays + length
"""

import logging
import re
from typing import Optional

import numpy as np

from src.database import get_db

# Fixed vocabulary sizes — no build step needed.
VOCAB_SIZES = {
    "period": 5,             # 0=padding, 1-4=regulation quarters
    "clock_bucket": 721,     # 0-720 seconds (12 minutes * 60)
    "home_score_bucket": 51, # 0-50 (5-point bins: 0, 1-5, 6-10, ..., 246-250)
    "away_score_bucket": 51, # Same as home
    "margin_bucket": 121,    # -60 to +60 → buckets 0-120
}


class GameStateTokenizer:
    """
    Tokenizer for GameStates table rows.

    Converts each GameStates row into 5 integer tokens representing
    the score trajectory at that point in the game. Fixed vocabularies
    mean no vocabulary build step is needed (unlike PBP tokenizer).
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            from src.config import config
            db_path = config["database"]["path"]
        self.db_path = db_path

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return VOCAB_SIZES.copy()

    @staticmethod
    def _clock_to_seconds(clock_str: str) -> int:
        """Convert NBA clock string (PT{M}M{S.SS}S) to seconds remaining."""
        if not clock_str:
            return 0
        try:
            match = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
            if match:
                minutes = int(match.group(1))
                seconds = int(float(match.group(2)))
                total = minutes * 60 + seconds
                return min(total, 720)
            return 0
        except Exception:
            return 0

    @staticmethod
    def _score_to_bucket(score: int) -> int:
        """Convert a score to a 5-point bin. 0→0, 1-5→1, 6-10→2, ..., 246-250→50."""
        if score <= 0:
            return 0
        bucket = (score - 1) // 5 + 1
        return min(bucket, 50)

    @staticmethod
    def _margin_to_bucket(margin: int) -> int:
        """Convert home_margin to bucket 0-120. margin=-60→0, 0→60, +60→120."""
        clamped = max(-60, min(60, margin))
        return clamped + 60  # shift so -60 → 0

    def tokenize_game(self, game_id: str) -> Optional[dict]:
        """
        Tokenize a game's score trajectory from the GameStates table.

        Queries GameStates for regulation plays only (period <= 4), converts
        each row to 5 integer tokens.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with 5 numpy arrays (one per feature) + 'length', or None if no data.
        """
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT period, clock, home_score, away_score, home_margin
                FROM GameStates
                WHERE game_id = ? AND period <= 4
                ORDER BY play_id
                """,
                (game_id,),
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        periods = []
        clock_buckets = []
        home_score_buckets = []
        away_score_buckets = []
        margin_buckets = []

        for period, clock, home_score, away_score, home_margin in rows:
            periods.append(min(max(period, 1), 4))
            clock_buckets.append(self._clock_to_seconds(clock))
            home_score_buckets.append(self._score_to_bucket(home_score))
            away_score_buckets.append(self._score_to_bucket(away_score))
            margin_buckets.append(self._margin_to_bucket(home_margin))

        return {
            "periods": np.array(periods, dtype=np.int64),
            "clock_buckets": np.array(clock_buckets, dtype=np.int64),
            "home_score_buckets": np.array(home_score_buckets, dtype=np.int64),
            "away_score_buckets": np.array(away_score_buckets, dtype=np.int64),
            "margin_buckets": np.array(margin_buckets, dtype=np.int64),
            "length": len(rows),
        }
