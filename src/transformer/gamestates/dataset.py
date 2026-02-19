"""
PyTorch Dataset for GameStates Model.

Same structure as NBAGameDataset but uses GameStateSequenceBuilder
to fetch score trajectory data instead of play-by-play.

Usage:
    tokenizer = GameStateTokenizer()
    dataset = GameStatesDataset(tokenizer=tokenizer, seasons=["2022-2023"])
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_gamestates)
"""

import logging
from functools import lru_cache
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.transformer.gamestates.tokenizer import GameStateTokenizer
from src.transformer.gamestates.sequence_builder import (
    GameStateSequenceBuilder,
    sequence_to_arrays,
)

# Feature keys in the history dicts (used for collation)
FEATURE_KEYS = [
    "periods", "clock_buckets", "home_score_buckets",
    "away_score_buckets", "margin_buckets",
]


class GameStatesDataset(Dataset):
    """PyTorch Dataset for NBA game prediction using GameStates score trajectories."""

    def __init__(
        self,
        tokenizer: GameStateTokenizer,
        seasons: list[str],
        n_history_games: int = 5,
        min_history_games: int = 3,
        db_path: Optional[str] = None,
        cache_size: int = 2000,
    ):
        self.tokenizer = tokenizer
        self.seasons = seasons
        self.n_history_games = n_history_games
        self.min_history_games = min_history_games

        if db_path is None:
            from src.config import config
            db_path = config["database"]["path"]
        self.db_path = db_path

        self.builder = GameStateSequenceBuilder(
            tokenizer=tokenizer,
            n_games=n_history_games,
            db_path=db_path,
        )

        logging.info(f"Loading game IDs for seasons: {seasons}")
        self.game_ids = self.builder.get_training_game_ids(
            seasons=seasons,
            min_history_games=min_history_games,
        )
        logging.info(f"Found {len(self.game_ids)} valid games")

        self._get_sequence_cached = lru_cache(maxsize=cache_size)(
            self._get_sequence_uncached
        )

    def _get_sequence_uncached(self, game_id: str) -> Optional[dict]:
        """Get sequence arrays for a game (uncached)."""
        sequence = self.builder.build_sequence(game_id)
        if sequence is None:
            return None

        if (
            len(sequence.home_history.game_data) < self.min_history_games
            or len(sequence.away_history.game_data) < self.min_history_games
        ):
            return None

        return sequence_to_arrays(sequence)

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> dict:
        game_id = self.game_ids[idx]
        arrays = self._get_sequence_cached(game_id)

        if arrays is None:
            logging.warning(f"Failed to build sequence for game {game_id}")
            return self._empty_sample(game_id)

        return self._arrays_to_tensors(arrays)

    def _arrays_to_tensors(self, arrays: dict) -> dict:
        """Convert numpy arrays to PyTorch tensors."""
        result = {
            "game_id": arrays["target_game_id"],
            "target_date": arrays["target_date"],
            "home_team": arrays["home_team"],
            "away_team": arrays["away_team"],
            "target_home_score": torch.tensor(
                arrays["home_score"] if arrays["home_score"] is not None else 0.0,
                dtype=torch.float32,
            ),
            "target_away_score": torch.tensor(
                arrays["away_score"] if arrays["away_score"] is not None else 0.0,
                dtype=torch.float32,
            ),
            "final_home_score": torch.tensor(
                arrays["final_home_score"] if arrays.get("final_home_score") is not None else 0.0,
                dtype=torch.float32,
            ),
            "final_away_score": torch.tensor(
                arrays["final_away_score"] if arrays.get("final_away_score") is not None else 0.0,
                dtype=torch.float32,
            ),
            "is_overtime": arrays.get("is_overtime", False),
        }

        for prefix in ["home_history", "away_history"]:
            hist = arrays[prefix]
            if hist is not None:
                result[prefix] = {
                    key: torch.from_numpy(arr)
                    for key, arr in hist.items()
                }
            else:
                result[prefix] = None

        return result

    def _empty_sample(self, game_id: str) -> dict:
        """Return a placeholder sample for failed sequence builds."""
        return {
            "game_id": game_id,
            "target_date": "",
            "home_team": "",
            "away_team": "",
            "target_home_score": torch.tensor(0.0),
            "target_away_score": torch.tensor(0.0),
            "final_home_score": torch.tensor(0.0),
            "final_away_score": torch.tensor(0.0),
            "is_overtime": False,
            "home_history": None,
            "away_history": None,
        }

    def clear_cache(self):
        self._get_sequence_cached.cache_clear()
        self.builder.clear_cache()


def collate_gamestates(batch: list[dict]) -> Optional[dict]:
    """
    Custom collate function for GameStates batches.

    Pads variable-length histories in both games and rows dimensions.
    """
    batch = [b for b in batch if b["home_history"] is not None]

    if not batch:
        return None

    result = {
        "game_ids": [b["game_id"] for b in batch],
        "target_dates": [b["target_date"] for b in batch],
        "home_teams": [b["home_team"] for b in batch],
        "away_teams": [b["away_team"] for b in batch],
        "target_home_scores": torch.stack([b["target_home_score"] for b in batch]),
        "target_away_scores": torch.stack([b["target_away_score"] for b in batch]),
        "final_home_scores": torch.stack([b["final_home_score"] for b in batch]),
        "final_away_scores": torch.stack([b["final_away_score"] for b in batch]),
        "is_overtime": [b["is_overtime"] for b in batch],
    }

    for prefix in ["home_history", "away_history"]:
        histories = [b[prefix] for b in batch]

        # Find max dimensions across the batch
        max_games = max(h["periods"].shape[0] for h in histories)
        max_rows = max(h["periods"].shape[1] for h in histories)

        collated = {}
        for key in histories[0].keys():
            if key == "game_lengths":
                padded = []
                for h in histories:
                    arr = h[key]
                    if len(arr) < max_games:
                        padding = torch.zeros(max_games - len(arr), dtype=arr.dtype)
                        arr = torch.cat([arr, padding])
                    padded.append(arr)
                collated[key] = torch.stack(padded)
            else:
                padded = []
                for h in histories:
                    arr = h[key]
                    n_games, n_rows = arr.shape

                    if n_rows < max_rows:
                        row_padding = torch.zeros(
                            (n_games, max_rows - n_rows), dtype=arr.dtype
                        )
                        arr = torch.cat([arr, row_padding], dim=1)

                    if n_games < max_games:
                        game_padding = torch.zeros(
                            (max_games - n_games, max_rows), dtype=arr.dtype
                        )
                        arr = torch.cat([arr, game_padding], dim=0)

                    padded.append(arr)

                collated[key] = torch.stack(padded)

        result[prefix] = collated

    return result
