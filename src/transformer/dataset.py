"""
PyTorch Dataset for NBA Transformer Model.

This module bridges the SequenceBuilder (which fetches and tokenizes games from
the database) and PyTorch's training loop. It implements PyTorch's Dataset
interface, which is the standard way to feed data into a model during training.

Key concepts for deep learning newcomers:
  - A "dataset" tells PyTorch how many samples exist and how to get each one.
  - A "dataloader" wraps the dataset and handles batching, shuffling, and
    parallel loading -- you typically don't write that yourself.
  - A "collate function" takes a list of individual samples and merges them
    into a single batch. This is where variable-length padding happens.

Usage:
    tokenizer = PBPTokenizer()
    tokenizer.load("data/tokenized/tokenizer.json")

    dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=["2022-2023", "2023-2024"],
        split="train",
        n_history_games=5,
    )

    # collate_games is a custom collate function that handles padding
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_games)
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transformer.sequence_builder import SequenceBuilder, sequence_to_arrays
from src.transformer.tokenizer import PBPTokenizer


class NBAGameDataset(Dataset):
    """
    PyTorch Dataset for NBA game prediction.

    Extends PyTorch's Dataset class, which is the standard interface for providing
    data to a training loop. PyTorch requires two methods:
      - __len__():      returns how many samples are in the dataset
      - __getitem__(i): returns the i-th sample as tensors

    PyTorch's DataLoader calls these methods internally to build batches:
    it picks random (or sequential) indices, calls __getitem__ for each,
    and then uses a "collate function" to merge them into a batch.
    """

    # Default train/val/test split configuration.
    # Data is split by SEASON (not randomly) to prevent temporal leakage --
    # the model never sees future seasons during training.
    #
    # NOTE: These represent the FULL dataset available across all 23+ seasons.
    # For actual experiments, the trainer passes explicit seasons from the YAML config,
    # which overrides these defaults. See configs/transformer/full_baseline.yaml.
    #
    # Phase 1a baseline uses 5 recent seasons (2018-2023) for faster iteration.
    # Future ablations will scale up to 10, 15, or all 23 seasons.
    SPLIT_CONFIG = {
        "train": {
            # ~23 seasons of historical data for training
            "seasons": [
                "2000-2001", "2001-2002", "2002-2003", "2003-2004", "2004-2005",
                "2005-2006", "2006-2007", "2007-2008", "2008-2009", "2009-2010",
                "2010-2011", "2011-2012", "2012-2013", "2013-2014", "2014-2015",
                "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
                "2020-2021", "2021-2022", "2022-2023",
            ],
        },
        "val": {
            # 1 season held out for validation (tuning hyperparameters)
            "seasons": ["2023-2024"],
        },
        "test": {
            # Most recent seasons held out for final evaluation
            "seasons": ["2024-2025", "2025-2026"],
        },
    }

    def __init__(
        self,
        tokenizer: PBPTokenizer,
        seasons: Optional[list[str]] = None,
        split: Optional[str] = None,
        n_history_games: int = 5,
        min_history_games: int = 3,
        db_path: Optional[str] = None,
        cache_size: int = 1000,
    ):
        """
        Initialize dataset.

        Args:
            tokenizer: Initialized PBPTokenizer
            seasons: List of seasons to include. If None, uses split config.
            split: One of "train", "val", "test". Used to determine seasons.
            n_history_games: Number of historical games per team (context window size)
            min_history_games: Minimum history required for a valid sample
            db_path: Database path. If None, uses config default.
            cache_size: Max number of sequences to keep in the LRU cache
        """
        # Must provide either explicit seasons or a named split
        if seasons is None and split is None:
            raise ValueError("Must specify either 'seasons' or 'split'")

        # If no explicit seasons, look them up from the split configuration
        if seasons is None:
            if split not in self.SPLIT_CONFIG:
                raise ValueError(f"Unknown split: {split}")
            seasons = self.SPLIT_CONFIG[split]["seasons"]

        self.tokenizer = tokenizer
        self.seasons = seasons
        self.split = split
        self.n_history_games = n_history_games
        self.min_history_games = min_history_games

        if db_path is None:
            from src.config import config
            db_path = config["database"]["path"]

        self.db_path = db_path

        # Create the SequenceBuilder that knows how to fetch and tokenize games
        self.builder = SequenceBuilder(
            tokenizer=tokenizer,
            n_games=n_history_games,
            db_path=db_path,
        )

        # Pre-compute the list of valid game IDs for this dataset.
        # This filters out early-season games where teams lack enough history.
        # The list is stored so __len__ and __getitem__ can index into it.
        logging.info(f"Loading game IDs for seasons: {seasons}")
        self.game_ids = self.builder.get_training_game_ids(
            seasons=seasons,
            min_history_games=min_history_games,
        )
        logging.info(f"Found {len(self.game_ids)} valid games")

        # Setup LRU (Least Recently Used) cache for sequence data.
        # LRU cache keeps the most recently accessed `cache_size` sequences in
        # memory. When the cache is full and a new sequence is requested, the
        # least recently used one is evicted. This avoids redundant database
        # queries when the DataLoader re-accesses the same games across epochs.
        # We wrap the uncached method with functools.lru_cache at runtime.
        self._get_sequence_cached = lru_cache(maxsize=cache_size)(
            self._get_sequence_uncached
        )

    def _get_sequence_uncached(self, game_id: str) -> Optional[dict]:
        """
        Get sequence arrays for a game (uncached version).

        This is the "expensive" version that actually hits the database.
        It's wrapped by lru_cache in __init__ so repeated calls with the
        same game_id return the cached result instead of re-querying.
        """
        # Build the full MatchupSequence (fetches history from DB, tokenizes plays)
        sequence = self.builder.build_sequence(game_id)
        if sequence is None:
            return None

        # Double-check that both teams have enough historical games.
        # Even though get_training_game_ids pre-filters, some games might fail
        # to tokenize, leaving fewer valid history games than expected.
        if (
            len(sequence.home_history.games) < self.min_history_games
            or len(sequence.away_history.games) < self.min_history_games
        ):
            return None

        # Convert from Python objects to numpy arrays (with padding)
        return sequence_to_arrays(
            sequence,
            self.tokenizer,
        )

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        PyTorch's DataLoader calls this to know how many batches to create.
        Each "sample" is one game we want to predict.
        """
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single training sample by index.

        PyTorch's DataLoader calls this method repeatedly to build batches.
        For example, with batch_size=32, the DataLoader picks 32 random indices,
        calls __getitem__ for each, then passes the 32 results to collate_games()
        to merge them into one batch.

        Returns dict with:
            - home_history: dict of tensors for home team's prior games (MODEL INPUT)
            - away_history: dict of tensors for away team's prior games (MODEL INPUT)
            - home_roster: tensor of player IDs (MODEL INPUT)
            - away_roster: tensor of player IDs (MODEL INPUT)
            - target_home_score: float (LABEL -- what model predicts)
            - target_away_score: float (LABEL -- what model predicts)
            - game_id: str (metadata for tracking/debugging)
        """
        game_id = self.game_ids[idx]
        # Try the LRU-cached version first (fast path if already computed)
        arrays = self._get_sequence_cached(game_id)

        if arrays is None:
            # Sequence building failed (e.g., missing data in DB).
            # Return a placeholder of zeros so the batch doesn't crash.
            # The collate function will filter these out later.
            logging.warning(f"Failed to build sequence for game {game_id}")
            return self._empty_sample(game_id)

        return self._arrays_to_tensors(arrays)

    def _arrays_to_tensors(self, arrays: dict) -> dict:
        """
        Convert numpy arrays to PyTorch tensors.

        PyTorch models operate on torch.Tensor objects, not numpy arrays.
        This method converts all numeric data from numpy to PyTorch format.
        String metadata (game_id, team names) stays as plain Python strings.
        """
        result = {
            # Metadata (strings, not tensors -- used for logging/debugging)
            "game_id": arrays["target_game_id"],
            "target_date": arrays["target_date"],
            "home_team": arrays["home_team"],
            "away_team": arrays["away_team"],
            # Labels/targets: the scores the model tries to predict.
            # Stored as float32 because the loss function expects floats.
            # Unplayed games default to 0.0 (shouldn't happen during training).
            "target_home_score": torch.tensor(
                arrays["home_score"] if arrays["home_score"] is not None else 0.0,
                dtype=torch.float32,
            ),
            "target_away_score": torch.tensor(
                arrays["away_score"] if arrays["away_score"] is not None else 0.0,
                dtype=torch.float32,
            ),
            # Final scores (including OT) for evaluation comparison.
            # These differ from training targets for OT games.
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

        # Convert history arrays (each is a dict of numpy arrays -> dict of tensors)
        for prefix in ["home_history", "away_history"]:
            hist = arrays[prefix]
            if hist is not None:
                result[prefix] = {
                    key: torch.from_numpy(arr)
                    for key, arr in hist.items()
                }
            else:
                result[prefix] = None

        # Convert roster arrays (1D arrays of player token IDs)
        result["home_roster"] = torch.from_numpy(arrays["home_roster"])
        result["away_roster"] = torch.from_numpy(arrays["away_roster"])

        return result

    def _empty_sample(self, game_id: str) -> dict:
        """
        Return a placeholder sample filled with zeros for failed sequence builds.

        When sequence building fails (e.g., missing play-by-play data in the DB),
        we can't just skip the sample -- PyTorch's DataLoader expects __getitem__
        to always return something. Instead, we return this zero-filled placeholder.
        The collate_games() function will detect these (home_history is None) and
        filter them out before building the batch, so they never reach the model.
        """
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
            "home_history": None,   # None signals this is a failed sample
            "away_history": None,
            "home_roster": torch.zeros(0, dtype=torch.int64),
            "away_roster": torch.zeros(0, dtype=torch.int64),
        }

    def clear_cache(self):
        """Clear the sequence cache."""
        self._get_sequence_cached.cache_clear()
        self.builder.clear_cache()

    @property
    def vocab_sizes(self) -> dict:
        """Get vocabulary sizes for model configuration."""
        return self.tokenizer.vocab_sizes


# ---------------------------------------------------------------------------
# Custom collate function for batching
# ---------------------------------------------------------------------------

def collate_games(batch: list[dict]) -> dict:
    """
    Custom collate function for batching variable-length game data.

    WHY IS THIS NEEDED?
    PyTorch's default collate function assumes all samples have the same shape,
    but our samples have variable lengths in TWO dimensions:
      1. Number of historical games (some teams may have fewer if games failed
         to tokenize)
      2. Number of plays per game (games vary from ~250 to ~500+ plays)

    This function PADS all samples in the batch with zeros so they match the
    longest sample, then stacks them into uniform tensors. The model uses
    attention masking (based on game_lengths) to ignore the padded positions.

    It also filters out any failed samples (where home_history is None) before
    padding, so one bad sample doesn't crash the entire batch.

    Args:
        batch: List of sample dicts from NBAGameDataset.__getitem__

    Returns:
        Single dict of batched tensors with padding, or None if all samples failed
    """
    # STEP 1: Remove failed samples (those where sequence building returned None).
    # These are the placeholder samples from _empty_sample().
    batch = [b for b in batch if b["home_history"] is not None]

    if not batch:
        # Every sample in this batch failed -- return None so the training
        # loop can skip this batch entirely.
        return None

    # STEP 2: Collate simple scalar values.
    # Strings go into plain lists; score tensors get stacked into a 1D tensor.
    result = {
        "game_ids": [b["game_id"] for b in batch],
        "target_dates": [b["target_date"] for b in batch],
        "home_teams": [b["home_team"] for b in batch],
        "away_teams": [b["away_team"] for b in batch],
        # torch.stack combines individual scalar tensors into a 1D batch tensor
        # e.g., [tensor(110), tensor(98), tensor(105)] -> tensor([110, 98, 105])
        "target_home_scores": torch.stack([b["target_home_score"] for b in batch]),
        "target_away_scores": torch.stack([b["target_away_score"] for b in batch]),
        # Final scores (including OT) for evaluation comparison
        "final_home_scores": torch.stack([b["final_home_score"] for b in batch]),
        "final_away_scores": torch.stack([b["final_away_score"] for b in batch]),
        "is_overtime": [b["is_overtime"] for b in batch],
    }

    # STEP 3: Collate history tensors (the most complex part).
    # Each sample has a history dict with shape (n_games, n_plays).
    # We need to pad BOTH dimensions across the batch:
    #   - Pad n_games: if one sample has 5 history games and another has 4,
    #     pad the shorter one with a row of zeros.
    #   - Pad n_plays: if one game has 300 plays and another has 410,
    #     pad the shorter one with trailing zeros.
    for prefix in ["home_history", "away_history"]:
        histories = [b[prefix] for b in batch]

        # Find the maximum dimensions across all samples in this batch.
        # All samples will be padded to these dimensions.
        max_games = max(h["action_type_ids"].shape[0] for h in histories)
        max_plays = max(h["action_type_ids"].shape[1] for h in histories)

        collated = {}
        for key in histories[0].keys():
            if histories[0][key].dim() == 1:
                # 1D arrays (one value per game): game_lengths, days_before_target,
                # season_game_number. Only needs padding in the games dimension.
                padded = []
                for h in histories:
                    arr = h[key]
                    if len(arr) < max_games:
                        # Pad with zeros for "missing" games
                        padding = torch.zeros(max_games - len(arr), dtype=arr.dtype)
                        arr = torch.cat([arr, padding])
                    padded.append(arr)
                collated[key] = torch.stack(padded)
            else:
                # 2D tensors (n_games, n_plays) need padding in BOTH dimensions.
                padded = []
                for h in histories:
                    arr = h[key]
                    n_games, n_plays = arr.shape

                    # First, pad the plays dimension (add columns of zeros on the right)
                    if n_plays < max_plays:
                        play_padding = torch.zeros(
                            (n_games, max_plays - n_plays), dtype=arr.dtype
                        )
                        arr = torch.cat([arr, play_padding], dim=1)

                    # Then, pad the games dimension (add rows of zeros at the bottom)
                    if n_games < max_games:
                        game_padding = torch.zeros(
                            (max_games - n_games, max_plays), dtype=arr.dtype
                        )
                        arr = torch.cat([arr, game_padding], dim=0)

                    padded.append(arr)

                # Stack all samples into a 3D tensor: (batch_size, max_games, max_plays)
                collated[key] = torch.stack(padded)

        result[prefix] = collated

    # STEP 4: Collate rosters (pad to the largest roster in the batch).
    # Teams have different numbers of players, so shorter rosters get zero-padded.
    for prefix in ["home_roster", "away_roster"]:
        rosters = [b[prefix] for b in batch]
        max_roster = max(len(r) for r in rosters)

        padded = []
        for r in rosters:
            if len(r) < max_roster:
                padding = torch.zeros(max_roster - len(r), dtype=r.dtype)
                r = torch.cat([r, padding])
            padded.append(r)

        # Stack into shape: (batch_size, max_roster_size)
        result[prefix] = torch.stack(padded)

    return result


def test_dataset():
    """Test dataset functionality."""
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.INFO)

    # Load tokenizer
    tokenizer_path = "data/tokenized/test_tokenizer.json"
    tokenizer = PBPTokenizer()

    try:
        tokenizer.load(tokenizer_path)
        print("Loaded existing tokenizer")
    except FileNotFoundError:
        print("Building tokenizer...")
        tokenizer.build_vocab_from_db(seasons=["2023-2024"], min_count=5)
        tokenizer.save(tokenizer_path)

    # Create dataset for a single season (smaller for testing)
    print("\nCreating dataset for 2023-2024 season...")
    dataset = NBAGameDataset(
        tokenizer=tokenizer,
        seasons=["2023-2024"],
        n_history_games=5,
        min_history_games=3,
        cache_size=100,
    )

    print(f"Dataset size: {len(dataset)} games")
    print(f"Vocab sizes: {dataset.vocab_sizes}")

    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    print(f"  Game ID: {sample['game_id']}")
    print(f"  Target: {sample['home_team']} vs {sample['away_team']}")
    print(f"  Score: {sample['target_home_score'].item():.0f} - {sample['target_away_score'].item():.0f}")
    print(f"  Home history shape: {sample['home_history']['action_type_ids'].shape}")
    print(f"  Away history shape: {sample['away_history']['action_type_ids'].shape}")
    print(f"  Home roster size: {len(sample['home_roster'])}")
    print(f"  Away roster size: {len(sample['away_roster'])}")

    # Test DataLoader with batching
    print("\nTesting DataLoader with batch_size=4...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_games,
        num_workers=0,  # Use 0 for testing
    )

    batch = next(iter(dataloader))
    print(f"  Batch game_ids: {batch['game_ids']}")
    print(f"  Target scores shape: {batch['target_home_scores'].shape}")
    print(f"  Home history action_type_ids shape: {batch['home_history']['action_type_ids'].shape}")
    print(f"  Away history action_type_ids shape: {batch['away_history']['action_type_ids'].shape}")
    print(f"  Home roster shape: {batch['home_roster'].shape}")
    print(f"  Away roster shape: {batch['away_roster'].shape}")

    # Profile memory usage
    print("\nMemory usage:")
    import sys
    for key, val in batch.items():
        if isinstance(val, dict):
            total = sum(v.element_size() * v.nelement() for v in val.values())
            print(f"  {key}: {total / 1024:.1f} KB")
        elif isinstance(val, torch.Tensor):
            print(f"  {key}: {val.element_size() * val.nelement() / 1024:.1f} KB")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(test_dataset())
