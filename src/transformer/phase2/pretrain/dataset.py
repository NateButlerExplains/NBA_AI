"""
Pre-training Dataset for Phase 3 Experiment 2.

Each sample = one team's full season (~82 games).
Random masking at 40% creates unique samples each epoch.
"""

import logging
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transformer.phase2.cache_builder import TEAM_TO_IDX, PerGameFeatures, load_cache

logger = logging.getLogger(__name__)

# Fixed NBA normalization constants (same as Phase2 dataset)
SCORE_MEAN = 108.0
SCORE_STD = 13.0
MARGIN_STD = 13.0
POINTS_NORM = 30.0


def _normalize_scores(team_score: float, opp_score: float) -> list[float]:
    """Normalize to [team, opponent, margin, total] — same as Phase 2."""
    return [
        (team_score - SCORE_MEAN) / SCORE_STD,
        (opp_score - SCORE_MEAN) / SCORE_STD,
        (team_score - opp_score) / MARGIN_STD,
        (team_score + opp_score - 216.0) / 20.0,
    ]


class PretrainDataset(Dataset):
    """
    Pre-training dataset: full-season game sequences with random masking.

    Each sample is one (team, season) pair. Random 40% of positions are masked.
    Target: predict team_score, opp_score, margin at masked positions.
    """

    def __init__(
        self,
        game_features: dict[str, PerGameFeatures],
        season_index: dict,
        player_id_map: dict[int, int],
        seasons: list[str],
        mask_ratio: float = 0.4,
        max_players_per_game: int = 15,
        min_games: int = 10,
    ):
        self.game_features = game_features
        self.season_index = season_index
        self.player_id_map = player_id_map
        self.mask_ratio = mask_ratio
        self.max_players = max_players_per_game

        # Enumerate valid (team, season) pairs
        self.samples = []
        for (team, season), game_ids in season_index.items():
            if season in seasons and len(game_ids) >= min_games:
                # Verify team is in TEAM_TO_IDX (filter out any non-NBA teams)
                if team in TEAM_TO_IDX:
                    self.samples.append((team, season, game_ids))

        logger.info(f"PretrainDataset: {len(self.samples)} team-season pairs "
                    f"from {len(seasons)} seasons")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        team, season, game_ids = self.samples[idx]
        n_games = len(game_ids)

        # Encode all games in chronological order
        scores = np.zeros((n_games, 4), dtype=np.float32)
        opponent_ids = np.zeros(n_games, dtype=np.int64)
        location = np.zeros(n_games, dtype=np.int64)
        player_ids = np.zeros((n_games, self.max_players), dtype=np.int64)
        player_points = np.zeros((n_games, self.max_players), dtype=np.float32)
        player_mask = np.ones((n_games, self.max_players), dtype=bool)  # True=padding

        # Target scores at all positions (will extract masked ones later)
        target_scores = np.zeros((n_games, 3), dtype=np.float32)

        # Compute days_between: days from each game to last game in season
        game_dates = []
        for game_id in game_ids:
            gf = self.game_features[game_id]
            game_dates.append(gf.game_date)

        # Parse dates and compute days from last game
        parsed_dates = []
        for d in game_dates:
            try:
                parsed_dates.append(datetime.strptime(d[:10], "%Y-%m-%d"))
            except (ValueError, TypeError):
                parsed_dates.append(None)

        last_date = parsed_dates[-1] if parsed_dates[-1] else parsed_dates[0]
        days_between = np.zeros(n_games, dtype=np.int64)
        for i, pd in enumerate(parsed_dates):
            if pd and last_date:
                days_between[i] = min((last_date - pd).days, 179)
            else:
                days_between[i] = 0

        for i, game_id in enumerate(game_ids):
            gf = self.game_features[game_id]

            # Team-relative encoding
            if gf.home_team == team:
                team_score, opp_score = gf.home_score, gf.away_score
                opponent_ids[i] = TEAM_TO_IDX.get(gf.away_team, 0)
                location[i] = 1  # home
                players = gf.home_player_points
            else:
                team_score, opp_score = gf.away_score, gf.home_score
                opponent_ids[i] = TEAM_TO_IDX.get(gf.home_team, 0)
                location[i] = 0  # away
                players = gf.away_player_points

            scores[i] = _normalize_scores(team_score, opp_score)

            # Raw target scores (normalized)
            target_scores[i, 0] = (team_score - SCORE_MEAN) / SCORE_STD
            target_scores[i, 1] = (opp_score - SCORE_MEAN) / SCORE_STD
            target_scores[i, 2] = (team_score - opp_score) / MARGIN_STD

            # Player contributions
            for j, (pid, pts) in enumerate(players[:self.max_players]):
                player_ids[i, j] = self.player_id_map.get(pid, 0)
                player_points[i, j] = pts / POINTS_NORM
                player_mask[i, j] = False

        # Random masking: select mask_ratio positions
        n_mask = max(1, int(n_games * self.mask_ratio))
        mask_indices = np.array(sorted(random.sample(range(n_games), n_mask)), dtype=np.int64)

        return {
            "scores": torch.from_numpy(scores),
            "opponent_ids": torch.from_numpy(opponent_ids),
            "location": torch.from_numpy(location),
            "player_ids": torch.from_numpy(player_ids),
            "player_points": torch.from_numpy(player_points),
            "player_mask": torch.from_numpy(player_mask),
            "days_between": torch.from_numpy(days_between),
            "mask_indices": torch.from_numpy(mask_indices),
            "target_scores": torch.from_numpy(target_scores),
            "n_games": n_games,
        }


def collate_pretrain(batch: list[dict]) -> dict:
    """
    Collate pre-training samples with variable-length padding.

    Pads: n_games, n_players, n_masked.
    """
    B = len(batch)
    max_games = max(s["n_games"] for s in batch)
    max_players = max(s["player_ids"].shape[1] for s in batch)
    max_masked = max(s["mask_indices"].shape[0] for s in batch)

    scores = torch.zeros(B, max_games, 4)
    opponent_ids = torch.zeros(B, max_games, dtype=torch.int64)
    location_t = torch.zeros(B, max_games, dtype=torch.int64)
    player_ids = torch.zeros(B, max_games, max_players, dtype=torch.int64)
    player_points = torch.zeros(B, max_games, max_players)
    player_mask = torch.ones(B, max_games, max_players, dtype=torch.bool)
    days_between = torch.zeros(B, max_games, dtype=torch.int64)
    game_mask = torch.ones(B, max_games, dtype=torch.bool)  # True=padding
    mask_indices = torch.zeros(B, max_masked, dtype=torch.int64)
    mask_padding = torch.ones(B, max_masked, dtype=torch.bool)  # True=padding
    target_scores = torch.zeros(B, max_games, 3)

    for i, s in enumerate(batch):
        ng = s["n_games"]
        np_ = s["player_ids"].shape[1]
        nm = s["mask_indices"].shape[0]

        scores[i, :ng] = s["scores"]
        opponent_ids[i, :ng] = s["opponent_ids"]
        location_t[i, :ng] = s["location"]
        player_ids[i, :ng, :np_] = s["player_ids"]
        player_points[i, :ng, :np_] = s["player_points"]
        player_mask[i, :ng, :np_] = s["player_mask"]
        days_between[i, :ng] = s["days_between"]
        game_mask[i, :ng] = False
        mask_indices[i, :nm] = s["mask_indices"]
        mask_padding[i, :nm] = False
        target_scores[i, :ng] = s["target_scores"]

    return {
        "scores": scores,
        "opponent_ids": opponent_ids,
        "location": location_t,
        "player_ids": player_ids,
        "player_points": player_points,
        "player_mask": player_mask,
        "days_between": days_between,
        "game_mask": game_mask,
        "mask_indices": mask_indices,
        "mask_padding": mask_padding,
        "target_scores": target_scores,
    }
