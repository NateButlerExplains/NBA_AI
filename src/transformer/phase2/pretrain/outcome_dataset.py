"""
Per-Match Outcome Pre-training Dataset (Phase 3 Experiment 6).

Each sample = 1 game from one team's perspective (home or away).
~62K samples from 31K games x 2 perspectives.

Target: win/loss (BCE) + normalized margin (MSE).
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transformer.phase2.cache_builder import TEAM_TO_IDX, PerGameFeatures
from src.transformer.phase2.dataset import SCORE_MEAN, SCORE_STD, MARGIN_STD, TOTAL_MEAN, TOTAL_STD
from src.transformer.phase2.dataset import POINTS_NORM, STAT_NORMS

logger = logging.getLogger(__name__)


class OutcomePretrainDataset(Dataset):
    """
    Dataset for per-match outcome pre-training.

    Each sample contains a single game from one team's perspective:
    - Player stats (with scores masked during forward pass)
    - Target: win (0/1) and normalized margin
    """

    def __init__(
        self,
        game_features: dict[str, PerGameFeatures],
        player_id_map: dict[int, int],
        seasons: list[str],
        max_players_per_game: int = 15,
        n_player_stats: int = 16,
    ):
        self.game_features = game_features
        self.player_id_map = player_id_map
        self.max_players = max_players_per_game
        self.n_player_stats = n_player_stats

        # Build sample list: (game_id, is_home) for each perspective
        self.samples = []
        season_set = set(seasons)
        for gid, gf in game_features.items():
            if gf.season not in season_set:
                continue
            # Only include games with player data
            has_home = bool(gf.home_player_stats) if n_player_stats > 0 else bool(gf.home_player_points)
            has_away = bool(gf.away_player_stats) if n_player_stats > 0 else bool(gf.away_player_points)
            if has_home:
                self.samples.append((gid, True))
            if has_away:
                self.samples.append((gid, False))

        logger.info(f"OutcomePretrainDataset: {len(self.samples)} samples from "
                     f"{len(season_set)} seasons")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        game_id, is_home = self.samples[idx]
        gf = self.game_features[game_id]
        P = self.max_players
        use_stats = self.n_player_stats > 0

        # Determine team perspective
        if is_home:
            team_score, opp_score = gf.home_score, gf.away_score
            opponent = gf.away_team
            loc = 1  # home
            players_stats = gf.home_player_stats
            players_points = gf.home_player_points
        else:
            team_score, opp_score = gf.away_score, gf.home_score
            opponent = gf.home_team
            loc = 0  # away
            players_stats = gf.away_player_stats
            players_points = gf.away_player_points

        # Score features (will be masked in model, but included for shape)
        scores = np.array([
            (team_score - SCORE_MEAN) / SCORE_STD,
            (opp_score - SCORE_MEAN) / SCORE_STD,
            (team_score - opp_score) / MARGIN_STD,
            (team_score + opp_score - TOTAL_MEAN) / TOTAL_STD,
        ], dtype=np.float32)

        # Opponent and location
        opponent_id = TEAM_TO_IDX.get(opponent, 0)

        # Player features
        player_ids = np.zeros(P, dtype=np.int64)
        player_points_arr = np.zeros(P, dtype=np.float32)
        player_mask = np.ones(P, dtype=bool)  # True=padding

        result = {
            "scores": torch.from_numpy(scores).unsqueeze(0),  # (1, 4)
            "opponent_ids": torch.tensor([opponent_id], dtype=torch.int64),  # (1,)
            "location": torch.tensor([loc], dtype=torch.int64),  # (1,)
        }

        if use_stats and players_stats:
            player_stats_arr = np.zeros((P, self.n_player_stats), dtype=np.float32)
            player_positions = np.full(P, 3, dtype=np.int64)
            player_pm_available = np.zeros(P, dtype=np.float32)

            for j, (pid, stats, pos_idx, pm_avail) in enumerate(players_stats[:P]):
                player_ids[j] = self.player_id_map.get(pid, 0)
                normalized = np.array(stats, dtype=np.float32) / STAT_NORMS
                normalized[15] = 0.0  # Zero out plus_minus — it leaks the outcome
                player_stats_arr[j] = normalized
                player_points_arr[j] = stats[1] / POINTS_NORM
                player_positions[j] = pos_idx
                player_pm_available[j] = 0.0  # Always mask pm_available flag too
                player_mask[j] = False

            result["player_stats"] = torch.from_numpy(player_stats_arr).unsqueeze(0)  # (1, P, 16)
            result["player_positions"] = torch.from_numpy(player_positions).unsqueeze(0)  # (1, P)
            result["player_pm_available"] = torch.from_numpy(player_pm_available).unsqueeze(0)  # (1, P)
        else:
            for j, (pid, pts) in enumerate(players_points[:P]):
                player_ids[j] = self.player_id_map.get(pid, 0)
                player_points_arr[j] = pts / POINTS_NORM
                player_mask[j] = False

        result["player_ids"] = torch.from_numpy(player_ids).unsqueeze(0)  # (1, P)
        result["player_points"] = torch.from_numpy(player_points_arr).unsqueeze(0)  # (1, P)
        result["player_mask"] = torch.from_numpy(player_mask).unsqueeze(0)  # (1, P)

        # Targets
        margin = team_score - opp_score
        result["target_win"] = torch.tensor(1.0 if margin > 0 else 0.0, dtype=torch.float32)
        result["target_margin"] = torch.tensor(margin / MARGIN_STD, dtype=torch.float32)

        return result


def collate_outcome(batch: list[dict]) -> dict:
    """Collate outcome pre-training samples. All have fixed shapes, just stack."""
    result = {}
    keys = batch[0].keys()

    for key in keys:
        tensors = [s[key] for s in batch]
        if tensors[0].dim() == 0:
            result[key] = torch.stack(tensors)
        else:
            # Pad player dimension to max across batch
            max_p = max(t.shape[-1] for t in tensors)
            if all(t.shape == tensors[0].shape for t in tensors):
                result[key] = torch.stack(tensors)
            else:
                result[key] = torch.stack(tensors)

    return result
