"""
Phase 2 Dataset and Collate Function.

Constructs training samples from cached features with full-season context,
per-player contributions, GameStates dynamics, and home/away augmentation.

Changes from baseline:
- Team-relative scores: scores[0] is always the encoded team's score
- Team-relative GameStates: swap home/away score buckets and invert margin when team was away
- Player form tensors: per-roster-player (points, days) appearance history
"""

import logging
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.transformer.phase2.cache_builder import TEAM_TO_IDX, PerGameFeatures
from src.transformer.phase2.sequence_builder import Phase2SequenceBuilder

logger = logging.getLogger(__name__)

# Fixed NBA normalization constants
SCORE_MEAN = 108.0
SCORE_STD = 13.0
MARGIN_STD = 13.0
TOTAL_MEAN = 216.0
TOTAL_STD = 20.0
POINTS_NORM = 30.0


def _normalize_scores(team_score: float, opp_score: float) -> list[float]:
    """Normalize game scores to [team, opponent, margin, total] (team-relative)."""
    return [
        (team_score - SCORE_MEAN) / SCORE_STD,
        (opp_score - SCORE_MEAN) / SCORE_STD,
        (team_score - opp_score) / MARGIN_STD,
        (team_score + opp_score - TOTAL_MEAN) / TOTAL_STD,
    ]


def _encode_team_context(
    context: list[PerGameFeatures],
    team: str,
    days_before: list[int],
    recent_indices: list[int],
    gs_cache: dict,
    player_id_map: dict[int, int],
    max_players: int = 15,
    max_roster: int = 15,
    roster_ids: Optional[np.ndarray] = None,
    max_appearances: int = 40,
    enable_player_form: bool = False,
) -> dict:
    """Encode a team's context games into tensor arrays.

    Scores and GameStates are now team-relative: scores[0] is always the
    encoded team's score, margin is from the team's perspective.
    """
    n_games = len(context)

    scores = np.zeros((n_games, 4), dtype=np.float32)
    opponent_ids = np.zeros(n_games, dtype=np.int64)
    location = np.zeros(n_games, dtype=np.int64)
    player_ids = np.zeros((n_games, max_players), dtype=np.int64)
    player_points = np.zeros((n_games, max_players), dtype=np.float32)
    player_mask = np.ones((n_games, max_players), dtype=bool)  # True=padding
    days_before_arr = np.zeros(n_games, dtype=np.int64)
    is_recent = np.zeros(n_games, dtype=bool)

    # Track per-game team-is-away flag for GameStates swapping
    team_is_away = np.zeros(n_games, dtype=bool)

    for i, gf in enumerate(context):
        # Opponent and location relative to THIS team
        if gf.home_team == team:
            opponent_ids[i] = TEAM_TO_IDX.get(gf.away_team, 0)
            location[i] = 1  # home
            players = gf.home_player_points
            # Team-relative scores: team is home, so team_score=home, opp=away
            scores[i] = _normalize_scores(gf.home_score, gf.away_score)
        else:
            opponent_ids[i] = TEAM_TO_IDX.get(gf.home_team, 0)
            location[i] = 0  # away
            players = gf.away_player_points
            team_is_away[i] = True
            # Team-relative scores: team is away, so team_score=away, opp=home
            scores[i] = _normalize_scores(gf.away_score, gf.home_score)

        # Player contributions (remap raw NBA IDs to sequential indices)
        for j, (pid, pts) in enumerate(players[:max_players]):
            player_ids[i, j] = player_id_map.get(pid, 0)
            player_points[i, j] = pts / POINTS_NORM
            player_mask[i, j] = False

        # Days before target
        days_before_arr[i] = min(days_before[i], 179)

    # Mark recent games
    for idx in recent_indices:
        if idx < n_games:
            is_recent[idx] = True

    # Collect GameStates for recent games (with team-relative swapping)
    gs_periods_list = []
    gs_clock_list = []
    gs_home_score_list = []
    gs_away_score_list = []
    gs_margin_list = []
    gs_lengths_list = []

    for idx in recent_indices:
        if idx >= n_games:
            continue
        gf = context[idx]
        gs_data = gs_cache.get(gf.game_id)
        if gs_data is not None:
            gs_periods_list.append(gs_data["periods"])
            gs_clock_list.append(gs_data["clock_buckets"])

            if team_is_away[idx]:
                # Team was away: swap home/away score buckets, invert margin
                gs_home_score_list.append(gs_data["away_score_buckets"])
                gs_away_score_list.append(gs_data["home_score_buckets"])
                # margin_bucket = home_margin + 60, invert: 120 - bucket
                gs_margin_list.append(120 - gs_data["margin_buckets"])
            else:
                gs_home_score_list.append(gs_data["home_score_buckets"])
                gs_away_score_list.append(gs_data["away_score_buckets"])
                gs_margin_list.append(gs_data["margin_buckets"])

            gs_lengths_list.append(gs_data["length"])
        else:
            # Empty placeholder
            gs_periods_list.append(np.zeros(1, dtype=np.int64))
            gs_clock_list.append(np.zeros(1, dtype=np.int64))
            gs_home_score_list.append(np.zeros(1, dtype=np.int64))
            gs_away_score_list.append(np.zeros(1, dtype=np.int64))
            gs_margin_list.append(np.zeros(1, dtype=np.int64))
            gs_lengths_list.append(1)

    # Pad GS sequences to same length
    n_recent = len(gs_periods_list)
    if n_recent > 0:
        max_gs_rows = max(len(p) for p in gs_periods_list)
        gs_periods = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_clock = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_home_score = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_away_score = np.zeros((n_recent, max_gs_rows), dtype=np.int64)
        gs_margin = np.zeros((n_recent, max_gs_rows), dtype=np.int64)

        for j in range(n_recent):
            length = len(gs_periods_list[j])
            gs_periods[j, :length] = gs_periods_list[j]
            gs_clock[j, :length] = gs_clock_list[j]
            gs_home_score[j, :length] = gs_home_score_list[j]
            gs_away_score[j, :length] = gs_away_score_list[j]
            gs_margin[j, :length] = gs_margin_list[j]

        gs_lengths = np.array(gs_lengths_list, dtype=np.int64)
    else:
        gs_periods = np.zeros((1, 1), dtype=np.int64)
        gs_clock = np.zeros((1, 1), dtype=np.int64)
        gs_home_score = np.zeros((1, 1), dtype=np.int64)
        gs_away_score = np.zeros((1, 1), dtype=np.int64)
        gs_margin = np.zeros((1, 1), dtype=np.int64)
        gs_lengths = np.array([1], dtype=np.int64)

    result = {
        "scores": scores,
        "opponent_ids": opponent_ids,
        "location": location,
        "player_ids": player_ids,
        "player_points": player_points,
        "player_mask": player_mask,
        "days_before": days_before_arr,
        "is_recent": is_recent,
        "gs_periods": gs_periods,
        "gs_clock_buckets": gs_clock,
        "gs_home_score_buckets": gs_home_score,
        "gs_away_score_buckets": gs_away_score,
        "gs_margin_buckets": gs_margin,
        "gs_lengths": gs_lengths,
    }

    # Player form tensors: per-roster-player appearance history
    if enable_player_form and roster_ids is not None:
        R = len(roster_ids)
        A = max_appearances

        form_points = np.zeros((R, A), dtype=np.float32)
        form_days = np.zeros((R, A), dtype=np.int64)
        form_mask = np.ones((R, A), dtype=bool)  # True=padding

        for r, rid in enumerate(roster_ids):
            if rid == 0:
                continue  # Padding player
            app_idx = 0
            # Scan all context games for this player's appearances
            for g in range(n_games):
                if app_idx >= A:
                    break
                # Check player_ids for this game
                for p in range(max_players):
                    if player_ids[g, p] == rid:
                        form_points[r, app_idx] = player_points[g, p]
                        form_days[r, app_idx] = days_before_arr[g]
                        form_mask[r, app_idx] = False
                        app_idx += 1
                        break

        result["roster_form_points"] = form_points
        result["roster_form_days"] = form_days
        result["roster_form_mask"] = form_mask

    return result


class Phase2Dataset(Dataset):
    """
    Phase 2 dataset with full-season context and per-player contributions.

    Returns tensor dicts with home_ and away_ prefixed keys.
    """

    def __init__(
        self,
        game_features: dict[str, PerGameFeatures],
        season_index: dict,
        gs_cache: dict,
        seasons: list[str],
        player_id_map: dict[int, int] | None = None,
        n_recent_games: int = 5,
        max_context_games: int = 82,
        min_history_games: int = 3,
        max_players_per_game: int = 15,
        max_roster_size: int = 15,
        enable_augmentation: bool = False,
        enable_player_form: bool = False,
        max_player_appearances: int = 40,
    ):
        self.game_features = game_features
        self.gs_cache = gs_cache
        self.player_id_map = player_id_map or {}
        self.max_players = max_players_per_game
        self.max_roster = max_roster_size
        self.enable_augmentation = enable_augmentation
        self.enable_player_form = enable_player_form
        self.max_player_appearances = max_player_appearances

        self.builder = Phase2SequenceBuilder(
            game_features=game_features,
            season_index=season_index,
            n_recent_games=n_recent_games,
            max_context_games=max_context_games,
        )

        self.game_ids = self.builder.get_training_game_ids(
            seasons=seasons,
            min_history=min_history_games,
        )

        logger.info(f"Phase2Dataset: {len(self.game_ids)} games for seasons {seasons}")

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Optional[dict]:
        game_id = self.game_ids[idx]
        sample = self.builder.build_sample(game_id)

        if sample is None:
            return self.__getitem__((idx + 1) % len(self.game_ids))

        # Build roster arrays first (needed for form tensors)
        home_roster = np.zeros(self.max_roster, dtype=np.int64)
        away_roster = np.zeros(self.max_roster, dtype=np.int64)
        for i, pid in enumerate(sample.home_roster[:self.max_roster]):
            home_roster[i] = self.player_id_map.get(pid, 0)
        for i, pid in enumerate(sample.away_roster[:self.max_roster]):
            away_roster[i] = self.player_id_map.get(pid, 0)

        # Encode both teams
        home_data = _encode_team_context(
            sample.home_context, sample.home_team,
            sample.home_days_before, sample.home_recent_indices,
            self.gs_cache, self.player_id_map, self.max_players, self.max_roster,
            roster_ids=home_roster,
            max_appearances=self.max_player_appearances,
            enable_player_form=self.enable_player_form,
        )
        away_data = _encode_team_context(
            sample.away_context, sample.away_team,
            sample.away_days_before, sample.away_recent_indices,
            self.gs_cache, self.player_id_map, self.max_players, self.max_roster,
            roster_ids=away_roster,
            max_appearances=self.max_player_appearances,
            enable_player_form=self.enable_player_form,
        )

        # Build result dict
        result = {}

        # Add prefixed team data
        for prefix, data in [("home_", home_data), ("away_", away_data)]:
            for key, value in data.items():
                result[prefix + key] = torch.from_numpy(value)

        # Target game features
        result["home_roster"] = torch.from_numpy(home_roster)
        result["away_roster"] = torch.from_numpy(away_roster)
        result["home_rest_days"] = torch.tensor(
            min(sample.home_rest_days, 29), dtype=torch.int64
        )
        result["away_rest_days"] = torch.tensor(
            min(sample.away_rest_days, 29), dtype=torch.int64
        )

        # Labels
        result["target_home_score"] = torch.tensor(
            sample.target_home_score, dtype=torch.float32
        )
        result["target_away_score"] = torch.tensor(
            sample.target_away_score, dtype=torch.float32
        )

        # Home/away augmentation: 50% chance of swapping
        if self.enable_augmentation and random.random() < 0.5:
            result = self._swap_home_away(result)

        return result

    @staticmethod
    def _swap_home_away(result: dict) -> dict:
        """Swap home and away teams (labels negated)."""
        swapped = {}

        for key, value in result.items():
            if key.startswith("home_"):
                new_key = "away_" + key[5:]
                swapped[new_key] = value
            elif key.startswith("away_"):
                new_key = "home_" + key[5:]
                swapped[new_key] = value
            elif key == "target_home_score":
                swapped["target_away_score"] = value
            elif key == "target_away_score":
                swapped["target_home_score"] = value
            else:
                swapped[key] = value

        return swapped


def collate_phase2(batch: list[Optional[dict]]) -> Optional[dict]:
    """
    Collate Phase 2 samples, padding variable-length dimensions.

    Pads: n_games (max across batch), max_players (max across batch),
          max_gs_rows (max across batch).
    Adds game_mask (bool, True=padding) per team.
    """
    # Filter None samples
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    B = len(batch)

    # Check if player form tensors are present
    has_form = "home_roster_form_points" in batch[0]

    # Determine max dimensions
    max_home_games = max(s["home_scores"].shape[0] for s in batch)
    max_away_games = max(s["away_scores"].shape[0] for s in batch)
    max_home_players = max(s["home_player_ids"].shape[1] for s in batch)
    max_away_players = max(s["away_player_ids"].shape[1] for s in batch)
    max_home_gs_recent = max(s["home_gs_periods"].shape[0] for s in batch)
    max_away_gs_recent = max(s["away_gs_periods"].shape[0] for s in batch)
    max_home_gs_rows = max(s["home_gs_periods"].shape[1] for s in batch)
    max_away_gs_rows = max(s["away_gs_periods"].shape[1] for s in batch)

    result = {}

    # Pad per-team sequences
    for prefix, max_games, max_players, max_gs_recent, max_gs_rows in [
        ("home_", max_home_games, max_home_players, max_home_gs_recent, max_home_gs_rows),
        ("away_", max_away_games, max_away_players, max_away_gs_recent, max_away_gs_rows),
    ]:
        # Game-level features
        scores = torch.zeros(B, max_games, 4)
        opponent_ids = torch.zeros(B, max_games, dtype=torch.int64)
        location = torch.zeros(B, max_games, dtype=torch.int64)
        days_before = torch.zeros(B, max_games, dtype=torch.int64)
        is_recent = torch.zeros(B, max_games, dtype=torch.bool)
        game_mask = torch.ones(B, max_games, dtype=torch.bool)  # True=padding

        # Player-level features
        player_ids = torch.zeros(B, max_games, max_players, dtype=torch.int64)
        player_points = torch.zeros(B, max_games, max_players)
        player_mask = torch.ones(B, max_games, max_players, dtype=torch.bool)

        # GameStates features
        gs_periods = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_clock = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_home_score = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_away_score = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_margin = torch.zeros(B, max_gs_recent, max_gs_rows, dtype=torch.int64)
        gs_lengths = torch.zeros(B, max_gs_recent, dtype=torch.int64)

        for i, s in enumerate(batch):
            ng = s[prefix + "scores"].shape[0]
            np_ = s[prefix + "player_ids"].shape[1]
            nr = s[prefix + "gs_periods"].shape[0]
            ngr = s[prefix + "gs_periods"].shape[1]

            scores[i, :ng] = s[prefix + "scores"]
            opponent_ids[i, :ng] = s[prefix + "opponent_ids"]
            location[i, :ng] = s[prefix + "location"]
            days_before[i, :ng] = s[prefix + "days_before"]
            is_recent[i, :ng] = s[prefix + "is_recent"]
            game_mask[i, :ng] = False

            player_ids[i, :ng, :np_] = s[prefix + "player_ids"]
            player_points[i, :ng, :np_] = s[prefix + "player_points"]
            player_mask[i, :ng, :np_] = s[prefix + "player_mask"]

            gs_periods[i, :nr, :ngr] = s[prefix + "gs_periods"]
            gs_clock[i, :nr, :ngr] = s[prefix + "gs_clock_buckets"]
            gs_home_score[i, :nr, :ngr] = s[prefix + "gs_home_score_buckets"]
            gs_away_score[i, :nr, :ngr] = s[prefix + "gs_away_score_buckets"]
            gs_margin[i, :nr, :ngr] = s[prefix + "gs_margin_buckets"]
            gs_lengths[i, :nr] = s[prefix + "gs_lengths"]

        result[prefix + "scores"] = scores
        result[prefix + "opponent_ids"] = opponent_ids
        result[prefix + "location"] = location
        result[prefix + "days_before"] = days_before
        result[prefix + "is_recent"] = is_recent
        result[prefix + "game_mask"] = game_mask
        result[prefix + "player_ids"] = player_ids
        result[prefix + "player_points"] = player_points
        result[prefix + "player_mask"] = player_mask
        result[prefix + "gs_periods"] = gs_periods
        result[prefix + "gs_clock_buckets"] = gs_clock
        result[prefix + "gs_home_score_buckets"] = gs_home_score
        result[prefix + "gs_away_score_buckets"] = gs_away_score
        result[prefix + "gs_margin_buckets"] = gs_margin
        result[prefix + "gs_lengths"] = gs_lengths

    # Player form tensors (fixed shape R×A, just stack)
    if has_form:
        for prefix in ["home_", "away_"]:
            result[prefix + "roster_form_points"] = torch.stack(
                [s[prefix + "roster_form_points"] for s in batch]
            )
            result[prefix + "roster_form_days"] = torch.stack(
                [s[prefix + "roster_form_days"] for s in batch]
            )
            result[prefix + "roster_form_mask"] = torch.stack(
                [s[prefix + "roster_form_mask"] for s in batch]
            )

    # Target game features (no padding needed)
    result["home_roster"] = torch.stack([s["home_roster"] for s in batch])
    result["away_roster"] = torch.stack([s["away_roster"] for s in batch])
    result["home_rest_days"] = torch.stack([s["home_rest_days"] for s in batch])
    result["away_rest_days"] = torch.stack([s["away_rest_days"] for s in batch])
    result["target_home_scores"] = torch.stack([s["target_home_score"] for s in batch])
    result["target_away_scores"] = torch.stack([s["target_away_score"] for s in batch])

    return result
