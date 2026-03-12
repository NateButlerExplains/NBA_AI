"""
Dataset and collate function for Phase 4 generative model.

Loads pre-built caches (game states + context) and dynamically assembles
context windows for each target game at training time.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.generative.cache_builder import TEAM_TO_IDX, normalize_team
from src.generative.config import GenerativeDataConfig

logger = logging.getLogger(__name__)

# Normalization constants for the 16 PlayerBox stats
# [min, pts, oreb, dreb, ast, stl, blk, tov, pf, fga, fgm, fg3a, fg3m, fta, ftm, plus_minus]
STAT_NORMS = [
    48.0,   # min (max ~48)
    50.0,   # pts
    10.0,   # oreb
    15.0,   # dreb
    15.0,   # ast
    5.0,    # stl
    5.0,    # blk
    10.0,   # tov
    6.0,    # pf
    30.0,   # fga
    20.0,   # fgm
    15.0,   # fg3a
    10.0,   # fg3m
    15.0,   # fta
    12.0,   # ftm
    50.0,   # plus_minus
]


class GenerativeDataset(Dataset):
    """PyTorch dataset for the Phase 4 generative model.

    Loads pre-built caches and dynamically constructs context windows
    (prior same-season games) for each target game.

    Parameters
    ----------
    data_config : GenerativeDataConfig
        Data configuration (cache dir, seasons, limits).
    split : str
        One of "train", "val", "test".
    seasons : list[str], optional
        Explicit season list (overrides split).
    """

    def __init__(
        self,
        data_config: GenerativeDataConfig,
        split: Optional[str] = None,
        seasons: Optional[list[str]] = None,
    ) -> None:
        self.config = data_config
        self.cache_dir = Path(data_config.cache_dir)
        self.states_dir = self.cache_dir / "states"
        self.max_context_games = data_config.max_context_games
        self.min_history_games = data_config.min_history_games
        self.max_players = data_config.max_players_per_game
        self.max_seq_len = data_config.max_seq_len

        # Determine seasons for this split
        if seasons is not None:
            self.seasons = seasons
        elif split == "train":
            self.seasons = data_config.train_seasons
        elif split == "val":
            self.seasons = data_config.val_seasons
        elif split == "test":
            self.seasons = data_config.test_seasons
        else:
            raise ValueError(f"Must provide split or seasons, got split={split}")

        self.split = split or "custom"

        # Load context caches
        self.season_games = self._load_season_games()
        self.game_features = torch.load(
            self.cache_dir / "context" / "game_features.pt", weights_only=False
        )
        self.player_id_map = torch.load(
            self.cache_dir / "context" / "player_id_map.pt", weights_only=False
        )
        self.team_id_map = torch.load(
            self.cache_dir / "context" / "team_id_map.pt", weights_only=False
        )

        # Build list of eligible game_ids (those with state cache files + enough history)
        self.game_ids = self._build_game_list()
        logger.info(
            f"GenerativeDataset({self.split}): {len(self.game_ids)} games "
            f"from {len(self.seasons)} seasons"
        )

        # Pre-compute stat normalization tensor
        self._stat_norms = torch.tensor(STAT_NORMS, dtype=torch.float32)

    def _load_season_games(self) -> dict[tuple[str, str], list[str]]:
        """Load season_games cache and convert keys back to (team, season) tuples."""
        raw = torch.load(
            self.cache_dir / "context" / "season_games.pt", weights_only=False
        )
        result = {}
        for key_str, game_ids in raw.items():
            team, season = key_str.split("|", 1)
            result[(team, season)] = game_ids
        return result

    def _build_game_list(self) -> list[str]:
        """Find all games in target seasons that have state caches and enough history."""
        game_ids = []

        for game_id, features in self.game_features.items():
            # Check season
            season = self._get_season(game_id)
            if season not in self.seasons:
                continue

            # Check state cache exists
            state_path = self.states_dir / f"{game_id}.pt"
            if not state_path.exists():
                continue

            # Check enough history for both teams
            home = features["home_team"]
            away = features["away_team"]

            home_prior = self._get_prior_games(home, season, game_id)
            away_prior = self._get_prior_games(away, season, game_id)

            if len(home_prior) < self.min_history_games:
                continue
            if len(away_prior) < self.min_history_games:
                continue

            game_ids.append(game_id)

        # Sort chronologically (game_id encodes season in positions 2-5)
        game_ids.sort()
        return game_ids

    def _get_season(self, game_id: str) -> str:
        """Derive season string from game_id or game_features."""
        feat = self.game_features.get(game_id)
        if feat and "date" in feat:
            date_str = feat["date"]
            if date_str and len(date_str) >= 4:
                try:
                    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    year = dt.year
                    month = dt.month
                    if month >= 9:
                        return f"{year}-{year + 1}"
                    else:
                        return f"{year - 1}-{year}"
                except ValueError:
                    pass
        # Fallback: game_id prefix (00220 = 2022-23 season)
        # Game IDs: first 3 chars = type, chars 3-4 = season decade
        # This is unreliable — try matching against known seasons
        for season in self.seasons:
            team_key = next(
                ((t, season) for t in TEAM_TO_IDX
                 if game_id in self.season_games.get((t, season), [])),
                None,
            )
            if team_key:
                return season
        return ""

    def _get_prior_games(
        self, team: str, season: str, target_game_id: str
    ) -> list[str]:
        """Get team's prior games in the season before the target game."""
        all_games = self.season_games.get((team, season), [])
        prior = []
        for gid in all_games:
            if gid == target_game_id:
                break
            prior.append(gid)
        return prior

    def __len__(self) -> int:
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Load a single training sample.

        Returns dict with game state tensors + context tensors for both teams,
        or None if data is corrupted/missing.
        """
        game_id = self.game_ids[idx]

        # Load game state cache
        try:
            state_cache = torch.load(
                self.states_dir / f"{game_id}.pt", weights_only=False
            )
        except Exception:
            return None

        states = state_cache["states"]  # (T, 7)
        score_events = state_cache["score_events"]  # (T-1,)
        clock_targets = state_cache["clock_targets"]  # (T-1,)
        final_margin = float(state_cache["final_margin"])

        # Truncate to max_seq_len
        if states.shape[0] > self.max_seq_len:
            states = states[: self.max_seq_len]
            score_events = score_events[: self.max_seq_len - 1]
            clock_targets = clock_targets[: self.max_seq_len - 1]

        # Build context for both teams
        features = self.game_features[game_id]
        home_team = features["home_team"]
        away_team = features["away_team"]
        target_date = features["date"]

        season = self._get_season(game_id)

        sample = {
            "game_id": game_id,
            "states": states,
            "score_events": score_events,
            "clock_targets": clock_targets,
            "final_margin": torch.tensor(final_margin, dtype=torch.float32),
        }

        # Build context for home and away
        for side, team, is_home in [
            ("home", home_team, True),
            ("away", away_team, False),
        ]:
            ctx = self._build_team_context(team, season, game_id, target_date, is_home)
            if ctx is None:
                return None
            for key, val in ctx.items():
                sample[f"{side}_{key}"] = val

        return sample

    def _build_team_context(
        self,
        team: str,
        season: str,
        target_game_id: str,
        target_date: str,
        is_home_in_target: bool,
    ) -> Optional[dict]:
        """Build context tensors for one team.

        Returns dict with keys:
            scores, opponents, locations, player_ids, player_stats,
            player_mask, days_before, game_mask, rest_days
        """
        prior_game_ids = self._get_prior_games(team, season, target_game_id)

        # Limit to most recent max_context_games
        if len(prior_game_ids) > self.max_context_games:
            prior_game_ids = prior_game_ids[-self.max_context_games :]

        G = len(prior_game_ids)
        P = self.max_players

        # Pre-allocate tensors
        scores = torch.zeros(G, 4, dtype=torch.float32)
        opponents = torch.zeros(G, dtype=torch.long)
        locations = torch.zeros(G, dtype=torch.long)
        player_ids = torch.zeros(G, P, dtype=torch.long)
        player_stats = torch.zeros(G, P, 16, dtype=torch.float32)
        player_mask = torch.zeros(G, P, dtype=torch.bool)
        days_before = torch.zeros(G, dtype=torch.float32)

        target_dt = self._parse_date(target_date)

        for g_idx, gid in enumerate(prior_game_ids):
            feat = self.game_features.get(gid)
            if feat is None:
                continue

            # Determine if team was home or away in this prior game
            if feat["home_team"] == team:
                team_score = feat["home_score"]
                opp_score = feat["away_score"]
                opp_team = feat["away_team"]
                loc = 1  # home
                team_players = feat.get("home_player_stats", [])
            else:
                team_score = feat["away_score"]
                opp_score = feat["home_score"]
                opp_team = feat["home_team"]
                loc = 0  # away
                team_players = feat.get("away_player_stats", [])

            # Score features (normalized)
            margin = team_score - opp_score
            total = team_score + opp_score
            scores[g_idx] = torch.tensor(
                [team_score / 150.0, opp_score / 150.0, margin / 50.0, total / 300.0],
                dtype=torch.float32,
            )

            # Opponent index
            opp_idx = self.team_id_map.get(normalize_team(opp_team), 0)
            opponents[g_idx] = opp_idx

            # Location
            locations[g_idx] = loc

            # Days before target
            game_dt = self._parse_date(feat["date"])
            if target_dt and game_dt:
                delta = (target_dt - game_dt).days
                days_before[g_idx] = min(float(max(delta, 0)), 180.0)

            # Player data
            for p_idx, (pid, stats) in enumerate(team_players[:P]):
                mapped_pid = self.player_id_map.get(pid, 0)
                player_ids[g_idx, p_idx] = mapped_pid

                # Normalize stats
                stat_tensor = torch.tensor(stats, dtype=torch.float32)
                player_stats[g_idx, p_idx] = stat_tensor / self._stat_norms
                player_mask[g_idx, p_idx] = True

        # Game mask: all prior games are valid
        game_mask = torch.ones(G, dtype=torch.bool)

        # Rest days: days since most recent prior game
        rest_days = 1  # default
        if G > 0:
            last_game_feat = self.game_features.get(prior_game_ids[-1])
            if last_game_feat and target_dt:
                last_dt = self._parse_date(last_game_feat["date"])
                if last_dt:
                    rest_days = min((target_dt - last_dt).days, 30)
                    rest_days = max(rest_days, 0)

        return {
            "scores": scores,
            "opponents": opponents,
            "locations": locations,
            "player_ids": player_ids,
            "player_stats": player_stats,
            "player_mask": player_mask,
            "days_before": days_before,
            "game_mask": game_mask,
            "rest_days": torch.tensor(rest_days, dtype=torch.long),
        }

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string to datetime, returning None on failure."""
        if not date_str or len(date_str) < 10:
            return None
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            return None


def generative_collate(batch: list[Optional[dict]]) -> Optional[dict]:
    """Collate variable-length generative samples into padded batch tensors.

    Pads:
      - Game states to batch-max T
      - Context games to batch-max G per side
      - Players already padded to max_players_per_game in dataset

    Returns None if all samples in batch are None (no valid data).
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    B = len(batch)

    # --- Game state padding ---
    max_T = max(b["states"].shape[0] for b in batch)
    max_events = max_T - 1  # score_events and clock_targets have T-1 entries

    states = torch.zeros(B, max_T, 7, dtype=torch.float32)
    score_events = torch.zeros(B, max_events, dtype=torch.int64)
    clock_targets = torch.zeros(B, max_events, dtype=torch.float32)
    state_mask = torch.zeros(B, max_events, dtype=torch.bool)
    final_margin = torch.stack([b["final_margin"] for b in batch])

    for i, b in enumerate(batch):
        T = b["states"].shape[0]
        states[i, :T] = b["states"]
        E = b["score_events"].shape[0]
        score_events[i, :E] = b["score_events"]
        clock_targets[i, :E] = b["clock_targets"]
        state_mask[i, :E] = True

    result = {
        "states": states,
        "score_events": score_events,
        "clock_targets": clock_targets,
        "state_mask": state_mask,
        "final_margin": final_margin,
    }

    # --- Context padding (per side) ---
    for side in ["home", "away"]:
        max_G = max(b[f"{side}_scores"].shape[0] for b in batch)
        P = batch[0][f"{side}_player_ids"].shape[1]  # already fixed at max_players

        side_scores = torch.zeros(B, max_G, 4, dtype=torch.float32)
        side_opponents = torch.zeros(B, max_G, dtype=torch.long)
        side_locations = torch.zeros(B, max_G, dtype=torch.long)
        side_player_ids = torch.zeros(B, max_G, P, dtype=torch.long)
        side_player_stats = torch.zeros(B, max_G, P, 16, dtype=torch.float32)
        side_player_mask = torch.zeros(B, max_G, P, dtype=torch.bool)
        side_days_before = torch.zeros(B, max_G, dtype=torch.float32)
        side_game_mask = torch.zeros(B, max_G, dtype=torch.bool)
        side_rest_days = torch.stack([b[f"{side}_rest_days"] for b in batch])

        for i, b in enumerate(batch):
            G = b[f"{side}_scores"].shape[0]
            side_scores[i, :G] = b[f"{side}_scores"]
            side_opponents[i, :G] = b[f"{side}_opponents"]
            side_locations[i, :G] = b[f"{side}_locations"]
            side_player_ids[i, :G] = b[f"{side}_player_ids"]
            side_player_stats[i, :G] = b[f"{side}_player_stats"]
            side_player_mask[i, :G] = b[f"{side}_player_mask"]
            side_days_before[i, :G] = b[f"{side}_days_before"]
            side_game_mask[i, :G] = True

        result[f"{side}_scores"] = side_scores
        result[f"{side}_opponents"] = side_opponents
        result[f"{side}_locations"] = side_locations
        result[f"{side}_player_ids"] = side_player_ids
        result[f"{side}_player_stats"] = side_player_stats
        result[f"{side}_player_mask"] = side_player_mask
        result[f"{side}_days_before"] = side_days_before
        result[f"{side}_game_mask"] = side_game_mask
        result[f"{side}_rest_days"] = side_rest_days

    return result
