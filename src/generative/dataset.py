"""
Dataset and collate function for Phase 4 generative model.

Loads pre-built caches (game states + context) and dynamically assembles
context windows for each target game at training time.

Supports three context modes:
  - Full context (Exp 5): Player-level history + rolling stats + team indices
  - Simplified (Exp 3-4): Rolling stats + team indices only
  - Original (Exp 1-2): Player-level history only
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
    48.0,  # min (max ~48)
    50.0,  # pts
    10.0,  # oreb
    15.0,  # dreb
    15.0,  # ast
    5.0,  # stl
    5.0,  # blk
    10.0,  # tov
    6.0,  # pf
    30.0,  # fga
    20.0,  # fgm
    15.0,  # fg3a
    10.0,  # fg3m
    15.0,  # fta
    12.0,  # ftm
    50.0,  # plus_minus
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
    use_full_context : bool
        If True, build full player-level context AND include rolling stats +
        team indices (Exp 5 mode). Requires both player-level caches and
        rolling_stats.pt.
    """

    def __init__(
        self,
        data_config: GenerativeDataConfig,
        split: Optional[str] = None,
        seasons: Optional[list[str]] = None,
        use_simplified_context: bool = False,
        use_full_context: bool = False,
        use_scoring_events_only: bool = False,
        max_scoring_events: int = 200,
    ) -> None:
        self.config = data_config
        self.cache_dir = Path(data_config.cache_dir)
        self.states_dir = self.cache_dir / "states"
        self.max_context_games = data_config.max_context_games
        self.min_history_games = data_config.min_history_games
        self.max_players = data_config.max_players_per_game
        self.max_seq_len = data_config.max_seq_len
        self.use_simplified_context = use_simplified_context
        self.use_full_context = use_full_context
        self.use_scoring_events_only = use_scoring_events_only
        self.max_scoring_events = max_scoring_events

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

        if self.use_full_context:
            # Full context needs BOTH player-level maps AND rolling stats
            self.player_id_map = torch.load(
                self.cache_dir / "context" / "player_id_map.pt", weights_only=False
            )
            self.team_id_map = torch.load(
                self.cache_dir / "context" / "team_id_map.pt", weights_only=False
            )
            self.rolling_stats = torch.load(
                self.cache_dir / "context" / "rolling_stats.pt", weights_only=False
            )
        elif self.use_simplified_context:
            # Load pre-computed rolling stats
            self.rolling_stats = torch.load(
                self.cache_dir / "context" / "rolling_stats.pt", weights_only=False
            )
        else:
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

            if self.use_full_context:
                # Full context needs both rolling stats AND enough player history
                if game_id not in self.rolling_stats:
                    continue
                home = features["home_team"]
                away = features["away_team"]
                home_prior = self._get_prior_games(home, season, game_id)
                away_prior = self._get_prior_games(away, season, game_id)
                if len(home_prior) < self.min_history_games:
                    continue
                if len(away_prior) < self.min_history_games:
                    continue
            elif self.use_simplified_context:
                # Just need rolling stats to exist for this game
                if game_id not in self.rolling_stats:
                    continue
            else:
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
                (
                    (t, season)
                    for t in TEAM_TO_IDX
                    if game_id in self.season_games.get((t, season), [])
                ),
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

        if self.use_scoring_events_only:
            sample_data = self._compress_to_scoring_events(
                states, score_events, final_margin
            )
            if sample_data is None:
                return None
        else:
            # Truncate to max_seq_len
            if states.shape[0] > self.max_seq_len:
                states = states[: self.max_seq_len]
                score_events = score_events[: self.max_seq_len - 1]
                clock_targets = clock_targets[: self.max_seq_len - 1]
            sample_data = {
                "states": states,
                "score_events": score_events,
                "clock_targets": clock_targets,
            }

        # Build context for both teams
        features = self.game_features[game_id]
        home_team = features["home_team"]
        away_team = features["away_team"]
        target_date = features["date"]

        season = self._get_season(game_id)

        sample = {
            "game_id": game_id,
            "final_margin": torch.tensor(final_margin, dtype=torch.float32),
            **sample_data,
        }

        if self.use_full_context:
            # Full context: player-level history + rolling stats + team indices
            for side, team, is_home in [
                ("home", home_team, True),
                ("away", away_team, False),
            ]:
                ctx = self._build_team_context(
                    team, season, game_id, target_date, is_home
                )
                if ctx is None:
                    return None
                for key, val in ctx.items():
                    sample[f"{side}_{key}"] = val

            # Also include rolling stats and team indices (for FullContextEncoder)
            rs = self.rolling_stats[game_id]
            sample["home_rolling_stats"] = torch.tensor(
                rs["home_stats"], dtype=torch.float32
            )
            sample["away_rolling_stats"] = torch.tensor(
                rs["away_stats"], dtype=torch.float32
            )
            sample["home_team_idx"] = torch.tensor(
                rs["home_team_idx"], dtype=torch.long
            )
            sample["away_team_idx"] = torch.tensor(
                rs["away_team_idx"], dtype=torch.long
            )
        elif self.use_simplified_context:
            # Use pre-computed rolling stats
            rs = self.rolling_stats[game_id]
            sample["home_rolling_stats"] = torch.tensor(
                rs["home_stats"], dtype=torch.float32
            )
            sample["away_rolling_stats"] = torch.tensor(
                rs["away_stats"], dtype=torch.float32
            )
            sample["home_team_idx"] = torch.tensor(
                rs["home_team_idx"], dtype=torch.long
            )
            sample["away_team_idx"] = torch.tensor(
                rs["away_team_idx"], dtype=torch.long
            )
        else:
            # Build full context for home and away
            for side, team, is_home in [
                ("home", home_team, True),
                ("away", away_team, False),
            ]:
                ctx = self._build_team_context(
                    team, season, game_id, target_date, is_home
                )
                if ctx is None:
                    return None
                for key, val in ctx.items():
                    sample[f"{side}_{key}"] = val

        return sample

    def _compress_to_scoring_events(
        self,
        states: torch.Tensor,
        score_events: torch.Tensor,
        final_margin: float,
    ) -> Optional[dict]:
        """Compress full state sequence to scoring events only.

        Filters to positions where a score change occurs, adds inter-event
        time, and remaps event classes:
          Original: {0:none, 1:h+1, 2:h+2, 3:h+3, 4:a+1, 5:a+2, 6:a+3}
          Compressed: {0:h+1, 1:h+2, 2:h+3, 3:a+1, 4:a+2, 5:a+3, 6:game_end}

        Handles both 7-dim base states and 17-dim enriched states:
          - 7-dim: inter_event_time appended as dim 7 -> 8-dim output
          - 17-dim: inter_event_time appended as dim 17 -> 18-dim output

        Returns dict with:
            states: (S+1, D+1) — scoring event states + initial state, with inter-event time
            score_events: (S,) — remapped targets (what event happens AFTER each state)
            clock_targets: (S,) — clock of next scoring event
        where S = number of scoring events (capped at max_scoring_events).
        """
        # Find positions where scoring occurs (in score_events, which is T-1 long)
        scoring_mask = score_events > 0  # bool, (T-1,)
        scoring_indices = scoring_mask.nonzero(as_tuple=True)[
            0
        ]  # indices into score_events

        if len(scoring_indices) < 2:
            return None

        # Cap at max scoring events
        if len(scoring_indices) > self.max_scoring_events:
            scoring_indices = scoring_indices[: self.max_scoring_events]

        S = len(scoring_indices)

        # Build compressed state sequence:
        # Position 0 = initial game state (0-0, P1, 720s)
        # Positions 1..S = states at each scoring event
        # The state at scoring_indices[i] in the original is the state BEFORE
        # the score event. We want the state AFTER, which is states[scoring_indices[i]+1].

        # Actually, let's think about this carefully:
        # score_events[i] describes what happens between states[i] and states[i+1]
        # So if score_events[k] > 0, the scoring event occurs at position k,
        # and states[k+1] reflects the updated score.
        #
        # For the compressed sequence:
        # - Input state at position j = the state AFTER the j-th scoring event
        #   (i.e., states[scoring_indices[j] + 1])
        # - Target at position j = the (j+1)-th scoring event class
        # - The initial state (before any scoring) = states[0]

        # Gather state vectors: initial + after each scoring event
        state_indices = torch.zeros(S + 1, dtype=torch.long)
        state_indices[0] = 0  # initial state
        state_indices[1:] = scoring_indices + 1  # state after each scoring event

        compressed_states_base = states[state_indices]  # (S+1, D) where D=7 or 17

        # Compute inter-event time (seconds between consecutive scoring events)
        # game_progress is at index 2, range [0, 1] over 2880 seconds
        progress = compressed_states_base[:, 2]  # (S+1,)
        time_seconds = progress * 2880.0  # absolute elapsed seconds
        inter_event_time = torch.zeros(S + 1, dtype=torch.float32)
        inter_event_time[1:] = time_seconds[1:] - time_seconds[:-1]
        # Normalize: typical inter-event time ~20-30s, max ~120s
        inter_event_time = inter_event_time / 120.0

        # Append inter-event time as the last dimension
        # 7-dim -> 8-dim, 17-dim -> 18-dim
        compressed_states = torch.cat(
            [compressed_states_base, inter_event_time.unsqueeze(-1)], dim=-1
        )  # (S+1, D+1)

        # Build targets: remap classes (remove no_score class)
        # Original: 1→h+1, 2→h+2, 3→h+3, 4→a+1, 5→a+2, 6→a+3
        # Remapped: 0→h+1, 1→h+2, 2→h+3, 3→a+1, 4→a+2, 5→a+3, 6→game_end
        raw_events = score_events[scoring_indices]  # (S,) values 1-6
        remapped_events = raw_events - 1  # shift to 0-5

        # Targets for the compressed sequence:
        # Position j predicts what the NEXT scoring event is.
        # Positions 0..S-2 predict events at scoring_indices[1..S-1]
        # Position S-1: game_end (class 6) for 7-class mode,
        #               or drop last position for 6-class mode (rules engine handles termination)
        if self.use_scoring_events_only and self.max_scoring_events > 0 and S > 1:
            # Check if we're in 6-class mode (no game_end class)
            # In 6-class mode, we simply use all S positions predicting the next event,
            # and the last position predicts the last scoring event (no game_end sentinel)
            # We detect 6-class mode by the absence of game_end in valid classes
            # For backward compat, we still include game_end=6 here;
            # the loss function will handle it by ignoring targets >= n_classes
            pass
        targets = torch.zeros(S, dtype=torch.int64)
        targets[:-1] = remapped_events[1:]  # next event's class (0-5)
        targets[-1] = 6  # game_end sentinel (ignored by 6-class CE via ignore_index)

        # Clock targets: game_progress of the next scoring event
        # Using game_progress (index 2, monotonic 0→1) instead of clock_norm
        # (index 1, resets each period) for better rollout tracking
        clock_targets = torch.zeros(S, dtype=torch.float32)
        for j in range(S - 1):
            next_state_idx = scoring_indices[j + 1] + 1
            clock_targets[j] = states[next_state_idx, 2]  # game_progress
        # Last position: game over = progress 1.0
        clock_targets[-1] = 1.0

        return {
            "states": compressed_states,  # (S+1, D+1)
            "score_events": targets,  # (S,)
            "clock_targets": clock_targets,  # (S,)
        }

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

        When game_features contain 3-element tuples (pid, stats, pos_idx),
        also includes:
            player_positions: (G, P) position indices 0-4
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
        player_positions = torch.full((G, P), 4, dtype=torch.long)  # default: unknown
        days_before = torch.zeros(G, dtype=torch.float32)

        # Track whether we have position data (3-element tuples)
        has_positions = False

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

            # Player data — handle both 2-element (pid, stats) and 3-element (pid, stats, pos_idx)
            for p_idx, entry in enumerate(team_players[:P]):
                if len(entry) == 3:
                    pid, stats, pos_idx = entry
                    player_positions[g_idx, p_idx] = pos_idx
                    has_positions = True
                else:
                    pid, stats = entry

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

        result = {
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

        # Include player positions when available (full context / enriched cache)
        if has_positions or self.use_full_context:
            result["player_positions"] = player_positions

        return result

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

    Handles three context modes:
      - Full context (Exp 5): player-level + rolling stats + positions
      - Simplified (Exp 3-4): rolling stats only
      - Original (Exp 1-2): player-level only

    Returns None if all samples in batch are None (no valid data).
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    B = len(batch)

    # --- Game state padding ---
    max_T = max(b["states"].shape[0] for b in batch)
    state_dim = batch[0]["states"].shape[-1]  # 7, 8, or 18
    max_events = max_T - 1  # score_events and clock_targets have T-1 entries

    states = torch.zeros(B, max_T, state_dim, dtype=torch.float32)
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

    # Derive home_win from final_margin for pre-decoder BCE loss
    home_win = (final_margin > 0).float()  # (B,)

    result = {
        "states": states,
        "score_events": score_events,
        "clock_targets": clock_targets,
        "state_mask": state_mask,
        "final_margin": final_margin,
        "home_win": home_win,
    }

    # --- Extract game_progress for outcome loss position weighting ---
    # game_progress is at index 2 in the state vector (all modes)
    # We need the input positions (states[:, :-1, 2]) to match the decoder output shape
    game_progress = states[:, :-1, 2]  # (B, max_events)
    result["game_progress"] = game_progress

    # --- Determine context mode ---
    has_player_context = f"home_scores" in batch[0]
    has_rolling_stats = "home_rolling_stats" in batch[0]
    is_full_context = has_player_context and has_rolling_stats

    if is_full_context:
        # Full context mode (Exp 5): player-level + rolling stats + team indices
        # Stack rolling stats and team indices
        result["home_rolling_stats"] = torch.stack(
            [b["home_rolling_stats"] for b in batch]
        )
        result["away_rolling_stats"] = torch.stack(
            [b["away_rolling_stats"] for b in batch]
        )
        result["home_team_idx"] = torch.stack([b["home_team_idx"] for b in batch])
        result["away_team_idx"] = torch.stack([b["away_team_idx"] for b in batch])

        # Pad player-level context (per side)
        has_positions = f"home_player_positions" in batch[0]
        for side in ["home", "away"]:
            max_G = max(b[f"{side}_scores"].shape[0] for b in batch)
            P = batch[0][f"{side}_player_ids"].shape[1]

            side_scores = torch.zeros(B, max_G, 4, dtype=torch.float32)
            side_opponents = torch.zeros(B, max_G, dtype=torch.long)
            side_locations = torch.zeros(B, max_G, dtype=torch.long)
            side_player_ids = torch.zeros(B, max_G, P, dtype=torch.long)
            side_player_stats = torch.zeros(B, max_G, P, 16, dtype=torch.float32)
            side_player_mask = torch.zeros(B, max_G, P, dtype=torch.bool)
            side_days_before = torch.zeros(B, max_G, dtype=torch.float32)
            side_game_mask = torch.zeros(B, max_G, dtype=torch.bool)
            side_rest_days = torch.stack([b[f"{side}_rest_days"] for b in batch])

            # Player positions: default to unknown (4)
            side_player_positions = (
                torch.full((B, max_G, P), 4, dtype=torch.long)
                if has_positions
                else None
            )

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

                if has_positions and f"{side}_player_positions" in b:
                    side_player_positions[i, :G] = b[f"{side}_player_positions"]

            result[f"{side}_scores"] = side_scores
            result[f"{side}_opponents"] = side_opponents
            result[f"{side}_locations"] = side_locations
            result[f"{side}_player_ids"] = side_player_ids
            result[f"{side}_player_stats"] = side_player_stats
            result[f"{side}_player_mask"] = side_player_mask
            result[f"{side}_days_before"] = side_days_before
            result[f"{side}_game_mask"] = side_game_mask
            result[f"{side}_rest_days"] = side_rest_days

            if has_positions and side_player_positions is not None:
                result[f"{side}_player_positions"] = side_player_positions

    elif has_rolling_stats and not has_player_context:
        # Simplified context: fixed-size tensors, just stack
        result["home_rolling_stats"] = torch.stack(
            [b["home_rolling_stats"] for b in batch]
        )  # (B, N)
        result["away_rolling_stats"] = torch.stack(
            [b["away_rolling_stats"] for b in batch]
        )  # (B, N)
        result["home_team_idx"] = torch.stack(
            [b["home_team_idx"] for b in batch]
        )  # (B,)
        result["away_team_idx"] = torch.stack(
            [b["away_team_idx"] for b in batch]
        )  # (B,)
    else:
        # Original full context padding (per side, no rolling stats)
        has_positions = f"home_player_positions" in batch[0]
        for side in ["home", "away"]:
            max_G = max(b[f"{side}_scores"].shape[0] for b in batch)
            P = batch[0][f"{side}_player_ids"].shape[1]

            side_scores = torch.zeros(B, max_G, 4, dtype=torch.float32)
            side_opponents = torch.zeros(B, max_G, dtype=torch.long)
            side_locations = torch.zeros(B, max_G, dtype=torch.long)
            side_player_ids = torch.zeros(B, max_G, P, dtype=torch.long)
            side_player_stats = torch.zeros(B, max_G, P, 16, dtype=torch.float32)
            side_player_mask = torch.zeros(B, max_G, P, dtype=torch.bool)
            side_days_before = torch.zeros(B, max_G, dtype=torch.float32)
            side_game_mask = torch.zeros(B, max_G, dtype=torch.bool)
            side_rest_days = torch.stack([b[f"{side}_rest_days"] for b in batch])

            # Player positions: only if present in the batch
            side_player_positions = (
                torch.full((B, max_G, P), 4, dtype=torch.long)
                if has_positions
                else None
            )

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

                if has_positions and f"{side}_player_positions" in b:
                    side_player_positions[i, :G] = b[f"{side}_player_positions"]

            result[f"{side}_scores"] = side_scores
            result[f"{side}_opponents"] = side_opponents
            result[f"{side}_locations"] = side_locations
            result[f"{side}_player_ids"] = side_player_ids
            result[f"{side}_player_stats"] = side_player_stats
            result[f"{side}_player_mask"] = side_player_mask
            result[f"{side}_days_before"] = side_days_before
            result[f"{side}_game_mask"] = side_game_mask
            result[f"{side}_rest_days"] = side_rest_days

            if has_positions and side_player_positions is not None:
                result[f"{side}_player_positions"] = side_player_positions

    return result
