"""
Cache Builder for Phase 4 Generative Model.

Pre-processes all data from the database into tensor caches for training.
Two cache types:
  1. Game State Cache — per-game state tensors, score events, and clock targets
  2. Context Cache — season-level game features, player stats, and mappings

Usage:
    python scripts/build_generative_cache.py --seasons 2008-2009 2009-2010 ...
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from src.database import get_db

logger = logging.getLogger(__name__)

# 30 NBA team tricodes -> integer indices
TEAM_TO_IDX = {
    "ATL": 0,
    "BOS": 1,
    "BKN": 2,
    "CHA": 3,
    "CHI": 4,
    "CLE": 5,
    "DAL": 6,
    "DEN": 7,
    "DET": 8,
    "GSW": 9,
    "HOU": 10,
    "IND": 11,
    "LAC": 12,
    "LAL": 13,
    "MEM": 14,
    "MIA": 15,
    "MIL": 16,
    "MIN": 17,
    "NOP": 18,
    "NYK": 19,
    "OKC": 20,
    "ORL": 21,
    "PHI": 22,
    "PHX": 23,
    "POR": 24,
    "SAC": 25,
    "SAS": 26,
    "TOR": 27,
    "UTA": 28,
    "WAS": 29,
}

# Historical team codes -> current franchise codes
HISTORICAL_TO_CURRENT = {
    "NJN": "BKN",  # New Jersey Nets -> Brooklyn Nets
    "SEA": "OKC",  # Seattle SuperSonics -> Oklahoma City Thunder
    "NOH": "NOP",  # New Orleans Hornets -> New Orleans Pelicans
    "NOK": "NOP",  # New Orleans/OKC Hornets -> New Orleans Pelicans
    "VAN": "MEM",  # Vancouver Grizzlies -> Memphis Grizzlies
    "CHH": "CHA",  # Charlotte Hornets (old) -> Charlotte Hornets (current)
}

# 16 PlayerBox stats in order
STAT_COLUMNS = [
    "min",
    "pts",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "fga",
    "fgm",
    "fg3a",
    "fg3m",
    "fta",
    "ftm",
    "plus_minus",
]


def parse_clock(clock_str: str) -> float:
    """Parse 'MM:SS' or 'M:SS' clock string to seconds remaining.

    Returns seconds as a float. Handles missing/malformed clocks gracefully.
    """
    if not clock_str or not isinstance(clock_str, str):
        return 0.0
    try:
        parts = clock_str.strip().split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60.0 + seconds
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def normalize_team(abbrev: str) -> str:
    """Map historical team abbreviations to current."""
    return HISTORICAL_TO_CURRENT.get(abbrev, abbrev)


# Position mapping: NBA position strings -> 5-class index
# 0: Guard (PG, SG, G)
# 1: Forward (SF, PF, F)
# 2: Center (C)
# 3: Guard-Forward (GF, FG)
# 4: Unknown (NULL or empty)
POSITION_MAP = {
    "PG": 0,
    "SG": 0,
    "G": 0,
    "SF": 1,
    "PF": 1,
    "F": 1,
    "C": 2,
    "GF": 3,
    "FG": 3,
}
POSITION_UNKNOWN = 4


class GenerativeCacheBuilder:
    """Builds tensor caches from the database for the generative model."""

    def __init__(
        self,
        db_path: str,
        cache_dir: str,
        seasons: list[str],
        enrich_states: bool = False,
        n_rolling_features: int = 24,
    ):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.seasons = seasons
        self.enrich_states = enrich_states
        self.n_rolling_features = n_rolling_features

        # Sub-directories
        self.states_dir = self.cache_dir / "states"
        self.context_dir = self.cache_dir / "context"

        # Tracking
        self.eligible_game_ids: list[str] = []
        self.all_player_ids: set[int] = set()

        # Historical position lookup (built during context cache)
        self.position_lookup: dict[int, int] = {}  # player_id -> position_idx

        # Statistics
        self.stats = {
            "total_games": 0,
            "games_with_states": 0,
            "games_skipped": 0,
            "total_states": 0,
            "class_counts": np.zeros(7, dtype=np.int64),
        }

    def build(self):
        """Build all caches."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir.mkdir(parents=True, exist_ok=True)

        self._fetch_eligible_games()
        self._build_game_state_cache()
        self._build_context_cache()
        self._build_rolling_stats_cache()
        self._save_metadata()

    def _fetch_eligible_games(self):
        """Query all eligible game_ids: status=3, not All-Star, in requested seasons."""
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT g.game_id
            FROM Games g
            WHERE g.status = 3
              AND g.season IN ({season_placeholders})
              AND g.game_id NOT LIKE '003%'
            ORDER BY g.date_time_utc
        """
        with get_db(self.db_path) as conn:
            rows = conn.execute(query, self.seasons).fetchall()

        self.eligible_game_ids = [row[0] for row in rows]
        self.stats["total_games"] = len(self.eligible_game_ids)
        logger.info(
            f"Found {len(self.eligible_game_ids)} eligible games across {len(self.seasons)} seasons"
        )

    def _build_game_state_cache(self):
        """Parse GameStates -> state tensors + score events + clock targets.

        Saves one .pt file per game in states/ directory.
        When enrich_states=True, also fetches players_data and computes
        additional per-state features (10 new dims: player scoring + dynamics).
        """
        logger.info("Building game state cache...")

        chunk_size = 500
        game_ids = self.eligible_game_ids

        # Determine columns to fetch
        if self.enrich_states:
            select_cols = (
                "game_id, play_id, period, clock, home_score, away_score, players_data"
            )
            n_extra = 1  # players_data column
        else:
            select_cols = "game_id, play_id, period, clock, home_score, away_score"
            n_extra = 0

        # Fetch and process in chunks
        processed = 0
        skipped = 0

        for chunk_start in tqdm(
            range(0, len(game_ids), chunk_size), desc="Game states", unit="chunk"
        ):
            chunk = game_ids[chunk_start : chunk_start + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))

            with get_db(self.db_path) as conn:
                rows = conn.execute(
                    f"""
                    SELECT {select_cols}
                    FROM GameStates
                    WHERE game_id IN ({placeholders})
                    ORDER BY game_id, play_id
                    """,
                    chunk,
                ).fetchall()

            # Group rows by game_id
            game_rows = defaultdict(list)
            for row in rows:
                game_rows[row[0]].append(row[1:])

            for game_id in chunk:
                if game_id not in game_rows:
                    skipped += 1
                    continue

                result = self._process_game_states(game_id, game_rows[game_id])
                if result is None:
                    skipped += 1
                    continue

                # Save individual game cache
                save_path = self.states_dir / f"{game_id}.pt"
                torch.save(result, save_path)
                processed += 1

                # Update statistics
                n = result["n_states"]
                self.stats["total_states"] += n
                if n > 1:
                    events = result["score_events"]
                    for cls in range(7):
                        self.stats["class_counts"][cls] += int((events == cls).sum())

        self.stats["games_with_states"] = processed
        self.stats["games_skipped"] = skipped

        logger.info(f"Game state cache: {processed} games saved, {skipped} skipped")
        if processed > 0:
            avg_states = self.stats["total_states"] / processed
            logger.info(f"Average states per game: {avg_states:.1f}")
            state_dim = 17 if self.enrich_states else 7
            logger.info(f"State dimension: {state_dim}")
            logger.info("Score event class distribution:")
            class_names = [
                "no_score",
                "home+1",
                "home+2",
                "home+3",
                "away+1",
                "away+2",
                "away+3",
            ]
            total_events = self.stats["class_counts"].sum()
            for i, name in enumerate(class_names):
                count = self.stats["class_counts"][i]
                pct = 100.0 * count / total_events if total_events > 0 else 0.0
                logger.info(f"  {name}: {count:,} ({pct:.1f}%)")

    def _process_game_states(self, game_id: str, rows: list) -> Optional[dict]:
        """Process raw GameStates rows for a single game into tensors.

        Args:
            game_id: The game identifier.
            rows: List of tuples ordered by play_id. Format depends on enrich_states:
                  - False: (play_id, period, clock, home_score, away_score)
                  - True:  (play_id, period, clock, home_score, away_score, players_data)

        Returns:
            Dict with states, score_events, clock_targets, n_states,
            final_home_score, final_away_score, final_margin.
            When enrich_states=True, states are 17-dim (7 base + 10 enriched).
            None if game has insufficient data.
        """
        # Truncate at regulation end (period <= 4)
        reg_rows = [r for r in rows if r[1] is not None and r[1] <= 4]

        if len(reg_rows) < 2:
            return None

        # Build state vectors
        state_list = []

        # Track scoring runs for dynamics features (enriched mode only)
        home_run = 0
        away_run = 0

        for idx, row in enumerate(reg_rows):
            play_id = row[0]
            period = row[1]
            clock = row[2]
            h_score = row[3] or 0
            a_score = row[4] or 0

            seconds_remaining = parse_clock(clock)

            period_norm = period / 4.0
            clock_norm = seconds_remaining / 720.0
            elapsed = (period - 1) * 720.0 + (720.0 - seconds_remaining)
            game_progress = elapsed / 2880.0
            home_score_norm = h_score / 150.0
            away_score_norm = a_score / 150.0
            margin_norm = (h_score - a_score) / 50.0
            total_norm = (h_score + a_score) / 300.0

            base_state = [
                period_norm,
                clock_norm,
                game_progress,
                home_score_norm,
                away_score_norm,
                margin_norm,
                total_norm,
            ]

            if self.enrich_states:
                players_data_str = row[5]
                enriched = self._extract_enriched_features(
                    players_data_str,
                    h_score,
                    a_score,
                    period,
                    seconds_remaining,
                    home_run,
                    away_run,
                )
                base_state.extend(enriched)

                # Update scoring runs based on score delta from previous state
                if idx > 0:
                    prev_h = reg_rows[idx - 1][3] or 0
                    prev_a = reg_rows[idx - 1][4] or 0
                    h_delta = h_score - prev_h
                    a_delta = a_score - prev_a

                    # Handle score corrections (negative deltas) as no-ops
                    if h_delta > 0 and a_delta <= 0:
                        home_run = min(home_run + 1, 5)
                        away_run = 0
                    elif a_delta > 0 and h_delta <= 0:
                        away_run = min(away_run + 1, 5)
                        home_run = 0
                    elif h_delta > 0 and a_delta > 0:
                        # Simultaneous — reset both
                        home_run = 0
                        away_run = 0
                    # If both deltas <= 0 (no-score or correction), keep runs unchanged

            state_list.append(base_state)

        states = torch.tensor(state_list, dtype=torch.float32)
        n_states = states.shape[0]

        # Build score events: classify delta between consecutive states
        score_events = []
        for i in range(n_states - 1):
            h_curr = reg_rows[i][3] or 0
            a_curr = reg_rows[i][4] or 0
            h_next = reg_rows[i + 1][3] or 0
            a_next = reg_rows[i + 1][4] or 0

            h_delta = h_next - h_curr
            a_delta = a_next - a_curr

            if h_delta > 0 and a_delta > 0:
                # Both teams scored simultaneously — prioritize home
                h_delta = min(h_delta, 3)
                event = h_delta  # 1, 2, or 3
            elif h_delta > 0:
                h_delta = min(h_delta, 3)
                event = h_delta  # 1=home+1, 2=home+2, 3=home+3
            elif a_delta > 0:
                a_delta = min(a_delta, 3)
                event = a_delta + 3  # 4=away+1, 5=away+2, 6=away+3
            else:
                event = 0  # no_score

            score_events.append(event)

        score_events = torch.tensor(score_events, dtype=torch.int64)

        # Clock targets: next state's clock_norm
        clock_targets = states[1:, 1].clone()  # column 1 = clock_norm

        # Final scores at regulation end
        final_h = reg_rows[-1][3] or 0
        final_a = reg_rows[-1][4] or 0

        return {
            "states": states,
            "score_events": score_events,
            "clock_targets": clock_targets,
            "n_states": n_states,
            "final_home_score": final_h,
            "final_away_score": final_a,
            "final_margin": final_h - final_a,
        }

    @staticmethod
    def _extract_enriched_features(
        players_data_str: Optional[str],
        h_score: int,
        a_score: int,
        period: int,
        seconds_remaining: float,
        home_run: int,
        away_run: int,
    ) -> list[float]:
        """Extract 10 enriched features from players_data JSON and game dynamics.

        Player scoring features (6):
          - Top-1 scorer cumulative points per team (2, /50.0)
          - Scoring concentration (Herfindahl index) per team (2, default 0.0)
          - Active scorer count per team (2, /15.0)

        Game dynamics features (4):
          - clutch_indicator: 1.0 if Q4+, clock < 120s, |margin| <= 10
          - is_second_half: 1.0 if period >= 3
          - scoring_run_home (/5.0)
          - scoring_run_away (/5.0)

        Returns list of 10 floats.
        """
        # Parse players_data
        home_top1 = 0.0
        away_top1 = 0.0
        home_hhi = 0.0
        away_hhi = 0.0
        home_scorers = 0.0
        away_scorers = 0.0

        if players_data_str:
            try:
                pd = json.loads(players_data_str)
            except (json.JSONDecodeError, TypeError):
                pd = {"home": {}, "away": {}}
        else:
            pd = {"home": {}, "away": {}}

        for side, prefix in [("home", "home"), ("away", "away")]:
            team_players = pd.get(side, {})
            if not team_players:
                continue

            pts_list = [p.get("points", 0) for p in team_players.values()]
            total_pts = sum(pts_list)
            n_scorers = sum(1 for p in pts_list if p > 0)

            top1 = max(pts_list) if pts_list else 0

            # Herfindahl index: sum of (individual_share)^2
            # Ranges from 1/n (perfectly even) to 1.0 (one player has all)
            if total_pts > 0:
                hhi = sum((p / total_pts) ** 2 for p in pts_list if p > 0)
            else:
                hhi = 0.0

            if side == "home":
                home_top1 = top1
                home_hhi = hhi
                home_scorers = n_scorers
            else:
                away_top1 = top1
                away_hhi = hhi
                away_scorers = n_scorers

        # Game dynamics
        margin = h_score - a_score
        clutch = (
            1.0
            if (period >= 4 and seconds_remaining < 120.0 and abs(margin) <= 10)
            else 0.0
        )
        is_second_half = 1.0 if period >= 3 else 0.0

        return [
            home_top1 / 50.0,
            away_top1 / 50.0,
            home_hhi,
            away_hhi,
            home_scorers / 15.0,
            away_scorers / 15.0,
            clutch,
            is_second_half,
            home_run / 5.0,
            away_run / 5.0,
        ]

    def _build_context_cache(self):
        """Build season-level context from Games + PlayerBox.

        Produces:
          - season_games: {(team_abbrev, season) -> [game_id, ...]}
          - game_features: {game_id -> {home_team, away_team, home_score, away_score,
                            date, home_player_stats, away_player_stats}}
          - player_id_map: {person_id -> contiguous index}
          - team_id_map: {team_abbrev -> index}
        """
        logger.info("Building context cache...")

        # Fetch game metadata (scores come from TeamBox, not Games table)
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT g.game_id, g.home_team, g.away_team,
                   g.date_time_utc, g.season
            FROM Games g
            WHERE g.status = 3
              AND g.season IN ({season_placeholders})
              AND g.game_id NOT LIKE '003%'
            ORDER BY g.date_time_utc
        """
        with get_db(self.db_path) as conn:
            game_rows = conn.execute(query, self.seasons).fetchall()

        logger.info(f"Fetched {len(game_rows)} games for context cache")

        # Batch-fetch scores from TeamBox
        game_scores = self._batch_query_scores([r[0] for r in game_rows])
        n_missing_scores = len(game_rows) - len(game_scores)
        logger.info(f"Fetched scores for {len(game_scores)} games from TeamBox")
        if n_missing_scores > 0:
            logger.warning(
                f"Missing TeamBox scores for {n_missing_scores} games — will default to 0"
            )

        # Build season_games index and collect game metadata
        season_games = defaultdict(list)
        game_meta = {}

        for game_id, home, away, date_utc, season in game_rows:
            home = normalize_team(home)
            away = normalize_team(away)
            game_date = str(date_utc)[:10] if date_utc else ""

            scores = game_scores.get(game_id, {})
            h_score = scores.get(home, 0)
            a_score = scores.get(away, 0)

            game_meta[game_id] = {
                "home_team": home,
                "away_team": away,
                "home_score": h_score,
                "away_score": a_score,
                "date": game_date,
                "season": season,
            }

            season_games[(home, season)].append(game_id)
            season_games[(away, season)].append(game_id)

        # Batch fetch PlayerBox data
        game_id_list = list(game_meta.keys())
        playerbox = self._batch_query_playerbox(game_id_list)

        n_with_pb = sum(1 for gid in game_id_list if gid in playerbox)
        logger.info(f"PlayerBox data: {n_with_pb}/{len(game_id_list)} games")

        # Build game_features with player stats
        game_features = {}
        for game_id, meta in tqdm(
            game_meta.items(), desc="Context features", unit="game"
        ):
            home = meta["home_team"]
            away = meta["away_team"]

            pb = playerbox.get(game_id, {})
            home_stats = pb.get(home, [])
            away_stats = pb.get(away, [])

            # Collect player IDs (tuples are 2 or 3 elements depending on enrich_states)
            for entry in home_stats:
                self.all_player_ids.add(entry[0])
            for entry in away_stats:
                self.all_player_ids.add(entry[0])

            game_features[game_id] = {
                "home_team": home,
                "away_team": away,
                "home_score": meta["home_score"],
                "away_score": meta["away_score"],
                "date": meta["date"],
                "home_player_stats": home_stats,
                "away_player_stats": away_stats,
            }

        # Build player_id_map: NBA person_id -> contiguous index (0 to N-1)
        sorted_pids = sorted(self.all_player_ids)
        player_id_map = {pid: idx for idx, pid in enumerate(sorted_pids)}

        # team_id_map is just TEAM_TO_IDX
        team_id_map = dict(TEAM_TO_IDX)

        # Save context caches
        season_games_dict = {f"{k[0]}|{k[1]}": v for k, v in season_games.items()}
        torch.save(season_games_dict, self.context_dir / "season_games.pt")
        logger.info(f"Saved season_games: {len(season_games_dict)} team-season entries")

        torch.save(game_features, self.context_dir / "game_features.pt")
        logger.info(f"Saved game_features: {len(game_features)} games")

        torch.save(player_id_map, self.context_dir / "player_id_map.pt")
        logger.info(
            f"Saved player_id_map: {len(player_id_map)} players (indices 0-{len(player_id_map) - 1})"
        )

        torch.save(team_id_map, self.context_dir / "team_id_map.pt")
        logger.info(f"Saved team_id_map: {len(team_id_map)} teams")

        # Log per-season counts
        season_counts = defaultdict(int)
        for meta in game_meta.values():
            season_counts[meta["season"]] += 1
        for s, c in sorted(season_counts.items()):
            logger.info(f"  {s}: {c} games")

    def _batch_query_scores(self, game_ids: list[str]) -> dict:
        """Batch query TeamBox for game scores.

        Returns {game_id: {team_abbrev: pts}}.
        """
        if not game_ids:
            return {}

        result: dict[str, dict[str, int]] = {}
        chunk_size = 500

        with get_db(self.db_path) as conn:
            for i in range(0, len(game_ids), chunk_size):
                chunk = game_ids[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))

                rows = conn.execute(
                    f"""
                    SELECT tb.game_id, t.abbreviation, tb.pts
                    FROM TeamBox tb
                    JOIN Teams t ON tb.team_id = t.team_id
                    WHERE tb.game_id IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()

                for game_id, abbrev, pts in rows:
                    abbrev = normalize_team(abbrev)
                    if game_id not in result:
                        result[game_id] = {}
                    result[game_id][abbrev] = pts or 0

        return result

    def _batch_query_playerbox(self, game_ids: list[str]) -> dict:
        """Batch query PlayerBox for multiple games.

        When enrich_states=False:
          Returns {game_id: {team_abbrev: [(player_id, [16 floats]), ...]}}.
        When enrich_states=True:
          Returns {game_id: {team_abbrev: [(player_id, [16 floats], position_idx), ...]}}.
          Also builds self.position_lookup for historical position fallback.

        Limited to 13 players per team per game, sorted by minutes DESC.
        """
        if not game_ids:
            return {}

        result = {}
        chunk_size = 500
        max_players = 13

        # When enriching, also fetch position and build historical lookup
        if self.enrich_states:
            self._build_position_lookup()

        with get_db(self.db_path) as conn:
            for i in tqdm(
                range(0, len(game_ids), chunk_size),
                desc="PlayerBox queries",
                unit="chunk",
            ):
                chunk = game_ids[i : i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))

                if self.enrich_states:
                    select_sql = f"""
                        SELECT pb.game_id, pb.player_id, t.abbreviation,
                               pb.min, pb.pts, pb.oreb, pb.dreb, pb.ast, pb.stl, pb.blk,
                               pb.tov, pb.pf, pb.fga, pb.fgm, pb.fg3a, pb.fg3m, pb.fta,
                               pb.ftm, pb.plus_minus, pb.position
                        FROM PlayerBox pb
                        JOIN Teams t ON pb.team_id = t.team_id
                        WHERE pb.game_id IN ({placeholders})
                          AND pb.min > 0
                        ORDER BY pb.game_id, t.abbreviation, pb.min DESC
                    """
                else:
                    select_sql = f"""
                        SELECT pb.game_id, pb.player_id, t.abbreviation,
                               pb.min, pb.pts, pb.oreb, pb.dreb, pb.ast, pb.stl, pb.blk,
                               pb.tov, pb.pf, pb.fga, pb.fgm, pb.fg3a, pb.fg3m, pb.fta,
                               pb.ftm, pb.plus_minus
                        FROM PlayerBox pb
                        JOIN Teams t ON pb.team_id = t.team_id
                        WHERE pb.game_id IN ({placeholders})
                          AND pb.min > 0
                        ORDER BY pb.game_id, t.abbreviation, pb.min DESC
                    """

                rows = conn.execute(select_sql, chunk).fetchall()

                for row in rows:
                    game_id = row[0]
                    player_id = row[1]
                    team_abbrev = normalize_team(row[2])

                    # Extract 16 stats, coalesce NULL to 0.0
                    stats = [
                        float(row[3 + j]) if row[3 + j] is not None else 0.0
                        for j in range(16)
                    ]

                    if game_id not in result:
                        result[game_id] = {}
                    if team_abbrev not in result[game_id]:
                        result[game_id][team_abbrev] = []

                    if self.enrich_states:
                        # Position is at index 19 (after 3 header + 16 stats)
                        raw_pos = row[19]
                        pos_idx = self._resolve_position(player_id, raw_pos)
                        result[game_id][team_abbrev].append((player_id, stats, pos_idx))
                    else:
                        result[game_id][team_abbrev].append((player_id, stats))

        # Trim to max_players per team (already sorted by minutes DESC)
        for game_id in result:
            for team in result[game_id]:
                result[game_id][team] = result[game_id][team][:max_players]

        return result

    def _build_position_lookup(self):
        """Build a historical position lookup from all PlayerBox rows with position data.

        Creates self.position_lookup: {player_id -> position_idx} using the most
        frequently occurring position for each player.
        """
        logger.info("Building historical position lookup...")

        with get_db(self.db_path) as conn:
            rows = conn.execute("""
                SELECT player_id, position, COUNT(*) as cnt
                FROM PlayerBox
                WHERE position IS NOT NULL AND position != '' AND position != 'NA'
                GROUP BY player_id, position
                ORDER BY player_id, cnt DESC
            """).fetchall()

        # For each player, take the position with the highest count
        player_positions: dict[int, tuple[str, int]] = {}  # pid -> (best_pos, best_cnt)
        for player_id, position, cnt in rows:
            if (
                player_id not in player_positions
                or cnt > player_positions[player_id][1]
            ):
                player_positions[player_id] = (position, cnt)

        self.position_lookup = {}
        for pid, (pos_str, _) in player_positions.items():
            self.position_lookup[pid] = POSITION_MAP.get(pos_str, POSITION_UNKNOWN)

        logger.info(
            f"Position lookup built: {len(self.position_lookup)} players, "
            f"{sum(1 for v in self.position_lookup.values() if v == POSITION_UNKNOWN)} unknown"
        )

    def _resolve_position(self, player_id: int, raw_position: Optional[str]) -> int:
        """Resolve a player's position, falling back to historical lookup.

        Args:
            player_id: The player's ID.
            raw_position: The position string from the current game's PlayerBox row.

        Returns:
            Position index (0-4).
        """
        # Try the current game's position first
        if raw_position and raw_position.strip() and raw_position.strip() != "NA":
            pos_idx = POSITION_MAP.get(raw_position.strip(), None)
            if pos_idx is not None:
                return pos_idx

        # Fall back to historical lookup
        return self.position_lookup.get(player_id, POSITION_UNKNOWN)

    def _build_rolling_stats_cache(self):
        """Build pre-computed rolling aggregate stats per team per game.

        Computes N features per team using a 20-game rolling window (shift-by-1
        to prevent leakage). N=24 by default, N=28 when n_rolling_features=28.

        Base 24 features cover scoring, efficiency, pace, momentum, and schedule.
        Extended 4 features (indices 24-27): opponent_avg_win_pct, is_back_to_back,
        season_type, season_era.

        Saves to context/rolling_stats.pt as {game_id: {
            home_stats: [N floats], away_stats: [N floats],
            home_team_idx: int, away_team_idx: int
        }}
        """
        import pandas as pd

        n_feat = self.n_rolling_features
        logger.info(f"Building rolling stats cache ({n_feat} features)...")

        # Fetch TeamBox + Games data
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT g.game_id, g.date_time_utc, g.home_team, g.away_team, g.season,
                   t.abbreviation as team_abbrev,
                   tb.pts, tb.pts_allowed, tb.fga, tb.fgm, tb.fg3a, tb.fg3m,
                   tb.fta, tb.ftm, tb.tov, tb.ast, tb.reb,
                   g.season_type
            FROM TeamBox tb
            JOIN Games g ON tb.game_id = g.game_id
            JOIN Teams t ON tb.team_id = t.team_id
            WHERE g.season IN ({season_placeholders})
              AND g.status = 3
              AND g.game_id NOT LIKE '003%'
            ORDER BY g.date_time_utc
        """
        with get_db(self.db_path) as conn:
            rows = conn.execute(query, self.seasons).fetchall()

        cols = [
            "game_id",
            "date",
            "home_team",
            "away_team",
            "season",
            "team_abbrev",
            "pts",
            "pts_allowed",
            "fga",
            "fgm",
            "fg3a",
            "fg3m",
            "fta",
            "ftm",
            "tov",
            "ast",
            "reb",
            "season_type",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["team_abbrev"] = df["team_abbrev"].map(
            lambda x: HISTORICAL_TO_CURRENT.get(x, x)
        )
        df["home_team"] = df["home_team"].map(lambda x: HISTORICAL_TO_CURRENT.get(x, x))
        df["away_team"] = df["away_team"].map(lambda x: HISTORICAL_TO_CURRENT.get(x, x))
        df["date"] = pd.to_datetime(df["date"])

        # Determine if team was home in this game
        df["is_home"] = (df["team_abbrev"] == df["home_team"]).astype(float)

        # Opponent for each row
        df["opponent"] = df.apply(
            lambda r: (
                r["away_team"] if r["team_abbrev"] == r["home_team"] else r["home_team"]
            ),
            axis=1,
        )

        # Derived stats
        df["margin"] = df["pts"] - df["pts_allowed"]
        df["total"] = df["pts"] + df["pts_allowed"]
        df["win"] = (df["margin"] > 0).astype(float)
        # Effective FG%
        df["efg"] = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].clip(lower=1)
        # 3PT attempt rate
        df["fg3a_rate"] = df["fg3a"] / df["fga"].clip(lower=1)
        # FT rate
        df["ft_rate"] = df["fta"] / df["fga"].clip(lower=1)
        # True shooting %
        df["ts_pct"] = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"]).clip(lower=1))
        # Simplified pace: possessions ≈ fga - 0.44*fta + tov
        df["pace"] = df["fga"] - 0.44 * df["fta"] + df["tov"]
        # Assist ratio
        df["ast_ratio"] = df["ast"] / df["fgm"].clip(lower=1)
        # Turnover rate
        df["tov_rate"] = df["tov"] / df["pace"].clip(lower=1)

        # Pre-compute per-team season win% for opponent strength lookup
        # {(team, season) -> cumulative_win_pct at each game index}
        team_season_win_pcts = {}
        if n_feat >= 28:
            for (team, season), group in df.groupby(["team_abbrev", "season"]):
                g = group.sort_values("date").reset_index(drop=True)
                wins = g["win"].values
                # Cumulative win% at each game (shifted: game i uses wins through i-1)
                cum_wins = np.cumsum(wins)
                cum_games = np.arange(1, len(wins) + 1, dtype=np.float64)
                # Shifted: win% before each game
                shifted_pct = np.zeros(len(wins))
                shifted_pct[1:] = cum_wins[:-1] / cum_games[:-1]
                shifted_pct[0] = 0.5  # default for first game
                team_season_win_pcts[(team, season)] = dict(
                    zip(g["game_id"].tolist(), shifted_pct.tolist())
                )

        # Sort and compute per-team rolling stats
        df = df.sort_values(["team_abbrev", "season", "date"])
        window = 20
        min_p = 3

        rolling_features = {}  # game_id -> {team_abbrev -> features_list}

        for (team, season), group in df.groupby(["team_abbrev", "season"]):
            g = group.reset_index(drop=True)
            n = len(g)
            if n < min_p:
                continue

            # Compute rolling stats (shifted by 1 to exclude current game)
            for i in range(min_p, n):
                window_df = g.iloc[max(0, i - window) : i]  # prior games only

                game_id = g.iloc[i]["game_id"]
                game_date = g.iloc[i]["date"]

                # Rest days
                prev_date = g.iloc[i - 1]["date"]
                rest_days = (game_date - prev_date).days
                rest_days = min(max(rest_days, 0), 7)

                # Win streak
                recent_wins = window_df["win"].values
                streak = 0
                for w in reversed(recent_wins):
                    if w == 1.0:
                        streak += 1
                    else:
                        streak = -(len(recent_wins) - sum(recent_wins))
                        # Count consecutive losses from end
                        streak = 0
                        for w2 in reversed(recent_wins):
                            if w2 == 0.0:
                                streak -= 1
                            else:
                                break
                        break

                # Compute EWM stats
                ewm10 = (
                    window_df["margin"]
                    .ewm(span=min(10, len(window_df)), min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ewm20 = (
                    window_df["margin"]
                    .ewm(span=min(20, len(window_df)), min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ewm_win10 = (
                    window_df["win"]
                    .ewm(span=min(10, len(window_df)), min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ewm_ortg = (
                    window_df["pts"]
                    .ewm(span=min(10, len(window_df)), min_periods=1)
                    .mean()
                    .iloc[-1]
                )
                ewm_drtg = (
                    window_df["pts_allowed"]
                    .ewm(span=min(10, len(window_df)), min_periods=1)
                    .mean()
                    .iloc[-1]
                )

                # 24 base features (pre-normalized to ~[0, 1] or [-1, 1])
                features = [
                    window_df["pts"].mean() / 150.0,  # 0: avg pts scored
                    window_df["pts_allowed"].mean() / 150.0,  # 1: avg pts allowed
                    window_df["margin"].mean() / 50.0,  # 2: avg margin
                    window_df["total"].mean() / 300.0,  # 3: avg total
                    (
                        window_df["pts"].std() / 30.0 if len(window_df) > 1 else 0.0
                    ),  # 4: pts std
                    (
                        window_df["margin"].std() / 30.0 if len(window_df) > 1 else 0.0
                    ),  # 5: margin std
                    window_df["win"].mean(),  # 6: win pct
                    max(-10, min(10, streak)) / 10.0,  # 7: streak
                    window_df["efg"].mean(),  # 8: eFG%
                    window_df["fg3a_rate"].mean(),  # 9: 3PT attempt rate
                    window_df["ft_rate"].mean(),  # 10: FT rate
                    window_df["ts_pct"].mean(),  # 11: TS%
                    window_df["pace"].mean() / 100.0,  # 12: pace
                    window_df["ast_ratio"].mean() / 3.0,  # 13: ast ratio
                    window_df["pts_allowed"].mean()
                    / 150.0,  # 14: defensive rating proxy
                    window_df["tov_rate"].mean(),  # 15: TOV rate
                    rest_days / 7.0,  # 16: rest days
                    float(g.iloc[i]["is_home"]),  # 17: is home
                    i / 82.0,  # 18: games played (season progress)
                    ewm10 / 50.0,  # 19: EWM margin span 10
                    ewm20 / 50.0,  # 20: EWM margin span 20
                    ewm_win10,  # 21: EWM win% span 10
                    ewm_ortg / 120.0,  # 22: EWM offensive rating
                    ewm_drtg / 120.0,  # 23: EWM defensive rating
                ]

                # Extended features (24-27) for Exp 5
                if n_feat >= 28:
                    # 24: opponent_avg_win_pct — mean win% of opponents in window
                    opp_win_pcts = []
                    for _, wrow in window_df.iterrows():
                        opp = wrow["opponent"]
                        opp_season = wrow["season"]
                        opp_gid = wrow["game_id"]
                        opp_pcts = team_season_win_pcts.get((opp, opp_season), {})
                        opp_wp = opp_pcts.get(opp_gid, 0.5)
                        opp_win_pcts.append(opp_wp)
                    opponent_avg_wp = (
                        float(np.mean(opp_win_pcts)) if opp_win_pcts else 0.5
                    )

                    # 25: is_back_to_back — 1 if rest_days == 1
                    is_b2b = 1.0 if rest_days == 1 else 0.0

                    # 26: season_type — 0.0 regular, 1.0 playoffs
                    raw_st = g.iloc[i]["season_type"]
                    season_type_val = 1.0 if raw_st == "Post Season" else 0.0

                    # 27: season_era — year / 2030.0
                    try:
                        year = int(season.split("-")[0])
                    except (ValueError, IndexError):
                        year = 2020
                    season_era = year / 2030.0

                    features.extend(
                        [
                            opponent_avg_wp,  # 24
                            is_b2b,  # 25
                            season_type_val,  # 26
                            season_era,  # 27
                        ]
                    )

                if game_id not in rolling_features:
                    rolling_features[game_id] = {}
                rolling_features[game_id][team] = features

        # Now build per-game dict keyed by game_id
        rolling_stats = {}
        for game_id, teams_data in rolling_features.items():
            # Look up home/away teams from the game
            game_rows_df = df[df["game_id"] == game_id]
            if game_rows_df.empty:
                continue

            home_team = game_rows_df.iloc[0]["home_team"]
            away_team = game_rows_df.iloc[0]["away_team"]

            if home_team not in teams_data or away_team not in teams_data:
                continue

            rolling_stats[game_id] = {
                "home_stats": teams_data[home_team],
                "away_stats": teams_data[away_team],
                "home_team_idx": TEAM_TO_IDX.get(home_team, 0),
                "away_team_idx": TEAM_TO_IDX.get(away_team, 0),
            }

        torch.save(rolling_stats, self.context_dir / "rolling_stats.pt")
        logger.info(
            f"Saved rolling_stats: {len(rolling_stats)} games with {n_feat} features per team"
        )

    def _save_metadata(self):
        """Save dataset statistics and build metadata."""
        state_dim = 17 if self.enrich_states else 7
        metadata = {
            "seasons": self.seasons,
            "total_games": self.stats["total_games"],
            "games_with_states": self.stats["games_with_states"],
            "games_skipped": self.stats["games_skipped"],
            "total_states": int(self.stats["total_states"]),
            "avg_states_per_game": (
                self.stats["total_states"] / self.stats["games_with_states"]
                if self.stats["games_with_states"] > 0
                else 0.0
            ),
            "n_players": len(self.all_player_ids),
            "enrich_states": self.enrich_states,
            "state_dim": state_dim,
            "n_rolling_features": self.n_rolling_features,
            "n_positions_in_lookup": len(self.position_lookup),
            "class_distribution": {
                "no_score": int(self.stats["class_counts"][0]),
                "home+1": int(self.stats["class_counts"][1]),
                "home+2": int(self.stats["class_counts"][2]),
                "home+3": int(self.stats["class_counts"][3]),
                "away+1": int(self.stats["class_counts"][4]),
                "away+2": int(self.stats["class_counts"][5]),
                "away+3": int(self.stats["class_counts"][6]),
            },
        }

        meta_path = self.cache_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")


def build_cache_cli():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build generative model cache")
    parser.add_argument(
        "--db-path", default="data/NBA_AI_full.sqlite", help="Path to SQLite database"
    )
    parser.add_argument(
        "--cache-dir", default="data/generative_cache", help="Output cache directory"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to cache (default: 2008-2009 through 2025-2026)",
    )
    args = parser.parse_args()

    seasons = args.seasons or [f"{y}-{y + 1}" for y in range(2008, 2026)]

    builder = GenerativeCacheBuilder(args.db_path, args.cache_dir, seasons)
    builder.build()


if __name__ == "__main__":
    build_cache_cli()
