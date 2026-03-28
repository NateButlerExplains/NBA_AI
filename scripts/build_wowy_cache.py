#!/usr/bin/env python3
"""
Build WOWY (With-Or-Without-You) cache from PBP data.

Processes all PBP games through LineupTracker to extract stints, then computes:
1. Pairwise player WOWY statistics (net rating together vs apart)
2. 5-man lineup stint aggregates (offensive/defensive ratings)

Output: SQLite tables in data/l2_cache/wowy.sqlite
    - Stints: raw stint-level data (team, players, duration, points, possessions)
    - PairwiseWOWY: per-pair-per-season aggregated stats
    - LineupStints: per-5man-lineup-per-season aggregated stats

Resume-capable: tracks processed game_ids in a progress table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db
from src.pbp.event_parser import parse_events
from src.pbp.lineup_tracker import LineupTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "l2_cache"
WOWY_DB_PATH = CACHE_DIR / "wowy.sqlite"

# Historical abbreviation -> modern franchise team_id
ABBR_TO_TEAM_ID = {
    "NJN": 1610612751,
    "BKN": 1610612751,
    "SEA": 1610612760,
    "OKC": 1610612760,
    "VAN": 1610612763,
    "MEM": 1610612763,
    "CHH": 1610612766,
    "CHA": 1610612766,
    "NOH": 1610612740,
    "NOK": 1610612740,
    "NOP": 1610612740,
    "WSB": 1610612764,
    "WAS": 1610612764,
}

# Period duration in seconds (regulation = 12 min, OT = 5 min)
REGULAR_PERIOD_SECS = 720.0
OT_PERIOD_SECS = 300.0

# Minimum valid lineup size to track (should be 5, but allow 4 for edge cases)
MIN_LINEUP_SIZE = 5

# Batch size for committing stint data
COMMIT_BATCH = 200

LOG_INTERVAL = 500


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------


def init_wowy_db(db_path: Path) -> sqlite3.Connection:
    """Create WOWY database with required tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS Stints (
            stint_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            season TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            player_ids TEXT NOT NULL,       -- sorted comma-separated player IDs
            n_players INTEGER NOT NULL DEFAULT 5,
            period INTEGER NOT NULL,
            start_clock REAL NOT NULL,      -- seconds remaining at stint start
            end_clock REAL NOT NULL,        -- seconds remaining at stint end
            duration_secs REAL NOT NULL,    -- stint duration in seconds
            pts_for INTEGER NOT NULL DEFAULT 0,
            pts_against INTEGER NOT NULL DEFAULT 0,
            fga INTEGER NOT NULL DEFAULT 0,
            fta INTEGER NOT NULL DEFAULT 0,
            tov INTEGER NOT NULL DEFAULT 0,
            oreb INTEGER NOT NULL DEFAULT 0,
            dreb INTEGER NOT NULL DEFAULT 0,
            opp_fga INTEGER NOT NULL DEFAULT 0,
            opp_fta INTEGER NOT NULL DEFAULT 0,
            opp_tov INTEGER NOT NULL DEFAULT 0,
            opp_oreb INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS PairwiseWOWY (
            player_a INTEGER NOT NULL,
            player_b INTEGER NOT NULL,
            season TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            minutes_together REAL NOT NULL DEFAULT 0,
            minutes_a_only REAL NOT NULL DEFAULT 0,
            minutes_b_only REAL NOT NULL DEFAULT 0,
            pts_for_together INTEGER NOT NULL DEFAULT 0,
            pts_against_together INTEGER NOT NULL DEFAULT 0,
            poss_together REAL NOT NULL DEFAULT 0,
            pts_for_a_only INTEGER NOT NULL DEFAULT 0,
            pts_against_a_only INTEGER NOT NULL DEFAULT 0,
            poss_a_only REAL NOT NULL DEFAULT 0,
            pts_for_b_only INTEGER NOT NULL DEFAULT 0,
            pts_against_b_only INTEGER NOT NULL DEFAULT 0,
            poss_b_only REAL NOT NULL DEFAULT 0,
            stints_together INTEGER NOT NULL DEFAULT 0,
            games_together INTEGER NOT NULL DEFAULT 0,
            net_rtg_together REAL,
            net_rtg_a_only REAL,
            net_rtg_b_only REAL,
            PRIMARY KEY (player_a, player_b, season, team_id)
        );

        CREATE TABLE IF NOT EXISTS LineupStints (
            lineup_hash TEXT NOT NULL,
            season TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            player_ids TEXT NOT NULL,       -- sorted comma-separated
            total_minutes REAL NOT NULL DEFAULT 0,
            total_pts_for INTEGER NOT NULL DEFAULT 0,
            total_pts_against INTEGER NOT NULL DEFAULT 0,
            total_poss REAL NOT NULL DEFAULT 0,
            off_rating REAL,
            def_rating REAL,
            n_stints INTEGER NOT NULL DEFAULT 0,
            n_games INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (lineup_hash, season)
        );

        CREATE TABLE IF NOT EXISTS ProcessedGames (
            game_id TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL,
            n_stints INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ok'
        );

        CREATE INDEX IF NOT EXISTS idx_stints_game ON Stints(game_id);
        CREATE INDEX IF NOT EXISTS idx_stints_season ON Stints(season);
        CREATE INDEX IF NOT EXISTS idx_stints_team ON Stints(team_id);
        CREATE INDEX IF NOT EXISTS idx_pairwise_player_a ON PairwiseWOWY(player_a);
        CREATE INDEX IF NOT EXISTS idx_pairwise_player_b ON PairwiseWOWY(player_b);
        CREATE INDEX IF NOT EXISTS idx_lineup_team ON LineupStints(team_id);
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Stint extraction
# ---------------------------------------------------------------------------


def _period_duration(period: int) -> float:
    """Get total seconds for a period."""
    if period <= 4:
        return REGULAR_PERIOD_SECS
    return OT_PERIOD_SECS  # OT periods


def _clock_to_elapsed(period: int, clock_secs: float) -> float:
    """Convert (period, clock_remaining) to total elapsed game seconds."""
    # Elapsed in completed prior periods
    elapsed = 0.0
    for p in range(1, period):
        elapsed += _period_duration(p)
    # Add elapsed in current period
    elapsed += _period_duration(period) - clock_secs
    return elapsed


def _build_player_names(
    nba_conn: sqlite3.Connection, game_id: str
) -> dict[int, tuple[int, list[str]]]:
    """
    Build player_names dict for LineupTracker from PlayerBox + GameRotationRoster.

    Returns: {player_id: (team_id, [name_variant1, name_variant2, ...])}
    """
    player_names: dict[int, tuple[int, list[str]]] = {}

    # From PlayerBox
    cursor = nba_conn.execute(
        """
        SELECT player_id, team_id, player_name
        FROM PlayerBox
        WHERE game_id = ? AND min > 0
        """,
        (game_id,),
    )
    for pid, tid_str, full_name in cursor:
        tid = int(tid_str)
        names = []
        if full_name:
            # "LeBron James" -> ["james", "l. james", "lebron james"]
            parts = full_name.strip().split()
            if parts:
                last = parts[-1].lower()
                names.append(last)
                if len(parts) >= 2:
                    first_initial = parts[0][0].upper()
                    names.append(f"{first_initial}. {last}")
                    names.append(full_name.lower())
                    # Also first name for unique names
                    names.append(parts[0].lower())
        player_names[pid] = (tid, names)

    # From GameRotationRoster (for problematic games)
    cursor = nba_conn.execute(
        "SELECT person_id, team_id, player_first, player_last FROM GameRotationRoster WHERE game_id = ?",
        (game_id,),
    )
    for pid, tid, first_name, last_name in cursor:
        if pid in player_names:
            # Extend existing name variants
            existing_tid, existing_names = player_names[pid]
            if last_name:
                ln = last_name.lower()
                if ln not in existing_names:
                    existing_names.append(ln)
            if first_name and last_name:
                fi = first_name[0].upper()
                variant = f"{fi}. {last_name.lower()}"
                if variant not in existing_names:
                    existing_names.append(variant)
                full = f"{first_name} {last_name}".lower()
                if full not in existing_names:
                    existing_names.append(full)
        else:
            names = []
            if last_name:
                names.append(last_name.lower())
            if first_name and last_name:
                names.append(f"{first_name[0].upper()}. {last_name.lower()}")
                names.append(f"{first_name} {last_name}".lower())
            player_names[pid] = (tid, names)

    return player_names


def _get_scoring_points(evt) -> int:
    """Determine points scored on this event (0 if not a scoring event)."""
    if evt.is_field_goal and evt.is_made:
        return 3 if evt.is_3pt else 2
    if evt.is_free_throw and evt.is_ft_made:
        return 1
    return 0


def extract_stints_for_game(
    nba_conn: sqlite3.Connection,
    game_id: str,
    home_team_id: int,
    away_team_id: int,
    season: str,
) -> list[dict]:
    """
    Extract all stints for a game.

    A stint is a contiguous stretch where the same 5 players are on court
    for a given team. Stints end when a substitution changes the lineup.

    Returns list of stint dicts.
    """
    # Load PBP events
    cursor = nba_conn.execute(
        "SELECT log_data FROM PBP_Logs WHERE game_id = ? ORDER BY play_id",
        (game_id,),
    )
    events_raw = []
    for (log_data,) in cursor:
        try:
            events_raw.append(json.loads(log_data))
        except (json.JSONDecodeError, TypeError):
            continue

    if not events_raw:
        return []

    # Parse events
    events = parse_events(events_raw)
    if not events:
        return []

    # Build player name lookup for LineupTracker
    player_names = _build_player_names(nba_conn, game_id)

    # Initialize tracker
    tracker = LineupTracker(events, home_team_id, away_team_id, player_names)

    # Process all periods
    periods = sorted(set(e.period for e in events if e.period > 0))
    if not periods:
        return []

    all_stints = []

    for period in periods:
        tracker.initialize_period(period)

        # Get initial lineup snapshot
        home_lineup = frozenset(tracker.on_court.get(home_team_id, set()))
        away_lineup = frozenset(tracker.on_court.get(away_team_id, set()))

        # Current stint tracking for each team
        stint_data = {
            home_team_id: _new_stint(
                game_id,
                season,
                home_team_id,
                home_lineup,
                period,
                _period_duration(period),  # start of period = full clock
            ),
            away_team_id: _new_stint(
                game_id,
                season,
                away_team_id,
                away_lineup,
                period,
                _period_duration(period),
            ),
        }

        period_events = [e for e in events if e.period == period]

        # Group consecutive substitution events at the same clock time for the
        # same team so we process them atomically (v3 format has separate in/out
        # events that must be paired to avoid transient 4/6-player lineups).
        i = 0
        while i < len(period_events):
            evt = period_events[i]

            if evt.is_period_start:
                i += 1
                continue

            # Collect a batch of substitution events at the same clock time
            if evt.is_substitution and evt.team_id in (home_team_id, away_team_id):
                sub_batch: list = [evt]
                clock = evt.clock_seconds
                j = i + 1
                while j < len(period_events):
                    next_evt = period_events[j]
                    if (
                        next_evt.is_substitution
                        and next_evt.team_id == evt.team_id
                        and abs(next_evt.clock_seconds - clock) < 0.5
                    ):
                        sub_batch.append(next_evt)
                        j += 1
                    else:
                        break

                sub_team = evt.team_id

                # Close current stint for this team
                old_stint = stint_data[sub_team]
                if old_stint["players"] and len(old_stint["players"]) >= 4:
                    old_stint["end_clock"] = clock
                    old_stint["duration_secs"] = (
                        old_stint["start_clock"] - old_stint["end_clock"]
                    )
                    if old_stint["duration_secs"] > 0.1:
                        all_stints.append(old_stint)

                # Process all subs in the batch
                for sub_evt in sub_batch:
                    tracker.process_substitution(sub_evt)

                # Start new stint with the post-substitution lineup
                new_lineup = frozenset(tracker.on_court.get(sub_team, set()))
                stint_data[sub_team] = _new_stint(
                    game_id,
                    season,
                    sub_team,
                    new_lineup,
                    period,
                    clock,
                )

                i = j
                continue

            # Non-substitution event: record stats on the current stints

            # Determine scoring
            pts = _get_scoring_points(evt)
            scoring_team = evt.team_id if pts > 0 else 0

            # Update stint stats for scoring
            if pts > 0 and scoring_team in (home_team_id, away_team_id):
                opp_team = (
                    away_team_id if scoring_team == home_team_id else home_team_id
                )
                # Points for the scoring team
                if stint_data[scoring_team]["players"]:
                    stint_data[scoring_team]["pts_for"] += pts
                # Points against for the opponent
                if stint_data[opp_team]["players"]:
                    stint_data[opp_team]["pts_against"] += pts

            # Track possession-related events
            if evt.team_id in (home_team_id, away_team_id):
                tid = evt.team_id
                opp = away_team_id if tid == home_team_id else home_team_id

                if evt.is_field_goal and stint_data[tid]["players"]:
                    stint_data[tid]["fga"] += 1
                    stint_data[opp]["opp_fga"] += 1

                if evt.is_free_throw and stint_data[tid]["players"]:
                    stint_data[tid]["fta"] += 1
                    stint_data[opp]["opp_fta"] += 1

                if evt.is_turnover and stint_data[tid]["players"]:
                    stint_data[tid]["tov"] += 1
                    stint_data[opp]["opp_tov"] += 1

                if evt.is_rebound and stint_data[tid]["players"]:
                    if evt.reb_is_offensive:
                        stint_data[tid]["oreb"] += 1
                        stint_data[opp]["opp_oreb"] += 1
                    else:
                        stint_data[tid]["dreb"] += 1

            # Period end: close all stints
            if evt.is_period_end:
                for tid in (home_team_id, away_team_id):
                    s = stint_data[tid]
                    if s["players"] and len(s["players"]) >= 4:
                        s["end_clock"] = 0.0
                        s["duration_secs"] = s["start_clock"] - s["end_clock"]
                        if s["duration_secs"] > 0.1:
                            all_stints.append(s)
                break  # Done with this period

            i += 1

        else:
            # No period_end event found — close stints at clock 0
            for tid in (home_team_id, away_team_id):
                s = stint_data[tid]
                if s["players"] and len(s["players"]) >= 4:
                    s["end_clock"] = 0.0
                    s["duration_secs"] = s["start_clock"] - s["end_clock"]
                    if s["duration_secs"] > 0.1:
                        all_stints.append(s)

    return all_stints


def _new_stint(
    game_id: str,
    season: str,
    team_id: int,
    players: frozenset[int],
    period: int,
    start_clock: float,
) -> dict:
    """Create a new stint tracking dict."""
    return {
        "game_id": game_id,
        "season": season,
        "team_id": team_id,
        "players": players,
        "period": period,
        "start_clock": start_clock,
        "end_clock": start_clock,
        "duration_secs": 0.0,
        "pts_for": 0,
        "pts_against": 0,
        "fga": 0,
        "fta": 0,
        "tov": 0,
        "oreb": 0,
        "dreb": 0,
        "opp_fga": 0,
        "opp_fta": 0,
        "opp_tov": 0,
        "opp_oreb": 0,
    }


# ---------------------------------------------------------------------------
# Possession estimation
# ---------------------------------------------------------------------------


def estimate_possessions(stint: dict) -> float:
    """
    Estimate possessions from box-score proxies.

    Standard formula: Poss ≈ FGA - OREB + TOV + 0.44 * FTA
    """
    return stint["fga"] - stint["oreb"] + stint["tov"] + 0.44 * stint["fta"]


def estimate_opp_possessions(stint: dict) -> float:
    """Estimate opponent possessions."""
    return (
        stint["opp_fga"]
        - stint["opp_oreb"]
        + stint["opp_tov"]
        + 0.44 * stint["opp_fta"]
    )


# ---------------------------------------------------------------------------
# WOWY aggregation
# ---------------------------------------------------------------------------


def _lineup_hash(player_ids: frozenset[int]) -> str:
    """Create a stable hash for a 5-man lineup."""
    sorted_ids = sorted(player_ids)
    key = ",".join(str(p) for p in sorted_ids)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _sorted_player_ids_str(players: frozenset[int]) -> str:
    """Sorted comma-separated player IDs string."""
    return ",".join(str(p) for p in sorted(players))


# ---------------------------------------------------------------------------
# Insert stints into WOWY database
# ---------------------------------------------------------------------------


def insert_stints(wowy_conn: sqlite3.Connection, stints: list[dict]):
    """Insert stint records into the database.

    Accepts stints with 4-6 players. Stints with < 4 or > 6 are skipped.
    The n_players column records the actual lineup size so downstream
    aggregation can filter to exactly-5 lineups.
    """
    if not stints:
        return

    rows = []
    for s in stints:
        n = len(s["players"]) if s["players"] else 0
        if n < 4 or n > 6:
            continue
        rows.append(
            (
                s["game_id"],
                s["season"],
                s["team_id"],
                _sorted_player_ids_str(s["players"]),
                n,
                s["period"],
                s["start_clock"],
                s["end_clock"],
                s["duration_secs"],
                s["pts_for"],
                s["pts_against"],
                s["fga"],
                s["fta"],
                s["tov"],
                s["oreb"],
                s["dreb"],
                s["opp_fga"],
                s["opp_fta"],
                s["opp_tov"],
                s["opp_oreb"],
            )
        )

    wowy_conn.executemany(
        """INSERT INTO Stints (
            game_id, season, team_id, player_ids, n_players, period,
            start_clock, end_clock, duration_secs,
            pts_for, pts_against, fga, fta, tov, oreb, dreb,
            opp_fga, opp_fta, opp_tov, opp_oreb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


# ---------------------------------------------------------------------------
# Aggregation: build PairwiseWOWY and LineupStints from Stints
# ---------------------------------------------------------------------------


def aggregate_pairwise_wowy(wowy_conn: sqlite3.Connection):
    """
    Aggregate Stints into PairwiseWOWY table.

    For each pair of players on the same team in a season:
    - minutes_together: when both on court
    - minutes_a_only: when A on court, B off
    - minutes_b_only: when B on court, A off
    - net_rtg_*: (pts_for - pts_against) / possessions * 100
    """
    logger.info("Aggregating pairwise WOWY statistics...")

    # Clear existing data
    wowy_conn.execute("DELETE FROM PairwiseWOWY")

    # Process by season and team for memory efficiency
    cursor = wowy_conn.execute(
        "SELECT DISTINCT season, team_id FROM Stints WHERE n_players = 5 ORDER BY season, team_id"
    )
    season_teams = list(cursor)
    logger.info(f"Processing {len(season_teams)} season-team combinations")

    total_pairs = 0
    for idx, (season, team_id) in enumerate(season_teams):
        # Load all stints for this season-team
        cursor = wowy_conn.execute(
            """SELECT player_ids, duration_secs, pts_for, pts_against,
                      fga, fta, tov, oreb, game_id,
                      opp_fga, opp_fta, opp_tov, opp_oreb
               FROM Stints WHERE season = ? AND team_id = ? AND n_players = 5""",
            (season, team_id),
        )
        stints = list(cursor)

        # Build player sets and pair accumulators
        # player -> {stints where player is present}
        player_stints: dict[int, list[int]] = defaultdict(list)

        stint_records = []
        for i, (
            pids_str,
            dur,
            pf,
            pa,
            fga,
            fta,
            tov,
            oreb,
            gid,
            opp_fga,
            opp_fta,
            opp_tov,
            opp_oreb,
        ) in enumerate(stints):
            pids = frozenset(int(p) for p in pids_str.split(","))
            poss = max(0.0, fga - oreb + tov + 0.44 * fta)
            stint_records.append(
                {
                    "players": pids,
                    "duration": dur,
                    "pts_for": pf,
                    "pts_against": pa,
                    "poss": poss,
                    "game_id": gid,
                }
            )
            for pid in pids:
                player_stints[pid].append(i)

        players = sorted(player_stints.keys())

        # Build pair stats
        pair_data: dict[tuple[int, int], dict] = {}

        for ai, pa in enumerate(players):
            pa_stint_set = set(player_stints[pa])
            for bi in range(ai + 1, len(players)):
                pb = players[bi]
                pb_stint_set = set(player_stints[pb])

                together_idxs = pa_stint_set & pb_stint_set
                a_only_idxs = pa_stint_set - pb_stint_set
                b_only_idxs = pb_stint_set - pa_stint_set

                if not together_idxs:
                    continue  # No overlapping stints

                # Accumulate
                mins_tog = 0.0
                pts_for_tog = 0
                pts_ag_tog = 0
                poss_tog = 0.0
                games_tog = set()

                for si in together_idxs:
                    s = stint_records[si]
                    mins_tog += s["duration"]
                    pts_for_tog += s["pts_for"]
                    pts_ag_tog += s["pts_against"]
                    poss_tog += s["poss"]
                    games_tog.add(s["game_id"])

                mins_a = 0.0
                pts_for_a = 0
                pts_ag_a = 0
                poss_a = 0.0
                for si in a_only_idxs:
                    s = stint_records[si]
                    mins_a += s["duration"]
                    pts_for_a += s["pts_for"]
                    pts_ag_a += s["pts_against"]
                    poss_a += s["poss"]

                mins_b = 0.0
                pts_for_b = 0
                pts_ag_b = 0
                poss_b = 0.0
                for si in b_only_idxs:
                    s = stint_records[si]
                    mins_b += s["duration"]
                    pts_for_b += s["pts_for"]
                    pts_ag_b += s["pts_against"]
                    poss_b += s["poss"]

                # Net ratings (per 100 possessions)
                net_rtg_tog = (
                    (pts_for_tog - pts_ag_tog) / poss_tog * 100
                    if poss_tog > 0
                    else None
                )
                net_rtg_a = (
                    (pts_for_a - pts_ag_a) / poss_a * 100 if poss_a > 0 else None
                )
                net_rtg_b = (
                    (pts_for_b - pts_ag_b) / poss_b * 100 if poss_b > 0 else None
                )

                pair_data[(pa, pb)] = {
                    "minutes_together": mins_tog / 60.0,
                    "minutes_a_only": mins_a / 60.0,
                    "minutes_b_only": mins_b / 60.0,
                    "pts_for_together": pts_for_tog,
                    "pts_against_together": pts_ag_tog,
                    "poss_together": poss_tog,
                    "pts_for_a_only": pts_for_a,
                    "pts_against_a_only": pts_ag_a,
                    "poss_a_only": poss_a,
                    "pts_for_b_only": pts_for_b,
                    "pts_against_b_only": pts_ag_b,
                    "poss_b_only": poss_b,
                    "stints_together": len(together_idxs),
                    "games_together": len(games_tog),
                    "net_rtg_together": net_rtg_tog,
                    "net_rtg_a_only": net_rtg_a,
                    "net_rtg_b_only": net_rtg_b,
                }

        # Insert pairs
        if pair_data:
            rows = []
            for (pa, pb), d in pair_data.items():
                rows.append(
                    (
                        pa,
                        pb,
                        season,
                        team_id,
                        d["minutes_together"],
                        d["minutes_a_only"],
                        d["minutes_b_only"],
                        d["pts_for_together"],
                        d["pts_against_together"],
                        d["poss_together"],
                        d["pts_for_a_only"],
                        d["pts_against_a_only"],
                        d["poss_a_only"],
                        d["pts_for_b_only"],
                        d["pts_against_b_only"],
                        d["poss_b_only"],
                        d["stints_together"],
                        d["games_together"],
                        d["net_rtg_together"],
                        d["net_rtg_a_only"],
                        d["net_rtg_b_only"],
                    )
                )
            wowy_conn.executemany(
                """INSERT INTO PairwiseWOWY (
                    player_a, player_b, season, team_id,
                    minutes_together, minutes_a_only, minutes_b_only,
                    pts_for_together, pts_against_together, poss_together,
                    pts_for_a_only, pts_against_a_only, poss_a_only,
                    pts_for_b_only, pts_against_b_only, poss_b_only,
                    stints_together, games_together,
                    net_rtg_together, net_rtg_a_only, net_rtg_b_only
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            total_pairs += len(rows)
            wowy_conn.commit()

        if (idx + 1) % 50 == 0:
            logger.info(
                f"  Pairwise: [{idx+1}/{len(season_teams)}] "
                f"season-teams processed, {total_pairs} pairs so far"
            )

    logger.info(f"Pairwise WOWY complete: {total_pairs} total pairs")


def aggregate_lineup_stints(wowy_conn: sqlite3.Connection):
    """Aggregate Stints into LineupStints table (5-man lineup aggregates)."""
    logger.info("Aggregating 5-man lineup statistics...")

    wowy_conn.execute("DELETE FROM LineupStints")

    cursor = wowy_conn.execute("SELECT DISTINCT season FROM Stints ORDER BY season")
    seasons = [r[0] for r in cursor]

    total_lineups = 0
    for season in seasons:
        cursor = wowy_conn.execute(
            """SELECT player_ids, team_id, duration_secs, pts_for, pts_against,
                      fga, fta, tov, oreb, game_id,
                      opp_fga, opp_fta, opp_tov, opp_oreb
               FROM Stints WHERE season = ? AND n_players = 5""",
            (season,),
        )

        # Accumulate by lineup
        lineup_agg: dict[str, dict] = {}

        for (
            pids_str,
            team_id,
            dur,
            pf,
            pa,
            fga,
            fta,
            tov,
            oreb,
            gid,
            opp_fga,
            opp_fta,
            opp_tov,
            opp_oreb,
        ) in cursor:
            pids = frozenset(int(p) for p in pids_str.split(","))
            lhash = _lineup_hash(pids)

            if lhash not in lineup_agg:
                lineup_agg[lhash] = {
                    "team_id": team_id,
                    "player_ids": pids_str,
                    "total_minutes": 0.0,
                    "total_pts_for": 0,
                    "total_pts_against": 0,
                    "total_fga": 0,
                    "total_fta": 0,
                    "total_tov": 0,
                    "total_oreb": 0,
                    "total_opp_fga": 0,
                    "total_opp_fta": 0,
                    "total_opp_tov": 0,
                    "total_opp_oreb": 0,
                    "n_stints": 0,
                    "games": set(),
                }

            agg = lineup_agg[lhash]
            agg["total_minutes"] += dur / 60.0
            agg["total_pts_for"] += pf
            agg["total_pts_against"] += pa
            agg["total_fga"] += fga
            agg["total_fta"] += fta
            agg["total_tov"] += tov
            agg["total_oreb"] += oreb
            agg["total_opp_fga"] += opp_fga
            agg["total_opp_fta"] += opp_fta
            agg["total_opp_tov"] += opp_tov
            agg["total_opp_oreb"] += opp_oreb
            agg["n_stints"] += 1
            agg["games"].add(gid)

        # Compute ratings and insert
        rows = []
        for lhash, agg in lineup_agg.items():
            off_poss = max(
                0.0,
                agg["total_fga"]
                - agg["total_oreb"]
                + agg["total_tov"]
                + 0.44 * agg["total_fta"],
            )
            def_poss = max(
                0.0,
                agg["total_opp_fga"]
                - agg["total_opp_oreb"]
                + agg["total_opp_tov"]
                + 0.44 * agg["total_opp_fta"],
            )
            total_poss = (off_poss + def_poss) / 2.0  # average

            off_rtg = agg["total_pts_for"] / off_poss * 100 if off_poss > 0 else None
            def_rtg = (
                agg["total_pts_against"] / def_poss * 100 if def_poss > 0 else None
            )

            rows.append(
                (
                    lhash,
                    season,
                    agg["team_id"],
                    agg["player_ids"],
                    agg["total_minutes"],
                    agg["total_pts_for"],
                    agg["total_pts_against"],
                    total_poss,
                    off_rtg,
                    def_rtg,
                    agg["n_stints"],
                    len(agg["games"]),
                )
            )

        if rows:
            wowy_conn.executemany(
                """INSERT INTO LineupStints (
                    lineup_hash, season, team_id, player_ids,
                    total_minutes, total_pts_for, total_pts_against,
                    total_poss, off_rating, def_rating,
                    n_stints, n_games
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            total_lineups += len(rows)
            wowy_conn.commit()

        logger.info(f"  Season {season}: {len(rows)} unique lineups")

    logger.info(f"Lineup stints complete: {total_lineups} total lineups")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_stints(
    nba_conn: sqlite3.Connection,
    wowy_conn: sqlite3.Connection,
    game_ids: list[str],
    max_games: int = 50,
) -> dict:
    """
    Validate stint extraction against known game scores.

    Checks:
    1. Stint scores approximately sum to final game score (from TeamBox or PBP)
    2. No negative possessions
    3. No impossibly long stints
    4. Stint durations approximately sum to game duration
    """
    issues = {
        "score_mismatch": 0,
        "negative_poss": 0,
        "long_stint": 0,
        "duration_issue": 0,
        "checked": 0,
    }

    for gid in game_ids[:max_games]:
        # Get actual game score - try TeamBox first, then PBP running scores
        actual_scores: dict[int, int] = {}

        cursor = nba_conn.execute(
            "SELECT team_id, pts FROM TeamBox WHERE game_id = ?",
            (gid,),
        )
        for r in cursor:
            if r[1] is not None:
                actual_scores[int(r[0])] = r[1]

        if not actual_scores:
            # Fall back to PBP final scores: find the last event with scoreHome/scoreAway
            cursor = nba_conn.execute(
                "SELECT log_data FROM PBP_Logs WHERE game_id = ? ORDER BY play_id DESC LIMIT 30",
                (gid,),
            )
            # Get home/away team IDs
            cursor2 = nba_conn.execute(
                "SELECT home_team, away_team FROM Games WHERE game_id = ?", (gid,)
            )
            game_row = cursor2.fetchone()
            if not game_row:
                continue

            # Resolve team IDs
            cursor2 = nba_conn.execute("SELECT team_id, abbreviation FROM Teams")
            team_lookup_local = {r[1]: int(r[0]) for r in cursor2}

            home_tid = team_lookup_local.get(game_row[0]) or ABBR_TO_TEAM_ID.get(
                game_row[0], 0
            )
            away_tid = team_lookup_local.get(game_row[1]) or ABBR_TO_TEAM_ID.get(
                game_row[1], 0
            )

            for (log_data,) in cursor:
                try:
                    evt_raw = json.loads(log_data)
                    sh = evt_raw.get("scoreHome", "")
                    sa = evt_raw.get("scoreAway", "")
                    if sh and sa and sh != "" and sa != "":
                        try:
                            if home_tid:
                                actual_scores[home_tid] = int(sh)
                            if away_tid:
                                actual_scores[away_tid] = int(sa)
                            break
                        except (ValueError, TypeError):
                            continue
                except (json.JSONDecodeError, TypeError):
                    continue

        if not actual_scores:
            continue

        # Get stint totals (include all stints, not just 5-player, for score validation)
        cursor = wowy_conn.execute(
            "SELECT team_id, SUM(pts_for), SUM(duration_secs) FROM Stints WHERE game_id = ? GROUP BY team_id",
            (gid,),
        )
        stint_scores = {}
        stint_durations = {}
        for tid, pts, dur in cursor:
            stint_scores[tid] = pts or 0
            stint_durations[tid] = dur or 0

        # Check score alignment (allow up to 5 point discrepancy for edge cases)
        for tid, actual_pts in actual_scores.items():
            stint_pts = stint_scores.get(tid, 0)
            if actual_pts and abs(actual_pts - stint_pts) > 5:
                issues["score_mismatch"] += 1
                if issues["score_mismatch"] <= 5:
                    logger.warning(
                        f"Score mismatch game {gid} team {tid}: "
                        f"actual={actual_pts}, stints={stint_pts}, "
                        f"diff={actual_pts - stint_pts}"
                    )

        # Check total stint duration vs expected game duration
        for tid, dur in stint_durations.items():
            # Regular game = 4 * 720 = 2880 seconds
            # Duration should be close to this (within 10%)
            if dur > 0 and dur < 2000:  # less than ~33 minutes tracked
                issues["duration_issue"] += 1
                if issues["duration_issue"] <= 3:
                    logger.warning(
                        f"Low stint duration game {gid} team {tid}: "
                        f"{dur:.0f}s ({dur/60:.1f} min)"
                    )

        # Check for negative possessions
        cursor = wowy_conn.execute(
            "SELECT stint_id, fga, oreb, tov, fta FROM Stints WHERE game_id = ?",
            (gid,),
        )
        for sid, fga, oreb, tov, fta in cursor:
            poss = fga - oreb + tov + 0.44 * fta
            if poss < -1:  # allow small floating point error
                issues["negative_poss"] += 1

        # Check for stints > 15 minutes (unlikely)
        cursor = wowy_conn.execute(
            "SELECT COUNT(*) FROM Stints WHERE game_id = ? AND duration_secs > 900",
            (gid,),
        )
        long = cursor.fetchone()[0]
        if long:
            issues["long_stint"] += long

        issues["checked"] += 1

    return issues


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_game_list(nba_conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    """
    Get all eligible games with PBP data.

    Returns: [(game_id, home_team, away_team, season), ...]
    """
    cursor = nba_conn.execute("""
        SELECT DISTINCT g.game_id, g.home_team, g.away_team, g.season
        FROM Games g
        JOIN PBP_Logs p ON g.game_id = p.game_id
        WHERE g.status = 3
        AND g.season_type IN ('Regular Season', 'Post Season', 'PlayIn')
        ORDER BY g.season, g.game_id
    """)
    return list(cursor)


def resolve_team_id(abbr: str, team_lookup: dict[str, int]) -> int:
    """Resolve team abbreviation to numeric team_id."""
    # Direct lookup
    tid = team_lookup.get(abbr)
    if tid:
        return tid
    # Historical mapping
    tid = ABBR_TO_TEAM_ID.get(abbr)
    if tid:
        return tid
    return 0


def build_stint_cache(
    nba_conn: sqlite3.Connection,
    wowy_conn: sqlite3.Connection,
    test_mode: bool = False,
    max_games: int = 0,
):
    """
    Extract stints from all PBP games.

    Args:
        nba_conn: connection to main NBA database
        wowy_conn: connection to WOWY cache database
        test_mode: if True, process only 100 games for validation
        max_games: if > 0, process at most this many games
    """
    # Build team abbreviation lookup
    cursor = nba_conn.execute("SELECT team_id, abbreviation FROM Teams")
    team_lookup = {row[1]: int(row[0]) for row in cursor}

    # Get already-processed games
    cursor = wowy_conn.execute("SELECT game_id FROM ProcessedGames")
    processed = set(row[0] for row in cursor)
    logger.info(f"Already processed: {len(processed)} games")

    # Get full game list
    all_games = get_game_list(nba_conn)
    logger.info(f"Total eligible games: {len(all_games)}")

    # Filter to unprocessed
    games = [(gid, ht, at, s) for gid, ht, at, s in all_games if gid not in processed]
    logger.info(f"Games to process: {len(games)}")

    if test_mode:
        games = games[:100]
        logger.info("TEST MODE: processing 100 games only")
    elif max_games > 0:
        games = games[:max_games]
        logger.info(f"Limiting to {max_games} games")

    start_time = time.time()
    total_stints = 0
    errors = 0
    batch_stints: list[dict] = []
    batch_games: list[tuple[str, str, int, str]] = []

    for i, (game_id, home_team, away_team, season) in enumerate(games):
        try:
            home_tid = resolve_team_id(home_team, team_lookup)
            away_tid = resolve_team_id(away_team, team_lookup)

            if not home_tid or not away_tid:
                logger.debug(
                    f"Skipping {game_id}: cannot resolve team IDs "
                    f"({home_team}->{home_tid}, {away_team}->{away_tid})"
                )
                batch_games.append(
                    (
                        game_id,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        0,
                        "skip_team_id",
                    )
                )
                continue

            stints = extract_stints_for_game(
                nba_conn, game_id, home_tid, away_tid, season
            )

            batch_stints.extend(stints)
            n_stints = len(stints)
            total_stints += n_stints
            batch_games.append(
                (
                    game_id,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    n_stints,
                    "ok",
                )
            )

        except Exception as e:
            errors += 1
            batch_games.append(
                (
                    game_id,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    0,
                    f"error: {str(e)[:100]}",
                )
            )
            if errors <= 5:
                logger.warning(f"Error processing {game_id}: {e}")

        # Commit batch
        if (i + 1) % COMMIT_BATCH == 0 or (i + 1) == len(games):
            if batch_stints:
                insert_stints(wowy_conn, batch_stints)
            wowy_conn.executemany(
                "INSERT OR REPLACE INTO ProcessedGames (game_id, processed_at, n_stints, status) "
                "VALUES (?, ?, ?, ?)",
                batch_games,
            )
            wowy_conn.commit()
            batch_stints = []
            batch_games = []

        if (i + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(games) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{i+1}/{len(games)}] {rate:.1f} games/sec, "
                f"{total_stints} stints, {errors} errors, "
                f"ETA {eta/60:.1f} min"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Stint extraction complete: {len(games)} games in {elapsed:.1f}s, "
        f"{total_stints} stints, {errors} errors"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build WOWY cache from PBP data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only 100 games for validation",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Process at most N games (0 = all)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip stint extraction, only run aggregation",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after stint extraction",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize WOWY database
    wowy_conn = init_wowy_db(WOWY_DB_PATH)
    logger.info(f"WOWY database: {WOWY_DB_PATH}")

    if not args.aggregate_only:
        with get_db() as nba_conn:
            build_stint_cache(
                nba_conn,
                wowy_conn,
                test_mode=args.test,
                max_games=args.max_games,
            )

            if args.validate or args.test:
                logger.info("Running validation checks...")
                # Get a sample of processed games
                cursor = wowy_conn.execute(
                    "SELECT game_id FROM ProcessedGames WHERE status = 'ok' LIMIT 100"
                )
                sample_ids = [r[0] for r in cursor]
                issues = validate_stints(nba_conn, wowy_conn, sample_ids)
                logger.info(f"Validation results: {issues}")

    # Run aggregation
    logger.info("Running aggregation...")
    aggregate_lineup_stints(wowy_conn)
    aggregate_pairwise_wowy(wowy_conn)

    # Summary stats
    cursor = wowy_conn.execute("SELECT COUNT(*) FROM Stints")
    n_stints = cursor.fetchone()[0]
    cursor = wowy_conn.execute("SELECT COUNT(*) FROM PairwiseWOWY")
    n_pairs = cursor.fetchone()[0]
    cursor = wowy_conn.execute("SELECT COUNT(*) FROM LineupStints")
    n_lineups = cursor.fetchone()[0]
    cursor = wowy_conn.execute(
        "SELECT COUNT(*) FROM ProcessedGames WHERE status = 'ok'"
    )
    n_ok = cursor.fetchone()[0]
    cursor = wowy_conn.execute(
        "SELECT COUNT(*) FROM ProcessedGames WHERE status != 'ok'"
    )
    n_err = cursor.fetchone()[0]

    logger.info(
        f"\nSummary:\n"
        f"  Games processed: {n_ok} ok, {n_err} errors/skipped\n"
        f"  Stints: {n_stints}\n"
        f"  Pairwise WOWY pairs: {n_pairs}\n"
        f"  Unique 5-man lineups: {n_lineups}"
    )

    wowy_conn.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
