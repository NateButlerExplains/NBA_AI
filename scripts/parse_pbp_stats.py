#!/usr/bin/env python3
"""
parse_pbp_stats.py

Parse PbP_Logs table to compute enriched per-game player stats and store
them in the PBPPlayerGameStats table.

Handles two PBP format eras:
  - Older format (pre-2022 + some 2022-23): actionType = "Made Shot" / "Missed Shot"
  - Newer v3 format (some 2022-23+): actionType = "2pt" / "3pt" / "block" / "steal" etc.

Usage:
    python scripts/parse_pbp_stats.py --season 2024-25
    python scripts/parse_pbp_stats.py --all
    python scripts/parse_pbp_stats.py --game 0022400100
    python scripts/parse_pbp_stats.py --all --no-resume
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Turnover sub-type classification (all lowered for matching)
LIVE_BALL_TO_SUBTYPES = {
    "lost ball",
    "bad pass",
    "stolen",
    "lost ball turnover",
    "bad pass turnover",
}

DEAD_BALL_TO_SUBTYPES = {
    "traveling",
    "travel",
    "offensive foul",
    "offensive foul turnover",
    "backcourt",
    "backcourt turnover",
    "out of bounds",
    "out of bounds lost ball",
    "out of bounds lost ball turnover",
    "out of bounds bad pass",
    "out of bounds bad pass turnover",
    "double dribble",
    "3 second violation",
    "3 second",
    "5 second violation",
    "5 second",
    "5 sec inbound",
    "8 second violation",
    "8 second",
    "step out of bounds",
    "palming",
    "palming turnover",
    "discontinued dribble",
    "lane violation",
    "jump ball violation",
    "kicked ball violation",
    "inbound",
    "inbound turnover",
    "shot clock",
    "shot clock turnover",
    "swinging elbows",
    "basket from below",
    "opposite basket",
    "illegal assist",
    "punched ball",
    "too many players",
}

# 3-point line distance threshold (feet)
THREE_PT_DISTANCE = (
    22  # conservative — actual 3PT line is 22ft at corners, 23.75 at arc
)

# Corner 3 Y-coordinate threshold (court half-width pixels)
CORNER_Y_MAX = 90  # |yLegacy| or yLegacy <= 90 for corner region
CORNER_X_MIN = 220  # |xLegacy| >= 220 for corner region


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS PBPPlayerGameStats (
    player_id INTEGER,
    game_id TEXT,
    at_rim_fgm INTEGER DEFAULT 0,
    at_rim_fga INTEGER DEFAULT 0,
    short_mid_fgm INTEGER DEFAULT 0,
    short_mid_fga INTEGER DEFAULT 0,
    long_mid_fgm INTEGER DEFAULT 0,
    long_mid_fga INTEGER DEFAULT 0,
    corner3_fgm INTEGER DEFAULT 0,
    corner3_fga INTEGER DEFAULT 0,
    arc3_fgm INTEGER DEFAULT 0,
    arc3_fga INTEGER DEFAULT 0,
    assisted_2pt INTEGER DEFAULT 0,
    unassisted_2pt INTEGER DEFAULT 0,
    assisted_3pt INTEGER DEFAULT 0,
    unassisted_3pt INTEGER DEFAULT 0,
    live_ball_turnovers INTEGER DEFAULT 0,
    dead_ball_turnovers INTEGER DEFAULT 0,
    putback_fgm INTEGER DEFAULT 0,
    self_oreb INTEGER DEFAULT 0,
    blocks_made INTEGER DEFAULT 0,
    shots_blocked INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO PBPPlayerGameStats (
    player_id, game_id,
    at_rim_fgm, at_rim_fga,
    short_mid_fgm, short_mid_fga,
    long_mid_fgm, long_mid_fga,
    corner3_fgm, corner3_fga,
    arc3_fgm, arc3_fga,
    assisted_2pt, unassisted_2pt,
    assisted_3pt, unassisted_3pt,
    live_ball_turnovers, dead_ball_turnovers,
    putback_fgm,
    self_oreb,
    blocks_made,
    shots_blocked
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

STAT_FIELDS = [
    "at_rim_fgm",
    "at_rim_fga",
    "short_mid_fgm",
    "short_mid_fga",
    "long_mid_fgm",
    "long_mid_fga",
    "corner3_fgm",
    "corner3_fga",
    "arc3_fgm",
    "arc3_fga",
    "assisted_2pt",
    "unassisted_2pt",
    "assisted_3pt",
    "unassisted_3pt",
    "live_ball_turnovers",
    "dead_ball_turnovers",
    "putback_fgm",
    "self_oreb",
    "blocks_made",
    "shots_blocked",
]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(events: list[dict]) -> str:
    """Detect whether a game's PBP events use the older or v3 format.

    The v3 format uses actionType values like '2pt', '3pt', 'block', 'steal',
    'freethrow', etc.  The older format uses 'Made Shot', 'Missed Shot',
    'Free Throw', etc.

    We check the first few non-period events for a reliable signal.
    """
    for ev in events:
        at = ev.get("actionType", "")
        if at in ("2pt", "3pt", "block", "steal", "freethrow", "turnover", "rebound"):
            return "v3"
        if at in ("Made Shot", "Missed Shot", "Free Throw", "Rebound", "Turnover"):
            return "older"
        # jumpball / period events exist in both — skip them
    # Fallback: check for 'timeActual' (v3-only field) or 'qualifiers'
    for ev in events[:10]:
        if "timeActual" in ev or "qualifiers" in ev:
            return "v3"
    return "older"


# ---------------------------------------------------------------------------
# Shot zone classification
# ---------------------------------------------------------------------------


def classify_zone_v3(ev: dict) -> str:
    """Classify shot zone from a v3-format event using the `area` field."""
    area = (ev.get("area") or "").strip()
    if area == "Restricted Area":
        return "at_rim"
    if area == "In The Paint (Non-RA)":
        return "short_mid"
    if area == "Mid-Range":
        return "long_mid"
    if area in ("Left Corner 3", "Right Corner 3"):
        return "corner3"
    if area == "Above the Break 3":
        return "arc3"
    # If area is missing/unrecognized, fall back to distance/coordinates
    return classify_zone_by_distance(ev)


def classify_zone_by_distance(ev: dict) -> str:
    """Classify shot zone using shotDistance and coordinates (older format
    or as fallback for v3 when area is missing)."""
    dist = ev.get("shotDistance")
    x = ev.get("xLegacy")
    y = ev.get("yLegacy")
    desc = (ev.get("description") or "").upper()
    action_type = ev.get("actionType", "")

    # Determine if the shot is a 3-pointer
    is_3pt = False
    if action_type == "3pt":
        is_3pt = True
    elif "3PT" in desc:
        is_3pt = True

    # Try numeric distance first
    if dist is not None and dist != "" and dist != 0:
        try:
            d = float(dist)
        except (ValueError, TypeError):
            d = None
    else:
        d = None

    if is_3pt:
        # Corner vs arc 3
        if _is_corner_3(x, y):
            return "corner3"
        return "arc3"

    if d is not None:
        if d <= 4:
            return "at_rim"
        if d <= 14:
            return "short_mid"
        # d > 14 and not 3PT → long mid-range
        return "long_mid"

    # No distance, no 3PT indicator — try coordinates
    if x is not None and y is not None:
        try:
            xf, yf = float(x), float(y)
        except (ValueError, TypeError):
            return "unknown"
        # Rough distance from basket (basket at ~0,0 in legacy coords)
        approx_dist = (xf**2 + yf**2) ** 0.5 / 10.0  # legacy coords ~ 10x feet
        if approx_dist <= 4:
            return "at_rim"
        if approx_dist <= 14:
            return "short_mid"
        return "long_mid"

    return "unknown"


def _is_corner_3(x, y) -> bool:
    """Check if legacy coordinates indicate a corner 3."""
    if x is None or y is None:
        return False
    try:
        xf, yf = float(x), float(y)
    except (ValueError, TypeError):
        return False
    return abs(xf) >= CORNER_X_MIN and abs(yf) <= CORNER_Y_MAX


# ---------------------------------------------------------------------------
# Assist detection (older format)
# ---------------------------------------------------------------------------

# Pattern: "(PlayerName N AST)" in description of made shots
_AST_PATTERN = re.compile(r"\(\w[\w\s\.\-\']*\s+\d+\s+AST\)", re.IGNORECASE)


def has_assist_older(ev: dict) -> bool:
    """Detect if a made shot in the older format was assisted."""
    # Some older events might still have assistPersonId
    if ev.get("assistPersonId"):
        return True
    desc = ev.get("description") or ""
    return bool(_AST_PATTERN.search(desc))


# ---------------------------------------------------------------------------
# Determine if shot is 2PT or 3PT
# ---------------------------------------------------------------------------


def is_3pt_shot(ev: dict, fmt: str) -> bool:
    """Return True if the event is a 3-point field goal attempt."""
    if fmt == "v3":
        return ev.get("actionType") == "3pt"
    # Older format
    desc = (ev.get("description") or "").upper()
    return "3PT" in desc


# ---------------------------------------------------------------------------
# Putback detection helpers
# ---------------------------------------------------------------------------


def _parse_clock_seconds(clock_str: str) -> float | None:
    """Parse 'PT11M44.00S' or 'PT44.00S' to total seconds."""
    if not clock_str:
        return None
    m = re.match(r"PT(?:(\d+)M)?(\d+(?:\.\d+)?)S", clock_str)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = float(m.group(2))
    return minutes * 60 + seconds


def _events_within_seconds(ev1: dict, ev2: dict, max_seconds: float = 2.0) -> bool:
    """Check if two events in the same period are within max_seconds of each other."""
    if ev1.get("period") != ev2.get("period"):
        return False
    t1 = _parse_clock_seconds(ev1.get("clock", ""))
    t2 = _parse_clock_seconds(ev2.get("clock", ""))
    if t1 is None or t2 is None:
        return False
    # Clock counts down, so earlier event has higher clock value
    return abs(t1 - t2) <= max_seconds


# ---------------------------------------------------------------------------
# Process a single game
# ---------------------------------------------------------------------------


def process_game(events: list[dict]) -> dict[int, dict[str, int]]:
    """Process all PBP events for one game and return per-player stat dicts.

    Args:
        events: List of parsed JSON dicts, sorted by play_id (action order).

    Returns:
        {player_id: {stat_name: count, ...}, ...}
    """
    fmt = detect_format(events)
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {f: 0 for f in STAT_FIELDS})

    # Build an index of miss events for self-OREB / putback detection.
    # miss_index[i] holds the event if it's a missed FG, keyed by list position.
    # We'll scan backwards from rebounds/putbacks to find the preceding miss.

    # Pre-parse: find all shot events (field goals) and their indices
    # for quick lookup when processing rebounds
    shot_events_by_idx: dict[int, dict] = {}
    for i, ev in enumerate(events):
        if ev.get("isFieldGoal") == 1:
            shot_events_by_idx[i] = ev

    for i, ev in enumerate(events):
        pid = ev.get("personId")
        if not pid or pid == 0:
            continue

        action_type = ev.get("actionType", "")
        sub_type = (ev.get("subType") or "").strip()
        desc = ev.get("description") or ""

        # =================================================================
        # FIELD GOALS (made and missed)
        # =================================================================
        if ev.get("isFieldGoal") == 1:
            shot_result = (ev.get("shotResult") or "").lower()
            is_made = shot_result == "made"
            is_missed = shot_result == "missed"

            if not (is_made or is_missed):
                # Older format uses actionType to indicate made/missed
                if action_type == "Made Shot":
                    is_made = True
                elif action_type == "Missed Shot":
                    is_missed = True
                else:
                    continue

            three = is_3pt_shot(ev, fmt)

            # --- Shot zone ---
            if fmt == "v3":
                zone = classify_zone_v3(ev)
            else:
                zone = classify_zone_by_distance(ev)

            # Update FGA
            if zone == "at_rim":
                stats[pid]["at_rim_fga"] += 1
            elif zone == "short_mid":
                stats[pid]["short_mid_fga"] += 1
            elif zone == "long_mid":
                stats[pid]["long_mid_fga"] += 1
            elif zone == "corner3":
                stats[pid]["corner3_fga"] += 1
            elif zone == "arc3":
                stats[pid]["arc3_fga"] += 1
            # "unknown" zone — don't count in any zone bucket

            if is_made:
                # Update FGM
                if zone == "at_rim":
                    stats[pid]["at_rim_fgm"] += 1
                elif zone == "short_mid":
                    stats[pid]["short_mid_fgm"] += 1
                elif zone == "long_mid":
                    stats[pid]["long_mid_fgm"] += 1
                elif zone == "corner3":
                    stats[pid]["corner3_fgm"] += 1
                elif zone == "arc3":
                    stats[pid]["arc3_fgm"] += 1

                # --- Assisted / Unassisted ---
                if fmt == "v3":
                    assisted = bool(ev.get("assistPersonId"))
                else:
                    assisted = has_assist_older(ev)

                if three:
                    if assisted:
                        stats[pid]["assisted_3pt"] += 1
                    else:
                        stats[pid]["unassisted_3pt"] += 1
                else:
                    if assisted:
                        stats[pid]["assisted_2pt"] += 1
                    else:
                        stats[pid]["unassisted_2pt"] += 1

                # --- Putback detection ---
                is_putback = False
                if fmt == "v3":
                    descriptor = (ev.get("descriptor") or "").lower()
                    if "putback" in descriptor:
                        is_putback = True
                else:
                    # Older format: check subType for "Putback"
                    if "putback" in sub_type.lower():
                        is_putback = True

                # Also check: made shot within 2 seconds of own OREB
                if not is_putback:
                    # Look backwards for an OREB by same player within 2 sec
                    for j in range(i - 1, max(i - 5, -1), -1):
                        prev = events[j]
                        prev_at = prev.get("actionType", "").lower()
                        prev_sub = (prev.get("subType") or "").lower()
                        is_oreb = False
                        if fmt == "v3":
                            if prev_at == "rebound" and prev_sub == "offensive":
                                is_oreb = True
                        else:
                            if prev_at == "rebound":
                                prev_desc = (prev.get("description") or "").upper()
                                if "OFF:" in prev_desc:
                                    # Parse "Off:N" — if N > 0 at this point, it's offensive
                                    # Actually, all OREBs show up as Rebound with "Off:" count
                                    # The OREB count increments, but the key is the rebound
                                    # type is determined by which team gets it.
                                    # Simpler: check if rebound player == shooter on same team
                                    is_oreb = True
                            elif (
                                prev_at == ""
                                and "REBOUND" in (prev.get("description") or "").upper()
                            ):
                                # Sometimes actionType is empty but description has REBOUND
                                is_oreb = True

                        if is_oreb and prev.get("personId") == pid:
                            if _events_within_seconds(prev, ev, max_seconds=2.0):
                                is_putback = True
                                break
                        # Stop scanning at period boundaries or unrelated events
                        if prev.get("period") != ev.get("period"):
                            break

                if is_putback:
                    stats[pid]["putback_fgm"] += 1

            # --- Shots blocked (shooter's perspective) ---
            if is_missed:
                was_blocked = False
                if fmt == "v3":
                    # v3: the missed shot itself often has blockPersonId
                    if ev.get("blockPersonId"):
                        was_blocked = True
                    else:
                        # Also check if next event is a block
                        if i + 1 < len(events):
                            nxt = events[i + 1]
                            if nxt.get("actionType") == "block" and nxt.get(
                                "period"
                            ) == ev.get("period"):
                                was_blocked = True
                else:
                    # Older: check if description says "blocked" or if next/nearby
                    # event is a BLOCK with empty actionType
                    if "blocked" in desc.lower():
                        was_blocked = True
                    else:
                        # Look for a BLOCK event with same actionNumber or within 1
                        for j in range(i + 1, min(i + 3, len(events))):
                            nxt = events[j]
                            nxt_desc = (nxt.get("description") or "").upper()
                            if "BLOCK" in nxt_desc and nxt.get("period") == ev.get(
                                "period"
                            ):
                                # Verify same clock time (block happens simultaneously)
                                if nxt.get("clock") == ev.get("clock"):
                                    was_blocked = True
                                    break
                            # Don't look too far
                            if nxt.get("period") != ev.get("period"):
                                break

                if was_blocked:
                    stats[pid]["shots_blocked"] += 1

        # =================================================================
        # BLOCKS MADE
        # =================================================================
        elif fmt == "v3" and action_type == "block":
            stats[pid]["blocks_made"] += 1

        elif fmt == "older":
            # Older format: blocks show up as events with empty actionType
            # and "BLOCK" in description, e.g. "Durant BLOCK (1 BLK)"
            if action_type == "" and "BLOCK" in desc.upper() and "BLK" in desc.upper():
                stats[pid]["blocks_made"] += 1

        # =================================================================
        # REBOUNDS — check for self-OREB
        # =================================================================
        if fmt == "v3" and action_type == "rebound" and sub_type.lower() == "offensive":
            # Check if the preceding missed shot was by the same player
            # v3 has shotActionNumber linking rebound to the missed shot
            shot_action_num = ev.get("shotActionNumber")
            if shot_action_num is not None:
                # Find the missed shot with that actionNumber
                for j in range(i - 1, max(i - 10, -1), -1):
                    prev = events[j]
                    if prev.get("actionNumber") == shot_action_num:
                        if prev.get("personId") == pid and prev.get("isFieldGoal") == 1:
                            stats[pid]["self_oreb"] += 1
                        break
            else:
                # Fallback: look for preceding miss by same player
                for j in range(i - 1, max(i - 5, -1), -1):
                    prev = events[j]
                    if (
                        prev.get("isFieldGoal") == 1
                        and (prev.get("shotResult") or "").lower() == "missed"
                    ):
                        if prev.get("personId") == pid:
                            stats[pid]["self_oreb"] += 1
                        break
                    if prev.get("period") != ev.get("period"):
                        break

        elif fmt == "older" and action_type == "Rebound":
            # Determine if offensive rebound
            reb_desc = desc.upper()
            # Offensive rebound: check "Off:N" where N > previous
            # Actually, to determine OREB vs DREB in older format we need
            # to check team alignment: rebounder's team vs shooter's team.
            # If same team → OREB, different → DREB.
            is_oreb = False
            # Find the preceding missed shot
            preceding_miss = None
            preceding_miss_idx = None
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = events[j]
                if prev.get("isFieldGoal") == 1:
                    prev_result = (prev.get("shotResult") or "").lower()
                    prev_at = prev.get("actionType", "")
                    if prev_result == "missed" or prev_at == "Missed Shot":
                        preceding_miss = prev
                        preceding_miss_idx = j
                        break
                if prev.get("period") != ev.get("period"):
                    break

            if preceding_miss is not None:
                # OREB if rebounder's team == shooter's team
                if ev.get("teamId") == preceding_miss.get("teamId"):
                    is_oreb = True

            if is_oreb and preceding_miss is not None:
                if preceding_miss.get("personId") == pid:
                    stats[pid]["self_oreb"] += 1

        # Also handle the case where actionType is empty but description has REBOUND
        # (some older format events)
        elif fmt == "older" and action_type == "" and "REBOUND" in desc.upper():
            # Same logic as above
            preceding_miss = None
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = events[j]
                if prev.get("isFieldGoal") == 1:
                    prev_result = (prev.get("shotResult") or "").lower()
                    prev_at = prev.get("actionType", "")
                    if prev_result == "missed" or prev_at == "Missed Shot":
                        preceding_miss = prev
                        break
                if prev.get("period") != ev.get("period"):
                    break
            if preceding_miss is not None:
                if ev.get("teamId") == preceding_miss.get("teamId"):
                    # It's an OREB — check self
                    if preceding_miss.get("personId") == pid:
                        stats[pid]["self_oreb"] += 1

        # =================================================================
        # TURNOVERS
        # =================================================================
        if action_type == "Turnover" or (fmt == "v3" and action_type == "turnover"):
            to_sub = (sub_type or "").lower().strip()
            # Also check descriptor for v3
            if fmt == "v3":
                descriptor = (ev.get("descriptor") or "").lower().strip()
                # Merge subType and descriptor for classification
                to_key = to_sub if to_sub else descriptor
            else:
                to_key = to_sub

            if _is_live_ball_turnover(to_key, desc):
                stats[pid]["live_ball_turnovers"] += 1
            else:
                stats[pid]["dead_ball_turnovers"] += 1

    return stats


def _is_live_ball_turnover(to_key: str, desc: str) -> bool:
    """Classify a turnover as live-ball or dead-ball.

    Live-ball turnovers give the opponent a fastbreak opportunity:
    lost ball, bad pass, stolen.
    """
    if not to_key:
        # Try to infer from description
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in ["lost ball", "bad pass", "stolen"]):
            return True
        return False

    # Check against known sets
    if to_key in LIVE_BALL_TO_SUBTYPES:
        return True
    if to_key in DEAD_BALL_TO_SUBTYPES:
        return False

    # Fuzzy matching for variants
    for live in LIVE_BALL_TO_SUBTYPES:
        if live in to_key or to_key in live:
            return True

    # Default: if "steal" appears in context, it's live
    if "steal" in to_key or "stolen" in desc.lower():
        return True

    # Default to dead-ball for anything else (violations, etc.)
    return False


# ---------------------------------------------------------------------------
# Game loading & batching
# ---------------------------------------------------------------------------


def load_game_events(conn: sqlite3.Connection, game_id: str) -> list[dict]:
    """Load and parse all PBP events for a single game, sorted by play_id."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT play_id, log_data FROM PbP_Logs WHERE game_id = ? ORDER BY play_id",
        (game_id,),
    )
    events = []
    for _play_id, log_data in cursor:
        if log_data:
            try:
                events.append(json.loads(log_data))
            except json.JSONDecodeError:
                continue
    return events


def get_game_ids_for_season(conn: sqlite3.Connection, season: str) -> list[str]:
    """Get all game_ids from Games table for a given season that have PBP data."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT g.game_id
        FROM Games g
        INNER JOIN PbP_Logs p ON g.game_id = p.game_id
        WHERE g.season = ?
        ORDER BY g.game_id
        """,
        (season,),
    )
    return [row[0] for row in cursor.fetchall()]


def get_all_game_ids(conn: sqlite3.Connection) -> list[str]:
    """Get all game_ids that have PBP data."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT game_id FROM PbP_Logs ORDER BY game_id")
    return [row[0] for row in cursor.fetchall()]


def get_existing_game_ids(conn: sqlite3.Connection) -> set[str]:
    """Get set of game_ids already processed in PBPPlayerGameStats."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT game_id FROM PBPPlayerGameStats")
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return set()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_games(
    game_ids: list[str],
    resume: bool = True,
    db_path: str = None,
) -> None:
    """Process a list of game_ids and write stats to PBPPlayerGameStats."""
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    try:
        # Create table if needed
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()

        # Filter out already-processed games if resuming
        if resume:
            existing = get_existing_game_ids(conn)
            original_count = len(game_ids)
            game_ids = [gid for gid in game_ids if gid not in existing]
            skipped = original_count - len(game_ids)
            if skipped > 0:
                print(f"Resuming: skipping {skipped} already-processed games")

        total = len(game_ids)
        if total == 0:
            print("No games to process.")
            return

        print(f"Processing {total} games...")
        t_start = time.time()
        games_done = 0
        total_players = 0
        batch_rows: list[tuple] = []
        batch_size = 50  # Commit every N games

        for idx, game_id in enumerate(game_ids):
            events = load_game_events(conn, game_id)
            if not events:
                games_done += 1
                continue

            player_stats = process_game(events)

            for pid, pstats in player_stats.items():
                row = (
                    pid,
                    game_id,
                    pstats["at_rim_fgm"],
                    pstats["at_rim_fga"],
                    pstats["short_mid_fgm"],
                    pstats["short_mid_fga"],
                    pstats["long_mid_fgm"],
                    pstats["long_mid_fga"],
                    pstats["corner3_fgm"],
                    pstats["corner3_fga"],
                    pstats["arc3_fgm"],
                    pstats["arc3_fga"],
                    pstats["assisted_2pt"],
                    pstats["unassisted_2pt"],
                    pstats["assisted_3pt"],
                    pstats["unassisted_3pt"],
                    pstats["live_ball_turnovers"],
                    pstats["dead_ball_turnovers"],
                    pstats["putback_fgm"],
                    pstats["self_oreb"],
                    pstats["blocks_made"],
                    pstats["shots_blocked"],
                )
                batch_rows.append(row)
                total_players += 1

            games_done += 1

            # Commit in batches
            if games_done % batch_size == 0 and batch_rows:
                conn.executemany(INSERT_SQL, batch_rows)
                conn.commit()
                batch_rows = []

            # Progress reporting every 100 games
            if games_done % 100 == 0:
                elapsed = time.time() - t_start
                rate = games_done / elapsed if elapsed > 0 else 0
                eta = (total - games_done) / rate if rate > 0 else 0
                print(
                    f"  [{games_done}/{total}] "
                    f"{rate:.1f} games/sec, "
                    f"ETA {eta / 60:.1f} min"
                )

        # Final commit
        if batch_rows:
            conn.executemany(INSERT_SQL, batch_rows)
            conn.commit()

        elapsed = time.time() - t_start
        print(
            f"\nDone. Processed {games_done} games, "
            f"{total_players} player-game rows in {elapsed:.1f}s"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def normalize_season(s: str) -> str:
    """Normalize season string to 'YYYY-YYYY' format used in Games table.

    Accepts: '2024-25', '2024-2025', '2024'
    Returns: '2024-2025'
    """
    s = s.strip()
    if re.match(r"^\d{4}-\d{4}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}$", s):
        start = int(s[:4])
        return f"{start}-{start + 1}"
    if re.match(r"^\d{4}$", s):
        start = int(s)
        return f"{start}-{start + 1}"
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Parse PBP logs into enriched per-game player stats."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--season",
        type=str,
        help="Process one season (e.g. '2024-25' or '2024-2025')",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all seasons",
    )
    group.add_argument(
        "--game",
        type=str,
        help="Process a single game_id (for testing)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip games already in PBPPlayerGameStats (default: True)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path (default: from config)",
    )

    args = parser.parse_args()
    db_path = args.db or DB_PATH

    # Determine game_ids to process
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    try:
        if args.game:
            game_ids = [args.game]
            # Verify it exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM PbP_Logs WHERE game_id = ?",
                (args.game,),
            )
            count = cursor.fetchone()[0]
            if count == 0:
                print(f"Error: No PBP data found for game_id '{args.game}'")
                sys.exit(1)
            print(f"Processing single game: {args.game} ({count} events)")

        elif args.season:
            season = normalize_season(args.season)
            game_ids = get_game_ids_for_season(conn, season)
            if not game_ids:
                print(f"Error: No games with PBP data found for season '{season}'")
                # Show available seasons
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT season FROM Games ORDER BY season")
                seasons = [row[0] for row in cursor.fetchall()]
                print(f"Available seasons: {', '.join(seasons)}")
                sys.exit(1)
            print(f"Season {season}: {len(game_ids)} games with PBP data")

        else:  # --all
            game_ids = get_all_game_ids(conn)
            print(f"All seasons: {len(game_ids)} games with PBP data")

    finally:
        conn.close()

    process_games(game_ids, resume=args.resume, db_path=db_path)


if __name__ == "__main__":
    main()
