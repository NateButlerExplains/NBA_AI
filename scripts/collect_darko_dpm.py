#!/usr/bin/env python3
"""
collect_darko_dpm.py

Collects per-game DPM (Daily Plus-Minus) history from the DARKO API for all
NBA players in our database. Stores results in a DarkoDPM table.

The DARKO API is undocumented and free, so we rate-limit aggressively (2s
between requests by default) to be respectful.

Usage:
    # Full collection (~82 minutes at 2s delay)
    python scripts/collect_darko_dpm.py

    # Test with first 10 players
    python scripts/collect_darko_dpm.py --limit 10

    # Resume interrupted collection (default behavior)
    python scripts/collect_darko_dpm.py --resume

    # Fresh start (drops existing data)
    python scripts/collect_darko_dpm.py --no-resume

    # Adjust rate limiting
    python scripts/collect_darko_dpm.py --delay 3.0
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH
from src.logging_config import setup_logging

# --- Constants ---

DARKO_API_URL = "https://www.darko.app/api/player/{nba_id}/history?full=1"

REQUEST_TIMEOUT = 30  # seconds

import random

# Retry config
MAX_RETRIES = 2
RATE_LIMIT_BACKOFF = 30  # seconds to wait on 429

BATCH_COMMIT_SIZE = 50  # commit every N players

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS DarkoDPM (
    person_id INTEGER,
    game_date TEXT,
    season TEXT,
    team TEXT,
    dpm REAL,
    o_dpm REAL,
    d_dpm REAL,
    box_dpm REAL,
    box_odpm REAL,
    box_ddpm REAL,
    bayes_rapm_off REAL,
    bayes_rapm_def REAL,
    bayes_rapm_total REAL,
    age REAL,
    career_game_num INTEGER,
    seconds_played REAL,
    position TEXT,
    x_minutes REAL,
    PRIMARY KEY (person_id, game_date)
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO DarkoDPM (
    person_id, game_date, season, team, dpm, o_dpm, d_dpm,
    box_dpm, box_odpm, box_ddpm, bayes_rapm_off, bayes_rapm_def,
    bayes_rapm_total, age, career_game_num, seconds_played, position,
    x_minutes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# --- Helper Functions ---


def safe_float(value):
    """Convert a value to float, returning None if invalid."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value):
    """Convert a value to int, returning None if invalid."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def get_player_list(db_path: str) -> list[tuple[int, str]]:
    """
    Get all unique player IDs from PlayerBox for seasons >= '2001-02'.

    Joins with Players table to get player names for progress display.

    Returns:
        List of (person_id, display_name) tuples, sorted by person_id.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT pb.player_id,
                   COALESCE(p.first_name || ' ' || p.last_name, 'Unknown') as name
            FROM PlayerBox pb
            JOIN Games g ON pb.game_id = g.game_id
            LEFT JOIN Players p ON pb.player_id = p.person_id
            WHERE g.season >= '2001-02'
            ORDER BY pb.player_id
            """)
        return [(row[0], row[1]) for row in cursor.fetchall()]


def get_completed_player_ids(db_path: str) -> set[int]:
    """
    Get set of person_ids that already have data in DarkoDPM.

    Used for resume functionality.

    Returns:
        Set of person_id integers already in the table.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        # Check if table exists first
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='DarkoDPM'"
        )
        if cursor.fetchone() is None:
            return set()
        cursor.execute("SELECT DISTINCT person_id FROM DarkoDPM")
        return {row[0] for row in cursor.fetchall()}


def fetch_darko_history(nba_id: int, delay: float) -> list[dict] | None:
    """
    Fetch DPM history for a single player from the DARKO API.

    Handles retries for timeouts and rate limiting (429).

    Args:
        nba_id: NBA player ID (same as person_id in our DB).
        delay: Base delay between requests in seconds.

    Returns:
        List of row dicts from the API, or None if player not found (404)
        or unrecoverable error.
    """
    url = DARKO_API_URL.format(nba_id=nba_id)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                return data.get("rows", [])

            elif response.status_code == 404:
                logging.debug(f"Player {nba_id} not found in DARKO (404)")
                return None

            elif response.status_code == 429:
                if attempt < MAX_RETRIES:
                    logging.warning(
                        f"Rate limited (429) for player {nba_id}. "
                        f"Backing off {RATE_LIMIT_BACKOFF}s..."
                    )
                    time.sleep(RATE_LIMIT_BACKOFF)
                    continue
                else:
                    logging.error(
                        f"Rate limited (429) for player {nba_id} after "
                        f"{MAX_RETRIES} retries. Skipping."
                    )
                    return None

            else:
                logging.warning(
                    f"Unexpected status {response.status_code} for player "
                    f"{nba_id}. Response: {response.text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    continue
                return None

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                logging.warning(
                    f"Timeout for player {nba_id} (attempt {attempt + 1}/"
                    f"{MAX_RETRIES + 1}). Retrying..."
                )
                time.sleep(delay)
                continue
            else:
                logging.error(
                    f"Timeout for player {nba_id} after {MAX_RETRIES + 1} "
                    f"attempts. Skipping."
                )
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error for player {nba_id}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                continue
            return None

    return None


def parse_rows(person_id: int, rows: list[dict]) -> list[tuple]:
    """
    Parse DARKO API rows into tuples for SQL insertion.

    Args:
        person_id: NBA player ID.
        rows: List of row dicts from the DARKO API.

    Returns:
        List of tuples matching INSERT_SQL parameter order.
    """
    records = []
    for row in rows:
        record = (
            person_id,
            row.get("date"),
            row.get("season"),
            row.get("team_name"),
            safe_float(row.get("dpm")),
            safe_float(row.get("o_dpm")),
            safe_float(row.get("d_dpm")),
            safe_float(row.get("box_dpm")),
            safe_float(row.get("box_odpm")),
            safe_float(row.get("box_ddpm")),
            safe_float(row.get("bayes_rapm_off")),
            safe_float(row.get("bayes_rapm_def")),
            safe_float(row.get("bayes_rapm_total")),
            safe_float(row.get("age")),
            safe_int(row.get("career_game_num")),
            safe_float(row.get("seconds_played")),
            row.get("position"),
            safe_float(row.get("x_minutes")),
        )
        records.append(record)
    return records


def create_table(db_path: str):
    """Create the DarkoDPM table if it doesn't exist."""
    with get_db(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()


def print_summary(
    total_players: int,
    fetched: int,
    skipped_resume: int,
    not_found: int,
    errors: int,
    total_games: int,
    elapsed: float,
):
    """Print summary statistics after collection."""
    print("\n" + "=" * 60)
    print("DARKO DPM Collection Summary")
    print("=" * 60)
    print(f"  Total players in DB:    {total_players:,}")
    print(f"  Skipped (resume):       {skipped_resume:,}")
    print(f"  Fetched this run:       {fetched:,}")
    print(f"  Not in DARKO (404):     {not_found:,}")
    print(f"  Errors:                 {errors:,}")
    print(f"  Total game-rows saved:  {total_games:,}")
    print(f"  Elapsed time:           {elapsed / 60:.1f} minutes")
    if fetched > 0:
        print(f"  Avg games/player:       {total_games / max(fetched, 1):.0f}")
        print(f"  Avg time/player:        {elapsed / fetched:.1f}s")
    print("=" * 60)


def print_db_summary(db_path: str):
    """Print summary of what's in the DarkoDPM table."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM DarkoDPM")
        total_rows = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT person_id) FROM DarkoDPM")
        total_players = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM DarkoDPM")
        min_date, max_date = cursor.fetchone()
        cursor.execute(
            "SELECT season, COUNT(*) FROM DarkoDPM GROUP BY season ORDER BY season"
        )
        season_counts = cursor.fetchall()

    print(f"\n  DarkoDPM table: {total_rows:,} rows, {total_players:,} players")
    print(f"  Date range: {min_date} to {max_date}")
    print(f"  Seasons: {len(season_counts)}")
    if season_counts:
        print(f"    First: {season_counts[0][0]} ({season_counts[0][1]:,} rows)")
        print(f"    Last:  {season_counts[-1][0]} ({season_counts[-1][1]:,} rows)")


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-game DPM history from DARKO API for all NBA players."
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip players already in DB (default: True). Use --no-resume for fresh start.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API requests in seconds (default: 3.0). Be respectful.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch N players (for testing).",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help=f"Database path. Default: {DB_PATH}",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())

    db_path = args.db_path or DB_PATH
    logging.info(f"Database: {db_path}")
    logging.info(f"Rate limit delay: {args.delay}s between requests")

    # --- Get player list ---
    logging.info("Querying player list from PlayerBox (seasons >= 2001-02)...")
    players = get_player_list(db_path)
    logging.info(f"Found {len(players):,} unique players")

    if args.limit:
        players = players[: args.limit]
        logging.info(f"Limited to first {args.limit} players (--limit)")

    # --- Create table ---
    create_table(db_path)

    # --- Check resume state ---
    skipped_resume = 0
    if args.resume:
        completed = get_completed_player_ids(db_path)
        if completed:
            logging.info(
                f"Resume mode: {len(completed):,} players already collected. "
                f"Skipping those."
            )
    else:
        completed = set()
        logging.info("Fresh start mode (--no-resume). All players will be fetched.")

    # --- Collect ---
    total_players = len(players)
    fetched = 0
    not_found = 0
    errors = 0
    total_games = 0
    batch_records = []
    batch_count = 0

    start_time = time.time()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    try:
        for i, (person_id, name) in enumerate(players):
            # Skip if already collected
            if person_id in completed:
                skipped_resume += 1
                continue

            # Fetch from DARKO API
            rows = fetch_darko_history(person_id, args.delay)

            if rows is None:
                # 404 or error
                not_found += 1
                print(
                    f"Player {i + 1}/{total_players}: {name} ({person_id}) "
                    f"- not in DARKO"
                )
            elif len(rows) == 0:
                not_found += 1
                print(
                    f"Player {i + 1}/{total_players}: {name} ({person_id}) "
                    f"- 0 games (empty)"
                )
            else:
                records = parse_rows(person_id, rows)
                total_games += len(records)
                fetched += 1
                batch_records.extend(records)
                batch_count += 1
                print(
                    f"Player {i + 1}/{total_players}: {name} ({person_id}) "
                    f"- {len(records)} games fetched"
                )

            # Batch commit
            if batch_count >= BATCH_COMMIT_SIZE and batch_records:
                cursor = conn.cursor()
                cursor.executemany(INSERT_SQL, batch_records)
                conn.commit()
                logging.debug(
                    f"Committed batch of {batch_count} players "
                    f"({len(batch_records)} rows)"
                )
                batch_records = []
                batch_count = 0

            # Rate limit with jitter — every player that reaches here made an API call
            # (resumed players were skipped via continue above)
            jitter = random.uniform(0, 1.0)
            time.sleep(args.delay + jitter)

        # Final batch commit
        if batch_records:
            cursor = conn.cursor()
            cursor.executemany(INSERT_SQL, batch_records)
            conn.commit()
            logging.debug(
                f"Final commit: {batch_count} players ({len(batch_records)} rows)"
            )

    except KeyboardInterrupt:
        # Save progress on interrupt
        logging.warning("Interrupted! Saving progress...")
        if batch_records:
            cursor = conn.cursor()
            cursor.executemany(INSERT_SQL, batch_records)
            conn.commit()
            logging.info(f"Saved {len(batch_records)} pending rows before exit.")
    finally:
        conn.close()

    elapsed = time.time() - start_time

    # --- Summary ---
    print_summary(
        total_players=total_players,
        fetched=fetched,
        skipped_resume=skipped_resume,
        not_found=not_found,
        errors=errors,
        total_games=total_games,
        elapsed=elapsed,
    )
    print_db_summary(db_path)


if __name__ == "__main__":
    main()
