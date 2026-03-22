#!/usr/bin/env python3
"""
collect_rapm_archive.py

Download historical Regularized Adjusted Plus/Minus (RAPM) data from
basketball-analytics.gitlab.io/rapm-data/ and store it in the SQLite database.

Data source: https://basketball-analytics.gitlab.io/rapm-data/
Coverage: 1996-97 through 2018-19 (23 seasons), regular season + playoffs
Format: JSON files with structure {"data": [[rank, name, team, poss, orapm, drapm, rapm], ...]}

The archive does NOT include player IDs — only player names and 3-letter team
abbreviations. The script attempts a best-effort match to NBA API person_id via
the Players table (first_name + last_name), but unmatched players are still
stored with person_id = NULL. A summary of match rates is printed at the end.

Usage:
    # Download all seasons and insert into the database
    python scripts/collect_rapm_archive.py

    # Download only regular season data (skip playoffs)
    python scripts/collect_rapm_archive.py --regular-season-only

    # Download only playoff data
    python scripts/collect_rapm_archive.py --playoffs-only

    # Dry run (download and parse but don't write to DB)
    python scripts/collect_rapm_archive.py --dry-run

    # Use a specific database path
    python scripts/collect_rapm_archive.py --db-path data/NBA_AI_full.sqlite

    # Specific seasons only
    python scripts/collect_rapm_archive.py --seasons 2017-18 2018-19
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db

# ============================================================
# Constants
# ============================================================

BASE_URL = "https://basketball-analytics.gitlab.io/rapm-data/data"

# All available seasons (1996-97 through 2018-19)
ALL_SEASONS = [
    "1996-97",
    "1997-98",
    "1998-99",
    "1999-00",
    "2000-01",
    "2001-02",
    "2002-03",
    "2003-04",
    "2004-05",
    "2005-06",
    "2006-07",
    "2007-08",
    "2008-09",
    "2009-10",
    "2010-11",
    "2011-12",
    "2012-13",
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
]

# JSON file URL patterns:
#   Regular season: {BASE_URL}/{season}-rapm.json
#   Playoffs:       {BASE_URL}/{season}-playoffs-rapm.json

# Delay between HTTP requests to be polite to the server
REQUEST_DELAY_SECONDS = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Table DDL
# ============================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS RAPMArchive (
    person_id       INTEGER,
    player_name     TEXT NOT NULL,
    season          TEXT NOT NULL,
    season_type     TEXT NOT NULL DEFAULT 'Regular Season',
    team            TEXT,
    possessions     INTEGER,
    o_rapm          REAL,
    d_rapm          REAL,
    rapm            REAL,
    PRIMARY KEY (player_name, season, season_type)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_rapm_archive_person_id
    ON RAPMArchive(person_id);
"""


# ============================================================
# Download helpers
# ============================================================


def build_url(season: str, playoffs: bool = False) -> str:
    """Build the JSON file URL for a given season and type."""
    if playoffs:
        return f"{BASE_URL}/{season}-playoffs-rapm.json"
    else:
        return f"{BASE_URL}/{season}-rapm.json"


def download_json(url: str) -> Optional[dict]:
    """Download and parse a JSON file. Returns None on failure."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.warning("Not found: %s", url)
        else:
            logger.error("HTTP error fetching %s: %s", url, e)
        return None
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error("Error fetching %s: %s", url, e)
        return None


# ============================================================
# Parsing
# ============================================================


def parse_season_data(
    raw: dict,
    season: str,
    season_type: str,
) -> List[dict]:
    """
    Parse the raw JSON into a list of row dicts.

    The JSON format is:
        {"data": [[rank, player_name, team, possessions, o_rapm, d_rapm, rapm], ...]}

    Returns list of dicts ready for DB insertion.
    """
    rows = []
    data = raw.get("data", [])
    for entry in data:
        if len(entry) < 7:
            logger.warning(
                "Skipping malformed entry in %s %s: %s", season, season_type, entry
            )
            continue

        _rank, player_name, team, possessions, o_rapm, d_rapm, rapm = entry[:7]

        rows.append(
            {
                "player_name": str(player_name).strip(),
                "season": season,
                "season_type": season_type,
                "team": str(team).strip() if team else None,
                "possessions": int(possessions) if possessions is not None else None,
                "o_rapm": float(o_rapm) if o_rapm is not None else None,
                "d_rapm": float(d_rapm) if d_rapm is not None else None,
                "rapm": float(rapm) if rapm is not None else None,
            }
        )

    return rows


# ============================================================
# Player ID matching
# ============================================================


def build_player_lookup(db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Build a name -> person_id lookup from the Players table.

    Uses "First Last" format as the key (lowercased for case-insensitive matching).
    For duplicate names, the most recent player (highest to_year) wins.

    Returns:
        Dict mapping lowercase "first last" -> person_id
    """
    lookup: Dict[str, int] = {}

    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT person_id, first_name, last_name, from_year, to_year FROM Players"
        )
        rows = cursor.fetchall()

    # Track best match by recency for duplicate names
    best_year: Dict[str, int] = {}

    for person_id, first_name, last_name, from_year, to_year in rows:
        if not first_name or not last_name:
            continue
        key = f"{first_name} {last_name}".lower().strip()
        yr = to_year or 0
        if key not in lookup or yr > best_year.get(key, 0):
            lookup[key] = person_id
            best_year[key] = yr

    return lookup


def match_player_ids(
    all_rows: List[dict],
    db_path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Attempt to fill in person_id for each row by matching player_name
    against the Players table.

    Modifies rows in place. Returns (matched_count, total_count).
    """
    lookup = build_player_lookup(db_path)

    matched = 0
    for row in all_rows:
        key = row["player_name"].lower().strip()
        pid = lookup.get(key)
        if pid is not None:
            row["person_id"] = pid
            matched += 1
        else:
            row["person_id"] = None

    return matched, len(all_rows)


# ============================================================
# Database operations
# ============================================================


def create_table(db_path: Optional[str] = None) -> None:
    """Create the RAPMArchive table if it doesn't exist."""
    with get_db(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(CREATE_INDEX_SQL)
    logger.info("RAPMArchive table ready.")


def insert_rows(rows: List[dict], db_path: Optional[str] = None) -> int:
    """
    Insert rows into RAPMArchive using INSERT OR REPLACE to handle re-runs.
    Returns the number of rows inserted.
    """
    if not rows:
        return 0

    sql = """
        INSERT OR REPLACE INTO RAPMArchive
            (person_id, player_name, season, season_type, team, possessions, o_rapm, d_rapm, rapm)
        VALUES
            (:person_id, :player_name, :season, :season_type, :team, :possessions, :o_rapm, :d_rapm, :rapm)
    """

    with get_db(db_path) as conn:
        conn.executemany(sql, rows)

    return len(rows)


# ============================================================
# Main pipeline
# ============================================================


def collect_all(
    seasons: List[str],
    include_regular: bool = True,
    include_playoffs: bool = True,
    dry_run: bool = False,
    db_path: Optional[str] = None,
) -> None:
    """Download, parse, match IDs, and store all RAPM archive data."""

    all_rows: List[dict] = []
    download_tasks: List[Tuple[str, str, str]] = []  # (url, season, season_type)

    for season in seasons:
        if include_regular:
            download_tasks.append(
                (
                    build_url(season, playoffs=False),
                    season,
                    "Regular Season",
                )
            )
        if include_playoffs:
            download_tasks.append(
                (
                    build_url(season, playoffs=True),
                    season,
                    "Playoffs",
                )
            )

    logger.info(
        "Downloading %d files for %d seasons...", len(download_tasks), len(seasons)
    )

    for i, (url, season, season_type) in enumerate(download_tasks):
        logger.info("[%d/%d] %s %s", i + 1, len(download_tasks), season, season_type)

        raw = download_json(url)
        if raw is None:
            logger.warning("  Skipped (download failed).")
            continue

        rows = parse_season_data(raw, season, season_type)
        logger.info("  Parsed %d players.", len(rows))
        all_rows.extend(rows)

        # Be polite
        if i < len(download_tasks) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Total rows parsed: %d", len(all_rows))

    if not all_rows:
        logger.warning("No data collected. Exiting.")
        return

    # --- Player ID matching ---
    logger.info("Matching player names to NBA API person_id via Players table...")
    matched, total = match_player_ids(all_rows, db_path)
    unmatched = total - matched
    logger.info(
        "Matched %d / %d players (%.1f%%). %d unmatched (stored with person_id = NULL).",
        matched,
        total,
        100.0 * matched / total if total else 0,
        unmatched,
    )

    # Print a sample of unmatched names for debugging
    unmatched_names = sorted(
        {r["player_name"] for r in all_rows if r["person_id"] is None}
    )
    if unmatched_names:
        sample = unmatched_names[:20]
        logger.info(
            "Sample unmatched names (%d total): %s%s",
            len(unmatched_names),
            ", ".join(sample),
            "..." if len(unmatched_names) > 20 else "",
        )

    if dry_run:
        logger.info("Dry run — skipping database writes.")
        # Print a summary table
        logger.info("Sample data (first 10 rows):")
        for row in all_rows[:10]:
            logger.info(
                "  %-25s %-8s %-4s poss=%6s oRAPM=%7.3f dRAPM=%7.3f RAPM=%7.3f pid=%s",
                row["player_name"],
                row["season"],
                row["team"],
                row["possessions"],
                row["o_rapm"] or 0,
                row["d_rapm"] or 0,
                row["rapm"] or 0,
                row["person_id"],
            )
        return

    # --- Database write ---
    create_table(db_path)
    inserted = insert_rows(all_rows, db_path)
    logger.info("Inserted %d rows into RAPMArchive.", inserted)

    # --- Verification ---
    with get_db(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM RAPMArchive")
        count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(DISTINCT season) FROM RAPMArchive")
        n_seasons = cursor.fetchone()[0]
        cursor = conn.execute(
            "SELECT COUNT(*) FROM RAPMArchive WHERE person_id IS NOT NULL"
        )
        n_matched = cursor.fetchone()[0]

    logger.info(
        "Verification: %d total rows, %d seasons, %d with person_id.",
        count,
        n_seasons,
        n_matched,
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical RAPM data and store in SQLite."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Specific seasons to download (e.g., 2017-18 2018-19). Default: all 23 seasons.",
    )
    parser.add_argument(
        "--regular-season-only",
        action="store_true",
        help="Download only regular season data.",
    )
    parser.add_argument(
        "--playoffs-only",
        action="store_true",
        help="Download only playoff data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse but do not write to the database.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database. Defaults to project config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seasons = args.seasons if args.seasons else ALL_SEASONS

    # Validate season format
    for s in seasons:
        if s not in ALL_SEASONS:
            logger.error(
                "Invalid season '%s'. Must be one of: %s",
                s,
                ", ".join(ALL_SEASONS),
            )
            sys.exit(1)

    include_regular = not args.playoffs_only
    include_playoffs = not args.regular_season_only

    if not include_regular and not include_playoffs:
        logger.error("Cannot specify both --regular-season-only and --playoffs-only.")
        sys.exit(1)

    logger.info(
        "RAPM Archive Collector — %d seasons, regular=%s, playoffs=%s, dry_run=%s",
        len(seasons),
        include_regular,
        include_playoffs,
        args.dry_run,
    )

    collect_all(
        seasons=seasons,
        include_regular=include_regular,
        include_playoffs=include_playoffs,
        dry_run=args.dry_run,
        db_path=args.db_path,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
