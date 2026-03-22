#!/usr/bin/env python3
"""
collect_databallr.py

Downloads NBA player data from databallr.com's public API and stores it
in the project SQLite database.

Three data sources:
  1. SQ RAPM (ShotQuality RAPM) — current season only
     Endpoint: api.databallr.com/api/supabase/v3_sq_rapm
  2. Player stats with metrics — multi-year, includes wingspan + height + multi-year RAPM
     Endpoint: api.databallr.com/api/supabase/player_stats_with_metrics?year=YYYY
  3. Multi-year RAPM with O/D splits — extracted from player_stats_with_metrics

Tables created:
  - DataballrSQRAPM: ShotQuality RAPM decomposition (current season)
  - DataballrPlayerMetrics: Full player metrics per season (181 fields)
  - DataballrWingspan: Player physical measurements (wingspan + height)
  - DataballrMultiYearRAPM: 2Y/3Y/4Y/5Y RAPM with offensive/defensive splits

Usage:
    python scripts/collect_databallr.py                  # Current season only
    python scripts/collect_databallr.py --all-years      # All years 2008-2026
    python scripts/collect_databallr.py --years 2024 2025 2026
    python scripts/collect_databallr.py --sq-rapm-only   # Just SQ RAPM
    python scripts/collect_databallr.py --wingspan-only   # Just wingspan data
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
BASE_API = "https://api.databallr.com/api/supabase"
SQ_RAPM_URL = f"{BASE_API}/v3_sq_rapm"
PLAYER_STATS_URL = f"{BASE_API}/player_stats_with_metrics"

# The SQ RAPM endpoint ignores year/season params — always returns current season.
# The player_stats_with_metrics endpoint accepts ?year=YYYY (2008-2026 confirmed).

# Fields of interest from the SQ RAPM endpoint (18 fields total)
SQ_RAPM_FIELDS = [
    "nba_id",
    "player_name",
    "team_abbreviation",
    "year",
    "possessions",
    "off_poss",
    "def_poss",
    "net_ts",
    "oTS",
    "dTS",
    "oSQ",
    "dSQ",
    "oCONTEST",
    "dCONTEST",
    "oFT",
    "dFT",
    "oRESID",
    "dRESID",
]

# Multi-year RAPM fields extracted from player_stats_with_metrics
MULTI_YEAR_RAPM_FIELDS = [
    "nba_id",
    "Name",
    "TeamAbbreviation",
    "year",
    "playoffs",
    # Single-year RAPM
    "off_rapm",
    "def_rapm",
    "total_rapm",
    # 2-year
    "two_year_orapm",
    "two_year_drapm",
    "two_year_rapm",
    # 3-year
    "three_year_orapm",
    "three_year_drapm",
    "three_year_rapm",
    # 4-year
    "four_year_orapm",
    "four_year_drapm",
    "four_year_rapm",
    # 5-year
    "five_year_orapm",
    "five_year_drapm",
    "five_year_rapm",
]

# Wingspan / physical measurement fields
WINGSPAN_FIELDS = [
    "nba_id",
    "Name",
    "TeamAbbreviation",
    "year",
    "wingspan_in",
    "height_wo_shoes_in",
]

# Rate limiting: polite delay between requests
REQUEST_DELAY = 3.0  # seconds — be respectful to free API


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------
def fetch_json(url: str, params: dict = None, retries: int = 3) -> list[dict]:
    """Fetch JSON array from a databallr API endpoint with retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                logger.warning("Unexpected response type: %s", type(data))
                return []
            return data
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                logger.warning("404 for %s — endpoint not found", url)
                return []
            logger.warning("HTTP error (attempt %d/%d): %s", attempt, retries, e)
        except requests.exceptions.RequestException as e:
            logger.warning("Request error (attempt %d/%d): %s", attempt, retries, e)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error (attempt %d/%d): %s", attempt, retries, e)

        if attempt < retries:
            wait = 2**attempt
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    logger.error("Failed to fetch %s after %d attempts", url, retries)
    return []


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
def create_tables(conn):
    """Create databallr tables if they don't exist."""

    # SQ RAPM — ShotQuality RAPM decomposition
    conn.execute("""
        CREATE TABLE IF NOT EXISTS DataballrSQRAPM (
            nba_id          INTEGER NOT NULL,
            player_name     TEXT,
            team_abbreviation TEXT,
            year            INTEGER NOT NULL,
            possessions     INTEGER,
            off_poss        INTEGER,
            def_poss        INTEGER,
            net_ts          REAL,
            oTS             REAL,
            dTS             REAL,
            oSQ             REAL,
            dSQ             REAL,
            oCONTEST        REAL,
            dCONTEST        REAL,
            oFT             REAL,
            dFT             REAL,
            oRESID          REAL,
            dRESID          REAL,
            PRIMARY KEY (nba_id, year)
        )
    """)

    # Full player metrics — store as JSON blob since there are 181 fields
    # that may change over time. Key lookups use nba_id + year + playoffs.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS DataballrPlayerMetrics (
            nba_id          INTEGER NOT NULL,
            name            TEXT,
            team_abbreviation TEXT,
            year            INTEGER NOT NULL,
            playoffs        INTEGER NOT NULL DEFAULT 0,
            data_json       TEXT NOT NULL,
            PRIMARY KEY (nba_id, year, playoffs)
        )
    """)

    # Wingspan — deduplicated physical measurements
    conn.execute("""
        CREATE TABLE IF NOT EXISTS DataballrWingspan (
            nba_id              INTEGER PRIMARY KEY,
            name                TEXT,
            wingspan_in         REAL,
            height_wo_shoes_in  REAL,
            updated_year        INTEGER
        )
    """)

    # Multi-year RAPM with O/D splits
    conn.execute("""
        CREATE TABLE IF NOT EXISTS DataballrMultiYearRAPM (
            nba_id              INTEGER NOT NULL,
            name                TEXT,
            team_abbreviation   TEXT,
            year                INTEGER NOT NULL,
            playoffs            INTEGER NOT NULL DEFAULT 0,
            off_rapm            REAL,
            def_rapm            REAL,
            total_rapm          REAL,
            two_year_orapm      REAL,
            two_year_drapm      REAL,
            two_year_rapm       REAL,
            three_year_orapm    REAL,
            three_year_drapm    REAL,
            three_year_rapm     REAL,
            four_year_orapm     REAL,
            four_year_drapm     REAL,
            four_year_rapm      REAL,
            five_year_orapm     REAL,
            five_year_drapm     REAL,
            five_year_rapm      REAL,
            PRIMARY KEY (nba_id, year, playoffs)
        )
    """)

    conn.commit()
    logger.info("Tables created/verified")


# ---------------------------------------------------------------------------
# Data insertion
# ---------------------------------------------------------------------------
def insert_sq_rapm(conn, records: list[dict]) -> int:
    """Insert SQ RAPM records. Returns count of rows inserted."""
    if not records:
        return 0

    sql = """
        INSERT OR REPLACE INTO DataballrSQRAPM
            (nba_id, player_name, team_abbreviation, year,
             possessions, off_poss, def_poss,
             net_ts, oTS, dTS, oSQ, dSQ,
             oCONTEST, dCONTEST, oFT, dFT, oRESID, dRESID)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = []
    for r in records:
        nba_id = r.get("nba_id")
        if nba_id is None:
            continue
        rows.append(
            (
                nba_id,
                r.get("player_name"),
                r.get("team_abbreviation"),
                r.get("year"),
                r.get("possessions"),
                r.get("off_poss"),
                r.get("def_poss"),
                r.get("net_ts"),
                r.get("oTS"),
                r.get("dTS"),
                r.get("oSQ"),
                r.get("dSQ"),
                r.get("oCONTEST"),
                r.get("dCONTEST"),
                r.get("oFT"),
                r.get("dFT"),
                r.get("oRESID"),
                r.get("dRESID"),
            )
        )

    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def insert_player_metrics(conn, records: list[dict], year: int) -> int:
    """Insert full player metrics as JSON blobs. Returns count."""
    if not records:
        return 0

    sql = """
        INSERT OR REPLACE INTO DataballrPlayerMetrics
            (nba_id, name, team_abbreviation, year, playoffs, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    rows = []
    for r in records:
        nba_id = r.get("nba_id")
        if nba_id is None:
            continue
        rows.append(
            (
                nba_id,
                r.get("Name"),
                r.get("TeamAbbreviation"),
                r.get("year", year),
                r.get("playoffs", 0),
                json.dumps(r, default=str),
            )
        )

    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def insert_wingspan(conn, records: list[dict]) -> int:
    """Insert/update wingspan data. Deduplicates by nba_id, keeping most recent year."""
    if not records:
        return 0

    sql = """
        INSERT INTO DataballrWingspan
            (nba_id, name, wingspan_in, height_wo_shoes_in, updated_year)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(nba_id) DO UPDATE SET
            name = excluded.name,
            wingspan_in = COALESCE(excluded.wingspan_in, DataballrWingspan.wingspan_in),
            height_wo_shoes_in = COALESCE(excluded.height_wo_shoes_in, DataballrWingspan.height_wo_shoes_in),
            updated_year = MAX(excluded.updated_year, DataballrWingspan.updated_year)
    """
    count = 0
    for r in records:
        nba_id = r.get("nba_id")
        wingspan = r.get("wingspan_in")
        height = r.get("height_wo_shoes_in")
        if nba_id is None:
            continue
        # Only insert if at least one measurement exists
        if wingspan is None and height is None:
            continue
        conn.execute(
            sql,
            (
                nba_id,
                r.get("Name"),
                wingspan,
                height,
                r.get("year"),
            ),
        )
        count += 1

    conn.commit()
    return count


def insert_multi_year_rapm(conn, records: list[dict], year: int) -> int:
    """Insert multi-year RAPM data. Returns count."""
    if not records:
        return 0

    sql = """
        INSERT OR REPLACE INTO DataballrMultiYearRAPM
            (nba_id, name, team_abbreviation, year, playoffs,
             off_rapm, def_rapm, total_rapm,
             two_year_orapm, two_year_drapm, two_year_rapm,
             three_year_orapm, three_year_drapm, three_year_rapm,
             four_year_orapm, four_year_drapm, four_year_rapm,
             five_year_orapm, five_year_drapm, five_year_rapm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = []
    for r in records:
        nba_id = r.get("nba_id")
        if nba_id is None:
            continue
        # Skip if no RAPM data at all
        has_rapm = any(
            r.get(f) is not None
            for f in [
                "off_rapm",
                "def_rapm",
                "total_rapm",
                "two_year_rapm",
                "three_year_rapm",
                "four_year_rapm",
                "five_year_rapm",
            ]
        )
        if not has_rapm:
            continue

        rows.append(
            (
                nba_id,
                r.get("Name"),
                r.get("TeamAbbreviation"),
                r.get("year", year),
                r.get("playoffs", 0),
                r.get("off_rapm"),
                r.get("def_rapm"),
                r.get("total_rapm"),
                r.get("two_year_orapm"),
                r.get("two_year_drapm"),
                r.get("two_year_rapm"),
                r.get("three_year_orapm"),
                r.get("three_year_drapm"),
                r.get("three_year_rapm"),
                r.get("four_year_orapm"),
                r.get("four_year_drapm"),
                r.get("four_year_rapm"),
                r.get("five_year_orapm"),
                r.get("five_year_drapm"),
                r.get("five_year_rapm"),
            )
        )

    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Collection routines
# ---------------------------------------------------------------------------
def collect_sq_rapm(conn) -> int:
    """Download and store ShotQuality RAPM (current season only)."""
    logger.info("Fetching SQ RAPM from %s", SQ_RAPM_URL)
    records = fetch_json(SQ_RAPM_URL)
    if not records:
        logger.warning("No SQ RAPM data returned")
        return 0

    year = records[0].get("year", "unknown")
    logger.info("SQ RAPM: %d players for season %s", len(records), year)
    count = insert_sq_rapm(conn, records)
    logger.info("Inserted %d SQ RAPM rows", count)
    return count


def collect_player_stats_for_year(conn, year: int) -> dict:
    """Download player_stats_with_metrics for a single year.

    Returns dict with counts: {metrics, wingspan, rapm}.
    """
    logger.info("Fetching player_stats_with_metrics for %d", year)
    records = fetch_json(PLAYER_STATS_URL, params={"year": year, "limit": 1500})
    if not records:
        logger.warning("No player stats data for %d", year)
        return {"metrics": 0, "wingspan": 0, "rapm": 0}

    logger.info("Year %d: %d player records received", year, len(records))

    # 1. Full player metrics
    metrics_count = insert_player_metrics(conn, records, year)

    # 2. Wingspan (deduplicated across years)
    wingspan_count = insert_wingspan(conn, records)

    # 3. Multi-year RAPM
    rapm_count = insert_multi_year_rapm(conn, records, year)

    logger.info(
        "Year %d: %d metrics, %d wingspan, %d RAPM rows inserted",
        year,
        metrics_count,
        wingspan_count,
        rapm_count,
    )
    return {"metrics": metrics_count, "wingspan": wingspan_count, "rapm": rapm_count}


def collect_player_stats(conn, years: list[int]) -> dict:
    """Download player stats for multiple years with rate limiting."""
    totals = {"metrics": 0, "wingspan": 0, "rapm": 0}

    for i, year in enumerate(years):
        counts = collect_player_stats_for_year(conn, year)
        for key in totals:
            totals[key] += counts[key]

        # Rate limit between requests
        if i < len(years) - 1:
            time.sleep(REQUEST_DELAY)

    return totals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download NBA player data from databallr.com"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Specific years to download (default: current season 2026)",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Download all available years (2008-2026)",
    )
    parser.add_argument(
        "--sq-rapm-only",
        action="store_true",
        help="Only download SQ RAPM data (current season)",
    )
    parser.add_argument(
        "--wingspan-only",
        action="store_true",
        help="Only download wingspan data (from player_stats endpoint)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2008,
        help="Start year for --all-years (default: 2008)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="End year for --all-years (default: 2026)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override database path (default: from config)",
    )
    args = parser.parse_args()

    # Determine years to fetch
    if args.all_years:
        years = list(range(args.start_year, args.end_year + 1))
    elif args.years:
        years = args.years
    else:
        years = [2026]  # Current season default

    logger.info("=" * 60)
    logger.info("databallr.com data collection")
    logger.info("=" * 60)

    db_kwargs = {"db_path": args.db_path} if args.db_path else {}

    with get_db(**db_kwargs) as conn:
        create_tables(conn)

        # --- SQ RAPM ---
        if not args.wingspan_only:
            sq_count = collect_sq_rapm(conn)
            logger.info("SQ RAPM total: %d rows", sq_count)

        # --- Player stats (metrics + wingspan + multi-year RAPM) ---
        if not args.sq_rapm_only:
            logger.info("Collecting player stats for years: %s", years)
            totals = collect_player_stats(conn, years)
            logger.info("-" * 40)
            logger.info("Player metrics total: %d rows", totals["metrics"])
            logger.info("Wingspan total: %d rows (deduplicated)", totals["wingspan"])
            logger.info("Multi-year RAPM total: %d rows", totals["rapm"])

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Collection complete")

    # Print table row counts
    with get_db(**db_kwargs) as conn:
        cursor = conn.cursor()
        for table in [
            "DataballrSQRAPM",
            "DataballrPlayerMetrics",
            "DataballrWingspan",
            "DataballrMultiYearRAPM",
        ]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            count = cursor.fetchone()[0]
            logger.info("  %-30s %6d rows", table, count)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
