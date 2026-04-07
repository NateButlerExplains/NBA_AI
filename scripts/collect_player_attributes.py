"""
collect_player_attributes.py

Collects player attribute data from the NBA API using 2 API calls:
1. PlayerIndex (historical_nullable='1') — all ~5,122 historical players
2. DraftHistory — complete draft history with overall pick numbers

Merges with existing player_experience.json for birth_year data,
then saves everything to a PlayerAttributes table in the SQLite database.

Usage:
    python scripts/collect_player_attributes.py
    python scripts/collect_player_attributes.py --log_level=DEBUG
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH
from src.logging_config import setup_logging

# --- Constants ---

PLAYER_EXPERIENCE_PATH = (
    PROJECT_ROOT / "data" / "phase3_cache" / "player_experience.json"
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS PlayerAttributes (
    person_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    position TEXT,
    height TEXT,
    height_inches INTEGER,
    weight INTEGER,
    country TEXT,
    college TEXT,
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER,
    undrafted INTEGER DEFAULT 0,
    from_year INTEGER,
    to_year INTEGER,
    birth_year INTEGER
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO PlayerAttributes (
    person_id, first_name, last_name, position, height, height_inches,
    weight, country, college, draft_year, draft_round, draft_number,
    undrafted, from_year, to_year, birth_year
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# --- Helper Functions ---


def parse_height(height_str):
    """
    Parse height string like "6-6" into total inches (78).

    Args:
        height_str: Height string in "feet-inches" format, e.g. "6-6"

    Returns:
        int or None: Total height in inches, or None if unparseable
    """
    if not height_str or not isinstance(height_str, str):
        return None
    parts = height_str.strip().split("-")
    if len(parts) != 2:
        return None
    try:
        feet = int(parts[0])
        inches = int(parts[1])
        return feet * 12 + inches
    except (ValueError, TypeError):
        return None


def safe_int(value):
    """Convert a value to int, returning None if it's empty, None, or invalid."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def fetch_player_index():
    """
    Fetch all historical players from the PlayerIndex endpoint.

    Returns:
        list[dict]: List of player records with keys matching the API response headers.
    """
    from nba_api.stats.endpoints.playerindex import PlayerIndex

    logging.info("Fetching PlayerIndex (historical_nullable='1')...")
    try:
        player_index = PlayerIndex(historical_nullable="1", timeout=60)
        df = player_index.get_data_frames()[0]
        logging.info(f"PlayerIndex returned {len(df)} players")
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch PlayerIndex: {e}")
        raise


def fetch_draft_history():
    """
    Fetch complete draft history from the DraftHistory endpoint.

    Returns:
        dict: Mapping of person_id -> draft record dict
    """
    from nba_api.stats.endpoints.drafthistory import DraftHistory

    logging.info("Fetching DraftHistory...")
    try:
        draft_history = DraftHistory(timeout=60)
        df = draft_history.get_data_frames()[0]
        logging.info(f"DraftHistory returned {len(df)} records")

        # Build lookup by person_id
        draft_lookup = {}
        for record in df.to_dict("records"):
            person_id = record.get("PERSON_ID")
            if person_id is not None:
                draft_lookup[int(person_id)] = record
        return draft_lookup
    except Exception as e:
        logging.error(f"Failed to fetch DraftHistory: {e}")
        raise


def load_birth_years():
    """
    Load birth year data from player_experience.json.

    The file maps person_id (as string) -> birth_year (integer).

    Returns:
        dict: Mapping of person_id (int) -> birth_year (int)
    """
    if not PLAYER_EXPERIENCE_PATH.exists():
        logging.warning(f"player_experience.json not found at {PLAYER_EXPERIENCE_PATH}")
        return {}

    with open(PLAYER_EXPERIENCE_PATH) as f:
        raw = json.load(f)

    # Convert string keys to int
    birth_years = {}
    for pid_str, birth_year in raw.items():
        try:
            birth_years[int(pid_str)] = int(birth_year)
        except (ValueError, TypeError):
            continue

    logging.info(f"Loaded {len(birth_years)} birth years from player_experience.json")
    return birth_years


def build_player_records(player_index_data, draft_lookup, birth_years):
    """
    Merge PlayerIndex, DraftHistory, and birth year data into unified records.

    Args:
        player_index_data: List of dicts from PlayerIndex endpoint
        draft_lookup: Dict mapping person_id -> DraftHistory record
        birth_years: Dict mapping person_id -> birth_year

    Returns:
        list[tuple]: List of tuples ready for SQL insertion
    """
    records = []

    for player in player_index_data:
        person_id = safe_int(player.get("PERSON_ID"))
        if person_id is None:
            continue

        first_name = player.get("PLAYER_FIRST_NAME") or ""
        last_name = player.get("PLAYER_LAST_NAME") or ""
        position = player.get("POSITION") or None
        height = player.get("HEIGHT") or None
        height_inches = parse_height(height)
        weight = safe_int(player.get("WEIGHT"))
        country = player.get("COUNTRY") or None
        college = player.get("COLLEGE") or None
        from_year = safe_int(player.get("FROM_YEAR"))
        to_year = safe_int(player.get("TO_YEAR"))

        # Draft info from PlayerIndex (fallback)
        draft_year_pi = safe_int(player.get("DRAFT_YEAR"))
        draft_round_pi = safe_int(player.get("DRAFT_ROUND"))
        draft_number_pi = safe_int(player.get("DRAFT_NUMBER"))

        # Draft info from DraftHistory (preferred for overall pick)
        draft_rec = draft_lookup.get(person_id)
        if draft_rec:
            draft_year = safe_int(draft_rec.get("SEASON")) or draft_year_pi
            draft_round = safe_int(draft_rec.get("ROUND_NUMBER")) or draft_round_pi
            draft_number = safe_int(draft_rec.get("OVERALL_PICK")) or draft_number_pi
            undrafted = 0
        else:
            # Not in DraftHistory — use PlayerIndex values if available
            draft_year = draft_year_pi
            draft_round = draft_round_pi
            # PlayerIndex DRAFT_NUMBER might be round pick, not overall
            draft_number = draft_number_pi
            # Mark as undrafted if no draft info at all
            undrafted = 1 if (draft_year is None or draft_year == 0) else 0

        # Birth year from player_experience.json
        birth_year = birth_years.get(person_id)

        records.append(
            (
                person_id,
                first_name,
                last_name,
                position,
                height,
                height_inches,
                weight,
                country,
                college,
                draft_year,
                draft_round,
                draft_number,
                undrafted,
                from_year,
                to_year,
                birth_year,
            )
        )

    return records


def save_to_database(records):
    """
    Create the PlayerAttributes table and insert all records.

    Args:
        records: List of tuples for SQL insertion

    Returns:
        int: Number of rows inserted
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Create table
        cursor.execute(CREATE_TABLE_SQL)

        # Insert all records
        cursor.executemany(INSERT_SQL, records)

        # Verify
        cursor.execute("SELECT COUNT(*) FROM PlayerAttributes")
        count = cursor.fetchone()[0]

        conn.commit()

    return count


def print_summary(records):
    """Print summary statistics about the collected data."""
    total = len(records)
    has_height = sum(1 for r in records if r[5] is not None)  # height_inches
    has_weight = sum(1 for r in records if r[6] is not None)  # weight
    has_draft = sum(1 for r in records if r[9] is not None and r[9] != 0)  # draft_year
    undrafted = sum(1 for r in records if r[12] == 1)  # undrafted flag
    has_birth = sum(1 for r in records if r[15] is not None)  # birth_year
    has_college = sum(1 for r in records if r[8] is not None and r[8] != "")  # college
    has_country = sum(1 for r in records if r[7] is not None and r[7] != "")  # country

    print("\n" + "=" * 60)
    print("PlayerAttributes Collection Summary")
    print("=" * 60)
    print(f"  Total players:       {total:,}")
    print(f"  With height:         {has_height:,} ({100*has_height/total:.1f}%)")
    print(f"  With weight:         {has_weight:,} ({100*has_weight/total:.1f}%)")
    print(f"  With draft data:     {has_draft:,} ({100*has_draft/total:.1f}%)")
    print(f"  Undrafted:           {undrafted:,} ({100*undrafted/total:.1f}%)")
    print(f"  With birth year:     {has_birth:,} ({100*has_birth/total:.1f}%)")
    print(f"  With college:        {has_college:,} ({100*has_college/total:.1f}%)")
    print(f"  With country:        {has_country:,} ({100*has_country/total:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Collect player attributes from NBA API into PlayerAttributes table."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help=f"Database path. Default: {DB_PATH}",
    )
    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())

    db_path = args.db_path or DB_PATH
    logging.info(f"Database: {db_path}")
    logging.info("Starting player attributes collection (2 API calls)...")

    # --- API Call 1: PlayerIndex ---
    player_index_data = fetch_player_index()

    # Rate-limit delay between API calls
    logging.debug("Waiting 1.0s between API calls...")
    time.sleep(1.0)

    # --- API Call 2: DraftHistory ---
    draft_lookup = fetch_draft_history()

    # --- Load birth years from existing cache ---
    birth_years = load_birth_years()

    # --- Merge data ---
    logging.info("Merging PlayerIndex + DraftHistory + birth years...")
    records = build_player_records(player_index_data, draft_lookup, birth_years)
    logging.info(f"Built {len(records)} player records")

    # --- Save to database ---
    logging.info("Saving to PlayerAttributes table...")
    saved_count = save_to_database(records)
    logging.info(f"Saved {saved_count} rows to PlayerAttributes")

    # --- Summary ---
    print_summary(records)

    logging.info("Done.")


if __name__ == "__main__":
    main()
