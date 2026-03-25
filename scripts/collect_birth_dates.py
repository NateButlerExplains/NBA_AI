"""
collect_birth_dates.py

Collects real birth dates for all players in the PlayerAttributes table
using the NBA API CommonPlayerInfo endpoint.

Rate limiting strategy:
- 10 requests per burst, then 15-second pause
- After every 100 requests, 60-second cooling period
- On error: 60-second backoff, 3 consecutive errors = 120-second pause
- Progress saved to JSON for resumability

Expected runtime: ~45-60 minutes for ~5,000 players
"""

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonplayerinfo

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DB_PATH = "data/NBA_AI_full.sqlite"
PROGRESS_FILE = "data/birth_date_progress.json"

BURST_SIZE = 10
BURST_PAUSE = 15  # seconds between bursts
COOLING_INTERVAL = 100  # cool down after this many requests
COOLING_PAUSE = 60  # seconds for cooling period
ERROR_PAUSE = 60  # seconds after a single error
CONSECUTIVE_ERROR_PAUSE = 120  # seconds after 3 consecutive errors
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/birth_date_collection.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_progress():
    """Load progress from JSON file. Returns dict of person_id -> birth_year."""
    path = Path(PROGRESS_FILE)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        # Keys stored as strings in JSON, convert to int
        return {int(k): v for k, v in data.items()}
    return {}


def save_progress(progress):
    """Save progress to JSON file."""
    # Convert int keys to strings for JSON
    with open(PROGRESS_FILE, "w") as f:
        json.dump({str(k): v for k, v in progress.items()}, f)


def fetch_birth_year(person_id):
    """
    Fetch birth year for a single player from CommonPlayerInfo.

    Returns:
        int or None: Birth year, or None if not available
    """
    info = commonplayerinfo.CommonPlayerInfo(
        player_id=person_id, timeout=REQUEST_TIMEOUT
    )
    data = info.get_dict()

    result_sets = data.get("resultSets", [])
    for rs in result_sets:
        if rs["name"] == "CommonPlayerInfo" and rs["rowSet"]:
            headers = rs["headers"]
            row = rs["rowSet"][0]
            birth_idx = headers.index("BIRTHDATE")
            birthdate_str = row[birth_idx]

            if birthdate_str and isinstance(birthdate_str, str):
                # Format: "1984-12-30T00:00:00"
                year_str = birthdate_str.split("-")[0]
                try:
                    return int(year_str)
                except ValueError:
                    logger.warning(
                        f"Could not parse birth year from '{birthdate_str}' "
                        f"for person_id={person_id}"
                    )
                    return None
    return None


def update_database(progress):
    """Batch-update PlayerAttributes.birth_year from progress dict."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    count = 0
    for person_id, birth_year in progress.items():
        if birth_year is not None:
            cursor.execute(
                "UPDATE PlayerAttributes SET birth_year = ? WHERE person_id = ?",
                (birth_year, person_id),
            )
            count += 1
    conn.commit()
    conn.close()
    logger.info(f"Updated {count} rows in PlayerAttributes")


def main():
    logger.info("=" * 60)
    logger.info("Birth date collection starting")
    logger.info("=" * 60)

    # Get all person_ids from database — only players active 2001+
    # (older IDs like <100000 timeout on the API consistently)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT pa.person_id FROM PlayerAttributes pa "
        "JOIN Players p ON pa.person_id = p.person_id "
        "WHERE p.from_year >= 2001 "
        "ORDER BY pa.person_id"
    )
    all_person_ids = [row[0] for row in cursor.fetchall()]
    conn.close()

    logger.info(f"Total players in PlayerAttributes: {len(all_person_ids)}")

    # Load existing progress
    progress = load_progress()
    logger.info(f"Already fetched: {len(progress)} players")

    # Determine which players still need fetching
    # Skip only successfully fetched players (not None failures, which we retry)
    successfully_fetched = {k for k, v in progress.items() if v is not None}
    remaining = [pid for pid in all_person_ids if pid not in successfully_fetched]
    logger.info(f"Remaining to fetch: {len(remaining)}")

    if not remaining:
        logger.info("All players already fetched. Updating database...")
        update_database(progress)
        return

    # Fetch birth dates
    consecutive_errors = 0
    burst_count = 0
    fetched_this_run = 0
    errors_this_run = 0
    start_time = time.time()

    for i, person_id in enumerate(remaining):
        # Cooling period after every COOLING_INTERVAL requests
        if fetched_this_run > 0 and fetched_this_run % COOLING_INTERVAL == 0:
            logger.info(
                f"Cooling period: {COOLING_PAUSE}s pause after "
                f"{fetched_this_run} requests..."
            )
            time.sleep(COOLING_PAUSE)
            burst_count = 0

        # Rate limiting: pause after each burst
        if burst_count >= BURST_SIZE:
            logger.debug(f"Burst complete, pausing {BURST_PAUSE}s...")
            time.sleep(BURST_PAUSE)
            burst_count = 0

        # Attempt to fetch with retries
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                birth_year = fetch_birth_year(person_id)
                progress[person_id] = birth_year
                fetched_this_run += 1
                burst_count += 1
                consecutive_errors = 0
                success = True
                break
            except Exception as e:
                consecutive_errors += 1
                errors_this_run += 1
                error_msg = str(e)[:120]
                logger.warning(
                    f"Error fetching person_id={person_id} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}"
                )

                if consecutive_errors >= 3:
                    logger.warning(
                        f"3 consecutive errors, pausing {CONSECUTIVE_ERROR_PAUSE}s..."
                    )
                    time.sleep(CONSECUTIVE_ERROR_PAUSE)
                    consecutive_errors = 0
                    burst_count = 0  # Reset burst counter after long pause
                else:
                    logger.debug(f"Error backoff: {ERROR_PAUSE}s...")
                    time.sleep(ERROR_PAUSE)

        if not success:
            # Mark as None so we don't retry endlessly
            progress[person_id] = None
            logger.error(
                f"Failed to fetch person_id={person_id} after {MAX_RETRIES} retries"
            )

        # Progress logging every 100 players
        if (fetched_this_run % 100 == 0 and fetched_this_run > 0) or (
            i == len(remaining) - 1
        ):
            elapsed = time.time() - start_time
            rate = fetched_this_run / elapsed * 60 if elapsed > 0 else 0
            successful = sum(1 for v in progress.values() if v is not None)
            logger.info(
                f"Progress: {fetched_this_run}/{len(remaining)} fetched "
                f"({successful} total with birth year) | "
                f"Errors: {errors_this_run} | "
                f"Rate: {rate:.0f}/min | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # Save progress every 50 players
        if fetched_this_run % 50 == 0:
            save_progress(progress)

    # Final save
    save_progress(progress)

    elapsed = time.time() - start_time
    successful = sum(1 for v in progress.values() if v is not None)
    failed = sum(1 for v in progress.values() if v is None)

    logger.info("=" * 60)
    logger.info("Collection complete!")
    logger.info(f"Total fetched: {fetched_this_run}")
    logger.info(f"Successful: {successful}, Failed/No data: {failed}")
    logger.info(f"Total errors: {errors_this_run}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 60)

    # Update database
    logger.info("Updating database...")
    update_database(progress)

    # Verification
    logger.info("Verifying results...")
    verify_results()


def verify_results():
    """Verify the birth year data looks correct."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT person_id, first_name, last_name, birth_year "
        "FROM PlayerAttributes WHERE birth_year IS NOT NULL"
    )
    players = cursor.fetchall()
    years = [p[3] for p in players]

    logger.info(f"Players with birth_year: {len(years)}")
    if years:
        logger.info(f"Range: {min(years)} to {max(years)}")
        logger.info(
            f"Players with birth_year < 1950: " f"{sum(1 for y in years if y < 1950)}"
        )
        logger.info(
            f"Players with birth_year 1950-1969: "
            f"{sum(1 for y in years if 1950 <= y < 1970)}"
        )
        logger.info(
            f"Players with birth_year 1970-1989: "
            f"{sum(1 for y in years if 1970 <= y < 1990)}"
        )
        logger.info(
            f"Players with birth_year 1990-2005: "
            f"{sum(1 for y in years if 1990 <= y <= 2005)}"
        )
        logger.info(
            f"Players with birth_year > 2005: " f"{sum(1 for y in years if y > 2005)}"
        )

    # Spot check known players
    for pid, name in [(2544, "LeBron"), (201939, "Curry"), (203999, "Jokic")]:
        row = conn.execute(
            "SELECT birth_year FROM PlayerAttributes WHERE person_id = ?", (pid,)
        ).fetchone()
        if row:
            logger.info(f"  {name}: birth_year = {row[0]}")

    conn.close()


if __name__ == "__main__":
    main()
