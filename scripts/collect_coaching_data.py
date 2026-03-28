"""
collect_coaching_data.py

Collects head coach (and assistant coach) data for all teams across seasons
2001-02 through 2025-26 using the NBA API CommonTeamRoster endpoint.

Rate limiting strategy (VERY conservative):
- 5 requests per burst, then 20-second pause
- After every 50 requests, 60-second cooling period
- On error: 60-second backoff, 3 consecutive errors = 120-second pause
- Max 3 retries per request, then skip
- Progress saved to JSON for resumability

Expected runtime: ~50-60 minutes for 750 calls (30 teams x 25 seasons)
"""

import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DB_PATH = "data/NBA_AI_full.sqlite"
PROGRESS_FILE = "data/coaching_progress.json"

BURST_SIZE = 5
BURST_PAUSE = 20  # seconds between bursts
COOLING_INTERVAL = 50  # cool down after this many requests
COOLING_PAUSE = 60  # seconds for cooling period
ERROR_PAUSE = 60  # seconds after a single error
CONSECUTIVE_ERROR_PAUSE = 120  # seconds after 3 consecutive errors
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Seasons to collect: 2001-02 through 2025-26
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2001, 2026)]

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/coaching_collection.log"),
    ],
)
logger = logging.getLogger(__name__)


def create_coaches_table():
    """Create the Coaches table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Coaches (
            team_id TEXT NOT NULL,
            season TEXT NOT NULL,
            coach_id TEXT,
            coach_name TEXT,
            is_head_coach INTEGER,
            PRIMARY KEY (team_id, season, coach_id)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Coaches table ready")


def get_team_ids():
    """Get all 30 team IDs from the database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT team_id, abbreviation FROM Teams ORDER BY abbreviation"
    ).fetchall()
    conn.close()
    return rows


def load_progress():
    """Load progress from JSON file. Returns set of 'team_id|season' strings."""
    path = Path(PROGRESS_FILE)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_progress(completed):
    """Save progress to JSON file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


def fetch_coaches(team_id, season):
    """
    Fetch coaching staff for a single team-season from CommonTeamRoster.

    Returns:
        list of dicts: Each dict has team_id, season, coach_id, coach_name, is_head_coach
    """
    roster = commonteamroster.CommonTeamRoster(
        team_id=str(team_id), season=season, timeout=REQUEST_TIMEOUT
    )
    dfs = roster.get_data_frames()

    if len(dfs) < 2 or dfs[1].empty:
        logger.warning(f"No coaches data for team={team_id} season={season}")
        return []

    coaches_df = dfs[1]
    records = []

    for _, row in coaches_df.iterrows():
        coach_id = str(row["COACH_ID"])
        coach_name = row["COACH_NAME"]
        is_assistant = row.get("IS_ASSISTANT", None)

        # IS_ASSISTANT: 1=Head Coach, 2=Assistant Coach, 3=Trainer, 4=Asst Trainer
        is_head_coach = 1 if is_assistant == 1 else 0

        records.append(
            {
                "team_id": str(team_id),
                "season": season,
                "coach_id": coach_id,
                "coach_name": coach_name,
                "is_head_coach": is_head_coach,
            }
        )

    return records


def insert_coaches(records):
    """Insert coach records into the database."""
    if not records:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executemany(
        """
        INSERT OR REPLACE INTO Coaches (team_id, season, coach_id, coach_name, is_head_coach)
        VALUES (:team_id, :season, :coach_id, :coach_name, :is_head_coach)
        """,
        records,
    )
    conn.commit()
    conn.close()


def verify_results():
    """Verify the collected coaching data."""
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM Coaches").fetchone()[0]
    head_coaches = conn.execute(
        "SELECT COUNT(*) FROM Coaches WHERE is_head_coach = 1"
    ).fetchone()[0]
    assistants = conn.execute(
        "SELECT COUNT(*) FROM Coaches WHERE is_head_coach = 0"
    ).fetchone()[0]
    team_seasons = conn.execute(
        "SELECT COUNT(DISTINCT team_id || '|' || season) FROM Coaches"
    ).fetchone()[0]

    logger.info(f"Total coach records: {total}")
    logger.info(f"Head coaches: {head_coaches}")
    logger.info(f"Assistants/trainers: {assistants}")
    logger.info(f"Unique team-seasons: {team_seasons}")

    # Check for team-seasons with multiple head coaches (mid-season changes)
    multi_hc = conn.execute("""
        SELECT team_id, season, COUNT(*) as n, GROUP_CONCAT(coach_name, ', ')
        FROM Coaches
        WHERE is_head_coach = 1
        GROUP BY team_id, season
        HAVING COUNT(*) > 1
    """).fetchall()

    if multi_hc:
        logger.info(f"\nTeam-seasons with multiple head coaches ({len(multi_hc)}):")
        for row in multi_hc:
            logger.info(f"  {row[0]} {row[1]}: {row[2]} coaches — {row[3]}")

    # Check for team-seasons with NO head coach
    no_hc = conn.execute("""
        SELECT DISTINCT c.team_id, c.season
        FROM Coaches c
        WHERE NOT EXISTS (
            SELECT 1 FROM Coaches c2
            WHERE c2.team_id = c.team_id AND c2.season = c.season AND c2.is_head_coach = 1
        )
    """).fetchall()

    if no_hc:
        logger.info(f"\nTeam-seasons with NO head coach listed ({len(no_hc)}):")
        for row in no_hc:
            logger.info(f"  {row[0]} {row[1]}")

    # Spot-check known coaches
    spot_checks = [
        ("1610612748", "MIA", "Spoelstra"),  # MIA — Spoelstra since 2008
        ("1610612759", "SAS", "Popovich"),  # SAS — Popovich
        ("1610612744", "GSW", "Kerr"),  # GSW — Kerr since 2014
    ]
    logger.info("\nSpot checks:")
    for tid, abbrev, name_fragment in spot_checks:
        rows = conn.execute(
            """
            SELECT season, coach_name FROM Coaches
            WHERE team_id = ? AND is_head_coach = 1 AND coach_name LIKE ?
            ORDER BY season
            """,
            (tid, f"%{name_fragment}%"),
        ).fetchall()
        if rows:
            seasons = [r[0] for r in rows]
            logger.info(
                f"  {abbrev} {rows[0][1]}: {len(rows)} seasons "
                f"({seasons[0]} to {seasons[-1]})"
            )
        else:
            logger.info(f"  {abbrev} {name_fragment}: NOT FOUND")

    conn.close()


def main():
    logger.info("=" * 60)
    logger.info("Coaching data collection starting")
    logger.info("=" * 60)

    # Create table
    create_coaches_table()

    # Get teams
    teams = get_team_ids()
    logger.info(f"Teams: {len(teams)}")
    logger.info(f"Seasons: {len(SEASONS)} ({SEASONS[0]} to {SEASONS[-1]})")

    # Build full list of (team_id, abbrev, season) combos
    all_combos = []
    for team_id, abbrev in teams:
        for season in SEASONS:
            all_combos.append((team_id, abbrev, season))

    logger.info(f"Total API calls needed: {len(all_combos)}")

    # Load progress
    completed = load_progress()
    logger.info(f"Already completed: {len(completed)}")

    remaining = [
        (tid, abbrev, season)
        for tid, abbrev, season in all_combos
        if f"{tid}|{season}" not in completed
    ]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All team-seasons already collected!")
        verify_results()
        return

    # Collect data
    consecutive_errors = 0
    burst_count = 0
    fetched_this_run = 0
    errors_this_run = 0
    start_time = time.time()
    total_coaches_inserted = 0

    for i, (team_id, abbrev, season) in enumerate(remaining):
        # Cooling period
        if fetched_this_run > 0 and fetched_this_run % COOLING_INTERVAL == 0:
            logger.info(
                f"Cooling period: {COOLING_PAUSE}s pause after "
                f"{fetched_this_run} requests..."
            )
            time.sleep(COOLING_PAUSE)
            burst_count = 0

        # Burst pause
        if burst_count >= BURST_SIZE:
            logger.debug(f"Burst complete, pausing {BURST_PAUSE}s...")
            time.sleep(BURST_PAUSE)
            burst_count = 0

        # Attempt fetch with retries
        key = f"{team_id}|{season}"
        success = False

        for attempt in range(MAX_RETRIES):
            try:
                records = fetch_coaches(team_id, season)
                insert_coaches(records)
                total_coaches_inserted += len(records)

                completed.add(key)
                fetched_this_run += 1
                burst_count += 1
                consecutive_errors = 0
                success = True

                head_names = [r["coach_name"] for r in records if r["is_head_coach"]]
                logger.debug(
                    f"  {abbrev} {season}: {len(records)} coaches "
                    f"(HC: {', '.join(head_names) if head_names else 'none'})"
                )
                break

            except Exception as e:
                consecutive_errors += 1
                errors_this_run += 1
                error_msg = str(e)[:120]
                logger.warning(
                    f"Error fetching {abbrev} {season} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}): {error_msg}"
                )

                if consecutive_errors >= 3:
                    logger.warning(
                        f"3 consecutive errors, pausing {CONSECUTIVE_ERROR_PAUSE}s..."
                    )
                    time.sleep(CONSECUTIVE_ERROR_PAUSE)
                    consecutive_errors = 0
                    burst_count = 0
                else:
                    logger.debug(f"Error backoff: {ERROR_PAUSE}s...")
                    time.sleep(ERROR_PAUSE)

        if not success:
            logger.error(
                f"Failed to fetch {abbrev} {season} after {MAX_RETRIES} retries — skipping"
            )

        # Progress logging every 25 calls
        if fetched_this_run > 0 and fetched_this_run % 25 == 0:
            elapsed = time.time() - start_time
            rate = fetched_this_run / elapsed * 60 if elapsed > 0 else 0
            logger.info(
                f"Progress: {fetched_this_run}/{len(remaining)} fetched | "
                f"Coaches inserted: {total_coaches_inserted} | "
                f"Errors: {errors_this_run} | "
                f"Rate: {rate:.1f}/min | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # Save progress every 25 calls
        if fetched_this_run % 25 == 0:
            save_progress(completed)

    # Final save
    save_progress(completed)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Collection complete!")
    logger.info(f"Total fetched: {fetched_this_run}")
    logger.info(f"Total coaches inserted: {total_coaches_inserted}")
    logger.info(f"Total errors: {errors_this_run}")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info("=" * 60)

    # Verify
    logger.info("Verifying results...")
    verify_results()


if __name__ == "__main__":
    main()
