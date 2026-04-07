#!/usr/bin/env python3
"""
collect_pbpstats.py

Collects enriched per-game player stats from the PBPStats API and stores them
in the PBPPlayerGameStats table in our SQLite database.

Uses the free REST API at https://api.pbpstats.com/get-game-stats to fetch
per-game player-level stats that go beyond standard box scores:
  - Shot zone FGM/FGA (AtRim, ShortMidRange, LongMidRange, Corner3, Arc3)
  - Assisted vs unassisted scoring
  - Turnover types (live/dead ball, bad pass, lost ball)
  - Shot quality average
  - Offensive/defensive possessions and usage
  - Assist breakdown by zone
  - Recovered blocks, self offensive rebounds

Data coverage: The API covers 2000-01 through the current season.

Usage:
    python scripts/collect_pbpstats.py --season 2025-2026
    python scripts/collect_pbpstats.py --all
    python scripts/collect_pbpstats.py --all --concurrency 5
    python scripts/collect_pbpstats.py --season 2025-2026 --include-playoffs
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aiohttp
from tqdm import tqdm

from src.database import get_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.pbpstats.com"
GAME_STATS_URL = f"{API_BASE}/get-game-stats"

# Seasons supported by PBPStats (2000-01 onward)
FIRST_SEASON_YEAR = 2000  # "2000-2001" in our DB format

# Stats we extract from the API response for each player.
# Maps our DB column name -> PBPStats API field name.
STAT_MAPPING = {
    # Shot zones
    "at_rim_fgm": "AtRimFGM",
    "at_rim_fga": "AtRimFGA",
    "short_mid_fgm": "ShortMidRangeFGM",
    "short_mid_fga": "ShortMidRangeFGA",
    "long_mid_fgm": "LongMidRangeFGM",
    "long_mid_fga": "LongMidRangeFGA",
    "corner3_fgm": "Corner3FGM",
    "corner3_fga": "Corner3FGA",
    "arc3_fgm": "Arc3FGM",
    "arc3_fga": "Arc3FGA",
    # Assisted / unassisted
    "pts_assisted_2s": "PtsAssisted2s",
    "pts_unassisted_2s": "PtsUnassisted2s",
    "pts_assisted_3s": "PtsAssisted3s",
    "pts_unassisted_3s": "PtsUnassisted3s",
    # Turnovers
    "live_ball_turnovers": "LiveBallTurnovers",
    "dead_ball_turnovers": "DeadBallTurnovers",
    "bad_pass_turnovers": "BadPassTurnovers",
    "lost_ball_turnovers": "LostBallTurnovers",
    # Shot quality
    "shot_quality_avg": "ShotQualityAvg",
    # Possessions
    "off_poss": "OffPoss",
    "def_poss": "DefPoss",
    # Other
    "self_oreb": "SelfOReb",
    "recovered_blocks": "RecoveredBlocks",
}

# DB column names in insertion order (player_id, game_id, then stats)
DB_COLUMNS = ["player_id", "game_id"] + list(STAT_MAPPING.keys())

# SQL for the table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS PBPPlayerGameStats (
    player_id INTEGER,
    game_id TEXT,
    -- Shot zones
    at_rim_fgm INTEGER,
    at_rim_fga INTEGER,
    short_mid_fgm INTEGER,
    short_mid_fga INTEGER,
    long_mid_fgm INTEGER,
    long_mid_fga INTEGER,
    corner3_fgm INTEGER,
    corner3_fga INTEGER,
    arc3_fgm INTEGER,
    arc3_fga INTEGER,
    -- Assisted/unassisted
    pts_assisted_2s REAL,
    pts_unassisted_2s REAL,
    pts_assisted_3s REAL,
    pts_unassisted_3s REAL,
    -- Turnovers
    live_ball_turnovers INTEGER,
    dead_ball_turnovers INTEGER,
    bad_pass_turnovers INTEGER,
    lost_ball_turnovers INTEGER,
    -- Shot quality
    shot_quality_avg REAL,
    -- Possessions
    off_poss INTEGER,
    def_poss INTEGER,
    -- Other
    self_oreb INTEGER,
    recovered_blocks INTEGER,
    PRIMARY KEY (player_id, game_id)
);
"""

INSERT_SQL = f"""
INSERT OR IGNORE INTO PBPPlayerGameStats ({', '.join(DB_COLUMNS)})
VALUES ({', '.join('?' for _ in DB_COLUMNS)})
"""

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def create_table():
    """Create the PBPPlayerGameStats table if it doesn't exist."""
    with get_db() as conn:
        conn.execute(CREATE_TABLE_SQL)
    log.info("PBPPlayerGameStats table ready.")


def get_existing_game_ids() -> set:
    """Return the set of game_ids that already have data in PBPPlayerGameStats."""
    with get_db() as conn:
        try:
            cursor = conn.execute("SELECT DISTINCT game_id FROM PBPPlayerGameStats")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return set()


def get_game_ids_for_season(season: str, include_playoffs: bool = False) -> list[str]:
    """
    Get all completed regular-season game_ids for a given season from our DB.

    Args:
        season: Season string in our DB format, e.g. "2024-2025"
        include_playoffs: If True, also include Post Season games

    Returns:
        List of game_id strings sorted ascending.
    """
    season_types = ["Regular Season"]
    if include_playoffs:
        season_types.append("Post Season")

    placeholders = ", ".join("?" for _ in season_types)
    query = f"""
        SELECT game_id FROM Games
        WHERE season = ? AND status = 3
          AND season_type IN ({placeholders})
        ORDER BY game_id
    """

    with get_db() as conn:
        cursor = conn.execute(query, [season] + season_types)
        return [row[0] for row in cursor.fetchall()]


def save_player_rows(rows: list[tuple]):
    """Batch-insert player stat rows into the database."""
    if not rows:
        return
    with get_db() as conn:
        conn.executemany(INSERT_SQL, rows)


# ---------------------------------------------------------------------------
# Season format conversion
# ---------------------------------------------------------------------------


def db_season_to_api_season(db_season: str) -> str:
    """
    Convert our DB season format to PBPStats API format.
    "2024-2025" -> "2024-25"
    """
    start, end = db_season.split("-")
    return f"{start}-{end[2:]}"


def get_all_seasons() -> list[str]:
    """
    Return all seasons from our DB that are within PBPStats coverage
    (2000-01 onward).
    """
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT season FROM Games WHERE status = 3 ORDER BY season"
        )
        seasons = [row[0] for row in cursor.fetchall()]

    # Filter to seasons within PBPStats coverage
    return [s for s in seasons if int(s.split("-")[0]) >= FIRST_SEASON_YEAR]


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------


def parse_game_response(game_id: str, data: dict) -> list[tuple]:
    """
    Parse the API response for a single game into DB rows.

    The API returns:
    {
      "stats": {
        "Home": {
          "FullGame": [
            { "EntityId": 12345, "Name": "Player Name", stat_field: value, ... },
            ...
          ],
          "1": [...], "2": [...], ...
        },
        "Away": { ... }
      },
      ...
    }

    We extract only the "FullGame" period stats for actual players
    (EntityId != 0, Name != "Team").
    """
    rows = []
    stats = data.get("stats", {})

    for side in ("Home", "Away"):
        side_data = stats.get(side, {})
        full_game_players = side_data.get("FullGame", [])

        for player in full_game_players:
            entity_id = player.get("EntityId", 0)
            name = player.get("Name", "")

            # Skip team aggregate rows
            if entity_id == 0 or name == "Team":
                continue

            # Build the row
            row = [entity_id, game_id]
            for db_col, api_field in STAT_MAPPING.items():
                value = player.get(api_field)
                # Keep None for missing fields (will be NULL in DB)
                row.append(value)

            rows.append(tuple(row))

    return rows


async def fetch_game(
    session: aiohttp.ClientSession,
    game_id: str,
    semaphore: asyncio.Semaphore,
    retry_count: int = 3,
    retry_delay: float = 5.0,
) -> tuple[str, list[tuple] | None, str | None]:
    """
    Fetch enriched player stats for a single game from the API.

    Returns:
        (game_id, rows, error_message)
        rows is None if the request failed after all retries.
    """
    params = {"GameId": game_id, "Type": "Player"}

    for attempt in range(retry_count):
        async with semaphore:
            try:
                async with session.get(
                    GAME_STATS_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rows = parse_game_response(game_id, data)
                        return (game_id, rows, None)
                    elif resp.status == 404:
                        # Game not available in PBPStats
                        return (game_id, [], f"404 Not Found")
                    elif resp.status == 429:
                        # Rate limited -- back off
                        wait = retry_delay * (2**attempt)
                        log.warning(
                            "Rate limited on %s, waiting %.1fs (attempt %d/%d)",
                            game_id,
                            wait,
                            attempt + 1,
                            retry_count,
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        text = await resp.text()
                        error = f"HTTP {resp.status}: {text[:200]}"
                        if attempt < retry_count - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return (game_id, None, error)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error = f"{type(e).__name__}: {e}"
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return (game_id, None, error)

    return (game_id, None, "Exhausted retries")


async def process_season(
    season: str,
    include_playoffs: bool = False,
    concurrency: int = 3,
    batch_size: int = 50,
):
    """
    Fetch and store enriched stats for all games in a season.

    Args:
        season: Season in DB format, e.g. "2024-2025"
        include_playoffs: Include Post Season games
        concurrency: Max concurrent API requests
        batch_size: Number of games to process before writing to DB
    """
    log.info("Processing season %s ...", season)

    # Get game IDs from our database
    all_game_ids = get_game_ids_for_season(season, include_playoffs)
    if not all_game_ids:
        log.warning("No games found for season %s", season)
        return

    # Filter out games we already have
    existing = get_existing_game_ids()
    game_ids = [gid for gid in all_game_ids if gid not in existing]

    log.info(
        "Season %s: %d total games, %d already collected, %d remaining",
        season,
        len(all_game_ids),
        len(all_game_ids) - len(game_ids),
        len(game_ids),
    )

    if not game_ids:
        log.info("Season %s: nothing to do.", season)
        return

    semaphore = asyncio.Semaphore(concurrency)
    stats = {"success": 0, "empty": 0, "error": 0, "players": 0}
    failed_games = []

    async with aiohttp.ClientSession() as session:
        # Process in batches to limit memory and provide incremental saves
        for batch_start in range(0, len(game_ids), batch_size):
            batch = game_ids[batch_start : batch_start + batch_size]
            tasks = [fetch_game(session, gid, semaphore) for gid in batch]

            batch_rows = []
            desc = f"{season} [{batch_start+1}-{min(batch_start+len(batch), len(game_ids))}/{len(game_ids)}]"

            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=desc,
                unit="game",
            ):
                game_id, rows, error = await coro

                if error and rows is None:
                    stats["error"] += 1
                    failed_games.append((game_id, error))
                    log.debug("Failed %s: %s", game_id, error)
                elif rows is not None and len(rows) == 0:
                    stats["empty"] += 1
                    if error:
                        log.debug("Empty %s: %s", game_id, error)
                else:
                    stats["success"] += 1
                    stats["players"] += len(rows)
                    batch_rows.extend(rows)

            # Save batch to database
            if batch_rows:
                save_player_rows(batch_rows)
                log.info("Saved batch: %d player-game rows", len(batch_rows))

    # Summary
    log.info(
        "Season %s complete: %d games OK (%d players), %d empty, %d errors",
        season,
        stats["success"],
        stats["players"],
        stats["empty"],
        stats["error"],
    )

    if failed_games:
        log.warning("Failed games for %s:", season)
        for gid, err in failed_games[:20]:
            log.warning("  %s: %s", gid, err)
        if len(failed_games) > 20:
            log.warning("  ... and %d more", len(failed_games) - 20)


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


def report_coverage():
    """Print a summary of current PBPPlayerGameStats coverage."""
    with get_db() as conn:
        try:
            cursor = conn.execute("""
                SELECT
                    g.season,
                    COUNT(DISTINCT g.game_id) as total_games,
                    COUNT(DISTINCT p.game_id) as covered_games,
                    COUNT(p.player_id) as player_rows
                FROM Games g
                LEFT JOIN PBPPlayerGameStats p ON g.game_id = p.game_id
                WHERE g.status = 3 AND g.season_type = 'Regular Season'
                GROUP BY g.season
                ORDER BY g.season
            """)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            log.info("PBPPlayerGameStats table does not exist yet.")
            return

    print(f"\n{'Season':<12} {'Total':>6} {'Covered':>8} {'Pct':>6} {'Players':>9}")
    print("-" * 45)
    total_games = 0
    total_covered = 0
    total_players = 0

    for season, games, covered, players in rows:
        start_year = int(season.split("-")[0])
        if start_year < FIRST_SEASON_YEAR:
            continue
        pct = (covered / games * 100) if games > 0 else 0
        print(f"{season:<12} {games:>6} {covered:>8} {pct:>5.1f}% {players:>9}")
        total_games += games
        total_covered += covered
        total_players += players

    print("-" * 45)
    pct = (total_covered / total_games * 100) if total_games > 0 else 0
    print(
        f"{'TOTAL':<12} {total_games:>6} {total_covered:>8} {pct:>5.1f}% {total_players:>9}"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect enriched per-game player stats from PBPStats API"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--season",
        type=str,
        help='Season in DB format, e.g. "2024-2025"',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all seasons from 2000-2001 onward",
    )
    group.add_argument(
        "--coverage",
        action="store_true",
        help="Print coverage report and exit",
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Include Post Season games (default: regular season only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent API requests (default: 3, be respectful)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Games per batch before writing to DB (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.coverage:
        report_coverage()
        return

    # Create table if needed
    create_table()

    if args.all:
        seasons = get_all_seasons()
        log.info(
            "Will process %d seasons: %s ... %s", len(seasons), seasons[0], seasons[-1]
        )
    else:
        seasons = [args.season]
        # Validate season format
        try:
            start, end = args.season.split("-")
            assert len(start) == 4 and len(end) == 4
            assert int(start) >= FIRST_SEASON_YEAR
        except (ValueError, AssertionError):
            log.error(
                "Invalid season format: %s (expected e.g. '2024-2025', min '2000-2001')",
                args.season,
            )
            sys.exit(1)

    start_time = time.time()

    for season in seasons:
        asyncio.run(
            process_season(
                season,
                include_playoffs=args.include_playoffs,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
            )
        )

    elapsed = time.time() - start_time
    log.info("All done in %.1f minutes.", elapsed / 60)

    # Print final coverage
    report_coverage()


if __name__ == "__main__":
    main()
