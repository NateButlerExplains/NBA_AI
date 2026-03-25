"""
collect_lineup_stats.py

Collects supplemental lineup statistics from NBA API endpoints:
1. LeagueDashLineups — Base + FourFactors for 2-man and 5-man lineups
2. LeagueLineupViz — Advanced efficiency for 5-man lineups (50+ min)
3. TeamPlayerOnOffDetails — Individual on/off impact per team (Advanced)

Rate limiting: burst of 5 requests, then 15s pause.
On error: wait 30s and retry once; on second failure, skip and log.
Progress saved to data/lineup_stats_progress.json for resume capability.

Usage:
    python scripts/collect_lineup_stats.py [--step 1|2|3|4|all] [--reset]

    --step 1: Create tables only
    --step 2: Collect LeagueDashLineups
    --step 3: Collect LeagueLineupViz
    --step 4: Collect TeamPlayerOnOffDetails
    --step all: Run all steps (default)
    --reset: Clear progress file and start fresh
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nba_api.stats.endpoints import (
    leaguedashlineups,
    leaguelineupviz,
    teamplayeronoffdetails,
)

from src.config import config

DB_PATH = config["database"]["path"]
PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "lineup_stats_progress.json",
)

# Seasons to collect
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2007, 2025)]  # 2007-08 to 2024-25

# Rate limiting
BURST_SIZE = 5
BURST_PAUSE = 15  # seconds between bursts
RETRY_WAIT = 30  # seconds before retry on error
REQUEST_TIMEOUT = 120  # seconds per request

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress() -> dict:
    """Load progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "lineups_done": [],  # list of "season|group_qty|measure" strings
        "lineupsv_done": [],  # list of "season" strings
        "onoff_done": [],  # list of "team_id|season" strings
    }


def save_progress(progress: dict):
    """Save progress to JSON file."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def reset_progress():
    """Delete progress file."""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log.info("Progress file deleted.")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Burst-based rate limiter with retry logic."""

    def __init__(self, burst_size=BURST_SIZE, burst_pause=BURST_PAUSE):
        self.burst_size = burst_size
        self.burst_pause = burst_pause
        self.call_count = 0
        self.total_calls = 0
        self.start_time = time.time()

    def pre_request(self):
        """Call before each API request. Pauses if burst limit reached."""
        if self.call_count > 0 and self.call_count % self.burst_size == 0:
            log.info(
                f"  Burst pause: {self.burst_pause}s after {self.call_count} calls "
                f"({self.total_calls} total)"
            )
            time.sleep(self.burst_pause)
        elif self.call_count > 0:
            # Small delay between every request to be conservative
            time.sleep(2)

    def post_request(self):
        """Call after each successful request."""
        self.call_count += 1
        self.total_calls += 1

    def elapsed(self) -> str:
        """Return elapsed time as a formatted string."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}m {seconds}s"


def api_call_with_retry(func, rate_limiter, label=""):
    """
    Execute an API call with rate limiting and retry logic.

    Args:
        func: Callable that makes the API request and returns data.
        rate_limiter: RateLimiter instance.
        label: Description for logging.

    Returns:
        Result from func(), or None on failure.
    """
    rate_limiter.pre_request()

    try:
        result = func()
        rate_limiter.post_request()
        return result
    except Exception as e:
        error_str = str(e).lower()
        log.warning(f"  Error on {label}: {e}")

        # Permanent errors — don't retry
        if any(code in error_str for code in ("400", "403", "404", "not found")):
            log.error(f"  Permanent error for {label}, skipping.")
            return None

        # Transient error — retry once after wait
        log.info(f"  Retrying {label} in {RETRY_WAIT}s...")
        time.sleep(RETRY_WAIT)

        try:
            rate_limiter.pre_request()
            result = func()
            rate_limiter.post_request()
            log.info(f"  Retry succeeded for {label}")
            return result
        except Exception as e2:
            log.error(f"  Retry failed for {label}: {e2}. Skipping.")
            return None


# ---------------------------------------------------------------------------
# Step 1: Create tables
# ---------------------------------------------------------------------------


def create_tables(db_path=DB_PATH):
    """Create LineupStats and PlayerOnOff tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LineupStats (
            group_id TEXT,
            group_name TEXT,
            team_id INTEGER,
            team_abbreviation TEXT,
            season TEXT,
            group_quantity INTEGER,
            measure_type TEXT,
            gp INTEGER,
            w INTEGER,
            l INTEGER,
            min REAL,
            plus_minus REAL,
            -- Base stats (nullable for non-Base measures)
            pts REAL, fgm REAL, fga REAL, fg_pct REAL,
            fg3m REAL, fg3a REAL, fg3_pct REAL,
            ftm REAL, fta REAL, ft_pct REAL,
            oreb REAL, dreb REAL, reb REAL,
            ast REAL, tov REAL, stl REAL, blk REAL,
            -- Four Factors (nullable)
            efg_pct REAL, fta_rate REAL, tm_tov_pct REAL, oreb_pct REAL,
            opp_efg_pct REAL, opp_fta_rate REAL, opp_tov_pct REAL, opp_oreb_pct REAL,
            -- Advanced (from LineupViz, nullable)
            off_rating REAL, def_rating REAL, net_rating REAL, pace REAL, ts_pct REAL,
            PRIMARY KEY (group_id, season, measure_type)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PlayerOnOff (
            player_id INTEGER,
            player_name TEXT,
            team_id INTEGER,
            team_abbreviation TEXT,
            season TEXT,
            court_status TEXT,
            gp INTEGER, w INTEGER, l INTEGER,
            min REAL, plus_minus REAL,
            off_rating REAL, def_rating REAL, net_rating REAL,
            PRIMARY KEY (player_id, team_id, season, court_status)
        )
    """)

    conn.commit()
    conn.close()
    log.info("Tables created: LineupStats, PlayerOnOff")


# ---------------------------------------------------------------------------
# Step 2: Collect LeagueDashLineups
# ---------------------------------------------------------------------------


def collect_league_dash_lineups(db_path=DB_PATH):
    """
    Collect LeagueDashLineups for all seasons, 2-man and 5-man groups,
    Base and FourFactors measures.

    Total calls: 18 seasons x 2 groups x 2 measures = 72
    """
    progress = load_progress()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rate_limiter = RateLimiter()

    group_quantities = [2, 5]
    measures = ["Base", "Four Factors"]

    # Count remaining work
    total_combos = len(SEASONS) * len(group_quantities) * len(measures)
    done_count = len(progress["lineups_done"])
    remaining = total_combos - done_count
    log.info(f"LeagueDashLineups: {remaining}/{total_combos} combinations remaining")

    total_rows = 0

    for season in SEASONS:
        for gq in group_quantities:
            for measure in measures:
                key = f"{season}|{gq}|{measure}"
                if key in progress["lineups_done"]:
                    continue

                label = f"LeagueDashLineups season={season} gq={gq} measure={measure}"
                log.info(f"Fetching {label}...")

                def make_call(s=season, g=gq, m=measure):
                    resp = leaguedashlineups.LeagueDashLineups(
                        group_quantity=str(g),
                        season=s,
                        measure_type_detailed_defense=m,
                        timeout=REQUEST_TIMEOUT,
                    )
                    return resp.get_dict()

                data = api_call_with_retry(make_call, rate_limiter, label)
                if data is None:
                    continue

                # Parse and insert
                rows_inserted = 0
                for rs in data.get("resultSets", []):
                    headers = [h.upper() for h in rs["headers"]]
                    for row in rs["rowSet"]:
                        row_dict = dict(zip(headers, row))

                        group_id = row_dict.get("GROUP_ID", "")
                        group_name = row_dict.get("GROUP_NAME", "")
                        team_id = row_dict.get("TEAM_ID")
                        team_abbr = row_dict.get("TEAM_ABBREVIATION", "")

                        if measure == "Base":
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO LineupStats (
                                    group_id, group_name, team_id, team_abbreviation,
                                    season, group_quantity, measure_type,
                                    gp, w, l, min, plus_minus,
                                    pts, fgm, fga, fg_pct,
                                    fg3m, fg3a, fg3_pct,
                                    ftm, fta, ft_pct,
                                    oreb, dreb, reb,
                                    ast, tov, stl, blk
                                ) VALUES (
                                    ?, ?, ?, ?, ?, ?, ?,
                                    ?, ?, ?, ?, ?,
                                    ?, ?, ?, ?,
                                    ?, ?, ?,
                                    ?, ?, ?,
                                    ?, ?, ?,
                                    ?, ?, ?, ?
                                )
                            """,
                                (
                                    group_id,
                                    group_name,
                                    team_id,
                                    team_abbr,
                                    season,
                                    gq,
                                    "Base",
                                    row_dict.get("GP"),
                                    row_dict.get("W"),
                                    row_dict.get("L"),
                                    row_dict.get("MIN"),
                                    row_dict.get("PLUS_MINUS"),
                                    row_dict.get("PTS"),
                                    row_dict.get("FGM"),
                                    row_dict.get("FGA"),
                                    row_dict.get("FG_PCT"),
                                    row_dict.get("FG3M"),
                                    row_dict.get("FG3A"),
                                    row_dict.get("FG3_PCT"),
                                    row_dict.get("FTM"),
                                    row_dict.get("FTA"),
                                    row_dict.get("FT_PCT"),
                                    row_dict.get("OREB"),
                                    row_dict.get("DREB"),
                                    row_dict.get("REB"),
                                    row_dict.get("AST"),
                                    row_dict.get("TOV"),
                                    row_dict.get("STL"),
                                    row_dict.get("BLK"),
                                ),
                            )
                        elif measure == "Four Factors":
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO LineupStats (
                                    group_id, group_name, team_id, team_abbreviation,
                                    season, group_quantity, measure_type,
                                    gp, w, l, min,
                                    efg_pct, fta_rate, tm_tov_pct, oreb_pct,
                                    opp_efg_pct, opp_fta_rate, opp_tov_pct, opp_oreb_pct
                                ) VALUES (
                                    ?, ?, ?, ?, ?, ?, ?,
                                    ?, ?, ?, ?,
                                    ?, ?, ?, ?,
                                    ?, ?, ?, ?
                                )
                            """,
                                (
                                    group_id,
                                    group_name,
                                    team_id,
                                    team_abbr,
                                    season,
                                    gq,
                                    "FourFactors",
                                    row_dict.get("GP"),
                                    row_dict.get("W"),
                                    row_dict.get("L"),
                                    row_dict.get("MIN"),
                                    row_dict.get("EFG_PCT"),
                                    row_dict.get("FTA_RATE"),
                                    row_dict.get("TM_TOV_PCT"),
                                    row_dict.get("OREB_PCT"),
                                    row_dict.get("OPP_EFG_PCT"),
                                    row_dict.get("OPP_FTA_RATE"),
                                    row_dict.get("OPP_TOV_PCT"),
                                    row_dict.get("OPP_OREB_PCT"),
                                ),
                            )

                        rows_inserted += 1

                conn.commit()
                total_rows += rows_inserted
                log.info(f"  Inserted {rows_inserted} rows for {key}")

                # Mark done and save progress
                progress["lineups_done"].append(key)
                save_progress(progress)

    conn.close()
    log.info(
        f"LeagueDashLineups complete: {total_rows} total rows inserted "
        f"in {rate_limiter.elapsed()}"
    )
    return total_rows


# ---------------------------------------------------------------------------
# Step 3: Collect LeagueLineupViz
# ---------------------------------------------------------------------------


def collect_league_lineup_viz(db_path=DB_PATH):
    """
    Collect LeagueLineupViz for 5-man lineups with 50+ minutes.
    Updates existing LineupStats rows with advanced metrics, or inserts new ones.

    Total calls: 18 seasons x 1 = 18
    """
    progress = load_progress()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rate_limiter = RateLimiter()

    remaining = len(SEASONS) - len(progress["lineupsv_done"])
    log.info(f"LeagueLineupViz: {remaining}/{len(SEASONS)} seasons remaining")

    total_rows = 0

    for season in SEASONS:
        if season in progress["lineupsv_done"]:
            continue

        label = f"LeagueLineupViz season={season}"
        log.info(f"Fetching {label}...")

        def make_call(s=season):
            resp = leaguelineupviz.LeagueLineupViz(
                minutes_min=50,
                group_quantity="5",
                season=s,
                timeout=REQUEST_TIMEOUT,
            )
            return resp.get_dict()

        data = api_call_with_retry(make_call, rate_limiter, label)
        if data is None:
            continue

        rows_updated = 0
        for rs in data.get("resultSets", []):
            headers = [h.upper() for h in rs["headers"]]
            for row in rs["rowSet"]:
                row_dict = dict(zip(headers, row))

                group_id = row_dict.get("GROUP_ID", "")
                group_name = row_dict.get("GROUP_NAME", "")
                team_id = row_dict.get("TEAM_ID")
                team_abbr = row_dict.get("TEAM_ABBREVIATION", "")

                # Try to update existing Base row first
                cursor.execute(
                    """
                    UPDATE LineupStats SET
                        off_rating = ?, def_rating = ?, net_rating = ?,
                        pace = ?, ts_pct = ?
                    WHERE group_id = ? AND season = ? AND measure_type = 'Base'
                """,
                    (
                        row_dict.get("OFF_RATING"),
                        row_dict.get("DEF_RATING"),
                        row_dict.get("NET_RATING"),
                        row_dict.get("PACE"),
                        row_dict.get("TS_PCT"),
                        group_id,
                        season,
                    ),
                )

                if cursor.rowcount == 0:
                    # No existing Base row — insert a minimal row with advanced stats
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO LineupStats (
                            group_id, group_name, team_id, team_abbreviation,
                            season, group_quantity, measure_type,
                            min, off_rating, def_rating, net_rating, pace, ts_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            group_id,
                            group_name,
                            team_id,
                            team_abbr,
                            season,
                            5,
                            "Base",
                            row_dict.get("MIN"),
                            row_dict.get("OFF_RATING"),
                            row_dict.get("DEF_RATING"),
                            row_dict.get("NET_RATING"),
                            row_dict.get("PACE"),
                            row_dict.get("TS_PCT"),
                        ),
                    )

                rows_updated += 1

        conn.commit()
        total_rows += rows_updated
        log.info(f"  Updated/inserted {rows_updated} rows for {season}")

        progress["lineupsv_done"].append(season)
        save_progress(progress)

    conn.close()
    log.info(
        f"LeagueLineupViz complete: {total_rows} total rows updated/inserted "
        f"in {rate_limiter.elapsed()}"
    )
    return total_rows


# ---------------------------------------------------------------------------
# Step 4: Collect TeamPlayerOnOffDetails
# ---------------------------------------------------------------------------


def collect_team_player_on_off(db_path=DB_PATH):
    """
    Collect TeamPlayerOnOffDetails (Advanced measure) for all teams and seasons.

    Total calls: 30 teams x 18 seasons = 540
    Estimated time: ~36 minutes with burst limiting
    """
    progress = load_progress()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rate_limiter = RateLimiter()

    # Get team IDs from database
    team_rows = cursor.execute(
        "SELECT DISTINCT team_id FROM Teams WHERE team_id > 0 ORDER BY team_id"
    ).fetchall()
    team_ids = [r[0] for r in team_rows]
    log.info(f"Found {len(team_ids)} teams")

    total_combos = len(team_ids) * len(SEASONS)
    done_count = len(progress["onoff_done"])
    remaining = total_combos - done_count
    log.info(
        f"TeamPlayerOnOffDetails: {remaining}/{total_combos} combinations remaining"
    )

    total_rows = 0

    for team_id in team_ids:
        for season in SEASONS:
            key = f"{team_id}|{season}"
            if key in progress["onoff_done"]:
                continue

            label = f"OnOff team={team_id} season={season}"
            log.info(f"Fetching {label}...")

            def make_call(tid=team_id, s=season):
                resp = teamplayeronoffdetails.TeamPlayerOnOffDetails(
                    team_id=tid,
                    season=s,
                    measure_type_detailed_defense="Advanced",
                    timeout=REQUEST_TIMEOUT,
                )
                return resp.get_dict()

            data = api_call_with_retry(make_call, rate_limiter, label)
            if data is None:
                # Still mark as done to avoid infinite retries on persistent failures
                progress["onoff_done"].append(key)
                save_progress(progress)
                continue

            rows_inserted = 0

            # Process both On and Off court result sets
            for rs in data.get("resultSets", []):
                rs_name = rs.get("name", "")
                # Skip the Overall result set
                if "Overall" in rs_name:
                    continue

                headers = [h.upper() for h in rs["headers"]]
                for row in rs["rowSet"]:
                    row_dict = dict(zip(headers, row))

                    player_id = row_dict.get("VS_PLAYER_ID")
                    player_name = row_dict.get("VS_PLAYER_NAME", "")
                    court_status = row_dict.get("COURT_STATUS", "")

                    if not player_id or not court_status:
                        continue

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO PlayerOnOff (
                            player_id, player_name, team_id, team_abbreviation,
                            season, court_status,
                            gp, w, l, min, plus_minus,
                            off_rating, def_rating, net_rating
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            player_id,
                            player_name,
                            row_dict.get("TEAM_ID"),
                            row_dict.get("TEAM_ABBREVIATION", ""),
                            season,
                            court_status,
                            row_dict.get("GP"),
                            row_dict.get("W"),
                            row_dict.get("L"),
                            row_dict.get("MIN"),
                            row_dict.get("PLUS_MINUS"),
                            row_dict.get("OFF_RATING"),
                            row_dict.get("DEF_RATING"),
                            row_dict.get("NET_RATING"),
                        ),
                    )
                    rows_inserted += 1

            conn.commit()
            total_rows += rows_inserted
            if rows_inserted > 0:
                log.info(f"  Inserted {rows_inserted} rows for {key}")

            progress["onoff_done"].append(key)
            save_progress(progress)

    conn.close()
    log.info(
        f"TeamPlayerOnOffDetails complete: {total_rows} total rows inserted "
        f"in {rate_limiter.elapsed()}"
    )
    return total_rows


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_data(db_path=DB_PATH):
    """Check row counts and data sanity."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    log.info("=" * 60)
    log.info("DATA VERIFICATION")
    log.info("=" * 60)

    # LineupStats counts
    cursor.execute("SELECT COUNT(*) FROM LineupStats")
    total_lineups = cursor.fetchone()[0]
    log.info(f"LineupStats total rows: {total_lineups:,}")

    cursor.execute("""
        SELECT group_quantity, measure_type, COUNT(*)
        FROM LineupStats
        GROUP BY group_quantity, measure_type
        ORDER BY group_quantity, measure_type
    """)
    for gq, mt, cnt in cursor.fetchall():
        log.info(f"  group_quantity={gq}, measure_type={mt}: {cnt:,} rows")

    cursor.execute("""
        SELECT season, COUNT(*)
        FROM LineupStats
        WHERE measure_type = 'Base' AND group_quantity = 5
        GROUP BY season ORDER BY season
    """)
    log.info("  5-man Base rows per season:")
    for season, cnt in cursor.fetchall():
        log.info(f"    {season}: {cnt:,}")

    # Advanced stats coverage
    cursor.execute("""
        SELECT COUNT(*) FROM LineupStats
        WHERE off_rating IS NOT NULL AND measure_type = 'Base'
    """)
    adv_count = cursor.fetchone()[0]
    log.info(f"  Base rows with advanced stats (off_rating): {adv_count:,}")

    # Sanity check: off_rating range
    cursor.execute("""
        SELECT MIN(off_rating), AVG(off_rating), MAX(off_rating)
        FROM LineupStats WHERE off_rating IS NOT NULL
    """)
    row = cursor.fetchone()
    if row and row[0] is not None:
        log.info(f"  off_rating range: [{row[0]:.1f}, {row[2]:.1f}], avg={row[1]:.1f}")

    # PlayerOnOff counts
    cursor.execute("SELECT COUNT(*) FROM PlayerOnOff")
    total_onoff = cursor.fetchone()[0]
    log.info(f"\nPlayerOnOff total rows: {total_onoff:,}")

    cursor.execute("""
        SELECT court_status, COUNT(*)
        FROM PlayerOnOff
        GROUP BY court_status
    """)
    for status, cnt in cursor.fetchall():
        log.info(f"  court_status={status}: {cnt:,} rows")

    cursor.execute("""
        SELECT season, COUNT(*)
        FROM PlayerOnOff
        WHERE court_status = 'On'
        GROUP BY season ORDER BY season
    """)
    log.info("  'On' rows per season:")
    for season, cnt in cursor.fetchall():
        log.info(f"    {season}: {cnt:,}")

    # Sanity check: net_rating range
    cursor.execute("""
        SELECT MIN(net_rating), AVG(net_rating), MAX(net_rating)
        FROM PlayerOnOff WHERE net_rating IS NOT NULL
    """)
    row = cursor.fetchone()
    if row and row[0] is not None:
        log.info(f"  net_rating range: [{row[0]:.1f}, {row[2]:.1f}], avg={row[1]:.1f}")

    # Distinct players
    cursor.execute("SELECT COUNT(DISTINCT player_id) FROM PlayerOnOff")
    n_players = cursor.fetchone()[0]
    log.info(f"  Distinct players: {n_players:,}")

    conn.close()
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Collect lineup stats from NBA API")
    parser.add_argument(
        "--step",
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset progress and start fresh",
    )
    args = parser.parse_args()

    if args.reset:
        reset_progress()

    overall_start = time.time()

    if args.step in ("1", "all"):
        log.info("=" * 60)
        log.info("STEP 1: Create tables")
        log.info("=" * 60)
        create_tables()

    if args.step in ("2", "all"):
        log.info("=" * 60)
        log.info("STEP 2: Collect LeagueDashLineups (72 API calls)")
        log.info("=" * 60)
        collect_league_dash_lineups()

    if args.step in ("3", "all"):
        log.info("=" * 60)
        log.info("STEP 3: Collect LeagueLineupViz (18 API calls)")
        log.info("=" * 60)
        collect_league_lineup_viz()

    if args.step in ("4", "all"):
        log.info("=" * 60)
        log.info("STEP 4: Collect TeamPlayerOnOffDetails (540 API calls)")
        log.info("=" * 60)
        collect_team_player_on_off()

    # Always verify at the end
    try:
        verify_data()
    except Exception as e:
        log.warning(f"Verification failed (tables may not exist yet): {e}")

    elapsed = time.time() - overall_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log.info(f"\nTotal collection time: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
