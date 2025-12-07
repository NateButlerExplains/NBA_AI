"""
Betting data collection module with 3-tier data source strategy.

=============================================================================
3-TIER FETCHING STRATEGY
=============================================================================

Tier 1 - ESPN API (Live Window: -7 to +2 days):
  - Primary source for real-time and recent data
  - Provides DraftKings odds via summary endpoint
  - Includes spreads, totals, moneylines with odds (-110, etc.)
  - Used automatically by web app for current games

Tier 2 - Covers.com Matchups (On-Demand Finalization):
  - Used for games >7 days old that need closing line finalization
  - Fetches by date (1 API call per date with unfinalized games)
  - Provides closing spreads and totals (no odds)
  - Triggered automatically when ESPN window expires

Tier 3 - Covers.com Team Schedules (Historical Backfill):
  - Bulk backfill via CLI with --backfill flag
  - Fetches all home games per team (30 API calls per season)
  - Best for populating historical seasons
  - Rate-limited to respect Covers.com

=============================================================================
SINGLE ROW PER GAME DESIGN
=============================================================================

Each game has ONE row in the Betting table that gets updated over time:
  1. Created with opening lines (Tier 1 ESPN, ~1-2 days before game)
  2. Updated with current lines (Tier 1 ESPN, as game approaches)
  3. Finalized with closing lines (Tier 1 ESPN or Tier 2 Covers)

The lines_finalized flag indicates we have the final closing line and
won't update this game anymore.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests
from tqdm import tqdm

from src.config import config

logger = logging.getLogger(__name__)

DB_PATH = config["database"]["path"]

# ESPN API configuration
ESPN_SCOREBOARD_URL = (
    "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
ESPN_SUMMARY_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"

# Fetching thresholds
FUTURE_CUTOFF_DAYS = 2  # Don't fetch for games more than 2 days out
ESPN_LOOKBACK_DAYS = 7  # ESPN API has data for ~7 days back
COVERS_FINALIZATION_DELAY_DAYS = 1  # Wait 1 day after ESPN window before using Covers


def _get_current_season(now: datetime) -> str:
    """
    Determine current NBA season based on date.
    NBA season runs Oct-June, so Oct-Dec is start year, Jan-Sept is end year.

    Args:
        now: Current datetime

    Returns:
        Season string like "2024-2025"
    """
    year = now.year
    month = now.month

    # Oct-Dec: current year is start of season (e.g., Oct 2024 → 2024-2025)
    if month >= 10:
        return f"{year}-{year + 1}"
    # Jan-Sept: current year is end of season (e.g., Mar 2025 → 2024-2025)
    else:
        return f"{year - 1}-{year}"


# =============================================================================
# Schema Definition - Single Row Per Game
# =============================================================================

BETTING_SCHEMA = """
CREATE TABLE IF NOT EXISTS Betting (
    -- Primary key = one row per game
    game_id TEXT PRIMARY KEY,                 -- NBA game_id (FK to Games)
    
    -- ESPN mapping (for Tier 1 fetches)
    espn_event_id TEXT,                       -- ESPN event ID for reference
    
    -- Closing Spread (from home team perspective, negative = home favored)
    spread REAL,                              -- Home team spread: -10.5 means home favored by 10.5
    spread_result TEXT,                       -- 'W', 'L', 'P' (home team perspective)
    spread_home_odds INTEGER,                 -- Spread odds for home: -110 (ESPN only)
    spread_away_odds INTEGER,                 -- Spread odds for away: -110 (ESPN only)
    
    -- Closing Total (Over/Under)
    total REAL,                               -- Over/under line: 224.5
    ou_result TEXT,                           -- 'O', 'U', 'P' (over/under/push)
    over_odds INTEGER,                        -- Over odds: -105 (ESPN only)
    under_odds INTEGER,                       -- Under odds: -115 (ESPN only)
    
    -- Moneylines (ESPN only, Covers doesn't provide these)
    home_moneyline INTEGER,                   -- Home ML: -485 (favored) or +150 (underdog)
    away_moneyline INTEGER,                   -- Away ML: +370 (underdog) or -200 (favored)
    
    -- Metadata
    source TEXT NOT NULL DEFAULT 'ESPN',      -- 'ESPN', 'Covers', 'Manual'
    lines_finalized INTEGER NOT NULL DEFAULT 0,  -- 1 = closing lines confirmed, won't update
    
    -- Timestamps
    created_at TEXT NOT NULL,                 -- When this row was created (ISO 8601)
    updated_at TEXT NOT NULL,                 -- When this row was last updated (ISO 8601)
    
    -- Constraints
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_betting_lines_finalized ON Betting(lines_finalized);
CREATE INDEX IF NOT EXISTS idx_betting_source ON Betting(source);
"""


def create_betting_tables(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create Betting table (single row per game schema)."""
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    try:
        conn.executescript(BETTING_SCHEMA)
        conn.commit()
        logger.info("Betting table created/verified")
    finally:
        if close_conn:
            conn.close()


# =============================================================================
# ESPN API Functions
# =============================================================================


def get_espn_event_id(
    game_id: str, game_date: str, home_team: str, away_team: str
) -> Optional[str]:
    """
    Get ESPN event ID for an NBA game.

    Uses ESPNGameMapping cache if available, otherwise fetches from ESPN scoreboard.

    Args:
        game_id: NBA game ID
        game_date: Game date in YYYY-MM-DD format
        home_team: Home team tricode (e.g., "BOS")
        away_team: Away team tricode (e.g., "LAL")

    Returns:
        ESPN event ID or None if not found
    """
    # Check cache first
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT espn_event_id FROM ESPNGameMapping WHERE nba_game_id = ?",
            (game_id,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

    # Fetch from ESPN scoreboard
    date_formatted = game_date.replace("-", "")
    url = f"{ESPN_SCOREBOARD_URL}?dates={date_formatted}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])
        for event in events:
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            if len(competitors) >= 2:
                # ESPN: competitors[0] is home, competitors[1] is away
                espn_home = competitors[0].get("team", {}).get("abbreviation", "")
                espn_away = competitors[1].get("team", {}).get("abbreviation", "")

                # Normalize team abbreviations (ESPN uses different codes sometimes)
                if _teams_match(home_team, espn_home) and _teams_match(
                    away_team, espn_away
                ):
                    espn_id = event["id"]

                    # Cache the mapping
                    _cache_espn_mapping(
                        game_id, espn_id, game_date, home_team, away_team
                    )
                    return espn_id

        logger.warning(
            f"No ESPN match found for {away_team}@{home_team} on {game_date}"
        )
        return None

    except requests.RequestException as e:
        logger.error(f"ESPN scoreboard request failed: {e}")
        return None


def _teams_match(nba_code: str, espn_code: str) -> bool:
    """Check if team codes match (handles ESPN abbreviation differences)."""
    # Normalize both codes to a common format
    # NBA uses: GSW, SAS, NYK, NOP, UTA, WAS, PHX
    # ESPN uses: GS, SA, NY, NO, UTAH, WSH, PHX (usually same as NBA)

    # Map ESPN codes to NBA standard
    espn_to_nba = {
        "GS": "GSW",
        "SA": "SAS",
        "NY": "NYK",
        "NO": "NOP",
        "UTAH": "UTA",
        "WSH": "WAS",
        "PHO": "PHX",  # ESPN sometimes uses PHO
    }

    nba_normalized = nba_code.upper()
    espn_normalized = espn_to_nba.get(espn_code.upper(), espn_code.upper())

    return nba_normalized == espn_normalized


def _cache_espn_mapping(
    game_id: str, espn_id: str, game_date: str, home_team: str, away_team: str
) -> None:
    """Cache ESPN game mapping for future lookups."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ESPNGameMapping 
                (nba_game_id, espn_event_id, game_date, home_team, away_team)
                VALUES (?, ?, ?, ?, ?)
            """,
                (game_id, espn_id, game_date, home_team, away_team),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"Failed to cache ESPN mapping: {e}")


def fetch_espn_betting_data(espn_event_id: str, home_team: str) -> Optional[dict]:
    """
    Fetch betting data from ESPN summary endpoint.

    Args:
        espn_event_id: ESPN event ID
        home_team: Home team tricode (for spread conversion)

    Returns:
        Dict with betting data or None if unavailable
    """
    url = f"{ESPN_SUMMARY_URL}?event={espn_event_id}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        pickcenter = data.get("pickcenter", [])
        if not pickcenter:
            return None

        # Prefer DraftKings, fall back to first available
        dk_odds = next(
            (
                p
                for p in pickcenter
                if p.get("provider", {}).get("name") == "Draft Kings"
            ),
            None,
        )
        odds_data = dk_odds if dk_odds else pickcenter[0]

        # Extract betting data
        # NOTE: ESPN returns spread in 'details' field as string (e.g., 'BOS -8.5')
        # NOT as numeric 'spread' field - must parse and convert to home perspective
        result = {
            "source": odds_data.get("provider", {}).get("name", "unknown"),
            "spread": _parse_spread_from_details(odds_data.get("details"), home_team),
            "total": odds_data.get("overUnder"),
            "over_odds": _convert_odds(odds_data.get("overOdds")),
            "under_odds": _convert_odds(odds_data.get("underOdds")),
            "home_moneyline": None,
            "away_moneyline": None,
            "spread_home_odds": None,
            "spread_away_odds": None,
        }

        # Extract team-specific odds
        home_odds = odds_data.get("homeTeamOdds", {})
        away_odds = odds_data.get("awayTeamOdds", {})

        if home_odds:
            result["home_moneyline"] = home_odds.get("moneyLine")
            result["spread_home_odds"] = _convert_odds(home_odds.get("spreadOdds"))

        if away_odds:
            result["away_moneyline"] = away_odds.get("moneyLine")
            result["spread_away_odds"] = _convert_odds(away_odds.get("spreadOdds"))

        return result

    except requests.RequestException as e:
        logger.error(f"ESPN summary request failed for event {espn_event_id}: {e}")
        return None


def _parse_spread_from_details(
    details: Optional[str], home_team: str
) -> Optional[float]:
    """
    Parse spread from ESPN 'details' field and convert to home team perspective.

    ESPN provides spread as: 'TEAM_ABBREV +/-POINTS' (e.g., 'BOS -8.5', 'LAL +3')
    We convert to home team perspective: negative = home favored

    Args:
        details: Spread string from ESPN (e.g., 'BOS -8.5')
        home_team: Home team tricode (e.g., 'BOS')

    Returns:
        Float spread from home perspective, or None if parsing fails

    Examples:
        details='BOS -8.5', home_team='BOS' → -8.5 (home favored by 8.5)
        details='BOS -8.5', home_team='LAL' → +8.5 (away favored, home underdog)
        details='LAL +3', home_team='LAL' → +3 (home underdog by 3)
        details='LAL +3', home_team='BOS' → -3 (away underdog, home favored)
    """
    if not details:
        return None

    try:
        # Parse 'BOS -8.5' format
        parts = details.strip().split()
        if len(parts) != 2:
            logger.warning(f"Unexpected spread format: '{details}'")
            return None

        team_abbrev = parts[0]
        spread_str = parts[1]  # e.g., '-8.5' or '+3'

        # Convert to float (handles both '+3' and '-8.5')
        spread_value = float(spread_str)

        # Convert to home team perspective
        # If the team in the details is the home team, use value as-is
        # If it's the away team, flip the sign
        if team_abbrev == home_team:
            return spread_value
        else:
            return -spread_value

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse spread from '{details}': {e}")
        return None


def _convert_odds(odds_value) -> Optional[int]:
    """Convert odds to integer (ESPN sometimes returns floats)."""
    if odds_value is None:
        return None
    try:
        return int(float(odds_value))
    except (ValueError, TypeError):
        return None


# =============================================================================
# Core Betting Data Functions
# =============================================================================


def should_fetch_betting(
    game_datetime: datetime, game_status: str, now: Optional[datetime] = None
) -> tuple[bool, str]:
    """
    Determine if we should fetch betting data for a game.

    Strategy:
    - Future > 2 days: Skip (odds not available yet)
    - Future 1-2 days: Fetch from ESPN (opening lines)
    - Game started or completed: Fetch closing lines
    - After game > 7 days old: Skip (ESPN no longer has data)

    Args:
        game_datetime: Game scheduled start time (UTC)
        game_status: Game status ("Scheduled", "In Progress", "Final", etc.)
        now: Current time (for testing), defaults to datetime.utcnow()

    Returns:
        Tuple of (should_fetch: bool, source: str)
        source can be: "espn", "too_far_future", "too_old", "skip"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Make game_datetime timezone-aware if it isn't
    if game_datetime.tzinfo is None:
        game_datetime = game_datetime.replace(tzinfo=timezone.utc)

    delta = game_datetime - now
    days_away = delta.total_seconds() / 86400  # Convert to days

    # Game is too far in the future
    if days_away > FUTURE_CUTOFF_DAYS:
        return False, "too_far_future"

    # Game is too old - ESPN no longer has data (Covers.com scraping implemented separately)
    if game_status in ("Final", "Completed") and days_away < -ESPN_LOOKBACK_DAYS:
        return False, "too_old"

    # Game is within ESPN's data window
    if (
        game_status in ("Final", "Completed", "In Progress")
        or days_away >= -ESPN_LOOKBACK_DAYS
    ):
        return True, "espn"

    # Default: skip
    return False, "skip"


def fetch_betting_for_game(
    game_id: str,
    game_date: str,
    home_team: str,
    away_team: str,
    game_datetime: datetime,
    game_status: str,
) -> Optional[dict]:
    """
    Fetch betting data for a single game from ESPN API (Tier 1).

    Data availability:
    - Upcoming games (1-2 days): Opening/current lines
    - Recent completed games (< 7 days): Closing lines
    - Older games (> 7 days): Returns None (use Tier 2/3 Covers)

    Args:
        game_id: NBA game ID
        game_date: Game date (YYYY-MM-DD)
        home_team: Home team tricode
        away_team: Away team tricode
        game_datetime: Game start time (UTC)
        game_status: Game status

    Returns:
        Dict with betting data or None
    """
    # Check if we should fetch from ESPN
    should_fetch, source = should_fetch_betting(game_datetime, game_status)
    if not should_fetch:
        logger.debug(f"Skipping ESPN betting for {game_id}: {source}")
        return None

    # Get ESPN event ID
    espn_id = get_espn_event_id(game_id, game_date, home_team, away_team)
    if not espn_id:
        logger.warning(f"No ESPN ID found for {game_id}")
        return None

    # Fetch from ESPN
    betting_data = fetch_espn_betting_data(espn_id, home_team)

    if not betting_data:
        logger.debug(f"No betting data available for {game_id} (ESPN: {espn_id})")
        return None

    # Add metadata
    betting_data["game_id"] = game_id
    betting_data["espn_event_id"] = espn_id
    betting_data["source"] = betting_data.get("source", "ESPN")

    # Determine if lines should be finalized
    # Finalized when game is completed and we have closing line
    if game_status in ("Final", "Completed") and betting_data.get("spread") is not None:
        betting_data["lines_finalized"] = 1
    else:
        betting_data["lines_finalized"] = 0

    return betting_data


def save_betting_data(
    betting_data_list: list[dict], conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Save betting data to Betting table (single row per game).

    Uses UPSERT pattern: inserts new rows, updates existing ones.

    Args:
        betting_data_list: List of betting data dicts
        conn: Optional database connection

    Returns:
        Number of rows inserted/updated
    """
    if not betting_data_list:
        return 0

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    saved = 0
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        for data in betting_data_list:
            game_id = data["game_id"]

            # Check if row exists and if it's finalized
            cursor = conn.execute(
                "SELECT lines_finalized, spread FROM Betting WHERE game_id = ?",
                (game_id,),
            )
            existing = cursor.fetchone()

            if existing:
                # Don't update if already finalized (unless new data also finalized with better spread)
                if existing[0] == 1 and existing[1] is not None:
                    # Already have finalized data with spread, skip
                    continue

                # Update existing row
                conn.execute(
                    """
                    UPDATE Betting SET
                        espn_event_id = COALESCE(?, espn_event_id),
                        spread = COALESCE(?, spread),
                        spread_result = COALESCE(?, spread_result),
                        spread_home_odds = COALESCE(?, spread_home_odds),
                        spread_away_odds = COALESCE(?, spread_away_odds),
                        total = COALESCE(?, total),
                        ou_result = COALESCE(?, ou_result),
                        over_odds = COALESCE(?, over_odds),
                        under_odds = COALESCE(?, under_odds),
                        home_moneyline = COALESCE(?, home_moneyline),
                        away_moneyline = COALESCE(?, away_moneyline),
                        source = ?,
                        lines_finalized = MAX(lines_finalized, ?),
                        updated_at = ?
                    WHERE game_id = ?
                """,
                    (
                        data.get("espn_event_id"),
                        data.get("spread"),
                        data.get("spread_result"),
                        data.get("spread_home_odds"),
                        data.get("spread_away_odds"),
                        data.get("total"),
                        data.get("ou_result"),
                        data.get("over_odds"),
                        data.get("under_odds"),
                        data.get("home_moneyline"),
                        data.get("away_moneyline"),
                        data.get("source", "ESPN"),
                        data.get("lines_finalized", 0),
                        now,
                        game_id,
                    ),
                )
            else:
                # Insert new row
                conn.execute(
                    """
                    INSERT INTO Betting (
                        game_id, espn_event_id, spread, spread_result,
                        spread_home_odds, spread_away_odds, total, ou_result,
                        over_odds, under_odds, home_moneyline, away_moneyline,
                        source, lines_finalized, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        game_id,
                        data.get("espn_event_id"),
                        data.get("spread"),
                        data.get("spread_result"),
                        data.get("spread_home_odds"),
                        data.get("spread_away_odds"),
                        data.get("total"),
                        data.get("ou_result"),
                        data.get("over_odds"),
                        data.get("under_odds"),
                        data.get("home_moneyline"),
                        data.get("away_moneyline"),
                        data.get("source", "ESPN"),
                        data.get("lines_finalized", 0),
                        now,
                        now,
                    ),
                )
            saved += 1

        conn.commit()
        logger.debug(f"Saved {saved} betting records")

    finally:
        if close_conn:
            conn.close()

    return saved


# Legacy alias for backwards compatibility
save_betting_lines = save_betting_data


# =============================================================================
# 3-Tier Batch Operations
# =============================================================================


def update_betting_data(
    game_ids: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
    season: Optional[str] = None,
    skip_finalized: bool = True,
    use_covers: bool = True,
) -> dict:
    """
    Update betting data using 3-tier strategy.

    TIER 1 - ESPN API (games within -7 to +2 days):
        - Used for real-time and recent data
        - Provides full odds (spreads, totals, moneylines with vig)
        - Primary source for automatic web app updates

    TIER 2 - Covers Matchups (completed games >7 days old):
        - Used for finalizing older games that ESPN no longer serves
        - Fetches by date (efficient: 1 call per date)
        - Only triggered for dates with unfinalized completed games
        - Provides closing spreads/totals (no odds)

    TIER 3 - Covers Team Schedules (historical backfill):
        - Used via CLI with --backfill flag
        - Not triggered by this function (use update_betting_backfill)

    Args:
        game_ids: Specific game IDs to update, or None for auto-detection
        date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
        season: Specific season to update (e.g., "2024-2025")
                If None, uses current season (automatic mode for web app)
        skip_finalized: If True, skip games with finalized lines (default: True)
        use_covers: If True, use Tier 2 Covers for games outside ESPN window

    Returns:
        Dict with stats: {"espn_fetched": int, "covers_fetched": int,
                         "saved": int, "skipped": int, "errors": int}
    """
    stats = {
        "espn_fetched": 0,
        "covers_fetched": 0,
        "saved": 0,
        "skipped": 0,
        "already_finalized": 0,
        "errors": 0,
    }

    now = datetime.now(timezone.utc)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Ensure table exists
        create_betting_tables(conn)

        # Determine season filter
        if season:
            target_season = season
        elif game_ids or date_range:
            target_season = None
        else:
            target_season = _get_current_season(now)

        # Get games that need betting data
        games = _get_games_needing_betting_data(
            conn, game_ids, date_range, target_season, now
        )

        if not games:
            logger.info("No games need betting data updates")
            return stats

        logger.info(
            f"Processing betting data for {len(games)} games "
            f"(season: {target_season or 'mixed'})"
        )

        # Get existing finalized game_ids
        existing_finalized = set()
        if skip_finalized:
            cursor = conn.execute(
                "SELECT game_id FROM Betting WHERE lines_finalized = 1 AND spread IS NOT NULL"
            )
            existing_finalized = {row[0] for row in cursor.fetchall()}

        # Partition games into ESPN window vs Covers needed
        espn_games = []
        covers_dates_needed = set()

        for game in games:
            game_id = game["game_id"]

            if skip_finalized and game_id in existing_finalized:
                stats["already_finalized"] += 1
                continue

            game_datetime = datetime.strptime(
                game["date_time_est"], "%Y-%m-%dT%H:%M:%SZ"
            )
            days_ago = (now - game_datetime.replace(tzinfo=timezone.utc)).days

            # ESPN window: -FUTURE_CUTOFF_DAYS to +ESPN_LOOKBACK_DAYS
            if -FUTURE_CUTOFF_DAYS <= days_ago <= ESPN_LOOKBACK_DAYS:
                espn_games.append(game)
            elif (
                use_covers
                and game["status"] in ("Final", "Completed")
                and days_ago > ESPN_LOOKBACK_DAYS
            ):
                # Game is outside ESPN window, use Covers
                game_date = game["date_time_est"].split("T")[0]
                covers_dates_needed.add(game_date)

        # TIER 1: Fetch from ESPN
        if espn_games:
            espn_stats = _fetch_espn_batch(espn_games, conn)
            stats["espn_fetched"] += espn_stats["fetched"]
            stats["saved"] += espn_stats["saved"]
            stats["skipped"] += espn_stats["skipped"]
            stats["errors"] += espn_stats["errors"]

        # TIER 2: Fetch from Covers for dates outside ESPN window
        if covers_dates_needed and use_covers:
            covers_stats = _fetch_covers_batch(
                list(covers_dates_needed), conn, target_season
            )
            stats["covers_fetched"] += covers_stats["fetched"]
            stats["saved"] += covers_stats["saved"]
            stats["errors"] += covers_stats["errors"]

    logger.info(f"Betting update complete: {stats}")
    return stats


def _get_games_needing_betting_data(
    conn: sqlite3.Connection,
    game_ids: Optional[list[str]],
    date_range: Optional[tuple[str, str]],
    season: Optional[str],
    now: datetime,
) -> list[sqlite3.Row]:
    """
    Get games that need betting data from the database.

    Returns games that are:
    1. Completed without finalized betting data
    2. Upcoming within +2 days of now
    """
    if game_ids:
        placeholders = ",".join(["?"] * len(game_ids))
        query = f"""
            SELECT g.game_id, g.date_time_est, g.home_team, g.away_team, g.status
            FROM Games g
            WHERE g.game_id IN ({placeholders})
        """
        return conn.execute(query, game_ids).fetchall()

    if date_range:
        query = """
            SELECT g.game_id, g.date_time_est, g.home_team, g.away_team, g.status
            FROM Games g
            WHERE date(g.date_time_est) BETWEEN ? AND ?
              AND g.season_type IN ('Regular Season', 'Post Season')
            ORDER BY g.date_time_est
        """
        return conn.execute(query, date_range).fetchall()

    # Season mode - get games needing betting data
    future_cutoff = (now + timedelta(days=FUTURE_CUTOFF_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    query = """
        SELECT g.game_id, g.date_time_est, g.home_team, g.away_team, g.status
        FROM Games g
        LEFT JOIN Betting b ON g.game_id = b.game_id
        WHERE g.season = ?
          AND g.season_type IN ('Regular Season', 'Post Season')
          AND (g.status IN ('Final', 'Completed') OR g.date_time_est <= ?)
          AND (b.lines_finalized IS NULL OR b.lines_finalized = 0 OR b.spread IS NULL)
        ORDER BY g.date_time_est
    """
    return conn.execute(query, (season, future_cutoff)).fetchall()


def _fetch_espn_batch(games: list, conn: sqlite3.Connection) -> dict:
    """
    Fetch betting data from ESPN API for a batch of games (Tier 1).
    """
    stats = {"fetched": 0, "saved": 0, "skipped": 0, "errors": 0}
    betting_data_batch = []
    BATCH_SIZE = 50

    pbar = tqdm(games, desc="Tier 1: ESPN API", unit="game")
    for game in pbar:
        game_id = game["game_id"]
        pbar.set_postfix(fetched=stats["fetched"], errors=stats["errors"])

        try:
            game_datetime_str = game["date_time_est"]
            game_datetime = datetime.strptime(game_datetime_str, "%Y-%m-%dT%H:%M:%SZ")
            game_date = game_datetime_str.split("T")[0]

            betting_data = fetch_betting_for_game(
                game_id=game_id,
                game_date=game_date,
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_datetime=game_datetime,
                game_status=game["status"],
            )

            if betting_data:
                betting_data_batch.append(betting_data)
                stats["fetched"] += 1

                if len(betting_data_batch) >= BATCH_SIZE:
                    stats["saved"] += save_betting_data(betting_data_batch, conn)
                    betting_data_batch = []
            else:
                stats["skipped"] += 1

        except Exception as e:
            logger.error(f"ESPN error for {game_id}: {e}")
            stats["errors"] += 1

    pbar.close()

    if betting_data_batch:
        stats["saved"] += save_betting_data(betting_data_batch, conn)

    return stats


def _fetch_covers_batch(
    dates: list[str], conn: sqlite3.Connection, season: Optional[str]
) -> dict:
    """
    Fetch betting data from Covers.com matchups pages (Tier 2).

    Args:
        dates: List of dates (YYYY-MM-DD) to fetch
        conn: Database connection
        season: Season string for context
    """
    from src.database_updater.covers import fetch_matchups_for_date

    stats = {"fetched": 0, "saved": 0, "errors": 0}

    logger.info(f"Tier 2: Fetching Covers data for {len(dates)} dates")

    # Get mapping of (date, home_team, away_team) -> game_id
    game_lookup = {}
    for date_str in dates:
        cursor = conn.execute(
            """
            SELECT game_id, home_team, away_team
            FROM Games
            WHERE date(date_time_est) = ?
              AND season_type IN ('Regular Season', 'Post Season')
        """,
            (date_str,),
        )
        for row in cursor.fetchall():
            key = (date_str, row["home_team"], row["away_team"])
            game_lookup[key] = row["game_id"]

    pbar = tqdm(dates, desc="Tier 2: Covers matchups", unit="date")
    for date_str in pbar:
        pbar.set_postfix(fetched=stats["fetched"], errors=stats["errors"])

        try:
            game_date = date.fromisoformat(date_str)
            covers_games = fetch_matchups_for_date(game_date)

            betting_data_batch = []
            for cg in covers_games:
                key = (date_str, cg.home_team, cg.away_team)
                game_id = game_lookup.get(key)

                if not game_id:
                    logger.debug(
                        f"No DB match for Covers game: {cg.away_team}@{cg.home_team} on {date_str}"
                    )
                    continue

                # Use results from Covers directly (already parsed from page)
                betting_data = {
                    "game_id": game_id,
                    "spread": cg.spread,
                    "spread_result": cg.spread_result,
                    "total": cg.total,
                    "ou_result": cg.ou_result,
                    "source": "Covers",
                    "lines_finalized": 1 if cg.spread is not None else 0,
                }
                betting_data_batch.append(betting_data)
                stats["fetched"] += 1

            if betting_data_batch:
                stats["saved"] += save_betting_data(betting_data_batch, conn)

        except Exception as e:
            logger.error(f"Covers error for {date_str}: {e}")
            stats["errors"] += 1

    pbar.close()
    return stats


def update_betting_backfill(season: str) -> dict:
    """
    Tier 3: Historical backfill using Covers.com team schedule pages.

    This fetches all home games for all 30 teams in a season from Covers.
    Use this for bulk historical data collection, not for regular updates.

    Makes 30 requests with rate limiting (~90 seconds total).

    Args:
        season: Season to backfill (e.g., "2024-2025")

    Returns:
        Dict with stats: {"fetched": int, "matched": int, "saved": int, "errors": int}
    """
    from src.database_updater.covers import NBA_TO_COVERS_SLUG, fetch_team_schedule

    stats = {"fetched": 0, "matched": 0, "saved": 0, "errors": 0}

    logger.info(f"Tier 3: Backfilling season {season} from Covers team schedules")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # Ensure table exists
        create_betting_tables(conn)

        # Build lookup: (date, home_team, away_team) -> game_id
        cursor = conn.execute(
            """
            SELECT game_id, date(date_time_est) as game_date, 
                   home_team, away_team
            FROM Games
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season')
        """,
            (season,),
        )

        game_lookup = {}
        for row in cursor.fetchall():
            key = (row["game_date"], row["home_team"], row["away_team"])
            game_lookup[key] = row["game_id"]

        logger.info(f"Found {len(game_lookup)} games in DB for {season}")

        # Fetch each team's schedule
        pbar = tqdm(
            NBA_TO_COVERS_SLUG.items(),
            desc="Tier 3: Covers team schedules",
            unit="team",
        )
        for tricode, slug in pbar:
            pbar.set_postfix(team=tricode, fetched=stats["fetched"])

            try:
                covers_games = fetch_team_schedule(slug, season)

                betting_data_batch = []
                for cg in covers_games:
                    # Fill in home team
                    cg.home_team = tricode

                    if not cg.away_team:
                        continue

                    # Format date for lookup
                    game_date_str = cg.game_date.strftime("%Y-%m-%d")
                    key = (game_date_str, tricode, cg.away_team)
                    game_id = game_lookup.get(key)

                    if not game_id:
                        logger.debug(
                            f"No DB match: {cg.away_team}@{tricode} on {game_date_str}"
                        )
                        continue

                    stats["matched"] += 1

                    # Use results from Covers directly (already parsed from page)
                    betting_data = {
                        "game_id": game_id,
                        "spread": cg.spread,
                        "spread_result": cg.spread_result,
                        "total": cg.total,
                        "ou_result": cg.ou_result,
                        "source": "Covers",
                        "lines_finalized": 1 if cg.spread is not None else 0,
                    }
                    betting_data_batch.append(betting_data)
                    stats["fetched"] += 1

                if betting_data_batch:
                    stats["saved"] += save_betting_data(betting_data_batch, conn)

            except Exception as e:
                logger.error(f"Covers error for {tricode}: {e}")
                stats["errors"] += 1

        pbar.close()

    logger.info(f"Tier 3 backfill complete: {stats}")
    return stats


# =============================================================================
# Query Functions
# =============================================================================


def get_betting_data(game_id: str) -> Optional[dict]:
    """
    Get betting data for a game.

    Args:
        game_id: NBA game ID

    Returns:
        Betting data dict or None
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM Betting WHERE game_id = ?",
            (game_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    from src.logging_config import setup_logging

    parser = argparse.ArgumentParser(
        description="Fetch and store NBA betting data using 3-tier strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update current season (Tier 1 ESPN + Tier 2 Covers auto)
  python -m src.database_updater.betting
  
  # Update specific season
  python -m src.database_updater.betting --season=2024-2025
  
  # Backfill historical season (Tier 3 Covers team schedules)
  python -m src.database_updater.betting --backfill --season=2023-2024
  
  # Force re-fetch even if finalized
  python -m src.database_updater.betting --season=2024-2025 --force
  
  # Update specific games
  python -m src.database_updater.betting --game_ids=0022400123,0022400124
        """,
    )
    parser.add_argument("--game_ids", type=str, help="Comma-separated game IDs")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start_date", type=str, help="Start date for range")
    parser.add_argument("--end_date", type=str, help="End date for range")
    parser.add_argument(
        "--season",
        type=str,
        help="Season to process (e.g., 2024-2025). Uses current season if not specified.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Use Tier 3 (Covers team schedules) for bulk historical backfill. Requires --season.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch data even if already finalized",
    )
    parser.add_argument(
        "--no-covers",
        action="store_true",
        dest="no_covers",
        help="Disable Tier 2 Covers fetching (ESPN only)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--create_tables", action="store_true", help="Create Betting table and exit"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.create_tables:
        create_betting_tables()
        print("Betting table created successfully")
        exit(0)

    # Validate backfill requires season
    if args.backfill:
        if not args.season:
            parser.error("--backfill requires --season to be specified")

        # Run Tier 3 backfill
        stats = update_betting_backfill(args.season)
        print(f"\nTier 3 Backfill complete for {args.season}:")
        print(f"  Fetched from Covers:  {stats['fetched']}")
        print(f"  Matched to DB games:  {stats['matched']}")
        print(f"  Saved:                {stats['saved']}")
        print(f"  Errors:               {stats['errors']}")
        exit(0)

    # Determine what to fetch
    game_ids = None
    date_range = None
    season = args.season

    if args.game_ids:
        game_ids = [g.strip() for g in args.game_ids.split(",")]
    elif args.date:
        date_range = (args.date, args.date)
    elif args.start_date and args.end_date:
        date_range = (args.start_date, args.end_date)

    # Run 3-tier update
    stats = update_betting_data(
        game_ids=game_ids,
        date_range=date_range,
        season=season,
        skip_finalized=not args.force,
        use_covers=not args.no_covers,
    )

    print(f"\nBetting data update complete:")
    print(f"  Tier 1 - ESPN:        {stats['espn_fetched']}")
    print(f"  Tier 2 - Covers:      {stats['covers_fetched']}")
    print(f"  Saved:                {stats['saved']}")
    print(f"  Already finalized:    {stats['already_finalized']}")
    print(f"  Skipped (no data):    {stats['skipped']}")
    print(f"  Errors:               {stats['errors']}")
