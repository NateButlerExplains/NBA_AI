#!/usr/bin/env python3
"""
backfill_boxscores.py

Backfill historical PlayerBox and TeamBox data from ESPN/Sportsdataverse
parquet files. This is a standalone script separate from the normal NBA API
collection pipeline.

Data source: sportsdataverse/sportsdataverse-data GitHub releases (ESPN data)
Coverage: 2001-02 through 2024-25 seasons
Format: Parquet files with player and team boxscore stats

Usage:
    # Dry run for one season
    python scripts/backfill_boxscores.py --dry-run --seasons 2023

    # Full backfill against the full database
    python scripts/backfill_boxscores.py --db-path data/NBA_AI_full.sqlite

    # Specific seasons only
    python scripts/backfill_boxscores.py --seasons 2003 2004 2005

    # Skip download (use cached parquet files)
    python scripts/backfill_boxscores.py --skip-download

    # Validate only (no inserts)
    python scripts/backfill_boxscores.py --validate-only
"""

import argparse
import logging
import os
import sqlite3
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db
from src.utils import NBATeamConverter

# ============================================================
# Constants
# ============================================================

PLAYER_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_player_boxscores/player_box_{year}.parquet"
)
TEAM_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_team_boxscores/team_box_{year}.parquet"
)

# Season end-years for backfill (2002 = 2001-02 season, 2025 = 2024-25 season)
DEFAULT_BACKFILL_SEASONS = list(range(2002, 2026))

# All-Star and non-standard team abbreviations to filter out
ALLSTAR_ABBREVS = {"EAST", "WEST", "GIA", "LEB", "USA", "WORLD"}

# ESPN abbreviation → all possible historical abbreviations used in our Games table.
# NBATeamConverter.get_abbreviation() returns the CURRENT franchise name, but
# the Games table stores the abbreviation correct for the era (e.g. NJN, NOH, SEA).
# This maps ESPN abbreviations to all variants we should try during game matching.
ESPN_HISTORICAL_ABBREVS = {
    "NJ": ["NJN", "BKN"],           # New Jersey Nets (→ Brooklyn Nets)
    "NO": ["NOH", "NOK", "NOP"],    # New Orleans Hornets / Oklahoma City Hornets / Pelicans
    "SEA": ["SEA", "OKC"],          # Seattle SuperSonics (→ Oklahoma City Thunder)
    "CHA": ["CHA", "CHH"],          # Charlotte (Bobcats/Hornets overlap)
    "VAN": ["VAN", "MEM"],          # Vancouver Grizzlies (→ Memphis)
}

# ESPN season_type values
REGULAR_SEASON = 2
PLAYOFFS = 3
PLAY_IN = 5
VALID_SEASON_TYPES = {REGULAR_SEASON, PLAYOFFS, PLAY_IN}

# Default cache directory for downloaded parquet files
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "backfill", "espn")

logger = logging.getLogger(__name__)


# ============================================================
# Name normalization
# ============================================================

def normalize_name(name: str) -> str:
    """
    Normalize a player name for matching: strip diacritics, lowercase, collapse whitespace.

    Examples:
        "Luka Dončić" -> "luka doncic"
        "Jonas Valančiūnas" -> "jonas valanciunas"
        "Dennis Schröder" -> "dennis schroder"
    """
    # NFKD decomposition separates base characters from combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    # Filter out combining marks (category 'M')
    ascii_name = "".join(c for c in nfkd if not unicodedata.category(c).startswith("M"))
    # Lowercase and collapse whitespace
    return " ".join(ascii_name.lower().split())


# ============================================================
# Download layer
# ============================================================

def download_parquet(url: str, dest_path: str, skip_if_cached: bool = True) -> bool:
    """
    Download a parquet file from GitHub releases.

    Returns True if file is available (downloaded or cached), False on failure.
    """
    if skip_if_cached and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.debug(f"Using cached: {os.path.basename(dest_path)}")
        return True

    try:
        logger.info(f"Downloading {os.path.basename(dest_path)}...")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"Downloaded {os.path.basename(dest_path)} ({size_mb:.2f} MB)")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_season_files(
    year: int, cache_dir: str, skip_download: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Download both player and team parquet files for a season."""
    player_path = os.path.join(cache_dir, f"player_box_{year}.parquet")
    team_path = os.path.join(cache_dir, f"team_box_{year}.parquet")

    if skip_download:
        p_ok = os.path.exists(player_path) and os.path.getsize(player_path) > 0
        t_ok = os.path.exists(team_path) and os.path.getsize(team_path) > 0
        return (player_path if p_ok else None, team_path if t_ok else None)

    p_ok = download_parquet(PLAYER_BOX_URL.format(year=year), player_path)
    t_ok = download_parquet(TEAM_BOX_URL.format(year=year), team_path)

    return (player_path if p_ok else None, team_path if t_ok else None)


# ============================================================
# Lookup table builders
# ============================================================

def build_player_lookup(db_path: str) -> Dict[str, int]:
    """
    Build {normalized_name: person_id} from Players table.

    For duplicate names (same name, different era), keeps the higher person_id
    (more recent player) since they're more likely to appear in the data.
    """
    lookup = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT person_id, first_name, last_name FROM Players")
        for person_id, first_name, last_name in cursor.fetchall():
            if first_name and last_name:
                key = normalize_name(f"{first_name} {last_name}")
                # Keep higher person_id for duplicates (more recent player)
                if key not in lookup or person_id > lookup[key]:
                    lookup[key] = person_id

    logger.info(f"Built player lookup: {len(lookup)} unique names")
    return lookup


def build_team_lookup(db_path: str) -> Dict[str, str]:
    """
    Build {lowercase_abbreviation: team_id} from Teams table.

    Also resolves ESPN-specific abbreviations (GS, NO, NY, SA, UTAH, WSH)
    via NBATeamConverter.
    """
    lookup = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT team_id, abbreviation FROM Teams")
        for team_id, abbreviation in cursor.fetchall():
            lookup[abbreviation.lower()] = str(team_id)

    # Add ESPN-specific abbreviation mappings
    espn_specific = ["GS", "NO", "NY", "SA", "UTAH", "WSH"]
    for espn_abbrev in espn_specific:
        try:
            nba_abbrev = NBATeamConverter.get_abbreviation(espn_abbrev)
            if nba_abbrev.lower() in lookup:
                lookup[espn_abbrev.lower()] = lookup[nba_abbrev.lower()]
        except ValueError:
            logger.warning(f"Could not resolve ESPN abbreviation: {espn_abbrev}")

    # Add historical ESPN abbreviations → same franchise team_id
    # ESPN uses 'NJ' for Nets, 'SEA' for SuperSonics, 'VAN' for Grizzlies, etc.
    # These should all map to the current franchise's team_id.
    for espn_abbrev, variants in ESPN_HISTORICAL_ABBREVS.items():
        if espn_abbrev.lower() not in lookup:
            # Find the team_id from any known variant
            for variant in variants:
                if variant.lower() in lookup:
                    lookup[espn_abbrev.lower()] = lookup[variant.lower()]
                    break

    logger.info(f"Built team lookup: {len(lookup)} abbreviations")
    return lookup


def build_game_lookup(
    db_path: str, nba_season: str
) -> Dict[Tuple[str, str, str], str]:
    """
    Build {(date_str, home_abbrev, away_abbrev): game_id} for one season.

    Stores entries keyed by both the UTC date and UTC date-1 to handle
    the ESPN Eastern time → UTC offset. ESPN dates are Eastern, so a
    9 PM ET game on Jan 15 appears as Jan 16 in our UTC-based Games table.
    When looking up by ESPN date, we try the ESPN date matching against
    our UTC dates, and also ESPN date+1.
    """
    lookup = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT game_id, date_time_utc, home_team, away_team
            FROM Games
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season')
              AND status = 3
            """,
            (nba_season,),
        )
        for game_id, date_time_utc, home_team, away_team in cursor.fetchall():
            if not date_time_utc:
                continue
            # Extract date from UTC timestamp
            date_str = date_time_utc[:10]  # "YYYY-MM-DD"
            home_upper = home_team.upper()
            away_upper = away_team.upper()

            # Primary key: UTC date
            key = (date_str, home_upper, away_upper)
            lookup[key] = game_id

            # Also store date-1 to handle timezone offset:
            # If UTC date is Jan 16, the ESPN Eastern date is Jan 15
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                prev_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
                prev_key = (prev_date, home_upper, away_upper)
                if prev_key not in lookup:
                    lookup[prev_key] = game_id
            except ValueError:
                pass

    logger.info(
        f"Built game lookup for {nba_season}: {len(lookup)} entries "
        f"(with timezone duplicates)"
    )
    return lookup


# ============================================================
# Mapping helpers
# ============================================================

def map_espn_team_abbrev(espn_abbrev: str, team_lookup: Dict[str, str]) -> Optional[str]:
    """Map an ESPN team abbreviation to our team_id (TEXT)."""
    key = espn_abbrev.lower()
    if key in team_lookup:
        return team_lookup[key]

    # Try NBATeamConverter normalization
    try:
        nba_abbrev = NBATeamConverter.get_abbreviation(espn_abbrev)
        if nba_abbrev.lower() in team_lookup:
            return team_lookup[nba_abbrev.lower()]
    except ValueError:
        pass

    logger.warning(f"Could not map ESPN team abbreviation: {espn_abbrev}")
    return None


def map_espn_player(
    athlete_name: str, player_lookup: Dict[str, int]
) -> Optional[int]:
    """Map an ESPN athlete_display_name to our person_id."""
    key = normalize_name(athlete_name)
    return player_lookup.get(key)


def check_game_has_boxscores(cursor: sqlite3.Cursor, game_id: str) -> bool:
    """Check if a game already has PlayerBox data."""
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM PlayerBox WHERE game_id = ?)", (game_id,)
    )
    return cursor.fetchone()[0] == 1


# ============================================================
# Data transformation
# ============================================================

def _safe_int(val) -> Optional[int]:
    """Convert a value to int, returning None for NaN/None."""
    if pd.isna(val):
        return None
    return int(val)


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None for NaN/None."""
    if pd.isna(val):
        return None
    return float(val)


def _compute_pct(made, attempted) -> Optional[float]:
    """Compute shooting percentage as decimal (0.0-1.0)."""
    made_val = _safe_int(made)
    attempted_val = _safe_int(attempted)
    if attempted_val is None or attempted_val == 0:
        return None
    if made_val is None:
        return None
    return round(made_val / attempted_val, 3)


def transform_player_box_row(
    row: pd.Series, game_id: str, team_id: str, person_id: int
) -> dict:
    """Transform one ESPN player box row into our PlayerBox schema."""
    # Parse plus_minus from string ("+18", "-7", "0") to int
    pm_val = None
    if pd.notna(row.get("plus_minus")):
        try:
            pm_val = int(float(str(row["plus_minus"]).replace("+", "")))
        except (ValueError, TypeError):
            pm_val = None

    return {
        "player_id": person_id,
        "game_id": game_id,
        "team_id": team_id,
        "player_name": row.get("athlete_display_name", ""),
        "position": row.get("athlete_position_abbreviation") if pd.notna(row.get("athlete_position_abbreviation")) else None,
        "min": _safe_float(row.get("minutes")),
        "pts": _safe_int(row.get("points")),
        "reb": _safe_int(row.get("rebounds")),
        "ast": _safe_int(row.get("assists")),
        "stl": _safe_int(row.get("steals")),
        "blk": _safe_int(row.get("blocks")),
        "tov": _safe_int(row.get("turnovers")),
        "pf": _safe_int(row.get("fouls")),
        "oreb": _safe_int(row.get("offensive_rebounds")),
        "dreb": _safe_int(row.get("defensive_rebounds")),
        "fga": _safe_int(row.get("field_goals_attempted")),
        "fgm": _safe_int(row.get("field_goals_made")),
        "fg_pct": _compute_pct(row.get("field_goals_made"), row.get("field_goals_attempted")),
        "fg3a": _safe_int(row.get("three_point_field_goals_attempted")),
        "fg3m": _safe_int(row.get("three_point_field_goals_made")),
        "fg3_pct": _compute_pct(
            row.get("three_point_field_goals_made"),
            row.get("three_point_field_goals_attempted"),
        ),
        "fta": _safe_int(row.get("free_throws_attempted")),
        "ftm": _safe_int(row.get("free_throws_made")),
        "ft_pct": _compute_pct(row.get("free_throws_made"), row.get("free_throws_attempted")),
        "plus_minus": pm_val,
    }


def transform_team_box_row(
    row: pd.Series, game_id: str, team_id: str, opponent_score: int
) -> dict:
    """Transform one ESPN team box row into our TeamBox schema."""
    team_score = int(row["team_score"])

    # ESPN percentages are in format 44.9 (percentage), our DB uses 0.449 (decimal)
    def pct_to_decimal(val):
        if pd.isna(val):
            return None
        return round(float(val) / 100.0, 3)

    return {
        "team_id": team_id,
        "game_id": game_id,
        "pts": team_score,
        "pts_allowed": opponent_score,
        "reb": _safe_int(row.get("total_rebounds")),
        "ast": _safe_int(row.get("assists")),
        "stl": _safe_int(row.get("steals")),
        "blk": _safe_int(row.get("blocks")),
        "tov": _safe_int(row.get("turnovers")),
        "pf": _safe_int(row.get("fouls")),
        "fga": _safe_int(row.get("field_goals_attempted")),
        "fgm": _safe_int(row.get("field_goals_made")),
        "fg_pct": pct_to_decimal(row.get("field_goal_pct")),
        "fg3a": _safe_int(row.get("three_point_field_goals_attempted")),
        "fg3m": _safe_int(row.get("three_point_field_goals_made")),
        "fg3_pct": pct_to_decimal(row.get("three_point_field_goal_pct")),
        "fta": _safe_int(row.get("free_throws_attempted")),
        "ftm": _safe_int(row.get("free_throws_made")),
        "ft_pct": pct_to_decimal(row.get("free_throw_pct")),
        "plus_minus": team_score - opponent_score,
    }


# ============================================================
# Save layer
# ============================================================

def save_player_records(cursor: sqlite3.Cursor, records: List[dict]) -> int:
    """Insert player box records using INSERT OR REPLACE (matches boxscores.py SQL)."""
    if not records:
        return 0

    values = [
        (
            r["player_id"], r["game_id"], r["team_id"], r["player_name"], r["position"],
            r["min"], r["pts"], r["reb"], r["ast"], r["stl"], r["blk"], r["tov"], r["pf"],
            r["oreb"], r["dreb"], r["fga"], r["fgm"], r["fg_pct"],
            r["fg3a"], r["fg3m"], r["fg3_pct"],
            r["fta"], r["ftm"], r["ft_pct"], r["plus_minus"],
        )
        for r in records
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO PlayerBox (
            player_id, game_id, team_id, player_name, position,
            min, pts, reb, ast, stl, blk, tov, pf,
            oreb, dreb, fga, fgm, fg_pct,
            fg3a, fg3m, fg3_pct,
            fta, ftm, ft_pct, plus_minus
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def save_team_records(cursor: sqlite3.Cursor, records: List[dict]) -> int:
    """Insert team box records using INSERT OR REPLACE (matches boxscores.py SQL)."""
    if not records:
        return 0

    values = [
        (
            r["team_id"], r["game_id"], r["pts"], r["pts_allowed"],
            r["reb"], r["ast"], r["stl"], r["blk"], r["tov"], r["pf"],
            r["fga"], r["fgm"], r["fg_pct"],
            r["fg3a"], r["fg3m"], r["fg3_pct"],
            r["fta"], r["ftm"], r["ft_pct"], r["plus_minus"],
        )
        for r in records
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO TeamBox (
            team_id, game_id, pts, pts_allowed, reb, ast, stl, blk, tov, pf,
            fga, fgm, fg_pct, fg3a, fg3m, fg3_pct,
            fta, ftm, ft_pct, plus_minus
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


# ============================================================
# Game matching
# ============================================================

def _get_abbrev_variants(espn_abbrev: str) -> List[str]:
    """
    Get all possible abbreviation variants for an ESPN team abbreviation.

    Returns the current NBA abbreviation plus any historical variants
    (e.g. ESPN 'NO' → ['NOP', 'NOH', 'NOK']).
    """
    upper = espn_abbrev.upper()

    # Check historical mappings first
    if upper in ESPN_HISTORICAL_ABBREVS:
        return ESPN_HISTORICAL_ABBREVS[upper]

    # Try NBATeamConverter for current franchise name
    try:
        nba = NBATeamConverter.get_abbreviation(espn_abbrev)
        return [nba] if nba != upper else [upper]
    except ValueError:
        return [upper]


def match_espn_game_to_nba(
    team_box_game: pd.DataFrame,
    game_lookup: Dict[Tuple[str, str, str], str],
    exclude_game_ids: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Match ESPN game rows to an NBA game_id using date + home/away teams.

    Handles historical franchise abbreviations by trying all known variants
    (e.g. ESPN 'NO' could be NOH, NOK, or NOP in our Games table depending
    on the season).

    Args:
        team_box_game: Team box rows for one ESPN game (should be 2 rows).
        game_lookup: {(date, home_abbrev, away_abbrev): nba_game_id}
        exclude_game_ids: Set of NBA game_ids to skip (already matched/filled).
            Handles back-to-back same-matchup games where date offsets collide.

    Returns:
        NBA game_id or None if no match.
    """
    home_rows = team_box_game[team_box_game["team_home_away"] == "home"]
    away_rows = team_box_game[team_box_game["team_home_away"] == "away"]

    if len(home_rows) == 0 or len(away_rows) == 0:
        return None

    home_row = home_rows.iloc[0]
    away_row = away_rows.iloc[0]

    game_date = str(home_row["game_date"])[:10]
    home_espn_abbrev = str(home_row["team_abbreviation"])
    away_espn_abbrev = str(away_row["team_abbreviation"])

    # Get all possible abbreviation variants for each team
    home_variants = _get_abbrev_variants(home_espn_abbrev)
    away_variants = _get_abbrev_variants(away_espn_abbrev)

    # Build list of dates to try: ESPN date (Eastern) and ESPN date + 1 (UTC offset)
    dates_to_try = [game_date]
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d")
        next_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        dates_to_try.append(next_date)
    except ValueError:
        pass

    # Try all combinations of date × home_variant × away_variant
    # Skip game_ids that are already matched (handles back-to-back same-team games)
    for date_str in dates_to_try:
        for home_nba in home_variants:
            for away_nba in away_variants:
                key = (date_str, home_nba, away_nba)
                if key in game_lookup:
                    candidate = game_lookup[key]
                    if exclude_game_ids and candidate in exclude_game_ids:
                        continue
                    return candidate

    return None


# ============================================================
# Per-season processing
# ============================================================

def process_season(
    year: int,
    player_box_path: str,
    team_box_path: str,
    db_path: str,
    player_lookup: Dict[str, int],
    team_lookup: Dict[str, str],
    dry_run: bool = False,
) -> dict:
    """
    Process one season of ESPN data and insert into database.

    Args:
        year: End-year of season (e.g., 2003 for 2002-03)
        player_box_path: Path to player_box_{year}.parquet
        team_box_path: Path to team_box_{year}.parquet
        db_path: Database path
        player_lookup: Player name -> person_id mapping
        team_lookup: Team abbreviation -> team_id mapping
        dry_run: If True, don't actually insert

    Returns:
        Dict with processing stats.
    """
    nba_season = f"{year - 1}-{year}"
    stats = {
        "season": nba_season,
        "total_espn_games": 0,
        "games_matched": 0,
        "games_skipped_existing": 0,
        "games_unmatched": 0,
        "games_filtered": 0,
        "player_records_inserted": 0,
        "team_records_inserted": 0,
        "players_unmatched": 0,
        "errors": [],
    }

    # Load parquet files
    try:
        player_df = pd.read_parquet(player_box_path)
        team_df = pd.read_parquet(team_box_path)
    except Exception as e:
        stats["errors"].append(f"Failed to load parquet files: {e}")
        return stats

    # Filter to valid season types (regular season + playoffs)
    player_df = player_df[player_df["season_type"].isin(VALID_SEASON_TYPES)]
    team_df = team_df[team_df["season_type"].isin(VALID_SEASON_TYPES)]

    # Filter out All-Star games
    team_df = team_df[~team_df["team_abbreviation"].isin(ALLSTAR_ABBREVS)]
    player_df = player_df[~player_df["team_abbreviation"].isin(ALLSTAR_ABBREVS)]

    # Build game lookup for this season
    game_lookup = build_game_lookup(db_path, nba_season)

    # Get unique ESPN game IDs from team box, sorted by date.
    # Chronological order is critical for back-to-back same-matchup games:
    # if PHX vs POR plays Nov 4 and Nov 5, processing Nov 4 first ensures it
    # claims the correct NBA game_id before Nov 5 is processed.
    game_dates = team_df.groupby("game_id")["game_date"].first()
    espn_game_ids = game_dates.sort_values().index.values
    stats["total_espn_games"] = len(espn_game_ids)

    # Track matched NBA game_ids to handle back-to-back same-matchup games
    # where date offsets can cause one ESPN game to incorrectly match another
    matched_nba_ids: Set[str] = set()

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        for espn_gid in tqdm(
            espn_game_ids,
            desc=f"Processing {nba_season}",
            unit="game",
            leave=False,
        ):
            try:
                game_team_rows = team_df[team_df["game_id"] == espn_gid]
                if len(game_team_rows) != 2:
                    stats["games_filtered"] += 1
                    continue

                # Check for All-Star teams (redundant safety)
                abbrevs = set(game_team_rows["team_abbreviation"].values)
                if abbrevs & ALLSTAR_ABBREVS:
                    stats["games_filtered"] += 1
                    continue

                # Match to NBA game_id, excluding already-matched games
                nba_game_id = match_espn_game_to_nba(
                    game_team_rows, game_lookup, exclude_game_ids=matched_nba_ids
                )
                if nba_game_id is None:
                    stats["games_unmatched"] += 1
                    logger.debug(
                        f"Unmatched ESPN game {espn_gid}: "
                        f"{game_team_rows.iloc[0].get('game_date', '?')} "
                        f"{game_team_rows.iloc[0].get('team_abbreviation', '?')} vs "
                        f"{game_team_rows.iloc[1].get('team_abbreviation', '?')}"
                    )
                    continue

                # Track this match to prevent back-to-back collisions
                matched_nba_ids.add(nba_game_id)

                # Skip if game already has boxscore data
                if check_game_has_boxscores(cursor, nba_game_id):
                    stats["games_skipped_existing"] += 1
                    continue

                # Extract home/away rows
                home_rows = game_team_rows[game_team_rows["team_home_away"] == "home"]
                away_rows = game_team_rows[game_team_rows["team_home_away"] == "away"]
                if len(home_rows) == 0 or len(away_rows) == 0:
                    stats["errors"].append(f"No home/away split for ESPN game {espn_gid}")
                    continue

                home_row = home_rows.iloc[0]
                away_row = away_rows.iloc[0]

                home_team_id = map_espn_team_abbrev(
                    str(home_row["team_abbreviation"]), team_lookup
                )
                away_team_id = map_espn_team_abbrev(
                    str(away_row["team_abbreviation"]), team_lookup
                )
                if not home_team_id or not away_team_id:
                    stats["errors"].append(
                        f"Team mapping failed for ESPN game {espn_gid}"
                    )
                    continue

                home_score = int(home_row["team_score"])
                away_score = int(away_row["team_score"])

                # Transform and save team box records
                home_team_record = transform_team_box_row(
                    home_row, nba_game_id, home_team_id, away_score
                )
                away_team_record = transform_team_box_row(
                    away_row, nba_game_id, away_team_id, home_score
                )

                if not dry_run:
                    save_team_records(cursor, [home_team_record, away_team_record])
                stats["team_records_inserted"] += 2

                # Transform and save player box records
                game_player_rows = player_df[player_df["game_id"] == espn_gid]
                player_records = []

                for _, prow in game_player_rows.iterrows():
                    # Skip players with NaN minutes (truly inactive / DNP)
                    if pd.isna(prow.get("minutes")):
                        continue

                    person_id = map_espn_player(
                        str(prow["athlete_display_name"]), player_lookup
                    )
                    if person_id is None:
                        stats["players_unmatched"] += 1
                        logger.debug(
                            f"Unmatched player: {prow['athlete_display_name']} "
                            f"(ESPN game {espn_gid})"
                        )
                        continue

                    player_team_id = map_espn_team_abbrev(
                        str(prow["team_abbreviation"]), team_lookup
                    )
                    if player_team_id is None:
                        continue

                    record = transform_player_box_row(
                        prow, nba_game_id, player_team_id, person_id
                    )
                    player_records.append(record)

                if not dry_run and player_records:
                    save_player_records(cursor, player_records)
                stats["player_records_inserted"] += len(player_records)
                stats["games_matched"] += 1

            except Exception as e:
                stats["errors"].append(f"Error processing ESPN game {espn_gid}: {e}")
                logger.warning(f"Error processing ESPN game {espn_gid}: {e}")
                continue

        # Commit after entire season
        if not dry_run:
            conn.commit()

    return stats


# ============================================================
# Validation
# ============================================================

def validate_backfill(db_path: str) -> dict:
    """Run basic validation on backfilled data."""
    results = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Coverage by season
        cursor.execute(
            """
            SELECT g.season,
                   COUNT(*) as total_completed,
                   SUM(CASE WHEN pb.game_id IS NOT NULL THEN 1 ELSE 0 END) as has_boxscores
            FROM Games g
            LEFT JOIN (SELECT DISTINCT game_id FROM PlayerBox) pb ON g.game_id = pb.game_id
            WHERE g.status = 3
              AND g.season_type IN ('Regular Season', 'Post Season')
            GROUP BY g.season
            ORDER BY g.season
            """
        )
        coverage = {}
        for season, total, has_box in cursor.fetchall():
            coverage[season] = {"total": total, "has_boxscores": has_box}
        results["coverage"] = coverage

        # Check for games with wrong team record count
        cursor.execute(
            """
            SELECT game_id, COUNT(*) as team_count
            FROM TeamBox
            GROUP BY game_id
            HAVING team_count != 2
            LIMIT 20
            """
        )
        bad_team_counts = cursor.fetchall()
        results["bad_team_counts"] = bad_team_counts

        # Check for extreme stat values
        cursor.execute(
            """
            SELECT COUNT(*) FROM PlayerBox
            WHERE pts < 0 OR pts > 100 OR min < 0 OR min > 65
            """
        )
        results["extreme_stats"] = cursor.fetchone()[0]

        # Total records
        cursor.execute("SELECT COUNT(*) FROM PlayerBox")
        results["total_player_records"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM TeamBox")
        results["total_team_records"] = cursor.fetchone()[0]

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical boxscore data from ESPN/Sportsdataverse"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Database path (default: from config)",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Season end-years to process (e.g., 2003 2004). Default: all 2002-2025",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing to database",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached parquet files only (don't download)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing backfilled data",
    )
    parser.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help=f"Directory for cached parquet files (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve database path
    if args.db_path:
        db_path = os.path.abspath(args.db_path)
    else:
        from src.database import DB_PATH
        db_path = DB_PATH

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Using database: {db_path}")

    # Validate-only mode
    if args.validate_only:
        logger.info("Running validation only...")
        results = validate_backfill(db_path)
        print("\n=== Backfill Validation Results ===")
        print(f"Total PlayerBox records: {results['total_player_records']:,}")
        print(f"Total TeamBox records: {results['total_team_records']:,}")
        print(f"Games with wrong team count: {len(results['bad_team_counts'])}")
        print(f"Records with extreme stats: {results['extreme_stats']}")
        print("\nCoverage by season:")
        for season, data in sorted(results["coverage"].items()):
            pct = (data["has_boxscores"] / data["total"] * 100) if data["total"] > 0 else 0
            print(f"  {season}: {data['has_boxscores']}/{data['total']} ({pct:.1f}%)")
        return

    # Determine seasons to process
    seasons = args.seasons if args.seasons else DEFAULT_BACKFILL_SEASONS
    logger.info(f"Processing {len(seasons)} seasons: {min(seasons)}-{max(seasons)}")

    if args.dry_run:
        logger.info("DRY RUN — no data will be written to database")

    # Build one-time lookup tables
    logger.info("Building lookup tables...")
    player_lookup = build_player_lookup(db_path)
    team_lookup = build_team_lookup(db_path)

    # Process each season
    overall_stats = {
        "seasons_processed": 0,
        "seasons_skipped": 0,
        "total_games_matched": 0,
        "total_games_skipped": 0,
        "total_games_unmatched": 0,
        "total_player_records": 0,
        "total_team_records": 0,
        "total_players_unmatched": 0,
        "total_errors": 0,
    }

    for year in seasons:
        nba_season = f"{year - 1}-{year}"

        # Download parquet files
        player_path, team_path = download_season_files(
            year, args.cache_dir, args.skip_download
        )

        if not player_path or not team_path:
            logger.warning(f"Skipping {nba_season}: parquet files not available")
            overall_stats["seasons_skipped"] += 1
            continue

        # Process season
        season_stats = process_season(
            year=year,
            player_box_path=player_path,
            team_box_path=team_path,
            db_path=db_path,
            player_lookup=player_lookup,
            team_lookup=team_lookup,
            dry_run=args.dry_run,
        )

        # Log season summary
        logger.info(
            f"{nba_season}: "
            f"matched={season_stats['games_matched']}, "
            f"skipped={season_stats['games_skipped_existing']}, "
            f"unmatched={season_stats['games_unmatched']}, "
            f"players={season_stats['player_records_inserted']}, "
            f"teams={season_stats['team_records_inserted']}, "
            f"player_unmatch={season_stats['players_unmatched']}, "
            f"errors={len(season_stats['errors'])}"
        )

        # Accumulate overall stats
        overall_stats["seasons_processed"] += 1
        overall_stats["total_games_matched"] += season_stats["games_matched"]
        overall_stats["total_games_skipped"] += season_stats["games_skipped_existing"]
        overall_stats["total_games_unmatched"] += season_stats["games_unmatched"]
        overall_stats["total_player_records"] += season_stats["player_records_inserted"]
        overall_stats["total_team_records"] += season_stats["team_records_inserted"]
        overall_stats["total_players_unmatched"] += season_stats["players_unmatched"]
        overall_stats["total_errors"] += len(season_stats["errors"])

    # Print overall summary
    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Seasons processed:    {overall_stats['seasons_processed']}")
    print(f"Seasons skipped:      {overall_stats['seasons_skipped']}")
    print(f"Games matched:        {overall_stats['total_games_matched']:,}")
    print(f"Games skipped (exist):{overall_stats['total_games_skipped']:,}")
    print(f"Games unmatched:      {overall_stats['total_games_unmatched']:,}")
    print(f"Player records:       {overall_stats['total_player_records']:,}")
    print(f"Team records:         {overall_stats['total_team_records']:,}")
    print(f"Players unmatched:    {overall_stats['total_players_unmatched']:,}")
    print(f"Errors:               {overall_stats['total_errors']}")
    if args.dry_run:
        print("\n(DRY RUN — no data was written)")
    print("=" * 60)

    # Run validation after backfill
    if not args.dry_run and overall_stats["total_games_matched"] > 0:
        logger.info("Running post-backfill validation...")
        val_results = validate_backfill(db_path)
        print(f"\nPost-backfill: {val_results['total_player_records']:,} total PlayerBox records")
        print(f"Post-backfill: {val_results['total_team_records']:,} total TeamBox records")
        if val_results["bad_team_counts"]:
            print(f"WARNING: {len(val_results['bad_team_counts'])} games with wrong team count")
        if val_results["extreme_stats"] > 0:
            print(f"WARNING: {val_results['extreme_stats']} records with extreme stat values")


if __name__ == "__main__":
    main()
