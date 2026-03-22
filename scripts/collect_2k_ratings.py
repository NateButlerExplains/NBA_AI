"""
collect_2k_ratings.py

Collects NBA 2K player ratings data for use in player archetype modeling.
Stores all skill attributes (0-99 scale) per player per edition year into
the TwoKRatings table in the SQLite database.

Data sources (in priority order):
  1. nba2kapi.com REST API — free, ~2,500+ current-edition players with all
     37 skill attributes. Sourced from 2kratings.com. Current edition only.
  2. MikeYan01/nba2k-player-ratings GitHub repo — pre-scraped JSON with all
     37 attributes for the current edition. Fallback if API is unavailable.
  3. Direct scraping of 2kratings.com — supports historical editions
     (NBA 2K16 through 2K26) by crawling team roster pages and individual
     player detail pages. Used to fill in prior years.

Player matching: Names from 2K data are fuzzy-matched against the Players
table (person_id, first_name, last_name) in our SQLite DB using
rapidfuzz (if available) or difflib.

Usage:
    python scripts/collect_2k_ratings.py
    python scripts/collect_2k_ratings.py --editions 2024 2025 2026
    python scripts/collect_2k_ratings.py --source api
    python scripts/collect_2k_ratings.py --source github
    python scripts/collect_2k_ratings.py --source scrape --editions 2020 2021 2022 2023 2024 2025 2026
    python scripts/collect_2k_ratings.py --log_level DEBUG
    python scripts/collect_2k_ratings.py --dry-run
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH
from src.logging_config import setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 37 canonical skill attributes we want, in a stable order.
# Attribute names as they appear in the MikeYan01 JSON / nba2kapi response.
SKILL_ATTRIBUTES = [
    "closeShot",
    "midRangeShot",
    "threePointShot",
    "freeThrow",
    "shotIQ",
    "offensiveConsistency",
    "speed",
    "agility",
    "strength",
    "vertical",
    "stamina",
    "hustle",
    "overallDurability",
    "layup",
    "standingDunk",
    "drivingDunk",
    "postHook",
    "postFade",
    "postControl",
    "drawFoul",
    "hands",
    "passAccuracy",
    "ballHandle",
    "speedWithBall",
    "passIQ",
    "passVision",
    "interiorDefense",
    "perimeterDefense",
    "steal",
    "block",
    "helpDefenseIQ",
    "passPerception",
    "defensiveConsistency",
    "offensiveRebound",
    "defensiveRebound",
    "intangibles",
    "overallAttribute",
]

# Human-readable names matching the JSON keys (for the attributes JSON blob).
# Maps camelCase key -> readable display name.
ATTRIBUTE_DISPLAY_NAMES = {
    "closeShot": "Close Shot",
    "midRangeShot": "Mid-Range Shot",
    "threePointShot": "Three-Point Shot",
    "freeThrow": "Free Throw",
    "shotIQ": "Shot IQ",
    "offensiveConsistency": "Offensive Consistency",
    "speed": "Speed",
    "agility": "Agility",
    "strength": "Strength",
    "vertical": "Vertical",
    "stamina": "Stamina",
    "hustle": "Hustle",
    "overallDurability": "Durability",
    "layup": "Layup",
    "standingDunk": "Standing Dunk",
    "drivingDunk": "Driving Dunk",
    "postHook": "Post Hook",
    "postFade": "Post Fade",
    "postControl": "Post Control",
    "drawFoul": "Draw Foul",
    "hands": "Hands",
    "passAccuracy": "Pass Accuracy",
    "ballHandle": "Ball Handle",
    "speedWithBall": "Speed with Ball",
    "passIQ": "Pass IQ",
    "passVision": "Pass Vision",
    "interiorDefense": "Interior Defense",
    "perimeterDefense": "Perimeter Defense",
    "steal": "Steal",
    "block": "Block",
    "helpDefenseIQ": "Help Defense IQ",
    "passPerception": "Pass Perception",
    "defensiveConsistency": "Defensive Consistency",
    "offensiveRebound": "Offensive Rebound",
    "defensiveRebound": "Defensive Rebound",
    "intangibles": "Intangibles",
    "overallAttribute": "Overall Rating",
}

# --- 2kratings.com scraping ---

TWOK_RATINGS_BASE_URL = "https://www.2kratings.com"

# Current 30 NBA team slugs as used in 2kratings.com URLs.
CURRENT_TEAM_SLUGS = [
    "atlanta-hawks",
    "boston-celtics",
    "brooklyn-nets",
    "charlotte-hornets",
    "chicago-bulls",
    "cleveland-cavaliers",
    "dallas-mavericks",
    "denver-nuggets",
    "detroit-pistons",
    "golden-state-warriors",
    "houston-rockets",
    "indiana-pacers",
    "los-angeles-clippers",
    "los-angeles-lakers",
    "memphis-grizzlies",
    "miami-heat",
    "milwaukee-bucks",
    "minnesota-timberwolves",
    "new-orleans-pelicans",
    "new-york-knicks",
    "oklahoma-city-thunder",
    "orlando-magic",
    "philadelphia-76ers",
    "phoenix-suns",
    "portland-trail-blazers",
    "sacramento-kings",
    "san-antonio-spurs",
    "toronto-raptors",
    "utah-jazz",
    "washington-wizards",
]

# Attribute CSS selectors — 2kratings.com uses index-based attribute boxes
# within ".content .card .card-body .list-no-bullet li .attribute-box".
# The ordering matches:
#   Outside Scoring (6): closeShot, midRangeShot, threePointShot, freeThrow, shotIQ, offensiveConsistency
#   Inside Scoring (8):  layup, standingDunk, drivingDunk, postHook, postFade, postControl, drawFoul, hands
#   Athleticism (7):     speed, agility, strength, vertical, stamina, hustle, overallDurability
#   Playmaking (5):      passAccuracy, ballHandle, speedWithBall, passIQ, passVision
#   Defense (7):         interiorDefense, perimeterDefense, steal, block, helpDefenseIQ, passPerception, defensiveConsistency
#   Rebounding (2):      offensiveRebound, defensiveRebound
SCRAPE_ATTRIBUTE_ORDER = [
    # Outside Scoring
    "closeShot",
    "midRangeShot",
    "threePointShot",
    "freeThrow",
    "shotIQ",
    "offensiveConsistency",
    # Inside Scoring
    "layup",
    "standingDunk",
    "drivingDunk",
    "postHook",
    "postFade",
    "postControl",
    "drawFoul",
    "hands",
    # Athleticism
    "speed",
    "agility",
    "strength",
    "vertical",
    "stamina",
    "hustle",
    "overallDurability",
    # Playmaking
    "passAccuracy",
    "ballHandle",
    "speedWithBall",
    "passIQ",
    "passVision",
    # Defense
    "interiorDefense",
    "perimeterDefense",
    "steal",
    "block",
    "helpDefenseIQ",
    "passPerception",
    "defensiveConsistency",
    # Rebounding
    "offensiveRebound",
    "defensiveRebound",
]

# GitHub raw URL for MikeYan01's pre-scraped data
GITHUB_LEAGUE_JSON_URL = "https://raw.githubusercontent.com/MikeYan01/nba2k-player-ratings/master/data/league.json"

# nba2kapi.com settings
NBA2K_API_BASE = "https://api.nba2kdb.com/api"
NBA2K_API_KEY_ENV = "NBA2K_API_KEY"  # Environment variable name for the API key

# Requests settings
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}
REQUEST_TIMEOUT = 30  # seconds
SCRAPE_DELAY = 2.0  # seconds between requests to 2kratings.com (be respectful)
API_DELAY = 0.5  # seconds between API requests

# SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS TwoKRatings (
    person_id INTEGER,
    edition_year INTEGER,
    overall_rating INTEGER,
    attributes TEXT,
    PRIMARY KEY (person_id, edition_year)
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO TwoKRatings (person_id, edition_year, overall_rating, attributes)
VALUES (?, ?, ?, ?)
"""

# ---------------------------------------------------------------------------
# Name normalization and matching
# ---------------------------------------------------------------------------

# Common 2K name -> NBA API name mappings for players whose names differ
# between sources (suffixes, accents, abbreviations, etc.)
MANUAL_NAME_OVERRIDES = {
    # Suffix differences
    "Kelly Oubre": "Kelly Oubre Jr.",
    "Derrick Jones": "Derrick Jones Jr.",
    "Tim Hardaway": "Tim Hardaway Jr.",
    "Larry Nance": "Larry Nance Jr.",
    "Gary Trent": "Gary Trent Jr.",
    "Marcus Morris": "Marcus Morris Sr.",
    "Jaren Jackson": "Jaren Jackson Jr.",
    "Otto Porter": "Otto Porter Jr.",
    "Wendell Carter": "Wendell Carter Jr.",
    "Michael Porter": "Michael Porter Jr.",
    "Robert Williams": "Robert Williams III",
    "Lonnie Walker": "Lonnie Walker IV",
    "Kevin Porter": "Kevin Porter Jr.",
    "Jabari Smith": "Jabari Smith Jr.",
    "Kenyon Martin": "Kenyon Martin Jr.",
    "Troy Brown": "Troy Brown Jr.",
    "Gary Payton II": "Gary Payton II",
    "Trey Murphy": "Trey Murphy III",
    "Dennis Smith": "Dennis Smith Jr.",
    "Kris Dunn": "Kris Dunn",
    # Name spelling differences
    "Nic Claxton": "Nicolas Claxton",
    "Naz Reid": "Naz Reid",
    "PJ Washington": "P.J. Washington",
    "CJ McCollum": "CJ McCollum",
    "OG Anunoby": "OG Anunoby",
    "RJ Barrett": "RJ Barrett",
    "AJ Griffin": "AJ Griffin",
    "TJ McConnell": "T.J. McConnell",
    "TJ Warren": "T.J. Warren",
    "KJ Martin": "Kenyon Martin Jr.",
    "Herb Jones": "Herbert Jones",
    "Moe Wagner": "Moritz Wagner",
    "Ish Smith": "Ishmael Smith",
    "Cedi Osman": "Cedi Osman",
    "Alexandre Sarr": "Alex Sarr",
}


def normalize_name(name: str) -> str:
    """
    Normalize a player name for comparison.

    Strips accents (via ASCII folding), lowercases, removes suffixes like
    Jr./Sr./III/IV/II, removes periods and extra whitespace.
    """
    if not name:
        return ""

    # Try to handle Unicode accents
    try:
        import unicodedata

        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
    except Exception:
        pass

    name = name.lower().strip()

    # Normalize various apostrophe/quote characters to a standard one
    name = re.sub(r"[\u2018\u2019\u2032\u0060\u00b4]", "'", name)

    # Remove common suffixes for matching
    name = re.sub(r"\b(jr|sr|jr\.|sr\.|iii|iv|ii|i)\b\.?", "", name)

    # Remove periods and extra spaces
    name = name.replace(".", "")
    name = re.sub(r"\s+", " ", name).strip()

    return name


def build_player_lookup(db_path: str = None) -> dict:
    """
    Build lookup structures from the Players table for name matching.

    Returns:
        dict with keys:
            'exact': {normalized_name: person_id}
            'first_last': {(first, last): person_id}
            'all_players': [(person_id, first_name, last_name, full_name, from_year, to_year)]
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT person_id, first_name, last_name, full_name, from_year, to_year "
            "FROM Players"
        )
        rows = cursor.fetchall()

    exact_lookup = {}
    first_last_lookup = {}
    all_players = []

    for person_id, first_name, last_name, full_name, from_year, to_year in rows:
        all_players.append(
            (person_id, first_name, last_name, full_name, from_year, to_year)
        )

        # "First Last" normalized
        constructed = f"{first_name} {last_name}".strip()
        norm = normalize_name(constructed)
        if norm:
            exact_lookup[norm] = person_id

        # Also store (first, last) tuple for partial matching
        fn = normalize_name(first_name)
        ln = normalize_name(last_name)
        if fn and ln:
            first_last_lookup[(fn, ln)] = person_id

    return {
        "exact": exact_lookup,
        "first_last": first_last_lookup,
        "all_players": all_players,
    }


def match_player_name(
    twok_name: str,
    player_lookup: dict,
    edition_year: int = None,
    min_score: float = 0.85,
) -> Optional[int]:
    """
    Match a 2K player name to a person_id from our Players table.

    Strategy:
      1. Check manual overrides first.
      2. Try exact normalized match.
      3. Try (first, last) tuple match.
      4. Fall back to fuzzy matching against all players.

    Args:
        twok_name: Player name as it appears in 2K data.
        player_lookup: Dict from build_player_lookup().
        edition_year: Edition year (e.g. 2026) for filtering active players.
        min_score: Minimum fuzzy match score (0-1) to accept. Default 0.85.

    Returns:
        person_id (int) or None if no match found.
    """
    if not twok_name:
        return None

    # Step 1: Manual override
    override_name = MANUAL_NAME_OVERRIDES.get(twok_name)
    if override_name:
        norm_override = normalize_name(override_name)
        pid = player_lookup["exact"].get(norm_override)
        if pid:
            return pid

    # Step 2: Exact normalized match
    norm_name = normalize_name(twok_name)
    pid = player_lookup["exact"].get(norm_name)
    if pid:
        return pid

    # Step 3: (first, last) match
    parts = norm_name.split()
    if len(parts) >= 2:
        first = parts[0]
        last = " ".join(parts[1:])
        pid = player_lookup["first_last"].get((first, last))
        if pid:
            return pid

    # Step 4: Fuzzy matching
    # Try rapidfuzz first (much faster), fall back to difflib
    best_score = 0.0
    best_pid = None

    try:
        from rapidfuzz import fuzz as rfuzz

        for (
            person_id,
            first_name,
            last_name,
            full_name,
            from_year,
            to_year,
        ) in player_lookup["all_players"]:
            # Filter by active years if edition_year provided
            # Edition year N corresponds to season (N-1)-(N), so player active in year N-1
            if edition_year and from_year and to_year:
                season_year = edition_year - 1
                if season_year < from_year or season_year > to_year:
                    continue

            candidate = normalize_name(f"{first_name} {last_name}")
            score = rfuzz.ratio(norm_name, candidate) / 100.0

            if score > best_score:
                best_score = score
                best_pid = person_id
    except ImportError:
        # Fall back to difflib SequenceMatcher
        for (
            person_id,
            first_name,
            last_name,
            full_name,
            from_year,
            to_year,
        ) in player_lookup["all_players"]:
            if edition_year and from_year and to_year:
                season_year = edition_year - 1
                if season_year < from_year or season_year > to_year:
                    continue

            candidate = normalize_name(f"{first_name} {last_name}")
            score = SequenceMatcher(None, norm_name, candidate).ratio()

            if score > best_score:
                best_score = score
                best_pid = person_id

    if best_score >= min_score and best_pid is not None:
        logging.debug(
            f"  Fuzzy match: '{twok_name}' -> person_id={best_pid} (score={best_score:.3f})"
        )
        return best_pid

    logging.debug(f"  No match for '{twok_name}' (best fuzzy score={best_score:.3f})")
    return None


# ---------------------------------------------------------------------------
# Source 1: nba2kapi.com REST API
# ---------------------------------------------------------------------------


def fetch_from_api(api_key: str, edition_year: int = 2026) -> list[dict]:
    """
    Fetch all players from the nba2kapi.com REST API.

    The API paginates with cursor-based pagination, returning up to 100
    players per request. We paginate until hasMore is False.

    Args:
        api_key: API key for nba2kapi.com (from NBA2K_API_KEY env var).
        edition_year: The 2K edition year (used only for labeling; the API
                      returns the current edition only).

    Returns:
        list[dict]: Player records with 'name', 'overallAttribute', and
                    all skill attribute fields.
    """
    logging.info(f"Fetching from nba2kapi.com API (edition {edition_year})...")
    all_players = []
    cursor = None
    page = 0

    headers = {"X-API-Key": api_key}

    while True:
        page += 1
        params = {"limit": 100, "teamType": "curr"}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                f"{NBA2K_API_BASE}/players",
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logging.error(f"API request failed on page {page}: {e}")
            break

        if not data.get("success"):
            logging.error(f"API returned success=false on page {page}: {data}")
            break

        players = data.get("data", [])
        if not players:
            break

        all_players.extend(players)
        logging.info(
            f"  Page {page}: fetched {len(players)} players (total: {len(all_players)})"
        )

        pagination = data.get("pagination", {})
        if not pagination.get("hasMore", False):
            break

        cursor = pagination.get("nextCursor")
        time.sleep(API_DELAY)

    logging.info(f"API total: {len(all_players)} players")
    return _normalize_api_records(all_players, edition_year)


def _normalize_api_records(records: list[dict], edition_year: int) -> list[dict]:
    """
    Normalize API response records into our standard format.

    The API uses the same camelCase keys as the MikeYan01 JSON, plus
    some extras (slug, positions, height, weight, images, _id).
    We extract just the fields we care about.
    """
    normalized = []
    for rec in records:
        player = {
            "name": rec.get("name", ""),
            "team": rec.get("team", ""),
            "edition_year": edition_year,
            "overallAttribute": _safe_int(
                rec.get("overall") or rec.get("overallAttribute")
            ),
        }
        # Extract all skill attributes
        for attr in SKILL_ATTRIBUTES:
            if attr == "overallAttribute":
                continue  # already handled above
            val = rec.get(attr)
            if val is not None:
                player[attr] = _safe_int(val)
        normalized.append(player)
    return normalized


# ---------------------------------------------------------------------------
# Source 2: GitHub (MikeYan01/nba2k-player-ratings)
# ---------------------------------------------------------------------------


def fetch_from_github(edition_year: int = 2026) -> list[dict]:
    """
    Fetch pre-scraped 2K ratings from MikeYan01's GitHub repo.

    The repo contains league.json with all current-edition players and
    their full 37 attributes. This is a snapshot of whatever edition
    was current when the repo was last updated.

    Args:
        edition_year: The edition year to label the data with.

    Returns:
        list[dict]: Player records in normalized format.
    """
    logging.info(f"Fetching from GitHub (MikeYan01/nba2k-player-ratings)...")

    try:
        resp = requests.get(
            GITHUB_LEAGUE_JSON_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        raw_players = resp.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch GitHub data: {e}")
        return []

    logging.info(f"GitHub: fetched {len(raw_players)} players")

    normalized = []
    for rec in raw_players:
        player = {
            "name": rec.get("name", ""),
            "team": rec.get("team", ""),
            "edition_year": edition_year,
            "overallAttribute": _safe_int(rec.get("overallAttribute")),
        }
        for attr in SKILL_ATTRIBUTES:
            if attr == "overallAttribute":
                continue
            val = rec.get(attr)
            if val is not None:
                player[attr] = _safe_int(val)
        normalized.append(player)
    return normalized


# ---------------------------------------------------------------------------
# Source 3: Direct scraping of 2kratings.com
# ---------------------------------------------------------------------------


def fetch_from_scrape(edition_year: int = 2026) -> list[dict]:
    """
    Scrape 2kratings.com for a specific edition's player ratings.

    For the current edition, URLs are like:
        https://www.2kratings.com/teams/boston-celtics
        https://www.2kratings.com/lebron-james

    For historical editions, URLs follow the pattern:
        https://www.2kratings.com/nba-2k{YY}/teams/boston-celtics
        https://www.2kratings.com/nba-2k{YY}/lebron-james

    where YY is the 2-digit edition suffix (e.g., 25 for NBA 2K25).

    Args:
        edition_year: The 2K edition year (e.g. 2025 for NBA 2K25).

    Returns:
        list[dict]: Player records in normalized format.
    """
    # Determine the current edition by checking the site's title
    current_edition = _detect_current_edition()
    is_current = (
        (edition_year == current_edition) if current_edition else (edition_year == 2026)
    )

    if is_current:
        prefix = ""
        logging.info(f"Scraping 2kratings.com (current edition = {edition_year})...")
    else:
        yy = edition_year % 100
        prefix = f"/nba-2k{yy}"
        logging.info(
            f"Scraping 2kratings.com (historical edition NBA 2K{yy}, prefix='{prefix}')..."
        )

    # Step 1: Get player URLs from team pages
    all_player_urls = []
    for team_slug in CURRENT_TEAM_SLUGS:
        team_url = f"{TWOK_RATINGS_BASE_URL}{prefix}/teams/{team_slug}"
        logging.debug(f"  Fetching team: {team_url}")

        try:
            resp = requests.get(
                team_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 403:
                logging.warning(
                    f"  403 Forbidden for {team_url} — 2kratings.com may be blocking requests"
                )
                return []
            if resp.status_code == 404:
                logging.warning(
                    f"  404 Not Found for {team_url} — edition may not exist"
                )
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            logging.warning(f"  Failed to fetch team page {team_slug}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        entries = soup.select("tbody .entry-font a")
        for entry in entries:
            href = entry.get("href")
            if href:
                full_url = urljoin(TWOK_RATINGS_BASE_URL, href)
                all_player_urls.append((full_url, team_slug))

        logging.debug(f"  {team_slug}: found {len(entries)} player links")
        time.sleep(SCRAPE_DELAY)

    logging.info(f"Found {len(all_player_urls)} player URLs across all teams")

    # Step 2: Scrape individual player pages
    all_players = []
    for i, (player_url, team_slug) in enumerate(all_player_urls):
        if (i + 1) % 50 == 0:
            logging.info(f"  Scraping player {i + 1}/{len(all_player_urls)}...")

        player_data = _scrape_player_page(player_url, team_slug, edition_year)
        if player_data:
            all_players.append(player_data)

        time.sleep(SCRAPE_DELAY)

    logging.info(f"Scraped {len(all_players)} players for edition {edition_year}")
    return all_players


def _detect_current_edition() -> Optional[int]:
    """
    Try to detect the current edition from the 2kratings.com homepage.

    Returns:
        int: The current edition year (e.g. 2026), or None if detection fails.
    """
    try:
        resp = requests.get(
            TWOK_RATINGS_BASE_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            match = re.search(r"NBA 2K(\d{2})", resp.text)
            if match:
                yy = int(match.group(1))
                # Convert 2-digit to 4-digit year (26 -> 2026)
                year = 2000 + yy
                logging.debug(f"Detected current edition: NBA 2K{yy} ({year})")
                return year
    except Exception:
        pass
    return None


def _scrape_player_page(url: str, team_slug: str, edition_year: int) -> Optional[dict]:
    """
    Scrape a single player's detail page from 2kratings.com.

    Extracts the player name, overall rating, and all 35 skill attributes
    from the attribute boxes on the page.

    Args:
        url: Full URL to the player's detail page.
        team_slug: Team slug for labeling.
        edition_year: Edition year for labeling.

    Returns:
        dict with player data, or None on failure.
    """
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 403:
            logging.warning(f"  403 Forbidden: {url}")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"  Failed to fetch player page {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract name
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else ""
    if not name:
        logging.debug(f"  No name found at {url}")
        return None

    # Extract overall rating
    overall_box = soup.select_one(".attribute-box-player")
    overall = _safe_int(overall_box.get_text(strip=True)) if overall_box else None

    # Extract detailed attributes
    attribute_boxes = soup.select(
        ".content .card .card-body .list-no-bullet li .attribute-box"
    )

    player = {
        "name": name,
        "team": _slug_to_team_name(team_slug),
        "edition_year": edition_year,
        "overallAttribute": overall,
    }

    # Map attribute boxes to attribute names by index
    for idx, attr_name in enumerate(SCRAPE_ATTRIBUTE_ORDER):
        if idx < len(attribute_boxes):
            try:
                val_text = attribute_boxes[idx].get_text(strip=True)
                player[attr_name] = _safe_int(val_text)
            except (IndexError, AttributeError):
                pass

    return player


def _slug_to_team_name(slug: str) -> str:
    """Convert a team slug like 'los-angeles-lakers' to 'Los Angeles Lakers'."""
    return " ".join(word.capitalize() for word in slug.split("-"))


# ---------------------------------------------------------------------------
# Data processing and DB insertion
# ---------------------------------------------------------------------------


def process_and_save(
    player_records: list[dict],
    player_lookup: dict,
    db_path: str = None,
    dry_run: bool = False,
) -> dict:
    """
    Match 2K player names to person_ids and save to the TwoKRatings table.

    Args:
        player_records: List of normalized player dicts from any source.
        player_lookup: Dict from build_player_lookup().
        db_path: Optional DB path override.
        dry_run: If True, do not write to DB, just report what would happen.

    Returns:
        dict with stats: matched, unmatched, saved, by_edition.
    """
    stats = {
        "matched": 0,
        "unmatched": 0,
        "saved": 0,
        "unmatched_names": [],
        "by_edition": {},
    }
    rows_to_insert = []

    for rec in player_records:
        name = rec.get("name", "")
        edition_year = rec.get("edition_year", 2026)
        overall = rec.get("overallAttribute")

        person_id = match_player_name(name, player_lookup, edition_year=edition_year)

        if person_id is None:
            stats["unmatched"] += 1
            stats["unmatched_names"].append((name, edition_year))
            continue

        stats["matched"] += 1
        stats["by_edition"].setdefault(edition_year, 0)
        stats["by_edition"][edition_year] += 1

        # Build attributes JSON blob
        attrs = {}
        for attr in SKILL_ATTRIBUTES:
            if attr == "overallAttribute":
                continue
            val = rec.get(attr)
            if val is not None:
                display_name = ATTRIBUTE_DISPLAY_NAMES.get(attr, attr)
                attrs[display_name] = val

        attrs_json = json.dumps(attrs, separators=(",", ":"))

        rows_to_insert.append((person_id, edition_year, overall, attrs_json))

    if dry_run:
        logging.info(f"Dry run: would insert {len(rows_to_insert)} rows")
        stats["saved"] = 0
    else:
        if rows_to_insert:
            with get_db(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(CREATE_TABLE_SQL)
                cursor.executemany(INSERT_SQL, rows_to_insert)
                cursor.execute("SELECT COUNT(*) FROM TwoKRatings")
                total = cursor.fetchone()[0]
                conn.commit()

            stats["saved"] = len(rows_to_insert)
            logging.info(
                f"Saved {stats['saved']} rows to TwoKRatings (total in table: {total})"
            )
        else:
            logging.warning("No rows to insert")

    return stats


def _safe_int(value) -> Optional[int]:
    """Convert a value to int, returning None if invalid."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------


def print_summary(stats: dict):
    """Print a summary of the collection results."""
    print("\n" + "=" * 60)
    print("TwoKRatings Collection Summary")
    print("=" * 60)
    print(f"  Total records processed:  {stats['matched'] + stats['unmatched']}")
    print(f"  Matched to person_id:     {stats['matched']}")
    print(f"  Unmatched:                {stats['unmatched']}")
    print(f"  Saved to DB:              {stats['saved']}")

    if stats["by_edition"]:
        print("\n  By edition:")
        for year in sorted(stats["by_edition"].keys()):
            print(
                f"    NBA 2K{year % 100:02d} ({year}):  {stats['by_edition'][year]} players"
            )

    # Show first N unmatched names
    unmatched = stats.get("unmatched_names", [])
    if unmatched:
        n_show = min(20, len(unmatched))
        print(f"\n  Unmatched names (showing {n_show}/{len(unmatched)}):")
        for name, year in unmatched[:n_show]:
            print(f"    - {name} (2K{year % 100:02d})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _update_scrape_delay(new_delay: float):
    """Update the module-level SCRAPE_DELAY variable."""
    global SCRAPE_DELAY
    SCRAPE_DELAY = new_delay


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect NBA 2K player ratings and store in TwoKRatings table. "
            "Supports multiple data sources: nba2kapi.com API, GitHub pre-scraped "
            "JSON, or direct scraping of 2kratings.com."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["api", "github", "scrape", "auto"],
        default="auto",
        help=(
            "Data source. 'api' = nba2kapi.com (requires NBA2K_API_KEY env var), "
            "'github' = MikeYan01 pre-scraped JSON, "
            "'scrape' = direct scraping of 2kratings.com (supports historical editions), "
            "'auto' = try API first, fall back to GitHub. Default: auto"
        ),
    )
    parser.add_argument(
        "--editions",
        type=int,
        nargs="+",
        default=[2026],
        help=(
            "Edition years to collect (e.g. 2024 2025 2026). "
            "API and GitHub only support the current edition. "
            "Scrape mode supports 2K16 (2016) through current. Default: 2026"
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="nba2kapi.com API key. If not provided, reads from NBA2K_API_KEY env var.",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write to DB. Just fetch data, match names, and report.",
    )
    parser.add_argument(
        "--scrape-delay",
        type=float,
        default=SCRAPE_DELAY,
        help=f"Delay between scrape requests in seconds. Default: {SCRAPE_DELAY}",
    )
    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())

    db_path = args.db_path or DB_PATH
    source = args.source
    editions = sorted(set(args.editions))

    # Update scrape delay module-level variable if user provided a custom value
    _update_scrape_delay(args.scrape_delay)

    logging.info(f"Database: {db_path}")
    logging.info(f"Source: {source}")
    logging.info(f"Editions: {editions}")

    # Build player lookup from DB
    logging.info("Building player name lookup from Players table...")
    player_lookup = build_player_lookup(db_path)
    logging.info(
        f"  Loaded {len(player_lookup['exact'])} exact names, "
        f"{len(player_lookup['all_players'])} total players"
    )

    # Collect data from each edition
    all_records = []

    for edition_year in editions:
        logging.info(
            f"\n--- Edition: NBA 2K{edition_year % 100:02d} ({edition_year}) ---"
        )

        records = []

        if source == "api":
            api_key = args.api_key or _get_api_key()
            if not api_key:
                logging.error(
                    "No API key provided. Set NBA2K_API_KEY env var or use --api-key. "
                    "Get a free key at https://www.nba2kapi.com/dashboard"
                )
                sys.exit(1)
            records = fetch_from_api(api_key, edition_year)

        elif source == "github":
            records = fetch_from_github(edition_year)

        elif source == "scrape":
            records = fetch_from_scrape(edition_year)

        elif source == "auto":
            # Try API first (if key available), then GitHub
            api_key = args.api_key or _get_api_key()
            if api_key:
                logging.info("Auto mode: trying API first...")
                records = fetch_from_api(api_key, edition_year)

            if not records:
                logging.info("Auto mode: falling back to GitHub...")
                records = fetch_from_github(edition_year)

            if not records:
                logging.warning(
                    f"Auto mode: no data fetched for edition {edition_year}. "
                    f"Try --source scrape for historical editions, or set NBA2K_API_KEY "
                    f"for the current edition."
                )

        if records:
            logging.info(
                f"Fetched {len(records)} player records for edition {edition_year}"
            )
            all_records.extend(records)
        else:
            logging.warning(f"No records fetched for edition {edition_year}")

    if not all_records:
        logging.error("No data collected from any source. Exiting.")
        sys.exit(1)

    # Process and save
    logging.info(f"\nTotal records to process: {len(all_records)}")
    stats = process_and_save(
        all_records,
        player_lookup,
        db_path=db_path,
        dry_run=args.dry_run,
    )

    # Summary
    print_summary(stats)

    logging.info("Done.")


def _get_api_key() -> Optional[str]:
    """Get API key from environment variable."""
    import os

    key = os.environ.get(NBA2K_API_KEY_ENV)
    if key:
        logging.debug("Found API key in environment")
    return key


if __name__ == "__main__":
    main()
