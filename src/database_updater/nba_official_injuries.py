"""
nba_official_injuries.py

Fetches and parses NBA's official daily injury report PDFs.
Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_05PM.pdf

This provides granular injury data including:
- body_part: Ankle, Knee, Hamstring, etc.
- injury_type: Sprain, Strain, Soreness, Surgery, etc.
- injury_side: Left, Right
- status: Out, Questionable, Doubtful, Probable, Available

Used for: Historical backfill and daily updates to complement ESPN real-time data.

Functions:
    - update_nba_official_injuries(days_back=1): Updates recent injury reports
    - fetch_injury_report(date): Fetches and parses a single day's PDF
    - parse_injury_pdf(pdf_content): Parses PDF content to extract injuries
"""

import io
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import pdfplumber
import requests
from tqdm import tqdm

from src.config import config

DB_PATH = config["database"]["path"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PDF_URL_TEMPLATE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_05PM.pdf"
)

# Cache configuration
INJURY_CACHE_TODAY_HOURS = 2  # Refetch today's injuries every 2 hours


def parse_injury_reason(reason: str) -> tuple:
    """
    Parse injury reason text to extract body_part, injury_type, injury_side, and category.

    Returns:
        tuple: (body_part, injury_type, injury_side, category)
    """
    if not reason or pd.isna(reason):
        return None, None, None, None

    reason = reason.upper()

    # Filter out non-injury related absences
    non_injury_keywords = [
        "GLEAGUE",
        "G LEAGUE",
        "G-LEAGUE",
        "TWO-WAY",
        "TRADE",
        "PERSONAL",
        "REST",
        "COACH",
        "NOT WITH TEAM",
        "SUSPENSION",
        "RETURN TO COMPETITION",
        "RECONDITIONING",
    ]
    if any(keyword in reason for keyword in non_injury_keywords):
        return None, None, None, "Non-Injury"

    # Extract side
    side = None
    if "LEFT" in reason:
        side = "Left"
    elif "RIGHT" in reason:
        side = "Right"

    # Map body parts
    body_parts_map = {
        "ANKLE": "Ankle",
        "KNEE": "Knee",
        "HAMSTRING": "Hamstring",
        "FOOT": "Foot",
        "BACK": "Back",
        "HIP": "Hip",
        "SHOULDER": "Shoulder",
        "HAND": "Hand",
        "FINGER": "Finger",
        "WRIST": "Wrist",
        "ELBOW": "Elbow",
        "CALF": "Calf",
        "THIGH": "Thigh",
        "GROIN": "Groin",
        "RIB": "Ribs",
        "ACHILLES": "Achilles",
        "QUAD": "Quad",
        "TOE": "Toe",
        "HEAD": "Head",
        "NECK": "Neck",
        "CONCUSSION": "Head",
        "ABDOMINAL": "Abdomen",
        "ABDOMEN": "Abdomen",
        "ILLNESS": "Illness",
        "COVID": "Illness",
        "LEG": "Leg",
        "ARM": "Arm",
        "PATELLAR": "Knee",
        "ACL": "Knee",
        "MCL": "Knee",
        "MENISCUS": "Knee",
        "PLANTAR": "Foot",
        "LUMBAR": "Back",
        "FACE": "Face",
        "EYE": "Eye",
        "NOSE": "Face",
        "JAW": "Face",
        "THUMB": "Hand",
        "FOREARM": "Arm",
        "BICEP": "Arm",
        "TRICEP": "Arm",
        "PELVIS": "Hip",
        "GLUTE": "Hip",
        "ADDUCTOR": "Groin",
        "OBLIQUE": "Abdomen",
    }

    body_part = None
    for key, val in body_parts_map.items():
        if key in reason:
            body_part = val
            break

    # Map injury types
    injury_types_map = {
        "SPRAIN": "Sprain",
        "STRAIN": "Strain",
        "SORENESS": "Soreness",
        "SURGERY": "Surgery",
        "FRACTURE": "Fracture",
        "CONTUSION": "Contusion",
        "TENDINITIS": "Tendinitis",
        "TENDONITIS": "Tendinitis",
        "TORN": "Tear",
        "TEAR": "Tear",
        "INFLAMMATION": "Inflammation",
        "ILLNESS": "Illness",
        "DISLOCATION": "Dislocation",
        "IMPINGEMENT": "Impingement",
        "BRUISE": "Contusion",
        "BROKEN": "Fracture",
        "BONE": "Fracture",
    }

    injury_type = None
    for key, val in injury_types_map.items():
        if key in reason:
            injury_type = val
            break

    return body_part, injury_type, side, "Injury"


def parse_injury_pdf(pdf_content: bytes) -> pd.DataFrame:
    """Parse NBA injury report PDF content."""
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_content))
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        pdf.close()
    except Exception:
        return pd.DataFrame()

    records = []
    lines = all_text.split("\n")

    current_date = None
    current_time = None
    current_matchup = None

    for line in lines:
        line = line.strip()
        if (
            not line
            or line.startswith("Page")
            or line.startswith("Injury Report:")
            or line.startswith("GameDate")
        ):
            continue

        # Match: "MM/DD/YYYY HH:MM(ET) ABC@XYZ PlayerInfo..."
        date_match = re.match(
            r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\(ET\)\s+([A-Z]{3}@[A-Z]{3})\s*(.*)",
            line,
        )
        if date_match:
            current_date = date_match.group(1)
            current_time = date_match.group(2)
            current_matchup = date_match.group(3)
            rest = date_match.group(4)
        elif re.match(r"(\d{2}:\d{2})\(ET\)\s+([A-Z]{3}@[A-Z]{3})", line):
            match = re.match(r"(\d{2}:\d{2})\(ET\)\s+([A-Z]{3}@[A-Z]{3})\s*(.*)", line)
            if match:
                current_time = match.group(1)
                current_matchup = match.group(2)
                rest = match.group(3)
        else:
            rest = line

        # Match player line
        player_match = re.match(
            r"^([A-Za-z\'\-]+,\s*[A-Za-z\'\-]+(?:\s*(?:Jr\.?|Sr\.?|III|IV|II|V))?)\s+"
            r"(Out|Available|Questionable|Doubtful|Probable)\s+(.*)$",
            rest,
        )
        if player_match and current_date:
            player_name = player_match.group(1).strip()
            status = player_match.group(2)
            reason = player_match.group(3).strip()

            body_part, injury_type, side, category = parse_injury_reason(reason)

            # Include ALL absences (injuries + rest/personal/etc.)
            records.append(
                {
                    "game_date": current_date,
                    "game_time": current_time,
                    "matchup": current_matchup,
                    "player_name": player_name,
                    "status": status,
                    "reason": reason,
                    "body_part": body_part,
                    "injury_type": injury_type,
                    "injury_side": side,
                    "category": category
                    or "Injury",  # Default to Injury if not classified
                }
            )

    return pd.DataFrame(records)


def fetch_injury_report(date: datetime) -> pd.DataFrame:
    """Fetch and parse injury report for a specific date."""
    url = PDF_URL_TEMPLATE.format(date=date.strftime("%Y-%m-%d"))

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            df = parse_injury_pdf(resp.content)
            if len(df) > 0:
                df["report_date"] = date.strftime("%Y-%m-%d")
                return df
    except Exception as e:
        logging.debug(f"Error fetching {date.strftime('%Y-%m-%d')}: {e}")

    return pd.DataFrame()


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching to Players table."""
    if not name:
        return ""
    # Split attached suffixes (WalkerIV -> Walker IV)
    name = re.sub(r"([a-z])(II|III|IV|Jr|Sr)([,\s]|$)", r"\1 \2\3", name)
    # Remove suffixes entirely for matching
    name = re.sub(
        r"\s+(Jr\.?|Sr\.?|III|II|IV)(\s|$|,)", r"\2", name, flags=re.IGNORECASE
    )
    # Remove periods, apostrophes and extra spaces
    name = name.replace(".", "").replace("'", "").strip()
    # Handle special chars (ć -> c, etc)
    replacements = {
        "ć": "c",
        "č": "c",
        "ž": "z",
        "š": "s",
        "đ": "d",
        "ö": "o",
        "ü": "u",
        "ä": "a",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.lower()


def _ensure_injury_cache_table(db_path: str = DB_PATH):
    """Create InjuryCache table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS InjuryCache (
                report_date TEXT PRIMARY KEY,
                last_fetched_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _get_injury_fetch_time(
    report_date: str, db_path: str = DB_PATH
) -> Optional[datetime]:
    """Get the last fetch time for a specific injury report date."""
    _ensure_injury_cache_table(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_fetched_at FROM InjuryCache WHERE report_date = ?",
                (report_date,),
            )
            result = cursor.fetchone()
            if result:
                return datetime.fromisoformat(result[0])
            return None
    except sqlite3.OperationalError:
        return None


def _update_injury_cache(report_date: str, db_path: str = DB_PATH):
    """Update the injury cache with current fetch timestamp."""
    _ensure_injury_cache_table(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        fetch_time = datetime.now().isoformat()
        cursor.execute(
            """
            INSERT INTO InjuryCache (report_date, last_fetched_at)
            VALUES (?, ?)
            ON CONFLICT(report_date) DO UPDATE SET last_fetched_at = excluded.last_fetched_at
            """,
            (report_date, fetch_time),
        )
        conn.commit()


def _should_fetch_injury_date(report_date: datetime, db_path: str = DB_PATH) -> bool:
    """
    Determine if an injury report date should be fetched.

    Cache strategy:
    - Today's date: Refetch if last fetch was >2 hours ago
    - Past dates: Once fetched, never refetch (permanent cache)
    """
    date_str = report_date.strftime("%Y-%m-%d")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    is_today = report_date.date() == today.date()

    last_fetch = _get_injury_fetch_time(date_str, db_path)

    if last_fetch is None:
        # Never fetched - fetch it
        return True

    if is_today:
        # Today: check if cache expired (2 hours)
        hours_since_fetch = (datetime.now() - last_fetch).total_seconds() / 3600
        if hours_since_fetch > INJURY_CACHE_TODAY_HOURS:
            logging.debug(
                f"Today's injury cache expired ({hours_since_fetch:.1f}h old) - refetching"
            )
            return True
        else:
            logging.debug(
                f"Today's injury cache fresh ({hours_since_fetch:.1f}h old) - skipping"
            )
            return False
    else:
        # Past date: permanent cache
        return False


def build_player_lookup(db_path: str = DB_PATH) -> dict:
    """Build a lookup dict mapping normalized names to NBA player IDs."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT person_id, first_name, last_name, full_name FROM Players"
        )
        players = cursor.fetchall()

    player_lookup = {}
    for person_id, first_name, last_name, full_name in players:
        if last_name and first_name:
            key1 = normalize_player_name(f"{last_name}, {first_name}")
            player_lookup[key1] = person_id
        if full_name:
            key2 = normalize_player_name(full_name)
            player_lookup[key2] = person_id

    return player_lookup


def _ensure_injury_unique_constraint(db_path: str = DB_PATH):
    """Add unique constraint to prevent duplicate injury records."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check if constraint already exists by trying to query index
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_injury_unique'"
        )
        if cursor.fetchone():
            return  # Already exists

        # Remove existing duplicates before adding constraint
        logging.debug(
            "Removing duplicate injury records before adding unique constraint..."
        )
        cursor.execute(
            """
            DELETE FROM InjuryReports
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM InjuryReports
                GROUP BY nba_player_id, player_name, report_timestamp, source, team
            )
        """
        )
        removed = cursor.rowcount
        if removed > 0:
            logging.info(f"Removed {removed} duplicate injury records")

        # Add unique constraint via index (SQLite doesn't support ALTER TABLE ADD CONSTRAINT)
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_injury_unique 
            ON InjuryReports(nba_player_id, player_name, report_timestamp, source, team)
        """
        )
        conn.commit()


def save_injury_records(df: pd.DataFrame, db_path: str = DB_PATH) -> dict:
    """Save injury records to database with player ID matching and UPSERT logic.

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    if df.empty:
        return {"added": 0, "updated": 0, "total": 0}

    # Ensure unique constraint exists
    _ensure_injury_unique_constraint(db_path)

    # Build player lookup for matching
    player_lookup = build_player_lookup(db_path)

    conn = sqlite3.connect(db_path)

    db_records = []
    for _, row in df.iterrows():
        matchup = row.get("matchup", "")
        away_team = matchup.split("@")[0] if "@" in matchup else None
        home_team = matchup.split("@")[1] if "@" in matchup else None

        # Format player name
        player_name = row["player_name"]
        if "," in player_name and ", " not in player_name:
            player_name = player_name.replace(",", ", ")

        # Match to NBA player ID
        normalized_name = normalize_player_name(player_name)
        nba_player_id = player_lookup.get(normalized_name)

        # Determine injury location category
        leg_parts = [
            "Ankle",
            "Knee",
            "Hamstring",
            "Calf",
            "Thigh",
            "Foot",
            "Toe",
            "Achilles",
            "Quad",
            "Groin",
            "Leg",
        ]
        injury_location = "Leg" if row["body_part"] in leg_parts else "Other"

        # Derive season from report_date (Oct-Dec = current year, Jan-Sep = previous year)
        report_date = row["report_date"]
        if report_date:
            year = int(report_date[:4])
            month = int(report_date[5:7])
            if month >= 10:  # Oct-Dec = start of new season
                season = f"{year}-{year + 1}"
            else:  # Jan-Sep = end of previous season
                season = f"{year - 1}-{year}"
        else:
            season = None

        db_records.append(
            {
                "nba_player_id": nba_player_id,
                "player_name": player_name,
                "team": away_team or home_team,
                "status": row["status"],
                "injury_type": row["injury_type"],
                "body_part": row["body_part"],
                "injury_location": injury_location,
                "injury_side": row["injury_side"],
                "category": row.get("category", "Injury"),
                "report_timestamp": row["report_date"],
                "source": "NBA_Official",
                "season": season,
            }
        )

    # Check which records already exist (for counting)
    cursor = conn.cursor()
    existing_keys = set()

    if db_records:
        # Build list of unique keys to check
        keys_to_check = [
            (
                r["nba_player_id"],
                r["player_name"],
                r["report_timestamp"],
                r["source"],
                r["team"],
            )
            for r in db_records
        ]

        placeholders = ",".join(["(?,?,?,?,?)"] * len(keys_to_check))
        flat_params = [item for key in keys_to_check for item in key]

        cursor.execute(
            f"""
            SELECT nba_player_id, player_name, report_timestamp, source, team
            FROM InjuryReports
            WHERE (nba_player_id, player_name, report_timestamp, source, team) IN (VALUES {placeholders})
        """,
            flat_params,
        )

        existing_keys = set(cursor.fetchall())

    # Count added vs updated
    added_count = 0
    updated_count = 0

    for record in db_records:
        key = (
            record["nba_player_id"],
            record["player_name"],
            record["report_timestamp"],
            record["source"],
            record["team"],
        )
        if key in existing_keys:
            updated_count += 1
        else:
            added_count += 1

    # Use INSERT OR REPLACE to handle duplicates
    records_df = pd.DataFrame(db_records)

    # Pandas to_sql doesn't support OR REPLACE, so use executemany
    columns = list(records_df.columns)
    placeholders_str = ",".join(["?"] * len(columns))

    cursor.executemany(
        f"""
        INSERT OR REPLACE INTO InjuryReports ({','.join(columns)})
        VALUES ({placeholders_str})
        """,
        records_df.values.tolist(),
    )

    conn.commit()
    conn.close()

    return {"added": added_count, "updated": updated_count, "total": len(records_df)}


def update_nba_official_injuries(
    days_back: int = 1, season: str = None, db_path: str = DB_PATH, stage_logger=None
) -> dict:
    """
    Update NBA Official injury reports for recent days or entire season.

    This is meant to be called as part of the daily pipeline to fetch
    the latest injury report PDFs.

    Args:
        days_back: Number of days to look back (default 1 = yesterday + today)
        season: Season string (e.g., "2024-2025") for season-wide gap filling
        db_path: Path to database
        stage_logger: Optional StageLogger for tracking

    Returns:
        dict: {"added": int, "updated": int, "total": int}
    """
    from src.utils import determine_current_season, get_season_start_date

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # If season provided, fetch all missing dates in season
    if season:
        current_season = determine_current_season()

        # Determine season date range
        if season == current_season:
            # Current season: from actual season start to today
            season_start = get_season_start_date(season, db_path)
            season_end = today
        else:
            # Historical season: from actual season start to May 31 next year
            season_start = get_season_start_date(season, db_path)
            season_end_year = int(season.split("-")[1])
            season_end = datetime(season_end_year, 5, 31)

        # Generate all dates in season
        all_dates = []
        current = season_start
        while current <= season_end:
            all_dates.append(current)
            current += timedelta(days=1)
    else:
        # Generate dates to check (recent days mode)
        all_dates = [today - timedelta(days=i) for i in range(days_back + 1)]
        all_dates.reverse()  # Process oldest first

    conn = sqlite3.connect(db_path)

    # Filter dates using smart caching:
    # - Today: refetch if >2 hours old
    # - Past dates: permanent cache (only fetch if never fetched)
    dates = [dt for dt in all_dates if _should_fetch_injury_date(dt, db_path)]

    if len(dates) == 0:
        logging.debug(f"All {len(all_dates)} days already cached, nothing to fetch")
        conn.close()

        # Get total count for reporting
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM InjuryReports WHERE source='NBA_Official'"
            )
            total = cursor.fetchone()[0]

        return {"added": 0, "updated": 0, "total": total}

    logging.debug(
        f"Checking NBA Official injury reports for {len(dates)} days ({len(all_dates) - len(dates)} cached)..."
    )

    total_added = 0
    total_updated = 0
    api_calls = 0

    # Use tqdm for progress bar only if fetching more than 7 days
    iterator = (
        tqdm(dates, desc="Fetching injury reports", unit="day")
        if len(dates) > 7
        else dates
    )

    for dt in iterator:
        date_str = dt.strftime("%Y-%m-%d")

        # Fetch the report
        df = fetch_injury_report(dt)
        api_calls += 1

        if stage_logger:
            stage_logger.log_api_call()

        if not df.empty:
            counts = save_injury_records(df, db_path)
            total_added += counts["added"]
            total_updated += counts["updated"]

            logging.debug(
                f"NBA Official injuries for {date_str}: +{counts['added']} ~{counts['updated']}"
            )
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "saved", "records": counts["total"]})
        else:
            logging.debug(f"NBA Official injuries for {date_str}: no report available")
            if isinstance(iterator, tqdm):
                iterator.set_postfix({"status": "not found"})

        # Update cache timestamp for this date (whether we got data or not)
        _update_injury_cache(date_str, db_path)

        time.sleep(0.1)  # Be nice to NBA servers

    conn.close()

    # Get total count
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM InjuryReports WHERE source='NBA_Official'")
        total = cursor.fetchone()[0]

    logging.debug(
        f"NBA Official injury update complete: +{total_added} ~{total_updated} (total: {total})"
    )

    return {"added": total_added, "updated": total_updated, "total": total}


def backfill_injury_reports(
    start_date: str, end_date: str, db_path: str = DB_PATH, batch_size: int = 50
) -> int:
    """
    Backfill NBA Official injury reports for a date range.

    This is for historical data collection. Uses batch saving and progress bars
    for efficient processing of large date ranges.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database
        batch_size: Number of days to process before saving (default: 50)

    Returns:
        Number of records inserted
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError("start_date must be before end_date")

    # Generate all dates in range
    current_dt = start_dt
    dates = []
    while current_dt <= end_dt:
        dates.append(current_dt)
        current_dt += timedelta(days=1)

    logging.info(
        f"Backfilling NBA Official injury reports for {len(dates)} days ({start_date} to {end_date})..."
    )

    conn = sqlite3.connect(db_path)
    total_inserted = 0
    total_cached = 0
    total_not_found = 0

    # Batch collection for efficient saving
    batch_dfs = []

    # Progress bar for large backfills
    with tqdm(dates, desc="Backfilling injury reports", unit="day") as pbar:
        for dt in pbar:
            date_str = dt.strftime("%Y-%m-%d")

            # Check if we already have data for this date
            existing = pd.read_sql(
                "SELECT COUNT(*) as cnt FROM InjuryReports WHERE source = 'NBA_Official' AND report_timestamp = ?",
                conn,
                params=(date_str,),
            )["cnt"].iloc[0]

            if existing > 0:
                logging.debug(f"{date_str}: already have {existing} records")
                total_cached += 1
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "inserted": total_inserted,
                        "not_found": total_not_found,
                    }
                )
                continue

            # Fetch the report
            df = fetch_injury_report(dt)
            if not df.empty:
                batch_dfs.append(df)
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "batch": len(batch_dfs),
                        "pending": total_inserted,
                    }
                )

                # Save batch if it reaches batch_size
                if len(batch_dfs) >= batch_size:
                    combined_df = pd.concat(batch_dfs, ignore_index=True)
                    count = save_injury_records(combined_df, db_path)
                    logging.info(
                        f"Saved batch: {count} records from {len(batch_dfs)} days"
                    )
                    total_inserted += count
                    batch_dfs = []  # Clear batch
            else:
                logging.debug(f"{date_str}: no report available")
                total_not_found += 1
                pbar.set_postfix(
                    {
                        "cached": total_cached,
                        "inserted": total_inserted,
                        "not_found": total_not_found,
                    }
                )

            # Rate limiting - be respectful to NBA servers
            time.sleep(0.2)

    # Save any remaining records in the last batch
    if batch_dfs:
        combined_df = pd.concat(batch_dfs, ignore_index=True)
        count = save_injury_records(combined_df, db_path)
        logging.info(f"Saved final batch: {count} records from {len(batch_dfs)} days")
        total_inserted += count

    conn.close()

    logging.info(
        f"Backfill complete: {total_inserted} new records, {total_cached} cached, {total_not_found} not found ({len(dates)} days total)"
    )
    return total_inserted


def main():
    """CLI entry point for injury data collection."""
    import argparse

    from src.logging_config import setup_logging

    parser = argparse.ArgumentParser(
        description="Collect NBA Official injury reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update recent reports (default: yesterday + today)
  python -m src.database_updater.nba_official_injuries
  
  # Update last 7 days
  python -m src.database_updater.nba_official_injuries --days-back 7
  
  # Backfill historical range
  python -m src.database_updater.nba_official_injuries --backfill --start 2024-10-01 --end 2024-12-01
        """,
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Number of days to look back (default: 1)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical data (requires --start and --end)",
    )
    parser.add_argument(
        "--start", type=str, help="Start date for backfill (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, help="End date for backfill (YYYY-MM-DD)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()
    setup_logging(log_level=args.log_level.upper())

    if args.backfill:
        if not args.start or not args.end:
            parser.error("--backfill requires both --start and --end dates")

        count = backfill_injury_reports(args.start, args.end)
        print(f"✓ Backfill complete: {count} new records")
    else:
        count = update_nba_official_injuries(days_back=args.days_back)
        print(f"✓ Update complete: {count} new records")


if __name__ == "__main__":
    main()
