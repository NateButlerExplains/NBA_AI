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

from src.config import config

DB_PATH = config["database"]["path"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PDF_URL_TEMPLATE = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_05PM.pdf"
)


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

            if category == "Injury":
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


def save_injury_records(df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Save injury records to database with player ID matching."""
    if df.empty:
        return 0

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
                "report_timestamp": row["report_date"],
                "source": "NBA_Official",
            }
        )

    records_df = pd.DataFrame(db_records)
    records_df.to_sql("InjuryReports", conn, if_exists="append", index=False)

    conn.close()
    return len(records_df)


def update_nba_official_injuries(days_back: int = 1, db_path: str = DB_PATH) -> int:
    """
    Update NBA Official injury reports for recent days.

    This is meant to be called as part of the daily pipeline to fetch
    the latest injury report PDFs.

    Args:
        days_back: Number of days to look back (default 1 = yesterday + today)
        db_path: Path to database

    Returns:
        Number of records inserted
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate dates to check
    dates = [today - timedelta(days=i) for i in range(days_back + 1)]
    dates.reverse()  # Process oldest first

    logging.info(f"Checking NBA Official injury reports for {len(dates)} days...")

    conn = sqlite3.connect(db_path)

    total_inserted = 0
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")

        # Check if we already have data for this date
        existing = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM InjuryReports WHERE source = 'NBA_Official' AND report_timestamp = ?",
            conn,
            params=(date_str,),
        )["cnt"].iloc[0]

        if existing > 0:
            logging.debug(
                f"NBA Official injuries for {date_str}: already have {existing} records"
            )
            continue

        # Fetch the report
        df = fetch_injury_report(dt)
        if not df.empty:
            count = save_injury_records(df, db_path)
            logging.info(
                f"NBA Official injuries for {date_str}: inserted {count} records"
            )
            total_inserted += count
        else:
            logging.debug(f"NBA Official injuries for {date_str}: no report available")

        time.sleep(0.1)  # Be nice to NBA servers

    conn.close()

    if total_inserted > 0:
        logging.info(
            f"NBA Official injury update complete: {total_inserted} new records"
        )
    else:
        logging.debug("NBA Official injury update: no new records")

    return total_inserted
