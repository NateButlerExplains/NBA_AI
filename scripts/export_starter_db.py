#!/usr/bin/env python3
"""
Export a starter database with current season data only.

Creates a lightweight database (~500MB-1GB) that includes:
- Current season games, box scores, PBP, GameStates
- All players, teams, and player attributes (reference data)
- Current season betting lines and injury reports
- Current season features and predictions (Baseline only)

This allows new users to:
- Run the web app and see games
- Run the daily pipeline to collect new data
- Generate Baseline/Linear/Tree predictions (no models needed for Baseline)

It does NOT include:
- Historical seasons (needed for model training)
- Trained model checkpoints (kept private)
- L1 vectors, caches, or other derived data

Usage:
    python scripts/export_starter_db.py
    python scripts/export_starter_db.py --season 2025-2026 --output data/NBA_AI_starter.sqlite
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_starter_db(source_path: str, output_path: str, season: str):
    """Export a starter database with one season of data."""

    logger.info(f"Exporting starter DB from {source_path}")
    logger.info(f"Season: {season}")
    logger.info(f"Output: {output_path}")

    # Connect to source
    src = sqlite3.connect(source_path)
    src.execute("PRAGMA journal_mode=WAL")

    # Create fresh output
    out_path = Path(output_path)
    if out_path.exists():
        out_path.unlink()

    dst = sqlite3.connect(output_path)

    # Copy schema for all tables
    tables = src.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
    ).fetchall()

    for name, sql in tables:
        if sql:
            dst.execute(sql)
    dst.commit()

    # Tables with season filter (game-related)
    game_filtered = {
        "Games": f"SELECT * FROM Games WHERE season='{season}'",
        "PlayerBox": f"SELECT * FROM PlayerBox WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "TeamBox": f"SELECT * FROM TeamBox WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "GameStates": f"SELECT * FROM GameStates WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "PbP_Logs": f"SELECT * FROM PbP_Logs WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "Betting": f"SELECT * FROM Betting WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "Features": f"SELECT * FROM Features WHERE game_id IN (SELECT game_id FROM Games WHERE season='{season}')",
        "InjuryReports": f"SELECT * FROM InjuryReports WHERE season='{season}'",
        "InjuryCache": f"SELECT * FROM InjuryCache WHERE report_date >= '2025-10-01'",
    }

    # Tables copied in full (reference data)
    full_copy = [
        "Players", "Teams", "PlayerAttributes", "PlayersCache",
        "ScheduleCache", "CoversAttempts", "ESPNGameMapping",
    ]

    # Tables to skip (training data, not needed for starter)
    skip = [
        "DarkoDPM", "DataballrMultiYearRAPM", "DataballrPlayerMetrics",
        "DataballrSQRAPM", "DataballrWingspan", "RAPMArchive",
        "Coaches", "LineupStats", "TwoKRatings",
        "PBPPlayerGameStatsV2", "GameRotationRoster", "PipelineRuns",
        "Predictions",
    ]

    for table, query in game_filtered.items():
        try:
            rows = src.execute(query).fetchall()
            if rows:
                cols = [desc[0] for desc in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
                placeholders = ",".join(["?"] * len(cols))
                dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                dst.commit()
            logger.info(f"  {table}: {len(rows):,} rows")
        except Exception as e:
            logger.warning(f"  {table}: skipped ({e})")

    for table in full_copy:
        try:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                cols = [desc[0] for desc in src.execute(f"SELECT * FROM {table} LIMIT 1").description]
                placeholders = ",".join(["?"] * len(cols))
                dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                dst.commit()
            logger.info(f"  {table}: {len(rows):,} rows (full)")
        except Exception as e:
            logger.warning(f"  {table}: skipped ({e})")

    for table in skip:
        logger.info(f"  {table}: skipped (training data)")

    # Create indexes
    dst.execute("CREATE INDEX IF NOT EXISTS idx_playerbox_game ON PlayerBox(game_id)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_playerbox_player ON PlayerBox(player_id)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_gamestates_final ON GameStates(game_id, is_final_state)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_games_team_date ON Games(home_team, status, date_time_utc)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_games_away_date ON Games(away_team, status, date_time_utc)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_betting_lines ON Betting(lines_finalized)")
    dst.execute("CREATE INDEX IF NOT EXISTS idx_teambox_team ON TeamBox(team_id, game_id)")
    dst.commit()

    # Set WAL mode
    dst.execute("PRAGMA journal_mode=WAL")

    # VACUUM to compact
    logger.info("Compacting database...")
    dst.execute("VACUUM")

    src.close()
    dst.close()

    # Report size
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Done: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export starter database")
    parser.add_argument("--season", default="2025-2026", help="Season to export")
    parser.add_argument("--output", default="data/NBA_AI_starter.sqlite", help="Output path")
    args = parser.parse_args()

    export_starter_db(str(DB_PATH), args.output, args.season)


if __name__ == "__main__":
    main()
