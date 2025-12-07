#!/usr/bin/env python3
"""
Export release databases from NBA_AI_full.sqlite master database.

This script creates:
1. NBA_AI_current.sqlite - Only 2025-2026 season (for production)
2. NBA_AI_dev.sqlite - Three seasons: 2023-2024, 2024-2025, 2025-2026 (for development)

Both databases include all tables with proper schema.
"""

import sqlite3
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DB = DATA_DIR / "NBA_AI_full.sqlite"
CURRENT_DB = DATA_DIR / "NBA_AI_current.sqlite"
DEV_DB = DATA_DIR / "NBA_AI_dev.sqlite"

# Database configurations
DATABASES = {
    "NBA_AI_current.sqlite": {
        "path": CURRENT_DB,
        "seasons": ["2025-2026"],
        "description": "Production database (current season only)",
    },
    "NBA_AI_dev.sqlite": {
        "path": DEV_DB,
        "seasons": ["2023-2024", "2024-2025", "2025-2026"],
        "description": "Development database (3 seasons)",
    },
}

# Tables to copy
SEASON_TABLES = [
    "Games",
    "PbP_Logs",
    "GameStates",
    "PlayerBox",
    "TeamBox",
    "Features",
    "Predictions",
    "Betting",
    "InjuryReports",
]

REFERENCE_TABLES = [
    "Players",
    "Teams",
]

OTHER_TABLES = [
    "ESPNGameMapping",
    "ScheduleCache",
]


def get_table_schema(conn, table_name):
    """Get CREATE TABLE statement for a table."""
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    )
    result = cursor.fetchone()
    return result[0] if result else None


def copy_schema(source_conn, target_conn):
    """Copy all table schemas from source to target."""
    cursor = source_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        if table == "sqlite_sequence":
            continue

        schema = get_table_schema(source_conn, table)
        if schema:
            try:
                target_conn.execute(schema)
                print(f"  ✓ Created table: {table}")
            except Exception as e:
                print(f"  ✗ Failed to create table {table}: {e}")


def copy_season_data(source_conn, target_conn, table_name, seasons):
    """Copy data for specific seasons."""
    cursor = source_conn.cursor()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    column_list = ", ".join(columns)
    placeholders = ", ".join(["?" for _ in columns])

    # Build season filter
    season_filter = ", ".join([f"'{s}'" for s in seasons])

    # Special handling for tables without game_id column
    if table_name == "InjuryReports":
        # InjuryReports has season column (computed from report_timestamp)
        query = f"""
            SELECT * FROM {table_name}
            WHERE season IN ({season_filter})
        """
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows:
                target_conn.executemany(
                    f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})",
                    rows,
                )
                return len(rows)
            return 0
        except Exception as e:
            print(f"    Error copying {table_name}: {e}")
            return 0

    elif table_name == "ESPNGameMapping":
        # ESPNGameMapping uses nba_game_id instead of game_id
        query = f"""
            SELECT {table_name}.*
            FROM {table_name}
            JOIN Games ON {table_name}.nba_game_id = Games.game_id
            WHERE Games.season IN ({season_filter})
        """
    else:
        # Standard tables with game_id
        query = f"""
            SELECT {table_name}.*
            FROM {table_name}
            JOIN Games ON {table_name}.game_id = Games.game_id
            WHERE Games.season IN ({season_filter})
        """

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        if rows:
            target_conn.executemany(
                f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})",
                rows,
            )
            return len(rows)
        return 0
    except Exception as e:
        print(f"    Error copying {table_name}: {e}")
        return 0


def copy_games_data(source_conn, target_conn, seasons):
    """Copy Games table for specific seasons."""
    cursor = source_conn.cursor()

    # Get column names
    cursor.execute("PRAGMA table_info(Games)")
    columns = [row[1] for row in cursor.fetchall()]
    column_list = ", ".join(columns)
    placeholders = ", ".join(["?" for _ in columns])

    # Build season filter
    season_filter = ", ".join([f"'{s}'" for s in seasons])

    query = f"SELECT * FROM Games WHERE season IN ({season_filter})"

    cursor.execute(query)
    rows = cursor.fetchall()

    if rows:
        target_conn.executemany(
            f"INSERT INTO Games ({column_list}) VALUES ({placeholders})", rows
        )
        return len(rows)
    return 0


def copy_reference_data(source_conn, target_conn, table_name):
    """Copy entire reference tables (Players, Teams)."""
    cursor = source_conn.cursor()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    column_list = ", ".join(columns)
    placeholders = ", ".join(["?" for _ in columns])

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    if rows:
        target_conn.executemany(
            f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})", rows
        )
        return len(rows)
    return 0


def export_database(db_config):
    """Export a database with specific seasons."""
    db_path = db_config["path"]
    seasons = db_config["seasons"]
    description = db_config["description"]

    print(f"\n{'='*60}")
    print(f"Exporting: {db_path.name}")
    print(f"Description: {description}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"{'='*60}\n")

    # Remove existing database
    if db_path.exists():
        print(f"Removing existing database...")
        db_path.unlink()

    # Connect to databases
    source_conn = sqlite3.connect(SOURCE_DB)
    target_conn = sqlite3.connect(db_path)

    try:
        # Step 1: Copy schema
        print("Step 1: Copying schema...")
        copy_schema(source_conn, target_conn)

        # Step 2: Copy Games table
        print("\nStep 2: Copying Games table...")
        game_count = copy_games_data(source_conn, target_conn, seasons)
        print(f"  ✓ Copied {game_count:,} games")

        # Step 3: Copy season-specific tables
        print("\nStep 3: Copying season-specific tables...")
        for table in SEASON_TABLES:
            if table == "Games":
                continue  # Already copied

            # Check if table exists in source
            cursor = source_conn.cursor()
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            if not cursor.fetchone():
                print(f"  ⊘ Skipping {table} (doesn't exist in source)")
                continue

            count = copy_season_data(source_conn, target_conn, table, seasons)
            print(f"  ✓ Copied {count:,} rows from {table}")

        # Step 4: Copy reference tables
        print("\nStep 4: Copying reference tables...")
        for table in REFERENCE_TABLES:
            cursor = source_conn.cursor()
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            if not cursor.fetchone():
                print(f"  ⊘ Skipping {table} (doesn't exist in source)")
                continue

            count = copy_reference_data(source_conn, target_conn, table)
            print(f"  ✓ Copied {count:,} rows from {table}")

        # Step 5: Copy other tables (ESPNGameMapping, ScheduleCache)
        print("\nStep 5: Copying other tables...")
        for table in OTHER_TABLES:
            cursor = source_conn.cursor()
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
            )
            if not cursor.fetchone():
                print(f"  ⊘ Skipping {table} (doesn't exist in source)")
                continue

            try:
                if table == "ESPNGameMapping":
                    # ESPNGameMapping has nba_game_id, not game_id
                    count = copy_season_data(source_conn, target_conn, table, seasons)
                else:
                    # For ScheduleCache, copy everything
                    count = copy_reference_data(source_conn, target_conn, table)
                print(f"  ✓ Copied {count:,} rows from {table}")
            except Exception as e:
                print(f"  ⊘ Error with {table}: {e}")

        # Commit changes
        target_conn.commit()

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        cursor = target_conn.cursor()
        cursor.execute(
            "SELECT season, COUNT(*) FROM Games GROUP BY season ORDER BY season"
        )
        for season, count in cursor.fetchall():
            print(f"  {season}: {count:,} games")

        # Get database size
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"\nDatabase size: {size_mb:.1f} MB")
        print(f"Location: {db_path}")
        print("\n✅ Export completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        source_conn.close()
        target_conn.close()


def main():
    """Main export function."""
    if not SOURCE_DB.exists():
        print(f"❌ Source database not found: {SOURCE_DB}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("NBA AI - Release Database Export")
    print("=" * 60)
    print(f"\nSource: {SOURCE_DB}")
    print(f"Size: {SOURCE_DB.stat().st_size / (1024**3):.1f} GB")

    # Export both databases
    for db_name, db_config in DATABASES.items():
        export_database(db_config)

    print("\n" + "=" * 60)
    print("ALL EXPORTS COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run database evaluator on both databases")
    print("2. Run test suite to verify")
    print("3. Compress for release:")
    print(f"   cd data && zip -9 NBA_AI_current.zip NBA_AI_current.sqlite")
    print(f"   cd data && zip -9 NBA_AI_dev.zip NBA_AI_dev.sqlite")
    print()


if __name__ == "__main__":
    main()
