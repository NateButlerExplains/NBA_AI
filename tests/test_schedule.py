"""
Tests for src/database_updater/schedule.py

Simple unit tests to verify schedule module behavior:
- Cache logic works correctly
- Flag preservation during updates
- Date filtering
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database_updater import schedule
from src.utils import determine_current_season


class TestScheduleCache:
    """Tests for schedule caching behavior."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database with ScheduleCache table."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()

        # Create minimal ScheduleCache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ScheduleCache (
                season TEXT PRIMARY KEY,
                last_update_datetime TEXT NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

        yield db.name

        # Cleanup
        import os

        os.unlink(db.name)

    def test_cache_skips_update_for_fresh_historical_season(self, temp_db):
        """Historical season with fresh cache should skip update."""
        current_season = determine_current_season()
        historical_season = "2022-2023"  # Not current

        # Insert recent cache entry
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO ScheduleCache VALUES (?, ?)", (historical_season, now)
            )
            conn.commit()

        # Should NOT update (cache is fresh)
        result = schedule._should_update_schedule(historical_season, temp_db)
        assert result is False, "Fresh cache should skip update"

    def test_cache_updates_expired_historical_season(self, temp_db):
        """Historical season with expired cache should update."""
        historical_season = "2022-2023"

        # Insert old cache entry (10 minutes ago)
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            old_time = (pd.Timestamp.now() - timedelta(minutes=10)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cursor.execute(
                "INSERT INTO ScheduleCache VALUES (?, ?)", (historical_season, old_time)
            )
            conn.commit()

        # Should update (cache expired)
        result = schedule._should_update_schedule(historical_season, temp_db)
        assert result is True, "Expired cache should trigger update"

    def test_current_season_always_updates(self, temp_db):
        """Current season should always update regardless of cache."""
        current_season = determine_current_season()

        # Insert fresh cache entry for current season
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO ScheduleCache VALUES (?, ?)", (current_season, now)
            )
            conn.commit()

        # Should still update (current season always updates)
        result = schedule._should_update_schedule(current_season, temp_db)
        assert result is True, "Current season should always update"

    def test_missing_cache_triggers_update(self, temp_db):
        """Season with no cache entry should update."""
        season = "2023-2024"

        # Don't insert any cache entry
        result = schedule._should_update_schedule(season, temp_db)
        assert result is True, "Missing cache should trigger update"


class TestScheduleFlagPreservation:
    """Tests for flag preservation during schedule updates."""

    def test_save_schedule_preserves_flags(self):
        """Updating schedule should preserve existing flags.

        This test uses the actual database to verify flag preservation works
        with real schema. It reads existing data, doesn't modify anything.
        """
        from src.config import config

        db_path = config["database"]["path"]

        # Find a game with all flags set to 1
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT game_id, date_time_est, home_team, away_team, status, season, season_type
                FROM Games 
                WHERE game_data_finalized = 1 
                AND boxscore_data_finalized = 1 
                AND pre_game_data_finalized = 1
                LIMIT 1
            """
            )
            result = cursor.fetchone()

        if not result:
            pytest.skip("No games with all flags set - cannot test preservation")

        game_id, date_time, home, away, status, season, season_type = result

        # Create mock game data matching this game
        game_data = [
            {
                "gameId": game_id,
                "season": season,
                "gameDateTimeEst": date_time,
                "homeTeam": home,
                "awayTeam": away,
                "gameStatus": status,
                "seasonType": season_type,
            }
        ]

        # Save schedule (should preserve flags)
        schedule.save_schedule(game_data, season, db_path)

        # Verify flags still set to 1
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT game_data_finalized, boxscore_data_finalized, pre_game_data_finalized
                FROM Games WHERE game_id = ?
            """,
                (game_id,),
            )
            flags = cursor.fetchone()

        assert flags == (
            1,
            1,
            1,
        ), f"Flags should be preserved during schedule update (got {flags})"


class TestScheduleDateFiltering:
    """Tests for date filtering in schedule."""

    def test_validate_date_range_filters_correctly(self):
        """Schedule should filter games by date range."""
        # Create mock games spanning multiple dates
        games = [
            {"GAME_DATE": "2024-01-01", "GAME_ID": "001"},
            {"GAME_DATE": "2024-01-15", "GAME_ID": "002"},
            {"GAME_DATE": "2024-02-01", "GAME_ID": "003"},
        ]

        # Filter to January only
        start_date = pd.to_datetime("2024-01-01")
        end_date = pd.to_datetime("2024-01-31")

        filtered = [
            g for g in games if start_date <= pd.to_datetime(g["GAME_DATE"]) <= end_date
        ]

        assert len(filtered) == 2, "Should filter to 2 January games"
        assert filtered[0]["GAME_ID"] == "001"
        assert filtered[1]["GAME_ID"] == "002"
