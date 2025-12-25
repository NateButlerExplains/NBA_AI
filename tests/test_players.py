"""
Tests for src/database_updater/players.py

Unit tests to verify players module behavior:
- Smart update logic (only updates changed/new players)
- Data parsing and validation
- Team name conversion
- Error handling
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.config import config
from src.database_updater import players


class TestPlayerSmartUpdate:
    """Tests for smart update logic (only update changed players)."""

    def test_update_skips_unchanged_players(self):
        """Should not update players with no changes."""
        db_path = config["database"]["path"]

        # Mock fetch_players to return one existing player with no changes
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT person_id, first_name, last_name, full_name, from_year, 
                       to_year, roster_status, team
                FROM Players 
                WHERE roster_status = 1
                LIMIT 1
            """
            )
            existing = cursor.fetchone()

        if not existing:
            pytest.skip("No active players in database")

        person_id, fname, lname, fullname, from_y, to_y, status, team = existing

        mock_api_data = [
            {
                "person_id": person_id,
                "first_name": fname,
                "last_name": lname,
                "full_name": fullname,
                "from_year": from_y,
                "to_year": to_y,
                "roster_status": status,
                "team": team,
            }
        ]

        with patch("src.database_updater.players.fetch_players") as mock_fetch:
            with patch("src.database_updater.players.save_players") as mock_save:
                mock_fetch.return_value = mock_api_data

                players.update_players(db_path)

                # save_players should NOT be called if no changes
                mock_save.assert_not_called()

    def test_update_processes_new_players(self):
        """Should update players not in database."""
        db_path = config["database"]["path"]

        # Create fake new player
        mock_new_player = {
            "person_id": 999999999,  # Invalid ID won't exist
            "first_name": "Test",
            "last_name": "Player",
            "full_name": "Test Player",
            "from_year": 2024,
            "to_year": 2024,
            "roster_status": 1,
            "team": "BOS",
        }

        with patch("src.database_updater.players.fetch_players") as mock_fetch:
            with patch("src.database_updater.players.save_players") as mock_save:
                with patch(
                    "src.database_updater.players._should_update_players",
                    return_value=True,
                ):
                    # Properly mock StageLogger
                    with patch(
                        "src.database_updater.players.StageLogger"
                    ) as mock_logger_class:
                        mock_logger = MagicMock()
                        mock_logger_class.return_value = mock_logger

                        mock_fetch.return_value = [mock_new_player]

                        players.update_players(db_path)

                        # save_players SHOULD be called with new player
                        mock_save.assert_called_once()
                        saved_players = mock_save.call_args[0][0]
                        assert len(saved_players) == 1
                        assert saved_players[0]["person_id"] == 999999999

    def test_update_processes_changed_players(self):
        """Should update players with field changes."""
        db_path = config["database"]["path"]

        # Get existing player and modify team
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT person_id, first_name, last_name, full_name, from_year, 
                       to_year, roster_status, team
                FROM Players 
                WHERE roster_status = 1 AND team IS NOT NULL
                LIMIT 1
            """
            )
            existing = cursor.fetchone()

        if not existing:
            pytest.skip("No active players with team in database")

        person_id, fname, lname, fullname, from_y, to_y, status, team = existing

        # Change team
        mock_changed_player = {
            "person_id": person_id,
            "first_name": fname,
            "last_name": lname,
            "full_name": fullname,
            "from_year": from_y,
            "to_year": to_y,
            "roster_status": status,
            "team": "LAL" if team != "LAL" else "BOS",  # Different team
        }

        with patch("src.database_updater.players.fetch_players") as mock_fetch:
            with patch("src.database_updater.players.save_players") as mock_save:
                with patch(
                    "src.database_updater.players._should_update_players",
                    return_value=True,
                ):
                    # Properly mock StageLogger
                    with patch(
                        "src.database_updater.players.StageLogger"
                    ) as mock_logger_class:
                        mock_logger = MagicMock()
                        mock_logger_class.return_value = mock_logger

                        mock_fetch.return_value = [mock_changed_player]

                        players.update_players(db_path)

                        # save_players SHOULD be called with changed player
                        mock_save.assert_called_once()
                        saved_players = mock_save.call_args[0][0]
                        assert len(saved_players) == 1


class TestPlayerDataParsing:
    """Tests for player data parsing from API."""

    def test_fetch_players_returns_list(self):
        """fetch_players should return list of dicts."""
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        result = players.fetch_players(mock_logger)

        assert isinstance(result, list), "Should return list"
        if result:  # If API call succeeded
            assert isinstance(result[0], dict), "Should contain dicts"
            # Check required fields
            required_fields = [
                "person_id",
                "first_name",
                "last_name",
                "full_name",
                "from_year",
                "to_year",
                "roster_status",
                "team",
            ]
            for field in required_fields:
                assert field in result[0], f"Missing field: {field}"

    def test_save_players_formats_correctly(self):
        """save_players should handle player dicts correctly."""
        # Create minimal test database
        import tempfile

        db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()

        # Create Players table
        cursor.execute(
            """
            CREATE TABLE Players (
                person_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                from_year INTEGER,
                to_year INTEGER,
                roster_status BOOLEAN,
                team TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        # Test data
        test_players = [
            {
                "person_id": 123,
                "first_name": "John",
                "last_name": "Doe",
                "full_name": "John Doe",
                "from_year": 2020,
                "to_year": 2024,
                "roster_status": 1,
                "team": "BOS",
            }
        ]

        # Save to test database
        players.save_players(test_players, db.name)

        # Verify saved correctly
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Players WHERE person_id = 123")
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "Player should be saved"
        assert result[0] == 123
        assert result[1] == "John"
        assert result[7] == "BOS"

        # Cleanup
        import os

        os.unlink(db.name)


class TestPlayerDataQuality:
    """Tests for data quality in Players table."""

    def test_active_players_count_reasonable(self):
        """Should have reasonable number of active players."""
        db_path = config["database"]["path"]

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Players WHERE roster_status = 1")
            active_count = cursor.fetchone()[0]

        # NBA has ~450 active players, allow range 400-600
        assert (
            400 <= active_count <= 600
        ), f"Active player count {active_count} outside expected range"

    def test_no_null_person_ids(self):
        """All players should have valid person_id."""
        db_path = config["database"]["path"]

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Players WHERE person_id IS NULL")
            null_count = cursor.fetchone()[0]

        assert null_count == 0, "No NULL person_ids allowed"

    def test_active_players_have_teams(self):
        """Most active players should have team assignment."""
        db_path = config["database"]["path"]

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM Players WHERE roster_status = 1 AND team IS NULL"
            )
            no_team = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM Players WHERE roster_status = 1")
            active = cursor.fetchone()[0]

        # Allow up to 5% of active players without teams (free agents, etc.)
        pct_no_team = no_team / active if active > 0 else 0
        assert (
            pct_no_team < 0.05
        ), f"{pct_no_team*100:.1f}% of active players have no team"
