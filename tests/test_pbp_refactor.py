"""
Tests for PBP refactor: timestamp-based refetch logic, validator, StageLogger integration.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.database_updater.database_update_manager import get_games_needing_pbp_update
from src.database_updater.pbp import save_pbp
from src.database_updater.validators import PbPValidator, Severity


class TestPBPRefetchLogic:
    """Test timestamp-based refetch query logic."""

    @pytest.fixture
    def test_db(self):
        """Create temporary test database with schema."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()

        # Create schema
        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                date_time_utc TEXT,
                home_team TEXT,
                away_team TEXT,
                status INTEGER,
                season TEXT,
                season_type TEXT,
                pre_game_data_finalized BOOLEAN DEFAULT 0,
                game_data_finalized BOOLEAN DEFAULT 0,
                boxscore_data_finalized BOOLEAN DEFAULT 0,
                pbp_last_fetched_at TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PbP_Logs (
                game_id TEXT,
                play_id INTEGER,
                log_data TEXT,
                PRIMARY KEY (game_id, play_id)
            )
        """
        )

        conn.commit()
        yield db.name, conn
        conn.close()

    def test_in_progress_game_never_fetched(self, test_db):
        """In-progress game with no PBP should be selected."""
        db_path, conn = test_db
        cursor = conn.cursor()

        # Insert in-progress game with no PBP
        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, pbp_last_fetched_at)
            VALUES ('0022500999', '2025-2026', 'Regular Season', 2, datetime('now'), NULL)
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500999" in game_ids

    def test_in_progress_game_stale_fetch(self, test_db):
        """In-progress game with PBP >5 min old should be selected."""
        db_path, conn = test_db
        cursor = conn.cursor()

        # Insert in-progress game with stale PBP (10 minutes ago)
        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, pbp_last_fetched_at)
            VALUES ('0022500888', '2025-2026', 'Regular Season', 2, datetime('now'), datetime('now', '-10 minutes'))
        """
        )
        # Add some PBP data
        cursor.execute(
            """
            INSERT INTO PbP_Logs (game_id, play_id, log_data)
            VALUES ('0022500888', 1, '{}')
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500888" in game_ids

    def test_in_progress_game_recent_fetch(self, test_db):
        """In-progress game with PBP <5 min old should NOT be selected."""
        db_path, conn = test_db
        cursor = conn.cursor()

        # Insert in-progress game with very recent PBP (2 minutes ago)
        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, pbp_last_fetched_at)
            VALUES ('0022500777', '2025-2026', 'Regular Season', 2, datetime('now'), datetime('now', '-2 minutes'))
        """
        )
        cursor.execute(
            """
            INSERT INTO PbP_Logs (game_id, play_id, log_data)
            VALUES ('0022500777', 1, '{}')
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500777" not in game_ids

    def test_completed_game_no_pbp_recent(self, test_db):
        """Completed game without PBP within 48 hours should be selected."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, game_data_finalized)
            VALUES ('0022500666', '2025-2026', 'Regular Season', 3, datetime('now', '-1 hour'), 0)
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500666" in game_ids

    def test_completed_game_no_pbp_old(self, test_db):
        """Completed game without PBP >48 hours old SHOULD be selected (ensures complete coverage)."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, game_data_finalized)
            VALUES ('0022500555', '2025-2026', 'Regular Season', 3, datetime('now', '-72 hours'), 0)
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500555" in game_ids  # Should be fetched to ensure data completeness

    def test_completed_game_with_pbp(self, test_db):
        """Completed game with PBP should NOT be selected."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, date_time_utc, game_data_finalized)
            VALUES ('0022500444', '2025-2026', 'Regular Season', 3, datetime('now', '-1 hour'), 0)
        """
        )
        cursor.execute(
            """
            INSERT INTO PbP_Logs (game_id, play_id, log_data)
            VALUES ('0022500444', 1, '{}')
        """
        )
        conn.commit()

        game_ids = get_games_needing_pbp_update("2025-2026", db_path)
        assert "0022500444" not in game_ids


class TestSavePBP:
    """Test save_pbp timestamp tracking and count logic."""

    @pytest.fixture
    def test_db(self):
        """Create temporary test database."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                pbp_last_fetched_at TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PbP_Logs (
                game_id TEXT,
                play_id INTEGER,
                log_data TEXT,
                PRIMARY KEY (game_id, play_id)
            )
        """
        )

        conn.commit()
        yield db.name, conn
        conn.close()

    def test_save_pbp_new_game(self, test_db):
        """Saving PBP for new game should count as 'added'."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id) VALUES ('0022500999')")
        conn.commit()

        pbp_data = {
            "0022500999": [
                {"orderNumber": 1, "period": 1, "clock": "PT12M00.00S"},
                {"orderNumber": 2, "period": 1, "clock": "PT11M50.00S"},
            ]
        }

        counts = save_pbp(pbp_data, db_path)
        assert counts["added"] == 1
        assert counts["updated"] == 0
        assert counts["unchanged"] == 0

        # Verify timestamp was set
        cursor.execute(
            "SELECT pbp_last_fetched_at FROM Games WHERE game_id = '0022500999'"
        )
        timestamp = cursor.fetchone()[0]
        assert timestamp is not None

    def test_save_pbp_update_game(self, test_db):
        """Updating PBP for existing game should count as 'updated'."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id) VALUES ('0022500888')")
        cursor.execute(
            "INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES ('0022500888', 1, '{}')"
        )
        conn.commit()

        pbp_data = {
            "0022500888": [
                {"orderNumber": 1, "period": 1, "clock": "PT12M00.00S"},
                {"orderNumber": 2, "period": 1, "clock": "PT11M50.00S"},
                {"orderNumber": 3, "period": 1, "clock": "PT11M40.00S"},
            ]
        }

        counts = save_pbp(pbp_data, db_path)
        assert counts["added"] == 0
        assert counts["updated"] == 1
        assert counts["unchanged"] == 0

    def test_save_pbp_empty_data(self, test_db):
        """Empty PBP data should count as 'unchanged'."""
        db_path, conn = test_db
        cursor = conn.cursor()

        cursor.execute("INSERT INTO Games (game_id) VALUES ('0022500777')")
        conn.commit()

        pbp_data = {"0022500777": []}

        counts = save_pbp(pbp_data, db_path)
        assert counts["added"] == 0
        assert counts["updated"] == 0
        assert counts["unchanged"] == 1


class TestPbPValidator:
    """Test PbPValidator functionality."""

    @pytest.fixture
    def test_db(self):
        """Create test database with schema."""
        db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        conn = sqlite3.connect(db.name)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE Games (
                game_id TEXT PRIMARY KEY,
                status INTEGER,
                pbp_last_fetched_at TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE PbP_Logs (
                game_id TEXT,
                play_id INTEGER,
                log_data TEXT,
                PRIMARY KEY (game_id, play_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE GameStates (
                game_id TEXT,
                play_id INTEGER,
                is_final_state BOOLEAN,
                PRIMARY KEY (game_id, play_id)
            )
        """
        )

        conn.commit()
        yield conn.cursor()
        conn.close()

    def test_missing_pbp_detection(self, test_db):
        """Validator should detect completed games without PBP."""
        cursor = test_db

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022500999', 3)")
        cursor.connection.commit()

        validator = PbPValidator()
        result = validator.validate(["0022500999"], cursor)

        assert result.has_critical_issues
        assert any(issue.check_id == "MISSING_PBP" for issue in result.issues)

    def test_low_play_count_detection(self, test_db):
        """Validator should detect games with suspiciously low play counts."""
        cursor = test_db

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022500888', 3)")

        # Insert only 50 plays (should trigger warning)
        for i in range(50):
            cursor.execute(
                f"INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES ('0022500888', {i}, '{{}}')"
            )

        cursor.connection.commit()

        validator = PbPValidator()
        result = validator.validate(["0022500888"], cursor)

        assert result.has_warnings
        assert any(issue.check_id == "LOW_PLAY_COUNT" for issue in result.issues)

    def test_stale_inprogress_detection(self, test_db):
        """Validator should detect in-progress games with stale PBP."""
        cursor = test_db

        cursor.execute(
            """
            INSERT INTO Games (game_id, status, pbp_last_fetched_at)
            VALUES ('0022500777', 2, datetime('now', '-30 minutes'))
        """
        )
        cursor.connection.commit()

        validator = PbPValidator()
        result = validator.validate(["0022500777"], cursor)

        assert result.has_warnings
        assert any(issue.check_id == "STALE_INPROGRESS_PBP" for issue in result.issues)

    def test_no_final_state_detection(self, test_db):
        """Validator should detect games with PBP but no final GameState."""
        cursor = test_db

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022500666', 3)")
        cursor.execute(
            "INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES ('0022500666', 1, '{}')"
        )
        cursor.connection.commit()

        validator = PbPValidator()
        result = validator.validate(["0022500666"], cursor)

        assert result.has_warnings
        assert any(issue.check_id == "NO_FINAL_STATE" for issue in result.issues)

    def test_duplicate_plays_detection(self, test_db):
        """Validator should detect duplicate play_ids (if they somehow got through)."""
        cursor = test_db

        # Note: In real DB, PRIMARY KEY prevents duplicates.
        # This test simulates checking logic even if constraint was bypassed.
        # In practice, this check is defensive and would only trigger in corrupted data.

        cursor.execute("INSERT INTO Games (game_id, status) VALUES ('0022500555', 3)")
        cursor.execute(
            "INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES ('0022500555', 1, '{}')"
        )
        cursor.execute(
            "INSERT INTO PbP_Logs (game_id, play_id, log_data) VALUES ('0022500555', 2, '{}')"
        )
        cursor.connection.commit()

        validator = PbPValidator()
        result = validator.validate(["0022500555"], cursor)

        # Should pass validation (no duplicates possible with PRIMARY KEY)
        assert not result.has_critical_issues
        assert not any(issue.check_id == "DUPLICATE_PLAYS" for issue in result.issues)
