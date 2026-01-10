"""
test_health_check.py

Tests for the health check CLI.
"""

import json
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.health_check import CheckResult, CheckStatus, HealthReport, SeasonHealthChecker

# =============================================================================
# CheckResult Tests
# =============================================================================


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        result = CheckResult(
            stage="Games",
            category="completeness",
            check_name="game_count",
            status=CheckStatus.PASS,
            message="Found 1320 games",
            expected=1230,
            actual=1320,
        )
        d = result.to_dict()

        assert d["stage"] == "Games"
        assert d["category"] == "completeness"
        assert d["check_name"] == "game_count"
        assert d["status"] == "pass"
        assert d["message"] == "Found 1320 games"
        assert d["expected"] == 1230
        assert d["actual"] == 1320

    def test_to_dict_with_details(self):
        """Test to_dict with details."""
        result = CheckResult(
            stage="Games",
            category="integrity",
            check_name="no_duplicates",
            status=CheckStatus.CRITICAL,
            message="Found duplicates",
            details={"duplicates": ["123", "456"]},
        )
        d = result.to_dict()

        assert d["details"] == {"duplicates": ["123", "456"]}

    def test_status_values(self):
        """Test all status enum values."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.CRITICAL.value == "critical"
        assert CheckStatus.SKIP.value == "skip"


# =============================================================================
# HealthReport Tests
# =============================================================================


class TestHealthReport:
    """Tests for HealthReport class."""

    def test_empty_report(self):
        """Test empty report defaults."""
        report = HealthReport(season="2024-2025")

        assert report.season == "2024-2025"
        assert report.passed == 0
        assert report.warnings == 0
        assert report.critical == 0
        assert report.skipped == 0
        assert report.exit_code == 0

    def test_add_results(self):
        """Test adding results."""
        report = HealthReport(season="2024-2025")
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="game_count",
                status=CheckStatus.PASS,
                message="OK",
            )
        )
        report.add(
            CheckResult(
                stage="PbP",
                category="completeness",
                check_name="coverage",
                status=CheckStatus.WARN,
                message="Low coverage",
            )
        )

        assert len(report.results) == 2
        assert report.passed == 1
        assert report.warnings == 1

    def test_exit_code_all_pass(self):
        """Test exit code 0 when all pass."""
        report = HealthReport(season="2024-2025")
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="test",
                status=CheckStatus.PASS,
                message="OK",
            )
        )
        report.add(
            CheckResult(
                stage="Games",
                category="integrity",
                check_name="test2",
                status=CheckStatus.PASS,
                message="OK",
            )
        )

        assert report.exit_code == 0

    def test_exit_code_with_warnings(self):
        """Test exit code 1 with warnings."""
        report = HealthReport(season="2024-2025")
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="test",
                status=CheckStatus.PASS,
                message="OK",
            )
        )
        report.add(
            CheckResult(
                stage="Games",
                category="integrity",
                check_name="test2",
                status=CheckStatus.WARN,
                message="Warning",
            )
        )

        assert report.exit_code == 1

    def test_exit_code_with_critical(self):
        """Test exit code 2 with critical."""
        report = HealthReport(season="2024-2025")
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="test",
                status=CheckStatus.CRITICAL,
                message="Failed",
            )
        )

        assert report.exit_code == 2

    def test_exit_code_critical_over_warning(self):
        """Test critical takes precedence over warning."""
        report = HealthReport(season="2024-2025")
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="test",
                status=CheckStatus.WARN,
                message="Warning",
            )
        )
        report.add(
            CheckResult(
                stage="Games",
                category="integrity",
                check_name="test2",
                status=CheckStatus.CRITICAL,
                message="Critical",
            )
        )

        assert report.exit_code == 2

    def test_summary_table_format(self):
        """Test summary table contains expected elements."""
        report = HealthReport(season="2024-2025")
        report.start_time = datetime(2024, 1, 1, 12, 0, 0)
        report.end_time = datetime(2024, 1, 1, 12, 0, 5)
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="game_count",
                status=CheckStatus.PASS,
                message="OK",
            )
        )

        table = report.summary_table()

        assert "HEALTH CHECK REPORT: 2024-2025" in table
        assert "Games" in table
        assert "[completeness]" in table
        assert "game_count" in table
        assert "SUMMARY:" in table
        assert "1 passed" in table
        assert "Duration: 5.0s" in table

    def test_to_json_format(self):
        """Test JSON output format."""
        report = HealthReport(season="2024-2025")
        report.start_time = datetime(2024, 1, 1, 12, 0, 0)
        report.end_time = datetime(2024, 1, 1, 12, 0, 5)
        report.pipeline_ran = True
        report.add(
            CheckResult(
                stage="Games",
                category="completeness",
                check_name="game_count",
                status=CheckStatus.PASS,
                message="OK",
                expected=1230,
                actual=1230,
            )
        )

        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["season"] == "2024-2025"
        assert data["pipeline_ran"] is True
        assert data["summary"]["passed"] == 1
        assert data["summary"]["exit_code"] == 0
        assert len(data["results"]) == 1
        assert data["results"][0]["stage"] == "Games"


# =============================================================================
# SeasonHealthChecker Tests
# =============================================================================


class TestSeasonHealthChecker:
    """Tests for SeasonHealthChecker class."""

    def test_init(self):
        """Test checker initialization."""
        checker = SeasonHealthChecker(season="2024-2025", db_path=":memory:")

        assert checker.season == "2024-2025"
        assert checker.db_path == ":memory:"
        assert checker.report.season == "2024-2025"

    def test_shortened_season_constants(self):
        """Test shortened season game counts are defined."""
        assert SeasonHealthChecker.SHORTENED_SEASONS["2011-2012"] == 990
        assert SeasonHealthChecker.SHORTENED_SEASONS["2019-2020"] == 1059
        assert SeasonHealthChecker.SHORTENED_SEASONS["2020-2021"] == 1080

    def test_data_availability_constants(self):
        """Test data availability season thresholds."""
        assert SeasonHealthChecker.PLAYERBOX_START_SEASON == "2023-2024"
        assert SeasonHealthChecker.BETTING_START_SEASON == "2007-2008"

    def test_per_game_constants(self):
        """Test per-game expected record counts."""
        assert SeasonHealthChecker.PBP_MIN_PLAYS == 200
        assert SeasonHealthChecker.PBP_MAX_PLAYS == 800
        assert SeasonHealthChecker.TEAMBOX_EXPECTED == 2
        assert SeasonHealthChecker.PLAYERBOX_MIN == 22
        assert SeasonHealthChecker.PLAYERBOX_MAX == 36


# =============================================================================
# Integration Tests (require database)
# =============================================================================


@pytest.fixture
def test_db(tmp_path):
    """Create a minimal test database."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create minimal schema
    cursor.execute(
        """
        CREATE TABLE Games (
            game_id TEXT PRIMARY KEY,
            season TEXT,
            season_type TEXT,
            status INTEGER,
            date_time_utc TEXT,
            game_data_finalized INTEGER DEFAULT 0,
            boxscore_data_finalized INTEGER DEFAULT 0,
            pre_game_data_finalized INTEGER DEFAULT 0
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE PbP_Logs (
            game_id TEXT,
            action_number INTEGER,
            log_data TEXT,
            PRIMARY KEY (game_id, action_number)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE GameStates (
            game_id TEXT,
            action_number INTEGER,
            is_final_state INTEGER DEFAULT 0,
            PRIMARY KEY (game_id, action_number)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE PlayerBox (
            game_id TEXT,
            person_id INTEGER,
            PRIMARY KEY (game_id, person_id)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE TeamBox (
            game_id TEXT,
            team_id INTEGER,
            PRIMARY KEY (game_id, team_id)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE Features (
            game_id TEXT PRIMARY KEY,
            feature_set TEXT,
            save_datetime TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE Predictions (
            game_id TEXT,
            predictor TEXT,
            prediction_datetime TEXT,
            prediction_set TEXT,
            PRIMARY KEY (game_id, predictor)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE Betting (
            game_id TEXT PRIMARY KEY,
            lines_finalized INTEGER DEFAULT 0
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE Players (
            person_id INTEGER PRIMARY KEY,
            full_name TEXT
        )
    """
    )

    conn.commit()
    conn.close()

    return db_path


class TestSeasonHealthCheckerIntegration:
    """Integration tests with actual database."""

    def test_empty_database(self, test_db):
        """Test checks on empty database."""
        checker = SeasonHealthChecker(season="2024-2025", db_path=test_db)
        report = checker.run_all()

        # Should have critical for no games
        assert report.critical > 0

    def test_games_check_pass(self, test_db):
        """Test games check passes with data."""
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        # Insert test games
        for i in range(1230):
            cursor.execute(
                """
                INSERT INTO Games (game_id, season, season_type, status, date_time_utc)
                VALUES (?, '2024-2025', 'Regular Season', 3, '2024-10-22T19:30:00Z')
            """,
                (f"002420{i:04d}",),
            )

        conn.commit()
        conn.close()

        checker = SeasonHealthChecker(season="2024-2025", db_path=test_db)
        checker._check_games()

        # Find the game_count result
        game_count_result = next(
            (r for r in checker.report.results if r.check_name == "game_count"), None
        )
        assert game_count_result is not None
        assert game_count_result.status == CheckStatus.PASS

    def test_flag_consistency_critical(self, test_db):
        """Test flag consistency detects issues."""
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        # Insert a game with game_data_finalized=1 but no PbP
        cursor.execute(
            """
            INSERT INTO Games (game_id, season, season_type, status, game_data_finalized)
            VALUES ('0024200001', '2024-2025', 'Regular Season', 3, 1)
        """
        )

        conn.commit()
        conn.close()

        checker = SeasonHealthChecker(season="2024-2025", db_path=test_db)
        checker._check_flag_consistency()

        # Find the game_data_has_pbp result
        pbp_flag_result = next(
            (r for r in checker.report.results if r.check_name == "game_data_has_pbp"),
            None,
        )
        assert pbp_flag_result is not None
        assert pbp_flag_result.status == CheckStatus.CRITICAL

    def test_players_check_pass(self, test_db):
        """Test players check passes with data."""
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        # Insert 500+ players
        for i in range(600):
            cursor.execute(
                """
                INSERT INTO Players (person_id, full_name)
                VALUES (?, ?)
            """,
                (i, f"Player {i}"),
            )

        conn.commit()
        conn.close()

        checker = SeasonHealthChecker(season="2024-2025", db_path=test_db)
        checker._check_players()

        player_count_result = next(
            (r for r in checker.report.results if r.check_name == "player_count"), None
        )
        assert player_count_result is not None
        assert player_count_result.status == CheckStatus.PASS

    def test_boxscores_skipped_for_old_season(self, test_db):
        """Test boxscores check skipped for seasons before 2023-2024."""
        checker = SeasonHealthChecker(season="2022-2023", db_path=test_db)
        checker._check_boxscores()

        boxscore_result = next(
            (r for r in checker.report.results if r.stage == "Boxscores"), None
        )
        assert boxscore_result is not None
        assert boxscore_result.status == CheckStatus.SKIP
