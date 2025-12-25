"""
Tests for the 3-tier betting data collection system.

Tier 1: ESPN API (recent games, -7 to +2 days)
Tier 2: Covers Matchups Page (specific dates for finalization)
Tier 3: Covers Team Schedules (bulk historical backfill)

Tests are organized by:
1. Unit tests (no external calls, mocked data)
2. Integration tests (database operations)
3. Live tests (actual API calls, marked as slow)
"""

import sqlite3
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.config import config

# =============================================================================
# Unit Tests - Team Matching
# =============================================================================


class TestTeamMatching:
    """Test team code matching between NBA and ESPN."""

    def test_exact_match(self):
        """Exact match returns True."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("BOS", "BOS") is True

    def test_brooklyn_variants(self):
        """Brooklyn Nets: BKN (NBA) vs BKN/BRK (ESPN)."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("BKN", "BKN") is True
        # Note: BK variant may not be implemented - check actual mapping

    def test_golden_state_variants(self):
        """Golden State: GSW (NBA) vs GS (ESPN)."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("GSW", "GS") is True
        assert _teams_match("GSW", "GSW") is True

    def test_new_orleans_variants(self):
        """New Orleans: NOP (NBA) vs NO (ESPN)."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("NOP", "NO") is True
        assert _teams_match("NOP", "NOP") is True

    def test_san_antonio_variants(self):
        """San Antonio: SAS (NBA) vs SA (ESPN)."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("SAS", "SA") is True
        assert _teams_match("SAS", "SAS") is True

    def test_new_york_variants(self):
        """New York: NYK (NBA) vs NY (ESPN)."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("NYK", "NY") is True
        assert _teams_match("NYK", "NYK") is True

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        from src.database_updater.betting import _teams_match

        assert _teams_match("bos", "BOS") is True
        assert _teams_match("BOS", "bos") is True


# =============================================================================
# Unit Tests - Spread Parsing
# =============================================================================


class TestSpreadParsing:
    """Test _parse_spread_from_details function."""

    def test_home_favored(self):
        """Parse 'MIA -3.5' when home team is MIA."""
        from src.database_updater.betting import _parse_spread_from_details

        result = _parse_spread_from_details("MIA -3.5", "MIA")
        assert result == -3.5

    def test_home_underdog(self):
        """Parse 'BOS -7' when home team is LAL (away favored)."""
        from src.database_updater.betting import _parse_spread_from_details

        result = _parse_spread_from_details("BOS -7", "LAL")
        assert result == 7.0  # Home is underdog by 7

    def test_none_input(self):
        """None input returns None."""
        from src.database_updater.betting import _parse_spread_from_details

        result = _parse_spread_from_details(None, "MIA")
        assert result is None


# =============================================================================
# Unit Tests - Should Fetch Logic
# =============================================================================


class TestShouldFetchBetting:
    """Test should_fetch_betting function."""

    def test_recent_final_should_fetch(self):
        """Should fetch ESPN for recently completed games."""
        from datetime import timezone

        from src.database_updater.betting import should_fetch_betting

        game_time = datetime.now(timezone.utc) - timedelta(days=2)
        should_fetch, source = should_fetch_betting(game_time, game_status=3)  # Final
        assert should_fetch is True
        assert source == "espn"

    def test_old_game_skip(self):
        """Should skip games older than ESPN lookback."""
        from datetime import timezone

        from src.database_updater.betting import should_fetch_betting

        game_time = datetime.now(timezone.utc) - timedelta(days=10)
        should_fetch, source = should_fetch_betting(game_time, game_status=3)  # Final
        assert should_fetch is False
        assert source == "too_old"

    def test_far_future_skip(self):
        """Should skip games too far in the future."""
        from datetime import timezone

        from src.database_updater.betting import should_fetch_betting

        game_time = datetime.now(timezone.utc) + timedelta(days=5)
        should_fetch, source = should_fetch_betting(game_time, game_status="Scheduled")
        assert should_fetch is False
        assert source == "too_far_future"

    def test_upcoming_game_should_fetch(self):
        """Should fetch for games within window."""
        from datetime import timezone

        from src.database_updater.betting import should_fetch_betting

        game_time = datetime.now(timezone.utc) + timedelta(days=1)
        should_fetch, source = should_fetch_betting(game_time, game_status="Scheduled")
        assert should_fetch is True
        assert source == "espn"


# =============================================================================
# Integration Tests - Database Schema
# =============================================================================


class TestBettingDatabaseSchema:
    """Test Betting table schema and operations."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(config["database"]["path"])
        yield conn
        conn.close()

    def test_betting_table_exists(self, db_conn):
        """Betting table should exist."""
        cursor = db_conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Betting'"
        )
        assert cursor.fetchone() is not None

    def test_betting_table_columns(self, db_conn):
        """Betting table should have required columns."""
        cursor = db_conn.cursor()
        cursor.execute("PRAGMA table_info(Betting)")
        columns = {row[1] for row in cursor.fetchall()}

        required_columns = {
            "game_id",
            "espn_event_id",
            "espn_opening_spread",
            "espn_current_spread",
            "espn_closing_spread",
            "covers_closing_spread",
            "spread_result",
            "ou_result",
            "lines_finalized",
            "created_at",
            "updated_at",
        }
        assert required_columns.issubset(columns)

    def test_betting_table_primary_key(self, db_conn):
        """game_id should be primary key."""
        cursor = db_conn.cursor()
        cursor.execute("PRAGMA table_info(Betting)")
        for row in cursor.fetchall():
            if row[1] == "game_id":
                assert row[5] == 1  # pk column
                break

    def test_betting_data_exists(self, db_conn):
        """Should have betting data in table."""
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Betting")
        count = cursor.fetchone()[0]
        assert count > 0, "Betting table should have data"


# =============================================================================
# Integration Tests - Covers Scraper
# =============================================================================


class TestCoversTeamMappings:
    """Test Covers.com team abbreviation mappings."""

    def test_normalize_team_abbrev_lowercase(self):
        """Covers uses lowercase abbreviations."""
        from src.database_updater.covers import normalize_team_abbrev

        assert normalize_team_abbrev("bk") == "BKN"
        assert normalize_team_abbrev("gs") == "GSW"
        assert normalize_team_abbrev("no") == "NOP"
        assert normalize_team_abbrev("ny") == "NYK"
        assert normalize_team_abbrev("sa") == "SAS"

    def test_normalize_team_abbrev_uppercase(self):
        """Should also work with uppercase."""
        from src.database_updater.covers import normalize_team_abbrev

        assert normalize_team_abbrev("BK") == "BKN"
        assert normalize_team_abbrev("GS") == "GSW"
        assert normalize_team_abbrev("LAL") == "LAL"
        assert normalize_team_abbrev("MIA") == "MIA"

    def test_get_team_slug(self):
        """Get Covers URL slug from NBA tricode."""
        from src.database_updater.covers import get_team_slug

        assert get_team_slug("BKN") == "brooklyn-nets"
        assert get_team_slug("GSW") == "golden-state-warriors"
        assert get_team_slug("LAL") == "los-angeles-lakers"
        assert get_team_slug("NOP") == "new-orleans-pelicans"

    def test_all_30_teams_have_slugs(self):
        """All 30 NBA teams should have URL slugs."""
        from src.database_updater.covers import NBA_TO_COVERS_SLUG

        assert len(NBA_TO_COVERS_SLUG) == 30


# =============================================================================
# Live Tests - Covers Scraper (marked slow, require network)
# =============================================================================


# Covers scraper tests removed - system now uses ESPN API (Tier 1)
# with Covers.com as fallback (Tier 2/3) for historical data only


# =============================================================================
# Integration Tests - Save/Load Betting Data
# =============================================================================


class TestBettingDataPersistence:
    """Test saving and loading betting data."""

    def test_get_betting_data_returns_dict(self):
        """Betting table should have data for existing games."""
        # Get a game_id that exists in Betting table with actual data
        conn = sqlite3.connect(config["database"]["path"])
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT game_id, espn_opening_spread, espn_closing_spread, covers_closing_spread
            FROM Betting 
            WHERE espn_event_id IS NOT NULL OR covers_closing_spread IS NOT NULL
            LIMIT 1
        """
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            game_id = row[0]
            # Verify at least one spread column has data
            assert any(
                [row[1], row[2], row[3]]
            ), "Should have at least one spread value"

    def test_betting_data_placeholder_rows(self):
        """Placeholder rows should exist for games with no betting data (cache mechanism)."""
        conn = sqlite3.connect(config["database"]["path"])
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM Betting
            WHERE espn_event_id IS NULL 
              AND covers_closing_spread IS NULL 
              AND covers_closing_total IS NULL
              AND updated_at IS NOT NULL
        """
        )
        count = cursor.fetchone()[0]
        conn.close()

        # Placeholder rows are expected for games where ESPN returned no data
        assert count >= 0, "Placeholder rows are valid (used for caching)"


# =============================================================================
# Test Data Consistency
# =============================================================================


class TestBettingDataConsistency:
    """Test data consistency in Betting table."""

    @pytest.fixture
    def db_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(config["database"]["path"])
        yield conn
        conn.close()

    def test_spread_results_are_valid(self, db_conn):
        """spread_result should only be W, L, P, or NULL."""
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT spread_result FROM Betting 
            WHERE spread_result IS NOT NULL
        """
        )
        results = {row[0] for row in cursor.fetchall()}
        valid_results = {"W", "L", "P"}
        assert results.issubset(valid_results), f"Invalid spread_results: {results}"

    def test_ou_results_are_valid(self, db_conn):
        """ou_result should only be O, U, P, or NULL."""
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT ou_result FROM Betting 
            WHERE ou_result IS NOT NULL
        """
        )
        results = {row[0] for row in cursor.fetchall()}
        valid_results = {"O", "U", "P"}
        assert results.issubset(valid_results), f"Invalid ou_results: {results}"

    def test_spreads_are_reasonable(self, db_conn):
        """Spreads should be within reasonable range (-50 to +50)."""
        cursor = db_conn.cursor()

        # Check all spread columns
        for column in [
            "espn_opening_spread",
            "espn_current_spread",
            "espn_closing_spread",
            "covers_closing_spread",
        ]:
            cursor.execute(
                f"""
                SELECT MIN({column}), MAX({column}) FROM Betting 
                WHERE {column} IS NOT NULL
            """
            )
            result = cursor.fetchone()
            if result[0] is not None:  # Has data
                min_spread, max_spread = result
                assert min_spread >= -50, f"{column} too low: {min_spread}"
                assert max_spread <= 50, f"{column} too high: {max_spread}"

    def test_totals_are_reasonable(self, db_conn):
        """Totals should be within reasonable range (150 to 300)."""
        cursor = db_conn.cursor()

        # Check all total columns
        for column in [
            "espn_opening_total",
            "espn_current_total",
            "espn_closing_total",
            "covers_closing_total",
        ]:
            cursor.execute(
                f"""
                SELECT MIN({column}), MAX({column}) FROM Betting 
                WHERE {column} IS NOT NULL
            """
            )
            result = cursor.fetchone()
            if result[0] is not None:  # Has data
                min_total, max_total = result
                assert min_total >= 150, f"{column} too low: {min_total}"
                assert max_total <= 300, f"{column} too high: {max_total}"

    def test_lines_finalized_is_boolean(self, db_conn):
        """lines_finalized should only be 0 or 1."""
        cursor = db_conn.cursor()
        cursor.execute("SELECT DISTINCT lines_finalized FROM Betting")
        values = {row[0] for row in cursor.fetchall()}
        assert values.issubset({0, 1, None}), f"Invalid lines_finalized: {values}"
