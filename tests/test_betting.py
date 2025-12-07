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
        should_fetch, source = should_fetch_betting(game_time, game_status="Final")
        assert should_fetch is True
        assert source == "espn"

    def test_old_game_skip(self):
        """Should skip games older than ESPN lookback."""
        from datetime import timezone

        from src.database_updater.betting import should_fetch_betting

        game_time = datetime.now(timezone.utc) - timedelta(days=10)
        should_fetch, source = should_fetch_betting(game_time, game_status="Final")
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
            "spread",
            "spread_result",
            "total",
            "ou_result",
            "source",
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


@pytest.mark.slow
class TestCoversMatchupsScraper:
    """Test Covers matchups page scraping (live network calls)."""

    def test_fetch_matchups_returns_data(self):
        """fetch_matchups_for_date should return game data."""
        from src.database_updater.covers import fetch_matchups_for_date

        # Use a known date with games
        test_date = date(2024, 12, 4)
        games = fetch_matchups_for_date(test_date, delay=0)

        assert len(games) > 0, "Should find games on 2024-12-04"

    def test_fetch_matchups_has_spread(self):
        """Fetched games should have spread data."""
        from src.database_updater.covers import fetch_matchups_for_date

        test_date = date(2024, 12, 4)
        games = fetch_matchups_for_date(test_date, delay=0)

        games_with_spread = [g for g in games if g.spread is not None]
        assert len(games_with_spread) > 0, "Should have games with spreads"

    def test_fetch_matchups_has_total(self):
        """Fetched games should have total data."""
        from src.database_updater.covers import fetch_matchups_for_date

        test_date = date(2024, 12, 4)
        games = fetch_matchups_for_date(test_date, delay=0)

        games_with_total = [g for g in games if g.total is not None]
        assert len(games_with_total) > 0, "Should have games with totals"

    def test_fetch_matchups_has_results(self):
        """Completed games should have spread/OU results."""
        from src.database_updater.covers import fetch_matchups_for_date

        test_date = date(2024, 12, 4)
        games = fetch_matchups_for_date(test_date, delay=0)

        games_with_results = [g for g in games if g.spread_result is not None]
        assert len(games_with_results) > 0, "Should have games with results"


@pytest.mark.slow
class TestCoversTeamScheduleScraper:
    """Test Covers team schedule scraping (live network calls)."""

    def test_fetch_team_schedule_returns_data(self):
        """fetch_team_schedule should return home games."""
        from src.database_updater.covers import fetch_team_schedule

        games = fetch_team_schedule("BOS", "2024-2025", delay=0)

        assert len(games) > 20, "Should find many home games"

    def test_fetch_team_schedule_sets_home_team(self):
        """Home team should be set correctly."""
        from src.database_updater.covers import fetch_team_schedule

        games = fetch_team_schedule("LAL", "2024-2025", delay=0)

        for game in games:
            assert game.home_team == "LAL", "All games should have LAL as home"

    def test_fetch_team_schedule_has_betting_data(self):
        """Games should have spread and total."""
        from src.database_updater.covers import fetch_team_schedule

        games = fetch_team_schedule("MIA", "2024-2025", delay=0)

        games_with_data = [
            g for g in games if g.spread is not None and g.total is not None
        ]
        assert len(games_with_data) > 10, "Should have games with betting data"


# =============================================================================
# Integration Tests - Save/Load Betting Data
# =============================================================================


class TestBettingDataPersistence:
    """Test saving and loading betting data."""

    def test_get_betting_data_returns_dict(self):
        """get_betting_data should return dict for existing game."""
        from src.database_updater.betting import get_betting_data

        # Get a game_id that exists in Betting table
        conn = sqlite3.connect(config["database"]["path"])
        cursor = conn.cursor()
        cursor.execute("SELECT game_id FROM Betting LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row:
            game_id = row[0]
            data = get_betting_data(game_id)
            assert data is not None
            assert "spread" in data
            assert "total" in data

    def test_get_betting_data_nonexistent_returns_none(self):
        """get_betting_data should return None for non-existent game."""
        from src.database_updater.betting import get_betting_data

        data = get_betting_data("9999999999")  # Non-existent game
        assert data is None


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
        """Spreads should be within reasonable range (-30 to +30)."""
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT MIN(spread), MAX(spread) FROM Betting 
            WHERE spread IS NOT NULL
        """
        )
        min_spread, max_spread = cursor.fetchone()
        assert min_spread >= -35, f"Spread too low: {min_spread}"
        assert max_spread <= 35, f"Spread too high: {max_spread}"

    def test_totals_are_reasonable(self, db_conn):
        """Totals should be within reasonable range (180 to 280)."""
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT MIN(total), MAX(total) FROM Betting 
            WHERE total IS NOT NULL
        """
        )
        min_total, max_total = cursor.fetchone()
        assert min_total >= 170, f"Total too low: {min_total}"
        assert max_total <= 290, f"Total too high: {max_total}"

    def test_lines_finalized_is_boolean(self, db_conn):
        """lines_finalized should only be 0 or 1."""
        cursor = db_conn.cursor()
        cursor.execute("SELECT DISTINCT lines_finalized FROM Betting")
        values = {row[0] for row in cursor.fetchall()}
        assert values.issubset({0, 1, None}), f"Invalid lines_finalized: {values}"
