"""
test_data_pipeline.py

Tests for verifying data pipeline functionality:
- Data completeness for all seasons
- Automatic updating capability
- Player and injury data linking
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from src.config import config

DB_PATH = config["database"]["path"]


class TestDataCompleteness:
    """Tests to verify data is complete for all seasons."""

    @pytest.fixture
    def db_connection(self):
        """Create database connection for tests."""
        conn = sqlite3.connect(DB_PATH)
        yield conn
        conn.close()

    def test_all_regular_season_games_have_pbp(self, db_connection):
        """All completed Regular Season games should have PbP data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status IN ('Completed', 'Final')
            AND g.season_type = 'Regular Season'
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} Regular Season games missing PbP data"

    def test_all_post_season_games_have_pbp(self, db_connection):
        """All completed Post Season games should have PbP data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status IN ('Completed', 'Final')
            AND g.season_type = 'Post Season'
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} Post Season games missing PbP data"

    def test_all_completed_games_have_boxscores(self, db_connection):
        """All completed Regular/Post Season games should have boxscore data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status IN ('Completed', 'Final')
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} games missing PlayerBox data"

    def test_all_completed_games_have_game_states(self, db_connection):
        """All completed Regular/Post Season games should have GameStates."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status IN ('Completed', 'Final')
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (SELECT 1 FROM GameStates gs WHERE gs.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} games missing GameStates data"

    def test_game_data_finalized_flag_accurate(self, db_connection):
        """Games with game_data_finalized=1 should have PbP and BoxScore data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.game_data_finalized = 1
            AND (
                NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
                OR NOT EXISTS (SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id)
            )
        """
        )
        inconsistent = cursor.fetchone()[0]
        assert (
            inconsistent == 0
        ), f"{inconsistent} games have finalized=1 but missing data"

    def test_seasons_have_expected_game_counts(self, db_connection):
        """Each season should have reasonable game counts."""
        cursor = db_connection.cursor()

        # NBA regular season has 1230 games (30 teams * 82 games / 2)
        expected_regular = 1230

        for season in ["2023-2024", "2024-2025"]:
            cursor.execute(
                """
                SELECT COUNT(*) FROM Games 
                WHERE season = ? AND season_type = 'Regular Season'
            """,
                (season,),
            )
            count = cursor.fetchone()[0]
            assert (
                count == expected_regular
            ), f"{season} has {count} regular season games, expected {expected_regular}"


class TestPlayerData:
    """Tests for player data completeness and linking."""

    @pytest.fixture
    def db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        yield conn
        conn.close()

    def test_players_table_has_data(self, db_connection):
        """Players table should have active player data."""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM Players WHERE roster_status = 1")
        active_players = cursor.fetchone()[0]
        # NBA has ~450 active players
        assert (
            active_players >= 400
        ), f"Only {active_players} active players, expected 400+"

    def test_playerbox_references_valid_players(self, db_connection):
        """Most PlayerBox entries should reference valid Players."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM PlayerBox pb
            WHERE NOT EXISTS (SELECT 1 FROM Players p WHERE p.person_id = pb.player_id)
        """
        )
        unlinked = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM PlayerBox")
        total = cursor.fetchone()[0]
        # Allow some unlinked (very new players might not be in Players yet)
        assert (
            unlinked / total < 0.01
        ), f"{unlinked} of {total} PlayerBox entries unlinked"


class TestInjuryData:
    """Tests for injury data completeness and linking."""

    @pytest.fixture
    def db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        yield conn
        conn.close()

    def test_injury_reports_have_recent_data(self, db_connection):
        """InjuryReports should have data from the last few days."""
        cursor = db_connection.cursor()
        # Allow 3 days of lag (weekends, etc.)
        three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT MAX(report_timestamp) FROM InjuryReports 
            WHERE source = 'NBA_Official'
        """
        )
        latest = cursor.fetchone()[0]
        assert (
            latest >= three_days_ago
        ), f"Latest injury report is {latest}, expected >= {three_days_ago}"

    def test_injury_player_id_match_rate(self, db_connection):
        """Most injury reports should have nba_player_id linked."""
        cursor = db_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM InjuryReports WHERE nba_player_id IS NOT NULL"
        )
        matched = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM InjuryReports")
        total = cursor.fetchone()[0]
        match_rate = matched / total if total > 0 else 0
        assert (
            match_rate >= 0.95
        ), f"Injury player match rate {match_rate:.1%}, expected >= 95%"

    def test_injury_data_covers_all_seasons(self, db_connection):
        """InjuryReports should have data for all tracked seasons."""
        cursor = db_connection.cursor()
        for season_start in ["2023-10", "2024-10", "2025-10"]:
            cursor.execute(
                """
                SELECT COUNT(*) FROM InjuryReports 
                WHERE report_timestamp LIKE ? || '%'
            """,
                (season_start,),
            )
            count = cursor.fetchone()[0]
            assert count > 0, f"No injury data found for season starting {season_start}"


class TestPipelineFunctionality:
    """Tests for pipeline update functionality."""

    def test_players_module_imports(self):
        """Players module should import without errors."""
        from src.database_updater.players import (
            fetch_players,
            save_players,
            update_players,
        )

        assert callable(update_players)
        assert callable(fetch_players)
        assert callable(save_players)

    def test_injuries_module_imports(self):
        """Injuries module should import without errors."""
        from src.database_updater.nba_official_injuries import (
            build_player_lookup,
            normalize_player_name,
            update_nba_official_injuries,
        )

        assert callable(update_nba_official_injuries)
        assert callable(normalize_player_name)
        assert callable(build_player_lookup)

    def test_database_update_manager_imports(self):
        """Database update manager should import without errors."""
        from src.database_updater.database_update_manager import (
            update_database,
            update_game_data,
            update_injury_data,
            update_pre_game_data,
        )

        assert callable(update_database)
        assert callable(update_game_data)
        assert callable(update_injury_data)
        assert callable(update_pre_game_data)

    def test_name_normalization(self):
        """Player name normalization should handle edge cases."""
        from src.database_updater.nba_official_injuries import normalize_player_name

        # Test suffix handling
        assert normalize_player_name("WalkerIV, Lonnie") == "walker, lonnie"
        assert normalize_player_name("Williams III, Robert") == "williams, robert"
        assert normalize_player_name("Payton Jr., Gary") == "payton, gary"

        # Test special characters
        assert normalize_player_name("Jokić, Nikola") == "jokic, nikola"
        assert normalize_player_name("Schröder, Dennis") == "schroder, dennis"

        # Test normal names
        assert normalize_player_name("James, LeBron") == "james, lebron"


class TestDataFreshness:
    """Tests to verify data is being updated."""

    @pytest.fixture
    def db_connection(self):
        conn = sqlite3.connect(DB_PATH)
        yield conn
        conn.close()

    def test_recent_games_are_completed(self, db_connection):
        """Games from several days ago should be marked completed with data."""
        cursor = db_connection.cursor()
        # Check games from 3-7 days ago (giving time for updates)
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT COUNT(*) FROM Games
            WHERE date_time_est >= ? AND date_time_est < ?
            AND season_type = 'Regular Season'
            AND status NOT IN ('Completed', 'Final')
        """,
            (week_ago, three_days_ago),
        )
        not_completed = cursor.fetchone()[0]

        # Should have very few games not marked completed from 3-7 days ago
        assert (
            not_completed <= 2
        ), f"{not_completed} games from 3-7 days ago still not completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
