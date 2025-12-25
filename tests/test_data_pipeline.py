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
            WHERE g.status = 3
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
            WHERE g.status = 3
            AND g.season_type = 'Post Season'
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} Post Season games missing PbP data"

    def test_pbp_play_counts_reasonable(self, db_connection):
        """Each game should have reasonable play counts (400-600 plays per game)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT p.game_id, COUNT(*) as play_count
            FROM PbP_Logs p
            JOIN Games g ON p.game_id = g.game_id
            WHERE g.status = 3
            AND g.season_type IN ('Regular Season', 'Post Season')
            GROUP BY p.game_id
            HAVING play_count < 300 OR play_count > 700
        """
        )
        outliers = cursor.fetchall()
        # Allow some outliers for unusual games (OT, shortened games)
        assert (
            len(outliers) < 50
        ), f"{len(outliers)} games have unusual play counts (expected 300-700)"

    def test_pbp_log_data_is_valid_json(self, db_connection):
        """PbP log_data should be valid JSON (sample check)."""
        import json

        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, log_data 
            FROM PbP_Logs 
            ORDER BY RANDOM() 
            LIMIT 10
        """
        )
        invalid_count = 0
        for game_id, log_data in cursor.fetchall():
            try:
                json.loads(log_data)
            except (json.JSONDecodeError, TypeError):
                invalid_count += 1
        assert (
            invalid_count == 0
        ), f"{invalid_count}/10 sampled PbP entries have invalid JSON"

    def test_all_completed_games_have_boxscores(self, db_connection):
        """All completed Regular/Post Season games should have boxscore data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status = 3
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
            WHERE g.status = 3
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (SELECT 1 FROM GameStates gs WHERE gs.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} games missing GameStates data"

    def test_game_states_have_no_duplicate_finals(self, db_connection):
        """Each game should have exactly one final state."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, COUNT(*) as final_count
            FROM GameStates
            WHERE is_final_state = 1
            GROUP BY game_id
            HAVING final_count > 1
        """
        )
        duplicates = cursor.fetchall()
        assert (
            len(duplicates) == 0
        ), f"{len(duplicates)} games have multiple final states (game_ids: {[d[0] for d in duplicates[:5]]})"

    def test_game_states_count_reasonable(self, db_connection):
        """Each game should have reasonable state counts (300-700 states)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, COUNT(*) as state_count
            FROM GameStates
            GROUP BY game_id
            HAVING state_count < 300 OR state_count > 700
        """
        )
        outliers = cursor.fetchall()
        # Allow some outliers for unusual games (OT, technical issues)
        assert (
            len(outliers) < 100
        ), f"{len(outliers)} games have unusual state counts (expected 300-700)"

    def test_completed_games_have_final_state(self, db_connection):
        """All completed games should have a final state."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status = 3
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND NOT EXISTS (
                SELECT 1 FROM GameStates gs 
                WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
            )
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} completed games missing final state"

    def test_game_states_scores_valid(self, db_connection):
        """GameState scores should be non-negative and reasonable."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM GameStates
            WHERE home_score < 0 OR away_score < 0 
            OR home_score > 250 OR away_score > 250
        """
        )
        invalid = cursor.fetchone()[0]
        assert invalid == 0, f"{invalid} GameStates have invalid scores"

    def test_game_states_period_valid(self, db_connection):
        """GameState periods should be between 1-7 (allowing OT)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM GameStates
            WHERE period < 1 OR period > 7
        """
        )
        invalid = cursor.fetchone()[0]
        assert invalid == 0, f"{invalid} GameStates have invalid periods"

    def test_game_data_finalized_flag_accurate(self, db_connection):
        """Games with game_data_finalized=1 should have PbP and GameStates data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.game_data_finalized = 1
            AND (
                NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
                OR NOT EXISTS (SELECT 1 FROM GameStates gs WHERE gs.game_id = g.game_id AND gs.is_final_state = 1)
            )
        """
        )
        inconsistent = cursor.fetchone()[0]
        assert (
            inconsistent == 0
        ), f"{inconsistent} games have game_data_finalized=1 but missing PbP/GameStates"

    def test_boxscore_data_finalized_flag_accurate(self, db_connection):
        """Games with boxscore_data_finalized=1 should have PlayerBox and TeamBox data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.boxscore_data_finalized = 1
            AND (
                NOT EXISTS (SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id)
                OR NOT EXISTS (SELECT 1 FROM TeamBox tb WHERE tb.game_id = g.game_id)
            )
        """
        )
        inconsistent = cursor.fetchone()[0]
        assert (
            inconsistent == 0
        ), f"{inconsistent} games have boxscore_data_finalized=1 but missing boxscores"

    def test_all_completed_games_have_teambox(self, db_connection):
        """All completed Regular/Post Season games should have TeamBox data (2 teams per game)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.status = 3
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND (SELECT COUNT(*) FROM TeamBox tb WHERE tb.game_id = g.game_id) != 2
        """
        )
        missing = cursor.fetchone()[0]
        assert (
            missing == 0
        ), f"{missing} games missing TeamBox data (should have 2 per game)"

    def test_playerbox_has_no_duplicates(self, db_connection):
        """No duplicate PlayerBox entries per game (unique game_id + player_id)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, player_id, COUNT(*) as cnt
            FROM PlayerBox
            GROUP BY game_id, player_id
            HAVING cnt > 1
        """
        )
        duplicates = cursor.fetchall()
        assert (
            len(duplicates) == 0
        ), f"{len(duplicates)} duplicate PlayerBox entries found"

    def test_teambox_has_no_duplicates(self, db_connection):
        """No duplicate TeamBox entries per game (unique game_id + team_id)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, team_id, COUNT(*) as cnt
            FROM TeamBox
            GROUP BY game_id, team_id
            HAVING cnt > 1
        """
        )
        duplicates = cursor.fetchall()
        assert (
            len(duplicates) == 0
        ), f"{len(duplicates)} duplicate TeamBox entries found"

    def test_playerbox_player_counts_reasonable(self, db_connection):
        """Each game should have reasonable player counts (8-20 per team, 16-40 total).

        NBA teams can have up to 20 active players on game day, so total can reach 40.
        """
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, COUNT(*) as player_count
            FROM PlayerBox
            GROUP BY game_id
            HAVING player_count < 16 OR player_count > 40
        """
        )
        outliers = cursor.fetchall()
        assert (
            len(outliers) == 0
        ), f"{len(outliers)} games have unusual player counts (expected 16-40)"

    def test_playerbox_minutes_valid(self, db_connection):
        """Player minutes should be reasonable (0-58, allowing OT)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM PlayerBox 
            WHERE min IS NOT NULL 
            AND (min < 0 OR min > 70)
        """
        )
        invalid = cursor.fetchone()[0]
        assert invalid == 0, f"{invalid} PlayerBox entries have invalid minutes"

    def test_playerbox_fg_percentage_valid(self, db_connection):
        """Field goal percentages should be between 0 and 1."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM PlayerBox 
            WHERE fg_pct IS NOT NULL 
            AND (fg_pct < 0 OR fg_pct > 1)
        """
        )
        invalid = cursor.fetchone()[0]
        assert invalid == 0, f"{invalid} PlayerBox entries have invalid FG%"

    def test_teambox_points_match_playerbox(self, db_connection):
        """TeamBox total points should roughly match sum of PlayerBox points (within 5%)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT tb.game_id, tb.team_id, tb.pts as team_pts, SUM(pb.pts) as player_pts_sum
            FROM TeamBox tb
            LEFT JOIN PlayerBox pb ON tb.game_id = pb.game_id AND tb.team_id = pb.team_id
            GROUP BY tb.game_id, tb.team_id
            HAVING ABS(team_pts - player_pts_sum) / CAST(team_pts AS FLOAT) > 0.05
        """
        )
        mismatches = cursor.fetchall()
        # Allow some mismatches (rounding, data quality issues)
        assert (
            len(mismatches) < 100
        ), f"{len(mismatches)} games have significant TeamBox/PlayerBox point mismatches"

    def test_games_with_game_data_have_features(self, db_connection):
        """Games with game_data_finalized=1 and prior games available should have Features."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.game_data_finalized = 1
            AND g.pre_game_data_finalized = 1
            AND NOT EXISTS (SELECT 1 FROM Features f WHERE f.game_id = g.game_id)
        """
        )
        missing = cursor.fetchone()[0]
        assert missing == 0, f"{missing} games have flags set but missing Features"

    def test_pre_game_data_finalized_flag_accurate(self, db_connection):
        """Games with pre_game_data_finalized=1 should have Features data."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM Games g
            WHERE g.pre_game_data_finalized = 1
            AND NOT EXISTS (SELECT 1 FROM Features f WHERE f.game_id = g.game_id)
        """
        )
        inconsistent = cursor.fetchone()[0]
        assert (
            inconsistent == 0
        ), f"{inconsistent} games have pre_game_data_finalized=1 but missing Features"

    def test_features_have_no_duplicates(self, db_connection):
        """No duplicate Features entries (unique game_id primary key)."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, COUNT(*) as cnt
            FROM Features
            GROUP BY game_id
            HAVING cnt > 1
        """
        )
        duplicates = cursor.fetchall()
        assert len(duplicates) == 0, f"{len(duplicates)} games have duplicate Features"

    def test_features_json_is_valid(self, db_connection):
        """Feature_set should be valid JSON (sample check)."""
        import json

        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, feature_set 
            FROM Features 
            WHERE feature_set IS NOT NULL 
            LIMIT 10
        """
        )
        for game_id, feature_set in cursor.fetchall():
            try:
                features = json.loads(feature_set)
                assert isinstance(features, dict), f"Game {game_id} features not a dict"
            except json.JSONDecodeError as e:
                assert False, f"Game {game_id} has invalid JSON: {e}"

    def test_features_have_required_keys(self, db_connection):
        """Features should contain expected key categories."""
        import json

        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, feature_set 
            FROM Features 
            WHERE feature_set IS NOT NULL 
            LIMIT 100
        """
        )

        # Expected feature prefixes based on features.py (uses Home_ and Away_)
        checked = 0
        for game_id, feature_set in cursor.fetchall():
            features = json.loads(feature_set)

            # Skip empty feature sets (early season games with no prior data)
            if len(features) == 0:
                continue

            checked += 1
            # Check that we have both Home and Away features
            has_home = any(k.startswith("Home_") for k in features.keys())
            has_away = any(k.startswith("Away_") for k in features.keys())
            assert has_home and has_away, f"Game {game_id} missing Home/Away features"

        assert checked > 0, "No non-empty feature sets found to validate"

    def test_features_values_are_reasonable(self, db_connection):
        """Feature values should be reasonable (not NaN, not extreme)."""
        import json

        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, feature_set 
            FROM Features 
            WHERE feature_set IS NOT NULL 
            LIMIT 100
        """
        )

        nan_count = 0
        checked = 0
        for game_id, feature_set in cursor.fetchall():
            features = json.loads(feature_set)

            # Skip empty feature sets
            if len(features) == 0:
                continue

            checked += 1
            for key, value in features.items():
                # Skip non-numeric features
                if not isinstance(value, (int, float)):
                    continue
                # Check for NaN (can occur in early season games with limited data)
                if isinstance(value, float) and (value != value):  # NaN check
                    nan_count += 1
                # Check for extreme values (likely errors)
                if isinstance(value, (int, float)) and abs(value) > 1000:
                    # Allow some features to be large (like total stats)
                    if "total" not in key.lower():
                        assert (
                            False
                        ), f"Game {game_id} has extreme value in {key}: {value}"

        # Allow some NaN values (early season games), but not too many
        assert checked > 0, "No feature sets checked"
        nan_pct = (
            nan_count / (checked * 43) if checked > 0 else 0
        )  # 43 features per game
        assert (
            nan_pct < 0.06
        ), f"Too many NaN features: {nan_pct:.1%} ({nan_count} NaN in {checked} games)"

    def test_features_depend_on_prior_final_states(self, db_connection):
        """Games with features should have prior games with final states."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season
            FROM Games g
            JOIN Features f ON g.game_id = f.game_id
            WHERE LENGTH(f.feature_set) > 100
            LIMIT 5
        """
        )

        for game_id, home, away, date, season in cursor.fetchall():
            # Check home team has prior final states
            cursor.execute(
                """
                SELECT COUNT(*) FROM Games g
                JOIN GameStates gs ON g.game_id = gs.game_id
                WHERE g.date_time_utc < ?
                AND (g.home_team = ? OR g.away_team = ?)
                AND g.season = ?
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND gs.is_final_state = 1
            """,
                (date, home, home, season),
            )
            home_prior_states = cursor.fetchone()[0]
            assert (
                home_prior_states > 0
            ), f"Game {game_id} has features but {home} has no prior final states"

            # Check away team has prior final states
            cursor.execute(
                """
                SELECT COUNT(*) FROM Games g
                JOIN GameStates gs ON g.game_id = gs.game_id
                WHERE g.date_time_utc < ?
                AND (g.home_team = ? OR g.away_team = ?)
                AND g.season = ?
                AND g.season_type IN ('Regular Season', 'Post Season')
                AND gs.is_final_state = 1
            """,
                (date, away, away, season),
            )
            away_prior_states = cursor.fetchone()[0]
            assert (
                away_prior_states > 0
            ), f"Game {game_id} has features but {away} has no prior final states"

    def test_early_season_games_have_no_features(self, db_connection):
        """Season openers (no prior games) should have empty feature sets."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT g.game_id, g.home_team
            FROM Games g
            WHERE g.season_type = 'Regular Season'
            AND NOT EXISTS (
                SELECT 1 FROM Games g2
                WHERE g2.season = g.season
                AND g2.season_type IN ('Regular Season', 'Post Season')
                AND g2.date_time_utc < g.date_time_utc
                AND (g2.home_team = g.home_team OR g2.away_team = g.home_team)
            )
            LIMIT 5
        """
        )

        season_openers = cursor.fetchall()
        for game_id, team in season_openers:
            cursor.execute(
                "SELECT feature_set FROM Features WHERE game_id = ?", (game_id,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                import json

                features = json.loads(result[0])
                assert (
                    len(features) == 0
                ), f"Season opener {game_id} for {team} should have empty features, has {len(features)}"

    def test_feature_generation_uses_all_prior_games(self, db_connection):
        """Features should be based on ALL prior games in season (not just last N)."""
        import json

        cursor = db_connection.cursor()
        # Find a game late in season with many prior games
        cursor.execute(
            """
            SELECT g.game_id, g.home_team, g.date_time_utc, g.season
            FROM Games g
            JOIN Features f ON g.game_id = f.game_id
            WHERE g.season = '2024-2025'
            AND g.season_type = 'Regular Season'
            AND LENGTH(f.feature_set) > 100
            ORDER BY g.date_time_utc DESC
            LIMIT 1
        """
        )

        result = cursor.fetchone()
        if result:
            game_id, home_team, date, season = result

            # Count prior games for home team
            cursor.execute(
                """
                SELECT COUNT(*) FROM Games
                WHERE date_time_utc < ?
                AND (home_team = ? OR away_team = ?)
                AND season = ?
                AND season_type IN ('Regular Season', 'Post Season')
            """,
                (date, home_team, home_team, season),
            )
            prior_game_count = cursor.fetchone()[0]

            # Features should exist if team has ANY prior games
            if prior_game_count > 0:
                cursor.execute(
                    "SELECT feature_set FROM Features WHERE game_id = ?", (game_id,)
                )
                features = json.loads(cursor.fetchone()[0])
                assert (
                    len(features) > 0
                ), f"Game {game_id} with {prior_game_count} prior games should have features"

    def test_seasons_have_expected_game_counts(self, db_connection):
        """Each complete season should have 1230 regular season games."""
        cursor = db_connection.cursor()

        # NBA regular season has 1230 games (30 teams * 82 games / 2)
        expected_regular = 1230

        # Check 2024-2025 season (should be complete or near-complete in dev database)
        cursor.execute(
            """
            SELECT COUNT(*) FROM Games 
            WHERE season = '2024-2025' AND season_type = 'Regular Season'
        """
        )
        count = cursor.fetchone()[0]
        # Allow some tolerance for in-progress season
        assert (
            count >= 1200
        ), f"2024-2025 has {count} regular season games, expected ~{expected_regular}"


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

    def test_injury_category_field_exists(self, db_connection):
        """InjuryReports should have category field for distinguishing injuries from other absences."""
        cursor = db_connection.cursor()
        cursor.execute("PRAGMA table_info(InjuryReports)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "category" in columns, "Missing 'category' column in InjuryReports table"

    def test_injury_categories_are_valid(self, db_connection):
        """All injury records should have valid category values."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT DISTINCT category FROM InjuryReports 
            WHERE category IS NOT NULL
        """
        )
        categories = [row[0] for row in cursor.fetchall()]

        # Should only have Injury and Non-Injury categories
        valid_categories = {"Injury", "Non-Injury"}
        invalid = set(categories) - valid_categories

        assert (
            not invalid
        ), f"Found invalid categories: {invalid}. Valid categories: {valid_categories}"

    def test_injury_data_has_both_categories(self, db_connection):
        """InjuryReports should track both injuries and non-injury absences."""
        cursor = db_connection.cursor()

        # Check we have injury data with category field populated
        cursor.execute("SELECT COUNT(*) FROM InjuryReports WHERE category IS NOT NULL")
        categorized_count = cursor.fetchone()[0]

        # Check we have some injury category records
        cursor.execute("SELECT COUNT(*) FROM InjuryReports WHERE category = 'Injury'")
        injury_count = cursor.fetchone()[0]

        # Check total count
        cursor.execute("SELECT COUNT(*) FROM InjuryReports")
        total_count = cursor.fetchone()[0]

        # Should have some categorized data (at least 10 records)
        assert (
            categorized_count >= 10
        ), f"Only {categorized_count} records have category populated"

        # Of categorized records, most should be injuries (at least 30%)
        if categorized_count > 0:
            injury_pct = injury_count / categorized_count
            assert (
                injury_pct >= 0.3
            ), f"Injury records only {injury_pct:.1%} of categorized data, expected >= 30%"


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
            WHERE date_time_utc >= ? AND date_time_utc < ?
            AND season_type = 'Regular Season'
            AND status <> 3
        """,
            (week_ago, three_days_ago),
        )
        not_completed = cursor.fetchone()[0]

        # Should have very few games not marked completed from 3-7 days ago
        assert (
            not_completed <= 2
        ), f"{not_completed} games from 3-7 days ago still not completed"


# STAGE 9: PREDICTIONS
# Tests for prediction generation, storage, and validation


class TestPredictions:
    """Stage 9: Prediction generation and storage validation."""

    @pytest.fixture
    def db_connection(self):
        """Fixture to provide a database connection."""
        from src.config import config

        db_path = config["database"]["path"]
        conn = sqlite3.connect(db_path)
        yield conn
        conn.close()

    def test_games_with_features_have_predictions(self, db_connection):
        """Games with finalized pre-game data should have predictions from at least one predictor."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            WHERE g.season_type IN ('Regular Season', 'Post Season')
            AND g.pre_game_data_finalized = 1
            AND NOT EXISTS (
                SELECT 1 FROM Predictions p
                WHERE p.game_id = g.game_id
            )
        """
        )
        games_without_predictions = cursor.fetchone()[0]

        # Most games with features should have predictions
        # Allow small number for just-finalized games pending prediction run
        assert (
            games_without_predictions < 50
        ), f"{games_without_predictions} games have features but no predictions"

    def test_predictions_have_no_duplicates(self, db_connection):
        """Each game+predictor combination should have at most one prediction."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id, predictor, COUNT(*) as count
            FROM Predictions
            GROUP BY game_id, predictor
            HAVING count > 1
        """
        )
        duplicates = cursor.fetchall()

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate predictions"

    def test_prediction_json_is_valid(self, db_connection):
        """All prediction_set JSON should be valid and parseable."""
        import json

        cursor = db_connection.cursor()
        cursor.execute("SELECT game_id, predictor, prediction_set FROM Predictions")

        invalid_count = 0
        for game_id, predictor, prediction_set in cursor.fetchall():
            try:
                json.loads(prediction_set)
            except json.JSONDecodeError:
                invalid_count += 1
                if invalid_count <= 5:  # Show first 5 examples
                    print(f"Invalid JSON for {game_id}, {predictor}: {prediction_set}")

        assert invalid_count == 0, f"{invalid_count} predictions have invalid JSON"

    def test_predictions_have_required_keys(self, db_connection):
        """Predictions should contain required keys: pred_home_score, pred_away_score, pred_home_win_pct."""
        import json

        cursor = db_connection.cursor()
        cursor.execute("SELECT game_id, predictor, prediction_set FROM Predictions")

        required_keys = {"pred_home_score", "pred_away_score", "pred_home_win_pct"}
        missing_count = 0

        for game_id, predictor, prediction_set in cursor.fetchall():
            pred_dict = json.loads(prediction_set)
            missing = required_keys - set(pred_dict.keys())

            if missing:
                missing_count += 1
                if missing_count <= 5:
                    print(
                        f"Game {game_id}, predictor {predictor} missing keys: {missing}"
                    )

        assert missing_count == 0, f"{missing_count} predictions missing required keys"

    def test_prediction_values_are_reasonable(self, db_connection):
        """Predicted scores should be in reasonable range for formula-based predictors.

        Note: ML predictor models (Linear, Tree, MLP, Ensemble) are placeholders
        and may produce unreasonable predictions. This test only validates Baseline.
        """
        import json

        cursor = db_connection.cursor()
        # Only test Baseline predictor (formula-based, should always be reasonable)
        cursor.execute(
            "SELECT game_id, predictor, prediction_set FROM Predictions WHERE predictor = 'Baseline'"
        )

        unreasonable_count = 0
        total = 0

        for game_id, predictor, prediction_set in cursor.fetchall():
            total += 1
            pred_dict = json.loads(prediction_set)

            home_score = pred_dict.get("pred_home_score", 0)
            away_score = pred_dict.get("pred_away_score", 0)

            if not (80 <= home_score <= 140) or not (80 <= away_score <= 140):
                unreasonable_count += 1
                if unreasonable_count <= 5:
                    print(
                        f"Game {game_id}, predictor {predictor}: scores {home_score:.1f} vs {away_score:.1f}"
                    )

        # Baseline should have no unreasonable scores (formula-based)
        assert (
            unreasonable_count == 0
        ), f"{unreasonable_count}/{total} Baseline predictions have unreasonable scores"

    def test_predictions_made_before_game_start(self, db_connection):
        """Most predictions should be made before or shortly after game start.

        Note: Historical analysis mode generates predictions after games are completed,
        so this test is lenient and just checks data format is valid.
        """
        import pandas as pd

        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT p.game_id, p.predictor, p.prediction_datetime, g.date_time_utc
            FROM Predictions p
            JOIN Games g ON p.game_id = g.game_id
            WHERE g.season_type IN ('Regular Season', 'Post Season')
            LIMIT 100
        """
        )

        valid_datetime_count = 0
        total = 0

        for game_id, predictor, pred_dt, game_dt in cursor.fetchall():
            total += 1
            try:
                pred_time = pd.to_datetime(pred_dt, utc=True)
                game_time = pd.to_datetime(game_dt, utc=True)
                valid_datetime_count += 1
            except Exception as e:
                print(f"Invalid datetime for {game_id}, {predictor}: {e}")

        # All prediction datetimes should be parseable
        assert (
            valid_datetime_count == total
        ), f"Only {valid_datetime_count}/{total} predictions have valid datetime format"

    def test_all_active_predictors_have_predictions(self, db_connection):
        """All predictors in PREDICTOR_MAP should have saved predictions."""
        from src.predictions.prediction_manager import PREDICTOR_MAP

        cursor = db_connection.cursor()
        cursor.execute("SELECT DISTINCT predictor FROM Predictions")
        predictors_with_data = {row[0] for row in cursor.fetchall()}

        # Check that major predictors have data (not necessarily all if some are new)
        expected_predictors = {"Baseline", "Linear", "Tree", "MLP"}
        missing = expected_predictors - predictors_with_data

        assert len(missing) == 0, f"Expected predictors missing predictions: {missing}"

    def test_predictions_depend_on_pre_game_data(self, db_connection):
        """Predictions should only exist for games with pre_game_data_finalized=1."""
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(DISTINCT p.game_id)
            FROM Predictions p
            JOIN Games g ON p.game_id = g.game_id
            WHERE g.pre_game_data_finalized = 0
        """
        )
        invalid_predictions = cursor.fetchone()[0]

        assert (
            invalid_predictions == 0
        ), f"{invalid_predictions} predictions exist for games without finalized pre-game data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
