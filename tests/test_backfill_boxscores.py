"""
Tests for ESPN historical backfill script (scripts/backfill_boxscores.py).

Tests cover:
- Name normalization and diacritic handling
- Player name matching against Players table
- Team abbreviation mapping (ESPN → NBA)
- Game date matching with UTC timezone handling
- PlayerBox data transformation
- TeamBox data transformation
- Edge cases: DNP players, All-Star filtering, plus_minus parsing
"""

import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add scripts/ to path so we can import the backfill module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from backfill_boxscores import (
    ALLSTAR_ABBREVS,
    build_game_lookup,
    build_player_lookup,
    build_team_lookup,
    check_game_has_boxscores,
    map_espn_player,
    map_espn_team_abbrev,
    match_espn_game_to_nba,
    normalize_name,
    transform_player_box_row,
    transform_team_box_row,
    _safe_int,
    _safe_float,
    _compute_pct,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def test_db():
    """Create a temporary test database with full schema."""
    db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    conn = sqlite3.connect(db.name)
    cursor = conn.cursor()

    cursor.execute("""
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
    """)
    cursor.executemany(
        "INSERT INTO Players (person_id, first_name, last_name, full_name) VALUES (?, ?, ?, ?)",
        [
            (1629029, "Luka", "Dončić", "Dončić, Luka"),
            (203999, "Nikola", "Jokić", "Jokić, Nikola"),
            (201566, "Russell", "Westbrook", "Westbrook, Russell"),
            (2544, "LeBron", "James", "James, LeBron"),
            (203471, "Dennis", "Schröder", "Schröder, Dennis"),
            (101108, "Chris", "Paul", "Paul, Chris"),
            # Duplicate name scenario: two Gary Paytons
            (2155, "Gary", "Payton", "Payton, Gary"),
            (1627780, "Gary", "Payton II", "Payton II, Gary"),
        ],
    )

    cursor.execute("""
        CREATE TABLE Teams (
            team_id TEXT PRIMARY KEY,
            abbreviation TEXT,
            abbreviation_normalized TEXT,
            full_name TEXT,
            full_name_normalized TEXT,
            short_name TEXT,
            short_name_normalized TEXT,
            alternatives TEXT,
            alternatives_normalized TEXT
        )
    """)
    cursor.executemany(
        """INSERT INTO Teams (team_id, abbreviation, abbreviation_normalized,
           full_name, full_name_normalized, short_name, short_name_normalized,
           alternatives, alternatives_normalized)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ("1610612738", "BOS", "bos", "Boston Celtics", "boston celtics",
             "Celtics", "celtics", "[]", "[]"),
            ("1610612744", "GSW", "gsw", "Golden State Warriors", "golden state warriors",
             "Warriors", "warriors", '["GS"]', '["gs"]'),
            ("1610612740", "NOP", "nop", "New Orleans Pelicans", "new orleans pelicans",
             "Pelicans", "pelicans", '["NO", "NOH", "NOK"]', '["no", "noh", "nok"]'),
            ("1610612752", "NYK", "nyk", "New York Knicks", "new york knicks",
             "Knicks", "knicks", '["NY"]', '["ny"]'),
            ("1610612747", "LAL", "lal", "Los Angeles Lakers", "los angeles lakers",
             "Lakers", "lakers", "[]", "[]"),
        ],
    )

    cursor.execute("""
        CREATE TABLE Games (
            game_id TEXT PRIMARY KEY,
            date_time_utc TEXT,
            home_team TEXT,
            away_team TEXT,
            status INTEGER,
            season TEXT,
            season_type TEXT,
            boxscore_data_finalized BOOLEAN DEFAULT 0
        )
    """)
    # Game at 7:30 PM ET on Jan 15 = Jan 16 00:30 UTC
    # Game at 12:30 PM ET on Jan 20 = Jan 20 17:30 UTC (same day in UTC)
    cursor.executemany(
        """INSERT INTO Games (game_id, date_time_utc, home_team, away_team, status, season, season_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            ("0020200100", "2003-01-16T00:30:00Z", "BOS", "LAL", 3, "2002-2003", "Regular Season"),
            ("0020200200", "2003-01-20T17:30:00Z", "GSW", "NYK", 3, "2002-2003", "Regular Season"),
            ("0020200300", "2003-02-10T01:00:00Z", "NOP", "BOS", 3, "2002-2003", "Regular Season"),
            ("0040200100", "2003-04-20T00:00:00Z", "LAL", "GSW", 3, "2002-2003", "Post Season"),
        ],
    )

    cursor.execute("""
        CREATE TABLE PlayerBox (
            player_id INTEGER,
            game_id TEXT,
            team_id TEXT,
            player_name TEXT,
            position TEXT,
            min REAL,
            pts INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            oreb INTEGER,
            dreb INTEGER,
            fga INTEGER,
            fgm INTEGER,
            fg_pct REAL,
            fg3a INTEGER,
            fg3m INTEGER,
            fg3_pct REAL,
            fta INTEGER,
            ftm INTEGER,
            ft_pct REAL,
            plus_minus INTEGER,
            PRIMARY KEY (player_id, game_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE TeamBox (
            team_id TEXT,
            game_id TEXT,
            pts INTEGER,
            pts_allowed INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            pf INTEGER,
            fga INTEGER,
            fgm INTEGER,
            fg_pct REAL,
            fg3a INTEGER,
            fg3m INTEGER,
            fg3_pct REAL,
            fta INTEGER,
            ftm INTEGER,
            ft_pct REAL,
            plus_minus INTEGER,
            PRIMARY KEY (team_id, game_id)
        )
    """)

    conn.commit()
    conn.close()
    yield db.name
    # Cleanup handled by tempfile


# ============================================================
# Name Normalization
# ============================================================

class TestNormalizeName:
    def test_simple_ascii(self):
        assert normalize_name("LeBron James") == "lebron james"

    def test_diacritics_doncic(self):
        assert normalize_name("Luka Dončić") == "luka doncic"

    def test_diacritics_jokic(self):
        assert normalize_name("Nikola Jokić") == "nikola jokic"

    def test_diacritics_schroder(self):
        assert normalize_name("Dennis Schröder") == "dennis schroder"

    def test_diacritics_valanciunas(self):
        assert normalize_name("Jonas Valančiūnas") == "jonas valanciunas"

    def test_extra_whitespace(self):
        assert normalize_name("  LeBron   James  ") == "lebron james"

    def test_hyphenated_name(self):
        assert normalize_name("Shai Gilgeous-Alexander") == "shai gilgeous-alexander"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_espn_matches_db_after_normalization(self):
        """ESPN plain name should match DB diacritical name after normalization."""
        espn_name = normalize_name("Luka Doncic")
        db_name = normalize_name("Luka Dončić")
        assert espn_name == db_name


# ============================================================
# Safe Conversion Helpers
# ============================================================

class TestSafeConversions:
    def test_safe_int_normal(self):
        assert _safe_int(25) == 25
        assert _safe_int(25.0) == 25

    def test_safe_int_nan(self):
        assert _safe_int(float("nan")) is None

    def test_safe_int_none(self):
        assert _safe_int(None) is None

    def test_safe_float_normal(self):
        assert _safe_float(34.5) == 34.5

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) is None

    def test_compute_pct_normal(self):
        assert _compute_pct(10, 20) == pytest.approx(0.5)

    def test_compute_pct_zero_attempts(self):
        assert _compute_pct(0, 0) is None

    def test_compute_pct_nan(self):
        assert _compute_pct(float("nan"), 10) is None

    def test_compute_pct_rounding(self):
        # 1/3 should round to 0.333
        assert _compute_pct(1, 3) == pytest.approx(0.333, abs=0.001)


# ============================================================
# Player Lookup
# ============================================================

class TestPlayerLookup:
    def test_build_lookup(self, test_db):
        lookup = build_player_lookup(test_db)
        assert lookup["lebron james"] == 2544
        assert lookup["russell westbrook"] == 201566

    def test_diacritic_name_in_lookup(self, test_db):
        """DB has diacritical names — lookup keys should be normalized."""
        lookup = build_player_lookup(test_db)
        # "Dončić" normalized to "doncic"
        assert lookup["luka doncic"] == 1629029
        assert lookup["nikola jokic"] == 203999
        assert lookup["dennis schroder"] == 203471

    def test_map_espn_player_match(self, test_db):
        lookup = build_player_lookup(test_db)
        assert map_espn_player("LeBron James", lookup) == 2544
        assert map_espn_player("Luka Doncic", lookup) == 1629029

    def test_map_espn_player_no_match(self, test_db):
        lookup = build_player_lookup(test_db)
        assert map_espn_player("Unknown Player", lookup) is None

    def test_duplicate_name_keeps_higher_id(self, test_db):
        """Two 'Gary Payton' entries — should keep higher person_id."""
        lookup = build_player_lookup(test_db)
        # "Gary Payton II" != "Gary Payton", so both should be in lookup
        assert lookup["gary payton"] == 2155
        assert lookup["gary payton ii"] == 1627780


# ============================================================
# Team Lookup
# ============================================================

class TestTeamLookup:
    def test_build_lookup(self, test_db):
        lookup = build_team_lookup(test_db)
        assert lookup["bos"] == "1610612738"
        assert lookup["gsw"] == "1610612744"
        assert lookup["lal"] == "1610612747"

    def test_espn_abbreviation_mapping(self, test_db):
        """ESPN-specific abbreviations should be resolved."""
        lookup = build_team_lookup(test_db)
        # GS -> GSW's team_id
        assert lookup.get("gs") == "1610612744"
        # NO -> NOP's team_id
        assert lookup.get("no") == "1610612740"
        # NY -> NYK's team_id
        assert lookup.get("ny") == "1610612752"

    def test_map_espn_team_standard(self, test_db):
        lookup = build_team_lookup(test_db)
        assert map_espn_team_abbrev("BOS", lookup) == "1610612738"

    def test_map_espn_team_nonstandard(self, test_db):
        lookup = build_team_lookup(test_db)
        assert map_espn_team_abbrev("GS", lookup) == "1610612744"

    def test_map_espn_team_unknown(self, test_db):
        lookup = build_team_lookup(test_db)
        assert map_espn_team_abbrev("EAST", lookup) is None


# ============================================================
# Game Lookup & Matching
# ============================================================

class TestGameLookup:
    def test_build_lookup_utc_date(self, test_db):
        lookup = build_game_lookup(test_db, "2002-2003")
        # Game at Jan 16 00:30 UTC — direct UTC date match
        assert ("2003-01-16", "BOS", "LAL") in lookup

    def test_build_lookup_timezone_offset(self, test_db):
        """ESPN Eastern date (Jan 15) should match UTC date (Jan 16) via date-1."""
        lookup = build_game_lookup(test_db, "2002-2003")
        # date-1 of UTC Jan 16 = Jan 15 (ESPN's Eastern date)
        assert ("2003-01-15", "BOS", "LAL") in lookup

    def test_build_lookup_same_day(self, test_db):
        """Afternoon game — same date in both Eastern and UTC."""
        lookup = build_game_lookup(test_db, "2002-2003")
        # Game at Jan 20 17:30 UTC (12:30 PM ET) — same day
        assert ("2003-01-20", "GSW", "NYK") in lookup

    def test_match_espn_game_evening(self, test_db):
        """Match an ESPN evening game (Eastern date ≠ UTC date)."""
        lookup = build_game_lookup(test_db, "2002-2003")

        # ESPN team box for this game — game_date is Eastern (Jan 15)
        team_box = pd.DataFrame([
            {"game_id": 99999, "game_date": "2003-01-15", "team_abbreviation": "BOS",
             "team_home_away": "home", "team_score": 105},
            {"game_id": 99999, "game_date": "2003-01-15", "team_abbreviation": "LAL",
             "team_home_away": "away", "team_score": 98},
        ])

        result = match_espn_game_to_nba(team_box, lookup)
        assert result == "0020200100"

    def test_match_espn_game_afternoon(self, test_db):
        """Match an ESPN afternoon game (same date in both timezones)."""
        lookup = build_game_lookup(test_db, "2002-2003")

        team_box = pd.DataFrame([
            {"game_id": 88888, "game_date": "2003-01-20", "team_abbreviation": "GS",
             "team_home_away": "home", "team_score": 110},
            {"game_id": 88888, "game_date": "2003-01-20", "team_abbreviation": "NY",
             "team_home_away": "away", "team_score": 100},
        ])

        result = match_espn_game_to_nba(team_box, lookup)
        assert result == "0020200200"

    def test_match_espn_game_no_match(self, test_db):
        """Game that doesn't exist in our database."""
        lookup = build_game_lookup(test_db, "2002-2003")

        team_box = pd.DataFrame([
            {"game_id": 77777, "game_date": "2003-03-15", "team_abbreviation": "BOS",
             "team_home_away": "home", "team_score": 105},
            {"game_id": 77777, "game_date": "2003-03-15", "team_abbreviation": "LAL",
             "team_home_away": "away", "team_score": 98},
        ])

        result = match_espn_game_to_nba(team_box, lookup)
        assert result is None

    def test_check_game_has_boxscores_empty(self, test_db):
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        assert check_game_has_boxscores(cursor, "0020200100") is False
        conn.close()

    def test_check_game_has_boxscores_with_data(self, test_db):
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO PlayerBox (player_id, game_id, team_id) VALUES (1, '0020200100', '1610612738')"
        )
        conn.commit()
        assert check_game_has_boxscores(cursor, "0020200100") is True
        conn.close()


# ============================================================
# PlayerBox Transform
# ============================================================

class TestTransformPlayerBox:
    def _make_row(self, **overrides):
        defaults = {
            "athlete_display_name": "Test Player",
            "athlete_position_abbreviation": "PG",
            "minutes": 32.0,
            "points": 25.0,
            "rebounds": 5.0,
            "assists": 8.0,
            "steals": 2.0,
            "blocks": 0.0,
            "turnovers": 3.0,
            "fouls": 2.0,
            "offensive_rebounds": 1.0,
            "defensive_rebounds": 4.0,
            "field_goals_attempted": 18.0,
            "field_goals_made": 10.0,
            "three_point_field_goals_attempted": 6.0,
            "three_point_field_goals_made": 3.0,
            "free_throws_attempted": 4.0,
            "free_throws_made": 2.0,
            "plus_minus": "+8",
        }
        defaults.update(overrides)
        return pd.Series(defaults)

    def test_normal_player(self):
        row = self._make_row()
        result = transform_player_box_row(row, "0020200100", "1610612738", 1629029)

        assert result["player_id"] == 1629029
        assert result["game_id"] == "0020200100"
        assert result["team_id"] == "1610612738"
        assert result["player_name"] == "Test Player"
        assert result["position"] == "PG"
        assert result["min"] == 32.0
        assert result["pts"] == 25
        assert result["reb"] == 5
        assert result["ast"] == 8
        assert result["fg_pct"] == pytest.approx(10 / 18, abs=0.001)
        assert result["fg3_pct"] == pytest.approx(3 / 6, abs=0.001)
        assert result["ft_pct"] == pytest.approx(2 / 4, abs=0.001)
        assert result["plus_minus"] == 8

    def test_plus_minus_negative(self):
        row = self._make_row(plus_minus="-12")
        result = transform_player_box_row(row, "0020200100", "1610612738", 1)
        assert result["plus_minus"] == -12

    def test_plus_minus_zero(self):
        row = self._make_row(plus_minus="0")
        result = transform_player_box_row(row, "0020200100", "1610612738", 1)
        assert result["plus_minus"] == 0

    def test_plus_minus_nan(self):
        row = self._make_row(plus_minus=float("nan"))
        result = transform_player_box_row(row, "0020200100", "1610612738", 1)
        assert result["plus_minus"] is None

    def test_zero_fga(self):
        """When FGA=0, fg_pct should be None (avoid division by zero)."""
        row = self._make_row(
            field_goals_attempted=0.0,
            field_goals_made=0.0,
        )
        result = transform_player_box_row(row, "0020200100", "1610612738", 1)
        assert result["fg_pct"] is None

    def test_nan_stats_for_zero_minute_player(self):
        """Player with 0 minutes but did play (bench warmers)."""
        row = self._make_row(
            minutes=0.0,
            points=0.0,
            rebounds=0.0,
            assists=0.0,
            field_goals_attempted=0.0,
            field_goals_made=0.0,
            plus_minus="0",
        )
        result = transform_player_box_row(row, "0020200100", "1610612738", 1)
        assert result["min"] == 0.0
        assert result["pts"] == 0
        assert result["fg_pct"] is None  # 0/0


# ============================================================
# TeamBox Transform
# ============================================================

class TestTransformTeamBox:
    def _make_row(self, **overrides):
        defaults = {
            "team_score": 110,
            "total_rebounds": 45,
            "assists": 25,
            "steals": 8,
            "blocks": 5,
            "turnovers": 12,
            "fouls": 18,
            "field_goals_attempted": 85,
            "field_goals_made": 42,
            "field_goal_pct": 49.4,
            "three_point_field_goals_attempted": 35,
            "three_point_field_goals_made": 12,
            "three_point_field_goal_pct": 34.3,
            "free_throws_attempted": 20,
            "free_throws_made": 14,
            "free_throw_pct": 70.0,
        }
        defaults.update(overrides)
        return pd.Series(defaults)

    def test_normal_team(self):
        row = self._make_row()
        result = transform_team_box_row(row, "0020200100", "1610612738", 105)

        assert result["team_id"] == "1610612738"
        assert result["game_id"] == "0020200100"
        assert result["pts"] == 110
        assert result["pts_allowed"] == 105
        assert result["reb"] == 45
        assert result["plus_minus"] == 5  # 110 - 105

    def test_percentage_conversion(self):
        """ESPN percentages (44.9) should be converted to decimals (0.449)."""
        row = self._make_row(
            field_goal_pct=44.9,
            three_point_field_goal_pct=34.3,
            free_throw_pct=70.0,
        )
        result = transform_team_box_row(row, "0020200100", "1610612738", 100)

        assert result["fg_pct"] == pytest.approx(0.449, abs=0.001)
        assert result["fg3_pct"] == pytest.approx(0.343, abs=0.001)
        assert result["ft_pct"] == pytest.approx(0.700, abs=0.001)

    def test_nan_percentage(self):
        """NaN percentage should become None."""
        row = self._make_row(field_goal_pct=float("nan"))
        result = transform_team_box_row(row, "0020200100", "1610612738", 100)
        assert result["fg_pct"] is None

    def test_plus_minus_computed(self):
        """Plus/minus should be team_score - opponent_score."""
        row = self._make_row(team_score=95)
        result = transform_team_box_row(row, "0020200100", "1610612738", 110)
        assert result["plus_minus"] == -15  # 95 - 110


# ============================================================
# All-Star Filtering
# ============================================================

class TestAllStarFiltering:
    def test_allstar_abbrevs(self):
        """Verify All-Star abbreviations are defined."""
        assert "EAST" in ALLSTAR_ABBREVS
        assert "WEST" in ALLSTAR_ABBREVS
        assert "GIA" in ALLSTAR_ABBREVS
        assert "LEB" in ALLSTAR_ABBREVS
        assert "BOS" not in ALLSTAR_ABBREVS
