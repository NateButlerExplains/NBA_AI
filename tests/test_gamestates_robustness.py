"""
Quick test to validate GameStates parsing handles both PBP sources correctly.
Tests edge cases: missing fields, empty logs, both Live and Stats endpoints.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database_updater.game_states import create_game_states


def test_live_endpoint_data():
    """Test parsing with Live CDN data (orderNumber)."""
    games_info = {
        "0022400001": {
            "home": "BOS",
            "away": "NYK",
            "date_time_utc": "2024-10-22T19:30:00",
            "pbp_logs": [
                {
                    "orderNumber": 1,
                    "actionId": 1,
                    "period": 1,
                    "clock": "PT11M59.00S",
                    "scoreHome": "0",
                    "scoreAway": "0",
                    "actionType": "jumpball",
                    "subType": "",
                    "description": "Jump Ball",
                    "teamTricode": "BOS",
                },
                {
                    "orderNumber": 2,
                    "period": 1,
                    "clock": "PT11M45.00S",
                    "scoreHome": "2",
                    "scoreAway": "0",
                    "actionType": "2pt",
                    "subType": "layup",
                    "description": "Tatum Layup (2 PTS)",
                    "personId": 1627759,
                    "playerNameI": "Tatum, J.",
                    "teamTricode": "BOS",
                    "pointsTotal": 2,
                },
                {
                    "orderNumber": 492,
                    "period": 4,
                    "clock": "PT00M00.00S",
                    "scoreHome": "108",
                    "scoreAway": "104",
                    "actionType": "game",
                    "subType": "end",
                    "description": "Game End",
                },
            ],
        }
    }

    result = create_game_states(games_info)
    assert "0022400001" in result, "Game ID should be in result"
    assert len(result["0022400001"]) == 3, "Should have 3 game states"
    assert result["0022400001"][-1]["is_final_state"], "Last state should be final"
    assert result["0022400001"][0]["home_score"] == 0, "First state score should be 0"
    assert result["0022400001"][1]["home_score"] == 2, "Second state score should be 2"
    print("✓ Live endpoint data parsing works")


def test_stats_endpoint_data():
    """Test parsing with Stats API data (actionId)."""
    games_info = {
        "0022400002": {
            "home": "LAL",
            "away": "GSW",
            "date_time_utc": "2024-10-22T22:00:00",
            "pbp_logs": [
                {
                    "actionId": 1,
                    "period": 1,
                    "clock": "PT11M59.00S",
                    "scoreHome": "0",
                    "scoreAway": "0",
                    "actionType": "jumpball",
                    "subType": "",
                    "description": "Jump Ball",
                    "teamTricode": "LAL",
                },
                {
                    "actionId": 2,
                    "period": 1,
                    "clock": "PT11M45.00S",
                    "scoreHome": "2",
                    "scoreAway": "0",
                    "actionType": "2pt",
                    "subType": "layup",
                    "description": "James Layup (2 PTS)",
                    "personId": 2544,
                    "playerNameI": "James, L.",
                    "teamTricode": "LAL",
                },
                {
                    "actionId": 500,
                    "period": 4,
                    "clock": "PT00M00.00S",
                    "scoreHome": "112",
                    "scoreAway": "109",
                    "actionType": "game",
                    "subType": "end",
                    "description": "Game End",
                },
            ],
        }
    }

    result = create_game_states(games_info)
    assert "0022400002" in result, "Game ID should be in result"
    assert len(result["0022400002"]) == 3, "Should have 3 game states"
    assert result["0022400002"][-1]["is_final_state"], "Last state should be final"
    assert result["0022400002"][0]["home_score"] == 0, "First state score should be 0"
    print("✓ Stats endpoint data parsing works")


def test_missing_fields():
    """Test parsing with missing optional fields."""
    games_info = {
        "0022400003": {
            "home": "MIA",
            "away": "PHI",
            "date_time_utc": "2024-10-23T19:00:00",
            "pbp_logs": [
                {
                    "orderNumber": 1,
                    # Missing scores (should default to 0)
                    "period": 1,
                    "clock": "PT11M59.00S",
                    "actionType": "jumpball",
                    "subType": "",
                    "description": "Jump Ball",
                },
                {
                    "orderNumber": 2,
                    "scoreHome": "2",
                    "scoreAway": "0",
                    # Missing clock (should default)
                    # Missing period (should default)
                    "actionType": "2pt",
                    "description": "Basket",
                },
            ],
        }
    }

    result = create_game_states(games_info)
    assert "0022400003" in result, "Game ID should be in result"
    assert len(result["0022400003"]) == 2, "Should have 2 game states"
    assert result["0022400003"][0]["home_score"] == 0, "Missing score should default to 0"
    assert result["0022400003"][1]["clock"] == "PT00M00.00S", "Missing clock defaults to PT00M00.00S"
    assert result["0022400003"][1]["period"] == 1, "Missing period should default to 1"
    print("✓ Missing fields handled correctly")


def test_empty_logs_after_filtering():
    """Test handling of games where all logs are filtered out."""
    games_info = {
        "0022400004": {
            "home": "DAL",
            "away": "PHX",
            "date_time_utc": "2024-10-23T20:00:00",
            "pbp_logs": [
                {
                    "orderNumber": 1,
                    "period": 1,
                    "clock": "PT11M59.00S",
                    # Missing 'description' field - should be filtered out
                },
                {
                    "orderNumber": 2,
                    "period": 1,
                    "clock": "PT11M45.00S",
                    # Missing 'description' field - should be filtered out
                },
            ],
        }
    }

    result = create_game_states(games_info)
    assert "0022400004" in result, "Game ID should be in result"
    assert result["0022400004"] == [], "Should have empty list when all logs filtered"
    print("✓ Empty logs after filtering handled correctly")


def test_in_progress_game():
    """Test that in-progress games don't get marked as final."""
    games_info = {
        "0022400005": {
            "home": "DEN",
            "away": "LAC",
            "date_time_utc": "2024-10-23T21:00:00",
            "pbp_logs": [
                {
                    "orderNumber": 1,
                    "period": 1,
                    "clock": "PT11M59.00S",
                    "scoreHome": "0",
                    "scoreAway": "0",
                    "actionType": "jumpball",
                    "description": "Jump Ball",
                },
                {
                    "orderNumber": 250,
                    "period": 2,
                    "clock": "PT08M30.00S",
                    "scoreHome": "48",
                    "scoreAway": "52",
                    "actionType": "2pt",
                    "description": "Basket",
                    # Last play but NOT a game end - should NOT be final
                },
            ],
        }
    }

    result = create_game_states(games_info)
    assert "0022400005" in result, "Game ID should be in result"
    assert not result["0022400005"][-1]["is_final_state"], "In-progress game should NOT be final"
    print("✓ In-progress game not marked as final")


def test_player_tracking_live():
    """Test player points tracking with Live endpoint."""
    games_info = {
        "0022400006": {
            "home": "BOS",
            "away": "NYK",
            "date_time_utc": "2024-10-24T19:30:00",
            "pbp_logs": [
                {
                    "orderNumber": 1,
                    "period": 1,
                    "clock": "PT11M45.00S",
                    "scoreHome": "2",
                    "scoreAway": "0",
                    "actionType": "2pt",
                    "description": "Tatum Layup (2 PTS)",
                    "personId": 1627759,
                    "playerNameI": "Tatum, J.",
                    "teamTricode": "BOS",
                    "pointsTotal": 2,
                },
                {
                    "orderNumber": 2,
                    "period": 1,
                    "clock": "PT11M30.00S",
                    "scoreHome": "5",
                    "scoreAway": "0",
                    "actionType": "3pt",
                    "description": "Tatum 3PT (5 PTS)",
                    "personId": 1627759,
                    "playerNameI": "Tatum, J.",
                    "teamTricode": "BOS",
                    "pointsTotal": 5,
                },
            ],
        }
    }

    result = create_game_states(games_info)
    players_data = result["0022400006"][1]["players_data"]
    assert "home" in players_data, "Should have home players"
    assert 1627759 in players_data["home"], "Should track Tatum"
    assert players_data["home"][1627759]["points"] == 5, "Should have 5 points"
    print("✓ Player tracking works with Live endpoint")


def test_player_tracking_stats():
    """Test player points tracking with Stats endpoint (regex parsing)."""
    games_info = {
        "0022400007": {
            "home": "LAL",
            "away": "GSW",
            "date_time_utc": "2024-10-24T22:00:00",
            "pbp_logs": [
                {
                    "actionId": 1,
                    "period": 1,
                    "clock": "PT11M45.00S",
                    "scoreHome": "2",
                    "scoreAway": "0",
                    "actionType": "2pt",
                    "description": "James Layup (2 PTS)",
                    "personId": 2544,
                    "playerNameI": "James, L.",
                    "teamTricode": "LAL",
                },
                {
                    "actionId": 2,
                    "period": 1,
                    "clock": "PT11M30.00S",
                    "scoreHome": "5",
                    "scoreAway": "0",
                    "actionType": "3pt",
                    "description": "James 3PT (5 PTS)",
                    "personId": 2544,
                    "playerNameI": "James, L.",
                    "teamTricode": "LAL",
                },
            ],
        }
    }

    result = create_game_states(games_info)
    players_data = result["0022400007"][1]["players_data"]
    assert "home" in players_data, "Should have home players"
    assert 2544 in players_data["home"], "Should track LeBron"
    assert players_data["home"][2544]["points"] == 5, "Should have 5 points from regex"
    print("✓ Player tracking works with Stats endpoint (regex)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing GameStates Parsing Robustness")
    print("=" * 60 + "\n")

    try:
        test_live_endpoint_data()
        test_stats_endpoint_data()
        test_missing_fields()
        test_empty_logs_after_filtering()
        test_in_progress_game()
        test_player_tracking_live()
        test_player_tracking_stats()

        print("\n" + "=" * 60)
        print("✓ All tests passed! GameStates parsing is robust.")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
