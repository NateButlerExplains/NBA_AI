"""
insightbet_api.py

InsightBet-specific API endpoints for the NBA_AI Flask app.
Provides game data, injury reports, and player stats formatted for
the InsightBet React frontend.

Routes:
- GET /api/insightbet/games?date=YYYY-MM-DD  — Games for a date with odds and predictions
- GET /api/insightbet/games/<game_id>        — Detailed game data
- GET /api/insightbet/injuries?date=YYYY-MM-DD — Current injury reports
- GET /api/insightbet/players/<player_id>    — Player stats
"""

import logging
import sqlite3
from datetime import datetime

from flask import Blueprint, jsonify, request

from src.config import config
from src.games_api.games import get_games, get_games_for_date

DB_PATH = config["database"]["path"]
DEFAULT_PREDICTOR = config.get("default_predictor", "Baseline")

insightbet = Blueprint("insightbet", __name__)


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _format_odds(spread_value):
    if spread_value is None:
        return -110
    return int(spread_value)


def _format_game(game_id, raw_game):
    """Convert raw game dict from games_api into InsightBet format."""
    # Extract prediction data (nested dict keyed by predictor name)
    predictions_raw = raw_game.get("predictions") or {}
    prediction = {}
    if isinstance(predictions_raw, dict) and predictions_raw:
        pred_data = next(iter(predictions_raw.values()), {})
        prediction = pred_data if isinstance(pred_data, dict) else {}

    spread = raw_game.get("opening_spread")
    home_ml = -110 if spread is None else (int(-110 - (abs(spread) * 10)) if spread < 0 else int(110 + (abs(spread) * 10)))
    away_ml = -110 if spread is None else (int(110 + (abs(spread) * 10)) if spread < 0 else int(-110 - (abs(spread) * 10)))

    home_win_prob = float(prediction.get("home_win_probability") or 0.5)
    away_win_prob = 1.0 - home_win_prob

    # Extract latest score from game_states
    game_states = raw_game.get("game_states") or []
    latest_state = game_states[-1] if game_states else {}

    return {
        "id": game_id,
        "gameDate": raw_game.get("date_time_utc", ""),
        "status": _map_status(raw_game.get("status", "")),
        "homeTeam": {
            "name": raw_game.get("home_team", ""),
            "abbr": raw_game.get("home_team", "")[:3].upper(),
        },
        "awayTeam": {
            "name": raw_game.get("away_team", ""),
            "abbr": raw_game.get("away_team", "")[:3].upper(),
        },
        "homeScore": latest_state.get("home_score"),
        "awayScore": latest_state.get("away_score"),
        "oddsData": {
            "moneyline": {"home": home_ml, "away": away_ml},
            "spread": {
                "home": spread if spread else 0,
                "away": -spread if spread else 0,
                "value": -110,
            },
            "total": {"over": 220.5, "under": 220.5},
        },
        "predictions": {
            "homeWinProb": round(home_win_prob, 3),
            "awayWinProb": round(away_win_prob, 3),
            "expectedHomeScore": prediction.get("predicted_home_score"),
            "expectedAwayScore": prediction.get("predicted_away_score"),
            "confidence": round(max(home_win_prob, away_win_prob), 2),
            "predictor": next(iter(predictions_raw.keys()), DEFAULT_PREDICTOR),
        },
    }


def _map_status(status_str):
    if not status_str:
        return "upcoming"
    s = str(status_str).lower()
    if "final" in s or "complete" in s:
        return "completed"
    if "progress" in s or "live" in s or "half" in s:
        return "live"
    return "upcoming"


def _get_mock_games():
    """Return mock game data when database is unavailable."""
    return {
        "202404240": {
            "id": "202404240",
            "date_time_utc": "2026-04-24T19:00:00Z",
            "status": "upcoming",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
            "opening_spread": -5.5,
            "predictions": {
                "Baseline": {
                    "home_win_probability": 0.65,
                    "predicted_home_score": 112,
                    "predicted_away_score": 105,
                }
            },
        },
        "202404241": {
            "id": "202404241",
            "date_time_utc": "2026-04-24T21:00:00Z",
            "status": "upcoming",
            "home_team": "Denver Nuggets",
            "away_team": "Miami Heat",
            "opening_spread": 3.0,
            "predictions": {
                "Baseline": {
                    "home_win_probability": 0.58,
                    "predicted_home_score": 108,
                    "predicted_away_score": 103,
                }
            },
        },
    }


@insightbet.route("/insightbet/games", methods=["GET"])
def get_insightbet_games():
    """Return games for a date in InsightBet format."""
    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        raw = get_games_for_date(date_str, predictor=DEFAULT_PREDICTOR)
        # get_games_for_date returns dict keyed by game_id
        games = [_format_game(game_id, g) for game_id, g in raw.items()]
        return jsonify({"games": games, "date": date_str, "count": len(games)})
    except Exception as e:
        logging.warning("Database not available, using mock games for date %s: %s", date_str, str(e))
        # Fallback to mock data when database is unavailable
        raw = _get_mock_games()
        games = [_format_game(game_id, g) for game_id, g in raw.items()]
        return jsonify({"games": games, "date": date_str, "count": len(games), "source": "mock"})


@insightbet.route("/insightbet/games/<game_id>", methods=["GET"])
def get_insightbet_game(game_id):
    """Return detailed game data for a single game."""
    try:
        raw = get_games([game_id], predictor=DEFAULT_PREDICTOR)
        if not raw or game_id not in raw:
            return jsonify({"error": "Game not found"}), 404
        return jsonify({"game": _format_game(game_id, raw[game_id])})
    except Exception as e:
        logging.exception("Error fetching InsightBet game %s", game_id)
        return jsonify({"error": "Failed to fetch game", "detail": str(e)}), 500


@insightbet.route("/insightbet/injuries", methods=["GET"])
def get_insightbet_injuries():
    """Return current injury reports."""
    date_str = request.args.get("date")

    try:
        conn = _get_db()
        cursor = conn.cursor()

        if date_str:
            cursor.execute("""
                SELECT * FROM InjuryReports
                WHERE date(report_timestamp) = date(?)
                ORDER BY report_timestamp DESC
            """, (date_str,))
        else:
            cursor.execute("""
                SELECT * FROM InjuryReports
                ORDER BY report_timestamp DESC
                LIMIT 50
            """)

        rows = cursor.fetchall()
        conn.close()

        injuries = []
        for row in rows:
            d = dict(row)
            status = d.get("status", "").upper()
            severity = "high" if status == "OUT" else ("medium" if status in ("QUESTIONABLE", "DOUBTFUL") else "low")
            injuries.append({
                "playerId": str(d.get("nba_player_id", "")),
                "playerName": d.get("player_name", ""),
                "team": d.get("team", ""),
                "status": status or "UNKNOWN",
                "severity": severity,
                "reason": (d.get("injury_type") or "") + (" - " + d.get("body_part") if d.get("body_part") else ""),
                "returnDate": None,
            })

        return jsonify({"injuries": injuries, "count": len(injuries)})
    except Exception as e:
        logging.exception("Error fetching InsightBet injuries")
        return jsonify({"injuries": [], "count": 0, "note": "Injury data unavailable", "detail": str(e)})


@insightbet.route("/insightbet/players/<player_id>", methods=["GET"])
def get_insightbet_player(player_id):
    """Return player stats for props and analysis."""
    try:
        conn = _get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM Players WHERE player_id = ?
        """, (player_id,))
        player = cursor.fetchone()

        if not player:
            conn.close()
            return jsonify({"error": "Player not found"}), 404

        player_dict = dict(player)

        cursor.execute("""
            SELECT * FROM PlayerBox
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT 10
        """, (player_id,))
        recent = [dict(r) for r in cursor.fetchall()]

        conn.close()

        return jsonify({
            "id": str(player_id),
            "name": player_dict.get("player_name", player_dict.get("name", "")),
            "team": player_dict.get("team_abbreviation", player_dict.get("team", "")),
            "position": player_dict.get("position", ""),
            "recentPerformance": recent[:5],
            "seasonStats": recent[0] if recent else {},
        })
    except Exception as e:
        logging.exception("Error fetching InsightBet player %s", player_id)
        return jsonify({"error": "Failed to fetch player", "detail": str(e)}), 500
