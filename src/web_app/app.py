"""
app.py

This module sets up a Flask web application to display NBA game data, including game schedules, team details, and predictions.
It integrates with external APIs to fetch and process data, and utilizes a machine learning predictor for game predictions.

Core Functions:
- create_app(predictor): Initializes and configures the Flask application, including setting up routes and the app secret key.

Routes:
- home(): Renders the home page with the NBA game schedule for a specific date.
- get_game_data(): Fetches game data for a given date or game ID and processes it for display.

Helper Functions:
- add_header(response): Adds headers to the response to prevent caching of the pages.

Usage:
Typically run via a entry point in the root directory of the project.
"""

import logging
import time
from datetime import datetime, timedelta

from flask import Flask, flash, jsonify, render_template, request

from src.config import config
from src.games_api.api import api as api_blueprint
from src.games_api.games import get_games, get_games_for_date
from src.utils import validate_date_format
from src.web_app.game_data_processor import get_user_datetime, process_game_data

# Configuration variables
DB_PATH = config["database"]["path"]
WEB_APP_SECRET_KEY = config["web_app"]["secret_key"]


def create_app(predictor):
    """
    Initializes and configures the Flask application.

    Args:
        predictor (str): A predictor used for generating game predictions.

    Returns:
        Flask: The configured Flask application instance.
    """
    app = Flask(__name__)
    app.secret_key = WEB_APP_SECRET_KEY

    # Store the predictor in the app configuration
    app.config["PREDICTOR"] = predictor

    # Register the API blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")

    @app.route("/")
    def home():
        """
        Renders the home page with NBA game schedule and details for a specific date.

        - Defaults to the current date if no date is provided or if an invalid date is entered.
        - Displays a list of games, including links to detailed game information.

        Returns:
            str: Rendered HTML page of the home screen with games table.
        """
        current_date_local = get_user_datetime(as_eastern_tz=False)
        current_date_str = current_date_local.strftime("%Y-%m-%d")
        query_date_str = request.args.get("date", current_date_str)

        try:
            validate_date_format(query_date_str)
            query_date = datetime.strptime(query_date_str, "%Y-%m-%d")
        except Exception as e:
            flash("Invalid date format. Showing games for today.", "error")
            query_date_str = current_date_str
            query_date = current_date_local

        query_date_display_str = query_date.strftime("%b %d")
        next_date = query_date + timedelta(days=1)
        prev_date = query_date - timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")
        prev_date_str = prev_date.strftime("%Y-%m-%d")

        return render_template(
            "index.html",
            query_date_str=query_date_str,
            query_date_display_str=query_date_display_str,
            prev_date=prev_date_str,
            next_date=next_date_str,
        )

    @app.route("/get-game-data")
    def get_game_data():
        """
        Fetches and processes game data for a given date or game ID.

        - Supports querying by either 'date' or 'game_id'.
        - Retrieves game data directly from games module (no internal HTTP call).

        Returns:
            Response: JSON response containing processed game data or error message.
        """
        start_time = time.perf_counter()
        try:
            predictor = app.config["PREDICTOR"]

            # Determine the type of input (date or game_id)
            if "date" in request.args:
                # Use provided date or default to the current date if not provided
                inbound_query_date_str = request.args.get("date")
                if inbound_query_date_str is None or inbound_query_date_str == "":
                    current_date_local = get_user_datetime(as_eastern_tz=False)
                    query_date_str = current_date_local.strftime("%Y-%m-%d")
                else:
                    query_date_str = inbound_query_date_str

                # Call get_games_for_date directly (no HTTP overhead)
                game_data = get_games_for_date(
                    query_date_str,
                    predictor=predictor,
                    update_predictions=True,
                )

            elif "game_id" in request.args:
                game_id = request.args.get("game_id")
                game_ids = [g.strip() for g in game_id.split(",") if g.strip()]

                # Validate we have at least one game_id
                if not game_ids:
                    return (
                        jsonify({"error": "game_id parameter cannot be empty."}),
                        400,
                    )

                # Call get_games directly (no HTTP overhead)
                game_data = get_games(
                    game_ids,
                    predictor=predictor,
                    update_predictions=True,
                )

            else:
                return (
                    jsonify({"error": "Either 'date' or 'game_id' must be provided."}),
                    400,
                )

            outbound_game_data = process_game_data(game_data)

            elapsed = time.perf_counter() - start_time

            # Summary log line at INFO level (similar style to pipeline stages)
            if "date" in request.args:
                logging.info(
                    f"[Frontend] {query_date_str}: {len(game_data)} games | {elapsed:.1f}s"
                )
            else:
                # game_id request - show the game_id(s)
                game_id_display = (
                    game_ids[0] if len(game_ids) == 1 else f"{len(game_ids)} games"
                )
                logging.info(f"[Frontend] {game_id_display} | {elapsed:.1f}s")

            return jsonify(outbound_game_data)

        except ValueError as e:
            return (
                jsonify({"error": str(e)}),
                400,
            )
        except Exception as e:
            logging.exception("Error in get_game_data")
            return (
                jsonify({"error": f"Unable to fetch game data: {str(e)}"}),
                500,
            )

    @app.after_request
    def add_header(response):
        """
        Adds headers to the response to prevent caching of the pages.

        Args:
            response (Response): The HTTP response object.

        Returns:
            Response: The modified response object with added headers.
        """
        response.headers["Cache-Control"] = "no-store"
        return response

    return app
