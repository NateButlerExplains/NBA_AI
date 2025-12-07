"""
database_update_manager.py

Description:
This module handles the core process of updating the database with game data, including schedule updates, play-by-play logs,
game states, prior states, feature sets, and predictions.
It consists of functions to:
- Update the schedule for a given season.
- Update play-by-play logs and game states for games needing updates.
- Update prior states and feature sets for games with incomplete pre-game data.
- Update predictions for games needing updated predictions.

Functions:
    - update_database(season="Current", predictor=None, db_path=DB_PATH): Orchestrates the full update process for the specified season.
    - update_game_data(season, db_path=DB_PATH): Updates play-by-play logs and game states for games needing updates.
    - update_pre_game_data(season, db_path=DB_PATH): Updates prior states and feature sets for games with incomplete pre-game data.
    - update_prediction_data(season, predictor, db_path=DB_PATH): Generates and saves predictions for upcoming games.
    - get_games_needing_game_state_update(season, db_path=DB_PATH): Retrieves game_ids for games needing game state updates.
    - get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH): Retrieves game_ids for games with incomplete pre-game data.
    - get_games_for_prediction_update(season, predictor, db_path=DB_PATH): Retrieves game_ids for games needing updated predictions.
    - main(): Main function to handle command-line arguments and orchestrate the update process.

Usage:
- Typically run as part of a larger data processing pipeline.
- Script can be run directly from the command line (project root) to update the database with the latest game data and predictions.
    python -m src.database_updater.database_update_manager --log_level=DEBUG --season=2023-2024 --predictor=Linear
- Successful execution will update the database with the latest game data and predictions for the specified season.
"""

import argparse
import logging
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from src.config import config
from src.database_updater.betting import update_betting_data
from src.database_updater.boxscores import get_boxscores, save_boxscores
from src.database_updater.game_states import create_game_states, save_game_states
from src.database_updater.nba_official_injuries import update_nba_official_injuries
from src.database_updater.pbp import get_pbp, save_pbp
from src.database_updater.players import update_players
from src.database_updater.prior_states import (
    determine_prior_states_needed,
    load_prior_states,
)
from src.database_updater.schedule import update_schedule
from src.logging_config import setup_logging
from src.predictions.features import create_feature_sets, save_feature_sets
from src.predictions.prediction_manager import (
    make_pre_game_predictions,
    save_predictions,
)
from src.utils import log_execution_time, lookup_basic_game_info

# Configuration
DB_PATH = config["database"]["path"]


@log_execution_time()
def update_database(
    season="Current",
    predictor=None,
    db_path=DB_PATH,
):
    """
    Orchestrates the full update process for the specified season.

    Parameters:
        season (str): The season to update (default is "Current").
        predictor: The prediction model to use (default is None).
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    # STEP 1: Update Schedule
    update_schedule(season)

    # STEP 2: Update Players List
    update_players(db_path)

    # STEP 3: Update Injury Data
    update_injury_data(season, db_path)

    # STEP 4: Update Betting Data
    update_betting_lines(season, db_path)

    # STEP 5: Update Play-by-Play Data
    update_pbp_data(season, db_path)

    # STEP 6: Update Game States (parsed from PbP)
    update_game_state_data(season, db_path)

    # STEP 7: Update Boxscore Data
    update_boxscore_data(season, db_path)

    # STEP 8: Update Pre Game Data (Prior States, Feature Sets)
    update_pre_game_data(season, db_path)

    # STEP 9: Update Predictions
    if predictor:
        update_prediction_data(season, predictor, db_path)


@log_execution_time()
def update_pbp_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Collects play-by-play logs for games needing updates.

    This is Step 5 of the pipeline - fetches raw PBP data from NBA API.
    Automatically uses live endpoint for in-progress games.
    Does NOT parse into GameStates (that's Step 6).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    game_ids = get_games_needing_pbp_update(season, db_path)

    if not game_ids:
        logging.info("No games need PBP data updates.")
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.info(f"Processing {total_games} PBP games in {total_chunks} chunks.")

    pbar = (
        tqdm(total=total_chunks, desc="PBP chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            pbp_data = get_pbp(chunk_game_ids, pbp_endpoint="both")
            save_pbp(pbp_data, db_path)

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(f"Error processing PBP chunk starting at index {i}: {str(e)}")
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    logging.info(
        f"Step 5 complete: Play-by-Play data collected for {total_games} games."
    )


@log_execution_time()
def update_game_state_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Parses play-by-play logs into structured GameStates.

    This is Step 6 of the pipeline - converts raw PBP into game state snapshots.
    Handles both completed and in-progress games.
    Requires PBP data to already exist (Step 5 must run first).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    game_ids = get_games_needing_game_state_update(season, db_path)

    if not game_ids:
        logging.info("No games need GameState updates.")
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.info(
            f"Processing {total_games} GameState games in {total_chunks} chunks."
        )

    pbar = (
        tqdm(total=total_chunks, desc="GameState chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            basic_game_info = lookup_basic_game_info(chunk_game_ids, db_path)

            # Load PBP data for parsing
            pbp_data = {}
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                for game_id in chunk_game_ids:
                    cursor.execute(
                        "SELECT log_data FROM PbP_Logs WHERE game_id = ?", (game_id,)
                    )
                    rows = cursor.fetchall()
                    if rows:
                        import json

                        pbp_data[game_id] = [json.loads(row[0]) for row in rows]

            # Create GameStates from PBP
            game_state_inputs = {
                game_id: {
                    "home": basic_game_info[game_id]["home"],
                    "away": basic_game_info[game_id]["away"],
                    "date_time_est": basic_game_info[game_id]["date_time_est"],
                    "pbp_logs": game_info,
                }
                for game_id, game_info in pbp_data.items()
            }

            game_states = create_game_states(game_state_inputs)
            save_game_states(game_states)

            # Set game_data_finalized flag for games with complete PBP/GameStates
            pbp_finalized = _mark_pbp_games_finalized(chunk_game_ids, db_path)
            if pbp_finalized:
                logging.debug(
                    f"Marked {len(pbp_finalized)} games with PBP/GameStates finalized."
                )

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(
                f"Error processing GameState chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    logging.info(
        f"Step 6 complete: GameStates created and game_data_finalized flags updated."
    )


@log_execution_time()
def update_boxscore_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Collects boxscore data (PlayerBox, TeamBox) for games needing updates.

    This is Step 7 of the pipeline - fetches boxscore statistics from NBA API.
    Independent of PBP/GameStates (can run in parallel if needed).

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    game_ids = get_games_needing_boxscores(season, db_path)

    if not game_ids:
        logging.info("No games need boxscore updates.")
        return

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    if total_chunks > 1:
        logging.info(
            f"Processing {total_games} boxscore games in {total_chunks} chunks."
        )

    pbar = (
        tqdm(total=total_chunks, desc="Boxscore chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in range(0, total_games, chunk_size):
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            boxscore_data = get_boxscores(
                chunk_game_ids, check_game_status=True, db_path=db_path
            )
            save_boxscores(boxscore_data, db_path)

            # Set boxscore_data_finalized flag for games with complete boxscore data
            boxscore_finalized = _mark_boxscore_games_finalized(chunk_game_ids, db_path)
            if boxscore_finalized:
                logging.debug(
                    f"Marked {len(boxscore_finalized)} games with boxscores finalized."
                )

            if pbar:
                pbar.update(1)
        except Exception as e:
            logging.error(
                f"Error processing boxscore chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    logging.info(
        f"Step 7 complete: Boxscores collected and boxscore_data_finalized flags updated."
    )


@log_execution_time()
def update_game_data(season, db_path=DB_PATH, chunk_size=100):
    """
    DEPRECATED: Legacy function that combines PBP, GameStates, and Boxscores.

    Use the individual functions instead:
    - update_pbp_data() for play-by-play collection
    - update_game_state_data() for parsing PBP into states
    - update_boxscore_data() for boxscore collection

    This function is kept for backwards compatibility but will call
    the new modular functions internally.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    logging.warning(
        "update_game_data() is deprecated. Use update_pbp_data(), update_game_state_data(), update_boxscore_data() instead."
    )

    # Call new modular functions
    update_pbp_data(season, db_path, chunk_size)
    update_game_state_data(season, db_path, chunk_size)
    update_boxscore_data(season, db_path, chunk_size)


# Helper functions for determining which games need updates


def get_games_needing_pbp_update(season, db_path):
    """Get list of game_ids needing PBP data collection (includes in-progress games)."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT game_id FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND status IN ('Completed', 'Final', 'In Progress')
            AND game_data_finalized = 0
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs WHERE PbP_Logs.game_id = Games.game_id)
            ORDER BY date_time_est
            """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]
    """
    Updates play-by-play logs and game states for games needing updates.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """

    game_ids = get_games_needing_game_state_update(season, db_path)

    total_games = len(game_ids)
    total_chunks = (
        total_games + chunk_size - 1
    ) // chunk_size  # Ceiling division to calculate total chunks

    # Only log chunk information if there will be more than 1 chunk
    if total_chunks > 1:
        logging.info(f"Processing {total_games} games in {total_chunks} chunks.")

    # Process the games in chunks
    chunk_iterator = range(0, total_games, chunk_size)
    pbar = (
        tqdm(total=total_chunks, desc="Game data chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in chunk_iterator:
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            basic_game_info = lookup_basic_game_info(chunk_game_ids, db_path)
            pbp_data = get_pbp(chunk_game_ids)
            save_pbp(pbp_data, db_path)

            game_state_inputs = {
                game_id: {
                    "home": basic_game_info[game_id]["home"],
                    "away": basic_game_info[game_id]["away"],
                    "date_time_est": basic_game_info[game_id]["date_time_est"],
                    "pbp_logs": game_info,
                }
                for game_id, game_info in pbp_data.items()
            }

            game_states = create_game_states(game_state_inputs)
            save_game_states(game_states)

            # Set game_data_finalized flag for games with complete PBP/GameStates
            pbp_finalized = _mark_pbp_games_finalized(chunk_game_ids, db_path)
            if pbp_finalized:
                logging.debug(
                    f"Marked {len(pbp_finalized)} games with PBP/GameStates finalized."
                )

            # Collect boxscore data for completed games
            boxscore_data = get_boxscores(chunk_game_ids)
            save_boxscores(boxscore_data, db_path)

            # Set boxscore_data_finalized flag for games with complete boxscore data
            boxscore_finalized = _mark_boxscore_games_finalized(chunk_game_ids, db_path)
            if boxscore_finalized:
                logging.debug(
                    f"Marked {len(boxscore_finalized)} games with boxscores finalized."
                )

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            logging.error(f"Error processing chunk starting at index {i}: {str(e)}")
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()

    logging.info(
        f"Stage 3 complete. Updated game_data_finalized and boxscore_data_finalized flags."
    )

    # ADDITIONAL CHECK: Collect boxscores for games that have PBP but no PlayerBox
    # This handles cases where boxscores were added to the pipeline after initial collection
    missing_boxscores = get_games_needing_boxscores_only(season, db_path)

    if missing_boxscores:
        total_missing = len(missing_boxscores)
        total_chunks = (total_missing + chunk_size - 1) // chunk_size

        logging.info(
            f"Found {total_missing} games with PBP but missing PlayerBox. Collecting boxscores..."
        )

        pbar = (
            tqdm(total=total_chunks, desc="Boxscore backfill chunks", unit="chunk")
            if total_chunks > 1
            else None
        )

        for i in range(0, total_missing, chunk_size):
            chunk = missing_boxscores[i : i + chunk_size]

            try:
                boxscore_data = get_boxscores(chunk, check_game_status=False)
                save_boxscores(boxscore_data, db_path)

                # Mark games as finalized now that boxscores are backfilled
                boxscore_finalized = _mark_boxscore_games_finalized(chunk, db_path)
                if boxscore_finalized:
                    logging.debug(
                        f"Marked {len(boxscore_finalized)} backfilled games as finalized."
                    )

                if pbar:
                    pbar.update(1)
            except Exception as e:
                logging.error(f"Error collecting boxscores for chunk: {e}")
                if pbar:
                    pbar.update(1)
                continue

        if pbar:
            pbar.close()


@log_execution_time()
def update_injury_data(season, db_path=DB_PATH):
    """
    Collects NBA Official injury reports for the specified season.

    Smart fetching strategy based on season:
    - Current season: Fetch recent days (yesterday + today) for daily updates
    - Historical seasons: Fetch entire season date range for backfill

    Parameters:
        season (str): The season to update (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    from src.utils import determine_current_season

    current_season = determine_current_season()

    # For current season, just fetch recent days (daily update mode)
    if season == "Current" or season == current_season:
        try:
            count = update_nba_official_injuries(days_back=1, db_path=db_path)
            if count > 0:
                logging.info(f"Injury data: {count} new records for recent days.")
            else:
                logging.debug("Injury data: no new records for recent days.")
        except Exception as e:
            logging.error(f"Error collecting NBA Official injury data: {e}")
        return

    # For historical seasons, determine the full date range
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(date_time_est), MAX(date_time_est)
                FROM Games
                WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
            """,
                (season,),
            )
            result = cursor.fetchone()

            if not result or not result[0]:
                logging.warning(
                    f"No games found for season {season} - skipping injury data"
                )
                return

            first_game = pd.to_datetime(result[0]).date()
            last_game = pd.to_datetime(result[1]).date()
            today = datetime.now().date()

            # Smart fetching: For completed historical seasons, only check recent days
            # For current/recent seasons, check entire range
            season_is_historical = last_game < (today - timedelta(days=30))

            if season_is_historical:
                # Historical season - only check last 7 days for corrections/updates
                days_back = 7
                logging.info(
                    f"Historical season {season}: checking last {days_back} days for injury updates"
                )
            else:
                # Current/recent season - check full range
                end_date = min(last_game, today)
                days_back = (end_date - first_game).days
                logging.info(
                    f"Fetching injury reports for {season} ({first_game} to {end_date}, {days_back} days)"
                )

            count = update_nba_official_injuries(days_back=days_back, db_path=db_path)
            if count > 0:
                logging.info(f"Injury data: {count} new records for {season}.")
            else:
                logging.debug(f"Injury data: no new records for {season}.")

    except Exception as e:
        logging.error(f"Error collecting injury data for {season}: {e}")


@log_execution_time()
def update_betting_lines(season, db_path=DB_PATH):
    """
    Collects betting data (spreads, totals, moneylines) from ESPN DraftKings.

    Tiered fetching strategy:
    - Games > 2 days in future: Skip (not available yet)
    - Games -7 days to +2 days: Fetch from ESPN API (DraftKings odds)
    - Games older than 7 days: Use Covers for backfill

    ESPN provides DraftKings odds approximately 1-2 days before games
    and retains them for about 5-7 days after completion.

    Parameters:
        season (str): The season to update (e.g., "2024-2025" or "Current").
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    from src.utils import determine_current_season

    # Convert "Current" to actual season
    if season == "Current":
        season = determine_current_season()

    try:
        stats = update_betting_data(season=season, use_covers=True)
        if stats["saved"] > 0:
            logging.info(
                f"Betting data complete: {stats['saved']} lines saved, "
                f"{stats['skipped']} skipped."
            )
        else:
            logging.debug("Betting data: no new lines available.")
    except Exception as e:
        logging.error(f"Error collecting betting data: {e}")


@log_execution_time()
def update_pre_game_data(season, db_path=DB_PATH, chunk_size=100):
    """
    Updates prior states and feature sets for games with incomplete pre-game data.

    Parameters:
        season (str): The season to update.
        db_path (str): The path to the database (default is from config).
        chunk_size (int): Number of games to process at a time (default is 100).

    Returns:
        None
    """
    game_ids = get_games_with_incomplete_pre_game_data(season, db_path)

    total_games = len(game_ids)
    total_chunks = (total_games + chunk_size - 1) // chunk_size

    # Only log chunk information if there will be more than 1 chunk
    if total_chunks > 1:
        logging.info(
            f"Processing {total_games} games for pre-game data in {total_chunks} chunks."
        )

    # Process the games in chunks
    chunk_iterator = range(0, total_games, chunk_size)
    pbar = (
        tqdm(total=total_chunks, desc="Pre-game data chunks", unit="chunk")
        if total_chunks > 1
        else None
    )

    for i in chunk_iterator:
        chunk_game_ids = game_ids[i : i + chunk_size]

        try:
            prior_states_needed = determine_prior_states_needed(chunk_game_ids, db_path)
            prior_states_dict = load_prior_states(prior_states_needed, db_path)
            feature_sets = create_feature_sets(prior_states_dict, db_path)
            save_feature_sets(feature_sets, db_path)

            # Categorize games and prepare data for database update
            games_update_data = []
            for game_id, states in prior_states_dict.items():
                pre_game_data_finalized = int(
                    not states["missing_prior_states"]["home"]
                    and not states["missing_prior_states"]["away"]
                )
                games_update_data.append((pre_game_data_finalized, game_id))

            # Update database in a single transaction
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    UPDATE Games
                    SET pre_game_data_finalized = ?
                    WHERE game_id = ?
                """,
                    games_update_data,
                )
                conn.commit()

            # Update progress bar
            if pbar:
                pbar.update(1)

        except Exception as e:
            logging.error(
                f"Error processing pre-game data chunk starting at index {i}: {str(e)}"
            )
            if pbar:
                pbar.update(1)
            continue

    if pbar:
        pbar.close()


@log_execution_time()
def update_prediction_data(season, predictor, db_path=DB_PATH):
    """
    Generates and saves predictions for upcoming games.

    Parameters:
        season (str): The season to update.
        predictor: The prediction model to use.
        db_path (str): The path to the database (default is from config).

    Returns:
        None
    """
    # Get game_ids for games needing updated predictions
    game_ids = get_games_for_prediction_update(season, predictor, db_path)

    # Generate and save predictions
    predictions = make_pre_game_predictions(game_ids, predictor, save=True)


def get_games_needing_boxscores(season, db_path):
    """Get list of game_ids needing boxscore data collection (includes in-progress games)."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT game_id FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND status IN ('Completed', 'Final', 'In Progress')
            AND boxscore_data_finalized = 0
            AND NOT EXISTS (SELECT 1 FROM PlayerBox WHERE PlayerBox.game_id = Games.game_id)
            ORDER BY date_time_est
            """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]


@log_execution_time()
def get_games_needing_game_state_update(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing GameState creation from PBP logs.

    This function looks for games that have PBP data but no GameStates yet.
    Includes in-progress games to enable live updates.

    Parameters:
        season (str): The season to filter games by.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids for games that need GameState parsing.
    """
    with sqlite3.connect(db_path) as db_connection:
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT game_id 
            FROM Games 
            WHERE season = ?
              AND season_type IN ('Regular Season', 'Post Season') 
              AND status IN ('Completed', 'Final', 'In Progress')
              AND game_data_finalized = 0
              AND EXISTS (SELECT 1 FROM PbP_Logs WHERE PbP_Logs.game_id = Games.game_id)
              AND NOT EXISTS (SELECT 1 FROM GameStates WHERE GameStates.game_id = Games.game_id AND is_final_state = 1)
            ORDER BY date_time_est;
        """,
            (season,),
        )

        games_to_update = cursor.fetchall()

    return [game_id for (game_id,) in games_to_update]


@log_execution_time()
def get_games_needing_boxscores_only(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games that have PBP data but are missing boxscores.

    This handles cases where PBP/GameStates succeeded but boxscore collection failed.

    Parameters:
        season (str): The season to check.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need boxscore collection.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            AND g.game_data_finalized = 1
            AND g.boxscore_data_finalized = 0
            ORDER BY g.date_time_est
        """,
            (season,),
        )
        return [row[0] for row in cursor.fetchall()]


@log_execution_time()
def get_games_with_incomplete_pre_game_data(season, db_path=DB_PATH):
    """
    Retrieves game_ids for games with incomplete pre-game data.

    Parameters:
        season (str): The season to filter games by.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need to have their pre_game_data_finalized flag updated.
    """
    query = """
    SELECT game_id
    FROM Games
    WHERE season = ?
      AND season_type IN ("Regular Season", "Post Season")
      AND pre_game_data_finalized = 0
      AND game_data_finalized = 1
      AND (status = 'Completed' OR status = 'In Progress')
    
    UNION

    SELECT g1.game_id
    FROM Games g1
    WHERE g1.season = ?
      AND g1.season_type IN ("Regular Season", "Post Season")
      AND g1.pre_game_data_finalized = 0
      AND g1.status = 'Not Started'
      AND NOT EXISTS (
          SELECT 1
          FROM Games g2
          WHERE g2.season = ?
            AND g2.season_type IN ("Regular Season", "Post Season")
            AND g2.date_time_est < g1.date_time_est
            AND (g2.home_team = g1.home_team OR g2.away_team = g1.home_team OR g2.home_team = g1.away_team OR g2.away_team = g1.away_team)
            AND (g2.game_data_finalized = 0 OR g2.boxscore_data_finalized = 0)
      )
    """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (season, season, season))
        results = cursor.fetchall()

    return [row[0] for row in results]


def _mark_pbp_games_finalized(game_ids, db_path=DB_PATH):
    """
    Marks games as having finalized PBP/GameStates data:
    - PBP_Logs (at least one play)
    - GameStates (with is_final_state=1)

    This is the core play-by-play data that can succeed independently of boxscores.

    Parameters:
        game_ids (list): List of game IDs to check.
        db_path (str): Path to database.

    Returns:
        list: Game IDs that were marked as finalized.
    """
    finalized = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            # Check if PBP and GameStates exist
            cursor.execute(
                """
                SELECT 
                    (SELECT COUNT(*) FROM PbP_Logs WHERE game_id = ?) > 0 as has_pbp,
                    (SELECT COUNT(*) FROM GameStates WHERE game_id = ? AND is_final_state = 1) > 0 as has_final_state
                """,
                (game_id, game_id),
            )

            row = cursor.fetchone()
            has_pbp, has_final_state = row

            if has_pbp and has_final_state:
                cursor.execute(
                    "UPDATE Games SET game_data_finalized = 1 WHERE game_id = ?",
                    (game_id,),
                )
                finalized.append(game_id)

        conn.commit()

    return finalized


def _mark_boxscore_games_finalized(game_ids, db_path=DB_PATH):
    """
    Marks games as having finalized boxscore data:
    - PlayerBox (at least one player)
    - TeamBox (both teams)

    Boxscores are collected separately and can fail independently of PBP/GameStates.

    Parameters:
        game_ids (list): List of game IDs to check.
        db_path (str): Path to database.

    Returns:
        list: Game IDs that were marked as finalized.
    """
    finalized = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            # Check if boxscore data exists
            cursor.execute(
                """
                SELECT 
                    (SELECT COUNT(*) FROM PlayerBox WHERE game_id = ?) > 0 as has_players,
                    (SELECT COUNT(*) FROM TeamBox WHERE game_id = ?) >= 2 as has_teams
                """,
                (game_id, game_id),
            )

            row = cursor.fetchone()
            has_players, has_teams = row

            if has_players and has_teams:
                cursor.execute(
                    "UPDATE Games SET boxscore_data_finalized = 1 WHERE game_id = ?",
                    (game_id,),
                )
                finalized.append(game_id)

        conn.commit()

    return finalized


@log_execution_time()
def get_games_for_prediction_update(season, predictor, db_path=DB_PATH):
    """
    Retrieves game_ids for games needing updated predictions.

    Returns games that have pre_game_data_finalized = 1 but no predictions yet.
    This allows predictions to be generated for past games on-demand when the web
    app requests them.

    Parameters:
        season (str): The season to update.
        predictor (str): The predictor to check for existing predictions.
        db_path (str): The path to the database (default is from config).

    Returns:
        list: A list of game_ids that need updated predictions.
    """
    query = """
        SELECT g.game_id
        FROM Games g
        LEFT JOIN Predictions p ON g.game_id = p.game_id AND p.predictor = ?
        WHERE g.season = ?
            AND g.season_type IN ("Regular Season", "Post Season")
            AND g.pre_game_data_finalized = 1
            AND p.game_id IS NULL
        """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (predictor, season))
        result = cursor.fetchall()

    game_ids = [row[0] for row in result]

    return game_ids


def main():
    """
    Main function to handle command-line arguments and orchestrate the update process.
    """
    parser = argparse.ArgumentParser(
        description="Update the database with the latest game data and predictions."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The logging level. Default is INFO. DEBUG provides more details.",
    )
    parser.add_argument(
        "--season",
        default="Current",
        type=str,
        help="The season to update. Default is 'Current'.",
    )
    parser.add_argument(
        "--predictor",
        default=None,
        type=str,
        help="The predictor to use for predictions.",
    )

    args = parser.parse_args()
    log_level = args.log_level.upper()
    setup_logging(log_level=log_level)

    update_database(
        args.season,
        args.predictor,
    )


if __name__ == "__main__":
    main()
