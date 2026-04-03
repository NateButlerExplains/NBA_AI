"""
live_predictions.py

Compute live spread and win probability for in-progress NBA games.
Uses the existing logistic win-probability model from prediction_utils
and blends the pre-game spread prediction with the observed margin
based on game progress.
"""

import logging

from src.predictions.prediction_utils import calculate_home_win_prob

logger = logging.getLogger(__name__)

# NBA regulation: 4 quarters x 12 minutes = 48 minutes
REGULATION_MINUTES = 48.0
QUARTER_MINUTES = 12.0
OT_MINUTES = 5.0


def parse_iso_clock(clock_str):
    """
    Parse ISO 8601 duration clock string (e.g., 'PT08M30.00S') into minutes remaining
    in the current period.

    Args:
        clock_str (str): Clock string in PT##M##.##S format.

    Returns:
        float: Minutes remaining in the current period.
    """
    try:
        stripped = clock_str.lstrip("PT").rstrip("S")
        parts = stripped.split("M")
        minutes = int(parts[0])
        seconds = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        return minutes + seconds / 60.0
    except (ValueError, IndexError, AttributeError):
        logger.warning(f"Failed to parse clock string: {clock_str}")
        return 0.0


def compute_minutes_remaining(period, clock_str):
    """
    Compute total minutes remaining in the game from period and clock.

    Args:
        period (int): Current period (1-4 for regulation, 5+ for OT).
        clock_str (str): ISO 8601 clock string for time remaining in current period.

    Returns:
        float: Total minutes remaining in the game (can exceed 0 for OT).
    """
    period_minutes_left = parse_iso_clock(clock_str)

    if period <= 4:
        # Regulation: remaining quarters after current + current period time
        remaining_full_quarters = 4 - period
        return remaining_full_quarters * QUARTER_MINUTES + period_minutes_left
    else:
        # Overtime: only current OT period time remains (no future OTs assumed)
        return period_minutes_left


def compute_game_progress(period, clock_str):
    """
    Compute fraction of game completed (0.0 to 1.0).

    For regulation (periods 1-4): based on 48-minute game.
    For OT: progress can exceed 1.0 but is capped at 0.99 to avoid
    division issues, since the game is still in progress.

    Args:
        period (int): Current period (1-4 for regulation, 5+ for OT).
        clock_str (str): ISO 8601 clock string.

    Returns:
        float: Fraction of game completed, between 0.0 and ~1.0.
    """
    period_minutes_left = parse_iso_clock(clock_str)

    if period <= 4:
        minutes_elapsed = (period - 1) * QUARTER_MINUTES + (
            QUARTER_MINUTES - period_minutes_left
        )
        progress = minutes_elapsed / REGULATION_MINUTES
    else:
        # OT: regulation done + OT time elapsed
        ot_number = period - 4
        ot_elapsed = (ot_number - 1) * OT_MINUTES + (OT_MINUTES - period_minutes_left)
        total_elapsed = REGULATION_MINUTES + ot_elapsed
        total_possible = REGULATION_MINUTES + ot_number * OT_MINUTES
        progress = total_elapsed / total_possible

    return min(max(progress, 0.0), 0.99)


class LivePredictor:
    """Compute live spread and win probability from current game state."""

    def compute_live_prediction(self, game_data):
        """
        Compute live spread and win probability from current game state.

        Args:
            game_data: dict with keys:
                - home_score (int/float): Current home team score
                - away_score (int/float): Current away team score
                - period (int): Current period (1-4 regulation, 5+ OT)
                - clock (str): ISO 8601 clock string (e.g., 'PT08M30.00S')
                - pred_spread (float or None): Pre-game predicted spread
                  (model convention: positive = home advantage)

        Returns:
            dict with:
                - live_spread (float): Blended spread estimate
                - live_win_prob (float): Home team win probability (0 to 1)
                - game_progress (float): Fraction completed (0 to 1)
                - minutes_remaining (float): Minutes left in the game
        """
        home_score = game_data["home_score"]
        away_score = game_data["away_score"]
        period = game_data["period"]
        clock_str = game_data["clock"]
        pred_spread = game_data.get("pred_spread")

        minutes_remaining = compute_minutes_remaining(period, clock_str)
        game_progress = compute_game_progress(period, clock_str)

        # Live win probability from logistic model
        live_win_prob = calculate_home_win_prob(
            home_score, away_score, minutes_remaining
        )

        # Live spread: blend pre-game prediction with observed margin
        current_margin = home_score - away_score
        if pred_spread is not None:
            live_spread = (
                1 - game_progress
            ) * pred_spread + game_progress * current_margin
        else:
            # No pre-game spread available — use current margin only
            live_spread = current_margin

        return {
            "live_spread": round(live_spread, 1),
            "live_win_prob": live_win_prob,
            "game_progress": game_progress,
            "minutes_remaining": round(minutes_remaining, 1),
        }
