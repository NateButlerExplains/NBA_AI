"""Deterministic rules engine for NBA game state management.

Consolidates all non-learned game logic into one module:
- Score application (6-class events)
- Clock advancement with monotonicity enforcement
- Game termination (regulation + overtime)
- State vector construction (18-dim)
- Period/clock derivation from game_progress

All public functions operate on batched tensors for efficiency.
"""

import torch

# ---------------------------------------------------------------------------
# Score delta table: 6-class (no game_end — termination is deterministic)
# Class mapping: {0:h+1, 1:h+2, 2:h+3, 3:a+1, 4:a+2, 5:a+3}
# ---------------------------------------------------------------------------
SCORE_DELTAS_6CLASS = torch.tensor(
    [
        [1, 0],  # 0: home+1
        [2, 0],  # 1: home+2
        [3, 0],  # 2: home+3
        [0, 1],  # 3: away+1
        [0, 2],  # 4: away+2
        [0, 3],  # 5: away+3
    ],
    dtype=torch.float32,
)

# Constants
REGULATION_PERIODS = 4
REGULATION_PERIOD_SECONDS = 720.0  # 12 minutes
OT_PERIOD_SECONDS = 300.0  # 5 minutes
MAX_OT_PERIODS = 3  # Cap rollout at 3 OT periods
TOTAL_REGULATION_SECONDS = REGULATION_PERIODS * REGULATION_PERIOD_SECONDS  # 2880

# Normalization constants (match dataset.py / rollout.py conventions)
SCORE_NORM = 150.0
MARGIN_NORM = 50.0
TOTAL_NORM = 300.0
SCORING_RUN_CAP = 5.0
ACTIVE_SCORER_NORM = 15.0
TOP_SCORER_NORM = 50.0
INTER_EVENT_TIME_NORM = 120.0  # seconds, for normalization


def apply_event(
    home_scores: torch.Tensor,
    away_scores: torch.Tensor,
    event_class: torch.Tensor,
    scoring_run_home: torch.Tensor,
    scoring_run_away: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a 6-class scoring event to cumulative scores and scoring runs.

    Args:
        home_scores: (B,) cumulative home points.
        away_scores: (B,) cumulative away points.
        event_class: (B,) int64, class indices 0-5.
        scoring_run_home: (B,) consecutive home-only scoring events.
        scoring_run_away: (B,) consecutive away-only scoring events.

    Returns:
        Tuple of (home_scores, away_scores, scoring_run_home, scoring_run_away),
        all updated in-place friendly (new tensors returned).
    """
    deltas = SCORE_DELTAS_6CLASS.to(home_scores.device)  # (6, 2)
    # Clamp event_class to valid range to avoid index errors
    clamped = event_class.clamp(0, 5).long()
    event_deltas = deltas[clamped]  # (B, 2)

    home_delta = event_deltas[:, 0]
    away_delta = event_deltas[:, 1]

    new_home = home_scores + home_delta
    new_away = away_scores + away_delta

    # Update scoring runs
    home_scored = home_delta > 0  # (B,) bool
    away_scored = away_delta > 0  # (B,) bool

    new_run_home = torch.where(
        home_scored,
        scoring_run_home + 1,
        torch.where(away_scored, torch.zeros_like(scoring_run_home), scoring_run_home),
    )
    new_run_away = torch.where(
        away_scored,
        scoring_run_away + 1,
        torch.where(home_scored, torch.zeros_like(scoring_run_away), scoring_run_away),
    )

    return new_home, new_away, new_run_home, new_run_away


def advance_clock(
    game_progress: torch.Tensor,
    predicted: torch.Tensor,
    is_delta: bool = False,
    min_delta: float = 0.001,
) -> torch.Tensor:
    """Advance game clock with monotonicity enforcement.

    Args:
        game_progress: (B,) current progress in [0, 1].
        predicted: (B,) model's predicted next progress (absolute mode)
            or positive delta increment (delta mode).
        is_delta: if True, ``predicted`` is a positive increment to add.
        min_delta: minimum delta per step (safety floor, delta mode only).

    Returns:
        (B,) new game_progress, guaranteed >= current and in [0, 1].
    """
    if is_delta:
        delta = predicted.clamp(min=min_delta)
        return (game_progress + delta).clamp(0.0, 1.0)
    new_progress = torch.max(predicted, game_progress)
    return new_progress.clamp(0.0, 1.0)


def check_game_end(
    game_progress: torch.Tensor,
    home_scores: torch.Tensor,
    away_scores: torch.Tensor,
    is_overtime: torch.Tensor,
    ot_period: torch.Tensor,
    ot_progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check game termination conditions and handle overtime transitions.

    Regulation ends when game_progress >= 1.0 AND scores are not tied.
    If tied at regulation end, overtime begins.
    OT ends when OT period is complete AND scores are not tied.

    Args:
        game_progress: (B,) regulation progress in [0, 1].
        home_scores: (B,) cumulative home points.
        away_scores: (B,) cumulative away points.
        is_overtime: (B,) bool, whether currently in overtime.
        ot_period: (B,) int, current OT period number (1-based, 0 if not OT).
        ot_progress: (B,) float, progress within current OT period [0, 1].

    Returns:
        Tuple of (active_mask, is_overtime, ot_period, ot_progress):
        - active_mask: (B,) bool, True if game should continue.
        - is_overtime: (B,) bool, updated overtime status.
        - ot_period: (B,) int, updated OT period.
        - ot_progress: (B,) float, updated OT progress.
    """
    tied = home_scores == away_scores
    B = game_progress.shape[0]
    device = game_progress.device
    active = torch.ones(B, dtype=torch.bool, device=device)

    # --- Regulation end ---
    reg_ended = (game_progress >= 1.0) & ~is_overtime
    reg_ended_tied = reg_ended & tied
    reg_ended_decided = reg_ended & ~tied

    # Decided regulation games end
    active[reg_ended_decided] = False

    # Tied regulation games go to overtime
    new_is_overtime = is_overtime.clone()
    new_ot_period = ot_period.clone()
    new_ot_progress = ot_progress.clone()

    new_is_overtime[reg_ended_tied] = True
    new_ot_period[reg_ended_tied] = 1
    new_ot_progress[reg_ended_tied] = 0.0

    # --- Overtime end ---
    ot_period_done = is_overtime & (new_ot_progress >= 1.0)
    ot_decided = ot_period_done & ~tied
    ot_still_tied = ot_period_done & tied

    # Decided OT games end
    active[ot_decided] = False

    # Still tied after OT: start next OT period (up to cap)
    can_extend = ot_still_tied & (new_ot_period < MAX_OT_PERIODS)
    new_ot_period[can_extend] = new_ot_period[can_extend] + 1
    new_ot_progress[can_extend] = 0.0

    # Hit OT cap while still tied: force end (rare edge case)
    capped = ot_still_tied & (new_ot_period >= MAX_OT_PERIODS)
    active[capped] = False

    return active, new_is_overtime, new_ot_period, new_ot_progress


def derive_period_clock(
    game_progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive period number and normalized clock from game_progress.

    Args:
        game_progress: (B,) in [0, 1] for regulation.

    Returns:
        Tuple of:
        - period: (B,) float, quarter number (1-4).
        - clock_norm: (B,) float, normalized clock within quarter (1.0=start, 0.0=end).
    """
    # Quarter index: 0-3
    quarter_index = (game_progress * 4).clamp(0, 3.99).floor()
    # Fractional progress within quarter
    quarter_frac = game_progress * 4 - quarter_index
    period = quarter_index + 1  # 1-4
    clock_norm = 1.0 - quarter_frac  # 1.0 at quarter start, 0.0 at quarter end
    return period, clock_norm


def derive_clock_seconds(
    game_progress: torch.Tensor,
    is_overtime: torch.Tensor,
    ot_progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive period number and clock in seconds from progress values.

    Args:
        game_progress: (B,) regulation progress [0, 1].
        is_overtime: (B,) bool.
        ot_progress: (B,) OT progress within current period [0, 1].

    Returns:
        Tuple of:
        - period: (B,) int, period number (1-4 reg, 5+ OT).
        - clock_seconds: (B,) float, seconds remaining in current period.
    """
    # Regulation
    reg_period, reg_clock_norm = derive_period_clock(game_progress)
    reg_clock_seconds = reg_clock_norm * REGULATION_PERIOD_SECONDS

    # For OT games, period is 4 + ot_period_number, clock from ot_progress
    # (ot_period is tracked externally; here we just use ot_progress for clock)
    ot_clock_seconds = (1.0 - ot_progress.clamp(0, 1)) * OT_PERIOD_SECONDS

    period = torch.where(is_overtime, torch.full_like(reg_period, 5.0), reg_period)
    clock_seconds = torch.where(is_overtime, ot_clock_seconds, reg_clock_seconds)

    return period, clock_seconds


def build_state_vector(
    period: torch.Tensor,
    clock_seconds: torch.Tensor,
    game_progress: torch.Tensor,
    home_scores: torch.Tensor,
    away_scores: torch.Tensor,
    inter_event_time: torch.Tensor,
    player_features: torch.Tensor,
    dynamics_features: torch.Tensor,
) -> torch.Tensor:
    """Construct the normalized 18-dim state vector.

    Layout:
      [0]  period_norm          = period / 4.0
      [1]  clock_norm           = clock_seconds / 720.0 (reg) or / 300.0 (OT)
      [2]  game_progress        = raw [0, 1]
      [3]  home_score_norm      = home_scores / 150.0
      [4]  away_score_norm      = away_scores / 150.0
      [5]  margin_norm          = (home - away) / 50.0
      [6]  total_norm           = (home + away) / 300.0
      [7]  inter_event_time     = pre-normalized (seconds / 120.0)
      [8:14] player_features    = 6 features (pre-normalized)
      [14:18] dynamics_features = 4 features (pre-normalized)

    Args:
        period: (B,) period number (1-4+ float).
        clock_seconds: (B,) seconds remaining in current period.
        game_progress: (B,) in [0, 1].
        home_scores: (B,) cumulative home points.
        away_scores: (B,) cumulative away points.
        inter_event_time: (B,) normalized inter-event time.
        player_features: (B, 6) normalized player scoring features.
        dynamics_features: (B, 4) normalized dynamics features.

    Returns:
        (B, 18) normalized state vector.
    """
    B = period.shape[0]
    device = period.device

    state = torch.zeros(B, 18, device=device, dtype=period.dtype)

    state[:, 0] = period / 4.0
    state[:, 1] = clock_seconds / REGULATION_PERIOD_SECONDS
    state[:, 2] = game_progress
    state[:, 3] = home_scores / SCORE_NORM
    state[:, 4] = away_scores / SCORE_NORM
    state[:, 5] = (home_scores - away_scores) / MARGIN_NORM
    state[:, 6] = (home_scores + away_scores) / TOTAL_NORM
    state[:, 7] = inter_event_time
    state[:, 8:14] = player_features
    state[:, 14:18] = dynamics_features

    return state


def build_dynamics_features(
    game_progress: torch.Tensor,
    period: torch.Tensor,
    clock_seconds: torch.Tensor,
    margin: torch.Tensor,
    scoring_run_home: torch.Tensor,
    scoring_run_away: torch.Tensor,
) -> torch.Tensor:
    """Compute the 4 game dynamics features (pre-normalized).

    Layout:
      [0] clutch_indicator: 1.0 if period >= 4, clock < 120s, |margin| <= 10
      [1] scoring_run_home: capped at 5, / 5.0
      [2] scoring_run_away: capped at 5, / 5.0
      [3] is_second_half: 1.0 if period >= 3

    Args:
        game_progress: (B,) — unused here but kept for API consistency.
        period: (B,) float, period number.
        clock_seconds: (B,) seconds remaining in current period.
        margin: (B,) raw margin (home - away).
        scoring_run_home: (B,) raw consecutive home scoring events.
        scoring_run_away: (B,) raw consecutive away scoring events.

    Returns:
        (B, 4) normalized dynamics features.
    """
    B = period.shape[0]
    device = period.device

    dynamics = torch.zeros(B, 4, device=device, dtype=period.dtype)

    # Clutch indicator
    clutch = (
        (period >= 4).float()
        * (clock_seconds < 120.0).float()
        * (margin.abs() <= 10.0).float()
    )
    dynamics[:, 0] = clutch

    # Scoring runs (capped and normalized)
    dynamics[:, 1] = scoring_run_home.clamp(max=SCORING_RUN_CAP) / SCORING_RUN_CAP
    dynamics[:, 2] = scoring_run_away.clamp(max=SCORING_RUN_CAP) / SCORING_RUN_CAP

    # Second half indicator
    dynamics[:, 3] = (period >= 3).float()

    return dynamics


def build_player_features_zeros(batch_size: int, device: torch.device) -> torch.Tensor:
    """Build zeroed player features for rollout initialization.

    During rollout, player scoring features are tracked externally and
    injected. This helper produces the correct-shape zero tensor for
    game start or when player data is unavailable.

    Returns:
        (B, 6) zero tensor.
    """
    return torch.zeros(batch_size, 6, device=device)


def normalize_player_features(
    top1_home_pts: torch.Tensor,
    top1_away_pts: torch.Tensor,
    hhi_home: torch.Tensor,
    hhi_away: torch.Tensor,
    active_scorers_home: torch.Tensor,
    active_scorers_away: torch.Tensor,
) -> torch.Tensor:
    """Normalize raw player scoring features into the 6-dim vector.

    Args:
        top1_home_pts: (B,) top scorer cumulative points, home team.
        top1_away_pts: (B,) top scorer cumulative points, away team.
        hhi_home: (B,) Herfindahl index of home scoring concentration.
        hhi_away: (B,) Herfindahl index of away scoring concentration.
        active_scorers_home: (B,) count of home players with pts > 0.
        active_scorers_away: (B,) count of away players with pts > 0.

    Returns:
        (B, 6) normalized player features.
    """
    B = top1_home_pts.shape[0]
    device = top1_home_pts.device

    features = torch.zeros(B, 6, device=device, dtype=top1_home_pts.dtype)
    features[:, 0] = top1_home_pts / TOP_SCORER_NORM
    features[:, 1] = top1_away_pts / TOP_SCORER_NORM
    features[:, 2] = hhi_home  # Already in [0, 1]
    features[:, 3] = hhi_away  # Already in [0, 1]
    features[:, 4] = active_scorers_home / ACTIVE_SCORER_NORM
    features[:, 5] = active_scorers_away / ACTIVE_SCORER_NORM

    return features
