"""Competing baseline: Poisson scoring model for NBA game prediction.

Estimates per-team scoring rates (λ for +1, +2, +3 events) from a rolling
window of prior games, adjusts for opponent defense and home court, then
runs Monte Carlo simulation to predict final scores.

This establishes a floor that the neural generative model must beat to
justify its complexity.

Usage::

    python scripts/poisson_baseline.py --split test --n-rollouts 10000
    python scripts/poisson_baseline.py --split val --window 20
"""

import argparse
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.database import get_db
from src.generative.cache_builder import HISTORICAL_TO_CURRENT, normalize_team

logger = logging.getLogger(__name__)

DB_PATH = "data/NBA_AI_full.sqlite"

TRAIN_SEASONS = [f"{y}-{y+1}" for y in range(2008, 2023)]
VAL_SEASONS = ["2023-2024"]
TEST_SEASONS = ["2024-2025", "2025-2026"]

# Scoring event point values
POINT_VALUES = [1, 2, 3]


def fetch_games(db_path: str, seasons: list[str]) -> list[dict]:
    """Fetch completed games with scores, ordered chronologically."""
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT g.game_id, g.home_team, g.away_team, g.date_time_utc, g.season
        FROM Games g
        WHERE g.status = 3
          AND g.season IN ({placeholders})
          AND g.game_id NOT LIKE '003%'
        ORDER BY g.date_time_utc
    """
    with get_db(db_path) as conn:
        rows = conn.execute(query, seasons).fetchall()

    # Fetch scores from TeamBox
    game_ids = [r[0] for r in rows]
    scores = _fetch_scores(db_path, game_ids)

    games = []
    for game_id, home, away, date_utc, season in rows:
        home = normalize_team(home)
        away = normalize_team(away)
        s = scores.get(game_id, {})
        h_score = s.get(home, 0)
        a_score = s.get(away, 0)
        if h_score == 0 and a_score == 0:
            continue
        games.append(
            {
                "game_id": game_id,
                "home": home,
                "away": away,
                "date": str(date_utc)[:10],
                "season": season,
                "home_score": h_score,
                "away_score": a_score,
                "margin": h_score - a_score,
            }
        )
    return games


def _fetch_scores(db_path: str, game_ids: list[str]) -> dict:
    """Batch fetch team scores from TeamBox."""
    result = {}
    chunk_size = 500
    with get_db(db_path) as conn:
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT tb.game_id, t.abbreviation, tb.pts
                FROM TeamBox tb
                JOIN Teams t ON tb.team_id = t.team_id
                WHERE tb.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for gid, abbrev, pts in rows:
                abbrev = normalize_team(abbrev)
                if gid not in result:
                    result[gid] = {}
                result[gid][abbrev] = pts or 0
    return result


def fetch_scoring_events(db_path: str, seasons: list[str]) -> dict:
    """Fetch per-game scoring event counts from GameStates.

    Returns {game_id: {team: {1: count, 2: count, 3: count}}}.
    Counts how many +1, +2, +3 scoring events each team had.
    """
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT game_id, home_score, away_score, home, away
        FROM GameStates
        WHERE game_id IN (
            SELECT game_id FROM Games
            WHERE status = 3
              AND season IN ({placeholders})
              AND game_id NOT LIKE '003%'
        )
        AND period <= 4
        ORDER BY game_id, play_id
    """

    logger.info("Fetching scoring events from GameStates...")
    result = {}
    current_game = None
    prev_home = 0
    prev_away = 0
    home_team = None
    away_team = None

    with get_db(db_path) as conn:
        cursor = conn.execute(query, seasons)
        for row in cursor:
            gid, h_score, a_score, home, away = row
            h_score = h_score or 0
            a_score = a_score or 0
            home = normalize_team(home)
            away = normalize_team(away)

            if gid != current_game:
                current_game = gid
                prev_home = 0
                prev_away = 0
                home_team = home
                away_team = away
                result[gid] = {
                    home_team: {1: 0, 2: 0, 3: 0},
                    away_team: {1: 0, 2: 0, 3: 0},
                }

            h_delta = h_score - prev_home
            a_delta = a_score - prev_away

            # Handle score corrections (negative deltas) as no-ops
            if h_delta > 0:
                pts = min(h_delta, 3)
                result[gid][home_team][pts] += 1
            if a_delta > 0:
                pts = min(a_delta, 3)
                result[gid][away_team][pts] += 1

            prev_home = h_score
            prev_away = a_score

    logger.info(f"Processed scoring events for {len(result)} games")
    return result


def compute_team_rates(
    games: list[dict],
    scoring_events: dict,
    window: int = 20,
) -> dict:
    """Compute per-team rolling scoring rates.

    Returns {team: {1: rate, 2: rate, 3: rate}} as events per game,
    updated after each game.
    """
    # Track per-team game history
    team_history: dict[str, list[dict]] = defaultdict(list)

    # Per-game rates (computed before the game, using rolling window)
    game_rates = {}

    for game in games:
        gid = game["game_id"]
        home = game["home"]
        away = game["away"]

        # Compute rates for both teams using prior games
        rates = {}
        for team in [home, away]:
            hist = team_history[team]
            if len(hist) < 3:
                # Not enough history — use league average
                rates[team] = {1: 15.0, 2: 25.0, 3: 10.0}
            else:
                recent = hist[-window:]
                rates[team] = {
                    pts: np.mean([g[pts] for g in recent]) for pts in POINT_VALUES
                }

        game_rates[gid] = rates

        # Update history with this game's actual events
        if gid in scoring_events:
            for team in [home, away]:
                if team in scoring_events[gid]:
                    team_history[team].append(scoring_events[gid][team])

    return game_rates


def compute_league_averages(
    games: list[dict],
    scoring_events: dict,
) -> dict:
    """Compute league-wide average scoring rates for home/away adjustment."""
    home_rates = {1: [], 2: [], 3: []}
    away_rates = {1: [], 2: [], 3: []}

    for game in games:
        gid = game["game_id"]
        if gid not in scoring_events:
            continue
        home = game["home"]
        away = game["away"]
        if home in scoring_events[gid]:
            for pts in POINT_VALUES:
                home_rates[pts].append(scoring_events[gid][home][pts])
        if away in scoring_events[gid]:
            for pts in POINT_VALUES:
                away_rates[pts].append(scoring_events[gid][away][pts])

    return {
        "home": {pts: np.mean(v) for pts, v in home_rates.items()},
        "away": {pts: np.mean(v) for pts, v in away_rates.items()},
        "overall": {
            pts: np.mean(home_rates[pts] + away_rates[pts]) for pts in POINT_VALUES
        },
    }


def simulate_game(
    home_rates: dict[int, float],
    away_rates: dict[int, float],
    opp_def_adj_home: float,
    opp_def_adj_away: float,
    home_court_factor: float,
    n_rollouts: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate a game using Poisson scoring model.

    Args:
        home_rates: {1: λ, 2: λ, 3: λ} home team's scoring event rates per game
        away_rates: {1: λ, 2: λ, 3: λ} away team's scoring event rates per game
        opp_def_adj_home: Opponent defensive adjustment for home team (multiplier)
        opp_def_adj_away: Opponent defensive adjustment for away team (multiplier)
        home_court_factor: Home court scoring multiplier (typically 1.01-1.03)
        n_rollouts: Number of simulations
        rng: Random number generator

    Returns:
        dict with spread_mean, spread_std, win_prob, home_scores, away_scores
    """
    home_scores = np.zeros(n_rollouts)
    away_scores = np.zeros(n_rollouts)

    for pts in POINT_VALUES:
        # Adjusted rates
        h_lambda = max(home_rates[pts] * opp_def_adj_home * home_court_factor, 0.1)
        a_lambda = max(away_rates[pts] * opp_def_adj_away / home_court_factor, 0.1)

        # Sample event counts from Poisson
        h_events = rng.poisson(h_lambda, n_rollouts)
        a_events = rng.poisson(a_lambda, n_rollouts)

        # Add points
        home_scores += h_events * pts
        away_scores += a_events * pts

    spreads = home_scores - away_scores

    return {
        "spread_mean": float(np.mean(spreads)),
        "spread_std": float(np.std(spreads)),
        "win_prob": float(np.mean(spreads > 0)),
        "home_scores": home_scores.tolist(),
        "away_scores": away_scores.tolist(),
    }


def compute_opponent_defense(
    games: list[dict],
    scoring_events: dict,
    league_avg: dict,
    window: int = 20,
) -> dict:
    """Compute per-team defensive adjustment factor.

    defensive_adj = (opponent_scoring_rate_allowed / league_average)
    < 1.0 means good defense (opponents score less), > 1.0 means bad defense.

    Returns {game_id: {team: adj_factor}} for each team playing in each game.
    """
    # Track points allowed per team
    team_pts_allowed: dict[str, list[float]] = defaultdict(list)
    game_adj = {}

    for game in games:
        gid = game["game_id"]
        home = game["home"]
        away = game["away"]

        # Compute defensive adjustment for each team's opponent
        adj = {}
        for team, opp in [(home, away), (away, home)]:
            hist = team_pts_allowed.get(opp, [])
            if len(hist) < 3:
                adj[team] = 1.0  # No adjustment
            else:
                recent_allowed = np.mean(hist[-window:])
                league_mean = sum(
                    league_avg["overall"][pts] * pts for pts in POINT_VALUES
                )
                if league_mean > 0:
                    adj[team] = recent_allowed / league_mean
                else:
                    adj[team] = 1.0

        game_adj[gid] = adj

        # Update points allowed history
        team_pts_allowed[away].append(
            game["home_score"]
        )  # away team allowed home_score
        team_pts_allowed[home].append(
            game["away_score"]
        )  # home team allowed away_score

    return game_adj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poisson scoring baseline for NBA prediction"
    )
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--window", type=int, default=20, help="Rolling window size")
    parser.add_argument(
        "--n-rollouts", type=int, default=10000, help="Monte Carlo rollouts per game"
    )
    parser.add_argument(
        "--home-court", type=float, default=1.02, help="Home court scoring factor"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    eval_seasons = VAL_SEASONS if args.split == "val" else TEST_SEASONS
    all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS

    # Fetch all games chronologically (need history for rolling stats)
    logger.info("Fetching games...")
    all_games = fetch_games(DB_PATH, all_seasons)
    logger.info(f"Total games: {len(all_games)}")

    # Fetch scoring events from GameStates
    scoring_events = fetch_scoring_events(DB_PATH, all_seasons)

    # Compute rolling rates for all games
    logger.info("Computing rolling scoring rates...")
    game_rates = compute_team_rates(all_games, scoring_events, window=args.window)

    # Compute league averages from training data
    train_games = [g for g in all_games if g["season"] in TRAIN_SEASONS]
    league_avg = compute_league_averages(train_games, scoring_events)
    logger.info(
        f"League avg events/game: "
        f"+1={league_avg['overall'][1]:.1f}, "
        f"+2={league_avg['overall'][2]:.1f}, "
        f"+3={league_avg['overall'][3]:.1f}"
    )

    # Compute opponent defensive adjustments
    opp_defense = compute_opponent_defense(
        all_games, scoring_events, league_avg, window=args.window
    )

    # Filter to evaluation split
    eval_games = [g for g in all_games if g["season"] in eval_seasons]
    logger.info(f"Evaluating {args.split} split: {len(eval_games)} games")

    # Run simulations
    all_spreads_pred = []
    all_spreads_true = []
    all_win_probs = []
    all_win_true = []

    for game in tqdm(eval_games, desc="Simulating"):
        gid = game["game_id"]
        if gid not in game_rates:
            continue

        rates = game_rates[gid]
        adj = opp_defense.get(gid, {game["home"]: 1.0, game["away"]: 1.0})

        result = simulate_game(
            home_rates=rates[game["home"]],
            away_rates=rates[game["away"]],
            opp_def_adj_home=adj.get(game["home"], 1.0),
            opp_def_adj_away=adj.get(game["away"], 1.0),
            home_court_factor=args.home_court,
            n_rollouts=args.n_rollouts,
            rng=rng,
        )

        true_margin = game["margin"]
        true_win = 1 if true_margin > 0 else 0

        all_spreads_pred.append(result["spread_mean"])
        all_spreads_true.append(true_margin)
        all_win_probs.append(result["win_prob"])
        all_win_true.append(true_win)

    # Compute metrics
    spreads_pred = np.array(all_spreads_pred)
    spreads_true = np.array(all_spreads_true)
    win_probs = np.array(all_win_probs)
    win_true = np.array(all_win_true)

    mae = float(np.mean(np.abs(spreads_pred - spreads_true)))
    rmse = float(np.sqrt(np.mean((spreads_pred - spreads_true) ** 2)))
    win_acc = float(np.mean((win_probs > 0.5) == win_true))

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(win_true, win_probs))
    except (ValueError, ImportError):
        auc = float("nan")

    # Calibration (10-bin ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (win_probs >= bin_boundaries[i]) & (win_probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_pred = win_probs[mask].mean()
            avg_true = win_true[mask].mean()
            ece += mask.sum() / len(win_probs) * abs(avg_pred - avg_true)

    print(f"\n{'=' * 60}")
    print(f"Poisson Scoring Baseline ({args.split})")
    print(f"{'=' * 60}")
    print(f"Games evaluated:    {len(spreads_pred)}")
    print(f"Rollouts per game:  {args.n_rollouts}")
    print(f"Rolling window:     {args.window}")
    print(f"Home court factor:  {args.home_court}")
    print(f"{'=' * 60}")
    print(f"Spread MAE:         {mae:.2f}")
    print(f"Spread RMSE:        {rmse:.2f}")
    print(f"Win AUC:            {auc:.3f}")
    print(f"Win Accuracy:       {win_acc:.1%}")
    print(f"Win ECE:            {ece:.4f}")
    print(f"{'=' * 60}")
    print()
    print("Comparison targets:")
    print(f"  Phase 3 Exp 9 (ensemble):  MAE 10.66, AUC 0.718, Acc 66.5%")
    print(f"  Phase 4 Exp 4 (generative): MAE 11.76, AUC 0.662, Acc 61.4%")
    print(
        f"  Poisson baseline:           MAE {mae:.2f}, AUC {auc:.3f}, Acc {win_acc:.1%}"
    )


if __name__ == "__main__":
    main()
