"""
Test player stat predictions from Phase 5 L1 decoder.

Compares predicted stats (from next_game_head) against actual PlayerBox data
for completed games. Reports MAE for key stats: points, rebounds, assists,
minutes, steals, blocks.

Usage:
    python -m scripts.test_player_stat_predictions
    python -m scripts.test_player_stat_predictions --n-games 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db
from src.pipeline.phase5_predictor import Phase5Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"


def load_actual_stats(game_ids: list[str]) -> dict[str, dict[str, dict]]:
    """
    Load actual PlayerBox stats for completed games.

    Returns: {game_id: {"home": {player_id_str: {stat_name: value}},
                         "away": {player_id_str: {...}}}}
    """
    result = {}
    with get_db(str(DB_PATH)) as conn:
        for game_id in game_ids:
            # Get game info
            game = conn.execute(
                "SELECT home_team, away_team FROM Games WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            if not game:
                continue
            home_team, away_team = game

            # Get team_ids for home/away
            home_tid = conn.execute(
                "SELECT team_id FROM Teams WHERE abbreviation = ?",
                (home_team,),
            ).fetchone()
            away_tid = conn.execute(
                "SELECT team_id FROM Teams WHERE abbreviation = ?",
                (away_team,),
            ).fetchone()
            if not home_tid or not away_tid:
                continue

            game_data = {"home": {}, "away": {}}

            # Load all player box stats for this game
            rows = conn.execute(
                """
                SELECT player_id, team_id, min, pts, oreb, dreb, ast, stl, blk,
                       tov, pf, fga, fgm, fg3a, fg3m, fta, ftm, plus_minus,
                       p.full_name
                FROM PlayerBox pb
                LEFT JOIN Players p ON pb.player_id = p.person_id
                WHERE pb.game_id = ? AND pb.min > 0
                """,
                (game_id,),
            ).fetchall()

            for row in rows:
                pid = row[0]
                tid = row[1]
                side = "home" if tid == home_tid[0] else "away"
                game_data[side][str(pid)] = {
                    "minutes": row[2] or 0,
                    "points": row[3] or 0,
                    "oreb": row[4] or 0,
                    "dreb": row[5] or 0,
                    "rebounds": (row[4] or 0) + (row[5] or 0),
                    "assists": row[6] or 0,
                    "steals": row[7] or 0,
                    "blocks": row[8] or 0,
                    "turnovers": row[9] or 0,
                    "fouls": row[10] or 0,
                    "fga": row[11] or 0,
                    "fgm": row[12] or 0,
                    "fg3a": row[13] or 0,
                    "fg3m": row[14] or 0,
                    "fta": row[15] or 0,
                    "ftm": row[16] or 0,
                    "plus_minus": row[17] or 0,
                    "name": row[18] or "",
                }

            result[game_id] = game_data

    return result


def compare_predictions(predictions: dict, actuals: dict, game_ids: list[str]) -> None:
    """Compare predicted vs actual player stats and print report."""
    # Accumulate errors per stat
    stat_keys = [
        ("pred_points", "points"),
        ("pred_rebounds", "rebounds"),
        ("pred_assists", "assists"),
        ("pred_minutes", "minutes"),
        ("pred_steals", "steals"),
        ("pred_blocks", "blocks"),
        ("pred_turnovers", "turnovers"),
        ("pred_fga", "fga"),
        ("pred_fgm", "fgm"),
        ("pred_fg3a", "fg3a"),
        ("pred_fg3m", "fg3m"),
        ("pred_fta", "fta"),
        ("pred_ftm", "ftm"),
    ]

    errors = {sk[0]: [] for sk in stat_keys}
    n_matched = 0
    n_games_with_preds = 0

    for game_id in game_ids:
        if game_id not in predictions or game_id not in actuals:
            continue

        pred = predictions[game_id].get("pred_players", {"home": {}, "away": {}})
        actual = actuals[game_id]

        has_preds = False
        for side in ("home", "away"):
            for pid_str, pred_stats in pred.get(side, {}).items():
                if pid_str in actual.get(side, {}):
                    actual_stats = actual[side][pid_str]
                    n_matched += 1
                    has_preds = True

                    for pred_key, actual_key in stat_keys:
                        p = pred_stats.get(pred_key, 0)
                        a = actual_stats.get(actual_key, 0)
                        errors[pred_key].append(abs(p - a))

        if has_preds:
            n_games_with_preds += 1

    # Print report
    print("\n" + "=" * 70)
    print("PLAYER STAT PREDICTION ACCURACY REPORT")
    print("=" * 70)
    print(f"Games evaluated: {n_games_with_preds}")
    print(f"Player-games matched: {n_matched}")
    print()

    if n_matched == 0:
        print("No player predictions matched actual box scores.")
        return

    print(f"{'Stat':<15} {'MAE':>8} {'Median AE':>10} {'Std':>8} {'N':>6}")
    print("-" * 50)
    for pred_key, actual_key in stat_keys:
        errs = errors[pred_key]
        if errs:
            mae = np.mean(errs)
            med = np.median(errs)
            std = np.std(errs)
            print(f"{actual_key:<15} {mae:8.2f} {med:10.2f} {std:8.2f} {len(errs):6d}")

    # Print some example predictions vs actuals
    print("\n" + "-" * 70)
    print("SAMPLE PREDICTIONS (first 3 games, top 5 players by pred_points)")
    print("-" * 70)

    shown = 0
    for game_id in game_ids:
        if shown >= 3:
            break
        if game_id not in predictions or game_id not in actuals:
            continue
        pred = predictions[game_id].get("pred_players", {"home": {}, "away": {}})
        actual = actuals[game_id]

        if not pred.get("home") and not pred.get("away"):
            continue

        shown += 1
        print(f"\nGame {game_id}:")
        for side in ("home", "away"):
            side_preds = pred.get(side, {})
            side_actual = actual.get(side, {})
            if not side_preds:
                continue

            # Sort by predicted points
            sorted_pids = sorted(
                side_preds.keys(),
                key=lambda p: side_preds[p].get("pred_points", 0),
                reverse=True,
            )

            print(f"  {side.upper()} team:")
            print(
                f"    {'Name':<22} {'PredPts':>7} {'ActPts':>7} "
                f"{'PredReb':>7} {'ActReb':>7} {'PredAst':>7} {'ActAst':>7}"
            )
            for pid in sorted_pids[:5]:
                pp = side_preds[pid]
                ap = side_actual.get(pid, {})
                name = ap.get("name", f"ID:{pid}")[:22]
                print(
                    f"    {name:<22} "
                    f"{pp.get('pred_points', 0):7.1f} "
                    f"{ap.get('points', '?'):>7} "
                    f"{pp.get('pred_rebounds', 0):7.1f} "
                    f"{ap.get('rebounds', '?'):>7} "
                    f"{pp.get('pred_assists', 0):7.1f} "
                    f"{ap.get('assists', '?'):>7}"
                )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test player stat predictions from Phase 5 L1 decoder"
    )
    parser.add_argument(
        "--n-games", type=int, default=10, help="Number of recent games to evaluate"
    )
    parser.add_argument(
        "--game-ids",
        type=str,
        default=None,
        help="Comma-separated game IDs to evaluate (overrides --n-games)",
    )
    args = parser.parse_args()

    # Select games to evaluate
    if args.game_ids:
        game_ids = args.game_ids.split(",")
    else:
        with get_db(str(DB_PATH)) as conn:
            rows = conn.execute(
                """
                SELECT game_id FROM Games
                WHERE status = 3
                AND season = '2024-2025'
                AND season_type = 'Regular Season'
                ORDER BY date_time_utc DESC
                LIMIT ?
                """,
                (args.n_games,),
            ).fetchall()
            game_ids = [r[0] for r in rows]

    logger.info(f"Evaluating player stat predictions for {len(game_ids)} games")

    # Run predictions
    t0 = time.time()
    predictor = Phase5Predictor()
    predictions = predictor.make_pre_game_predictions(game_ids)
    t1 = time.time()
    logger.info(
        f"Predictions complete in {t1 - t0:.1f}s "
        f"({len(predictions)}/{len(game_ids)} games)"
    )

    # Load actual stats
    actuals = load_actual_stats(game_ids)
    logger.info(f"Loaded actual stats for {len(actuals)} games")

    # Compare
    compare_predictions(predictions, actuals, game_ids)


if __name__ == "__main__":
    main()
