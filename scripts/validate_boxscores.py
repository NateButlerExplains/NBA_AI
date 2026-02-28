#!/usr/bin/env python3
"""
validate_boxscores.py

Validate the existing NBA API boxscore collection pipeline.
Runs data quality checks, coverage reports, and optionally fills gaps.

Usage:
    # Run all validation steps
    python scripts/validate_boxscores.py --all

    # Coverage report only
    python scripts/validate_boxscores.py --check-coverage

    # Validate data quality
    python scripts/validate_boxscores.py --validate-data

    # Run unit tests
    python scripts/validate_boxscores.py --run-unit-tests

    # Check finalization logic
    python scripts/validate_boxscores.py --check-finalization

    # Fill gaps for current season via NBA API
    python scripts/validate_boxscores.py --fill-gaps --season 2025-2026
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_db, DB_PATH
from src.database_updater.validators import BoxscoresValidator

logger = logging.getLogger(__name__)


def check_coverage(db_path: str) -> dict:
    """
    Report boxscore coverage by season.

    Returns dict keyed by season with total/covered/finalized counts.
    """
    coverage = {}
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT g.season,
                   COUNT(*) as total_completed,
                   SUM(CASE WHEN pb.game_id IS NOT NULL THEN 1 ELSE 0 END) as has_playerbox,
                   SUM(CASE WHEN tb.game_id IS NOT NULL THEN 1 ELSE 0 END) as has_teambox,
                   SUM(CASE WHEN g.boxscore_data_finalized = 1 THEN 1 ELSE 0 END) as finalized
            FROM Games g
            LEFT JOIN (SELECT DISTINCT game_id FROM PlayerBox) pb ON g.game_id = pb.game_id
            LEFT JOIN (SELECT DISTINCT game_id FROM TeamBox) tb ON g.game_id = tb.game_id
            WHERE g.status = 3
              AND g.season_type IN ('Regular Season', 'Post Season')
            GROUP BY g.season
            ORDER BY g.season
            """
        )
        for season, total, has_pb, has_tb, finalized in cursor.fetchall():
            coverage[season] = {
                "total_completed": total,
                "has_playerbox": has_pb,
                "has_teambox": has_tb,
                "finalized": finalized,
            }

    return coverage


def validate_existing_data(db_path: str) -> dict:
    """
    Run BoxscoresValidator against all games with boxscore data.

    Processes in chunks of 500 to avoid huge IN clauses.
    """
    results = {
        "total_games_checked": 0,
        "critical_issues": [],
        "warnings": [],
        "info": [],
    }

    validator = BoxscoresValidator()

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Get all game_ids that have boxscore data
        cursor.execute("SELECT DISTINCT game_id FROM PlayerBox ORDER BY game_id")
        all_game_ids = [row[0] for row in cursor.fetchall()]

        if not all_game_ids:
            logger.warning("No games with boxscore data found")
            return results

        logger.info(f"Validating {len(all_game_ids)} games with boxscore data...")

        # Process in chunks
        chunk_size = 500
        for i in range(0, len(all_game_ids), chunk_size):
            chunk = all_game_ids[i : i + chunk_size]
            result = validator.validate(chunk, cursor)
            results["total_games_checked"] += result.total_checked

            for issue in result.issues:
                entry = f"[{issue.check_id}] {issue.message} (count={issue.count})"
                if issue.severity.value == "CRITICAL":
                    results["critical_issues"].append(entry)
                elif issue.severity.value == "WARNING":
                    results["warnings"].append(entry)
                else:
                    results["info"].append(entry)

    return results


def run_unit_tests() -> bool:
    """Run pytest on the boxscores test suite. Returns True if all pass."""
    test_file = os.path.join(PROJECT_ROOT, "tests", "test_boxscores_refactor.py")
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False

    logger.info(f"Running pytest on {test_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


def check_finalization(db_path: str, sample_size: int = 20) -> dict:
    """
    Check games with boxscore data but not finalized.

    Runs finalization logic on a sample and reports results.
    """
    from src.database_updater.database_update_manager import (
        _mark_boxscore_games_finalized,
    )

    results = {
        "unfinalized_with_data": 0,
        "sample_checked": 0,
        "newly_finalized": 0,
        "still_unfinalized": 0,
        "reasons": [],
    }

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Find games with boxscore data but not finalized
        cursor.execute(
            """
            SELECT g.game_id
            FROM Games g
            WHERE g.status = 3
              AND g.boxscore_data_finalized = 0
              AND EXISTS (SELECT 1 FROM PlayerBox pb WHERE pb.game_id = g.game_id)
            ORDER BY g.date_time_utc DESC
            """
        )
        unfinalized = [row[0] for row in cursor.fetchall()]
        results["unfinalized_with_data"] = len(unfinalized)

        if not unfinalized:
            logger.info("All games with boxscore data are finalized")
            return results

        # Try finalizing a sample
        sample = unfinalized[:sample_size]
        results["sample_checked"] = len(sample)

        finalized = _mark_boxscore_games_finalized(sample, db_path)
        results["newly_finalized"] = len(finalized)
        results["still_unfinalized"] = len(sample) - len(finalized)

        # Diagnose why unfinalized games weren't finalized
        still_unfin = set(sample) - set(finalized)
        for game_id in list(still_unfin)[:5]:
            cursor.execute(
                """
                SELECT team_id, SUM(min) as total_min
                FROM PlayerBox
                WHERE game_id = ?
                GROUP BY team_id
                """,
                (game_id,),
            )
            rows = cursor.fetchall()
            for team_id, total_min in rows:
                if total_min is not None and total_min < 239:
                    results["reasons"].append(
                        f"{game_id}: team {team_id} has {total_min:.1f} min (need 239+)"
                    )

    return results


def fill_gaps(db_path: str, season: str) -> dict:
    """
    Use existing pipeline to fill boxscore gaps for a season.

    Safety: aborts if more than 200 games need fetching (rate limit protection).
    """
    from src.database_updater.database_update_manager import (
        get_games_needing_boxscores,
        update_boxscore_data,
    )

    results = {"games_needed": 0, "filled": False, "error": None}

    try:
        games = get_games_needing_boxscores(season, db_path)
        results["games_needed"] = len(games)

        if len(games) == 0:
            logger.info(f"No gaps to fill for {season}")
            return results

        if len(games) > 200:
            results["error"] = (
                f"Too many games ({len(games)}) — aborting to protect API rate limits. "
                f"Use the ESPN backfill script for large gaps."
            )
            logger.warning(results["error"])
            return results

        logger.info(f"Filling {len(games)} games for {season} via NBA API...")
        update_boxscore_data(season, db_path)
        results["filled"] = True

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Error filling gaps: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate the NBA API boxscore collection pipeline"
    )
    parser.add_argument("--db-path", default=None, help="Database path (default: from config)")
    parser.add_argument("--check-coverage", action="store_true", help="Show coverage by season")
    parser.add_argument("--validate-data", action="store_true", help="Run BoxscoresValidator")
    parser.add_argument("--run-unit-tests", action="store_true", help="Run pytest suite")
    parser.add_argument("--check-finalization", action="store_true", help="Check finalization logic")
    parser.add_argument("--fill-gaps", action="store_true", help="Fill gaps via NBA API")
    parser.add_argument("--season", default="2025-2026", help="Season for --fill-gaps (default: 2025-2026)")
    parser.add_argument("--all", action="store_true", help="Run all validation steps (except fill-gaps)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = args.db_path if args.db_path else DB_PATH
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Using database: {db_path}")

    run_any = args.all or args.check_coverage or args.validate_data or args.run_unit_tests or args.check_finalization or args.fill_gaps
    if not run_any:
        parser.print_help()
        sys.exit(0)

    exit_code = 0

    # 1. Coverage report
    if args.all or args.check_coverage:
        print("\n" + "=" * 70)
        print("BOXSCORE COVERAGE BY SEASON")
        print("=" * 70)
        coverage = check_coverage(db_path)
        print(f"{'Season':<13} {'Completed':>10} {'PlayerBox':>10} {'TeamBox':>10} {'Finalized':>10} {'Coverage':>10}")
        print("-" * 70)
        total_completed = 0
        total_pb = 0
        for season, data in sorted(coverage.items()):
            pct = (data["has_playerbox"] / data["total_completed"] * 100) if data["total_completed"] > 0 else 0
            print(
                f"{season:<13} {data['total_completed']:>10,} {data['has_playerbox']:>10,} "
                f"{data['has_teambox']:>10,} {data['finalized']:>10,} {pct:>9.1f}%"
            )
            total_completed += data["total_completed"]
            total_pb += data["has_playerbox"]
        print("-" * 70)
        total_pct = (total_pb / total_completed * 100) if total_completed > 0 else 0
        print(f"{'TOTAL':<13} {total_completed:>10,} {total_pb:>10,} {'':>10} {'':>10} {total_pct:>9.1f}%")

    # 2. Data quality validation
    if args.all or args.validate_data:
        print("\n" + "=" * 70)
        print("DATA QUALITY VALIDATION")
        print("=" * 70)
        val_results = validate_existing_data(db_path)
        print(f"Games checked: {val_results['total_games_checked']}")
        if val_results["critical_issues"]:
            print(f"\nCRITICAL ISSUES ({len(val_results['critical_issues'])}):")
            for issue in val_results["critical_issues"]:
                print(f"  {issue}")
            exit_code = 1
        else:
            print("No critical issues found.")
        if val_results["warnings"]:
            print(f"\nWARNINGS ({len(val_results['warnings'])}):")
            for issue in val_results["warnings"][:10]:
                print(f"  {issue}")
            if len(val_results["warnings"]) > 10:
                print(f"  ... and {len(val_results['warnings']) - 10} more")
        if val_results["info"]:
            print(f"\nINFO ({len(val_results['info'])})")

    # 3. Unit tests
    if args.all or args.run_unit_tests:
        print("\n" + "=" * 70)
        print("UNIT TESTS")
        print("=" * 70)
        passed = run_unit_tests()
        if not passed:
            print("UNIT TESTS FAILED")
            exit_code = 1
        else:
            print("All unit tests passed.")

    # 4. Finalization check
    if args.all or args.check_finalization:
        print("\n" + "=" * 70)
        print("FINALIZATION CHECK")
        print("=" * 70)
        fin_results = check_finalization(db_path)
        print(f"Games with data but not finalized: {fin_results['unfinalized_with_data']}")
        if fin_results["sample_checked"] > 0:
            print(f"Sample checked: {fin_results['sample_checked']}")
            print(f"Newly finalized: {fin_results['newly_finalized']}")
            print(f"Still unfinalized: {fin_results['still_unfinalized']}")
            if fin_results["reasons"]:
                print("Reasons for non-finalization:")
                for reason in fin_results["reasons"]:
                    print(f"  {reason}")

    # 5. Fill gaps (opt-in only, not included in --all)
    if args.fill_gaps:
        print("\n" + "=" * 70)
        print(f"FILLING GAPS FOR {args.season}")
        print("=" * 70)
        gap_results = fill_gaps(db_path, args.season)
        print(f"Games needing boxscores: {gap_results['games_needed']}")
        if gap_results["error"]:
            print(f"Error: {gap_results['error']}")
        elif gap_results["filled"]:
            print("Gaps filled successfully.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
