"""
database_evaluator.py

Unified database evaluation and sanity checking for NBA AI.

Combines coverage monitoring, integrity validation, and data quality checks
into a single comprehensive tool that understands the NBA AI data pipeline workflow.

Key Features:
- Coverage: Do we have data for all completed games?
- Integrity: Is the data internally consistent?
- Quality: Are scores/stats reasonable?
- Workflow-aware: Understands pre-game predictions, schedule fetching, etc.

Usage:
    # Quick health check for a season
    python -m src.database_evaluator --season=2024-2025

    # Detailed validation with all checks
    python -m src.database_evaluator --season=2024-2025 --detailed

    # Check specific game
    python -m src.database_evaluator --game_id=0022500354

    # Fix auto-fixable issues
    python -m src.database_evaluator --season=2024-2025 --fix

Categories of Checks:
    1. COVERAGE: Data completeness for completed games
    2. INTEGRITY: Cross-table consistency (PBP ↔ GameStates ↔ Features)
    3. SCORES: Score validation (monotonicity, negatives, consistency)
    4. FLAGS: Finalization flag logic
    5. QUALITY: Reasonable ranges and distributions
"""

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.config import config
from src.logging_config import setup_logging

DB_PATH = config["database"]["path"]


class EvaluationIssue:
    """Represents a single validation issue."""

    def __init__(
        self,
        check_id: str,
        category: str,
        severity: str,  # 'critical', 'warning', 'info'
        message: str,
        count: int = 0,
        sample_data: list = None,
        fixable: bool = False,
    ):
        self.check_id = check_id
        self.category = category
        self.severity = severity
        self.message = message
        self.count = count
        self.sample_data = sample_data or []
        self.fixable = fixable

    def to_dict(self):
        return {
            "check_id": self.check_id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
            "sample_data": self.sample_data[:5],  # Limit to 5 samples
            "fixable": self.fixable,
        }


class DatabaseEvaluator:
    """Main database evaluation orchestrator."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.issues = []
        self.stats = {}

    def evaluate(
        self,
        season: str = None,
        game_id: str = None,
        detailed: bool = False,
    ) -> Dict:
        """
        Run database evaluation.

        Args:
            season: Season to check (e.g., "2024-2025")
            game_id: Specific game to check
            detailed: Run all detailed checks (slower)

        Returns:
            Evaluation report dictionary
        """
        self.issues = []
        self.stats = {}

        if game_id:
            return self._evaluate_game(game_id)
        elif season:
            return self._evaluate_season(season, detailed)
        else:
            raise ValueError("Must provide either season or game_id")

    def _evaluate_game(self, game_id: str) -> Dict:
        """Detailed evaluation of a single game."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get game info
            cursor.execute(
                """
                SELECT game_id, home_team, away_team, date_time_est, status,
                       game_data_finalized, pre_game_data_finalized, 
                       boxscore_data_finalized
                FROM Games WHERE game_id = ?
                """,
                (game_id,),
            )

            row = cursor.fetchone()
            if not row:
                return {"error": f"Game {game_id} not found in Games table"}

            game_info = dict(row)

            # Count data across tables
            data_counts = {}
            for table in [
                "PbP_Logs",
                "GameStates",
                "PlayerBox",
                "TeamBox",
                "Features",
                "Predictions",
            ]:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE game_id = ?", (game_id,)
                )
                data_counts[table] = cursor.fetchone()[0]

            # Check Betting table
            cursor.execute("SELECT COUNT(*) FROM Betting WHERE game_id = ?", (game_id,))
            data_counts["Betting"] = cursor.fetchone()[0]

            # Check InjuryReports (by date, not game_id)
            cursor.execute(
                "SELECT date_time_est FROM Games WHERE game_id = ?", (game_id,)
            )
            game_date_row = cursor.fetchone()
            if game_date_row:
                game_date = game_date_row[0][:10]  # Extract YYYY-MM-DD
                cursor.execute(
                    "SELECT COUNT(*) FROM InjuryReports WHERE report_timestamp = ?",
                    (game_date,),
                )
                data_counts["InjuryReports"] = cursor.fetchone()[0]
            else:
                data_counts["InjuryReports"] = 0

            # Check for final state
            cursor.execute(
                """
                SELECT COUNT(*) FROM GameStates 
                WHERE game_id = ? AND is_final_state = 1
                """,
                (game_id,),
            )
            has_final_state = cursor.fetchone()[0] > 0

            # Score validation for completed games
            score_issues = []
            if game_info["status"] in ("Completed", "Final"):
                score_issues = self._check_game_scores(game_id, cursor)

            return {
                "game_id": game_id,
                "matchup": f"{game_info['away_team']} @ {game_info['home_team']}",
                "date": game_info["date_time_est"],
                "status": game_info["status"],
                "flags": {
                    "game_data_finalized": bool(game_info["game_data_finalized"]),
                    "pre_game_data_finalized": bool(
                        game_info["pre_game_data_finalized"]
                    ),
                    "boxscore_data_finalized": bool(
                        game_info["boxscore_data_finalized"]
                    ),
                },
                "data_counts": data_counts,
                "has_final_state": has_final_state,
                "score_issues": score_issues,
                "quality": "PASS" if len(score_issues) == 0 else "FAIL",
            }

    def _evaluate_season(self, season: str, detailed: bool = False) -> Dict:
        """Evaluate entire season."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total completed games
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM Games
                WHERE season = ?
                AND season_type IN ('Regular Season', 'Post Season')
                AND status IN ('Completed', 'Final')
                """,
                (season,),
            )
            total_completed = cursor.fetchone()[0]

            if total_completed == 0:
                self.issues.append(
                    EvaluationIssue(
                        check_id="season_empty",
                        category="COVERAGE",
                        severity="critical",
                        message=f"Season {season} has 0 completed games - database may be empty or incorrectly filtered",
                        count=0,
                    )
                )
                return {
                    "season": season,
                    "error": "No completed games found",
                    "issues": [issue.to_dict() for issue in self.issues],
                }

            self.stats["total_completed_games"] = total_completed

            # Sanity check: Flag suspiciously low game counts
            # Regular season: ~1,230 games (82 games × 30 teams / 2)
            # Post season: ~90 games (variable, but typically 60-90)
            # Current season (in progress): Variable, but should be proportional to date
            EXPECTED_REGULAR_SEASON_GAMES = 1230
            EXPECTED_PLAYOFF_GAMES = 60  # Minimum

            # Get current date to determine if season is complete
            from datetime import datetime as dt

            current_year = dt.now().year
            season_end_year = int(season.split("-")[1])
            is_historical = season_end_year < current_year

            # Only flag historical seasons with suspiciously low counts
            if is_historical and total_completed < 100:
                self.issues.append(
                    EvaluationIssue(
                        check_id="season_game_count_low",
                        category="COVERAGE",
                        severity="critical",
                        message=f"Historical season {season} has only {total_completed} completed games (expected ~{EXPECTED_REGULAR_SEASON_GAMES} for full season)",
                        count=total_completed,
                    )
                )
            elif is_historical and total_completed < 1000:
                self.issues.append(
                    EvaluationIssue(
                        check_id="season_game_count_incomplete",
                        category="COVERAGE",
                        severity="warning",
                        message=f"Historical season {season} has {total_completed} completed games (expected ~{EXPECTED_REGULAR_SEASON_GAMES} for full regular season)",
                        count=total_completed,
                    )
                )

            # Run coverage checks
            self._check_coverage(season, cursor)

            # Run integrity checks (workflow-aware)
            self._check_integrity(season, cursor)

            # Run score validation on completed games
            if detailed:
                self._check_scores(season, cursor)

            # Run flag validation (workflow-aware)
            self._check_flags(season, cursor)

            # Return evaluation report
            from datetime import datetime

            return {
                "season": season,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": self.stats,
                "issues": [issue.to_dict() for issue in self.issues],
                "summary": {
                    "total_checks": len(self.issues),  # All checks performed
                    "total_issues": len(self.issues),
                    "critical": len(
                        [i for i in self.issues if i.severity == "critical"]
                    ),
                    "warnings": len(
                        [i for i in self.issues if i.severity == "warning"]
                    ),
                    "info": len([i for i in self.issues if i.severity == "info"]),
                },
            }

    def _check_coverage(self, season: str, cursor):
        """Check data coverage for completed games."""
        total = self.stats["total_completed_games"]

        # PBP Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN PbP_Logs p ON g.game_id = p.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        pbp_count = cursor.fetchone()[0]
        self.stats["pbp_coverage"] = f"{pbp_count}/{total} ({pbp_count/total*100:.1f}%)"

        if pbp_count < total * 0.95:
            self.issues.append(
                EvaluationIssue(
                    "COV-001",
                    "COVERAGE",
                    "warning",
                    f"Incomplete PBP coverage: {pbp_count}/{total} games",
                    total - pbp_count,
                )
            )

        # GameStates Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN GameStates gs ON g.game_id = gs.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        states_count = cursor.fetchone()[0]
        self.stats["gamestates_coverage"] = (
            f"{states_count}/{total} ({states_count/total*100:.1f}%)"
        )

        if states_count < total * 0.95:
            self.issues.append(
                EvaluationIssue(
                    "COV-002",
                    "COVERAGE",
                    "warning",
                    f"Incomplete GameStates coverage: {states_count}/{total} games",
                    total - states_count,
                )
            )

        # PlayerBox Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN PlayerBox pb ON g.game_id = pb.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        playerbox_count = cursor.fetchone()[0]
        self.stats["playerbox_coverage"] = (
            f"{playerbox_count}/{total} ({playerbox_count/total*100:.1f}%)"
        )

        if playerbox_count < total * 0.95:
            self.issues.append(
                EvaluationIssue(
                    "COV-003",
                    "COVERAGE",
                    "warning",
                    f"Incomplete PlayerBox coverage: {playerbox_count}/{total} games",
                    total - playerbox_count,
                )
            )

        # TeamBox Coverage
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN TeamBox tb ON g.game_id = tb.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        teambox_count = cursor.fetchone()[0]
        self.stats["teambox_coverage"] = (
            f"{teambox_count}/{total} ({teambox_count/total*100:.1f}%)"
        )

        if teambox_count < total * 0.95:
            self.issues.append(
                EvaluationIssue(
                    "COV-004",
                    "COVERAGE",
                    "warning",
                    f"Incomplete TeamBox coverage: {teambox_count}/{total} games",
                    total - teambox_count,
                )
            )

        # Features Coverage (only for games with prior games available)
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN Features f ON g.game_id = f.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            AND g.pre_game_data_finalized = 1
            """,
            (season,),
        )
        features_count = cursor.fetchone()[0]

        # Total games that should have features (pre_game_data_finalized=1)
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM Games
            WHERE season = ?
            AND season_type IN ('Regular Season', 'Post Season')
            AND status IN ('Completed', 'Final')
            AND pre_game_data_finalized = 1
            """,
            (season,),
        )
        features_expected = cursor.fetchone()[0]

        if features_expected > 0:
            self.stats["features_coverage"] = (
                f"{features_count}/{features_expected} ({features_count/features_expected*100:.1f}%)"
            )

            if features_count < features_expected * 0.95:
                self.issues.append(
                    EvaluationIssue(
                        "COV-005",
                        "COVERAGE",
                        "warning",
                        f"Incomplete Features coverage: {features_count}/{features_expected} finalized games",
                        features_expected - features_count,
                    )
                )

        # Predictions Coverage (all games should have predictions)
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN Predictions p ON g.game_id = p.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        predictions_count = cursor.fetchone()[0]
        self.stats["predictions_coverage"] = (
            f"{predictions_count}/{total} ({predictions_count/total*100:.1f}%)"
        )

        if predictions_count < total * 0.95:
            self.issues.append(
                EvaluationIssue(
                    "COV-006",
                    "COVERAGE",
                    "warning",
                    f"Incomplete Predictions coverage: {predictions_count}/{total} games",
                    total - predictions_count,
                )
            )

        # Betting Coverage (expect ~90% - not all games have lines)
        cursor.execute(
            """
            SELECT COUNT(DISTINCT g.game_id)
            FROM Games g
            INNER JOIN Betting b ON g.game_id = b.game_id
            WHERE g.season = ?
            AND g.season_type IN ('Regular Season', 'Post Season')
            AND g.status IN ('Completed', 'Final')
            """,
            (season,),
        )
        betting_count = cursor.fetchone()[0]
        self.stats["betting_coverage"] = (
            f"{betting_count}/{total} ({betting_count/total*100:.1f}%)"
        )

        if betting_count < total * 0.80:  # Lower threshold for betting
            self.issues.append(
                EvaluationIssue(
                    "COV-007",
                    "COVERAGE",
                    "info",
                    f"Low Betting coverage: {betting_count}/{total} games (expected ~90%)",
                    total - betting_count,
                )
            )

    def _check_integrity(self, season: str, cursor):
        """Check data integrity (workflow-aware - only for finalized games)."""

        # Check: game_data_finalized=1 should have PBP
        cursor.execute(
            """
            SELECT g.game_id, g.home_team, g.away_team
            FROM Games g
            WHERE g.season = ?
            AND g.game_data_finalized = 1
            AND NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = g.game_id)
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "INT-001",
                    "INTEGRITY",
                    "critical",
                    "Games marked finalized but missing PBP data",
                    len(results),
                    [{"game_id": r[0], "teams": f"{r[2]}@{r[1]}"} for r in results],
                    fixable=True,
                )
            )

        # Check: game_data_finalized=1 should have final GameState
        cursor.execute(
            """
            SELECT g.game_id, g.home_team, g.away_team
            FROM Games g
            WHERE g.season = ?
            AND g.game_data_finalized = 1
            AND NOT EXISTS (
                SELECT 1 FROM GameStates gs 
                WHERE gs.game_id = g.game_id AND gs.is_final_state = 1
            )
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "INT-002",
                    "INTEGRITY",
                    "critical",
                    "Games marked finalized but missing final GameState",
                    len(results),
                    [{"game_id": r[0], "teams": f"{r[2]}@{r[1]}"} for r in results],
                    fixable=True,
                )
            )

        # Check: Null critical fields
        cursor.execute(
            """
            SELECT game_id, home_team, away_team
            FROM Games
            WHERE season = ?
            AND (home_team IS NULL OR away_team IS NULL OR date_time_est IS NULL)
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "INT-003",
                    "INTEGRITY",
                    "critical",
                    "Games with NULL critical fields (home_team, away_team, date_time_est)",
                    len(results),
                    [{"game_id": r[0]} for r in results],
                    fixable=False,
                )
            )

    def _check_flags(self, season: str, cursor):
        """Check finalization flag logic (workflow-aware)."""

        # Check: pre_game_data_finalized=1 should have Features
        # (But NOT require game_data_finalized - pre-game happens BEFORE game!)
        cursor.execute(
            """
            SELECT g.game_id, g.home_team, g.away_team, g.status
            FROM Games g
            WHERE g.season = ?
            AND g.pre_game_data_finalized = 1
            AND NOT EXISTS (SELECT 1 FROM Features f WHERE f.game_id = g.game_id)
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "FLAG-001",
                    "FLAGS",
                    "critical",
                    "Games with pre_game_data_finalized=1 but no Features",
                    len(results),
                    [
                        {"game_id": r[0], "teams": f"{r[2]}@{r[1]}", "status": r[3]}
                        for r in results
                    ],
                    fixable=True,
                )
            )

    def _check_scores(self, season: str, cursor):
        """Check score consistency for completed games."""

        # Check: Negative scores
        cursor.execute(
            """
            SELECT gs.game_id, gs.play_id, gs.home_score, gs.away_score
            FROM GameStates gs
            INNER JOIN Games g ON gs.game_id = g.game_id
            WHERE g.season = ?
            AND (gs.home_score < 0 OR gs.away_score < 0)
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "SCORE-001",
                    "SCORES",
                    "critical",
                    "Negative scores detected",
                    len(results),
                    [
                        {"game_id": r[0], "play_id": r[1], "scores": f"{r[2]}-{r[3]}"}
                        for r in results
                    ],
                    fixable=False,
                )
            )

        # Check: Multiple different final scores
        cursor.execute(
            """
            SELECT gs.game_id, 
                   COUNT(DISTINCT gs.home_score || '-' || gs.away_score) as unique_finals
            FROM GameStates gs
            INNER JOIN Games g ON gs.game_id = g.game_id
            WHERE g.season = ?
            AND gs.is_final_state = 1
            GROUP BY gs.game_id
            HAVING COUNT(DISTINCT gs.home_score || '-' || gs.away_score) > 1
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "SCORE-002",
                    "SCORES",
                    "critical",
                    "Games with multiple different final scores",
                    len(results),
                    [{"game_id": r[0], "unique_finals": r[1]} for r in results],
                    fixable=False,
                )
            )

        # Check: Score monotonicity (scores shouldn't decrease)
        # This is expensive, so only sample 100 games
        cursor.execute(
            """
            WITH ScoreChanges AS (
                SELECT 
                    gs.game_id,
                    gs.play_id,
                    gs.home_score - LAG(gs.home_score) OVER (
                        PARTITION BY gs.game_id ORDER BY gs.play_id
                    ) as home_diff,
                    gs.away_score - LAG(gs.away_score) OVER (
                        PARTITION BY gs.game_id ORDER BY gs.play_id
                    ) as away_diff
                FROM GameStates gs
                INNER JOIN Games g ON gs.game_id = g.game_id
                WHERE g.season = ?
            )
            SELECT DISTINCT game_id, play_id, home_diff, away_diff
            FROM ScoreChanges
            WHERE home_diff < 0 OR away_diff < 0
            LIMIT 10
            """,
            (season,),
        )
        results = cursor.fetchall()

        if results:
            self.issues.append(
                EvaluationIssue(
                    "SCORE-003",
                    "SCORES",
                    "critical",
                    "Scores decreased within a game (non-monotonic)",
                    len(results),
                    [
                        {
                            "game_id": r[0],
                            "play_id": r[1],
                            "home_diff": r[2],
                            "away_diff": r[3],
                        }
                        for r in results
                    ],
                    fixable=False,
                )
            )

    def _check_game_scores(self, game_id: str, cursor) -> List[str]:
        """Check score issues for a single game."""
        issues = []

        # Check for negative scores
        cursor.execute(
            """
            SELECT play_id, home_score, away_score
            FROM GameStates
            WHERE game_id = ? AND (home_score < 0 OR away_score < 0)
            """,
            (game_id,),
        )
        if cursor.fetchone():
            issues.append("Negative scores detected")

        # Check score monotonicity
        cursor.execute(
            """
            SELECT play_id, home_score, away_score
            FROM GameStates
            WHERE game_id = ?
            ORDER BY play_id
            """,
            (game_id,),
        )

        prev_home, prev_away = 0, 0
        for play_id, home_score, away_score in cursor.fetchall():
            if home_score < prev_home or away_score < prev_away:
                issues.append(f"Score decreased at play {play_id}")
                break
            prev_home, prev_away = home_score, away_score

        return issues

    def fix_issues(self, season: str = None) -> int:
        """
        Auto-fix issues where possible.

        Currently supports:
        - Resetting incorrect finalization flags

        Returns:
            Number of issues fixed
        """
        fixed = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Fix INT-001: Reset game_data_finalized if no PBP
            if season:
                cursor.execute(
                    """
                    UPDATE Games
                    SET game_data_finalized = 0
                    WHERE season = ?
                    AND game_data_finalized = 1
                    AND NOT EXISTS (SELECT 1 FROM PbP_Logs p WHERE p.game_id = Games.game_id)
                    """,
                    (season,),
                )
                fixed += cursor.rowcount

                # Fix INT-002: Reset game_data_finalized if no final GameState
                cursor.execute(
                    """
                    UPDATE Games
                    SET game_data_finalized = 0
                    WHERE season = ?
                    AND game_data_finalized = 1
                    AND NOT EXISTS (
                        SELECT 1 FROM GameStates gs 
                        WHERE gs.game_id = Games.game_id AND gs.is_final_state = 1
                    )
                    """,
                    (season,),
                )
                fixed += cursor.rowcount

                # Fix FLAG-001: Reset pre_game_data_finalized if no Features
                cursor.execute(
                    """
                    UPDATE Games
                    SET pre_game_data_finalized = 0
                    WHERE season = ?
                    AND pre_game_data_finalized = 1
                    AND NOT EXISTS (SELECT 1 FROM Features f WHERE f.game_id = Games.game_id)
                    """,
                    (season,),
                )
                fixed += cursor.rowcount

            conn.commit()

        self.logger.info(f"Fixed {fixed} issues")
        return fixed


def print_report(report: Dict):
    """Print human-readable evaluation report."""
    print(f"\n{'='*80}")

    if "game_id" in report:
        # Single game report
        print(f"GAME EVALUATION: {report['game_id']}")
        print(f"{'='*80}")
        print(f"Matchup: {report['matchup']}")
        print(f"Date: {report['date']}")
        print(f"Status: {report['status']}")
        print(f"\nFlags:")
        for flag, value in report["flags"].items():
            symbol = "✅" if value else "❌"
            print(f"  {symbol} {flag}: {value}")
        print(f"\nData Counts:")
        for table, count in report["data_counts"].items():
            print(f"  {table}: {count}")
        print(f"\nHas Final State: {report['has_final_state']}")

        if report["score_issues"]:
            print(f"\n❌ SCORE ISSUES:")
            for issue in report["score_issues"]:
                print(f"  - {issue}")
        else:
            print(f"\n✅ NO SCORE ISSUES")

        print(f"\nQuality: {report['quality']}")

    else:
        # Season report
        print(f"DATABASE EVALUATION: {report['season']}")
        print(f"{'='*80}")
        print(f"Generated: {report['timestamp']}")

        if "warning" in report:
            print(f"\n⚠️  {report['warning']}")
            return

        print(f"\nSTATS:")
        for key, value in report["stats"].items():
            print(f"  {key}: {value}")

        print(f"\nSUMMARY:")
        print(f"  Total Checks: {report['summary']['total_checks']}")
        print(f"  Critical Issues: {report['summary']['critical']}")
        print(f"  Warnings: {report['summary']['warnings']}")
        print(f"  Info: {report['summary']['info']}")

        if report["issues"]:
            print(f"\n{'='*80}")
            print("ISSUES FOUND:")
            print(f"{'='*80}\n")

            for issue in report["issues"]:
                severity_symbol = {
                    "critical": "🔴",
                    "warning": "🟡",
                    "info": "🔵",
                }.get(issue["severity"], "⚪")

                fixable = " ✓ Fixable" if issue["fixable"] else " ✗ Manual"

                print(f"{severity_symbol} [{issue['check_id']}] {issue['message']}")
                print(f"   Count: {issue['count']} |{fixable}")

                if issue["sample_data"]:
                    print(f"   Sample: {issue['sample_data'][0]}")
                print()
        else:
            print(f"\n✅ NO ISSUES FOUND")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="NBA AI Database Evaluator - Unified sanity checking tool"
    )
    parser.add_argument(
        "--season", type=str, help="Season to evaluate (e.g., 2024-2025)"
    )
    parser.add_argument("--game_id", type=str, help="Evaluate specific game")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed checks (slower, includes score validation)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible (resets incorrect flags)",
    )
    parser.add_argument("--output", type=str, help="Export report to JSON file")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)

    evaluator = DatabaseEvaluator(DB_PATH)

    # Run evaluation
    if args.fix and not args.season:
        print("Error: --fix requires --season")
        return

    if args.fix:
        print(f"Fixing issues for season {args.season}...")
        fixed = evaluator.fix_issues(args.season)
        print(f"✅ Fixed {fixed} issues")
        print("\nRe-running evaluation...")

    report = evaluator.evaluate(
        season=args.season,
        game_id=args.game_id,
        detailed=args.detailed,
    )

    # Print report
    print_report(report)

    # Export if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to {args.output}")


if __name__ == "__main__":
    main()
