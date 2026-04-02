"""
Pipeline monitoring: tracks runs, logs results, checks staleness.

Creates and manages the PipelineRuns table in the project database.
Used by the PipelineOrchestrator to record each run's outcome.

Usage:
    from src.pipeline.monitoring import PipelineMonitor
    monitor = PipelineMonitor()
    run_id = monitor.start_run("pre_game", "2025-2026")
    # ... do work ...
    monitor.complete_run(run_id, "success", games=8, predictions=8,
                         errors=[], warnings=[], metrics={"mae": 10.1})
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from src.database import DB_PATH, get_db
from src.utils import get_current_eastern_datetime

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Records and queries pipeline run history in the PipelineRuns table."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self._ensure_table()

    def _ensure_table(self):
        """Create PipelineRuns table if it doesn't exist."""
        with get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS PipelineRuns (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    season TEXT,
                    games_processed INTEGER DEFAULT 0,
                    predictions_generated INTEGER DEFAULT 0,
                    errors TEXT,
                    metrics TEXT,
                    warnings TEXT
                )
            """)

    def start_run(self, run_type: str, season: str) -> str:
        """
        Create a run record with status 'running'.

        Args:
            run_type: "post_game" or "pre_game"
            season: e.g. "2025-2026"

        Returns:
            run_id (UUID string)
        """
        run_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO PipelineRuns
                    (run_id, run_type, started_at, status, season, errors, warnings, metrics)
                VALUES (?, ?, ?, 'running', ?, '[]', '[]', '{}')
                """,
                (run_id, run_type, now, season),
            )

        logger.info(f"Pipeline run started: {run_type} [{run_id[:8]}]")
        return run_id

    def complete_run(
        self,
        run_id: str,
        status: str,
        games: int = 0,
        predictions: int = 0,
        errors: list | None = None,
        warnings: list | None = None,
        metrics: dict | None = None,
    ):
        """
        Update a run record on completion.

        Args:
            run_id: The UUID returned by start_run().
            status: "success", "partial", or "failed".
            games: Number of games processed.
            predictions: Number of predictions generated.
            errors: List of error message strings.
            warnings: List of warning message strings.
            metrics: Arbitrary metrics dict (timings, counts, etc.).
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE PipelineRuns
                SET completed_at = ?,
                    status = ?,
                    games_processed = ?,
                    predictions_generated = ?,
                    errors = ?,
                    warnings = ?,
                    metrics = ?
                WHERE run_id = ?
                """,
                (
                    now,
                    status,
                    games,
                    predictions,
                    json.dumps(errors or []),
                    json.dumps(warnings or []),
                    json.dumps(metrics or {}),
                    run_id,
                ),
            )

        logger.info(
            f"Pipeline run completed: {status} [{run_id[:8]}] "
            f"— {games} games, {predictions} predictions"
        )

    def get_recent_runs(self, n: int = 10) -> list[dict]:
        """
        Get the N most recent pipeline runs, newest first.

        Returns:
            List of dicts with all PipelineRuns columns.
        """
        with get_db(self.db_path) as conn:
            conn.row_factory = _dict_factory
            rows = conn.execute(
                """
                SELECT * FROM PipelineRuns
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()

        # Parse JSON fields back into Python objects
        for row in rows:
            for field in ("errors", "warnings", "metrics"):
                if row.get(field) and isinstance(row[field], str):
                    try:
                        row[field] = json.loads(row[field])
                    except (json.JSONDecodeError, TypeError):
                        pass

        return rows

    def check_staleness(self) -> list[str]:
        """
        Check if today's games have predictions. Return warning strings.

        Uses Eastern Time since NBA schedules are in ET.
        """
        warnings = []
        now_et = get_current_eastern_datetime()
        today_str = now_et.strftime("%Y-%m-%d")

        with get_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Count today's upcoming games (status=1 means scheduled)
            cursor.execute(
                """
                SELECT COUNT(*) FROM Games
                WHERE date(date_time_utc) = ?
                AND status = 1
                """,
                (today_str,),
            )
            n_upcoming = cursor.fetchone()[0]

            if n_upcoming == 0:
                return warnings

            # Count how many of those have predictions
            cursor.execute(
                """
                SELECT COUNT(DISTINCT g.game_id) FROM Games g
                JOIN Predictions p ON g.game_id = p.game_id
                WHERE date(g.date_time_utc) = ?
                AND g.status = 1
                """,
                (today_str,),
            )
            n_with_preds = cursor.fetchone()[0]

            if n_with_preds == 0:
                warnings.append(
                    f"No predictions for any of {n_upcoming} games scheduled today ({today_str})"
                )
            elif n_with_preds < n_upcoming:
                warnings.append(
                    f"Only {n_with_preds}/{n_upcoming} games have predictions for today ({today_str})"
                )

            # Check most recent successful pre_game run
            cursor.execute("""
                SELECT started_at FROM PipelineRuns
                WHERE run_type = 'pre_game' AND status IN ('success', 'partial')
                ORDER BY started_at DESC LIMIT 1
                """)
            row = cursor.fetchone()
            if row is None:
                warnings.append("No successful pre_game pipeline run found in history")
            else:
                last_run = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                hours_ago = (datetime.utcnow() - last_run).total_seconds() / 3600
                if hours_ago > 24:
                    warnings.append(
                        f"Last successful pre_game run was {hours_ago:.1f} hours ago"
                    )

        return warnings


def _dict_factory(cursor, row):
    """sqlite3 row_factory that returns dicts."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}
