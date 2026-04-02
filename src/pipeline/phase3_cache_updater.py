"""
Phase 3 Cache Updater: detect new games and rebuild the Phase 3 cache.

The Phase 3 transformer cache consists of several interrelated files
(game_features.pt, season_index.pt, gamestates_cache.pt, player_mapping.json,
team_mapping.json). These are tightly coupled -- e.g. player_mapping indices
must be consistent across all games. For the MVP, we detect new games and
trigger a full cache rebuild when needed, rather than attempting incremental
updates that risk index inconsistencies.

Usage:
    from src.pipeline.phase3_cache_updater import Phase3CacheUpdater
    updater = Phase3CacheUpdater()
    new_games = updater.find_new_games()
    if new_games:
        n_added = updater.append_new_games()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.database import get_db

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "NBA_AI_full.sqlite"

# All seasons the Phase 3 cache covers (matches Exp 5 config)
DEFAULT_CACHE_SEASONS = [
    "2008-2009",
    "2009-2010",
    "2010-2011",
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]


class Phase3CacheUpdater:
    """Detect and apply cache updates for the Phase 3 transformer model."""

    def __init__(
        self,
        cache_dir: str = "data/phase3_cache",
        seasons: Optional[list[str]] = None,
        db_path: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.seasons = seasons or DEFAULT_CACHE_SEASONS
        self.db_path = str(db_path) if db_path else str(DB_PATH)
        self._cached_game_ids: Optional[set[str]] = None

    def _load_cached_game_ids(self) -> set[str]:
        """Load the set of game_ids currently in the cache."""
        if self._cached_game_ids is not None:
            return self._cached_game_ids

        features_path = self.cache_dir / "game_features.pt"
        if not features_path.exists():
            logger.info("No existing cache found at %s", self.cache_dir)
            self._cached_game_ids = set()
            return self._cached_game_ids

        try:
            raw = torch.load(features_path, weights_only=False)
            self._cached_game_ids = set(raw.keys())
            logger.info(
                "Loaded %d game IDs from existing cache at %s",
                len(self._cached_game_ids),
                features_path,
            )
        except Exception as e:
            logger.error("Failed to load cache from %s: %s", features_path, e)
            self._cached_game_ids = set()

        return self._cached_game_ids

    def _query_db_game_ids(self) -> set[str]:
        """Query the database for all finalized game IDs in the target seasons.

        Uses Games table only (no GameStates join) since game_data_finalized=1
        already implies GameStates exist. This avoids a slow join on the large
        GameStates table.
        """
        season_placeholders = ",".join(["?"] * len(self.seasons))
        query = f"""
            SELECT game_id
            FROM Games
            WHERE season IN ({season_placeholders})
              AND status = 3
              AND game_data_finalized = 1
              AND season_type IN ('Regular Season', 'Post Season')
        """
        with get_db(self.db_path) as conn:
            rows = conn.execute(query, self.seasons).fetchall()

        db_ids = {row[0] for row in rows}
        logger.info(
            "Found %d finalized games in DB for seasons %s..%s",
            len(db_ids),
            self.seasons[0],
            self.seasons[-1],
        )
        return db_ids

    def find_new_games(self) -> list[str]:
        """
        Find games in the DB that are not yet in the cache.

        Returns:
            Sorted list of game IDs present in DB but missing from cache.
        """
        cached = self._load_cached_game_ids()
        db_ids = self._query_db_game_ids()
        new_ids = sorted(db_ids - cached)

        if new_ids:
            logger.info(
                "Found %d new games not in cache (cache has %d, DB has %d)",
                len(new_ids),
                len(cached),
                len(db_ids),
            )
        else:
            logger.info(
                "Cache is up to date (%d games, DB has %d)",
                len(cached),
                len(db_ids),
            )

        return new_ids

    def is_fresh(self) -> bool:
        """Check whether the cache is up to date with the database."""
        return len(self.find_new_games()) == 0

    def append_new_games(self, dry_run: bool = False) -> int:
        """
        Append new games to the cache. Since the Phase 3 cache files are
        tightly coupled (player_mapping indices, season_index references),
        this triggers a full rebuild when new games are detected.

        Args:
            dry_run: If True, report what would be done without modifying the cache.

        Returns:
            Number of new games that were (or would be) added.
        """
        new_games = self.find_new_games()
        if not new_games:
            logger.info("No new games to add -- cache is already fresh")
            return 0

        if dry_run:
            logger.info(
                "[DRY RUN] Would rebuild cache with %d new games "
                "(total would be %d)",
                len(new_games),
                len(self._load_cached_game_ids()) + len(new_games),
            )
            return len(new_games)

        logger.info(
            "Rebuilding Phase 3 cache with %d new games at %s",
            len(new_games),
            self.cache_dir,
        )

        # Import here to avoid circular imports and heavy module loading at init
        from src.transformer.phase2.cache_builder import build_cache

        result = build_cache(
            seasons=self.seasons,
            cache_dir=str(self.cache_dir),
            db_path=self.db_path,
        )

        n_total = result["n_games"]
        logger.info(
            "Cache rebuild complete: %d total games (%d new)",
            n_total,
            len(new_games),
        )

        # Invalidate the cached game ID set so next call re-reads from disk
        self._cached_game_ids = None

        return len(new_games)

    def get_cache_stats(self) -> dict:
        """Return summary statistics about the current cache state."""
        cached = self._load_cached_game_ids()

        stats = {
            "cache_dir": str(self.cache_dir),
            "n_cached_games": len(cached),
            "cache_exists": (self.cache_dir / "game_features.pt").exists(),
            "has_gamestates": (self.cache_dir / "gamestates_cache.pt").exists(),
            "has_player_mapping": (self.cache_dir / "player_mapping.json").exists(),
            "has_season_index": (self.cache_dir / "season_index.pt").exists(),
        }

        # Check file sizes
        for fname in ["game_features.pt", "gamestates_cache.pt", "season_index.pt"]:
            fpath = self.cache_dir / fname
            if fpath.exists():
                stats[f"{fname}_size_mb"] = round(fpath.stat().st_size / 1e6, 1)

        return stats
