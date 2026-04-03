"""Build Exp 5 extended cache with enriched states, positions, and 28 rolling features.

This builds to data/exp5_cache (separate from data/generative_cache used by Exp 1-4).

Usage:
    python scripts/build_exp5_cache.py
    python scripts/build_exp5_cache.py --seasons 2023-2024 2024-2025  # subset for testing
"""

import argparse
import logging

from src.generative.cache_builder import GenerativeCacheBuilder


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build Phase 4 Exp 5 extended cache (enriched states + positions + 28 rolling features)"
    )
    parser.add_argument(
        "--db-path",
        default="data/NBA_AI_full.sqlite",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/exp5_cache",
        help="Output cache directory (default: data/exp5_cache)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to cache (default: 2008-2009 through 2025-2026)",
    )
    args = parser.parse_args()

    seasons = args.seasons or [f"{y}-{y + 1}" for y in range(2008, 2026)]

    builder = GenerativeCacheBuilder(
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        seasons=seasons,
        enrich_states=True,
        n_rolling_features=28,
    )
    builder.build()


if __name__ == "__main__":
    main()
