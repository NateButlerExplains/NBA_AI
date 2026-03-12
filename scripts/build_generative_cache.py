"""Build generative model cache from database."""

import argparse
import logging

from src.generative.cache_builder import GenerativeCacheBuilder


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build Phase 4 generative model cache from database"
    )
    parser.add_argument("--db-path", default="data/NBA_AI_full.sqlite",
                        help="Path to SQLite database")
    parser.add_argument("--cache-dir", default="data/generative_cache",
                        help="Output cache directory")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to cache (default: 2008-2009 through 2025-2026)")
    args = parser.parse_args()

    # If no seasons specified, build all available
    seasons = args.seasons or [f"{y}-{y + 1}" for y in range(2008, 2026)]

    builder = GenerativeCacheBuilder(args.db_path, args.cache_dir, seasons)
    builder.build()


if __name__ == "__main__":
    main()
