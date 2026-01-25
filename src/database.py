"""
database.py

Centralized database connection management for NBA_AI project.

This module provides a single source of truth for database connections,
ensuring consistent configuration across the codebase:
- WAL mode for concurrent read access during writes
- busy_timeout to wait instead of failing on lock contention

Usage:
    from src.database import get_db, create_connection, DB_PATH

    # Using context manager (preferred for simple operations)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Games")

    # Using custom database path (e.g., for tests)
    with get_db(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Games")

    # For functions that accept optional connection parameters
    def my_func(conn=None):
        close_conn = False
        if conn is None:
            conn = create_connection()
            close_conn = True
        try:
            # ... use conn ...
        finally:
            if close_conn:
                conn.close()
"""

import sqlite3
from contextlib import contextmanager

from src.config import config

# Default database path from configuration
DB_PATH = config["database"]["path"]


@contextmanager
def get_db(db_path: str = None):
    """
    Context manager for SQLite database connections.

    Configures SQLite for better concurrent access:
    - WAL mode: Allows concurrent reads during writes
    - busy_timeout: Wait up to 5 seconds if database is locked

    Args:
        db_path: Path to the database file. Defaults to DB_PATH from config.

    Yields:
        sqlite3.Connection: Database connection with WAL mode and busy_timeout set.

    Example:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Games")
            rows = cursor.fetchall()
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Create a database connection with WAL mode and busy_timeout configured.

    Use this for functions that accept an optional connection parameter and
    need to manage the connection lifecycle manually. For simple operations,
    prefer the get_db() context manager.

    Args:
        db_path: Path to the database file. Defaults to DB_PATH from config.

    Returns:
        sqlite3.Connection: Database connection with WAL mode and busy_timeout set.
        Caller is responsible for closing the connection.

    Example:
        def save_data(data, conn=None):
            close_conn = False
            if conn is None:
                conn = create_connection()
                close_conn = True
            try:
                # ... use conn ...
            finally:
                if close_conn:
                    conn.close()
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn
