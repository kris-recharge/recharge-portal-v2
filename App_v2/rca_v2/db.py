import os
import sqlite3
from pathlib import Path

import psycopg

# Base dir is App_v2/rca_v2
BASE_DIR = Path(__file__).resolve().parent

# Local fallback: your existing SQLite file for when you run locally
SQLITE_PATH = BASE_DIR.parent.parent / "database" / "lynkwell_data.db"


def using_postgres() -> bool:
    """Return True when we should use Postgres."""
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if db_url:
        return True

    # Back-compat: allow explicit APP_MODE=render/cloud
    app_mode = (os.getenv("APP_MODE") or "local").strip().lower()
    return app_mode in {"render", "cloud", "prod", "production"}


def param_placeholder() -> str:
    """Return the correct parameter placeholder token for the active backend."""
    return "%s" if using_postgres() else "?"


def get_conn():
    """Return a DB connection.

    - If DATABASE_URL is set (or APP_MODE indicates cloud), use Postgres via psycopg.
    - Otherwise, fall back to the local SQLite file.
    """
    if using_postgres():
        db_url = (os.getenv("DATABASE_URL") or "").strip()
        if not db_url:
            raise RuntimeError(
                "DATABASE_URL is not set; cannot use Postgres. "
                "Set DATABASE_URL or run with local SQLite."
            )
        return psycopg.connect(db_url)

    # Local dev: SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn