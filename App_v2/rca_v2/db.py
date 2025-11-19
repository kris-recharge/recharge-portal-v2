import os
import sqlite3
from pathlib import Path

import psycopg

# Base dir is App_v2/rca_v2
BASE_DIR = Path(__file__).resolve().parent

# Local fallback: your existing SQLite file for when you run on Wednesday
# (adjust the relative path to match whatever you had before if needed)
SQLITE_PATH = BASE_DIR.parent.parent / "database" / "lynkwell_data.db"

# Render / cloud: use Postgres when DATABASE_URL is set

DATABASE_URL = os.getenv("DATABASE_URL")


def param_placeholder() -> str:
    """
    Return the correct parameter placeholder token for the active backend.

    - For Postgres/psycopg (when DATABASE_URL is set), use %s
    - For local SQLite (no DATABASE_URL), use ?
    """
    if DATABASE_URL:
        # psycopg uses the 'pyformat' style with %s placeholders
        return "%s"
    # sqlite3 uses the 'qmark' style with ? placeholders
    return "?"


def get_conn():
    """
    Return a database connection.

    - On Render (or anywhere DATABASE_URL is set), use Postgres via psycopg.
    - Locally, fall back to the SQLite file so your existing workflow still works.
    """
    if DATABASE_URL:
        # Postgres connection for Render
        return psycopg.connect(DATABASE_URL)

    # Local dev: SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn