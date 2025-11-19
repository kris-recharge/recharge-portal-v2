import os
import sqlite3
from pathlib import Path

import psycopg

APP_MODE = os.getenv("APP_MODE", "local")

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

    - For Postgres/psycopg (APP_MODE == 'render'), use %s
    - For local SQLite (APP_MODE != 'render'), use ?
    """
    if APP_MODE == "render":
        return "%s"
    return "?"


def get_conn():
    """
    Return a database connection.

    - On Render (APP_MODE == 'render'), use Postgres via psycopg with DATABASE_URL.
    - Locally, fall back to the SQLite file so your existing workflow still works.
    """
    if APP_MODE == "render":
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set in the environment for Render/APP_MODE=render")
        # Postgres connection for Render
        return psycopg.connect(DATABASE_URL)

    # Local dev: SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn