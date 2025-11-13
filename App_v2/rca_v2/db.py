from .config import DB_BACKEND, SQLITE_PATH, RENDER_DB_URL, PGSSLMODE
import pandas as pd

def get_conn():
    if DB_BACKEND == "postgres":
        from sqlalchemy import create_engine
        url = RENDER_DB_URL
        if not url:
            raise RuntimeError("RENDER_DB_URL is not set; cannot connect to Postgres.")
        # Ensure sslmode is present (Render typically requires it)
        sslmode = PGSSLMODE or "require"
        if "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + f"sslmode={sslmode}"
        engine = create_engine(url, pool_pre_ping=True)
        return engine.connect()
    else:
        import sqlite3, os
        # Ensure directory exists for local DB file
        directory = os.path.dirname(SQLITE_PATH)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def param_placeholder(n: int) -> str:
    # psycopg2 uses %s; SQLite uses ?
    token = "%s" if DB_BACKEND == "postgres" else "?"
    return ",".join([token] * n)