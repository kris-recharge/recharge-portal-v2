from .config import DB_BACKEND, SQLITE_PATH, RENDER_DB_URL
import pandas as pd

def get_conn():
    if DB_BACKEND == "postgres":
        from sqlalchemy import create_engine
        url = RENDER_DB_URL
        if url and "sslmode" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        return create_engine(url).connect()
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def param_placeholder(n: int) -> str:
    # For SQLite we use "?", for Postgres via SQLAlchemy we can still pass "?" safely with pandas
    return ",".join(["?"] * n)