import os
import sqlite3
import socket
from urllib.parse import urlparse, parse_qs, unquote
from pathlib import Path

import psycopg

# Base dir is App_v2/rca_v2
BASE_DIR = Path(__file__).resolve().parent

# Local fallback: your existing SQLite file for when you run locally
SQLITE_PATH = BASE_DIR.parent.parent / "database" / "lynkwell_data.db"


def _resolve_ipv4(hostname: str) -> str | None:
    """Resolve a hostname to an IPv4 address (A record). Returns None if not found."""
    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not infos:
            return None
        # sockaddr is (address, port)
        return infos[0][4][0]
    except Exception:
        return None


def _url_to_conninfo(db_url: str) -> str:
    """Convert a postgres URL into a libpq-style conninfo string.

    We also add hostaddr=<ipv4> when possible to avoid IPv6-only connection attempts.
    """
    u = urlparse(db_url)
    hostname = u.hostname or ""
    port = u.port or 5432

    user = unquote(u.username) if u.username else ""
    password = unquote(u.password) if u.password else ""

    dbname = (u.path or "").lstrip("/")

    q = parse_qs(u.query or "")
    sslmode = (q.get("sslmode", [""])[0] or "require")

    hostaddr = _resolve_ipv4(hostname) if hostname else None

    parts = []
    if hostname:
        parts.append(f"host={hostname}")
    if hostaddr:
        parts.append(f"hostaddr={hostaddr}")
    parts.append(f"port={port}")
    if dbname:
        parts.append(f"dbname={dbname}")
    if user:
        parts.append(f"user={user}")
    if password:
        # libpq conninfo requires escaping spaces/backslashes; our passwords are typically URL-safe
        parts.append(f"password={password}")
    if sslmode:
        parts.append(f"sslmode={sslmode}")

    return " ".join(parts)


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
        # If DATABASE_URL is a postgres URL, convert to libpq conninfo and
        # add hostaddr=<ipv4> when available to avoid IPv6 network issues.
        try:
            conninfo = _url_to_conninfo(db_url) if "://" in db_url else db_url
            return psycopg.connect(conninfo)
        except Exception:
            # Fall back to raw URL/conninfo
            return psycopg.connect(db_url)

    # Local dev: SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn