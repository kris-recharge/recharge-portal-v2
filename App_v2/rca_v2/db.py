"""
Postgres-only DB connector for App_v2.

- Requires DATABASE_URL (Supabase Postgres URL or libpq conninfo).
- Uses psycopg (v3).
- Adds hostaddr=<ipv4> when possible to avoid IPv6-only connection attempts.
- Autocommit enabled (read-heavy app; avoids idle-in-transaction issues).

RLS support:
- Optionally applies Supabase/PostgREST-compatible request.jwt.claims to the session
  (via `set_config('request.jwt.claims', <json>, false)`) so Postgres RLS policies
  can enforce per-user access.
"""

import os
import json
import socket
from urllib.parse import urlparse, parse_qs, unquote


import psycopg

# Explicit exports expected by older modules (e.g., loaders.py)
__all__ = [
    "get_conn",
    "using_postgres",
    "param_placeholder",
    "apply_jwt_claims",
    "clear_jwt_claims",
]


def using_postgres() -> bool:
    """App_v2 is Postgres-only in this deployment."""
    return True


def param_placeholder() -> str:
    """psycopg placeholder style."""
    return "%s"


def _resolve_ipv4(hostname: str) -> str | None:
    """Resolve hostname to an IPv4 address (A record). Returns None if not found."""
    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not infos:
            return None
        return infos[0][4][0]
    except Exception:
        return None


def _url_to_conninfo(db_url: str) -> str:
    """Convert a postgres URL into a libpq-style conninfo string.

    Also adds hostaddr=<ipv4> when possible to avoid IPv6-only connection attempts.
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

    parts: list[str] = []
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
        parts.append(f"password={password}")
    if sslmode:
        parts.append(f"sslmode={sslmode}")

    return " ".join(parts)


def apply_jwt_claims(conn, claims) -> None:
    """Apply Supabase/PostgREST-style JWT claims to this DB session.

    This enables Postgres RLS policies that reference `auth.uid()` / JWT claims.

    Parameters
    ----------
    conn:
        psycopg Connection
    claims:
        Either a dict of decoded JWT claims, or a JSON string.

    Notes
    -----
    - We intentionally set `is_local = false` so the config persists for the
      session even with autocommit enabled.
    - Supabase/PostgREST sets `request.jwt.claims` (plural). We follow that.
    """
    if claims is None:
        return

    if isinstance(claims, dict):
        claims_json = json.dumps(claims)
    else:
        claims_json = str(claims)

    # Best-effort: set request.jwt.claims for RLS.
    # Using set_config with is_local=false so it persists for the session.
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('request.jwt.claims', %s, false)", (claims_json,))
    except Exception:
        # If this fails, we leave the session unmodified.
        return


def clear_jwt_claims(conn) -> None:
    """Clear any previously-applied JWT claims from the DB session."""
    try:
        with conn.cursor() as cur:
            # Empty JSON object is a safe default
            cur.execute("SELECT set_config('request.jwt.claims', %s, false)", ("{}",))
    except Exception:
        return


def get_conn(jwt_claims=None):
    """Return a Postgres connection (psycopg). Requires DATABASE_URL.

    Parameters
    ----------
    jwt_claims:
        Optional decoded JWT claims (dict) or JSON string. If provided, the
        connection session will be tagged with `request.jwt.claims` so Postgres
        RLS policies can enforce per-user access.

    Backwards compatible:
        Existing callers can continue calling `get_conn()` with no args.
    """
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is not set. This deployment is Postgres-only.\n"
            "Set DATABASE_URL to your Supabase Postgres connection string."
        )

    # Accept either a URL or a libpq conninfo string
    conninfo = _url_to_conninfo(db_url) if "://" in db_url else db_url

    conn = psycopg.connect(
        conninfo,
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "10")),
        application_name=os.getenv("APP_NAME", "app_v2"),
    )
    conn.autocommit = True

    # Optional: cap statement runtime (ms)
    stmt_timeout_ms = os.getenv("PG_STATEMENT_TIMEOUT_MS", "").strip()
    if stmt_timeout_ms:
        try:
            ms = int(stmt_timeout_ms)
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = %s", (ms,))
        except Exception:
            pass

    # Optional: apply RLS JWT claims to this session
    if jwt_claims is not None:
        apply_jwt_claims(conn, jwt_claims)

    return conn
