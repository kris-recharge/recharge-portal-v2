import pandas as pd
import numpy as np
from typing import Sequence, Optional
from sqlalchemy import inspect
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine, Connection
from .config import DB_BACKEND, AK_TZ
from .db import get_conn, param_placeholder
from .constants import EVSE_DISPLAY

def _ts_col() -> str:
    """Return the proper timestamp column reference for the current backend."""
    return '"timestamp"' if DB_BACKEND == "postgres" else "timestamp"

def _make_placeholders(n: int) -> str:
    """Return the correct placeholder list for IN (...) given backend and count n."""
    if n <= 0:
        return ""
    if DB_BACKEND == "postgres":
        return ",".join(["%s"] * n)
    # default: sqlite
    return ",".join(["?"] * n)


def _ts_cast_sql() -> str:
    """
    Return an expression that safely casts the TEXT `timestamp` column into timestamptz on Postgres.
    Falls back to plain column on sqlite.
    """
    if DB_BACKEND == "postgres":
        # convert trailing Z to +00:00 to satisfy timestamptz
        return '(regexp_replace("timestamp", \'Z$\', \'+00:00\'))::timestamptz'
    return _ts_col()

def _ts_has_iso_sql() -> str:
    """Predicate ensuring the timestamp looks ISO-like before casting (Postgres safety)."""
    if DB_BACKEND == "postgres":
        return "\"timestamp\" ~ '^\\d{4}-\\d{2}-\\d{2}T'"
    return "1=1"

def _stations_pred_sa(stations):
    """
    For SQLAlchemy text() queries on Postgres: build an expanding IN predicate
    and a matching bindparam. Returns (predicate_sql, bindparam_or_none).
    """
    if not stations:
        return "1=1", None
    return "station_id IN :stations", bindparam("stations", expanding=True)


def _sa_engine(conn_or_engine) -> Optional[Engine]:
    """Return a SQLAlchemy Engine if `conn_or_engine` is an Engine or Connection; otherwise None."""
    try:
        # If it's already an Engine
        if isinstance(conn_or_engine, Engine):
            return conn_or_engine
        # If it's a SQLAlchemy Connection, return its bound engine
        if isinstance(conn_or_engine, Connection):
            return conn_or_engine.engine
    except Exception:
        pass
    return None

def _first_existing(conn, names: Sequence[str]) -> Optional[str]:
    """Return the first table name that exists from `names` for the current backend.

    Strategy:
      • If we have a SQLAlchemy Engine/Connection (Postgres path on Render):
          1) Try SQLAlchemy inspector with schema="public" (Postgres).
          2) If not found, probe with a lightweight "SELECT 1 FROM ..." for each candidate.
      • Otherwise, fall back to sqlite's sqlite_master check (raw sqlite3 path) — only if the
        connection actually has a .cursor() attribute (to avoid 'Connection has no cursor').
    """
    eng = _sa_engine(conn)

    # ---------- Postgres / SQLAlchemy path ----------
    if eng is not None and DB_BACKEND == "postgres":
        try:
            insp = inspect(eng)
        except Exception:
            insp = None

        schema = "public"
        for n in names:
            # 1) Inspector check
            try:
                if insp is not None and insp.has_table(n, schema=schema):
                    return n
            except Exception:
                pass
            # 2) Lightweight probe
            for target in (f'{schema}."{n}"', f"{schema}.{n}"):
                try:
                    with eng.connect() as c2:
                        c2.exec_driver_sql(f"SELECT 1 FROM {target} LIMIT 1")
                    return n
                except Exception:
                    continue
        return None  # do not fall into sqlite path with a SQLAlchemy engine

    # ---------- sqlite fallback (local) ----------
    try:
        # only attempt if this object actually behaves like sqlite3 connection
        if hasattr(conn, "cursor"):
            cur = conn.cursor()
            for n in names:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (n,))
                if cur.fetchone() is not None:
                    return n
    except Exception:
        pass

    return None

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Ensure a 'station_id' column exists for downstream code
    if "station_id" not in df.columns:
        for alt in ("evse_id", "external_id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "station_id"})
                break
    return df

# -----------------------------
# Meter values + Authorize data
# -----------------------------
def load_meter_values(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_meter_values for the given stations and UTC window.
    Postgres path uses SQLAlchemy text() with proper timestamptz casting;
    sqlite path retains the existing parameter style.
    """
    conn = get_conn()
    eng = _sa_engine(conn)

    # ---------- Postgres path (Render) ----------
    if DB_BACKEND == "postgres" and eng is not None:
        table = _first_existing(eng, ["realtime_meter_values", "meter_values"])
        if not table:
            return pd.DataFrame()

        ts_cast = _ts_cast_sql()
        ts_pred = f"{ts_cast} BETWEEN :start AND :end"
        has_iso = _ts_has_iso_sql()
        pred_stations, bp = _stations_pred_sa(stations)

        sql = text(f"""
            SELECT
                station_id, connector_id, transaction_id, "timestamp",
                power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v, hbv_v
            FROM public.{table}
            WHERE {pred_stations}
              AND {has_iso}
              AND {ts_pred}
            ORDER BY {ts_cast} ASC
        """)
        if bp is not None:
            sql = sql.bindparams(bp)

        params = {"start": start_utc, "end": end_utc}
        if stations:
            params["stations"] = tuple(stations)

        with eng.connect() as c:
            df = pd.read_sql(sql, c, params=params)

        df = _normalize_ids(df)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            for ccol in ["power_w","energy_wh","soc","amperage_offered","amperage_import","power_offered_w","voltage_v","hbv_v"]:
                if ccol in df.columns:
                    df[ccol] = pd.to_numeric(df[ccol], errors="coerce")
        return df

    # ---------- sqlite fallback (local dev) ----------
    ph = param_placeholder()
    try:
        table = _first_existing(conn, ["realtime_meter_values", "meter_values"])
        if not table:
            conn.close()
            return pd.DataFrame()
        where_in = ""
        params = []
        if stations:
            where_in = f"station_id IN ({_make_placeholders(len(stations))}) AND "
            params.extend(list(stations))
        sql = f"""
          SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
                 power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
          FROM {table}
          WHERE {where_in} {_ts_col()} BETWEEN {ph} AND {ph}
          ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
        """
        params.extend([start_utc, end_utc])
        try:
            df = pd.read_sql(sql, conn, params=params)
            df = _normalize_ids(df)
            if (df is None or df.empty) and stations and len(stations) > 1:
                frames = []
                single_sql = f"""
                  SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
                         power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
                  FROM {table}
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {ph} AND {ph}
                  ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
                """
                for sid in stations:
                    f = pd.read_sql(single_sql, conn, params=[sid, start_utc, end_utc])
                    if f is not None and not f.empty:
                        frames.append(f)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    df = _normalize_ids(df)
        except Exception:
            df = pd.DataFrame()
            if stations and len(stations) >= 1:
                frames = []
                single_sql = f"""
                  SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
                         power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
                  FROM {table}
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {ph} AND {ph}
                  ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
                """
                for sid in stations:
                    try:
                        f = pd.read_sql(single_sql, conn, params=[sid, start_utc, end_utc])
                        if f is not None and not f.empty:
                            frames.append(f)
                    except Exception:
                        pass
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    df = _normalize_ids(df)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in ["power_w","energy_wh","soc","amperage_offered","amperage_import","power_offered_w","voltage_v"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_authorize(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_authorize rows (VID: id_tags) with a small time pad around the window.
    """
    start_pad = (pd.to_datetime(start_utc) - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_pad   = (pd.to_datetime(end_utc)   + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_conn()
    eng = _sa_engine(conn)

    # ---------- Postgres path ----------
    if DB_BACKEND == "postgres" and eng is not None:
        table = _first_existing(eng, ["realtime_authorize", "authorize"])
        if not table:
            return pd.DataFrame()

        ts_cast = _ts_cast_sql()
        ts_pred = f"{ts_cast} BETWEEN :start AND :end"
        has_iso = _ts_has_iso_sql()
        pred_stations, bp = _stations_pred_sa(stations)

        sql = text(f"""
            SELECT station_id, "timestamp", id_tag
            FROM public.{table}
            WHERE {pred_stations}
              AND {has_iso}
              AND {ts_pred}
            ORDER BY {ts_cast} ASC
        """)
        if bp is not None:
            sql = sql.bindparams(bp)

        params = {"start": start_pad, "end": end_pad}
        if stations:
            params["stations"] = tuple(stations)

        with eng.connect() as c:
            df = pd.read_sql(sql, c, params=params)

        df = _normalize_ids(df)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["id_tag"] = df["id_tag"].astype(str)
            df = df[df["id_tag"].str.startswith("VID:", na=False)]
        return df

    # ---------- sqlite fallback ----------
    ph = param_placeholder()
    try:
        table = _first_existing(conn, ["realtime_authorize", "authorize"])
        if not table:
            conn.close()
            return pd.DataFrame()
        where_in = ""
        params = []
        if stations:
            where_in = f"station_id IN ({_make_placeholders(len(stations))}) AND "
            params.extend(list(stations))
        sql = f"""
          SELECT station_id, {_ts_col()} AS timestamp, id_tag
          FROM {table}
          WHERE {where_in} {_ts_col()} BETWEEN {ph} AND {ph}
          ORDER BY station_id, {_ts_col()}
        """
        params.extend([start_pad, end_pad])
        df = pd.read_sql(sql, conn, params=params)
        df = _normalize_ids(df)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["id_tag"] = df["id_tag"].astype(str)
        df = df[df["id_tag"].str.startswith("VID:", na=False)]
    return df

# -----------------------------
# Status + Connectivity loaders
# -----------------------------
def _in_clause_and_params(stations, start_utc: str, end_utc: str):
    """
    Build an 'IN (...)' fragment (or empty) and parameter list. BETWEEN placeholders are handled at callsites.
    """
    placeholders = _make_placeholders(len(stations)) if stations else ""
    where_in = f"station_id IN ({placeholders}) AND " if stations else ""
    base_params = list(stations) if stations else []
    return where_in, base_params

def load_status_history(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_status_notifications for the window; newest first; AKDT strings + friendly names.
    Includes vendor_error_code if present.
    """
    conn = get_conn()
    eng = _sa_engine(conn)

    # ---------- Postgres path ----------
    if DB_BACKEND == "postgres" and eng is not None:
        table = _first_existing(eng, ["realtime_status_notifications", "status_notifications"])
        if not table:
            return pd.DataFrame(columns=["AKDT","Location","station_id","connector_id","status","error_code","vendor_error_code"])

        ts_cast = _ts_cast_sql()
        ts_pred = f"{ts_cast} BETWEEN :start AND :end"
        has_iso = _ts_has_iso_sql()
        pred_stations, bp = _stations_pred_sa(stations)

        # Try selecting vendor_error_code if present
        try_sql = text(f"""
            SELECT station_id, connector_id, "timestamp", status, error_code, vendor_error_code
            FROM public.{table}
            WHERE {pred_stations}
              AND {has_iso}
              AND {ts_pred}
            ORDER BY {ts_cast} DESC
        """)
        fb_sql = text(f"""
            SELECT station_id, connector_id, "timestamp", status, error_code
            FROM public.{table}
            WHERE {pred_stations}
              AND {has_iso}
              AND {ts_pred}
            ORDER BY {ts_cast} DESC
        """)
        if bp is not None:
            try_sql = try_sql.bindparams(bp)
            fb_sql  = fb_sql.bindparams(bp)

        params = {"start": start_utc, "end": end_utc}
        if stations:
            params["stations"] = tuple(stations)

        with eng.connect() as c:
            try:
                df = pd.read_sql(try_sql, c, params=params)
            except Exception:
                df = pd.read_sql(fb_sql, c, params=params)

        df = _normalize_ids(df)
    else:
        # ---------- sqlite fallback ----------
        ph = param_placeholder()
        try:
            table = _first_existing(conn, ["realtime_status_notifications", "status_notifications"])
            if not table:
                conn.close()
                return pd.DataFrame(columns=["AKDT","Location","station_id","connector_id","status","error_code","vendor_error_code"])

            where_in, base_params = _in_clause_and_params(stations, start_utc, end_utc)
            base_cols = 'station_id, connector_id, {_ts} AS timestamp, status, error_code'.format(_ts=_ts_col())
            try_cols = base_cols + ', vendor_error_code'

            sql_try = f"""
              SELECT {try_cols}
              FROM {table}
              WHERE {where_in} {_ts_col()} BETWEEN {ph} AND {ph}
              ORDER BY {_ts_col()} DESC
            """
            sql_fallback = f"""
              SELECT {base_cols}
              FROM {table}
              WHERE {where_in} {_ts_col()} BETWEEN {ph} AND {ph}
              ORDER BY {_ts_col()} DESC
            """
            params = base_params + [start_utc, end_utc]

            try:
                df = pd.read_sql(sql_try, conn, params=params)
                df = _normalize_ids(df)
            except Exception:
                df = pd.read_sql(sql_fallback, conn, params=params)
                df = _normalize_ids(df)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    if df is None or df.empty:
        return pd.DataFrame(columns=["AKDT","Location","station_id","connector_id","status","error_code","vendor_error_code"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["AKDT"] = df["timestamp"].dt.tz_convert(AK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Location"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])

    cols = ["AKDT","Location","station_id","connector_id","status","error_code"]
    if "vendor_error_code" in df.columns:
        cols.append("vendor_error_code")
    return df[[c for c in cols if c in df.columns]].sort_values("AKDT", ascending=False, kind="mergesort")

def load_connectivity(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_websocket CONNECT/DISCONNECT events; newest first; AKDT strings + friendly names.
    """
    conn = get_conn()
    eng = _sa_engine(conn)

    # ---------- Postgres path ----------
    if DB_BACKEND == "postgres" and eng is not None:
        table = _first_existing(eng, ["realtime_websocket", "connectivity_logs"])
        if not table:
            return pd.DataFrame(columns=["AKDT","Location","station_id","connection_id","Connectivity"])

        ts_cast = _ts_cast_sql()
        ts_pred = f"{ts_cast} BETWEEN :start AND :end"
        has_iso = _ts_has_iso_sql()
        pred_stations, bp = _stations_pred_sa(stations)

        sql = text(f"""
            SELECT station_id, connection_id, event, "timestamp"
            FROM public.{table}
            WHERE {pred_stations}
              AND {has_iso}
              AND {ts_pred}
            ORDER BY {ts_cast} DESC
        """)
        if bp is not None:
            sql = sql.bindparams(bp)

        params = {"start": start_utc, "end": end_utc}
        if stations:
            params["stations"] = tuple(stations)

        with eng.connect() as c:
            df = pd.read_sql(sql, c, params=params)

        df = _normalize_ids(df)
    else:
        # ---------- sqlite fallback ----------
        ph = param_placeholder()
        try:
            table = _first_existing(conn, ["realtime_websocket", "connectivity_logs"])
            if not table:
                conn.close()
                return pd.DataFrame(columns=["AKDT","Location","station_id","connection_id","Connectivity"])

            where_in, base_params = _in_clause_and_params(stations, start_utc, end_utc)
            sql = f"""
              SELECT station_id, connection_id, event, {_ts_col()} AS timestamp
              FROM {table}
              WHERE {where_in} {_ts_col()} BETWEEN {ph} AND {ph}
              ORDER BY {_ts_col()} DESC
            """
            params = base_params + [start_utc, end_utc]
            df = pd.read_sql(sql, conn, params=params)
            df = _normalize_ids(df)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    if df is None or df.empty:
        return pd.DataFrame(columns=["AKDT","Location","station_id","connection_id","Connectivity"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["AKDT"] = df["timestamp"].dt.tz_convert(AK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    ev = df.get("event").astype(str).str.upper()
    df["Connectivity"] = np.where(ev.str.contains("DISCONNECT"), "websocket DISCONNECT", "websocket CONNECT")
    df["Location"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])
    cols = ["AKDT","Location","station_id","connection_id","Connectivity"]
    return df[[c for c in cols if c in df.columns]].sort_values("AKDT", ascending=False, kind="mergesort")

def load_tritium_error_codes() -> pd.DataFrame:
    """
    Return columns: platform, code, impact, description
    from the `tritium_error_codes` table. Returns empty DataFrame if missing.
    """
    conn = get_conn()
    try:
        df = pd.read_sql(
            "SELECT platform, code, impact, description FROM tritium_error_codes",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df