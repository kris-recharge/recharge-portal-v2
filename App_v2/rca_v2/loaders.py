import os
import logging
import pandas as pd
import numpy as np
from .config import DB_BACKEND, AK_TZ
from .db import get_conn
from .constants import EVSE_DISPLAY

logger = logging.getLogger(__name__)

# Heuristic: treat as Postgres if either the config says so OR the DATABASE_URL is a Postgres URI.
_DB_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = (
    DB_BACKEND == "postgres"
    or _DB_URL.startswith("postgres://")
    or _DB_URL.startswith("postgresql://")
)

def _ts_col() -> str:
    """Return the proper timestamp column reference for the current backend."""
    return '"timestamp"' if IS_POSTGRES else "timestamp"

def _make_placeholders(n: int) -> str:
    """Return the correct placeholder list for IN (...) given backend and count n."""
    if n <= 0:
        return ""
    if IS_POSTGRES:
        # psycopg / Postgres uses %s style placeholders
        return ",".join(["%s"] * n)
    # default: sqlite qmark style
    return ",".join(["?"] * n)

def _between_placeholders() -> str:
    """Return the placeholder pair for `BETWEEN ... AND ...` based on backend."""
    if IS_POSTGRES:
        # psycopg / Postgres uses %s style placeholders
        return "%s AND %s"
    # default: sqlite qmark style
    return "? AND ?"

# -----------------------------
# Meter values + Authorize data
# -----------------------------
def load_meter_values(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_meter_values for the given stations and UTC window.
    Returns a DataFrame with parsed dtypes.
    """
    if not stations:
        return pd.DataFrame()

    placeholders = _make_placeholders(len(stations))
    between = _between_placeholders()
    sql = f"""
      SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
             power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
      FROM realtime_meter_values
      WHERE station_id IN ({placeholders})
        AND {_ts_col()} BETWEEN {between}
      ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
    """
    params = list(stations) + [start_utc, end_utc]
    conn = get_conn()
    try:
        try:
            df = pd.read_sql(sql, conn, params=params)
            # Fallback: if a multi-station query returns empty but single-station queries succeed,
            # run per-EVSE and concatenate (addresses IN-clause driver quirks).
            if (df is None or df.empty) and len(stations) > 1:
                frames = []
                single_between = _between_placeholders()
                single_sql = f"""
                  SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
                         power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
                  FROM realtime_meter_values
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {single_between}
                  ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
                """
                for sid in stations:
                    f = pd.read_sql(single_sql, conn, params=[sid, start_utc, end_utc])
                    if f is not None and not f.empty:
                        frames.append(f)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
        except Exception as e:
            logger.exception(
                "load_meter_values primary query failed (backend=%s, stations=%s, window=%s→%s)",
                "postgres" if IS_POSTGRES else "sqlite",
                stations,
                start_utc,
                end_utc,
            )
            # As an additional hard fallback, also attempt per-station queries
            df = pd.DataFrame()
            if len(stations) >= 1:
                frames = []
                single_between = _between_placeholders()
                single_sql = f"""
                  SELECT station_id, connector_id, transaction_id, {_ts_col()} AS timestamp,
                         power_w, energy_wh, soc, amperage_offered, amperage_import, power_offered_w, voltage_v
                  FROM realtime_meter_values
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {single_between}
                  ORDER BY station_id, connector_id, transaction_id, {_ts_col()}
                """
                for sid in stations:
                    try:
                        f = pd.read_sql(single_sql, conn, params=[sid, start_utc, end_utc])
                        if f is not None and not f.empty:
                            frames.append(f)
                    except Exception as e:
                        logger.exception(
                            "load_meter_values single-station query failed for %s (backend=%s)",
                            sid,
                            "postgres" if IS_POSTGRES else "sqlite",
                        )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()

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
    if not stations:
        return pd.DataFrame()

    start_pad = (pd.to_datetime(start_utc) - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_pad   = (pd.to_datetime(end_utc)   + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    placeholders = _make_placeholders(len(stations))
    between = _between_placeholders()
    sql = f"""
      SELECT station_id, {_ts_col()} AS timestamp, id_tag
      FROM realtime_authorize
      WHERE station_id IN ({placeholders})
        AND {_ts_col()} BETWEEN {between}
      ORDER BY station_id, {_ts_col()}
    """
    params = list(stations) + [start_pad, end_pad]
    conn = get_conn()
    try:
        try:
            df = pd.read_sql(sql, conn, params=params)
            # Fallback: if a multi-station query returns empty but single-station queries succeed,
            # run per-EVSE and concatenate (addresses IN-clause driver quirks).
            if (df is None or df.empty) and len(stations) > 1:
                frames = []
                single_between = _between_placeholders()
                single_sql = f"""
                  SELECT station_id, {_ts_col()} AS timestamp, id_tag
                  FROM realtime_authorize
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {single_between}
                  ORDER BY station_id, {_ts_col()}
                """
                for sid in stations:
                    f = pd.read_sql(single_sql, conn, params=[sid, start_pad, end_pad])
                    if f is not None and not f.empty:
                        frames.append(f)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
        except Exception as e:
            logger.exception(
                "load_authorize primary query failed (backend=%s, stations=%s, window=%s→%s)",
                "postgres" if IS_POSTGRES else "sqlite",
                stations,
                start_pad,
                end_pad,
            )
            # As an additional hard fallback, also attempt per-station queries
            df = pd.DataFrame()
            if len(stations) >= 1:
                frames = []
                single_between = _between_placeholders()
                single_sql = f"""
                  SELECT station_id, {_ts_col()} AS timestamp, id_tag
                  FROM realtime_authorize
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {single_between}
                  ORDER BY station_id, {_ts_col()}
                """
                for sid in stations:
                    try:
                        f = pd.read_sql(single_sql, conn, params=[sid, start_pad, end_pad])
                        if f is not None and not f.empty:
                            frames.append(f)
                    except Exception as e:
                        logger.exception(
                            "load_authorize single-station query failed for %s (backend=%s)",
                            sid,
                            "postgres" if IS_POSTGRES else "sqlite",
                        )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["id_tag"] = df["id_tag"].astype(str)
        df = df[df["id_tag"].str.startswith("VID:", na=False)]
    return df

# -----------------------------
# Status + Connectivity loaders
# -----------------------------
def _in_clause_and_params(stations, start_utc: str, end_utc: str):
    """Build the `station_id IN (...) AND` piece and param list for the window.

    The timestamp BETWEEN placeholders are handled separately per backend.
    """
    placeholders = _make_placeholders(len(stations)) if stations else ""
    where_in = f"station_id IN ({placeholders}) AND " if stations else ""
    return where_in, list(stations) + [start_utc, end_utc]

def load_status_history(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_status_notifications for the window; newest first; AKDT strings + friendly names.
    Includes vendor_error_code if the column exists (falls back cleanly if not).
    """
    where_in, params = _in_clause_and_params(stations, start_utc, end_utc)
    between = _between_placeholders()

    # Try selecting vendor_error_code if present; fallback to a reduced column set if not
    base_cols = 'station_id, connector_id, {_ts} AS timestamp, status, error_code'.format(_ts=_ts_col())
    try_cols = base_cols + ', vendor_error_code'

    sql_try = f"""
      SELECT {try_cols}
      FROM realtime_status_notifications
      WHERE {where_in} {_ts_col()} BETWEEN {between}
      ORDER BY {_ts_col()} DESC
    """
    sql_fallback = f"""
      SELECT {base_cols}
      FROM realtime_status_notifications
      WHERE {where_in} {_ts_col()} BETWEEN {between}
      ORDER BY {_ts_col()} DESC
    """

    conn = get_conn()
    try:
        try:
            df = pd.read_sql(sql_try, conn, params=params)
        except Exception:
            df = pd.read_sql(sql_fallback, conn, params=params)
    finally:
        conn.close()

    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "AKDT",
            "Location",
            "station_id",
            "connector_id",
            "status",
            "error_code",
            "vendor_error_code",
            "impact",
            "description",
        ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["AKDT"] = df["timestamp"].dt.tz_convert(AK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Location"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])

    # Optional Tritium enrichment: map vendor_error_code -> impact / description
    if "vendor_error_code" in df.columns:
        try:
            tec = load_tritium_error_codes()
            if tec is not None and not tec.empty:
                tec = tec.copy()
                # Ensure numeric compare where possible
                tec["code"] = pd.to_numeric(tec["code"], errors="coerce")
                df["vendor_error_code"] = pd.to_numeric(df["vendor_error_code"], errors="coerce")

                df = df.merge(
                    tec[["code", "impact", "description"]],
                    left_on="vendor_error_code",
                    right_on="code",
                    how="left",
                )
                # Drop the helper join column if present
                if "code" in df.columns:
                    df = df.drop(columns=["code"])
        except Exception:
            # Fail silently if lookup table is missing or malformed
            pass

    cols = [
        "AKDT",
        "Location",
        "station_id",
        "connector_id",
        "status",
        "error_code",
    ]
    if "vendor_error_code" in df.columns:
        cols.append("vendor_error_code")
    if "impact" in df.columns:
        cols.append("impact")
    if "description" in df.columns:
        cols.append("description")

    return df[[c for c in cols if c in df.columns]].sort_values(
        "AKDT", ascending=False, kind="mergesort"
    )

def load_connectivity(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_websocket CONNECT/DISCONNECT events; newest first; AKDT strings + friendly names.
    """
    where_in, params = _in_clause_and_params(stations, start_utc, end_utc)
    between = _between_placeholders()
    sql = f"""
      SELECT station_id, connection_id, event, {_ts_col()} AS timestamp
      FROM realtime_websocket
      WHERE {where_in} {_ts_col()} BETWEEN {between}
      ORDER BY {_ts_col()} DESC
    """
    conn = get_conn()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

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