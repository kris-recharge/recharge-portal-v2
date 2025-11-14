import pandas as pd
import numpy as np
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

def _table_exists(cur, name: str) -> bool:
    try:
        if DB_BACKEND == "postgres":
            # to_regclass returns the OID (as regclass) or NULL if it doesn't exist
            cur.execute("SELECT to_regclass(%s)", (name,))
            return cur.fetchone()[0] is not None
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
            return cur.fetchone() is not None
    except Exception:
        return False

def _first_existing(conn, names):
    cur = conn.cursor()
    for n in names:
        if _table_exists(cur, n):
            return n
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
    Returns a DataFrame with parsed dtypes.
    """
    # If stations is empty/None, we will query all stations in the time window
    ph = param_placeholder()
    conn = get_conn()
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
            # Fallback: if a multi-station query returns empty but single-station queries succeed,
            # run per-EVSE and concatenate (addresses IN-clause driver quirks).
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
            # As an additional hard fallback, also attempt per-station queries
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
    # If stations is empty/None, we will query all stations in the time window
    start_pad = (pd.to_datetime(start_utc) - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_pad   = (pd.to_datetime(end_utc)   + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ph = param_placeholder()
    conn = get_conn()
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
        try:
            df = pd.read_sql(sql, conn, params=params)
            df = _normalize_ids(df)
            # Fallback: if a multi-station query returns empty but single-station queries succeed,
            # run per-EVSE and concatenate (addresses IN-clause driver quirks).
            if (df is None or df.empty) and stations and len(stations) > 1:
                frames = []
                single_sql = f"""
                  SELECT station_id, {_ts_col()} AS timestamp, id_tag
                  FROM {table}
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {ph} AND {ph}
                  ORDER BY station_id, {_ts_col()}
                """
                for sid in stations:
                    f = pd.read_sql(single_sql, conn, params=[sid, start_pad, end_pad])
                    if f is not None and not f.empty:
                        frames.append(f)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    df = _normalize_ids(df)
        except Exception:
            # As an additional hard fallback, also attempt per-station queries
            df = pd.DataFrame()
            if stations and len(stations) >= 1:
                frames = []
                single_sql = f"""
                  SELECT station_id, {_ts_col()} AS timestamp, id_tag
                  FROM {table}
                  WHERE station_id IN ({_make_placeholders(1)})
                    AND {_ts_col()} BETWEEN {ph} AND {ph}
                  ORDER BY station_id, {_ts_col()}
                """
                for sid in stations:
                    try:
                        f = pd.read_sql(single_sql, conn, params=[sid, start_pad, end_pad])
                        if f is not None and not f.empty:
                            frames.append(f)
                    except Exception:
                        pass
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    df = _normalize_ids(df)
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
    Includes vendor_error_code if the column exists (falls back cleanly if not).
    """
    ph = param_placeholder()
    conn = get_conn()
    try:
        table = _first_existing(conn, ["realtime_status_notifications", "status_notifications"])
        if not table:
            conn.close()
            return pd.DataFrame(columns=["AKDT","Location","station_id","connector_id","status","error_code","vendor_error_code"])

        where_in, base_params = _in_clause_and_params(stations, start_utc, end_utc)

        # Try selecting vendor_error_code if present; fallback to a reduced column set if not
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
        conn.close()

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
    ph = param_placeholder()
    conn = get_conn()
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