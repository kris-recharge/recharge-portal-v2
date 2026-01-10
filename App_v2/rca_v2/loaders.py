import logging
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Optional
from .config import AK_TZ
from .db import get_conn, using_postgres, param_placeholder
from .constants import EVSE_DISPLAY

logger = logging.getLogger(__name__)


def _ts_col() -> str:
    """Return the proper timestamp column reference for the current backend."""
    # Postgres/Supabase uses `received_at` in ocpp_events
    return 'received_at' if using_postgres() else "timestamp"


def _make_placeholders(n: int) -> str:
    """Return the correct placeholder list for IN (...) given backend and count n."""
    if n <= 0:
        return ""
    ph = param_placeholder()
    return ",".join([ph] * n)


def _between_placeholders() -> str:
    """Return the placeholder pair for `BETWEEN ... AND ...` based on backend."""
    ph = param_placeholder()
    return f"{ph} AND {ph}"


#
# -----------------------------
# Supabase OCPP parsing helpers
# -----------------------------

def _safe_get(d: Any, *keys: str) -> Any:
    """Safely descend into nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_sampled_values(sampled: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Extract a few common metrics from a MeterValues.sampledValue list."""
    out: Dict[str, Optional[float]] = {
        "power_w": None,
        "energy_wh": None,
        "soc": None,
        "voltage_v": None,
        "amperage_import": None,
        "amperage_offered": None,
        "power_offered_w": None,
    }

    for sv in sampled or []:
        if not isinstance(sv, dict):
            continue
        meas = (sv.get("measurand") or "").strip()
        unit = (sv.get("unit") or "").strip()
        val = _to_float(sv.get("value"))

        # Common OCPP 1.6 measurands
        if meas in {"Power.Active.Import", "Power.Active.Import"}:
            # often W
            if val is not None:
                out["power_w"] = val
        elif meas in {"Energy.Active.Import.Register", "Energy.Active.Import.Register"}:
            # often Wh or kWh
            if val is not None:
                if unit.lower() == "kwh":
                    out["energy_wh"] = val * 1000.0
                else:
                    out["energy_wh"] = val
        elif meas in {"SoC", "Soc", "SOC"}:
            if val is not None:
                out["soc"] = val
        elif meas in {"Voltage"}:
            if val is not None:
                out["voltage_v"] = val
        elif meas in {"Current.Import"}:
            if val is not None:
                out["amperage_import"] = val
        elif meas in {"Current.Offered"}:
            if val is not None:
                out["amperage_offered"] = val
        elif meas in {"Power.Offered"}:
            if val is not None:
                out["power_offered_w"] = val

    return out


def _parse_meter_values_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Parse ocpp_events MeterValues rows into the schema expected by the app."""
    frames: List[pd.DataFrame] = []

    for r in rows:
        station_id = r.get("station_id")
        connector_id = r.get("connector_id")
        transaction_id = r.get("transaction_id")
        received_at = r.get("timestamp")
        payload = r.get("action_payload")

        # payload may come in as dict (psycopg) or string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None

        meter_values = None
        if isinstance(payload, dict):
            meter_values = payload.get("meterValue")

        if not isinstance(meter_values, list) or not meter_values:
            # still emit a single row at received_at
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "station_id": station_id,
                            "connector_id": connector_id,
                            "transaction_id": transaction_id,
                            "timestamp": received_at,
                        }
                    ]
                )
            )
            continue

        out_rows: List[Dict[str, Any]] = []
        for mv in meter_values:
            if not isinstance(mv, dict):
                continue
            ts = mv.get("timestamp") or received_at
            sampled = mv.get("sampledValue")
            metrics = _extract_sampled_values(sampled if isinstance(sampled, list) else [])
            out_rows.append(
                {
                    "station_id": station_id,
                    "connector_id": connector_id,
                    "transaction_id": transaction_id,
                    "timestamp": ts,
                    **metrics,
                }
            )

        if out_rows:
            frames.append(pd.DataFrame(out_rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df


# -----------------------------
# Fallback: parse Start/StopTransaction rows for minimal meter values
# -----------------------------
def _parse_start_stop_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a minimal meter-values-like dataframe from Start/StopTransaction events.

    This enables session building on Supabase when `MeterValues` is not available.
    We emit two rows per transaction (start and stop) with energy_wh populated from
    meterStart/meterStop when present.
    """
    out: List[Dict[str, Any]] = []

    for r in rows or []:
        station_id = r.get("station_id")
        received_at = r.get("timestamp")
        action = r.get("action")
        payload = r.get("action_payload")

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None

        if not isinstance(payload, dict):
            payload = {}

        # Connector id can be in different places depending on platform
        connector_id = (
            payload.get("connectorId")
            or _safe_get(payload, "connectorId")
            or _safe_get(payload, "connector_id")
        )

        # Transaction id is often only in payload on Supabase
        tx = (
            r.get("transaction_id")
            or payload.get("transactionId")
            or payload.get("transaction_id")
        )

        # Meter start/stop (usually Wh)
        meter_start = payload.get("meterStart")
        meter_stop = payload.get("meterStop")

        energy_wh = None
        if action == "StartTransaction" and meter_start is not None:
            energy_wh = _to_float(meter_start)
        elif action == "StopTransaction" and meter_stop is not None:
            energy_wh = _to_float(meter_stop)

        out.append(
            {
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": str(tx) if tx is not None else None,
                "timestamp": received_at,
                "energy_wh": energy_wh,
                # Other metrics are unknown in this fallback
                "power_w": None,
                "soc": None,
                "voltage_v": None,
                "amperage_import": None,
                "amperage_offered": None,
                "power_offered_w": None,
            }
        )

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    return df

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

    # Supabase/Postgres: pull from ocpp_events and parse MeterValues payload JSON
    if using_postgres():
        placeholders = _make_placeholders(len(stations))
        between = _between_placeholders()
        sql = f"""
          SELECT asset_id AS station_id,
                 connector_id,
                 transaction_id,
                 {_ts_col()} AS timestamp,
                 action_payload
          FROM ocpp_events
          WHERE asset_id IN ({placeholders})
            AND action = 'MeterValues'
            AND {_ts_col()} BETWEEN {between}
          ORDER BY asset_id, connector_id, transaction_id, {_ts_col()}
        """
        params = list(stations) + [start_utc, end_utc]
        conn = get_conn()
        try:
            raw = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

        if raw is None or raw.empty:
            # Fallback: some platforms do not emit MeterValues but do emit Start/StopTransaction.
            # Build a minimal dataframe from those events so sessions can still be produced.
            placeholders2 = _make_placeholders(len(stations))
            between2 = _between_placeholders()
            sql2 = f"""
              SELECT asset_id AS station_id,
                     connector_id,
                     COALESCE(transaction_id::text, action_payload->>'transactionId', action_payload->>'transaction_id') AS transaction_id,
                     {_ts_col()} AS timestamp,
                     action,
                     action_payload
              FROM ocpp_events
              WHERE asset_id IN ({placeholders2})
                AND action IN ('StartTransaction','StopTransaction')
                AND {_ts_col()} BETWEEN {between2}
              ORDER BY asset_id, {_ts_col()}
            """
            params2 = list(stations) + [start_utc, end_utc]
            conn2 = get_conn()
            try:
                raw2 = pd.read_sql(sql2, conn2, params=params2)
            finally:
                conn2.close()

            if raw2 is None or raw2.empty:
                return pd.DataFrame()

            df2 = _parse_start_stop_rows(raw2.to_dict(orient='records'))
            if not df2.empty:
                df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True, errors="coerce")
                for c in [
                    "power_w",
                    "energy_wh",
                    "soc",
                    "amperage_offered",
                    "amperage_import",
                    "power_offered_w",
                    "voltage_v",
                ]:
                    if c in df2.columns:
                        df2[c] = pd.to_numeric(df2[c], errors="coerce")
            return df2

        rows = raw.to_dict(orient="records")
        df = _parse_meter_values_rows(rows)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            for c in [
                "power_w",
                "energy_wh",
                "soc",
                "amperage_offered",
                "amperage_import",
                "power_offered_w",
                "voltage_v",
            ]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # Legacy/SQLite fallback
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
                "postgres" if using_postgres() else "sqlite",
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
                    except Exception:
                        logger.exception(
                            "load_meter_values single-station query failed for %s (backend=%s)",
                            sid,
                            "postgres" if using_postgres() else "sqlite",
                        )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in [
            "power_w",
            "energy_wh",
            "soc",
            "amperage_offered",
            "amperage_import",
            "power_offered_w",
            "voltage_v",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_authorize(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_authorize rows (VID: id_tags) with a small time pad around the window.
    """
    if not stations:
        return pd.DataFrame()

    # Supabase/Postgres: Authorize events live in ocpp_events
    if using_postgres():
        start_pad = (pd.to_datetime(start_utc) - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_pad = (pd.to_datetime(end_utc) + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        placeholders = _make_placeholders(len(stations))
        between = _between_placeholders()
        sql = f"""
          SELECT asset_id AS station_id,
                 {_ts_col()} AS timestamp,
                 (action_payload->>'idTag') AS id_tag
          FROM ocpp_events
          WHERE asset_id IN ({placeholders})
            AND action = 'Authorize'
            AND {_ts_col()} BETWEEN {between}
          ORDER BY asset_id, {_ts_col()}
        """
        params = list(stations) + [start_pad, end_pad]
        conn = get_conn()
        try:
            df = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

        if df is None or df.empty:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["id_tag"] = df["id_tag"].astype(str)
        df = df[df["id_tag"].str.startswith("VID:", na=False)]
        return df

    # Legacy/SQLite fallback
    start_pad = (pd.to_datetime(start_utc) - pd.Timedelta(hours=2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    end_pad = (pd.to_datetime(end_utc) + pd.Timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
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
        except Exception:
            logger.exception(
                "load_authorize primary query failed (backend=%s, stations=%s, window=%s→%s)",
                "postgres" if using_postgres() else "sqlite",
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
                    except Exception:
                        logger.exception(
                            "load_authorize single-station query failed for %s (backend=%s)",
                            sid,
                            "postgres" if using_postgres() else "sqlite",
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
    if using_postgres():
        # Supabase/Postgres: StatusNotification events live in ocpp_events
        placeholders = _make_placeholders(len(stations)) if stations else ""
        where_in = f"asset_id IN ({placeholders}) AND " if stations else ""
        between = _between_placeholders()

        sql = f"""
          SELECT asset_id AS station_id,
                 connector_id,
                 {_ts_col()} AS timestamp,
                 (action_payload->>'status') AS status,
                 (action_payload->>'errorCode') AS error_code,
                 (action_payload->>'vendorErrorCode') AS vendor_error_code
          FROM ocpp_events
          WHERE {where_in} action = 'StatusNotification'
            AND {_ts_col()} BETWEEN {between}
          ORDER BY {_ts_col()} DESC
        """
        params = (list(stations) if stations else []) + [start_utc, end_utc]

        conn = get_conn()
        try:
            df = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "AKDT",
                    "Location",
                    "station_id",
                    "connector_id",
                    "status",
                    "error_code",
                    "vendor_error_code",
                    "impact",
                    "description",
                ]
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["AKDT"] = df["timestamp"].dt.tz_convert(AK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
        df["Location"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])

        # Optional Tritium enrichment
        if "vendor_error_code" in df.columns:
            try:
                tec = load_tritium_error_codes()
                if tec is not None and not tec.empty:
                    tec = tec.copy()
                    tec["code"] = pd.to_numeric(tec["code"], errors="coerce")
                    df["vendor_error_code"] = pd.to_numeric(df["vendor_error_code"], errors="coerce")
                    df = df.merge(
                        tec[["code", "impact", "description"]],
                        left_on="vendor_error_code",
                        right_on="code",
                        how="left",
                    )
                    if "code" in df.columns:
                        df = df.drop(columns=["code"])
            except Exception:
                logger.exception("load_status_history: Tritium enrichment failed")

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

    # Legacy/SQLite fallback
    where_in, params = _in_clause_and_params(stations, start_utc, end_utc)
    between = _between_placeholders()

    # Try selecting vendor_error_code if present; fallback to a reduced column set if not
    base_cols = "station_id, connector_id, {_ts} AS timestamp, status, error_code".format(
        _ts=_ts_col()
    )
    try_cols = base_cols + ", vendor_error_code"

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
            # This can be e.g. missing vendor_error_code column or SQL error
            logger.exception(
                "load_status_history extended-column query failed; falling back (backend=%s, stations=%s)",
                "postgres" if using_postgres() else "sqlite",
                stations,
            )
            try:
                df = pd.read_sql(sql_fallback, conn, params=params)
            except Exception:
                logger.exception(
                    "load_status_history fallback query failed completely (backend=%s, stations=%s)",
                    "postgres" if using_postgres() else "sqlite",
                    stations,
                )
                df = pd.DataFrame()
    finally:
        conn.close()

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "AKDT",
                "Location",
                "station_id",
                "connector_id",
                "status",
                "error_code",
                "vendor_error_code",
                "impact",
                "description",
            ]
        )

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
                df["vendor_error_code"] = pd.to_numeric(
                    df["vendor_error_code"], errors="coerce"
                )

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
            logger.exception("load_status_history: Tritium enrichment failed")

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
    if using_postgres():
        # Supabase/Postgres: websocket CONNECT/DISCONNECT is not yet normalized.
        # Return an empty frame with expected columns for now.
        return pd.DataFrame(
            columns=["AKDT", "Location", "station_id", "connection_id", "Connectivity"]
        )
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
        try:
            df = pd.read_sql(sql, conn, params=params)
        except Exception:
            logger.exception(
                "load_connectivity query failed (backend=%s, stations=%s, window=%s→%s)",
                "postgres" if using_postgres() else "sqlite",
                stations,
                start_utc,
                end_utc,
            )
            df = pd.DataFrame()
    finally:
        conn.close()

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["AKDT", "Location", "station_id", "connection_id", "Connectivity"]
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["AKDT"] = df["timestamp"].dt.tz_convert(AK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    ev = df.get("event").astype(str).str.upper()
    df["Connectivity"] = np.where(
        ev.str.contains("DISCONNECT"), "websocket DISCONNECT", "websocket CONNECT"
    )
    df["Location"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])
    cols = ["AKDT", "Location", "station_id", "connection_id", "Connectivity"]
    return df[[c for c in cols if c in df.columns]].sort_values(
        "AKDT", ascending=False, kind="mergesort"
    )


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