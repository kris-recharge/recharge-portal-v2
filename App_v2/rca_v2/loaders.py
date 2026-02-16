import logging
import os
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Optional
from .config import AK_TZ
from .db import get_conn, using_postgres, param_placeholder
from .constants import EVSE_DISPLAY


logger = logging.getLogger(__name__)


# --- Parameter normalization helpers ---
def _normalize_station_list(stations: Any) -> List[str]:
    """Normalize `stations` into a flat List[str] suitable for SQL params.

    This guards against callers accidentally passing pandas Series/Index, numpy arrays,
    or nested list-likes. psycopg cannot adapt a pandas Series object.
    """
    if stations is None:
        return []

    # First: unwrap common array-like containers
    if isinstance(stations, (pd.Series, pd.Index, np.ndarray)):
        stations = stations.tolist()

    # Ensure list-like
    if isinstance(stations, (set, tuple)):
        stations = list(stations)
    elif not isinstance(stations, list):
        stations = [stations]

    # Flatten one level of nested list-likes (including Series/ndarray inside lists)
    flat: List[Any] = []
    for s in stations:
        if s is None:
            continue
        if isinstance(s, (pd.Series, pd.Index, np.ndarray)):
            flat.extend(s.tolist())
        elif isinstance(s, (list, tuple, set)):
            flat.extend(list(s))
        else:
            flat.append(s)

    # Allow callers to pass either asset_id (as_...) or a friendly display name.
    # If a friendly name is provided, map it back to its asset_id.
    reverse_display = {v: k for k, v in (EVSE_DISPLAY or {}).items()}

    out: List[str] = []
    for s in flat:
        # Convert numpy scalar -> python scalar
        if hasattr(s, "item") and callable(getattr(s, "item")):
            try:
                s = s.item()
            except Exception:
                pass

        ss = str(s)
        # If the station value matches a display name, convert back to asset id
        if ss in reverse_display:
            ss = str(reverse_display[ss])
        out.append(ss)

    return out


def _normalize_time_param(x: Any):
    """Normalize start/end params into a single scalar time value.

    Why:
      - Callers sometimes accidentally pass a pandas Series/Index (e.g. a whole column).
      - psycopg/Postgres cannot bind a Series to a timestamptz placeholder.

    Behavior:
      - If a list-like / Series / Index / ndarray is provided, use its first element.
      - Always parse to a UTC timestamp when possible.
      - For Postgres, return a timezone-aware python datetime.
      - For SQLite, return an ISO-like string.
    """
    if x is None:
        return None if using_postgres() else ""

    # Unwrap common list-likes: Series/Index/ndarray/list/tuple/set
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple, set)):
        try:
            seq = list(x) if not isinstance(x, np.ndarray) else x.tolist()
        except Exception:
            seq = []
        x = seq[0] if seq else None
        if x is None:
            return None if using_postgres() else ""

    # Unwrap numpy scalar
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            x = x.item()
        except Exception:
            pass

    # If it's already a pandas Timestamp, keep it; otherwise try to parse
    ts = None
    if isinstance(x, pd.Timestamp):
        ts = x
    else:
        try:
            ts = pd.to_datetime(x, utc=True, errors="coerce")
        except Exception:
            ts = None

    if ts is None or ts is pd.NaT:
        # Last resort: stringify (but avoid passing giant Series strings)
        return None if using_postgres() else str(x)

    # Ensure UTC
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize("UTC")
        except Exception:
            pass

    if using_postgres():
        # psycopg prefers python datetime for timestamptz params
        try:
            return ts.to_pydatetime()
        except Exception:
            return ts

    # SQLite path: keep as a consistent string
    try:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(ts)


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
    """Parse ocpp_events MeterValues rows into the schema expected by the app.

    Supports two formats:
      1) Direct OCPP MeterValues payload stored as JSON in `action_payload` with key `meterValue`.
      2) Webhook-wrapped payload where `action_payload.data.log.action == 'MeterValues'` and
         `action_payload.data.log.payload` is a JSON string containing the OCPP payload.
    """

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

        # Unwrap webhook-style payload if present
        inner = None
        if isinstance(payload, dict):
            log_action = _safe_get(payload, "data", "log", "action")
            log_payload = _safe_get(payload, "data", "log", "payload")
            if log_action == "MeterValues" and isinstance(log_payload, str):
                try:
                    inner = json.loads(log_payload)
                except Exception:
                    inner = None

        mv_source = inner if isinstance(inner, dict) else (payload if isinstance(payload, dict) else None)

        # Fill connector/tx from inner if DB columns are null
        if mv_source is not None:
            if connector_id in (None, ""):
                connector_id = mv_source.get("connectorId")
            if transaction_id in (None, ""):
                transaction_id = mv_source.get("transactionId")

        try:
            connector_id = int(connector_id) if connector_id is not None else None
        except Exception:
            connector_id = None

        if transaction_id is not None:
            try:
                transaction_id = str(transaction_id)
            except Exception:
                pass

        meter_values = mv_source.get("meterValue") if mv_source is not None else None

        if not isinstance(meter_values, list) or not meter_values:
            frames.append(pd.DataFrame([{
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": transaction_id,
                "timestamp": received_at,
            }]))
            continue

        out_rows: List[Dict[str, Any]] = []
        for mv in meter_values:
            if not isinstance(mv, dict):
                continue
            ts = mv.get("timestamp") or received_at
            sampled = mv.get("sampledValue")
            metrics = _extract_sampled_values(sampled if isinstance(sampled, list) else [])
            out_rows.append({
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": transaction_id,
                "timestamp": ts,
                **metrics,
            })

        if out_rows:
            frames.append(pd.DataFrame(out_rows))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Fallback: parse Start/StopTransaction rows for minimal meter values
# -----------------------------
def _parse_start_stop_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a minimal meter-values-like dataframe from Start/StopTransaction events.

    This enables session building on Supabase when `MeterValues` is not available.

    Important Supabase nuance we've observed:
      - StartTransaction often has connector_id + meterStart, but NO transaction_id
      - StopTransaction often has transaction_id + meterStop, but NO connector_id

    To make downstream session pairing possible, we:
      - always emit an `action` column (StartTransaction / StopTransaction)
      - infer missing StopTransaction connector_id by pairing it to the most recent
        unmatched StartTransaction on the same station_id (stack/last-in-first-out)

    The resulting dataframe keeps the schema the app expects while being explicit
    enough for debugging and future improvements.
    """

    if not rows:
        return pd.DataFrame()

    # Normalize + sort rows so pairing is deterministic
    norm: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        # Ensure timestamp is a datetime for sorting; keep original string too if needed
        ts = rr.get("timestamp")
        rr["_ts_sort"] = pd.to_datetime(ts, utc=True, errors="coerce")
        norm.append(rr)

    # Sort: station_id, timestamp, Start before Stop when same timestamp
    def _action_rank(a: Any) -> int:
        return 0 if a == "StartTransaction" else 1

    norm.sort(key=lambda x: (
        str(x.get("station_id") or ""),
        x.get("_ts_sort") if x.get("_ts_sort") is not pd.NaT else pd.Timestamp.min.tz_localize("UTC"),
        _action_rank(x.get("action")),
    ))

    out: List[Dict[str, Any]] = []
    # Track open starts per station (LIFO stack): [{"ts":..., "connector_id":...}, ...]
    open_starts: Dict[str, List[Dict[str, Any]]] = {}

    for r in norm:
        station_id = r.get("station_id")
        received_at = r.get("timestamp")
        action = r.get("action")
        payload = r.get("action_payload")

        # payload may come in as dict (psycopg) or string
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            payload = {}

        # Connector id can be in different places depending on platform
        connector_id = (
            r.get("connector_id")
            or payload.get("connectorId")
            or _safe_get(payload, "connectorId")
            or _safe_get(payload, "connector_id")
        )
        try:
            connector_id = int(connector_id) if connector_id is not None else None
        except Exception:
            connector_id = None

        # Transaction id may be in table column or inside the JSON
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

        # Pairing heuristic: infer missing Stop connector_id using the most recent
        # unmatched Start on the same station.
        if station_id is not None:
            sid = str(station_id)
            open_starts.setdefault(sid, [])

            if action == "StartTransaction":
                if connector_id is not None:
                    open_starts[sid].append({"ts": r.get("_ts_sort"), "connector_id": connector_id})
            elif action == "StopTransaction":
                if connector_id is None and open_starts[sid]:
                    connector_id = open_starts[sid].pop().get("connector_id")

        out.append(
            {
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": str(tx) if tx is not None else None,
                "timestamp": received_at,
                "action": action,
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

    df = pd.DataFrame(out)
    # Ensure timestamp is parseable/consistent
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    return df

# -----------------------------
# EVSE pricing loader
# -----------------------------

def load_evse_pricing(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """Load EVSE pricing rules for the given stations and time window.

    Expected Postgres/Supabase table: `evse_pricing`
      - station_id (text)
      - effective_start (timestamptz)
      - effective_end (timestamptz, nullable)
      - connection_fee (numeric)
      - price_per_kwh (numeric)
      - price_per_min (numeric)
      - idle_fee_per_min (numeric)
      - idle_grace_min (numeric)

    We return all pricing rows whose effective window overlaps [start_utc, end_utc].
    """

    if not stations:
        return pd.DataFrame(
            columns=[
                "station_id",
                "effective_start",
                "effective_end",
                "connection_fee",
                "price_per_kwh",
                "price_per_min",
                "idle_fee_per_min",
                "idle_grace_min",
            ]
        )

    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)

    if not stations:
        return pd.DataFrame(
            columns=[
                "station_id",
                "effective_start",
                "effective_end",
                "connection_fee",
                "price_per_kwh",
                "price_per_min",
                "idle_fee_per_min",
                "idle_grace_min",
            ]
        )

    # Supabase/Postgres
    if using_postgres():
        placeholders = _make_placeholders(len(stations))
        ph = param_placeholder()

        # Overlap logic:
        #   effective_start <= end_utc
        #   AND (effective_end IS NULL OR effective_end >= start_utc)
        sql = f"""
          SELECT station_id,
                 effective_start,
                 effective_end,
                 connection_fee,
                 price_per_kwh,
                 price_per_min,
                 idle_fee_per_min,
                 idle_grace_min
          FROM evse_pricing
          WHERE station_id IN ({placeholders})
            AND effective_start <= {ph}
            AND (effective_end IS NULL OR effective_end >= {ph})
          ORDER BY station_id, effective_start
        """

        # Params order must match placeholders in SQL
        params = list(stations) + [end_utc, start_utc]
        conn = get_conn()
        try:
            df = pd.read_sql(sql, conn, params=params)
        except Exception:
            logger.exception(
                "load_evse_pricing failed (stations=%s, window=%s→%s)",
                stations,
                start_utc,
                end_utc,
            )
            df = pd.DataFrame()
        finally:
            conn.close()

        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "station_id",
                    "effective_start",
                    "effective_end",
                    "connection_fee",
                    "price_per_kwh",
                    "price_per_min",
                    "idle_fee_per_min",
                    "idle_grace_min",
                ]
            )

        # Normalize dtypes
        for c in ["effective_start", "effective_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

        for c in [
            "connection_fee",
            "price_per_kwh",
            "price_per_min",
            "idle_fee_per_min",
            "idle_grace_min",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # Legacy/SQLite path: pricing table may not exist in local SQLite.
    # Return an empty frame with the expected columns.
    return pd.DataFrame(
        columns=[
            "station_id",
            "effective_start",
            "effective_end",
            "connection_fee",
            "price_per_kwh",
            "price_per_min",
            "idle_fee_per_min",
            "idle_grace_min",
        ]
    )

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
    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)
    if not stations:
        return pd.DataFrame()

    # Supabase/Postgres: pull from ocpp_events and parse MeterValues payload JSON
    if using_postgres():
        placeholders = _make_placeholders(len(stations))
        between = _between_placeholders()
        station_expr = "COALESCE(asset_id, action_payload->'data'->'asset'->>'id')"
        sql = f"""
          SELECT {station_expr} AS station_id,
                 connector_id,
                 transaction_id,
                 {_ts_col()} AS timestamp,
                 action_payload
          FROM ocpp_events
          WHERE {station_expr} IN ({placeholders})
            AND (
                  action = 'MeterValues'
                  OR (action IS NULL AND action_payload->'data'->'log'->>'action' = 'MeterValues')
                )
            AND {_ts_col()} BETWEEN {between}
          ORDER BY {station_expr}, connector_id, transaction_id, {_ts_col()}
        """
        params = list(stations) + [start_utc, end_utc]
        conn = get_conn()
        try:
            raw = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

        if raw is None or raw.empty:
            # Second attempt: MeterValues are often only present in the raw webhook table.
            # The webhook wrapper looks like:
            #   {"data":{"log":{"action":"MeterValues","payload":"{...OCPP...}"},"asset":{"id":"as_..."},"timestamp":"..."},"createdAt":"..."}
            # We select the full wrapper JSON as action_payload so `_parse_meter_values_rows` can unwrap `data.log.payload`.
            placeholders_mv = _make_placeholders(len(stations))
            between_mv = _between_placeholders()
            mv_station_expr = "(t.payload->'data'->'asset'->>'id')"
            mv_ts_expr = "COALESCE((t.payload->>'createdAt')::timestamptz, (t.payload->'data'->>'timestamp')::timestamptz, t.received_at)"

            sql_mv = f"""
              SELECT {mv_station_expr} AS station_id,
                     NULL::int AS connector_id,
                     NULL::text AS transaction_id,
                     {mv_ts_expr} AS timestamp,
                     t.payload AS action_payload
              FROM lynkwell_webhook_raw t
              WHERE (t.payload->'data'->'log'->>'action') = 'MeterValues'
                AND {mv_station_expr} IN ({placeholders_mv})
                AND {mv_ts_expr} BETWEEN {between_mv}
              ORDER BY {mv_station_expr}, {mv_ts_expr}
            """
            params_mv = list(stations) + [start_utc, end_utc]
            conn_mv = get_conn()
            try:
                raw_mv = pd.read_sql(sql_mv, conn_mv, params=params_mv)
            finally:
                conn_mv.close()

            if raw_mv is not None and not raw_mv.empty:
                rows_mv = raw_mv.to_dict(orient="records")
                df_mv = _parse_meter_values_rows(rows_mv)
                if not df_mv.empty and "action" not in df_mv.columns:
                    df_mv["action"] = "MeterValues"
                if not df_mv.empty:
                    df_mv["timestamp"] = pd.to_datetime(df_mv["timestamp"], utc=True, errors="coerce")
                    for c in [
                        "power_w",
                        "energy_wh",
                        "soc",
                        "amperage_offered",
                        "amperage_import",
                        "power_offered_w",
                        "voltage_v",
                    ]:
                        if c in df_mv.columns:
                            df_mv[c] = pd.to_numeric(df_mv[c], errors="coerce")
                # Return MeterValues from webhook_raw; do NOT fall through to Start/Stop fallback.
                return df_mv

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

        if not df.empty and "action" not in df.columns:
            df["action"] = "MeterValues"

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
    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)
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
    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)
    placeholders = _make_placeholders(len(stations)) if stations else ""
    where_in = f"station_id IN ({placeholders}) AND " if stations else ""
    return where_in, list(stations) + [start_utc, end_utc]


def load_status_history(stations, start_utc: str, end_utc: str) -> pd.DataFrame:
    """
    Load realtime_status_notifications for the window; newest first; AKDT strings + friendly names.
    Includes vendor_error_code if the column exists (falls back cleanly if not).
    """
    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)
    if using_postgres():
        # Supabase/Postgres: StatusNotification events live in ocpp_events
        # To prevent accidental "only a few hours" views when StatusNotification volume is high,
        # we apply a HIGH default row limit that can be tuned via env var.
        # Set STATUS_HISTORY_MAX_ROWS=0 to disable limiting entirely.
        try:
            max_rows = int(os.getenv("STATUS_HISTORY_MAX_ROWS", "20000"))
        except Exception:
            max_rows = 20000
        limit_clause = f"LIMIT {max_rows}" if max_rows and max_rows > 0 else ""

        logger.info(
            "load_status_history: stations=%s window=%s→%s max_rows=%s",
            (len(stations) if stations else 0),
            start_utc,
            end_utc,
            max_rows,
        )

        placeholders = _make_placeholders(len(stations)) if stations else ""
        between = _between_placeholders()

        # Broaden to allow both direct and webhook-wrapped StatusNotification events.
        station_expr = "COALESCE(asset_id, action_payload->'data'->'asset'->>'id')"
        is_statusnotification = "(action = 'StatusNotification' OR (action IS NULL AND action_payload->'data'->'log'->>'action' = 'StatusNotification'))"

        sql = f"""
          SELECT {station_expr} AS station_id,
                 connector_id,
                 {_ts_col()} AS timestamp,
                 (action_payload->>'status') AS status,
                 (action_payload->>'errorCode') AS error_code,
                 (action_payload->>'vendorErrorCode') AS vendor_error_code
          FROM ocpp_events
          WHERE { (f"{station_expr} IN ({placeholders}) AND " if stations else "") } {is_statusnotification}
            AND {_ts_col()} BETWEEN {between}
          ORDER BY {_ts_col()} DESC
          {limit_clause}
        """
        params = (list(stations) if stations else []) + [start_utc, end_utc]

        conn = get_conn()
        try:
            df = pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

        if df is None or df.empty:
            # Diagnostics: verify the DB has rows for this same filter.
            try:
                conn2 = get_conn()
                cur = conn2.cursor()
                # Build the same WHERE clause, but only return counts + min/max.
                where_station = ""
                params2 = []
                if stations:
                    where_station = f"{station_expr} IN ({placeholders}) AND "
                    params2.extend(list(stations))
                params2.extend([start_utc, end_utc])

                diag_sql = f"""
                  SELECT count(*) AS n,
                         min({_ts_col()}) AS min_ts,
                         max({_ts_col()}) AS max_ts
                  FROM ocpp_events
                  WHERE {where_station}{is_statusnotification}
                    AND {_ts_col()} BETWEEN {between}
                """
                cur.execute(diag_sql, tuple(params2))
                n, min_ts, max_ts = cur.fetchone()
                logger.warning(
                    "load_status_history EMPTY: stations=%s window=%s→%s diag_count=%s diag_window=%s→%s",
                    (len(stations) if stations else 0),
                    start_utc,
                    end_utc,
                    n,
                    min_ts,
                    max_ts,
                )
                cur.close()
                conn2.close()
            except Exception:
                logger.exception("load_status_history EMPTY: diagnostic query failed")

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

        try:
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()
            logger.info(
                "load_status_history: returned_rows=%s returned_window=%s→%s",
                len(df),
                min_ts,
                max_ts,
            )
        except Exception:
            logger.exception("load_status_history: failed to log returned window")

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
    stations = _normalize_station_list(stations)
    start_utc = _normalize_time_param(start_utc)
    end_utc = _normalize_time_param(end_utc)
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