import json
import pandas as pd
from typing import Any, Dict, List
from .config import AK_TZ
from .constants import EVSE_LOCATION, CONNECTOR_TYPE

def _to_float(val):
    try:
        return float(val)
    except Exception:
        return None

def _parse_meter_values_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out = []
    for r in rows:
        station_id = r.get("station_id")
        connector_id = r.get("connector_id")
        transaction_id = r.get("transaction_id")
        timestamp = r.get("timestamp")
        action = r.get("action")
        payload = r.get("action_payload")

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None

        energy_wh = None
        power_w = None
        soc = None
        voltage_v = None
        amperage_import = None
        amperage_offered = None
        power_offered_w = None

        if isinstance(payload, dict):
            # Parse nested meter values if present
            meter_values = payload.get("meterValue") or payload.get("meterValues")
            if meter_values and isinstance(meter_values, list):
                # Take the last meter value reading in the list
                last_mv = meter_values[-1]
                if isinstance(last_mv, dict):
                    energy_wh = _to_float(last_mv.get("energyActiveImportRegister"))
                    power_w = _to_float(last_mv.get("powerActiveImport"))
                    soc = _to_float(last_mv.get("stateOfCharge"))
                    voltage_v = _to_float(last_mv.get("voltage"))
                    amperage_import = _to_float(last_mv.get("currentImport"))
                    amperage_offered = _to_float(last_mv.get("currentOffer"))
                    power_offered_w = _to_float(last_mv.get("powerOffer"))

        out.append(
            {
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": transaction_id,
                "timestamp": timestamp,
                "energy_wh": energy_wh,
                "power_w": power_w,
                "soc": soc,
                "voltage_v": voltage_v,
                "amperage_import": amperage_import,
                "amperage_offered": amperage_offered,
                "power_offered_w": power_offered_w,
            }
        )
    df = pd.DataFrame(out)
    return df


def _parse_start_stop_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a minimal meter-values-like dataframe from StartTransaction/StopTransaction.

    Enables session building even when MeterValues are not present.
    Produces: station_id, connector_id, transaction_id, timestamp, energy_wh
    and includes the other expected numeric columns as null.
    """
    out: List[Dict[str, Any]] = []

    for r in rows:
        station_id = r.get("station_id")
        connector_id = r.get("connector_id")
        transaction_id = r.get("transaction_id")
        received_at = r.get("timestamp")
        action = r.get("action")
        payload = r.get("action_payload")

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = None

        ts = received_at
        if isinstance(payload, dict):
            ts = payload.get("timestamp") or received_at

        energy_wh = None
        if isinstance(payload, dict):
            if action == "StartTransaction":
                energy_wh = _to_float(payload.get("meterStart"))
            elif action == "StopTransaction":
                energy_wh = _to_float(payload.get("meterStop"))

        out.append(
            {
                "station_id": station_id,
                "connector_id": connector_id,
                "transaction_id": transaction_id,
                "timestamp": ts,
                "energy_wh": energy_wh,
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

    return pd.DataFrame(out)


def load_meter_values(stations, start_utc, end_utc):
    from .db import get_conn, using_postgres, _make_placeholders, _between_placeholders, _ts_col

    # Supabase/Postgres: pull from ocpp_events. Prefer MeterValues; also pull
    # Start/StopTransaction as a fallback so sessions can be built.
    if using_postgres():
        placeholders = _make_placeholders(len(stations))
        between = _between_placeholders()
        params = list(stations) + [start_utc, end_utc]

        conn = get_conn()
        try:
            df_parts: List[pd.DataFrame] = []

            # 1) MeterValues (if present)
            sql_mv = f"""
              SELECT asset_id AS station_id,
                     connector_id,
                     transaction_id,
                     {_ts_col()} AS timestamp,
                     action,
                     action_payload
              FROM ocpp_events
              WHERE asset_id IN ({placeholders})
                AND action = 'MeterValues'
                AND {_ts_col()} BETWEEN {between}
              ORDER BY asset_id, connector_id, transaction_id, {_ts_col()}
            """
            raw_mv = pd.read_sql(sql_mv, conn, params=params)
            if raw_mv is not None and not raw_mv.empty:
                df_parts.append(_parse_meter_values_rows(raw_mv.to_dict(orient="records")))

            # 2) Start/StopTransaction fallback (meterStart/meterStop)
            sql_tx = f"""
              SELECT asset_id AS station_id,
                     connector_id,
                     transaction_id,
                     {_ts_col()} AS timestamp,
                     action,
                     action_payload
              FROM ocpp_events
              WHERE asset_id IN ({placeholders})
                AND action IN ('StartTransaction','StopTransaction')
                AND {_ts_col()} BETWEEN {between}
              ORDER BY asset_id, connector_id, transaction_id, {_ts_col()}
            """
            raw_tx = pd.read_sql(sql_tx, conn, params=params)
            if raw_tx is not None and not raw_tx.empty:
                df_parts.append(_parse_start_stop_rows(raw_tx.to_dict(orient="records")))

        finally:
            conn.close()

        if not df_parts:
            return pd.DataFrame()

        df = pd.concat([d for d in df_parts if d is not None and not d.empty], ignore_index=True)
        if df is None or df.empty:
            return pd.DataFrame()

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

        df = df.sort_values(["station_id", "connector_id", "transaction_id", "timestamp"], kind="mergesort")
        return df

    # SQLite/legacy path below (not changed)
    # ... (existing code not shown) ...