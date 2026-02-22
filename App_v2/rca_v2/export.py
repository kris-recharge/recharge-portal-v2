

"""Excel export helpers for ReCharge Alaska Portal v2.

This module centralizes all Excel export logic so `app_v2.py` can stay lean.

Design goals:
- Works with both Postgres-backed data (Supabase) and local SQLite dev data.
- Defensive about column names (schemas can evolve).
- Keeps sheet creation and formatting in one place.

Expected usage from app_v2.py:

    from rca_v2.export import build_export_xlsx_bytes

    xlsx_bytes = build_export_xlsx_bytes(
        sessions_df=sessions_df,
        meter_values_df=meter_values_df,
        status_df=status_df,
        connectivity_events_df=connectivity_events_df,
        evse_display=evse_display,
        tz_name="America/Anchorage",
        start_utc=start_utc,
        end_utc=end_utc,
    )

Then feed `xlsx_bytes` to st.download_button.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, Optional, Tuple

import re

import pandas as pd

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ----------------------------
# Public API
# ----------------------------


def build_export_xlsx_bytes(
    *,
    sessions_df: pd.DataFrame | None = None,
    meter_values_df: pd.DataFrame | None = None,
    status_df: pd.DataFrame | None = None,
    connectivity_events_df: pd.DataFrame | None = None,
    # Back-compat aliases used by older app_v2.py exports
    connectivity_df: pd.DataFrame | None = None,
    stations: Any = None,
    evse_display: Dict[str, str] | None = None,
    tz_name: str = "America/Anchorage",
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    filename_prefix: str = "recharge_export",
    authorize_methods_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> bytes:
    """Build an .xlsx export (bytes).

    Sheets:
      - Sessions
      - MeterValues (window)
      - Status
      - Connectivity Data
      - Connectivity Summary

    The caller controls what data to include by passing dataframes.

    Notes:
      - This is the canonical export entrypoint used by `app_v2.py`.
      - It is intentionally tolerant of schema differences between local SQLite and
        Supabase/Postgres.
    """

    evse_display = evse_display or {}

    # ----------------------------
    # Backwards-compatible argument handling
    # ----------------------------
    # Some call-sites still pass `connectivity_df` instead of `connectivity_events_df`.
    if connectivity_events_df is None and connectivity_df is not None:
        connectivity_events_df = connectivity_df

    # Older call-sites passed extra keyword args (e.g. `stations`). We don't need them
    # in the export module; ignore safely.
    _ = stations
    kwargs.pop("stations", None)
    kwargs.pop("connectivity_df", None)

    sheets: Dict[str, pd.DataFrame] = {}

    # --- Sessions ---
    if sessions_df is not None and not sessions_df.empty:
        sessions_sheet = prep_sessions_sheet(
            sessions_df,
            evse_display,
            tz_name=tz_name,
            authorize_methods_df=authorize_methods_df,
        )
        sheets["Sessions"] = sessions_sheet
    else:
        sheets["Sessions"] = pd.DataFrame()

    # --- MeterValues (window) ---
    if meter_values_df is not None and not meter_values_df.empty:
        sheets["MeterValues (window)"] = prep_metervalues_sheet(
            meter_values_df, evse_display, tz_name=tz_name
        )
    else:
        sheets["MeterValues (window)"] = pd.DataFrame()

    # --- Status ---
    if status_df is not None and not status_df.empty:
        sheets["Status"] = prep_status_sheet(status_df, evse_display, tz_name=tz_name)
    else:
        sheets["Status"] = pd.DataFrame()

    # --- Connectivity ---
    if connectivity_events_df is not None and not connectivity_events_df.empty:
        conn_data = prep_connectivity_data_sheet(connectivity_events_df, evse_display, tz_name=tz_name)
        conn_summary = prep_connectivity_summary_sheet(conn_data)
        sheets["Connectivity Data"] = conn_data
        sheets["Connectivity Summary"] = conn_summary
    else:
        sheets["Connectivity Data"] = pd.DataFrame()
        sheets["Connectivity Summary"] = pd.DataFrame()

    return write_xlsx_bytes(
        sheets,
        tz_name=tz_name,
        start_utc=start_utc,
        end_utc=end_utc,
        filename_prefix=filename_prefix,
    )


def build_export_xlsx(
    *,
    sess_df: pd.DataFrame | None = None,
    mv_df: pd.DataFrame | None = None,
    status_df: pd.DataFrame | None = None,
    connectivity_events_df: pd.DataFrame | None = None,
    # Back-compat alias
    connectivity_df: pd.DataFrame | None = None,
    stations: Any = None,
    evse_display: Dict[str, str] | None = None,
    tz_name: str = "America/Anchorage",
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    filename_prefix: str = "recharge_export",
    authorize_methods_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> bytes:
    return build_export_xlsx_bytes(
        sessions_df=sess_df,
        meter_values_df=mv_df,
        status_df=status_df,
        connectivity_events_df=connectivity_events_df,
        connectivity_df=connectivity_df,
        stations=stations,
        evse_display=evse_display,
        tz_name=tz_name,
        start_utc=start_utc,
        end_utc=end_utc,
        filename_prefix=filename_prefix,
        authorize_methods_df=authorize_methods_df,
        **kwargs,
    )


# ----------------------------
# Sheet prep helpers
# ----------------------------


def prep_sessions_sheet(
    df: pd.DataFrame,
    evse_display: Dict[str, str],
    *,
    tz_name: str,
    authorize_methods_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare Sessions sheet.

    Requirements from Kris:
      - remove redundant connector column D (connector #)
      - remove redundant asset_id/station_id column (they called it Column K earlier)
      - keep friendly EVSE name

    Because column names vary between local/cloud, this function is defensive.
    """

    out = df.copy()

    # Normalize station id column
    station_col = _first_existing(out, [
        "station_id",
        "asset_id",
        "evse_id",
        "station",
    ])

    # Add/normalize friendly name
    if station_col:
        out["EVSE"] = out[station_col].map(evse_display).fillna(out[station_col])

    # Convert time columns to local
    out = _convert_time_cols(
        out,
        tz_name=tz_name,
        cols=["timestamp", "received_at", "ts"],
        assume_utc_for_naive=True,
    )

    # Prefer consistent column names if present
    rename_map = {}
    # Start
    for c in ["start_time", "start", "Start Date/Time", "Start Time"]:
        if c in out.columns:
            rename_map[c] = "Start Time (local)"
            break
    # End
    for c in ["end_time", "end", "End Date/Time", "End Time"]:
        if c in out.columns:
            rename_map[c] = "End Time (local)"
            break

    # Energy
    if "energy_kwh" in out.columns:
        rename_map["energy_kwh"] = "Energy Delivered (kWh)"
    elif "kwh" in out.columns:
        rename_map["kwh"] = "Energy Delivered (kWh)"

    # Duration
    if "duration_min" in out.columns:
        rename_map["duration_min"] = "Duration (min)"

    # SoC
    if "soc_start" in out.columns:
        rename_map["soc_start"] = "SoC Start (%)"
    if "soc_end" in out.columns:
        rename_map["soc_end"] = "SoC End (%)"

    # Id Tag
    if "id_tag" in out.columns:
        rename_map["id_tag"] = "ID Tag"

    out = out.rename(columns=rename_map)

    # --- Authorization columns (raw + guessed method) ---
    # Kris wants both:
    #  - the raw authorize string (ID Tag)
    #  - a best-guess method/category (RFID / CC / AutoCharge / App / Unknown)

    # Ensure a stable raw authorize column.
    if "ID Tag" in out.columns and "Authorize (raw)" not in out.columns:
        out["Authorize (raw)"] = out["ID Tag"].where(out["ID Tag"].notna(), pd.NA)

    # If the upstream already provided a method, normalize its label.
    if "authorization_method" in out.columns and "Auth Method (guess)" not in out.columns:
        out["Auth Method (guess)"] = out["authorization_method"]

    # Otherwise, try to look it up from an authorize_methods dataframe if provided.
    # Prefer matching by (station_id, transaction_id) when possible (most reliable),
    # otherwise fall back to matching by ID Tag.
    if authorize_methods_df is not None and not authorize_methods_df.empty:
        try:
            am = authorize_methods_df.copy()

            # Choose the best tag column from authorize_methods.
            # Prefer id_tag_effective (true token). If sessions only carries VID tokens,
            # we can also match against id_tag_vid.
            tag_col = None
            for c in ["id_tag_effective", "id_tag", "id_tag_vid"]:
                if c in am.columns:
                    tag_col = c
                    break

            needed = [c for c in ["station_id", "asset_id", "transaction_id", "authorization_method", "confidence"] if c in am.columns]
            if tag_col is not None:
                needed.append(tag_col)

            if "authorization_method" in am.columns and needed:
                am = am[needed].dropna(subset=["authorization_method"], how="any")

                # Normalize station id column name to station_id
                if "station_id" not in am.columns and "asset_id" in am.columns:
                    am = am.rename(columns={"asset_id": "station_id"})

                # Normalize types
                if "station_id" in am.columns:
                    am["station_id"] = am["station_id"].astype(str)
                if "transaction_id" in am.columns:
                    am["transaction_id"] = am["transaction_id"].astype(str)
                if tag_col is not None:
                    am[tag_col] = am[tag_col].astype(str)

                # De-dupe: keep the most recent classification per key
                # (authorize_received_at may exist; if so, use it to sort)
                if "authorize_received_at" in authorize_methods_df.columns:
                    am2 = authorize_methods_df.copy()
                    if "asset_id" in am2.columns and "station_id" not in am2.columns:
                        am2 = am2.rename(columns={"asset_id": "station_id"})
                    if tag_col and tag_col in am2.columns:
                        am2[tag_col] = am2[tag_col].astype(str)
                    if "station_id" in am2.columns:
                        am2["station_id"] = am2["station_id"].astype(str)
                    if "transaction_id" in am2.columns:
                        am2["transaction_id"] = am2["transaction_id"].astype(str)
                    am2["authorize_received_at"] = pd.to_datetime(am2["authorize_received_at"], errors="coerce")
                    am2 = am2.sort_values("authorize_received_at")
                    cols_keep = [c for c in ["station_id", "transaction_id", tag_col, "authorization_method", "confidence"] if c in am2.columns]
                    am = am2[cols_keep].dropna(subset=["authorization_method"], how="any")

                am = am.drop_duplicates()

                out_id = out.copy()

                # Determine station id column present in sessions df
                sess_station_col = station_col if station_col in out_id.columns else (
                    "station_id" if "station_id" in out_id.columns else ("asset_id" if "asset_id" in out_id.columns else None)
                )

                # Normalize types in sessions df
                if sess_station_col:
                    out_id[sess_station_col] = out_id[sess_station_col].astype(str)
                if "transaction_id" in out_id.columns:
                    out_id["transaction_id"] = out_id["transaction_id"].astype(str)

                # 1) Preferred merge by (station_id, transaction_id)
                if sess_station_col and "transaction_id" in out_id.columns and "station_id" in am.columns and "transaction_id" in am.columns:
                    out_id = out_id.merge(
                        am,
                        how="left",
                        left_on=[sess_station_col, "transaction_id"],
                        right_on=["station_id", "transaction_id"],
                        suffixes=("", "_am"),
                    )
                # 2) Enhanced fallback merge by ID Tag, supporting both effective/raw tags and VID mapping
                elif "ID Tag" in out_id.columns:
                    out_id["ID Tag"] = out_id["ID Tag"].astype(str)

                    # First try matching against the primary tag_col (id_tag_effective/id_tag/id_tag_vid)
                    if tag_col is not None and tag_col in am.columns:
                        out_id = out_id.merge(
                            am[[tag_col, "authorization_method", "confidence"]].drop_duplicates(),
                            how="left",
                            left_on="ID Tag",
                            right_on=tag_col,
                            suffixes=("", "_am"),
                        )

                    # If Sessions carries VID tokens, authorize_methods may store the actual token
                    # in id_tag_effective and the VID mapping in id_tag_vid. Try id_tag_vid as a
                    # second pass to fill anything still missing.
                    if "id_tag_vid" in am.columns:
                        need_fill = (
                            "authorization_method" not in out_id.columns
                            or out_id["authorization_method"].isna().any()
                        )
                        if need_fill:
                            out2 = out_id.merge(
                                am[["id_tag_vid", "authorization_method", "confidence"]].drop_duplicates(),
                                how="left",
                                left_on="ID Tag",
                                right_on="id_tag_vid",
                                suffixes=("", "_am2"),
                            )

                            # Coalesce from the 2nd pass only where missing
                            if "authorization_method" in out_id.columns and "authorization_method_am2" in out2.columns:
                                out2["authorization_method"] = out_id["authorization_method"].where(
                                    out_id["authorization_method"].notna(),
                                    out2["authorization_method_am2"],
                                )
                            elif "authorization_method_am2" in out2.columns and "authorization_method" not in out2.columns:
                                out2["authorization_method"] = out2["authorization_method_am2"]

                            if "confidence" in out_id.columns and "confidence_am2" in out2.columns:
                                out2["confidence"] = out_id["confidence"].where(
                                    out_id["confidence"].notna(),
                                    out2["confidence_am2"],
                                )
                            elif "confidence_am2" in out2.columns and "confidence" not in out2.columns:
                                out2["confidence"] = out2["confidence_am2"]

                            out_id = out2.drop(columns=[c for c in ["authorization_method_am2", "confidence_am2"] if c in out2.columns], errors="ignore")

                if "authorization_method" in out_id.columns:
                    # Prefer the table's method (passthrough). If the column already exists,
                    # only fill missing values from the table (don't overwrite real values).
                    if "Auth Method (guess)" not in out_id.columns:
                        out_id["Auth Method (guess)"] = out_id["authorization_method"]
                    else:
                        out_id["Auth Method (guess)"] = out_id["Auth Method (guess)"].where(
                            out_id["Auth Method (guess)"].notna(),
                            out_id["authorization_method"],
                        )

                # Keep confidence if present (useful in exports)
                if "confidence" in out_id.columns and "Auth Method Confidence" not in out_id.columns:
                    out_id["Auth Method Confidence"] = out_id["confidence"]

                # Drop extra join columns
                drop_extra = [c for c in ["station_id_am", "confidence", "authorization_method", tag_col, "id_tag_vid"] if c in out_id.columns]
                out = out_id.drop(columns=drop_extra, errors="ignore")
        except Exception:
            pass


    # Normalize placeholder strings that can appear after merges
    for c in ["Auth Method (guess)", "Auth Method Confidence"]:
        if c in out.columns:
            out[c] = out[c].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    # Final fallback heuristic: fill anything still missing
    if "Auth Method (guess)" not in out.columns or out["Auth Method (guess)"].isna().any():
        def _guess_auth_method(x: Any) -> str:
            try:
                s = str(x)
            except Exception:
                return "Unknown"
            if not s or s.lower() == "nan":
                return "Unknown"
            if s.startswith("VID:"):
                return "AutoCharge"
            # 14-char hex tokens match CC reader in LynkWell-confirmed data
            if re.fullmatch(r"[0-9A-Fa-f]{14}", s):
                return "CC"
            # 20-char opaque A–Z0–9 tokens match App authorization
            if re.fullmatch(r"[A-Z0-9]{20}", s):
                return "App"
            # other common RFID-like hex tokens
            if re.fullmatch(r"[0-9A-Fa-f]{8,32}", s):
                return "RFID"
            return "Unknown"

        if "ID Tag" in out.columns:
            guessed = out["ID Tag"].apply(_guess_auth_method)
            if "Auth Method (guess)" not in out.columns:
                out["Auth Method (guess)"] = guessed
            else:
                out["Auth Method (guess)"] = out["Auth Method (guess)"].fillna(guessed)

    def _soc_to_fraction(val: Any) -> Any:
        try:
            x = float(val)
        except Exception:
            return val
        if pd.isna(x):
            return x
        # Already fraction
        if 0.0 <= x <= 1.0:
            return x
        # Percent -> fraction
        if 1.0 < x <= 100.0:
            return x / 100.0
        return x

    for soc_col in ["SoC Start (%)", "SoC End (%)", "SoC Start", "SoC End"]:
        if soc_col in out.columns:
            out[soc_col] = out[soc_col].apply(_soc_to_fraction)

    # Flag attempted/failed sessions (keeps them visible in export without forcing them into totals).
    # Convention: anything under 1 kWh is considered an "attempt" for reporting.
    if "Energy Delivered (kWh)" in out.columns:
        try:
            out["Attempted (<1 kWh)"] = pd.to_numeric(out["Energy Delivered (kWh)"], errors="coerce").fillna(0) < 1
        except Exception:
            pass

    # Drop columns Kris called redundant
    # - Connector number (often connector_id or connector_number)
    # - Remove connector_id only if both connector type and transaction id are present
    drop_cols = []
    drop_cols += [c for c in ["connector_num", "connector_number", "connector", "Connector #"] if c in out.columns]

    # Keep connector_id for reporting (Kris wants it in the export).
    has_connector_type = any(c in out.columns for c in ["connector_type", "Connector Type"])
    has_txn = any(c in out.columns for c in ["transaction_id", "Transaction ID"])
    # Intentionally do NOT drop connector_id.

    # Remove the raw station id columns if we have friendly EVSE,
    # but only if the column is exactly asset_id or station_id
    if "EVSE" in out.columns and station_col in {"station_id", "asset_id"} and station_col in out.columns:
        drop_cols.append(station_col)

    # Also drop any explicitly requested "asset_id" column
    if "asset_id" in out.columns:
        drop_cols.append("asset_id")

    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    # Prefer a stable export column order.
    preferred = [
        "Start Time (local)",
        "End Time (local)",
        "EVSE",
        "Connector Type",
        "Max Power (kW)",
        "Energy Delivered (kWh)",
        "Duration (min)",
        "SoC Start (%)",
        "SoC End (%)",
        "SoC Start",
        "SoC End",
        "ID Tag",
        "Authorize (raw)",
        "Auth Method (guess)",
        "Auth Method Confidence",
        "Estimated Revenue ($)",
        "transaction_id",
        "connector_id",
        # Add the new flag at the end of preferred columns
        "Attempted (<1 kWh)",
    ]

    # Some datasets use different casing/labels; allow those too.
    preferred += [
        "connector_type",
        "max_power_kw",
        "Max Power",
        "energy_kwh",
        "Energy Delivered",
        "duration_min",
        "soc_start",
        "soc_end",
        "id_tag",
        "Transaction ID",
        "Connector ID",
    ]

    out = _order_cols(out, preferred)

    # If connector_id exists, place it immediately after transaction_id if both exist.
    cols = list(out.columns)
    if "connector_id" in cols:
        if "transaction_id" in cols:
            cols.remove("connector_id")
            cols.insert(cols.index("transaction_id") + 1, "connector_id")
            out = out[cols]

    return out


def prep_metervalues_sheet(df: pd.DataFrame, evse_display: Dict[str, str], *, tz_name: str) -> pd.DataFrame:
    """Prepare MeterValues (window) sheet."""
    out = df.copy()

    station_col = _first_existing(out, ["station_id", "asset_id", "evse_id"])
    if station_col:
        out["EVSE"] = out[station_col].map(evse_display).fillna(out[station_col])

    out = _convert_time_cols(out, tz_name=tz_name, cols=["timestamp", "received_at", "ts"])

    # Prefer a friendly ordering
    preferred = [
        "timestamp",
        "received_at",
        "ts",
        "EVSE",
        "connector_id",
        "transaction_id",
        "power_w",
        "energy_wh",
        "energy_kwh",
        "soc",
        "voltage_v",
        "current_a",
    ]
    out = _order_cols(out, preferred)

    return out


def prep_status_sheet(df: pd.DataFrame, evse_display: Dict[str, str], *, tz_name: str) -> pd.DataFrame:
    """Prepare Status sheet."""
    out = df.copy()

    station_col = _first_existing(out, ["station_id", "asset_id", "evse_id"])
    if station_col:
        out["EVSE"] = out[station_col].map(evse_display).fillna(out[station_col])

    # Convert common UTC timestamp columns to local clock time (tz-naive for Excel)
    out = _convert_time_cols(out, tz_name=tz_name, cols=["timestamp", "received_at", "ts"])

    # Normalize timestamp column for export.
    # Some upstream dataframes already include a local-time column named AKDT/AKST.
    ts_col = _first_existing(
        out,
        [
            "Timestamp (AKST)",
            "Timestamp (AKDT)",
            "AKST",
            "AKDT",
            "timestamp",
            "received_at",
            "ts",
        ],
    )
    if ts_col and ts_col in out.columns and ts_col != "Timestamp (AKST)":
        out = out.rename(columns={ts_col: "Timestamp (AKST)"})

    # If the timestamp is present but still object/string, try to coerce to datetime.
    if "Timestamp (AKST)" in out.columns:
        try:
            out["Timestamp (AKST)"] = pd.to_datetime(out["Timestamp (AKST)"], errors="coerce")
        except Exception:
            pass

    # Normalize vendor error meta to requested labels
    if "vendor_error_impact" in out.columns:
        out = out.rename(columns={"vendor_error_impact": "impact"})
    if "vendor_error_desc" in out.columns:
        out = out.rename(columns={"vendor_error_desc": "description"})
    elif "vendor_error_description" in out.columns:
        out = out.rename(columns={"vendor_error_description": "description"})

    preferred = [
        "Timestamp (AKST)",
        "EVSE",
        "connector_id",
        "status",
        "impact",
        "vendor_error_code",
        "description",
    ]

    out = out.drop(
        columns=[c for c in ["Location", "location", "station_id", "error_code"] if c in out.columns],
        errors="ignore",
    )
    out = _order_cols(out, preferred)

    return out


def prep_connectivity_data_sheet(df: pd.DataFrame, evse_display: Dict[str, str], *, tz_name: str) -> pd.DataFrame:
    """Build per-event connectivity rows and compute durations.

    Input expectations (best-effort):
      - station/asset id column: station_id|asset_id|evse_id
      - timestamp column: timestamp|received_at|ts
      - event/action column: event|action|type (values include CONNECT/DISCONNECT)

    Output columns:
      - Start (local)
      - End (local)
      - Duration (min)
      - EVSE
      - station_id (kept for traceability)
    """

    src = df.copy()

    station_col = _first_existing(src, ["station_id", "asset_id", "evse_id"])
    ts_col = _first_existing(src, ["timestamp", "received_at", "ts"])
    ev_col = _first_existing(src, ["event", "action", "type"])

    if not station_col or not ts_col or not ev_col:
        # Can't compute; return raw-ish with best-effort friendly mapping
        if station_col:
            src["EVSE"] = src[station_col].map(evse_display).fillna(src[station_col])
        return src

    # Ensure timestamps are datetime
    src[ts_col] = pd.to_datetime(src[ts_col], utc=True, errors="coerce")

    # Standardize event names
    src[ev_col] = src[ev_col].astype(str).str.upper()

    # Sort per EVSE
    src = src.sort_values([station_col, ts_col])

    rows = []
    for sid, g in src.groupby(station_col, sort=False):
        g = g.sort_values(ts_col)
        last_connect = None
        last_disconnect = None

        # We create a duration row whenever we see a transition.
        for _, r in g.iterrows():
            ev = r[ev_col]
            t = r[ts_col]
            if pd.isna(t):
                continue

            if "CONNECT" in ev:
                # if we have a preceding DISCONNECT, duration is disconnect->connect (downtime)
                if last_disconnect is not None and t >= last_disconnect:
                    rows.append(
                        {
                            "station_id": sid,
                            "EVSE": evse_display.get(str(sid), str(sid)),
                            "Start (UTC)": last_disconnect,
                            "End (UTC)": t,
                            "Duration (min)": (t - last_disconnect).total_seconds() / 60.0,
                            "Type": "DISCONNECT→CONNECT",
                        }
                    )
                    last_disconnect = None
                last_connect = t

            elif "DISCONNECT" in ev:
                # if we have a preceding CONNECT, duration is connect->disconnect (uptime)
                if last_connect is not None and t >= last_connect:
                    rows.append(
                        {
                            "station_id": sid,
                            "EVSE": evse_display.get(str(sid), str(sid)),
                            "Start (UTC)": last_connect,
                            "End (UTC)": t,
                            "Duration (min)": (t - last_connect).total_seconds() / 60.0,
                            "Type": "CONNECT→DISCONNECT",
                        }
                    )
                    last_connect = None
                last_disconnect = t

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Convert to local
    out["Start (local)"] = _to_local(out["Start (UTC)"], tz_name)
    out["End (local)"] = _to_local(out["End (UTC)"], tz_name)

    out = out.drop(columns=["Start (UTC)", "End (UTC)"])

    # Ordering
    out = _order_cols(
        out,
        ["Start (local)", "End (local)", "Duration (min)", "Type", "EVSE", "station_id"],
    )

    # Sort newest first for export readability
    if "Start (local)" in out.columns:
        out = out.sort_values("Start (local)", ascending=False)

    return out


def prep_connectivity_summary_sheet(conn_data_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize connectivity minutes by EVSE."""

    if conn_data_df is None or conn_data_df.empty:
        return pd.DataFrame()

    src = conn_data_df.copy()

    # We only sum minutes for downtime rows by default (DISCONNECT→CONNECT),
    # but if Type isn't present, just sum all.
    if "Type" in src.columns:
        downtime = src[src["Type"] == "DISCONNECT→CONNECT"].copy()
        if downtime.empty:
            downtime = src
    else:
        downtime = src

    if "Duration (min)" not in downtime.columns:
        return pd.DataFrame()

    group_cols = [c for c in ["EVSE", "station_id"] if c in downtime.columns]
    if not group_cols:
        group_cols = ["EVSE"] if "EVSE" in downtime.columns else []

    out = (
        downtime.groupby(group_cols, dropna=False)["Duration (min)"]
        .sum()
        .reset_index()
        .rename(columns={"Duration (min)": "Total Minutes"})
    )

    out["Total Hours"] = out["Total Minutes"] / 60.0

    # Sort largest downtime first
    out = out.sort_values("Total Minutes", ascending=False)

    return out


# ----------------------------
# Writer
# ----------------------------


def write_xlsx_bytes(
    sheets: Dict[str, pd.DataFrame],
    *,
    tz_name: str,
    start_utc: datetime | None,
    end_utc: datetime | None,
    filename_prefix: str,
) -> bytes:
    """Write sheets to XLSX bytes.

    Uses xlsxwriter if available, otherwise openpyxl.
    """

    output = BytesIO()

    # Prefer xlsxwriter (faster + better column widths)
    engine = "xlsxwriter"
    try:
        __import__("xlsxwriter")
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(output, engine=engine, datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for name, df in sheets.items():
            safe_name = _safe_sheet_name(name)
            df_out = _excel_sanitize_df(df if df is not None else pd.DataFrame())
            df_out.to_excel(writer, sheet_name=safe_name, index=False)

            # Best-effort: autosize columns for xlsxwriter
            if engine == "xlsxwriter":
                try:
                    worksheet = writer.sheets[safe_name]
                    _autosize_xlsxwriter(worksheet, df)
                except Exception:
                    pass

        # Add a tiny "Meta" sheet for traceability
        meta_rows = []
        if start_utc is not None:
            meta_rows.append({"key": "start_utc", "value": str(start_utc)})
        if end_utc is not None:
            meta_rows.append({"key": "end_utc", "value": str(end_utc)})
        meta_rows.append({"key": "tz_name", "value": tz_name})
        meta_rows.append({"key": "generated_at_utc", "value": str(pd.Timestamp.utcnow())})
        meta_df = pd.DataFrame(meta_rows)
        _excel_sanitize_df(meta_df).to_excel(writer, sheet_name="Meta", index=False)

        if engine == "xlsxwriter":
            try:
                worksheet = writer.sheets["Meta"]
                _autosize_xlsxwriter(worksheet, meta_df)
            except Exception:
                pass

    return output.getvalue()


# ----------------------------
# Utilities
# ----------------------------


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _order_cols(df: pd.DataFrame, preferred: Iterable[str]) -> pd.DataFrame:
    cols = list(df.columns)
    out_cols = [c for c in preferred if c in cols]
    out_cols += [c for c in cols if c not in out_cols]
    return df[out_cols]


def _convert_time_cols(
    df: pd.DataFrame,
    *,
    tz_name: str,
    cols: Iterable[str],
    assume_utc_for_naive: bool = True,
) -> pd.DataFrame:
    """Convert listed datetime columns to local time (Excel-safe tz-naive).

    - tz-aware values are converted to `tz_name`.
    - tz-naive values:
        * assume_utc_for_naive=True  -> treat as UTC, convert to local
        * assume_utc_for_naive=False -> assume already local, leave as-is
    """
    out = df
    for c in cols:
        if c not in out.columns:
            continue
        try:
            s = pd.to_datetime(out[c], errors="coerce")

            # tz-aware -> convert to local
            if pd.api.types.is_datetime64tz_dtype(s):
                out[c] = _to_local(s, tz_name)
                continue

            # tz-naive -> either treat as UTC and convert, or leave as-is
            if assume_utc_for_naive:
                try:
                    s_utc = s.dt.tz_localize("UTC")
                    out[c] = _to_local(s_utc, tz_name)
                except Exception:
                    out[c] = s
            else:
                out[c] = s
        except Exception:
            pass
    return out


def _to_local(series: pd.Series, tz_name: str) -> pd.Series:
    """Convert to local time and strip tzinfo (Excel-safe).

    Excel writers (xlsxwriter/openpyxl) cannot write tz-aware datetimes.
    We keep the local *clock time* but return tz-naive values.
    """
    s = pd.to_datetime(series, utc=True, errors="coerce")

    try:
        local = s.dt.tz_convert(tz_name)
    except Exception:
        local = s

    # Strip timezone to make Excel happy (keep displayed clock time)
    try:
        return local.dt.tz_localize(None)
    except Exception:
        return pd.to_datetime(local, errors="coerce")
def _excel_sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe for Excel writers.

    Pandas Excel writers cannot handle tz-aware datetimes. This converts any
    tz-aware datetime columns to tz-naive (keeping the displayed clock time).
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    out = df.copy()

    for c in out.columns:
        try:
            col = out[c]

            # If it's already datetime with tz, strip tz
            if pd.api.types.is_datetime64tz_dtype(col):
                out[c] = col.dt.tz_localize(None)
                continue

            # If it's object that contains tz-aware timestamps, try coercing
            if col.dtype == object:
                parsed = pd.to_datetime(col, errors="ignore")
                if isinstance(parsed, pd.Series) and pd.api.types.is_datetime64tz_dtype(parsed):
                    out[c] = parsed.dt.tz_localize(None)
        except Exception:
            continue

    return out


def _safe_sheet_name(name: str) -> str:
    # Excel sheet names max 31 chars and cannot contain : \/ ? * [ ]
    bad = [":", "\\", "/", "?", "*", "[", "]"]
    out = name
    for b in bad:
        out = out.replace(b, "-")
    out = out.strip()
    return out[:31] if len(out) > 31 else out


def _autosize_xlsxwriter(worksheet: Any, df: pd.DataFrame) -> None:
    """Autosize columns in an xlsxwriter worksheet."""
    if df is None:
        return
    # Cap width so it doesn't get absurd
    max_width = 70

    for i, col in enumerate(df.columns):
        try:
            series = df[col]
            # Convert to strings for width calc, but keep NaNs short
            max_len = max(
                [len(str(col))]
                + [len("" if pd.isna(v) else str(v)) for v in series.head(500)]
            )
            width = min(max_len + 2, max_width)
            worksheet.set_column(i, i, width)
        except Exception:
            continue