

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

from .constants import get_platform_map

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

    # ----------------------------
    # Backwards-compatible aliases for authorize methods
    # ----------------------------
    # Older call-sites may pass the authorize table under slightly different names.
    # Make this a *passthrough* so updates in Supabase flow into exports without
    # having to patch app_v2.py each time.
    if authorize_methods_df is None:
        for k in [
            "authorize_methods",
            "authorize_df",
            "authorize_methods",
            "auth_methods_df",
            "authorization_df",
        ]:
            if k in kwargs and isinstance(kwargs.get(k), pd.DataFrame):
                authorize_methods_df = kwargs.get(k)
                break

    # Don't forward these legacy keys further.
    for k in [
        "authorize_methods",
        "authorize_df",
        "auth_methods_df",
        "authorization_df",
    ]:
        kwargs.pop(k, None)

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

    Columns produced (in order):
      Start Time (local) | End Time (local) | EVSE | Connector Type |
      Max Power (kW) | Duration (min) | Energy Delivered (kWh) |
      SoC Start (%) | SoC End (%) | ID Tag | Authorize (raw) | Auth Method |
      Estimated Revenue ($) | transaction_id

    Auth enrichment joins authorize_methods on:
      1. (station_id, transaction_id)  — primary, for completed sessions
      2. (ID Tag → id_tag_vid)         — fallback for VID-initiated sessions

    Columns intentionally removed:
      - Auth Method Confidence  (not useful for reporting)
      - connector_id            (redundant given Connector Type)
      - Attempted (<1 kWh)      (removed per user request)
      - Outcome                 (removed per user request)
    """

    out = df.copy()

    # --- Station id → EVSE friendly name ---
    station_col = _first_existing(out, ["station_id", "asset_id", "evse_id", "station"])
    if station_col:
        out["EVSE"] = out[station_col].map(evse_display).fillna(out[station_col])

    # --- Convert UTC timestamps to local ---
    out = _convert_time_cols(out, tz_name=tz_name, cols=["timestamp", "received_at", "ts"], assume_utc_for_naive=True)

    # --- Rename columns to export-friendly labels ---
    rename_map = {}
    for c in ["start_time", "start", "Start Date/Time", "Start Time"]:
        if c in out.columns:
            rename_map[c] = "Start Time (local)"
            break
    for c in ["end_time", "end", "End Date/Time", "End Time"]:
        if c in out.columns:
            rename_map[c] = "End Time (local)"
            break
    if "energy_kwh" in out.columns:
        rename_map["energy_kwh"] = "Energy Delivered (kWh)"
    elif "kwh" in out.columns:
        rename_map["kwh"] = "Energy Delivered (kWh)"
    if "duration_min" in out.columns:
        rename_map["duration_min"] = "Duration (min)"
    if "soc_start" in out.columns:
        rename_map["soc_start"] = "SoC Start (%)"
    if "soc_end" in out.columns:
        rename_map["soc_end"] = "SoC End (%)"
    if "id_tag" in out.columns:
        rename_map["id_tag"] = "ID Tag"
    out = out.rename(columns=rename_map)

    # --- SoC normalisation (percent → fraction for Excel percentage formatting) ---
    def _soc_to_fraction(val: Any) -> Any:
        try:
            x = float(val)
        except Exception:
            return val
        if pd.isna(x):
            return x
        if 0.0 <= x <= 1.0:
            return x
        if 1.0 < x <= 100.0:
            return x / 100.0
        return x

    for soc_col in ["SoC Start (%)", "SoC End (%)", "SoC Start", "SoC End"]:
        if soc_col in out.columns:
            out[soc_col] = out[soc_col].apply(_soc_to_fraction)

    # --- Authorization enrichment ---
    # Populate two columns from the authorize_methods table:
    #   Authorize (raw)  — id_tag presented at the charger (app token / CC token / VID)
    #   Auth Method      — "App", "CC", or "AutoCharge" pulled directly from the DB;
    #                      heuristic fallback only for unmatched rows
    #
    # Root cause of previous blank column: sessions have VID tokens in ID Tag;
    # old code replaced them with id_tag_effective from the join, but many
    # authorize_methods rows have null transaction_id so the join produced no match
    # → the VID got replaced with NaN → blank export column.
    #
    # Fix: pull id_tag (not id_tag_effective) and use a clean two-step join.

    def _clean_str(series: pd.Series) -> pd.Series:
        """Strip and blank-out null-like strings for reliable joining."""
        return series.astype(str).str.strip().replace({"nan": "", "None": "", "none": "", "NaN": ""})

    if authorize_methods_df is not None and not authorize_methods_df.empty:
        try:
            am = authorize_methods_df.copy()
            if "station_id" not in am.columns and "asset_id" in am.columns:
                am = am.rename(columns={"asset_id": "station_id"})

            # Normalise join-key columns
            for col in ["station_id", "transaction_id", "id_tag_vid"]:
                if col in am.columns:
                    am[col] = _clean_str(am[col])

            # Determine session station_id column
            sess_sid = station_col if station_col and station_col in out.columns else \
                _first_existing(out, ["station_id", "asset_id"])

            work = out.copy()
            if sess_sid and sess_sid in work.columns:
                work[sess_sid] = _clean_str(work[sess_sid])
            if "transaction_id" in work.columns:
                work["transaction_id"] = _clean_str(work["transaction_id"])
            if "ID Tag" in work.columns:
                work["ID Tag"] = _clean_str(work["ID Tag"])

            work["_auth_matched"] = False

            # --- Join 1: (station_id, transaction_id) ---
            # Reliable for all sessions that completed a transaction.
            if sess_sid and "transaction_id" in am.columns and "transaction_id" in work.columns:
                am_txn = (
                    am[am["transaction_id"].str.len() > 0]
                    [[c for c in ["station_id", "transaction_id", "id_tag", "authorization_method"] if c in am.columns]]
                    .drop_duplicates(subset=["station_id", "transaction_id"], keep="last")
                )
                if not am_txn.empty:
                    merged = work.merge(
                        am_txn.rename(columns={"id_tag": "_am_tag", "authorization_method": "_am_method"}),
                        how="left",
                        left_on=[sess_sid, "transaction_id"],
                        right_on=["station_id", "transaction_id"],
                        suffixes=("", "_amj"),
                    )
                    # Drop the extra station_id column produced by the merge
                    merged = merged.drop(columns=[c for c in ["station_id_amj", "station_id"] if c in merged.columns and c != sess_sid], errors="ignore")
                    matched = (
                        merged["_am_tag"].notna()
                        & (merged["_am_tag"].astype(str).str.lower() != "nan")
                        & (merged["_am_tag"].astype(str) != "")
                    )
                    work.loc[matched.values, "Authorize (raw)"] = merged.loc[matched, "_am_tag"].values
                    work.loc[matched.values, "Auth Method"] = merged.loc[matched, "_am_method"].values
                    work.loc[matched.values, "_auth_matched"] = True

            # --- Join 2: ID Tag → id_tag_vid ---
            # Catches VID-initiated sessions where transaction_id was null in authorize_methods.
            if "id_tag_vid" in am.columns and "ID Tag" in work.columns:
                am_vid = (
                    am[am["id_tag_vid"].str.len() > 0]
                    [[c for c in ["id_tag_vid", "id_tag", "authorization_method"] if c in am.columns]]
                    .drop_duplicates(subset=["id_tag_vid"], keep="last")
                    .rename(columns={"id_tag": "_am_tag_vid", "authorization_method": "_am_method_vid"})
                )
                if not am_vid.empty:
                    unmatched = ~work["_auth_matched"]
                    if unmatched.any():
                        merged_vid = work[unmatched].merge(
                            am_vid, how="left", left_on="ID Tag", right_on="id_tag_vid"
                        )
                        vid_matched = (
                            merged_vid["_am_tag_vid"].notna()
                            & (merged_vid["_am_tag_vid"].astype(str).str.lower() != "nan")
                            & (merged_vid["_am_tag_vid"].astype(str) != "")
                        )
                        um_idx = work[unmatched].index
                        work.loc[um_idx[vid_matched.values], "Authorize (raw)"] = merged_vid.loc[vid_matched, "_am_tag_vid"].values
                        work.loc[um_idx[vid_matched.values], "Auth Method"] = merged_vid.loc[vid_matched, "_am_method_vid"].values
                        work.loc[um_idx[vid_matched.values], "_auth_matched"] = True

            work = work.drop(columns=["_auth_matched"], errors="ignore")
            out = work

        except Exception:
            pass

    # --- Fallback heuristic for unmatched rows ---
    # Only fills Auth Method when the DB join left it blank.
    # Patterns confirmed against LynkWell authorize_methods data:
    #   VID:*               → AutoCharge  (vehicle-initiated)
    #   14-char hex         → CC          (payment terminal / credit card)
    #   20-char A-Z0-9      → App         (mobile app token)
    def _guess_auth_method(x: Any) -> str:
        try:
            s = str(x).strip()
        except Exception:
            return "Unknown"
        if not s or s.lower() in ("nan", "none", ""):
            return "Unknown"
        if s.startswith("VID:"):
            return "AutoCharge"
        if re.fullmatch(r"[0-9A-Fa-f]{14}", s):
            return "CC"
        if re.fullmatch(r"[A-Z0-9]{20}", s):
            return "App"
        if re.fullmatch(r"[0-9A-Fa-f]{8,32}", s):
            return "RFID"
        return "Unknown"

    if "ID Tag" in out.columns:
        need_guess = (
            "Auth Method" not in out.columns
            or out["Auth Method"].isna()
            | out["Auth Method"].astype(str).str.lower().isin(["nan", "none", ""])
        )
        if need_guess.any() if hasattr(need_guess, "any") else need_guess:
            guessed = out["ID Tag"].apply(_guess_auth_method)
            if "Auth Method" not in out.columns:
                out["Auth Method"] = guessed
            else:
                mask = out["Auth Method"].isna() | out["Auth Method"].astype(str).str.lower().isin(["nan", "none", ""])
                out.loc[mask, "Auth Method"] = guessed[mask]

    # --- Drop unwanted columns ---
    drop_cols = []
    # Connector number variants (always remove)
    drop_cols += [c for c in ["connector_num", "connector_number", "connector", "Connector #"] if c in out.columns]
    # connector_id — removed per user request (Connector Type is sufficient)
    drop_cols += [c for c in ["connector_id", "Connector ID"] if c in out.columns]
    # Columns removed per user request
    drop_cols += [c for c in ["Outcome", "outcome", "Attempted (<1 kWh)", "Auth Method Confidence",
                               "Auth Method (guess)"] if c in out.columns]
    # Raw station id when friendly EVSE name is present
    if "EVSE" in out.columns and station_col in {"station_id", "asset_id"} and station_col in out.columns:
        drop_cols.append(station_col)
    if "asset_id" in out.columns:
        drop_cols.append("asset_id")
    # Leftover join columns
    drop_cols += [c for c in ["authorization_method", "id_tag_vid", "id_tag_effective",
                               "id_tag_effective_txn", "id_tag_effective_vid",
                               "confidence"] if c in out.columns]

    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    # --- Final column ordering ---
    preferred = [
        "Start Time (local)",
        "End Time (local)",
        "EVSE",
        "Connector Type",
        "Max Power (kW)",
        "Duration (min)",
        "Energy Delivered (kWh)",
        "SoC Start (%)",
        "SoC End (%)",
        "SoC Start",
        "SoC End",
        "ID Tag",
        "Authorize (raw)",
        "Auth Method",
        "Estimated Revenue ($)",
        "transaction_id",
        # Alternate column name variants (pushed to end if present)
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
    ]

    out = _order_cols(out, preferred)
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

    # Autel often encodes human-readable fault text in error_code.
    # If description is missing/empty, fall back to error_code.
    if "error_code" in out.columns:
        if "description" not in out.columns:
            out["description"] = out["error_code"]
        else:
            desc = out["description"]
            # Treat None/empty/"None" as missing
            missing = desc.isna() | desc.astype(str).str.strip().isin(["", "None", "nan"])
            out.loc[missing, "description"] = out.loc[missing, "error_code"]

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

    # OCPP-based connectivity (BootNotification vs Heartbeat/MeterValues)
    if src[ev_col].str.contains("BOOTNOTIFICATION|HEARTBEAT|METERVALUES").any():
        platform_map = get_platform_map()
        rows = []
        for sid, g in src.groupby(station_col, sort=False):
            g = g.sort_values(ts_col)
            platform = platform_map.get(str(sid), "")
            heartbeat_types = {"METERVALUES"} if platform == "RTM" else {"HEARTBEAT"}

            last_hb_ts = None
            last_hb_type = ""
            for _, r in g.iterrows():
                ev = r[ev_col]
                t = r[ts_col]
                if pd.isna(t):
                    continue
                if ev in heartbeat_types:
                    last_hb_ts = t
                    last_hb_type = ev
                    continue
                if ev == "BOOTNOTIFICATION":
                    duration = (t - last_hb_ts).total_seconds() / 60.0 if last_hb_ts is not None else None
                    rows.append(
                        {
                            "station_id": sid,
                            "EVSE": evse_display.get(str(sid), str(sid)),
                            "Start (UTC)": last_hb_ts,
                            "End (UTC)": t,
                            "Duration (min)": duration,
                            "Type": "BOOTNOTIFICATION",
                            "Heartbeat Source": last_hb_type.title() if last_hb_type else ("MeterValues" if platform == "RTM" else "Heartbeat"),
                        }
                    )

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        out["Start (local)"] = _to_local(out["Start (UTC)"], tz_name)
        out["End (local)"] = _to_local(out["End (UTC)"], tz_name)
        out = out.drop(columns=["Start (UTC)", "End (UTC)"])
        out = _order_cols(
            out,
            ["Start (local)", "End (local)", "Duration (min)", "Type", "Heartbeat Source", "EVSE", "station_id"],
        )
        if "End (local)" in out.columns:
            out = out.sort_values("End (local)", ascending=False)
        return out

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
