import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from rca_v2.config import APP_MODE
from rca_v2.ui import render_sidebar, sessions_table_single_select
from rca_v2.loaders import (
    load_meter_values,
    load_authorize,
    load_status_history,
    load_connectivity,
)
from rca_v2.sessions import build_sessions
from rca_v2.charts import session_detail_figure, heatmap_count, heatmap_duration
from rca_v2.constants import get_evse_display

from rca_v2.admintab import render_admin_tab

st.set_page_config(page_title="ReCharge Alaska — Portal v2", layout="wide")

with st.sidebar:
    stations, start_utc, end_utc = render_sidebar()

EVSE_DISPLAY = get_evse_display()

# Treat "no selection" as ALL EVSEs (matches sidebar hint)
if not stations:
    stations = list(EVSE_DISPLAY.keys())
    st.session_state["__v2_all_evse"] = True
else:
    st.session_state["__v2_all_evse"] = False

# Build tabs; only show Admin when running locally
if APP_MODE == "local":
    TAB_TITLES = ["Charging Sessions", "Status History", "Connectivity", "Data Export", "Admin (local-only)"]
    t1, t2, t3, t4, t5 = st.tabs(TAB_TITLES)
else:
    TAB_TITLES = ["Charging Sessions", "Status History", "Connectivity", "Data Export"]
    t1, t2, t3, t4 = st.tabs(TAB_TITLES)
    t5 = None

with t1:
    st.subheader("Charging Sessions")
    with st.spinner("Loading data…"):
        # Some SQLite builds mis-handle `IN (?)` when given multiple params, which can
        # cause results to collapse to only the last EVSE (e.g., Autel Maxi).
        # To be bulletproof on all environments, load per‑EVSE and concat.
        def _load_per_evse(loader_fn, ids, start_iso, end_iso):
            if not ids:
                # No filter means all EVSEs; let the loader decide by passing None
                try:
                    return loader_fn(None, start_iso, end_iso)
                except TypeError:
                    return loader_fn([], start_iso, end_iso)
            if isinstance(ids, (set, tuple)):
                ids = list(ids)
            frames = []
            for sid in ids:
                try:
                    df = loader_fn([sid], start_iso, end_iso)
                except TypeError:
                    df = loader_fn(sid, start_iso, end_iso)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        mv = _load_per_evse(load_meter_values, stations, start_utc, end_utc)
        auth = _load_per_evse(load_authorize, stations, start_utc, end_utc)
    sess, heat = build_sessions(mv, auth)
    # Defensive: re-apply EVSE filter at the session level (guards against loader quirks)
    if isinstance(stations, (list, tuple, set)) and len(stations) > 0 and "station_id" in sess.columns:
        _wanted = {str(s) for s in stations}
        sess = sess[sess["station_id"].astype(str).isin(_wanted)]
    if sess.empty:
        st.warning("No sessions found."); st.stop()
    # Filter out non-charging/invalid rows (no tx, bad connector, or no power/energy)
    sess = sess.copy()
    if "transaction_id" in sess.columns:
        sess = sess[sess["transaction_id"].notna()]
        sess = sess[sess["transaction_id"].astype(str).str.strip().ne("")]
        sess = sess[~sess["transaction_id"].astype(str).str.lower().isin(["none", "nan"])]
    if "Connector #" in sess.columns:
        sess = sess[sess["Connector #"].isin([1, 2])]
    # keep only sessions with activity (either power or energy)
    pow_ok = pd.to_numeric(sess.get("Max Power (kW)"), errors="coerce").fillna(0) > 0
    eng_ok = pd.to_numeric(sess.get("Energy Delivered (kWh)"), errors="coerce").fillna(0) > 0
    sess = sess[pow_ok | eng_ok]
    if sess.empty:
        st.warning("No valid charging sessions found in this window."); st.stop()
    # Normalize and sort sessions by start time (most recent first)
    if "Start Date/Time" in sess.columns:
        try:
            _sdt = pd.to_datetime(sess["Start Date/Time"], errors="coerce", utc=True)
            sess = sess.assign(_sdt=_sdt).sort_values("_sdt", ascending=False, kind="mergesort").drop(columns=["_sdt"])
        except Exception:
            pass
    # Always derive friendly EVSE name from station_id, overriding any legacy column
    # that might contain site labels like "ARG" or "Glennallen".
    if "station_id" in sess.columns:
        sess["EVSE"] = (
            sess["station_id"].map(EVSE_DISPLAY)
            .fillna(sess["station_id"].astype(str))
        )
    elif "EVSE" not in sess.columns and "Location" in sess.columns:
        # Fallback if station_id is missing in this frame
        sess["EVSE"] = sess["Location"].astype(str)
    cols = ["Start Date/Time","End Date/Time","EVSE","Connector #","Connector Type",
            "Max Power (kW)","Energy Delivered (kWh)","Duration (min)","SoC Start","SoC End","ID Tag"]
    cols = [c for c in cols if c in sess.columns]
    # Render the sessions table and capture the selected (station_id, transaction_id)
    selected_sid, selected_tx = sessions_table_single_select(sess)

    st.markdown("#### Charge Session Details")
    if selected_sid and selected_tx:
        st.plotly_chart(
            session_detail_figure(mv, selected_sid, selected_tx),
            use_container_width=True,
            config={"displaylogo": False},
        )
    else:
        st.info("Select a session above to view details.")

    # Heatmaps — stacked full width (to match v1 layout)
    st.plotly_chart(
        heatmap_count(heat, "Session Start Density (by Day & Hour)"),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.plotly_chart(
        heatmap_duration(heat, "Average Session Duration (min)"),
        use_container_width=True,
        config={"displaylogo": False},
    )

    # Cache latest results for the Data Export tab
    st.session_state["__v2_last_sessions"] = sess.copy()
    st.session_state["__v2_last_meter"] = mv.copy()


with t2:
    st.subheader("Status History")

    # Top row: badge + vendor-only toggle
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("sorted newest first")
    with c2:
        show_only_vendor = st.checkbox("Show only vendor_error_code", value=False)

    status_df = load_status_history(stations, start_utc, end_utc)
    if status_df.empty:
        st.info("No status notifications in this window for the selected EVSE.")
    else:
        df = status_df.copy()

        # Friendly EVSE name
        if "station_id" in df.columns:
            df["EVSE"] = df["station_id"].map(EVSE_DISPLAY).fillna(df["station_id"])
        else:
            df["EVSE"] = df.get("EVSE", "")

        # Build AK-local timestamp for sort/print
        AK = "America/Anchorage"
        if "AKDT_dt" in df.columns:
            ts = pd.to_datetime(df["AKDT_dt"], errors="coerce")
            try:
                ts = ts.dt.tz_convert(AK)
            except Exception:
                pass
        elif "AKDT" in df.columns:
            ts = pd.to_datetime(df["AKDT"], errors="coerce")
            try:
                ts = ts.dt.tz_localize(AK)
            except Exception:
                pass
        else:
            ts = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True).dt.tz_convert(AK)
        df["_ts"] = ts

        # Optional: only rows that actually have a vendor_error_code
        if show_only_vendor and "vendor_error_code" in df.columns:
            v = df["vendor_error_code"].astype(str).str.strip()
            df = df[(v != "") & (~v.str.lower().eq("none")) & (~v.eq("0"))]

        # Tritium enrichment (best‑effort; no‑op if helper not present)
        try:
            from rca_v2.loaders import load_tritium_error_codes  # added in loaders.py
            from rca_v2.constants import PLATFORM_MAP
            codes = load_tritium_error_codes()
            if isinstance(codes, pd.DataFrame) and not codes.empty and "vendor_error_code" in df.columns:
                # normalize codes table: use same keys and strip to just digits
                codes2 = codes.rename(columns={"platform": "Platform", "code": "code_key"}).copy()
                codes2["code_key"] = (
                    codes2["code_key"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
                )

                base = df.copy()
                # map platform by station_id (PLATFORM_MAP is keyed by station_id)
                base["Platform"] = base.get("station_id", "").map(PLATFORM_MAP).fillna("")

                # normalize vendor_error_code -> digits only
                base["code_key"] = (
                    base["vendor_error_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
                )

                # join on (Platform, code_key)
                base = base.merge(
                    codes2[["Platform", "code_key", "impact", "description"]],
                    on=["Platform", "code_key"],
                    how="left",
                )
                df = base
        except Exception:
            # Keep table functional even if enrichment isn't wired yet
            pass

        # Build display & sort newest-first
        display = df.copy()
        display["Date/Time (AK Local)"] = display["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

        wanted = [
            "Date/Time (AK Local)",
            "EVSE",
            "connector_id",
            "status",
            "error_code",
            "vendor_error_code",
            "impact",
            "description",
        ]
        final = [c for c in wanted if c in display.columns]
        display = display.sort_values("_ts", ascending=False, kind="mergesort")
        st.dataframe(display[final], use_container_width=True, hide_index=True)

with t3:
    st.subheader("Connectivity")
    conn_df = load_connectivity(stations, start_utc, end_utc)
    if conn_df.empty:
        st.info("No websocket CONNECT/DISCONNECT events in this window.")
    else:
        df = conn_df.copy()

        # Timestamp (AK local) for sorting and duration math
        if "AKDT_dt" in df.columns:
            ts_ak = pd.to_datetime(df["AKDT_dt"], errors="coerce")
        elif "AKDT" in df.columns:
            ts_ak = pd.to_datetime(df["AKDT"], errors="coerce")
        else:
            # Fallback: if only UTC exists, convert to AK
            ts_utc = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=True)
            try:
                ts_ak = ts_utc.dt.tz_convert("America/Anchorage")
            except Exception:
                ts_ak = ts_utc
        df["_ts"] = ts_ak

        # Friendly EVSE names
        if "station_id" in df.columns:
            df["EVSE"] = df["station_id"].map(EVSE_DISPLAY).fillna(df.get("Location"))
        else:
            df["EVSE"] = df.get("Location")

        # Normalize event label to a single column
        if "Connectivity" in df.columns:
            df["Connectivity"] = df["Connectivity"].astype(str)
        elif "event" in df.columns:
            df["Connectivity"] = df["event"].astype(str)
        else:
            df["Connectivity"] = ""

        # Sort ASC within station for duration calc
        sort_cols = ["station_id", "_ts"] if "station_id" in df.columns else ["EVSE", "_ts"]
        df = df.sort_values(sort_cols, kind="mergesort")

        # Duration (min) between previous DISCONNECT and this CONNECT per station
        evt = df["Connectivity"].str.upper()
        prev_evt = evt.shift(1)
        prev_ts = df["_ts"].shift(1)
        same_station = (
            df["station_id"].eq(df["station_id"].shift(1))
            if "station_id" in df.columns else
            df["EVSE"].eq(df["EVSE"].shift(1))
        )

        mask = same_station & evt.str.contains("CONNECT") & prev_evt.str.contains("DISCONNECT")
        df["Duration (min)"] = np.where(
            mask,
            (df["_ts"] - prev_ts).dt.total_seconds() / 60.0,
            np.nan,
        )

        # Build display (newest first)
        display = pd.DataFrame({
            "Date/Time (AK Local)": df["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "EVSE": df["EVSE"],
            "Connectivity": df["Connectivity"],
            "Duration (min)": pd.to_numeric(df["Duration (min)"], errors="coerce").round(2),
        })
        display = display.assign(__ts=df["_ts"]).sort_values("__ts", ascending=False, kind="mergesort").drop(columns="__ts")

        st.dataframe(display, use_container_width=True, hide_index=True)

with t4:
    st.subheader("Data Export")
    sess_last = st.session_state.get("__v2_last_sessions", pd.DataFrame())
    mv_last = st.session_state.get("__v2_last_meter", pd.DataFrame())

    # Build Status and Connectivity exports on-demand using the current filters
    status_src = load_status_history(stations, start_utc, end_utc)
    status_export = pd.DataFrame()
    if isinstance(status_src, pd.DataFrame) and not status_src.empty:
        s = status_src.copy()

        # Friendly EVSE name
        if "station_id" in s.columns:
            s["EVSE"] = s["station_id"].map(EVSE_DISPLAY).fillna(s["station_id"])
        else:
            s["EVSE"] = s.get("EVSE", "")

        # AK‑local timestamp for sort/print
        AK = "America/Anchorage"
        if "AKDT_dt" in s.columns:
            ts = pd.to_datetime(s["AKDT_dt"], errors="coerce")
            try:
                ts = ts.dt.tz_convert(AK)
            except Exception:
                pass
        elif "AKDT" in s.columns:
            ts = pd.to_datetime(s["AKDT"], errors="coerce")
            try:
                ts = ts.dt.tz_localize(AK)
            except Exception:
                pass
        else:
            ts = pd.to_datetime(s.get("timestamp"), errors="coerce", utc=True).dt.tz_convert(AK)
        s["_ts"] = ts

        # Tritium enrichment (adds impact/description when available)
        # Always create the columns so they appear in the export, even if the lookup is unavailable.
        if "vendor_error_code" not in s.columns:
            s["vendor_error_code"] = ""
        if "impact" not in s.columns:
            s["impact"] = ""
        if "description" not in s.columns:
            s["description"] = ""
        try:
            from rca_v2.loaders import load_tritium_error_codes
            from rca_v2.constants import PLATFORM_MAP
            codes = load_tritium_error_codes()
            if isinstance(codes, pd.DataFrame) and not codes.empty and "vendor_error_code" in s.columns:
                # normalize codes table: same join keys and digits-only code
                codes2 = codes.rename(columns={"platform": "Platform", "code": "code_key"}).copy()
                codes2["code_key"] = (
                    codes2["code_key"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
                )

                base = s.copy()
                base["Platform"] = base.get("station_id", "").map(PLATFORM_MAP).fillna("")
                base["code_key"] = (
                    base["vendor_error_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
                )

                s = base.merge(
                    codes2[["Platform", "code_key", "impact", "description"]],
                    on=["Platform", "code_key"],
                    how="left",
                )

                # Ensure the columns exist even if merge produced all-NaN
                if "impact" not in s.columns:
                    s["impact"] = ""
                if "description" not in s.columns:
                    s["description"] = ""
        except Exception:
            # best‑effort enrichment; continue silently if the lookup isn't available
            pass

        status_export = s.copy()
        status_export["Date/Time (AK Local)"] = status_export["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        _status_cols = [
            "Date/Time (AK Local)",
            "EVSE",
            "connector_id",
            "status",
            "error_code",
            "vendor_error_code",
            "impact",
            "description",
        ]
        status_export = status_export.sort_values("_ts", ascending=False, kind="mergesort")

    conn_src = load_connectivity(stations, start_utc, end_utc)
    conn_export = pd.DataFrame()
    if isinstance(conn_src, pd.DataFrame) and not conn_src.empty:
        c = conn_src.copy()

        # Timestamp (AK local)
        if "AKDT_dt" in c.columns:
            ts_ak = pd.to_datetime(c["AKDT_dt"], errors="coerce")
        elif "AKDT" in c.columns:
            ts_ak = pd.to_datetime(c["AKDT"], errors="coerce")
        else:
            ts_utc = pd.to_datetime(c.get("timestamp"), errors="coerce", utc=True)
            try:
                ts_ak = ts_utc.dt.tz_convert("America/Anchorage")
            except Exception:
                ts_ak = ts_utc
        c["_ts"] = ts_ak

        # Friendly EVSE names
        if "station_id" in c.columns:
            c["EVSE"] = c["station_id"].map(EVSE_DISPLAY).fillna(c.get("Location"))
        else:
            c["EVSE"] = c.get("Location")

        # Normalize event label
        if "Connectivity" in c.columns:
            c["Connectivity"] = c["Connectivity"].astype(str)
        elif "event" in c.columns:
            c["Connectivity"] = c["event"].astype(str)
        else:
            c["Connectivity"] = ""

        # Compute duration (min) between previous DISCONNECT and this CONNECT per station
        sort_cols = ["station_id", "_ts"] if "station_id" in c.columns else ["EVSE", "_ts"]
        c = c.sort_values(sort_cols, kind="mergesort")
        evt = c["Connectivity"].str.upper()
        prev_evt = evt.shift(1)
        prev_ts = c["_ts"].shift(1)
        same_station = (
            c["station_id"].eq(c["station_id"].shift(1))
            if "station_id" in c.columns else
            c["EVSE"].eq(c["EVSE"].shift(1))
        )
        mask = same_station & evt.str.contains("CONNECT") & prev_evt.str.contains("DISCONNECT")
        c["Duration (min)"] = np.where(
            mask,
            (c["_ts"] - prev_ts).dt.total_seconds() / 60.0,
            np.nan,
        )

        conn_export = pd.DataFrame({
            "Date/Time (AK Local)": c["_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "EVSE": c["EVSE"],
            "Connectivity": c["Connectivity"],
            "Duration (min)": pd.to_numeric(c["Duration (min)"], errors="coerce").round(2),
        })
        conn_export = conn_export.assign(__ts=c["_ts"]).sort_values("__ts", ascending=False, kind="mergesort").drop(columns="__ts")

    if sess_last.empty and mv_last.empty:
        st.info("No data available to export from this view. Visit the Charging Sessions tab first.")
    else:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            if not sess_last.empty:
                sess_last.to_excel(xw, sheet_name="Sessions", index=False)
            if not mv_last.empty:
                mvx = mv_last.copy()
                # Excel can't handle tz-aware datetimes. Provide AKDT/UTC strings and drop the tz-aware column.
                if "timestamp" in mvx.columns:
                    ts = pd.to_datetime(mvx["timestamp"], errors="coerce", utc=True)
                    mvx["timestamp_akdt"] = ts.dt.tz_convert("America/Anchorage").dt.strftime("%Y-%m-%d %H:%M:%S")
                    mvx["timestamp_utc"] = ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
                    mvx = mvx.drop(columns=["timestamp"])
                mvx.to_excel(xw, sheet_name="MeterValues (window)", index=False)
            # Extra sheets: Status & Connectivity (matching the on-screen tables)
            if not status_export.empty:
                se_cols = [c for c in _status_cols if c in status_export.columns]
                status_export[se_cols].to_excel(xw, sheet_name="Status", index=False)
            if not conn_export.empty:
                conn_export.to_excel(xw, sheet_name="Connectivity", index=False)
        data = bio.getvalue()
        st.download_button(
            "⬇ Download Excel",
            data=data,
            file_name="recharge_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Exports Sessions, MeterValues (window), plus Status and Connectivity tables for the current filters."
        )

if APP_MODE == "local" and t5 is not None:
    with t5:
        render_admin_tab()