import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import inspect
import plotly.graph_objects as go

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
from rca_v2.db import get_conn
from rca_v2.auth import require_auth

from rca_v2.admintab import render_admin_tab


st.set_page_config(page_title="ReCharge Alaska — Portal v2", layout="wide")

# Gate the web build behind a simple login (disabled for local runs)
if APP_MODE != "local":
    require_auth()

# Admin wildcard propagated by auth.require_auth: {"*"} means unrestricted
_admin_all = False
try:
    _allowed = st.session_state.get("_allowed_evse")
    if isinstance(_allowed, set) and ("*" in _allowed):
        _admin_all = True
except Exception:
    _admin_all = False
st.session_state["_admin_all"] = _admin_all

# Helper: future‑proof sizing for st.dataframe across Streamlit versions
# Uses width="stretch" on newer Streamlit; falls back to use_container_width=True on older.

def _df_stretch_kwargs():
    try:
        sig = inspect.signature(st.dataframe)
        if "width" in sig.parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}

with st.sidebar:
    stations, start_utc, end_utc = render_sidebar()

    # Optional diagnostics to verify DB connectivity & table counts on Render
    with st.expander("Diagnostics", expanded=False):
        try:
            raw_conn = get_conn()

            # Open an executable connection across both SQLAlchemy Engines and sqlite3 connections
            exec_conn = raw_conn
            if hasattr(raw_conn, "connect") and not hasattr(raw_conn, "cursor"):
                # SQLAlchemy Engine -> Connection
                exec_conn = raw_conn.connect()

            def _run(sql: str, params=None):
                """Best-effort SQL runner that works for SQLAlchemy (exec_driver_sql/execute) and sqlite3."""
                params = params or {}
                # SQLAlchemy Connection (preferred)
                if hasattr(exec_conn, "exec_driver_sql"):
                    res = exec_conn.exec_driver_sql(sql, params)
                    try:
                        return res.fetchall()
                    except Exception:
                        return []
                if hasattr(exec_conn, "execute"):
                    res = exec_conn.execute(sql, params)
                    try:
                        return res.fetchall()
                    except Exception:
                        return []
                # Raw DB-API (e.g., sqlite3)
                cur = exec_conn.cursor()
                if isinstance(params, (list, tuple)):
                    cur.execute(sql, params)
                elif isinstance(params, dict):
                    # Fall back to tuple of dict values for sqlite3
                    cur.execute(sql, tuple(params.values()))
                else:
                    cur.execute(sql)
                rows = cur.fetchall()
                return rows

            # Detect backend
            backend = "unknown"
            try:
                v = _run("select version()")
                if v and "PostgreSQL" in str(v[0][0]):
                    backend = "postgres"
                elif v and "SQLite" in str(v[0][0]):
                    backend = "sqlite"
            except Exception:
                try:
                    _ = _run("pragma user_version")
                    backend = "sqlite"
                except Exception:
                    pass

            st.success(f"DB connected ({backend})")

            expected_tables = [
                "sessions",
                "session_logs",
                "meter_values",
                "realtime_meter_values",
                "status_notifications",
                "realtime_status_notifications",
                "realtime_websocket",
                "connectivity_logs",
            ]

            # Enumerate tables present
            if backend == "postgres":
                tbl_rows = _run("select table_name from information_schema.tables where table_schema='public'")
                present_tables = {r[0] for r in tbl_rows}
            else:
                tbl_rows = _run("select name from sqlite_master where type='table'")
                present_tables = {r[0] for r in tbl_rows}

            st.write("**Tables present**")
            st.json(sorted(present_tables))

            missing_tables = [t for t in expected_tables if t not in present_tables]
            if missing_tables:
                st.warning(f"Missing tables: {', '.join(missing_tables)}")

            # Helper to list columns for a table
            def _list_columns(table_name: str):
                if backend == "postgres":
                    rows = _run(
                        "select column_name, data_type from information_schema.columns where table_schema='public' and table_name=%(t)s",
                        {"t": table_name},
                    )
                    return [r[0] for r in rows]
                else:
                    # sqlite PRAGMA table_info returns: cid, name, type, ...
                    rows = _run(f"pragma table_info({table_name})")
                    return [r[1] for r in rows]

            # Column inventory for meter tables (check for 'hbv_v')
            expected_meter_cols = [
                "station_id","connector_id","transaction_id","timestamp",
                "power_w","energy_wh","soc","amperage_offered","amperage_import",
                "power_offered_w","voltage_v","hbv_v"
            ]
            col_report = {}
            for tbl in ["realtime_meter_values", "meter_values"]:
                if tbl in present_tables:
                    cols = _list_columns(tbl)
                    miss = [c for c in expected_meter_cols if c not in cols]
                    col_report[tbl] = {"columns": cols, "missing_expected": miss}
            if col_report:
                st.write("**Column inventory (meter tables)**")
                st.json(col_report)
                if any("hbv_v" in v.get("missing_expected", []) for v in col_report.values()):
                    st.info(
                        "Hint: your Postgres tables are missing 'hbv_v'. To match the local SQLite schema, run:\n"
                        "```sql\n"
                        "ALTER TABLE public.realtime_meter_values ADD COLUMN hbv_v double precision;\n"
                        "ALTER TABLE public.meter_values ADD COLUMN hbv_v double precision;\n"
                        "```"
                    )

            # Window sanity counts (uses current sidebar date range)
            def _window_counts(table_name: str):
                if backend == "postgres":
                    sql = f"""
                    select
                      min((regexp_replace("timestamp",'Z$','+00:00'))::timestamptz) as min_ts,
                      max((regexp_replace("timestamp",'Z$','+00:00'))::timestamptz) as max_ts,
                      count(*) as n
                    from public.{table_name}
                    where "timestamp" ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}T'
                      and (regexp_replace("timestamp",'Z$','+00:00'))::timestamptz
                          between %(s)s and %(e)s
                    """
                    rows = _run(sql, {"s": start_utc, "e": end_utc})
                else:
                    sql = f"""
                    select min(timestamp), max(timestamp), count(*)
                    from {table_name}
                    where timestamp between ? and ?
                    """
                    rows = _run(sql, [start_utc, end_utc])
                return rows[0] if rows else None

            window_report = {}
            for tname in ["realtime_meter_values","meter_values","sessions","status_notifications","realtime_websocket"]:
                if tname in present_tables:
                    r = _window_counts(tname)
                    if r:
                        window_report[tname] = {"min": str(r[0]), "max": str(r[1]), "rows": int(r[2]) if r[2] is not None else None}
            if window_report:
                st.write("**Rows in current window**")
                st.json(window_report)

        except Exception as e:
            st.error(f"DB error: {e}")
        finally:
            # Close connections if applicable
            try:
                if 'exec_conn' in locals() and hasattr(exec_conn, "close"):
                    exec_conn.close()
            except Exception:
                pass
            try:
                if 'raw_conn' in locals() and hasattr(raw_conn, "close"):
                    raw_conn.close()
            except Exception:
                pass

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
    if (not st.session_state.get("_admin_all")) and (not st.session_state.get("__v2_all_evse", False)) and isinstance(stations, (list, tuple, set)) and len(stations) > 0 and "station_id" in sess.columns:
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
    # Rebuild heatmaps from sessions to ensure AK-local bucketing and robust de-duplication.
    try:
        s2 = sess.copy()

        # --- Build AK‑local, hour‑floored start times (robust across sources) ---
        AK = "America/Anchorage"
        starts_ak = None
        if "_start" in s2.columns:
            # Preferred: UTC ISO from build_sessions
            su = pd.to_datetime(s2["_start"], errors="coerce", utc=True)
            starts_ak = su.dt.tz_convert(AK).dt.floor("H")
        elif "AKDT_dt" in s2.columns:
            su = pd.to_datetime(s2["AKDT_dt"], errors="coerce")
            # Localize if naive; convert if tz‑aware
            if getattr(su.dt, "tz", None) is None:
                starts_ak = su.dt.tz_localize(AK).dt.floor("H")
            else:
                starts_ak = su.dt.tz_convert(AK).dt.floor("H")
        else:
            sd = pd.to_datetime(s2.get("Start Date/Time"), errors="coerce")
            if getattr(sd.dt, "tz", None) is None:
                starts_ak = sd.dt.tz_localize(AK).dt.floor("H")
            else:
                starts_ak = sd.dt.tz_convert(AK).dt.floor("H")

        # Keep only rows with a valid start time
        valid = starts_ak.notna()
        s2 = s2.loc[valid].copy()
        starts_ak = starts_ak.loc[valid]

        # Working columns for grouping
        s2["_start_ak"] = starts_ak
        s2["_dow"] = s2["_start_ak"].dt.dayofweek  # Mon=0 .. Sun=6
        s2["_hour"] = s2["_start_ak"].dt.hour

        # --- De‑duplication: one row per session ---
        # Prefer transaction_id; otherwise fall back to (station_id, _start_ak) or (EVSE, _start_ak).
        if "transaction_id" in s2.columns:
            s2 = s2.sort_values("_start_ak").drop_duplicates(subset=["transaction_id"], keep="first")
        else:
            dedupe_keys = []
            if "station_id" in s2.columns:
                dedupe_keys = ["station_id", "_start_ak"]
            elif "EVSE" in s2.columns:
                dedupe_keys = ["EVSE", "_start_ak"]
            if dedupe_keys:
                s2 = s2.sort_values("_start_ak").drop_duplicates(subset=dedupe_keys, keep="first")

        # ---------- Count heatmap: session starts by day/hour ----------
        # Use explicit groupby(size) instead of crosstab to avoid dtype/engine quirks seen on Render.
        # Ensure day-of-week and hour are clean integers first.
        _d = pd.to_numeric(s2["_dow"], errors="coerce").astype("Int64")
        _h = pd.to_numeric(s2["_hour"], errors="coerce").astype("Int64")
        tmp = (
            pd.DataFrame({"dow": _d, "hour": _h})
            .dropna()
            .astype({"dow": int, "hour": int})
        )

        counts = (
            tmp.groupby(["dow", "hour"])
               .size()
               .unstack(fill_value=0)
        )

        # Stable grid Sun..Sat and hours 0..23
        sun_first = [6, 0, 1, 2, 3, 4, 5]
        counts = (
            counts
            .reindex(index=sun_first, fill_value=0)
            .reindex(columns=range(24), fill_value=0)
        )
        counts.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

        # Build text labels that hide zeros (match local style)
        count_vals = counts.values.astype(float)
        count_text = np.where(count_vals > 0, counts.values.astype(int).astype(str), "")

        # Upper bound for color scale (avoid a flat palette when all zeros)
        zmax_count = int(max(1, int(np.nanmax(counts.values)))) if counts.size else 1

        fig_count = go.Figure(
            data=go.Heatmap(
                z=counts.values,
                x=[f"{h:02d}" for h in counts.columns],
                y=list(counts.index),
                colorscale="Blues",
                zmin=0,
                zmax=zmax_count,
                colorbar=dict(
                    title="Starts",
                    tickcolor="black",
                    tickfont=dict(color="black"),
                    titlefont=dict(color="black"),
                ),
                text=count_text,
                texttemplate="%{text}",
                textfont=dict(color="black"),
                hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Starts: %{z:.0f}<extra></extra>",
                xgap=1,
                ygap=1,
            )
        )
        fig_count.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            xaxis_title="Hour (0–23)",
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )
        fig_count.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False, linecolor="black")
        fig_count.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False, linecolor="black")

        # ---------- Duration heatmap: average minutes by day/hour ----------
        dur = pd.to_numeric(s2.get("Duration (min)"), errors="coerce")
        s2["_dur"] = dur

        dur_pivot = (
            s2.groupby(["_dow", "_hour"])["_dur"]
              .mean()
              .unstack()
        )
        dur_pivot = dur_pivot.reindex(index=sun_first, fill_value=np.nan).reindex(columns=range(24), fill_value=np.nan)
        dur_pivot.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

        # Round for on‑cell annotation
        _vals = dur_pivot.values.astype(float)
        # Hide NaNs and zeros in cell labels
        dur_text = np.where(np.isnan(_vals) | (_vals <= 0), "", np.round(_vals, 1))

        fig_dur = go.Figure(
            data=go.Heatmap(
                z=dur_pivot.values,
                x=[f"{h:02d}" for h in dur_pivot.columns],
                y=list(dur_pivot.index),
                colorscale="Blues",
                zmin=0,
                colorbar=dict(
                    title="Avg min",
                    tickcolor="black",
                    tickfont=dict(color="black"),
                    titlefont=dict(color="black"),
                ),
                text=dur_text,
                texttemplate="%{text}",
                textfont=dict(color="black"),
                hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Avg min: %{z:.1f}<extra></extra>",
                xgap=1,
                ygap=1,
            )
        )
        fig_dur.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            xaxis_title="Hour (0–23)",
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )
        fig_dur.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=False, linecolor="black")
        fig_dur.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False, linecolor="black")

        # Render both heatmaps
        st.plotly_chart(fig_count, use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(fig_dur, use_container_width=True, config={"displaylogo": False})

    except Exception:
        # Fallback to previous implementation if anything goes sideways
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
        st.dataframe(display[final], hide_index=True, **_df_stretch_kwargs())

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

        st.dataframe(display, hide_index=True, **_df_stretch_kwargs())

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
                sx = sess_last.copy()

                # Normalize connector column for export — keep only a single "Connector id"
                # Accept several possible source column spellings and drop the rest.
                rename_map = {}
                if "connector_id" in sx.columns and "Connector id" not in sx.columns:
                    rename_map["connector_id"] = "Connector id"
                if "Connector Id" in sx.columns:
                    rename_map["Connector Id"] = "Connector id"
                if rename_map:
                    sx = sx.rename(columns=rename_map)
                # If we only have "Connector #", promote it to "Connector id"
                if "Connector id" not in sx.columns and "Connector #" in sx.columns:
                    sx["Connector id"] = sx["Connector #"]

                # Drop any legacy connector columns so only "Connector id" remains
                drop_candidates = ["Connector #", "Connect #", "connector_id", "Connector Id"]
                sx = sx.drop(columns=[c for c in drop_candidates if c in sx.columns], errors="ignore")

                sx.to_excel(xw, sheet_name="Sessions", index=False)
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