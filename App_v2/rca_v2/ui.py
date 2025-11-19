import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date, time
from typing import Optional
from .config import AK_TZ, UTC
from .constants import EVSE_DISPLAY, display_name, get_all_station_ids

def _round_up_to_hour(dt: datetime) -> datetime:
    base = dt.replace(minute=0, second=0, microsecond=0)
    if dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return base
    return base + timedelta(hours=1)

def _find_logo_path() -> Optional[str]:
    here = os.path.dirname(__file__)
    app_root = os.path.abspath(os.path.join(here, ".."))
    candidates = [
        os.path.join(app_root, "ReCharge Logo_REVA.png"),
        os.path.join(here,     "ReCharge Logo_REVA.png"),
        os.path.join(app_root, "logo.png"),
        os.path.join(here,     "logo.png"),
        os.path.join(app_root, "logo.jpg"),
        os.path.join(here,     "logo.jpg"),
        os.path.join(app_root, "logo.jpeg"),
        os.path.join(here,     "logo.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def render_sidebar():
    # Logo
    logo_path = _find_logo_path()
    if logo_path:
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("### ReCharge Alaska")

    # ---------- EVSE Filter (friendly names only) ----------
    st.markdown("#### EVSE Filter")
    # Use union of all known station_ids so overrides / archived items still appear
    all_keys = get_all_station_ids()
    pairs = sorted([(display_name(k), k) for k in all_keys], key=lambda x: x[0])
    labels = [p[0] for p in pairs]
    friendly_to_key = dict(pairs)
    # Clean visual by default: no chips shown; empty selection means "all EVSEs"
    sel_labels = st.multiselect(" ", options=labels, default=[], label_visibility="collapsed")
    stations = [friendly_to_key[x] for x in sel_labels] if sel_labels else all_keys
    st.caption("No selection = all EVSEs")

    # ---------- Default rolling 7‑day window (AK local) ----------
    now_ak = datetime.now(AK_TZ)
    end_default = _round_up_to_hour(now_ak)
    start_default = end_default - timedelta(days=7)

    # Use v2‑scoped keys so we don't collide with other apps
    ks = {
        "sd": "v2_start_date",
        "st": "v2_start_time",
        "ed": "v2_end_date",
        "et": "v2_end_time",
    }

    # ---------- Compact Start/End inputs ----------
    # Row 1: Start Date / Start Time
    st.caption("Start (AK Local)")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        start_date: date = st.date_input(
            " ", value=start_default.date(), key=ks["sd"], label_visibility="collapsed"
        )
    with c2:
        start_time: time = st.time_input(
            " ", value=start_default.time().replace(second=0, microsecond=0),
            step=60*60, key=ks["st"], label_visibility="collapsed"
        )

    # Row 2: End Date / End Time
    st.caption("End (AK Local)")
    c3, c4 = st.columns(2, gap="small")
    with c3:
        end_date: date = st.date_input(
            " ", value=end_default.date(), min_value=start_date, key=ks["ed"], label_visibility="collapsed"
        )
    with c4:
        end_time: time = st.time_input(
            " ", value=end_default.time().replace(second=0, microsecond=0),
            step=60*60, key=ks["et"], label_visibility="collapsed"
        )

    # Coerce end >= start
    start_dt = datetime.combine(start_date, start_time).replace(tzinfo=AK_TZ)
    end_dt   = datetime.combine(end_date, end_time).replace(tzinfo=AK_TZ)
    if end_dt < start_dt:
        end_dt = start_dt

    # Convert to UTC strings for queries
    start_utc = start_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc   = end_dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    st.divider()
    st.caption(f"Query window (UTC): {start_utc} ➜ {end_utc}")
    return stations, start_utc, end_utc


# --- Charging Sessions table (single-select) ---

def sessions_table_single_select(session_summary: pd.DataFrame):
    """
    Render the sessions table with a single-select checkbox column and return
    (station_id, transaction_id) for the selected row. If nothing is selected,
    the newest (top) session is used.

    Returns:
        Tuple[str | None, str | None]
    """
    if session_summary is None or session_summary.empty:
        st.info("No sessions found.")
        return (None, None)

    df = session_summary.copy()

    # Build human-visible columns (assumes these are already present upstream)
    show_cols = [
        "Start Date/Time",
        "End Date/Time",
        "EVSE",
        "connector_id",
        "Connector Type",
        "Max Power (kW)",
        "Energy Delivered (kWh)",
        "Duration (min)",
        "SoC Start",
        "SoC End",
        "ID Tag",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    # Stable key to track selection across reruns
    df["__session_key__"] = df["station_id"].astype(str) + "|" + df["transaction_id"].astype(str)

    # Use a session-scoped key so we can control which row is selected on each rerun
    prev_selected_key = st.session_state.get("v2_selected_session_key", "")
    df["__sel__"] = df["__session_key__"].eq(prev_selected_key)

    # Build a widget key that depends on the previously selected session.
    # This forces Streamlit to instantiate a fresh data_editor when the selection changes,
    # so only our computed "__sel__" row remains checked after rerun.
    safe_suffix = (prev_selected_key or "none").replace("|", "_").replace(" ", "_")
    editor_key = f"v2_sessions_editor_{safe_suffix}"

    edited = st.data_editor(
        df[["__sel__"] + show_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "__sel__": st.column_config.CheckboxColumn(
                " ", help="Select a session to show details below", default=False
            ),
            "connector_id": st.column_config.NumberColumn("Connector #", help="Connector on EVSE", format="%d"),
            "Max Power (kW)": st.column_config.NumberColumn("Max Power (kW)", format="%.2f"),
            "Energy Delivered (kWh)": st.column_config.NumberColumn("Energy Delivered (kWh)", format="%.2f"),
            "Duration (min)": st.column_config.NumberColumn("Duration (min)", format="%.2f"),
            "SoC Start": st.column_config.NumberColumn("SoC Start", format="%d"),
            "SoC End": st.column_config.NumberColumn("SoC End", format="%d"),
        },
        disabled=[c for c in show_cols],
        height=480,
        key=editor_key,
    )

    # Enforce single-select: take the last True row if multiple are checked
    sel_idx = edited.index[edited["__sel__"] == True].tolist()
    if sel_idx:
        chosen_idx = sel_idx[-1]
    else:
        chosen_idx = edited.index[0]

    chosen_key = df.loc[chosen_idx, "__session_key__"]
    st.session_state["v2_selected_session_key"] = chosen_key

    sid, tx = chosen_key.split("|", 1)
    return sid, tx