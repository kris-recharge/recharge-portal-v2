import os
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta, date, time
from typing import Optional
from .config import AK_TZ, UTC
from .constants import EVSE_DISPLAY, display_name, get_all_station_ids
from .auth import get_portal_user, filter_allowed_evse_ids, user_label

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

def render_sidebar(*, allowed_evse_ids: Optional[list[str]] = None, user_email_override: Optional[str] = None, logout_url_override: Optional[str] = None):
    # Logo
    logo_path = _find_logo_path()
    if logo_path:
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("### ReCharge Alaska")

    # Portal-provided user context (preferred): we read request headers forwarded by Caddy.
    portal = get_portal_user()

    # `portal` may be either a dict (older/local paths) or a PortalUser object (new auth.py).
    def _pget(key: str, default=None):
        if portal is None:
            return default
        if isinstance(portal, dict):
            return portal.get(key, default)
        return getattr(portal, key, default)

    # Allow app-level overrides (used for local/dev or explicit embedding overrides)
    email = (user_email_override or (_pget("email", "") or "")).strip()
    logout_url = (logout_url_override or (_pget("logout_url", "") or "")).strip()

    # Allowed EVSEs: prefer explicit arg, else portal header, else (if we know the user) deny-by-default
    allowed_from_portal = _pget("allowed_evse_ids", []) or []

    # If allowed_from_portal is a JSON/text blob, normalize to list[str]
    if isinstance(allowed_from_portal, str):
        s = allowed_from_portal.strip()
        parsed: list[str] = []
        # JSON list (preferred)
        if s.startswith("["):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    parsed = [str(x).strip() for x in v if str(x).strip()]
            except Exception:
                parsed = []
        # Postgres text array like {"a","b"} or {a,b}
        elif s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            if inner:
                parts = inner.split(",")
                for p in parts:
                    p = p.strip().strip('"').strip("'")
                    if p:
                        parsed.append(p)
        # Fallback: comma-separated
        elif "," in s:
            parsed = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]

        allowed_from_portal = parsed
    if allowed_evse_ids is None:
        if allowed_from_portal:
            allowed_evse_ids = list(allowed_from_portal)
        elif email:
            # SECURITY: user identified but allow-list missing -> show nothing
            allowed_evse_ids = []

    # --- Account / Logout ---
    # Prefer an explicit logout_url from the portal, but fall back to our known route.
    if not logout_url:
        logout_url = "/api/auth/logout"

    # Keep the sidebar clean: always show who is signed in (when we know),
    # and show logout next to it.
    if email:
        left, right = st.columns([3, 2])
        with left:
            st.caption(user_label(email))
        with right:
            st.link_button("Log out", logout_url, use_container_width=True)
    else:
        # If we don't know the email, still show a logout button (portal session is cookie-based)
        st.link_button("Log out", logout_url, use_container_width=True)

    st.divider()

    # ---------- EVSE Filter (friendly names only, honor allowed_evse_ids if provided) ----------
    st.markdown("#### EVSE Filter")

    # All known EVSE station_ids from the DB
    all_keys = get_all_station_ids()

    # Filter EVSEs by allow-list when provided.
    visible_keys = filter_allowed_evse_ids(all_keys, allowed_evse_ids)

    # ---------- Optional auth debug (set RCA_AUTH_DEBUG=1 in env) ----------
    auth_debug = os.getenv("RCA_AUTH_DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if auth_debug:
        try:
            raw_allowed = allowed_from_portal
        except Exception:
            raw_allowed = None

        debug_payload = {
            "portal_type": type(portal).__name__ if portal is not None else None,
            "email": email,
            "logout_url": logout_url,
            "allowed_evse_ids_arg": allowed_evse_ids,
            "allowed_from_portal_normalized": raw_allowed,
            "all_station_ids_count": len(all_keys) if all_keys else 0,
            "all_station_ids_sample": list(all_keys)[:10] if all_keys else [],
            "visible_station_ids_count": len(visible_keys) if visible_keys else 0,
            "visible_station_ids": list(visible_keys) if visible_keys else [],
        }
        st.divider()
        st.caption("Auth debug (RCA_AUTH_DEBUG=1)")
        st.code(json.dumps(debug_payload, indent=2), language="json")

    # Build label → station_id mapping from the visible keys
    pairs = sorted([(display_name(k), k) for k in visible_keys], key=lambda x: x[0])
    labels = [p[0] for p in pairs]
    friendly_to_key = dict(pairs)

    # If user has no visible EVSEs, show a note but still render the controls
    if not visible_keys:
        st.caption("You don't currently have access to any EVSEs.")

    # Clean visual by default: no chips shown; empty selection means \"all allowed EVSEs\"
    sel_labels = st.multiselect(" ", options=labels, default=[], label_visibility="collapsed")
    stations = [friendly_to_key[x] for x in sel_labels] if sel_labels else visible_keys
    st.caption("No selection = all allowed EVSEs")

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
        "Estimated Revenue ($)",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    # Stable key to track selection across reruns
    df["__session_key__"] = df["station_id"].astype(str) + "|" + df["transaction_id"].astype(str)

    # Display the table (read-only). Selection happens via the dropdown below
    # because this Streamlit version doesn't support row-click selection.
    st.dataframe(
        df[show_cols],
        hide_index=True,
        use_container_width=True,
        height=480,
        key="v2_sessions_table",
    )
    st.caption("Use the selector below to view details for a specific session.")

    # Build a compact label list for single-select
    labels = []
    for _, row in df.iterrows():
        label = f"{row.get('Start Date/Time','')} | {row.get('EVSE','')} | Tx {row.get('transaction_id','')}"
        labels.append(label)

    # Map labels -> session key
    label_to_key = dict(zip(labels, df["__session_key__"].tolist()))
    prev_selected_key = st.session_state.get("v2_selected_session_key", "")
    default_label = None
    if prev_selected_key:
        for label, key in label_to_key.items():
            if key == prev_selected_key:
                default_label = label
                break
    if default_label is None and labels:
        default_label = labels[0]

    selected_label = st.selectbox(
        "Select a session for details",
        options=labels,
        index=labels.index(default_label) if default_label in labels else 0,
        key="v2_sessions_select",
    )

    chosen_key = label_to_key.get(selected_label) if selected_label else df.loc[df.index[0], "__session_key__"]
    st.session_state["v2_selected_session_key"] = chosen_key

    sid, tx = chosen_key.split("|", 1)
    return sid, tx
