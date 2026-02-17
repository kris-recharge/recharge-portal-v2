# App_v2/rca_v2/admintab.py
from __future__ import annotations

import json, os, stat
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Import base constants, then merged helpers if available
from . import constants as C

# ---- Runtime files ----
OVERRIDES_PATH = Path(__file__).with_name("runtime_overrides.json")


# ---------- small IO helpers ----------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, obj: Dict[str, Any], chmod_600: bool = False) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    if chmod_600:
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

def _merged_map(base: Dict[str, Any], key: str) -> Dict[str, Any]:
    ov = _read_json(OVERRIDES_PATH).get(key, {})
    return {**base, **ov}

def _merged_list(key: str) -> List[str]:
    return list(_read_json(OVERRIDES_PATH).get(key, []))

def _upsert_override(key: str, payload: Any) -> Dict[str, Any]:
    ov = _read_json(OVERRIDES_PATH)
    ov[key] = payload
    _write_json(OVERRIDES_PATH, ov)
    return ov


# ---------- Supabase (REST) ----------
def _load_supabase_creds() -> Dict[str, str]:
    """Load Supabase REST credentials.

    We prefer env vars (from .env / docker-compose env_file):
      - NEXT_PUBLIC_SUPABASE_URL (or SUPABASE_URL)
      - SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_SERVICE_KEY

    You can also temporarily override in the current Streamlit session via
    st.session_state['__sb_url'] / st.session_state['__sb_key'].
    """
    url = (
        st.session_state.get("__sb_url")
        or os.getenv("SUPABASE_URL", "")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
    )

    key = (
        st.session_state.get("__sb_key")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.getenv("SUPABASE_SERVICE_ROLE", "")
        or os.getenv("SUPABASE_SERVICE_KEY", "")
    )
    return {"url": url, "service_key": key}


def _sb_headers(key: str) -> Dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def _sb_table_url(base_url: str, table: str) -> str:
    return f"{base_url.rstrip('/')}/rest/v1/{table}"

def _sb_get_users(url: str, key: str) -> pd.DataFrame:
    import requests
    r = requests.get(_sb_table_url(url, "portal_users") + "?select=*", headers=_sb_headers(key), timeout=20)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data) if data else pd.DataFrame()

def _sb_insert_user(url: str, key: str, row: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    r = requests.post(
        _sb_table_url(url, "portal_users"),
        headers=_sb_headers(key),
        json=row,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()[0] if r.json() else {}

def _sb_update_user(url: str, key: str, user_id: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    r = requests.patch(
        _sb_table_url(url, "portal_users") + f"?id=eq.{user_id}",
        headers=_sb_headers(key),
        json=patch,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()[0] if r.json() else {}


def _sb_get_pricing(url: str, key: str) -> pd.DataFrame:
    """Fetch EVSE pricing rows from Supabase."""
    import requests
    # order by most-recent start first
    q = "?select=*&order=effective_start.desc.nullslast"
    r = requests.get(_sb_table_url(url, "evse_pricing") + q, headers=_sb_headers(key), timeout=20)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame(data) if data else pd.DataFrame()


def _sb_insert_pricing(url: str, key: str, row: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    r = requests.post(
        _sb_table_url(url, "evse_pricing"),
        headers=_sb_headers(key),
        json=row,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()[0] if r.json() else {}


def _sb_update_pricing(url: str, key: str, pricing_id: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    r = requests.patch(
        _sb_table_url(url, "evse_pricing") + f"?id=eq.{pricing_id}",
        headers=_sb_headers(key),
        json=patch,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()[0] if r.json() else {}


def _coerce_numeric(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _ts_to_iso_z(ts: Optional[pd.Timestamp]) -> Optional[str]:
    """Convert a pandas Timestamp (or None) to RFC3339 string with timezone."""
    if ts is None:
        return None
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            return None
    if ts.tzinfo is None:
        # treat naive timestamps as UTC
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    # Supabase likes RFC3339
    return ts.isoformat()


def _pricing_payload(
    station_id: str,
    connection_fee: Optional[float],
    price_per_kwh: Optional[float],
    price_per_min: Optional[float],
    idle_fee_per_min: Optional[float],
    effective_start: Optional[pd.Timestamp],
    effective_end: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "station_id": station_id,
        "connection_fee": connection_fee,
        "price_per_kwh": price_per_kwh,
        "price_per_min": price_per_min,
        "idle_fee_per_min": idle_fee_per_min,
        "effective_start": _ts_to_iso_z(effective_start),
        "effective_end": _ts_to_iso_z(effective_end),
    }
    # remove nulls so we don't overwrite columns with null accidentally
    return {k: v for k, v in row.items() if v is not None}


# ==========================================================
# Public Entry
# ==========================================================
def render_admin_tab():
    st.subheader("Admin")
    st.caption("Local admin tools: EVSE & Locations, automation schedules, and Supabase user access.")

    # Resolve current, merged views
    try:
        evse_display = C.get_evse_display()      # merged map if helper exists
    except Exception:
        evse_display = getattr(C, "EVSE_DISPLAY", {})
    try:
        platform_map = C.get_platform_map()
    except Exception:
        platform_map = getattr(C, "PLATFORM_MAP", {})
    try:
        archived_set = set(C.get_archived_station_ids())
    except Exception:
        archived_set = set(_merged_list("archived_station_ids"))

    st.divider()
    st.markdown("### EVSE & Locations")

    # Show current table
    df = pd.DataFrame(
        [{"station_id": sid,
          "display": evse_display.get(sid, ""),
          "location": getattr(C, "EVSE_LOCATION", {}).get(sid, ""),
          "platform": platform_map.get(sid, ""),
          "archived": sid in archived_set}
         for sid in sorted(set(evse_display.keys()) | set(platform_map.keys()))]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Add / Update EVSE"):
        c1, c2 = st.columns([1, 1])
        with c1:
            sid = st.text_input("station_id (e.g., as_LYHe6mZTRKiFfziSNJFvJ)").strip()
            disp = st.text_input("Display name (EVSE)", value="")
            loc = st.text_input("Location label (e.g., ARG - Left / Glennallen)", value="")
        with c2:
            platform = st.selectbox("Platform", ["", "RT50", "RTM", "Autel", "Other"])
            archived = st.checkbox("Archive this EVSE", value=False)

        if st.button("Save EVSE"):
            if not sid:
                st.error("station_id is required.")
            else:
                ov = _read_json(OVERRIDES_PATH)
                ev_map = ov.get("evse_display", {})
                lo_map = ov.get("evse_location", {})
                pf_map = ov.get("platform_map", {})
                ar_list = set(ov.get("archived_station_ids", []))

                if disp:
                    ev_map[sid] = disp
                if loc:
                    lo_map[sid] = loc
                if platform:
                    pf_map[sid] = platform
                if archived:
                    ar_list.add(sid)
                else:
                    ar_list.discard(sid)

                ov["evse_display"] = ev_map
                ov["evse_location"] = lo_map
                ov["platform_map"] = pf_map
                ov["archived_station_ids"] = sorted(ar_list)
                _write_json(OVERRIDES_PATH, ov)
                st.success(f"Saved overrides for {sid}")

    with st.expander("Archive / Unarchive EVSEs"):
        all_sids = sorted(evse_display.keys())
        current_arch = set(_merged_list("archived_station_ids"))
        selection = st.multiselect("Archived EVSEs", options=all_sids, default=sorted(current_arch))
        if st.button("Apply archive set"):
            ov = _read_json(OVERRIDES_PATH)
            ov["archived_station_ids"] = selection
            _write_json(OVERRIDES_PATH, ov)
            st.success("Archive list updated.")

    st.divider()
    st.markdown("### Supabase Users")

    creds = _load_supabase_creds()
    with st.expander("Supabase connection"):
        st.caption(
            "Credentials are loaded from environment variables (.env / docker-compose env_file). "
            "You can temporarily override for this session (not saved to disk)."
        )

        u = st.text_input("Supabase URL", value=creds["url"], help="From NEXT_PUBLIC_SUPABASE_URL or SUPABASE_URL")
        k = st.text_input(
            "Service role key",
            value=creds["service_key"],
            type="password",
            help="From SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_SERVICE_KEY",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Use values for this session"):
                st.session_state["__sb_url"] = u
                st.session_state["__sb_key"] = k
                st.success("Using these Supabase credentials for this session.")
        with c2:
            if st.button("Clear session override"):
                st.session_state.pop("__sb_url", None)
                st.session_state.pop("__sb_key", None)
                st.success("Cleared session override; back to environment variables.")

        # Simple status hints
        st.write(
            {
                "has_url": bool(u.strip()),
                "has_service_role_key": bool(k.strip()),
                "url_source": "env/session",
                "key_source": "env/session",
            }
        )

    url, key = _load_supabase_creds().values()
    if url and key:
        try:
            users = _sb_get_users(url, key)
        except Exception as e:
            st.error(f"Could not load users: {e}")
            users = pd.DataFrame()
    else:
        st.info("Enter/Save Supabase URL and service key to manage users.")
        users = pd.DataFrame()

    if not users.empty:
        st.dataframe(users, use_container_width=True, hide_index=True)

    with st.expander("Add / Update user"):
        email = st.text_input("Email").strip()
        name = st.text_input("Name").strip()
        allowed = st.multiselect("Allowed EVSEs", options=sorted(evse_display.keys()))
        is_active = st.checkbox("Active", value=True)
        mode = st.radio("Action", ["Create", "Update (by id)", "Deactivate (by id)"], horizontal=True)
        user_id = st.text_input("Existing user id (for update/deactivate)")

        if st.button("Submit user action"):
            if not url or not key:
                st.error("Missing Supabase credentials.")
            else:
                try:
                    if mode == "Create":
                        row = {"email": email, "name": name,
                               "allowed_evse_ids": allowed, "active": is_active}
                        out = _sb_insert_user(url, key, row)
                        st.success(f"Created {out.get('email', '')}")
                        st.rerun()
                    elif mode.startswith("Update"):
                        if not user_id:
                            st.error("user id required for update")
                        else:
                            patch = {"email": email or None, "name": name or None,
                                     "allowed_evse_ids": allowed, "active": is_active}
                            # remove Nones so we don't overwrite with null unintentionally
                            patch = {k: v for k, v in patch.items() if v is not None}
                            out = _sb_update_user(url, key, user_id, patch)
                            st.success(f"Updated {out.get('email', user_id)}")
                            st.rerun()
                    else:
                        if not user_id:
                            st.error("user id required")
                        else:
                            out = _sb_update_user(url, key, user_id, {"active": False})
                            st.success(f"Deactivated {out.get('email', user_id)}")
                            st.rerun()
                except Exception as e:
                    st.error(f"Supabase action failed: {e}")

    st.divider()
    st.markdown("### Supabase Pricing")
    st.caption(
        "Manage EVSE pricing in Supabase (connection fee, $/kWh, $/min, idle $/min, and effective date ranges). "
        "These rows become the source of truth for estimated revenue and future billing views."
    )

    if not url or not key:
        st.info("Enter/Save Supabase URL and service key above to manage pricing.")
    else:

        # Load pricing table if it exists
        try:
            pricing = _sb_get_pricing(url, key)
        except Exception as e:
            st.warning(
                "Could not load `evse_pricing` from Supabase. If the table doesn't exist yet, create it and try again. "
                f"\n\nDetails: {e}"
            )
            pricing = pd.DataFrame()

        if not pricing.empty:
            # Human friendly view
            friendly = dict(evse_display)
            pricing_view = pricing.copy()
            if "station_id" in pricing_view.columns:
                pricing_view.insert(
                    1,
                    "evse",
                    pricing_view["station_id"].map(lambda x: friendly.get(x, "")),
                )
            st.dataframe(pricing_view, use_container_width=True, hide_index=True)
        else:
            st.caption("No pricing rows found (or table not created yet).")

        with st.expander("Add pricing rule"):
            # Pick EVSE by friendly name, store station_id
            station_ids = sorted(evse_display.keys())
            labels = [f"{evse_display.get(sid, sid)}  ({sid})" for sid in station_ids]
            label_to_sid = {lab: sid for lab, sid in zip(labels, station_ids)}

            chosen = st.selectbox("EVSE", options=labels) if labels else ""
            station_id = label_to_sid.get(chosen, "")

            c1, c2 = st.columns([1, 1])
            with c1:
                connection_fee = st.text_input("Connection fee ($)", value="0")
                price_kwh = st.text_input("Price per kWh ($/kWh)", value="")
            with c2:
                price_min = st.text_input("Price per minute ($/min)", value="")
                idle_min = st.text_input("Idle fee ($/min)", value="0")

            # Effective date range (AK local UI -> store UTC)
            AK_TZ = "America/Anchorage"

            c3, c4 = st.columns([1, 1])
            with c3:
                st.caption("Effective start (AK local)")
                eff_start_date = st.date_input(
                    "Start date",
                    value=pd.Timestamp.now(tz=AK_TZ).date(),
                    key="pricing_eff_start_date",
                )
                eff_start_time = st.time_input(
                    "Start time",
                    value=pd.Timestamp.now(tz=AK_TZ).replace(second=0, microsecond=0).time(),
                    key="pricing_eff_start_time",
                )

            with c4:
                eff_end_enabled = st.checkbox("Set an effective end", value=False)
                eff_end_date = None
                eff_end_time = None
                if eff_end_enabled:
                    st.caption("Effective end (AK local)")
                    eff_end_date = st.date_input(
                        "End date",
                        value=pd.Timestamp.now(tz=AK_TZ).date(),
                        key="pricing_eff_end_date",
                    )
                    eff_end_time = st.time_input(
                        "End time",
                        value=pd.Timestamp.now(tz=AK_TZ).replace(second=0, microsecond=0).time(),
                        key="pricing_eff_end_time",
                    )

            if st.button("Create pricing rule"):
                if not station_id:
                    st.error("No EVSE selected.")
                else:
                    c_fee = _coerce_numeric(connection_fee)
                    p_kwh = _coerce_numeric(price_kwh)
                    p_min = _coerce_numeric(price_min)
                    i_min = _coerce_numeric(idle_min)

                    if p_kwh is None and p_min is None:
                        st.error("Provide at least one of: $/kWh or $/min")
                    else:
                        # Build AK-local timestamps then convert to UTC
                        start_local = pd.Timestamp.combine(eff_start_date, eff_start_time).tz_localize(AK_TZ)
                        start_ts = start_local.tz_convert("UTC")

                        end_ts = None
                        if eff_end_enabled and eff_end_date is not None and eff_end_time is not None:
                            end_local = pd.Timestamp.combine(eff_end_date, eff_end_time).tz_localize(AK_TZ)
                            end_ts = end_local.tz_convert("UTC")
                        if end_ts is not None and end_ts <= start_ts:
                            st.error("Effective end must be after effective start.")
                        else:
                            row = _pricing_payload(
                                station_id=station_id,
                                connection_fee=c_fee,
                                price_per_kwh=p_kwh,
                                price_per_min=p_min,
                                idle_fee_per_min=i_min,
                                effective_start=start_ts,
                                effective_end=end_ts,
                            )
                            try:
                                out = _sb_insert_pricing(url, key, row)
                                st.success(f"Created pricing rule id={out.get('id', '')} for {station_id}")
                                # Refresh the page so the new row shows up immediately
                                st.rerun()
                            except Exception as e:
                                # Try to surface HTTP status/body for PostgREST errors
                                status = getattr(getattr(e, "response", None), "status_code", None)
                                body = getattr(getattr(e, "response", None), "text", None)
                                if status is not None:
                                    st.error(f"Failed to create pricing rule (HTTP {status}).")
                                    if body:
                                        st.code(body[:2000])
                                else:
                                    st.error(f"Failed to create pricing rule: {e}")

        with st.expander("Update pricing rule (by id)"):
            pricing_id = st.text_input("Pricing row id")
            st.caption("Only fields you enter here will be updated.")

            c1, c2 = st.columns([1, 1])
            with c1:
                connection_fee_u = st.text_input("Connection fee ($) [optional]", value="")
                price_kwh_u = st.text_input("Price per kWh ($/kWh) [optional]", value="")
            with c2:
                price_min_u = st.text_input("Price per minute ($/min) [optional]", value="")
                idle_min_u = st.text_input("Idle fee ($/min) [optional]", value="")

            AK_TZ = "America/Anchorage"

            c3, c4 = st.columns([1, 1])
            with c3:
                eff_start_u_enabled = st.checkbox("Update effective start", value=False)
                eff_start_u_date = None
                eff_start_u_time = None
                if eff_start_u_enabled:
                    st.caption("New effective start (AK local)")
                    eff_start_u_date = st.date_input(
                        "New start date",
                        value=pd.Timestamp.now(tz=AK_TZ).date(),
                        key="pricing_update_start_date",
                    )
                    eff_start_u_time = st.time_input(
                        "New start time",
                        value=pd.Timestamp.now(tz=AK_TZ).replace(second=0, microsecond=0).time(),
                        key="pricing_update_start_time",
                    )

            with c4:
                eff_end_u_enabled = st.checkbox("Update effective end", value=False)
                eff_end_u_date = None
                eff_end_u_time = None
                if eff_end_u_enabled:
                    st.caption("New effective end (AK local)")
                    eff_end_u_date = st.date_input(
                        "New end date",
                        value=pd.Timestamp.now(tz=AK_TZ).date(),
                        key="pricing_update_end_date",
                    )
                    eff_end_u_time = st.time_input(
                        "New end time",
                        value=pd.Timestamp.now(tz=AK_TZ).replace(second=0, microsecond=0).time(),
                        key="pricing_update_end_time",
                    )

            if st.button("Apply pricing update"):
                if not pricing_id.strip():
                    st.error("Pricing row id is required.")
                else:
                    patch: Dict[str, Any] = {}

                    c_fee = _coerce_numeric(connection_fee_u)
                    if c_fee is not None:
                        patch["connection_fee"] = c_fee

                    p_kwh = _coerce_numeric(price_kwh_u)
                    if p_kwh is not None:
                        patch["price_per_kwh"] = p_kwh

                    p_min = _coerce_numeric(price_min_u)
                    if p_min is not None:
                        patch["price_per_min"] = p_min

                    i_min = _coerce_numeric(idle_min_u)
                    if i_min is not None:
                        patch["idle_fee_per_min"] = i_min

                    if eff_start_u_enabled and eff_start_u_date is not None and eff_start_u_time is not None:
                        start_local = pd.Timestamp.combine(eff_start_u_date, eff_start_u_time).tz_localize(AK_TZ)
                        patch["effective_start"] = _ts_to_iso_z(start_local.tz_convert("UTC"))

                    if eff_end_u_enabled:
                        if eff_end_u_date is not None and eff_end_u_time is not None:
                            end_local = pd.Timestamp.combine(eff_end_u_date, eff_end_u_time).tz_localize(AK_TZ)
                            patch["effective_end"] = _ts_to_iso_z(end_local.tz_convert("UTC"))
                        else:
                            patch["effective_end"] = None

                    if not patch:
                        st.warning("No fields provided to update.")
                    else:
                        try:
                            out = _sb_update_pricing(url, key, pricing_id.strip(), patch)
                            st.success(f"Updated pricing rule id={out.get('id', pricing_id.strip())}")
                            st.rerun()
                        except Exception as e:
                            status = getattr(getattr(e, "response", None), "status_code", None)
                            body = getattr(getattr(e, "response", None), "text", None)
                            if status is not None:
                                st.error(f"Failed to update pricing rule (HTTP {status}).")
                                if body:
                                    st.code(body[:2000])
                            else:
                                st.error(f"Failed to update pricing rule: {e}")