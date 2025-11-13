# App_v2/rca_v2/admintab.py
from __future__ import annotations

import json, os, stat, subprocess, textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# Import base constants, then merged helpers if available
from . import constants as C

# ---- Runtime files ----
OVERRIDES_PATH = Path(__file__).with_name("runtime_overrides.json")
SECRETS_DIR = Path.home() / ".recharge_admin"
SECRETS_DIR.mkdir(parents=True, exist_ok=True)
SUPABASE_SECRETS = SECRETS_DIR / "supabase.json"


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
    cfg = _read_json(SUPABASE_SECRETS)
    url = st.session_state.get("__sb_url") or cfg.get("url") or os.getenv("SUPABASE_URL", "")
    key = st.session_state.get("__sb_key") or cfg.get("service_key") or os.getenv("SUPABASE_SERVICE_KEY", "")
    return {"url": url, "service_key": key}

def _save_supabase_creds(url: str, key: str) -> None:
    _write_json(SUPABASE_SECRETS, {"url": url, "service_key": key}, chmod_600=True)

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
    r = requests.post(_sb_table_url(url, "portal_users"), headers=_sb_headers(key), data=json.dumps(row), timeout=20)
    r.raise_for_status()
    return r.json()[0] if r.json() else {}

def _sb_update_user(url: str, key: str, user_id: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    import requests
    r = requests.patch(_sb_table_url(url, "portal_users") + f"?id=eq.{user_id}",
                       headers=_sb_headers(key), data=json.dumps(patch), timeout=20)
    r.raise_for_status()
    return r.json()[0] if r.json() else {}

# ---------- Cron helpers ----------
def _read_crontab() -> str:
    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""

def _write_crontab(new_body: str) -> bool:
    try:
        p = subprocess.run(["crontab", "-"], input=new_body, text=True, capture_output=True, check=False)
        return p.returncode == 0
    except Exception:
        return False


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
    st.markdown("### Automation (cron)")

    path_hint = str(Path.home() / "LynkWell DataSync" / "nightly_ingest.sh")
    st.caption(f"Target script: `{path_hint}`")
    current = _read_crontab()
    st.text_area("Current crontab (read-only)", value=current, height=140, disabled=True)

    sched = st.text_input("New schedule (cron expression)",
                          value='7,37 * * * *', help="Example: twice per hour at :07 and :37")
    line = f'{sched} "$HOME/LynkWell DataSync/nightly_ingest.sh" >> "$HOME/LynkWell DataSync/logs/ingest.log" 2>&1'
    st.code(line)

    if st.button("Replace crontab lines for nightly_ingest.sh"):
        # Remove existing ingest lines, append new one
        body_lines = [ln for ln in current.splitlines() if "nightly_ingest.sh" not in ln]
        body_lines.append(line)
        ok = _write_crontab("\n".join(body_lines) + "\n")
        st.success("Crontab updated.") if ok else st.error("Failed to update crontab. Try running Streamlit from Terminal.")

    st.divider()
    st.markdown("### Supabase Users")

    creds = _load_supabase_creds()
    with st.expander("Supabase connection"):
        u = st.text_input("Supabase URL", value=creds["url"])
        k = st.text_input("Service role key", value=creds["service_key"], type="password")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Save credentials (local only)"):
                _save_supabase_creds(u, k)
                st.success("Saved. Stored at ~/.recharge_admin/supabase.json (chmod 600).")
        with c2:
            st.caption("Service key is stored locally, never in the repo.")

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
                    else:
                        if not user_id:
                            st.error("user id required")
                        else:
                            out = _sb_update_user(url, key, user_id, {"active": False})
                            st.success(f"Deactivated {out.get('email', user_id)}")
                except Exception as e:
                    st.error(f"Supabase action failed: {e}")