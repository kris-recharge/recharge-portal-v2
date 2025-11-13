import os
import hmac
import time
from typing import Dict, Optional, Set

import streamlit as st

# Session keys
_SESSION_OK = "_auth_ok"
_SESSION_USER = "_auth_user"
_SESSION_TS = "_auth_ts"

# Default 12 hours, override via env var PORTAL_SESSION_MAX_AGE (seconds)
_MAX_AGE = int(os.getenv("PORTAL_SESSION_MAX_AGE", "43200"))


def _load_users_from_env() -> Dict[str, str]:
    """Load allowed users from env.
    Supports either:
      - PORTAL_USERS="user1:pass1; user2:pass2"
      - or legacy PORTAL_USER / PORTAL_PASSWORD
    """
    users: Dict[str, str] = {}

    multi = os.getenv("PORTAL_USERS", "").strip()
    if multi:
        for pair in multi.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                u, p = pair.split(":", 1)
                u, p = u.strip(), p.strip()
                if u and p:
                    users[u] = p

    # Legacy single-user fallback
    u1 = os.getenv("PORTAL_USER", "").strip()
    p1 = os.getenv("PORTAL_PASSWORD", "").strip()
    if u1 and p1:
        users.setdefault(u1, p1)

    return users


def _seed_allowed_evse() -> Optional[Set[str]]:
    """Optionally seed a default allowed‑EVSE set for the session from env.
    Format: PORTAL_ALLOWED_EVSE="as_xxx, as_yyy, as_zzz".
    This is a temporary convenience until full Supabase gating is wired.
    """
    raw = os.getenv("PORTAL_ALLOWED_EVSE", "").strip()
    if not raw:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def logout() -> None:
    for k in (_SESSION_OK, _SESSION_USER, _SESSION_TS, "_allowed_evse"):
        st.session_state.pop(k, None)
    st.experimental_rerun()


def require_auth() -> None:
    """Minimal env‑backed front door for the web build.

    - If no users are configured in env, this is a no‑op (useful for local dev).
    - Supports multiple users via PORTAL_USERS or a single legacy user via
      PORTAL_USER / PORTAL_PASSWORD.
    - Adds a Sign out control in the sidebar.
    - Optional session timeout via PORTAL_SESSION_MAX_AGE.
    - Optionally seeds `_allowed_evse` from PORTAL_ALLOWED_EVSE.
    """
    users = _load_users_from_env()
    if not users:
        # No auth configured → allow through (local/dev convenience)
        return

    # Check active session and expiry
    now = int(time.time())
    if st.session_state.get(_SESSION_OK) and st.session_state.get(_SESSION_TS):
        age = now - int(st.session_state[_SESSION_TS])
        if age < _MAX_AGE:
            # Active session → show who and offer logout
            with st.sidebar:
                user = st.session_state.get(_SESSION_USER, "")
                if user:
                    st.caption(f"Signed in as **{user}**")
                if st.button("Sign out"):
                    logout()
            return
        else:
            # Expired → clear
            for k in (_SESSION_OK, _SESSION_USER, _SESSION_TS):
                st.session_state.pop(k, None)

    # --- Login form ---
    st.title("🔒 ReCharge Alaska — Portal v2")
    with st.form("login"):
        u = st.text_input("Username", autocomplete="username")
        p = st.text_input("Password", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Sign in")

    if ok:
        pw = users.get(u)
        if pw and hmac.compare_digest(p, pw):
            st.session_state[_SESSION_OK] = True
            st.session_state[_SESSION_USER] = u
            st.session_state[_SESSION_TS] = now
            seed = _seed_allowed_evse()
            if seed is not None:
                st.session_state["_allowed_evse"] = seed
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
            st.stop()
    else:
        st.stop()