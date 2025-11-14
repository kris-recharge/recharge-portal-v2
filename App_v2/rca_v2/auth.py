import os
import hmac
import time
from typing import Dict, Optional, Set

import streamlit as st
import requests

# Session keys
_SESSION_OK = "_auth_ok"
_SESSION_USER = "_auth_user"
_SESSION_TS = "_auth_ts"
_LOGIN_REDIRECT_TS = "_login_redirect_ts"

# Default 12 hours, override via env var PORTAL_SESSION_MAX_AGE (seconds)
_MAX_AGE = int(os.getenv("PORTAL_SESSION_MAX_AGE", "43200"))

# Supabase environment (either direct or NEXT_PUBLIC_* forwarded)
SB_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SB_ANON = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Where to send users to sign in (Next.js front door).
# You can override with PORTAL_LOGIN_URL, else we fall back to NEXT_PUBLIC_PORTAL_URL, else a sensible default.
LOGIN_URL = (
    os.getenv("PORTAL_LOGIN_URL")
    or os.getenv("NEXT_PUBLIC_PORTAL_URL")
    or "https://recharge-portal.onrender.com/login"
)

def _redirect_to_login() -> None:
    """Hard redirect to the front-door login; guard against tight loops."""
    now = int(time.time())
    last = st.session_state.get(_LOGIN_REDIRECT_TS)
    if last and (now - int(last) < 8):
        # We just redirected recently → show a button instead of looping
        st.warning("We just tried to send you to the sign-in page. Click below to continue.")
        try:
            st.link_button("Go to sign-in", LOGIN_URL)
        except Exception:
            st.markdown(f"[Go to sign-in]({LOGIN_URL})")
        st.stop()

    st.session_state[_LOGIN_REDIRECT_TS] = now
    st.markdown(f'<meta http-equiv="refresh" content="0; url={LOGIN_URL}">', unsafe_allow_html=True)
    st.info("Redirecting to sign-in… If nothing happens, click below.")
    try:
        st.link_button("Go to sign-in", LOGIN_URL)
    except Exception:
        st.markdown(f"[Go to sign-in]({LOGIN_URL})")
    st.stop()

def _verify_supabase_token_and_get_email(token: str) -> Optional[str]:
    """
    Validate a Supabase access token by calling /auth/v1/user.
    Returns the user's email on success, else None.
    """
    if not token or not SB_URL or not SB_ANON:
        return None
    try:
        r = requests.get(
            f"{SB_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SB_ANON},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json() or {}
            email = (data.get("email") or "").strip().lower()
            return email or None
    except Exception:
        pass
    return None

def _get_query_params() -> Dict[str, str]:
    """Return URL query params as a simple dict[str, str] across Streamlit versions."""
    try:
        if hasattr(st, "query_params"):
            raw = st.query_params
            # In modern Streamlit this is a QueryParams proxy with .to_dict()
            if hasattr(raw, "to_dict"):
                d = raw.to_dict()
            else:
                try:
                    d = dict(raw)  # best-effort
                except Exception:
                    d = {}
        else:
            d = st.experimental_get_query_params() or {}
    except Exception:
        d = {}

    out: Dict[str, str] = {}
    for k, v in d.items():
        out[k] = v[0] if isinstance(v, list) else v
    return out

def _bootstrap_auth_from_query_params() -> Optional[str]:
    """
    Reads ?sb= / ?access_token= / ?token= from the URL, verifies with Supabase,
    stores session, and scrubs the token from the URL bar.
    """
    qp = _get_query_params()
    raw = qp.get("sb") or qp.get("access_token") or qp.get("token")
    email = _verify_supabase_token_and_get_email(raw) if raw else None
    if email:
        st.session_state[_SESSION_OK] = True
        st.session_state[_SESSION_USER] = email
        st.session_state[_SESSION_TS] = int(time.time())
        seed = _seed_allowed_evse()
        if seed is not None:
            st.session_state["_allowed_evse"] = seed
        # Remove token-bearing params from the browser URL
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        return email
    return None


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
    """SSO-only gating.

    Order:
      1) If existing session valid → allow
      2) If ?sb / ?access_token / ?token present → verify via Supabase and allow
      3) Else → redirect to the Next.js login (no local form)
    """
    now = int(time.time())

    # Existing valid session?
    if st.session_state.get(_SESSION_OK) and st.session_state.get(_SESSION_TS):
        age = now - int(st.session_state[_SESSION_TS])
        if age < _MAX_AGE:
            with st.sidebar:
                user = st.session_state.get(_SESSION_USER, "")
                if user:
                    st.caption(f"Signed in as **{user}**")
                if st.button("Sign out"):
                    logout()
            return
        else:
            for k in (_SESSION_OK, _SESSION_USER, _SESSION_TS):
                st.session_state.pop(k, None)

    # Supabase SSO pass-through via query params
    email = _bootstrap_auth_from_query_params()
    if email:
        with st.sidebar:
            st.caption(f"Signed in as **{email}**")
            if st.button("Sign out"):
                logout()
        return

    # No valid session and no token → send to login
    _redirect_to_login()