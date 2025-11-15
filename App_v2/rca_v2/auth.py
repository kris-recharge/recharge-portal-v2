import os
import hmac
import time
import json
from typing import Dict, Optional, Set
from urllib.parse import urlparse, parse_qs

import streamlit as st
import requests
from streamlit.components.v1 import html

# Session keys
_SESSION_OK = "_auth_ok"
_SESSION_USER = "_auth_user"
_SESSION_TS = "_auth_ts"
_LOGIN_REDIRECT_TS = "_login_redirect_ts"

# Debug flag (turn on in Render env: DEBUG_AUTH=1)
DEBUG_AUTH = os.getenv("DEBUG_AUTH", os.getenv("PORTAL_DEBUG_AUTH", "0")).strip().lower() in {"1", "true", "yes", "on"}

# Default 12 hours, override via env var PORTAL_SESSION_MAX_AGE (seconds)
_MAX_AGE = int(os.getenv("PORTAL_SESSION_MAX_AGE", "43200"))

# Supabase environment (either direct or NEXT_PUBLIC_* forwarded)
SB_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SB_ANON = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Treat anyone on this domain as admin (wildcard EVSE access)
ADMIN_DOMAIN = os.getenv("PORTAL_ADMIN_DOMAIN", "rechargealaska.net").strip().lower()

# Where to send users to sign in (Next.js front door).
# You can override with PORTAL_LOGIN_URL, else we fall back to NEXT_PUBLIC_PORTAL_URL, else a sensible default.
LOGIN_URL = (
    os.getenv("PORTAL_LOGIN_URL")
    or os.getenv("NEXT_PUBLIC_PORTAL_URL")
    or "https://recharge-portal.onrender.com/login"
)

def _dbg(msg: str, obj: Optional[object] = None) -> None:
    if not DEBUG_AUTH:
        return
    try:
        if obj is not None:
            try:
                pretty = json.dumps(obj, indent=2, default=str)[:4000]
            except Exception:
                pretty = str(obj)[:4000]
            st.write(f"🔎 {msg}:\n\n```\n{pretty}\n```")
        else:
            st.write(f"🔎 {msg}")
    except Exception:
        # Best-effort log to console
        print(f"[AUTH-DEBUG] {msg}: {obj!r}")

def _redirect_to_login() -> None:
    """Hard redirect to the front-door login; guard against tight loops."""
    now = int(time.time())
    last = st.session_state.get(_LOGIN_REDIRECT_TS)
    _dbg("Redirecting to login", {"login_url": LOGIN_URL, "last_redirect_ts": last})
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
        _dbg("Verifying token against Supabase", {"SB_URL": SB_URL, "has_ANON": bool(SB_ANON), "token_preview": (token[:12] + "...") if token else None})
        r = requests.get(
            f"{SB_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SB_ANON},
            timeout=8,
        )
        data = r.json() or {}
        _dbg("Supabase /auth/v1/user response", {"status": r.status_code, "json": data})
        if r.status_code == 200:
            email = (data.get("email") or "").strip().lower()
            return email or None
    except Exception as e:
        _dbg("Token verify exception", str(e))
    return None

def _get_query_params() -> Dict[str, str]:
    """Return URL query params as a simple dict[str, str] across Streamlit versions."""
    raw_qp = {}
    try:
        if hasattr(st, "query_params"):
            qp = st.query_params
            if hasattr(qp, "to_dict"):
                raw_qp = qp.to_dict()
            else:
                try:
                    raw_qp = dict(qp)
                except Exception:
                    raw_qp = {}
        else:
            raw_qp = st.experimental_get_query_params() or {}
    except Exception:
        raw_qp = {}

    # Normalize to single-value dict
    out: Dict[str, str] = {}
    for k, v in raw_qp.items():
        out[k] = v[0] if isinstance(v, list) else v

    _dbg("Parsed query params", {"raw": raw_qp, "normalized": out})
    return out

def _promote_hash_token_to_query_from_hash() -> None:
    """If the auth token arrives in the URL fragment (e.g. #sb=TOKEN),
    convert it to a query param (?sb=TOKEN) and reload once. This helps
    avoid loops when the front door uses a hash-based redirect."""
    # Inject a tiny JS shim; no-op if there is no hash token.
    js = r"""
    <script>
    (function () {
      try {
        var h = window.location.hash || "";
        if (!h) return;
        var m = h.match(/[#&](sb|access_token|token)=([^&]+)/);
        if (!m) return;
        var key = m[1];
        var val = m[2];
        var u = new URL(window.location.href);
        // Only promote if no token already in the query string
        if (!u.searchParams.get('sb') && !u.searchParams.get('access_token') && !u.searchParams.get('token')) {
          // Normalize to ?sb=
          u.searchParams.set('sb', val);
          u.hash = '';
          window.location.replace(u.toString());
        }
      } catch (e) {
        // swallow
      }
    })();
    </script>
    """
    try:
      html(js, height=0)
    except Exception:
      pass

def _bootstrap_auth_from_query_params() -> Optional[str]:
    """
    Reads ?sb= / ?access_token= / ?token= from the URL, verifies with Supabase,
    stores session, and scrubs the token from the URL bar.
    """
    qp = _get_query_params()
    _dbg("Bootstrap from query", qp)
    raw = qp.get("sb") or qp.get("access_token") or qp.get("token")
    email = _verify_supabase_token_and_get_email(raw) if raw else None
    if email:
        st.session_state[_SESSION_OK] = True
        st.session_state[_SESSION_USER] = email
        st.session_state[_SESSION_TS] = int(time.time())
        seed = _seed_allowed_evse()
        if seed is not None:
            st.session_state["_allowed_evse"] = seed
        # If email is on the admin domain, grant wildcard EVSE access
        try:
            email_l = email.strip().lower()
            if ADMIN_DOMAIN and email_l.endswith("@" + ADMIN_DOMAIN):
                st.session_state["_allowed_evse"] = {"*"}
        except Exception:
            pass
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
    if DEBUG_AUTH:
        st.sidebar.markdown("**Auth Debug:** ON")
    now = int(time.time())

    # Promote hash token (if any) to query string before we attempt to parse
    _promote_hash_token_to_query_from_hash()

    # Existing valid session?
    if st.session_state.get(_SESSION_OK) and st.session_state.get(_SESSION_TS):
        age = now - int(st.session_state[_SESSION_TS])
        if age < _MAX_AGE:
            user = st.session_state.get(_SESSION_USER, "") or ""
            # Ensure admin wildcard is present on every refresh
            try:
                if ADMIN_DOMAIN and isinstance(user, str) and user.lower().endswith("@" + ADMIN_DOMAIN):
                    st.session_state.setdefault("_allowed_evse", {"*"})
            except Exception:
                pass
            with st.sidebar:
                if user:
                    st.caption(f"Signed in as **{user}**")
                if st.button("Sign out"):
                    logout()
            _dbg("Existing session valid", {"user": st.session_state.get(_SESSION_USER), "age": age, "max_age": _MAX_AGE})
            return
        else:
            for k in (_SESSION_OK, _SESSION_USER, _SESSION_TS):
                st.session_state.pop(k, None)
            _dbg("Session expired; cleared")

    # Supabase SSO pass-through via query params
    email = _bootstrap_auth_from_query_params()
    if email:
        _dbg("Signed in via URL token", {"email": email})
        with st.sidebar:
            st.caption(f"Signed in as **{email}**")
            if st.button("Sign out"):
                logout()
        return

    # No valid session and no token → send to login
    if DEBUG_AUTH:
        qp = _get_query_params()
        st.warning("No valid session and no token found; redirecting to sign‑in.")
        _dbg("Final state before redirect", {
            "has_session": bool(st.session_state.get(_SESSION_OK)),
            "qp": qp,
            "SB_URL": SB_URL,
            "LOGIN_URL": LOGIN_URL,
        })

    # Debug escape: visit with ?noredirect=1 to inspect without bouncing
    qp = _get_query_params()
    if str(qp.get("noredirect", "")).lower() in {"1", "true", "yes"}:
        st.warning("Auth redirect suppressed by ?noredirect=1. You are not signed in.")
        return

    _redirect_to_login()