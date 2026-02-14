"""Auth helpers for the Streamlit dashboard.

This module is intentionally small and dependency-light.

In the web deployment, the Next.js portal `/api/auth/verify` is the source of truth.

Important note about Streamlit:
- The Streamlit UI is maintained over a WebSocket (`/_stcore/stream`). Browsers do not send
  arbitrary custom headers on that WebSocket upgrade.
- Because of that, relying on `x-portal-*` headers being visible inside Streamlit is brittle.

So, when possible, we call the portal verify endpoint from Streamlit (forwarding the browser
cookies) to retrieve the user's email and `allowed_evse_ids`.

When running locally (no portal), these helpers can fall back to a configurable dev mode.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

import streamlit as st


# -----------------------------
# Models
# -----------------------------


@dataclass(frozen=True)
class PortalUser:
    email: str | None
    user_id: str | None
    allowed_evse_ids: list[str] | None  # None => no restriction (local/dev)


# -----------------------------
# Header access
# -----------------------------


def _get_request_headers() -> dict[str, str]:
    """Best-effort access to request headers.

    IMPORTANT CONTEXT
    - Your browser DevTools shows only the headers the browser *sent*.
    - In your setup, identity/authorization headers (x-portal-*) are injected
      server-side by Caddy after forward_auth succeeds, and then forwarded
      upstream to Streamlit.

    Streamlit itself does not expose request headers consistently across
    versions/runtimes, so we try multiple approaches.

    Returns a **lower-cased** dict of header -> value.
    """

    def _normalize(raw_obj: Any) -> dict[str, str]:
        out: dict[str, str] = {}
        if not raw_obj:
            return out

        # raw_obj may be Mapping-like, or something convertible to dict
        try:
            items = dict(raw_obj).items()
        except Exception:
            try:
                items = raw_obj.items()  # type: ignore[attr-defined]
            except Exception:
                return out

        for k, v in items:
            if v is None:
                continue
            out[str(k).lower()] = str(v)
        return out

    # 1) Streamlit "public" context headers (varies by version)
    try:
        ctx = getattr(st, "context", None)
        if ctx is not None:
            raw = getattr(ctx, "headers", None)
            h = _normalize(raw)
            if h:
                return h
    except Exception:
        pass

    # 2) Streamlit internal websocket headers (commonly works with reverse proxies)
    # NOTE: this is a private API, but it's the most reliable way to access headers
    # across many Streamlit versions.
    try:
        # Newer/older versions have different names.
        try:
            from streamlit.web.server.websocket_headers import (  # type: ignore
                _get_websocket_headers as _ws_headers,
            )
        except Exception:
            from streamlit.web.server.websocket_headers import (  # type: ignore
                get_websocket_headers as _ws_headers,
            )

        raw = _ws_headers()  # type: ignore[misc]
        h = _normalize(raw)
        if h:
            return h
    except Exception:
        pass

    # 3) No headers available (common in local/dev or some deployments)
    return {}


def _debug_headers_if_enabled(headers: dict[str, str]) -> None:
    """Print request header diagnostics when RCA_AUTH_DEBUG=1.

    We only print a safe subset of headers (x-*, forwarded/proxy) to avoid leaking
    cookies or other secrets.
    """
    if os.getenv("RCA_AUTH_DEBUG") != "1":
        return

    safe: dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk.startswith("x-") or lk.startswith("cf-") or lk.startswith("forwarded") or lk.startswith("x-forwarded"):
            safe[lk] = v

    # Print to server logs (Render / Docker logs)
    print("RCA_AUTH_DEBUG headers_seen_by_streamlit:")
    print(json.dumps(safe, indent=2, sort_keys=True))


def _parse_allowed_evse(value: str | None) -> list[str] | None:
    """Parse allowed EVSE header value.

    Supports:
    - JSON array: ["id1","id2"]
    - comma-separated: id1,id2
    - empty/None: returns [] (explicitly none allowed)

    IMPORTANT: Returning None means "no restriction".
    Returning [] means "restricted to nothing".
    """
    if value is None:
        return None

    s = str(value).strip()
    if s == "":
        return []

    # JSON list
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                out: list[str] = []
                for item in data:
                    if item is None:
                        continue
                    item_s = str(item).strip()
                    if item_s:
                        out.append(item_s)
                return out
        except Exception:
            # fall through to comma parsing
            pass

    # Comma-separated
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


# -----------------------------
# Portal verify (Option 1)
# -----------------------------


def _portal_verify_url() -> str:
    """Return the portal verify URL for server-side calls from Streamlit.

    In Docker, Streamlit can reach the Next.js container via the service name.
    Override with RCA_PORTAL_VERIFY_URL if needed.
    """
    return os.getenv("RCA_PORTAL_VERIFY_URL", "http://recharge_web:3000/api/auth/verify")


def _call_portal_verify(cookie_header: str | None) -> tuple[int, dict[str, str], dict[str, Any] | None]:
    """Call the portal `/api/auth/verify` endpoint.

    Returns (status_code, response_headers_lower, json_body_or_none).

    We forward the browser Cookie header so the portal can authenticate the user.
    """
    url = _portal_verify_url()

    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    if cookie_header:
        req.add_header("Cookie", cookie_header)

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = getattr(resp, "status", 200)
            headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            body_bytes = resp.read() or b""

        body_json: dict[str, Any] | None = None
        if body_bytes:
            try:
                body_json = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except Exception:
                body_json = None

        return status, headers, body_json

    except urllib.error.HTTPError as e:
        headers = {str(k).lower(): str(v) for k, v in getattr(e, "headers", {}).items()}  # type: ignore[arg-type]
        # Try to parse the error body (often empty)
        try:
            raw = e.read()  # type: ignore[attr-defined]
        except Exception:
            raw = b""
        body_json: dict[str, Any] | None = None
        if raw:
            try:
                body_json = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                body_json = None
        return int(getattr(e, "code", 0) or 0), headers, body_json

    except Exception:
        # Network error / timeout
        return 0, {}, None


def _portal_user_from_verify_response(headers: dict[str, str], body: dict[str, Any] | None) -> PortalUser:
    """Extract PortalUser from verify response headers/body.

    Preferred: JSON body fields (if your verify endpoint returns them).
    Fallback: response headers (x-portal-*).
    """

    def _s(x: Any) -> str | None:
        if x is None:
            return None
        s = str(x).strip()
        return s or None

    email: str | None = None
    user_id: str | None = None
    allowed: list[str] | None = None

    if isinstance(body, dict):
        email = _s(body.get("email") or body.get("user_email") or body.get("userEmail"))
        user_id = _s(body.get("user_id") or body.get("userId") or body.get("id"))
        # allowed IDs can be returned as list or string
        allowed_val = body.get("allowed_evse_ids") or body.get("allowedEvseIds") or body.get("allowed_evse")
        if allowed_val is not None:
            if isinstance(allowed_val, list):
                allowed = [str(v).strip() for v in allowed_val if str(v).strip()]
            else:
                allowed = _parse_allowed_evse(_s(allowed_val))

    if email is None:
        email = _s(headers.get("x-portal-user-email") or headers.get("x-debug-portal-email") or headers.get("x-portal-email"))
    if user_id is None:
        user_id = _s(headers.get("x-portal-user-id") or headers.get("x-debug-portal-userid") or headers.get("x-portal-userid"))
    if allowed is None:
        allowed_raw = headers.get("x-portal-allowed-evse-ids") or headers.get("x-portal-allowed-evse") or headers.get("x-debug-portal-allowed-evse-alt") or headers.get("x-debug-portal-allowed-evse")
        allowed = _parse_allowed_evse(allowed_raw)

    return PortalUser(email=email, user_id=user_id, allowed_evse_ids=allowed)


# -----------------------------
# Public API
# -----------------------------


def get_portal_user() -> PortalUser:
    """Return the current portal user identity/authorization.

    Preferred (production): call the portal `/api/auth/verify` endpoint from Streamlit,
    forwarding the browser cookies.

    Fallback: try to read proxy-injected headers from the Streamlit request context.
    """

    h = _get_request_headers()
    _debug_headers_if_enabled(h)

    cookie = h.get("cookie")

    # Option 1: ask the portal who the user is (cookie-based)
    status, resp_headers, body = _call_portal_verify(cookie)
    if status == 200:
        u = _portal_user_from_verify_response(resp_headers, body)
        # If portal says ok but provides no identity, treat as unauthenticated
        if u.email or u.user_id or (u.allowed_evse_ids is not None):
            if os.getenv("RCA_AUTH_DEBUG") == "1":
                debug = {
                    "portal_verify_status": status,
                    "portal_verify_url": _portal_verify_url(),
                    "portal_verify_has_body": bool(body),
                    "portal_verify_email": u.email or "",
                    "portal_verify_allowed_count": (len(u.allowed_evse_ids) if isinstance(u.allowed_evse_ids, list) else None),
                }
                print("RCA_AUTH_DEBUG portal_verify_result:")
                print(json.dumps(debug, indent=2, sort_keys=True))
            return u

    # If verify explicitly says unauthorized, lock down.
    if status in (401, 403):
        return PortalUser(email=None, user_id=None, allowed_evse_ids=[])

    # Fallback: attempt to use headers visible to Streamlit (best-effort)
    def _h(*names: str) -> str | None:
        for name in names:
            v = h.get(name.lower())
            if v is None:
                continue
            s = str(v).strip()
            if s != "":
                return s
        return None

    email = _h(
        "x-portal-user-email",
        "x-portal-email",
        "x-debug-portal-email",
        "x-user-email",
        "x-auth-request-email",
        "cf-access-authenticated-user-email",
    )

    user_id = _h(
        "x-portal-user-id",
        "x-portal-userid",
        "x-debug-portal-userid",
        "x-user-id",
        "x-auth-request-user",
    )

    allowed_raw = _h(
        "x-portal-allowed-evse-ids",
        "x-portal-allowed-evse",
        "x-allowed-evse-ids",
        "x-debug-portal-allowed-evse-alt",
        "x-debug-portal-allowed-evse",
    )

    # If no identity is present at all, treat as local/dev (no restriction)
    if not email and not user_id and allowed_raw is None:
        # Local/dev: allow all by returning None (the caller must decide if this is OK)
        return PortalUser(email=None, user_id=None, allowed_evse_ids=None)

    allowed = _parse_allowed_evse(allowed_raw)

    if os.getenv("RCA_AUTH_DEBUG") == "1":
        present = {
            "fallback_x_portal_user_email": "x-portal-user-email" in h,
            "fallback_x_portal_user_id": "x-portal-user-id" in h,
            "fallback_x_portal_allowed_evse_ids": "x-portal-allowed-evse-ids" in h,
            "fallback_x_debug_portal_email": "x-debug-portal-email" in h,
            "fallback_cookie_present": bool(cookie),
            "portal_verify_status": status,
        }
        print("RCA_AUTH_DEBUG portal_fallback_header_presence:")
        print(json.dumps(present, indent=2, sort_keys=True))

    return PortalUser(email=email, user_id=user_id, allowed_evse_ids=allowed)


def filter_allowed_evse_ids(all_evse_ids: Iterable[str], allowed_evse_ids: list[str] | None) -> list[str]:
    """Filter a list of EVSE ids by an allowed list.

    Security model (production-safe):
    - allowed_evse_ids is None => DENY (return [])
    - allowed_evse_ids is [] => DENY (return [])
    - allowed_evse_ids contains "__ALL__" => allow all
    - otherwise => return intersection

    NOTE:
    We intentionally treat None as "deny" so that a portal user
    with no allowed_evse_ids configured cannot see any EVSE data.
    Local/dev environments should explicitly bypass this at a higher layer
    (e.g., by not supplying portal headers).
    """
    all_list = list(all_evse_ids)

    # Hard deny if nothing configured
    if not allowed_evse_ids:
        return []

    # Explicit superuser flag
    if "__ALL__" in allowed_evse_ids:
        return all_list

    allowed_set = set(allowed_evse_ids)
    return [x for x in all_list if x in allowed_set]


def require_portal_auth(redirect_url: str = "https://dashboard.rechargealaska.net/login") -> PortalUser:
    """Hard-stop the Streamlit app if portal headers are missing.

    Use this ONLY in the web deployment if you want Streamlit to never be directly
    usable without the portal.

    In local/dev, you typically should NOT call this.
    """
    u = get_portal_user()
    if not u.email:
        st.error("Access to this dashboard is restricted.")
        st.markdown(f"Please sign in via the portal: [{redirect_url}]({redirect_url})")
        st.stop()
    return u


def user_label(user: PortalUser) -> str:
    if user.email:
        return user.email
    if user.user_id:
        return user.user_id
    return ""


def debug_portal_headers() -> dict[str, str]:
    """Return lower-cased request headers for troubleshooting (do not display in production by default)."""
    return _get_request_headers()