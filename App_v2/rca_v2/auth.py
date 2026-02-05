

"""Auth helpers for the Streamlit dashboard.

This module is intentionally small and dependency-light.

In the web deployment, Caddy forward_auth calls the Next.js portal `/api/auth/verify`
which (when valid) returns useful identity/authorization information via response headers.
Caddy forwards those headers to Streamlit.

We read those headers (when available) to:
- display the signed-in user next to the logout control
- restrict EVSE visibility to the user's `allowed_evse_ids`

When running locally (no proxy headers), these helpers gracefully fall back to
"no restriction".
"""

from __future__ import annotations

import json
import os
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

    Streamlit has evolved header access over time. We try a few approaches so this
    keeps working across versions.

    Returns a **lower-cased** dict of header -> value.
    """
    headers: dict[str, str] = {}

    # Streamlit 1.27+ (varies by version): st.context.headers may exist.
    try:
        ctx = getattr(st, "context", None)
        if ctx is not None:
            raw = getattr(ctx, "headers", None)
            if raw:
                # raw is a Mapping-like
                for k, v in dict(raw).items():
                    if v is None:
                        continue
                    headers[str(k).lower()] = str(v)
                return headers
    except Exception:
        pass

    # Older fallback: experimental_get_query_params exists, but not headers.
    # We intentionally do not attempt private Streamlit internals here.
    return headers


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
# Public API
# -----------------------------



def get_portal_user() -> PortalUser:
    """Return the current portal user identity/authorization, if present.

    Expected headers (case-insensitive) in the web deployment (via Caddy forward_auth):
    - x-portal-user-email
    - x-portal-user-id
    - x-portal-allowed-evse-ids   (JSON list or comma-separated)

    Backwards-compatible/alternate header names we also accept:
    - x-portal-email, x-user-email, x-auth-request-email
    - x-portal-allowed-evse (older)

    If headers are not present (typical local run), allowed_evse_ids is None.
    """

    h = _get_request_headers()
    _debug_headers_if_enabled(h)

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
        "x-user-email",
        "x-auth-request-email",
        "cf-access-authenticated-user-email",
    )

    user_id = _h(
        "x-portal-user-id",
        "x-portal-userid",
        "x-user-id",
        "x-auth-request-user",
    )

    # Prefer the explicit allow-list header name.
    allowed_raw = _h(
        "x-portal-allowed-evse-ids",
        "x-portal-allowed-evse",
        "x-allowed-evse-ids",
    )

    # If the proxy isn't providing portal headers at all, treat as local/dev.
    if not email and not user_id and allowed_raw is None:
        return PortalUser(email=None, user_id=None, allowed_evse_ids=None)

    allowed = _parse_allowed_evse(allowed_raw)

    return PortalUser(email=email, user_id=user_id, allowed_evse_ids=allowed)


def filter_allowed_evse_ids(all_evse_ids: Iterable[str], allowed_evse_ids: list[str] | None) -> list[str]:
    """Filter a list of EVSE ids by an allowed list.

    - allowed_evse_ids is None => return all (no restriction)
    - allowed_evse_ids is [] => return []
    """
    all_list = list(all_evse_ids)
    if allowed_evse_ids is None:
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