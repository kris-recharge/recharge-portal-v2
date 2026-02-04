import os
import json

import streamlit as st
st.set_page_config(page_title="ReCharge Alaska — Portal v2", layout="wide")

import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import requests

# ---------------- Sidebar: Account / Logout ----------------

from streamlit.components.v1 import html

with st.sidebar:
    st.markdown("### Account")

    # Use a real button + JS redirect so logout happens in the SAME tab.
    # A plain markdown link often opens a new tab and leaves the Streamlit
    # websocket session alive (making /app still usable until refresh).
    if st.button("Log out"):
        html(
            """
            <script>
              // Logout should happen in the SAME tab/window.
              // Do a fetch to clear cookies, then navigate to /login.
              fetch("/api/auth/logout", { method: "GET", credentials: "include" })
                .catch(() => {})
                .finally(() => {
                  window.top.location.href = "/login";
                });
            </script>
            """,
            height=0,
        )
        st.stop()

# -----------------------------------------------------------

# Optional soft gate: require a shared embed token when running behind a portal.
REQUIRED_EMBED_ACCESS_TOKEN = os.getenv("EMBED_ACCESS_TOKEN")


def check_embed_token() -> None:
    """
    Block direct access to the dashboard unless a valid embed token is present.

    This is a soft gate intended to discourage direct hits to the Streamlit URL.
    The primary authentication and authorization still live in the Next.js + Supabase portal.
    """
    # If no token is configured in the environment, do nothing (gate disabled).
    if not REQUIRED_EMBED_ACCESS_TOKEN:
        return

    # Streamlit 1.27+ has st.query_params; older releases use experimental_get_query_params.
    try:
        qp = st.query_params  # type: ignore[attr-defined]
    except Exception:
        qp = st.experimental_get_query_params()

    token = ""

    if isinstance(qp, dict):
        # qp may be Mapping[str, List[str]] or Mapping[str, str]
        val = qp.get("token")
        if isinstance(val, list):
            token = val[0] if val else ""
        else:
            token = val or ""
    else:
        # Newer Mapping-like object
        token = qp.get("token", "")

    if token != REQUIRED_EMBED_ACCESS_TOKEN:
        st.error("Access to this dashboard is restricted.")
        st.markdown(
            "Please sign in via the ReCharge Portal:\n\n"
            "[Open ReCharge Portal](https://dashboard.rechargealaska.net/login)"
        )
        st.stop()


# --- Begin: Portal user helpers (query param extraction, Supabase authorization) ---

def _get_request_headers() -> dict:
    """Best-effort access to the current request headers.

    Streamlit exposes headers via `st.context.headers` in newer versions.
    If unavailable, return an empty dict.
    """
    try:
        h = getattr(st, "context", None)
        h = getattr(h, "headers", None)
        if h is None:
            return {}
        # st.context.headers is a mapping-like object
        return {str(k).lower(): str(v) for k, v in dict(h).items()}
    except Exception:
        return {}


def _extract_header(headers: dict, key: str) -> str:
    """Fetch a header value (case-insensitive) as a simple string."""
    if not headers:
        return ""
    return (headers.get(key.lower()) or "").strip()


def get_current_user_email() -> str | None:
    """Best-effort extraction of the portal user's email.

    Priority order:
      1) Headers injected by the portal/proxy (most reliable)
      2) Query string (?email=... or ?user_email=...) for iframe embeds
    """
    headers = _get_request_headers()

    # Common header names you may choose to inject from Next/Caddy.
    for k in (
        "x-portal-user-email",
        "x-user-email",
        "x-auth-request-email",
        "x-forwarded-email",
    ):
        v = _extract_header(headers, k)
        if v:
            return v

    # Fallback to query params
    qp = _get_query_params()
    email = _extract_query_param(qp, "email")
    if not email:
        email = _extract_query_param(qp, "user_email")
    email = (email or "").strip()
    return email or None


def fetch_allowed_evse_ids_for_email(email: str | None):
    """Look up the set of station_ids this email is allowed to see from Supabase.

    Returns:
        - set([...])  : authorization data found and parsed
        - set()       : user is known but has no allowed EVSEs
        - None        : Supabase not configured / unreachable (no restriction)
    """
    # Prefer a dedicated SUPABASE_URL, but fall back to the public URL used by Next.
    supabase_url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not email or not supabase_url or not service_key:
        # Treat missing config as "no restriction" so the app still works locally.
        return None

    try:
        resp = requests.get(
            f"{supabase_url}/rest/v1/portal_users",
            params={
                "email": f"eq.{email}",
                "active": "eq.true",
                "select": "allowed_evse_ids",
                "limit": "1",
            },
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
            },
            timeout=5,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception:
        # Fail open: if Supabase is unavailable, don't break the dashboard.
        return None

    if not rows:
        # User exists in auth but has no portal_user_allowed_evse row.
        return set()

    raw_ids = rows[0].get("allowed_evse_ids") or []

    # Supabase normally returns a JSON array here; normalize defensively.
    if isinstance(raw_ids, str):
        try:
            parsed = json.loads(raw_ids)
            if isinstance(parsed, list):
                raw_ids = parsed
            else:
                raw_ids = [parsed]
        except Exception:
            raw_ids = [raw_ids]

    return {str(sid) for sid in raw_ids}


def fetch_allowed_evse_ids_from_headers():
    """Read allowed EVSE ids from headers, if provided by the portal/proxy.

    Header values may be:
      - JSON array string: ["id1","id2"]
      - Comma-separated: id1,id2

    Returns:
      - set([...]) if an allowlist header is present (even if empty)
      - None if no allowlist header is present
    """
    headers = _get_request_headers()

    # Header names we recognize (in priority order)
    header_names = (
        "x-portal-allowed-evse",
        "x-portal-allowed-evse-ids",
        "x-allowed-evse-ids",
    )

    present_name = None
    raw = None

    # IMPORTANT: distinguish "missing header" from "present but empty".
    # `headers.get()` returns None for missing keys; we must NOT treat that as "".
    for name in header_names:
        if name in headers:
            present_name = name
            raw = headers.get(name)
            break

    if present_name is None:
        # No allowlist header at all
        return None

    raw = "" if raw is None else str(raw)

    if raw.strip() == "":
        # Header present but empty -> explicit empty allowlist
        return set()

    raw = raw.strip()

    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {str(x) for x in parsed}
        if parsed is None:
            return set()
        return {str(parsed)}
    except Exception:
        pass

    # Fallback: comma-separated
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return {str(p) for p in parts}

def _get_query_params():
    """
    Helper to safely access the current query parameters in Streamlit.
    """
    try:
        return st.query_params  # type: ignore[attr-defined]
    except Exception:
        return st.experimental_get_query_params()


def _extract_query_param(qp, key: str) -> str:
    """
    Normalize a single query-parameter value to a simple string.
    """
    try:
        val = qp.get(key)
    except Exception:
        return ""
    if isinstance(val, list):
        return val[0] if val else ""
    return val or ""



# --- End: Portal user helpers ---

from rca_v2.config import APP_MODE
from rca_v2.ui import render_sidebar, sessions_table_single_select
from rca_v2.loaders import (
    load_meter_values,
    load_authorize,
    load_status_history,
    load_connectivity,
)
from rca_v2.sessions import build_sessions
from rca_v2.charts import (
    session_detail_figure,
    heatmap_count,
    heatmap_duration,
    daily_session_counts_and_energy_figures,
)
from rca_v2.constants import get_evse_display, connector_type_for
from rca_v2.export import build_export_xlsx_bytes



EVSE_DISPLAY = get_evse_display()

# Soft gate: require the correct embed token when configured.
check_embed_token()

# Identify current portal user (if any).
current_email = get_current_user_email()
if current_email:
    st.session_state["__portal_user_email"] = current_email

# If we are behind the portal/proxy, we should have identity headers.
# In that case, we MUST fail-closed if the portal does not provide an EVSE allowlist.
_headers = _get_request_headers()
_running_behind_portal = bool(_extract_header(_headers, "x-portal-user-email") or _extract_header(_headers, "x-portal-user-id"))

# Prefer explicit allowed EVSE ids passed via headers (best for portal deployments).
ALLOWED_STATIONS = fetch_allowed_evse_ids_from_headers()

# If we're behind the portal and it didn't provide an allowlist, fail closed.
# (Otherwise a misconfigured env would show ALL EVSEs.)
if _running_behind_portal and ALLOWED_STATIONS is None:
    st.error(
        "Authorization scope missing (no EVSE allowlist provided by the portal). "
        "Please contact ReCharge Alaska support."
    )
    st.stop()

# If headers didn't specify allowed EVSE ids, fall back to Supabase lookup by email.
if ALLOWED_STATIONS is None:
    ALLOWED_STATIONS = fetch_allowed_evse_ids_for_email(current_email)

if isinstance(ALLOWED_STATIONS, set):
    if not ALLOWED_STATIONS:
        st.error(
            "Your account does not have access to any EVSEs. "
            "Please contact ReCharge Alaska if you believe this is an error."
        )
        st.stop()

    # Restrict the global EVSE display map to the allowed set only.
    EVSE_DISPLAY = {
        sid: name
        for sid, name in EVSE_DISPLAY.items()
        if sid in ALLOWED_STATIONS
    }
    st.session_state["__portal_allowed_station_ids"] = ALLOWED_STATIONS
else:
    # None = fail-open / local dev: no authorization filter applied.
    st.session_state["__portal_allowed_station_ids"] = None

with st.sidebar:
    # Pass the filtered EVSE_DISPLAY into the sidebar renderer so that
    # users only see EVSEs they are authorized for. Fall back gracefully
    # if the function signature does not accept any arguments.
    try:
        stations, start_utc, end_utc = render_sidebar(EVSE_DISPLAY)
    except TypeError:
        # Older/local versions of render_sidebar() may not take parameters.
        stations, start_utc, end_utc = render_sidebar()

# Enforce EVSE authorization on the selected station list as a second line of defense.
_allowed = st.session_state.get("__portal_allowed_station_ids")
if isinstance(_allowed, (set, list, tuple)) and _allowed:
    allowed_set = {str(s) for s in _allowed}
    if stations:
        stations = [s for s in stations if str(s) in allowed_set]


# Best-effort enrichment of status rows with Tritium error metadata.
def _enrich_status_with_tritium(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort enrichment of status rows with Tritium error metadata.

    Joins the provided frame on vendor_error_code (digits only) to a
    lookup table loaded via load_tritium_error_codes(). If the lookup
    table is missing or has an unexpected shape, this function is a
    no-op and simply returns the original frame.
    """
    try:
        from rca_v2.loaders import load_tritium_error_codes

        if "vendor_error_code" not in df.columns:
            # Nothing to enrich
            return df

        codes = load_tritium_error_codes()
        if not isinstance(codes, pd.DataFrame) or codes.empty:
            return df

        base = df.copy()
        # Normalize the vendor_error_code column to a digits-only key
        base["code_key"] = (
            base["vendor_error_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
        )

        # Normalize the codes table. Support either `code` or `code_key`.
        if "code" in codes.columns:
            codes2 = codes.rename(columns={"code": "code_key"}).copy()
        elif "code_key" in codes.columns:
            codes2 = codes.copy()
        else:
            return df

        codes2["code_key"] = (
            codes2["code_key"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
        )

        # Ensure the impact/description columns exist on the lookup side
        for col in ("impact", "description"):
            if col not in codes2.columns:
                codes2[col] = ""

        # Join only on the digits-only error code; platform-specific
        # differences can be handled later if needed.
        merged = base.merge(
            codes2[["code_key", "impact", "description"]],
            on="code_key",
            how="left",
            suffixes=("", "_tr"),
        )

        # Ensure the output always has impact/description columns
        if "impact" not in merged.columns:
            merged["impact"] = ""
        if "description" not in merged.columns:
            merged["description"] = ""

        return merged
    except Exception:
        # Keep the calling code resilient; if anything goes wrong,
        # just return the original frame unchanged.
        return df


# Helper to load data per-EVSE and concatenate, to avoid SQLite IN () quirks.
def _load_per_evse(loader_fn, ids, start_iso, end_iso):
    """
    Helper to load data per‑EVSE and concatenate, to avoid SQLite IN () quirks.
    """
    if not ids:
        # No filter means all EVSEs; let the loader decide by passing None or []
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


# Helper to sanitize DataFrame datetime columns for Excel export (remove timezone info)
def _excel_safe_datetimes(df: pd.DataFrame, local_tz: str = "America/Anchorage") -> pd.DataFrame:
    """Return a copy of df with timezone-aware datetimes converted to tz-naive.

    Excel writers (xlsxwriter/openpyxl) do not support timezone-aware datetimes.
    We convert tz-aware datetime columns to `local_tz` and then drop the tzinfo.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    # Handle true datetime64 columns first
    for col in out.columns:
        s = out[col]
        # pandas datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                # If tz-aware, convert to local then make naive
                tz = getattr(getattr(s.dt, "tz", None), "zone", None) or getattr(s.dt, "tz", None)
                if tz is not None:
                    out[col] = s.dt.tz_convert(local_tz).dt.tz_localize(None)
                else:
                    # tz-naive already
                    out[col] = s
            except Exception:
                # If anything weird happens, try a safe coercion path
                try:
                    tmp = pd.to_datetime(s, errors="coerce")
                    tz2 = getattr(getattr(tmp.dt, "tz", None), "zone", None) or getattr(tmp.dt, "tz", None)
                    if tz2 is not None:
                        out[col] = tmp.dt.tz_convert(local_tz).dt.tz_localize(None)
                    else:
                        out[col] = tmp
                except Exception:
                    pass

    return out




def _make_station_key(stations):
    """Normalize a stations list/set into a hashable cache key.

    We convert to a sorted tuple of stringified IDs so that different
    list orderings produce the same cache key.
    """
    if isinstance(stations, (list, set, tuple)):
        return tuple(sorted(str(s) for s in stations))
    elif stations:
        return (str(stations),)
    else:
        return tuple()


@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def cached_status_history(station_key, start_utc, end_utc):
    """Cached wrapper around load_status_history.

    station_key is the normalized, hashable representation of the
    selected EVSEs (see _make_station_key).
    """
    stations = list(station_key)
    return load_status_history(stations, start_utc, end_utc)


@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def cached_connectivity(station_key, start_utc, end_utc):
    """Cached wrapper around load_connectivity.

    station_key is the normalized, hashable representation of the
    selected EVSEs (see _make_station_key).
    """
    stations = list(station_key)
    return load_connectivity(stations, start_utc, end_utc)


@st.cache_data(show_spinner=False, ttl=300, max_entries=20)
def load_mv_auth_and_sessions(stations_key, start_iso, end_iso):
    """
    Cached loader for meter values, authorize records, and built sessions/heatmaps.

    stations_key should be a hashable representation of the selected EVSEs
    (e.g., a sorted tuple of station IDs). This keeps interactive clicks fast:
    changing only the selected row will reuse cached data for the same filters.
    """
    # Normalize stations_key back to a simple list for the per‑EVSE loader
    if isinstance(stations_key, (list, set, tuple)):
        stations_list = list(stations_key)
    elif stations_key:
        stations_list = [stations_key]
    else:
        stations_list = []

    mv = _load_per_evse(load_meter_values, stations_list, start_iso, end_iso)
    auth = _load_per_evse(load_authorize, stations_list, start_iso, end_iso)
    sess, heat = build_sessions(mv, auth)
    return mv, auth, sess, heat

# Treat "no selection" as ALL EVSEs (matches sidebar hint)
if not stations:
    stations = list(EVSE_DISPLAY.keys())
    st.session_state["__v2_all_evse"] = True
else:
    st.session_state["__v2_all_evse"] = False

# Build tabs (Admin tab removed from the shared web deployment)
TAB_TITLES = ["Charging Sessions", "Status History", "Connectivity", "Data Export"]
t1, t2, t3, t4 = st.tabs(TAB_TITLES)

with t1:
    st.subheader("Charging Sessions")
    with st.spinner("Loading data…"):
        # Use a cached loader so that interactive clicks re-use data for the same filters.
        stations_key = _make_station_key(stations)
        mv, auth, sess, heat = load_mv_auth_and_sessions(stations_key, start_utc, end_utc)
    # Defensive: re-apply EVSE filter at the session level (guards against loader quirks)
    if isinstance(stations, (list, tuple, set)) and len(stations) > 0 and "station_id" in sess.columns:
        _wanted = {str(s) for s in stations}
        sess = sess[sess["station_id"].astype(str).isin(_wanted)]
    if sess.empty:
        st.warning("No sessions found."); st.stop()
    # Filter out invalid rows (must have a transaction_id and a valid connector)
    # NOTE: We intentionally KEEP 0 kWh sessions here so attempted/failed sessions
    # are visible in the table and export. Daily Totals will exclude them via a
    # success threshold.
    sess = sess.copy()
    if "transaction_id" in sess.columns:
        sess = sess[sess["transaction_id"].notna()]
        sess = sess[sess["transaction_id"].astype(str).str.strip().ne("")]
        sess = sess[~sess["transaction_id"].astype(str).str.lower().isin(["none", "nan"])]

    if "Connector #" in sess.columns:
        sess = sess[sess["Connector #"].isin([1, 2])]

    # Coerce numeric columns we rely on
    sess["Energy Delivered (kWh)"] = pd.to_numeric(sess.get("Energy Delivered (kWh)"), errors="coerce")
    sess["Max Power (kW)"] = pd.to_numeric(sess.get("Max Power (kW)"), errors="coerce")
    sess["Duration (min)"] = pd.to_numeric(sess.get("Duration (min)"), errors="coerce")

    # Define success threshold (kWh). Anything below this is treated as an attempt/failed.
    SUCCESS_KWH_THRESHOLD = 0.5

    # Add an outcome flag for UI + export (does not affect raw values)
    if "Energy Delivered (kWh)" in sess.columns:
        sess["Outcome"] = np.where(
            sess["Energy Delivered (kWh)"].fillna(0.0) >= SUCCESS_KWH_THRESHOLD,
            "Success",
            "Attempt/Failed",
        )
    else:
        sess["Outcome"] = "Unknown"

    # If everything is empty after basic validation, stop.
    if sess.empty:
        st.warning("No sessions found in this window."); st.stop()
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

    # Effective-dated connector type override (e.g., Delta CHAdeMO -> NACS on a cutover date)
    # This ensures the table reflects your authoritative mapping rules instead of whatever
    # legacy value is stored on the session row.
    #
    # IMPORTANT: We compute both AK-local and UTC-aware start timestamps so that the
    # effective-dating logic works regardless of whether the incoming Start Date/Time is
    # tz-aware (UTC) or tz-naive (already AK-local).
    try:
        if {"Connector #", "Start Date/Time"}.issubset(set(sess.columns)):
            AK_TZ = "America/Anchorage"

            # Parse session start times.
            # First attempt: parse as UTC-aware (works if the source string is UTC or has tz info).
            start_parsed_utc = pd.to_datetime(
                sess["Start Date/Time"],
                errors="coerce",
                utc=True,
            )

            # Also parse without forcing UTC so we can detect tz-naive inputs.
            start_parsed_raw = pd.to_datetime(sess["Start Date/Time"], errors="coerce")
            try:
                raw_tz = getattr(start_parsed_raw.dt, "tz", None)
            except Exception:
                raw_tz = None

            if raw_tz is not None:
                # tz-aware raw -> convert to AK and UTC
                try:
                    start_ak = start_parsed_raw.dt.tz_convert(AK_TZ)
                except Exception:
                    start_ak = start_parsed_raw
                try:
                    start_utc_ts = start_parsed_raw.dt.tz_convert("UTC")
                except Exception:
                    start_utc_ts = start_parsed_utc
            else:
                # tz-naive raw -> assume already AK-local, then convert to UTC
                try:
                    start_ak = start_parsed_raw.dt.tz_localize(
                        AK_TZ,
                        ambiguous="NaT",
                        nonexistent="shift_forward",
                    )
                except Exception:
                    start_ak = start_parsed_raw

                try:
                    start_utc_ts = start_ak.dt.tz_convert("UTC")
                except Exception:
                    start_utc_ts = start_parsed_utc

            # Build an override column by calling connector_type_for(...)
            # and also apply a safety-net rule for the Delta cutover.
            # Cutover is effective 2026-01-30 00:00:00 AK local.
            CUTOFF_AK_TS = pd.Timestamp("2026-01-30 00:00:00", tz=AK_TZ)

            def _ct_override(row):
                # Prefer station_id mapping when available
                sid = row.get("station_id")
                cid = row.get("Connector #")
                sdt_utc = row.get("__start_utc")
                sdt_ak = row.get("__start_ak")

                # Normalize connector id
                try:
                    cid_int = int(cid)
                except Exception:
                    return None

                # --- Safety net: Delta connector 1 changed from CHAdeMO -> NACS on 2026-01-30 (AK local) ---
                # We key off the friendly EVSE name (robust match) since station_id strings can vary by platform.
                try:
                    evse_name = str(row.get("EVSE") or "")
                except Exception:
                    evse_name = ""

                evse_key = " ".join(evse_name.strip().lower().split())

                # Ensure we have a comparable AK-local timestamp
                sdt_ak_ts = None
                try:
                    if pd.notna(sdt_ak):
                        sdt_ak_ts = pd.to_datetime(sdt_ak, errors="coerce")
                except Exception:
                    sdt_ak_ts = None

                # Apply Delta cutover only for connector 1
                if cid_int == 1 and sdt_ak_ts is not None and pd.notna(sdt_ak_ts):
                    if "delta" in evse_key and ("left" in evse_key or "right" in evse_key):
                        try:
                            return "NACS" if sdt_ak_ts >= CUTOFF_AK_TS else "CHAdeMO"
                        except Exception:
                            # If tz comparison fails for any reason, fall back to date-based check
                            try:
                                return "NACS" if sdt_ak_ts.date() >= CUTOFF_AK_TS.date() else "CHAdeMO"
                            except Exception:
                                return None

                # --- Primary rule engine ---
                # If station_id exists, ask constants.connector_type_for() using UTC-aware timestamp.
                if sid is None or pd.isna(sdt_utc):
                    return None
                try:
                    return connector_type_for(str(sid), cid_int, sdt_utc)
                except Exception:
                    return None

            sess = sess.copy()
            sess["__start_ak"] = start_ak
            sess["__start_utc"] = start_utc_ts

            _override = sess.apply(_ct_override, axis=1)

            if "Connector Type" in sess.columns:
                # Prefer override when present, otherwise keep existing
                sess["Connector Type"] = _override.where(_override.notna(), sess["Connector Type"])
            else:
                sess["Connector Type"] = _override

            sess = sess.drop(columns=["__start_ak", "__start_utc"], errors="ignore")
    except Exception:
        # Never break the sessions tab just because an override rule failed.
        pass
    cols = [
        "Start Date/Time",
        "End Date/Time",
        "EVSE",
        "Connector #",
        "Connector Type",
        "Max Power (kW)",
        "Energy Delivered (kWh)",
        "Duration (min)",
        "Outcome",
        "SoC Start",
        "SoC End",
        "ID Tag",
    ]
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

    # Daily totals (AK-local, based on session START time)
    st.markdown("#### Daily Totals")

    # Build daily totals directly here so it works reliably in Docker environments
    # where timezone databases can differ.
    try:
        sess_for_daily = sess.copy()

        # Parse the session START time.
        # IMPORTANT:
        # - In some deployments, `Start Date/Time` is already an AK-local *naive* timestamp string.
        # - In others, it may be UTC/tz-aware.
        # We detect tz-awareness and only convert when appropriate.
        if "Start Date/Time" in sess_for_daily.columns:
            start_parsed = pd.to_datetime(
                sess_for_daily["Start Date/Time"],
                errors="coerce",
            )
        elif "start_time" in sess_for_daily.columns:
            start_parsed = pd.to_datetime(
                sess_for_daily["start_time"],
                errors="coerce",
            )
        else:
            start_parsed = pd.Series([pd.NaT] * len(sess_for_daily), index=sess_for_daily.index)

        AK_TZ = "America/Anchorage"

        # If tz-aware, convert to AK. If tz-naive, assume it is already AK-local and localize.
        try:
            tzinfo = getattr(start_parsed.dt, "tz", None)
        except Exception:
            tzinfo = None

        if tzinfo is not None:
            # tz-aware timestamps (often UTC) -> convert to AK
            try:
                start_ak = start_parsed.dt.tz_convert(AK_TZ)
            except Exception:
                # As a last resort, strip tz and treat as local
                start_ak = start_parsed.dt.tz_localize(None)
        else:
            # tz-naive timestamps -> assume already AK-local
            try:
                start_ak = start_parsed.dt.tz_localize(
                    AK_TZ,
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                )
            except Exception:
                # Fallback: keep as naive
                start_ak = start_parsed

        sess_for_daily["__start_ak"] = start_ak
        # Grouping key: AK-local calendar day
        sess_for_daily["__day_ak"] = pd.to_datetime(sess_for_daily["__start_ak"], errors="coerce").dt.date

        # Energy (kWh) for the day is based on the session START day.
        energy_col = "Energy Delivered (kWh)" if "Energy Delivered (kWh)" in sess_for_daily.columns else None
        if energy_col:
            sess_for_daily["__energy_kwh"] = pd.to_numeric(sess_for_daily[energy_col], errors="coerce").fillna(0.0)
        else:
            sess_for_daily["__energy_kwh"] = 0.0
        # Exclude attempts/failed sessions from Daily Totals.
        # We keep them in the table/export, but do not count them in KPI charts.
        SUCCESS_KWH_THRESHOLD = 0.5
        sess_for_daily = sess_for_daily[sess_for_daily["__energy_kwh"].fillna(0.0) >= SUCCESS_KWH_THRESHOLD]

        # Drop rows without a start time
        sess_for_daily = sess_for_daily.dropna(subset=["__day_ak"])

        daily = (
            sess_for_daily
            .groupby("__day_ak", as_index=False)
            .agg(
                sessions=("__day_ak", "size"),
                energy_kwh=("__energy_kwh", "sum"),
            )
            .sort_values("__day_ak")
        )

        # Plotly figures
        import plotly.express as px

        fig_daily_count = px.bar(
            daily,
            x="__day_ak",
            y="sessions",
            labels={"__day_ak": "Date", "sessions": "Sessions"},
            title="Sessions per Day",
        )
        fig_daily_count.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        fig_daily_energy = px.bar(
            daily,
            x="__day_ak",
            y="energy_kwh",
            labels={"__day_ak": "Date", "energy_kwh": "kWh"},
            title="Energy Delivered per Day (kWh)",
        )
        fig_daily_energy.update_layout(margin=dict(l=10, r=10, t=40, b=10))

        st.plotly_chart(
            fig_daily_count,
            use_container_width=True,
            config={"displaylogo": False},
            key="daily_totals_count",
        )

        st.plotly_chart(
            fig_daily_energy,
            use_container_width=True,
            config={"displaylogo": False},
            key="daily_totals_energy",
        )

    except Exception as e:
        # Don’t fail the whole page; show the reason so we can diagnose quickly.
        st.warning(f"Unable to render daily totals charts: {e}")

    # Heatmaps — stacked full width (to match v1 layout)
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

    stations_key = _make_station_key(stations)
    status_df = cached_status_history(stations_key, start_utc, end_utc)
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

        # Tritium enrichment (best‑effort; no‑op if lookup isn’t wired yet)
        df = _enrich_status_with_tritium(df)

        # Optional: only rows that actually have a vendor_error_code
        # Apply this after Tritium enrichment so we filter on the final column.
        if show_only_vendor and "vendor_error_code" in df.columns:
            v_raw = df["vendor_error_code"]
            v = v_raw.astype(str).str.strip()

            keep = (
                ~v_raw.isna()
                & (v != "")
                & (~v.str.lower().isin(["none", "nan"]))
                & (~v.isin(["0", "0000", "0.0"]))
            )

            df = df[keep]

            if df.empty:
                st.info(
                    "No status rows with vendor_error_code in this window for the selected EVSE."
                )
                st.stop()

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
        st.dataframe(display[final], use_container_width=True, hide_index=True)

with t3:
    st.subheader("Connectivity")
    stations_key = _make_station_key(stations)
    conn_df = cached_connectivity(stations_key, start_utc, end_utc)
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

        st.dataframe(display, use_container_width=True, hide_index=True)

with t4:
    st.subheader("Data Export")

    # Export-specific date/time range controls (AK local), separate from the main sidebar.
    AK = "America/Anchorage"
    try:
        start_local = pd.to_datetime(start_utc).tz_convert(AK)
    except Exception:
        start_local = pd.Timestamp.now(tz=AK) - pd.Timedelta(days=1)
    try:
        end_local = pd.to_datetime(end_utc).tz_convert(AK)
    except Exception:
        end_local = pd.Timestamp.now(tz=AK)

    # Two rows of inputs: start/end date on top, start/end time below.
    row1_col1, row1_spacer, row1_col2 = st.columns([1, 0.3, 1])
    row2_col1, row2_spacer, row2_col2 = st.columns([1, 0.3, 1])

    with row1_col1:
        export_start_date = st.date_input(
            "Start date (export)",
            value=start_local.date(),
            key="export_start_date",
        )
    with row1_col2:
        export_end_date = st.date_input(
            "End date (export)",
            value=end_local.date(),
            key="export_end_date",
        )

    with row2_col1:
        export_start_time = st.time_input(
            "Start time (export)",
            value=start_local.time().replace(microsecond=0),
            key="export_start_time",
        )
    with row2_col2:
        export_end_time = st.time_input(
            "End time (export)",
            value=end_local.time().replace(microsecond=0),
            key="export_end_time",
        )

    # Build export window in UTC based on the export-specific controls.
    try:
        start_dt_local = pd.Timestamp(
            datetime.combine(export_start_date, export_start_time),
            tz=AK,
        )
    except Exception:
        start_dt_local = start_local
    try:
        end_dt_local = pd.Timestamp(
            datetime.combine(export_end_date, export_end_time),
            tz=AK,
        )
    except Exception:
        end_dt_local = end_local

    # NOTE: Excel writers do not support timezone-aware datetimes.
    # We keep UTC semantics but drop tzinfo before passing to the exporter.
    export_start_utc = start_dt_local.tz_convert("UTC")
    export_end_utc = end_dt_local.tz_convert("UTC")

    export_start_utc_naive = export_start_utc.tz_localize(None)
    export_end_utc_naive = export_end_utc.tz_localize(None)

    if export_end_utc <= export_start_utc:
        st.warning("Export end time must be after start time.")
        st.stop()

    sess_last = st.session_state.get("__v2_last_sessions", pd.DataFrame())
    mv_last = st.session_state.get("__v2_last_meter", pd.DataFrame())

    # Excel cannot write timezone-aware datetimes; make export frames tz-naive.
    sess_last = _excel_safe_datetimes(sess_last)
    mv_last = _excel_safe_datetimes(mv_last)

    # Load Status + Connectivity for the export window.
    # IMPORTANT: these are loaded for the export-specific date/time inputs (not the main sidebar window).
    try:
        export_start_iso = export_start_utc.isoformat()
        export_end_iso = export_end_utc.isoformat()

        # Always use the same station filter the user selected (or "all EVSE" fallback).
        export_stations = stations

        status_export = _load_per_evse(load_status_history, export_stations, export_start_iso, export_end_iso)
        conn_export = _load_per_evse(load_connectivity, export_stations, export_start_iso, export_end_iso)

        # Make them Excel-safe too.
        status_export = _excel_safe_datetimes(status_export)
        conn_export = _excel_safe_datetimes(conn_export)
    except Exception:
        status_export = pd.DataFrame()
        conn_export = pd.DataFrame()

    if sess_last.empty and mv_last.empty:
        st.info("No data available to export from this view. Visit the Charging Sessions tab first.")
        st.stop()

    try:
        xlsx_bytes = build_export_xlsx_bytes(
            sessions_df=sess_last,
            meter_values_df=mv_last,
            status_df=status_export,
            connectivity_df=conn_export,
            start_utc=export_start_utc_naive,
            end_utc=export_end_utc_naive,
            evse_display=EVSE_DISPLAY,
        )

        # Center the download button beneath the export controls
        spacer_left, center_col, spacer_right = st.columns([1, 1, 1])
        with center_col:
            st.download_button(
                "⬇ Download Excel",
                data=xlsx_bytes,
                file_name="recharge_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=(
                    "Exports Sessions, MeterValues (window), Status, Connectivity Data, and Connectivity Summary "
                    "for the selected window and EVSE filter."
                ),
            )
    except Exception as e:
        st.error(f"Export failed: {e}")
