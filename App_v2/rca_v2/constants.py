EVSE_DISPLAY = {
    "as_c8rCuPHDd7sV1ynHBVBiq": "ARG - Right",
    "as_cnIGqQ0DoWdFCo7zSrN01": "ARG - Left",
    "as_oXoa7HXphUu5riXsSW253": "Delta - Right",
    "as_xTUHfTKoOvKSfYZhhdlhT": "Delta - Left",
    "as_LYHe6mZTRKiFfziSNJFvJ": "Glennallen",
}

EVSE_LOCATION = {
    "as_c8rCuPHDd7sV1ynHBVBiq": "ARG",
    "as_cnIGqQ0DoWdFCo7zSrN01": "ARG",
    "as_oXoa7HXphUu5riXsSW253": "Delta Junction",
    "as_xTUHfTKoOvKSfYZhhdlhT": "Delta Junction",
    "as_LYHe6mZTRKiFfziSNJFvJ": "Glennallen",
}

CONNECTOR_TYPE = {
    ("as_c8rCuPHDd7sV1ynHBVBiq", 1): "CCS",
    ("as_c8rCuPHDd7sV1ynHBVBiq", 2): "CCS",
    ("as_cnIGqQ0DoWdFCo7zSrN01", 1): "NACS",
    ("as_cnIGqQ0DoWdFCo7zSrN01", 2): "CCS",
    ("as_LYHe6mZTRKiFfziSNJFvJ", 1): "NACS",
    ("as_LYHe6mZTRKiFfziSNJFvJ", 2): "CCS",
    ("as_oXoa7HXphUu5riXsSW253", 1): "NACS",
    ("as_oXoa7HXphUu5riXsSW253", 2): "CCS",
    ("as_xTUHfTKoOvKSfYZhhdlhT", 1): "NACS",
    ("as_xTUHfTKoOvKSfYZhhdlhT", 2): "CCS",
}

# --- Effective-dated connector changes ---
# Delta Junction connector 1 changed from CHAdeMO to NACS starting 2026-01-30 (Anchorage local date).
from datetime import datetime, date

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

_AK_TZ_NAME = "America/Anchorage"
_DELTA_CONN1_CUTOFF_AK_DATE = date(2026, 1, 30)

# Station IDs for the two Delta units
_DELTA_STATIONS = {
    "as_oXoa7HXphUu5riXsSW253",  # Delta - Right
    "as_xTUHfTKoOvKSfYZhhdlhT",  # Delta - Left
}


def _to_datetime(value) -> datetime | None:
    """Best-effort conversion to a timezone-aware datetime.

    Accepts:
      - datetime
      - ISO-8601-like strings ("2026-01-30T01:23:45Z", etc.)
      - objects that implement .to_pydatetime() (e.g. pandas Timestamp)

    Returns None if parsing fails.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        # pandas Timestamp and similar
        if hasattr(value, "to_pydatetime"):
            try:
                dt = value.to_pydatetime()
            except Exception:
                dt = None
        else:
            dt = None

        if dt is None and isinstance(value, str):
            s = value.strip()
            # Handle trailing Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(s)
            except Exception:
                return None

    # Ensure tz-aware
    if dt.tzinfo is None:
        # Assume UTC if no tzinfo is present
        dt = dt.replace(tzinfo=ZoneInfo("UTC") if ZoneInfo else None)

    return dt


def connector_type_for(
    station_id: str,
    connector_id: int,
    session_start_utc=None,
) -> str:
    """Return the connector type for a station/connector at a given time.

    - Uses `CONNECTOR_TYPE` + runtime overrides as the base.
    - Applies effective-dated logic for known connector changes.

    For Delta Junction units, connector 1 is:
      - CHAdeMO for 2026-01-29 and earlier (Anchorage local date)
      - NACS for 2026-01-30 and later (Anchorage local date)

    If `session_start_utc` is None or unparseable, we return the current mapping.
    """
    base_map = get_connector_type()
    current = base_map.get((station_id, connector_id), "")

    # Only Delta connector 1 has a date-based change right now
    if station_id in _DELTA_STATIONS and int(connector_id) == 1:
        dt = _to_datetime(session_start_utc)
        if dt and ZoneInfo:
            ak = dt.astimezone(ZoneInfo(_AK_TZ_NAME))
            if ak.date() < _DELTA_CONN1_CUTOFF_AK_DATE:
                return "CHAdeMO"
            return "NACS"

    return current

# Map station_id → Tritium platform (used for vendor error code enrichment).
# Leave non‑Tritium units as "" so they won't join to the Tritium codes table.
PLATFORM_MAP = {
    # Delta Junction (Autel)
    "as_oXoa7HXphUu5riXsSW253": "MaxiCharger",  # Delta — Right
    "as_xTUHfTKoOvKSfYZhhdlhT": "MaxiCharger",  # Delta — Left

    # Anchorage (ARG — Tritium RTM)
    "as_c8rCuPHDd7sV1ynHBVBiq": "RTM",  # ARG — Right
    "as_cnIGqQ0DoWdFCo7zSrN01": "RTM",  # ARG — Left

    # Non‑Tritium (no enrichment)
    "as_LYHe6mZTRKiFfziSNJFvJ": "MaxiCharger",
}

# --- add near the bottom of constants.py ---
from pathlib import Path
import json

_OVR = Path(__file__).with_name("runtime_overrides.json")

def _ovr() -> dict:
    try:
        return json.loads(_OVR.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_chargers_from_db() -> list:
    """Query public.chargers JOIN public.sites for EVSEs not in the hardcoded map.

    Used to pick up newly-registered EVSEs without a code deploy.
    Returns an empty list on any error (DB unavailable, missing table, etc.).
    """
    known = set(EVSE_DISPLAY.keys())
    try:
        from .db import get_conn
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.external_id,
                       c.name,
                       s.name  AS site_name,
                       c.make,
                       c.connector_types
                FROM   public.chargers c
                LEFT   JOIN public.sites s ON c.site_id = s.id
                WHERE  c.external_id IS NOT NULL
            """)
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "external_id":     r[0],
                "name":            r[1] or "",
                "site_name":       r[2] or "",
                "make":            r[3] or "",
                "connector_types": r[4] or {},
            }
            for r in rows
            if r[0] not in known
        ]
    except Exception:
        return []


def get_evse_display() -> dict:
    base = dict(EVSE_DISPLAY)
    for r in _load_chargers_from_db():
        if r["external_id"] not in base and r["name"]:
            base[r["external_id"]] = r["name"]
    ov = _ovr().get("evse_display", {})
    base.update(ov)
    return base

def get_platform_map() -> dict:
    base = dict(PLATFORM_MAP) if "PLATFORM_MAP" in globals() else {}
    for r in _load_chargers_from_db():
        if r["external_id"] not in base and r["make"]:
            base[r["external_id"]] = r["make"]
    ov = _ovr().get("platform_map", {})
    base.update(ov)
    return base

def get_archived_station_ids() -> list[str]:
    return list(_ovr().get("archived_station_ids", []))

def get_evse_location() -> dict:
    base = dict(EVSE_LOCATION) if "EVSE_LOCATION" in globals() else {}
    for r in _load_chargers_from_db():
        if r["external_id"] not in base and r["site_name"]:
            base[r["external_id"]] = r["site_name"]
    ov = _ovr().get("evse_location", {})
    base.update(ov)
    return base


def get_connector_type() -> dict:
    """
    Return CONNECTOR_TYPE merged with DB connector_types and runtime overrides.

    Priority (lowest → highest):
      1. Hardcoded CONNECTOR_TYPE (existing well-known EVSEs)
      2. DB chargers.connector_types JSONB (new EVSEs registered via admin)
      3. runtime_overrides.json connector_type (emergency manual overrides)
    """
    base = dict(CONNECTOR_TYPE) if "CONNECTOR_TYPE" in globals() else {}
    # Layer: DB connector_types JSONB (new EVSEs registered via admin)
    for r in _load_chargers_from_db():
        sid = r["external_id"]
        for cid_str, ctype in (r.get("connector_types") or {}).items():
            try:
                key = (sid, int(cid_str))
                if key not in base:
                    base[key] = ctype
            except (ValueError, TypeError):
                continue
    # Layer: runtime_overrides (manual overrides, highest priority)
    ov = _ovr().get("connector_type", {})
    try:
        base.update(ov)
    except Exception:
        # Be defensive in case overrides are stringified, e.g. "('as_x', 1)"
        for k, v in ov.items():
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                try:
                    sid, cid = k.strip("()").split(",", 1)
                    sid = sid.strip().strip("'").strip('"')
                    cid = int(cid)
                    base[(sid, cid)] = v
                except Exception:
                    continue
    return base


def get_all_station_ids() -> list:
    """
    Convenience: merged set of all known station_ids from display/location/platform maps.
    """
    s = set(get_evse_display().keys()) | set(get_evse_location().keys()) | set(get_platform_map().keys())
    return sorted(s)


def display_name(station_id: str) -> str:
    """
    Convenience accessor for a single EVSE display name.
    """
    return get_evse_display().get(station_id, station_id)


def location_label(station_id: str) -> str:
    """
    Convenience accessor for a single EVSE location label.
    """
    return get_evse_location().get(station_id, "")