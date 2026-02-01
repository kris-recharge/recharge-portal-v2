EVSE_DISPLAY = {
    "as_c8rCuPHDd7sV1ynHBVBiq": "ARG - Right",
    "as_cnIGqQ0DoWdFCo7zSrN01": "ARG - Left",
    "as_oXoa7HXphUu5riXsSW253": "Delta - Right",
    "as_xTUHfTKoOvKSfYZhhdlhT": "Delta - Left",
    "as_LYHe6mZTRKiFfziSNJFvJ": "Autel Maxi",
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
    # Delta Junction (Tritium)
    "as_oXoa7HXphUu5riXsSW253": "RT50",  # Delta — Right
    "as_xTUHfTKoOvKSfYZhhdlhT": "RT50",  # Delta — Left

    # Anchorage (ARG — Tritium RTM)
    "as_c8rCuPHDd7sV1ynHBVBiq": "RTM",  # ARG — Right
    "as_cnIGqQ0DoWdFCo7zSrN01": "RTM",  # ARG — Left

    # Non‑Tritium (no enrichment)
    "as_LYHe6mZTRKiFfziSNJFvJ": "",
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

def get_evse_display() -> dict:
    ov = _ovr().get("evse_display", {})
    base = dict(EVSE_DISPLAY)  # existing constant
    base.update(ov)
    return base

def get_platform_map() -> dict:
    ov = _ovr().get("platform_map", {})
    base = dict(PLATFORM_MAP) if "PLATFORM_MAP" in globals() else {}
    base.update(ov)
    return base

def get_archived_station_ids() -> list[str]:
    return list(_ovr().get("archived_station_ids", []))

def get_evse_location() -> dict:
    ov = _ovr().get("evse_location", {})
    base = dict(EVSE_LOCATION) if "EVSE_LOCATION" in globals() else {}
    base.update(ov)
    return base


def get_connector_type() -> dict:
    """
    Return CONNECTOR_TYPE with any runtime overrides merged in.

    Overrides schema (runtime_overrides.json):
      {
        "connector_type": {
          "(station_id, connector_id)": "CCS"
        }
      }

    NOTE: For practicality we expect overrides to use the same native
    mapping structure as CONNECTOR_TYPE, i.e. keys are tuples
    (station_id, connector_id). If the JSON uses string keys, your
    admin tab can normalize them before writing.
    """
    ov = _ovr().get("connector_type", {})
    base = dict(CONNECTOR_TYPE) if "CONNECTOR_TYPE" in globals() else {}
    # Merge; if ov contains tuple-like string keys this won't match —
    # we keep the simple case here; admin writer should store tuples.
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