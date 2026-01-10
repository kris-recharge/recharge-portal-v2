import numpy as np
import pandas as pd

from .config import AK_TZ
from .constants import EVSE_LOCATION, CONNECTOR_TYPE


def _fmt_ak(dt, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a UTC timestamp into Alaska local time."""
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    try:
        return ts.tz_convert(AK_TZ).strftime(fmt)
    except Exception:
        # Fallback if ts lost tz info
        try:
            return ts.tz_localize("UTC").tz_convert(AK_TZ).strftime(fmt)
        except Exception:
            return ""


def _first_nonzero(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    nz = s[s > 0]
    return (nz.iloc[0] if not nz.empty else (s.iloc[0] if not s.empty else None))


def build_sessions(df: pd.DataFrame, auth: pd.DataFrame):
    """Build charging sessions and a simple heatmap frame.

    Expected df columns:
      - station_id, connector_id, transaction_id, timestamp
      - energy_wh (optional but enables kWh), power_w, soc, voltage_v, etc.

    Expected auth columns:
      - station_id, timestamp, id_tag
    """

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["start_ts", "dur_min"])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Ensure IDs exist
    for c in ["station_id", "connector_id", "transaction_id"]:
        if c not in df.columns:
            df[c] = None

    df = df.sort_values(["station_id", "connector_id", "transaction_id", "timestamp"], kind="mergesort")

    groups = df.groupby(["station_id", "connector_id", "transaction_id"], dropna=True, sort=False)

    rows = []
    starts = []
    durs = []

    auth_df = None
    if auth is not None and not auth.empty:
        auth_df = auth.copy()
        auth_df["timestamp"] = pd.to_datetime(auth_df["timestamp"], utc=True, errors="coerce")
        auth_df = auth_df.dropna(subset=["timestamp"]).sort_values(["station_id", "timestamp"], kind="mergesort")

    for (sid, conn_id, tx), g in groups:
        if tx is None or (not isinstance(tx, str) and pd.isna(tx)):
            continue

        ts_start = g["timestamp"].min()
        ts_end = g["timestamp"].max()
        if pd.isna(ts_start) or pd.isna(ts_end):
            continue

        dur_min = (ts_end - ts_start).total_seconds() / 60.0
        starts.append(ts_start)
        durs.append(dur_min)

        # Power and energy
        pmax = pd.to_numeric(g.get("power_w", pd.Series(dtype=float)), errors="coerce").max()
        max_kw = (pmax / 1000.0) if pd.notna(pmax) else np.nan

        e = pd.to_numeric(g.get("energy_wh", pd.Series(dtype=float)), errors="coerce")
        e_min = e.min() if not e.empty else np.nan
        e_max = e.max() if not e.empty else np.nan
        e_kwh = (e_max - e_min) / 1000.0 if pd.notna(e_min) and pd.notna(e_max) and e_max >= e_min else np.nan

        # SoC (optional)
        soc_start = _first_nonzero(g.get("soc", pd.Series(dtype=float)))
        soc_series = pd.to_numeric(g.get("soc", pd.Series(dtype=float)), errors="coerce").dropna()
        soc_end = soc_series.iloc[-1] if not soc_series.empty else None

        # Find an authorization close to start
        id_tag = ""
        if auth_df is not None:
            a_sid = auth_df[auth_df["station_id"] == sid]
            if not a_sid.empty:
                mask = (a_sid["timestamp"] >= ts_start - pd.Timedelta(minutes=60)) & (a_sid["timestamp"] <= ts_start + pd.Timedelta(minutes=5))
                cand = a_sid.loc[mask]
                if not cand.empty:
                    idx = (cand["timestamp"] - ts_start).abs().sort_values().index
                    id_tag = str(cand.loc[idx[0], "id_tag"])

        conn_num = pd.to_numeric(conn_id, errors="coerce")
        conn_int = int(conn_num) if pd.notna(conn_num) else None

        rows.append(
            {
                "Start Date/Time": _fmt_ak(ts_start),
                "End Date/Time": _fmt_ak(ts_end),
                "EVSE": EVSE_LOCATION.get(sid, ""),
                "Connector #": conn_int,
                "Connector Type": CONNECTOR_TYPE.get((sid, conn_int), ""),
                "Max Power (kW)": round(max_kw, 2) if pd.notna(max_kw) else None,
                "Energy Delivered (kWh)": round(e_kwh, 2) if pd.notna(e_kwh) else None,
                "Duration (min)": round(dur_min, 2) if pd.notna(dur_min) else None,
                "SoC Start": int(round(soc_start)) if pd.notna(soc_start) else None,
                "SoC End": int(round(soc_end)) if soc_end is not None else None,
                "ID Tag": id_tag,
                "station_id": sid,
                "transaction_id": tx,
                "connector_id": conn_int,
            }
        )

    sess = pd.DataFrame(rows)
    if not sess.empty:
        # Sort newest-first by end time string (already AK formatted). Keep stable tie-breakers.
        sess = sess.sort_values(["End Date/Time", "station_id", "transaction_id"], ascending=[False, True, True], kind="mergesort")

    heat = pd.DataFrame({"start_ts": starts, "dur_min": durs}) if starts else pd.DataFrame(columns=["start_ts", "dur_min"])
    return sess, heat