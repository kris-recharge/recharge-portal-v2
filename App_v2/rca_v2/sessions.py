import numpy as np
import pandas as pd
from .config import AK_TZ
from .constants import get_evse_display, CONNECTOR_TYPE

# Friendly EVSE display names (ARG - Left, etc.)
EVSE_DISPLAY = get_evse_display()

# Helper: format datetime in Alaska time zone safely
def _fmt_ak(dt, fmt="%Y-%m-%d %H:%M"):
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    try:
        return ts.tz_convert(AK_TZ).strftime(fmt)
    except Exception:
        try:
            return ts.tz_localize("UTC").tz_convert(AK_TZ).strftime(fmt)
        except Exception:
            return ""

def first_nonzero(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    nz = s[s > 0]
    return (nz.iloc[0] if not nz.empty else (s.iloc[0] if not s.empty else None))

def build_sessions(df, auth):
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Normalize timestamps to UTC and sort
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["station_id","connector_id","transaction_id","timestamp"])
    groups = df.groupby(["station_id","connector_id","transaction_id"], dropna=True, sort=False)

    rows, starts, durs = [], [], []
    if auth is not None and not auth.empty:
        auth = auth.copy()
        auth["timestamp"] = pd.to_datetime(auth["timestamp"], utc=True, errors="coerce")
        auth = auth.sort_values(["station_id","timestamp"])

    for (sid, conn_id, tx), g in groups:
        if not isinstance(tx, str) and pd.isna(tx):
            continue
        ts_start = pd.to_datetime(g["timestamp"].min(), utc=True, errors="coerce")
        ts_end   = pd.to_datetime(g["timestamp"].max(), utc=True, errors="coerce")
        # Guard: skip sessions without valid times
        if pd.isna(ts_start) or pd.isna(ts_end):
            continue

        # Append for heatmap (UTC)
        try:
            dur_val = (ts_end - ts_start).total_seconds() / 60.0
        except Exception:
            dur_val = np.nan
        starts.append(ts_start)
        durs.append(dur_val)

        # Robust numerics
        pmax = pd.to_numeric(g["power_w"], errors="coerce").max()
        max_kw = pmax / 1000.0 if pd.notna(pmax) else np.nan

        e_min = pd.to_numeric(g["energy_wh"], errors="coerce").min()
        e_max = pd.to_numeric(g["energy_wh"], errors="coerce").max()
        e_kwh = (e_max - e_min)/1000.0 if pd.notna(e_min) and pd.notna(e_max) and e_max >= e_min else np.nan

        soc_start = first_nonzero(g["soc"])
        soc_end_series = pd.to_numeric(g["soc"], errors="coerce").dropna()
        soc_end = (soc_end_series.iloc[-1] if not soc_end_series.empty else None)

        # Find a nearby authorization id_tag near the start
        id_tag = None
        if auth is not None and not auth.empty:
            a_sid = auth[auth["station_id"] == sid]
            if not a_sid.empty:
                mask = (a_sid["timestamp"] >= ts_start - pd.Timedelta(minutes=60)) & (a_sid["timestamp"] <= ts_start + pd.Timedelta(minutes=5))
                cand = a_sid.loc[mask]
                if not cand.empty:
                    idx = (cand["timestamp"] - ts_start).abs().sort_values().index
                    id_tag = cand.loc[idx[0], "id_tag"]

        # Connector handling
        conn_num = pd.to_numeric(conn_id, errors="coerce")
        conn_int = int(conn_num) if pd.notna(conn_num) else None

        row = {
            "Start Date/Time": _fmt_ak(ts_start),
            "End Date/Time":   _fmt_ak(ts_end),
            "EVSE": EVSE_DISPLAY.get(sid, "") or str(sid),
            "Connector #": conn_int,
            "Connector Type": CONNECTOR_TYPE.get((sid, conn_int), ""),
            "Max Power (kW)": round(max_kw, 2) if pd.notna(max_kw) else None,
            "Energy Delivered (kWh)": round(e_kwh, 2) if pd.notna(e_kwh) else None,
            "Duration (min)": round(dur_val, 2) if pd.notna(dur_val) else None,
            "SoC Start": int(round(soc_start)) if pd.notna(soc_start) else None,
            "SoC End":   int(round(soc_end)) if soc_end is not None else None,
            "ID Tag": id_tag or "",
            "station_id": sid,
            "transaction_id": tx,
            "connector_id": conn_int,
        }
        rows.append(row)

    sess = pd.DataFrame(rows)
    if not sess.empty:
        sess["_end_ts"] = pd.to_datetime(sess["End Date/Time"], errors="coerce")
        sess = sess.sort_values(["_end_ts","station_id","transaction_id"], ascending=[False, True, True], kind="mergesort")
        sess = sess.drop(columns=["_end_ts"])

    heat = pd.DataFrame({"start_ts": starts, "dur_min": durs}) if starts else pd.DataFrame(columns=["start_ts","dur_min"])
    return sess, heat