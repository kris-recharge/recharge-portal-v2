import numpy as np
import pandas as pd

from .config import AK_TZ
from .constants import EVSE_LOCATION, CONNECTOR_TYPE


def _to_num(x):
    return pd.to_numeric(x, errors="coerce")


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


def _normalize_pricing_df(pricing: pd.DataFrame) -> pd.DataFrame:
    """Normalize pricing df to a known schema.

    Expected (any reasonable aliases accepted):
      - station_id
      - effective_start (timestamptz)
      - effective_end (timestamptz, nullable)
      - connection_fee
      - price_per_kwh
      - price_per_min

    Optional (if present, will be used when the session df includes idle minutes):
      - idle_fee_per_min
      - idle_grace_min
    """
    p = pricing.copy()

    # Column aliases
    colmap = {}
    for c in p.columns:
        lc = str(c).strip().lower()
        if lc in {"station_id", "evse_id", "asset_id"}:
            colmap[c] = "station_id"
        elif lc in {"effective_start", "start_ts", "start_time", "pricing_start", "valid_from"}:
            colmap[c] = "effective_start"
        elif lc in {"effective_end", "end_ts", "end_time", "pricing_end", "valid_to"}:
            colmap[c] = "effective_end"
        elif lc in {"connection_fee", "flat_fee", "start_fee"}:
            colmap[c] = "connection_fee"
        elif lc in {"price_per_kwh", "kwh_price", "kwh_rate", "energy_rate"}:
            colmap[c] = "price_per_kwh"
        elif lc in {"price_per_min", "minute_price", "min_price", "time_rate"}:
            colmap[c] = "price_per_min"
        elif lc in {"idle_fee_per_min", "idle_fee", "idle_price", "idle_rate", "idle_per_min", "idle_min_price"}:
            colmap[c] = "idle_fee_per_min"
        elif lc in {"idle_grace_min", "idle_grace_minutes", "grace_minutes", "idle_grace", "idle_free_minutes"}:
            colmap[c] = "idle_grace_min"

    if colmap:
        p = p.rename(columns=colmap)

    # Required columns
    for req in ["station_id", "effective_start"]:
        if req not in p.columns:
            return pd.DataFrame(columns=[
                "station_id",
                "effective_start",
                "effective_end",
                "connection_fee",
                "price_per_kwh",
                "price_per_min",
                "idle_fee_per_min",
                "idle_grace_min",
            ])

    if "effective_end" not in p.columns:
        p["effective_end"] = pd.NaT
    if "idle_fee_per_min" not in p.columns:
        p["idle_fee_per_min"] = 0
    if "idle_grace_min" not in p.columns:
        p["idle_grace_min"] = 0

    # Parse timestamps
    p["effective_start"] = pd.to_datetime(p["effective_start"], utc=True, errors="coerce")
    p["effective_end"] = pd.to_datetime(p["effective_end"], utc=True, errors="coerce")

    # Numeric fields (default 0)
    for ncol in ["connection_fee", "price_per_kwh", "price_per_min", "idle_fee_per_min", "idle_grace_min"]:
        if ncol not in p.columns:
            p[ncol] = 0
        p[ncol] = _to_num(p[ncol]).fillna(0)

    p["station_id"] = p["station_id"].astype(str)
    p = p.dropna(subset=["effective_start"])  # must have start

    # Sort so we can pick "latest start" easily
    p = p.sort_values(["station_id", "effective_start"], kind="mergesort")
    return p


def _estimate_revenue_for_sessions(sess: pd.DataFrame, pricing: pd.DataFrame) -> pd.Series:
    """Return a float Series of estimated revenue for each session row."""
    if sess is None or sess.empty or pricing is None or pricing.empty:
        return pd.Series([np.nan] * (0 if sess is None else len(sess)))

    p = _normalize_pricing_df(pricing)
    if p.empty:
        return pd.Series([np.nan] * len(sess), index=sess.index)

    # We rely on internal UTC start timestamps when present.
    ts = sess.get("_start_ts_utc")
    if ts is None:
        return pd.Series([np.nan] * len(sess), index=sess.index)

    ts = pd.to_datetime(ts, utc=True, errors="coerce")

    kwh = _to_num(sess.get("Energy Delivered (kWh)")).fillna(0)
    mins = _to_num(sess.get("Duration (min)")).fillna(0)
    sid = sess.get("station_id").astype(str)
    # Optional idle minutes (only used if session df includes it)
    idle_cols = [
        "Idle Duration (min)",
        "Idle (min)",
        "Idle Time (min)",
        "Idle Minutes",
    ]
    idle_mins = None
    for c in idle_cols:
        if c in sess.columns:
            idle_mins = _to_num(sess.get(c)).fillna(0)
            break
    if idle_mins is None:
        idle_mins = pd.Series([0.0] * len(sess), index=sess.index)

    out = []
    for i in sess.index:
        s_id = sid.loc[i]
        t0 = ts.loc[i]
        if pd.isna(t0):
            out.append(np.nan)
            continue

        # Find pricing rows for this station where start <= t0 < end (or end is null)
        p_sid = p[p["station_id"] == s_id]
        if p_sid.empty:
            out.append(np.nan)
            continue

        mask = (p_sid["effective_start"] <= t0) & (
            p_sid["effective_end"].isna() | (t0 < p_sid["effective_end"])
        )
        cand = p_sid.loc[mask]
        if cand.empty:
            # If nothing matches the window, fall back to the latest pricing before t0
            cand = p_sid[p_sid["effective_start"] <= t0]

        if cand.empty:
            out.append(np.nan)
            continue

        row = cand.iloc[-1]  # latest effective_start
        idle_fee_per_min = float(row.get("idle_fee_per_min", 0))
        idle_grace_min = float(row.get("idle_grace_min", 0))
        idle_billable = max(0.0, float(idle_mins.loc[i]) - idle_grace_min)

        rev = (
            float(row.get("connection_fee", 0))
            + float(kwh.loc[i]) * float(row.get("price_per_kwh", 0))
            + float(mins.loc[i]) * float(row.get("price_per_min", 0))
            + idle_billable * idle_fee_per_min
        )
        out.append(round(rev, 2))

    return pd.Series(out, index=sess.index)


def _is_startlike(row: pd.Series) -> bool:
    # StartTransaction fallback: connector_id known, tx missing, energy often 0
    tx = row.get("transaction_id")
    conn = row.get("connector_id")
    if conn is None or (not isinstance(conn, str) and pd.isna(conn)):
        return False
    if tx is not None and not (not isinstance(tx, str) and pd.isna(tx)):
        return False
    return True


def _is_stoplike(row: pd.Series) -> bool:
    # StopTransaction fallback: tx present, connector_id often missing, energy often >0
    tx = row.get("transaction_id")
    if tx is None or (not isinstance(tx, str) and pd.isna(tx)):
        return False
    # If it only has one record in the whole tx and no power/soc, it is likely stop-only
    return True


def _build_sessions_from_start_stop(df: pd.DataFrame, auth_df: pd.DataFrame | None, pricing_df: pd.DataFrame | None = None):
    """Pair Start/StopTransaction fallback rows into sessions.

    Works when Start rows have connector_id but no transaction_id and Stop rows have transaction_id but no connector_id.
    Pairing rule:
      - Track open starts per station (and connector when available).
      - On stop: close the most recent open start for that station (prefer same connector if stop provides it).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["start_ts", "dur_min"])

    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"])

    # Normalize ids
    for c in ["station_id", "connector_id", "transaction_id"]:
        if c not in d.columns:
            d[c] = None

    # Ensure connector_id numeric where possible
    d["connector_id_num"] = pd.to_numeric(d["connector_id"], errors="coerce")

    # Energy as numeric
    if "energy_wh" in d.columns:
        d["energy_wh_num"] = pd.to_numeric(d["energy_wh"], errors="coerce")
    else:
        d["energy_wh_num"] = np.nan

    d = d.sort_values(["station_id", "timestamp"], kind="mergesort")

    rows = []
    starts = []
    durs = []

    # open starts: station_id -> list of dicts (stack) to support multiple connectors
    open_starts: dict[str, list[dict]] = {}

    def _pick_auth(sid: str, ts_start: pd.Timestamp) -> str:
        if auth_df is None or auth_df.empty:
            return ""
        a_sid = auth_df[auth_df["station_id"] == sid]
        if a_sid.empty:
            return ""
        mask = (a_sid["timestamp"] >= ts_start - pd.Timedelta(minutes=60)) & (a_sid["timestamp"] <= ts_start + pd.Timedelta(minutes=5))
        cand = a_sid.loc[mask]
        if cand.empty:
            return ""
        idx = (cand["timestamp"] - ts_start).abs().sort_values().index
        return str(cand.loc[idx[0], "id_tag"])

    for _, r in d.iterrows():
        sid = r.get("station_id")
        if sid is None or (not isinstance(sid, str) and pd.isna(sid)):
            continue

        if _is_startlike(r):
            conn_int = None
            cn = r.get("connector_id_num")
            if pd.notna(cn):
                conn_int = int(cn)
            open_starts.setdefault(str(sid), []).append(
                {
                    "ts": r["timestamp"],
                    "connector_id": conn_int,
                    "energy_wh": r.get("energy_wh_num"),
                    "row": r,
                }
            )
            continue

        if _is_stoplike(r):
            stack = open_starts.get(str(sid), [])
            if not stack:
                continue

            # If stop provides connector_id, prefer the most recent start on that connector
            stop_conn = None
            cn = r.get("connector_id_num")
            if pd.notna(cn):
                stop_conn = int(cn)

            pick_idx = None
            if stop_conn is not None:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i].get("connector_id") == stop_conn:
                        pick_idx = i
                        break
            if pick_idx is None:
                pick_idx = len(stack) - 1

            start_rec = stack.pop(pick_idx)
            if not stack:
                open_starts.pop(str(sid), None)

            ts_start = start_rec["ts"]
            ts_end = r["timestamp"]
            if pd.isna(ts_start) or pd.isna(ts_end) or ts_end < ts_start:
                continue

            dur_min = (ts_end - ts_start).total_seconds() / 60.0
            starts.append(ts_start)
            durs.append(dur_min)

            # Energy: stop - start (start often 0)
            e_start = start_rec.get("energy_wh")
            e_stop = r.get("energy_wh_num")
            e_kwh = np.nan
            if pd.notna(e_stop):
                if pd.notna(e_start) and e_stop >= e_start:
                    e_kwh = (e_stop - e_start) / 1000.0
                else:
                    # If start is missing/NaN, treat stop as delivered since 0
                    e_kwh = e_stop / 1000.0

            tx = r.get("transaction_id")
            tx = str(tx) if tx is not None and not (not isinstance(tx, str) and pd.isna(tx)) else ""

            conn_int = start_rec.get("connector_id") if stop_conn is None else stop_conn

            id_tag = _pick_auth(str(sid), ts_start)

            rows.append(
                {
                    "Start Date/Time": _fmt_ak(ts_start),
                    "End Date/Time": _fmt_ak(ts_end),
                    "EVSE": EVSE_LOCATION.get(str(sid), ""),
                    "Connector #": conn_int,
                    "Connector Type": CONNECTOR_TYPE.get((str(sid), conn_int), ""),
                    "Max Power (kW)": None,
                    "Energy Delivered (kWh)": round(e_kwh, 2) if pd.notna(e_kwh) else None,
                    "Duration (min)": round(dur_min, 2) if pd.notna(dur_min) else None,
                    "SoC Start": None,
                    "SoC End": None,
                    "ID Tag": id_tag,
                    "station_id": str(sid),
                    "transaction_id": tx,
                    "connector_id": conn_int,
                    "_start_ts_utc": ts_start,
                }
            )

    sess = pd.DataFrame(rows)
    if pricing_df is not None and not sess.empty:
        sess["Estimated Revenue ($)"] = _estimate_revenue_for_sessions(sess, pricing_df)
        # Keep a consistent, human-friendly column order
        col_order = [
            "Start Date/Time",
            "End Date/Time",
            "EVSE",
            "Connector #",
            "Connector Type",
            "Max Power (kW)",
            "Energy Delivered (kWh)",
            "Duration (min)",
            "Estimated Revenue ($)",
            "SoC Start",
            "SoC End",
            "ID Tag",
        ]
        keep = [c for c in col_order if c in sess.columns]
        rest = [c for c in sess.columns if c not in keep and not c.startswith("_")]
        sess = sess[keep + rest + [c for c in sess.columns if c.startswith("_")]]

    if not sess.empty:
        sess = sess.sort_values(["End Date/Time", "station_id", "transaction_id"], ascending=[False, True, True], kind="mergesort")

    heat = pd.DataFrame({"start_ts": starts, "dur_min": durs}) if starts else pd.DataFrame(columns=["start_ts", "dur_min"])

    if "_start_ts_utc" in sess.columns:
        sess = sess.drop(columns=["_start_ts_utc"])

    return sess, heat


def build_sessions(df: pd.DataFrame, auth: pd.DataFrame, pricing: pd.DataFrame | None = None):
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

    auth_df = None
    if auth is not None and not auth.empty:
        auth_df = auth.copy()
        auth_df["timestamp"] = pd.to_datetime(auth_df["timestamp"], utc=True, errors="coerce")
        auth_df = auth_df.dropna(subset=["timestamp"]).sort_values(["station_id", "timestamp"], kind="mergesort")

    groups = df.groupby(["station_id", "connector_id", "transaction_id"], dropna=False, sort=False)

    rows = []
    starts = []
    durs = []

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
                "_start_ts_utc": ts_start,
            }
        )

    sess = pd.DataFrame(rows)
    if pricing is not None and not sess.empty:
        sess["Estimated Revenue ($)"] = _estimate_revenue_for_sessions(sess, pricing)
        # Keep a consistent, human-friendly column order
        col_order = [
            "Start Date/Time",
            "End Date/Time",
            "EVSE",
            "Connector #",
            "Connector Type",
            "Max Power (kW)",
            "Energy Delivered (kWh)",
            "Duration (min)",
            "Estimated Revenue ($)",
            "SoC Start",
            "SoC End",
            "ID Tag",
        ]
        keep = [c for c in col_order if c in sess.columns]
        rest = [c for c in sess.columns if c not in keep and not c.startswith("_")]
        sess = sess[keep + rest + [c for c in sess.columns if c.startswith("_")]]

    heat = pd.DataFrame({"start_ts": starts, "dur_min": durs}) if starts else pd.DataFrame(columns=["start_ts", "dur_min"])

    # If we are operating on Supabase Start/StopTransaction fallback rows, the transaction_id
    # grouping path can produce "stop-only" sessions (single row per tx) with 0 duration and 0 kWh.
    # In that case (and also when sess is empty), prefer the Start/Stop pairing fallback.
    has_start_stop = False
    if "action" in df.columns:
        actions = set(df["action"].dropna().unique().tolist())
        has_start_stop = actions.issubset({"StartTransaction", "StopTransaction"}) and ("StartTransaction" in actions) and ("StopTransaction" in actions)

    def _is_stop_only_sessions(s: pd.DataFrame) -> bool:
        if s is None or s.empty:
            return True
        # Coerce numeric columns if present
        dur = pd.to_numeric(s.get("Duration (min)", pd.Series(dtype=float)), errors="coerce")
        kwh = pd.to_numeric(s.get("Energy Delivered (kWh)", pd.Series(dtype=float)), errors="coerce")
        # If all durations are 0/NaN and all kWh are 0/NaN, these are not usable sessions
        dur_ok = dur.fillna(0) > 0
        kwh_ok = kwh.fillna(0) > 0
        return not (dur_ok.any() or kwh_ok.any())

    if sess.empty or (has_start_stop and _is_stop_only_sessions(sess)):
        sess2, heat2 = _build_sessions_from_start_stop(df, auth_df, pricing_df=pricing)
        # If fallback produced something, return it. Otherwise fall back to tx grouping output.
        if sess2 is not None and not sess2.empty:
            return sess2, heat2

    if sess is not None and not sess.empty and "_start_ts_utc" in sess.columns:
        sess = sess.drop(columns=["_start_ts_utc"])

    return sess, heat