import plotly.graph_objects as go
import numpy as np
import pandas as pd

from .config import AK_TZ

# --- Helpers for heatmaps: tolerant timestamp/duration discovery ---
def _to_ak(series, ak_hint: bool = False) -> pd.Series:
    """
    Convert any datetime-like series to Alaska tz.
    - If tz-aware, convert to AK_TZ.
    - If tz-naive: treat *_ak as already in AK; otherwise assume UTC then convert.
    """
    s = pd.to_datetime(series, errors="coerce", utc=False)
    if pd.api.types.is_datetime64tz_dtype(s):
        return s.dt.tz_convert(AK_TZ)
    try:
        if ak_hint:
            return s.dt.tz_localize(AK_TZ)
        else:
            return s.dt.tz_localize("UTC").dt.tz_convert(AK_TZ)
    except Exception:
        # Last-resort: parse as UTC then convert
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(AK_TZ)

def _start_local_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a single Alaska-local timestamp series from any of the known columns.
    Prefers *_ak columns, then UTC-like ones.
    """
    candidates = [
        ("_start_ak", True),
        ("start_ak", True),
        ("start_local", True),
        ("start_ts", False),
        ("timestamp", False),
        ("_start", False),
        ("start", False),
    ]
    for name, ak_hint in candidates:
        if name in df.columns:
            return _to_ak(df[name], ak_hint)
    # No viable column
    return pd.Series(dtype="datetime64[ns]")

def _duration_minutes_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a duration (minutes) series from any known name; coerces to numeric.
    Missing -> empty float series.
    """
    for name in ["dur_min", "Duration (min)", "duration_min", "duration_minutes", "duration", "dur"]:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(index=df.index, dtype=float)

def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def session_detail_figure(mv, sid, tx):
    """
    Build the Charge Session Details figure from raw MeterValues without fabricating data.
    - Filters by EVSE (station) and transaction id when present
    - Parses UTC timestamps and converts to Alaska time
    - Aligns by timestamp index to avoid any accidental reindexing
    - Converts Power W→kW and Energy Wh→kWh where appropriate
    - Baselines energy to session start (so the curve starts at 0)
    - Plots kW & A on left axis; SoC/energy/volts on right axis
    """
    fig = go.Figure()
    if mv is None or mv.empty:
        return fig

    df = mv.copy()

    # -------------------------
    # Filter by EVSE + tx id
    # -------------------------
    sid_col = _first_col(df, ["station_id", "evse_id", "station", "EVSE"])
    if sid_col:
        df = df[df[sid_col].astype(str) == str(sid)]

    tx_col = _first_col(df, ["transaction_id", "tx_id", "transaction", "transactionId"])
    if tx_col and pd.notna(tx):
        df = df[df[tx_col].astype(str) == str(tx)]

    if df.empty:
        return fig

    # -------------------------
    # Timestamps → AK local
    # -------------------------
    ts_col = _first_col(df, ["timestamp", "ts", "time"])
    if ts_col is None:
        return fig

    # Parse to UTC (handles tz‑naive by assuming UTC), then convert to AK
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    ts_ak = ts.dt.tz_convert(AK_TZ)

    # Stable, time‑indexed frame for aligned numeric extraction
    df = df.assign(_t=ts_ak).sort_values("_t")
    dti = df["_t"]
    f = df.set_index("_t")
    # If duplicates exist at exactly the same instant, keep the last reading
    f = f[~f.index.duplicated(keep="last")]

    # -------------------------
    # Column discovery (prefer local wide schema names first)
    # -------------------------
    if all(c in f.columns for c in ["power_w", "amperage_offered", "soc", "energy_wh", "voltage_v"]):
        power_col  = "power_w"
        amps_col   = "amperage_offered"
        soc_col    = "soc"
        energy_col = "energy_wh"
        hvb_col    = "voltage_v"
    else:
        power_col  = _first_col(f, ["power_kw", "Power (kW)", "kW", "active_power_kw", "power_w", "Power (W)"])
        amps_col   = _first_col(f, ["amps_offered", "Amps Offered", "current_a", "offered_current_a", "amperage_offered"]) 
        soc_col    = _first_col(f, ["soc_pct", "SoC (%)", "soc", "soc_percent"]) 
        hvb_col    = _first_col(f, ["hvb_volts", "HVB (V)", "voltage_v", "pack_volts", "Voltage (V)"]) 
        energy_col = _first_col(f, [
            "energy_kwh", "Energy (kWh)", "energy_wh", "Energy (Wh)",
            "Energy.Active.Import.Register", "Energy.Active.Import.Register (Wh)"
        ])

    def col_to_numeric(col_name):
        if not col_name:
            return pd.Series(index=f.index, dtype=float)
        return pd.to_numeric(f[col_name], errors="coerce")

    # Power (prefer kW; convert from W when needed)
    P = col_to_numeric(power_col)
    if power_col:
        low = power_col.lower()
        if power_col == "power_w" or ("power" in low and ("(w)" in low or low.endswith("_w"))) or (P.dropna().quantile(0.95) > 1000):
            P = P / 1000.0

    # Current, SoC, Voltage
    A   = col_to_numeric(amps_col)
    SOC = col_to_numeric(soc_col)
    V   = col_to_numeric(hvb_col)

    # Energy: convert to kWh if expressed in Wh; baseline to session start
    E = col_to_numeric(energy_col)
    if energy_col:
        low = energy_col.lower()
        if ("wh" in low and "kwh" not in low) or (E.dropna().quantile(0.95) > 1000):
            E = E / 1000.0
        first_valid = E.dropna().iloc[0] if E.dropna().size else pd.NA
        if pd.notna(first_valid):
            E = E - float(first_valid)
        E = E.mask(E < 0)

    # Build aligned numeric matrix (no synthetic interpolation)
    s = pd.DataFrame(index=f.index)
    s["P"]   = P
    s["A"]   = A
    s["SOC"] = SOC
    s["E"]   = E
    s["V"]   = V

    # Light forward‑fill only for slowly‑changing signals
    s["SOC"] = s["SOC"].ffill(limit=3)
    s["E"]   = s["E"].ffill(limit=3)

    # Drop rows where everything is NaN to prevent odd hover behaviour
    s = s.dropna(how="all")
    if s.empty:
        return fig

    def add_line(col, name, axis="y", hover_fmt="%{y}"):
        series = s[col].dropna()
        if series.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=name,
                mode="lines",
                yaxis=axis,
                hovertemplate=hover_fmt + "<br>%{x|%b %-d, %H:%M:%S}<extra></extra>",
            )
        )

    # Separate axes per series
    add_line("P",   "Power (kW)",   axis="y",  hover_fmt="%{y:.2f} kW")
    add_line("A",   "Amps Offered", axis="y2", hover_fmt="%{y:.0f} A")
    add_line("SOC", "SoC (%)",      axis="y3", hover_fmt="%{y:.0f} %")
    add_line("E",   "Energy (kWh)", axis="y4", hover_fmt="%{y:.2f} kWh")
    add_line("V",   "HVB (V)",      axis="y5", hover_fmt="%{y:.0f} V")

    fig.update_layout(
        margin=dict(l=10, r=200, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Time (AK)"),
        # y (left): Power
        yaxis=dict(title="Power (kW)", autorange=True, fixedrange=False),
        # y2..y5 (right): Amps, SoC, Energy, HVB
        yaxis2=dict(
            title="Amps (A)",
            overlaying="y",
            side="right",
            anchor="free",
            position=1.00,
            autorange=True,
            fixedrange=False,
            title_standoff=6,
        ),
        yaxis3=dict(
            title="SoC (%)",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.98,
            autorange=True,
            fixedrange=False,
            title_standoff=6,
        ),
        yaxis4=dict(
            title="Energy (kWh)",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.96,
            autorange=True,
            fixedrange=False,
            title_standoff=6,
        ),
        yaxis5=dict(
            title="HVB (V)",
            overlaying="y",
            side="right",
            anchor="free",
            position=0.94,
            autorange=True,
            fixedrange=False,
            title_standoff=6,
        ),
    )
    return fig

def heatmap_count(heat, title):
    """Session start counts per day/hour with white→blue cells, black borders and labels.
    Uses Plotly texttemplate instead of an overlay text trace for consistent rendering
    across Plotly versions (esp. on Render).
    """
    if heat is None or heat.empty:
        return go.Figure()

    h = heat.copy()

    # Normalize time → Alaska local robustly (accepts *_ak, UTC, or naive)
    s_local = _start_local_series(h)
    if s_local.empty or s_local.isna().all():
        return go.Figure()

    h["_start_local"] = s_local
    h["_dow"] = s_local.dt.dayofweek    # Mon=0 .. Sun=6
    h["_hour"] = s_local.dt.hour

    # 7×24 matrix (Sun..Sat x 0..23)
    mat = (
        h.groupby(["_dow", "_hour"]).size().unstack(fill_value=0)
         .reindex(index=[6, 0, 1, 2, 3, 4, 5], fill_value=0)
         .reindex(columns=range(24), fill_value=0)
    )
    mat.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    # Heatmap values and labels
    z = mat.to_numpy(dtype=float)
    text = [[(str(int(v)) if np.isfinite(v) and v > 0 else "") for v in row] for row in z]
    zmax = float(np.nanpercentile(z, 95)) if np.nanmax(z) > 0 else 1.0

    blues = [[0.0, "#ffffff"], [1.0, "#08519c"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=blues,
        zmin=0,
        zmax=zmax,
        colorbar=dict(title="Starts"),
        hoverongaps=False,
        xgap=1,
        ygap=1,
        text=text,
        texttemplate="%{text}",
        textfont={"color": "black", "size": 12},
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Starts: %{z:.0f}<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(title="Hour (0-23)", type="category"),
        yaxis=dict(title="Day", type="category"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def heatmap_duration(heat, title):
    """Average session duration (minutes) per day/hour with white→blue cells, borders and labels.
    Uses Plotly texttemplate instead of an overlay text trace for consistent rendering.
    """
    if heat is None or heat.empty:
        return go.Figure()

    h = heat.copy()

    s_local = _start_local_series(h)
    if s_local.empty or s_local.isna().all():
        return go.Figure()
    h["_start_local"] = s_local
    h["_dow"] = s_local.dt.dayofweek
    h["_hour"] = s_local.dt.hour

    # Duration series (minutes)
    h["_dur"] = _duration_minutes_series(h)

    mat = (
        h.groupby(["_dow", "_hour"])["_dur"].mean().unstack(fill_value=0.0)
         .reindex(index=[6, 0, 1, 2, 3, 4, 5], fill_value=0.0)
         .reindex(columns=range(24), fill_value=0.0)
    )
    mat.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    z = mat.to_numpy(dtype=float)
    text = [[(f"{v:.1f}" if np.isfinite(v) and v > 0 else "") for v in row] for row in z]
    zmax = float(np.nanpercentile(z, 95)) if np.nanmax(z) > 0 else 1.0

    blues = [[0.0, "#ffffff"], [1.0, "#08519c"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=blues,
        zmin=0,
        zmax=zmax,
        colorbar=dict(title="Avg min"),
        hoverongaps=False,
        xgap=1,
        ygap=1,
        text=text,
        texttemplate="%{text}",
        textfont={"color": "black", "size": 12},
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg min: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(title="Hour (0-23)", type="category"),
        yaxis=dict(title="Day", type="category"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig