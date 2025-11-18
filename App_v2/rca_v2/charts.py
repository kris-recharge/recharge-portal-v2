import plotly.graph_objects as go
import numpy as np
import pandas as pd
from .config import AK_TZ

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
        """
        Helper to add a line with a clean hover:
        - We rely on layout(hovermode="x unified") to show the timestamp once
        - Each trace hover only shows the value + units (no repeated time)
        """
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
                hovertemplate=hover_fmt + "<extra></extra>",
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
    """Session start counts per day/hour with white→blue cells, black borders and labels."""
    if heat is None or heat.empty:
        return go.Figure()

    h = heat.copy()
    # Normalize time → AK local then derive day/hour bins
    h["start_local"] = h["start_ts"].dt.tz_convert(AK_TZ)
    h["dow"] = h["start_local"].dt.dayofweek
    h["hour"] = h["start_local"].dt.hour

    # 7×24 matrix, fill missing with 0 so we always render a full grid
    mat = (
        h.groupby(["dow", "hour"]).size().unstack(fill_value=0)
        .reindex(index=[6, 0, 1, 2, 3, 4, 5], fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    mat.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    # Ensure numeric array (avoid object dtype rendering issues). Also cap zmax
    z = mat.to_numpy(dtype=float)
    # Guard against the all‑zero window (plotly auto with zmax=0 -> weird/blank).  
    zmax = float(np.nanpercentile(z, 95)) if np.nanmax(z) > 0 else 1.0

    # Explicit white→blue ramp so zero really looks white on dark background
    blues = [[0.0, "#ffffff"], [1.0, "#08519c"]]

    heatmap = go.Heatmap(
        z=z.tolist(),
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=blues,
        zmin=0,
        zmax=zmax,
        colorbar=dict(title="Starts"),
        hoverongaps=False,
        # Black borders via gaps (background shows through in Streamlit dark theme)
        xgap=1,
        ygap=1,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Starts: %{z:.0f}<extra></extra>",
    )

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(title="Hour (0-23)", type="category"),
        yaxis=dict(title="Day", type="category", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Overlay numeric labels so they render across Plotly versions
    xs, ys, labels = [], [], []
    cols = list(mat.columns)
    rows = list(mat.index)
    for yi, row in enumerate(rows):
        for xi, col in enumerate(cols):
            val = z[yi, xi]
            if np.isfinite(val) and val != 0:
                xs.append(col)
                ys.append(row)
                labels.append(f"{int(round(val))}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="text", text=labels,
        textposition="middle center",
        textfont=dict(color="black", size=12),
        hoverinfo="skip", showlegend=False
    ))
    return fig

def heatmap_duration(heat, title):
    """Average session duration (minutes) per day/hour with white→blue cells, black borders and labels."""
    if heat is None or heat.empty:
        return go.Figure()

    h = heat.copy()
    h["start_local"] = h["start_ts"].dt.tz_convert(AK_TZ)
    h["dow"] = h["start_local"].dt.dayofweek
    h["hour"] = h["start_local"].dt.hour

    mat = (
        h.groupby(["dow", "hour"])  # mean minutes per bin
         ["dur_min"].mean().unstack(fill_value=0.0)
         .reindex(index=[6, 0, 1, 2, 3, 4, 5], fill_value=0.0)
         .reindex(columns=range(24), fill_value=0.0)
    )
    mat.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    z = mat.to_numpy(dtype=float)
    # Robust color range (avoid all‑black/blank when values are tiny or all zeros)
    zmax = float(np.nanpercentile(z, 95)) if np.nanmax(z) > 0 else 1.0

    blues = [[0.0, "#ffffff"], [1.0, "#08519c"]]

    heatmap = go.Heatmap(
        z=z.tolist(),
        x=list(mat.columns),
        y=list(mat.index),
        colorscale=blues,
        zmin=0,
        zmax=zmax,
        colorbar=dict(title="Avg min"),
        hoverongaps=False,
        xgap=1,
        ygap=1,
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg min: %{z:.1f}<extra></extra>",
    )

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(title="Hour (0-23)", type="category"),
        yaxis=dict(title="Day", type="category", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Overlay numeric labels (one decimal) for reliability across Plotly versions
    xs, ys, labels = [], [], []
    cols = list(mat.columns)
    rows = list(mat.index)
    for yi, row in enumerate(rows):
        for xi, col in enumerate(cols):
            val = z[yi, xi]
            if np.isfinite(val) and val != 0:
                xs.append(col)
                ys.append(row)
                labels.append(f"{val:.1f}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="text", text=labels,
        textposition="middle center",
        textfont=dict(color="black", size=12),
        hoverinfo="skip", showlegend=False
    ))
    return fig