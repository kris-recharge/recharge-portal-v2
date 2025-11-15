from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Day labels with Sunday at index 0 to match your UI
DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _to_ak(ts: pd.Series) -> pd.Series:
    """
    Coerce a timestamp-like Series to tz-aware UTC then convert to Alaska time.
    Returns a tz-aware Series.
    """
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    try:
        return s.dt.tz_convert("America/Anchorage")
    except Exception:
        # Fallback: return as-is (still tz-aware UTC) if conversion fails
        return s


def _blank_or_int(v):
    if pd.isna(v) or v == 0:
        return ""
    try:
        return str(int(v))
    except Exception:
        return ""


def _blank_or_float1(v):
    if pd.isna(v) or v == 0:
        return ""
    try:
        return f"{float(v):.1f}"
    except Exception:
        return ""


def _scatter_text_for_grid(grid: "pd.DataFrame", value_to_text) -> tuple[list[int], list[str], list[str]]:
    """
    Build x (hours), y (day labels), and text arrays for a scatter layer that writes numbers
    in the center of each heatmap cell. Empty strings are skipped.
    """
    xs: list[int] = []
    ys: list[str] = []
    ts: list[str] = []
    # grid index is 0..6 (Sun..Sat), columns 0..23
    for i, dow in enumerate(grid.index):
        for j, hour in enumerate(grid.columns):
            val = grid.iat[i, j]
            txt = value_to_text(val)
            if txt != "":
                xs.append(int(hour))
                ys.append(DAY_LABELS[int(dow)])
                ts.append(txt)
    return xs, ys, ts


def _heatmap_from_grid(
    grid: "pd.DataFrame",
    title: str,
    value_to_text,
    colorscale: str = "Blues",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Build a Plotly heatmap from a (7 x 24) DataFrame whose index is 0..6 (Sun..Sat)
    and columns are 0..23 (hours). We render the color map with a Heatmap trace
    and place the per-cell numbers using a separate Scatter(text) overlay so we
    avoid any Plotly/texttemplate inconsistencies across environments.
    """
    # Ensure a fixed 7x24 numeric grid
    grid = (
        grid.reindex(index=range(7), fill_value=0)
            .reindex(columns=range(24), fill_value=0)
            .copy()
    )
    z = grid.to_numpy(dtype=float, copy=True)

    # Heat layer
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(grid.columns),
            y=[DAY_LABELS[i] for i in grid.index],
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=None),
            xgap=1,
            ygap=1,
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Value: %{z:.1f}<extra></extra>",
            showscale=True,
        )
    )

    # Overlay text labels (only non-empty cells)
    xs, ys, ts = _scatter_text_for_grid(grid, value_to_text)
    if ts:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="text",
                text=ts,
                textposition="middle center",
                textfont=dict(color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Hour (0–23)", dtick=1),
        yaxis=dict(title="Day"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=30, b=40),
    )
    return fig


def heatmap_count(starts_df: pd.DataFrame, title: str = "Session Start Density (by Day & Hour)") -> go.Figure:
    """
    Expect a DataFrame with a single column 'start_ts' (tz-naive or tz-aware).
    Returns a Plotly heatmap where each cell is the number of starts for (day, hour).
    """
    if starts_df is None or starts_df.empty or "start_ts" not in starts_df.columns:
        empty = pd.DataFrame(0, index=range(7), columns=range(24))
        return _heatmap_from_grid(empty, title, _blank_or_int)

    ts_ak = _to_ak(starts_df["start_ts"])
    base = pd.DataFrame({
        "dow": (ts_ak.dt.dayofweek + 1) % 7,  # Sun=0..Sat=6
        "hour": ts_ak.dt.hour,
    }).dropna()

    if base.empty:
        empty = pd.DataFrame(0, index=range(7), columns=range(24))
        return _heatmap_from_grid(empty, title, _blank_or_int)

    grid = (
        base.groupby(["dow", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )

    cmax = float(np.nanmax(grid.to_numpy(dtype=float))) if grid.size else 0.0
    cmax = max(cmax, 1.0)

    return _heatmap_from_grid(grid, title, _blank_or_int, zmin=0, zmax=cmax)


def heatmap_duration(sessions_df: pd.DataFrame, title: str = "Average Session Duration (min)") -> go.Figure:
    """
    Expect a DataFrame with a start timestamp and a duration column.
    - Start column can be one of: 'start_ts', '_start', 'Start (UTC)', 'start_time'
    - Duration column can be one of: 'Duration (min)', 'duration_min', 'duration_mins', 'duration', 'dur_min'
    Cells with 0/NaN are left blank.
    """
    if sessions_df is None or sessions_df.empty:
        empty = pd.DataFrame(np.nan, index=range(7), columns=range(24))
        return _heatmap_from_grid(empty, title, _blank_or_float1)

    # Resolve columns
    ts_col = next((c for c in ["start_ts", "_start", "Start (UTC)", "start_time"] if c in sessions_df.columns), None)
    dur_col = next((c for c in ["Duration (min)", "duration_min", "duration_mins", "duration", "dur_min"] if c in sessions_df.columns), None)

    if ts_col is None or dur_col is None:
        empty = pd.DataFrame(np.nan, index=range(7), columns=range(24))
        return _heatmap_from_grid(empty, title, _blank_or_float1)

    ts_ak = _to_ak(sessions_df[ts_col])
    dur = pd.to_numeric(sessions_df[dur_col], errors="coerce")

    df = pd.DataFrame(
        {
            "dow": (ts_ak.dt.dayofweek + 1) % 7,  # Sun=0 .. Sat=6
            "hour": ts_ak.dt.hour,
            "dur": dur,
        }
    ).dropna(subset=["dur"])

    # Remove zeros/negatives from averaging
    df = df[df["dur"] > 0]

    # Clip colors at the 95th percentile (min 30, max 240 minutes for readability)
    zmax = float(np.nanpercentile(df["dur"], 95)) if not df.empty else 0.0
    zmax = float(np.clip(zmax, 30.0, 240.0)) if zmax else 60.0

    grid = df.groupby(["dow", "hour"])["dur"].mean().unstack()
    grid = grid.reindex(index=range(7), columns=range(24), fill_value=np.nan)
    return _heatmap_from_grid(grid, title, _blank_or_float1, zmin=0, zmax=zmax)


def session_detail_figure(*args, title: str = "Charge Session Details", **kwargs) -> go.Figure:
    """
    Compatibility wrapper for session details plotting.

    Supported call patterns:
      1) session_detail_figure(df)
      2) session_detail_figure(mv, selected_sid, selected_tx)
      3) session_detail_figure(mv=..., station_id=..., transaction_id=...)

    The dataframe rows used for plotting should include some combination of:
      - timestamp column: one of ["timestamp", "ts", "time", "_time", "start_ts", "Start (UTC)"]
      - power column: "power_w"
      - energy column: "energy_wh"

    The chart shows Power (kW) on the primary Y axis and Energy (kWh) on a secondary Y axis
    when available. If energy_wh is not present, a trapezoidal integral of power is used
    to approximate cumulative energy.
    """
    # ---- Parse inputs to obtain a dataframe of the selected session ----
    df: Optional[pd.DataFrame] = None

    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        # New style: just pass the filtered session frame
        df = args[0]
    elif len(args) >= 3 and isinstance(args[0], pd.DataFrame):
        # Legacy style: (mv, selected_sid, selected_tx)
        mv = args[0]
        selected_sid = args[1]
        selected_tx = args[2]
        try:
            df = mv[(mv["station_id"] == selected_sid) & (mv["transaction_id"] == selected_tx)].copy()
        except Exception:
            df = pd.DataFrame()
    else:
        # kwargs style
        if "df" in kwargs and isinstance(kwargs["df"], pd.DataFrame):
            df = kwargs["df"]
        elif {"mv", "station_id", "transaction_id"} <= set(kwargs):
            mv = kwargs.get("mv")
            selected_sid = kwargs.get("station_id")
            selected_tx = kwargs.get("transaction_id")
            try:
                df = mv[(mv["station_id"] == selected_sid) & (mv["transaction_id"] == selected_tx)].copy()
            except Exception:
                df = pd.DataFrame()

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Time (AK)",
        yaxis_title="kW / A / V / %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
    )

    if df is None or df.empty:
        return fig

    # ---- Detect timestamp column and convert to Alaska time for display ----
    tcol = next((c for c in ["timestamp", "ts", "time", "_time", "start_ts", "Start (UTC)"] if c in df.columns), None)
    if tcol is None:
        return fig

    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    try:
        t_ak = t.dt.tz_convert("America/Anchorage")
    except Exception:
        t_ak = t

    # ---- Add traces if available ----
    added = 0

    # Power (kW)
    if "power_w" in df.columns:
        y_kw = pd.to_numeric(df["power_w"], errors="coerce") / 1000.0
        fig.add_trace(
            go.Scatter(x=t_ak, y=y_kw, mode="lines", name="Power (kW)")
        )
        added += 1

    # Energy (kWh) or approximate from power
    if "energy_wh" in df.columns:
        y_kwh = pd.to_numeric(df["energy_wh"], errors="coerce") / 1000.0
        fig.add_trace(
            go.Scatter(x=t_ak, y=y_kwh, mode="lines", name="Energy (kWh)", yaxis="y2")
        )
        added += 1
    elif "power_w" in df.columns:
        # Approximate cumulative energy from power
        s = df[[tcol, "power_w"]].copy()
        s[tcol] = pd.to_datetime(s[tcol], errors="coerce", utc=True)
        s = s.sort_values(tcol)
        ts = (s[tcol].astype("int64") // 10**9).to_numpy()
        p_kw = pd.to_numeric(s["power_w"], errors="coerce").fillna(0).to_numpy() / 1000.0
        if len(p_kw) > 1:
            dt_h = np.diff(ts) / 3600.0
            e_kwh = np.concatenate([[0.0], np.cumsum(p_kw[:-1] * dt_h)])
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(s[tcol], utc=True).tz_convert("America/Anchorage"),
                    y=e_kwh,
                    mode="lines",
                    name="Energy (kWh)",
                    yaxis="y2",
                )
            )
            added += 1

    # Current (A)
    for col in ["amperage_import", "current_a"]:
        if col in df.columns:
            y_a = pd.to_numeric(df[col], errors="coerce")
            fig.add_trace(
                go.Scatter(x=t_ak, y=y_a, mode="lines", name="Amps (A)")
            )
            added += 1
            break

    # Offered current (A)
    for col in ["amperage_offered", "offered_current_a"]:
        if col in df.columns:
            y_oa = pd.to_numeric(df[col], errors="coerce")
            fig.add_trace(
                go.Scatter(x=t_ak, y=y_oa, mode="lines", name="Offered A")
            )
            added += 1
            break

    # Voltage (V) — charger DC bus
    for col in ["voltage_v", "dc_voltage_v"]:
        if col in df.columns:
            y_v = pd.to_numeric(df[col], errors="coerce")
            fig.add_trace(
                go.Scatter(x=t_ak, y=y_v, mode="lines", name="Voltage (V)")
            )
            added += 1
            break

    # HV Battery Voltage (V) if available
    if "hbv_v" in df.columns:
        y_hbv = pd.to_numeric(df["hbv_v"], errors="coerce")
        fig.add_trace(
            go.Scatter(x=t_ak, y=y_hbv, mode="lines", name="HBV (V)")
        )
        added += 1

    # State of Charge (%) if available
    for col in ["soc", "SoC", "SoC (%)"]:
        if col in df.columns:
            y_soc = pd.to_numeric(df[col], errors="coerce")
            fig.add_trace(
                go.Scatter(x=t_ak, y=y_soc, mode="lines", name="SoC (%)")
            )
            added += 1
            break

    # ---- Axes config ----
    if added >= 2:
        fig.update_layout(
            yaxis=dict(title="Power (kW)"),
            yaxis2=dict(title="Energy (kWh)", overlaying="y", side="right", showgrid=False),
        )
    else:
        fig.update_layout(yaxis=dict(title="Value"))

    return fig