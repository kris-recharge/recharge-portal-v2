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


def _heatmap_from_grid(
    grid: pd.DataFrame,
    title: str,
    value_to_text,
    colorscale: str = "Blues",
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """
    Build a Plotly heatmap from a (7 x 24) DataFrame whose index is 0..6 (Sun..Sat)
    and columns are 0..23 (hours).
    """
    # Ensure exact shape/order for robustness
    grid = (
        grid.reindex(index=range(7), fill_value=0)
        .reindex(columns=range(24), fill_value=0)
        .copy()
    )

    z = grid.values
    text = grid.applymap(value_to_text).values
    y_labels = [DAY_LABELS[i] for i in grid.index]
    x_labels = list(grid.columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Value: %{z}<extra></extra>",
            colorbar=dict(title=None),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Hour (0–23)"),
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
    df = pd.DataFrame(
        {
            "dow": (ts_ak.dt.dayofweek + 1) % 7,  # Sun=0 .. Sat=6
            "hour": ts_ak.dt.hour,
        }
    )
    grid = df.groupby(["dow", "hour"]).size().unstack(fill_value=0)
    return _heatmap_from_grid(grid, title, _blank_or_int)


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

    grid = df.groupby(["dow", "hour"])["dur"].mean().unstack()
    return _heatmap_from_grid(grid, title, _blank_or_float1)


def session_detail_figure(df: Optional[pd.DataFrame], title: str = "Session details") -> go.Figure:
    """
    Minimal placeholder so callers can safely import it.
    Extend later with real traces from your session telemetry frame.
    """
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value")
    return fig