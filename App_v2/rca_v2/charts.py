def _heatmap_from_grid(
    grid: pd.DataFrame,
    title: str,
    value_to_text,
    colorscale: str = "Blues",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Build a Plotly heatmap from a (7 x 24) DataFrame whose index is 0..6 (Sun..Sat)
    and columns are 0..23 (hours). Text is forced to strings (no None) so Plotly
    never falls back to x/y labels inside the cells.
    """
    # Coerce to fixed shape and numeric array for Plotly
    grid = (
        grid.reindex(index=range(7), fill_value=0)
            .reindex(columns=range(24), fill_value=0)
            .astype(float)
            .copy()
    )

    # Numeric Z for the heatmap
    z = grid.to_numpy(copy=True)

    # Build label text strictly as strings with blanks for 0/NaN
    text_df = grid.applymap(value_to_text)
    text_df = text_df.replace(to_replace=[None, np.nan, "None", "nan"], value="")
    text = text_df.astype(str).to_numpy()

    y_labels = [DAY_LABELS[i] for i in grid.index]
    x_labels = list(grid.columns)

    fig = go.Figure(
        data=[go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Value: %{z:.1f}<extra></extra>",
            colorbar=dict(title=None),
            xgap=1,  # draw cell borders
            ygap=1,
        )]
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Hour (0–23)"),
        yaxis=dict(title="Day"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=30, b=40),
    )
    # Black labels so they’re readable on white cells
    fig.update_traces(textfont=dict(color="black", size=10))
    return fig