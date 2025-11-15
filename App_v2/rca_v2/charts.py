from rca_v2 import charts

st.subheader("Session Start Density (by Day & Hour)")

starts_df = None
if session_summary is not None and not session_summary.empty:
    df0 = session_summary.copy()

    # Robustly find the start column
    start_col = None
    for c in ["_start", "start_ts", "start_time", "Start (UTC)"]:
        if c in df0.columns:
            start_col = c
            break

    if start_col is not None:
        # Normalize to UTC (tz-aware) so downstream AK conversion is stable
        df0["_start_ts"] = pd.to_datetime(df0[start_col], errors="coerce", utc=True)

        mask = df0["_start_ts"].notna()

        # Build a minimal starts frame
        keep_cols = []
        for c in ["station_id", "connector_id", "transaction_id"]:
            if c in df0.columns:
                keep_cols.append(c)
        keep_cols.append("_start_ts")

        starts_raw = (
            df0.loc[mask, keep_cols]
               .rename(columns={"_start_ts": "start_ts"})
               .copy()
        )

        # Transaction id fallbacks
        if "transaction_id" not in starts_raw.columns:
            if "tx" in df0.columns:
                starts_raw["transaction_id"] = df0.loc[mask, "tx"].astype(str)

        # De-duplicate one start per (EVSE/connector/transaction)
        key_cols = [c for c in ["station_id", "connector_id", "transaction_id"] if c in starts_raw.columns]
        starts_df = (
            starts_raw.sort_values("start_ts")
                      .drop_duplicates(subset=key_cols, keep="first")
                      .reset_index(drop=True)
        )

if starts_df is not None and not starts_df.empty:
    # Render the new, correct heatmap (counts per day/hour)
    st.plotly_chart(
        charts.heatmap_count(starts_df[["start_ts"]], "Session Start Density (by Day & Hour)"),
        use_container_width=True,
    )
else:
    st.info("No sessions found in this window.")