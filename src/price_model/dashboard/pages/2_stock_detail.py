"""Per-ticker view: prediction history (all models) and prediction vs. realized scatter."""

from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from price_model.dashboard._common import get_store, query_df

st.title("Stock detail")

store = get_store()
tickers = query_df(str(store.path), "SELECT DISTINCT ticker FROM predictions ORDER BY ticker")
if tickers.height == 0:
    st.warning("No predictions yet.")
    st.stop()

ticker = st.selectbox("Ticker", tickers["ticker"].to_list())

hist = query_df(
    str(store.path),
    f"""
    SELECT prediction_date, model_id, prediction
    FROM predictions
    WHERE ticker = '{ticker}'
    ORDER BY prediction_date
    """,
)
if hist.height == 0:
    st.info("No predictions stored for this ticker.")
    st.stop()

st.subheader(f"Predicted excess return — {ticker}")
fig = px.line(
    hist.to_pandas(),
    x="prediction_date",
    y="prediction",
    color="model_id",
    title=f"{ticker} predictions over time",
)
fig.add_hline(y=0, line_dash="dot", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent predictions")
st.dataframe(
    hist.sort("prediction_date", descending=True).head(30).to_pandas(),
    use_container_width=True,
)

st.caption(
    "Note: predictions are *excess* returns (over the universe mean) at the horizon "
    "configured in the experiment. Calibration of the absolute level is approximate; "
    "the rank / sign is what matters for portfolio tilts."
)
