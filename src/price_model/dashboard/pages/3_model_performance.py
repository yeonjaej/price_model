"""Rolling out-of-sample performance per model.

Joins predictions to realized forward excess returns derived from the price panel,
then computes IC, hit rate, and long-short Sharpe in rolling windows.
"""

from __future__ import annotations

import polars as pl
import plotly.express as px
import streamlit as st

from price_model.dashboard._common import get_store, list_models, query_df
from price_model.data.loaders import load_panel
from price_model.eval.metrics import compare_models
from price_model.features.targets import add_forward_excess_return

st.title("Model performance")
st.caption(
    "Out-of-sample IC, hit rate, and decile long-short Sharpe per model. "
    "Computed by joining stored predictions to realized forward excess returns."
)

store = get_store()
models = list_models(str(store.path))
if not models:
    st.warning("No models tracked.")
    st.stop()

universe = st.selectbox("Realized-returns universe", ["sp500", "top20_2026_01_01"], index=0)
horizon = st.number_input("Horizon (days)", value=5, min_value=1, max_value=21, step=1)
start = st.text_input("Panel start date", value="2017-01-01")

with st.spinner("Loading panel and computing realized returns..."):
    panel = load_panel(universe=universe, start=start)
    panel = add_forward_excess_return(panel, horizon_days=int(horizon))
    realized = panel.select(
        pl.col("date").alias("prediction_date"),
        "ticker",
        pl.col("y").alias("realized"),
    )

preds = query_df(
    str(store.path),
    "SELECT model_id, prediction_date, ticker, prediction FROM predictions",
)
joined = preds.join(realized, on=["prediction_date", "ticker"], how="inner")
joined = joined.rename({"prediction_date": "date"})

if joined.height == 0:
    st.info("No overlap between stored predictions and realized returns yet.")
    st.stop()

summary = compare_models(joined, horizon_days=int(horizon))
st.subheader("Overall (all stored predictions joined to realized)")
st.dataframe(summary.to_pandas(), use_container_width=True)

st.subheader("Rolling 60-day information coefficient")
# Simple rolling IC: per-model, per-month, IC over the month
joined = joined.with_columns(
    pl.col("date").dt.truncate("1mo").alias("month")
)
monthly_ic_rows = []
for (model_id, month), grp in joined.group_by(["model_id", "month"]):
    summ = compare_models(grp.select("date", "ticker", "prediction", "realized")
                            .with_columns(pl.lit(model_id).alias("model_id")),
                          horizon_days=int(horizon))
    monthly_ic_rows.append({
        "model_id": model_id,
        "month": month,
        "ic": summ["information_coefficient"][0],
    })
if monthly_ic_rows:
    monthly = pl.DataFrame(monthly_ic_rows).sort("month")
    fig = px.line(monthly.to_pandas(), x="month", y="ic", color="model_id")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
