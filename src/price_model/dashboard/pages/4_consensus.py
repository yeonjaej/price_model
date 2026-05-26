"""Cross-model consensus and disagreement on the latest prediction date.

The most actionable view in the whole dashboard: when models agree, you have a signal;
when they disagree, you don't.
"""

from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from price_model.dashboard._common import get_store, list_models, query_df

st.title("Consensus")

store = get_store()
models = list_models(str(store.path))
if len(models) < 2:
    st.info("Need at least 2 models in the store for a consensus view.")
    st.stop()

selected = st.multiselect("Models", models, default=models)
if len(selected) < 2:
    st.info("Pick at least 2 models.")
    st.stop()

model_filter = ",".join(f"'{m}'" for m in selected)
df = query_df(
    str(store.path),
    f"""
    WITH latest AS (
        SELECT model_id, MAX(prediction_date) AS d
        FROM predictions
        WHERE model_id IN ({model_filter})
        GROUP BY model_id
    )
    SELECT p.ticker, p.model_id, p.prediction
    FROM predictions p
    JOIN latest l USING (model_id)
    WHERE p.prediction_date = l.d
    """,
)
if df.height == 0:
    st.info("No data.")
    st.stop()

wide = df.pivot(on="model_id", index="ticker", values="prediction")
wide = wide.with_columns(
    pl.mean_horizontal(selected).alias("consensus"),
    pl.concat_list(selected).list.std().alias("disagreement"),
)

st.subheader("Consensus vs. disagreement")
st.caption(
    "Each point is a ticker. X = average prediction across models. Y = stdev. "
    "Top-right and top-left quadrants are high-disagreement names; bottom row is consensus."
)
fig = px.scatter(
    wide.to_pandas(),
    x="consensus",
    y="disagreement",
    hover_name="ticker",
    title="Consensus vs. disagreement",
)
fig.add_vline(x=0, line_dash="dot", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Full table")
st.dataframe(wide.sort("consensus", descending=True).to_pandas(),
             use_container_width=True)
