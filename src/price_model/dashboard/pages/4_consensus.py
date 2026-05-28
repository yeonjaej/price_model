"""Cross-model consensus and disagreement on the latest prediction date.

The most actionable view in the whole dashboard: when models agree, you have a signal;
when they disagree, you don't.
"""

from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from price_model.dashboard._common import (
    get_store,
    latest_predictions_sql,
    list_models,
    query_df,
)

st.title("Consensus")

st.info(
    "**The most actionable view in the dashboard.** For every ticker on the "
    "latest prediction date, we plot the *mean* prediction across selected "
    "models (consensus, x-axis) vs. the *stdev* across those same models "
    "(disagreement, y-axis). **Bottom-right = clean buys. Bottom-left = clean "
    "sells. Top half = noisy — models can't agree.**"
)

store = get_store()
models = list_models(str(store.path))
if len(models) < 2:
    st.info("Need at least 2 models in the store for a consensus view.")
    st.stop()

selected = st.multiselect(
    "Models",
    models,
    default=models,
    help="Pick the models whose forecasts go into the consensus/disagreement aggregation.",
)
if len(selected) < 2:
    st.info("Pick at least 2 models.")
    st.stop()

model_filter = ",".join(f"'{m}'" for m in selected)
dedup_cte = latest_predictions_sql(where=f"model_id IN ({model_filter})")
df = query_df(
    str(store.path),
    f"""
    WITH dedup AS ({dedup_cte}),
    latest AS (
        SELECT model_id, MAX(prediction_date) AS d
        FROM dedup
        GROUP BY model_id
    )
    SELECT d.ticker, d.model_id, d.prediction
    FROM dedup d
    JOIN latest l USING (model_id)
    WHERE d.prediction_date = l.d
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
st.caption("Each point is one ticker on the latest prediction date. Hover for the symbol.")
fig = px.scatter(
    wide.to_pandas(),
    x="consensus",
    y="disagreement",
    hover_name="ticker",
    text="ticker",
    title="Consensus vs. disagreement (latest prediction date)",
)
fig.update_traces(textposition="top center", textfont_size=10)
fig.add_vline(x=0, line_dash="dot", line_color="gray")
fig.update_layout(
    xaxis_title="Consensus = mean predicted 5-day excess return across selected models",
    yaxis_title="Disagreement = stdev of those predictions",
)
# Annotate the four quadrants with their interpretation
xs = wide["consensus"].to_list()
ys = wide["disagreement"].to_list()
if xs and ys:
    x_max, x_min = max(xs), min(xs)
    y_max = max(ys)
    annotations = [
        ("Clean buys", x_max * 0.7 if x_max > 0 else 0.0, y_max * 0.1, "rgba(0,128,0,0.7)"),
        ("Clean sells", x_min * 0.7 if x_min < 0 else 0.0, y_max * 0.1, "rgba(192,0,0,0.7)"),
        ("Noisy bulls", x_max * 0.7 if x_max > 0 else 0.0, y_max * 0.9, "rgba(120,120,120,0.7)"),
        ("Noisy bears", x_min * 0.7 if x_min < 0 else 0.0, y_max * 0.9, "rgba(120,120,120,0.7)"),
    ]
    for txt, ax, ay, color in annotations:
        fig.add_annotation(
            x=ax,
            y=ay,
            text=txt,
            showarrow=False,
            font={"size": 11, "color": color},
        )
st.plotly_chart(fig, width="stretch")

st.subheader("Full table")
st.caption(
    "Sorted by consensus (descending). The top of the table is what the model "
    "ensemble most strongly agrees is a buy on the latest prediction date; "
    "the bottom is what it most strongly agrees is a sell."
)
st.dataframe(wide.sort("consensus", descending=True).to_pandas(), width="stretch")

with st.expander("How to read the scatter", expanded=False):
    st.markdown(
        """
| Quadrant | Meaning | Action |
|---|---|---|
| **Bottom-right** (high consensus, low disagreement) | All models agree this is a strong buy | Overweight in a tilted portfolio |
| **Bottom-left** (low consensus, low disagreement) | All models agree this is a strong sell | Underweight or short candidate |
| **Top-right** (high consensus, high disagreement) | Models lean bullish but disagree on strength | Treat as low-confidence — investigate |
| **Top-left** (low consensus, high disagreement) | Models lean bearish but disagree | Treat as low-confidence — investigate |

The thinking is **direction × certainty**. A ticker with `+0.01` consensus and
`0.0005` disagreement is much more actionable than the same `+0.01` consensus
with `0.008` disagreement — in the latter case some model is shouting
`+0.02` while another is shouting `0.0` and the average is hiding the
disagreement. The y-axis surfaces that.

Reality check: on a 20-name liquid mega-cap universe most points cluster
near the bottom-center (small consensus, small disagreement) because the
universe is hard. The outliers are where the action is.
"""
    )
