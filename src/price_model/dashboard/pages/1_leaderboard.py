"""Today's top/bottom predicted names per model + the cross-model consensus."""

from __future__ import annotations

import polars as pl
import streamlit as st

from price_model.dashboard._common import (
    get_store,
    latest_predictions_sql,
    list_models,
    query_df,
)

st.title("Leaderboard")
st.caption("Most recent predicted excess return per (model, ticker).")

st.info(
    "**How to read the table.** Each row is a ticker, each model column is its "
    "predicted 5-day forward excess return (decimal log-return; `+0.01` ≈ +1% "
    "vs. universe mean over the next week). **`consensus`** is the mean across "
    "selected models; **`disagreement`** is their stdev. Tickers near the top "
    "of the table are the strongest *agreed-upon* buys on the latest date in "
    "the store; tickers at the bottom are the strongest agreed-upon shorts."
)

store = get_store()
models = list_models(str(store.path))
if not models:
    st.warning("No models in the store. Run an experiment first.")
    st.stop()

selected = st.multiselect(
    "Models",
    models,
    default=models,
    help="Each selected model contributes one column to the table and one term to consensus/disagreement.",
)
top_n = st.slider("Top / bottom N", min_value=5, max_value=20, value=10)

if not selected:
    st.info("Pick at least one model.")
    st.stop()

model_filter = ",".join(f"'{m}'" for m in selected)
# Dedup first, then filter to the latest prediction_date per model. Doing the
# dedup BEFORE the date filter ensures we don't accidentally pick a stale
# latest-date from an older run.
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
    SELECT d.model_id, d.prediction_date, d.ticker, d.prediction
    FROM dedup d
    JOIN latest l USING (model_id)
    WHERE d.prediction_date = l.d
    ORDER BY d.model_id, d.prediction DESC
    """,
)

if df.height == 0:
    st.info("Selected models have no predictions yet.")
    st.stop()

# Pivot to wide format: rows = ticker, columns = model_id
wide = df.pivot(on="model_id", index="ticker", values="prediction").sort("ticker")
wide = wide.with_columns(
    pl.mean_horizontal(selected).alias("consensus"),
    pl.concat_list(selected).list.std().alias("disagreement"),
)
wide = wide.sort("consensus", descending=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Top {top_n} (highest consensus)")
    st.dataframe(wide.head(top_n).to_pandas(), width="stretch")
with c2:
    st.subheader(f"Bottom {top_n} (lowest consensus)")
    st.dataframe(wide.tail(top_n).reverse().to_pandas(), width="stretch")

st.divider()
st.subheader("Highest disagreement names")
st.caption(
    "Where models disagree most. Treat as low-confidence signals — or interesting cases "
    "to investigate."
)
st.dataframe(
    wide.sort("disagreement", descending=True).head(top_n).to_pandas(),
    width="stretch",
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        """
- **Long the consensus longs, short the consensus shorts.** Top-of-table names
  with low `disagreement` are the cleanest signals. A real portfolio would tilt
  weight by `consensus` and trim weight by `disagreement`.
- **Be skeptical of high-`disagreement` names.** They're cases where the models
  see different things — usually because the ticker is at a regime boundary
  (e.g., trending stock that just broke a 200-day MA) or because one model's
  feature set picks up something the others miss. Worth investigating but not
  worth trusting.
- **The numbers look small because they are.** A 5-day excess return of `0.005`
  is a real edge — most days the cross-section moves on the order of 1-3% in
  total, so out-performing the average by half a percent over a week is a
  meaningful tilt. Don't confuse "small decimal" with "small effect."
- **Cross-check with the Model Performance page.** The IC and Sharpe there tell
  you which model_id is actually worth listening to. If a model has IC ≈ 0
  out-of-sample, ignore its column even when it has strong opinions today.
"""
    )
