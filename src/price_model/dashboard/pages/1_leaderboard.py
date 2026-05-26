"""Today's top/bottom predicted names per model + the cross-model consensus."""

from __future__ import annotations

import polars as pl
import streamlit as st

from price_model.dashboard._common import get_store, list_models, query_df

st.title("Leaderboard")
st.caption("Most recent predicted excess return per (model, ticker).")

store = get_store()
models = list_models(str(store.path))
if not models:
    st.warning("No models in the store. Run an experiment first.")
    st.stop()

selected = st.multiselect("Models", models, default=models)
top_n = st.slider("Top / bottom N", min_value=5, max_value=20, value=10)

if not selected:
    st.info("Pick at least one model.")
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
    SELECT p.model_id, p.prediction_date, p.ticker, p.prediction
    FROM predictions p
    JOIN latest l USING (model_id)
    WHERE p.prediction_date = l.d
    ORDER BY p.model_id, p.prediction DESC
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
    st.dataframe(wide.head(top_n).to_pandas(), use_container_width=True)
with c2:
    st.subheader(f"Bottom {top_n} (lowest consensus)")
    st.dataframe(wide.tail(top_n).reverse().to_pandas(), use_container_width=True)

st.divider()
st.subheader("Highest disagreement names")
st.caption(
    "Where models disagree most. Treat as low-confidence signals — or interesting cases "
    "to investigate."
)
st.dataframe(
    wide.sort("disagreement", descending=True).head(top_n).to_pandas(),
    use_container_width=True,
)
