"""Streamlit entry point — reads from the prediction store, never calls a model.

Run with:  python -m price_model.cli dashboard
       or: streamlit run src/price_model/dashboard/app.py

Pages live in dashboard/pages/ and are auto-discovered by Streamlit's multi-page system.
"""

from __future__ import annotations

import streamlit as st

from price_model.dashboard._common import get_store, store_health

st.set_page_config(
    page_title="price-model",
    page_icon="📈",
    layout="wide",
)

st.title("price-model")
st.caption(
    "Cross-sectional return predictor. Trained on the S&P 500, deployed on the top 20. "
    "Predictions land in a DuckDB store; this dashboard reads from there."
)

store = get_store()
health = store_health(store)

cols = st.columns(4)
cols[0].metric("Models tracked", health["n_models"])
cols[1].metric("Predictions stored", f"{health['n_predictions']:,}")
cols[2].metric("Latest prediction date", str(health["latest_date"] or "—"))
cols[3].metric("Tickers seen", health["n_tickers"])

st.divider()

st.subheader("Where to look")
st.markdown(
    """
- **Leaderboard** — today's top/bottom predicted names per model, with cross-model
  consensus + disagreement.
- **Stock detail** — per-ticker view: prediction history vs. realized, feature
  attribution snapshot.
- **Model performance** — rolling IC, hit rate, long-short Sharpe, by model.
- **Consensus** — where do models agree / disagree most strongly today.
"""
)

if health["n_predictions"] == 0:
    st.warning(
        "No predictions in the store yet. Run an experiment first:  "
        "`python -m price_model.cli run --experiment smoke`"
    )

st.divider()
st.caption(f"Store path: `{store.path}`")
