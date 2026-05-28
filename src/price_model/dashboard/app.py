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
    "Cross-sectional equity return predictor — benchmarks Fama-French, ARIMA/GARCH, "
    "LightGBM, and the Chronos foundation model on the S&P 500."
)

st.info(
    "**What a prediction means here.** Every model in this dashboard produces a "
    "**5-day forward excess return** for each stock — i.e. the model's forecast of how "
    "much the stock will out- or under-perform the cross-sectional average over the "
    "next 5 trading days, expressed as a log-return decimal. A prediction of `+0.005` "
    "means *+0.5% relative to the universe mean over the next week*. Models are honest "
    "about scale only on rank; treat the magnitude as a rough confidence indicator."
)

store = get_store()
health = store_health(store)

cols = st.columns(4)
cols[0].metric(
    "Models tracked",
    health["n_models"],
    help="Distinct model IDs in the store. Includes baselines (Zero), classical "
    "(ARIMA/GARCH/Fama-MacBeth) and ML (LightGBM variants).",
)
cols[1].metric(
    "Predictions stored",
    f"{health['n_predictions']:,}",
    help="Total (model_id, prediction_date, ticker) rows. One per walk-forward refit; "
    "the dashboard deduplicates to the latest generated_at when reading.",
)
cols[2].metric(
    "Latest prediction date",
    str(health["latest_date"] or "—"),
    help="Most recent prediction_date in the store. Predictions are 5 trading days "
    "ahead of this date.",
)
cols[3].metric(
    "Tickers seen",
    health["n_tickers"],
    help="Distinct tickers any model has ever predicted on. Universe membership "
    "depends on the experiment YAML.",
)

st.divider()

st.subheader("Pages")
st.markdown(
    """
- **Leaderboard** — top / bottom names per model on the latest date, plus
  cross-model consensus (average prediction) and disagreement (stdev).
  Use this to find names where models agree, and to spot outliers.
- **Stock detail** — pick a ticker, see every model's prediction history.
  When the model lines move together, the signal is robust; when they diverge,
  it's not.
- **Model performance** — out-of-sample evaluation: Information Coefficient,
  hit rate, decile long-short Sharpe, and a rolling-IC line so you can see
  whether the edge is stable, growing, or decaying.
- **Consensus** — scatter of average prediction vs. stdev across models for
  every ticker on the latest date. The quadrants tell you whether to trust
  any given name's signal.
"""
)

with st.expander("Reading the numbers — quick glossary", expanded=False):
    st.markdown(
        """
- **Prediction**: 5-day forward excess return, log-return decimal. `+0.01` ≈ 1% over the next week relative to the universe mean.
- **Information Coefficient (IC)**: Spearman rank correlation between predicted and realized returns, averaged across dates. Measures how well a model *ranks* names. >0.02 with t-stat >2 is real signal on this kind of universe; >0.05 is genuinely good.
- **Hit rate**: fraction of predictions where the sign agrees with the realized sign. 0.5 = coin flip. 0.52-0.55 is typical of weakly profitable signals.
- **Long-short Sharpe**: annualized Sharpe ratio of a daily-rebalanced portfolio that goes long the top decile, short the bottom decile. Above 1 is the bar for "would actually trade this."
- **Consensus / disagreement**: per ticker, the *mean* and *stdev* of predictions across the selected models on the latest date. High consensus + low disagreement = trust the signal. High disagreement = treat as noise.
"""
    )

if health["n_predictions"] == 0:
    st.warning(
        "No predictions in the store yet. Run an experiment first:  "
        "`python -m price_model.cli run --experiment smoke`"
    )

st.divider()
st.caption(f"Store path: `{store.path}`")
