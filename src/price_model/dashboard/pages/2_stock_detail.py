"""Per-ticker view: prediction history (all models) and prediction vs. realized scatter."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from price_model.dashboard._common import get_store, latest_predictions_sql, query_df

st.title("Stock detail")

st.info(
    "**Per-ticker view of every model's prediction history.** Each line is one "
    "model's forecast of the next 5-day excess return for the selected stock. "
    "The dotted gray line at zero marks the universe mean — anything above means "
    "the model expects out-performance; below means under-performance. **When the "
    "lines move together, the signal is robust. When they fan out, models disagree "
    "and you should be cautious.**"
)

store = get_store()
tickers = query_df(str(store.path), "SELECT DISTINCT ticker FROM predictions ORDER BY ticker")
if tickers.height == 0:
    st.warning("No predictions yet.")
    st.stop()

ticker = st.selectbox("Ticker", tickers["ticker"].to_list())

dedup_cte = latest_predictions_sql(where=f"ticker = '{ticker}'")
hist = query_df(
    str(store.path),
    f"""
    WITH latest AS ({dedup_cte})
    SELECT prediction_date, model_id, prediction
    FROM latest
    ORDER BY prediction_date
    """,
)
if hist.height == 0:
    st.info("No predictions stored for this ticker.")
    st.stop()

st.subheader(f"Predicted 5-day excess return — {ticker}")
fig = px.line(
    hist.to_pandas(),
    x="prediction_date",
    y="prediction",
    color="model_id",
    title=f"{ticker} predictions over time (decimal log-return; +0.01 ≈ +1% vs. universe over next 5 days)",
)
fig.add_hline(y=0, line_dash="dot", line_color="gray")
fig.update_layout(
    xaxis_title="Prediction date (the date the forecast was made)",
    yaxis_title="Predicted forward excess return (5-day, decimal)",
    legend_title="Model",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Recent predictions")
st.caption("Last 30 prediction rows for this ticker, most recent first.")
st.dataframe(
    hist.sort("prediction_date", descending=True).head(30).to_pandas(),
    width="stretch",
)

with st.expander("How to read this page", expanded=False):
    st.markdown(
        """
- **The dotted zero line is the cross-section mean.** Predictions are *excess*
  returns, not raw forecasts. A flat line at zero means the model has no
  cross-sectional view on this stock; positive means it expects out-performance,
  negative means under-performance.
- **Look for consensus across model lines.** When 3+ models agree on direction,
  that's a robust signal. When they diverge — one model says +0.01, another
  says -0.01 — the cross-model disagreement is your noise indicator.
- **Recent vs. historical.** The rightmost points are the *live* forecast for
  the next 5 days. Earlier points are walk-forward predictions made as if the
  model only knew data up to that date — they're the honest backtest record,
  not in-sample fits.
- **Calibration is approximate.** The model is trained to rank cross-sectionally,
  not to predict precise excess return magnitudes. Treat the size of the
  prediction as a rough confidence indicator, not a literal expected return.
"""
    )
