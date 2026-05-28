"""Rolling out-of-sample performance per model.

Joins predictions to realized forward excess returns derived from the price panel,
then computes IC, hit rate, and long-short Sharpe in rolling windows.
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
from price_model.data.loaders import load_panel
from price_model.eval.metrics import compare_models
from price_model.features.targets import add_forward_excess_return

st.title("Model performance")
st.caption(
    "Out-of-sample IC, hit rate, and decile long-short Sharpe per model. "
    "Computed by joining stored predictions to realized forward excess returns."
)

st.info(
    "**This is the page that tells you which models to trust.** Stored "
    "predictions are joined to the *realized* 5-day forward excess return for "
    "the same (date, ticker) and scored. **IC > 0.02 with t-stat > 2** is real "
    "signal on a mega-cap universe; **long-short Sharpe > 1** is the bar for "
    "being worth trading. Negative IC means the model is reliably wrong — "
    "still useful as a contrarian indicator but not as a buy list."
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
    f"""
    WITH latest AS ({latest_predictions_sql()})
    SELECT model_id, prediction_date, ticker, prediction FROM latest
    """,
)
joined = preds.join(realized, on=["prediction_date", "ticker"], how="inner")
joined = joined.rename({"prediction_date": "date"})

if joined.height == 0:
    st.info("No overlap between stored predictions and realized returns yet.")
    st.stop()

summary = compare_models(joined, horizon_days=int(horizon))
st.subheader("Overall scoreboard")
st.caption(
    "All stored predictions joined to their realized 5-day forward excess return. "
    "Higher is better for `information_coefficient`, `ic_t_stat`, `hit_rate`, and "
    "`long_short_sharpe`; lower is better for `mae` and `rmse`."
)
st.dataframe(summary.to_pandas(), width="stretch")

st.subheader("Information coefficient over time")
st.caption(
    "Monthly IC per model. A line steady above zero = persistent edge. A line "
    "crossing zero = the edge appears and disappears with regime. The dotted "
    "gray line at IC = 0 is the EMH null (no predictive power)."
)
# Simple rolling IC: per-model, per-month, IC over the month
joined = joined.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
monthly_ic_rows = []
for (model_id, month), grp in joined.group_by(["model_id", "month"]):
    summ = compare_models(
        grp.select("date", "ticker", "prediction", "realized").with_columns(
            pl.lit(model_id).alias("model_id")
        ),
        horizon_days=int(horizon),
    )
    monthly_ic_rows.append(
        {
            "model_id": model_id,
            "month": month,
            "ic": summ["information_coefficient"][0],
        }
    )
if monthly_ic_rows:
    monthly = pl.DataFrame(monthly_ic_rows).sort("month")
    fig = px.line(monthly.to_pandas(), x="month", y="ic", color="model_id")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Information Coefficient (Spearman rank corr.)",
        legend_title="Model",
    )
    st.plotly_chart(fig, width="stretch")

with st.expander("How to read the scoreboard", expanded=False):
    st.markdown(
        """
- **`information_coefficient` (IC)** — Spearman rank correlation between
  predicted and realized excess returns, averaged across dates. Measures how
  well a model *ranks* names. A typical signal on a liquid mega-cap universe
  is 0.01-0.05; institutional quant shops aim for sustained IC > 0.05.
- **`ic_t_stat`** — t-statistic for the daily IC series being non-zero. **>2
  ≈ 5%-significant**; **>3 is strong**. Helps distinguish a real edge from
  one good year of luck.
- **`hit_rate`** — fraction of (date, ticker) where `sign(prediction)` matches
  `sign(realized)`. 0.5 = coin flip. 0.52-0.55 is typical of weakly profitable
  signals; sustained 0.55+ is very good.
- **`mae` / `rmse`** — mean / root-mean-squared error vs. realized. These
  measure *calibration* (magnitude accuracy), which a cross-sectional ranker
  doesn't optimize for, so the values are typically similar across models even
  when their IC differs. Use IC, not MAE, to compare ranking ability.
- **`long_short_sharpe`** — annualized Sharpe of a daily-rebalanced portfolio:
  long the top-decile predicted names, short the bottom decile, equal-weighted.
  Above 1 is "would actually trade this." This is the strictest test in the
  table — small IC can still produce decent Sharpe if the signal is
  concentrated in the tails (which is what tree models often do).
- **`n_observations` / `n_dates`** — sample size. IC is over-fittable on small
  samples; the t-stat already corrects for this, but always look at n_dates
  before reading IC magnitudes.

### When numbers don't agree
- High IC, low Sharpe: model ranks accurately but the magnitudes are small
  enough that decile spreads don't compound to much. Common with linear models.
- Low IC, decent Sharpe: model is mostly noisy but right when it has high
  confidence. Common with deep tree models that overfit the middle of the
  distribution.
- Both negative: model is reliably anti-correlated with future returns.
  Treat it as a contrarian indicator, or check for a sign-flip bug.
"""
    )
