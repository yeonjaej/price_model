"""Print the exact forward-chain CV folds, alpha grid, per-alpha CV IC, and the
selected alpha for the regime-confined Ridge-6 and Lasso-6 headline models.

Fits each model directly on the regime-confined training block (2022-10-10 ->
2024-11-29) with verbose=True, so models/_cv.select_alpha_by_ic prints:
  - each purged forward-chain fold's train/val date ranges (embargo applied),
  - the alpha grid (count + range),
  - the top candidate alphas by mean cross-validation IC,
  - the SELECTED alpha.
Then reports the 2025+ out-of-sample IC / Sharpe of the refit model.

Usage: PYTHONPATH=src .venv/bin/python scripts/inspect_alpha_selection.py
"""
from __future__ import annotations

import warnings
from datetime import date

import polars as pl
from rich.console import Console

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.metrics import summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig

FEATS6 = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "distance_52w_high", "log_dollar_volume"]
TRAIN_START, TRAIN_END, OOS = date(2022, 10, 10), date(2024, 11, 29), date(2025, 1, 1)


def main() -> None:
    warnings.filterwarnings("ignore")
    con = Console()
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    m = build_feature_matrix(raw, FEATS6, "rank", 21).pipe(drop_warmup_rows, FEATS6).sort(["ticker", "date"])
    train = m.filter((pl.col("date") >= TRAIN_START) & (pl.col("date") <= TRAIN_END))
    test = m.filter(pl.col("date") >= OOS)
    target = m.select("date", "ticker", "y")

    for name, cls in [("Ridge-6", "RidgeCrossSectional"), ("Lasso-6", "LassoCrossSectional")]:
        con.rule(name)
        model = build_model(cls, ModelConfig(name, tuple(FEATS6), "y", {"cv": 3, "verbose": True}))
        model.fit(train)  # prints folds / grid / per-alpha IC / selected alpha
        preds = model.predict(test)
        j = preds.join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left")
        s = summarize(j.select("date", "ticker", "prediction", "realized"), horizon_days=21)
        con.print(
            f"[bold]{name}[/bold]: selected alpha = {model.selected_alpha():.4e}  |  "
            f"OOS 2025+ IC = {s.information_coefficient:+.4f}  Sharpe = {s.long_short_sharpe:+.2f}"
        )


if __name__ == "__main__":
    main()
