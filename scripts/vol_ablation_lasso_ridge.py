"""Does removing vol-cluster features help Lasso AND Ridge? (matched grader, bullish regime)

Resolves a tension:
  - the 9->6 ablation (Lasso, full-history) found dropping the vol cluster HELPED;
  - but Ridge reproduces Lasso's opposite-sign vol_ewm/idio_vol loading, suggesting
    the spread (vol_ewm - idio_vol) is SIGNAL, so dropping idio_vol should HURT.

Test: for Lasso and Ridge, on Panel A (rank), bullish-regime split, IC-alpha
selected by purged forward-chain folds, sweep feature subsets that peel the vol
cluster {idio_vol_20, max_return_21d, beta_60, vol_ewm_20} one at a time.

Usage: PYTHONPATH=src .venv/bin/python scripts/vol_ablation_lasso_ridge.py
"""
from __future__ import annotations
import numpy as np, polars as pl
from datetime import date
from sklearn.linear_model import Lasso, Ridge
from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.metrics import _per_date_ic, summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows

FULL = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "idio_vol_20",
        "max_return_21d", "distance_52w_high", "log_dollar_volume", "beta_60"]
SUBSETS = {
    "9 full":            FULL,
    "8  -idio_vol":      [f for f in FULL if f not in {"idio_vol_20"}],
    "7  -idio,-maxret":  [f for f in FULL if f not in {"idio_vol_20", "max_return_21d"}],
    "6  vol->vol_ewm":   [f for f in FULL if f not in {"idio_vol_20", "max_return_21d", "beta_60"}],
    "5  -all vol":       [f for f in FULL if f not in {"idio_vol_20", "max_return_21d", "beta_60", "vol_ewm_20"}],
}
TRAIN_LO, TRAIN_HI, TEST_LO = date(2022, 10, 10), date(2024, 11, 29), date(2025, 1, 2)


def purged_folds(dates, k=4, embargo=21):
    n = len(dates); c = n // k
    return [(dates[:max(0, i*c-embargo)], dates[i*c:((i+1)*c if i < k-1 else n)]) for i in range(1, k)]


def fold_ic(p, y, d):
    icdf = _per_date_ic(pl.DataFrame({"date": d, "prediction": p, "realized": y}))
    return float(icdf["ic"].mean()) if icdf.height else float("nan")


def search(Maker, grid, feats, fa, train, test, target):
    best_a, best = None, -np.inf
    for al in grid:
        ics = []
        for ftr, fva in fa:
            lm = Maker(al).fit(ftr.select(feats).to_numpy(), ftr["y"].to_numpy())
            p = fva.select(feats).fill_null(0.0).to_numpy() @ lm.coef_ + lm.intercept_
            ics.append(fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        mic = float(np.nanmean(ics))
        if mic > best:
            best, best_a = mic, al
    lm = Maker(best_a).fit(train.select(feats).to_numpy(), train["y"].to_numpy())
    Xte = test.select(feats).fill_null(0.0).to_numpy()
    pred = test.select("date", "ticker").with_columns(pl.Series("prediction", Xte @ lm.coef_ + lm.intercept_))
    j = pred.join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left")
    s = summarize(j.select("date", "ticker", "prediction", "realized"), horizon_days=21)
    return s.information_coefficient, s.long_short_sharpe


def main():
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    print(f"{'subset':18s} {'k':>2s}  {'Lasso_IC':>9s} {'Lasso_Sh':>9s}   {'Ridge_IC':>9s} {'Ridge_Sh':>9s}")
    for name, feats in SUBSETS.items():
        m = build_feature_matrix(raw, feats, "rank", 21).pipe(drop_warmup_rows, feats).sort(["ticker", "date"])
        train = m.filter((pl.col("date") >= TRAIN_LO) & (pl.col("date") <= TRAIN_HI)).drop_nulls(["y", *feats])
        test = m.filter(pl.col("date") >= TEST_LO).drop_nulls(feats)
        target = m.select("date", "ticker", "y")
        trd = sorted(train["date"].unique().to_list())
        fa = [(train.filter(pl.col("date").is_in(t)), train.filter(pl.col("date").is_in(v))) for t, v in purged_folds(trd)]
        lic, lsh = search(lambda a: Lasso(a, max_iter=5000, tol=1e-4), np.logspace(-5, -2.3, 20), feats, fa, train, test, target)
        ric, rsh = search(lambda a: Ridge(a), np.logspace(-3, 4, 20), feats, fa, train, test, target)
        print(f"{name:18s} {len(feats):>2d}  {lic:+9.4f} {lsh:+9.2f}   {ric:+9.4f} {rsh:+9.2f}", flush=True)


if __name__ == "__main__":
    main()
