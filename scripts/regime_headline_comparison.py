"""Regime-confined headline comparison: linear vs Optuna-tuned trees vs momentum.

EVERYTHING trains on the bullish regime ONLY (2022-10-10 -> 2024-11-29) and is
tested on 2025-01-02 -> 2026-05-22. This deliberately avoids training across the
2022 regime break (models trained on multiple regimes are not well understood
here; see README Discussion). Features are built on full price history (so the
3-year momentum warmup is satisfied) and then training ROWS are filtered to the
regime -- the CLI walk-forward cannot do this, hence this standalone script.

For each model we report gross IC / t / gross L/S Sharpe, annual turnover, and
net Sharpe at 3 / 10 / 20 bp (eval/turnover.compute_turnover_and_costs).

Trees (LightGBM / XGBoost / CatBoost) are Optuna-tuned via purged forward-chain
folds *within the regime train block*, scored by mean per-date IC, on BOTH the
9-feature rank panel and the 14-feature engineered zscore panel; each tree's row
uses whichever panel won on CV IC.

Usage:  PYTHONPATH=src .venv/bin/python scripts/regime_headline_comparison.py [TRIALS]
        TRIALS defaults to 30.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date

import numpy as np
import optuna
import polars as pl
from sklearn.linear_model import Lasso, Ridge

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.metrics import _per_date_ic
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_LO, TRAIN_HI, TEST_LO = date(2022, 10, 10), date(2024, 11, 29), date(2025, 1, 2)
FEATS_RANK9 = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "idio_vol_20",
               "max_return_21d", "distance_52w_high", "log_dollar_volume", "beta_60"]
FEATS_LIN6 = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20",
              "distance_52w_high", "log_dollar_volume"]
FEATS_ENG14 = ["abnormal_volume", "distance_52w_high", "mom60_minus_dist_ma200", "return_5d",
               "max_return_21d", "momentum_12_1", "rsi_14", "log_dollar_volume", "vol_ewm_20",
               "momentum_60_rank", "cs_return_dispersion_20", "sector_relative_return_5d",
               "momentum_504", "beta_60"]
CFG = "[bold]regime-confined[/bold] train 2022-10-10..2024-11-29 / test 2025-01-02+"


def purged_folds(dates, k=4, embargo=21):
    n = len(dates); c = n // k
    return [(dates[:max(0, i*c-embargo)], dates[i*c:((i+1)*c if i < k-1 else n)]) for i in range(1, k)]


def fold_ic(pred, y, d):
    icdf = _per_date_ic(pl.DataFrame({"date": d, "prediction": pred, "realized": y}))
    return float(icdf["ic"].mean()) if icdf.height else float("nan")


def tree_space(trial, cls):
    if cls == "LightGBMModel":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 200),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.95),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 50, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 50, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "bagging_freq": 5, "verbosity": -1, "seeds": [42]}
    if cls == "XGBoostModel":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "tree_method": "hist", "verbosity": 0, "seeds": [42]}
    return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50, log=True),
            "rsm": trial.suggest_float("rsm", 0.5, 0.95),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "bootstrap_type": "Bernoulli", "verbose": False,
            "allow_writing_files": False, "seeds": [42]}


def tune_tree(cls, feats, train, test, fa, n_trials):
    def objective(trial):
        params = tree_space(trial, cls)
        ics = []
        for ftr, fva in fa:
            m = build_model(cls, ModelConfig("t", tuple(feats), "y", params))
            m.fit(ftr)
            p = m.predict(fva)["prediction"].to_numpy()
            ics.append(fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        return float(np.nanmean(ics))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = {**tree_space(optuna.trial.FixedTrial(study.best_params), cls)}
    m = build_model(cls, ModelConfig("t", tuple(feats), "y", best))
    m.fit(train)
    pred = m.predict(test)
    return study.best_value, pred


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    from rich.console import Console; from rich.table import Table
    con = Console()
    con.print(f"Building panels; {CFG}; Optuna trials={n_trials}")
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    m_rank = build_feature_matrix(raw, FEATS_RANK9, "rank", 21).pipe(drop_warmup_rows, FEATS_RANK9).sort(["ticker", "date"])
    m_eng = build_feature_matrix(raw, FEATS_ENG14, "zscore", 21).pipe(drop_warmup_rows, FEATS_ENG14).sort(["ticker", "date"])
    target = m_rank.select("date", "ticker", "y")

    def split(m, feats):
        tr = m.filter((pl.col("date") >= TRAIN_LO) & (pl.col("date") <= TRAIN_HI)).drop_nulls(["y", *feats])
        te = m.filter(pl.col("date") >= TEST_LO).drop_nulls(feats)
        fa = [(tr.filter(pl.col("date").is_in(t)), tr.filter(pl.col("date").is_in(v)))
              for t, v in purged_folds(sorted(tr["date"].unique().to_list()))]
        return tr, te, fa

    tr_r, te_r, fa_r = split(m_rank, FEATS_RANK9)
    tr_e, te_e, fa_e = split(m_eng, FEATS_ENG14)
    rows = []

    def add(name, panel, pred_df):
        j = pred_df.join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left")
        s = compute_turnover_and_costs(j.select("date", "ticker", "prediction", "realized"),
                                       cost_bps=(3, 10, 20), horizon_days=21)
        rows.append((name, panel, s.gross_ic, s.gross_ic_t_stat, s.gross_long_short_sharpe,
                     s.annual_turnover, s.after_cost_sharpe_by_bp[3], s.after_cost_sharpe_by_bp[10],
                     s.after_cost_sharpe_by_bp[20]))

    def lin_search(Maker, grid, feats, tr, te, fa):
        best_a, best = None, -np.inf
        for al in grid:
            ics = []
            for ftr, fva in fa:
                lm = Maker(al).fit(ftr.select(feats).to_numpy(), ftr["y"].to_numpy())
                p = fva.select(feats).fill_null(0.0).to_numpy() @ lm.coef_ + lm.intercept_
                ics.append(fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
            mic = float(np.nanmean(ics))
            if mic > best: best, best_a = mic, al
        lm = Maker(best_a).fit(tr.select(feats).to_numpy(), tr["y"].to_numpy())
        X = te.select(feats).fill_null(0.0).to_numpy()
        return te.select("date", "ticker").with_columns(pl.Series("prediction", X @ lm.coef_ + lm.intercept_))

    # Linear (curated-6, rank) + 9-feature reference
    con.print("Lasso-6 ..."); add("Lasso", "rank-6", lin_search(lambda a: Lasso(a, max_iter=5000, tol=1e-4), np.logspace(-5, -2.3, 25), FEATS_LIN6, tr_r, te_r, fa_r))
    con.print("Ridge-6 ..."); add("Ridge", "rank-6", lin_search(lambda a: Ridge(a), np.logspace(-3, 4, 25), FEATS_LIN6, tr_r, te_r, fa_r))
    con.print("Lasso-9 (ref) ..."); add("Lasso", "rank-9", lin_search(lambda a: Lasso(a, max_iter=5000, tol=1e-4), np.logspace(-5, -2.3, 25), FEATS_RANK9, tr_r, te_r, fa_r))

    # Momentum baseline (no training): prediction = rank momentum_756 on test
    con.print("mom_756 baseline ...")
    add("mom_756", "factor", te_r.select("date", "ticker", pl.col("momentum_756").alias("prediction")))

    # Trees: Optuna on both panels, keep the better by CV IC
    for cls, label in [("LightGBMModel", "LightGBM"), ("XGBoostModel", "XGBoost"), ("CatBoostModel", "CatBoost")]:
        con.print(f"{label}: tuning rank-9 ...")
        cv_r, pred_r = tune_tree(cls, FEATS_RANK9, tr_r, te_r, fa_r, n_trials)
        con.print(f"{label}: tuning eng-14 ...")
        cv_e, pred_e = tune_tree(cls, FEATS_ENG14, tr_e, te_e, fa_e, n_trials)
        if cv_r >= cv_e:
            con.print(f"  {label}: rank-9 wins CV ({cv_r:+.4f} vs {cv_e:+.4f})")
            add(label, "rank-9", pred_r)
        else:
            con.print(f"  {label}: eng-14 wins CV ({cv_e:+.4f} vs {cv_r:+.4f})")
            add(label, "eng-14", pred_e)

    t = Table(title="Regime-confined headline (test 2025-01-02..2026-05-22, 21-day)")
    for c in ("model", "panel", "gross IC", "t", "gross Sh", "ann.turn", "net@3bp", "net@10bp", "net@20bp"):
        t.add_column(c, justify=("left" if c in ("model", "panel") else "right"))
    for name, panel, ic, tt, sh, to, n3, n10, n20 in rows:
        t.add_row(name, panel, f"{ic:+.4f}", f"{tt:+.2f}", f"{sh:+.2f}",
                  f"{to:.0f}x", f"{n3:+.2f}", f"{n10:+.2f}", f"{n20:+.2f}")
    con.print(t)


if __name__ == "__main__":
    main()
