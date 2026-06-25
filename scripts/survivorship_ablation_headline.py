"""Clean survivorship-bias ablation on the headline 9-feature L1 model.

Isolates the survivorship effect from the composition/source change that
confounds notebooks/00 (which compares a 156-name top-cap `sp500` list against
the 701-name `sp500_pit` union — two different sources AND two different
time-handlings at once).

Here all three arms are drawn from the SAME price source (the 701-name
sp500_pit panel, loaded once with pit_filter=False), and differ ONLY in which
cross-section is used on each date:

  1. PIT-on (honest)      — per-date S&P 500 membership (filter_panel_to_pit).
  2. Survivor-snapshot    — restrict to the CURRENT roster (members_on_date at
                            the panel's last date), used across ALL history.
                            This is the genuinely survivorship-biased arm.
  3. All-available        — all 701 names whenever they have data (no membership
                            filter). What load_panel(pit_filter=False) literally
                            does; includes failed names while they traded.

Features are rebuilt per arm so cross-sectional rank-normalization matches each
arm's own universe. Same walk-forward recipe as lasso_elasso_pit_h21
(rank-normalized, h=21, embargo 22d, refit 252d, min_train 504d).

  delta = (survivor-snapshot IC) - (PIT-on IC)  == the headline model's own
  survivorship-bias inflation, finally measured rather than extrapolated.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/survivorship_ablation_headline.py
"""

from __future__ import annotations

from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit, members_on_date
from price_model.eval.metrics import summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward

FEATS = [
    "momentum_12_1", "momentum_756", "return_1d",
    "vol_ewm_20", "idio_vol_20", "max_return_21d",
    "distance_52w_high", "log_dollar_volume", "beta_60",
]
OOS_START = date(2025, 1, 2)


def run_arm(name: str, raw: pl.DataFrame) -> dict:
    matrix = build_feature_matrix(raw, feature_names=FEATS, normalize_kind="rank", target_horizon=21)
    matrix = drop_warmup_rows(matrix, FEATS).sort(["ticker", "date"])
    target = matrix.select("date", "ticker", "y")
    model = build_model(
        "LassoCrossSectional",
        ModelConfig(model_id=f"surv_{name}", feature_cols=FEATS, params={"cv": 5}),
    )
    preds = run_walk_forward(
        matrix, model=model, feature_cols=FEATS, target_col="y",
        experiment_id="surv_ablation", horizon_days=21,
        refit_freq_days=252, embargo_days=22, min_train_days=504,
    )
    joined = join_with_realized(preds, target)
    full = summarize(joined, horizon_days=21)
    oos = summarize(joined.filter(pl.col("date") >= OOS_START), horizon_days=21)
    return {
        "name": name,
        "n_tickers": matrix["ticker"].n_unique(),
        "full_ic": full.information_coefficient,
        "oos_ic": oos.information_coefficient,
        "oos_t": oos.ic_t_stat,
        "oos_sharpe": oos.long_short_sharpe,
    }


def main() -> None:
    console = Console()
    console.print("Loading full sp500_pit price panel (701 names, no membership filter)...")
    raw = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False)
    last_date = raw["date"].max()
    current = members_on_date(last_date)
    in_panel = set(raw["ticker"].unique().to_list())
    current_in_panel = current & in_panel
    console.print(
        f"Panel: {len(in_panel)} tickers with data, last date {last_date}. "
        f"Current roster on {last_date}: {len(current)} members ({len(current_in_panel)} have data).\n"
    )

    arms = {
        "PIT-on (honest)": filter_panel_to_pit(raw),
        "survivor-snapshot": raw.filter(pl.col("ticker").is_in(list(current_in_panel))),
        "all-available": raw,
    }

    results = []
    for name, rawf in arms.items():
        console.print(f"Running arm: {name} ...")
        results.append(run_arm(name, rawf))

    table = Table(title=f"Headline L1 survivorship ablation (same source; OOS = {OOS_START}+)")
    table.add_column("arm", style="bold")
    table.add_column("tickers", justify="right")
    table.add_column("full-sample IC", justify="right")
    table.add_column("OOS IC", justify="right")
    table.add_column("OOS t", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    for r in results:
        table.add_row(
            r["name"], str(r["n_tickers"]),
            f"{r['full_ic']:+.4f}", f"{r['oos_ic']:+.4f}",
            f"{r['oos_t']:+.2f}", f"{r['oos_sharpe']:+.2f}",
        )
    console.print(table)

    by = {r["name"]: r for r in results}
    honest = by["PIT-on (honest)"]
    surv = by["survivor-snapshot"]
    for horizon, key in [("full-sample", "full_ic"), ("OOS 2025+", "oos_ic")]:
        d = surv[key] - honest[key]
        infl = (d / honest[key] * 100) if honest[key] else float("nan")
        console.print(
            f"\n[bold]{horizon} survivorship inflation[/bold] "
            f"(survivor − honest): {surv[key]:+.4f} − {honest[key]:+.4f} = "
            f"{d:+.4f}  ({infl:+.0f}%)"
        )


if __name__ == "__main__":
    main()
