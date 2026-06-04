"""Equal-weight rank-average ensemble of multiple models.

Why this exists
---------------
Data-driven stackers (Ridge, NNLS, time-varying softmax) all failed
on this project — see `scripts/ridge_stack_ensemble.py`,
`scripts/nnls_stack_ensemble.py`, `scripts/walk_forward_ensemble.py`.
The failure mode in every case was train-eval regime mismatch: weights
fit to historical regimes did not transfer to the eval slice.

A rank-average ensemble has the opposite property: ZERO parameters.
It cannot overfit by construction. The theoretical motivation is the
standard variance-reduction result — if k models share the same
direction of information (positive correlation with the target) and
have independent noise, equal-weight averaging reduces noise variance
by a factor of 1/k while preserving the mean signal. The break-even
condition is that the models be positively correlated with the target;
beyond that, the ensemble can only help or be neutral.

For long-horizon momentum factors (mom_378, mom_504, mom_756) on
2025+, the per-factor ICs are +0.026 to +0.050 (all the same sign).
The condition is satisfied. The ensemble should produce IC at least
as high as the best constituent, and typically slightly higher due to
noise cancellation between factors.

Usage
-----
    PYTHONPATH=src python scripts/rank_average_ensemble.py \\
        --models mom_378_factor mom_504_factor mom_756_factor \\
        --since 2025-01-01
    PYTHONPATH=src python scripts/rank_average_ensemble.py \\
        --models mom_378_factor mom_504_factor mom_756_factor \\
        --since 2025-01-01 --horizon 21
"""

from __future__ import annotations

import argparse
from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.metrics import summarize
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

DEFAULT_MODEL_IDS = ["mom_378_factor", "mom_504_factor", "mom_756_factor"]


def _load_predictions(store: PredictionStore, model_ids: list[str]) -> pl.DataFrame:
    """Pull (date, ticker, model_id, prediction) for the given model ids."""
    ids = ", ".join(f"'{m}'" for m in model_ids)
    sql = f"""
        SELECT prediction_date AS date, ticker, model_id, prediction
        FROM predictions
        WHERE model_id IN ({ids})
    """
    return store.query(sql)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=DEFAULT_MODEL_IDS,
        help="Model IDs to ensemble. Default: top-3 long-horizon momentum.",
    )
    ap.add_argument(
        "--since", default=None,
        help="Optional inclusive lower bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--until", default=None,
        help="Optional exclusive upper bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--horizon", type=int, default=5,
        help="Forward horizon (days) for realized target.",
    )
    args = ap.parse_args()

    console = Console()
    cutoff_since: date | None = (
        date.fromisoformat(args.since) if args.since else None
    )
    cutoff_until: date | None = (
        date.fromisoformat(args.until) if args.until else None
    )

    # 1. Load predictions for the constituent models.
    with PredictionStore(read_only=True) as store:
        available = set(store.list_models())
        target_ids = [m for m in args.models if m in available]
        missing = [m for m in args.models if m not in available]
        if missing:
            console.print(f"[yellow]Missing from store (skipping):[/yellow] {missing}")
        if len(target_ids) < 2:
            console.print("[red]Need at least 2 models in the store to ensemble.")
            return
        preds = _load_predictions(store, target_ids)
    console.print(
        f"Loaded {preds.height:,} prediction rows across "
        f"{len(target_ids)} constituents: {target_ids}"
    )

    # 2. Intersection of dates across constituents.
    by_model = preds.group_by("model_id").agg(pl.col("date").unique().alias("dates"))
    per_model_dates = [set(row["dates"]) for row in by_model.iter_rows(named=True)]
    common_dates = set.intersection(*per_model_dates) if per_model_dates else set()
    if cutoff_since:
        common_dates = {d for d in common_dates if d >= cutoff_since}
    if cutoff_until:
        common_dates = {d for d in common_dates if d < cutoff_until}
    if not common_dates:
        console.print("[red]Date intersection is empty.")
        return
    common_dates_sorted = sorted(common_dates)
    console.print(
        f"Eval slice: {len(common_dates_sorted):,} dates "
        f"[{common_dates_sorted[0]} → {common_dates_sorted[-1]}]"
    )
    preds = preds.filter(pl.col("date").is_in(list(common_dates)))

    # 3. Restrict to the (date, ticker) INTERSECTION across all constituents.
    # Without this filter, the ensemble averages partial-coverage rows where
    # only 2 of N constituents have a prediction — biasing IC downward when
    # the missing constituent is the strongest one (e.g. mom_756, which has
    # a 756-day warmup and thus the smallest universe). All constituent and
    # ensemble ICs reported below are on the SAME (date, ticker) set.
    counts_per_pair = (
        preds.group_by(["date", "ticker"])
        .agg(pl.col("model_id").n_unique().alias("n_constituents"))
    )
    full_coverage = (
        counts_per_pair.filter(pl.col("n_constituents") == len(target_ids))
        .drop("n_constituents")
    )
    preds = preds.join(full_coverage, on=["date", "ticker"], how="inner")
    console.print(
        f"After restricting to full-coverage (date, ticker) pairs: "
        f"{preds.height:,} rows."
    )

    # 4. Compute per-(date, model_id) cross-sectional rank, then average ranks
    # per (date, ticker) — that average rank IS the ensemble's prediction.
    # Ranking uses average method (fractional ranks for ties), matching the
    # HHRZ / typical cross-sectional-factor convention.
    ranked = preds.with_columns(
        pl.col("prediction")
        .rank(method="average")
        .over(["date", "model_id"])
        .alias("rank")
    )
    ensemble_preds = (
        ranked.group_by(["date", "ticker"])
        .agg(pl.col("rank").mean().alias("prediction"))
        .sort(["date", "ticker"])
    )

    # 4. Load realized targets and join.
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select(["date", "ticker", pl.col("y").alias("realized")])
    realized = realized.filter(pl.col("date").is_in(list(common_dates)))
    eval_df = ensemble_preds.join(realized, on=["date", "ticker"], how="inner")
    console.print(
        f"After join to realized targets: {eval_df.height:,} rows across "
        f"{eval_df['date'].n_unique():,} dates."
    )

    # 5. Compute IC for the ensemble.
    ensemble_summary = summarize(eval_df, horizon_days=args.horizon).as_dict()

    # 6. Also compute IC for each constituent on the SAME slice, so we can
    # quantify the ensemble's lift over its best constituent. This is the
    # "did averaging help" test.
    constituent_rows = []
    for mid in target_ids:
        sub = preds.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction"
        )
        sub = sub.join(realized, on=["date", "ticker"], how="inner")
        if sub.height == 0:
            continue
        summary = summarize(sub, horizon_days=args.horizon).as_dict()
        summary["model_id"] = mid
        constituent_rows.append(summary)
    constituent_rows.append({**ensemble_summary, "model_id": "ENSEMBLE_rank_avg"})

    result = (
        pl.DataFrame(constituent_rows)
        .select(
            "model_id",
            "n_observations",
            "n_dates",
            "information_coefficient",
            "ic_t_stat",
            "hit_rate",
            "long_short_sharpe",
        )
        .sort("information_coefficient", descending=True)
    )

    # 7. Print.
    window_parts = []
    if args.since:
        window_parts.append(f"since {args.since}")
    if args.until:
        window_parts.append(f"until {args.until}")
    window_desc = ", ".join(window_parts) if window_parts else "full sample"
    horizon_desc = f"horizon={args.horizon}d"
    rule_label = (
        f"[bold green]Rank-average ensemble vs constituents "
        f"({window_desc}, {horizon_desc})"
    )
    console.rule(rule_label)
    table = Table()
    for col in result.columns:
        table.add_column(col)
    for row in result.iter_rows():
        table.add_row(
            *[f"{v:+.4f}" if isinstance(v, float) else str(v) for v in row]
        )
    console.print(table)

    # 8. The headline number: ensemble IC minus best-constituent IC.
    ensemble_ic = ensemble_summary["information_coefficient"]
    best_constituent_ic = max(
        r["information_coefficient"] for r in constituent_rows
        if r["model_id"] != "ENSEMBLE_rank_avg"
    )
    lift = ensemble_ic - best_constituent_ic
    relative = lift / abs(best_constituent_ic) if best_constituent_ic != 0 else 0
    console.print(
        f"\n[bold]Ensemble lift over best constituent:[/bold] "
        f"{lift:+.4f} IC ({relative:+.1%} relative)"
    )
    if lift > 0:
        console.print(
            "[green]Ensemble strictly improves on its best constituent — "
            "noise-reduction was material.[/green]"
        )
    else:
        console.print(
            "[yellow]Ensemble does NOT improve on its best constituent — "
            "either constituents share too much noise, or one constituent "
            "carries all the signal and averaging dilutes it.[/yellow]"
        )


if __name__ == "__main__":
    main()
