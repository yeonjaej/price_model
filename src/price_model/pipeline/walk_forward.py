"""Walk-forward training and prediction.

Pseudo-code:

    for split in walk_forward_splits(...):
        train_panel = slice_train(panel, split)
        train_panel = drop_warmup_rows + drop_null_target rows
        model = build_model(...)
        model.fit(train_panel)
        test_panel = slice_test(panel, split)
        preds = model.predict(test_panel)
        store.write(preds, model_id=..., experiment_id=...)

We do this once per (model, split), so every model sees identical splits. Comparisons
downstream are honest.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime

import polars as pl

from price_model.data.splits import Split, slice_test, slice_train, walk_forward_splits
from price_model.models.base import Model
from price_model.serving.store import PredictionStore

log = logging.getLogger(__name__)


def run_walk_forward(
    panel: pl.DataFrame,
    *,
    model: Model,
    feature_cols: Sequence[str],
    target_col: str,
    experiment_id: str,
    horizon_days: int,
    refit_freq_days: int = 21,
    embargo_days: int = 6,
    min_train_days: int = 252 * 2,
    store: PredictionStore | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Train + predict in a walking window. Returns a frame of all out-of-sample predictions.

    Also writes predictions to `store` if provided.
    """
    panel = panel.sort(["ticker", "date"])
    actual_start = start or panel["date"].min()
    actual_end = end or panel["date"].max()
    if actual_start is None or actual_end is None:
        raise ValueError("Panel has no dates")

    if embargo_days < horizon_days:
        raise ValueError(f"embargo_days ({embargo_days}) must be >= horizon_days ({horizon_days})")

    all_preds: list[pl.DataFrame] = []
    splits: list[Split] = list(
        walk_forward_splits(
            start=actual_start,
            end=actual_end,
            refit_freq_days=refit_freq_days,
            embargo_days=embargo_days,
            min_train_days=min_train_days,
        )
    )
    log.info("Running %d walk-forward splits for model %s", len(splits), model.config.model_id)

    for i, split in enumerate(splits):
        train = slice_train(panel, split).drop_nulls(subset=[target_col, *feature_cols])
        if train.height == 0:
            log.warning("Split %d has no training rows after dropna; skipping", i)
            continue
        # Refresh a fresh model instance via the same class+config — but to keep the
        # interface tiny we just call .fit again on the same object. For stateful
        # models this is fine if `fit` overwrites internal state; for LightGBM it does.
        model.fit(train)

        test = slice_test(panel, split).drop_nulls(subset=feature_cols)
        if test.height == 0:
            continue
        preds = model.predict(test)
        all_preds.append(preds)

        if store is not None:
            store.write(
                preds,
                model_id=model.config.model_id,
                experiment_id=experiment_id,
                horizon_days=horizon_days,
                generated_at=datetime.combine(split.refit_date, datetime.min.time()),
            )

    if not all_preds:
        return pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "prediction": pl.Float64})
    return pl.concat(all_preds).sort(["date", "ticker"])


def join_with_realized(
    predictions: pl.DataFrame,
    panel_with_target: pl.DataFrame,
    target_col: str = "y",
) -> pl.DataFrame:
    """Attach realized targets to predictions for evaluation.

    Returns columns: (date, ticker, prediction, realized).
    """
    realized = panel_with_target.select("date", "ticker", pl.col(target_col).alias("realized"))
    return predictions.join(realized, on=["date", "ticker"], how="left")
