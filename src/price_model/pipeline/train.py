"""Train one model on the most recent slice of data, save artifact, and write today's
prediction.

Used for live deployment (separate from the walk-forward harness, which is for
backtesting). The contract: read config -> load panel -> build features -> train ->
predict on today's data -> write to store.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from price_model.data.loaders import load_panel
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.serving.store import PredictionStore

log = logging.getLogger(__name__)


def train_and_predict(
    *,
    experiment_id: str,
    model_id: str,
    model_class: str,
    universe: str,
    feature_names: list[str],
    horizon_days: int,
    start: date,
    train_through: date | None = None,
    predict_universe: str | None = None,
    params: dict[str, Any] | None = None,
    artifact_dir: Path = Path("artifacts/models"),
    store_path: Path | None = None,
) -> pl.DataFrame:
    """Fit a model on (start, train_through] and emit predictions for the latest date.

    If `predict_universe` is given, predictions are emitted for that universe instead of
    `universe` — useful for "train on S&P 500, predict on top 20".
    """
    panel = load_panel(universe=universe, start=start)
    matrix = build_feature_matrix(panel, feature_names, target_horizon=horizon_days)
    matrix = drop_warmup_rows(matrix, feature_names)

    if train_through is not None:
        train = matrix.filter(pl.col("date") <= pl.lit(train_through))
    else:
        train = matrix.drop_nulls(subset=["y"])

    config = ModelConfig(
        model_id=model_id,
        feature_cols=tuple(feature_names),
        target_col="y",
        params=params or {},
    )
    model = build_model(model_class, config)
    log.info("Training %s on %d rows", model_id, train.height)
    model.fit(train)

    if predict_universe and predict_universe != universe:
        pred_panel_raw = load_panel(universe=predict_universe, start=start)
        # Build features using the predict-universe panel — the normalization step
        # will be against THAT cross-section. Document this: if you want predictions
        # normalized against the broader S&P 500 cross-section, predict against the
        # joint panel and then filter.
        pred_matrix = build_feature_matrix(
            pred_panel_raw, feature_names, target_horizon=horizon_days
        )
        pred_matrix = drop_warmup_rows(pred_matrix, feature_names)
    else:
        pred_matrix = matrix

    latest_date = pred_matrix["date"].max()
    today_slice = pred_matrix.filter(pl.col("date") == pl.lit(latest_date))
    preds = model.predict(today_slice)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / model_id)

    store = PredictionStore(store_path) if store_path else PredictionStore()
    try:
        store.write(
            preds,
            model_id=model_id,
            experiment_id=experiment_id,
            horizon_days=horizon_days,
            generated_at=datetime.utcnow(),
        )
    finally:
        store.close()

    return preds
