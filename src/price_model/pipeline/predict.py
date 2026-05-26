"""Live prediction: take the latest data, run a fitted model, write to the store.

Designed for the nightly job. Loads a model artifact rather than refitting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from price_model.data.loaders import load_panel
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import MODEL_REGISTRY
from price_model.serving.store import PredictionStore

log = logging.getLogger(__name__)


def predict_today(
    *,
    model_id: str,
    model_class: str,
    experiment_id: str,
    universe: str,
    feature_names: list[str],
    horizon_days: int,
    start: str,
    artifact_dir: Path = Path("artifacts/models"),
    store_path: Path | None = None,
) -> pl.DataFrame:
    """Load a saved model, compute today's features, write predictions to the store."""
    model_path = artifact_dir / model_id
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")

    model_cls = MODEL_REGISTRY[model_class]
    model = model_cls.load(model_path)

    panel = load_panel(universe=universe, start=start)
    matrix = build_feature_matrix(panel, feature_names, target_horizon=horizon_days)
    matrix = drop_warmup_rows(matrix, feature_names)

    latest_date = matrix["date"].max()
    today = matrix.filter(pl.col("date") == pl.lit(latest_date))
    preds = model.predict(today)

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
