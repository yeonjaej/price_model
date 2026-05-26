"""Model abstraction.

A Model is anything that can be `fit` on a panel and asked to `predict` on a panel.
The contract is intentionally minimal: that's what makes ensembles, evaluation, and
the prediction-writing pipeline polymorphic in the model.

Predictions are returned as a long-format polars DataFrame:
    (date, ticker, prediction)        # point estimate (e.g. excess return)
Optionally also pred_lower, pred_upper for probabilistic models.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class ModelConfig:
    """Minimal model config shared by every model."""

    model_id: str  # e.g. "lightgbm_v1"
    feature_cols: Sequence[str]
    target_col: str = "y"
    params: dict[str, Any] = field(default_factory=dict)


class Model(ABC):
    """Abstract base for every model in the project."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._fitted: bool = False

    @abstractmethod
    def fit(self, panel: pl.DataFrame) -> None:
        """Fit on a panel containing feature columns + target column."""
        ...

    @abstractmethod
    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Predict on a panel. Returns (date, ticker, prediction) at minimum."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Model: ...

    # ------- shared helpers -------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} not fitted")

    def _format_predictions(
        self,
        panel: pl.DataFrame,
        predictions,
        lower=None,
        upper=None,
    ) -> pl.DataFrame:
        out = panel.select(["date", "ticker"]).with_columns(
            pl.Series("prediction", predictions, dtype=pl.Float64)
        )
        if lower is not None:
            out = out.with_columns(pl.Series("pred_lower", lower, dtype=pl.Float64))
        if upper is not None:
            out = out.with_columns(pl.Series("pred_upper", upper, dtype=pl.Float64))
        return out


def save_config(config: ModelConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "model_id": config.model_id,
                "feature_cols": list(config.feature_cols),
                "target_col": config.target_col,
                "params": config.params,
            },
            f,
            indent=2,
        )


def load_config(path: Path) -> ModelConfig:
    with path.open() as f:
        d = json.load(f)
    return ModelConfig(
        model_id=d["model_id"],
        feature_cols=tuple(d["feature_cols"]),
        target_col=d["target_col"],
        params=d.get("params", {}),
    )
