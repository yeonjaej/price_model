"""Prediction store — DuckDB table that every model writes to and every consumer reads from.

Single source of truth. Backtest predictions, walk-forward predictions, and live nightly
predictions all share this schema. The only difference is `generated_at`.

Why DuckDB:
- File-backed (single .duckdb file in artifacts/predictions/), trivial to back up.
- Real SQL — the dashboard can run "what's the IC for model X over the last 60 days?"
  without loading anything into Python.
- Plays well with parquet exports if you want to dump for sharing.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl

DEFAULT_PATH = Path("artifacts/predictions/predictions.duckdb")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    generated_at      TIMESTAMP NOT NULL,
    prediction_date   DATE      NOT NULL,
    target_date       DATE      NOT NULL,
    horizon_days      INTEGER   NOT NULL,
    ticker            VARCHAR   NOT NULL,
    model_id          VARCHAR   NOT NULL,
    experiment_id     VARCHAR   NOT NULL,
    prediction        DOUBLE,
    pred_lower        DOUBLE,
    pred_upper        DOUBLE,
    prediction_kind   VARCHAR   NOT NULL DEFAULT 'excess_return'
);

CREATE INDEX IF NOT EXISTS predictions_lookup
    ON predictions (model_id, prediction_date, ticker);
"""


class PredictionStore:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(SCHEMA_SQL)

    # ----- writes -----

    def write(
        self,
        predictions: pl.DataFrame,
        *,
        model_id: str,
        experiment_id: str,
        horizon_days: int,
        generated_at: datetime | None = None,
        prediction_kind: str = "excess_return",
    ) -> int:
        """Insert a batch of predictions.

        `predictions` must have columns: date (= prediction_date), ticker, prediction.
        Optional: pred_lower, pred_upper.
        """
        if predictions.height == 0:
            return 0
        generated_at = generated_at or datetime.utcnow()

        df = predictions.rename({"date": "prediction_date"})
        # Compute target_date = prediction_date + horizon_days
        df = df.with_columns(
            (pl.col("prediction_date").cast(pl.Date) + pl.duration(days=horizon_days))
            .alias("target_date"),
            pl.lit(generated_at).alias("generated_at"),
            pl.lit(horizon_days).cast(pl.Int32).alias("horizon_days"),
            pl.lit(model_id).alias("model_id"),
            pl.lit(experiment_id).alias("experiment_id"),
            pl.lit(prediction_kind).alias("prediction_kind"),
        )
        # Fill optional cols if missing
        for col in ("pred_lower", "pred_upper"):
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        df = df.select(
            "generated_at",
            "prediction_date",
            "target_date",
            "horizon_days",
            "ticker",
            "model_id",
            "experiment_id",
            "prediction",
            "pred_lower",
            "pred_upper",
            "prediction_kind",
        )
        self._conn.register("df_view", df.to_arrow())
        self._conn.execute("INSERT INTO predictions SELECT * FROM df_view")
        self._conn.unregister("df_view")
        return df.height

    # ----- reads -----

    def query(self, sql: str) -> pl.DataFrame:
        return pl.from_arrow(self._conn.execute(sql).arrow())  # type: ignore[return-value]

    def latest_predictions(
        self,
        model_ids: Iterable[str] | None = None,
        as_of: date | None = None,
    ) -> pl.DataFrame:
        as_of_clause = f"AND prediction_date <= DATE '{as_of}'" if as_of else ""
        model_clause = ""
        if model_ids:
            ids = ", ".join(f"'{m}'" for m in model_ids)
            model_clause = f"AND model_id IN ({ids})"
        sql = f"""
            SELECT *
            FROM predictions p
            WHERE prediction_date = (
                SELECT MAX(prediction_date) FROM predictions p2
                WHERE p2.model_id = p.model_id
                {as_of_clause}
            )
            {model_clause}
            ORDER BY model_id, ticker
        """
        return self.query(sql)

    def list_models(self) -> list[str]:
        df = self.query("SELECT DISTINCT model_id FROM predictions ORDER BY model_id")
        return df["model_id"].to_list() if df.height else []

    def clear_experiment(self, experiment_id: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM predictions WHERE experiment_id = ?", [experiment_id]
        )
        # DuckDB doesn't return affected rows from DELETE in all versions; ignore.
        return 0 if cur is None else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PredictionStore":
        return self

    def __exit__(self, *a) -> None:
        self.close()
