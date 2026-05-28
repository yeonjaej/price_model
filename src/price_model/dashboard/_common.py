"""Shared dashboard helpers — store access + small query wrappers cached per session.

`PredictionStore` is append-only — the walk-forward harness writes one row per
(model_id, prediction_date, ticker) per refit, and re-running an experiment piles
new rows on top of old ones without an upsert. The dashboard wants the *latest*
prediction for each cell, so every page goes through `latest_predictions_sql()`
rather than reading the raw table directly.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st

from price_model.serving.store import PredictionStore


@st.cache_resource
def get_store() -> PredictionStore:
    # Dashboard is read-only — never writes — so connect with read_only=True.
    # This lets the dashboard coexist with a concurrent CLI experiment run
    # (which holds an exclusive write lock by default) and with other
    # dashboard instances. Without this, two browser tabs or a parallel
    # `cli run` will fail with an IOException on DuckDB's file lock.
    return PredictionStore(read_only=True)


def store_health(store: PredictionStore) -> dict[str, Any]:
    n_rows = store.query("SELECT COUNT(*) AS c FROM predictions")["c"][0]
    if n_rows == 0:
        return {
            "n_predictions": 0,
            "n_models": 0,
            "n_tickers": 0,
            "latest_date": None,
        }
    summary = store.query(
        """
        SELECT
            COUNT(*) AS n_predictions,
            COUNT(DISTINCT model_id) AS n_models,
            COUNT(DISTINCT ticker) AS n_tickers,
            MAX(prediction_date) AS latest_date
        FROM predictions
        """
    )
    row = summary.row(0, named=True)
    return {
        "n_predictions": int(row["n_predictions"]),
        "n_models": int(row["n_models"]),
        "n_tickers": int(row["n_tickers"]),
        "latest_date": row["latest_date"],
    }


@st.cache_data(ttl=60)
def list_models(_store_path: str) -> list[str]:
    store = get_store()
    return store.list_models()


@st.cache_data(ttl=60)
def query_df(_store_path: str, sql: str) -> pl.DataFrame:
    """Cached SQL pass-through. _store_path is in the cache key so reloads invalidate."""
    store = get_store()
    return store.query(sql)


def latest_predictions_sql(where: str = "1=1") -> str:
    """SQL CTE returning one row per (model_id, prediction_date, ticker).

    Resolves the append-only-store duplicate problem: for each cell we keep the
    row with the largest `generated_at`, dropping older runs / refits / re-runs.

    `where` is interpolated into the inner SELECT — callers can filter by model
    or date range there. Output columns: model_id, prediction_date, ticker,
    prediction, generated_at.
    """
    return f"""
        WITH dedup AS (
            SELECT
                model_id,
                prediction_date,
                ticker,
                prediction,
                generated_at,
                ROW_NUMBER() OVER (
                    PARTITION BY model_id, prediction_date, ticker
                    ORDER BY generated_at DESC
                ) AS rn
            FROM predictions
            WHERE {where}
        )
        SELECT model_id, prediction_date, ticker, prediction, generated_at
        FROM dedup
        WHERE rn = 1
    """
