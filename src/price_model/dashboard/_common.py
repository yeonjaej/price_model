"""Shared dashboard helpers — store access + small query wrappers cached per session."""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st

from price_model.serving.store import PredictionStore


@st.cache_resource
def get_store() -> PredictionStore:
    return PredictionStore()


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
