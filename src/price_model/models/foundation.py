"""Chronos zero-shot foundation model for time-series forecasting.

Chronos (Amazon, 2024) is a pretrained encoder-decoder model trained on a large
corpus of time series. We use it zero-shot: feed in a per-ticker price history,
get a probabilistic forecast h days ahead, convert to a cross-sectional excess
return prediction so it's comparable to our other models.

Practical notes:
- `fit()` is a no-op (model is pretrained). It only captures historical context.
- First `predict()` downloads weights from HuggingFace (~30MB for chronos-t5-tiny,
  ~250MB for chronos-t5-small).
- Inference is slow on CPU — chronos-t5-tiny processes ~20 tickers/sec/date.
  For a full backtest use a GPU and/or restrict to a small universe.
- Requires `chronos-forecasting`, `torch`, and `transformers`. Install via:
  `pip install ".[chronos]"`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from price_model.models.base import Model, ModelConfig, load_config, save_config

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "model_name": "amazon/chronos-t5-tiny",  # smallest; bump to -small for accuracy
    "context_length": 256,                    # trailing days fed as context
    "num_samples": 20,                        # samples for probabilistic forecast
    "prediction_length": 5,                   # must match the target horizon
    "device": "cpu",                          # "cuda" if you have a GPU
    "log_every_n_dates": 25,                  # progress logging cadence
}


class ChronosZeroShot(Model):
    """Pretrained foundation-model wrapper conforming to the Model ABC."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._pipeline = None
        self._history: pl.DataFrame | None = None  # (date, ticker, adj_close)

    def _params(self) -> dict[str, Any]:
        return {**DEFAULT_PARAMS, **self.config.params}

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "Chronos is an optional dependency. Install with:\n"
                "    pip install '.[chronos]'\n"
                "(this pulls in torch, transformers, and chronos-forecasting)."
            ) from e

        params = self._params()
        device = params["device"]
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("Loading Chronos pipeline %s on %s", params["model_name"], device)
        self._pipeline = ChronosPipeline.from_pretrained(
            params["model_name"],
            device_map=device,
            torch_dtype=dtype,
        )

    def fit(self, panel: pl.DataFrame) -> None:
        """No training — only capture history for context lookup at predict time."""
        if "adj_close" not in panel.columns:
            raise ValueError("ChronosZeroShot requires 'adj_close' in the input panel")
        self._history = (
            panel.select("date", "ticker", "adj_close").sort(["ticker", "date"])
        )
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        self._ensure_pipeline()
        import torch

        if "adj_close" not in panel.columns:
            raise ValueError("ChronosZeroShot requires 'adj_close' in the predict panel")

        params = self._params()
        ctx_len = int(params["context_length"])
        num_samples = int(params["num_samples"])
        h = int(params["prediction_length"])
        log_every = int(params["log_every_n_dates"])

        # Combine stored history with anything new in the predict panel so context
        # stays fresh across dates within a single walk-forward window.
        combined = (
            pl.concat([
                self._history if self._history is not None else panel.select(
                    "date", "ticker", "adj_close"
                ),
                panel.select("date", "ticker", "adj_close"),
            ])
            .unique(subset=["date", "ticker"])
            .sort(["ticker", "date"])
        )

        # Per-ticker indexed history for fast slicing
        history_by_ticker: dict[str, pl.DataFrame] = {}
        for key, grp in combined.group_by("ticker"):
            t = key[0] if isinstance(key, tuple) else key
            history_by_ticker[t] = grp.sort("date")

        out_rows: list[dict[str, Any]] = []
        unique_dates = panel["date"].unique().sort().to_list()
        n_dates = len(unique_dates)

        for d_idx, d in enumerate(unique_dates):
            target_tickers = (
                panel.filter(pl.col("date") == pl.lit(d))["ticker"].to_list()
            )
            contexts: list[torch.Tensor] = []
            valid_tickers: list[str] = []
            current_prices: list[float] = []

            for t in target_tickers:
                hist = history_by_ticker.get(t)
                if hist is None:
                    continue
                hist_up_to_d = hist.filter(pl.col("date") <= pl.lit(d))
                if hist_up_to_d.height < 20:  # need some context
                    continue
                prices = hist_up_to_d["adj_close"].to_list()
                if ctx_len > 0:
                    prices = prices[-ctx_len:]
                contexts.append(torch.tensor(prices, dtype=torch.float32))
                valid_tickers.append(t)
                current_prices.append(float(prices[-1]))

            if not contexts:
                continue

            # forecast: [n_tickers, num_samples, h]
            forecast = self._pipeline.predict(  # type: ignore[union-attr]
                context=contexts,
                prediction_length=h,
                num_samples=num_samples,
            )
            # Median forecast at the end of the horizon
            median_forecast_h = forecast[:, :, -1].quantile(0.5, dim=1).cpu().numpy()
            q10 = forecast[:, :, -1].quantile(0.1, dim=1).cpu().numpy()
            q90 = forecast[:, :, -1].quantile(0.9, dim=1).cpu().numpy()

            cur = np.asarray(current_prices, dtype=np.float64)
            preds_raw = np.log(median_forecast_h / cur)
            lower_raw = np.log(np.clip(q10, 1e-9, None) / cur)
            upper_raw = np.log(np.clip(q90, 1e-9, None) / cur)

            # Cross-sectional excess: subtract today's mean prediction
            xs_mean = preds_raw.mean()
            preds = preds_raw - xs_mean
            lower = lower_raw - xs_mean
            upper = upper_raw - xs_mean

            for t, p, lo, hi in zip(valid_tickers, preds, lower, upper):
                out_rows.append({
                    "date": d, "ticker": t,
                    "prediction": float(p),
                    "pred_lower": float(lo),
                    "pred_upper": float(hi),
                })

            if log_every > 0 and (d_idx + 1) % log_every == 0:
                log.info("Chronos progress: %d/%d dates", d_idx + 1, n_dates)

        if not out_rows:
            return pl.DataFrame(
                schema={
                    "date": pl.Date, "ticker": pl.Utf8,
                    "prediction": pl.Float64,
                    "pred_lower": pl.Float64, "pred_upper": pl.Float64,
                }
            )
        return pl.DataFrame(out_rows).sort(["date", "ticker"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")

    @classmethod
    def load(cls, path: Path) -> "ChronosZeroShot":
        model = cls(load_config(path / "config.json"))
        model._fitted = True
        return model
