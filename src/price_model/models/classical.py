"""Classical time-series baselines — the textbook lineage.

Three univariate, per-ticker models that fit on each stock's own log-return series:

- **ArimaPerTicker**: ARIMA(p,d,q). Box-Jenkins lineage. Predicts forward log-return
  h steps ahead, then converts to a cross-sectional excess return by subtracting the
  date's mean prediction across tickers. Default order (1,0,1) — AR(1) + MA(1) on
  already-stationary log-returns (no differencing).

- **GarchVolForecaster**: GARCH(1,1). Models volatility clustering. *Not* a return
  forecaster — the prediction is 0 (mean='Zero') and the value sits in `pred_lower` /
  `pred_upper` as a ±2σ interval from the conditional variance forecast. Included to
  demonstrate understanding of ARCH effects and as a baseline that's honest about
  what GARCH does and doesn't do.

- **GbmMaximumLikelihood**: parametric Geometric Brownian Motion. Closed-form MLE
  on log-returns: μ_hat = sample mean, σ_hat = sample std. h-step-ahead expectation:
  (μ − σ²/2)·h, with a ±2σ√h interval. Subtracted to cross-sectional excess to be
  comparable with other models. The continuous-time framework underneath Black-Scholes.

All three use only the existing yfinance panel (adj_close → log-returns). Optional
dependencies are pulled in via `pip install ".[classical]"`:

- statsmodels (for ARIMA)
- arch (for GARCH; Kevin Sheppard's package, the de facto standard)

The modules import cleanly without these optional deps; the ImportError is deferred
to `fit()` so adding these classes to MODEL_REGISTRY doesn't break the core install.

Speed notes: ARIMA/GARCH per-ticker fits cost a few seconds each. With 156 tickers
and the default monthly-refit walk-forward harness, that's hours. For full backtests,
use the `classical.yaml` experiment which restricts to the 20-name deployment universe
and refits annually instead of monthly.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from price_model.models.base import Model, ModelConfig, load_config, save_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _log_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """Return panel sorted by (ticker, date) with a `log_return` column added."""
    c = pl.col("adj_close")
    return panel.sort(["ticker", "date"]).with_columns(
        (c.log() - c.log().shift(1)).over("ticker").alias("log_return")
    )


def _to_cross_sectional_excess(rows: list[dict]) -> list[dict]:
    """Subtract per-date mean prediction so the output is a cross-sectional excess
    return — comparable to LightGBM's target and to forward_excess_return realized."""
    if not rows:
        return rows
    by_date: dict = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    out: list[dict] = []
    for _date, drows in by_date.items():
        mean_pred = sum(r["prediction"] for r in drows) / len(drows)
        for r in drows:
            new_r = dict(r)
            new_r["prediction"] = r["prediction"] - mean_pred
            out.append(new_r)
    return out


def _per_ticker_series(panel: pl.DataFrame, ticker: str) -> np.ndarray:
    """Sorted log-return numpy array for a given ticker (NaNs dropped)."""
    sub = panel.filter(pl.col("ticker") == ticker).sort("date").drop_nulls("log_return")
    return sub["log_return"].to_numpy()


# ---------------------------------------------------------------------------
# ARIMA per ticker
# ---------------------------------------------------------------------------

DEFAULT_ARIMA_PARAMS: dict[str, Any] = {
    "order": (1, 0, 1),  # AR(1) + MA(1); d=0 because log-returns are stationary
    "trend": "c",
    "min_history": 252,  # ~1 year before fitting
    "horizon_days": 5,  # forecast horizon in trading days
}


class ArimaPerTicker(Model):
    """ARIMA(p,d,q) per ticker on log-returns; predicts cross-sectional excess return.

    Per EMH expectations, IC should be ≈ 0 on liquid mega-caps. That's the point of
    including it — a defensible baseline that says "past returns don't predict future
    returns."
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._fits: dict[str, Any] = {}

    def _params(self) -> dict[str, Any]:
        return {**DEFAULT_ARIMA_PARAMS, **self.config.params}

    def fit(self, panel: pl.DataFrame) -> None:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ArimaPerTicker requires statsmodels. Install with: pip install '.[classical]'"
            ) from e

        params = self._params()
        order = tuple(params["order"])
        trend = params["trend"]
        min_history = int(params["min_history"])

        panel = _log_returns(panel)
        tickers = panel["ticker"].unique().to_list()
        self._fits = {}
        for ticker in tickers:
            series = _per_ticker_series(panel, ticker)
            if len(series) < min_history:
                continue
            try:
                model = ARIMA(series, order=order, trend=trend)
                result = model.fit(method_kwargs={"warn_convergence": False})
                self._fits[ticker] = result
            except Exception as e:
                log.warning("ARIMA failed for %s: %s", ticker, e)
        self._fitted = True
        log.info(
            "ArimaPerTicker fitted %d/%d tickers (order=%s)",
            len(self._fits),
            len(tickers),
            order,
        )

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        h = int(self._params()["horizon_days"])

        rows: list[dict] = []
        unique_tickers = panel["ticker"].unique().to_list()
        for ticker in unique_tickers:
            result = self._fits.get(ticker)
            if result is None:
                continue
            try:
                forecast = result.forecast(steps=h)
                forward_logret = float(np.sum(forecast))
            except Exception as e:
                log.warning("ARIMA forecast failed for %s: %s", ticker, e)
                continue
            for d in panel.filter(pl.col("ticker") == ticker)["date"].unique().to_list():
                rows.append({"date": d, "ticker": ticker, "prediction": forward_logret})

        rows = _to_cross_sectional_excess(rows)
        if not rows:
            return pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.Utf8, "prediction": pl.Float64}
            )
        return pl.DataFrame(rows).sort(["date", "ticker"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "fits.pkl").open("wb") as f:
            pickle.dump(self._fits, f)

    @classmethod
    def load(cls, path: Path) -> ArimaPerTicker:
        m = cls(load_config(path / "config.json"))
        fits_path = path / "fits.pkl"
        if fits_path.exists():
            with fits_path.open("rb") as f:
                m._fits = pickle.load(f)
            m._fitted = True
        return m


# ---------------------------------------------------------------------------
# GARCH per ticker — forward volatility forecaster
# ---------------------------------------------------------------------------

DEFAULT_GARCH_PARAMS: dict[str, Any] = {
    "p": 1,
    "q": 1,
    "mean": "Zero",
    "min_history": 252,
    "horizon_days": 5,
}


class GarchVolForecaster(Model):
    """GARCH(1,1) per ticker on log-returns. Forecasts forward volatility.

    Important: this is a *volatility* model, not a *return* model. Point prediction
    is 0; the value is in `pred_lower` / `pred_upper`, a ±2σ interval where σ is the
    forecast forward total-vol over the horizon.

    Use cases:
    - Demonstrates understanding of ARCH effects + volatility clustering.
    - Honest baseline: GARCH IC for return prediction ≈ 0 by construction.
    - The forecast forward vol can be lifted into a Feature for LightGBM.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._fits: dict[str, Any] = {}

    def _params(self) -> dict[str, Any]:
        return {**DEFAULT_GARCH_PARAMS, **self.config.params}

    def fit(self, panel: pl.DataFrame) -> None:
        try:
            from arch import arch_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "GarchVolForecaster requires arch. Install with: pip install '.[classical]'"
            ) from e

        params = self._params()
        p, q = int(params["p"]), int(params["q"])
        mean = params["mean"]
        min_history = int(params["min_history"])

        panel = _log_returns(panel)
        tickers = panel["ticker"].unique().to_list()
        self._fits = {}
        for ticker in tickers:
            series = _per_ticker_series(panel, ticker)
            if len(series) < min_history:
                continue
            try:
                # arch convention: pass returns in percent for numerical stability
                am = arch_model(series * 100.0, p=p, q=q, mean=mean, vol="GARCH")
                result = am.fit(disp="off", show_warning=False)
                self._fits[ticker] = result
            except Exception as e:
                log.warning("GARCH failed for %s: %s", ticker, e)
        self._fitted = True
        log.info(
            "GarchVolForecaster fitted %d/%d tickers (p=%d, q=%d)",
            len(self._fits),
            len(tickers),
            p,
            q,
        )

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        h = int(self._params()["horizon_days"])

        rows: list[dict] = []
        for ticker in panel["ticker"].unique().to_list():
            result = self._fits.get(ticker)
            if result is None:
                continue
            try:
                forecast = result.forecast(horizon=h, reindex=False)
                var_path = forecast.variance.values[0]
                # arch returned variances are in (percent return)^2 — divide by 1e4
                total_var = float(np.sum(var_path)) / 1e4
                std = math.sqrt(max(total_var, 0.0))
            except Exception as e:
                log.warning("GARCH forecast failed for %s: %s", ticker, e)
                continue
            for d in panel.filter(pl.col("ticker") == ticker)["date"].unique().to_list():
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "prediction": 0.0,
                        "pred_lower": -2.0 * std,
                        "pred_upper": 2.0 * std,
                    }
                )

        if not rows:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "ticker": pl.Utf8,
                    "prediction": pl.Float64,
                    "pred_lower": pl.Float64,
                    "pred_upper": pl.Float64,
                }
            )
        return pl.DataFrame(rows).sort(["date", "ticker"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "fits.pkl").open("wb") as f:
            pickle.dump(self._fits, f)

    @classmethod
    def load(cls, path: Path) -> GarchVolForecaster:
        m = cls(load_config(path / "config.json"))
        fits_path = path / "fits.pkl"
        if fits_path.exists():
            with fits_path.open("rb") as f:
                m._fits = pickle.load(f)
            m._fitted = True
        return m


# ---------------------------------------------------------------------------
# Geometric Brownian Motion via MLE
# ---------------------------------------------------------------------------

DEFAULT_GBM_PARAMS: dict[str, Any] = {
    "min_history": 252,
    "horizon_days": 5,
}


class GbmMaximumLikelihood(Model):
    """Parametric GBM (Geometric Brownian Motion) fit per ticker via MLE.

    Continuous-time model:  dS_t = μ S_t dt + σ S_t dW_t
    Closed-form MLE on log-returns:
        μ_hat = mean(log_returns)
        σ_hat = std (log_returns, ddof=1)
    h-step-ahead expectation of log forward return: (μ − σ²/2) · h
    Standard deviation of that forecast: σ · √h

    The IC of GBM-as-return-predictor on liquid mega-caps should be ≈ 0 — historical
    drift estimates μ_hat are too noisy to predict cross-sectional rankings. The
    point of including it is to demonstrate the Black-Scholes machinery and provide
    an honest parametric baseline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._fits: dict[str, tuple[float, float]] = {}

    def _params(self) -> dict[str, Any]:
        return {**DEFAULT_GBM_PARAMS, **self.config.params}

    def fit(self, panel: pl.DataFrame) -> None:
        params = self._params()
        min_history = int(params["min_history"])

        panel = _log_returns(panel)
        self._fits = {}
        for ticker in panel["ticker"].unique().to_list():
            series = _per_ticker_series(panel, ticker)
            if len(series) < min_history:
                continue
            mu = float(np.mean(series))
            sigma = float(np.std(series, ddof=1))
            self._fits[ticker] = (mu, sigma)
        self._fitted = True
        log.info("GbmMaximumLikelihood fitted %d tickers", len(self._fits))

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        h = int(self._params()["horizon_days"])

        rows: list[dict] = []
        for ticker in panel["ticker"].unique().to_list():
            fit = self._fits.get(ticker)
            if fit is None:
                continue
            mu, sigma = fit
            point = (mu - 0.5 * sigma * sigma) * h
            std = sigma * math.sqrt(h)
            for d in panel.filter(pl.col("ticker") == ticker)["date"].unique().to_list():
                rows.append(
                    {
                        "date": d,
                        "ticker": ticker,
                        "prediction": point,
                        "pred_lower": point - 2.0 * std,
                        "pred_upper": point + 2.0 * std,
                    }
                )

        # Convert raw drift predictions → cross-sectional excess returns.
        # Note: we don't adjust pred_lower/pred_upper here; they remain in the
        # *raw* log-return scale and represent the GBM uncertainty around the
        # absolute forward return, not the excess. That's appropriate because
        # the interval is a property of the per-stock model, not the cross-section.
        rows = _to_cross_sectional_excess(rows)
        if not rows:
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "ticker": pl.Utf8,
                    "prediction": pl.Float64,
                    "pred_lower": pl.Float64,
                    "pred_upper": pl.Float64,
                }
            )
        return pl.DataFrame(rows).sort(["date", "ticker"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "fits.json").open("w") as f:
            json.dump({t: list(p) for t, p in self._fits.items()}, f)

    @classmethod
    def load(cls, path: Path) -> GbmMaximumLikelihood:
        m = cls(load_config(path / "config.json"))
        fits_path = path / "fits.json"
        if fits_path.exists():
            with fits_path.open() as f:
                data = json.load(f)
            m._fits = {t: tuple(p) for t, p in data.items()}
            m._fitted = True
        return m
