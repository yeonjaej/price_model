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


# ---------------------------------------------------------------------------
# Fama-French factor model — Fama-MacBeth two-pass procedure
# ---------------------------------------------------------------------------

DEFAULT_FAMA_FRENCH_PARAMS: dict[str, Any] = {
    "factors": ("MKT_RF", "SMB", "HML", "RMW", "CMA"),
    "min_history": 252,  # ~1y to estimate stable betas
    "horizon_days": 5,
    "lambda_window": 60,  # CS-regression window for factor risk premia (in trading days)
}


class FamaFrenchFactorModel(Model):
    """Two-pass Fama-MacBeth on KF 5 factors.

    Pipeline:
    1. **Time-series pass (per ticker)**: regress the ticker's daily excess return
       (r_i − r_f) on the K factor returns. Store the K-vector of factor loadings
       β_i = (β_i,MKT, β_i,SMB, …).
    2. **Cross-sectional pass (per date)**: regress the cross-section of next-day
       excess returns on the previously-estimated β's, yielding K factor risk
       premia λ_t = (λ_t,MKT, …) per date. We average the last `lambda_window`
       trading days of λ to get λ_bar.
    3. **Forecast**: r_hat_i = β_i · λ_bar  (h-step return ≈ h · daily forecast).

    The output is normalized cross-sectionally per date (subtract date mean) so
    predictions are comparable to LightGBM's excess-return target.

    Why this is interesting beyond the standalone IC:
    - On a 20-name liquid-mega-cap universe, factor premia are small relative to
      idiosyncratic noise — IC is typically modest (~0.005–0.01). That's the
      honest baseline.
    - The estimated β's are also exposed as features (see
      price_model.features.factor_loadings). LightGBM can consume those, which
      is the cross-pollination that gives this scaffold real lift.

    Data dependency: uses Ken French daily 5-factor returns (the
    fama_french.fetch() adapter handles download + cache). The first walk-forward
    split incurs a one-time download (~1MB).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Per-ticker time-series-pass betas: dict[ticker, np.ndarray of shape (K,)]
        self._betas: dict[str, np.ndarray] = {}
        # Average cross-sectional risk premium vector (shape (K,))
        self._lambda_bar: np.ndarray | None = None
        # Cache of training panel last-known date — used to pull the lambda window
        self._train_end_date: Any | None = None

    def _params(self) -> dict[str, Any]:
        return {**DEFAULT_FAMA_FRENCH_PARAMS, **self.config.params}

    def _factor_panel(self) -> pl.DataFrame:
        """Lazy KF factor download, deferred so import works without internet."""
        from price_model.data.sources import fama_french

        return fama_french.fetch()

    def fit(self, panel: pl.DataFrame) -> None:
        params = self._params()
        factors = list(params["factors"])
        min_history = int(params["min_history"])
        lambda_window = int(params["lambda_window"])

        ff = self._factor_panel().select(["date", *factors, "RF"])
        panel = _log_returns(panel).join(ff, on="date", how="left")
        # Excess return = ticker log return - daily risk-free rate
        panel = panel.with_columns(
            (pl.col("log_return") - pl.col("RF")).alias("_excess"),
        )
        # Drop rows where any factor or excess is null (warmup + KF lag)
        train = panel.drop_nulls(subset=["_excess", *factors])

        # --- Pass 1: per-ticker time-series regression ---
        K = len(factors)
        betas: dict[str, np.ndarray] = {}
        for ticker in train["ticker"].unique().to_list():
            sub = train.filter(pl.col("ticker") == ticker).sort("date")
            if sub.height < min_history:
                continue
            y = sub["_excess"].to_numpy()
            X = np.column_stack([sub[f].to_numpy() for f in factors])
            # OLS with intercept absorbed (alpha is what's left over; we drop it
            # because predictions are cross-sectionally demeaned anyway).
            X_with_const = np.column_stack([np.ones(len(y)), X])
            try:
                coefs, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
            except np.linalg.LinAlgError as e:
                log.warning("FF time-series regression failed for %s: %s", ticker, e)
                continue
            betas[ticker] = coefs[1:]  # drop intercept (alpha)

        # --- Pass 2: cross-sectional regression on each date with >= K+1 tickers ---
        # For each date t, regress cross-section of r_i,t on (β_i,MKT, …, β_i,CMA)
        # to get λ_t. Average the last `lambda_window` dates.
        train_with_betas: list[dict] = []
        for ticker, beta_vec in betas.items():
            sub = train.filter(pl.col("ticker") == ticker)
            for row in sub.iter_rows(named=True):
                rec = {"date": row["date"], "ticker": ticker, "_excess": row["_excess"]}
                for j, f in enumerate(factors):
                    rec[f"_beta_{f}"] = float(beta_vec[j])
                train_with_betas.append(rec)

        if not train_with_betas:
            log.warning("FamaFrenchFactorModel: no usable training rows; aborting fit")
            self._betas = {}
            self._lambda_bar = None
            self._fitted = True
            return

        beta_panel = pl.DataFrame(train_with_betas)
        dates_sorted = beta_panel["date"].unique().sort().to_list()
        # Use the most recent `lambda_window` dates as the in-sample window
        window_dates = set(dates_sorted[-lambda_window:])
        recent = beta_panel.filter(pl.col("date").is_in(list(window_dates)))

        lambda_acc = np.zeros(K)
        lambda_n = 0
        # Need at least 2 points to fit an intercept + at least one slope; in
        # practice we want K tickers so the K slopes aren't trivially underdetermined,
        # but with a 20-name deployment universe and K=5 that always holds.
        min_cs_n = max(2, K)
        for d in window_dates:
            day = recent.filter(pl.col("date") == d)
            if day.height < min_cs_n:
                continue
            y_t = day["_excess"].to_numpy()
            X_t = np.column_stack([day[f"_beta_{f}"].to_numpy() for f in factors])
            X_const = np.column_stack([np.ones(len(y_t)), X_t])
            try:
                coefs, *_ = np.linalg.lstsq(X_const, y_t, rcond=None)
            except np.linalg.LinAlgError:
                continue
            lambda_acc += coefs[1:]
            lambda_n += 1

        if lambda_n == 0:
            log.warning(
                "FamaFrenchFactorModel: no dates with enough tickers for CS regression; "
                "falling back to zero risk premia"
            )
            self._lambda_bar = np.zeros(K)
        else:
            self._lambda_bar = lambda_acc / lambda_n

        self._betas = betas
        self._train_end_date = dates_sorted[-1] if dates_sorted else None
        self._fitted = True
        log.info(
            "FamaFrenchFactorModel fitted %d tickers, lambda from %d CS-days, factors=%s",
            len(self._betas),
            lambda_n,
            factors,
        )

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        params = self._params()
        h = int(params["horizon_days"])

        if self._lambda_bar is None or not self._betas:
            # Degenerate fit — emit zeros so the harness still has rows to evaluate.
            unique = panel.select(["date", "ticker"]).unique().sort(["date", "ticker"])
            return unique.with_columns(pl.lit(0.0).alias("prediction"))

        # Daily forecast in excess-return units, scaled to horizon h.
        # The model's daily prediction for ticker i is β_i · λ_bar; over h days,
        # multiply by h (additive in expectation under iid assumption).
        rows: list[dict] = []
        for ticker in panel["ticker"].unique().to_list():
            beta_vec = self._betas.get(ticker)
            if beta_vec is None:
                continue
            daily_pred = float(np.dot(beta_vec, self._lambda_bar))
            horizon_pred = daily_pred * h
            for d in panel.filter(pl.col("ticker") == ticker)["date"].unique().to_list():
                rows.append({"date": d, "ticker": ticker, "prediction": horizon_pred})

        rows = _to_cross_sectional_excess(rows)
        if not rows:
            return pl.DataFrame(
                schema={"date": pl.Date, "ticker": pl.Utf8, "prediction": pl.Float64}
            )
        return pl.DataFrame(rows).sort(["date", "ticker"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        # Save betas + lambda as a single pickle for convenience
        state = {
            "betas": {t: b.tolist() for t, b in self._betas.items()},
            "lambda_bar": None if self._lambda_bar is None else self._lambda_bar.tolist(),
            "train_end_date": str(self._train_end_date) if self._train_end_date else None,
        }
        (path / "fits.json").write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path) -> FamaFrenchFactorModel:
        m = cls(load_config(path / "config.json"))
        fits_path = path / "fits.json"
        if fits_path.exists():
            data = json.loads(fits_path.read_text())
            m._betas = {t: np.array(b, dtype=float) for t, b in data.get("betas", {}).items()}
            lam = data.get("lambda_bar")
            m._lambda_bar = None if lam is None else np.array(lam, dtype=float)
            m._train_end_date = data.get("train_end_date")
            m._fitted = True
        return m
