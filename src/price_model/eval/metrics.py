"""Evaluation metrics — same definitions used across all models.

Conventions:
- All metrics are computed on `(prediction, realized)` pairs that share (date, ticker).
- Cross-sectional metrics are computed per-date and then averaged.
- "Realized" should be the same target the model was trained on (forward excess return).
- Where a metric isn't well-defined (e.g. all-constant predictions on a date), we return
  NaN for that date and skip it in the average.

The functions accept a polars DataFrame to keep the call sites simple. Any caller can
pass: SELECT prediction_date AS date, ticker, prediction, realized FROM ....

Significance and neutralization
-------------------------------
Two diagnostic helpers complement the standard `summarize` summary:

  shuffle_null_ic()       — empirical null distribution of mean IC computed by
                            permuting predictions cross-sectionally per date.
                            Use when the parametric IC t-stat's normality
                            assumption is suspect, or when reporting
                            non-parametric p-values is preferred.

  compute_ic_neutralized() — residualize both prediction and realized against
                            sector / size / beta columns via cross-sectional
                            OLS per date, then compute Spearman IC on the
                            residuals. Reveals whether gross IC is genuine
                            cross-sectional alpha or factor exposure.

Both are diagnostic functions, not part of the standard summary, so they
don't slow the default evaluation path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass
class MetricSummary:
    """Bundle of metrics for one (model, evaluation window)."""

    n_observations: int
    n_dates: int
    information_coefficient: float  # avg per-date Spearman
    ic_t_stat: float  # mean(IC) / (std(IC)/sqrt(n_dates))
    hit_rate: float  # fraction of correct sign predictions
    mae: float
    rmse: float
    long_short_sharpe: float  # annualized Sharpe of decile L/S portfolio

    def as_dict(self) -> dict[str, float]:
        return {
            "n_observations": self.n_observations,
            "n_dates": self.n_dates,
            "information_coefficient": self.information_coefficient,
            "ic_t_stat": self.ic_t_stat,
            "hit_rate": self.hit_rate,
            "mae": self.mae,
            "rmse": self.rmse,
            "long_short_sharpe": self.long_short_sharpe,
        }


def _per_date_ic(df: pl.DataFrame) -> pl.DataFrame:
    """Spearman IC per date. Returns (date, ic)."""
    rows = []
    for d, grp in df.group_by("date"):
        # Need at least 5 valid observations to compute a Spearman
        sub = grp.drop_nulls(subset=["prediction", "realized"])
        if sub.height < 5:
            continue
        # spearmanr returns nan if one side is constant
        rho, _ = spearmanr(sub["prediction"].to_numpy(), sub["realized"].to_numpy())
        if rho is not None and not math.isnan(rho):
            rows.append({"date": d[0] if isinstance(d, tuple) else d, "ic": float(rho)})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64})


def _long_short_returns(df: pl.DataFrame, top_frac: float = 0.2) -> pl.DataFrame:
    """Construct a daily decile long-short portfolio return series.

    For each date: take the top `top_frac` predicted names, equally long; bottom
    `top_frac`, equally short; return is mean(top.realized) - mean(bottom.realized).
    """
    rows = []
    for d, grp in df.group_by("date"):
        sub = grp.drop_nulls(subset=["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * top_frac))
        sorted_sub = sub.sort("prediction")
        bot = sorted_sub.head(k)["realized"].to_numpy()
        top = sorted_sub.tail(k)["realized"].to_numpy()
        ret = float(top.mean() - bot.mean())
        rows.append({"date": d[0] if isinstance(d, tuple) else d, "ret": ret})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64})


def summarize(df: pl.DataFrame, horizon_days: int = 5) -> MetricSummary:
    """Compute all standard metrics on a (date, ticker, prediction, realized) frame."""
    valid = df.drop_nulls(subset=["prediction", "realized"])
    if valid.height == 0:
        return MetricSummary(
            0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        )

    err = (valid["prediction"] - valid["realized"]).to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    hit_rate = float(
        np.mean(np.sign(valid["prediction"].to_numpy()) == np.sign(valid["realized"].to_numpy()))
    )

    ic_df = _per_date_ic(valid)
    if ic_df.height >= 2:
        ic_mean = float(ic_df["ic"].mean())
        ic_std = float(ic_df["ic"].std())
        t_stat = (ic_mean * math.sqrt(ic_df.height)) / ic_std if ic_std > 0 else float("nan")
    else:
        ic_mean = float("nan")
        t_stat = float("nan")

    ls = _long_short_returns(valid)
    if ls.height >= 20:
        per_day = float(ls["ret"].mean())
        per_day_std = float(ls["ret"].std())
        # Scale from per-horizon return to annualized Sharpe.
        # The realized side is a horizon-day forward return; daily refresh implies
        # we hold roughly horizon_days per signal, but for a rough comparison we
        # annualize by sqrt(252 / horizon_days).
        ann_factor = math.sqrt(252 / max(horizon_days, 1))
        sharpe = (per_day / per_day_std) * ann_factor if per_day_std > 0 else float("nan")
    else:
        sharpe = float("nan")

    return MetricSummary(
        n_observations=valid.height,
        n_dates=valid["date"].n_unique(),
        information_coefficient=ic_mean,
        ic_t_stat=t_stat,
        hit_rate=hit_rate,
        mae=mae,
        rmse=rmse,
        long_short_sharpe=sharpe,
    )


def compare_models(
    df: pl.DataFrame,
    model_ids: Iterable[str] | None = None,
    horizon_days: int = 5,
) -> pl.DataFrame:
    """Run `summarize` once per model_id. Returns a long-form comparison table."""
    if "model_id" not in df.columns:
        raise ValueError("DataFrame must have a model_id column for comparison")
    ids = list(model_ids) if model_ids else sorted(df["model_id"].unique().to_list())
    rows = []
    for mid in ids:
        sub = df.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction", "realized"
        )
        summary = summarize(sub, horizon_days=horizon_days).as_dict()
        summary["model_id"] = mid
        rows.append(summary)
    return pl.DataFrame(rows).select(
        "model_id",
        "n_observations",
        "n_dates",
        "information_coefficient",
        "ic_t_stat",
        "hit_rate",
        "mae",
        "rmse",
        "long_short_sharpe",
    )


# -----------------------------------------------------------------------------
# Diagnostic helpers — not part of the standard summary path.
# -----------------------------------------------------------------------------


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    n_periods: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> dict[str, float]:
    """Bailey-López de Prado (2014) Deflated Sharpe Ratio.

    Adjusts an observed Sharpe ratio for multiple-testing inflation. Given
    N trials (model variants tested across the project), the expected maximum
    Sharpe under a null of no skill grows as sqrt(2 * ln(N) / T). The DSR
    quantifies the probability that the true Sharpe is positive given the
    observed value, the number of trials, the sample length, and the higher
    moments of the return distribution.

    Inputs:
      - sharpe_obs:  observed Sharpe ratio (annualized or any consistent scale)
      - n_trials:    number of model variants attempted (multi-test count)
      - n_periods:   number of independent return periods used to estimate Sharpe
      - skew:        sample skewness of the strategy's per-period returns
      - kurt:        sample kurtosis (NOT excess kurtosis; gaussian = 3.0)

    Returns a dict with:
      - expected_max_sharpe_under_null
      - test_statistic
      - probability_true_sharpe_positive  (the DSR; one-sided test)

    Reference: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    Journal of Portfolio Management 40(5).
    """
    from scipy.stats import norm

    if n_trials < 1 or n_periods < 2:
        return {
            "expected_max_sharpe_under_null": float("nan"),
            "test_statistic": float("nan"),
            "probability_true_sharpe_positive": float("nan"),
        }
    # Expected maximum Sharpe under the null with N independent trials.
    # The formula uses Euler-Mascheroni gamma and the inverse-Mills ratio of
    # the maximum of N standard normals; the leading-order approximation is
    # sqrt(2 * ln(N)) which is the dominant term for moderate N.
    expected_max = float(np.sqrt(2.0 * np.log(max(n_trials, 1.0)) / n_periods))

    # Test statistic with higher-moment correction
    denom_sq = max(1.0 - skew * sharpe_obs + (kurt - 1.0) / 4.0 * sharpe_obs**2, 1e-12)
    test_stat = (
        (sharpe_obs - expected_max) * float(np.sqrt(n_periods - 1)) / float(np.sqrt(denom_sq))
    )
    prob_true_positive = float(norm.cdf(test_stat))
    return {
        "expected_max_sharpe_under_null": expected_max,
        "test_statistic": test_stat,
        "probability_true_sharpe_positive": prob_true_positive,
    }


@dataclass
class ShuffleNullResult:
    """Empirical null distribution of mean IC under random cross-sectional permutation.

    `observed_ic` is the model's actual mean per-date Spearman IC. `null_mean_ics`
    is the array of M shuffled mean ICs. `p_value_one_sided` is the fraction of
    shuffled mean ICs >= the observed (right-tailed test for positive signal).
    `null_std` is the std of the null distribution; observed_ic / null_std gives
    a non-parametric z-score that doesn't assume normality of per-date IC.
    """

    observed_ic: float
    null_mean_ics: np.ndarray
    p_value_one_sided: float
    null_mean: float
    null_std: float
    n_iterations: int

    def z_score(self) -> float:
        if self.null_std == 0:
            return float("nan")
        return (self.observed_ic - self.null_mean) / self.null_std


def shuffle_null_ic(
    df: pl.DataFrame,
    n_iterations: int = 1000,
    seed: int = 42,
    min_obs_per_date: int = 5,
) -> ShuffleNullResult:
    """Build an empirical null distribution of mean IC by cross-sectional permutation.

    For each iteration: within each date, permute the prediction column independently
    of the realized column, recompute per-date Spearman IC, average across dates.
    Returns the distribution of M shuffled mean ICs and a one-sided p-value
    relative to the observed mean IC.

    The shuffle preserves the marginal distribution of predictions and the
    cross-sectional structure of realized returns; it breaks only the alignment.
    Under the null hypothesis "predictions carry no information about forward
    realized returns," shuffled mean IC should be ~ N(0, σ_null²).

    Performance note: for M=1000 iterations on ~1500 dates with ~500 tickers
    per date, this runs in ~30-60 seconds. Spearman on shuffled data is the
    bottleneck. If you need finer p-value resolution use M=10000; this scales
    linearly.
    """
    valid = df.drop_nulls(subset=["prediction", "realized"])
    if valid.height == 0:
        return ShuffleNullResult(
            observed_ic=float("nan"),
            null_mean_ics=np.empty(0),
            p_value_one_sided=float("nan"),
            null_mean=float("nan"),
            null_std=float("nan"),
            n_iterations=0,
        )

    # Observed mean IC, computed once
    observed_ic_df = _per_date_ic(valid)
    if observed_ic_df.height == 0:
        return ShuffleNullResult(
            observed_ic=float("nan"),
            null_mean_ics=np.empty(0),
            p_value_one_sided=float("nan"),
            null_mean=float("nan"),
            null_std=float("nan"),
            n_iterations=0,
        )
    observed_ic = float(observed_ic_df["ic"].mean())

    # Group by date once, build per-date numpy arrays for fast permutation.
    rng = np.random.default_rng(seed)
    per_date_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for _d, grp in valid.group_by("date"):
        sub = grp.drop_nulls(subset=["prediction", "realized"])
        if sub.height < min_obs_per_date:
            continue
        per_date_arrays.append((sub["prediction"].to_numpy(), sub["realized"].to_numpy()))
    if not per_date_arrays:
        return ShuffleNullResult(
            observed_ic=observed_ic,
            null_mean_ics=np.empty(0),
            p_value_one_sided=float("nan"),
            null_mean=float("nan"),
            null_std=float("nan"),
            n_iterations=0,
        )

    null_mean_ics = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        per_date_ics = np.empty(len(per_date_arrays), dtype=np.float64)
        for j, (pred, realized) in enumerate(per_date_arrays):
            shuffled = rng.permutation(pred)
            rho, _ = spearmanr(shuffled, realized)
            # spearmanr returns SignificanceResult (newer scipy) — its rho can be
            # SupportsFloat but pyright's union type is wider. Coerce explicitly
            # so the numpy array assignment is unambiguous.
            rho_f = float(rho) if rho is not None else float("nan")
            per_date_ics[j] = rho_f if not math.isnan(rho_f) else 0.0
        null_mean_ics[i] = per_date_ics.mean()

    null_mean = float(null_mean_ics.mean())
    null_std = float(null_mean_ics.std())
    # One-sided right-tail: probability of seeing a mean IC at least as large
    # as observed, under the null of no information.
    p_value_one_sided = float((null_mean_ics >= observed_ic).mean())

    return ShuffleNullResult(
        observed_ic=observed_ic,
        null_mean_ics=null_mean_ics,
        p_value_one_sided=p_value_one_sided,
        null_mean=null_mean,
        null_std=null_std,
        n_iterations=n_iterations,
    )


def _residualize_per_date(
    df: pl.DataFrame,
    target_col: str,
    neutralize_cols: Sequence[str],
) -> pl.DataFrame:
    """Per-date cross-sectional OLS residualization.

    For each date, regress `target_col` on `neutralize_cols` via OLS and replace
    `target_col` with the residual. Adds an intercept implicitly by demeaning
    both target and regressors per date before solving the least-squares system.
    Returns the original frame with `target_col` overwritten by residuals.

    Rows with any null in `target_col` or `neutralize_cols` are passed through
    unchanged (residual = original value). Dates with fewer rows than
    regressors are passed through unchanged.
    """
    cols = list(neutralize_cols)
    if not cols:
        return df

    out_rows = []
    for d, grp in df.group_by("date"):
        date_key = d[0] if isinstance(d, tuple) else d
        sub = grp.drop_nulls(subset=[target_col, *cols])
        if sub.height <= len(cols) + 1:
            # Not enough rows to fit; pass through unchanged.
            out_rows.append(grp.with_columns(pl.col(target_col).alias(target_col)))
            continue
        X = sub.select(cols).to_numpy()
        y = sub[target_col].to_numpy()
        # Center both to absorb the intercept; OLS via lstsq on centered data.
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        try:
            beta, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
            preds = X_centered @ beta + y.mean()
            resid = y - preds
        except np.linalg.LinAlgError:
            # Singular system; pass through unchanged for this date.
            out_rows.append(grp)
            continue
        # Build a per-date residual frame and join back. Easier: replace target
        # column in `sub` and append `grp`'s rows that weren't in `sub` (null
        # ones) unchanged.
        sub_resid = sub.with_columns(pl.Series(target_col, resid, dtype=pl.Float64))
        # Rows that had nulls in neutralize_cols / target — pass through.
        # We rebuild grp by combining (sub_resid) + (rows of grp not in sub).
        # Use ticker as a unique id within date.
        sub_tickers = set(sub_resid["ticker"].to_list())
        passthrough = grp.filter(~pl.col("ticker").is_in(list(sub_tickers)))
        out_rows.append(pl.concat([sub_resid, passthrough], how="vertical_relaxed"))
        # date_key is unused but the loop iterator yields it; explicit no-op:
        _ = date_key

    if not out_rows:
        return df
    return pl.concat(out_rows, how="vertical_relaxed").sort(["date", "ticker"])


def compute_ic_neutralized(
    df: pl.DataFrame,
    neutralize_cols: Sequence[str],
    prediction_col: str = "prediction",
    realized_col: str = "realized",
) -> dict[str, float]:
    """Compute Spearman IC after residualizing both prediction and realized
    against `neutralize_cols` via per-date cross-sectional OLS.

    Returns a dict with:
      - ic_gross:        Spearman IC on raw prediction vs raw realized
      - ic_neutralized:  Spearman IC on (prediction | neutralize_cols)_resid
                         vs (realized | neutralize_cols)_resid
      - n_dates:         number of dates with valid IC
      - delta:           ic_gross - ic_neutralized (how much was factor exposure)

    The caller is responsible for joining the prediction/realized frame to
    the neutralization columns before calling. Typical usage:

        eval_df = predictions.join(realized, on=["date", "ticker"])
        eval_df = eval_df.join(panel[["date", "ticker", "log_dollar_volume",
                                      "beta_60", "sector_dummy"]],
                               on=["date", "ticker"])
        result = compute_ic_neutralized(
            eval_df,
            neutralize_cols=["log_dollar_volume", "beta_60"],
        )

    Sector dummies need to be one-hot-encoded into numeric columns by the
    caller; this helper expects all neutralize_cols to be float-typed.
    """
    cols = list(neutralize_cols)

    # Gross IC on raw columns
    gross_df = df.select(
        pl.col("date"),
        pl.col("ticker"),
        pl.col(prediction_col).alias("prediction"),
        pl.col(realized_col).alias("realized"),
    )
    gross_ic_df = _per_date_ic(gross_df.drop_nulls(["prediction", "realized"]))
    ic_gross = float(gross_ic_df["ic"].mean()) if gross_ic_df.height > 0 else float("nan")

    if not cols:
        return {
            "ic_gross": ic_gross,
            "ic_neutralized": ic_gross,
            "n_dates": int(gross_ic_df.height),
            "delta": 0.0,
        }

    # Residualize prediction and realized separately, then re-join and compute IC.
    pred_resid = _residualize_per_date(
        df.select("date", "ticker", prediction_col, *cols),
        target_col=prediction_col,
        neutralize_cols=cols,
    )
    realized_resid = _residualize_per_date(
        df.select("date", "ticker", realized_col, *cols),
        target_col=realized_col,
        neutralize_cols=cols,
    )
    joined = pred_resid.select("date", "ticker", pl.col(prediction_col).alias("prediction")).join(
        realized_resid.select("date", "ticker", pl.col(realized_col).alias("realized")),
        on=["date", "ticker"],
        how="inner",
    )
    neut_ic_df = _per_date_ic(joined.drop_nulls(["prediction", "realized"]))
    ic_neutralized = float(neut_ic_df["ic"].mean()) if neut_ic_df.height > 0 else float("nan")

    return {
        "ic_gross": ic_gross,
        "ic_neutralized": ic_neutralized,
        "n_dates": int(neut_ic_df.height),
        "delta": ic_gross - ic_neutralized,
    }
