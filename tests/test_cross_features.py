"""Tests for cross-sectional features that depend on the panel (not just one ticker).

Currently covers `beta_60` (RollingBeta60). The sector-relative / rank features
are smoke-tested indirectly through the pipeline tests.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

import price_model.features.cross_features  # noqa: F401  trigger registration
from price_model.features.base import FEATURE_REGISTRY


def _two_ticker_panel_with_known_beta(
    n_days: int = 400, beta_aaa: float = 1.5, seed: int = 11
) -> pl.DataFrame:
    """A 2-ticker panel where AAA is constructed to have a known beta to the
    cross-sectional mean return.

    With only two tickers, the cross-sectional mean on each date is
    `0.5 * (r_aaa + r_bbb)`. If we set:

        r_aaa(t) = beta_aaa * factor(t) + noise_aaa(t)
        r_bbb(t) = factor(t) + noise_bbb(t)

    then the "market" return (mean of the two) is roughly
    `0.5 * (beta_aaa + 1) * factor(t)` plus aggregated noise, and AAA's beta
    to that market series is approximately
    `beta_aaa / (0.5 * (beta_aaa + 1)) = 2 beta_aaa / (beta_aaa + 1)`.

    For `beta_aaa = 1.5` the expected rolling beta is `2 * 1.5 / 2.5 = 1.2`.
    The test asserts the recovered median beta is within a generous tolerance
    of that target — the goal is to confirm the formula direction, not the
    third decimal.
    """
    rng = np.random.default_rng(seed=seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    factor = rng.normal(0.0, 0.01, size=n_days)
    noise_aaa = rng.normal(0.0, 0.003, size=n_days)
    noise_bbb = rng.normal(0.0, 0.003, size=n_days)

    r_aaa = beta_aaa * factor + noise_aaa
    r_bbb = factor + noise_bbb

    rows = []
    log_aaa = np.cumsum(r_aaa)
    log_bbb = np.cumsum(r_bbb)
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": "AAA", "adj_close": float(100.0 * np.exp(log_aaa[i]))})
        rows.append({"date": d, "ticker": "BBB", "adj_close": float(100.0 * np.exp(log_bbb[i]))})
    return pl.DataFrame(rows).sort(["ticker", "date"])


def test_beta_60_registered():
    assert "beta_60" in FEATURE_REGISTRY


def test_beta_60_produces_output_column(synthetic_panel):
    feat = FEATURE_REGISTRY["beta_60"]
    out = feat.compute(synthetic_panel)
    assert "beta_60" in out.columns
    assert out["beta_60"].dtype == pl.Float64


def test_beta_60_warmup_leaves_61_nulls(synthetic_panel):
    """1 day for the diff + 60 for the rolling window → first 60 rows null."""
    feat = FEATURE_REGISTRY["beta_60"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")["beta_60"].to_list()
    # Implementation produces nulls until both the log-return and the 60-day
    # cov/var windows are populated. Exact null count = 60.
    leading_nulls = 0
    for v in one:
        if v is None:
            leading_nulls += 1
        else:
            break
    assert leading_nulls == 60, f"expected 60 leading nulls, got {leading_nulls}"


def test_beta_60_per_ticker_isolation_on_market_excluded_axis(synthetic_panel):
    """Perturbing one ticker's price LEVEL shouldn't change BBB's beta to
    floating-point tolerance: multiplying AAA's prices by a constant leaves
    AAA's daily log-returns mathematically identical (the constant log offset
    cancels in the diff), so the cross-sectional mean return is unchanged and
    BBB's beta is invariant. The numerical check uses an isclose tolerance
    because FP cancellation of the offset is not bit-exact under polars's
    reduction ordering.
    """
    feat = FEATURE_REGISTRY["beta_60"]
    baseline = feat.compute(synthetic_panel).filter(pl.col("ticker") == "BBB")["beta_60"]
    perturbed = synthetic_panel.with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.col("adj_close") * 3.0)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    after = feat.compute(perturbed).filter(pl.col("ticker") == "BBB")["beta_60"]
    base_arr = baseline.to_numpy()
    after_arr = after.to_numpy()
    # Same null mask
    assert np.array_equal(np.isnan(base_arr), np.isnan(after_arr))
    # Non-null values agree to floating-point tolerance
    mask = ~np.isnan(base_arr)
    assert np.allclose(base_arr[mask], after_arr[mask], rtol=1e-10, atol=1e-12)


def test_beta_60_recovers_known_beta_direction():
    """On a constructed panel where AAA has a higher loading on the common
    factor than BBB, AAA's recovered beta-to-market should exceed BBB's.

    With AAA loading 1.5 on factor and BBB loading 1.0, the 2-ticker market
    proxy is `0.5 * (r_aaa + r_bbb)` and the expected betas are:
        beta_aaa ≈ 2 * 1.5 / (1.5 + 1) = 1.2
        beta_bbb ≈ 2 * 1.0 / (1.5 + 1) = 0.8
    The test checks the ordering (AAA > BBB) and that the median sits
    in a wide tolerance band around the analytical targets.
    """
    panel = _two_ticker_panel_with_known_beta(n_days=800, beta_aaa=1.5)
    out = FEATURE_REGISTRY["beta_60"].compute(panel).drop_nulls("beta_60")

    med_aaa = float(out.filter(pl.col("ticker") == "AAA")["beta_60"].median())
    med_bbb = float(out.filter(pl.col("ticker") == "BBB")["beta_60"].median())

    assert med_aaa > med_bbb, f"AAA beta ({med_aaa:.3f}) should exceed BBB beta ({med_bbb:.3f})"
    # Generous bounds — the cross-sectional-mean proxy is noisy on N=2.
    assert 0.8 < med_aaa < 1.6, f"AAA beta {med_aaa:.3f} outside expected ~1.2 band"
    assert 0.4 < med_bbb < 1.2, f"BBB beta {med_bbb:.3f} outside expected ~0.8 band"


def test_beta_60_constant_prices_gives_nulls_or_zero():
    """Flat prices → zero return variance → beta is either null (var<eps) or 0."""
    panel = pl.DataFrame(
        [
            {"date": date(2020, 1, 2) + timedelta(days=i), "ticker": t, "adj_close": 100.0}
            for t in ("AAA", "BBB", "CCC")
            for i in range(200)
        ]
    ).sort(["ticker", "date"])
    out = FEATURE_REGISTRY["beta_60"].compute(panel)
    # All non-null betas (if any) must be exactly zero — the implementation's
    # `pl.when(var_y > 1e-12)` branch sets var-collapse rows to null instead.
    non_null = out.drop_nulls("beta_60")["beta_60"].to_list()
    assert all(abs(v) < 1e-9 for v in non_null)
