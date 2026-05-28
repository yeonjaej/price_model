"""Tests for the Fama-French scaffolding.

Three layers:
- KF CSV parser: deterministic, no network. Uses an inline mini-CSV that mimics
  the real Dartmouth ZIP layout (text header block, header row, YYYYMMDD rows).
- Rolling factor-loading features: monkeypatched fetch() so the feature contract
  (lookback, leakage discipline, output column) is exercised offline.
- FamaFrenchFactorModel: monkeypatched fetch() with synthetic factors so the
  Fama-MacBeth two-pass procedure is exercised end-to-end without network.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from price_model.data.sources import fama_french
from price_model.features.base import get_feature
from price_model.models import MODEL_REGISTRY, FamaFrenchFactorModel, build_model
from price_model.models.base import ModelConfig

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _synthetic_kf(n_days: int = 800, seed: int = 11) -> pl.DataFrame:
    """Generate a fake KF 5-factor daily frame for offline tests.

    Already in decimal units (as fetch() would return).
    """
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "date": dates,
            "MKT_RF": rng.normal(0.0003, 0.011, n_days),
            "SMB": rng.normal(0.0, 0.005, n_days),
            "HML": rng.normal(0.0, 0.005, n_days),
            "RMW": rng.normal(0.0, 0.004, n_days),
            "CMA": rng.normal(0.0, 0.004, n_days),
            "RF": np.full(n_days, 1e-4),
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def _synthetic_panel(
    n_days: int = 800,
    seed: int = 17,
    market_returns: np.ndarray | None = None,
) -> pl.DataFrame:
    """Three-ticker synthetic panel with controlled factor exposures.

    If `market_returns` is provided, each ticker's log return is `beta * market + idio`.
    Otherwise we draw a market series locally. To exercise the rolling-beta recovery
    test you must pass the SAME market series the KF fixture exposes — otherwise
    the regression sees two uncorrelated noise sources and beta lands at ~0.
    """
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    mkt = market_returns if market_returns is not None else rng.normal(0.0003, 0.011, n_days)
    rows = []
    for ticker, beta in [("AAA", 1.2), ("BBB", 0.8), ("CCC", 1.0)]:
        idio = rng.normal(0.0, 0.012, n_days)
        log_ret = beta * mkt + idio
        log_p = np.cumsum(log_ret)
        prices = 100.0 * np.exp(log_p)
        for i, p in enumerate(prices):
            rows.append({"date": dates[i], "ticker": ticker, "adj_close": float(p)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])


# ---------------------------------------------------------------------------
# KF CSV parser
# ---------------------------------------------------------------------------


KF_SAMPLE_CSV = """\

This file was created by CMPT_ME_BEME_RETS using the 202401 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

      ,Mkt-RF,SMB,HML,RMW,CMA,RF
20240102,  0.50, -0.20,  0.30,  0.10,  0.05, 0.02
20240103, -0.25,  0.15, -0.05, -0.02,  0.01, 0.02
20240104,  1.10,  0.40, -0.30,  0.20, -0.10, 0.02

Annual Factors:
"""


def test_kf_parser_reads_text_header_and_decimal_conversion():
    df = fama_french._parse_kf_csv(KF_SAMPLE_CSV)
    assert df.height == 3
    assert df.columns == ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    # Dates parsed correctly
    assert df["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    # Percent → decimal conversion
    assert df["MKT_RF"].to_list() == pytest.approx([0.005, -0.0025, 0.011])
    assert df["RF"].to_list() == pytest.approx([0.0002, 0.0002, 0.0002])


def test_kf_parser_stops_at_blank_line_before_annual_block():
    # Append a fake annual table after a blank line and confirm we ignore it.
    csv = KF_SAMPLE_CSV + "\n2024,  10.0, 2.0, 3.0, 1.0, 0.5, 5.0\n"
    df = fama_french._parse_kf_csv(csv)
    assert df.height == 3  # still only the 3 daily rows


def test_kf_parser_raises_without_header():
    bad = "Some text\n\nbut no header row anywhere\n"
    with pytest.raises(ValueError, match="header row"):
        fama_french._parse_kf_csv(bad)


# ---------------------------------------------------------------------------
# Rolling factor-loading features
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_kf(monkeypatch):
    """Replace fama_french.fetch with a synthetic frame and clear the lru_cache."""
    import price_model.features.factor_loadings as fl

    fake = _synthetic_kf(n_days=800)
    monkeypatch.setattr(fama_french, "fetch", lambda *a, **kw: fake)
    # Bust the lru_cache so the next call sees the patched fetch.
    fl._load_kf_factors.cache_clear()
    yield fake
    fl._load_kf_factors.cache_clear()


def test_mkt_beta_60_feature_runs_and_has_lookback_warmup(patched_kf):
    panel = _synthetic_panel(n_days=400)
    feat = get_feature("mkt_beta_60")
    out = feat.compute(panel.sort(["ticker", "date"]))

    assert "mkt_beta_60" in out.columns
    assert out.height == panel.height
    # First ~61 rows per ticker should be null (1 for diff + 60 for window)
    early = out.filter(pl.col("ticker") == "AAA").sort("date").head(60)
    assert early["mkt_beta_60"].null_count() >= 50  # generous — exact depends on join

    # After warmup, betas should be finite for the majority of rows
    late = out.filter(pl.col("ticker") == "AAA").sort("date").tail(200)
    assert late["mkt_beta_60"].null_count() < 30


def test_mkt_beta_60_recovers_true_beta_within_tolerance(patched_kf):
    """The synthetic panel uses beta_AAA=1.2 and beta_BBB=0.8 — when we feed the
    SAME market series into both the panel and the KF fixture, the median rolling
    beta over the sample should land near those true values.
    """
    mkt_series = patched_kf["MKT_RF"].to_numpy()
    panel = _synthetic_panel(n_days=800, market_returns=mkt_series)
    out = get_feature("mkt_beta_60").compute(panel.sort(["ticker", "date"]))

    for ticker, true_beta in [("AAA", 1.2), ("BBB", 0.8)]:
        sub = out.filter(pl.col("ticker") == ticker).drop_nulls("mkt_beta_60")
        median_beta = float(sub["mkt_beta_60"].median())
        assert abs(median_beta - true_beta) < 0.15, (
            f"Recovered beta {median_beta:.3f} for {ticker} is far from true {true_beta}"
        )


def test_factor_loading_features_registered():
    for name in ("mkt_beta_60", "smb_beta_60", "hml_beta_60", "rmw_beta_60", "cma_beta_60"):
        assert get_feature(name).name == name


def test_factor_spread_features_registered():
    for name in (
        "mkt_beta_xs_dispersion",
        "mkt_beta_x_mkt_20d",
        "smb_beta_x_smb_20d",
        "hml_beta_x_hml_20d",
    ):
        assert get_feature(name).name == name


def test_mkt_beta_xs_dispersion_median_is_zero_per_date(patched_kf):
    """The cross-sectional demean trivially satisfies median ≈ 0 within each date.

    Sanity check that the per-date median subtraction is actually happening.
    """
    panel = _synthetic_panel(n_days=400)
    out = get_feature("mkt_beta_xs_dispersion").compute(panel.sort(["ticker", "date"]))
    by_date = (
        out.drop_nulls("mkt_beta_xs_dispersion")
        .group_by("date")
        .agg(pl.col("mkt_beta_xs_dispersion").median().alias("m"))
    )
    # Median of (x - median(x)) is 0 by construction
    assert by_date["m"].abs().max() < 1e-9


def test_smb_beta_x_smb_20d_sign_flips_with_regime(patched_kf):
    """When SMB cumulates positive, ticker SMB beta and feature should share sign;
    when SMB cumulates negative, they should differ."""
    panel = _synthetic_panel(n_days=400)
    smb_beta_out = get_feature("smb_beta_60").compute(panel.sort(["ticker", "date"]))
    spread_out = get_feature("smb_beta_x_smb_20d").compute(panel.sort(["ticker", "date"]))

    # Join the two on (date, ticker) so we can compare per row
    joined = (
        smb_beta_out.select("date", "ticker", "smb_beta_60")
        .join(
            spread_out.select("date", "ticker", "smb_beta_x_smb_20d"),
            on=["date", "ticker"],
        )
        .drop_nulls()
    )

    # Use the synthetic KF feed directly to know each date's 20-day SMB cumulative
    kf = (
        patched_kf.select("date", "SMB")
        .sort("date")
        .with_columns(pl.col("SMB").rolling_sum(window_size=20).alias("smb_20d"))
    )
    joined = joined.join(kf.select("date", "smb_20d"), on="date").drop_nulls()

    # sign(beta) and sign(spread / smb_20d) should agree (since spread = beta * smb_20d)
    # Just check the product (beta * smb_20d) ≈ spread within float tolerance
    diff = (joined["smb_beta_60"] * joined["smb_20d"] - joined["smb_beta_x_smb_20d"]).abs().max()
    assert diff < 1e-9


# ---------------------------------------------------------------------------
# FamaFrenchFactorModel
# ---------------------------------------------------------------------------


def test_fama_french_model_registered():
    assert "FamaFrenchFactorModel" in MODEL_REGISTRY


def test_fama_french_round_trip(monkeypatch, tmp_path):
    """End-to-end fit + predict + save + load on synthetic data, no network."""
    fake = _synthetic_kf(n_days=800)
    monkeypatch.setattr(fama_french, "fetch", lambda *a, **kw: fake)

    cfg = ModelConfig(
        model_id="ff_test",
        feature_cols=(),
        # 3 tickers in the synthetic panel — restrict to 2 factors so the
        # cross-sectional regression isn't trivially underdetermined.
        params={
            "factors": ("MKT_RF", "SMB"),
            "min_history": 200,
            "horizon_days": 5,
            "lambda_window": 60,
        },
    )
    model = build_model("FamaFrenchFactorModel", cfg)
    panel = _synthetic_panel(n_days=600)
    model.fit(panel)

    assert len(model._betas) == 3  # AAA, BBB, CCC
    assert model._lambda_bar is not None
    assert model._lambda_bar.shape == (2,)  # configured 2 factors

    preds = model.predict(panel.tail(150))
    assert preds.height >= 3
    assert preds.columns == ["date", "ticker", "prediction"]

    # Cross-sectional excess: mean prediction per date is ~0
    by_date_mean = preds.group_by("date").agg(pl.col("prediction").mean().alias("m"))
    assert by_date_mean["m"].abs().max() < 1e-9

    # Save + load preserves betas and lambda
    model.save(tmp_path / "ff_test")
    loaded = FamaFrenchFactorModel.load(tmp_path / "ff_test")
    assert set(loaded._betas.keys()) == set(model._betas.keys())
    assert loaded._lambda_bar is not None
    np.testing.assert_allclose(loaded._lambda_bar, model._lambda_bar)


def test_fama_french_recovers_relative_beta_ordering(monkeypatch):
    """When beta_AAA > beta_CCC > beta_BBB and the market premium is positive,
    AAA should out-rank BBB in the forecast. We don't test point estimates because
    cross-sectional excess return subtracts the mean — only the ordering survives.
    """
    # Make the market premium clearly positive so high-beta = high forecast.
    rng = np.random.default_rng(99)
    n = 800
    start = date(2022, 1, 3)
    dates = [start + timedelta(days=i) for i in range(n)]
    mkt = rng.normal(0.001, 0.011, n)  # positive drift
    fake_kf = pl.DataFrame(
        {
            "date": dates,
            "MKT_RF": mkt,
            "SMB": np.zeros(n),
            "HML": np.zeros(n),
            "RMW": np.zeros(n),
            "CMA": np.zeros(n),
            "RF": np.full(n, 1e-4),
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    monkeypatch.setattr(fama_french, "fetch", lambda *a, **kw: fake_kf)

    # Panel: AAA β=1.5, BBB β=0.5, CCC β=1.0 — all driven by the same mkt
    rows = []
    for ticker, beta in [("AAA", 1.5), ("BBB", 0.5), ("CCC", 1.0)]:
        idio = rng.normal(0.0, 0.008, n)
        log_ret = beta * mkt + idio
        log_p = np.cumsum(log_ret)
        prices = 100.0 * np.exp(log_p)
        for i, p in enumerate(prices):
            rows.append({"date": dates[i], "ticker": ticker, "adj_close": float(p)})
    panel = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])

    cfg = ModelConfig(
        model_id="ff_order_test",
        feature_cols=(),
        params={
            "factors": ("MKT_RF",),  # single factor so 3 tickers suffices for CS regression
            "min_history": 200,
            "horizon_days": 5,
            "lambda_window": 100,
        },
    )
    model = build_model("FamaFrenchFactorModel", cfg)
    model.fit(panel)

    # Slice by date (not rows) so all 3 tickers appear in the prediction panel
    cutoff = panel["date"].unique().sort()[-20]
    test_panel = panel.filter(pl.col("date") >= cutoff)
    preds = model.predict(test_panel)

    # On any given date in the held-out tail, AAA > CCC > BBB ordering should hold
    by_date = preds.group_by("date").agg(
        pl.col("prediction").alias("ps"),
        pl.col("ticker").alias("ts"),
    )
    assert by_date.height >= 1
    for row in by_date.iter_rows(named=True):
        ranked = sorted(zip(row["ts"], row["ps"], strict=True), key=lambda kv: kv[1], reverse=True)
        order = [t for t, _ in ranked]
        assert order == ["AAA", "CCC", "BBB"], f"Bad ordering on {row['date']}: {order}"
