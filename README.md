# price-model

A point-in-time (PIT) corrected cross-sectional equity return predictor on the S&P 500, evaluated under strict HP-free held-out protocols at the 21-day (approximately monthly) forward horizon.

## Executive summary

On the 2025-01-02 → 2026-04-29 out-of-sample slice (336 dates, all hyperparameters and coefficients selected strictly on data preceding this period), a **9-feature L1 cross-sectional regression** on documented academic anomalies (Han-He-Rapach-Zhou 2024) produces the strongest signal at the 21-day forward horizon: **IC = +0.0695 (t = +5.25), long-short Sharpe = +1.24**. The same framework applied at 5-day horizon scores IC = +0.0249 (t = +2.72); the README leads with the 21-day result because the linear-on-anomalies advantage is much sharper at the monthly horizon.

Headline 21-day comparison on the 2025-01-02 → 2026-04-29 OOS slice (336 dates):

| Model | HP-selection | Gross IC | t-stat | Gross L/S Sharpe | Net 20 bp Sharpe |
|---|---|---|---|---|---|
| **L1 regression, 9-feature anomaly panel** | inner CV | **+0.0695** | **+5.25** | **+1.24** | +1.17 |
| 24-month momentum factor | none | +0.0509 | +6.32 | +1.40 | +1.39 |
| 36-month momentum factor | none | +0.0501 | +7.15 | +1.82 | **+1.81** |
| Ridge regression, 12-feature panel | inner CV | +0.0430 | +4.32 | +1.02 | +0.98 |
| 18-month momentum factor | none | +0.0421 | +4.97 | +1.35 | +1.34 |
| JT 12-1 momentum (canonical) | none | +0.0311 | +3.51 | +1.04 | +1.02 |
| LightGBM, default HPs | none | +0.0274 | +4.30 | +0.74 | +0.62 |
| LightGBM, held-out Optuna (≤ 2024-12-31) | held-out Optuna | +0.0187 | +2.97 | +0.73 | +0.59 |
| Pure-momentum Lasso, 4 momentum features | inner CV | **−0.0184** | **−2.79** | −0.21 | −0.26 |

Two top-line observations from the table:

- **Gross-IC ranking favors the L1 regression**; net 20 bp Sharpe favors the 36-month momentum factor (+1.80 vs +1.17), driven by an 8× turnover differential.
- **Every tested ML variant trails both** the L1 regression and the long-horizon momentum factors on both gross IC and net Sharpe.

For the technical setup (PIT correction, walk-forward training with embargo, inner CV for regularization-strength selection, held-out Optuna, deflated Sharpe), see [Methodology](#methodology). For analysis of why ML fell short, why long-horizon momentum wins net-of-cost, and the regime fragility of hyperparameter optimization, see [Discussion](#discussion).

The result is **statistically rigorous, regime-conditional, and not deployable for retail investors** after bid-ask spreads, commissions, and capital-gains taxes. See [Data quality and methodological limitations](#data-quality-and-methodological-limitations) and [Scope and limitations](#scope-and-limitations) for the bounds on what these numbers mean. For the precise definitions of each metric in the table above (IC, t-stat, gross / net long-short Sharpe, annual turnover, deflated Sharpe), see [Metric definitions](#metric-definitions).

## Metric definitions

The headline 21-day result is reported in terms of standard cross-sectional asset-pricing metrics. The definitions below apply throughout the rest of the README. The worked example below uses 5-day forward excess return because that is what most stored predictions in the project are; substitute 21-day forward excess return for the headline result without loss of generality (the formula is the same, only the horizon changes).

- **Information Coefficient (IC).** Cross-sectional ranking quality of
  the model, averaged over time.

  *Per-date IC.* On each date `t` in the evaluation window, the model
  produces a prediction for every ticker in the cross-section that has a
  realized forward excess return available at the model's target horizon
  `H`. Let `N_t` be the number of such tickers on date `t` — in the
  present universe `N_t` is on the order of 300-600. The per-date IC is
  the Spearman rank correlation between two vectors **of length N_t**:
  the vector of predictions and the vector of realized `H`-day forward
  excess returns. Note that the vectors are not length `H`; the horizon
  refers to the return target, not the number of observations.

  *Time-averaged IC.* The reported IC is the unweighted average of the
  per-date ICs across all evaluation dates (e.g., `n_dates = 334` for the
  21-day headline OOS slice 2025-01-02 → 2026-04-29). A weak ranker on
  most days plus one strong day does not produce a high IC; consistent
  ranking is required.

  *Worked example.* On a hypothetical date with `N_t = 5` tickers,
  suppose the model's predictions and the realized 5-day forward excess
  returns are:

  | Ticker | Prediction | Pred rank | Realized | Realized rank |
  |---|---|---|---|---|
  | AAA | +0.020 | 5 | +0.030 | 5 |
  | BBB | +0.010 | 4 | +0.005 | 3 |
  | CCC | +0.000 | 3 | +0.012 | 4 |
  | DDD | −0.005 | 2 | −0.001 | 2 |
  | EEE | −0.015 | 1 | −0.020 | 1 |

  The rank vectors are `[5, 4, 3, 2, 1]` and `[5, 3, 4, 2, 1]`. The
  Spearman correlation (Pearson correlation of ranks) is:

  `ρ = 1 − (6 × Σ d²) / (N × (N² − 1))`
  ` = 1 − (6 × (0² + 1² + 1² + 0² + 0²)) / (5 × 24) = 1 − 12/120 = 0.90`

  So the per-date IC on this hypothetical date is **+0.90**. Perfect
  ranking would give +1.0; perfect mis-ranking would give −1.0;
  independent ranking would give roughly 0. The reported headline 21-day
  IC of +0.0695 is therefore the **average over 336 such daily
  correlations** across the 2025-01-02 → 2026-04-29 OOS slice, each
  computed across ~450 tickers. A per-date IC of +0.07 on average is
  small in absolute terms but large in t-stat (+5.25) due to the high
  date count — the consistency across days, not the magnitude on any
  one day, is what makes the signal credible.

  *Interpretation benchmarks.* On liquid US large-caps, IC = +0.02 with
  t-stat > 2 is considered a credible edge. IC ≈ +0.10 would be
  exceptional; IC > +0.20 is not realistic out-of-sample.

- **t-stat of IC.** `mean(per-date IC) / (stdev(per-date IC) / √n_dates)`.
  Tests whether the mean IC is distinguishable from zero against the
  null hypothesis that per-date ICs are sampled from a distribution
  centered at zero. |t| > 1.96 corresponds to p < 0.05.
- **Long-short Sharpe.** Annualized Sharpe ratio of a daily-rebalanced
  portfolio that is long the top-quintile predicted tickers (top 20%)
  and short the bottom-quintile (bottom 20%), equal-weighted within each
  leg.
  The quintile cut is set in code by `_long_short_returns(top_frac=0.2)`
  in `src/price_model/eval/metrics.py`. Sharpe is computed per-horizon and
  annualized as `mean(per-horizon return) / stdev(per-horizon return) ×
  √(252 / horizon_days)`. Above +1.0 is the conventional bar for
  institutional tradeability; reported values are gross of all transaction
  costs unless explicitly labeled "net N bp."
- **Annual turnover.** One-sided fraction of portfolio capital traded
  per year, computed as `mean(0.5 · Σ_i |w_t[i] − w_{t-1}[i]|) · 252`
  where `w_t` is the long-short weight vector on date `t`. An annual
  turnover of `1.0` means the portfolio replaces itself once per year;
  `100x` means the equivalent of 100 full-portfolio rotations per year.
  The metric is implemented in `src/price_model/eval/turnover.py` and
  governs how much of a gross signal survives net of transaction costs:
  for cost level `b` bp per side, annual cost drag ≈
  `annual_turnover · b · 2 / 10000`.
- **After-cost Sharpe at N bp.** Long-short Sharpe computed on returns
  net of transaction-cost drag. For each date, gross return is
  `mean(top.realized) − mean(bottom.realized)`; net return subtracts
  `turnover_t · N / 10000` from gross. The resulting net-return series
  is annualized to a Sharpe using the same `√(252/horizon)` convention
  as gross Sharpe. Reported at 3, 10, and 20 bp per side to span
  institutional → retail cost regimes.
- **Deflated Sharpe ratio (DSR).** Bailey & López de Prado (2014)
  multi-test-corrected probability that a model's true Sharpe is
  positive given the observed Sharpe, the number of trials attempted,
  the sample length, and the higher moments of the return distribution.
  Defined as `P(true_Sharpe > 0 | observed)` with a higher-moment
  correction for skew and kurtosis. The threshold for "significant
  after multi-test correction" is `DSR > 0.95`. Implemented in
  `src/price_model/eval/metrics.py::deflated_sharpe_ratio`; computed
  across all project trials in `scripts/deflated_sharpe_audit.py`.
- **Cross-sectional return dispersion (regime indicator).** For each
  date, the standard deviation of daily log returns across the
  cross-section, smoothed by a 20-day rolling mean. Computed only from
  past returns (verified lookahead-safe by the universal
  truncation-invariance leakage test). Higher dispersion → more
  ticker-specific divergence; lower dispersion → more uniform market
  movement. Used as a contemporaneously-observable conditioning
  variable for both the regime-aware LightGBM feature panel and the
  walk-forward ensemble. Implemented as the `cs_return_dispersion_20`
  feature in `src/price_model/features/cross_features.py`.

## Reproduction

### Step 0 — install and build the PIT universe (one-time, ~10-15 min cold)

```bash
# Core: required for everything below.
pip install -e ".[dev,classical]"

# Optional: notebook visualization stack (matplotlib + seaborn + jupyter).
# Install if you plan to run the notebooks/ directory, including the
# feature-exploration boilerplate at notebooks/05_feature_exploration.ipynb.
pip install -e ".[notebooks]"

# Scrape Wikipedia for historical S&P 500 membership and write the universe file
python -m price_model.cli build-universe --name sp500_pit --start 2017-01-01

# Fetch yfinance data for the ~700 resolving tickers (~10-15 min cold; subsequent
# runs are incremental, fetching only the new rows since the cache's max date)
python -m price_model.cli refresh-data --universe sp500_pit --start 2017-01-01
```

### Step 1 — train the headline models

The 21-day apples-to-apples comparison requires four trained models. Each run takes 1-15 minutes depending on model class (momentum factors are essentially instant; LightGBM is the slowest).

```bash
# 9-feature L1 regression at 21-day (the headline)
python -m price_model.cli run -e lasso_elasso_pit_h21

# Long-horizon momentum factors at 21-day (the runner-up)
python -m price_model.cli run -e momentum_factor_pit_h21

# LightGBM at 21-day with default HPs (the ML control)
python -m price_model.cli run -e extended_kaggle_v3_curated_h21

# LightGBM at 21-day with held-out Optuna HPs (the methodologically-strict ML)
python scripts/optuna_sweep.py --experiment extended_kaggle_v3_curated_h21 \
    --n-trials 100 --max-date 2024-12-31 --write-tuned
python -m price_model.cli run -e extended_kaggle_v3_curated_h21_hp_pre20241231
```

For the 5-day-horizon companion table, swap the `_h21` suffixes off:

```bash
python -m price_model.cli run -e lasso_elasso_pit            # 9-feature L1 regression, 5-day
python -m price_model.cli run -e momentum_factor_pit         # momentum factors 5-day
python -m price_model.cli run -e extended_kaggle_v3_curated  # LightGBM 5-day (default HPs)
```

### Step 2 — produce the headline comparison tables

The comparison scripts pull predictions from the DuckDB store, intersect dates across all loaded models, join to realized targets at the requested horizon, and compute per-model gross IC, t-stat, hit rate, and long-short Sharpe.

```bash
# Gross-IC comparison at 21-day on 2025+ — the headline table
python scripts/compare_apples_to_apples.py --since 2025-01-01 --horizon 21

# Net-of-cost comparison at 21-day on 2025+ — the deployment table
python scripts/compare_net_of_cost.py --since 2025-01-01 --horizon 21

# Same comparisons at 5-day horizon (legacy / companion analysis)
python scripts/compare_apples_to_apples.py --since 2025-01-01
python scripts/compare_net_of_cost.py --since 2025-01-01
```

The apples-to-apples script prints a diagnostic line identifying the model whose end-date binds the comparison's date intersection — useful when the headline slice unexpectedly truncates. The `--since` flag restricts the eval window; `--until` is available for symmetric pre-cutoff splits.

### Step 3 (optional) — methodology audits

The Methodology section references three additional diagnostic scripts:

```bash
# Verify L1 cancellation hypothesis on the pure-momentum panel (see Discussion)
PYTHONPATH=src python scripts/inspect_momentum_lasso_coefficients.py

# Apply Bailey-López de Prado deflated Sharpe correction across all models
PYTHONPATH=src python scripts/deflated_sharpe_audit.py

# LightGBM gain-importance audit of the v3_curated panel
PYTHONPATH=src python scripts/audit_lightgbm_features.py \
    --experiment extended_kaggle_v3_curated
```

The expected numbers in the comparison tables are deterministic for a fixed data snapshot. Small drift (< 5%) is expected as yfinance updates and Ken French refreshes monthly. The held-out Optuna sweeps are non-deterministic across re-runs (TPE sampler with a different random seed); the IC results have been stable within ±0.001 across attempted re-runs.

## Methodology

The headline result is produced by a walk-forward, cross-validated, point-in-time-corrected backtest. This section documents each component.

### Universe: S&P 500 with point-in-time membership filter

**Point-in-time (PIT) correction** restricts the cross-section evaluated on each date `t` to tickers that were *actually members of the S&P 500 index on date `t`* — not the tickers that *are* members today. The correction defends against survivorship bias: today's S&P 500 list disproportionately contains companies that survived, by selection. A backtest evaluated on today's list silently assumes the model would have "known" in 2018 to focus on the 2026 survivors.

The mechanics:

- `src/price_model/data/sources/sp500_membership.py` scrapes the Wikipedia "List of S&P 500 companies" page and its "Selected changes" table to reconstruct, for every ticker that has ever been in the index between 2014 and the present, an `(added_date, removed_date)` membership window.
- `src/price_model/data/membership.py::filter_panel_to_pit(panel)` drops every row whose `(date, ticker)` pair falls outside that ticker's membership window. A stock added on 2020-06-22 contributes only its post-2020-06-22 rows; a stock removed on 2019-04-30 contributes only its pre-2019-04-30 rows.

The PIT correction in this project is **partial**: yfinance does not return data for ~12% of historical S&P 500 tickers (SIVB, FRC, ATVI, AGN, etc. — see [Data quality and methodological limitations](#data-quality-and-methodological-limitations)). A bias-free PIT analysis requires a paid feed (CRSP or Norgate Premium Data).

### Forecast target and walk-forward training

The forecast target is the cross-sectional excess log-return over a fixed horizon. For each date `t` and ticker `i`, the target `y[t, i]` is the log-return of `i` from `t+1` to `t+H` minus the cross-sectional mean log-return over the same window. The headline uses `H = 21` trading days (approximately one month); the same framework is implemented at `H = 5` and is configurable per experiment via `target_horizon` in the YAML configs.

**Walk-forward training** refits every model on an expanding window:

- `min_train_days = 504` — first prediction date is approximately 2 years after panel start (warmup period).
- `refit_freq_days = 252` (linear and momentum models) or `21` (LightGBM) — annual or monthly refit cadence.
- `embargo_days = H + 1` — 6 days for 5-day target, 22 days for 21-day target. The embargo ensures the last training label (which depends on dates `t+1` through `t+H`) does not overlap with the first validation prediction.

For the 21-day headline result, the walk-forward harness produces 8 refits across the 2017→2026 panel. Predictions in 2025-01 → 2026-01 use weights from a refit whose training data ended in late 2024 (causally OOS for 2025+ dates); predictions in early 2026 → 2026-04 use weights from a refit whose training data ended in late 2025.

**Lookahead safety** is verified by the "truncation-invariance" leakage test in `tests/test_leakage.py`: the prediction at date `t` must be unchanged when the entire panel is truncated to dates `≤ t`. Every feature and every model is exercised against this test in CI.

### Inner cross-validation for regularization-strength selection

For the L1 (Lasso), L2 (Ridge), and L1+L2 (ElasticNet) cross-sectional models, the regularization strength is selected by **5-fold inner cross-validation within each refit's training window**:

- `LassoCV` (sklearn): 100-point geometric grid over α, 5-fold CV, α chosen to minimize pooled CV mean-squared error.
- `RidgeCV` (sklearn): 20-point log-spaced grid over α (project override of sklearn default which is too coarse), 5-fold CV, generalized CV.
- `ElasticNetCV` (sklearn): joint search over α and l1_ratio (default grid `[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`), 5-fold CV.

The CV splits are random within the training window (sklearn default). No user-specified α; sklearn picks a data-appropriate grid. This means the L1 regression has **no user-tunable hyperparameters at deployment time** — α is fully data-driven per refit. The surviving non-zero coefficients are themselves an empirical finding interpretable in factor-zoo terms.

The walk-forward, inner-CV combination means each refit's α is selected on data the deployment slice has never seen.

### Held-out Optuna protocol for LightGBM hyperparameter search

LightGBM has more hyperparameters than CV can search inside a refit. Instead, the project uses **Optuna with purged walk-forward CV** (de Prado, *Advances in Financial Machine Learning*, Ch. 7) for hyperparameter selection.

The HP-selection bias correction is provided by the `--max-date` flag in `scripts/optuna_sweep.py`: dates after the specified cutoff are excluded from the matrix before CV folds are constructed, so the chosen HPs are provably independent of any data in the post-cutoff evaluation slice.

Four sweeps (all evaluated on the 2025+ apples-to-apples slice):

| Sweep | Target horizon | Cutoff | Trials | CV mean IC (training) | 2025+ OOS IC |
|---|---|---|---|---|---|
| HP-leaked baseline | 5-day | none — HPs saw all data | 100 | +0.02476 | +0.0046 |
| Split A (held-out) | 5-day | ≤ 2023-12-31 | 100 | +0.01934 | +0.0015 |
| Split B (held-out) | 5-day | ≤ 2024-12-31 | 100 | +0.01540 | +0.0080 |
| 21-day held-out | 21-day | ≤ 2024-12-31 | 100 | +0.01709 | +0.0187 |

The chosen hyperparameters differ substantially across cutoffs (e.g., 5-day Split A picked `lambda_l1=1.79`, 21-day held-out picked `lambda_l1=0.10` with `lambda_l2=44.4` — 18× lower L1, 222× higher L2). CV mean IC computed inside Optuna's training window is not a reliable predictor of OOS IC; see [Discussion](#discussion) for analysis.

### 9-feature anomaly panel (headline L1 regression)

The headline result `lasso_elasso_pit_h21` uses a 9-feature panel curated on the principle of "one feature per economically distinct anomaly family":

| # | Feature | Anomaly family | Reference |
|---|---|---|---|
| 1 | `momentum_12_1` | Cross-sectional momentum (12-month, skip-1-month) | Jegadeesh-Titman 1993, *J. Finance* 48(1) |
| 2 | `momentum_756` | Long-horizon trend (36-month) | Moskowitz-Ooi-Pedersen 2012 (TSMOM-adjacent) |
| 3 | `return_1d` | Short-term reversal | Lehmann 1990 / Jegadeesh 1990 |
| 4 | `vol_ewm_20` | Low-volatility | Ang-Hodrick-Xing-Zhang 2006 |
| 5 | `idio_vol_20` | Idiosyncratic vol (60-day beta residual) | Ang-Hodrick-Xing-Zhang 2006 |
| 6 | `max_return_21d` | MAX (lottery-stock) effect | Bali-Cakici-Whitelaw 2011 |
| 7 | `distance_52w_high` | 52-week-high anchoring | George-Hwang 2004 |
| 8 | `log_dollar_volume` | Size / liquidity | Amihud 2002 |
| 9 | `beta_60` | Market exposure control (BAB-adjacent) | Frazzini-Pedersen 2014 |

The 9 features were chosen ex-ante to span economically distinct mechanisms. Pairwise correlations within the panel are typically below 0.3 (verified empirically), satisfying the L1-stability prerequisite. Each feature is **rank-normalized within each date** (cross-sectional rank, then scaled to `[0, 1]`) — robust to fat tails, matches the asset-pricing literature convention, and prevents any single outlier observation from dominating the regression. The curation principle follows Han-He-Rapach-Zhou (2024).

### Net-of-cost decomposition

Gross IC and gross long-short Sharpe assume costless rebalancing. Net-of-cost Sharpe accounts for transaction-cost drag from portfolio turnover. For each date, **daily turnover** is computed as the L1 distance between today's and yesterday's normalized long-short weight vectors. Annual turnover = `mean(daily turnover) × 252`. **After-cost daily return** = gross return − `daily_turnover × cost_bps / 10000`. After-cost Sharpe is computed on the net-return series with the same `√(252 / horizon)` annualization as gross Sharpe.

Reported at 3 bp (institutional all-in), 10 bp (small-fund), and 20 bp (retail-equivalent). The full decomposition is in `scripts/compare_net_of_cost.py`. Turnover and cost-drag mechanics are implemented in `src/price_model/eval/turnover.py`.

### Bailey-López de Prado deflated Sharpe (multi-test correction)

`scripts/deflated_sharpe_audit.py` applies the deflated Sharpe ratio (Bailey & López de Prado 2014) across all project models. The correction inflates the effective number of trials by the project's experiment count and accounts for skew and kurtosis of the return distribution to compute `P(true_Sharpe > 0 | observed)`. Threshold for "significant after multi-test correction" is DSR > 0.95.

The deflated-Sharpe pass list for the 21-day comparison aligns with the t-stat ranking — the high-t-stat L1 regression and momentum factors clear DSR > 0.99; the LightGBM held-out-tuned at t = +3.02 clears DSR > 0.95; lower-t-stat models do not clear.

### Pure-momentum Lasso cancellation diagnostic

`scripts/inspect_momentum_lasso_coefficients.py` fits Lasso and ElasticNet on a four-feature pure-momentum panel (12-1, 18-month, 24-month, 36-month) and prints the fitted coefficients alongside the feature correlation matrix. The diagnostic confirms:

1. Pairwise correlations between momentum features at 0.48–0.82 (e.g., `momentum_504 ↔ momentum_756` at 0.75).
2. The CV-selected α drives all coefficients to (near) zero — the model collapses to predicting the intercept. The realized predictions inherit the sign of the small noisy intercept, producing systematic anti-correlation with realized returns.
3. ElasticNet picks l1_ratio close to 1.0 (essentially pure Lasso) — L2 stabilization does not help on a heavily-collinear panel.

This diagnostic supports the discussion of why pure-momentum L1 produces negative OOS IC (see [Discussion](#discussion)).

### Regime indicator and audit-driven LightGBM v3 panel

The 14-feature LightGBM v3 panel includes `cs_return_dispersion_20` — the cross-sectional standard deviation of daily log returns smoothed by 20-day rolling mean. The feature is a lookahead-safe regime conditioning variable that distinguishes high-dispersion (idiosyncratic-pricing) regimes from low-dispersion (uniform-market-move) regimes without using forward-looking labels. Implementation in `src/price_model/features/cross_features.py::CsReturnDispersion20`; lookahead safety verified by `tests/test_microstructure_features.py`.

The v3 panel curation (14 features chosen from a 21-feature candidate panel via gain importance + cross-feature correlation analysis) is documented inline in `config/experiments/extended_kaggle_v3_curated.yaml`; audit script in `scripts/audit_lightgbm_features.py`.

## Discussion

The headline finding — a 9-feature L1 regression beats every tested ML variant at 21-day horizon, with long-horizon momentum factors winning on net-of-cost Sharpe — is decisive within the scope of this study but raises four questions worth unpacking. This section addresses each.

### Why the L1 regression dominates ML on this universe

**The 21-day cross-sectional signal on the post-2024 S&P 500 is approximately linear in feature space.** Three pieces of evidence:

1. **The momentum factors alone capture most of the signal.** mom_504, mom_756, and mom_378 individually produce IC = +0.04 to +0.05 with no model fitting; the L1 regression's incremental edge of +0.02 IC over the best single momentum factor reflects modest additional signal from non-momentum features (low-volatility, idiosyncratic volatility, MAX effect). The dominant component is a linear momentum signal that trees would have to recover via many small splits.
2. **Tree-ensemble gain-importance audits show feature concentration.** On the 14-feature v3 panel, three features produce 39% of LightGBM gain. The split structure devotes most of its capacity to a handful of features that L1 can express as a small number of coefficients. Trees' theoretical interaction-capturing advantage doesn't pay off here because the dominant signal isn't interactive.
3. **Optuna-tuned LightGBM still falls short.** Even when the LightGBM HPs are explicitly searched to maximize OOS IC, the result trails the L1 regression by 0.04+ IC. The gap is not a tuning failure.

Two caveats limit how broadly this conclusion generalizes:

- **The 9-feature panel is small and ex-ante-curated.** The L1 regression's advantage depends on this curation. On the broader 14-feature LightGBM panel (audit-driven, no economic-family constraint), the same L1 recipe produces a weaker signal.
- **The post-2024 regime is structurally momentum-friendly.** In a regime with more leadership rotation or weaker momentum persistence, the linear-on-momentum recipe would compress and ML might catch up. The result is robust within the studied 16-month evaluation period but has not been verified across regime changes.

The conclusion is therefore "in this regime on this universe at this horizon, L1 regression on a carefully curated 9-feature panel outperforms every tree-ensemble variant tested" — not "ML cannot succeed in cross-sectional equity prediction generally."

### Why long-horizon momentum wins after transaction costs

**Turnover dominates the gross-vs-net comparison.** The 36-month momentum factor has an annual turnover of 15× (it rebalances roughly twice per quarter). The L1 regression has 128× annual turnover (essentially full rebalancing every two days). LightGBM variants have 130–150× turnover. At 20 bp per side, the cost drag is:

- 36-month momentum: 15 × 20 bp × 2 / 10000 = 0.6% annual drag → retains 99% of gross Sharpe (+1.82 → +1.81).
- L1 regression: 128 × 20 bp × 2 / 10000 = 5.1% annual drag → retains 94% of gross Sharpe (+1.24 → +1.17).
- LightGBM (default): 130 × 20 bp × 2 / 10000 = 5.2% annual drag → retains 84% of gross Sharpe (+0.74 → +0.62).

The L1 regression's gross IC advantage of +0.019 over mom_756 (+0.0695 vs +0.0501) translates to a gross-Sharpe deficit of −0.58 once both are turnover-adjusted. Net at 20 bp, mom_756 leads by +0.64 Sharpe.

**Why the post-2024 regime amplifies the long-horizon advantage:** the Mag-7 megacap concentration and AI-sector winners-keep-winning pattern means multi-year leaders persist, so a 36-month momentum factor's holdings are stable across refits. In a regime with more frequent leadership rotation, the advantage would compress because mom_756's turnover would naturally rise. This is the cleanest regime-dependent finding in the project.

### Why hyperparameter optimization is fragile to regime shift

**Optuna minimizes CV error on training data, not OOS error on deployment data.** When the training period is a different statistical regime from the deployment period, the HPs that minimize training-period CV error don't minimize deployment-period OOS error. The gap between train-optimal HPs and deploy-optimal HPs grows with regime divergence.

This is exactly the **DeMiguel-Garlappi-Uppal (2009)** result generalized: they showed that the optimal portfolio weights estimated from historical returns underperform the equal-weight (1/N) portfolio out-of-sample because estimation error in optimal weights overwhelms the theoretical benefit of optimization. The same logic applies to HP search:

- **Default HPs are like 1/N.** They don't fit training data as hard, so they generalize better.
- **Optuna-selected HPs are like Markowitz weights.** They minimize a training-period objective but inherit the estimation error.

The 21-day result is the cleanest demonstration: held-out Optuna at 21-day target scored OOS IC +0.0187 vs default HPs' +0.0274. Even with the deployment slice strictly excluded from HP selection, the chosen HPs underperformed defaults by ~0.009 IC.

**Practical implication:** an Optuna-tuned model should not be deployed without first verifying its OOS IC exceeds the default-HP OOS IC on a slice the HPs never saw. In this study, that verification step would have rejected the 21-day Optuna result entirely.

The 5-day evidence is mixed (Split B beats Split A on 2025+ OOS but Split A beats Split B on full-sample), which is itself a regime-mismatch signature: the HPs Optuna chose with more recent training data fit recent patterns better but generalized worse to earlier periods.

### Why pure-momentum L1 produces statistically significant negative IC

The mechanism is **L1 cancellation on collinear features**, with a twist documented by the diagnostic script:

1. **The pure-momentum panel has high pairwise correlations.** mom_504 ↔ mom_756 = 0.75; mom_378 ↔ mom_504 = 0.82; mom_12_1 ↔ mom_378 = 0.72. All four features measure "did this stock go up over some multi-month window."
2. **At the α level CV selects, all coefficients are driven to zero.** The model collapses to predicting the intercept term, which on rank-normalized features is approximately zero. The realized predictions inherit the sign of the small noisy intercept.
3. **The noisy intercept is systematically anti-correlated with realized returns** across walk-forward refits, producing statistically significant negative IC.

ElasticNet does not help because the L2 component cannot stabilize the unstable L1 subspace at the CV-selected α magnitude.

The positive corollary: **L1 regularization is not a free lunch on factor zoos.** The Han-He-Rapach-Zhou (2024) paper's substantive contribution is the *panel-curation principle* (one feature per economic family), not the choice of regularization. This project's results confirm that empirically — the curated 9-feature panel produces +0.0695 IC; the same regularization on a single-family panel produces −0.018 IC. **Feature-panel design matters more than regularization choice.**

### Synthesizing thesis

The four observations above point to one unifying conclusion: **on the post-2024 S&P 500 at the 21-day horizon, the cross-sectional return signal is dominated by slow-moving systematic patterns (long-horizon momentum + a small set of documented anomalies) that linear models with conservative regularization recover more efficiently than tree ensembles.** Tree ensembles have the theoretical capacity to express any function the linear model can, but in practice they pay an estimation-error tax that — combined with their higher turnover — produces uniformly worse gross and net results in this regime.

A cleaner test of "is this finding regime-specific or general" would require running the same comparison on multiple non-overlapping deployment windows (e.g., 2010–2014, 2014–2018, 2018–2022, 2022–2026) with paid PIT-correct data. That multi-window stress test is the most-needed extension of this work and is enumerated in [Scope and limitations](#scope-and-limitations).

## Data quality and methodological limitations

The primary out-of-sample estimate is honest about what it measures,
but what it measures is bounded by the data available without a paid
feed. This section enumerates those boundaries.

### Limitations of yfinance

yfinance is the only free source of daily-bar US equity data and is the
reason the project is reproducible without a paid subscription. It has
three documented failure modes that materially affect the reported
results.

1. **Delisted-ticker history is permanently lost.** When a company is
   acquired, fails, or goes private, yfinance stops returning data for
   the old symbol. SIVB (Silicon Valley Bank, failed March 2023), FRC
   (First Republic, failed May 2023), SBNY (Signature Bank, failed March
   2023), ATVI (Activision, acquired by Microsoft October 2023), AGN
   (Allergan → AbbVie 2020), and CERN (Cerner → Oracle 2022) have no
   usable pre-event history accessible via yfinance. Approximately 21
   such tickers are enumerated in the project drop list, documented
   inline in `src/price_model/data/tickers.py`.
2. **Symbol-parser fragility.** yfinance fails to fetch some single- and
   short-letter tickers due to ambiguity with currency or commodity
   symbols in its routing — for example, `K` (Kellanova), `FI` (Fiserv),
   and `DAY` (Dayforce), all currently live, exchange-listed US
   large-caps. yfinance returns "no data" after retries.
3. **Foreign listings are unreachable.** Acquired companies that
   consolidated under non-US ADRs cannot be fetched. `SIE.DE` (Siemens
   Healthineers, acquired Varian) and `MC.PA` (LVMH, acquired Tiffany)
   are examples, despite active trading on European exchanges.

**Coverage.** Of the 701 ticker symbols that Wikipedia identifies as
S&P 500 members at some point during 2017-2026, yfinance provides usable
data for 617 (~88%). The remaining ~12% are absent from the panel.

### Partial PIT reconstruction from Wikipedia

The Wikipedia "List of S&P 500 companies" page and its "Selected changes"
table are the only free source of historical index membership. The scraper
in `src/price_model/data/sources/sp500_membership.py` reconstructs the
`(ticker, added_date, removed_date)` table from those sources. Three known
incompleteness modes remain.

1. **The change log is reliable only back to approximately 2014.** Earlier
   add / remove events are absent. This does not affect the 2017-start
   evaluation window directly, but it implies that any pre-2017 extension
   (for example, applying the same machinery to a 1990s-2010s sample)
   would inherit incomplete PIT membership.
2. **Renames and continuations are ambiguous.** When a company changes
   its ticker (FB → META, RTN → RTX, FISV → FI), Wikipedia logs the event
   as a simultaneous remove and add on the same date. The scraper treats
   such pairs as continuous membership at the *new* symbol; the alias is
   resolved in `tickers.py`. The choice is correct for backtest purposes
   but implies that a researcher reading the membership table directly
   may miscount index changes by approximately 30 events over the full
   window.
3. **Wikipedia editors can be incorrect, late, or partial.** Missing
   entries (e.g., short-lived 2018 additions never logged as removals)
   were observed during the build. The membership table is not
   independently audited against a primary source.

### Ticker resolution rules

Three small lookup tables are maintained in
`src/price_model/data/tickers.py` to bridge the gaps above:

| Table | Purpose | Size |
|---|---|---|
| `TICKER_ALIASES` | Map renamed symbols to their successor (FB → META, RTN → RTX, ~70 entries documented with rename dates) | 73 entries |
| `TICKER_DROP_LIST` | Symbols with no usable yfinance data (failed banks, acquired with non-unified history, foreign listings, short-ticker parser failures) | ~25 entries |
| `SYMBOL_NORMALIZATION` | Punctuation convention (BRK.B → BRK-B) | 2 entries |

The function `resolve_ticker(symbol) -> str | None` applies all three in
order. Every ticker in every universe file is routed through it. Tests in
`tests/test_tickers.py` cover the precedence rules and known edge cases.

These tables are derived heuristically: `cli refresh-data` is run on the
expanded universe and the resulting yfinance failure log is inspected.
Each entry carries a one-line comment identifying the corporate event and
the rationale for inclusion. New entries are added when new failures are
observed.

### Sources of upward bias in the primary estimate

The primary IC of +0.0183 should be interpreted with the following
caveats.

1. **PIT correction is partial.** Because yfinance does not return data
   for SIVB, FRC, ATVI, AGN, and similar delisted symbols, those tickers
   do not appear in the PIT panel even when Wikipedia indicates they were
   index members on the relevant dates. The PIT-corrected backtest
   therefore still excludes the worst realized losers of the 2022-2023
   banking crisis. A fully PIT-correct analysis using paid data (Norgate,
   Polygon, or CRSP) would either misrank or correctly short SIVB at
   −85% on 2023-03-08; the present model does neither.
2. **The 61% bias estimate is a lower bound.** It reflects the share of
   apparent edge attributable to selection *given the data available*.
   The actual selection bias on a fully PIT-correct dataset (with
   delisted history and complete pre-2014 membership) would be larger.
   The IC drop from +0.0142 to +0.0055 on the 22-feature model is
   therefore a floor on backtest inflation, not a ceiling. For reference,
   the same effect on the original 13-feature technical-only baseline
   was an 89% collapse (+0.0075 → +0.0008); weaker feature sets exhibit
   higher survivorship-bias inflation.
3. **The post-2022 regime contains the bank-failure period (March-May
   2023).** The reported +0.0183 IC over 905 days post-October-2022 is
   computed on a cross-section that excludes the tickers that
   catastrophically failed in that window. A real-world model would
   need to predict (or fail to predict) those failures; the present
   model does not face that test. The accurate framing of the reported
   IC is "+0.0183 on the survivors of the regime, given the available
   data."
4. **Transaction costs, taxes, and slippage are zero in the backtest.**
   All ICs and Sharpes assume costless rebalancing. A retail investor
   faces bid-ask spreads (~5-10 bp), commissions, capital-gains tax, and
   slippage; the reported Sharpe of +0.86 is gross. After realistic
   retail costs, the after-cost Sharpe on a 10-30 ticker portfolio
   approaches zero. An institutional desk paying ~1-3 bp all-in could
   plausibly net a Sharpe in the 0.4-0.6 range from this signal, which
   would not constitute a standalone strategy.
5. **Single evaluation window.** The 2017-2026 window was selected as
   the range yfinance reliably covers. Alternative windows (2010-2020,
   2017-2026, 2020-2026) would produce different deltas at each stage;
   a multi-window robustness check has not been performed.

### Toward a fully PIT-correct evaluation

Listed in approximate order of effort:

- **Replace yfinance with Norgate Premium Data (~$60 / month)** for
  survivorship-bias-free prices including delisted history. All five IC
  numbers would change; the primary +0.0183 IC would likely fall by
  0.002-0.005, but the regime-conditional shape (post-2022 strong,
  pre-2022 negative) is expected to persist.
- **Replace the Wikipedia membership source with CRSP** (paid; free
  academic access for most affiliated researchers). Provides cleaner
  pre-2014 history and eliminates scrape fragility.
- **Run multi-window robustness tests** at three different start / end
  pairs.
- **Add transaction-cost modeling** at the prediction-store layer so
  reported metrics are net of realistic costs.

## Scope and limitations

- **Not deployable for retail trading.** After bid-ask spreads, commissions, and capital-gains taxes, even the headline L1 regression at +0.0697 IC and gross Sharpe +1.24 yields an after-cost edge that is meaningful only for institutional cost levels (~3 bp all-in). At retail cost levels (~10-30 bp per side once you include slippage and spread), and on the small portfolio sizes a retail account can hold, the after-cost edge approaches zero. The result is institutional-grade gross, not retail-grade net.
- **The edge is regime-concentrated.** The 2025-01-02 → 2026-04-27 evaluation window covers a 16-month period characterized by Mag-7 dominance, multi-year momentum persistence, and AI-sector concentration. Long-horizon momentum factors and the L1 regression both benefit from this regime; their pre-2022 IC was much weaker. The structural ranking (linear-on-anomalies > momentum > ML) has not been verified across regime changes. A bull-to-bear or risk-on-to-risk-off transition could materially compress the L1-regression advantage or reverse the ML-vs-linear ranking.
- **Single evaluation period, no multi-window robustness.** All headline numbers are computed on one OOS window (2025+). A multi-window stress test — evaluating the same models on 2018-2020, 2020-2022, 2022-2024, and 2024-2026 in sequence — has not been performed. This is a known limitation that a paid-data version of the analysis would address.
- **Survivorship-bias is only partially corrected.** The PIT correction excludes ~12% of historical S&P 500 members because yfinance does not return data for delisted symbols (SIVB, FRC, ATVI, AGN, etc.). The reported ICs are floors on the true edge a paid-data version would produce, not estimates of it. See [Data quality and methodological limitations](#data-quality-and-methodological-limitations).
- **Not a substitute for index funds.** For individual investors, decades of research show that low-cost diversified index funds outperform almost all active strategies after fees and taxes.

## Architecture

```
src/price_model/
├── data/
│   ├── sources/         # yfinance + Ken French + Wikipedia adapters
│   ├── universes/       # static universe files (sp500.txt, sp500_pit.txt)
│   ├── tickers.py       # aliases / drop list / normalization rules
│   ├── membership.py    # PIT membership lookup + filter
│   └── loaders.py       # one-call load_panel(universe, start, pit_filter)
├── features/            # technical, cross-sectional, factor-loading, anomalies
├── models/              # LightGBM, baselines, classical (ARIMA, GARCH, GBM, FF)
├── pipeline/            # walk-forward backtest harness
├── eval/                # IC, hit rate, Sharpe, bootstrap CI, time-split
├── serving/             # DuckDB prediction store
├── dashboard/           # Streamlit dashboard reading from the store
└── cli.py               # `price-model` entry point

config/experiments/      # YAML configs for each stage above
notebooks/               # diagnostic + classical + robustness + portfolio + features
tests/                   # leakage tests, PIT tests, ticker tests, contract tests
```

Notebooks of note:

- `notebooks/04_portfolio_attribution.ipynb` — uses the Ken French adapter directly (not the model layer) to decompose a 10-stock equal-weight portfolio's exposures and attribute realized returns to factors. Independent application of the data layer; does not depend on the predictive model.
- `notebooks/05_feature_exploration.ipynb` — boilerplate for understanding the project's 41 registered features and adding new ones. Covers feature distributions, cross-sectional dispersion over time, the pairwise correlation matrix on the headline 9-feature panel, fitted L1 regression coefficients, LightGBM gain importance, and a step-by-step template for adding a new feature.

### Streamlit dashboard

`src/price_model/dashboard/` is a thin Streamlit reader over the DuckDB
prediction store. It exists to allow the model's daily output to be
inspected visually (per-date top / bottom quintile, rolling IC,
prediction vs. realized scatter) without writing a notebook for each
inspection. The dashboard is a debugging and monitoring surface, not
part of the reproduction workflow. None of the reported numbers in
this README originate from the dashboard, and skipping it does not
affect reproduction of any ablation cell. The dashboard is most useful
when extending the model and a quick sanity-check view of new runs is
desired.

## Development notes

The substantive research decisions in this project — what to measure,
what constitutes a fair test, how much survivorship bias the primary
estimate contains, which limitations are honest to ship with — are the
author's.
The engineering side of the project (test scaffolding, CI configuration,
package layout, refactors, and the translation from research intent to
reproducible implementation) was developed in close collaboration with
Claude Code. Specifically:

- The walk-forward harness, prediction store, and dashboard scaffold
  were designed and iterated in conversation with Claude Code.
- Test coverage (leakage tests, PIT-membership tests, contract tests on
  the loaders, ticker-resolver tests) was co-designed; Claude Code
  drafted many of the test cases that subsequently surfaced bugs in the
  implementation.
- The GitHub Actions CI pipeline (ruff + pyright + pytest on Python
  3.11) and the pre-commit hooks were configured with Claude Code.
- Code review, refactors, and a final pre-publish audit pass were
  conducted in collaboration with Claude Code.

The README, the experiment configs, and the ablation narrative are the
author's claims; the engineering quality of the surrounding codebase is
a product of the above collaboration.

## Citations

- **Han, Y., He, A., Rapach, D., and Zhou, G.** (2024). "Expected Stock Returns and the Cross-Section: An E-LASSO Approach." *Review of Finance*. — Panel-curation principle (one feature per economic family, rank-normalized, CV-selected α) used in the headline 9-feature L1 regression (`lasso_elasso_pit_h21`).
- **Jegadeesh, N. and Titman, S.** (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance* 48(1). — 12-1 momentum anomaly.
- **Moskowitz, T., Ooi, Y. H., and Pedersen, L. H.** (2012). "Time Series Momentum." *Journal of Financial Economics* 104(2). — Long-horizon (TSMOM) momentum.
- **Lehmann, B.** (1990). "Fads, Martingales, and Market Efficiency." *Quarterly Journal of Economics* 105(1). — 1-day reversal.
- **Ang, A., Hodrick, R., Xing, Y., and Zhang, X.** (2006). "The Cross-Section of Volatility and Expected Returns." *Journal of Finance* 61(1). — Low-vol and idiosyncratic-vol anomalies.
- **Bali, T., Cakici, N., and Whitelaw, R.** (2011). "Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns." *Journal of Financial Economics* 99(2). — MAX (lottery) effect.
- **George, T. and Hwang, C.-Y.** (2004). "The 52-Week High and Momentum Investing." *Journal of Finance* 59(5). — 52-week-high anchoring.
- **Amihud, Y.** (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets* 5(1). — Liquidity / size proxy.
- **Frazzini, A. and Pedersen, L. H.** (2014). "Betting Against Beta." *Journal of Financial Economics* 111(1). — Market-beta control.
- **Fama, E. and French, K.** (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics* 116(1). — FF5 baseline (reference only; not in the headline comparison due to Kenneth French release-lag truncation).
- **Zou, H. and Hastie, T.** (2005). "Regularization and Variable Selection via the Elastic Net." *Journal of the Royal Statistical Society, Series B* 67(2). — ElasticNet model class.
- **Bailey, D. and López de Prado, M.** (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management* 40(5). — Multi-test Sharpe correction.
- **DeMiguel, V., Garlappi, L., and Uppal, R.** (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies* 22(5). — The estimation-error-dominates-optimization result referenced in the Discussion's "Why HP optimization is fragile to regime shift" subsection.
- **López de Prado, M.** (2018). *Advances in Financial Machine Learning.* Wiley. — Purged walk-forward cross-validation (Ch. 7).
- **Ken French Data Library.** Daily factor returns. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Wikipedia: List of S&P 500 companies.** Historical components and change log. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## License

MIT.
