# price-model

A point-in-time (PIT) corrected cross-sectional equity return predictor on the S&P 500, evaluated under strict HP-free held-out protocols at the 21-day (approximately monthly) forward horizon.

## Executive summary

On the 2025-01-02 → 2026-04-27 out-of-sample slice (334 dates, all hyperparameters and coefficients selected strictly on data preceding this period), a **9-feature L1 cross-sectional regression** on documented academic anomalies (per Han-He-Rapach-Zhou 2024) produces the strongest signal in the comparison at 21-day forward horizon: **IC = +0.0697, t = +5.24, long-short Sharpe = +1.24**. The same framework can be applied at 5-day horizon (the L1 regression scores IC = +0.0249, t = +2.72 there); the relative advantage of the linear approach widens dramatically at the monthly horizon classical asset-pricing studies use, so this README leads with the 21-day result.

Headline 21-day comparison on the 2025-01-02 → 2026-04-27 OOS slice (334 dates):

| Model | HP-selection | IC | t-stat | L/S Sharpe |
|---|---|---|---|---|
| **L1 regression, 9-feature anomaly panel** | inner CV | **+0.0697** | **+5.24** | **+1.24** |
| 24-month momentum factor | none | +0.0510 | +6.31 | +1.40 |
| 36-month momentum factor | none | +0.0503 | +7.14 | +1.82 |
| 18-month momentum factor | none | +0.0423 | +4.96 | +1.35 |
| JT 12-1 momentum (canonical) | none | +0.0312 | +3.51 | +1.04 |
| LightGBM, default HPs | none | +0.0278 | +4.35 | +0.75 |
| Ridge regression, 12-feature panel | inner CV | +0.0250 | +3.08 | +0.76 |
| LightGBM, held-out Optuna (≤ 2024-12-31) | held-out Optuna | +0.0191 | +3.02 | +0.75 |
| Pure-momentum Lasso, 4 momentum features | inner CV | **−0.0185** | **−2.79** | −0.21 |

**Deployment-relevant result.** The L1 regression produces the strongest gross signal, but the 36-month momentum factor produces the strongest *net* signal after transaction costs. Net 20 bp Sharpe ranking on the same 21-day OOS slice: 36-month momentum factor **+1.80** (15× annual turnover); 24-month momentum +1.39; 18-month momentum +1.33; **L1 regression +1.17** (128× turnover); LightGBM (default HPs) +0.63 (130× turnover); LightGBM (held-out Optuna) +0.61. The inversion is driven by an 8× turnover differential: the L1 regression retains 84% of its gross Sharpe at 20 bp; the 36-month momentum factor retains 99%. **On gross signal, the multi-anomaly L1 regression dominates; on deployment, the slow-moving single momentum factor dominates.** Both decisively beat every ML variant on both metrics.

Three structural claims about cross-sectional equity return prediction on liquid US large-caps at the 21-day forward horizon in the post-2024 regime:

**Claim 1 — Multi-anomaly L1 regression dominates ML.** L1 regression at 21-day produces IC +0.0697 vs the best tree-ensemble variant at +0.0278 — a 150% relative gap. Every LightGBM variant tested — including audit-curated feature panels, default hyperparameters, and a 100-trial Optuna sweep — fails to match the linear-on-documented-anomalies result. The signal the L1 model extracts is not accessible to gradient-boosted trees on this feature panel.

**Claim 2 — Hyperparameter optimization is fragile to regime shift.** At 21-day target horizon, **held-out Optuna (HPs only saw ≤ 2024-12-31) scores IC +0.0191 on the 2025+ slice, *lower* than the default-HP baseline at +0.0278.** Aggressive HP optimization underperformed defaults on an OOS slice it never trained on. The mechanism is visible in the chosen HPs: Optuna selected `lambda_l1 = 0.10, lambda_l2 = 44.4` at 21-day (dense + high L2 — a regime that fits training-period structure but does not generalize). The HP that minimizes train-set CV error and the HP that maximizes test-set OOS IC are not the same point, and the gap widens with regime divergence — consistent with DeMiguel-Garlappi-Uppal (2009)'s estimation-error-dominates-optimization result, generalized to HP search.

**Claim 3 — Linear regularization on collinear panels can produce significantly negative IC.** Pure-momentum Lasso on {12-1, 18-month, 24-month, 36-month} momentum features — pairwise correlations 0.6-0.8 — produces IC −0.019, t = −3.01 at 21-day horizon. Lasso and ElasticNet outputs are virtually identical (IC differ by 0.00005), meaning CV picked l1_ratio = 1.0 — L2 stabilization did not help. The 9-feature anomaly panel works because its features span economically distinct mechanisms (momentum, volatility, MAX effect, 52w-high, liquidity, beta) with low pairwise correlations; the pure-momentum panel fails because L1 cannot cleanly select among collinear siblings. **The feature panel matters more than the regularization method.**

Every number is reproducible from the YAML configs in `config/experiments/` and the comparison scripts in `scripts/`. All headline models in the table above are HP-selected on data strictly preceding the 2025-01-02 evaluation start. The [methodology appendix](#methodology-appendix) documents the held-out Optuna protocol, the deflated-Sharpe correction (Bailey & López de Prado 2014), the regime indicator (`cs_return_dispersion_20`), the audit-driven LightGBM panel curation, and the pure-momentum-Lasso cancellation diagnostic.

The result is **statistically rigorous, regime-conditional, and not deployable for retail investors** after bid-ask spreads, commissions, and capital-gains taxes. The numbers above are bounded by free-data quality and methodological choices that are enumerated explicitly — see [Data quality and methodological limitations](#data-quality-and-methodological-limitations) and [Scope and limitations](#scope-and-limitations).

### Metric definitions

- **Information Coefficient (IC).** Cross-sectional ranking quality of
  the model, averaged over time.

  *Per-date IC.* On each date `t` in the evaluation window, the model
  produces a prediction for every ticker in the cross-section that has a
  realized 5-day forward excess return available. Let `N_t` be the number
  of such tickers on date `t` — in the present universe `N_t` is on the
  order of 300-600. The per-date IC is the Spearman rank correlation
  between two vectors **of length N_t**: the vector of predictions and the
  vector of realized 5-day forward excess returns. Note that the vectors
  are not length 5; the "5" refers to the *horizon* of the return target,
  not the number of observations.

  *Time-averaged IC.* The reported IC is the unweighted average of the
  per-date ICs across all evaluation dates (`n_dates = 1758` full sample,
  `n_dates = 905` post-October-2022). A weak ranker on most days plus one
  strong day does not produce a high IC; consistent ranking is required.

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
  independent ranking would give roughly 0. The reported headline
  headline L1 regression 21-day IC of +0.0697 is therefore the **average over 334
  such daily correlations** across the 2025-01-02 → 2026-04-27 OOS slice,
  each computed across ~450 tickers. A per-date IC of +0.07 on average is
  small in absolute terms but large in t-stat (+5.24) due to the
  high date count — the consistency across days, not the magnitude on
  any one day, is what makes the signal credible.

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
  across all 24 project trials in `scripts/deflated_sharpe_audit.py`.
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

## Primary result: 9-feature L1 regression at 21-day horizon and three claims

The primary out-of-sample estimate is **IC = +0.0697 (t = +5.24, long-short Sharpe = +1.24) on the PIT-corrected universe at 21-day forward horizon, evaluated on the 2025-01-02 → 2026-04-27 OOS slice (334 dates)**. The model is a 9-feature L1 (Lasso) cross-sectional regression on documented academic anomalies, α selected by 5-fold inner CV on training data only — the recipe of Han-He-Rapach-Zhou (2024), applied to the project's sp500_pit panel with one feature per economically distinct anomaly family. The same recipe at 5-day target horizon scores IC = +0.0249 (t = +2.72) on the same eval slice; the 21-day result is reported as the headline because the linear-on-anomalies advantage is much sharper at the monthly horizon.

The structure of the rest of this section: three subsections, one per claim. Each pairs a comparison table with the supporting empirical evidence.

### Claim 1 — Multi-anomaly L1 regression dominates ML

The 21-day apples-to-apples comparison ranks **the L1 regression above every tested ML variant**:

| Model | 21-day OOS IC | t-stat | Δ vs L1 regression |
|---|---|---|---|
| **L1 regression, 9-feature anomaly panel** | **+0.0697** | **+5.24** | — |
| 24-month momentum factor | +0.0510 | +6.31 | −0.019 |
| 36-month momentum factor | +0.0503 | +7.14 | −0.019 |
| Ridge regression, 12-feature panel | +0.0250 | +3.08 | −0.045 |
| LightGBM, default HPs | +0.0278 | +4.35 | −0.042 |
| LightGBM, held-out Optuna (≤ 2024-12-31) | +0.0191 | +3.02 | −0.051 |

The same ranking holds at 5-day horizon (L1 regression +0.0249 vs best ML +0.0158) but the absolute gap is smaller because every model's IC is lower at 5-day. The 21-day horizon makes the linear-on-anomalies advantage stark.

Why the L1 regression works where ML fails: the 9-feature documented-anomaly panel spans **economically distinct mechanisms** (momentum, low-volatility, idiosyncratic volatility, MAX effect, 52-week-high anchoring, liquidity, market beta), with pairwise correlations typically below 0.3. L1 regularization can cleanly select among them, retaining 4–6 features per refit. The cross-sectional information ML extracts via tree splits on the same universe is — empirically — a strict subset of what the L1-on-curated-anomalies recipe extracts.

#### Net-of-cost refinement: gross-vs-net ranking inversion

The gross-IC ranking favors the L1 regression; the net-Sharpe ranking favors long-horizon momentum factors. Net-of-cost decomposition at 3, 10, and 20 bp per side on the same 21-day OOS slice:

| Model | Gross IC | Gross Sharpe | Annual turnover | Net @ 3bp | Net @ 10bp | Net @ 20bp |
|---|---|---|---|---|---|---|
| **36-month momentum factor** | +0.0503 | +1.82 | **15×** | +1.81 | +1.81 | **+1.80** |
| 24-month momentum factor | +0.0510 | +1.40 | 16× | +1.40 | +1.39 | +1.39 |
| 18-month momentum factor | +0.0423 | +1.35 | 21× | +1.35 | +1.34 | +1.33 |
| **L1 regression, 9-feature anomaly panel** | **+0.0697** | +1.24 | 128× | +1.23 | +1.21 | **+1.17** |
| JT 12-1 momentum (canonical) | +0.0312 | +1.04 | 25× | +1.04 | +1.03 | +1.02 |
| LightGBM, default HPs | +0.0278 | +0.75 | 130× | +0.74 | +0.69 | +0.63 |
| LightGBM, held-out Optuna | +0.0191 | +0.75 | 147× | +0.73 | +0.68 | +0.61 |

Two observations:

1. **The L1 regression retains 94% of its gross Sharpe at 20 bp** (+1.24 → +1.17). The 36-month momentum factor retains 99% (+1.82 → +1.80). The difference (5 percentage points) is small in relative terms but matters at the deployment margin: the momentum factor ends up with a +0.63 absolute Sharpe lead over the L1 regression on net basis, despite trailing it by +0.020 on gross IC.

2. **ML keeps losing on net.** The best ML variant (LightGBM, default HPs) drops from +0.75 gross to +0.63 at 20 bp. Held-out Optuna LightGBM drops to +0.61. Both are below every long-horizon momentum factor on net Sharpe by 0.4+ Sharpe units. The ML penalty is compounded by tree ensembles' higher turnover relative to their gross signal strength.

The net-of-cost story does not change which model classes win or lose; it just inverts the within-winner ranking between the L1 regression and the 36-month momentum factor. The headline finding remains "linear-on-documented-features beats tree ensembles," with the momentum factor (one feature) and the L1 regression (nine features) representing two valid points on that spectrum.

### Claim 2 — Hyperparameter optimization is fragile to regime shift

The headline 21-day comparison, evaluated on 2025-01-02 → 2026-04-27 (334 dates):

| Sweep | HP-selection cutoff | 2025+ OOS IC | t-stat |
|---|---|---|---|
| LightGBM, default HPs (no tuning) | — | **+0.0278** | +4.35 |
| LightGBM, held-out Optuna | ≤ 2024-12-31 | **+0.0191** | +3.02 |

At 21-day horizon, default HPs beat held-out Optuna by 0.009 IC. The Optuna sweep — even with the eval slice strictly excluded from HP selection — picked HPs (`lambda_l1 = 0.10, lambda_l2 = 44.4`, `learning_rate = 0.097`, 519 trees) optimized for the 2017–2024 training regime that did not generalize to 2025+.

This is consistent with DeMiguel-Garlappi-Uppal (2009)'s portfolio-optimization result generalized to HP optimization: estimation error in optimal parameters can overwhelm the theoretical benefit of optimization. The practical implication for cross-sectional equity ML: an Optuna-tuned model should not be deployed without first verifying its OOS IC exceeds the default-HP baseline. In this study, that verification step would have rejected the 21-day Optuna result entirely.

The same Optuna protocol was also run at 5-day target horizon under three cutoffs (no cutoff, ≤ 2023-12-31, ≤ 2024-12-31). The full-sample IC drops monotonically with cutoff tightness — quantifying ~19% selection-bias inflation in the no-cutoff version — but the OOS-on-2025+ ranking is mixed because of regime non-stationarity. The 5-day sweep details are in the methodology appendix; they support Claim 2 but don't constitute a cleaner test than the 21-day result reported above.

### Claim 3 — Linear regularization on collinear panels can produce significantly negative IC

A failure-mode experiment: fit Lasso and ElasticNet on **only the four momentum factors** (12-1, 18-month, 24-month, 36-month). Pairwise correlations within the panel range 0.48 to 0.82.

| Model | IC | t-stat | L/S Sharpe |
|---|---|---|---|
| Pure-momentum Lasso, 21-day | **−0.0185** | **−2.79** | −0.21 |
| Pure-momentum ElasticNet, 21-day | −0.0185 | −2.79 | −0.21 |

The results are **statistically significant negative IC** at t < −2.7. ElasticNet's CV consistently picked l1_ratio ≈ 1.0 (pure Lasso); L2 stabilization did not help. Lasso and ElasticNet predictions differ by IC 0.00005 — essentially identical.

The mechanism is the canonical L1 cancellation pathology on collinear regressors. With four features all measuring "stocks that went up over multiple years," Lasso's soft-thresholding cannot cleanly select among them. The chosen coefficients flip signs erratically across walk-forward refits, producing predictions that are anti-correlated with realized returns out of sample. The same recipe with diverse features (the 9 economically-distinct anomalies from Claim 1) produces the strongest signal in the comparison; on a collinear-only panel it produces statistically significant *negative* IC.

The diagnostic in `scripts/inspect_momentum_lasso_coefficients.py` confirms the cancellation pattern empirically.

This claim has a positive corollary: **L1 regularization is not a free lunch on factor zoos.** The Han-He-Rapach-Zhou (2024) paper's key contribution is not "use Lasso on more features" but rather "curate the feature panel to one-per-economic-family before applying L1." This project's results confirm that empirically — the curated 9-feature panel produces strong positive IC; the same regularization on a single-family panel produces statistically-significant negative IC.

### Legacy: v2 LightGBM survivorship-bias quantification

The original primary estimate of the project (before the L1-regression comparison was introduced) was the v2 LightGBM at 22 features. The 2×3 ablation matrix below remains valid as a survivorship-bias quantification on that panel — the more important finding from that analysis was *how much* IC the PIT correction removed, not the absolute IC magnitude. The headline result has since shifted to the 9-feature L1 regression, but the PIT mechanics defined here apply to every model in the project.

#### What "PIT correction" means in this project

**Point-in-time (PIT) correction** restricts the cross-section evaluated
on each date `t` to tickers that were *actually members of the S&P 500
index on date `t`* — not the tickers that *are* members today. The
correction defends against survivorship bias: today's S&P 500 list
disproportionately contains companies that survived, by selection. A
backtest evaluated on today's list silently assumes the model would have
"known" in 2018 to focus on the 2026 survivors.

The mechanics are isolated in two modules:

- `src/price_model/data/sources/sp500_membership.py` scrapes the
  Wikipedia "List of S&P 500 companies" page and its "Selected changes"
  table to reconstruct, for every ticker that has ever been in the
  index between 2014 and the present, an `(added_date, removed_date)`
  membership window.
- `src/price_model/data/membership.py::filter_panel_to_pit(panel)`
  takes the long-format `(date, ticker, ...)` price panel and drops
  every row whose `(date, ticker)` pair falls *outside* that ticker's
  membership window. A stock added to the index on 2020-06-22 therefore
  contributes only its post-2020-06-22 rows to training and evaluation;
  a stock removed on 2019-04-30 contributes only its pre-2019-04-30
  rows; a stock that joined in 2018 and left in 2022 contributes only
  the rows for dates between those events.

When `pit_filter=True` is passed to `load_panel(...)`, the membership
filter is applied immediately after the yfinance fetch and before
features or targets are constructed. The walk-forward harness then
trains and evaluates only on the PIT-filtered cross-section. When
`pit_filter=False`, the model sees today's universe back-projected
across all dates — the "subset" column of the matrix below.

The PIT correction in this project is partial because yfinance is
missing data for ~12% of historical S&P 500 tickers (SIVB, FRC, ATVI,
etc.; see [Data quality and methodological limitations](#data-quality-and-methodological-limitations)).
A truly bias-free PIT analysis requires a paid feed.

#### Universe × PIT × regime ablation (22-feature LightGBM)

| | Subset universe (160 modern-survivor tickers, PIT OFF) | PIT-corrected universe (617 historical tickers, PIT ON) |
|---|---|---|
| **Full sample** (1758 dates) | IC = +0.0142, t = +4.05, Sharpe = +0.39 | IC = +0.0055, t = +1.51, Sharpe = +0.21 |
| **Pre-Oct-2022** (853 dates) | IC = +0.0034, t = +0.64, Sharpe = −0.07 | IC = −0.0082, t = −1.38, Sharpe = −0.16 |
| **Post-Oct-2022** (905 dates) | IC = +0.0244, t = +5.37, Sharpe = +0.94 | **IC = +0.0183, t = +4.34, Sharpe = +0.86** ← primary estimate |

Three observations follow.

**1. Survivorship-bias collapse varies sharply by regime.** Across rows:
full-sample IC collapses from +0.0142 to +0.0055 under PIT correction
(a **61% reduction**); pre-October-2022 IC **flips sign** from a noisy
+0.0034 to a clear −0.0082; post-October-2022 IC collapses only 25%
(from +0.0244 to +0.0183). Survivorship bias dominates when the
underlying signal is weak (pre-2022 noise) and contributes far less when
the signal is strong (post-2022 regime). A practitioner running the
subset / full-sample backtest would observe +0.0142 IC at t = 4.05 and
infer a credible statistically significant edge, 61% of which would be
selection bias.

**2. The regime effect is large and robust to universe.** Down columns:
both universes show comparable regime shifts in Sharpe (subset:
−0.07 → +0.94; PIT: −0.16 → +0.86). The post-October-2022 regime
contains real cross-sectional signal that is not primarily a
survivorship artifact. The IC magnitudes shift similarly (subset:
+0.0034 → +0.0244, roughly 7×; PIT: −0.0082 → +0.0183, further amplified
by the sign flip).

**3. Pre-October-2022 IC on the PIT universe is negative with t = −1.38.**
The 22-feature model would have actively lost on the cross-section
before the regime break, not merely been flat. The finding is sharper
than "the edge is concentrated post-2022": the edge is post-2022 *and*
the opposite disedge is pre-2022. The features behave as a
regime-conditional intensifier, amplifying whichever direction the
cross-section is paying.

#### v2 LightGBM feature-set ablation

Holding the universe fixed at PIT-on and varying the feature set
(chronological order of additions):

| Features | Full sample IC | Full sample t | Post-Oct-2022 Sharpe |
|---|---|---|---|
| 13 technical only | +0.0008 | +0.24 | −0.04 |
| 13 + 3 academic anomalies | +0.0033 | +0.96 | +0.083 |
| **22 (+ 6 OHLCV / volume)** | **+0.0055** | **+1.51** | **+0.86** (regime split) |

The OHLCV / volume batch is the largest single-batch incremental
contribution. Each addition was tested on identical splits / embargo /
walk-forward settings in apples-to-apples runs; the YAML configs
(`extended_kaggle_v2_pit.yaml`, `extended_kaggle_v2_anomaly.yaml`,
`extended_kaggle_v2_ohlcv.yaml`) make this auditable.

## Reproduction

### Step 0 — install and build the PIT universe (one-time, ~10-15 min cold)

```bash
pip install -e ".[dev,classical]"

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

The methodology appendix references three additional diagnostic scripts:

```bash
# Verify L1 cancellation hypothesis on the pure-momentum panel (Claim 3)
PYTHONPATH=src python scripts/inspect_momentum_lasso_coefficients.py

# Apply Bailey-López de Prado deflated Sharpe correction across all models
PYTHONPATH=src python scripts/deflated_sharpe_audit.py

# LightGBM gain-importance audit of the v3_curated panel
PYTHONPATH=src python scripts/audit_lightgbm_features.py \
    --experiment extended_kaggle_v3_curated
```

The expected numbers in the comparison tables are deterministic for a fixed data snapshot. Small drift (< 5%) is expected as yfinance updates and Ken French refreshes monthly. The held-out Optuna sweeps are non-deterministic across re-runs (TPE sampler with a different random seed); the IC results have been stable within ±0.001 across attempted re-runs.

## Methodology appendix

### The 9-feature L1 regression anomaly panel

The headline result comes from `lasso_elasso_pit_h21`, which applies L1-regularized cross-sectional regression (Lasso) to a 9-feature anomaly panel. The panel-curation principle — one feature per economically distinct anomaly family, documented academic attribution required, rank-normalized within each date — follows Han-He-Rapach-Zhou (2024). The panel:

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

The 9 features were chosen ex-ante to span economically distinct mechanisms; pairwise correlations within the panel are typically below 0.3 (verified empirically), satisfying the L1-stability prerequisite. The α regularization strength is selected by 5-fold inner CV at each annual refit; the surviving non-zero coefficients are themselves an empirical finding interpretable in factor-zoo terms.

### Held-out Optuna protocol

The Optuna sweep optimizes mean cross-validation IC across purged walk-forward CV folds (de Prado, *Advances in Financial Machine Learning*, Ch. 7) within a fixed training window. The HP-selection bias correction is provided by the `--max-date` flag in `scripts/optuna_sweep.py`: dates after the specified cutoff are excluded from the matrix before CV folds are constructed, so the chosen HPs are provably independent of any data in the post-cutoff evaluation slice.

Four sweeps documented in the executive summary, all evaluated on the 2025+ apples-to-apples slice (350 dates at 5-day horizon, 334 dates at 21-day horizon):

| Sweep | Target horizon | Cutoff | Trials | CV mean IC (training) | 2025+ OOS IC | Notes |
|---|---|---|---|---|---|---|
| HP-leaked baseline | 5-day | none — HPs saw all data | 100 | +0.02476 | +0.0046 | Selection bias inflates CV IC; modest 2025+ OOS |
| Split A (held-out) | 5-day | ≤ 2023-12-31 | 100 | +0.01934 | +0.0015 | HPs only saw pre-2024; weak 2025+ OOS |
| Split B (held-out) | 5-day | ≤ 2024-12-31 | 100 | +0.01540 | +0.0080 | Recent HP-selection data → better 2025+ OOS than Split A |
| 21-day held-out | 21-day | ≤ 2024-12-31 | 100 | +0.01709 | +0.0191 | Underperforms 21-day default (+0.0278) |

The chosen hyperparameters differ substantially across cutoffs (e.g., 5-day Split A picked `lambda_l1=1.79`, 21-day held-out picked `lambda_l1=0.10` with `lambda_l2=44.4` — 18× lower L1, 222× higher L2), confirming HP non-stationarity across regimes. CV mean IC (computed on training data inside Optuna) is not a reliable predictor of 2025+ OOS IC; for the 21-day held-out sweep, CV mean IC of +0.01709 was lower than the eventual 2025+ OOS of +0.0191 — but still below the default-HP 2025+ OOS of +0.0278.

### Pure-momentum Lasso cancellation diagnostic

`scripts/inspect_momentum_lasso_coefficients.py` fits Lasso and ElasticNet on the four-feature pure-momentum panel and prints the fitted coefficients alongside the feature correlation matrix. The diagnostic confirms:

1. Pairwise correlations between momentum features at 0.48-0.82 (mom_504 ↔ mom_756 at 0.75, mom_378 ↔ mom_504 at 0.82).
2. The CV-selected α drives all coefficients to zero — the model collapses to predicting the intercept, which is approximately zero on rank-normalized data. The realized predictions inherit the sign of the (small, noisy) constant term, producing systematic anti-correlation with realized returns.
3. ElasticNet at this collinear panel picks l1_ratio ≈ 0.1 (near-Ridge) — confirming the L1 sparsity is the failure mode — but the alternate coefficient solutions also fail to generalize because the underlying issue is feature design, not regularization mix.

The diagnostic is mentioned in Claim 3 because the empirical mechanism is more nuanced than "L1 splits weight into cancelling signs." The actual mechanism is "L1 cannot find a stable subspace solution on heavily collinear features at the regularization scale CV selects, so it converges to all-zero."

### Walk-forward harness, embargo, refit cadence

All comparisons use the project's walk-forward harness in `src/price_model/pipeline/walk_forward.py`. Default parameters across the headline experiments:

- `min_train_days: 504` — first prediction is at least 2 years into the panel
- `refit_freq_days: 252` (linear/momentum models) or `21` (LightGBM) — annual or monthly refit
- `embargo_days: 6` (5-day target) or `22` (21-day target) — one day more than the target horizon to prevent label overlap leakage

Predictions for date `t` use weights trained on data through `t − embargo`. Lookahead safety verified via the "truncation-invariance" leakage test in `tests/test_leakage.py`: prediction at date `t` must be unchanged when the panel is truncated to dates ≤ `t`.

### Bailey-López de Prado deflated Sharpe (multi-test correction)

`scripts/deflated_sharpe_audit.py` applies the deflated Sharpe ratio (Bailey & López de Prado 2014) across all project models. The correction inflates the effective number of trials by the project's experiment count and accounts for skew and kurtosis of the return distribution to compute `P(true_Sharpe > 0 | observed)`. Threshold for "significant after multi-test correction" is DSR > 0.95.

The deflated-Sharpe pass list for the 21-day comparison aligns with the t-stat ranking — the high-t-stat L1 regression and momentum factors clear DSR > 0.99; the LightGBM held-out tuned at t=+3.02 clears DSR > 0.95; lower-t-stat models (XGBoost, CatBoost) do not clear.

### The regime indicator (`cs_return_dispersion_20`)

The 14-feature LightGBM v3 panel includes `cs_return_dispersion_20` — a contemporaneously-observable feature defined as the cross-sectional standard deviation of daily log returns, smoothed by 20-day rolling mean. The feature is documented as a lookahead-safe regime conditioning variable: when added to the panel, the LightGBM has access to a signal that distinguishes high-dispersion (idiosyncratic-pricing) regimes from low-dispersion (uniform-market-move) regimes without using any forward-looking regime labels. Implementation in `src/price_model/features/cross_features.py::CsReturnDispersion20`; lookahead safety verified by `tests/test_microstructure_features.py`.

The audit-driven 14-feature LightGBM panel is documented inline in `config/experiments/extended_kaggle_v3_curated.yaml`, including the LightGBM gain-importance and cross-feature correlation analysis that motivated the curation (audit script: `scripts/audit_lightgbm_features.py`).

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
notebooks/               # diagnostic + classical + robustness + portfolio
tests/                   # leakage tests, PIT tests, ticker tests, contract tests
```

The same data infrastructure also supports
`notebooks/04_portfolio_attribution.ipynb`, which uses the Ken French
adapter directly (not the model layer) to decompose a 10-stock
equal-weight portfolio's exposures and attribute realized returns to
factors. The notebook is an independent application of the data layer
and does not depend on the predictive model.

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
- **DeMiguel, V., Garlappi, L., and Uppal, R.** (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies* 22(5). — The estimation-error-dominates-optimization result referenced in Claim 2.
- **López de Prado, M.** (2018). *Advances in Financial Machine Learning.* Wiley. — Purged walk-forward cross-validation (Ch. 7).
- **Ken French Data Library.** Daily factor returns. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Wikipedia: List of S&P 500 companies.** Historical components and change log. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## License

MIT.
