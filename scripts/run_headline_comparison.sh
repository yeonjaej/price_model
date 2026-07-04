#!/usr/bin/env bash
# Headline tuned-tree comparison runner.
# Usage:  bash scripts/run_headline_comparison.sh [TRIALS]
#   TRIALS defaults to 40. Each tree is swept on BOTH its engineered (14-feat
#   zscore) and rank (9-feat) panel; the headline uses whichever wins.
#
# IMPORTANT: free the DuckDB write-lock first (close/restart any Jupyter kernel
# that opened the prediction store), or the `cli run` steps will fail with
# "Could not set lock on predictions.duckdb". The Optuna sweeps (step 1) do NOT
# touch the store, so they run regardless; the lock only matters from step 2 on.

set -u
TRIALS="${1:-40}"
cd /Users/jwayeonjae/Documents/Claude/Projects/price_model
export PYTHONPATH=src
PY=.venv/bin/python

SWEEPS=(
  extended_kaggle_v3_curated_h21
  lightgbm_rank9_h21
  extended_kaggle_v3_xgboost_h21
  xgboost_rank9_h21
  extended_kaggle_v3_catboost_h21
  catboost_rank9_h21
)

echo "########## STEP 1: Optuna sweeps (trials=${TRIALS}) ##########"
for EXP in "${SWEEPS[@]}"; do
  echo "===== sweeping ${EXP} ====="
  "$PY" scripts/optuna_sweep.py --experiment "$EXP" \
    --n-trials "$TRIALS" --n-folds 4 --embargo-days 22 \
    --max-date 2024-12-31 --write-tuned
done

echo "########## STEP 2: train tuned trees + linear + momentum ##########"
RUNS=(
  extended_kaggle_v3_curated_h21_hp_pre20241231
  lightgbm_rank9_h21_hp_pre20241231
  extended_kaggle_v3_xgboost_h21_hp_pre20241231
  xgboost_rank9_h21_hp_pre20241231
  extended_kaggle_v3_catboost_h21_hp_pre20241231
  catboost_rank9_h21_hp_pre20241231
  lasso_curated6_pit_h21
  momentum_factor_pit_h21
)
for EXP in "${RUNS[@]}"; do
  echo "===== running ${EXP} ====="
  "$PY" -m price_model.cli run -e "$EXP"
done

echo "########## STEP 3: comparison tables (2025+ OOS, 21-day) ##########"
echo "===== GROSS (apples-to-apples) ====="
"$PY" scripts/compare_apples_to_apples.py --since 2025-01-01 --horizon 21
echo "===== NET OF COST (3/10/20 bp) ====="
"$PY" scripts/compare_net_of_cost.py --since 2025-01-01 --horizon 21
echo "########## DONE ##########"
