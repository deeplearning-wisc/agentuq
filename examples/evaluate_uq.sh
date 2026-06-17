#!/usr/bin/env bash
# Evaluate how well UQ estimates predict task failure.
#
# This script computes AUROC, AUARC, Pearson/Spearman/Kendall correlations
# between uncertainty metrics and task outcomes (1 - reward).
#
# Two evaluation modes are supported:
#   1. "embedded" — reads uq_summary directly from result JSON (no sidecar needed)
#   2. "csv"      — reads from a uq_trajectory_summary.csv (from analyze-uq-logprobs)
#
# Prerequisites:
#   - A completed simulation run with UQ tracking (see run_with_uq.sh)
#   - pip install -e .
#
# Usage:
#   bash examples/evaluate_uq.sh

set -euo pipefail

# --- Configuration ---
RESULTS_FILE="./results/gpt-4.1_retail.json"
UQ_ANALYSIS_DIR="./uq_analysis"
UQ_EVAL_DIR="./uq_eval"

# --- Option A: Evaluate from embedded uq_summary ---
echo "=== Evaluating UQ (embedded mode) ==="
tau2 evaluate-uq \
    --mode embedded \
    --results "${RESULTS_FILE}" \
    --output-dir "${UQ_EVAL_DIR}/embedded"

echo "Results: ${UQ_EVAL_DIR}/embedded/uq_eval_metrics.json"

# --- Option B: Evaluate from CSV (supports scorer filtering) ---
echo ""
echo "=== Evaluating UQ (CSV mode) ==="
tau2 evaluate-uq \
    --mode csv \
    --uq-trajectory-csv "${UQ_ANALYSIS_DIR}/uq_trajectory_summary.csv" \
    --results "${RESULTS_FILE}" \
    --output-dir "${UQ_EVAL_DIR}/csv"

echo "Results: ${UQ_EVAL_DIR}/csv/uq_eval_metrics.json"

# --- Option C: Evaluate observation UQ with scorer filter ---
# After running score_observation_uq.sh and analyze-uq-logprobs,
# you can filter by scorer to evaluate only observation uncertainty:
#
# tau2 evaluate-uq \
#     --mode csv \
#     --uq-trajectory-csv "${UQ_ANALYSIS_DIR}/uq_trajectory_summary.csv" \
#     --results "${RESULTS_FILE}" \
#     --output-dir "${UQ_EVAL_DIR}/obs_uq" \
#     --scorer agent_llm
