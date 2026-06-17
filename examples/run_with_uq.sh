#!/usr/bin/env bash
# Run tau2-bench simulation with uncertainty quantification (UQ) tracking enabled.
#
# This script runs the benchmark with logprobs capture, then extracts and
# analyzes the token-level UQ data.
#
# Prerequisites:
#   - pip install -e .
#   - Set API keys in .env or environment variables (see .env.example)
#
# Usage:
#   bash examples/run_with_uq.sh

set -euo pipefail

# --- Configuration ---
DOMAIN="retail"                        # airline | retail | telecom
AGENT_LLM="gpt-4.1"                   # Agent LLM model name
USER_LLM="gpt-4.1"                    # User simulator LLM
NUM_TRIALS=1                           # Number of trials per task
OUTPUT_DIR="./results"                 # Directory for result JSON
UQ_LOGPROBS_DIR="./uq_logprobs"       # Directory for token-level sidecar files
UQ_ANALYSIS_DIR="./uq_analysis"       # Directory for aggregated UQ summaries

RESULTS_FILE="${OUTPUT_DIR}/${AGENT_LLM}_${DOMAIN}.json"

# --- Step 1: Run simulation with UQ tracking ---
# The key LLM arguments are logprobs=true and top_logprobs=20, which enable
# token-level logprob capture during generation.
echo "=== Step 1: Running simulation with UQ tracking ==="
tau2 run \
    --domain "${DOMAIN}" \
    --agent-llm "${AGENT_LLM}" \
    --user-llm "${USER_LLM}" \
    --num-trials "${NUM_TRIALS}" \
    --output-dir "${OUTPUT_DIR}" \
    --agent-llm-args '{"logprobs": true, "top_logprobs": 20}' \
    --user-llm-args '{"logprobs": true, "top_logprobs": 20}'

echo "Results saved to: ${RESULTS_FILE}"

# --- Step 2: Extract UQ data into sidecar JSONL files ---
# This extracts embedded uq_logprobs from the result JSON into separate
# per-task JSONL files for efficient analysis.
echo ""
echo "=== Step 2: Extracting UQ logprobs to sidecar files ==="
tau2 extract-uq-from-trajs \
    --results "${RESULTS_FILE}" \
    --output-dir "${UQ_LOGPROBS_DIR}"

echo "Sidecar files written to: ${UQ_LOGPROBS_DIR}"

# --- Step 3: Analyze and aggregate UQ data ---
# Aggregates token-level data into per-turn and per-trajectory summaries.
echo ""
echo "=== Step 3: Analyzing UQ logprobs ==="
tau2 analyze-uq-logprobs \
    --input "${UQ_LOGPROBS_DIR}" \
    --output-dir "${UQ_ANALYSIS_DIR}"

echo "Analysis complete. Outputs:"
echo "  - Turn-level:       ${UQ_ANALYSIS_DIR}/uq_turn_summary.csv"
echo "  - Trajectory-level: ${UQ_ANALYSIS_DIR}/uq_trajectory_summary.csv"
