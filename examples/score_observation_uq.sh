#!/usr/bin/env bash
# Score observation uncertainty using an auxiliary LLM.
#
# This estimates how "surprising" observations (user messages and tool results)
# are from the perspective of a scoring model. Two modes are available:
#
#   agent_llm     — Uses the agent's own LLM with its domain system prompt.
#                   Measures what the agent "expected" to observe.
#   auxiliary_llm — Uses a separate observer model with a generic world-model
#                   prompt. Provides an external perspective on observation surprise.
#
# The scorer replays the conversation history up to each observation, then
# measures the logprob of the actual observation tokens. Low-probability
# observations indicate high uncertainty / surprise.
#
# Prerequisites:
#   - A completed simulation run (see run_with_uq.sh)
#   - An OpenAI-compatible API endpoint that supports logprobs
#   - pip install -e .
#
# Usage:
#   bash examples/score_observation_uq.sh

set -euo pipefail

# --- Configuration ---
RESULTS_FILE="./results/gpt-4.1_retail.json"
OBS_UQ_DIR="./uq_obs"
UQ_ANALYSIS_DIR="./uq_analysis_obs"

# API endpoint for the scoring model (must support logprobs)
SCORER_API_BASE="http://127.0.0.1:8000/v1"

# --- Option A: Agent LLM mode ---
# Score observations using the same model that was used as the agent.
# The scorer sees the agent's domain-policy system prompt.
echo "=== Scoring observation UQ (agent_llm mode) ==="
tau2 score-observation-uq \
    --results "${RESULTS_FILE}" \
    --output-dir "${OBS_UQ_DIR}/agent_llm" \
    --scorer-mode agent_llm \
    --scorer-api-base "${SCORER_API_BASE}" \
    --rescore-backend chat_replay

echo "Sidecar files: ${OBS_UQ_DIR}/agent_llm/"

# --- Option B: Auxiliary LLM mode ---
# Score observations using a separate model with a generic observer prompt.
echo ""
echo "=== Scoring observation UQ (auxiliary_llm mode) ==="
tau2 score-observation-uq \
    --results "${RESULTS_FILE}" \
    --output-dir "${OBS_UQ_DIR}/auxiliary_llm" \
    --scorer-mode auxiliary_llm \
    --scorer-llm "openai/gpt-4o-mini" \
    --scorer-api-base "${SCORER_API_BASE}" \
    --rescore-backend chat_replay

echo "Sidecar files: ${OBS_UQ_DIR}/auxiliary_llm/"

# --- Step 2: Aggregate into summaries ---
echo ""
echo "=== Aggregating observation UQ ==="
tau2 analyze-uq-logprobs \
    --input "${OBS_UQ_DIR}/agent_llm" \
    --output-dir "${UQ_ANALYSIS_DIR}"

echo "Analysis: ${UQ_ANALYSIS_DIR}/uq_trajectory_summary.csv"
