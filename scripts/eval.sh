#!/bin/bash
# ==============================================================================
# Evaluation Script
# Description: Full evaluation pipeline (multi_eval + LLM-Judge + report)
# Usage:
#   source scripts/base_config.sh && bash scripts/eval.sh
#   bash scripts/eval.sh --skip-llm-judge        # Skip LLM-Judge (rule-based only)
#   bash scripts/eval.sh --input path/to/file    # Specify custom input path
# ==============================================================================

set -euo pipefail

# --- Load base config if not already loaded ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${MODEL_NAME:-}" ]; then
    source "${SCRIPT_DIR}/base_config.sh"
fi

cd "${PROJECT_ROOT}"

# --- Parse arguments ---
CUSTOM_INPUT=""
SKIP_LLM_JUDGE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-llm-judge) SKIP_LLM_JUDGE=true; shift ;;
        --input=*) CUSTOM_INPUT="${1#*=}"; shift ;;
        --input) CUSTOM_INPUT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# --- Derive paths ---
AGENT_DIR_NAME="Agent-${MODEL_NAME}"

if [ -n "${CUSTOM_INPUT}" ]; then
    INFER_FILE="${CUSTOM_INPUT}"
else
    INFER_FILE="${INFER_OUTPUT_DIR}/${AGENT_DIR_NAME}/test.jsonl"
fi

EVAL_DIR="${EVAL_OUTPUT_DIR}/${AGENT_DIR_NAME}"
mkdir -p "${EVAL_DIR}"

# --- Validate ---
if [ ! -f "${INFER_FILE}" ]; then
    echo "❌ Inference output not found: ${INFER_FILE}"
    echo "   Please run inference first: bash scripts/infer.sh"
    exit 1
fi

# --- Print config ---
echo "============================================================"
echo "🔬 GroupTravelbench Evaluation Pipeline"
echo "============================================================"
echo "📂 Input     : ${INFER_FILE}"
echo "📂 Output Dir: ${EVAL_DIR}"
echo "📂 Test File : ${INPUT_FILE}"
echo "🤖 Agent     : ${AGENT_DIR_NAME}"
if [ "${SKIP_LLM_JUDGE}" = "true" ]; then
    echo "🧠 LLM-Judge : SKIPPED"
else
    echo "🧠 LLM-Judge : ${LLM_JUDGE_MODEL} (thinking: ${LLM_JUDGE_ENABLE_THINKING:-true})"
fi
echo "============================================================"
echo

# --- Run full evaluation pipeline ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running evaluation pipeline..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Override OPENAI_API_KEY/BASE for LLM-Judge if per-role keys are configured
if [ -n "${LLM_JUDGE_API_KEY:-}" ]; then
    export OPENAI_API_KEY="${LLM_JUDGE_API_KEY}"
fi
if [ -n "${LLM_JUDGE_API_BASE:-}" ]; then
    export OPENAI_API_BASE="${LLM_JUDGE_API_BASE}"
fi

EVAL_CMD="python -u -m group_travelbench.evaluation.run_full_eval \
    --input \"${INFER_FILE}\" \
    --output-dir \"${EVAL_DIR}\" \
    --test-file \"${INPUT_FILE}\" \
    --agent-name \"${AGENT_DIR_NAME}\" \
    --llm-judge-model \"${LLM_JUDGE_MODEL}\" \
    --llm-judge-concurrency ${LLM_JUDGE_CONCURRENCY}"

if [ "${SKIP_LLM_JUDGE}" = "true" ]; then
    EVAL_CMD="${EVAL_CMD} --skip-llm-judge"
fi

# The judge is expected to be a strong thinking model (this work: Gemini3-Flash-Preview),
# so thinking stays ON unless LLM_JUDGE_ENABLE_THINKING is explicitly set to false.
if [ "${LLM_JUDGE_ENABLE_THINKING:-true}" = "false" ]; then
    EVAL_CMD="${EVAL_CMD} --disable-thinking"
fi

eval ${EVAL_CMD}

status=$?
echo
if [ ${status} -eq 0 ]; then
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo "📊 Report: ${EVAL_DIR}/report.md"
    echo "📄 Multi-eval: ${EVAL_DIR}/multi_eval_result.json"
    if [ "${SKIP_LLM_JUDGE}" = "false" ]; then
        echo "📄 LLM-Judge: ${EVAL_DIR}/llm_judge_result.json"
    fi
    echo "============================================================"
else
    echo "⚠️  Evaluation failed (exit code ${status})."
fi

exit ${status}