#!/bin/bash
# ==============================================================================
# Inference Script
# Description: Run multi-user travel planning inference against the offline sandbox
# Usage:
#   source scripts/base_config.sh && bash scripts/infer.sh
#   nohup bash scripts/infer.sh > Logs/infer.log 2>&1 &
#
# Any extra arguments are forwarded to `python -m group_travelbench run`,
# e.g. `bash scripts/infer.sh --no-resume`.
# ==============================================================================

set -euo pipefail

# --- Load base config if not already loaded ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${MODEL_NAME:-}" ]; then
    source "${SCRIPT_DIR}/base_config.sh"
fi

cd "${PROJECT_ROOT}"

# --- Derive paths ---
AGENT_LLM="${MODEL_NAME}"
AGENT_DIR_NAME="Agent-${AGENT_LLM}"
OUTPUT_DIR="${INFER_OUTPUT_DIR}/${AGENT_DIR_NAME}"
OUTPUT_FILE="${OUTPUT_DIR}/test.jsonl"
mkdir -p "${OUTPUT_DIR}"

# --- Build Agent LLM args ---
# Keep these defaults in sync with scripts/run_pipeline.sh.
if [ "${USE_CUSTOM_ENDPOINT}" = "true" ]; then
    if [ -z "${MODEL_SERVICE_URL:-}" ]; then
        echo "❌ USE_CUSTOM_ENDPOINT=true but MODEL_SERVICE_URL is not set."
        echo "   Please run: source scripts/vllm_server.sh"
        exit 1
    fi
    AGENT_LLM_ARGS="{\"max_tokens\": 12288, \"temperature\": 0.7, \"enable_thinking\": ${ENABLE_THINKING}, \"api_base\": \"${MODEL_SERVICE_URL}\"}"
else
    AGENT_LLM_ARGS="{\"max_tokens\": 12288, \"temperature\": 0.7, \"enable_thinking\": ${ENABLE_THINKING}}"
fi

# --- Inject per-role api_base / api_key overrides ---
# Agent overrides only apply when not pointing at a local vLLM endpoint.
if [ "${USE_CUSTOM_ENDPOINT}" != "true" ] && [ -n "${AGENT_API_BASE:-}" ]; then
    AGENT_LLM_ARGS=$(echo "${AGENT_LLM_ARGS}" | sed 's/}$//')
    AGENT_LLM_ARGS="${AGENT_LLM_ARGS}, \"api_base\": \"${AGENT_API_BASE}\"}"
fi
if [ -n "${AGENT_API_KEY:-}" ]; then
    AGENT_LLM_ARGS=$(echo "${AGENT_LLM_ARGS}" | sed 's/}$//')
    AGENT_LLM_ARGS="${AGENT_LLM_ARGS}, \"api_key\": \"${AGENT_API_KEY}\"}"
fi

# max_tokens covers thinking + reply: 16384 keeps a thinking user simulator
# from being truncated mid-reply (a truncated turn is fed back as a broken message).
USER_LLM_ARGS="{\"max_tokens\": 16384, \"temperature\": 0.0, \"enable_thinking\": ${USER_ENABLE_THINKING:-true}"
if [ -n "${USER_API_BASE:-}" ]; then
    USER_LLM_ARGS="${USER_LLM_ARGS}, \"api_base\": \"${USER_API_BASE}\""
fi
if [ -n "${USER_API_KEY:-}" ]; then
    USER_LLM_ARGS="${USER_LLM_ARGS}, \"api_key\": \"${USER_API_KEY}\""
fi
USER_LLM_ARGS="${USER_LLM_ARGS}}"

TOOL_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.0, \"enable_thinking\": ${TOOL_ENABLE_THINKING:-false}"
if [ -n "${TOOL_API_BASE:-}" ]; then
    TOOL_LLM_ARGS="${TOOL_LLM_ARGS}, \"api_base\": \"${TOOL_API_BASE}\""
fi
if [ -n "${TOOL_API_KEY:-}" ]; then
    TOOL_LLM_ARGS="${TOOL_LLM_ARGS}, \"api_key\": \"${TOOL_API_KEY}\""
fi
TOOL_LLM_ARGS="${TOOL_LLM_ARGS}}"

# --- Validate input ---
if [ ! -f "${INPUT_FILE}" ]; then
    echo "❌ Input file not found: ${INPUT_FILE}"
    exit 1
fi

if [ ! -d "${SANDBOX_CACHE_DIR}" ]; then
    echo "❌ Sandbox cache not found: ${SANDBOX_CACHE_DIR}"
    echo "   Please restore the data first: bash scripts/restore_data.sh"
    exit 1
fi

# --- Print config ---
echo "============================================================"
echo "🚀 GroupTravelbench Inference"
echo "============================================================"
echo "📂 Input  : ${INPUT_FILE} ($(wc -l < "${INPUT_FILE}") tasks)"
echo "📂 Output : ${OUTPUT_FILE}"
echo "🤖 Agent  : ${AGENT_LLM}"
echo "🧑 User   : ${USER_LLM}"
echo "🛠️ Tool   : ${TOOL_LLM}"
echo "🧠 Thinking: ${ENABLE_THINKING}"
echo "🌐 Endpoint: ${USE_CUSTOM_ENDPOINT} ${MODEL_SERVICE_URL:-<cloud>}"
echo "============================================================"
echo

# --- Run inference ---
python -u -m group_travelbench run \
    --file "${INPUT_FILE}" \
    --output "${OUTPUT_FILE}" \
    --agent-llm "${AGENT_LLM}" \
    --agent-llm-args "${AGENT_LLM_ARGS}" \
    --user-llm "${USER_LLM}" \
    --user-llm-args "${USER_LLM_ARGS}" \
    --tool-llm "${TOOL_LLM}" \
    --tool-llm-args "${TOOL_LLM_ARGS}" \
    --max-turn 15 \
    --convergence-interval 3 \
    --max-concurrency "${INFER_MAX_CONCURRENCY:-300}" \
    --sandbox-mode isolated \
    --cache-dir "${SANDBOX_CACHE_DIR}" \
    --num-trials 3 \
    "$@"

status=$?
echo
echo "============================================================"
if [ ${status} -eq 0 ]; then
    echo "✅ Inference completed successfully."
else
    echo "⚠️  Inference failed (exit code ${status})."
fi
echo "📂 Output: ${OUTPUT_FILE}"
echo "============================================================"

exit ${status}
