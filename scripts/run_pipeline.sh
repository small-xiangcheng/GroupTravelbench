#!/bin/bash
# ==============================================================================
# Full Pipeline: Inference → Evaluation → Report (fully offline)
# Description: One-command pipeline. Config is snapshot at launch, so modifying
#              base_config.sh after start will NOT affect this running instance.
#              Logs are auto-redirected to Logs/Agent-{MODEL_NAME}.log
#
# Usage:
#   # After configuring base_config.sh:
#   source scripts/base_config.sh && source scripts/vllm_server.sh  # if local model
#   bash scripts/run_pipeline.sh           # foreground (logs tee to file + console)
#   bash scripts/run_pipeline.sh --bg      # background (auto nohup + log redirect)
#
# Options:
#   --bg                  Run in background (nohup, auto log redirect)
#   --skip-infer          Skip inference (only run eval on existing output)
#   --skip-llm-judge      Skip LLM-Judge evaluation
#   --no-resume           Fresh run (truncate log, re-run all tasks)
#
# Parallel Safety:
#   Each launch snapshots all config into local variables at startup.
#   You can safely modify base_config.sh and launch another pipeline in a
#   different terminal — they will NOT interfere with each other.
# ==============================================================================

set -euo pipefail

# --- Load base config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${MODEL_NAME:-}" ]; then
    source "${SCRIPT_DIR}/base_config.sh"
fi

cd "${PROJECT_ROOT}"

# ==============================================================================
# Handle --bg mode: re-launch self in background with auto log redirect
# The key insight: we must pass all critical env vars to the child process so
# it does NOT re-read base_config.sh (which may have been modified by then).
# We do this by exporting all config vars before nohup, so the child inherits
# them and skips the "source base_config.sh" step (because MODEL_NAME is set).
# ==============================================================================
for arg in "$@"; do
    if [ "${arg}" = "--bg" ]; then
        # Remove --bg from args and re-launch in background
        FILTERED_ARGS=()
        for a in "$@"; do
            [ "${a}" != "--bg" ] && FILTERED_ARGS+=("${a}")
        done
        mkdir -p "${LOG_DIR:-${PROJECT_ROOT}/Logs}"
        _BG_LOG="${LOG_DIR:-${PROJECT_ROOT}/Logs}/Agent-${MODEL_NAME}.log"
        echo "🚀 Launching pipeline in background..."
        echo "📝 Log: ${_BG_LOG}"
        echo "   tail -f ${_BG_LOG}  # to follow progress"

        # Export all config so child process inherits them (won't re-source base_config.sh)
        export MODEL_NAME MODEL_PATH MODEL_PORT MODEL_MAX_LEN MODEL_GPU_LIST
        export MODEL_TP_SIZE MODEL_DP_SIZE MODEL_INFERENCE_MODE
        export USE_CUSTOM_ENDPOINT ENABLE_THINKING USER_ENABLE_THINKING TOOL_ENABLE_THINKING USER_LLM TOOL_LLM
        export INPUT_FILE INFER_OUTPUT_DIR EVAL_OUTPUT_DIR AGENT_DIR_NAME
        export LOG_DIR VLLM_LOG_DIR SANDBOX_CACHE_DIR OPENAI_API_KEY OPENAI_API_BASE
        export LLM_JUDGE_MODEL LLM_JUDGE_CONCURRENCY LLM_JUDGE_ENABLE_THINKING
        export LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-}" LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-}"
        export INFER_MAX_CONCURRENCY
        export MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-}"
        export EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-}" EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-}"
        # The embedding key is optional: keep it unset (rather than empty) when
        # absent, so the built-in placeholder in the code keeps working.
        if [ -n "${EMBEDDING_API_KEY:-}" ]; then export EMBEDDING_API_KEY; fi
        export AGENT_API_BASE="${AGENT_API_BASE:-}" AGENT_API_KEY="${AGENT_API_KEY:-}"
        export USER_API_BASE="${USER_API_BASE:-}" USER_API_KEY="${USER_API_KEY:-}"
        export TOOL_API_BASE="${TOOL_API_BASE:-}" TOOL_API_KEY="${TOOL_API_KEY:-}"

        # Determine log redirect mode: append (resume) or truncate (no-resume)
        _NO_RESUME=false
        for _a in "$@"; do
            [ "${_a}" = "--no-resume" ] && _NO_RESUME=true
        done

        if [ "${_NO_RESUME}" = "true" ]; then
            # Truncate log for fresh run
            if [ ${#FILTERED_ARGS[@]} -eq 0 ]; then
                nohup bash "${BASH_SOURCE[0]}" > "${_BG_LOG}" 2>&1 &
            else
                nohup bash "${BASH_SOURCE[0]}" "${FILTERED_ARGS[@]}" > "${_BG_LOG}" 2>&1 &
            fi
        else
            # Append log for resume
            if [ ${#FILTERED_ARGS[@]} -eq 0 ]; then
                nohup bash "${BASH_SOURCE[0]}" >> "${_BG_LOG}" 2>&1 &
            else
                nohup bash "${BASH_SOURCE[0]}" "${FILTERED_ARGS[@]}" >> "${_BG_LOG}" 2>&1 &
            fi
        fi
        echo "   PID: $!"
        exit 0
    fi
done

# ==============================================================================
# Snapshot all config into local variables (parallel-safe)
# ==============================================================================
_MODEL_NAME="${MODEL_NAME}"
_MODEL_PATH="${MODEL_PATH:-}"
_MODEL_PORT="${MODEL_PORT:-10000}"
_MODEL_MAX_LEN="${MODEL_MAX_LEN:-131072}"
_MODEL_GPU_LIST="${MODEL_GPU_LIST:-}"
_MODEL_TP_SIZE="${MODEL_TP_SIZE:-2}"
_MODEL_DP_SIZE="${MODEL_DP_SIZE:-2}"
_MODEL_INFERENCE_MODE="${MODEL_INFERENCE_MODE:-think}"
_USE_CUSTOM_ENDPOINT="${USE_CUSTOM_ENDPOINT:-false}"
_ENABLE_THINKING="${ENABLE_THINKING:-true}"
_USER_ENABLE_THINKING="${USER_ENABLE_THINKING:-true}"
_TOOL_ENABLE_THINKING="${TOOL_ENABLE_THINKING:-false}"
_USER_LLM="${USER_LLM:-gpt-4o}"
_TOOL_LLM="${TOOL_LLM:-gpt-4o}"
_INPUT_FILE="${INPUT_FILE:-${PROJECT_ROOT}/datas/test.jsonl}"
_INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-${PROJECT_ROOT}/infer_output}"
_EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/eval_output}"
_LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/Logs}"
_SANDBOX_CACHE_DIR="${SANDBOX_CACHE_DIR:-${PROJECT_ROOT}/sandbox_cache}"
_OPENAI_API_KEY="${OPENAI_API_KEY}"
_OPENAI_API_BASE="${OPENAI_API_BASE}"
_LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-4o}"
_LLM_JUDGE_CONCURRENCY="${LLM_JUDGE_CONCURRENCY:-30}"
_LLM_JUDGE_ENABLE_THINKING="${LLM_JUDGE_ENABLE_THINKING:-true}"
_LLM_JUDGE_API_BASE="${LLM_JUDGE_API_BASE:-}"
_LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-}"
_MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-}"
_INFER_MAX_CONCURRENCY="${INFER_MAX_CONCURRENCY:-300}"

# Per-role API overrides (fall back to global OPENAI_API_BASE/KEY if empty)
_AGENT_API_BASE="${AGENT_API_BASE:-}"
_AGENT_API_KEY="${AGENT_API_KEY:-}"
_USER_API_BASE="${USER_API_BASE:-}"
_USER_API_KEY="${USER_API_KEY:-}"
_TOOL_API_BASE="${TOOL_API_BASE:-}"
_TOOL_API_KEY="${TOOL_API_KEY:-}"

# Derived paths
_AGENT_DIR_NAME="Agent-${_MODEL_NAME}"
_INFER_DIR="${_INFER_OUTPUT_DIR}/${_AGENT_DIR_NAME}"
_INFER_FILE="${_INFER_DIR}/test.jsonl"
_EVAL_DIR="${_EVAL_OUTPUT_DIR}/${_AGENT_DIR_NAME}"
_LOG_FILE="${_LOG_DIR}/${_AGENT_DIR_NAME}.log"

# --- Parse arguments ---
SKIP_INFER=false
SKIP_LLM_JUDGE=false
NO_RESUME=false

for arg in "$@"; do
    case "${arg}" in
        --skip-infer) SKIP_INFER=true ;;
        --skip-llm-judge) SKIP_LLM_JUDGE=true ;;
        --no-resume) NO_RESUME=true ;;
    esac
done

# --- Ensure directories exist ---
mkdir -p "${_INFER_DIR}" "${_EVAL_DIR}" "${_LOG_DIR}"

# ==============================================================================
# Print Config Summary
# ==============================================================================
echo "============================================================"
echo "🚀 GroupTravelbench Full Pipeline"
echo "============================================================"
echo "📋 Agent       : ${_AGENT_DIR_NAME}"
echo "📂 Input       : ${_INPUT_FILE}"
echo "📂 Infer Output: ${_INFER_FILE}"
echo "📂 Eval Output : ${_EVAL_DIR}/"
echo "📝 Log File    : ${_LOG_FILE}"
echo "🤖 Agent Model : ${_MODEL_NAME}"
echo "🧑 User Model  : ${_USER_LLM}"
echo "🛠️ Tool Model  : ${_TOOL_LLM}"
echo "🧠 Thinking    : ${_ENABLE_THINKING}"
echo "🌐 Endpoint    : ${_USE_CUSTOM_ENDPOINT} ${_MODEL_SERVICE_URL:-<cloud>}"
if [ "${SKIP_LLM_JUDGE}" = "true" ]; then
    echo "📊 LLM-Judge   : SKIPPED"
else
    echo "📊 LLM-Judge   : ${_LLM_JUDGE_MODEL}"
fi
echo "⏭️  Skip Infer  : ${SKIP_INFER}"
echo "============================================================"
echo ""

# Re-export for child processes (snapshot values)
export OPENAI_API_KEY="${_OPENAI_API_KEY}"
export OPENAI_API_BASE="${_OPENAI_API_BASE}"

# ==============================================================================
# Phase 1: Inference
# ==============================================================================
if [ "${SKIP_INFER}" = "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏭️  [Phase 1] Inference SKIPPED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ ! -f "${_INFER_FILE}" ]; then
        echo "❌ Inference output not found: ${_INFER_FILE}"
        echo "   Cannot skip inference without existing output."
        exit 1
    fi
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔮 [Phase 1] Running Inference"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Build Agent LLM args
    if [ "${_USE_CUSTOM_ENDPOINT}" = "true" ]; then
        if [ -z "${_MODEL_SERVICE_URL}" ]; then
            echo "❌ USE_CUSTOM_ENDPOINT=true but MODEL_SERVICE_URL is not set."
            echo "   Please run: source scripts/vllm_server.sh"
            exit 1
        fi
        AGENT_LLM_ARGS="{\"max_tokens\": 12288, \"temperature\": 0.7, \"enable_thinking\": ${_ENABLE_THINKING}, \"api_base\": \"${_MODEL_SERVICE_URL}\"}"
    else
        AGENT_LLM_ARGS="{\"max_tokens\": 12288, \"temperature\": 0.7, \"enable_thinking\": ${_ENABLE_THINKING}}"
    fi

    # Inject per-role api_base / api_key overrides into LLM args JSON
    # Agent: AGENT_API_BASE / AGENT_API_KEY override (when USE_CUSTOM_ENDPOINT=false)
    if [ "${_USE_CUSTOM_ENDPOINT}" != "true" ] && [ -n "${_AGENT_API_BASE}" ]; then
        AGENT_LLM_ARGS=$(echo "${AGENT_LLM_ARGS}" | sed 's/}$//')
        AGENT_LLM_ARGS="${AGENT_LLM_ARGS}, \"api_base\": \"${_AGENT_API_BASE}\"}"
    fi
    if [ -n "${_AGENT_API_KEY}" ]; then
        AGENT_LLM_ARGS=$(echo "${AGENT_LLM_ARGS}" | sed 's/}$//')
        AGENT_LLM_ARGS="${AGENT_LLM_ARGS}, \"api_key\": \"${_AGENT_API_KEY}\"}"
    fi

    # User-simulator args (thinking ON by default: replies should be reasoned).
    # max_tokens covers thinking + reply: 16384 keeps a thinking turn from being
    # truncated (a truncated reply is recorded as a broken user message).
    USER_LLM_ARGS="{\"max_tokens\": 16384, \"temperature\": 0.0, \"enable_thinking\": ${_USER_ENABLE_THINKING}"
    if [ -n "${_USER_API_BASE}" ]; then
        USER_LLM_ARGS="${USER_LLM_ARGS}, \"api_base\": \"${_USER_API_BASE}\""
    fi
    if [ -n "${_USER_API_KEY}" ]; then
        USER_LLM_ARGS="${USER_LLM_ARGS}, \"api_key\": \"${_USER_API_KEY}\""
    fi
    USER_LLM_ARGS="${USER_LLM_ARGS}}"

    # Tool-simulator args (thinking OFF by default: it only fabricates tool payloads)
    TOOL_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.0, \"enable_thinking\": ${_TOOL_ENABLE_THINKING}"
    if [ -n "${_TOOL_API_BASE}" ]; then
        TOOL_LLM_ARGS="${TOOL_LLM_ARGS}, \"api_base\": \"${_TOOL_API_BASE}\""
    fi
    if [ -n "${_TOOL_API_KEY}" ]; then
        TOOL_LLM_ARGS="${TOOL_LLM_ARGS}, \"api_key\": \"${_TOOL_API_KEY}\""
    fi
    TOOL_LLM_ARGS="${TOOL_LLM_ARGS}}"

    if [ ! -f "${_INPUT_FILE}" ]; then
        echo "❌ Input file not found: ${_INPUT_FILE}"
        exit 1
    fi

    if [ ! -d "${_SANDBOX_CACHE_DIR}" ]; then
        echo "❌ Sandbox cache not found: ${_SANDBOX_CACHE_DIR}"
        echo "   Please restore the data first: bash scripts/restore_data.sh"
        exit 1
    fi

    echo "  Starting inference on $(wc -l < "${_INPUT_FILE}") tasks..."
    echo ""

    INFER_EXTRA_ARGS=""
    if [ "${NO_RESUME}" = "true" ]; then
        INFER_EXTRA_ARGS="--no-resume"
    fi

    python -u -m group_travelbench run \
        --file "${_INPUT_FILE}" \
        --output "${_INFER_FILE}" \
        --agent-llm "${_MODEL_NAME}" \
        --agent-llm-args "${AGENT_LLM_ARGS}" \
        --user-llm "${_USER_LLM}" \
        --user-llm-args "${USER_LLM_ARGS}" \
        --tool-llm "${_TOOL_LLM}" \
        --tool-llm-args "${TOOL_LLM_ARGS}" \
        --max-turn 15 \
        --convergence-interval 3 \
        --max-concurrency "${_INFER_MAX_CONCURRENCY}" \
        --sandbox-mode isolated \
        --cache-dir "${_SANDBOX_CACHE_DIR}" \
        --num-trials 3 \
        ${INFER_EXTRA_ARGS}

    echo ""
    echo "  ✅ Inference completed."
    echo ""
fi

# ==============================================================================
# Phase 2: Evaluation + Report
# ==============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 [Phase 2] Running Evaluation Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Override OPENAI_API_KEY/BASE for LLM-Judge if per-role keys are configured
if [ -n "${_LLM_JUDGE_API_KEY}" ]; then
    export OPENAI_API_KEY="${_LLM_JUDGE_API_KEY}"
fi
if [ -n "${_LLM_JUDGE_API_BASE}" ]; then
    export OPENAI_API_BASE="${_LLM_JUDGE_API_BASE}"
fi

EVAL_CMD="python -u -m group_travelbench.evaluation.run_full_eval \
    --input \"${_INFER_FILE}\" \
    --output-dir \"${_EVAL_DIR}\" \
    --test-file \"${_INPUT_FILE}\" \
    --agent-name \"${_AGENT_DIR_NAME}\" \
    --llm-judge-model \"${_LLM_JUDGE_MODEL}\" \
    --llm-judge-concurrency ${_LLM_JUDGE_CONCURRENCY}"

if [ "${SKIP_LLM_JUDGE}" = "true" ]; then
    EVAL_CMD="${EVAL_CMD} --skip-llm-judge"
fi

# The judge is expected to be a strong thinking model (this work: Gemini3-Flash-Preview),
# so thinking stays ON unless LLM_JUDGE_ENABLE_THINKING is explicitly set to false.
if [ "${_LLM_JUDGE_ENABLE_THINKING}" = "false" ]; then
    EVAL_CMD="${EVAL_CMD} --disable-thinking"
fi

eval ${EVAL_CMD}

# ==============================================================================
# Done
# ==============================================================================
echo ""
echo "============================================================"
echo "🎉 Full Pipeline Completed: ${_AGENT_DIR_NAME}"
echo "============================================================"
echo "📊 Report    : ${_EVAL_DIR}/report.md"
echo "📄 Multi-eval: ${_EVAL_DIR}/multi_eval_result.json"
if [ "${SKIP_LLM_JUDGE}" = "false" ]; then
    echo "📄 LLM-Judge : ${_EVAL_DIR}/llm_judge_result.json"
fi
echo "📝 Full log  : ${_LOG_FILE}"
echo "============================================================"
