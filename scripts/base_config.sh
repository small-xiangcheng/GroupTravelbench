#!/bin/bash
# ==============================================================================
# Base Configuration File
# Description: Central configuration for all GroupTravelbench scripts
# Usage: source scripts/base_config.sh
#
# Every value below is a placeholder — replace the ones your setup needs.
# ==============================================================================

# ------------------------------------------------------------------------------
# Agent Model Configuration
#
# MODEL_NAME is the model under evaluation. It also determines the output
# directory name (infer_output/Agent-${MODEL_NAME}), so keep it filesystem-safe.
# MODEL_PATH / MODEL_*_SIZE only matter when USE_CUSTOM_ENDPOINT=true.
# ------------------------------------------------------------------------------
export MODEL_PATH="/path/to/your/model"     # Local weights, only for vLLM deployment
export MODEL_NAME="your-model-name"
export MODEL_PORT=10001
export MODEL_MAX_LEN=131072
export MODEL_GPU_LIST="0,1,2,3"             # GPUs to use, e.g. "0,1,2,3"
export MODEL_TP_SIZE=1                      # Tensor Parallel size
export MODEL_DP_SIZE=4                      # Data Parallel size (DP * TP must equal GPU count)
export MODEL_INFERENCE_MODE="think"         # think or no-think

# ------------------------------------------------------------------------------
# Embedding Model Configuration
#
# Only needed for similarity-based example retrieval in the tool simulator.
# Without it the simulator falls back to random example retrieval.
# ------------------------------------------------------------------------------
export EMBEDDING_MODEL_PATH="/path/to/your/embedding-model"
export EMBEDDING_MODEL_NAME="your-embedding-model-name"
export EMBEDDING_PORT=40001
export EMBEDDING_MAX_LEN=32768
export EMBEDDING_GPU_LIST="4,5,6,7"         # GPUs to use, e.g. "4,5,6,7"
export EMBEDDING_TP_SIZE=2                  # Tensor Parallel size
export EMBEDDING_DP_SIZE=2                  # Data Parallel size (DP * TP must equal GPU count)
# Set after vllm_embedding_server.sh starts, or point at an existing service:
export EMBEDDING_SERVICE_URL="http://localhost:${EMBEDDING_PORT}/v1"
# Optional: API key for the embedding service. Only needed when the service sits
# behind an authenticated gateway (e.g. an internal proxy) — a local vLLM does
# not verify the key, and leaving it unset keeps the built-in placeholder.
# export EMBEDDING_API_KEY="your-embedding-api-key"

# ------------------------------------------------------------------------------
# Inference & Evaluation Configuration
# ------------------------------------------------------------------------------
export USE_CUSTOM_ENDPOINT=false            # true: self-deployed vLLM; false: cloud API
export ENABLE_THINKING=true                 # Agent (model under test) thinking
export USER_ENABLE_THINKING=true            # User simulator thinking; intended ON so replies are reasoned
export TOOL_ENABLE_THINKING=false           # Tool simulator thinking; intended OFF (it only fabricates tool payloads)
# Role-specific model requirements:
#   - User simulator: MUST be a strong THINKING model. It has to play the users
#     well enough to drive the conversation; a weak model degrades simulation
#     quality badly (this work uses DeepSeek-V4-Flash, thinking ON).
#   - Tool simulator: easy job — it only fabricates plausible tool payloads from
#     cached examples, so an instruction-following model without thinking is
#     enough (this work uses GPT4.1-0401, thinking OFF).
export USER_LLM="your-user-simulator-model" # User simulator (strong thinking model)
export TOOL_LLM="your-tool-simulator-model" # Tool simulator (cache-miss fallback; instruction-following is enough)
export INFER_MAX_CONCURRENCY=100            # Inference max concurrency

# ------------------------------------------------------------------------------
# Path Configuration (auto-derived, override if needed)
# ------------------------------------------------------------------------------
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INPUT_FILE="${PROJECT_ROOT}/datas/test.jsonl"
export INFER_OUTPUT_DIR="${PROJECT_ROOT}/infer_output"
export EVAL_OUTPUT_DIR="${PROJECT_ROOT}/eval_output"
export AGENT_DIR_NAME="Agent-${MODEL_NAME}"
export LOG_DIR="${PROJECT_ROOT}/Logs"
export VLLM_LOG_DIR="${PROJECT_ROOT}/logs"
export SANDBOX_CACHE_DIR="${PROJECT_ROOT}/sandbox_cache"

# ------------------------------------------------------------------------------
# API Configuration (Global Defaults)
#
# Any OpenAI-compatible endpoint works. These are the fallback values for every
# role (agent / user simulator / tool simulator / LLM-Judge).
# ------------------------------------------------------------------------------
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://your-api-endpoint/v1"

# ------------------------------------------------------------------------------
# Per-Role API Overrides (optional)
#
# Use these when a role needs a different endpoint or key than the global
# default above. Leave empty to inherit OPENAI_API_BASE / OPENAI_API_KEY.
# ------------------------------------------------------------------------------
export AGENT_API_BASE=""                    # Agent endpoint (overrides OPENAI_API_BASE)
export AGENT_API_KEY=""                     # Agent key (overrides OPENAI_API_KEY)
export USER_API_BASE=""                     # User-simulator endpoint
export USER_API_KEY=""                      # User-simulator key
export TOOL_API_BASE=""                     # Tool-simulator endpoint
export TOOL_API_KEY=""                      # Tool-simulator key

# ------------------------------------------------------------------------------
# Eval Configuration
# ------------------------------------------------------------------------------
# LLM-Judge: MUST be a strong THINKING model too — it reads whole multi-user
# conversations and has to reason about them (this work uses Gemini3-Flash-Preview).
export LLM_JUDGE_MODEL="your-judge-model"
export LLM_JUDGE_ENABLE_THINKING=true        # Keep ON when the judge is a thinking model
export LLM_JUDGE_CONCURRENCY=50
export LLM_JUDGE_API_BASE=""                # LLM-Judge endpoint (overrides OPENAI_API_BASE)
export LLM_JUDGE_API_KEY=""                 # LLM-Judge key (overrides OPENAI_API_KEY)
