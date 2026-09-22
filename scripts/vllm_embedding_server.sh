#!/bin/bash
# ==============================================================================
# vLLM Embedding Model Deployment Script
# Description: Deploy the embedding model using vLLM server
# Usage: source scripts/vllm_embedding_server.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
GPU_COUNT=$(echo "$EMBEDDING_GPU_LIST" | awk -F',' '{print NF}')
mkdir -p "$VLLM_LOG_DIR"

# Validate DP * TP = GPU count
EXPECTED_GPU_COUNT=$((EMBEDDING_DP_SIZE * EMBEDDING_TP_SIZE))
if [ "$EXPECTED_GPU_COUNT" -ne "$GPU_COUNT" ]; then
    echo "❌ Error: EMBEDDING_DP_SIZE($EMBEDDING_DP_SIZE) * EMBEDDING_TP_SIZE($EMBEDDING_TP_SIZE) = $EXPECTED_GPU_COUNT, but GPU_COUNT = $GPU_COUNT"
    echo "   Please ensure DP * TP equals the number of GPUs in EMBEDDING_GPU_LIST"
    return 1 2>/dev/null || exit 1
fi

net0_ip=$(ifconfig net0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
[ -z "$net0_ip" ] && net0_ip=$(hostname -I | awk '{print $1}')
log_file="${VLLM_LOG_DIR}/embedding_$(echo $net0_ip | tr '.' '_').log"

echo "🚀 Deploying embedding model: ${EMBEDDING_MODEL_NAME}"
echo "   GPU(s): ${EMBEDDING_GPU_LIST} (${GPU_COUNT} cards)"
echo "   TP: ${EMBEDDING_TP_SIZE}, DP: ${EMBEDDING_DP_SIZE}"
echo "   Port: ${EMBEDDING_PORT}"

# ------------------------------------------------------------------------------
# Start Server
# ------------------------------------------------------------------------------
nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${EMBEDDING_GPU_LIST}
    vllm serve ${EMBEDDING_MODEL_PATH} \
        --served-model-name ${EMBEDDING_MODEL_NAME} \
        --runner pooling \
        --max-model-len ${EMBEDDING_MAX_LEN} \
        --enable-prefix-caching \
        --tensor-parallel-size ${EMBEDDING_TP_SIZE} \
        --data-parallel-size ${EMBEDDING_DP_SIZE} \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port ${EMBEDDING_PORT}
" > "$log_file" 2>&1 &
VLLM_PID=$!

echo "⏳ Waiting for server to start (PID: $VLLM_PID)..."
WAIT_TIMEOUT=600
WAIT_ELAPSED=0
while true; do
    if grep -q "Started server process" "$log_file" 2>/dev/null; then
        echo "✅ Server started successfully!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ Server process exited unexpectedly. Check log: $log_file"
        tail -20 "$log_file" 2>/dev/null
        return 1 2>/dev/null || exit 1
    fi
    if [ "$WAIT_ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
        echo "❌ Timeout (${WAIT_TIMEOUT}s) waiting for server to start. Check log: $log_file"
        return 1 2>/dev/null || exit 1
    fi
    sleep 5
    WAIT_ELAPSED=$((WAIT_ELAPSED + 5))
done

# ------------------------------------------------------------------------------
# Output Results
# ------------------------------------------------------------------------------
echo -e "\n===== Deployment Complete ====="
export EMBEDDING_SERVICE_URL="http://$net0_ip:$EMBEDDING_PORT/v1"
export EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}"
echo "📍 Endpoint: $EMBEDDING_SERVICE_URL"
echo "📝 Log: $log_file"
echo ""
echo "Environment variables:"
echo "  EMBEDDING_SERVICE_URL=$EMBEDDING_SERVICE_URL"
echo "  EMBEDDING_MODEL_NAME=$EMBEDDING_MODEL_NAME"
# Only relevant when the service sits behind an authenticated gateway;
# a local vLLM accepts any key, so <unset> is fine here.
echo "  EMBEDDING_API_KEY=${EMBEDDING_API_KEY:-<unset>}"
