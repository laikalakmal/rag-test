#!/bin/bash
# ============================================================================
# scripts/run_d4_static_benchmark.sh
# ============================================================================
# Step 2 of the 6-Step Defense Evaluation Protocol:
# Static Baseline Benchmark for Defense D4 (Hierarchical Context Delimiters & Prompt Spotlighting).
#
# Runs the 100 handcrafted static attacks (data/attacks/static_attacks.json)
# across target models in memory-safe batches of 20 with explicit Ollama VRAM unloads.
#
# Target Models:
#   - mistral:7b
#   - llama3:latest
#   - gemma:7b
#   - vicuna:7b
#
# Usage:
#   ./scripts/run_d4_static_benchmark.sh [model|all] [start_offset] [batch_size]
# Examples:
#   ./scripts/run_d4_static_benchmark.sh mistral:7b
#   ./scripts/run_d4_static_benchmark.sh all
#   ./scripts/run_d4_static_benchmark.sh llama3:latest 40 20
# ============================================================================

set -e

TARGET_INPUT=${1:-all}
START_OFFSET=${2:-0}
BATCH_SIZE=${3:-20}
SINGLE_BATCH=${4:-false}
TOTAL_ATTACKS=100
LOG_DIR="logs/batch_eval_runs"
mkdir -p "$LOG_DIR"

JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-flash-lite}"

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"

# Resolve and export GCP_VM_IP if running against dedicated VM port
if [ -z "$GCP_VM_IP" ] && [ "$OLLAMA_PORT" = "11435" ]; then
    GCP_VM_IP=$(gcloud compute instances describe instance-20261013-025109 --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)
    if [ -n "$GCP_VM_IP" ]; then
        export GCP_VM_IP
        echo "Resolved dedicated static VM IP: $GCP_VM_IP"
    fi
fi

# ─────────────────────────────────────────────────────────
# Ollama Memory Clean & Health Checks
# ─────────────────────────────────────────────────────────

check_ollama() {
    if ! curl -sf "${OLLAMA_BASE_URL}/" > /dev/null 2>&1; then
        echo "ERROR: Ollama server is not running at ${OLLAMA_BASE_URL}/. Start it before running."
        exit 1
    fi
}

clean_memory() {
    echo "  [Ollama] Cleaning VRAM on ${OLLAMA_BASE_URL}..."
    local loaded_models
    loaded_models=$(curl -sf "${OLLAMA_BASE_URL}/api/ps" | python3 -c '
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get("models", []):
        print(m.get("name", ""))
except Exception:
    pass
' 2>/dev/null)

    if [ -z "$loaded_models" ]; then
        echo "  [Ollama] No models currently loaded in memory."
    else
        while IFS= read -r m; do
            if [ -n "$m" ]; then
                echo "  [Ollama] Unloading model: $m"
                curl -sf "${OLLAMA_BASE_URL}/api/generate" \
                    -d "{\"model\": \"$m\", \"keep_alive\": 0}" > /dev/null
            fi
        done <<< "$loaded_models"
        echo "  [Ollama] Waiting 5s for memory to settle..."
        sleep 5
        echo "  [Ollama] Memory cleaned."
    fi
}

cleanup_on_exit() {
    echo ""
    echo "Interrupted! Performing emergency Ollama memory cleanup..."
    clean_memory
    exit 1
}
trap cleanup_on_exit SIGINT SIGTERM

# ─────────────────────────────────────────────────────────
# Batch Runner for a Specific Model
# ─────────────────────────────────────────────────────────

run_model_batches() {
    local model=$1
    local offset=$START_OFFSET

    echo "=========================================================="
    echo " Starting D4 Static Benchmark for Model: $model"
    echo " Attacks: 100 Handcrafted Static Payloads (manual)"
    echo " Defense: SpotlightingDefense (D4 active in config/agent_config_d4.yaml)"
    echo " Judge:   $JUDGE_MODEL"
    echo " Batch:   Size $BATCH_SIZE, Starting at offset $offset"
    echo "=========================================================="

    clean_memory

    while [ "$offset" -lt "$TOTAL_ATTACKS" ]; do
        local timestamp
        timestamp=$(date +"%Y%m%d_%H%M%S")
        local safe_model_name
        safe_model_name=$(echo "$model" | tr ':' '_')
        local log_file="$LOG_DIR/d4_static_${safe_model_name}_offset${offset}_${timestamp}.log"

        echo ""
        echo "----------------------------------------------------------"
        echo " Running Batch: Model=$model | Offset=$offset | Size=$BATCH_SIZE"
        echo " Log: $log_file"
        echo "----------------------------------------------------------"

        OLLAMA_API_URL="${OLLAMA_BASE_URL}/api/generate" \
        venv/bin/python3 src/evaluation/run_evaluations.py \
            --dataset attacks \
            --attack-source manual \
            --sample-size "$BATCH_SIZE" \
            --offset "$offset" \
            --judge llm \
            --judge-model "$JUDGE_MODEL" \
            --agent-model "$model" \
            --config config/agent_config_d4.yaml \
            2>&1 | tee "$log_file"

        local exit_code=${PIPESTATUS[0]}
        if [ "$exit_code" -ne 0 ]; then
            echo "ERROR: Batch execution failed for model $model at offset $offset (exit code $exit_code). Halting."
            clean_memory
            exit 1
        fi

        echo "✓ Batch completed successfully (Offset $offset)."
        clean_memory

        if [ "$SINGLE_BATCH" = "true" ]; then
            echo "Single batch mode requested; stopping after offset $offset."
            break
        fi

        offset=$((offset + BATCH_SIZE))
    done

    echo "=========================================================="
    echo " Completed All Batches for Model: $model"
    echo "=========================================================="
}

# ─────────────────────────────────────────────────────────
# Main Execution Dispatcher
# ─────────────────────────────────────────────────────────

check_ollama

MODELS=()
if [ "$TARGET_INPUT" == "all" ]; then
    MODELS=("mistral:7b" "llama3:latest" "gemma:7b" "vicuna:7b")
else
    MODELS=("$TARGET_INPUT")
fi

for model in "${MODELS[@]}"; do
    run_model_batches "$model"
    # Reset starting offset for subsequent models in multi-model run
    START_OFFSET=0
done

echo ""
echo "=========================================================="
echo " Running Post-Evaluation Aggregation & Decomposition..."
echo "=========================================================="
venv/bin/python3 scripts/analyze_defense_evals.py

echo "Step 2 Static Benchmark complete!"
