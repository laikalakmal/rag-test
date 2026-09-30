#!/bin/bash
# Batchwise evaluation script with robust Ollama memory management
# Usage: ./run_bipia_batch_evaluation.sh [all|bipia|manual] [agent_model]
# Default agent model: gemma:7b, judge model: gemini-3.1-flash-lite

MODE=${1:-all}
AGENT_MODEL=${2:-}
BATCH_SIZE=20
OFFSET=${3:-0}
LOG_DIR="logs/batch_eval_runs"
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

check_ollama() {
    if ! curl -sf http://localhost:11434/ > /dev/null 2>&1; then
        echo "ERROR: Ollama is not running. Start it with 'ollama serve' and try again."
        exit 1
    fi
}

# Dynamically unload ALL currently loaded models via /api/ps
clean_memory() {
    echo "Cleaning Ollama memory..."
    local loaded_models
    loaded_models=$(curl -sf http://localhost:11434/api/ps | python3 -c '
    import sys, json
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        print(m['name'])
    ' 2>/dev/null)

    if [ -z "$loaded_models" ]; then
        echo "  No models currently loaded. Nothing to clean."
    else
        while IFS= read -r model; do
            echo "  Unloading: $model"
            curl -sf http://localhost:11434/api/generate \
                -d "{\"model\": \"$model\", \"keep_alive\": 0}" > /dev/null
        done <<< "$loaded_models"
        echo "  Waiting 5s for memory to settle..."
        sleep 5
        echo "  Memory cleaned."
    fi
}

# Trap CTRL+C so we always clean up before exiting
cleanup_on_exit() {
    echo ""
    echo "Interrupted! Cleaning memory before exit..."
    clean_memory
    exit 1
}
trap cleanup_on_exit SIGINT SIGTERM

run_batch() {
    local source=$1
    local total=$2
    local offset=$3
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/${source}_offset${offset}_${timestamp}.log"

    echo "=================================================="
    echo "Running $source batch (agent: $AGENT_MODEL): Offset $offset, Size $BATCH_SIZE"
    echo "Log: $log_file"
    echo "=================================================="

    venv/bin/python3 src/evaluation/run_evaluations.py \
        --dataset attacks \
        --attack-source "$source" \
        --sample-size "$BATCH_SIZE" \
        --offset "$offset" \
        --judge llm \
        --judge-model "${JUDGE_MODEL:-gemini-3.1-flash-lite}" \
        --agent-model "$AGENT_MODEL" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Batch failed at offset $offset (exit code $exit_code). Stopping."
        clean_memory
        exit 1
    fi

    clean_memory
}

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

check_ollama

if [[ "$MODE" == "all" || "$MODE" == "bipia" ]]; then
    echo "Starting BIPIA batched evaluation (200 items)..."
    while [ $OFFSET -lt 200 ]; do
        run_batch "bipia" 200 $OFFSET
        OFFSET=$((OFFSET + BATCH_SIZE))
    done
    echo "BIPIA batched evaluation complete!"
fi

if [[ "$MODE" == "all" || "$MODE" == "manual" ]]; then
    echo "Starting Manual batched evaluation (100 items)..."
    while [ $OFFSET -lt 100 ]; do
        run_batch "manual" 100 $OFFSET
        OFFSET=$((OFFSET + BATCH_SIZE))
    done
    echo "Manual batched evaluation complete!"
fi

echo "All evaluations done!"
