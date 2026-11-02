#!/bin/bash
# ============================================================================
# scripts/run_d4_adaptive_benchmark.sh
# ============================================================================
# Closed-Loop Adaptive Evaluation for Defense D4 (Hierarchical Context Delimiters & Spotlighting)
# on the Formal 100-Sample Benchmark.
#
# Target Models:
#   - mistral:7b
#   - llama3:latest
#   - gemma:7b
#   - vicuna:7b
#
# Usage:
#   ./scripts/run_d4_adaptive_benchmark.sh [model|all] [sample_size]
# Examples:
#   ./scripts/run_d4_adaptive_benchmark.sh all 0        # Full 100-sample benchmark across all 4 models
#   ./scripts/run_d4_adaptive_benchmark.sh mistral:7b 0 # Full 100-sample benchmark for mistral:7b
# ============================================================================

set -e

TARGET_INPUT=${1:-all}
SAMPLE_SIZE=${2:-0}
BENCHMARK_PATH="data/attacks/formal_benchmark_100.json"
LOG_DIR="logs/batch_eval_runs"
mkdir -p "$LOG_DIR"
mkdir -p "results/adaptive_feedback"

JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.5-flash-lite}"
MUTATOR_MODEL="${MUTATOR_MODEL:-gemini-3.5-flash-lite}"

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_BASE_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"

# ─────────────────────────────────────────────────────────
# Health Checks & Memory Unload
# ─────────────────────────────────────────────────────────

check_ollama() {
    if ! curl -sf "${OLLAMA_BASE_URL}/" > /dev/null 2>&1; then
        echo "ERROR: Ollama server is not running at ${OLLAMA_BASE_URL}/. Start it or establish tunnel before running."
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
    echo "Interrupted! Performing emergency memory cleanup..."
    clean_memory
    exit 1
}
trap cleanup_on_exit SIGINT SIGTERM

# ─────────────────────────────────────────────────────────
# Model Evaluation Runner
# ─────────────────────────────────────────────────────────

run_adaptive_for_model() {
    local model=$1
    local safe_model_name
    safe_model_name=$(echo "$model" | tr ':' '_')
    local output_file="results/adaptive_feedback/d4_formal_100_${safe_model_name}.json"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/d4_adaptive_${safe_model_name}_${timestamp}.log"

    echo "=========================================================="
    echo " Starting D4 Adaptive Closed-Loop Benchmark: $model"
    echo " Attacks:   $BENCHMARK_PATH"
    echo " Output:    $output_file"
    echo " Defense:   spotlighting (via config/agent_config_d4.yaml)"
    echo " Judge:     $JUDGE_MODEL"
    echo " Mutator:   $MUTATOR_MODEL"
    echo " Endpoint:  ${OLLAMA_BASE_URL}/api/generate"
    echo " Log:       $log_file"
    echo "=========================================================="

    clean_memory

    OLLAMA_NUM_THREADS="${OLLAMA_NUM_THREADS:-8}" \
    venv/bin/python3 scripts/run_adaptive_feedback_loop.py \
        --agent_model "$model" \
        --config config/agent_config_d4.yaml \
        --judge_model "$JUDGE_MODEL" \
        --mutator_model "$MUTATOR_MODEL" \
        --attacks_path "$BENCHMARK_PATH" \
        --sample_size "$SAMPLE_SIZE" \
        --output "$output_file" \
        --resume \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: Adaptive benchmark failed for model $model (exit code $exit_code). Halting."
        clean_memory
        exit 1
    fi

    echo "✓ Finished adaptive evaluation for model: $model"
    clean_memory
}

# ─────────────────────────────────────────────────────────
# Main Dispatcher
# ─────────────────────────────────────────────────────────

check_ollama

MODELS=()
if [ "$TARGET_INPUT" == "all" ]; then
    MODELS=("mistral:7b" "llama3:latest" "gemma:7b" "vicuna:7b")
else
    MODELS=("$TARGET_INPUT")
fi

for model in "${MODELS[@]}"; do
    run_adaptive_for_model "$model"
done

echo ""
echo "=========================================================="
echo " Closed-Loop Adaptive Evaluation Complete!"
echo " Next Steps:"
echo "   1. Decomposed Metrics: python3 scripts/compare_static_adaptive_asr.py --defense spotlighting"
echo "   2. Fate Funnel Plot:   python3 scripts/plot_attack_fate_funnel.py --defense spotlighting"
echo "=========================================================="
