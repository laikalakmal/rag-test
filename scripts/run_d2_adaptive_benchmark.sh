#!/bin/bash
# ============================================================================
# scripts/run_d2_adaptive_benchmark.sh
# ============================================================================
# Closed-Loop Adaptive Evaluation for Defense D2 (Pattern Matching Regex Blocklist)
# on the Formal 100-Sample Benchmark.
#
# Target Models:
#   - mistral:7b
#   - llama3:latest
#   - gemma:7b
#   - vicuna:7b
#
# Usage:
#   ./scripts/run_d2_adaptive_benchmark.sh [model|all] [sample_size]
# Examples:
#   ./scripts/run_d2_adaptive_benchmark.sh all 0        # Full 100-sample benchmark across all 4 models
#   ./scripts/run_d2_adaptive_benchmark.sh mistral:7b 0 # Full 100-sample benchmark for mistral:7b
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

OLLAMA_PORT="${OLLAMA_PORT:-11435}"
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

restart_and_flush_ollama() {
    echo "  [Periodic Refresh] Flushing OS kernel page caches & restarting Ollama service..."
    local vm_ip="${GCP_VM_IP:-34.57.115.40}"
    local ssh_key="$HOME/.ssh/gcp_ollama"
    if [ -f "$ssh_key" ]; then
        ssh -o StrictHostKeyChecking=no -i "$ssh_key" "laika@$vm_ip" \
            "sudo systemctl restart ollama && sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches" > /dev/null 2>&1 || true
        sleep 4
        # Verify tunnel
        local retries=0
        while ! curl -sf "${OLLAMA_BASE_URL}/" > /dev/null 2>&1; do
            retries=$((retries + 1))
            if [ "$retries" -ge 15 ]; then
                echo "  [Periodic Refresh] Tunnel recovery timed out."
                break
            fi
            sleep 2
        done
        echo "  [Periodic Refresh] Clean baseline confirmed. Ready."
    else
        clean_memory
    fi
}

cleanup_on_exit() {
    echo ""
    echo "Interrupted! Performing emergency Ollama memory cleanup on ${OLLAMA_BASE_URL}..."
    clean_memory
    exit 1
}
trap cleanup_on_exit SIGINT SIGTERM

# ─────────────────────────────────────────────────────────
# Model Execution
# ─────────────────────────────────────────────────────────

run_model_adaptive() {
    local model=$1
    local safe_model_name
    safe_model_name=$(echo "$model" | tr ':' '_')
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_file="results/adaptive_feedback/d2_formal_100_${safe_model_name}.json"
    local log_file="$LOG_DIR/d2_adaptive_${safe_model_name}_${timestamp}.log"

    echo "=========================================================="
    echo " Starting D2 Adaptive Closed-Loop Benchmark: $model"
    echo " Attacks:   $BENCHMARK_PATH"
    echo " Output:    $output_file"
    echo " Defense:   pattern_matching"
    echo " Judge:     $JUDGE_MODEL"
    echo " Mutator:   $MUTATOR_MODEL"
    echo " Endpoint:  ${OLLAMA_BASE_URL}/api/generate"
    echo " Log:       $log_file"
    echo "=========================================================="

    restart_and_flush_ollama

    OLLAMA_NUM_THREADS="${OLLAMA_NUM_THREADS:-8}" \
    venv/bin/python3 scripts/run_adaptive_feedback_loop.py \
        --agent_model "$model" \
        --defense pattern_matching \
        --judge_model "$JUDGE_MODEL" \
        --mutator_model "$MUTATOR_MODEL" \
        --attacks_path "$BENCHMARK_PATH" \
        --sample_size "$SAMPLE_SIZE" \
        --ollama_api_url "${OLLAMA_BASE_URL}/api/generate" \
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
    run_model_adaptive "$model"
done

echo ""
echo "=========================================================="
echo " Aggregating Adaptive Results into Summary CSV..."
echo "=========================================================="
venv/bin/python3 -c '
import glob
from scripts.analyze_adaptive_human_reviews import run_adaptive_analysis

files = sorted(glob.glob("results/adaptive_feedback/d2_formal_100_*.json"))
print(f"Found {len(files)} D2 adaptive files to analyze: {files}")
run_adaptive_analysis(files, output_dir="results", verbose=True, append=True)
'

echo "Adaptive Closed-Loop Benchmark for Defense D2 complete!"
