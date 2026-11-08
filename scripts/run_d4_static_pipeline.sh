#!/bin/bash
# ============================================================================
# scripts/run_d4_static_pipeline.sh
# ============================================================================
# Dedicated Cloud Evaluation Pipeline for Defense D4 (Spotlighting) Static Attacks
# Uses instance-20261013-025109 (10 vCPU AMD EPYC, 20 GiB RAM) on local port 11435
# Follows the /ollama-batched-evaluation skill:
#   - Automatic VM startup & autossh port forwarding
#   - Batches of 20 with explicit Ollama VRAM unloads
#   - Mandatory teardown on exit/cancellation (cost guardrail)
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:-mistral:7b}"
OFFSET="${2:-0}"
BATCH_SIZE="${3:-20}"
SINGLE_BATCH="${4:-false}"

ZONE="us-central1-a"
VM_NAME="instance-20261013-025109"
VM_UNUSED="instance-20261023-051940"
OLLAMA_PORT="11435"

teardown_all_vms() {
    echo ""
    echo "=========================================================="
    echo " [TEARDOWN] Stopping All Cloud Compute Instances..."
    echo "=========================================================="
    pkill -9 -f autossh 2>/dev/null || true
    pkill -9 -f "ssh.*1143[56]" 2>/dev/null || true
    local retry=0
    while [ "$retry" -lt 5 ]; do
        if gcloud compute instances stop "$VM_NAME" "$VM_UNUSED" --zone="$ZONE" --quiet 2>/dev/null; then
            echo " [TEARDOWN] Verified: All VMs stopped."
            break
        fi
        retry=$((retry + 1))
        echo " [TEARDOWN] Retrying gcloud compute instances stop ($retry/5) in 5s..."
        sleep 5
    done
    echo " [TEARDOWN] Verified: Process complete. autossh tunnels terminated."
}

trap teardown_all_vms EXIT INT TERM

start_vm_and_tunnel() {
    echo ""
    echo "=========================================================="
    echo " [STARTUP] Checking & Starting Dedicated Cloud VM ($VM_NAME)..."
    echo "=========================================================="
    local status
    status=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(status)' 2>/dev/null || echo "UNKNOWN")
    if [ "$status" != "RUNNING" ]; then
        echo "Starting $VM_NAME in zone $ZONE..."
        gcloud compute instances start "$VM_NAME" --zone="$ZONE"
        echo "Waiting 15s for Ollama and SSH daemon to initialize..."
        sleep 15
    fi

    local vm_ip
    vm_ip=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo "Target VM External IP: $vm_ip"
    export GCP_VM_IP="$vm_ip"

    # Establish or restart tunnel
    pkill -9 -f "autossh.*11435" 2>/dev/null || true
    pkill -9 -f "ssh.*11435" 2>/dev/null || true
    echo "Establishing autossh port forwarding tunnel (Port 11435 -> $vm_ip:11434)..."
    autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
        -o "StrictHostKeyChecking=no" -i ~/.ssh/gcp_ollama -f -N \
        -L 11435:localhost:11434 "laika@$vm_ip"

    sleep 5
    local retries=0
    while ! curl -sf "http://localhost:${OLLAMA_PORT}/" > /dev/null 2>&1; do
        retries=$((retries + 1))
        if [ "$retries" -ge 15 ]; then
            echo "ERROR: Failed to establish Ollama connection on http://localhost:${OLLAMA_PORT}/"
            exit 1
        fi
        echo "Waiting for Ollama tunnel on http://localhost:${OLLAMA_PORT}/ ($retries/15)..."
        sleep 3
    done
    echo "✓ Ollama server is responsive at http://localhost:${OLLAMA_PORT}/"
}

start_vm_and_tunnel

echo "=========================================================="
echo " Starting Defense D4 Static Evaluation Pipeline"
echo " Target Endpoint : http://localhost:${OLLAMA_PORT}"
echo " Model Tested    : $MODEL"
echo " Active Defense  : spotlighting (Hierarchical Delimiters)"
echo " Start Offset    : $OFFSET"
echo " Batch Size      : $BATCH_SIZE"
echo " Single Batch    : $SINGLE_BATCH"
echo "=========================================================="

OLLAMA_PORT="$OLLAMA_PORT" ./scripts/run_d4_static_benchmark.sh "$MODEL" "$OFFSET" "$BATCH_SIZE" "$SINGLE_BATCH"

echo "=========================================================="
echo " D4 Static Benchmark Run Completed for $MODEL"
echo "=========================================================="
