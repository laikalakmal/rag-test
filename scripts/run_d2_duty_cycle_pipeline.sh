#!/bin/bash
# ============================================================================
# scripts/run_d2_duty_cycle_pipeline.sh
# ============================================================================
# Enforces a strict 3-hour duty cycle on Defense D2 Static BIPIA Benchmarks:
# 1. Runs target models in 10-sample batches.
# 2. Tracks session elapsed wall-clock time.
# 3. Completely halts execution after ~2.5 - 2.8 hours:
#    - Shuts down the cloud compute VM completely (STATUS: TERMINATED)
#    - Terminates all autossh/ssh tunnels
#    - Ingests and decomposes security metrics (analyze_defense_evals.py)
#    - Synchronizes artifacts to permanent thesis vault
# 4. Enforces a 3-minute cooldown.
# 5. Automatically restarts for the next 3-hour session until all 200 samples
#    are completed across remaining models (gemma:7b, vicuna:7b).
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ZONE="us-central1-a"
VM_NAME="instance-20261013-025109"
VM_UNUSED="instance-20261023-051940"
OLLAMA_PORT="11435"
MAX_SESSION_SECS=9600  # 160 minutes (~2.6 hours) to guarantee stopping strictly before 3 hours
COOLDOWN_SECS=180      # 3 minutes cooldown between 3-hour blocks
TOTAL_ATTACKS=200
BATCH_SIZE=10

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
    echo " [STARTUP] Checking & Starting Dedicated Cloud VM..."
    echo "=========================================================="
    local status
    status=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(status)' 2>/dev/null || echo "UNKNOWN")
    if [ "$status" != "RUNNING" ]; then
        echo "Starting $VM_NAME in zone $ZONE..."
        gcloud compute instances start "$VM_NAME" --zone="$ZONE" --quiet
        echo "Waiting 15s for Ollama and SSH daemon to initialize..."
        sleep 15
    fi

    local vm_ip=""
    local retries=0
    while [ -z "$vm_ip" ] && [ "$retries" -lt 10 ]; do
        vm_ip=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "")
        if [ -n "$vm_ip" ]; then
            break
        fi
        retries=$((retries + 1))
        sleep 2
    done
    echo "Target VM External IP: $vm_ip"
    export GCP_VM_IP="$vm_ip"

    # Establish or restart tunnel
    pkill -9 -f "autossh.*11435" 2>/dev/null || true
    pkill -9 -f "ssh.*11435" 2>/dev/null || true
    sleep 2
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

get_next_bipia_offset() {
    local target_model="$1"
    python3 -c "
import glob, json
files = glob.glob('results/evaluation_report_*_llm.json')
completed = set()
for f in sorted(files):
    try:
        with open(f) as fp:
            data = json.load(fp)
        if data.get('agent_llm_model') != '${target_model}' or data.get('attack_source') != 'bipia':
            continue
        details = data.get('attack_evaluation', {}).get('details', [])
        if not details:
            continue
        if details[0].get('defense_metadata', {}).get('defense_name') != 'pattern_matching':
            continue
        for d in details:
            aid = d.get('attack_id', '')
            if aid.startswith('EXT_BIPIA_'):
                completed.add(int(aid.split('_')[-1]))
    except Exception:
        pass
next_offset = 0
while next_offset in completed:
    next_offset += ${BATCH_SIZE}
print(next_offset)
"
}

sync_thesis_vault() {
    echo ""
    echo "=========================================================="
    echo " [SYNC] Running Metrics Decomposition & Vault Synchronization"
    echo "=========================================================="
    venv/bin/python3 scripts/analyze_defense_evals.py
    venv/bin/python3 scripts/compare_static_adaptive_asr.py --defense pattern_matching --scope both || true

    local vault_dir="/data/my-vault/Research/thesis/artifacts/adaptive_attacks"
    mkdir -p "$vault_dir"
    cp -u results/defense_eval_*.csv "$vault_dir/" 2>/dev/null || true
    cp -u results/static_vs_adaptive_pattern_matching.csv "$vault_dir/" 2>/dev/null || true
    echo "✓ Thesis vault updated at $vault_dir"
}

TARGET_MODELS=("$@")
if [ ${#TARGET_MODELS[@]} -eq 0 ]; then
    TARGET_MODELS=("gemma:7b" "vicuna:7b")
fi

echo "=========================================================="
echo " Starting Strict 3-Hour Duty Cycle Pipeline"
echo " Target Models  : ${TARGET_MODELS[*]}"
echo " Max Session    : $((MAX_SESSION_SECS / 60)) minutes (~2.6 hours)"
echo " Batch Size     : $BATCH_SIZE samples"
echo " Defense Active : PatternMatchingDefense (D2)"
echo " Endpoint       : http://localhost:${OLLAMA_PORT}"
echo "=========================================================="

for model in "${TARGET_MODELS[@]}"; do
    echo ""
    echo "##########################################################"
    echo " PROCESSING MODEL: $model"
    echo "##########################################################"

    while true; do
        current_offset=$(get_next_bipia_offset "$model")
        echo "Current status for $model: starting from offset $current_offset / $TOTAL_ATTACKS"

        if [ "$current_offset" -ge "$TOTAL_ATTACKS" ]; then
            echo "✓ Model $model BIPIA benchmark is 100% complete ($TOTAL_ATTACKS/$TOTAL_ATTACKS)!"
            break
        fi

        # Start VM and tunnel for this 3-hour session
        start_vm_and_tunnel
        session_start=$(date +%s)
        session_batches=0

        while [ "$current_offset" -lt "$TOTAL_ATTACKS" ]; do
            now=$(date +%s)
            elapsed=$((now - session_start))

            # Stop session completely if approaching 3-hour mark
            if [ "$elapsed" -ge "$MAX_SESSION_SECS" ]; then
                echo ""
                echo "=========================================================="
                echo " [3-HOUR CEILING REACHED] Elapsed: $((elapsed / 60)) min. Halting session."
                echo "=========================================================="
                break
            fi

            echo ""
            echo "----------------------------------------------------------"
            echo " Session Run: Model=$model | Offset=$current_offset | Session Elapsed=$((elapsed / 60))m / $((MAX_SESSION_SECS / 60))m"
            echo "----------------------------------------------------------"

            OLLAMA_PORT="$OLLAMA_PORT" ./scripts/run_d2_bipia_benchmark.sh "$model" "$current_offset" "$BATCH_SIZE" true
            batch_exit=$?
            if [ $batch_exit -ne 0 ]; then
                echo "ERROR: Batch at offset $current_offset failed for $model (exit $batch_exit). Halting."
                teardown_all_vms
                exit 1
            fi

            session_batches=$((session_batches + 1))
            current_offset=$((current_offset + BATCH_SIZE))
            echo "✓ Completed batch ($session_batches in this session). Next offset is $current_offset."
        done

        # COMPLETE STOP: Teardown VM and tunnels
        echo ""
        echo "=========================================================="
        echo " [COMPLETE STOP] Stopping cloud VM and terminating tunnels..."
        echo "=========================================================="
        teardown_all_vms
        sync_thesis_vault

        if [ "$current_offset" -lt "$TOTAL_ATTACKS" ]; then
            echo ""
            echo "=========================================================="
            echo " [COOLDOWN] Session ended at offset $current_offset / $TOTAL_ATTACKS."
            echo " Cloud VM is completely STOPPED to save credit."
            echo " Resting for ${COOLDOWN_SECS}s before starting next 3-hour session..."
            echo "=========================================================="
            sleep "$COOLDOWN_SECS"
            echo "Resuming execution for $model..."
        fi
    done
done

echo ""
echo "=========================================================="
echo " ALL TARGET MODELS SUCCESSFULLY COMPLETED!"
echo "=========================================================="
sync_thesis_vault
teardown_all_vms
