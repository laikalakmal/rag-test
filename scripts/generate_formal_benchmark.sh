#!/bin/bash
set -e

echo "=========================================================="
echo " Generating Formal 100-Sample Adaptive Benchmark Dataset"
echo "=========================================================="
echo "Note: This uses the Gemini API to mutate 100 attacks. Due to API rate limits (10 req/min),"
echo "this process is executed sequentially and has built-in retry logic."
echo ""

source venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash-lite}"
echo "Using Gemini model: $GEMINI_MODEL (4,000 RPM, 150,000 RPD quota)"
echo ""

# 1. Generate 50 custom attacks
echo "Step 1/3: Generating 50 Custom RAG Adaptive Attacks..."
python3 src/attacks/create_adaptive_attacks.py \
    --num_seeds 50 \
    --seed_path data/attacks/static_attacks.json \
    --output data/attacks/adaptive_attacks_50.json \
    --model "$GEMINI_MODEL"

# 2. Generate 50 BIPIA attacks
echo -e "\nStep 2/3: Generating 50 BIPIA Adaptive Attacks..."
python3 src/attacks/create_adaptive_attacks.py \
    --num_seeds 50 \
    --seed_path src/corpus/data/attacks/bipia_processed.json \
    --output data/attacks/adaptive_attacks_bipia_50.json \
    --model "$GEMINI_MODEL"

# 3. Merge them into a single 100-sample benchmark
echo -e "\nStep 3/3: Merging datasets into formal_benchmark_100.json..."
python3 -c "
import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

custom = load_json('data/attacks/adaptive_attacks_50.json')
bipia = load_json('data/attacks/adaptive_attacks_bipia_50.json')

merged_attacks = custom.get('attacks', []) + bipia.get('attacks', [])

merged_data = {
    'metadata': {
        'description': 'Formal 100-Sample Adaptive Indirect Prompt Injection Benchmark',
        'total_attacks': len(merged_attacks),
        'sources': '50 Custom RAG + 50 BIPIA',
        'predicted_stage1_pass_count': sum(1 for a in merged_attacks if a.get('predicted_stage1_pass'))
    },
    'attacks': merged_attacks
}

with open('data/attacks/formal_benchmark_100.json', 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f'Successfully merged {len(merged_attacks)} attacks into data/attacks/formal_benchmark_100.json')
"
echo "Done!"
