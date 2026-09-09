import json
import glob
import os
import pandas as pd

results_dir = '/data/learn-ai/rag-test/results/'
files = glob.glob(os.path.join(results_dir, 'evaluation_report_*.json'))

attack_records = []
benign_records = []

# Try to load existing mapping from baseline_results.csv if possible
timestamp_to_model = {}
if os.path.exists('/data/learn-ai/rag-test/experiments/baseline_results.csv'):
    df_base = pd.read_csv('/data/learn-ai/rag-test/experiments/baseline_results.csv')
    for _, row in df_base.iterrows():
        timestamp_to_model[str(row['Report_Timestamp'])] = row['Agent_Model']

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        
    timestamp = data.get('timestamp', '')
    model = data.get('agent_llm_model')
    if not model and timestamp in timestamp_to_model:
        model = timestamp_to_model[timestamp]
    if not model:
        model = 'unknown'
        
    attack_src = data.get('attack_source', 'none')
    
    if 'attack_evaluation' in data:
        details = data['attack_evaluation'].get('details', [])
        valid_samples = 0
        success_count = 0
        for d in details:
            llm_judge = d.get('llm_judge', {})
            verdict = llm_judge.get('verdict')
            if isinstance(verdict, bool):
                valid_samples += 1
                if verdict:
                    success_count += 1
        asr = (success_count / valid_samples * 100) if valid_samples > 0 else 0
        attack_records.append({
            'Timestamp': timestamp,
            'Model': model,
            'Attack_Source': attack_src,
            'Total_Evaluated': data['attack_evaluation'].get('total_evaluated', 0),
            'Valid_Samples': valid_samples,
            'Successful_Attacks': success_count,
            'ASR_%': round(asr, 2)
        })
        
    if 'benign_evaluation' in data:
        details = data['benign_evaluation'].get('details', [])
        valid_samples = 0
        fp_count = 0
        for d in details:
            llm_judge = d.get('llm_judge', {})
            verdict = llm_judge.get('verdict')
            if isinstance(verdict, bool):
                valid_samples += 1
                if verdict:
                    fp_count += 1
        fpr = (fp_count / valid_samples * 100) if valid_samples > 0 else 0
        benign_records.append({
            'Timestamp': timestamp,
            'Model': model,
            'Benign_Total_Evaluated': data['benign_evaluation'].get('total_evaluated', 0),
            'Benign_Valid_Samples': valid_samples,
            'False_Positives': fp_count,
            'Benign_FPR_%': round(fpr, 2)
        })

df_attacks = pd.DataFrame(attack_records)
df_benign = pd.DataFrame(benign_records)

# Aggregate by model and attack source
if not df_attacks.empty:
    agg_attacks = df_attacks.groupby(['Model', 'Attack_Source']).agg({
        'Total_Evaluated': 'sum',
        'Valid_Samples': 'sum',
        'Successful_Attacks': 'sum'
    }).reset_index()
    agg_attacks['ASR_%'] = (agg_attacks['Successful_Attacks'] / agg_attacks['Valid_Samples'] * 100).round(2)
    agg_attacks.to_csv(os.path.join(results_dir, 'aggregated_attack_results.csv'), index=False)
    print("Attack results aggregated.")
else:
    print("No attack results found.")

if not df_benign.empty:
    agg_benign = df_benign.groupby(['Model']).agg({
        'Benign_Total_Evaluated': 'sum',
        'Benign_Valid_Samples': 'sum',
        'False_Positives': 'sum'
    }).reset_index()
    agg_benign['Benign_FPR_%'] = (agg_benign['False_Positives'] / agg_benign['Benign_Valid_Samples'] * 100).round(2)
    agg_benign.to_csv(os.path.join(results_dir, 'aggregated_benign_results.csv'), index=False)
    print("Benign results aggregated.")
else:
    print("No benign results found.")

