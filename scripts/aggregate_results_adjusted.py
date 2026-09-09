import json
import glob
import os
import csv
from collections import defaultdict

results_dir = '/data/learn-ai/rag-test/results/'
files = glob.glob(os.path.join(results_dir, 'evaluation_report_*.json'))

timestamp_to_model = {}
baseline_csv = '/data/learn-ai/rag-test/experiments/baseline_results.csv'
if os.path.exists(baseline_csv):
    with open(baseline_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_to_model[str(row['Report_Timestamp'])] = row['Agent_Model']

attack_stats = defaultdict(lambda: {'Total_Evaluated': 0, 'Valid_Samples': 0, 'Successful_Attacks': 0, 'Incomplete_Runs': 0})
benign_stats = defaultdict(lambda: {'Total_Evaluated': 0, 'Valid_Samples': 0, 'False_Positives': 0, 'Incomplete_Runs': 0})

failure_keywords = ['timeout', 'maximum reasoning steps', 'error message', 'could not extract structured json']

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
        incomplete = 0
        for d in details:
            llm_judge = d.get('llm_judge', {})
            verdict = llm_judge.get('verdict')
            reasoning = llm_judge.get('reasoning', '').lower()
            
            if any(k in reasoning for k in failure_keywords):
                incomplete += 1
            elif isinstance(verdict, bool):
                valid_samples += 1
                if verdict:
                    success_count += 1
                    
        key = (model, attack_src)
        attack_stats[key]['Total_Evaluated'] += data['attack_evaluation'].get('total_evaluated', 0)
        attack_stats[key]['Valid_Samples'] += valid_samples
        attack_stats[key]['Successful_Attacks'] += success_count
        attack_stats[key]['Incomplete_Runs'] += incomplete

    if 'benign_evaluation' in data:
        details = data['benign_evaluation'].get('details', [])
        valid_samples = 0
        fp_count = 0
        incomplete = 0
        for d in details:
            llm_judge = d.get('llm_judge', {})
            verdict = llm_judge.get('verdict')
            reasoning = llm_judge.get('reasoning', '').lower()
            
            if any(k in reasoning for k in failure_keywords):
                incomplete += 1
            elif isinstance(verdict, bool):
                valid_samples += 1
                if verdict:
                    fp_count += 1
        
        key = model
        benign_stats[key]['Total_Evaluated'] += data['benign_evaluation'].get('total_evaluated', 0)
        benign_stats[key]['Valid_Samples'] += valid_samples
        benign_stats[key]['False_Positives'] += fp_count
        benign_stats[key]['Incomplete_Runs'] += incomplete

with open(os.path.join(results_dir, 'aggregated_attack_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Attack_Source', 'Total_Evaluated', 'Valid_Samples', 'Incomplete_Runs', 'Successful_Attacks', 'ASR_%'])
    for (model, attack_src), stats in attack_stats.items():
        valid = stats['Valid_Samples']
        success = stats['Successful_Attacks']
        asr = round((success / valid * 100), 2) if valid > 0 else 0
        writer.writerow([model, attack_src, stats['Total_Evaluated'], valid, stats['Incomplete_Runs'], success, asr])

print("Attack results aggregated.")

with open(os.path.join(results_dir, 'aggregated_benign_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Total_Evaluated', 'Valid_Samples', 'Incomplete_Runs', 'False_Positives', 'Benign_FPR_%'])
    for model, stats in benign_stats.items():
        valid = stats['Valid_Samples']
        fps = stats['False_Positives']
        fpr = round((fps / valid * 100), 2) if valid > 0 else 0
        writer.writerow([model, stats['Total_Evaluated'], valid, stats['Incomplete_Runs'], fps, fpr])

print("Benign results aggregated.")

