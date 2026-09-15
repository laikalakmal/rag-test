import json
import glob
import os
import csv
from collections import defaultdict

def is_invalid_sample(judge_dict):
    if not judge_dict:
        return True
    
    if 'verdict' not in judge_dict:
        return True
        
    reasoning = judge_dict.get('reasoning', '').lower()
    raw_response = judge_dict.get('raw_response', '').lower()
    
    # Specific, precise failure signatures that indicate a technical crash
    # rather than a legitimate model evaluation.
    error_phrases = [
        "logger' is not defined",
        "maximum reasoning steps",
        "[fallback parse]",
        "cannot connect",
        "404 error",
        "llm call failed",
        "internal error",
        "programming error",
        "runtime error",
        "retrieval issue",
        "timeout error",
        "standard error message"
    ]
    
    for phrase in error_phrases:
        if phrase in reasoning or phrase in raw_response:
            return True
            
    return False

def main():
    results_dir = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(results_dir, '*_both.json'))
    
    # Store metrics per model and attack source: 
    # {(model_name, attack_source): {'total': 0, 'invalid': 0, 'successful': 0}}
    metrics = defaultdict(lambda: {'total': 0, 'invalid': 0, 'successful': 0})
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'attack_evaluation' not in data:
                continue
                
            model_name = data.get('agent_llm_model', 'unknown')
            attack_source = data.get('attack_source', 'unknown')
            details = data['attack_evaluation'].get('details', [])
            
            # Use tuple of (model_name, attack_source) as key
            key = (model_name, attack_source)
            
            for item in details:
                metrics[key]['total'] += 1
                
                judge = item.get('llm_judge', {})
                if is_invalid_sample(judge):
                    metrics[key]['invalid'] += 1
                else:
                    if judge.get('verdict') is True:
                        metrics[key]['successful'] += 1
                        
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    csv_path = os.path.join(results_dir, 'aggregated_attack_results.csv')
    print("Writing aggregated results to:", csv_path)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Attack_Source', 'Total_Evaluated', 'Valid_Samples', 'Successful_Attacks', 'ASR_%', 'Excluded_Errors'])
        
        for key in sorted(metrics.keys()):
            model, source = key
            stats = metrics[key]
            total = stats['total']
            invalid = stats['invalid']
            successes = stats['successful']
            valid = total - invalid
            
            asr = (successes / valid * 100) if valid > 0 else 0.0
            
            row = [model, source, total, valid, successes, round(asr, 2), invalid]
            writer.writerow(row)
            print(f"{model} ({source}): Total={total}, Valid={valid}, ASR={asr:.2f}%, Excluded={invalid}")

if __name__ == "__main__":
    main()
