import csv

with open('/data/learn-ai/rag-test/results/defense_eval_per_sample.csv', 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        if row.get('Dataset') == 'manual' and row.get('Injected_Chunk_Bypassed') == 'False' and row.get('Judge_Verdict') == 'True':
            print("="*80)
            print("Query:", row.get('Query', 'N/A'))
            print("Agent Answer Preview:", row.get('Agent_Answer_Preview', 'N/A'))
            print("Judge Reasoning:", row.get('Judge_Reasoning', 'N/A'))
            break
