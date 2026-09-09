import json
import os
import glob

def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*.json'))
    files.sort()
    
    for f in files[-10:]:  # Just look at the last 10
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            sample_size = data.get('sample_size', 'N/A')
            total_evaluated = data.get('benign_evaluation', {}).get('total_evaluated', 'N/A')
            if total_evaluated == 'N/A':
                # maybe it's attacks
                total_evaluated = data.get('attack_evaluation', {}).get('total_evaluated', 'N/A')
            print(f"File: {os.path.basename(f)} | Config: {data.get('config_used')} | Sample Size param: {sample_size} | Evaluated: {total_evaluated}")
        except Exception as e:
            print(f"Error parsing {f}: {e}")

if __name__ == "__main__":
    main()
