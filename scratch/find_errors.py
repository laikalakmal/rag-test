import json, glob, os

def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*.json'))
    error_phrases = set()
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            if 'benign_evaluation' in data:
                details = data['benign_evaluation'].get('details', [])
                for d in details:
                    judge = d.get('llm_judge', {})
                    if not judge:
                        continue
                    reasoning = judge.get('reasoning', '')
                    raw = judge.get('raw_response', '')
                    
                    if 'error' in reasoning.lower() or 'error' in raw.lower():
                        error_phrases.add(reasoning)
                    if 'maximum reasoning steps' in reasoning.lower():
                        error_phrases.add(reasoning)
                    if 'crash' in reasoning.lower():
                        error_phrases.add(reasoning)
                    if 'incomplete' in reasoning.lower():
                        error_phrases.add(reasoning)
        except Exception:
            pass

    print("Unique error reasonings:")
    for p in error_phrases:
        print(f" - {p}")

if __name__ == "__main__":
    main()
