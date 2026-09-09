import json, glob, os

def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*_both.json'))
    
    unique_reasonings = set()
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            if 'benign_evaluation' in data:
                details = data['benign_evaluation'].get('details', [])
                for d in details:
                    judge = d.get('llm_judge', {})
                    if not judge:
                        unique_reasonings.add("MISSING_JUDGE_OBJECT")
                        continue
                    
                    if 'verdict' not in judge:
                        unique_reasonings.add("MISSING_VERDICT")
                        
                    reasoning = judge.get('reasoning', '').lower()
                    raw = judge.get('raw_response', '').lower()
                    
                    # Heuristics for finding strange or error-like responses
                    if any(x in reasoning for x in ['error', 'fail', 'crash', 'exception', 'timeout', 'maximum', 'fallback', 'incomplete', 'unable to complete', 'not defined']):
                        unique_reasonings.add(judge.get('reasoning', ''))
                    if any(x in raw for x in ['error', 'fail', 'crash', 'exception', 'timeout', 'maximum', 'fallback', 'incomplete', 'unable to complete', 'not defined']):
                        unique_reasonings.add(judge.get('raw_response', ''))
        except Exception:
            pass

    print("Found suspicious reasonings:")
    for r in unique_reasonings:
        print(f" - {r}")

if __name__ == "__main__":
    main()
