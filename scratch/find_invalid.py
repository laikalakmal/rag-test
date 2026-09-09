import json, glob, os
def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*.json'))
    invalid_examples = []
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            if 'benign_evaluation' in data:
                details = data['benign_evaluation'].get('details', [])
                for d in details:
                    # check for invalid
                    invalid_reason = None
                    # e.g., missing judge
                    if not d.get('llm_judge'):
                        invalid_reason = "missing_judge"
                    elif 'error' in str(d.get('response', '')).lower():
                        invalid_reason = "response_error"
                    elif d.get('response', '') == '':
                        invalid_reason = "empty_response"
                    elif 'verdict' not in d.get('llm_judge', {}):
                        invalid_reason = "missing_verdict"
                        
                    if invalid_reason and len(invalid_examples) < 5:
                        invalid_examples.append((f, d.get('query_id'), invalid_reason, d))
        except Exception as e:
            pass

    for ex in invalid_examples:
        print(f"File: {os.path.basename(ex[0])} Query: {ex[1]} Reason: {ex[2]}")
        print(ex[3].get('llm_judge'))
        print("---")

if __name__ == "__main__":
    main()
