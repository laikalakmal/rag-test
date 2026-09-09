import json, glob, os
def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*.json'))
    gemma_errors = {}
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            if data.get('agent_llm_model') == 'gemma' and 'benign_evaluation' in data:
                details = data['benign_evaluation'].get('details', [])
                for d in details:
                    judge = d.get('llm_judge', {})
                    if not judge:
                        gemma_errors['missing_judge'] = gemma_errors.get('missing_judge', 0) + 1
                        continue
                    reasoning = judge.get('reasoning', '')
                    if 'error' in reasoning.lower():
                        gemma_errors[reasoning] = gemma_errors.get(reasoning, 0) + 1
        except Exception:
            pass
    for k, v in gemma_errors.items():
        print(f"Count {v}: {k}")
if __name__ == "__main__":
    main()
