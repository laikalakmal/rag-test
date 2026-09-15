import json, glob, os

files = glob.glob('/data/learn-ai/rag-test/results/*_both.json')
found = False
for f in files:
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            if 'attack_evaluation' in data:
                print(f"File: {os.path.basename(f)}")
                print(f"Attack source: {data.get('attack_source')}")
                details = data['attack_evaluation'].get('details', [])
                if details:
                    print(json.dumps(details[0], indent=2))
                found = True
                break
    except Exception:
        pass
