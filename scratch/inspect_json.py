import json, glob, os

files = glob.glob('/data/learn-ai/rag-test/results/*_both.json')
if files:
    with open(files[-1], 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=2)[:1000])
