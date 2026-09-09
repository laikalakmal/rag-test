import json
import sys

def main():
    path = '/data/learn-ai/rag-test/results/evaluation_report_20260819_144958_both.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = data.get('benign_evaluation', {}).get('details', [])
    print(f"Total samples: {len(results)}")
    
    judge_positives = [r for r in results if r['judge_scores'].get('llm_judge', 0) == 1]
    judge_negatives = [r for r in results if r['judge_scores'].get('llm_judge', 0) == 0]
    
    print(f"LLM Judge flagged {len(judge_positives)} as refusals.")
    for r in judge_positives:
        print(f"Query: {r['query']}")
        print(f"Response: {r['response']}")
        print(f"Judge Score: {r['judge_scores'].get('llm_judge')}")
        print("-" * 40)
        
    print(f"First 5 LLM Judge passed:")
    for r in judge_negatives[:5]:
        print(f"Query: {r['query']}")
        print(f"Response: {r['response'][:150]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()
