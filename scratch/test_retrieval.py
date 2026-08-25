import json
from src.rag.tools import SearchKnowledgeBase, ToolCallLog

print("Loading XSTest queries...")
with open("data/benign/benign_queries.json", "r") as f:
    queries = json.load(f)["queries"]

log = ToolCallLog()
search_tool = SearchKnowledgeBase(log=log)

print("\nTesting 3 random XSTest queries against the current corpus:\n")
for q in queries[:3]:
    query_text = q["query"]
    print(f"QUERY: {query_text}")
    results = search_tool.run(query_text)
    try:
        results_json = json.loads(results)
        for i, res in enumerate(results_json):
            title = res.get("title", "Unknown")
            text = res.get("text", "")[:100]
            print(f"  Result {i+1}: {title} - {text}...")
    except Exception as e:
        print(f"  Error parsing results or no results: {results[:100]}")
    print("-" * 50)
