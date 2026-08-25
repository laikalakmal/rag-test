import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.rag.llm_provider import build_ollama_llm

def main():
    print("Loading XSTest queries...")
    with open("data/benign/benign_queries.json", "r") as f:
        benign_data = json.load(f)
        
    queries = benign_data.get("queries", [])
    xstest_queries = [q for q in queries if "XSTEST" in q.get("query_id", "")]
    print(f"Found {len(xstest_queries)} XSTest queries.")
    
    if not xstest_queries:
        print("No XSTest queries found. Exiting.")
        return

    print("Loading chunked corpus...")
    corpus_file = "data/processed/chunked_corpus.json"
    with open(corpus_file, "r") as f:
        corpus = json.load(f)
        
    print(f"Current corpus size: {len(corpus)} chunks.")
    
    llm = build_ollama_llm()
    
    new_chunks = []
    chunk_index_start = len(corpus)
    
    print("Generating synthetic documents for XSTest queries using Llama3...")
    for i, q in enumerate(tqdm(xstest_queries)):
        prompt = (
            f"Write a short, factual, objective, Wikipedia-style paragraph (2-3 sentences) "
            f"that directly answers or relates to the following topic or query: '{q['query']}'. "
            f"Do not include conversational filler like 'Here is a paragraph'."
        )
        
        try:
            # We don't want it to fail on a single timeout, although _call has retries.
            response_text = llm._call(prompt)
            
            chunk = {
                "chunk_id": f"xstest_doc_{i}",
                "doc_id": f"xstest_source_{i}",
                "chunk_index": 0,
                "total_chunks": 1,
                "text": response_text.strip(),
                "title": f"Information on: {q['query'][:50]}...",
                "source": "synthetic_xstest",
                "trust_score": 0.9,
                "url": ""
            }
            new_chunks.append(chunk)
            
        except Exception as e:
            print(f"Failed to generate for query {q['query_id']}: {e}")
            
    print(f"Generated {len(new_chunks)} new chunks.")
    
    corpus.extend(new_chunks)
    
    print("Saving updated corpus...")
    with open(corpus_file, "w") as f:
        json.dump(corpus, f, indent=2)
        
    print("Done! You can now rebuild the vector database.")

if __name__ == "__main__":
    main()
