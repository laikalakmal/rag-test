import sys
import os
import csv
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.attacks.attack_loader import BenignQueryLoader

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading benign queries...")
    loader = BenignQueryLoader()
    queries = loader.get_all_queries()
    
    print("Loading attack_eval corpus chunks...")
    corpus_path = "vector_db/attack_eval/metadata.json"
    with open(corpus_path, "r") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} documents. Encoding all documents...")
    # To save time, we will only encode a small sample if it's too large, or we can just encode them all.
    # Since all-MiniLM-L6-v2 is fast, we encode all.
    docs_text = [chunk["text"] for chunk in chunks]
    docs_emb = embedding_model.encode(docs_text, show_progress_bar=True)
    
    results = []
    
    for query_obj in tqdm(queries, desc="Analyzing Benign Queries"):
        query_text = query_obj.get('query')
        topic = query_obj.get('topic')
        
        query_emb = embedding_model.encode(query_text)
        
        # Calculate similarity with all docs
        similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in docs_emb]
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_doc = docs_text[best_idx]
        
        results.append({
            'query_id': query_obj.get('query_id'),
            'topic': topic,
            'query': query_text,
            'top_doc_preview': best_doc[:50].replace('\n', ' '),
            'similarity_score': best_sim
        })
        
    os.makedirs('results/relevance_analysis', exist_ok=True)
    out_path = 'results/relevance_analysis/benign_relevance.csv'
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'topic', 'query', 'top_doc_preview', 'similarity_score'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {out_path}")
    
    # Print stats
    scores = [r['similarity_score'] for r in results]
    print(f"\nStats:")
    print(f"  Min score: {min(scores):.4f}")
    print(f"  Max score: {max(scores):.4f}")
    print(f"  Mean score: {np.mean(scores):.4f}")
    print(f"  Median score: {np.median(scores):.4f}")

if __name__ == '__main__':
    main()
