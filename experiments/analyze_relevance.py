import json
import os
import sys
import csv
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from attacks.attack_loader import AttackLoader
from rag.llm_provider import get_llm

def generate_query_for_carrier(llm, carrier_text: str) -> str:
    """Generate a realistic user query for the given carrier text."""
    prompt = (
        "You are a user asking a question to a search engine. "
        "Based on the following document excerpt, generate ONE short, realistic search query "
        "that would naturally retrieve this document. Do not include quotes or any other text.\n\n"
        f"Document Excerpt: {carrier_text}\n\nSearch Query:"
    )
    try:
        response = llm.invoke(prompt)
        # Handle string response vs AIMessage object depending on langchain version
        if hasattr(response, 'content'):
            return response.content.strip().strip('"').strip("'")
        return str(response).strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error generating query: {e}")
        # Fallback to a simple heuristic if LLM fails
        return carrier_text[:50] + "?"

def main():
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Loading LLM for query generation...")
    llm = get_llm()
    
    loader = AttackLoader()
    # Load all attacks (manual + bipia)
    loader.load_external_dataset(dataset_name="bipia", num_samples=200)
    attacks = loader.get_all_attacks()
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'relevance_analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'relevance_drop_metrics.csv')
    
    results = []
    
    print(f"Processing {len(attacks)} attacks...")
    for i, attack in enumerate(tqdm(attacks)):
        attack_id = attack.get("attack_id", f"unknown_{i}")
        carrier_text = attack.get("carrier_text", "")
        injected_text = attack.get("injected_text", "")
        full_document = attack.get("full_document", "")
        
        if not carrier_text or not full_document:
            continue
            
        # 1. Generate Query
        query = generate_query_for_carrier(llm, carrier_text)
        
        # 2. Embeddings
        query_emb = model.encode([query])
        clean_emb = model.encode([carrier_text])
        poisoned_emb = model.encode([full_document])
        
        # 3. Similarities
        clean_score = float(cosine_similarity(query_emb, clean_emb)[0][0])
        poisoned_score = float(cosine_similarity(query_emb, poisoned_emb)[0][0])
        drop = clean_score - poisoned_score
        
        results.append({
            "attack_id": attack_id,
            "category": attack.get("category", "unknown"),
            "clean_score": round(clean_score, 4),
            "poisoned_score": round(poisoned_score, 4),
            "drop": round(drop, 4),
            "query": query,
            "carrier_len": len(carrier_text),
            "injected_len": len(injected_text)
        })
        
        # Periodically save to not lose progress if it crashes
        if (i + 1) % 50 == 0 or (i + 1) == len(attacks):
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
    print(f"\nAnalysis complete! Results saved to {out_csv}")
    
    # Calculate some basic stats
    drops = [r["drop"] for r in results]
    clean_scores = [r["clean_score"] for r in results]
    poisoned_scores = [r["poisoned_score"] for r in results]
    
    print("\n--- Summary Statistics ---")
    print(f"Average Clean Score:    {np.mean(clean_scores):.4f}")
    print(f"Average Poisoned Score: {np.mean(poisoned_scores):.4f}")
    print(f"Average Drop:           {np.mean(drops):.4f}")
    print(f"Max Drop:               {np.max(drops):.4f}")
    print(f"Min Drop:               {np.min(drops):.4f}")

if __name__ == "__main__":
    main()
