"""
Add Wikipedia Documents to Attack Eval Corpus
==============================================
Reads the sourced Wikipedia articles and adds them as new chunks
to the attack_eval vector database, then rebuilds the FAISS index.
Also computes similarity scores to verify improvement.
"""

import sys
import os
import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from attacks.attack_loader import BenignQueryLoader


def main():
    print("=" * 70)
    print("ADDING WIKIPEDIA DOCUMENTS TO ATTACK_EVAL CORPUS")
    print("=" * 70)
    
    # 1. Load existing corpus metadata
    metadata_path = "vector_db/attack_eval/metadata.json"
    with open(metadata_path, "r") as f:
        existing_chunks = json.load(f)
    print(f"Existing chunks: {len(existing_chunks)}")
    
    # 2. Load Wikipedia documents
    wiki_path = "data/benign/wikipedia_docs/xstest_wikipedia_docs.json"
    with open(wiki_path, "r") as f:
        wiki_data = json.load(f)
    
    wiki_docs = [d for d in wiki_data['documents'] if d['status'] == 'found']
    print(f"Wikipedia docs found: {len(wiki_docs)}")
    
    # 3. Create new chunks from Wikipedia docs
    new_chunks = []
    
    for i, doc in enumerate(wiki_docs):
        new_chunks.append({
            "chunk_id": f"wiki_xstest_{i}",
            "doc_id": f"wikipedia_xstest_{doc['query_id']}",
            "chunk_index": 0,
            "total_chunks": 1,
            "title": doc['wiki_title'],
            "source": "wikipedia_xstest",
            "trust_score": 1.0,
            "url": doc['wiki_url'],
            "text": doc['content'],
            "xstest_query_id": doc['query_id'],
        })
    
    print(f"New chunks to add: {len(new_chunks)}")
    
    # 4. Merge and save updated metadata
    all_chunks = existing_chunks + new_chunks
    
    # Backup existing metadata
    import shutil
    backup_path = metadata_path + ".backup"
    shutil.copy2(metadata_path, backup_path)
    print(f"Backed up metadata to: {backup_path}")
    
    with open(metadata_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Updated metadata: {len(all_chunks)} total chunks")
    
    # 5. Rebuild FAISS index with all chunks
    print("\nRebuilding FAISS index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dim = model.get_sentence_embedding_dimension()
    
    texts = [c['text'] for c in all_chunks]
    batch_size = 256
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(emb)
        if (i // batch_size) % 20 == 0:
            print(f"  Processed {i}/{len(texts)} chunks...")
    
    matrix = np.vstack(all_embeddings).astype("float32")
    
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    
    index_path = "vector_db/attack_eval/faiss.index"
    faiss.write_index(index, index_path)
    print(f"FAISS index rebuilt: {index.ntotal} vectors, dim={dim}")
    
    # 6. Verify: compute similarity for a few benign queries
    print("\n" + "=" * 70)
    print("VERIFICATION: Benign Query Similarity (Wikipedia docs)")
    print("=" * 70)
    
    loader = BenignQueryLoader()
    queries = loader.get_all_queries()[:10]
    
    # Find Wikipedia-sourced chunks
    wiki_chunk_texts = [c['text'] for c in all_chunks if c.get('source') == 'wikipedia_xstest']
    wiki_chunk_embs = model.encode(wiki_chunk_texts, normalize_embeddings=True)
    
    for q in queries:
        q_emb = model.encode([q['query']], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, wiki_chunk_embs)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        # Also check against the synthetic_xstest docs
        synth_texts = [c['text'] for c in all_chunks if c.get('source') == 'synthetic_xstest']
        synth_embs = model.encode(synth_texts, normalize_embeddings=True)
        synth_sims = cosine_similarity(q_emb, synth_embs)[0]
        best_synth = max(synth_sims)
        
        print(f"\n  Query: {q['query'][:60]}...")
        print(f"    Wikipedia best sim: {best_sim:.4f}")
        print(f"    Synthetic best sim: {best_synth:.4f}")
        print(f"    Improvement: +{(best_sim - best_synth):.4f}")
    
    print("\n  Done!")


if __name__ == '__main__':
    main()
