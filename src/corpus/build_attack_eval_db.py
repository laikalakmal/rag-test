import json
import os
import sys
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_attacks():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from attacks.attack_loader import AttackLoader
    loader = AttackLoader(data_path="data/attacks/static_attacks.json")
    loader.load_external_dataset(dataset_name="bipia", num_samples=200)
    return loader.get_all_attacks()

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Load baseline chunks
    chunks_file = "data/processed/chunked_corpus.json"
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} baseline chunks.")
    
    # 2. Add attack carrier texts
    attacks = load_attacks()
    added = set()
    new_chunks = []
    
    # Handle string chunk_ids properly
    max_id = 0
    for c in chunks:
        try:
            cid = int(c["chunk_id"])
            if cid > max_id:
                max_id = cid
        except ValueError:
            pass
    
    for attack in attacks:
        carrier = attack.get("carrier_text")
        if not carrier or carrier in added:
            continue
        added.add(carrier)
        max_id += 1
        new_chunks.append({
            "chunk_id": max_id,
            "doc_id": f"attack_carrier_{max_id}",
            "chunk_index": 0,
            "total_chunks": 1,
            "title": "Reference Document",
            "source": "attack_carrier_texts",
            "trust_score": 1.0,
            "url": "",
            "text": carrier
        })
    
    chunks.extend(new_chunks)
    print(f"Added {len(new_chunks)} unique attack carrier texts as chunks. Total: {len(chunks)}.")
    
    # 3. Build Vector DB
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    
    out_dir = "vector_db/attack_eval"
    os.makedirs(out_dir, exist_ok=True)
    
    batch_size = 256
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    
    print(f"Embedding {len(texts)} chunks...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(texts)} chunks...")
            
    matrix = np.vstack(all_embeddings).astype("float32")
    
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(chunks, f, indent=2)
        
    print(f"Successfully built attack_eval vector DB in {out_dir}.")

if __name__ == "__main__":
    main()
