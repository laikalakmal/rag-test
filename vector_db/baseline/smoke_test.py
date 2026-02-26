import faiss, json
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1 — load the index
index = faiss.read_index("vector_db/baseline/faiss.index")
# print(index.ntotal)

metadata = json.load(open("vector_db/baseline/metadata.json"))
# print(len(metadata))          # should print 34427
# print(metadata[0].keys())     # shows the fields available

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query=""

while query=="":
    query = input("Enter your query:")


vec = model.encode([query], normalize_embeddings=True).astype("float32")
scores, ids = index.search(vec, k=5)  # top 5 results

for score, idx in zip(scores[0], ids[0]):
    m = metadata[idx]
    print(f"[{m['source']}] score={score:.3f}  {m['title']}")
    print(f"  {m['text'][:120]}\n")
