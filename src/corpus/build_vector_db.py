"""
build_vector_db.py
------------------
Embeds chunked documents using sentence-transformers and builds a FAISS
vector index. Output is saved to vector_db/baseline/ so the RAG bot can
load it at query time.

Run from project root:
    python3 src/corpus/build_vector_db.py

Input:  data/processed/chunked_corpus.json   (34,427 chunks)
Output: vector_db/baseline/faiss.index       (FAISS binary index)
        vector_db/baseline/metadata.json     (chunk metadata, parallel array)
        vector_db/baseline/build_stats.json  (embedding stats)
"""

import json
import time
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── paths ────────────────────────────────────────────────────────────────────
CHUNKS_FILE   = "data/processed/chunked_corpus.json"
OUTPUT_DIR    = "vector_db/baseline"
INDEX_FILE    = os.path.join(OUTPUT_DIR, "faiss.index")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
STATS_FILE    = os.path.join(OUTPUT_DIR, "build_stats.json")

# ── config ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
BATCH_SIZE      = 256   # chunks per embedding batch
SHOW_PROGRESS   = True  # print progress every N batches
PROGRESS_EVERY  = 10    # batches between progress prints


def load_chunks(path: str) -> list[dict]:
    print(f"Loading chunks from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks):,} chunks")
    return chunks


def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Embed all chunks and build a FAISS inner-product index.

    Returns
    -------
    index    : faiss.IndexFlatIP  — normalised vectors → cosine similarity
    metadata : list[dict]         — one entry per chunk (parallel to index)
    """
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    texts    = [c["text"] for c in chunks]
    total    = len(texts)
    n_batch  = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nEmbedding {total:,} chunks in {n_batch} batches of {BATCH_SIZE} …")
    all_embeddings = []
    t0 = time.time()

    for i in range(n_batch):
        batch_texts = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        embeddings  = model.encode(batch_texts, show_progress_bar=False,
                                   normalize_embeddings=True)
        all_embeddings.append(embeddings)

        if SHOW_PROGRESS and (i + 1) % PROGRESS_EVERY == 0:
            done = min((i + 1) * BATCH_SIZE, total)
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (total - done) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_batch}]  {done:,}/{total:,} chunks  "
                  f"({rate:.0f} chunks/s  ETA {eta:.0f}s)")

    matrix = np.vstack(all_embeddings).astype("float32")
    elapsed_total = time.time() - t0
    print(f"\nEmbedding complete: {total:,} vectors in {elapsed_total:.1f}s "
          f"({total/elapsed_total:.0f} chunks/s)")

    # Build FAISS index (Inner Product on normalised vectors = cosine similarity)
    print(f"\nBuilding FAISS IndexFlatIP ({dim}d) …")
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    print(f"  Index contains {index.ntotal:,} vectors")

    # Metadata: keep all fields except the text itself (text is in chunked_corpus)
    metadata = [
        {
            "chunk_id":    c["chunk_id"],
            "doc_id":      c["doc_id"],
            "chunk_index": c["chunk_index"],
            "total_chunks":c["total_chunks"],
            "title":       c["title"],
            "source":      c["source"],
            "trust_score": c["trust_score"],
            "url":         c.get("url", ""),
            "text":        c["text"],   # kept here for retrieval convenience
        }
        for c in chunks
    ]

    return index, metadata, dim, elapsed_total


def save_outputs(index, metadata, dim, elapsed, total):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FAISS binary index
    faiss.write_index(index, INDEX_FILE)
    print(f"\nSaved FAISS index  → {INDEX_FILE}")

    # Metadata (parallel array — position i matches vector i in the index)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    meta_mb = os.path.getsize(METADATA_FILE) / 1e6
    print(f"Saved metadata     → {METADATA_FILE}  ({meta_mb:.1f} MB)")

    # Build stats
    source_counts = {}
    for m in metadata:
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1

    stats = {
        "embedding_model":  EMBEDDING_MODEL,
        "embedding_dim":    dim,
        "total_vectors":    total,
        "batch_size":       BATCH_SIZE,
        "build_time_s":     round(elapsed, 2),
        "vectors_per_sec":  round(total / elapsed, 1),
        "source_breakdown": source_counts,
        "index_type":       "IndexFlatIP (cosine via L2-normalised vectors)",
        "index_file":       INDEX_FILE,
        "metadata_file":    METADATA_FILE,
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved build stats  → {STATS_FILE}")
    return stats


def verify_index(index, metadata):
    """Quick smoke-test: query a few vectors and check results look sensible."""
    print("\n── Smoke test ──────────────────────────────────────────────────")
    model = SentenceTransformer(EMBEDDING_MODEL)

    queries = [
        "What is a prompt injection attack?",
        "retrieval augmented generation",
        "machine learning security vulnerabilities",
    ]
    for q in queries:
        vec = model.encode([q], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(vec, k=3)
        print(f"\nQuery: '{q}'")
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
            m = metadata[idx]
            snippet = m["text"][:80].replace("\n", " ")
            print(f"  #{rank} [{m['source']}] score={score:.3f}  \"{snippet}…\"")


def main():
    print("=" * 60)
    print("  build_vector_db.py — RAG Vector Database Builder")
    print("=" * 60)

    chunks = load_chunks(CHUNKS_FILE)
    index, metadata, dim, elapsed = build_index(chunks)
    stats = save_outputs(index, metadata, dim, elapsed, len(chunks))

    print("\n── Summary ─────────────────────────────────────────────────────")
    for k, v in stats.items():
        if k != "source_breakdown":
            print(f"  {k:<22}: {v}")
    print("  source_breakdown     :", stats["source_breakdown"])

    verify_index(index, metadata)

    print("\n✓ Vector database built successfully.")
    print(f"  Load in RAG bot with:")
    print(f"    index    = faiss.read_index('{INDEX_FILE}')")
    print(f"    metadata = json.load(open('{METADATA_FILE}'))")


if __name__ == "__main__":
    main()
