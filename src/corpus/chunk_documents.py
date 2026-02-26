#!/usr/bin/env python3
"""
Split corpus documents into smaller chunks for RAG retrieval.
Reads  : data/processed/complete_corpus.json
Writes : data/processed/chunked_corpus.json
         data/processed/chunk_stats.json
"""

import json
from pathlib import Path

INPUT_FILE  = Path("data/processed/complete_corpus.json")
OUTPUT_FILE = Path("data/processed/chunked_corpus.json")
STATS_FILE  = Path("data/processed/chunk_stats.json")

# ── Chunking config ────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # target characters per chunk  (≈ 100-120 words)
CHUNK_OVERLAP = 50    # overlap between chunks to preserve context


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks by character count.
    Tries to break at sentence boundaries ('. ') to avoid cutting mid-sentence.
    Falls back to word boundaries if no sentence boundary is found.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take whatever remains
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break at a sentence boundary within the last 100 chars of the window
        boundary = text.rfind('. ', start, end)
        if boundary != -1 and boundary > start + chunk_size // 2:
            end = boundary + 1          # include the period
        else:
            # Fall back to word boundary
            boundary = text.rfind(' ', start, end)
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Next chunk starts with overlap
        start = end - overlap

    return chunks


def chunk_documents(documents):
    """
    Chunk all documents. Each chunk inherits the parent document's metadata.
    Short documents (arXiv abstracts, synthetic) that are already under
    chunk_size are kept as a single chunk without splitting.
    """
    chunks = []
    chunk_id = 1

    for doc in documents:
        text = doc["text"].strip()
        if not text:
            continue

        # Documents shorter than chunk size → keep as one chunk
        if len(text) <= CHUNK_SIZE:
            chunks.append({
                "chunk_id":    f"chunk_{chunk_id}",
                "doc_id":      doc["id"],
                "chunk_index": 0,
                "total_chunks": 1,
                "text":        text,
                "title":       doc["title"],
                "source":      doc["source"],
                "trust_score": doc["trust_score"],
                "url":         doc["url"],
            })
            chunk_id += 1
            continue

        # Split longer documents
        text_chunks = split_text(text)

        for idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id":    f"chunk_{chunk_id}",
                "doc_id":      doc["id"],
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "text":        chunk_text,
                "title":       doc["title"],
                "source":      doc["source"],
                "trust_score": doc["trust_score"],
                "url":         doc["url"],
            })
            chunk_id += 1

    return chunks


def build_stats(documents, chunks):
    """Build statistics comparing before and after chunking."""
    by_source = {}
    for c in chunks:
        src = c["source"]
        if src not in by_source:
            by_source[src] = {"chunks": 0, "chars": 0}
        by_source[src]["chunks"] += 1
        by_source[src]["chars"]  += len(c["text"])

    avg_chunk_len = sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0

    return {
        "input_documents":    len(documents),
        "output_chunks":      len(chunks),
        "avg_chunks_per_doc": round(len(chunks) / len(documents), 1) if documents else 0,
        "avg_chunk_length":   round(avg_chunk_len),
        "chunk_size_setting": CHUNK_SIZE,
        "overlap_setting":    CHUNK_OVERLAP,
        "by_source":          by_source,
    }


def main():
    print("=" * 70)
    print("Document Chunker")
    print("=" * 70)
    print(f"\nChunk size : {CHUNK_SIZE} characters  (~100-120 words)")
    print(f"Overlap    : {CHUNK_OVERLAP} characters")
    print(f"Input      : {INPUT_FILE}")
    print(f"Output     : {OUTPUT_FILE}")

    # ── Load corpus ────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading corpus...")

    if not INPUT_FILE.exists():
        print(f"\n❌ {INPUT_FILE} not found.")
        print("   Run: python3 src/corpus/combine_corpus.py first")
        return None

    with open(INPUT_FILE, encoding="utf-8") as f:
        documents = json.load(f)

    print(f"      ✓ Loaded {len(documents)} documents")

    # ── Chunk ──────────────────────────────────────────────────────────────
    print(f"\n[2/3] Chunking...")
    chunks = chunk_documents(documents)
    print(f"      ✓ Created {len(chunks)} chunks from {len(documents)} documents")

    # ── Save ───────────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"      ✓ Chunks → {OUTPUT_FILE}  ({size_mb:.2f} MB)")

    stats = build_stats(documents, chunks)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"      ✓ Stats  → {STATS_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nDocuments  →  Chunks : {stats['input_documents']} → {stats['output_chunks']}")
    print(f"Avg chunks per doc  : {stats['avg_chunks_per_doc']}")
    print(f"Avg chunk length    : {stats['avg_chunk_length']} characters")
    print(f"\nBreakdown by source:")
    for src, info in stats["by_source"].items():
        avg = info["chars"] // info["chunks"] if info["chunks"] else 0
        print(f"  {src:12s}: {info['chunks']:5d} chunks  (avg {avg} chars)")

    # Show sample chunks from different sources
    print("\n" + "-" * 70)
    print("Sample Chunks (one per source):")
    print("-" * 70)
    shown = set()
    for chunk in chunks:
        if chunk["source"] not in shown:
            print(f"\n[{chunk['source'].upper()}]  {chunk['title'][:55]}")
            print(f"  chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}  |  {len(chunk['text'])} chars")
            print(f"  \"{chunk['text'][:140]}...\"")
            shown.add(chunk["source"])
        if len(shown) == 3:
            break

    print("\n" + "=" * 70)
    print("Next step: python3 src/corpus/build_vector_db.py")
    print("=" * 70)

    return chunks


if __name__ == "__main__":
    print("\nThis script splits documents into smaller chunks for RAG retrieval.")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != "y":
        print("\n❌ Cancelled")
        exit(0)

    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
