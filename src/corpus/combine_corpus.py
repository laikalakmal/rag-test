#!/usr/bin/env python3
"""
Combine all raw corpus sources into a single JSON file.
Reads from data/raw/ and writes to data/processed/complete_corpus.json
"""

import json
from pathlib import Path


# ── Source file locations ──────────────────────────────────────────────────
SOURCES = {
    "wikipedia": Path("data/raw/wikipedia/wikipedia_500.json"),
    "arxiv":     Path("data/raw/arxiv/arxiv_150.json"),
    "synthetic": Path("data/raw/synthetic/synthetic_50.json"),
}

OUTPUT_FILE = Path("data/processed/complete_corpus.json")
STATS_FILE  = Path("data/processed/corpus_stats.json")


def load_source(name, path):
    """Load one source file, report status."""
    if not path.exists():
        print(f"  ⚠️  {name}: file not found at {path} — skipping")
        return []

    with open(path, encoding="utf-8") as f:
        docs = json.load(f)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {name:12s}: {len(docs):4d} documents  ({size_mb:.2f} MB)")
    return docs


def normalise(doc, index):
    """
    Return a document with a guaranteed consistent set of fields.
    Any source-specific extra fields (authors, categories, etc.) are
    kept inside an 'extra' dict so the rest of the pipeline always
    sees the same top-level keys.
    """
    core_keys = {"id", "title", "text", "source", "trust_score", "url"}

    normalised = {
        "id":          f"doc_{index}",          # new sequential id
        "title":       doc.get("title", ""),
        "text":        doc.get("text", ""),
        "source":      doc.get("source", "unknown"),
        "trust_score": doc.get("trust_score", 0.5),
        "url":         doc.get("url", ""),
        # any source-specific fields land here
        "extra":       {k: v for k, v in doc.items() if k not in core_keys}
    }
    return normalised


def build_stats(documents):
    """Build a summary statistics dict for the corpus."""
    sources = {}
    total_chars = 0

    for doc in documents:
        src = doc["source"]
        sources[src] = sources.get(src, 0) + 1
        total_chars += len(doc["text"])

    return {
        "total_documents":    len(documents),
        "total_characters":   total_chars,
        "average_length":     round(total_chars / len(documents)) if documents else 0,
        "by_source":          sources,
    }


def combine_corpus():
    print("=" * 70)
    print("Corpus Combiner")
    print("=" * 70)
    print("\nReading source files...")

    # ── Load all sources ───────────────────────────────────────────────────
    all_raw = []
    for name, path in SOURCES.items():
        all_raw.extend(load_source(name, path))

    if not all_raw:
        print("\n❌ No documents loaded. Run the download scripts first:")
        print("     python src/corpus/download_wikipedia.py")
        print("     python src/corpus/download_arxiv.py")
        print("     python src/corpus/create_synthetic.py")
        return None

    # ── Normalise ──────────────────────────────────────────────────────────
    print(f"\n[2/3] Normalising {len(all_raw)} documents...")
    documents = [normalise(doc, i + 1) for i, doc in enumerate(all_raw)]
    print(f"      ✓ All documents share the same field structure")

    # ── Save corpus ────────────────────────────────────────────────────────
    print(f"\n[3/3] Saving...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"      ✓ Corpus  → {OUTPUT_FILE}  ({size_mb:.2f} MB)")

    # ── Save stats ─────────────────────────────────────────────────────────
    stats = build_stats(documents)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"      ✓ Stats   → {STATS_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nTotal documents : {stats['total_documents']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average length  : {stats['average_length']:,} chars per document")
    print(f"\nBreakdown by source:")
    for src, count in stats["by_source"].items():
        pct = count / stats["total_documents"] * 100
        print(f"  {src:12s}: {count:4d}  ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("Next step: python src/corpus/chunk_documents.py")
    print("=" * 70)

    return documents


if __name__ == "__main__":
    print("\nThis script combines all raw corpus files into one.")
    print("Input  : data/raw/wikipedia/, data/raw/arxiv/, data/raw/synthetic/")
    print("Output : data/processed/complete_corpus.json")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != "y":
        print("\n❌ Cancelled")
        exit(0)

    try:
        result = combine_corpus()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
