#!/usr/bin/env python3
"""
Download arXiv paper abstracts for RAG corpus
Downloads from arXiv API: AI, ML, Security, NLP categories
"""

import json
from pathlib import Path


def download_arxiv(num_papers=150):
    """
    Download arXiv paper abstracts

    Args:
        num_papers: Number of papers to download (default: 150)
    """
    print("="*70)
    print("arXiv Paper Downloader")
    print("="*70)
    print(f"\nTarget: {num_papers} paper abstracts")
    print("Source: arXiv API (cs.AI, cs.LG, cs.CR, cs.CL)")
    print("="*70)

    try:
        import arxiv
    except ImportError:
        print("\n❌ ERROR: 'arxiv' library not installed")
        print("\nInstall it with:")
        print("  pip install arxiv")
        return None

    # Create output directory
    output_dir = Path("data/raw/arxiv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search arXiv
    print(f"\n[1/3] Searching arXiv...")
    print("      Categories: AI, Machine Learning, Security, NLP")

    try:
        search = arxiv.Search(
            query="cat:cs.AI OR cat:cs.LG OR cat:cs.CR OR cat:cs.CL",
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        # Fetch results
        print(f"\n[2/3] Downloading {num_papers} abstracts...")

        documents = []
        for i, result in enumerate(search.results(), 1):
            doc = {
                "id": f"arxiv_{i}",
                "title": result.title,
                "text": result.summary,
                "authors": [author.name for author in result.authors],
                "published": str(result.published.date()),
                "categories": result.categories,
                "url": result.entry_id,
                "source": "arxiv",
                "trust_score": 0.8
            }
            documents.append(doc)

            if i % 25 == 0:
                print(f"      Downloaded {i}/{num_papers} papers...")

        print(f"      ✓ Downloaded {len(documents)} abstracts")

    except Exception as e:
        print(f"\n❌ ERROR: Failed to fetch from arXiv")
        print(f"   {str(e)}")
        return None

    # Save to JSON
    print(f"\n[3/3] Saving to file...")

    output_file = output_dir / f"arxiv_{num_papers}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"      ✓ Saved to: {output_file}")
    print(f"      ✓ File size: {file_size_mb:.2f} MB")

    # Print summary
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nDownloaded: {len(documents)} arXiv abstracts")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Category breakdown
    category_counts = {}
    for doc in documents:
        for cat in doc["categories"]:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nTop categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"  {cat}: {count} papers")

    # Show sample papers
    print("\n" + "-"*70)
    print("Sample Papers (First 5):")
    print("-"*70)
    for i, doc in enumerate(documents[:5], 1):
        print(f"\n[{i}] {doc['title'][:70]}")
        print(f"    Authors: {', '.join(doc['authors'][:2])}{'...' if len(doc['authors']) > 2 else ''}")
        print(f"    Published: {doc['published']}")
        print(f"    Abstract: {doc['text'][:120]}...")

    print("\n" + "="*70)
    print("Next step: python src/corpus/create_synthetic.py")
    print("="*70)

    return documents


if __name__ == "__main__":
    # Configuration
    NUM_PAPERS = 150  # Change this to download more/less papers

    print("\n⚠️  IMPORTANT:")
    print(f"   This will download {NUM_PAPERS} arXiv paper abstracts")
    print("   Requires internet connection")
    print("   Estimated time: 2-5 minutes")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Cancelled by user")
        exit(0)

    try:
        documents = download_arxiv(num_papers=NUM_PAPERS)

        if documents:
            print("\n✅ arXiv download complete!")
            exit(0)
        else:
            print("\n❌ Download failed")
            exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
