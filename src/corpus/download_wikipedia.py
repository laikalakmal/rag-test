#!/usr/bin/env python3
"""
Download Wikipedia articles for RAG corpus
Downloads clean article text from HuggingFace Wikipedia dataset
"""

import json
import os
from itertools import islice
from pathlib import Path

def download_wikipedia(num_articles=500):
    """
    Download Wikipedia articles using HuggingFace datasets
    
    Args:
        num_articles: Number of articles to download (default: 500)
    """
    print("="*70)
    print("Wikipedia Article Downloader")
    print("="*70)
    print(f"\nTarget: {num_articles} articles")
    print("Source: HuggingFace datasets (wikimedia/wikipedia preferred)")
    print("\nNote: First run may still download large metadata/cache, but streaming reduces startup cost")
    print("="*70)
    
    # Import required library
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n❌ ERROR: 'datasets' library not installed")
        print("\nInstall it with:")
        print("  pip install datasets")
        return None
    
    # Create output directory
    output_dir = Path("data/raw/wikipedia")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print(f"\n[1/3] Downloading Wikipedia dataset...")
    print("      This may take 10-20 minutes on first run...")
    
    try:
        dataset = None
        wikipedia_sources = [
            ("wikimedia/wikipedia", "20231101.en"),
            ("wikimedia/wikipedia", "20220301.en"),
            ("wikipedia", "20220301.en"),
        ]

        last_error = None
        for dataset_name, config_name in wikipedia_sources:
            try:
                # Streaming avoids materializing the full split and significantly improves first-run speed.
                dataset = load_dataset(
                    dataset_name,
                    config_name,
                    split="train",
                    streaming=True,
                    cache_dir="data/cache"
                )
                print(f"      ✓ Connected to dataset stream")
                print(f"      ✓ Using source: {dataset_name} ({config_name})")
                break
            except Exception as source_error:
                last_error = source_error

        if dataset is None:
            raise RuntimeError(f"Unable to load Wikipedia dataset from known sources: {last_error}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to download dataset")
        print(f"   {str(e)}")
        return None
    
    # Process articles
    print(f"\n[2/3] Processing articles...")
    
    documents = []
    for i, article in enumerate(islice(dataset, num_articles), 1):
        title = article["title"]
        text = article["text"]

        # Extract article data
        doc = {
            "id": f"wikipedia_{i}",
            "title": title,
            "text": text,
            "source": "wikipedia",
            "trust_score": 0.9,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        }
        documents.append(doc)
        
        # Show progress every 100 articles
        if i % 100 == 0:
            print(f"      Processed {i}/{num_articles} articles...")
    
    print(f"      ✓ Processed {len(documents)} articles")
    
    # Save to JSON
    print(f"\n[3/3] Saving to file...")
    
    output_file = output_dir / f"wikipedia_{num_articles}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print(f"      ✓ Saved to: {output_file}")
    print(f"      ✓ File size: {file_size_mb:.2f} MB")
    
    # Print summary
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nDownloaded: {len(documents)} Wikipedia articles")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Show sample articles
    print("\n" + "-"*70)
    print("Sample Articles (First 5):")
    print("-"*70)
    for i, doc in enumerate(documents[:5], 1):
        print(f"\n[{i}] {doc['title']}")
        print(f"    URL: {doc['url']}")
        print(f"    Text preview: {doc['text'][:100]}...")
    
    print("\n" + "="*70)
    print("Next step: python src/corpus/download_arxiv.py")
    print("="*70)
    
    return documents

if __name__ == "__main__":
    # Configuration
    NUM_ARTICLES = 500  # Change this to download more/less articles
    
    print("\n⚠️  IMPORTANT:")
    print(f"   This will download {NUM_ARTICLES} Wikipedia articles")
    print("   First run downloads ~2GB (cached for future use)")
    print("   Estimated time: 10-20 minutes (first run)")
    print()
    
    # Ask for confirmation
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Cancelled by user")
        exit(0)
    
    # Run download
    try:
        documents = download_wikipedia(num_articles=NUM_ARTICLES)
        
        if documents:
            print("\n✅ Wikipedia download complete!")
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
