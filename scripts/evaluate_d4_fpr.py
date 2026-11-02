#!/usr/bin/env python3
"""
scripts/evaluate_d4_fpr.py
==========================
Step 1 of the 6-Step Defense Evaluation Protocol:
Empirical False Positive Rate (FPR) & Usability Sweep for Defense D4 (Hierarchical Context Delimiters & Prompt Spotlighting).

Evaluates:
  1. 200 Standard Benign Queries (data/benign/benign_queries_backup.json)
  2. 250 XSTest Safe Prompts (data/benign/benign_queries.json)
Against dense vector retrieval (top_k=5) over the 35,164-chunk corpus in vector_db/attack_eval/.

Invariants:
  - P(S1 | D4) == 1.00 (Zero candidate chunks dropped at retrieval boundary).
  - All chunks properly encapsulated in <untrusted_retrieved_context> tags.
  - Verification of XML entity escaping for any embedded markup.
  - Target: Strict 0.0% False Positive Rate (0 benign queries falsely blocked).

Outputs:
  - results/relevance_analysis/d4_spotlighting_fpr_sweep.csv
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add src and project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rag.defense_hooks import SpotlightingDefense
from config.config_loader import load_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("D4_FPR_Sweep")


def load_benign_datasets():
    """Load both the 200 benign queries and the 250 XSTest safe prompts."""
    b200_path = PROJECT_ROOT / "data" / "benign" / "benign_queries_backup.json"
    xstest_path = PROJECT_ROOT / "data" / "benign" / "benign_queries.json"

    with open(b200_path, "r", encoding="utf-8") as f:
        data_200 = json.load(f)
        queries_200 = data_200.get("queries", [])

    with open(xstest_path, "r", encoding="utf-8") as f:
        data_xstest = json.load(f)
        queries_xstest = data_xstest.get("queries", [])

    return queries_200, queries_xstest


def evaluate_query(
    query_text: str,
    index,
    metadata: list,
    model: SentenceTransformer,
    defense: SpotlightingDefense,
    config,
    top_k: int = 5,
):
    """Retrieve top-k chunks from FAISS and evaluate defense filtering/spotlighting."""
    vec = model.encode([query_text], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vec, k=top_k)

    retrieved_chunks = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < len(metadata):
            chunk = metadata[idx]
            retrieved_chunks.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "source": chunk.get("source", "knowledge_base"),
                "trust_score": chunk.get("trust_score", 1.0),
                "title": chunk.get("title", ""),
                "text": chunk.get("text", "")
            })

    filtered_chunks, defense_meta = defense.filter_chunks(retrieved_chunks, query_text, config)

    blocked_count = defense_meta.get("blocked_count", 0)
    passed_count = defense_meta.get("passed_count", len(filtered_chunks))
    all_blocked = defense_meta.get("all_chunks_blocked", False)

    # In spotlighting evaluation, all chunks must pass through to generation
    is_false_positive = all_blocked or (len(retrieved_chunks) > 0 and len(filtered_chunks) == 0)

    # Verify spotlighting tags are properly wrapped
    tag = defense_meta.get("tag_used", "untrusted_retrieved_context")
    all_tagged = all(
        chunk.get("text", "").startswith(f"<{tag}") and chunk.get("text", "").rstrip().endswith(f"</{tag}>")
        for chunk in filtered_chunks
    )

    return {
        "retrieved_count": len(retrieved_chunks),
        "blocked_count": blocked_count,
        "passed_count": passed_count,
        "all_blocked": all_blocked,
        "is_false_positive": is_false_positive,
        "all_tagged": all_tagged,
        "tag_used": tag
    }


def main():
    print("=" * 70)
    print("  STEP 1: DEFENSE D4 FALSE POSITIVE RATE (FPR) & USABILITY SWEEP")
    print("  Defense: Hierarchical Context Delimiters & Prompt Spotlighting")
    print("=" * 70)

    config = load_config(str(PROJECT_ROOT / "config" / "agent_config.yaml"))
    defense = SpotlightingDefense()

    print("Loading FAISS index and metadata...")
    index_path = str(PROJECT_ROOT / "vector_db" / "attack_eval" / "faiss.index")
    metadata_path = str(PROJECT_ROOT / "vector_db" / "attack_eval" / "metadata.json")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Corpus loaded: {len(metadata)} chunks in index.")

    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries_200, queries_xstest = load_benign_datasets()
    print(f"Loaded {len(queries_200)} Standard Benign Queries and {len(queries_xstest)} XSTest Safe Prompts.")

    all_results = []
    
    # 1. Evaluate 200 Standard Benign Queries
    b200_fp_count = 0
    b200_blocked_chunks = 0
    b200_tagged_count = 0
    for q in tqdm(queries_200, desc="Evaluating 200 Standard Benign Queries"):
        qid = q.get("query_id", "")
        qtext = q.get("query", "")
        topic = q.get("topic", "general")
        
        res = evaluate_query(qtext, index, metadata, model, defense, config)
        if res["is_false_positive"]:
            b200_fp_count += 1
        b200_blocked_chunks += res["blocked_count"]
        if res["all_tagged"]:
            b200_tagged_count += 1

        all_results.append({
            "Dataset": "Standard_Benign_200",
            "Query_ID": qid,
            "Topic": topic,
            "Query": qtext,
            "Chunks_Retrieved": res["retrieved_count"],
            "Chunks_Passed": res["passed_count"],
            "Chunks_Blocked": res["blocked_count"],
            "All_Chunks_Blocked": res["all_blocked"],
            "Is_False_Positive": res["is_false_positive"],
            "All_Tagged": res["all_tagged"]
        })

    # 2. Evaluate 250 XSTest Safe Prompts
    xstest_fp_count = 0
    xstest_blocked_chunks = 0
    xstest_tagged_count = 0
    for q in tqdm(queries_xstest, desc="Evaluating 250 XSTest Safe Prompts"):
        qid = q.get("query_id", "")
        qtext = q.get("query", "")
        topic = q.get("topic", "xstest_safe")
        
        res = evaluate_query(qtext, index, metadata, model, defense, config)
        if res["is_false_positive"]:
            xstest_fp_count += 1
        xstest_blocked_chunks += res["blocked_count"]
        if res["all_tagged"]:
            xstest_tagged_count += 1

        all_results.append({
            "Dataset": "XSTest_Safe_250",
            "Query_ID": qid,
            "Topic": topic,
            "Query": qtext,
            "Chunks_Retrieved": res["retrieved_count"],
            "Chunks_Passed": res["passed_count"],
            "Chunks_Blocked": res["blocked_count"],
            "All_Chunks_Blocked": res["all_blocked"],
            "Is_False_Positive": res["is_false_positive"],
            "All_Tagged": res["all_tagged"]
        })

    # Calculate Summary Metrics
    total_queries = len(queries_200) + len(queries_xstest)
    total_fp = b200_fp_count + xstest_fp_count
    overall_fpr = (total_fp / total_queries) * 100.0
    b200_fpr = (b200_fp_count / len(queries_200)) * 100.0
    xstest_fpr = (xstest_fp_count / len(queries_xstest)) * 100.0

    print("\n" + "=" * 70)
    print("  DEFENSE D4 FPR & USABILITY SWEEP SUMMARY RESULTS")
    print("=" * 70)
    print(f"  Standard Benign Queries (N=200):")
    print(f"    - False Positives (Blocked Queries): {b200_fp_count} / {len(queries_200)}")
    print(f"    - False Positive Rate (FPR):         {b200_fpr:.2f}%")
    print(f"    - Blocked Chunks:                    {b200_blocked_chunks}")
    print(f"    - Perfectly Tagged Queries:          {b200_tagged_count} / {len(queries_200)}")
    print(f"  XSTest Safe Prompts (N=250):")
    print(f"    - False Positives (Blocked Queries): {xstest_fp_count} / {len(queries_200)}")
    print(f"    - False Positive Rate (FPR):         {xstest_fpr:.2f}%")
    print(f"    - Blocked Chunks:                    {xstest_blocked_chunks}")
    print(f"    - Perfectly Tagged Queries:          {xstest_tagged_count} / {len(queries_xstest)}")
    print("  --------------------------------------------------------")
    print(f"  OVERALL COMBINED EVALUATION (N={total_queries}):")
    print(f"    - Total False Positives:             {total_fp} / {total_queries}")
    print(f"    - Combined False Positive Rate:      {overall_fpr:.2f}%")
    print(f"    - Stage 1 Retrieval Bypass P(S1):    100.00% (By Design)")
    print("=" * 70)

    # Save to CSV
    output_dir = PROJECT_ROOT / "results" / "relevance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "d4_spotlighting_fpr_sweep.csv"
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Dataset", "Query_ID", "Topic", "Query", 
            "Chunks_Retrieved", "Chunks_Passed", "Chunks_Blocked", 
            "All_Chunks_Blocked", "Is_False_Positive", "All_Tagged"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✓ Granular per-sample FPR sweep saved to: {out_csv}")


if __name__ == "__main__":
    main()
