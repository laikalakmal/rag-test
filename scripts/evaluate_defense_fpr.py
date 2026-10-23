#!/usr/bin/env python3
"""
scripts/evaluate_defense_fpr.py
================================
Step 1 of the 6-Step Defense Evaluation Protocol:
Empirical False Positive Rate (FPR) Sweep for Defense D2 (Pattern Matching Regex Blocklist).

Evaluates:
  1. 200 Standard Benign Queries (data/benign/benign_queries_backup.json)
  2. 250 XSTest Safe Prompts (data/benign/benign_queries.json)
Against dense vector retrieval (top_k=5) over the 35,164-chunk corpus in vector_db/attack_eval/.

Target:
  Strict 0.0% False Positive Rate (0 benign queries falsely blocked).

Outputs:
  - results/relevance_analysis/d2_pattern_matching_fpr_sweep.csv
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

from rag.defense_hooks import PatternMatchingDefense
from config.config_loader import load_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("D2_FPR_Sweep")


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
    defense: PatternMatchingDefense,
    config,
    top_k: int = 5,
):
    """Retrieve top-k chunks from FAISS and evaluate defense filtering."""
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

    # In benign evaluation, if all chunks are blocked, the agent terminates with defense_blocked (False Positive)
    is_false_positive = all_blocked or (len(retrieved_chunks) > 0 and len(filtered_chunks) == 0)

    matched_classes = set()
    matched_patterns = []
    for cs in defense_meta.get("chunk_scores", []):
        for mc in cs.get("matched_classes", []):
            matched_classes.add(mc)
        matched_patterns.extend(cs.get("matched_patterns", []))

    return {
        "retrieved_count": len(retrieved_chunks),
        "blocked_count": blocked_count,
        "passed_count": passed_count,
        "all_blocked": all_blocked,
        "is_false_positive": is_false_positive,
        "matched_classes": sorted(list(matched_classes)),
        "matched_patterns": matched_patterns,
    }


def main():
    print("=" * 70)
    print("  STEP 1: DEFENSE D2 FALSE POSITIVE RATE (FPR) SWEEP")
    print("=" * 70)

    config = load_config(str(PROJECT_ROOT / "config" / "agent_config.yaml"))
    defense = PatternMatchingDefense()

    print(f"Loading FAISS index and metadata...")
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
    for q in tqdm(queries_200, desc="Evaluating 200 Standard Benign Queries"):
        qid = q.get("query_id", "")
        qtext = q.get("query", "")
        topic = q.get("topic", "general")
        
        res = evaluate_query(qtext, index, metadata, model, defense, config)
        if res["is_false_positive"]:
            b200_fp_count += 1
        b200_blocked_chunks += res["blocked_count"]

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
            "Matched_Classes": "|".join(res["matched_classes"]),
            "Matched_Patterns": "; ".join(res["matched_patterns"][:3])
        })

    # 2. Evaluate 250 XSTest Safe Prompts
    xs_fp_count = 0
    xs_blocked_chunks = 0
    for q in tqdm(queries_xstest, desc="Evaluating 250 XSTest Safe Prompts"):
        qid = q.get("query_id", "")
        qtext = q.get("query", "")
        topic = q.get("topic", "xstest_edge_case")

        res = evaluate_query(qtext, index, metadata, model, defense, config)
        if res["is_false_positive"]:
            xs_fp_count += 1
        xs_blocked_chunks += res["blocked_count"]

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
            "Matched_Classes": "|".join(res["matched_classes"]),
            "Matched_Patterns": "; ".join(res["matched_patterns"][:3])
        })

    # Output Directory & File
    out_dir = PROJECT_ROOT / "results" / "relevance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "d2_pattern_matching_fpr_sweep.csv"

    fieldnames = [
        "Dataset", "Query_ID", "Topic", "Query",
        "Chunks_Retrieved", "Chunks_Passed", "Chunks_Blocked",
        "All_Chunks_Blocked", "Is_False_Positive",
        "Matched_Classes", "Matched_Patterns"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    b200_fpr = (b200_fp_count / len(queries_200)) * 100 if queries_200 else 0.0
    xs_fpr = (xs_fp_count / len(queries_xstest)) * 100 if queries_xstest else 0.0
    total_queries = len(queries_200) + len(queries_xstest)
    total_fps = b200_fp_count + xs_fp_count
    total_fpr = (total_fps / total_queries) * 100 if total_queries else 0.0

    print("\n" + "=" * 70)
    print("  FPR SWEEP RESULTS SUMMARY (Defense D2)")
    print("=" * 70)
    print(f"Standard Benign (N=200): False Positives = {b200_fp_count} | Blocked Chunks = {b200_blocked_chunks} | FPR = {b200_fpr:.2f}%")
    print(f"XSTest Safe     (N=250): False Positives = {xs_fp_count} | Blocked Chunks = {xs_blocked_chunks} | FPR = {xs_fpr:.2f}%")
    print(f"COMBINED TOTAL  (N=450): False Positives = {total_fps} | FPR = {total_fpr:.2f}%")
    print(f"Per-sample CSV saved to: {out_csv}")
    print("=" * 70)

    assert total_fps == 0, f"FAILED: Non-zero false positive rate detected ({total_fps} FPs)!"
    print("✓ Verification Passed: Defense D2 strictly satisfies the 0.0% FPR requirement.")


if __name__ == "__main__":
    main()
