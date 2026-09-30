"""
Threshold Recalibration Analysis (Option A)
============================================
Computes cosine similarity distributions for:
  1. Benign queries -> their best-matching corpus documents
  2. Benign queries -> poisoned attack documents (cross-scores)

Outputs:
  - results/relevance_analysis/benign_similarity_scores.csv
  - results/relevance_analysis/cross_similarity_scores.csv
  - results/relevance_analysis/threshold_sweep.csv
"""

import sys
import os
import csv
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from attacks.attack_loader import AttackLoader, BenignQueryLoader


def load_corpus_metadata():
    with open("vector_db/attack_eval/metadata.json", "r") as f:
        return json.load(f)


def compute_benign_scores(model, corpus_chunks):
    loader = BenignQueryLoader()
    queries = loader.get_all_queries()
    
    clean_chunks = [c for c in corpus_chunks if c.get('source') != 'attack_carrier_texts']
    print(f"  Clean corpus chunks (excluding attack carriers): {len(clean_chunks)}")
    
    clean_texts = [c['text'] for c in clean_chunks]
    print(f"  Encoding {len(clean_texts)} clean corpus texts...")
    corpus_embeddings = model.encode(clean_texts, batch_size=256, show_progress_bar=True,
                                      normalize_embeddings=True)
    
    results = []
    for q in tqdm(queries, desc="  Benign query scores"):
        query_text = q['query']
        query_emb = model.encode([query_text], normalize_embeddings=True)
        
        sims = cosine_similarity(query_emb, corpus_embeddings)[0]
        
        top_k = 5
        top_indices = np.argsort(sims)[-top_k:][::-1]
        top_scores = sims[top_indices]
        
        results.append({
            'type': 'benign',
            'id': q.get('query_id', ''),
            'topic': q.get('topic', ''),
            'query': query_text,
            'best_score': float(top_scores[0]),
            'avg_top5_score': float(np.mean(top_scores)),
            'best_doc_title': clean_chunks[top_indices[0]].get('title', ''),
            'best_doc_source': clean_chunks[top_indices[0]].get('source', ''),
        })
    
    return results


def compute_cross_scores(model, corpus_chunks):
    loader = BenignQueryLoader()
    queries = loader.get_all_queries()
    
    attack_loader = AttackLoader()
    attacks = attack_loader.get_all_attacks()
    
    poisoned_texts = []
    for a in attacks:
        full_doc = a.get('full_document', '')
        if full_doc:
            poisoned_texts.append(full_doc)
        elif a.get('carrier_text'):
            poisoned_texts.append(a['carrier_text'])
    
    print(f"  Encoding {len(poisoned_texts)} poisoned documents...")
    poisoned_embeddings = model.encode(poisoned_texts, batch_size=256,
                                        show_progress_bar=True,
                                        normalize_embeddings=True)
    
    results = []
    sample_queries = queries[:50]
    
    for q in tqdm(sample_queries, desc="  Cross-score analysis"):
        query_emb = model.encode([q['query']], normalize_embeddings=True)
        sims = cosine_similarity(query_emb, poisoned_embeddings)[0]
        
        results.append({
            'type': 'cross_score',
            'id': q.get('query_id', ''),
            'topic': q.get('topic', ''),
            'query': q['query'],
            'max_poisoned_sim': float(np.max(sims)),
            'mean_poisoned_sim': float(np.mean(sims)),
            'median_poisoned_sim': float(np.median(sims)),
        })
    
    return results


def threshold_sweep(benign_results, cross_results):
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    sweep_results = []
    for t in thresholds:
        blocked_benign = sum(1 for r in benign_results if r['best_score'] < t)
        total_benign = len(benign_results)
        fpr = blocked_benign / total_benign if total_benign > 0 else 0
        
        attacks_passing = sum(1 for r in cross_results if r['max_poisoned_sim'] >= t)
        total_attacks = len(cross_results)
        attack_pass_rate = attacks_passing / total_attacks if total_attacks > 0 else 0
        
        security = 1 - attack_pass_rate
        usability = 1 - fpr
        youden_j = security + usability - 1
        
        sweep_results.append({
            'threshold': t,
            'fpr': round(fpr, 4),
            'fpr_pct': round(fpr * 100, 2),
            'attack_pass_rate': round(attack_pass_rate, 4),
            'attack_pass_rate_pct': round(attack_pass_rate * 100, 2),
            'security': round(security, 4),
            'usability': round(usability, 4),
            'youden_j': round(youden_j, 4),
            'blocked_benign': blocked_benign,
            'total_benign': total_benign,
            'attacks_passing': attacks_passing,
            'total_attacks': total_attacks,
        })
    
    return sweep_results


def main():
    print("=" * 70)
    print("THRESHOLD RECALIBRATION ANALYSIS")
    print("=" * 70)
    
    print("\n[1/5] Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\n[2/5] Loading corpus...")
    corpus_chunks = load_corpus_metadata()
    print(f"  Total corpus chunks: {len(corpus_chunks)}")
    
    print("\n[3/5] Computing benign query similarity scores...")
    benign_results = compute_benign_scores(model, corpus_chunks)
    
    print("\n[4/5] Computing cross-score analysis (benign queries vs poisoned docs)...")
    cross_results = compute_cross_scores(model, corpus_chunks)
    
    print("\n[5/5] Running threshold sweep...")
    sweep = threshold_sweep(benign_results, cross_results)
    
    os.makedirs('results/relevance_analysis', exist_ok=True)
    
    benign_csv = 'results/relevance_analysis/benign_similarity_scores.csv'
    with open(benign_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'id', 'topic', 'query', 'best_score', 'avg_top5_score',
            'best_doc_title', 'best_doc_source'
        ])
        writer.writeheader()
        writer.writerows(benign_results)
    print(f"\n  Saved benign scores: {benign_csv}")
    
    cross_csv = 'results/relevance_analysis/cross_similarity_scores.csv'
    with open(cross_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'id', 'topic', 'query', 'max_poisoned_sim',
            'mean_poisoned_sim', 'median_poisoned_sim'
        ])
        writer.writeheader()
        writer.writerows(cross_results)
    print(f"  Saved cross-scores: {cross_csv}")
    
    sweep_csv = 'results/relevance_analysis/threshold_sweep.csv'
    with open(sweep_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'fpr', 'fpr_pct', 'attack_pass_rate',
            'attack_pass_rate_pct', 'security', 'usability', 'youden_j',
            'blocked_benign', 'total_benign', 'attacks_passing', 'total_attacks'
        ])
        writer.writeheader()
        writer.writerows(sweep)
    print(f"  Saved threshold sweep: {sweep_csv}")
    
    print("\n" + "=" * 70)
    print("BENIGN QUERY SIMILARITY DISTRIBUTION")
    print("=" * 70)
    benign_scores = [r['best_score'] for r in benign_results]
    print(f"  Count:  {len(benign_scores)}")
    print(f"  Min:    {min(benign_scores):.4f}")
    print(f"  Max:    {max(benign_scores):.4f}")
    print(f"  Mean:   {np.mean(benign_scores):.4f}")
    print(f"  Median: {np.median(benign_scores):.4f}")
    print(f"  Std:    {np.std(benign_scores):.4f}")
    
    print("\n" + "=" * 70)
    print("CROSS-SCORE DISTRIBUTION (Benign Query -> Poisoned Doc)")
    print("=" * 70)
    cross_max_scores = [r['max_poisoned_sim'] for r in cross_results]
    print(f"  Count:  {len(cross_max_scores)}")
    print(f"  Min:    {min(cross_max_scores):.4f}")
    print(f"  Max:    {max(cross_max_scores):.4f}")
    print(f"  Mean:   {np.mean(cross_max_scores):.4f}")
    print(f"  Median: {np.median(cross_max_scores):.4f}")
    
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Threshold':>10} {'FPR%':>8} {'Attack Pass%':>14} {'Security':>10} {'Usability':>10} {'Youden J':>10}")
    print("-" * 64)
    best_j = max(sweep, key=lambda x: x['youden_j'])
    for s in sweep:
        marker = " <-- OPTIMAL" if s['threshold'] == best_j['threshold'] else ""
        print(f"{s['threshold']:>10.2f} {s['fpr_pct']:>7.1f}% {s['attack_pass_rate_pct']:>13.1f}% "
              f"{s['security']:>10.4f} {s['usability']:>10.4f} {s['youden_j']:>10.4f}{marker}")
    
    print(f"\n  RECOMMENDED THRESHOLD: {best_j['threshold']}")
    print(f"  At this threshold: FPR={best_j['fpr_pct']}%, Attack Pass Rate={best_j['attack_pass_rate_pct']}%")
    print(f"  Youden's J = {best_j['youden_j']}")


if __name__ == '__main__':
    main()
