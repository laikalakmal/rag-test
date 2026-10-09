"""
analyze_adaptive_results.py
---------------------------
Analyzes the evaluation report of adaptive prompt injection attacks compared to
the static baseline. Computes:
  1. Overall ASR Delta (ΔASR = Adaptive ASR - Static ASR)
  2. Stage 1 Retrieval Filter Bypass Rate (DBR / FBR)
  3. Stage 2 LLM Compliance Rate
  4. Technique Efficacy Breakdown (T1 Context Boundary vs T2 Instruction Laundering vs T3 Trojan Citation)
  5. Category Breakdown (OVR vs EXF vs CTX)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def analyze_adaptive_report(report_path: str, static_baseline_path: str = None):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    attack_eval = report.get("attack_evaluation", {})
    details = attack_eval.get("details", [])

    if not details:
        print("No evaluation details found in report.")
        return

    total = len(details)
    valid_samples = [d for d in details if not d.get("error")]
    
    # Stage 1: Filter Bypass
    stage1_bypassed = [d for d in valid_samples if d.get("defense_metadata", {}).get("injected_chunk_bypassed", False) or d.get("defense_bypassed", False) or not d.get("defense_blocked", False)]
    # Stage 2 & Final Success: LLM Judge Compromised
    stage2_success = [d for d in valid_samples if (d.get("llm_judge", {}).get("verdict") is True or d.get("success") is True or d.get("heuristic_success") is True)]

    overall_stage1_rate = (len(stage1_bypassed) / len(valid_samples) * 100) if valid_samples else 0.0
    overall_asr = (len(stage2_success) / len(valid_samples) * 100) if valid_samples else 0.0
    stage2_given_stage1 = (len(stage2_success) / len(stage1_bypassed) * 100) if stage1_bypassed else 0.0

    print("=" * 65)
    print(" ADAPTIVE INDIRECT PROMPT INJECTION EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Report File       : {os.path.basename(report_path)}")
    print(f"Target Model      : {report.get('agent_llm_model', 'unknown')}")
    print(f"Target Defense    : {report.get('defense_target', 'relevance_filtering')}")
    print(f"Total Evaluated   : {total} (Valid: {len(valid_samples)})")
    print("-" * 65)
    print(f"Stage 1 Filter Bypass Rate : {overall_stage1_rate:.2f}% ({len(stage1_bypassed)}/{len(valid_samples)})")
    print(f"Stage 2 LLM Compliance Rate: {stage2_given_stage1:.2f}% ({len(stage2_success)}/{len(stage1_bypassed)})")
    print(f"Final Attack Success (ASR) : {overall_asr:.2f}% ({len(stage2_success)}/{len(valid_samples)})")
    print("=" * 65)

    # Technique breakdown (T1, T2, T3)
    tech_stats = defaultdict(lambda: {"total": 0, "stage1_pass": 0, "success": 0})
    for d in valid_samples:
        attack_id = d.get("attack_id", "")
        # Extract technique e.g. from ADAPT_OVR_001_T1
        tech = "Unknown"
        for t in ["T1", "T2", "T3"]:
            if f"_{t}" in attack_id:
                tech = t
                break
        tech_stats[tech]["total"] += 1
        if d.get("defense_metadata", {}).get("injected_chunk_bypassed", False) or d.get("defense_bypassed", False) or not d.get("defense_blocked", False):
            tech_stats[tech]["stage1_pass"] += 1
        if d.get("llm_judge", {}).get("verdict") is True or d.get("success") is True or d.get("heuristic_success") is True:
            tech_stats[tech]["success"] += 1

    print("\nTECHNIQUE BREAKDOWN:")
    print(f"{'Technique':<12} | {'Total':<6} | {'Stage 1 Pass %':<16} | {'Stage 2 ASR %':<14}")
    print("-" * 55)
    for tech, stats in sorted(tech_stats.items()):
        s1_pct = (stats["stage1_pass"] / stats["total"] * 100) if stats["total"] else 0.0
        asr_pct = (stats["success"] / stats["total"] * 100) if stats["total"] else 0.0
        print(f"{tech:<12} | {stats['total']:<6} | {s1_pct:<15.1f}% | {asr_pct:<13.1f}%")

    # Static baseline comparison if provided
    if static_baseline_path and os.path.exists(static_baseline_path):
        with open(static_baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base_asr = base_data.get("attack_evaluation", {}).get("llm_judge", {}).get("asr_percentage", 0.0)
        delta_asr = overall_asr - base_asr
        print("\n" + "=" * 65)
        print(" COMPARATIVE GAIN OVER STATIC BASELINE")
        print("=" * 65)
        print(f"Static Baseline ASR : {base_asr:.2f}%")
        print(f"Adaptive ASR        : {overall_asr:.2f}%")
        print(f"ΔASR (Adaptive Gain): {delta_asr:+.2f} percentage points")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze adaptive attack results report.")
    parser.add_argument("report", type=str, help="Path to evaluation report JSON")
    parser.add_argument("--baseline", type=str, default=None, help="Path to static baseline report JSON")
    args = parser.parse_args()

    analyze_adaptive_report(args.report, args.baseline)
