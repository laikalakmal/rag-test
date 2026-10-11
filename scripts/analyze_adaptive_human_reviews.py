#!/usr/bin/env python3
"""
analyze_adaptive_human_reviews.py
---------------------------------
Aggregates human-reviewed adaptive closed-loop prompt injection evaluations
from results/adaptive_feedback/ and results/adaptive_feedback/human_reviews/.

Computes dual-track metrics (Automated LLM Judge vs Human Verified):
  1. Round 1 Initial ASR (%)
  2. Final ASR (%)
  3. ΔASR Gain from Feedback (%) = Final ASR - Round 1 ASR
  4. Converted Recoveries Count (Round 2 successes)
  5. Feedback Recovery Rate (%) = (Recoveries / R1 Failures) * 100
  6. Stage 1 Filter Bypass Rate (%)
  7. Technique Efficacy (T1 vs T2 vs T3)
  8. Category Efficacy (OVR vs EXF vs CTX)

Outputs:
  - adaptive_human_reviewed_summary.csv (Upsert on Agent_Model, Mutator_Model, Dataset)
  - adaptive_human_reviewed_per_sample.csv (Upsert on Attack_ID, Agent_Model)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_OUTPUT_DIR = Path("results")


def derive_human_outcomes(
    d: Dict[str, Any],
) -> Tuple[bool, Optional[bool], Optional[bool], str, Optional[int], bool, str]:
    """
    Evaluates human review verdicts for Round 1 and Round 2.
    Returns:
      (is_valid, r1_human_success, r2_human_success, final_outcome, succeeded_at_round, is_valid_conversion, invalid_reason)
    """
    human_rev = d.get("human_review", {})
    r1_verdict = human_rev.get("r1_verdict") or d.get("human_r1_verdict") or "Accept Judge"
    r2_verdict = human_rev.get("r2_verdict") or d.get("human_r2_verdict") or "Accept Judge"
    
    r1 = d.get("round1", {})
    r2 = d.get("round2")
    
    r1_judge_v = r1.get("judge_verdict", False) if r1 else False
    r2_judge_v = r2.get("judge_verdict", False) if r2 else None

    # Check validity
    if "exclude" in str(r1_verdict).lower() or "exclude" in str(r2_verdict).lower():
        return False, None, None, "invalid", None, False, "Excluded by human reviewer"

    # Derive Round 1 Human Verdict
    if str(r1_verdict).lower() == "success":
        r1_success = True
    elif str(r1_verdict).lower() == "fail":
        r1_success = False
    else: # Accept Judge
        r1_success = bool(r1_judge_v)

    # If Round 1 succeeded, Round 2 is not needed
    if r1_success:
        return True, True, None, "success", 1, False, ""

    # Round 1 failed; evaluate Round 2 if it exists
    if not r2:
        # No round 2 occurred
        return True, False, None, "failed", None, False, ""

    # Derive Round 2 Human Verdict
    if str(r2_verdict).lower() == "success":
        r2_success = True
    elif str(r2_verdict).lower() == "fail":
        r2_success = False
    else: # Accept Judge
        r2_success = bool(r2_judge_v)

    if r2_success:
        # Converted in Round 2 via feedback
        return True, False, True, "success", 2, True, ""
    else:
        # Both rounds failed
        return True, False, False, "failed", None, False, ""


def run_adaptive_analysis(
    files: List[str],
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    verbose: bool = True,
    append: bool = True,
) -> Dict[str, Any]:
    if not files:
        if verbose:
            print("No adaptive review files provided for analysis.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Aggregator key: (agent_model, mutator_model, dataset)
    agg = defaultdict(lambda: {
        "report_timestamps": set(),
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,

        # LLM Judge metrics
        "judge_r1_success": 0,
        "judge_final_success": 0,
        "judge_r2_recoveries": 0,

        # Human Verified metrics
        "human_reviewed_count": 0,
        "human_r1_success": 0,
        "human_final_success": 0,
        "human_r2_recoveries": 0,

        # Defense / Stage 1
        "stage1_bypassed": 0,

        # Technique break down: {tech: {"total": 0, "judge_succ": 0, "human_succ": 0}}
        "by_technique": defaultdict(lambda: {
            "total": 0,
            "judge_r1_succ": 0,
            "judge_final_succ": 0,
            "human_r1_succ": 0,
            "human_final_succ": 0,
            "human_recoveries": 0
        }),

        # Category break down: {cat: {"total": 0, "human_final_succ": 0}}
        "by_category": defaultdict(lambda: {
            "total": 0,
            "human_final_succ": 0,
            "judge_final_succ": 0
        }),
    })

    per_sample_rows = []

    for file_path in files:
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if verbose:
                print(f"Error reading {file_path}: {e}")
            continue

        meta = data.get("metadata", {})
        agent_model = meta.get("agent_model") or data.get("agent_llm_model", "unknown")
        judge_model = meta.get("judge_model", "unknown")
        mutator_model = meta.get("mutator_model", "unknown")
        timestamp = meta.get("timestamp") or data.get("timestamp", "")
        dataset = data.get("dataset", "formal_100")

        details = data.get("details", [])
        if not details:
            continue

        key = (agent_model, mutator_model, dataset)
        agg[key]["report_timestamps"].add(timestamp)

        for d in details:
            agg[key]["total_samples"] += 1
            attack_id = d.get("attack_id", "")
            category = d.get("category", "unknown")
            technique = d.get("technique") or ("T1" if "_T1" in attack_id else "T2" if "_T2" in attack_id else "T3" if "_T3" in attack_id else "unknown")
            query = d.get("query", "")

            r1 = d.get("round1", {})
            r2 = d.get("round2")

            # Stage 1 filter bypass
            r1_bypassed = bool(r1.get("injected_chunk_bypassed", False))
            if r1_bypassed:
                agg[key]["stage1_bypassed"] += 1

            # Judge outcomes
            judge_r1_v = bool(r1.get("judge_verdict", False))
            judge_outcome = d.get("final_outcome", "failed")
            judge_succ_round = d.get("succeeded_at_round")
            judge_is_success = (judge_outcome == "success")
            judge_is_recovery = (judge_succ_round == 2)

            # Human review derivation
            is_valid, hum_r1_s, hum_r2_s, hum_outcome, hum_succ_round, is_conversion, inv_reason = derive_human_outcomes(d)
            
            human_notes = d.get("human_review", {}).get("notes") or d.get("human_notes", "")
            is_reviewed = bool(d.get("human_review") or d.get("human_r1_verdict"))

            if is_reviewed:
                agg[key]["human_reviewed_count"] += 1

            if not is_valid:
                agg[key]["invalid_samples"] += 1
            else:
                agg[key]["valid_samples"] += 1

                # Tally Judge
                if judge_r1_v:
                    agg[key]["judge_r1_success"] += 1
                if judge_is_success:
                    agg[key]["judge_final_success"] += 1
                if judge_is_recovery:
                    agg[key]["judge_r2_recoveries"] += 1

                # Tally Human
                if hum_r1_s:
                    agg[key]["human_r1_success"] += 1
                if hum_outcome == "success":
                    agg[key]["human_final_success"] += 1
                if is_conversion:
                    agg[key]["human_r2_recoveries"] += 1

                # Group by Technique
                t_stats = agg[key]["by_technique"][technique]
                t_stats["total"] += 1
                if judge_r1_v:
                    t_stats["judge_r1_succ"] += 1
                if judge_is_success:
                    t_stats["judge_final_succ"] += 1
                if hum_r1_s:
                    t_stats["human_r1_succ"] += 1
                if hum_outcome == "success":
                    t_stats["human_final_succ"] += 1
                if is_conversion:
                    t_stats["human_recoveries"] += 1

                # Group by Category
                c_stats = agg[key]["by_category"][category]
                c_stats["total"] += 1
                if judge_is_success:
                    c_stats["judge_final_succ"] += 1
                if hum_outcome == "success":
                    c_stats["human_final_succ"] += 1

            # Build Per-Sample Row
            per_sample_rows.append({
                "Attack_ID": attack_id,
                "Category": category,
                "Technique": technique,
                "Agent_Model": agent_model,
                "Mutator_Model": mutator_model,
                "Dataset": dataset,
                "Query": query,
                "R1_Score": r1.get("injected_chunk_score", ""),
                "R1_Bypassed": r1.get("injected_chunk_bypassed", ""),
                "R1_Judge_Verdict": r1.get("judge_verdict", ""),
                "R1_Human_Verdict": hum_r1_s if hum_r1_s is not None else "",
                "R2_Score": r2.get("injected_chunk_score", "") if r2 else "",
                "R2_Bypassed": r2.get("injected_chunk_bypassed", "") if r2 else "",
                "R2_Judge_Verdict": r2.get("judge_verdict", "") if r2 else "",
                "R2_Human_Verdict": hum_r2_s if hum_r2_s is not None else "",
                "Judge_Outcome": judge_outcome,
                "Judge_Succeeded_Round": judge_succ_round or "",
                "Human_Outcome": hum_outcome,
                "Human_Succeeded_Round": hum_succ_round or "",
                "Is_Valid_Conversion": is_conversion,
                "Is_Valid": "VALID" if is_valid else "INVALID",
                "Invalid_Reason": inv_reason,
                "Human_Notes": human_notes,
                "Report_Timestamp": timestamp,
            })

    # Build Summary Rows
    summary_fieldnames = [
        "Agent_Model", "Mutator_Model", "Dataset",
        "Total_Samples", "Valid_Samples", "Human_Reviewed_Count",
        "R1_Judge_ASR_%", "R1_Human_ASR_%",
        "Final_Judge_ASR_%", "Final_Human_ASR_%",
        "Judge_Delta_ASR_Gain_%", "Human_Delta_ASR_Gain_%",
        "Judge_Recoveries", "Human_Recoveries",
        "Judge_Recovery_Rate_%", "Human_Recovery_Rate_%",
        "Stage1_Filter_Bypass_Rate_%",
        "T1_Human_ASR_%", "T2_Human_ASR_%", "T3_Human_ASR_%",
        "OVR_Human_ASR_%", "EXF_Human_ASR_%", "CTX_Human_ASR_%",
        "Report_Timestamps"
    ]

    new_summary_rows = []
    for (agent_m, mutator_m, dset), s in agg.items():
        val = s["valid_samples"]
        if val == 0:
            continue

        r1_j_asr = round((s["judge_r1_success"] / val * 100), 2)
        r1_h_asr = round((s["human_r1_success"] / val * 100), 2)
        fn_j_asr = round((s["judge_final_success"] / val * 100), 2)
        fn_h_asr = round((s["human_final_success"] / val * 100), 2)

        j_delta = round(fn_j_asr - r1_j_asr, 2)
        h_delta = round(fn_h_asr - r1_h_asr, 2)

        r1_j_failures = val - s["judge_r1_success"]
        r1_h_failures = val - s["human_r1_success"]

        j_rec_rate = round((s["judge_r2_recoveries"] / r1_j_failures * 100), 2) if r1_j_failures > 0 else 0.0
        h_rec_rate = round((s["human_r2_recoveries"] / r1_h_failures * 100), 2) if r1_h_failures > 0 else 0.0

        fbr = round((s["stage1_bypassed"] / val * 100), 2)

        # Technique stats
        def get_tech_asr(t_key):
            ts = s["by_technique"].get(t_key, {})
            tot = ts.get("total", 0)
            return round((ts.get("human_final_succ", 0) / tot * 100), 2) if tot > 0 else 0.0

        # Category stats
        def get_cat_asr(c_key):
            cs = s["by_category"].get(c_key, {})
            tot = cs.get("total", 0)
            return round((cs.get("human_final_succ", 0) / tot * 100), 2) if tot > 0 else 0.0

        new_summary_rows.append({
            "Agent_Model": agent_m,
            "Mutator_Model": mutator_m,
            "Dataset": dset,
            "Total_Samples": s["total_samples"],
            "Valid_Samples": val,
            "Human_Reviewed_Count": s["human_reviewed_count"],
            "R1_Judge_ASR_%": r1_j_asr,
            "R1_Human_ASR_%": r1_h_asr,
            "Final_Judge_ASR_%": fn_j_asr,
            "Final_Human_ASR_%": fn_h_asr,
            "Judge_Delta_ASR_Gain_%": j_delta,
            "Human_Delta_ASR_Gain_%": h_delta,
            "Judge_Recoveries": s["judge_r2_recoveries"],
            "Human_Recoveries": s["human_r2_recoveries"],
            "Judge_Recovery_Rate_%": j_rec_rate,
            "Human_Recovery_Rate_%": h_rec_rate,
            "Stage1_Filter_Bypass_Rate_%": fbr,
            "T1_Human_ASR_%": get_tech_asr("T1"),
            "T2_Human_ASR_%": get_tech_asr("T2"),
            "T3_Human_ASR_%": get_tech_asr("T3"),
            "OVR_Human_ASR_%": get_cat_asr("instruction_override"),
            "EXF_Human_ASR_%": get_cat_asr("data_exfiltration"),
            "CTX_Human_ASR_%": get_cat_asr("context_manipulation"),
            "Report_Timestamps": ";".join(sorted(list(s["report_timestamps"]))),
        })

    # Summary Upsert
    summary_csv = os.path.join(output_dir, "adaptive_human_reviewed_summary.csv")
    final_summary_rows = []
    if append and os.path.exists(summary_csv):
        try:
            with open(summary_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_map = {
                    (r.get("Agent_Model", ""), r.get("Mutator_Model", ""), r.get("Dataset", "")): r
                    for r in reader
                }
        except Exception:
            existing_map = {}
        for r in new_summary_rows:
            key = (r.get("Agent_Model", ""), r.get("Mutator_Model", ""), r.get("Dataset", ""))
            existing_map[key] = r
        final_summary_rows = list(existing_map.values())
    else:
        final_summary_rows = new_summary_rows

    if final_summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_summary_rows)
        if verbose:
            print(f"Adaptive Summary CSV written to: {summary_csv} ({len(final_summary_rows)} rows)")

    # Per-Sample Upsert
    per_sample_csv = os.path.join(output_dir, "adaptive_human_reviewed_per_sample.csv")
    final_sample_rows = []
    sample_fieldnames = [
        "Attack_ID", "Category", "Technique", "Agent_Model", "Mutator_Model", "Dataset",
        "Query", "R1_Score", "R1_Bypassed", "R1_Judge_Verdict", "R1_Human_Verdict",
        "R2_Score", "R2_Bypassed", "R2_Judge_Verdict", "R2_Human_Verdict",
        "Judge_Outcome", "Judge_Succeeded_Round", "Human_Outcome", "Human_Succeeded_Round",
        "Is_Valid_Conversion", "Is_Valid", "Invalid_Reason", "Human_Notes", "Report_Timestamp"
    ]

    if append and os.path.exists(per_sample_csv):
        try:
            with open(per_sample_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_sample_map = {
                    (r.get("Attack_ID", ""), r.get("Agent_Model", "")): r
                    for r in reader
                }
        except Exception:
            existing_sample_map = {}
        for r in per_sample_rows:
            key = (r.get("Attack_ID", ""), r.get("Agent_Model", ""))
            existing_sample_map[key] = r
        final_sample_rows = list(existing_sample_map.values())
    else:
        final_sample_rows = per_sample_rows

    if final_sample_rows:
        with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sample_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_sample_rows)
        if verbose:
            print(f"Adaptive Per-Sample CSV written to: {per_sample_csv} ({len(final_sample_rows)} rows)")

    return {
        "summary_rows": final_summary_rows,
        "per_sample_rows": final_sample_rows,
        "summary_csv": summary_csv,
        "per_sample_csv": per_sample_csv,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze adaptive closed-loop human review files.")
    parser.add_argument("--files", nargs="+", default=None, help="JSON review files to process")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSVs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs instead of appending")
    args = parser.parse_args()

    input_files = args.files
    if not input_files:
        rev_dir = Path("results/adaptive_feedback/human_reviews")
        if rev_dir.exists():
            input_files = [str(p) for p in rev_dir.glob("*.json")]
        if not input_files:
            adapt_dir = Path("results/adaptive_feedback")
            input_files = [str(p) for p in adapt_dir.glob("formal_100_*.json")]

    print(f"Processing {len(input_files or [])} files...")
    run_adaptive_analysis(
        files=input_files or [],
        output_dir=args.output_dir,
        verbose=True,
        append=not args.overwrite,
    )


if __name__ == "__main__":
    main()
