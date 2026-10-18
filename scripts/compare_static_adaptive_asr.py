#!/usr/bin/env python3
"""
compare_static_adaptive_asr.py
Formal Decomposed Two-Stage Comparison between Static and Adaptive Indirect Prompt Injection
across 4 Target Models: mistral:7b, llama3:latest, gemma:7b, vicuna:7b.

Decomposition:
  ASR = P(S1) * P(S2 | S1)
  Where:
    P(S1)          = Stage 1 Filter Bypass Rate
    P(S2 | S1)     = Stage 2 Conditional LLM Compliance Rate given filter bypass
    ASR            = End-to-End Attack Success Rate

Evaluates across two scopes:
  1. "All": Entire static dataset (Manual + BIPIA) vs. entire adaptive benchmark (100 samples)
  2. "Custom_Matched": 50 Custom attack payloads that share exact parent attack IDs
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StaticAdaptiveComparison")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_CSV = PROJECT_ROOT / "results" / "human_reviewed_per_sample.csv"
ADAPTIVE_CSV = PROJECT_ROOT / "results" / "adaptive_human_reviewed_per_sample.csv"
BENCHMARK_JSON = PROJECT_ROOT / "data" / "attacks" / "formal_benchmark_100.json"
OUTPUT_CSV = PROJECT_ROOT / "results" / "static_vs_adaptive_comparison.csv"

TARGET_MODELS = ["mistral:7b", "llama3:latest", "gemma:7b", "vicuna:7b"]


def load_custom_parent_ids(benchmark_path: Path) -> set:
    """Loads the set of parent_attack_ids for the 50 Custom attacks in the formal benchmark."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    attacks = data if isinstance(data, list) else data.get("attacks", [])
    custom_parents = set(
        a["parent_attack_id"]
        for a in attacks
        if not str(a.get("parent_attack_id", "")).startswith("EXT")
    )
    logger.info(f"Loaded {len(custom_parents)} custom parent attack IDs from benchmark.")
    return custom_parents


def compute_static_metrics(df_static: pd.DataFrame, model: str, custom_parent_ids: set, scope: str) -> Dict[str, Any]:
    """Computes Stage 1, Stage 2, and End-to-End metrics for static evaluations."""
    m_df = df_static[df_static["Model"] == model].copy()
    if scope == "Custom_Matched":
        m_df = m_df[m_df["Attack_ID"].isin(custom_parent_ids)]

    if len(m_df) == 0:
        return {
            "Static_N_Runs": 0,
            "Static_N_Unique_Attacks": 0,
            "Static_P_S1_%": 0.0,
            "Static_Bypassed_Count": 0,
            "Static_P_S2_given_S1_%": 0.0,
            "Static_Success_Count": 0,
            "Static_ASR_%": 0.0,
        }

    # Deduplicate by Attack_ID to get unique-attack-level conservative metric:
    # If ANY run succeeded, attack succeeded; if ANY run bypassed filter, attack bypassed.
    dedup = m_df.groupby("Attack_ID").agg({
        "Injected_Chunk_Bypassed": "max",
        "Final_Attack_Success": "max",
    }).reset_index()

    n_unique = len(dedup)
    n_bypassed = int(dedup["Injected_Chunk_Bypassed"].sum())
    p_s1 = (n_bypassed / n_unique) * 100 if n_unique > 0 else 0.0

    bypassed_attacks = dedup[dedup["Injected_Chunk_Bypassed"] == True]
    n_success = int(bypassed_attacks["Final_Attack_Success"].sum())
    p_s2_given_s1 = (n_success / n_bypassed) * 100 if n_bypassed > 0 else 0.0

    # End-to-end ASR = total successful unique attacks / total unique attacks
    total_success = int(dedup["Final_Attack_Success"].sum())
    asr = (total_success / n_unique) * 100 if n_unique > 0 else 0.0

    return {
        "Static_N_Runs": len(m_df),
        "Static_N_Unique_Attacks": n_unique,
        "Static_P_S1_%": round(p_s1, 2),
        "Static_Bypassed_Count": n_bypassed,
        "Static_P_S2_given_S1_%": round(p_s2_given_s1, 2),
        "Static_Success_Count": total_success,
        "Static_ASR_%": round(asr, 2),
    }


def compute_adaptive_metrics(df_adaptive: pd.DataFrame, model: str, benchmark_custom_ids: set, scope: str) -> Dict[str, Any]:
    """Computes Stage 1, Stage 2 (R1 and Final), and End-to-End metrics for adaptive evaluations."""
    m_df = df_adaptive[df_adaptive["Agent_Model"] == model].copy()
    if scope == "Custom_Matched":
        m_df = m_df[m_df["Attack_ID"].isin(benchmark_custom_ids)]

    n_total = len(m_df)
    if n_total == 0:
        return {}

    # Round 1
    r1_bypassed = m_df[m_df["R1_Bypassed"] == True]
    n_r1_bypassed = len(r1_bypassed)
    p_s1_r1 = (n_r1_bypassed / n_total) * 100

    r1_success_given_s1 = r1_bypassed["R1_Human_Verdict"].apply(lambda x: str(x).lower() in ("true", "1")).sum()
    p_s2_r1_given_s1 = (r1_success_given_s1 / n_r1_bypassed) * 100 if n_r1_bypassed > 0 else 0.0
    r1_asr = (r1_success_given_s1 / n_total) * 100

    # Final State (Round 2 incorporated)
    r1_bypass_col = m_df["R1_Bypassed"].fillna(False).astype(bool)
    r2_bypass_col = m_df["R2_Bypassed"].fillna(False).astype(bool)
    any_bypassed = m_df[r1_bypass_col | r2_bypass_col]
    n_final_bypassed = len(any_bypassed)
    p_s1_final = (n_final_bypassed / n_total) * 100

    final_success = (m_df["Human_Outcome"] == "success").sum()
    final_success_given_s1 = (any_bypassed["Human_Outcome"] == "success").sum()
    p_s2_final_given_s1 = (final_success_given_s1 / n_final_bypassed) * 100 if n_final_bypassed > 0 else 0.0
    final_asr = (final_success / n_total) * 100

    return {
        "Adaptive_N": n_total,
        "Adaptive_P_S1_R1_%": round(p_s1_r1, 2),
        "Adaptive_P_S2_R1_given_S1_%": round(p_s2_r1_given_s1, 2),
        "Adaptive_ASR_R1_%": round(r1_asr, 2),
        "Adaptive_P_S1_Final_%": round(p_s1_final, 2),
        "Adaptive_P_S2_Final_given_S1_%": round(p_s2_final_given_s1, 2),
        "Adaptive_ASR_Final_%": round(final_asr, 2),
        "Adaptive_Feedback_Gain_pp": round(final_asr - r1_asr, 2),
    }


def main():
    logger.info("Starting Static vs. Adaptive ASR comparison analysis...")
    custom_parent_ids = load_custom_parent_ids(BENCHMARK_JSON)

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    attacks = bench_data if isinstance(bench_data, list) else bench_data.get("attacks", [])
    benchmark_custom_ids = set(
        a["attack_id"] for a in attacks if not str(a.get("parent_attack_id", "")).startswith("EXT")
    )

    # Load Static Data
    df_static = pd.read_csv(STATIC_CSV)
    df_static.columns = df_static.columns.str.strip()
    df_static = df_static[df_static["Is_Valid"] == "VALID"].copy()
    logger.info(f"Loaded {len(df_static)} valid static evaluation rows.")

    # Load Adaptive Data
    df_adaptive = pd.read_csv(ADAPTIVE_CSV)
    df_adaptive.columns = df_adaptive.columns.str.strip()
    df_adaptive = df_adaptive[df_adaptive["Is_Valid"] == "VALID"].copy()
    logger.info(f"Loaded {len(df_adaptive)} valid adaptive evaluation rows.")

    rows = []
    scopes = ["All", "Custom_Matched"]

    for scope in scopes:
        for model in TARGET_MODELS:
            s_metrics = compute_static_metrics(df_static, model, custom_parent_ids, scope)
            a_metrics = compute_adaptive_metrics(df_adaptive, model, benchmark_custom_ids, scope)

            delta_s1 = round(a_metrics.get("Adaptive_P_S1_Final_%", 0.0) - s_metrics["Static_P_S1_%"], 2)
            delta_s2 = round(a_metrics.get("Adaptive_P_S2_Final_given_S1_%", 0.0) - s_metrics["Static_P_S2_given_S1_%"], 2)
            delta_asr_final = round(a_metrics.get("Adaptive_ASR_Final_%", 0.0) - s_metrics["Static_ASR_%"], 2)
            delta_asr_r1 = round(a_metrics.get("Adaptive_ASR_R1_%", 0.0) - s_metrics["Static_ASR_%"], 2)

            row = {
                "Scope": scope,
                "Model": model,
                "Static_N_Unique_Attacks": s_metrics["Static_N_Unique_Attacks"],
                "Static_P_S1_%": s_metrics["Static_P_S1_%"],
                "Static_P_S2_given_S1_%": s_metrics["Static_P_S2_given_S1_%"],
                "Static_ASR_%": s_metrics["Static_ASR_%"],
                "Adaptive_N": a_metrics.get("Adaptive_N", 0),
                "Adaptive_P_S1_R1_%": a_metrics.get("Adaptive_P_S1_R1_%", 0.0),
                "Adaptive_P_S2_R1_given_S1_%": a_metrics.get("Adaptive_P_S2_R1_given_S1_%", 0.0),
                "Adaptive_ASR_R1_%": a_metrics.get("Adaptive_ASR_R1_%", 0.0),
                "Adaptive_P_S1_Final_%": a_metrics.get("Adaptive_P_S1_Final_%", 0.0),
                "Adaptive_P_S2_Final_given_S1_%": a_metrics.get("Adaptive_P_S2_Final_given_S1_%", 0.0),
                "Adaptive_ASR_Final_%": a_metrics.get("Adaptive_ASR_Final_%", 0.0),
                "Adaptive_Feedback_Gain_pp": a_metrics.get("Adaptive_Feedback_Gain_pp", 0.0),
                "Delta_P_S1_pp": delta_s1,
                "Delta_P_S2_pp": delta_s2,
                "Delta_ASR_Final_vs_Static_pp": delta_asr_final,
                "Delta_ASR_R1_vs_Static_pp": delta_asr_r1,
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved complete decomposed comparison table to {OUTPUT_CSV}")

    # Display to console
    print("\n" + "=" * 115)
    print("      FORMAL DECOMPOSED TWO-STAGE COMPARISON: STATIC VS. ADAPTIVE PROMPT INJECTION")
    print("      ASR = P(S1: Filter Bypass) x P(S2|S1: Model Compliance given Bypass)")
    print("=" * 115)

    for scope in scopes:
        print(f"\n>>> SCOPE: {scope.upper()} ({'Full Population' if scope == 'All' else '50 Matched Custom Payloads'})\n")
        sub = df_out[df_out["Scope"] == scope]
        cols = [
            "Model",
            "Static_P_S1_%",
            "Static_P_S2_given_S1_%",
            "Static_ASR_%",
            "Adaptive_P_S1_Final_%",
            "Adaptive_P_S2_Final_given_S1_%",
            "Adaptive_ASR_Final_%",
            "Delta_P_S1_pp",
            "Delta_P_S2_pp",
            "Delta_ASR_Final_vs_Static_pp",
        ]
        headers = [
            "Model",
            "Stat P(S1)",
            "Stat P(S2|S1)",
            "Stat ASR",
            "Adapt P(S1)",
            "Adapt P(S2|S1)",
            "Adapt ASR",
            "Δ P(S1)",
            "Δ P(S2)",
            "Δ ASR (Total)",
        ]
        print(sub[cols].to_string(index=False, header=headers))

    print("\n" + "=" * 115 + "\n")


if __name__ == "__main__":
    main()
