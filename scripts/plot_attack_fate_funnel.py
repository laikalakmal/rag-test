#!/usr/bin/env python3
"""
plot_attack_fate_funnel.py
Publication-Grade Dual-Condition Attack Fate Funnel Diagram for a Specific Defense.
Visualizes how attacks progress through the two-stage kill chain:
  Stage 1: Pre-retrieval / Retrieval Defense Gate (Blocked vs. Passed)
  Stage 2: Model Compliance Gate (Defended vs. Breached)
Compares Static vs. Adaptive Indirect Prompt Injection across 4 Target Models.

Parameters:
  --defense: The defense mechanism being evaluated (default: 'relevance_filtering')
  --input: Path to the comparison CSV file
  --output: Output image path
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "results" / "static_vs_adaptive_comparison.csv"

TARGET_MODELS = ["mistral:7b", "llama3:latest", "gemma:7b", "vicuna:7b"]
MODEL_TITLES = {
    "mistral:7b": "Mistral-7B (Strong Alignment)",
    "llama3:latest": "Llama-3-8B (Strong Alignment)",
    "gemma:7b": "Gemma-7B (Intermediate Alignment)",
    "vicuna:7b": "Vicuna-7B (Conversational / Weak Alignment)",
}

DEFENSE_DISPLAY_NAMES = {
    "relevance_filtering": "Defense D1: Semantic Relevance Filtering (τ = 0.40)",
    "pattern_matching": "Defense D2: Pattern Matching Regex Blocklist",
    "trust_scoring": "Defense D3: Source Trust Scoring Filtering",
    "hierarchical_delimiters": "Defense D4: Hierarchical System Prompt Spotlighting",
}


def draw_funnel_panel(ax, model_name, s_data, a_data):
    """Draws a dual-condition stage-by-stage fate diagram for a single model."""
    ax.set_title(MODEL_TITLES[model_name], fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-5, 115)

    # Color palette
    c_submitted = "#7f8c8d"
    c_filter_blocked = "#bdc3c7"
    c_filter_pass = "#3498db"
    c_defended = "#95a5a6"
    c_breach_r1 = "#e67e22"
    c_breach_r2 = "#e74c3c"
    c_static_breach = "#d35400"

    # ------------------ STATIC CONDITION ------------------
    s_p_s1 = s_data["Static_P_S1_%"]
    s_blocked = 100.0 - s_p_s1
    s_asr = s_data["Static_ASR_%"]
    s_defended = max(0.0, s_p_s1 - s_asr)

    # Stage 0: Submitted
    ax.bar(0.0, 100.0, width=0.46, color=c_submitted, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.text(0.0, 50.0, "100%\nInit", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)

    # Stage 1: Filter
    ax.bar(0.8, s_blocked, bottom=s_p_s1, width=0.46, color=c_filter_blocked, alpha=0.8, edgecolor="black", linewidth=0.8)
    if s_blocked > 8:
        ax.text(0.8, s_p_s1 + s_blocked / 2, f"Blocked\n{s_blocked:.1f}%", ha="center", va="center", color="#2c3e50", fontsize=7.5, fontweight="bold")

    ax.bar(0.8, s_p_s1, bottom=0.0, width=0.46, color=c_filter_pass, alpha=0.9, edgecolor="black", linewidth=0.8)
    if s_p_s1 > 5:
        ax.text(0.8, s_p_s1 / 2, f"Pass\n{s_p_s1:.1f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
    else:
        ax.text(0.8, s_p_s1 + 2, f"{s_p_s1:.1f}%", ha="center", va="bottom", color="#1f618d", fontsize=7.0, fontweight="bold")

    # Stage 2: Outcome
    if s_defended > 0:
        ax.bar(1.6, s_defended, bottom=s_asr, width=0.46, color=c_defended, alpha=0.85, edgecolor="black", linewidth=0.8)
        if s_defended > 8:
            ax.text(1.6, s_asr + s_defended / 2, f"Defended\n{s_defended:.1f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
    if s_asr > 0:
        ax.bar(1.6, s_asr, bottom=0.0, width=0.46, color=c_static_breach, alpha=0.95, edgecolor="black", linewidth=0.8)
        ax.text(1.6, s_asr / 2 if s_asr > 4 else s_asr + 2, f"Breached\n{s_asr:.1f}%", ha="center", va="center" if s_asr > 4 else "bottom", color="white" if s_asr > 4 else "#b9770e", fontsize=7.0, fontweight="bold")
    else:
        ax.text(1.6, -3.5, "0.0% Breached", ha="center", va="top", color="#78281f", fontsize=7.0, fontweight="bold")

    # Connectors for Static
    ax.annotate("", xy=(0.54, max(s_p_s1, 10)), xytext=(0.26, 100), arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0, linestyle=":"))
    ax.annotate("", xy=(1.34, max(s_asr, 5)), xytext=(1.06, max(s_p_s1, 5)), arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0, linestyle=":"))

    # Label for Static Condition
    ax.text(0.8, 106.0, "STATIC INJECTION", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#7f8c8d")

    # Divider line
    ax.axvline(2.1, color="#bdc3c7", linestyle="--", linewidth=1.2, alpha=0.7)

    # ------------------ ADAPTIVE CONDITION ------------------
    a_p_s1 = a_data["Adaptive_P_S1_Final_%"]
    a_blocked = max(0.0, 100.0 - a_p_s1)
    a_r1_asr = a_data["Adaptive_ASR_R1_%"]
    a_final_asr = a_data["Adaptive_ASR_Final_%"]
    a_r2_gain = max(0.0, a_final_asr - a_r1_asr)
    a_defended = max(0.0, 100.0 - a_final_asr)

    # Stage 0: Submitted
    ax.bar(2.6, 100.0, width=0.46, color=c_submitted, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.text(2.6, 50.0, "100%\nInit", ha="center", va="center", color="white", fontweight="bold", fontsize=8.0)

    # Stage 1: Filter
    if a_blocked > 0:
        ax.bar(3.4, a_blocked, bottom=a_p_s1, width=0.46, color=c_filter_blocked, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax.bar(3.4, a_p_s1, bottom=0.0, width=0.46, color="#27ae60", alpha=0.9, edgecolor="black", linewidth=0.8)
    ax.text(3.4, a_p_s1 / 2, f"Pass\n{a_p_s1:.1f}%", ha="center", va="center", color="white", fontsize=8.0, fontweight="bold")

    # Stage 2: Outcome
    if a_defended > 0:
        ax.bar(4.2, a_defended, bottom=a_final_asr, width=0.46, color=c_defended, alpha=0.85, edgecolor="black", linewidth=0.8)
        if a_defended > 8:
            ax.text(4.2, a_final_asr + a_defended / 2, f"Defended\n{a_defended:.1f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

    # R2 Converted portion
    if a_r2_gain > 0:
        ax.bar(4.2, a_r2_gain, bottom=a_r1_asr, width=0.46, color=c_breach_r2, alpha=0.95, edgecolor="black", linewidth=0.8)
        if a_r2_gain > 5:
            ax.text(4.2, a_r1_asr + a_r2_gain / 2, f"R2 Adapt\n+{a_r2_gain:.1f}%", ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")

    # R1 Breach portion
    if a_r1_asr > 0:
        ax.bar(4.2, a_r1_asr, bottom=0.0, width=0.46, color=c_breach_r1, alpha=0.95, edgecolor="black", linewidth=0.8)
        if a_r1_asr > 5:
            ax.text(4.2, a_r1_asr / 2, f"R1 Breach\n{a_r1_asr:.1f}%", ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")

    # Total ASR badge above final bar
    ax.text(4.2, a_final_asr + 3, f"Final ASR: {a_final_asr:.1f}%", ha="center", va="bottom", color="#c0392b", fontsize=8.0, fontweight="bold")

    # Connectors for Adaptive
    ax.annotate("", xy=(3.14, a_p_s1), xytext=(2.86, 100), arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2, linestyle="-"))
    ax.annotate("", xy=(3.94, a_final_asr), xytext=(3.66, a_p_s1), arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2, linestyle="-"))

    # Label for Adaptive Condition
    ax.text(3.4, 106.0, "ADAPTIVE (T1-T3 + FEEDBACK)", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#27ae60")

    # Delta badge
    delta_asr = a_final_asr - s_asr
    ax.text(2.1, 114.0, f"Net ΔASR: +{delta_asr:.1f}pp", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c", lw=1.2),
            fontsize=8.5, fontweight="bold", color="#922b21")

    # Formatting
    ax.set_xticks([0.0, 0.8, 1.6, 2.6, 3.4, 4.2])
    ax.set_xticklabels(["Submitted", "Filter", "Outcome", "Submitted", "Filter", "Outcome"], fontsize=7.5)
    ax.set_ylabel("Attacks (% of Initial)", fontsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Attack Fate Funnel for a Specified Defense")
    parser.add_argument(
        "--defense",
        type=str,
        default="relevance_filtering",
        help="Name of the defense being evaluated (default: relevance_filtering)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input comparison CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defense_name = args.defense.strip()

    # Determine input path
    if args.input is not None:
        input_csv = args.input
    else:
        def_csv = PROJECT_ROOT / "results" / f"static_vs_adaptive_{defense_name}.csv"
        input_csv = def_csv if def_csv.exists() else DEFAULT_COMPARISON_CSV

    # Determine output path
    if args.output is not None:
        output_plot = args.output
    else:
        output_plot = PROJECT_ROOT / "results" / f"attack_fate_funnel_{defense_name}.png"

    df = pd.read_csv(input_csv)
    # Filter by defense if defense column exists
    if "Defense" in df.columns:
        df = df[df["Defense"] == defense_name]

    df_all = df[df["Scope"] == "All"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, model in enumerate(TARGET_MODELS):
        r, c = coords[idx]
        ax = axes[r, c]
        matching = df_all[df_all["Model"] == model]
        if len(matching) > 0:
            m_row = matching.iloc[0]
            draw_funnel_panel(ax, model, m_row, m_row)

    # Master title & global legend
    defense_header = DEFENSE_DISPLAY_NAMES.get(defense_name, f"Defense: {defense_name.replace('_', ' ').title()}")
    fig.suptitle(
        f"Attack Fate Progression: Static vs. Adaptive Prompt Injection Kill Chain\n"
        f"Evaluated Against {defense_header}\n"
        f"Stage 1 (Defense Gate) vs. Stage 2 (LLM Compliance Gate)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    legend_patches = [
        mpatches.Patch(color="#7f8c8d", label="Submitted Attacks (100%)"),
        mpatches.Patch(color="#bdc3c7", label="Blocked at Stage 1 (Defense Drop)"),
        mpatches.Patch(color="#3498db", label="Passed Stage 1 (Static)"),
        mpatches.Patch(color="#27ae60", label="Passed Stage 1 (Adaptive Camouflage ~100%)"),
        mpatches.Patch(color="#95a5a6", label="Defended at Stage 2 (LLM Refusal)"),
        mpatches.Patch(color="#e67e22", label="Breached in Round 1 (Initial Adaptive)"),
        mpatches.Patch(color="#e74c3c", label="Converted in Round 2 (Closed-Loop Feedback)"),
        mpatches.Patch(color="#d35400", label="Breached (Static Verbatim)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9.0, framealpha=0.95, bbox_to_anchor=(0.5, 0.015))

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"[INFO] Attack fate funnel visualization saved successfully to {output_plot}")

    # Also keep legacy output path synchronized if default
    if defense_name == "relevance_filtering":
        legacy_plot = PROJECT_ROOT / "results" / "attack_fate_funnel.png"
        plt.savefig(legacy_plot, dpi=300, bbox_inches="tight")
        print(f"[INFO] Synchronized legacy visualization to {legacy_plot}")

    plt.close()


if __name__ == "__main__":
    main()
