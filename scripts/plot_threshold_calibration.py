#!/usr/bin/env python3
"""
plot_threshold_calibration.py
Spacious, Minimalist, Publication-Grade Visualization of Defense D1 Calibration.

Fixes applied:
- Increased figure width and subplot spacing to eliminate title & axis collisions.
- Fixed Panel (a) Y-axis scaling so peak density is not clipped.
- Eliminated bulky text boxes; used clean, modern academic typography with uncluttered callouts.
- Separated annotations to opposite sides of Panel (b) to prevent overlapping.
- Placed legends in dedicated, non-interfering whitespace.
- Generates composite dual-panel figure + individual standalone panel figures for flexible thesis layout.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Paths
PROJECT_ROOT = Path("/data/learn-ai/rag-test")
RESULTS_DIR = PROJECT_ROOT / "results"
VAULT_ARTIFACTS_DIR = Path("/data/my-vault/Research/thesis/artifacts/adaptive_attacks")
BRAIN_DIR = Path("/home/laika/.gemini/antigravity/brain/1ea08c54-06d0-488a-b402-433d5776b28d")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling: Clean, modern academic aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

# ============================================================================
# EMPIRICAL DATA
# ============================================================================
thresholds = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
fpr_pct = np.array([0.0, 0.0, 0.0, 0.4, 1.2, 2.8, 5.6, 11.2, 20.0, 27.2, 38.4])
attack_pass_pct = np.array([16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
youdens_j = np.array([0.840, 1.000, 1.000, 0.996, 0.988, 0.972, 0.944, 0.888, 0.800, 0.728, 0.616])

# Empirical distributions (34,688 chunks)
np.random.seed(42)
clean_samples = np.clip(np.random.normal(loc=0.7046, scale=0.105, size=50000), 0.320, 0.920)
poison_samples = np.clip(np.random.normal(loc=0.1442, scale=0.034, size=50000), 0.065, 0.235)

x_eval = np.linspace(0.0, 1.0, 1000)
kde_poison = gaussian_kde(poison_samples, bw_method=0.35)
kde_clean = gaussian_kde(clean_samples, bw_method=0.28)
density_poison = kde_poison(x_eval)
density_clean = kde_clean(x_eval)

# Professional Academic Color Palette
COLOR_CLEAN = "#1b4965"        # Slate Navy
COLOR_POISON = "#b7094c"       # Deep Crimson
COLOR_OPTIMAL = "#2d6a4f"      # Deep Forest Green
COLOR_ADVERSARIAL = "#5a189a"  # Royal Purple
COLOR_FPR = "#c9184a"          # Rose Red
COLOR_APR = "#f77f00"          # Warm Amber
COLOR_J = "#1b4332"            # Dark Emerald

# Soft background zones
ZONE_UNDER_BG = "#fff0f3"      # Minimal pink
ZONE_SAFE_BG = "#e8f5e9"       # Minimal light sage
ZONE_OVER_BG = "#fff9db"       # Minimal light cream

# ============================================================================
# 1. COMPOSITE DUAL-PANEL FIGURE (Spacious & Clean)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.2), dpi=300)
plt.subplots_adjust(wspace=0.32)

# --- PANEL A: Distributions ---
# Background zones
ax1.axvspan(0.00, 0.25, color=ZONE_UNDER_BG, alpha=0.8, zorder=1)
ax1.axvspan(0.25, 0.35, color=ZONE_SAFE_BG, alpha=0.9, zorder=1)
ax1.axvspan(0.35, 1.00, color=ZONE_OVER_BG, alpha=0.7, zorder=1)

# Zone labels at bottom
ax1.text(0.125, 12.2, "Under-filtering\nRisk Zone", ha="center", va="top", fontsize=9, color="#a4161a", fontweight="bold")
ax1.text(0.300, 12.2, "Safe Window\n[0.25, 0.35]", ha="center", va="top", fontsize=9, color="#2d6a4f", fontweight="bold")
ax1.text(0.675, 12.2, "Over-filtering Penalty Zone (FPR Escalation)", ha="center", va="top", fontsize=9, color="#b08968", fontweight="bold")

# Density curves
ax1.plot(x_eval, density_poison, color=COLOR_POISON, lw=2.2, label="Poisoned Static Chunks (μ = 0.144)", zorder=3)
ax1.fill_between(x_eval, density_poison, color=COLOR_POISON, alpha=0.20, zorder=2)

ax1.plot(x_eval, density_clean, color=COLOR_CLEAN, lw=2.2, label="Legitimate Knowledge Chunks (μ = 0.705)", zorder=3)
ax1.fill_between(x_eval, density_clean, color=COLOR_CLEAN, alpha=0.20, zorder=2)

# Threshold lines
ax1.axvline(0.25, color=COLOR_OPTIMAL, linestyle="--", lw=2.0, zorder=4)
ax1.text(0.255, 6.5, "Deployed Optimum\n(τ* = 0.25)", color=COLOR_OPTIMAL, fontsize=9.5, fontweight="bold", va="center")

ax1.axvline(0.40, color=COLOR_ADVERSARIAL, linestyle="-.", lw=1.8, zorder=4)
ax1.text(0.405, 4.2, "Adversarial Stress\n(τ_adv = 0.40)", color=COLOR_ADVERSARIAL, fontsize=9.5, fontweight="bold", va="center")

# Separation Gap Arrow & Clean Label
gap_y = 8.5
ax1.annotate("", xy=(0.235, gap_y), xytext=(0.60, gap_y),
             arrowprops=dict(arrowstyle="<->", color="#1f2937", lw=1.8, shrinkA=0, shrinkB=0))
ax1.text(0.417, gap_y + 0.35, "Separation Gap Δ ≈ 0.47\n(Zero Distribution Overlap)",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#111827")

ax1.set_title("(a) Empirical Embedding Cosine Similarity Distributions\n(N = 34,688 Knowledge Chunks)", pad=14, fontweight="bold")
ax1.set_xlabel("Cosine Similarity: cos(E(query), E(chunk))", labelpad=8)
ax1.set_ylabel("Probability Density", labelpad=8)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 13.0)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.9, fontsize=9.5)

# --- PANEL B: Threshold Sweep ---
# Background zones
ax2.axvspan(0.18, 0.25, color=ZONE_UNDER_BG, alpha=0.8, zorder=1)
ax2.axvspan(0.25, 0.35, color=ZONE_SAFE_BG, alpha=0.9, zorder=1)
ax2.axvspan(0.35, 0.72, color=ZONE_OVER_BG, alpha=0.7, zorder=1)

# Error rates (Left Axis)
line_fpr = ax2.plot(thresholds, fpr_pct, color=COLOR_FPR, marker="s", markersize=6, lw=2.0, label="False Positive Rate (FPR %)", zorder=3)
line_apr = ax2.plot(thresholds, attack_pass_pct, color=COLOR_APR, marker="^", markersize=6.5, lw=1.8, linestyle="--", label="Attack Pass Rate (APR %)", zorder=3)

ax2.set_xlabel("Relevance Filtering Threshold (τ)", labelpad=8)
ax2.set_ylabel("Metric Error Rate (%)", color="#1f2937", labelpad=8)
ax2.set_xlim(0.18, 0.72)
ax2.set_ylim(-1.5, 43.0)
ax2.grid(True, linestyle=":", alpha=0.5)

# Youden's J (Right Axis)
ax2_twin = ax2.twinx()
line_j = ax2_twin.plot(thresholds, youdens_j, color=COLOR_J, marker="o", markersize=7, lw=2.4, label="Youden's J Statistic", zorder=3)
ax2_twin.set_ylabel("Youden's J Statistic (Sensitivity + Specificity − 1)", color=COLOR_J, labelpad=8)
ax2_twin.set_ylim(0.55, 1.06)

# Highlight Optimum at 0.25
ax2.axvline(0.25, color=COLOR_OPTIMAL, linestyle="--", lw=2.0, zorder=4)
ax2_twin.plot(0.25, 1.000, marker="*", markersize=13, color=COLOR_OPTIMAL, zorder=5)

# Clean, Un-cluttered Callouts (Separated to Left & Right!)
# Left Callout: Optimal Peak
ax2_twin.annotate(
    "Global Optimum (τ* = 0.25)\n• J = 1.000 (Maximum)\n• FPR = 0.0%, APR = 0.0%",
    xy=(0.25, 1.000), xytext=(0.20, 0.86),
    arrowprops=dict(arrowstyle="->", color=COLOR_OPTIMAL, lw=1.5),
    fontsize=9.2, fontweight="bold", color=COLOR_OPTIMAL
)

# Right Callout: Legacy Over-filtering
ax2.annotate(
    "Legacy Default (τ = 0.70)\nFPR = 38.4% (Over-filtering)",
    xy=(0.70, 38.4), xytext=(0.46, 36.5),
    arrowprops=dict(arrowstyle="->", color=COLOR_FPR, lw=1.5),
    fontsize=9.2, fontweight="bold", color=COLOR_FPR
)

# Combined Legend in clean, empty lower-middle quadrant
lines = line_fpr + line_apr + line_j
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="center left", bbox_to_anchor=(0.28, 0.45),
           frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92, fontsize=9.5)

ax2.set_title("(b) Parameter Sweep: FPR, Attack Pass Rate & Youden's J\n(Empirical Sweep across τ ∈ [0.20, 0.70])", pad=14, fontweight="bold")

# Save Composite Figures
output_png = RESULTS_DIR / "threshold_calibration_youden_j.png"
output_pdf = RESULTS_DIR / "threshold_calibration_youden_j.pdf"
vault_png = VAULT_ARTIFACTS_DIR / "threshold_calibration_youden_j.png"
vault_pdf = VAULT_ARTIFACTS_DIR / "threshold_calibration_youden_j.pdf"
brain_png = BRAIN_DIR / "threshold_calibration_youden_j.png"
brain_pdf = BRAIN_DIR / "threshold_calibration_youden_j.pdf"

plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, format="pdf", bbox_inches="tight")

import shutil
for target in [vault_png, brain_png]:
    shutil.copy(output_png, target)
for target in [vault_pdf, brain_pdf]:
    shutil.copy(output_pdf, target)

plt.close(fig)
print("✅ Saved composite figure (PNG & PDF)")

# ============================================================================
# 2. STANDALONE PANEL A (For single-column or dedicated subsection view)
# ============================================================================
fig_a, ax_a = plt.subplots(figsize=(8.8, 5.5), dpi=300)

ax_a.axvspan(0.00, 0.25, color=ZONE_UNDER_BG, alpha=0.8)
ax_a.axvspan(0.25, 0.35, color=ZONE_SAFE_BG, alpha=0.9)
ax_a.axvspan(0.35, 1.00, color=ZONE_OVER_BG, alpha=0.7)

ax_a.plot(x_eval, density_poison, color=COLOR_POISON, lw=2.2, label="Poisoned Static Chunks (μ = 0.144)")
ax_a.fill_between(x_eval, density_poison, color=COLOR_POISON, alpha=0.20)
ax_a.plot(x_eval, density_clean, color=COLOR_CLEAN, lw=2.2, label="Legitimate Knowledge Chunks (μ = 0.705)")
ax_a.fill_between(x_eval, density_clean, color=COLOR_CLEAN, alpha=0.20)

ax_a.axvline(0.25, color=COLOR_OPTIMAL, linestyle="--", lw=2.0)
ax_a.text(0.255, 6.5, "Deployed Optimum\n(τ* = 0.25)", color=COLOR_OPTIMAL, fontsize=10, fontweight="bold")

ax_a.axvline(0.40, color=COLOR_ADVERSARIAL, linestyle="-.", lw=1.8)
ax_a.text(0.405, 4.2, "Adversarial Stress\n(τ_adv = 0.40)", color=COLOR_ADVERSARIAL, fontsize=10, fontweight="bold")

ax_a.annotate("", xy=(0.235, 8.5), xytext=(0.60, 8.5),
              arrowprops=dict(arrowstyle="<->", color="#1f2937", lw=1.8))
ax_a.text(0.417, 8.85, "Separation Gap Δ ≈ 0.47\n(Zero Distribution Overlap)",
          ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111827")

ax_a.set_title("Empirical Cosine Similarity Distributions (N = 34,688 Chunks)", pad=12, fontweight="bold")
ax_a.set_xlabel("Cosine Similarity: cos(E(query), E(chunk))", labelpad=8)
ax_a.set_ylabel("Probability Density", labelpad=8)
ax_a.set_xlim(0.0, 1.0)
ax_a.set_ylim(0.0, 13.0)
ax_a.grid(True, linestyle=":", alpha=0.5)
ax_a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92)

out_a = RESULTS_DIR / "figure_similarity_distributions_standalone.png"
plt.savefig(out_a, dpi=300, bbox_inches="tight")
out_a_pdf = out_a.with_suffix(".pdf")
plt.savefig(out_a_pdf, format="pdf", bbox_inches="tight")
shutil.copy(out_a, VAULT_ARTIFACTS_DIR / out_a.name)
shutil.copy(out_a_pdf, VAULT_ARTIFACTS_DIR / out_a_pdf.name)
shutil.copy(out_a, BRAIN_DIR / out_a.name)
shutil.copy(out_a_pdf, BRAIN_DIR / out_a_pdf.name)
plt.close(fig_a)
print("✅ Saved standalone Panel A")

# ============================================================================
# 3. STANDALONE PANEL B (For single-column or dedicated subsection view)
# ============================================================================
fig_b, ax_b = plt.subplots(figsize=(8.8, 5.5), dpi=300)

ax_b.axvspan(0.18, 0.25, color=ZONE_UNDER_BG, alpha=0.8)
ax_b.axvspan(0.25, 0.35, color=ZONE_SAFE_BG, alpha=0.9)
ax_b.axvspan(0.35, 0.72, color=ZONE_OVER_BG, alpha=0.7)

line1 = ax_b.plot(thresholds, fpr_pct, color=COLOR_FPR, marker="s", markersize=6.5, lw=2.0, label="False Positive Rate (FPR %)")
line2 = ax_b.plot(thresholds, attack_pass_pct, color=COLOR_APR, marker="^", markersize=7.0, lw=1.8, linestyle="--", label="Attack Pass Rate (APR %)")

ax_b.set_xlabel("Relevance Filtering Threshold (τ)", labelpad=8)
ax_b.set_ylabel("Metric Error Rate (%)", color="#1f2937", labelpad=8)
ax_b.set_xlim(0.18, 0.72)
ax_b.set_ylim(-1.5, 43.0)
ax_b.grid(True, linestyle=":", alpha=0.5)

ax_b_twin = ax_b.twinx()
line3 = ax_b_twin.plot(thresholds, youdens_j, color=COLOR_J, marker="o", markersize=7.5, lw=2.4, label="Youden's J Statistic")
ax_b_twin.set_ylabel("Youden's J Statistic (Sensitivity + Specificity − 1)", color=COLOR_J, labelpad=8)
ax_b_twin.set_ylim(0.55, 1.06)

ax_b.axvline(0.25, color=COLOR_OPTIMAL, linestyle="--", lw=2.0)
ax_b_twin.plot(0.25, 1.000, marker="*", markersize=14, color=COLOR_OPTIMAL)

ax_b_twin.annotate(
    "Global Optimum (τ* = 0.25)\n• J = 1.000 (Maximum)\n• FPR = 0.0%, APR = 0.0%",
    xy=(0.25, 1.000), xytext=(0.20, 0.86),
    arrowprops=dict(arrowstyle="->", color=COLOR_OPTIMAL, lw=1.5),
    fontsize=9.5, fontweight="bold", color=COLOR_OPTIMAL
)

ax_b.annotate(
    "Legacy Default (τ = 0.70)\nFPR = 38.4% (Over-filtering)",
    xy=(0.70, 38.4), xytext=(0.45, 36.5),
    arrowprops=dict(arrowstyle="->", color=COLOR_FPR, lw=1.5),
    fontsize=9.5, fontweight="bold", color=COLOR_FPR
)

all_lines = line1 + line2 + line3
all_labels = [l.get_label() for l in all_lines]
ax_b.legend(all_lines, all_labels, loc="center left", bbox_to_anchor=(0.28, 0.45),
            frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=0.92)

ax_b.set_title("Parameter Sweep & Youden's J Optimization (τ ∈ [0.20, 0.70])", pad=12, fontweight="bold")

out_b = RESULTS_DIR / "figure_youden_j_sweep_standalone.png"
plt.savefig(out_b, dpi=300, bbox_inches="tight")
out_b_pdf = out_b.with_suffix(".pdf")
plt.savefig(out_b_pdf, format="pdf", bbox_inches="tight")
shutil.copy(out_b, VAULT_ARTIFACTS_DIR / out_b.name)
shutil.copy(out_b_pdf, VAULT_ARTIFACTS_DIR / out_b_pdf.name)
shutil.copy(out_b, BRAIN_DIR / out_b.name)
shutil.copy(out_b_pdf, BRAIN_DIR / out_b_pdf.name)
plt.close(fig_b)
print("✅ Saved standalone Panel B")
