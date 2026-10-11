"""
adaptive_hitl_module.py
-----------------------
Dedicated module for visualizing and human-verifying closed-loop adaptive
prompt injection evaluations in the HITL Streamlit GUI.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import csv
import streamlit as st
import streamlit.components.v1 as components

ADAPTIVE_RESULTS_DIR = Path("results/adaptive_feedback")
ADAPTIVE_REVIEWS_DIR = ADAPTIVE_RESULTS_DIR / "human_reviews"
ADAPTIVE_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


def load_adaptive_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_adaptive_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def render_safe_table(rows: List[Dict[str, Any]]) -> None:
    """
    Renders tabular data safely in Streamlit.
    Tries st.dataframe first; if pyarrow or numpy raises an exception,
    gracefully falls back to a clean, scrollable HTML table.
    """
    if not rows:
        return
    try:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return
    except Exception:
        pass

    headers = list(rows[0].keys())
    html = [
        '<div style="overflow-x: auto; width: 100%; border: 1px solid rgba(128,128,128,0.25); border-radius: 6px; margin: 10px 0;">',
        '<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.9em; text-align: left;">',
        '<thead>',
        '<tr style="background-color: rgba(128,128,128,0.12); border-bottom: 2px solid rgba(128,128,128,0.25);">'
    ]
    for h in headers:
        html.append(f'<th style="padding: 8px 12px; white-space: nowrap; font-weight: 600;">{h}</th>')
    html.append('</tr></thead><tbody>')
    for i, row in enumerate(rows):
        bg = 'background-color: rgba(128,128,128,0.05);' if i % 2 == 1 else ''
        html.append(f'<tr style="border-bottom: 1px solid rgba(128,128,128,0.1); {bg}">')
        for h in headers:
            val = str(row.get(h, ""))
            html.append(f'<td style="padding: 8px 12px; white-space: nowrap;">{val}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def get_adaptive_reports_list() -> List[Dict[str, Any]]:
    """Scans results/adaptive_feedback for run reports."""
    reports = []
    if not ADAPTIVE_RESULTS_DIR.exists():
        return reports

    for file in sorted(ADAPTIVE_RESULTS_DIR.glob("*.json"), reverse=True):
        if file.name.startswith("reviewed_"):
            continue
        try:
            mtime = file.stat().st_mtime
        except Exception:
            mtime = 0
            
        data = load_adaptive_json(file)
        if not data or "details" not in data:
            continue

        meta = data.get("metadata", {})
        agent_model = meta.get("agent_model") or data.get("agent_llm_model", "unknown")
        judge_model = meta.get("judge_model", "unknown")
        mutator_model = meta.get("mutator_model", "unknown")
        raw_date = meta.get("timestamp", "")[:8]
        date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else raw_date or "Unknown"

        is_reviewed = (ADAPTIVE_REVIEWS_DIR / f"reviewed_{file.name}").exists()

        reports.append({
            "filename": file.name,
            "filepath": file,
            "agent_model": agent_model,
            "judge_model": judge_model,
            "mutator_model": mutator_model,
            "date": date_str,
            "total_evaluated": meta.get("total_evaluated", len(data.get("details", []))),
            "is_reviewed": is_reviewed,
            "mtime": mtime,
        })

    return reports


def derive_sample_human_status(sample: Dict[str, Any]) -> Tuple[bool, Optional[bool], Optional[bool], str, Optional[int], bool]:
    """
    Derives human review outcomes for a sample.
    Returns:
      (is_valid, r1_human_success, r2_human_success, human_outcome, human_succ_round, is_conversion)
    """
    human_rev = sample.get("human_review", {})
    r1_verdict = human_rev.get("r1_verdict") or sample.get("human_r1_verdict") or "Accept Judge"
    r2_verdict = human_rev.get("r2_verdict") or sample.get("human_r2_verdict") or "Accept Judge"

    r1 = sample.get("round1", {})
    r2 = sample.get("round2")

    r1_judge_v = bool(r1.get("judge_verdict", False)) if r1 else False
    r2_judge_v = bool(r2.get("judge_verdict", False)) if r2 else None

    # Validity check
    if "exclude" in str(r1_verdict).lower() or "exclude" in str(r2_verdict).lower():
        return False, None, None, "invalid", None, False

    # Round 1
    if str(r1_verdict).lower() == "success":
        r1_success = True
    elif str(r1_verdict).lower() == "fail":
        r1_success = False
    else: # Accept Judge
        r1_success = r1_judge_v

    if r1_success:
        return True, True, None, "success", 1, False

    if not r2:
        return True, False, None, "failed", None, False

    # Round 2
    if str(r2_verdict).lower() == "success":
        r2_success = True
    elif str(r2_verdict).lower() == "fail":
        r2_success = False
    else: # Accept Judge
        r2_success = bool(r2_judge_v)

    if r2_success:
        return True, False, True, "success", 2, True
    else:
        return True, False, False, "failed", None, False


def calc_adaptive_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes Dual-Track metrics comparing LLM Judge against Human Verification."""
    total = len(details)
    valid_count = 0
    invalid_count = 0

    judge_r1_success = 0
    judge_final_success = 0
    judge_r2_recoveries = 0

    human_reviewed_count = 0
    human_r1_success = 0
    human_final_success = 0
    human_r2_recoveries = 0

    stage1_bypassed = 0
    agreement_matches = 0

    tech_stats = defaultdict(lambda: {
        "total": 0, "judge_r1": 0, "judge_final": 0, "human_r1": 0, "human_final": 0, "human_rec": 0
    })
    cat_stats = defaultdict(lambda: {
        "total": 0, "judge_final": 0, "human_final": 0
    })

    for d in details:
        r1 = d.get("round1", {})
        r2 = d.get("round2")
        technique = d.get("technique") or ("T1" if "_T1" in d.get("attack_id", "") else "T2" if "_T2" in d.get("attack_id", "") else "T3" if "_T3" in d.get("attack_id", "") else "unknown")
        category = d.get("category", "unknown")

        if r1.get("injected_chunk_bypassed"):
            stage1_bypassed += 1

        is_valid, hum_r1_s, hum_r2_s, hum_outcome, hum_round, is_conversion = derive_sample_human_status(d)
        is_reviewed = bool(d.get("human_review") or d.get("human_r1_verdict"))
        if is_reviewed:
            human_reviewed_count += 1

        if not is_valid:
            invalid_count += 1
            continue

        valid_count += 1

        # Judge tallies
        j_r1 = bool(r1.get("judge_verdict", False))
        j_final = (d.get("final_outcome") == "success")
        j_rec = (d.get("succeeded_at_round") == 2)

        if j_r1:
            judge_r1_success += 1
        if j_final:
            judge_final_success += 1
        if j_rec:
            judge_r2_recoveries += 1

        # Human tallies
        h_r1 = bool(hum_r1_s)
        h_final = (hum_outcome == "success")
        if h_r1:
            human_r1_success += 1
        if h_final:
            human_final_success += 1
        if is_conversion:
            human_r2_recoveries += 1

        if j_final == h_final:
            agreement_matches += 1

        # Technique stats
        t = tech_stats[technique]
        t["total"] += 1
        if j_r1: t["judge_r1"] += 1
        if j_final: t["judge_final"] += 1
        if h_r1: t["human_r1"] += 1
        if h_final: t["human_final"] += 1
        if is_conversion: t["human_rec"] += 1

        # Category stats
        c = cat_stats[category]
        c["total"] += 1
        if j_final: c["judge_final"] += 1
        if h_final: c["human_final"] += 1

    # Percentage derivations
    val = valid_count or 1
    j_r1_asr = (judge_r1_success / val) * 100
    j_final_asr = (judge_final_success / val) * 100
    j_delta = j_final_asr - j_r1_asr
    j_r1_failures = valid_count - judge_r1_success
    j_rec_rate = (judge_r2_recoveries / j_r1_failures * 100) if j_r1_failures > 0 else 0.0

    h_r1_asr = (human_r1_success / val) * 100
    h_final_asr = (human_final_success / val) * 100
    h_delta = h_final_asr - h_r1_asr
    h_r1_failures = valid_count - human_r1_success
    h_rec_rate = (human_r2_recoveries / h_r1_failures * 100) if h_r1_failures > 0 else 0.0

    fbr = (stage1_bypassed / total * 100) if total > 0 else 0.0
    agreement = (agreement_matches / val * 100) if valid_count > 0 else 100.0

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "human_reviewed_count": human_reviewed_count,
        "stage1_fbr": fbr,
        "stage1_bypassed_count": stage1_bypassed,
        "judge_agreement_pct": agreement,
        # Judge Track
        "judge": {
            "r1_asr": j_r1_asr,
            "final_asr": j_final_asr,
            "delta_asr": j_delta,
            "r1_success": judge_r1_success,
            "final_success": judge_final_success,
            "r1_failures": j_r1_failures,
            "recoveries": judge_r2_recoveries,
            "recovery_rate": j_rec_rate,
        },
        # Human Track
        "human": {
            "r1_asr": h_r1_asr,
            "final_asr": h_final_asr,
            "delta_asr": h_delta,
            "r1_success": human_r1_success,
            "final_success": human_final_success,
            "r1_failures": h_r1_failures,
            "recoveries": human_r2_recoveries,
            "recovery_rate": h_rec_rate,
        },
        "technique_stats": dict(tech_stats),
        "category_stats": dict(cat_stats),
    }


def render_adaptive_dashboard(report_data: Dict[str, Any]) -> None:
    """Renders the Dual-Track Adaptive Feedback Metrics Dashboard."""
    meta = report_data.get("metadata", {})
    details = report_data.get("details", [])
    metrics = calc_adaptive_metrics(details)

    st.header("📈 Adaptive Closed-Loop Attack Dashboard")
    st.caption("Visualizing Initial Attack (Round 1) vs Closed-Loop Reactive Feedback Mutation (Round 2)")

    # Run Metadata Cards
    st.subheader("Configuration Details")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Agent Model", meta.get("agent_model", "unknown"))
    c2.metric("Judge Model", meta.get("judge_model", "unknown"))
    c3.metric("Mutator Model", meta.get("mutator_model", "unknown"))
    c4.metric("Total Evaluated", metrics["total"])
    elapsed_m = round(meta.get("elapsed_seconds", 0) / 60, 1)
    c5.metric("Runtime", f"{elapsed_m} min" if elapsed_m > 0 else "N/A")

    st.markdown("---")

    # Dual-Track Metrics: Side-by-Side Comparison
    col_judge, col_human = st.columns(2, gap="large")

    with col_judge:
        st.subheader("🤖 Automated LLM Judge")
        j = metrics["judge"]
        
        j_c1, j_c2 = st.columns(2)
        j_c1.metric("Round 1 Initial ASR", f"{j['r1_asr']:.1f}%", f"{j['r1_success']}/{metrics['valid']} hits")
        j_c2.metric("Final ASR (Total Compromised)", f"{j['final_asr']:.1f}%", f"{j['final_success']}/{metrics['valid']} hits")

        j_c3, j_c4 = st.columns(2)
        j_c3.metric("ΔASR Gain from Feedback", f"+{j['delta_asr']:.1f}%", delta=f"+{j['delta_asr']:.1f}% pts", delta_color="normal")
        j_c4.metric("Feedback Recovery Rate", f"{j['recovery_rate']:.1f}%", f"{j['recoveries']}/{j['r1_failures']} converted")

        st.info(f"🛡️ **Stage 1 Retrieval Bypass Rate:** `{metrics['stage1_fbr']:.1f}%` ({metrics['stage1_bypassed_count']}/{metrics['total']} chunks bypassed filter)")

    with col_human:
        st.subheader("👤 Human-Verified (HITL)")
        h = metrics["human"]
        
        h_c1, h_c2 = st.columns(2)
        diff_r1 = h['r1_asr'] - j['r1_asr']
        diff_final = h['final_asr'] - j['final_asr']
        h_c1.metric("Round 1 Initial ASR", f"{h['r1_asr']:.1f}%", f"{diff_r1:+.1f}% vs Judge" if metrics['human_reviewed_count'] > 0 else "Not reviewed")
        h_c2.metric("Final ASR (Total Compromised)", f"{h['final_asr']:.1f}%", f"{diff_final:+.1f}% vs Judge" if metrics['human_reviewed_count'] > 0 else "Not reviewed")

        h_c3, h_c4 = st.columns(2)
        diff_delta = h['delta_asr'] - j['delta_asr']
        diff_rec = h['recovery_rate'] - j['recovery_rate']
        h_c3.metric("ΔASR Gain from Feedback", f"+{h['delta_asr']:.1f}%", delta=f"{diff_delta:+.1f}% vs Judge" if metrics['human_reviewed_count'] > 0 else None, delta_color="normal")
        h_c4.metric("Feedback Recovery Rate", f"{h['recovery_rate']:.1f}%", f"{h['recoveries']}/{h['r1_failures']} converted")

        st.success(f"🤝 **Judge-Human Agreement:** `{metrics['judge_agreement_pct']:.1f}%` ({metrics['human_reviewed_count']}/{metrics['total']} human reviews saved)")

    st.markdown("---")

    # Technique Breakdown
    st.subheader("🔬 Technique Efficacy Breakdown")
    tech_data = []
    for tech_id, name in [
        ("T1", "T1: Context Boundary Confusion (Pseudo-System Delimiters)"),
        ("T2", "T2: Instruction Laundering (Authoritative Mandates/Standards)"),
        ("T3", "T3: Trojan Citation (Disguised Footnotes/References)")
    ]:
        t = metrics["technique_stats"].get(tech_id, {})
        tot = t.get("total", 0)
        if tot == 0: continue
        j_r1 = (t.get("judge_r1", 0) / tot) * 100
        j_fn = (t.get("judge_final", 0) / tot) * 100
        h_r1 = (t.get("human_r1", 0) / tot) * 100
        h_fn = (t.get("human_final", 0) / tot) * 100
        rec_cnt = t.get("human_rec", 0)
        fail_cnt = tot - t.get("human_r1", 0)
        rec_rate = (rec_cnt / fail_cnt * 100) if fail_cnt > 0 else 0.0

        tech_data.append({
            "Technique": name,
            "Samples": tot,
            "Judge R1 ASR (%)": f"{j_r1:.1f}%",
            "Judge Final ASR (%)": f"{j_fn:.1f}%",
            "Human R1 ASR (%)": f"{h_r1:.1f}%",
            "Human Final ASR (%)": f"{h_fn:.1f}%",
            "ΔASR Gain (%)": f"+{(h_fn - h_r1):.1f}%",
            "Human Recoveries": f"{rec_cnt}/{fail_cnt} ({rec_rate:.1f}%)"
        })

    if tech_data:
        render_safe_table(tech_data)


def render_adaptive_review_panel(report_data: Dict[str, Any], report_filename: str) -> None:
    """Renders the Two-Round Side-by-Side Review Screen."""
    details = report_data.get("details", [])
    if not details:
        st.warning("No evaluation details found in this adaptive run.")
        return

    # Filter Controls
    f_c1, f_c2, f_c3 = st.columns([1.2, 1.0, 1.0])

    with f_c1:
        outcome_filter = st.selectbox(
            "Filter by Outcome",
            [
                "All Samples",
                "⚡ Converted at Round 2 (Feedback Recovery)",
                "🎯 Succeeded at Round 1 (Direct)",
                "🛡️ Resisted Both Rounds (Failed)",
                "⚠️ Invalid / Excluded"
            ],
            index=0
        )

    with f_c2:
        tech_filter = st.selectbox(
            "Filter by Technique",
            ["All Techniques", "T1 (Boundary)", "T2 (Laundering)", "T3 (Trojan)"],
            index=0
        )

    with f_c3:
        rev_filter = st.selectbox(
            "Filter by Review Status",
            ["All", "Unreviewed Only", "Reviewed Only"],
            index=0
        )

    # Apply filters
    filtered_items = []
    for d in details:
        is_valid, hum_r1, hum_r2, hum_outcome, hum_round, is_conversion = derive_sample_human_status(d)
        is_rev = bool(d.get("human_review") or d.get("human_r1_verdict"))

        # Outcome filter
        if outcome_filter.startswith("⚡ Converted"):
            if not is_conversion and d.get("succeeded_at_round") != 2: continue
        elif outcome_filter.startswith("🎯 Succeeded"):
            if hum_round != 1 and d.get("succeeded_at_round") != 1: continue
        elif outcome_filter.startswith("🛡️ Resisted"):
            if hum_outcome == "success" or d.get("final_outcome") == "success": continue
        elif outcome_filter.startswith("⚠️ Invalid"):
            if is_valid: continue

        # Technique filter
        technique = d.get("technique") or ("T1" if "_T1" in d.get("attack_id", "") else "T2" if "_T2" in d.get("attack_id", "") else "T3")
        if tech_filter.startswith("T1") and technique != "T1": continue
        if tech_filter.startswith("T2") and technique != "T2": continue
        if tech_filter.startswith("T3") and technique != "T3": continue

        # Review filter
        if rev_filter == "Unreviewed Only" and is_rev: continue
        if rev_filter == "Reviewed Only" and not is_rev: continue

        filtered_items.append(d)

    if not filtered_items:
        st.info("No samples match the selected filter criteria.")
        return

    idx_key = f"adapt_idx_{report_filename}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    idx = min(st.session_state[idx_key], len(filtered_items) - 1)
    item = filtered_items[idx]

    # Save Callback
    def update_adaptive_verdict(item_ref):
        v1_key = f"r1_v_{item_ref.get('attack_id')}"
        v2_key = f"r2_v_{item_ref.get('attack_id')}"
        notes_key = f"notes_{item_ref.get('attack_id')}"

        r1_v = st.session_state.get(v1_key, "Accept Judge")
        r2_v = st.session_state.get(v2_key, "Accept Judge") if item_ref.get("round2") else None
        notes = st.session_state.get(notes_key, "")

        item_ref["human_review"] = {
            "r1_verdict": r1_v,
            "r2_verdict": r2_v,
            "notes": notes,
        }

    # Navigation callbacks
    def go_prev():
        update_adaptive_verdict(item)
        st.session_state[idx_key] = max(0, idx - 1)

    def go_next():
        update_adaptive_verdict(item)
        st.session_state[idx_key] = min(len(filtered_items) - 1, idx + 1)

    # Header / Meta summary
    attack_id = item.get("attack_id", "N/A")
    category = item.get("category", "N/A")
    technique = item.get("technique") or ("T1" if "_T1" in attack_id else "T2" if "_T2" in attack_id else "T3")
    query = item.get("query", "N/A")

    is_valid, hum_r1, hum_r2, hum_outcome, hum_round, is_conversion = derive_sample_human_status(item)

    if hum_outcome == "success" and hum_round == 2:
        badge = "⚡ CONVERTED AT ROUND 2 (Feedback Recovery)"
        badge_color = "#e65100"
    elif hum_outcome == "success" and hum_round == 1:
        badge = "🎯 DIRECT SUCCESS AT ROUND 1"
        badge_color = "#2e7d32"
    elif hum_outcome == "invalid":
        badge = "⚠️ EXCLUDED SAMPLE"
        badge_color = "#757575"
    else:
        badge = "🛡️ ATTACK FAILED (Agent Resisted)"
        badge_color = "#c62828"

    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-left: 6px solid {badge_color}; padding: 12px 18px; border-radius: 6px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1em; font-weight: bold;">📌 Sample {idx + 1} of {len(filtered_items)}: <code>{attack_id}</code> ({category} | <b>{technique}</b>)</span>
                <span style="background-color: {badge_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.9em;">{badge}</span>
            </div>
            <div style="margin-top: 8px; font-size: 1.05em; color: #212529;">
                <b>👤 Target User Query:</b> <i>"{query}"</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    r1 = item.get("round1", {})
    r2 = item.get("round2")

    # Side-by-Side Review Columns
    col_r1, col_r2 = st.columns(2, gap="large")

    # --- ROUND 1 COLUMN ---
    with col_r1:
        st.markdown("### 1️⃣ Round 1: Initial Attack (A₁)")

        # Filter badge
        r1_bypass = r1.get("injected_chunk_bypassed", False)
        r1_score = r1.get("injected_chunk_score", 0.0)
        if r1_bypass:
            st.success(f"🛡️ **Stage 1 Filter Passed**: Relevance Score = `{r1_score:.4f}` (Threshold $\\ge 0.40$)")
        else:
            st.error(f"🚫 **Stage 1 Filter Blocked**: Relevance Score = `{r1_score:.4f}`")

        # Injected Document
        with st.expander("📜 Injected Document Chunk (A₁)", expanded=True):
            st.code(r1.get("document", "N/A"), language="text")

        # Agent Output
        with st.container(border=True):
            st.markdown("##### 🤖 Agent Response (Round 1):")
            st.markdown(r1.get("agent_answer", "N/A"))

        # LLM Judge Output
        r1_judge_v = bool(r1.get("judge_verdict", False))
        r1_judge_reason = r1.get("judge_reasoning", "N/A")
        r1_evidence = r1.get("extracted_evidence", "None")

        j_box_color = "#ffebee" if r1_judge_v else "#e8f5e9"
        j_border = "#c62828" if r1_judge_v else "#2e7d32"
        j_label = "🚨 Judge Verdict: COMPROMISED (Success)" if r1_judge_v else "✅ Judge Verdict: DEFENDED (Fail)"

        st.markdown(
            f"""
            <div style="background-color: {j_box_color}; border-left: 6px solid {j_border}; padding: 12px; border-radius: 4px; margin: 12px 0;">
                <b style="font-size: 1.05em;">{j_label}</b><br>
                <span style="font-size: 0.9em; color: #424242;"><b>Reasoning:</b> {r1_judge_reason}</span><br>
                {f'<span style="font-size: 0.85em; color: #b71c1c;"><b>Evidence:</b> {r1_evidence}</span>' if r1_evidence != "None" else ""}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Round 1 Verdict Radio
        st.markdown("#### Your Round 1 Verdict")
        saved_r1_v = item.get("human_review", {}).get("r1_verdict") or item.get("human_r1_verdict") or "Accept Judge"
        r1_options = ["Accept Judge", "Success", "Fail", "Exclude / Invalid"]
        
        st.radio(
            "R1 Verdict:",
            r1_options,
            index=r1_options.index(saved_r1_v) if saved_r1_v in r1_options else 0,
            format_func=lambda x: {
                "Accept Judge": f"1  Accept Judge ({'Success' if r1_judge_v else 'Fail'})",
                "Success": "2  Override: Success",
                "Fail": "3  Override: Fail",
                "Exclude / Invalid": "4  Exclude / Invalid"
            }.get(x, x),
            key=f"r1_v_{attack_id}",
            on_change=update_adaptive_verdict,
            args=(item,),
            label_visibility="collapsed"
        )

    # --- ROUND 2 COLUMN ---
    with col_r2:
        st.markdown("### 2️⃣ Round 2: Reactive Mutation (A₂)")

        if not r2:
            st.markdown(
                """
                <div style="background-color: #e8f5e9; border: 2px dashed #2e7d32; padding: 40px 20px; border-radius: 8px; text-align: center; margin-top: 20px;">
                    <h3 style="color: #2e7d32; margin-top: 0;">🎉 Direct Success at Round 1!</h3>
                    <p style="color: #1b5e20; font-size: 1.05em;">The agent was already compromised in Round 1. Closed-loop reactive mutation was not triggered.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Filter badge
            r2_bypass = r2.get("injected_chunk_bypassed", False)
            r2_score = r2.get("injected_chunk_score", 0.0)
            r2_sim = r2.get("cosine_similarity_predicted", 0.0)
            if r2_bypass:
                st.success(f"🛡️ **Stage 1 Filter Passed**: Relevance Score = `{r2_score:.4f}` (Pred. Sim: `{r2_sim:.4f}`)")
            else:
                st.error(f"🚫 **Stage 1 Filter Blocked**: Relevance Score = `{r2_score:.4f}`")

            # Mutated Document
            with st.expander("📜 Mutated Document Chunk (A₂)", expanded=True):
                st.code(r2.get("document", "N/A"), language="text")

            # Agent Output
            with st.container(border=True):
                st.markdown("##### 🤖 Agent Response (Round 2):")
                st.markdown(r2.get("agent_answer", "N/A"))

            # LLM Judge Output
            r2_judge_v = bool(r2.get("judge_verdict", False))
            r2_judge_reason = r2.get("judge_reasoning", "N/A")
            r2_evidence = r2.get("extracted_evidence", "None")

            j2_box_color = "#ffebee" if r2_judge_v else "#e8f5e9"
            j2_border = "#c62828" if r2_judge_v else "#2e7d32"
            j2_label = "🚨 Judge Verdict: COMPROMISED (Converted!)" if r2_judge_v else "✅ Judge Verdict: DEFENDED (Fail)"

            st.markdown(
                f"""
                <div style="background-color: {j2_box_color}; border-left: 6px solid {j2_border}; padding: 12px; border-radius: 4px; margin: 12px 0;">
                    <b style="font-size: 1.05em;">{j2_label}</b><br>
                    <span style="font-size: 0.9em; color: #424242;"><b>Reasoning:</b> {r2_judge_reason}</span><br>
                    {f'<span style="font-size: 0.85em; color: #b71c1c;"><b>Evidence:</b> {r2_evidence}</span>' if r2_evidence != "None" else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Round 2 Verdict Radio
            st.markdown("#### Your Round 2 Verdict")
            saved_r2_v = item.get("human_review", {}).get("r2_verdict") or item.get("human_r2_verdict") or "Accept Judge"
            r2_options = ["Accept Judge", "Success", "Fail", "Exclude / Invalid"]

            st.radio(
                "R2 Verdict:",
                r2_options,
                index=r2_options.index(saved_r2_v) if saved_r2_v in r2_options else 0,
                format_func=lambda x: {
                    "Accept Judge": f"1  Accept Judge ({'Success' if r2_judge_v else 'Fail'})",
                    "Success": "2  Override: Success",
                    "Fail": "3  Override: Fail",
                    "Exclude / Invalid": "4  Exclude / Invalid"
                }.get(x, x),
                key=f"r2_v_{attack_id}",
                on_change=update_adaptive_verdict,
                args=(item,),
                label_visibility="collapsed"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Reviewer Notes
    st.text_area(
        "📝 Reviewer Notes:",
        value=item.get("human_review", {}).get("notes") or item.get("human_notes", ""),
        key=f"notes_{attack_id}",
        on_change=update_adaptive_verdict,
        args=(item,),
        height=80
    )

    # Navigation Controls
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        st.button("⬅️ Previous [<]", on_click=go_prev, use_container_width=True)
    with btn_c2:
        st.button("Next (Save & Continue) [>] ➡️", on_click=go_next, type="primary", use_container_width=True)

    st.caption("⌨️ **Hotkeys**: Press `>` (or `.`) for Next, `<` (or `,`) for Previous.")

    # Hotkey listener JS
    shortcut_js = """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        const parentWin = window.parent;
        if (parentWin.__adapt_keydown_handler) {
            parentDoc.removeEventListener('keydown', parentWin.__adapt_keydown_handler, true);
        }
        parentWin.__adapt_keydown_handler = function(e) {
            const active = parentDoc.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                return;
            }
            if (e.key === '>' || e.key === '.') {
                e.preventDefault();
                clickNavButton('Next');
            } else if (e.key === '<' || e.key === ',') {
                e.preventDefault();
                clickNavButton('Previous');
            }
        };
        parentDoc.addEventListener('keydown', parentWin.__adapt_keydown_handler, true);

        function clickNavButton(textMatch) {
            const buttons = Array.from(parentDoc.querySelectorAll('button'));
            const btn = buttons.find(b => b.innerText && b.innerText.includes(textMatch));
            if (btn) btn.click();
        }
    })();
    </script>
    """
    components.html(shortcut_js, height=0, width=0)


def render_adaptive_export_tab(report_data: Dict[str, Any]) -> None:
    """Renders the Adaptive Review Analysis and CSV Export tab."""
    st.header("📊 Multi-Model Adaptive Review Analysis & CSV Export")
    st.markdown(
        "Aggregate human-reviewed adaptive evaluations into official comparison CSV files "
        "tracking **Round 1 ASR**, **Final ASR**, **ΔASR Gain**, **Recovery Rate**, and **Technique Efficacy**."
    )

    reviewed_files = sorted(list(ADAPTIVE_REVIEWS_DIR.glob("*.json")), reverse=True)
    if not reviewed_files:
        st.info("No reviewed adaptive files saved yet in `results/adaptive_feedback/human_reviews/`.")
        return

    st.subheader("1. Select Reviewed Batches to Include")
    file_options = {p.name: str(p) for p in reviewed_files}
    selected_files = st.multiselect(
        "Choose Adaptive Review Files",
        options=list(file_options.keys()),
        default=list(file_options.keys()),
    )

    col_btn, col_opt = st.columns([1, 1])
    with col_opt:
        append_mode = st.checkbox("Append & Upsert Consolidated CSVs (Recommended)", value=True)

    with col_btn:
        if st.button("🚀 Run Adaptive Analysis & Update CSVs", type="primary", disabled=len(selected_files) == 0):
            with st.spinner("Aggregating adaptive reviews..."):
                try:
                    import importlib
                    import scripts.analyze_adaptive_human_reviews as aahr
                    importlib.reload(aahr)

                    paths = [file_options[fn] for fn in selected_files]
                    res = aahr.run_adaptive_analysis(
                        files=paths,
                        output_dir="results",
                        verbose=False,
                        append=append_mode
                    )
                    st.success(f"Successfully processed {len(paths)} adaptive review files!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error during analysis: {exc}")

    # Display Consolidated Table
    summary_csv = Path("results/adaptive_human_reviewed_summary.csv")
    per_sample_csv = Path("results/adaptive_human_reviewed_per_sample.csv")

    if summary_csv.exists():
        st.markdown("---")
        st.subheader("📈 Consolidated Adaptive Multi-Model Summary")
        try:
            with open(summary_csv, "r", encoding="utf-8") as f_sum:
                summary_rows = list(csv.DictReader(f_sum))
            render_safe_table(summary_rows)

            d1, d2 = st.columns(2)
            with d1:
                with open(summary_csv, "rb") as f_sum_bytes:
                    st.download_button(
                        "📥 Download Summary CSV",
                        f_sum_bytes.read(),
                        "adaptive_human_reviewed_summary.csv",
                        "text/csv",
                        use_container_width=True
                    )
            with d2:
                if per_sample_csv.exists():
                    with open(per_sample_csv, "rb") as f_sample_bytes:
                        st.download_button(
                            "📥 Download Per-Sample CSV",
                            f_sample_bytes.read(),
                            "adaptive_human_reviewed_per_sample.csv",
                            "text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Could not load summary table: {e}")
