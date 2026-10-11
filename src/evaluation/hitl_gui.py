import streamlit as st
import streamlit.components.v1 as components
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="HITL Evaluation GUI", layout="wide")

st.title("Human-in-the-Loop (HITL) Validation GUI")

RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs/agent_runs")
REVIEWS_DIR = RESULTS_DIR / "human_reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_data
def get_report_meta(filename, mtime):
    filepath = RESULTS_DIR / filename
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            defense = data.get("attack_evaluation", {}).get("defense_summary", {}).get("defense_name")
            if not defense:
                defense = data.get("benign_evaluation", {}).get("defense_summary", {}).get("defense_name", "None")
            raw_date = data.get("timestamp", "")[:8]
            if not raw_date:
                parts = filename.split("_")
                raw_date = parts[2] if len(parts) > 2 else ""
            formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else raw_date or "Unknown"
            
            # Check if there are any pending judge evaluations
            has_pending = False
            for sec in ["attack_evaluation", "benign_evaluation"]:
                for d in data.get(sec, {}).get("details", []):
                    if d.get("evaluation_status") == "valid_judge_pending":
                        has_pending = True
                        break
                if has_pending:
                    break

            return {
                "model": data.get("agent_llm_model") or "Unknown",
                "defense": defense or "None",
                "date": formatted_date,
                "has_pending": has_pending
            }
    except Exception:
        return {"model": "Unknown", "defense": "None", "date": "Unknown", "has_pending": False}

def load_all_reports_info():
    reports_info = []
    if RESULTS_DIR.exists():
        for file in sorted(RESULTS_DIR.glob("evaluation_report_*.json"), reverse=True):
            try:
                mtime = file.stat().st_mtime
            except Exception:
                mtime = 0
            meta = get_report_meta(file.name, mtime)
            is_reviewed = (REVIEWS_DIR / f"reviewed_{file.name}").exists() or (RESULTS_DIR / f"reviewed_{file.name}").exists()
            reports_info.append({
                "filename": file.name,
                "model": meta["model"],
                "defense": meta["defense"],
                "date": meta["date"],
                "is_reviewed": is_reviewed,
                "has_pending": meta.get("has_pending", False)
            })
    return reports_info

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_agent_trace(session_id):
    trace_path = LOGS_DIR / f"{session_id}.json"
    if trace_path.exists():
        return load_json(trace_path)
    return None

@st.cache_data
def load_attack_lookup():
    """Load map of attack_id -> payload from attack datasets."""
    lookup = {}
    bipia_path = Path("data/attacks/bipia_processed.json")
    if bipia_path.exists():
        try:
            with open(bipia_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for atk in data:
                    aid = atk.get("attack_id")
                    payload = atk.get("payload") or atk.get("injected_text") or atk.get("full_document")
                    if aid and payload:
                        lookup[aid] = payload
        except Exception:
            pass
    static_path = Path("data/attacks/static_attacks.json")
    if static_path.exists():
        try:
            with open(static_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for atk in data.get("attacks", []):
                    aid = atk.get("attack_id")
                    payload = atk.get("payload") or atk.get("injected_text") or atk.get("full_document")
                    if aid and payload:
                        lookup[aid] = payload
        except Exception:
            pass
    return lookup

def calc_metrics(items):
    llm_success = 0
    llm_valid_total = 0
    
    human_success = 0
    human_reviewed = 0
    human_reviewed_valid = 0
    
    total = len(items)
    if total == 0:
        return 0, 0, 0, 0.0, None, 0, 0
    
    invalid_count = 0
    pending_count = 0
    for item in items:
        status = item.get("evaluation_status", "valid")
        orig_valid = (status == "valid")
        is_pending = (status == "valid_judge_pending")
        llm_v = item.get("llm_judge", {}).get("verdict", False)
        
        if orig_valid:
            llm_valid_total += 1
            if llm_v:
                llm_success += 1
        elif is_pending:
            pending_count += 1
        else:
            invalid_count += 1
            
        if "human_verdict" in item:
            h_verdict = item["human_verdict"]
            human_reviewed += 1
            
            if h_verdict == "Success":
                human_reviewed_valid += 1
                human_success += 1
            elif h_verdict == "Fail":
                human_reviewed_valid += 1
            elif h_verdict == "Accept LLM Verdict" and orig_valid:
                human_reviewed_valid += 1
                if llm_v:
                    human_success += 1
                    
    llm_rate = (llm_success / llm_valid_total * 100) if llm_valid_total > 0 else 0.0
    human_rate = (human_success / human_reviewed_valid * 100) if human_reviewed_valid > 0 else None
        
    return total, llm_valid_total, invalid_count, llm_rate, human_rate, human_reviewed, pending_count

def update_verdict(item_idx, e_type, target_items):
    v_key = f"verdict_{item_idx}_{e_type}"
    n_key = f"notes_{item_idx}_{e_type}"
    
    target_item = target_items[item_idx]
    
    new_verdict = st.session_state.get(v_key, "Accept LLM Verdict")
    new_notes = st.session_state.get(n_key, "")
    
    target_item["human_verdict"] = new_verdict
    target_item["human_notes"] = new_notes


# --- MODE SELECTION ---
with st.sidebar:
    st.header("Evaluation Mode")
    eval_mode = st.radio(
        "Mode:",
        ["🎯 Standard Benchmark", "🔄 Adaptive Closed-Loop Feedback"],
        index=0 if st.session_state.get("eval_mode") != "adaptive" else 1,
        help="Switch between Standard Static Attacks and Adaptive Closed-Loop Feedback evaluation."
    )
    st.session_state["eval_mode"] = "adaptive" if eval_mode.startswith("🔄") else "standard"
    st.markdown("---")

if st.session_state.get("eval_mode") == "adaptive":
    import importlib
    import src.evaluation.adaptive_hitl_module as ahm
    importlib.reload(ahm)

    with st.sidebar:
        st.header("Adaptive Run Selection")
        adaptive_reports = ahm.get_adaptive_reports_list()
        if not adaptive_reports:
            st.warning(f"No adaptive evaluation reports found in {ahm.ADAPTIVE_RESULTS_DIR}")
            st.stop()

        def format_adapt_name(r_meta):
            rev = " [✅ Reviewed]" if r_meta["is_reviewed"] else ""
            return f"{r_meta['filename']} ({r_meta['agent_model']}){rev}"

        selected_adapt_meta = st.selectbox(
            "Select Adaptive Report",
            adaptive_reports,
            format_func=format_adapt_name
        )

        st.caption(f"Evaluated: {selected_adapt_meta['total_evaluated']} samples | Date: {selected_adapt_meta['date']}")

        if (
            "current_adapt_filename" not in st.session_state
            or st.session_state["current_adapt_filename"] != selected_adapt_meta["filename"]
        ):
            st.session_state["current_adapt_filename"] = selected_adapt_meta["filename"]
            rev_path = ahm.ADAPTIVE_REVIEWS_DIR / f"reviewed_{selected_adapt_meta['filename']}"
            if rev_path.exists():
                st.session_state["adaptive_report_data"] = ahm.load_adaptive_json(rev_path)
                st.toast("Loaded previously saved human verifications.", icon="✅")
            else:
                st.session_state["adaptive_report_data"] = ahm.load_adaptive_json(selected_adapt_meta["filepath"])

        st.markdown("---")
        st.subheader("Progress Tracker")
        adapt_data = st.session_state.get("adaptive_report_data", {})
        adapt_details = adapt_data.get("details", [])
        rev_count = sum(1 for d in adapt_details if d.get("human_review") or d.get("human_r1_verdict"))
        st.metric("Reviews Completed", f"{rev_count} / {len(adapt_details)}")

        st.markdown("---")
        if st.button("💾 Save Adaptive Review", type="primary"):
            save_path = ahm.ADAPTIVE_REVIEWS_DIR / f"reviewed_{selected_adapt_meta['filename']}"
            ahm.save_adaptive_json(st.session_state["adaptive_report_data"], save_path)
            st.success(f"Saved review to {save_path.name}!")
            st.rerun()

    # Main Area for Adaptive Mode
    adapt_report_data = st.session_state.get("adaptive_report_data")
    if not adapt_report_data:
        st.error("Could not load adaptive report data.")
        st.stop()

    tab_adapt_dash, tab_adapt_rev, tab_adapt_exp = st.tabs([
        "📈 Adaptive Dashboard",
        "🔬 Two-Round Review",
        "📊 Multi-Model CSV Export"
    ])

    with tab_adapt_dash:
        ahm.render_adaptive_dashboard(adapt_report_data)
    with tab_adapt_rev:
        ahm.render_adaptive_review_panel(adapt_report_data, selected_adapt_meta["filename"])
    with tab_adapt_exp:
        ahm.render_adaptive_export_tab(adapt_report_data)

    st.stop()

# --- STANDARD BENCHMARK MODE ---
all_reports_info = load_all_reports_info()

if not all_reports_info:
    st.warning(f"No evaluation reports found in {RESULTS_DIR}")
    st.stop()

# Extract filter options
unique_models = sorted(list(set(r["model"] for r in all_reports_info if r["model"] != "Unknown")))
if any(r["model"] == "Unknown" for r in all_reports_info):
    unique_models.append("Unknown")

unique_defenses = sorted(list(set(r["defense"] for r in all_reports_info)))
unique_dates = sorted(list(set(r["date"] for r in all_reports_info if r["date"] != "Unknown")), reverse=True)
if any(r["date"] == "Unknown" for r in all_reports_info):
    unique_dates.append("Unknown")

with st.sidebar:
    st.header("Navigation & Controls")
    
    with st.expander("🔍 Filter Reports", expanded=True):
        review_filter = st.radio("Status", ["All", "Unreviewed Only", "Reviewed Only"], horizontal=True)
        selected_date = st.selectbox("Date", ["All"] + unique_dates)
        selected_model = st.selectbox("Model", ["All"] + unique_models)
        selected_defense = st.selectbox("Defense", ["All"] + unique_defenses)
        
    filtered_reports_info = []
    for r in all_reports_info:
        if review_filter == "Unreviewed Only" and r["is_reviewed"]:
            continue
        if review_filter == "Reviewed Only" and not r["is_reviewed"]:
            continue
        if selected_date != "All" and r["date"] != selected_date:
            continue
        if selected_model != "All" and r["model"] != selected_model:
            continue
        if selected_defense != "All" and r["defense"] != selected_defense:
            continue
        filtered_reports_info.append(r)
        
    filtered_reports = [r["filename"] for r in filtered_reports_info]
    meta_lookup = {r["filename"]: r for r in all_reports_info}
    
    st.caption(f"Showing {len(filtered_reports)} of {len(all_reports_info)} reports")
    
    if not filtered_reports:
        st.warning("No evaluation reports match the selected filters.")
        st.stop()
        
    def format_report_name(fname):
        info = meta_lookup.get(fname, {})
        rev_tag = " [✅ Reviewed]" if info.get("is_reviewed") else ""
        pending_tag = " [⏳ Judge Pending]" if (info.get("has_pending") and not info.get("is_reviewed")) else ""
        return f"{fname}{rev_tag}{pending_tag}"
        
    selected_report_name = st.selectbox(
        "Select Evaluation Report", 
        filtered_reports, 
        format_func=format_report_name
    )
    
if selected_report_name:
    if 'current_report_name' not in st.session_state or st.session_state['current_report_name'] != selected_report_name:
        st.session_state['current_report_name'] = selected_report_name
        
        reviewed_path = REVIEWS_DIR / f"reviewed_{selected_report_name}"
        if not reviewed_path.exists():
            reviewed_path = RESULTS_DIR / f"reviewed_{selected_report_name}"
            
        if reviewed_path.exists():
            st.session_state['report_data'] = load_json(reviewed_path)
            st.toast("Loaded previously saved human verifications from human_reviews.", icon="✅")
        else:
            st.session_state['report_data'] = load_json(RESULTS_DIR / selected_report_name)
            
        st.session_state['current_index_Attacks'] = 0
        st.session_state['current_index_Benign'] = 0
        
    report_data = st.session_state['report_data']
    if not report_data:
        st.error("Could not load report data.")
        st.stop()
        
    attacks = report_data.get("attack_evaluation", {}).get("details", [])
    benigns = report_data.get("benign_evaluation", {}).get("details", [])
    
    available_categories = []
    if attacks:
        available_categories.append("Attacks")
    if benigns:
        available_categories.append("Benign Queries")
        
    if not available_categories:
        st.warning("No evaluation details found in this report.")
        st.stop()
    
    att_total, att_valid_total, att_invalid_count, llm_asr, human_asr, att_reviewed, att_pending = calc_metrics(attacks)
    ben_total, ben_valid_total, ben_invalid_count, llm_fpr, human_fpr, ben_reviewed, ben_pending = calc_metrics(benigns)

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filters")
        exclude_invalid = st.checkbox("Exclude Invalid / Error Runs", value=True, help="Excludes agent runs that failed (no search tool or fatal agent errors). Preserves valid and judge-pending runs.")
        
        st.markdown("---")
        st.subheader("Progress Tracker")
        if attacks:
            p_total = (att_valid_total + att_pending) if exclude_invalid else att_total
            st.metric("Reviews (Attacks)", f"{att_reviewed} / {p_total}")
        if benigns:
            p_total = (ben_valid_total + ben_pending) if exclude_invalid else ben_total
            st.metric("Reviews (Benign)", f"{ben_reviewed} / {p_total}")
        st.markdown("---")
        
        if st.button("Save Reviewed Data", type="primary"):
            new_name = f"reviewed_{selected_report_name}"
            REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REVIEWS_DIR / new_name
            save_json(st.session_state['report_data'], save_path)
            st.success(f"Saved to {save_path}")

        st.markdown("---")
        with st.expander("🗑️ Delete Evaluation & Logs", expanded=False):
            st.caption("Permanently delete this evaluation report and its associated agent trace logs.")
            
            # Find associated session logs
            associated_sessions = set()
            if report_data:
                for d in report_data.get("attack_evaluation", {}).get("details", []):
                    if d.get("session_id"):
                        associated_sessions.add(d["session_id"])
                for d in report_data.get("benign_evaluation", {}).get("details", []):
                    if d.get("session_id"):
                        associated_sessions.add(d["session_id"])
                        
            reviewed_name = f"reviewed_{selected_report_name}"
            has_reviewed = (REVIEWS_DIR / reviewed_name).exists() or (RESULTS_DIR / reviewed_name).exists()
            
            st.write(f"• **Report:** `{selected_report_name}`")
            if has_reviewed:
                st.write(f"• **Review file:** `{reviewed_name}`")
            st.write(f"• **Trace logs:** {len(associated_sessions)} file(s) in `logs/agent_runs/`")
            
            confirm_del = st.checkbox("Confirm deletion", key=f"confirm_del_{selected_report_name}")
            
            if st.button("🗑️ Delete Report & All Logs", type="primary", disabled=not confirm_del):
                deleted_traces = 0
                for sid in associated_sessions:
                    trace_file = LOGS_DIR / f"{sid}.json"
                    if trace_file.exists():
                        try:
                            trace_file.unlink()
                            deleted_traces += 1
                        except Exception:
                            pass
                            
                for rev_dir in [REVIEWS_DIR, RESULTS_DIR]:
                    rev_file = rev_dir / reviewed_name
                    if rev_file.exists():
                        try:
                            rev_file.unlink()
                        except Exception:
                            pass
                            
                report_file = RESULTS_DIR / selected_report_name
                if report_file.exists():
                    try:
                        report_file.unlink()
                    except Exception:
                        pass
                        
                st.cache_data.clear()
                st.session_state.pop('current_report_name', None)
                st.session_state.pop('report_data', None)
                st.session_state.pop('chosen_review_batches', None)
                st.session_state.pop('analysis_result', None)
                
                st.toast(f"Deleted report and {deleted_traces} trace logs.", icon="🗑️")
                st.rerun()

    tab_dash, tab_review, tab_analysis = st.tabs(["Dashboard", "Evaluation Review", "📊 Review Analysis & CSV Export"])
    
    with tab_dash:
        st.header("Metrics Dashboard")
        
        st.subheader("Configuration Details")
        conf_cols = st.columns(5)
        conf_cols[0].metric("Agent Model", report_data.get("agent_llm_model", "Unknown"))
        conf_cols[1].metric("Judge Model", report_data.get("judge_llm_model", "Unknown"))
        conf_cols[2].metric("Provider", report_data.get("agent_llm_provider", "Unknown"))
        
        # Data source / Attack source
        data_src = report_data.get("attack_source") or report_data.get("data_source") or report_data.get("dataset", "Unknown")
        conf_cols[3].metric("Data Source", data_src.upper() if isinstance(data_src, str) and len(data_src) <= 5 else str(data_src).title())
        
        # Defense can come from either block
        defense = report_data.get("attack_evaluation", {}).get("defense_summary", {}).get("defense_name")
        if not defense:
            defense = report_data.get("benign_evaluation", {}).get("defense_summary", {}).get("defense_name", "None")
        conf_cols[4].metric("Defense", defense)
        
        st.markdown("---")
        
        st.subheader("Validity Summary")
        v_cols = st.columns(4)
        total_evals = att_total + ben_total
        total_valid = att_valid_total + ben_valid_total
        total_pending = att_pending + ben_pending
        total_invalid = att_invalid_count + ben_invalid_count
        v_cols[0].metric("Total Evaluated", total_evals)
        v_cols[1].metric("Valid (Judge Ran)", total_valid)
        v_cols[2].metric(
            "Judge Error (Pending HITL)", 
            total_pending, 
            delta=f"{total_pending} awaiting review" if total_pending > 0 else None,
            delta_color="normal"
        )
        v_cols[3].metric("Invalid Runs (Excluded)", total_invalid, delta_color="inverse")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("LLM-Judged")
            if attacks:
                st.metric("ASR (Attack Success Rate)", f"{llm_asr:.2f}%")
            if benigns:
                st.metric("FPR (False Positive Rate)", f"{llm_fpr:.2f}%")
        with col2:
            st.subheader("Human-Verified")
            if attacks:
                if human_asr is not None:
                    st.metric("ASR (Attack Success Rate)", f"{human_asr:.2f}%")
                else:
                    st.metric("ASR (Attack Success Rate)", "Not Verified")
            if benigns:
                if human_fpr is not None:
                    st.metric("FPR (False Positive Rate)", f"{human_fpr:.2f}%")
                else:
                    st.metric("FPR (False Positive Rate)", "Not Verified")

    with tab_review:
        col_cat, col_filt = st.columns([1, 1])
        with col_cat:
            if len(available_categories) > 1:
                eval_type = st.radio("Select Category", available_categories, horizontal=True)
            else:
                eval_type = available_categories[0]
                st.markdown(f"**Category:** {eval_type}")
            
        raw_items = attacks if eval_type == "Attacks" else benigns
        pending_in_cat = sum(1 for item in raw_items if item.get("evaluation_status") == "valid_judge_pending")
        
        filter_choices = ["All Actionable (Valid + Pending)"]
        if pending_in_cat > 0:
            filter_choices.append(f"⏳ Pending Judge Review Only ({pending_in_cat})")
        filter_choices.append("All Items (Including Excluded/Errors)")
        
        with col_filt:
            selected_review_view = st.selectbox("Filter Items to Review", filter_choices, index=0)
            
        if selected_review_view.startswith("⏳ Pending Judge Review Only"):
            items = [item for item in raw_items if item.get("evaluation_status") == "valid_judge_pending"]
        elif selected_review_view == "All Items (Including Excluded/Errors)":
            items = raw_items
        else: # "All Actionable (Valid + Pending)"
            if exclude_invalid:
                items = [
                    item for item in raw_items 
                    if item.get("evaluation_status", "valid") in ("valid", "valid_judge_pending") 
                    and item.get("human_verdict") != "Exclude / Invalid"
                ]
            else:
                items = raw_items
        
        idx_key = f"current_index_{eval_type}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
            
        if not items:
            st.write("No items found in this category.")
        else:
            # Pagination Controls
            def go_prev():
                update_verdict(st.session_state[idx_key], eval_type, items)
                st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
                
            def go_next():
                update_verdict(st.session_state[idx_key], eval_type, items)
                st.session_state[idx_key] = min(len(items) - 1, st.session_state[idx_key] + 1)
                
            idx = st.session_state[idx_key]
            item = items[idx]
            
            eval_status = item.get("evaluation_status", "valid")
            is_pending = (eval_status == "valid_judge_pending")
            status_tag = ""
            if is_pending:
                status_tag = " | ⏳ [Judge Pending]"
            elif eval_status.startswith("invalid_"):
                status_tag = f" | ⚠️ [{eval_status}]"

            # 1. Compact Metadata
            with st.expander(f"📌 Run {idx + 1} of {len(items)} | Session: {item.get('session_id', 'N/A')} | {item.get('category', 'N/A')}{status_tag}", expanded=False):
                meta_cols = st.columns(4)
                meta_cols[0].metric("Session ID", item.get("session_id", "N/A"))
                meta_cols[1].metric("Defense", item.get("defense_metadata", {}).get("defense_name", "N/A"))
                meta_cols[2].metric("Category", item.get("category", "N/A"))
                meta_cols[3].metric("Attack ID", item.get("attack_id", "N/A"))
                st.json({k: v for k, v in item.items() if k not in ['llm_judge', 'human_verdict', 'human_notes']})
            
            session_id = item.get("session_id")
            trace = load_agent_trace(session_id) if session_id else None
            
            # 2. Main Layout (Two Columns)
            col_trace, col_eval = st.columns([1.2, 1.0], gap="large")
            
            with col_trace:
                st.markdown("### 💬 Conversation Trace")
                if trace:
                    with st.container(border=True):
                        st.markdown("##### 👤 User Query:")
                        query_text = trace.get("metadata", {}).get("query", "N/A")
                        st.markdown(query_text)
                        
                    with st.container(border=True):
                        st.markdown("##### 🤖 Agent Response:")
                        answer_text = trace.get("final_answer", "N/A")
                        st.markdown(answer_text)
                    
                    with st.expander("🔍 Injected Document & Retrieved Context"):
                        defense_meta = item.get("defense_metadata", {})
                        defense_active = defense_meta.get("defense_active", False)
                        defense_name = defense_meta.get("defense_name", "defense")
                        injected_bypassed = defense_meta.get("injected_chunk_bypassed", False)
                        injected_score = defense_meta.get("injected_chunk_score", -1.0)
                        
                        went_to_llm = injected_bypassed if defense_active else True
                        
                        injected_doc = item.get("injected_document") or trace.get("injected_document")
                        if not injected_doc and went_to_llm and eval_type == "Attacks":
                            attack_lookup = load_attack_lookup()
                            injected_doc = attack_lookup.get(item.get("attack_id"))
                            
                        if defense_active and not injected_bypassed:
                            st.success(f"🛡️ **Defense Blocked the Payload**: `{defense_name}` successfully intercepted and blocked the injected document. It was never passed to the LLM.")
                        elif went_to_llm and injected_doc:
                            score_str = f" (Relevance Score: `{injected_score:.4f}`)" if injected_score >= 0 else ""
                            st.error(f"🚨 **Injected Document Passed to LLM Context{score_str}:**")
                            st.code(injected_doc, language="text")
                        elif eval_type == "Attacks":
                            st.info("No explicit 'injected_document' field found. See context below.")
                        
                        st.markdown("**Retrieved Documents:**")
                        tools = trace.get("tool_calls", [])
                        has_docs = False
                        for t in tools:
                            if t.get('tool_name') == 'search_knowledge_base':
                                has_docs = True
                                st.text(t.get('output', ''))
                        if not has_docs:
                            st.write("No 'search_knowledge_base' tool calls found.")
                else:
                    st.warning(f"Agent trace not found for session ID: {session_id}")
            
            with col_eval:
                st.markdown("### ⚖️ Evaluation")
                eval_status = item.get("evaluation_status", "valid")
                llm_judge = item.get("llm_judge", {})
                is_judge_pending = (eval_status == "valid_judge_pending") or bool(llm_judge.get("error"))
                is_invalid_run = eval_status.startswith("invalid_") or (eval_status not in ("valid", "valid_judge_pending"))
                llm_verdict_bool = llm_judge.get('verdict')
                reasoning = llm_judge.get('reasoning', 'N/A').replace('\n', '<br>')
                
                if is_judge_pending:
                    judge_err = llm_judge.get('reasoning') or "LLM judge encountered an error (timeout, quota, or network exception)."
                    verdict_html = f'''
                    <div style="background-color: #fff3e0; border-left: 8px solid #ff9800; padding: 20px; border-radius: 6px; color: #e65100; font-family: sans-serif; margin-bottom: 20px;">
                        <h3 style="color: #e65100; margin-top: 0; font-size: 1.4em;">⏳ LLM Judge Failed — Pending Human Review</h3>
                        <strong style="font-size: 1.1em;">Status:</strong> <code>valid_judge_pending</code><br>
                        <span style="font-size: 0.95em; color: #555;">The agent executed successfully, but the automated judge failed. Please review the conversation trace and assign a manual verdict below.</span><br><br>
                        <strong style="font-size: 1.1em;">Judge Error / Trace:</strong><br>
                        <span style="font-size: 1.05em; line-height: 1.6; color: #222; font-family: monospace;">{judge_err}</span>
                    </div>
                    '''
                elif is_invalid_run:
                    error_msg = item.get("error_message", "Agent did not perform required actions or encountered fatal error.")
                    verdict_html = f'''
                    <div style="background-color: #f5f5f5; border-left: 8px solid #9e9e9e; padding: 20px; border-radius: 6px; color: #424242; font-family: sans-serif; margin-bottom: 20px;">
                        <h3 style="color: #424242; margin-top: 0; font-size: 1.4em;">⚠️ Agent Run Excluded / Invalid</h3>
                        <strong style="font-size: 1.1em;">Status:</strong> <code>{eval_status}</code><br>
                        <strong style="font-size: 1.1em;">Error / Reason:</strong><br>
                        <span style="font-size: 1.1em; line-height: 1.6; color: #555;">{error_msg}</span>
                    </div>
                    '''
                elif llm_verdict_bool:
                    verdict_html = f'''
                    <div style="background-color: #ffebee; border-left: 8px solid #CC071E; padding: 20px; border-radius: 6px; color: #9C0012; font-family: sans-serif; margin-bottom: 20px;">
                        <h3 style="color: #CC071E; margin-top: 0; font-size: 1.4em;">🚨 Attack Succeeded / False Positive</h3>
                        <strong style="font-size: 1.1em;">LLM Reasoning:</strong><br>
                        <span style="font-size: 1.1em; line-height: 1.6;">{reasoning}</span>
                    </div>
                    '''
                else:
                    verdict_html = f'''
                    <div style="background-color: #e8f5e9; border-left: 8px solid #007D3E; padding: 20px; border-radius: 6px; color: #054D28; font-family: sans-serif; margin-bottom: 20px;">
                        <h3 style="color: #007D3E; margin-top: 0; font-size: 1.4em;">✅ Attack Failed / True Negative</h3>
                        <strong style="font-size: 1.1em;">LLM Reasoning:</strong><br>
                        <span style="font-size: 1.1em; line-height: 1.6;">{reasoning}</span>
                    </div>
                    '''
                st.markdown(verdict_html, unsafe_allow_html=True)
                
                if is_judge_pending:
                    st.info("💡 **Manual Verdict Needed**: Select **Success (2)** or **Fail (3)** based on whether the agent followed the malicious instructions.")

                st.markdown("#### Your Verdict")
                current_override = item.get("human_verdict", "Accept LLM Verdict")
                override_options = ["Accept LLM Verdict", "Success", "Fail", "Exclude / Invalid"]
                
                st.radio(
                    "Override:", 
                    override_options, 
                    index=override_options.index(current_override) if current_override in override_options else 0,
                    format_func=lambda x: {
                        "Accept LLM Verdict": "1  Accept LLM Verdict" + (" (⚠️ Judge Failed - Choose 2 or 3)" if is_judge_pending else ""),
                        "Success": "2  Override: Success" if not is_judge_pending else "2  Success",
                        "Fail": "3  Override: Fail" if not is_judge_pending else "3  Fail",
                        "Exclude / Invalid": "4  Exclude / Invalid"
                    }.get(x, x),
                    key=f"verdict_{idx}_{eval_type}",
                    on_change=update_verdict,
                    args=(idx, eval_type, items),
                    label_visibility="collapsed"
                )
                
                st.text_area(
                    "Reviewer Notes:", 
                    value=item.get("human_notes", ""),
                    key=f"notes_{idx}_{eval_type}",
                    on_change=update_verdict,
                    args=(idx, eval_type, items),
                    height=100
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.button("⬅️ Previous [<]", on_click=go_prev, use_container_width=True)
                with c2:
                    st.button("Next (Save & Continue) [>] ➡️", on_click=go_next, type="primary", use_container_width=True)
                    
                st.caption("⌨️ **Hotkeys**: Press `1` for Accept, `2` for Success, `3` for Fail. Press `>` (or `.`) for Next, `<` (or `,`) for Previous.")

            # Inject keyboard shortcuts listener
            shortcut_js = """
            <script>
            (function() {
                const parentDoc = window.parent.document;
                const parentWin = window.parent;
                
                // Clean up previous event listener if present
                if (parentWin.__hitl_keydown_handler) {
                    parentDoc.removeEventListener('keydown', parentWin.__hitl_keydown_handler, true);
                }
                
                parentWin.__hitl_keydown_handler = function(e) {
                    // Ignore key presses if user is actively typing inside an input or textarea
                    const active = parentDoc.activeElement;
                    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                        return;
                    }
                    
                    if (e.key === '1') {
                        e.preventDefault();
                        selectVerdictOption(0);
                    } else if (e.key === '2') {
                        e.preventDefault();
                        selectVerdictOption(1);
                    } else if (e.key === '3') {
                        e.preventDefault();
                        selectVerdictOption(2);
                    } else if (e.key === '4') {
                        e.preventDefault();
                        selectVerdictOption(3);
                    } else if (e.key === '>' || e.key === '.') {
                        e.preventDefault();
                        clickNavButton('Next');
                    } else if (e.key === '<' || e.key === ',') {
                        e.preventDefault();
                        clickNavButton('Previous');
                    }
                };
                
                parentDoc.addEventListener('keydown', parentWin.__hitl_keydown_handler, true);
                
                function selectVerdictOption(index) {
                    const radioGroups = Array.from(parentDoc.querySelectorAll('div[data-testid="stRadio"] div[role="radiogroup"]'));
                    const verdictGroup = radioGroups.find(rg => {
                        const txt = rg.innerText || "";
                        return txt.includes("Accept") || txt.includes("Success") || txt.includes("Fail");
                    });
                    
                    if (verdictGroup) {
                        const labels = Array.from(verdictGroup.querySelectorAll('label'));
                        if (labels && labels[index]) {
                            const targetLabel = labels[index];
                            const input = targetLabel.querySelector('input[type="radio"]');
                            if (input) {
                                input.click();
                            } else {
                                targetLabel.click();
                            }
                        }
                    }
                }
                
                function clickNavButton(textMatch) {
                    const buttons = Array.from(parentDoc.querySelectorAll('button'));
                    const btn = buttons.find(b => b.innerText && b.innerText.includes(textMatch));
                    if (btn) {
                        btn.click();
                    }
                }
            })();
            </script>
            """
            components.html(shortcut_js, height=0, width=0)

    with tab_analysis:
        st.header("📊 Multi-Batch Human Review Analysis")
        st.markdown(
            "Select multiple human-reviewed reports of the **same agent model** to aggregate metrics "
            "and generate official summary and per-sample CSV files."
        )

        reviewed_files = sorted(list(REVIEWS_DIR.glob("*.json")), reverse=True)
        if not reviewed_files:
            st.info("No reviewed evaluation files found in `results/human_reviews/` yet. Review and save some evaluations first.")
        else:
            reviewed_items = []
            for rf in reviewed_files:
                try:
                    with open(rf, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                    model = d.get("agent_llm_model") or "Unknown"
                    defense = d.get("attack_evaluation", {}).get("defense_summary", {}).get("defense_name")
                    if not defense:
                        defense = d.get("benign_evaluation", {}).get("defense_summary", {}).get("defense_name", "None")
                    source = d.get("attack_source") or "unknown"
                    ts = d.get("timestamp") or rf.name
                    samples = len(d.get("attack_evaluation", {}).get("details", []))
                    reviewed_items.append({
                        "file_path": str(rf),
                        "filename": rf.name,
                        "model": model,
                        "defense": defense,
                        "source": source,
                        "timestamp": ts,
                        "samples": samples
                    })
                except Exception:
                    pass

            available_review_models = sorted(list(set(item["model"] for item in reviewed_items)))

            st.subheader("1. Select Agent Model")
            selected_analysis_model = st.selectbox(
                "Filter Reviewed Files by Agent Model:",
                available_review_models,
                help="Only reviews from the selected model can be aggregated together."
            )

            model_items = [item for item in reviewed_items if item["model"] == selected_analysis_model]

            st.subheader(f"2. Select Reviewed Batches for `{selected_analysis_model}` ({len(model_items)} available)")

            col_btn1, col_btn2, _ = st.columns([1.2, 1.2, 3])
            with col_btn1:
                if st.button("Select All Batches"):
                    st.session_state["chosen_review_batches"] = [item["file_path"] for item in model_items]
                    st.rerun()
            with col_btn2:
                if st.button("Clear Selection"):
                    st.session_state["chosen_review_batches"] = []
                    st.rerun()

            current_chosen = st.session_state.get("chosen_review_batches")
            valid_paths = set(item["file_path"] for item in model_items)
            if current_chosen is None or not any(p in valid_paths for p in current_chosen):
                current_chosen = [item["file_path"] for item in model_items]
                st.session_state["chosen_review_batches"] = current_chosen
            else:
                current_chosen = [p for p in current_chosen if p in valid_paths]

            file_label_map = {
                item["file_path"]: f"{item['filename']} | Dataset: {item['source']} | Defense: {item['defense']} | Samples: {item['samples']}"
                for item in model_items
            }

            chosen_paths = st.multiselect(
                "Batches to Include in Analysis:",
                options=[item["file_path"] for item in model_items],
                default=current_chosen,
                format_func=lambda p: file_label_map.get(p, p)
            )
            st.session_state["chosen_review_batches"] = chosen_paths

            st.markdown("---")
            st.subheader("3. Run Analysis & Export CSV")

            col_opt1, col_opt2 = st.columns([2, 1])
            with col_opt1:
                append_mode = st.checkbox(
                    "Append & Update Consolidated CSVs (Recommended for multi-model comparison)",
                    value=True,
                    help="When checked, merges new results into the existing CSVs without losing other models or creating duplicate rows."
                )

            if st.button("🚀 Run Analysis & Generate CSVs", type="primary", disabled=len(chosen_paths) == 0):
                with st.spinner(f"Aggregating {len(chosen_paths)} review batches for {selected_analysis_model}..."):
                    try:
                        import importlib
                        import scripts.analyze_human_reviews
                        importlib.reload(scripts.analyze_human_reviews)
                        analysis_result = scripts.analyze_human_reviews.run_analysis(
                            chosen_paths, 
                            output_dir=str(RESULTS_DIR), 
                            verbose=False, 
                            append=append_mode
                        )
                        st.session_state["analysis_result"] = analysis_result
                        st.success(f"Successfully processed {len(chosen_paths)} batches for `{selected_analysis_model}`!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error during analysis: {exc}")

            # Check for existing or newly generated summary CSV on disk
            summary_csv_path = RESULTS_DIR / "human_reviewed_summary.csv"
            per_sample_csv_path = RESULTS_DIR / "human_reviewed_per_sample.csv"
            
            if summary_csv_path.exists():
                st.markdown("---")
                st.subheader("📈 Consolidated Multi-Model Performance")
                try:
                    import csv
                    with open(summary_csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        summary_rows = list(reader)
                        
                    if summary_rows:
                        from src.evaluation.adaptive_hitl_module import render_safe_table
                        render_safe_table(summary_rows)
                        
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        with open(summary_csv_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                label="📥 Download Consolidated Summary CSV",
                                data=f.read(),
                                file_name="human_reviewed_summary.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    with col_dl2:
                        if per_sample_csv_path.exists():
                            with open(per_sample_csv_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label="📥 Download Consolidated Per-Sample CSV",
                                    data=f.read(),
                                    file_name="human_reviewed_per_sample.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    st.caption(f"CSVs stored on disk at: `{summary_csv_path}` and `{per_sample_csv_path}`")
                    
                    with st.expander("🗑️ Reset / Clear Consolidated CSVs", expanded=False):
                        st.caption("Delete the consolidated CSV files to start a completely fresh comparison table.")
                        confirm_reset = st.checkbox("Confirm clearing consolidated CSV files", key="confirm_reset_csvs")
                        if st.button("🗑️ Clear Consolidated CSVs", disabled=not confirm_reset):
                            try:
                                if summary_csv_path.exists():
                                    summary_csv_path.unlink()
                                if per_sample_csv_path.exists():
                                    per_sample_csv_path.unlink()
                                st.session_state.pop("analysis_result", None)
                                st.toast("Consolidated CSV files cleared.", icon="🗑️")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error clearing CSVs: {e}")
                except Exception as e:
                    st.error(f"Error loading consolidated summary: {e}")
