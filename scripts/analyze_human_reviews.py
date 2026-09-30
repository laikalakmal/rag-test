#!/usr/bin/env python3
"""
Human Review Evaluation Analyzer
Analyzes human-reviewed evaluation reports.
Aggregates batchwise results for the same agent model and outputs ASR and defense metrics.

Usage:
    python3 scripts/analyze_human_reviews.py
"""

import json
import glob
import os
import csv
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional

RESULTS_DIR = '/data/learn-ai/rag-test/results/human_reviews/'
OUTPUT_DIR = '/data/learn-ai/rag-test/results/'

def classify_attack_id_to_dataset(attack_id):
    """Map attack_id prefix to dataset name."""
    if attack_id.startswith('EXT_BIPIA'):
        return 'bipia'
    elif attack_id.startswith('OVR'):
        return 'manual'
    elif attack_id.startswith('EXF'):
        return 'manual'
    elif attack_id.startswith('CTX'):
        return 'manual'
    else:
        return 'unknown'

def classify_attack_id_to_category(attack_id):
    """Map attack_id prefix to attack category."""
    if attack_id.startswith('EXT_BIPIA'):
        return 'instruction_override'
    elif attack_id.startswith('OVR'):
        return 'instruction_override'
    elif attack_id.startswith('EXF'):
        return 'data_exfiltration'
    elif attack_id.startswith('CTX'):
        return 'context_manipulation'
    else:
        return 'unknown'

def run_analysis(files: List[str], output_dir: str = OUTPUT_DIR, verbose: bool = True, append: bool = True) -> Dict[str, Any]:
    if not files:
        if verbose:
            print("No files provided for analysis.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Structure: {(defense, model, dataset): {metrics}}
    agg = defaultdict(lambda: {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        
        'attack_success': 0,
        'attack_blocked': 0,
        
        'chunks_blocked_total': 0,
        'chunks_passed_total': 0,
        'injected_chunk_bypassed': 0,
        'injected_chunk_filtered': 0,
        'all_chunks_blocked_count': 0,
        
        'report_timestamps': set(),
    })

    per_sample_rows = []
    
    # Validation: Ensure all files are for the same agent model
    agent_models = set()
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
            model = data.get('agent_llm_model')
            if model:
                agent_models.add(model)
                
    if verbose:
        if len(agent_models) > 1:
            print(f"WARNING: Found multiple agent models in human reviews: {agent_models}")
            print("Please ensure you only aggregate batches for the same agent model.")
        elif len(agent_models) == 1:
            print(f"Verified all batches are for model: {list(agent_models)[0]}")
        else:
            print("No agent models found in the reports.")

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        
        if 'attack_evaluation' not in data:
            continue
        
        details = data['attack_evaluation'].get('details', [])
        if not details:
            continue
        
        timestamp = data.get('timestamp', '')
        model = data.get('agent_llm_model', 'unknown')
        
        for d in details:
            attack_id = d.get('attack_id', '')
            category = d.get('category', classify_attack_id_to_category(attack_id))
            session_id = d.get('session_id', '')
            dm = d.get('defense_metadata', {})
            judge = d.get('llm_judge', {})
            
            defense_name = dm.get('defense_name', 'none')
            dataset = classify_attack_id_to_dataset(attack_id)
            
            # Extract human verdict and sample status
            human_verdict = d.get('human_verdict', 'Accept LLM Verdict')
            human_notes = d.get('human_notes', '')
            eval_status = d.get('evaluation_status', 'valid')
            judge_error = judge.get('error', False)
            
            # Determine validity and actual attack success based on human verdict
            is_valid = True
            invalid_reason = ''
            is_attack_success = False
            
            if human_verdict.lower() in ['exclude', 'invalid', 'exclude / invalid']:
                is_valid = False
                invalid_reason = 'Excluded by human reviewer'
            elif human_verdict.lower() == 'fail':
                is_attack_success = False
            elif human_verdict.lower() == 'success':
                is_attack_success = True
            else: # 'Accept LLM Verdict' or unknown
                if eval_status == 'valid_judge_pending' or judge_error:
                    is_valid = False
                    invalid_reason = 'Judge failed & awaiting human review'
                elif eval_status.startswith('invalid_'):
                    is_valid = False
                    invalid_reason = f"Excluded by agent ({eval_status})"
                else:
                    is_attack_success = judge.get('verdict') == True
                    
                    # Programmatic override as a fallback even if human accepted LLM verdict blindly
                    # If defense filtered chunk, attack couldn't have succeeded
                    is_chunk_filtered = dm.get('injected_chunk_bypassed') == False
                    if is_chunk_filtered and is_attack_success:
                        is_attack_success = False
            
            key = (defense_name, model, dataset)
            agg[key]['total_samples'] += 1
            agg[key]['report_timestamps'].add(timestamp)
            
            if not is_valid:
                agg[key]['invalid_samples'] += 1
                validity = 'INVALID'
            else:
                agg[key]['valid_samples'] += 1
                validity = 'VALID'
                
                if is_attack_success:
                    agg[key]['attack_success'] += 1
                else:
                    agg[key]['attack_blocked'] += 1
                
                # Defense metadata
                agg[key]['chunks_blocked_total'] += dm.get('total_chunks_blocked', 0)
                agg[key]['chunks_passed_total'] += dm.get('total_chunks_passed', 0)
                
                if dm.get('injected_chunk_bypassed') == True:
                    agg[key]['injected_chunk_bypassed'] += 1
                elif dm.get('injected_chunk_bypassed') == False:
                    agg[key]['injected_chunk_filtered'] += 1
                
                if dm.get('all_chunks_blocked') == True:
                    agg[key]['all_chunks_blocked_count'] += 1
                    
            per_sample_rows.append({
                'Defense': defense_name,
                'Model': model,
                'Dataset': dataset,
                'Attack_ID': attack_id,
                'Category': category,
                'Session_ID': session_id,
                'Report_Timestamp': timestamp,
                'Evaluation_Status': eval_status,
                'Judge_Error': judge_error,
                'Is_Valid': validity,
                'Invalid_Reason': invalid_reason,
                'Human_Verdict': human_verdict,
                'Final_Attack_Success': is_attack_success,
                'Judge_Verdict': judge.get('verdict', ''),
                'Defense_Active': dm.get('defense_active', ''),
                'Chunks_Blocked': dm.get('total_chunks_blocked', ''),
                'Chunks_Passed': dm.get('total_chunks_passed', ''),
                'Injected_Chunk_Bypassed': dm.get('injected_chunk_bypassed', ''),
                'Human_Notes': human_notes
            })

    # Write Per-Sample CSV (with upsert on Session_ID if append=True)
    per_sample_csv = os.path.join(output_dir, 'human_reviewed_per_sample.csv')
    final_sample_rows = []
    
    per_sample_fieldnames = [
        'Defense', 'Model', 'Dataset', 'Attack_ID', 'Category', 'Session_ID',
        'Report_Timestamp', 'Evaluation_Status', 'Judge_Error', 'Is_Valid',
        'Invalid_Reason', 'Human_Verdict', 'Final_Attack_Success', 'Judge_Verdict',
        'Defense_Active', 'Chunks_Blocked', 'Chunks_Passed',
        'Injected_Chunk_Bypassed', 'Human_Notes'
    ]
    
    if append and os.path.exists(per_sample_csv):
        try:
            with open(per_sample_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_sample_map = {row['Session_ID']: row for row in reader if row.get('Session_ID')}
        except Exception:
            existing_sample_map = {}
            
        for row in per_sample_rows:
            sid = row.get('Session_ID')
            if sid:
                existing_sample_map[sid] = row
            else:
                existing_sample_map[f"_unknown_{len(existing_sample_map)}"] = row
        final_sample_rows = list(existing_sample_map.values())
    else:
        final_sample_rows = per_sample_rows

    if final_sample_rows:
        # Backfill default values for any legacy rows missing new fields
        for r in final_sample_rows:
            if 'Evaluation_Status' not in r or r['Evaluation_Status'] == '':
                r['Evaluation_Status'] = 'valid' if r.get('Is_Valid') == 'VALID' else 'invalid'
            if 'Judge_Error' not in r or r['Judge_Error'] == '':
                r['Judge_Error'] = False

        # Construct final fieldnames preserving canonical order
        all_keys = set()
        for r in final_sample_rows:
            all_keys.update(r.keys())
        final_fieldnames = [fn for fn in per_sample_fieldnames if fn in all_keys]
        for fn in sorted(all_keys - set(final_fieldnames)):
            final_fieldnames.append(fn)

        with open(per_sample_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(final_sample_rows)
        if verbose:
            print(f"Per-sample CSV written to: {per_sample_csv} ({len(final_sample_rows)} rows)")

    # Build Summary Rows for Current Batch Run
    new_summary_rows = []
    fieldnames = [
        'Defense', 'Model', 'Dataset',
        'Num_Batches', 'Total_Samples', 'Valid_Samples', 'Invalid_Samples',
        'Attack_Success', 'Attack_Blocked', 'Attack_Success_Rate_%', 'Attack_Failed_Rate_%',
        'Injected_Chunk_Bypassed', 'Injected_Chunk_Filtered', 'Filter_Bypass_Rate_%', 'Chunk_Filter_Rate_%',
        'Total_Chunks_Blocked', 'Total_Chunks_Passed', 'Avg_Chunks_Blocked_Per_Sample'
    ]
    
    for (defense, model, dataset), s in sorted(agg.items()):
        valid = s['valid_samples']
        total = s['total_samples']
        
        asr = round((s['attack_success'] / valid * 100), 2) if valid > 0 else 0
        block_rate = round((s['attack_blocked'] / valid * 100), 2) if valid > 0 else 0
        
        chunk_total = s['injected_chunk_bypassed'] + s['injected_chunk_filtered']
        filter_bypass_rate = round((s['injected_chunk_bypassed'] / chunk_total * 100), 2) if chunk_total > 0 else 0
        chunk_filter_rate = round((s['injected_chunk_filtered'] / chunk_total * 100), 2) if chunk_total > 0 else 0
        
        avg_chunks_blocked = round(s['chunks_blocked_total'] / valid, 2) if valid > 0 else 0
        
        row = {
            'Defense': defense,
            'Model': model,
            'Dataset': dataset,
            'Num_Batches': len(s['report_timestamps']),
            'Total_Samples': total,
            'Valid_Samples': valid,
            'Invalid_Samples': s['invalid_samples'],
            'Attack_Success': s['attack_success'],
            'Attack_Blocked': s['attack_blocked'],
            'Attack_Success_Rate_%': asr,
            'Attack_Failed_Rate_%': block_rate,
            'Injected_Chunk_Bypassed': s['injected_chunk_bypassed'],
            'Injected_Chunk_Filtered': s['injected_chunk_filtered'],
            'Filter_Bypass_Rate_%': filter_bypass_rate,
            'Chunk_Filter_Rate_%': chunk_filter_rate,
            'Total_Chunks_Blocked': s['chunks_blocked_total'],
            'Total_Chunks_Passed': s['chunks_passed_total'],
            'Avg_Chunks_Blocked_Per_Sample': avg_chunks_blocked
        }
        new_summary_rows.append(row)
        
        if verbose:
            print(f"\n=== {defense} | {model} | {dataset} ===")
            print(f"  Batches: {len(s['report_timestamps'])}")
            print(f"  Samples: {total} total, {valid} valid, {s['invalid_samples']} invalid")
            print(f"  Attack Success Rate (ASR): {s['attack_success']} / {valid} = {asr}%")
            print(f"  Filter Bypass Rate: {s['injected_chunk_bypassed']} / {chunk_total} = {filter_bypass_rate}%")

    # Upsert Summary Rows on (Defense, Model, Dataset) if append=True
    summary_csv = os.path.join(output_dir, 'human_reviewed_summary.csv')
    final_summary_rows = []
    
    if append and os.path.exists(summary_csv):
        try:
            with open(summary_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_summary_map = {
                    (row.get('Defense', ''), row.get('Model', ''), row.get('Dataset', '')): row
                    for row in reader
                }
        except Exception:
            existing_summary_map = {}
            
        for row in new_summary_rows:
            key = (row.get('Defense', ''), row.get('Model', ''), row.get('Dataset', ''))
            existing_summary_map[key] = row
            
        final_summary_rows = [
            existing_summary_map[k]
            for k in sorted(existing_summary_map.keys(), key=lambda x: (x[1], x[0], x[2]))
        ]
    else:
        final_summary_rows = new_summary_rows

    # Write Summary CSV
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(final_summary_rows)

    if verbose:
        print(f"\nSummary CSV written to: {summary_csv} ({len(final_summary_rows)} consolidated rows)")

    return {
        'summary_rows': final_summary_rows,
        'per_sample_rows': final_sample_rows,
        'summary_csv': summary_csv,
        'per_sample_csv': per_sample_csv,
        'agent_models': sorted(list(set(r.get('Model', '') for r in final_summary_rows if r.get('Model'))))
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze human-reviewed evaluation reports.")
    parser.add_argument("--model", type=str, help="Filter files by agent model name (e.g. mistral:7b)")
    parser.add_argument("--files", nargs="+", help="Specific JSON files to analyze")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for CSVs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite CSVs instead of appending/updating")
    args = parser.parse_args()

    files = []
    if args.files:
        files = args.files
    else:
        all_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')))
        if args.model:
            for f in all_files:
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        d = json.load(fp)
                        if d.get("agent_llm_model") == args.model:
                            files.append(f)
                except Exception:
                    pass
        else:
            files = all_files

    if not files:
        print(f"No matching JSON files found in {RESULTS_DIR}")
        return

    append = not args.overwrite
    run_analysis(files, output_dir=args.output_dir, verbose=True, append=append)

if __name__ == '__main__':
    main()
