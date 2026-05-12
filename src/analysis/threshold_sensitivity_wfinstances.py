#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis for WfInstances experiments.
Compares different posterior thresholds for edge inference.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from src.utils.result_paths import (
    WFINSTANCES_SRASEARCH_RESULTS_DIR,
    LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
    WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    prefer_existing_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKFLOWS = {
    "srasearch": {
        "data_dir": PROJECT_ROOT / "data" / "wfinstances_srasearch",
        "results_dir": prefer_existing_path(
            WFINSTANCES_SRASEARCH_RESULTS_DIR,
            LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
        ),
        "title": "SRASearch",
    },
    "epigenomics": {
        "data_dir": PROJECT_ROOT / "data" / "wfinstances_epigenomics", 
        "results_dir": prefer_existing_path(
            WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
            LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
        ),
        "title": "Epigenomics",
    },
}

# Thresholds to test (extended range to find FP-minimizing thresholds)
THRESHOLDS = [0.20, 0.25, 0.30, 1/3, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
IP_COV_TARGET = 0.95


def load_workflow_data(workflow_name):
    """Load workflow DAG from WfInstances JSON file."""
    config = WORKFLOWS[workflow_name]
    data_dir = config["data_dir"]
    
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")
    
    filepath = json_files[0]
    with open(filepath) as f:
        data = json.load(f)
    
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    task_ids = [t.get('name', t.get('id')) for t in tasks]
    n = len(task_ids)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    
    cover = np.zeros((n, n), dtype=np.int8)
    for t in tasks:
        name = t.get('name', t.get('id'))
        if name not in task_to_idx:
            continue
        src = task_to_idx[name]
        for child in t.get('children', []):
            if child in task_to_idx:
                cover[src, task_to_idx[child]] = 1
    
    return {
        'task_ids': task_ids,
        'cover': cover,
        'num_tasks': n,
        'num_edges': int(cover.sum()),
    }


def load_avg_H(workflow_name, ip_cov_target=0.95):
    """Load posterior mean H matrix."""
    config = WORKFLOWS[workflow_name]
    results_dir = config["results_dir"]
    
    ip_cov_str = f"ip_cov_{ip_cov_target:.2f}".replace('.', '_')
    exp_dir = results_dir / ip_cov_str
    
    avg_H_file = exp_dir / "avg_H.pkl"
    if avg_H_file.exists():
        with open(avg_H_file, 'rb') as f:
            return pickle.load(f)
    return None


def load_H_trace(workflow_name, ip_cov_target=0.95):
    """Load full H trace for marginal mode computation."""
    config = WORKFLOWS[workflow_name]
    results_dir = config["results_dir"]
    
    ip_cov_str = f"ip_cov_{ip_cov_target:.2f}".replace('.', '_')
    exp_dir = results_dir / ip_cov_str
    
    H_trace_file = exp_dir / "H_trace.pkl"
    if H_trace_file.exists():
        with open(H_trace_file, 'rb') as f:
            return pickle.load(f)
    return None


def transitive_reduction(adj_matrix):
    """Compute transitive reduction of adjacency matrix."""
    n = adj_matrix.shape[0]
    adj = adj_matrix.astype(bool).copy()
    
    # Floyd-Warshall for transitive closure
    closure = adj.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
    
    # Transitive reduction: remove edge (i,j) if there's a path through k
    reduction = closure.copy()
    for i in range(n):
        for j in range(n):
            if closure[i, j]:
                for k in range(n):
                    if k != i and k != j and closure[i, k] and closure[k, j]:
                        reduction[i, j] = False
                        break
    
    return reduction.astype(np.int8)


def compute_metrics(true_cover, inferred_cover):
    """Compute precision, recall, F1, and SHD."""
    true_cover = np.array(true_cover)
    inferred_cover = np.array(inferred_cover)
    
    tp = np.sum((true_cover == 1) & (inferred_cover == 1))
    fp = np.sum((true_cover == 0) & (inferred_cover == 1))
    fn = np.sum((true_cover == 1) & (inferred_cover == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    shd = fp + fn
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def compute_marginal_mode(H_trace, burn_in_fraction=0.5):
    """Compute marginal mode: for each edge, take the most frequent value."""
    if H_trace is None or len(H_trace) == 0:
        return None
    
    burn = int(len(H_trace) * burn_in_fraction)
    post = H_trace[burn:]
    
    if len(post) == 0:
        return None
    
    # Stack all H matrices
    H_stack = np.stack(post)
    
    # For each edge, count 1s vs 0s
    n = H_stack.shape[1]
    marginal_mode = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ones = np.sum(H_stack[:, i, j])
                zeros = len(post) - ones
                marginal_mode[i, j] = 1 if ones > zeros else 0
    
    return marginal_mode


def main():
    print("=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS (WfInstances, IP-Cov=0.95)")
    print("=" * 80)
    
    # Collect results for each workflow
    all_results = {t: {'f1': [], 'shd': [], 'fp': [], 'fn': [], 'precision': [], 'recall': []} for t in THRESHOLDS}
    all_results['marginal_mode'] = {'f1': [], 'shd': [], 'fp': [], 'fn': [], 'precision': [], 'recall': []}
    
    for wf_name, config in WORKFLOWS.items():
        print(f"\n--- {config['title']} ---")
        
        # Load data
        wf_data = load_workflow_data(wf_name)
        true_cover = wf_data['cover']
        avg_H = load_avg_H(wf_name, IP_COV_TARGET)
        H_trace = load_H_trace(wf_name, IP_COV_TARGET)
        
        if avg_H is None:
            print(f"  No avg_H found for {wf_name}")
            continue
        
        avg_H = np.array(avg_H)
        
        print(f"  True edges: {wf_data['num_edges']}")
        print(f"  Avg H range: [{avg_H.min():.3f}, {avg_H.max():.3f}]")
        print()
        
        # Test each threshold
        print(f"  {'Threshold':<12} {'F1':<8} {'Prec':<8} {'Recall':<8} {'FP':<6} {'FN':<6} {'SHD':<6} {'Edges':<6}")
        print("  " + "-" * 72)
        
        for threshold in THRESHOLDS:
            # Threshold then apply transitive reduction (same as experiment)
            thresholded = (avg_H >= threshold).astype(int)
            np.fill_diagonal(thresholded, 0)
            inferred = transitive_reduction(thresholded)
            
            metrics = compute_metrics(true_cover, inferred)
            
            thresh_str = f"{threshold:.2f}" if threshold != 1/3 else "1/3"
            print(f"  {thresh_str:<12} {metrics['f1']:<8.3f} {metrics['precision']:<8.3f} "
                  f"{metrics['recall']:<8.3f} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['shd']:<6} {inferred.sum():<6}")
            
            all_results[threshold]['f1'].append(metrics['f1'])
            all_results[threshold]['shd'].append(metrics['shd'])
            all_results[threshold]['fp'].append(metrics['fp'])
            all_results[threshold]['fn'].append(metrics['fn'])
            all_results[threshold]['precision'].append(metrics['precision'])
            all_results[threshold]['recall'].append(metrics['recall'])
        
        # Marginal mode (also apply transitive reduction)
        if H_trace is not None:
            marginal_raw = compute_marginal_mode(H_trace)
            if marginal_raw is not None:
                marginal = transitive_reduction(marginal_raw)
                metrics = compute_metrics(true_cover, marginal)
                print(f"  {'Marg. Mode':<12} {metrics['f1']:<8.3f} {metrics['precision']:<8.3f} "
                      f"{metrics['recall']:<8.3f} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['shd']:<6} {marginal.sum():<6}")
                all_results['marginal_mode']['f1'].append(metrics['f1'])
                all_results['marginal_mode']['shd'].append(metrics['shd'])
                all_results['marginal_mode']['fp'].append(metrics['fp'])
                all_results['marginal_mode']['fn'].append(metrics['fn'])
                all_results['marginal_mode']['precision'].append(metrics['precision'])
                all_results['marginal_mode']['recall'].append(metrics['recall'])
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (Mean across workflows)")
    print("=" * 80)
    
    print(f"\n{'Threshold':<25} {'F1':<10} {'Prec':<10} {'Recall':<10} {'FP↓':<8} {'FN':<8} {'SHD↓':<8}")
    print("-" * 80)
    
    results_for_table = []
    
    for threshold in THRESHOLDS:
        if all_results[threshold]['f1']:
            mean_f1 = np.mean(all_results[threshold]['f1'])
            mean_shd = np.mean(all_results[threshold]['shd'])
            mean_fp = np.mean(all_results[threshold]['fp'])
            mean_fn = np.mean(all_results[threshold]['fn'])
            mean_prec = np.mean(all_results[threshold]['precision'])
            mean_recall = np.mean(all_results[threshold]['recall'])
            thresh_str = f"α={threshold:.2f}" if threshold != 1/3 else "α=1/3 (Theoretical)"
            print(f"{thresh_str:<25} {mean_f1:<10.3f} {mean_prec:<10.3f} {mean_recall:<10.3f} "
                  f"{mean_fp:<8.1f} {mean_fn:<8.1f} {mean_shd:<8.1f}")
            results_for_table.append({
                'threshold': threshold,
                'f1': mean_f1,
                'shd': mean_shd,
                'fp': mean_fp,
                'fn': mean_fn,
                'precision': mean_prec,
                'recall': mean_recall
            })
    
    if all_results['marginal_mode']['f1']:
        mean_f1 = np.mean(all_results['marginal_mode']['f1'])
        mean_shd = np.mean(all_results['marginal_mode']['shd'])
        mean_fp = np.mean(all_results['marginal_mode']['fp'])
        mean_fn = np.mean(all_results['marginal_mode']['fn'])
        mean_prec = np.mean(all_results['marginal_mode']['precision'])
        mean_recall = np.mean(all_results['marginal_mode']['recall'])
        print(f"{'Marginal Mode':<25} {mean_f1:<10.3f} {mean_prec:<10.3f} {mean_recall:<10.3f} "
              f"{mean_fp:<8.1f} {mean_fn:<8.1f} {mean_shd:<8.1f}")
        results_for_table.append({
            'threshold': 'marginal',
            'f1': mean_f1,
            'shd': mean_shd,
            'fp': mean_fp,
            'fn': mean_fn,
            'precision': mean_prec,
            'recall': mean_recall
        })
    
    # Find best threshold by different criteria
    numeric_results = [r for r in results_for_table if r['threshold'] != 'marginal']
    
    best_f1_idx = np.argmax([r['f1'] for r in numeric_results])
    best_f1_thresh = numeric_results[best_f1_idx]['threshold']
    
    best_fp_idx = np.argmin([r['fp'] for r in numeric_results])
    best_fp_thresh = numeric_results[best_fp_idx]['threshold']
    best_fp_result = numeric_results[best_fp_idx]
    
    # Find threshold with acceptable FP (e.g., FP <= 3) and maximum F1
    acceptable_fp_results = [r for r in numeric_results if r['fp'] <= 3]
    if acceptable_fp_results:
        best_constrained_idx = np.argmax([r['f1'] for r in acceptable_fp_results])
        best_constrained = acceptable_fp_results[best_constrained_idx]
    else:
        # If no threshold achieves FP<=3, find the one with min FP
        best_constrained = numeric_results[best_fp_idx]
    
    print(f"\n--- THRESHOLD RECOMMENDATIONS ---")
    print(f"Best by F1:           α={best_f1_thresh:.2f}" if best_f1_thresh != 1/3 else f"Best by F1:           α=1/3")
    print(f"Best by FP (min FP):  α={best_fp_thresh:.2f} (FP={best_fp_result['fp']:.1f}, F1={best_fp_result['f1']:.3f})" 
          if best_fp_thresh != 1/3 else f"Best by FP (min FP):  α=1/3")
    print(f"Recommended (FP<=3):  α={best_constrained['threshold']:.2f} (FP={best_constrained['fp']:.1f}, F1={best_constrained['f1']:.3f})"
          if best_constrained['threshold'] != 1/3 else f"Recommended (FP<=3):  α=1/3")
    
    # Per-workflow optimal thresholds (minimize FP)
    print("\n" + "=" * 80)
    print("PER-WORKFLOW FP-MINIMIZING THRESHOLDS")
    print("=" * 80)
    
    workflow_optimal = {}
    for wf_name, config in WORKFLOWS.items():
        print(f"\n{config['title']}:")
        wf_results = []
        for t in THRESHOLDS:
            if all_results[t]['fp']:
                # Get per-workflow results
                wf_idx = list(WORKFLOWS.keys()).index(wf_name)
                if wf_idx < len(all_results[t]['fp']):
                    wf_results.append({
                        'threshold': t,
                        'fp': all_results[t]['fp'][wf_idx],
                        'fn': all_results[t]['fn'][wf_idx],
                        'f1': all_results[t]['f1'][wf_idx],
                        'precision': all_results[t]['precision'][wf_idx],
                        'recall': all_results[t]['recall'][wf_idx]
                    })
        
        if wf_results:
            # Find threshold with min FP
            min_fp = min(r['fp'] for r in wf_results)
            min_fp_results = [r for r in wf_results if r['fp'] == min_fp]
            # Among those, pick highest F1
            best = max(min_fp_results, key=lambda x: x['f1'])
            workflow_optimal[wf_name] = best['threshold']
            print(f"  Optimal α={best['threshold']:.2f}: FP={best['fp']}, FN={best['fn']}, F1={best['f1']:.3f}")
    
    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE (sorted by FP ascending)")
    print("=" * 80)
    
    print(r"""
\begin{table}[h]
\centering
\caption{Sensitivity Analysis of Posterior Thresholds (WfInstances, IP-Cov=0.95).}
\label{tab:threshold_comparison_wfinstances}
\small
\begin{tabular}{l c c c c}
\toprule
\textbf{Threshold} & \textbf{F1} & \textbf{FP} $\downarrow$ & \textbf{FN} & \textbf{SHD} $\downarrow$ \\
\midrule""")
    
    # Sort by FP ascending (to emphasize FP minimization)
    sorted_results = sorted(numeric_results, key=lambda x: (x['fp'], -x['f1']))
    
    for i, r in enumerate(sorted_results):
        thresh = r['threshold']
        if thresh == 1/3:
            thresh_str = r"$\alpha=1/3$"
        elif thresh == best_fp_thresh:
            thresh_str = f"$\\alpha={thresh:.2f}$ (Min FP)"
        else:
            thresh_str = f"$\\alpha={thresh:.2f}$"
        
        if i == 0:  # Best FP result
            print(f"{thresh_str} & {r['f1']:.3f} & \\textbf{{{r['fp']:.1f}}} & {r['fn']:.1f} & {r['shd']:.1f} \\\\")
        else:
            print(f"{thresh_str} & {r['f1']:.3f} & {r['fp']:.1f} & {r['fn']:.1f} & {r['shd']:.1f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}""")
    
    # Return workflow-optimal thresholds for use in plotting
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS FOR PLOTTING (copy to plot_wfinstances_po_comparison.py):")
    print("=" * 80)
    print("\nWORKFLOW_THRESHOLDS = {")
    for wf_name, thresh in workflow_optimal.items():
        print(f'    "{wf_name}": {thresh},  # Min FP threshold')
    print("}")


if __name__ == '__main__':
    main()
