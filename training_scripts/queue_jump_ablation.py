#!/usr/bin/env python3
"""
Queue Jump Ablation Study - Comparing BHPOP vs NLE-based Queue Jump
===================================================================

Compares two Bayesian likelihood models for partial order inference:

1. log_successors_queue_jump: BHPOP's frontier-softmax likelihood
   P(y_j | remaining) = (1-ε) * softmax(β * log(1 + successors(y_j))) + ε/|remaining|

2. queue_jump: NLE-counting based (similar to Nicholls' queue jump model)
   P(y_j | remaining) = (1-ε) * (NLE_first / NLE_total) + ε/|remaining|
   NOTE: Uses Python arbitrary precision integers (slow for n > 25)

Datasets (WfInstances/WfCommons):
- SRASearch: 22 tasks, 30 edges
- Epigenomics: 41 tasks, 48 edges (skipped for queue_jump - too large)

Settings:
- IP Coverage: 0.95
- Epsilon: 0.01 (fixed)

Usage:
    python queue_jump_ablation.py [--quick]
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project root setup
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# BHPOP imports
from src.utils.po_fun import BasicUtils
from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_po
from src.utils.wfcommons_loader import load_workflow_instances

# Import from systematic experiments
from training_scripts.systematic_experiments import (
    incomparable_pairs_from_closure,
    feasibility_of_orders,
    _sample_kahn,
    posterior_threshold_mean,
    baseline_and,
    baseline_majority,
    OUTPUT_DIR,
)

# =========================
# Configuration
# =========================
# Note: queue_jump requires exact NLE computation which is #P-complete
# NLE overflows 64-bit for graphs with many incomparable pairs (like Epigenomics)
# SRASearch (22 tasks) works fine, Epigenomics (41 tasks) overflows
# frontier_softmax_uni_utility: uniform over frontier (Q=0, no successor preference)
LIKELIHOODS = ["log_successors_queue_jump", "frontier_softmax_uni_utility", "queue_jump"]
MAX_TASKS_FOR_NLE = 30  # queue_jump only for graphs with <= 30 tasks (before overflow)

IP_COV_TARGET = 0.95
EPSILON = 0.01  # Fixed epsilon

# MCMC settings
NUM_ITERATIONS_FULL = 1_000_000
NUM_ITERATIONS_QUICK = 100_000
BURN_IN_FRACTION = 0.5
SEED = 42

# Data directories
EPIGENOMICS_DATA_DIR = PROJECT_ROOT / "data" / "wfinstances_epigenomics"
SRASEARCH_DATA_DIR = PROJECT_ROOT / "data" / "wfinstances_srasearch"

# Output
ABLATION_OUTPUT_DIR = OUTPUT_DIR / "queue_jump_ablation"
ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Data Loading
# =========================

def load_epigenomics_data() -> Dict[str, Any]:
    """Load Epigenomics workflow and build DAG structure."""
    json_files = list(EPIGENOMICS_DATA_DIR.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {EPIGENOMICS_DATA_DIR}")
    
    filepath = json_files[0]
    with open(filepath) as f:
        data = json.load(f)
    
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    task_ids = [t.get('name', t.get('id')) for t in tasks]
    n = len(task_ids)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    
    adj = np.zeros((n, n), dtype=np.int8)
    for t in tasks:
        name = t.get('name', t.get('id'))
        if name not in task_to_idx:
            continue
        src = task_to_idx[name]
        for child in t.get('children', []):
            if child in task_to_idx:
                adj[src, task_to_idx[child]] = 1
    
    closure = BasicUtils.transitive_closure(adj)
    cover = adj.copy()
    
    return {
        'name': 'epigenomics',
        'task_ids': task_ids,
        'closure': closure,
        'cover': cover,
        'num_tasks': n,
        'num_edges': int(cover.sum()),
    }


def load_srasearch_data() -> Dict[str, Any]:
    """Load SRASearch workflow and build DAG structure."""
    instances = load_workflow_instances(SRASEARCH_DATA_DIR)
    
    if not instances:
        raise ValueError(f"No instances found in {SRASEARCH_DATA_DIR}")
    
    first = instances[0]
    task_ids = first['task_ids']
    parents = first['parents']
    n = len(task_ids)
    
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    
    adj = np.zeros((n, n), dtype=np.int8)
    for child, parent_set in parents.items():
        if child not in task_to_idx:
            continue
        child_idx = task_to_idx[child]
        for p in parent_set:
            if p in task_to_idx:
                adj[task_to_idx[p], child_idx] = 1
    
    closure = BasicUtils.transitive_closure(adj)
    cover = adj.copy()
    
    return {
        'name': 'srasearch',
        'task_ids': task_ids,
        'closure': closure,
        'cover': cover,
        'num_tasks': n,
        'num_edges': int(cover.sum()),
    }


# =========================
# IP Coverage Functions
# =========================

def compute_ip_coverage(traces: List[List[int]], closure: np.ndarray) -> float:
    """
    Compute incomparable pair coverage of traces.
    
    Full coverage requires BOTH orderings (i<j and j<i) to be observed for each pair.
    """
    pairs = incomparable_pairs_from_closure(closure)
    if not pairs:
        return 1.0
    
    # Track coverage: 0=unseen, 1=seen i<j, 2=seen j<i, 3=seen both
    seen = {pair: 0 for pair in pairs}
    
    for trace in traces:
        pos = {a: t for t, a in enumerate(trace)}
        for (i, j) in pairs:
            if i in pos and j in pos:
                seen[(i, j)] |= 1 if pos[i] < pos[j] else 2
    
    # Count pairs with BOTH orderings seen (value == 3)
    fully_covered = sum(1 for v in seen.values() if v == 3)
    return fully_covered / len(pairs)


def generate_traces_to_target_coverage(
    cover: np.ndarray,
    closure: np.ndarray,
    target_cov: float,
    max_traces: int = 500,
    seed: int = 42,
) -> Tuple[List[List[int]], float]:
    """Generate synthetic traces until target IP coverage is reached."""
    rng = np.random.default_rng(seed)
    n = cover.shape[0]
    traces = []
    
    pairs = incomparable_pairs_from_closure(closure)
    if not pairs:
        trace = _sample_kahn(cover, rng)
        return [trace], 1.0
    
    current_cov = 0.0
    while current_cov < target_cov and len(traces) < max_traces:
        trace = _sample_kahn(cover, rng)
        traces.append(trace)
        current_cov = compute_ip_coverage(traces, closure)
    
    return traces, current_cov


# =========================
# Experiment Functions
# =========================

def run_single_experiment(
    workflow_name: str,
    cover: np.ndarray,
    closure: np.ndarray,
    traces: List[List[int]],
    realized_cov: float,
    likelihood: str,
    num_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """Run MCMC for a single workflow with given likelihood."""
    
    n = cover.shape[0]
    items = list(range(n))
    choice_sets = [list(range(n)) for _ in traces]
    
    print(f"    Running MCMC: {likelihood}, {num_iterations:,} iterations...")
    start_time = datetime.now()
    
    mcmc = mcmc_simulation_po(
        num_iterations=num_iterations,
        items=items,
        choice_sets=choice_sets,
        observed_orders=traces,
        dr=0.5,
        noise_option=likelihood,
        rho_prior=1.0,
        noise_beta_prior=1.0,
        K_prior=3,
        fixed_K=None,
        random_seed=seed,
        cycle_length=500,
        epsilon=EPSILON,
        softmax_beta_prior=(2.0, 1.0),
        softmax_beta_stepsize=0.1,
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"    MCMC completed in {elapsed:.1f}s")
    
    # Extract posterior
    H_trace = mcmc["H_trace"]
    burn = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn:]
    
    # Point estimate
    if post:
        mats = [h for h in post]
        final_H = posterior_threshold_mean(mats, threshold=0.5)
    else:
        final_H = np.zeros((n, n), dtype=np.int8)
    
    # Compute metrics
    p, r = precision_recall(cover, final_H)
    cover_f1 = f1_score(p, r)
    shd = structural_hamming_distance(cover, final_H)
    feas = feasibility_of_orders(traces, final_H)
    
    # Acceptance rate
    acc_trace = mcmc.get("acceptance_trace", [])
    acc_rate = np.mean(acc_trace) if acc_trace else 0.0
    
    return {
        "workflow": workflow_name,
        "likelihood": likelihood,
        "ip_cov_target": IP_COV_TARGET,
        "ip_cov_realized": realized_cov,
        "epsilon": EPSILON,
        "num_traces": len(traces),
        "cover_f1": cover_f1,
        "precision": p,
        "recall": r,
        "shd": shd,
        "feasibility": feas,
        "acceptance_rate": acc_rate,
        "runtime_seconds": elapsed,
        "final_H": final_H,
    }


def run_ablation(quick: bool = False):
    """Run the queue jump ablation study on epigenomics and srasearch."""
    
    num_iterations = NUM_ITERATIONS_QUICK if quick else NUM_ITERATIONS_FULL
    print(f"\n{'='*60}")
    print(f"Queue Jump Ablation Study (WfInstances)")
    print(f"{'='*60}")
    print(f"IP Coverage Target: {IP_COV_TARGET}")
    print(f"Epsilon: {EPSILON}")
    print(f"Iterations: {num_iterations:,}")
    print(f"Likelihoods: {LIKELIHOODS}")
    print(f"{'='*60}\n")
    
    # Load both datasets
    print("Loading data...")
    datasets = {}
    
    try:
        datasets['epigenomics'] = load_epigenomics_data()
        print(f"  Epigenomics: {datasets['epigenomics']['num_tasks']} tasks, "
              f"{datasets['epigenomics']['num_edges']} edges")
    except Exception as e:
        print(f"  Warning: Could not load epigenomics: {e}")
    
    try:
        datasets['srasearch'] = load_srasearch_data()
        print(f"  SRASearch: {datasets['srasearch']['num_tasks']} tasks, "
              f"{datasets['srasearch']['num_edges']} edges")
    except Exception as e:
        print(f"  Warning: Could not load srasearch: {e}")
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return None
    
    results = []
    
    for workflow_name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Workflow: {workflow_name}")
        print(f"  Tasks: {data['num_tasks']}, Edges: {data['num_edges']}")
        print(f"{'='*50}")
        
        closure = data['closure']
        cover = data['cover']
        n = data['num_tasks']
        
        # Generate traces to target IP coverage
        print(f"  Generating traces to IP-Cov={IP_COV_TARGET}...")
        traces, realized_cov = generate_traces_to_target_coverage(
            cover, closure, IP_COV_TARGET, max_traces=500, seed=SEED
        )
        print(f"  Generated {len(traces)} traces, realized IP-Cov: {realized_cov:.3f}")
        
        # Baselines
        and_cover = baseline_and(traces, n)
        p_and, r_and = precision_recall(cover, and_cover)
        f1_and = f1_score(p_and, r_and)
        
        maj_cover = baseline_majority(traces, n)
        p_maj, r_maj = precision_recall(cover, maj_cover)
        f1_maj = f1_score(p_maj, r_maj)
        
        print(f"  Baselines: AND F1={f1_and:.3f}, Majority F1={f1_maj:.3f}")
        
        # Add baseline results
        results.append({
            "workflow": workflow_name,
            "likelihood": "AND (intersection)",
            "ip_cov_target": IP_COV_TARGET,
            "ip_cov_realized": realized_cov,
            "epsilon": None,
            "num_traces": len(traces),
            "cover_f1": f1_and,
            "precision": p_and,
            "recall": r_and,
            "shd": structural_hamming_distance(cover, and_cover),
            "feasibility": feasibility_of_orders(traces, and_cover),
            "acceptance_rate": None,
            "runtime_seconds": 0,
        })
        
        results.append({
            "workflow": workflow_name,
            "likelihood": "Majority",
            "ip_cov_target": IP_COV_TARGET,
            "ip_cov_realized": realized_cov,
            "epsilon": None,
            "num_traces": len(traces),
            "cover_f1": f1_maj,
            "precision": p_maj,
            "recall": r_maj,
            "shd": structural_hamming_distance(cover, maj_cover),
            "feasibility": feasibility_of_orders(traces, maj_cover),
            "acceptance_rate": None,
            "runtime_seconds": 0,
        })
        
        # Run BHPOP likelihoods
        for lh in LIKELIHOODS:
            # Skip queue_jump (exact NLE) for large graphs - it's computationally intractable
            if lh == "queue_jump" and n > MAX_TASKS_FOR_NLE:
                print(f"\n  Likelihood: {lh}")
                print(f"    SKIPPED: Exact NLE computation is intractable for n={n} > {MAX_TASKS_FOR_NLE} tasks")
                print(f"    (NLE counting is #P-complete, only feasible for very small graphs)")
                results.append({
                    "workflow": workflow_name,
                    "likelihood": lh,
                    "ip_cov_target": IP_COV_TARGET,
                    "ip_cov_realized": realized_cov,
                    "epsilon": EPSILON,
                    "num_traces": len(traces),
                    "cover_f1": None,
                    "precision": None,
                    "recall": None,
                    "shd": None,
                    "feasibility": None,
                    "acceptance_rate": None,
                    "runtime_seconds": None,
                    "note": f"Skipped: NLE intractable for n={n}",
                })
                continue
            
            print(f"\n  Likelihood: {lh}")
            seed = SEED + list(datasets.keys()).index(workflow_name)
            
            try:
                result = run_single_experiment(
                    workflow_name=workflow_name,
                    cover=cover,
                    closure=closure,
                    traces=traces,
                    realized_cov=realized_cov,
                    likelihood=lh,
                    num_iterations=num_iterations,
                    seed=seed,
                )
                
                print(f"    F1: {result['cover_f1']:.3f}, SHD: {result['shd']}, "
                      f"Feas: {result['feasibility']:.3f}, Acc: {result['acceptance_rate']:.3f}")
                
                # Save individual results
                exp_dir = ABLATION_OUTPUT_DIR / f"{workflow_name}_{lh}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                with open(exp_dir / "final_H.pkl", "wb") as f:
                    pickle.dump(result["final_H"], f)
                
                # Store for summary (without large objects)
                result_summary = {k: v for k, v in result.items() if k != "final_H"}
                results.append(result_summary)
                
            except RuntimeError as e:
                print(f"    ERROR: {e}")
                print(f"    (This is expected for queue_jump on large graphs)")
                results.append({
                    "workflow": workflow_name,
                    "likelihood": lh,
                    "ip_cov_target": IP_COV_TARGET,
                    "ip_cov_realized": realized_cov,
                    "epsilon": EPSILON,
                    "num_traces": len(traces),
                    "cover_f1": None,
                    "precision": None,
                    "recall": None,
                    "shd": None,
                    "feasibility": None,
                    "acceptance_rate": None,
                    "runtime_seconds": None,
                    "note": f"Error: {str(e)[:100]}",
                })
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv(ABLATION_OUTPUT_DIR / "ablation_results.csv", index=False)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Queue Jump Ablation Results")
    print(f"{'='*70}")
    
    # Format for display (handle None/NaN)
    df_display = df[['workflow', 'likelihood', 'cover_f1', 'precision', 'recall', 'shd', 'feasibility']].copy()
    for col in ['cover_f1', 'precision', 'recall', 'feasibility']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_display['shd'] = df_display['shd'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    print(df_display.to_string(index=False))
    
    # Print note about NLE intractability
    if df['cover_f1'].isna().any():
        print(f"\nNote: queue_jump (NLE-based) is skipped for graphs with n > {MAX_TASKS_FOR_NLE} tasks")
        print("      because exact NLE computation is #P-complete and intractable for large graphs.")
    
    # Create comparison plot
    create_comparison_plot(df)
    
    print(f"\nResults saved to: {ABLATION_OUTPUT_DIR}")
    return df


def create_comparison_plot(df: pd.DataFrame):
    """Create a comparison plot of the methods."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    workflows = df['workflow'].unique()
    # Only include methods that have at least some valid results
    df_valid = df.dropna(subset=['cover_f1'])
    methods = df_valid['likelihood'].unique()
    x = np.arange(len(workflows))
    width = 0.18
    
    colors = {
        'log_successors_queue_jump': '#377eb8',  # Blue (BPOP)
        'frontier_softmax_uni_utility': '#ff7f00',  # Orange (Uniform frontier)
        'queue_jump': '#e41a1c',  # Red (Nicholls-style NLE)
        'AND (intersection)': '#4daf4a',  # Green
        'Majority': '#984ea3',  # Purple
    }
    labels = {
        'log_successors_queue_jump': 'BPOP (log-successors)',
        'frontier_softmax_uni_utility': 'BPOP (uniform utility)',
        'queue_jump': 'NLE Queue Jump (n≤15 only)',
        'AND (intersection)': 'AND (intersection)',
        'Majority': 'Majority',
    }
    
    metrics = [('cover_f1', 'F1 Score'), ('shd', 'SHD'), ('feasibility', 'Feasibility')]
    
    for ax, (metric, title) in zip(axes, metrics):
        for i, method in enumerate(methods):
            method_data = df_valid[df_valid['likelihood'] == method].set_index('workflow')
            values = []
            for w in workflows:
                if w in method_data.index:
                    v = method_data.loc[w, metric]
                    values.append(0 if pd.isna(v) else v)
                else:
                    values.append(0)
            offset = (i - len(methods)/2 + 0.5) * width
            color = colors.get(method, '#999999')
            label = labels.get(method, method)
            ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
        
        ax.set_xlabel('Workflow')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([w.capitalize() for w in workflows], fontsize=10)
        if metric == 'cover_f1' or metric == 'feasibility':
            ax.set_ylim(0, 1.05)
    
    # Single legend at top
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc='upper center', ncol=len(methods), fontsize=9, 
               bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle(f'Queue Jump Ablation (IP-Cov={IP_COV_TARGET}, ε={EPSILON})', 
                 fontsize=12, y=1.08)
    plt.tight_layout()
    plt.savefig(ABLATION_OUTPUT_DIR / "ablation_comparison.pdf", bbox_inches='tight')
    plt.savefig(ABLATION_OUTPUT_DIR / "ablation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {ABLATION_OUTPUT_DIR / 'ablation_comparison.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue Jump Ablation Study")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    args = parser.parse_args()
    
    run_ablation(quick=args.quick)
