#!/usr/bin/env python3
"""
WfInstances Montage Experiments for ICML Paper

Uses REAL Montage workflow instances from WfCommons WfInstances repository.
Focuses on the 2 smallest instances for manageable experiments:
  - montage-005d: 58 tasks, 114 edges (8 task types, 8 depth levels)
  - montage-01d: 103 tasks, 231 edges (8 task types, 8 depth levels)

Pipeline:
1. Load real Montage DAGs from WfInstances
2. Generate synthetic traces (random linear extensions via Kahn's algorithm)
3. Run BHPOP inference at different IP-Cov levels
4. Evaluate against ground truth (F1, SHD, feasibility)
5. Compare with baselines (AND, Majority, Process Mining)

Usage:
    python wfinstances_montage_experiments.py --num_traces 100 --num_iterations 100000
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from src
from src.utils.po_fun import BasicUtils
# The hierarchical sampler is not available in this build; use single-PO sampler as a stand-in.
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_po
from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance
from src.utils.result_paths import WFINSTANCES_MONTAGE_RESULTS_DIR

# Import shared functions from systematic_experiments.py
from training_scripts.systematic_experiments import (
    # IP-Cov (Incomparable Pair Coverage) functions
    incomparable_pairs_from_closure,
    cp_cov,  # This function computes incomparable pair coverage
    is_linear_extension_partial,
    feasibility_of_orders,
    greedy_subset_indices_to_target,
    # Synthetic trace generation
    _sample_kahn,
    # Posterior aggregation
    posterior_threshold_mean,
    # Baselines
    baseline_and,
    baseline_majority,
    # Plotting
    setup_icml_style,
    save_posterior_params_pdf,
    save_diagnostics_pdf,
    # PM4Py availability
    PM4PY_AVAILABLE,
)

# Import PM4Py baselines if available
if PM4PY_AVAILABLE:
    from training_scripts.systematic_experiments import (
        baseline_inductive_miner_cover,
        baseline_heuristics_miner_cover,
    )

# =============================================================================
# Configuration
# =============================================================================

# The 2 smallest Montage instances from WfInstances
MONTAGE_INSTANCES = [
    "montage-chameleon-2mass-005d-001.json",  # 58 tasks, 114 edges
    "montage-chameleon-2mass-01d-001.json",   # 103 tasks, 231 edges
]

DATA_DIR = PROJECT_ROOT / "data" / "wfinstances_montage_all"
OUTPUT_DIR = WFINSTANCES_MONTAGE_RESULTS_DIR

# Default experiment settings
DEFAULT_NUM_TRACES = 300  # Base traces + targeted generation
DEFAULT_NUM_ITERATIONS = 1_000_000  # 1 million iterations for faster experimentation
BURN_IN_FRACTION = 0.5
SEED = 42
PARALLEL_WORKERS = 4  # Number of parallel workers for IP-Cov experiments

# IP-Cov (Incomparable Pair Coverage) targets for systematic experiments
# 8 levels from 0.5 to 1.0 for comprehensive analysis (parallel execution)
IP_COV_TARGETS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Parallel execution - each IP-Cov target runs on 1 core
MAX_WORKERS = 8  # Number of parallel processes (1 per IP-Cov target)


# =============================================================================
# Montage Data Loading
# =============================================================================

def load_montage_instance(filepath: Path) -> Dict[str, Any]:
    """Load a Montage workflow instance from WfInstances and extract the DAG."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Navigate to specification
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    # Build task list and mappings
    task_names = []
    task_to_idx = {}
    edges = []
    
    for i, task in enumerate(tasks):
        name = task.get('name', task.get('id', f'task_{i}'))
        task_names.append(name)
        task_to_idx[name] = i
    
    # Extract edges from children
    for task in tasks:
        name = task.get('name', task.get('id', ''))
        if name not in task_to_idx:
            continue
        src_idx = task_to_idx[name]
        for child in task.get('children', []):
            if child in task_to_idx:
                dst_idx = task_to_idx[child]
                edges.append((src_idx, dst_idx))
    
    n = len(task_names)
    adj = np.zeros((n, n), dtype=np.int8)
    for src, dst in edges:
        adj[src, dst] = 1
    
    # Compute transitive closure and reduction
    closure = BasicUtils.transitive_closure(adj)
    cover = BasicUtils.transitive_reduction(closure.astype(int)).astype(np.int8)
    
    # Extract task types (for hierarchical analysis)
    def get_task_type(name: str) -> str:
        if '_ID' in name:
            return name.split('_ID')[0]
        return name.split('_')[0]
    
    task_types = [get_task_type(name) for name in task_names]
    unique_types = sorted(set(task_types))
    type_to_tasks = defaultdict(list)
    for i, t in enumerate(task_types):
        type_to_tasks[t].append(i)
    
    return {
        'name': filepath.stem,
        'task_names': task_names,
        'task_to_idx': task_to_idx,
        'task_types': task_types,
        'unique_types': unique_types,
        'type_to_tasks': dict(type_to_tasks),
        'adj': adj,
        'closure': closure,
        'cover': cover,
        'num_tasks': n,
        'num_edges': int(cover.sum()),
        'num_types': len(unique_types),
    }


def analyze_hierarchy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the hierarchical structure of the workflow."""
    n = data['num_tasks']
    closure = data['closure']
    task_types = data['task_types']
    
    # Compute depth levels
    levels = {}
    
    def get_level(idx):
        if idx in levels:
            return levels[idx]
        parents = np.where(closure[:, idx] == 1)[0]
        if len(parents) == 0:
            levels[idx] = 0
        else:
            levels[idx] = 1 + max(get_level(p) for p in parents)
        return levels[idx]
    
    for i in range(n):
        get_level(i)
    
    max_depth = max(levels.values()) if levels else 0
    
    # Type-level mapping
    type_levels = defaultdict(set)
    for i, level in levels.items():
        type_levels[task_types[i]].add(level)
    
    # Type dependencies
    type_deps = defaultdict(set)
    cover = data['cover']
    for i in range(n):
        for j in range(n):
            if cover[i, j] == 1:
                t_i, t_j = task_types[i], task_types[j]
                if t_i != t_j:
                    type_deps[t_i].add(t_j)
    
    return {
        'levels': levels,
        'max_depth': max_depth + 1,
        'type_levels': {k: sorted(v) for k, v in type_levels.items()},
        'type_deps': {k: sorted(v) for k, v in type_deps.items()},
    }


# =============================================================================
# Synthetic Trace Generation
# =============================================================================

def generate_synthetic_traces(
    cover: np.ndarray,
    num_traces: int,
    seed: int = 42,
    allow_duplicates: bool = False,
) -> List[List[int]]:
    """Generate synthetic traces as random linear extensions using Kahn's algorithm."""
    rng = np.random.default_rng(seed)
    traces = []
    seen = set()
    
    attempts = 0
    max_attempts = num_traces * 100
    
    while len(traces) < num_traces and attempts < max_attempts:
        trace = _sample_kahn(cover, rng)
        key = tuple(trace)
        
        if allow_duplicates or key not in seen:
            seen.add(key)
            traces.append(trace)
        attempts += 1
    
    if len(traces) < num_traces:
        print(f"  Warning: Only generated {len(traces)}/{num_traces} unique traces")
    
    return traces


def compute_incomparable_pair_coverage(
    traces: List[List[int]], 
    closure: np.ndarray
) -> Tuple[float, int, int]:
    """Compute incomparable pair coverage of traces."""
    pairs = incomparable_pairs_from_closure(closure)
    
    if not pairs:
        return 1.0, 0, 0
    
    # Check coverage (need both orderings)
    seen = {pair: 0 for pair in pairs}
    for trace in traces:
        pos = {a: t for t, a in enumerate(trace)}
        for (i, j) in pairs:
            if i in pos and j in pos:
                seen[(i, j)] |= 1 if pos[i] < pos[j] else 2
    
    covered = sum(1 for v in seen.values() if v == 3)
    return covered / len(pairs), covered, len(pairs)


def generate_targeted_traces(
    cover: np.ndarray,
    closure: np.ndarray,
    num_base_traces: int = 200,
    target_ip_cov: float = 0.98,
    max_targeted_traces: int = 500,
    seed: int = 42,
) -> List[List[int]]:
    """
    Generate traces that maximize incomparable pair coverage.
    
    Strategy:
    1. Generate base random traces
    2. Identify uncovered pairs (missing one or both orderings)
    3. Generate targeted traces that cover specific pairs
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Generate base random traces
    traces = []
    seen = set()
    attempts = 0
    max_attempts = num_base_traces * 50
    
    while len(traces) < num_base_traces and attempts < max_attempts:
        trace = _sample_kahn(cover, rng)
        key = tuple(trace)
        if key not in seen:
            seen.add(key)
            traces.append(trace)
        attempts += 1
    
    # Step 2: Compute current coverage and identify gaps
    pairs = incomparable_pairs_from_closure(closure)
    if not pairs:
        return traces
    
    def get_pair_coverage(traces_list):
        """Get coverage status for each pair."""
        coverage = {pair: 0 for pair in pairs}
        for trace in traces_list:
            pos = {a: t for t, a in enumerate(trace)}
            for (i, j) in pairs:
                if i in pos and j in pos:
                    coverage[(i, j)] |= 1 if pos[i] < pos[j] else 2
        return coverage
    
    coverage = get_pair_coverage(traces)
    
    # Step 3: Generate targeted traces for uncovered orderings
    targeted_added = 0
    max_targeted_attempts = max_targeted_traces * 20
    targeted_attempts = 0
    
    while targeted_attempts < max_targeted_attempts and targeted_added < max_targeted_traces:
        # Find pairs that need coverage
        uncovered = [(p, c) for p, c in coverage.items() if c != 3]
        if not uncovered:
            break  # Full coverage achieved!
        
        # Pick a random uncovered pair and determine which ordering we need
        pair, current_cov = uncovered[rng.integers(0, len(uncovered))]
        i, j = pair
        need_i_before_j = (current_cov & 1) == 0  # Need i < j
        need_j_before_i = (current_cov & 2) == 0  # Need j < i
        
        # Generate a trace and check if it helps
        trace = _sample_kahn(cover, rng)
        key = tuple(trace)
        
        if key in seen:
            targeted_attempts += 1
            continue
        
        pos = {a: t for t, a in enumerate(trace)}
        if i in pos and j in pos:
            ordering = 1 if pos[i] < pos[j] else 2
            needed = 1 if need_i_before_j else (2 if need_j_before_i else 0)
            
            if ordering == needed or needed == 0:
                # This trace helps!
                seen.add(key)
                traces.append(trace)
                targeted_added += 1
                
                # Update coverage for all pairs
                for (pi, pj) in pairs:
                    if pi in pos and pj in pos:
                        coverage[(pi, pj)] |= 1 if pos[pi] < pos[pj] else 2
        
        targeted_attempts += 1
        
        # Check if we've reached target
        current_ip_cov = sum(1 for v in coverage.values() if v == 3) / len(pairs)
        if current_ip_cov >= target_ip_cov:
            break
    
    final_cov = sum(1 for v in coverage.values() if v == 3) / len(pairs)
    print(f"  Targeted generation: {num_base_traces} base + {targeted_added} targeted = {len(traces)} total, IP-Cov: {final_cov:.3f}")
    
    return traces


# =============================================================================
# Experiment Runner (Per-Instance Single-PO Model)
# =============================================================================

def run_hierarchical_experiment(
    all_instance_data: List[Dict[str, Any]],
    all_instance_traces: List[List[List[int]]],
    ip_cov_target: float,
    num_iterations: int,
    seed: int,
    noise_option: str = "log_successors_queue_jump",
    output_dir: Path = None,
    run_baselines: bool = True,
) -> Dict[str, Any]:
    """
    Run BHPOP inference per Montage instance (single-PO per instance).
    The hierarchical sampler is unavailable in this build, so instances are fit independently.
    """
    
    print(f"\n{'='*70}")
    print(f"Per-Instance Experiment | IP-Cov Target: {ip_cov_target}")
    print(f"{'='*70}")
    
    # Build per-instance structures (single-PO inference per instance)
    instance_ids = []
    total_tasks = 0
    
    # Store per-instance data for evaluation
    instance_data_dict = {}
    
    for i, (inst_data, inst_traces) in enumerate(zip(all_instance_data, all_instance_traces)):
        instance_id = inst_data['name']
        instance_ids.append(instance_id)
        
        # Subsample traces to reach target IP-Cov
        true_closure = inst_data['closure']
        selected_idx, realized_cov = greedy_subset_indices_to_target(
            inst_traces, true_closure, ip_cov_target, seed + i
        )
        
        if not selected_idx:
            print(f"  WARNING: No valid traces for {instance_id}")
            continue
        
        traces_local = [inst_traces[j] for j in selected_idx]
        
        instance_data_dict[instance_id] = {
            'inst_data': inst_data,
            'traces_local': traces_local,
            'realized_cov': realized_cov,
        }
        
        print(f"  Instance {instance_id}: {len(traces_local)} traces, IP-Cov: {realized_cov:.3f}")
        total_tasks += inst_data['num_tasks']
    
    if not instance_ids:
        return {'error': 'No valid instances'}
    
    # Run baselines per instance
    baseline_results = {}
    if run_baselines:
        print(f"\n  Running baselines...")
        for instance_id, idata in instance_data_dict.items():
            inst_data = idata['inst_data']
            traces_local = idata['traces_local']
            n = inst_data['num_tasks']
            true_cover = inst_data['cover']
            task_names = inst_data['task_names']
            
            baseline_results[instance_id] = {}
            
            # AND baseline
            and_cover = baseline_and(traces_local, n)
            p, r = precision_recall(true_cover, and_cover)
            baseline_results[instance_id]['and_f1'] = f1_score(p, r)
            
            # Majority baseline
            maj_cover = baseline_majority(traces_local, n)
            p, r = precision_recall(true_cover, maj_cover)
            baseline_results[instance_id]['majority_f1'] = f1_score(p, r)
            
            # Process mining baselines
            if PM4PY_AVAILABLE:
                im_cover = baseline_inductive_miner_cover(traces_local, task_names, scenario_id=instance_id)
                p, r = precision_recall(true_cover, im_cover)
                baseline_results[instance_id]['inductive_miner_f1'] = f1_score(p, r)
                
                hm_cover = baseline_heuristics_miner_cover(traces_local, task_names, scenario_id=instance_id)
                p, r = precision_recall(true_cover, hm_cover)
                baseline_results[instance_id]['heuristics_miner_f1'] = f1_score(p, r)
    
    # Run MCMC per instance (single-PO inference)
    print(f"\n  Running MCMC ({num_iterations:,} iterations) for {len(instance_ids)} instances...")
    start_time = datetime.now()
    acceptance_rates = []
    mcmc_results_by_instance = {}

    results = {
        'ip_cov_target': ip_cov_target,
        'num_instances': len(instance_ids),
        'total_tasks': total_tasks,
        'elapsed_seconds': 0.0,
        'acceptance_rate': 0.0,
        'instance_results': {},
    }
    
    print(f"\n  Per-instance results:")
    print(f"  {'-'*60}")
    
    for i, instance_id in enumerate(instance_ids):
        idata = instance_data_dict[instance_id]
        inst_data = idata['inst_data']
        traces_local = idata['traces_local']
        n_local = inst_data['num_tasks']

        items = list(range(n_local))
        choice_sets = [items for _ in traces_local]

        mcmc_result = mcmc_simulation_po(
            num_iterations=num_iterations,
            items=items,
            choice_sets=choice_sets,
            observed_orders=traces_local,
            dr=0.5,  # Reduced from 0.95 for better acceptance (smaller rho proposals)
            noise_option=noise_option,
            rho_prior=1.0,
            noise_beta_prior=1.0,
            K_prior=3,
            fixed_K=None,
            random_seed=seed + i,
            cycle_length=500,
            epsilon=0.01,  # Trembling-hand epsilon
            softmax_beta_prior=(2.0, 1.0),
            softmax_beta_stepsize=0.1,
        )

        mcmc_results_by_instance[instance_id] = mcmc_result
        acceptance_rates.append(mcmc_result.get('overall_acceptance_rate', 0.0))
        H_trace = mcmc_result.get('H_trace', [])
        burn = int(len(H_trace) * BURN_IN_FRACTION)
        post = H_trace[burn:]
        
        # Extract posterior for this instance
        if not post:
            print(f"  {instance_id}: No posterior samples!")
            continue
        
        inferred_local = posterior_threshold_mean(post, threshold=0.5)
        
        # Evaluate against ground truth
        true_cover = inst_data['cover']
        p, r = precision_recall(true_cover, inferred_local)
        f1 = f1_score(p, r)
        shd = structural_hamming_distance(true_cover, inferred_local)
        feas = feasibility_of_orders(traces_local, inferred_local)
        
        results['instance_results'][instance_id] = {
            'num_tasks': inst_data['num_tasks'],
            'num_edges': inst_data['num_edges'],
            'num_traces': len(traces_local),
            'ip_cov_realized': idata['realized_cov'],
            'bhpop_f1': f1,
            'bhpop_shd': shd,
            'bhpop_feasibility': feas,
            'baselines': baseline_results.get(instance_id, {}),
        }
        
        bl = baseline_results.get(instance_id, {})
        print(f"  {instance_id}: BHPOP F1={f1:.3f}, AND F1={bl.get('and_f1', 0):.3f}, Majority F1={bl.get('majority_f1', 0):.3f}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    results['elapsed_seconds'] = elapsed
    results['acceptance_rate'] = float(np.mean(acceptance_rates)) if acceptance_rates else 0.0
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Mean acceptance rate: {results['acceptance_rate']*100:.1f}%")

    # Save diagnostics
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use unique filename for each IP-Cov target
        ip_str = f"{ip_cov_target:.2f}".replace('.', '_')
        diag_path = output_dir / f"diagnostics_ip{ip_str}.pdf"
        
        with PdfPages(diag_path) as pdf:
            # Add per-instance diagnostics and comparison plots
            for i, instance_id in enumerate(instance_ids):
                if instance_id not in instance_data_dict:
                    continue
                    
                idata = instance_data_dict[instance_id]
                inst_data = idata['inst_data']
                traces_local = idata['traces_local']
                n_local = inst_data['num_tasks']
                mcmc_result = mcmc_results_by_instance.get(instance_id)
                if mcmc_result is None:
                    continue

                save_diagnostics_pdf(mcmc_result, diag_path, pdf_pages=pdf)
                save_posterior_params_pdf(mcmc_result, diag_path, pdf_pages=pdf)
                
                H_trace = mcmc_result.get('H_trace', [])
                burn = int(len(H_trace) * BURN_IN_FRACTION)
                post = H_trace[burn:]
                if not post:
                    continue
                
                inferred_local = posterior_threshold_mean(post, threshold=0.5)
                
                true_cover = inst_data['cover']
                
                setup_icml_style()
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f"{instance_id} | IP-Cov={idata['realized_cov']:.2f}", fontsize=12)
                
                axes[0].imshow(true_cover, cmap='Blues', aspect='auto')
                axes[0].set_title(f"Ground Truth ({int(true_cover.sum())} edges)")
                
                axes[1].imshow(inferred_local, cmap='Oranges', aspect='auto')
                r = results['instance_results'].get(instance_id, {})
                axes[1].set_title(f"BHPOP ({int(inferred_local.sum())} edges, F1={r.get('bhpop_f1', 0):.3f})")
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"\n  Diagnostics saved: {diag_path}")
    
    return results


# =============================================================================
# Parallel Experiment Wrapper
# =============================================================================

def run_single_ip_cov_experiment(
    exp_id: int,
    ip_target: float,
    all_instance_data: List[Dict[str, Any]],
    all_instance_traces: List[List[List[int]]],
    num_iterations: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Wrapper for running a single IP-Cov experiment (for parallel execution).
    """
    try:
        result = run_hierarchical_experiment(
            all_instance_data=all_instance_data,
            all_instance_traces=all_instance_traces,
            ip_cov_target=ip_target,
            num_iterations=num_iterations,
            seed=seed + int(ip_target * 100),
            output_dir=output_dir,
        )
        result['exp_id'] = exp_id
        result['success'] = True
        return result
    except Exception as e:
        import traceback
        return {
            'exp_id': exp_id,
            'ip_cov_target': ip_target,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WfInstances Montage Experiments for ICML Paper"
    )
    parser.add_argument(
        "--num_traces", type=int, default=DEFAULT_NUM_TRACES,
        help=f"Number of synthetic traces per instance (default: {DEFAULT_NUM_TRACES})"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=DEFAULT_NUM_ITERATIONS,
        help=f"MCMC iterations (default: {DEFAULT_NUM_ITERATIONS:,})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--ip_cov_targets", type=float, nargs='+', default=IP_COV_TARGETS,
        help=f"IP-Cov targets (default: {IP_COV_TARGETS})"
    )
    parser.add_argument(
        "--max_workers", type=int, default=PARALLEL_WORKERS,
        help=f"Number of parallel workers for IP-Cov experiments (default: {PARALLEL_WORKERS})"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WfInstances Montage Per-Instance Experiments")
    print("=" * 70)
    print(f"Instances (assessors): {len(MONTAGE_INSTANCES)}")
    print(f"Traces per instance: {args.num_traces}")
    print(f"MCMC iterations: {args.num_iterations:,}")
    print(f"IP-Cov targets: {args.ip_cov_targets}")
    print(f"Output: {output_dir}")
    
    # Check data directory
    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("Run this first to download Montage instances.")
        return
    
    # Load all instances and generate traces upfront
    all_instance_data = []
    all_instance_traces = []
    
    for i, filename in enumerate(MONTAGE_INSTANCES):
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            print(f"\nWARNING: File not found: {filepath}")
            continue
        
        # Load instance
        data = load_montage_instance(filepath)
        hierarchy = analyze_hierarchy(data)
        data['hierarchy'] = hierarchy
        
        print(f"\nInstance {i+1}: {filename}")
        print(f"  Tasks: {data['num_tasks']}, Edges: {data['num_edges']}")
        print(f"  Task types: {data['num_types']}, Depth: {hierarchy['max_depth']}")
        
        # Generate synthetic traces with targeted coverage
        print(f"  Generating traces (target IP-Cov: 0.98)...")
        traces = generate_targeted_traces(
            data['cover'], 
            data['closure'],
            num_base_traces=args.num_traces,
            target_ip_cov=0.98,
            max_targeted_traces=args.num_traces * 2,
            seed=args.seed + i,
        )
        
        full_cov, covered, total = compute_incomparable_pair_coverage(traces, data['closure'])
        print(f"  Final pool: {len(traces)} traces, IP-Cov: {full_cov:.3f} ({covered}/{total} pairs)")
        
        all_instance_data.append(data)
        all_instance_traces.append(traces)
    
    if not all_instance_data:
        print("\nERROR: No valid instances loaded!")
        return
    
    # Run per-instance experiments at each IP-Cov target IN PARALLEL
    all_results = []
    
    print(f"\n{'#'*70}")
    print(f"# Running {len(args.ip_cov_targets)} per-instance experiments IN PARALLEL")
    print(f"# (each with {len(all_instance_data)} instances)")
    print(f"# Parallel workers: {args.max_workers}")
    print(f"# Iterations per experiment: {args.num_iterations:,}")
    print(f"# Estimated time: ~{len(args.ip_cov_targets) * args.num_iterations / 1_000_000 / args.max_workers:.1f} hours")
    print(f"{'#'*70}")
    
    start_time = datetime.now()
    
    # Build experiment list
    experiments = [
        {
            'exp_id': i,
            'ip_target': ip_target,
            'all_instance_data': all_instance_data,
            'all_instance_traces': all_instance_traces,
            'num_iterations': args.num_iterations,
            'seed': args.seed,
            'output_dir': output_dir,
        }
        for i, ip_target in enumerate(args.ip_cov_targets)
    ]
    
    # Run in parallel
    print(f"\nSubmitting {len(experiments)} experiments to {args.max_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single_ip_cov_experiment, **exp): exp
            for exp in experiments
        }
        
        completed = 0
        for future in as_completed(futures):
            exp = futures[future]
            result = future.result()
            all_results.append(result)
            completed += 1
            
            if result.get('success', False):
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                avg_time = elapsed / completed
                remaining = avg_time * (len(experiments) - completed)
                
                print(f"\n✓ [{completed}/{len(experiments)}] Completed IP-Cov={result['ip_cov_target']} "
                      f"(Exp {result['exp_id']})")
                print(f"  Elapsed: {elapsed:.1f} min | Est. remaining: {remaining:.1f} min")
            else:
                print(f"\n✗ [{completed}/{len(experiments)}] Failed IP-Cov={result.get('ip_cov_target')} "
                      f"(Exp {result.get('exp_id')}): {result.get('error')}")
    
    # Sort results by IP-Cov target for consistent output
    all_results.sort(key=lambda x: x.get('ip_cov_target', 0))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Print comparison table
    print("\nBHPOP vs Baselines by IP-Cov (Per-Instance Model):")
    print("-" * 100)
    print(f"{'Instance':<40} {'IP-Cov':>7} {'BHPOP':>8} {'AND':>8} {'Majority':>10}")
    print("-" * 100)
    
    for result in all_results:
        if 'error' in result:
            continue
        for instance_id, ir in result.get('instance_results', {}).items():
            bl = ir.get('baselines', {})
            print(f"{instance_id:<40} {ir.get('ip_cov_realized', 0):>7.2f} "
                  f"{ir.get('bhpop_f1', 0):>8.3f} "
                  f"{bl.get('and_f1', 0):>8.3f} "
                  f"{bl.get('majority_f1', 0):>10.3f}")
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_traces': args.num_traces,
            'num_iterations': args.num_iterations,
            'seed': args.seed,
            'ip_cov_targets': args.ip_cov_targets,
            'instances': MONTAGE_INSTANCES,
        },
        'results': all_results,
    }
    
    # Use unique summary filename if running single IP-Cov target
    if len(args.ip_cov_targets) == 1:
        ip_str = f"{args.ip_cov_targets[0]:.2f}".replace('.', '_')
        summary_path = output_dir / f"summary_ip{ip_str}.json"
    else:
        summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print(f"\n✅ Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
