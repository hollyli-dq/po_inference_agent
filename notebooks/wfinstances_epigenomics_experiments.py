#!/usr/bin/env python3
"""
WfInstances Epigenomics Experiments for Benchmarking

Uses REAL Epigenomics workflow DAG from WfCommons WfInstances repository.
Single graph recovery experiment with synthetic traces.

Workflow: Epigenomics (1 sequence, 100k)
  - 41 tasks, 48 edges
  - 576 incomparable pairs (70% parallelism)
  - Pipeline structure with 9 parallel lanes

Pipeline:
1. Load real Epigenomics DAG from WfInstances
2. Generate synthetic traces at target IP-Cov levels
3. Run BHPOP inference at 4 IP-Cov levels
4. Evaluate against ground truth (F1, SHD, feasibility)
5. Compare with baselines (AND, Majority)

Usage:
    python wfinstances_epigenomics_experiments.py --num_iterations 1000000
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from src
from src.utils.po_fun import BasicUtils
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_po
from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance

# Import shared functions from systematic_experiments.py
from systematic_experiments import (
    incomparable_pairs_from_closure,
    feasibility_of_orders,
    _sample_kahn,
    posterior_threshold_mean,
    baseline_and,
    baseline_majority,
    setup_icml_style,
    save_posterior_params_pdf,
    save_diagnostics_pdf,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "wfinstances_epigenomics"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "wfinstances_epigenomics_results"

# Default experiment settings
DEFAULT_NUM_ITERATIONS = 1_000_000  # 1 million iterations
BURN_IN_FRACTION = 0.5
SEED = 42

# 4 IP-Cov (Incomparable Pair Coverage) targets
IP_COV_TARGETS = [0.5, 0.7, 0.85, 0.95]


# =============================================================================
# Data Loading
# =============================================================================

def load_epigenomics_data() -> Dict[str, Any]:
    """Load Epigenomics workflow and build DAG structure."""
    # Find the JSON file
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {DATA_DIR}")
    
    filepath = json_files[0]
    with open(filepath) as f:
        data = json.load(f)
    
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    # Build task list and mappings
    task_ids = [t.get('name', t.get('id')) for t in tasks]
    n = len(task_ids)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    idx_to_task = {i: t for t, i in task_to_idx.items()}
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    for t in tasks:
        name = t.get('name', t.get('id'))
        if name not in task_to_idx:
            continue
        src = task_to_idx[name]
        for child in t.get('children', []):
            if child in task_to_idx:
                adj[src, task_to_idx[child]] = 1
    
    # Compute transitive closure
    closure = BasicUtils.transitive_closure(adj)
    cover = adj.copy()  # Already the transitive reduction
    
    return {
        'name': filepath.stem,
        'task_ids': task_ids,
        'task_to_idx': task_to_idx,
        'idx_to_task': idx_to_task,
        'adj': adj,
        'closure': closure,
        'cover': cover,
        'num_tasks': n,
        'num_edges': int(cover.sum()),
    }


# =============================================================================
# IP Coverage Functions
# =============================================================================

def compute_ip_coverage(
    traces: List[List[int]], 
    closure: np.ndarray
) -> Tuple[float, int, int, Dict]:
    """Compute incomparable pair coverage of traces."""
    pairs = incomparable_pairs_from_closure(closure)
    
    if not pairs:
        return 1.0, 0, 0, {}
    
    seen = {pair: 0 for pair in pairs}
    
    for trace in traces:
        pos = {a: t for t, a in enumerate(trace)}
        for (i, j) in pairs:
            if i in pos and j in pos:
                seen[(i, j)] |= 1 if pos[i] < pos[j] else 2
    
    fully_covered = sum(1 for v in seen.values() if v == 3)
    partially_covered = sum(1 for v in seen.values() if v in [1, 2])
    uncovered = sum(1 for v in seen.values() if v == 0)
    
    details = {
        'fully_covered': fully_covered,
        'partially_covered': partially_covered,
        'uncovered': uncovered,
        'total_pairs': len(pairs),
    }
    
    return fully_covered / len(pairs), fully_covered, len(pairs), details


def generate_traces_to_target_ip_cov(
    cover: np.ndarray,
    closure: np.ndarray,
    target_ip_cov: float,
    max_traces: int = 1000,
    seed: int = 42,
) -> Tuple[List[List[int]], float]:
    """
    Generate synthetic traces to reach target IP coverage.
    
    Returns:
        (traces, realized_ip_cov)
    """
    rng = np.random.default_rng(seed)
    pairs = incomparable_pairs_from_closure(closure)
    
    if not pairs:
        # No incomparable pairs - just return one trace
        trace = _sample_kahn(cover, rng)
        return [trace], 1.0
    
    traces = []
    seen_traces = set()
    coverage = {pair: 0 for pair in pairs}
    
    attempts = 0
    max_attempts = max_traces * 100
    
    while len(traces) < max_traces and attempts < max_attempts:
        trace = _sample_kahn(cover, rng)
        key = tuple(trace)
        attempts += 1
        
        if key in seen_traces:
            continue
        
        # Check if this trace improves coverage
        pos = {a: t for t, a in enumerate(trace)}
        improves = False
        
        for (i, j) in pairs:
            if coverage[(i, j)] == 3:
                continue
            if i in pos and j in pos:
                ordering = 1 if pos[i] < pos[j] else 2
                if (coverage[(i, j)] | ordering) != coverage[(i, j)]:
                    improves = True
                    break
        
        # Always add first few traces, then only if they improve
        if len(traces) < 5 or improves:
            seen_traces.add(key)
            traces.append(trace)
            
            # Update coverage
            for (i, j) in pairs:
                if i in pos and j in pos:
                    coverage[(i, j)] |= 1 if pos[i] < pos[j] else 2
            
            current_cov = sum(1 for v in coverage.values() if v == 3) / len(pairs)
            if current_cov >= target_ip_cov:
                break
    
    final_cov = sum(1 for v in coverage.values() if v == 3) / len(pairs)
    return traces, final_cov


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(
    data: Dict[str, Any],
    traces: List[List[int]],
    ip_cov_target: float,
    realized_ip_cov: float,
    num_iterations: int,
    seed: int,
    noise_option: str = "log_successors_queue_jump",
) -> Dict[str, Any]:
    """Run BHPOP inference on a single graph with given traces."""
    n = data['num_tasks']
    true_cover = data['cover']
    
    # Prepare MCMC inputs
    items = list(range(n))
    choice_sets = [items for _ in traces]
    
    print(f"  Running MCMC: {len(traces)} traces, {num_iterations:,} iterations...")
    start_time = datetime.now()
    
    mcmc_result = mcmc_simulation_po(
        num_iterations=num_iterations,
        items=items,
        choice_sets=choice_sets,
        observed_orders=traces,
        dr=0.5,
        noise_option=noise_option,
        rho_prior=1.0,
        noise_beta_prior=1.0,
        K_prior=3,
        fixed_K=None,
        random_seed=seed,
        cycle_length=500,
        epsilon=0.01,
        softmax_beta_prior=(2.0, 1.0),
        softmax_beta_stepsize=0.1,
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Extract posterior
    H_trace = mcmc_result.get('H_trace', [])
    burn = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn:]
    
    if not post:
        return {'error': 'No posterior samples'}
    
    # Compute inferred cover
    inferred = posterior_threshold_mean(post, threshold=0.5)
    
    # Evaluate
    p, r = precision_recall(true_cover, inferred)
    f1 = f1_score(p, r)
    shd = structural_hamming_distance(true_cover, inferred)
    feas = feasibility_of_orders(traces, inferred)
    
    # Baselines
    and_cover = baseline_and(traces, n)
    p_and, r_and = precision_recall(true_cover, and_cover)
    f1_and = f1_score(p_and, r_and)
    
    maj_cover = baseline_majority(traces, n)
    p_maj, r_maj = precision_recall(true_cover, maj_cover)
    f1_maj = f1_score(p_maj, r_maj)
    
    return {
        'ip_cov_target': ip_cov_target,
        'ip_cov_realized': realized_ip_cov,
        'num_traces': len(traces),
        'num_iterations': num_iterations,
        'elapsed_seconds': elapsed,
        'acceptance_rate': mcmc_result.get('overall_acceptance_rate', 0.0),
        # BHPOP metrics
        'bhpop_precision': p,
        'bhpop_recall': r,
        'bhpop_f1': f1,
        'bhpop_shd': shd,
        'bhpop_feasibility': feas,
        'bhpop_inferred_edges': int(inferred.sum()),
        # Baselines
        'and_f1': f1_and,
        'majority_f1': f1_maj,
        # MCMC result for diagnostics
        'mcmc_result': mcmc_result,
        'inferred_cover': inferred,
    }


# =============================================================================
# Parallel Worker Function
# =============================================================================

def run_single_ip_cov_experiment(
    exp_idx: int,
    ip_target: float,
    data: Dict[str, Any],
    num_iterations: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Worker function for parallel execution of a single IP-Cov experiment."""
    try:
        # Generate synthetic traces to target IP-Cov
        traces, realized_cov = generate_traces_to_target_ip_cov(
            data['cover'],
            data['closure'],
            target_ip_cov=ip_target,
            max_traces=500,
            seed=seed + exp_idx,
        )
        
        print(f"[IP-Cov={ip_target}] Traces: {len(traces)}, Realized: {realized_cov*100:.1f}%")
        
        # Run experiment
        result = run_single_experiment(
            data=data,
            traces=traces,
            ip_cov_target=ip_target,
            realized_ip_cov=realized_cov,
            num_iterations=num_iterations,
            seed=seed + exp_idx * 100,
        )
        
        if 'error' in result:
            return {'ip_cov_target': ip_target, 'error': result['error']}
        
        # Save diagnostics (matching systematic_experiments.py format)
        exp_dir = output_dir / f"ip_cov_{ip_target:.2f}".replace('.', '_')
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        mcmc = result['mcmc_result']
        H_trace = mcmc.get('H_trace', [])
        
        # Save likelihood trace
        ll_trace = mcmc.get('log_likelihood_currents', [])
        if ll_trace:
            np.save(exp_dir / "likelihood_trace.npy", np.asarray(ll_trace, dtype=float))
        
        # Save H trace
        with open(exp_dir / "H_trace.pkl", "wb") as f:
            pickle.dump(H_trace, f)
        
        # Save parameter traces
        param_traces = {}
        for key in ['rho_trace', 'K_trace', 'prob_noise_trace', 'softmax_beta_trace', 'epsilon_trace', 'U_trace']:
            if key in mcmc:
                param_traces[key] = mcmc[key]
        if param_traces:
            with open(exp_dir / "param_traces.pkl", "wb") as f:
                pickle.dump(param_traces, f)
        
        # Save final H and avg H
        burn = int(len(H_trace) * BURN_IN_FRACTION)
        post = H_trace[burn:]
        if post:
            final_H = result['inferred_cover']
            avg_H = np.mean(np.stack(post), axis=0)
        else:
            n = data['num_tasks']
            final_H = np.zeros((n, n), dtype=np.int8)
            avg_H = np.zeros((n, n), dtype=float)
        
        with open(exp_dir / "final_H.pkl", "wb") as f:
            pickle.dump(final_H, f)
        with open(exp_dir / "avg_H.pkl", "wb") as f:
            pickle.dump(avg_H, f)
        
        # Save true cover
        np.save(exp_dir / "true_cover.npy", data['cover'])
        
        # Create combined diagnostics PDF
        diag_path = exp_dir / "diagnostics.pdf"
        with PdfPages(diag_path) as pdf:
            save_diagnostics_pdf(mcmc, diag_path, pdf_pages=pdf)
            save_posterior_params_pdf(mcmc, diag_path, pdf_pages=pdf,
                                     param_traces_path=exp_dir / "param_traces.pkl")
            
            # Add comparison plot
            setup_icml_style()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"Epigenomics | IP-Cov={realized_cov:.2f} | F1={result['bhpop_f1']:.3f}")
            
            axes[0].imshow(data['cover'], cmap='Blues', aspect='auto')
            axes[0].set_title(f"Ground Truth ({data['num_edges']} edges)")
            
            axes[1].imshow(result['inferred_cover'], cmap='Oranges', aspect='auto')
            axes[1].set_title(f"BHPOP ({result['bhpop_inferred_edges']} edges)")
            
            diff = result['inferred_cover'].astype(int) - data['cover'].astype(int)
            axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            axes[2].set_title(f"Difference (SHD={result['bhpop_shd']})")
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"[IP-Cov={ip_target}] ✓ F1={result['bhpop_f1']:.3f}, Time={result['elapsed_seconds']:.1f}s")
        
        # Return clean result (without large objects)
        return {k: v for k, v in result.items() if k not in ['mcmc_result', 'inferred_cover']}
        
    except Exception as e:
        import traceback
        return {
            'ip_cov_target': ip_target,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


# =============================================================================
# Main
# =============================================================================

# Default number of parallel workers
DEFAULT_WORKERS = 4

def main():
    parser = argparse.ArgumentParser(
        description="WfInstances Epigenomics Experiments"
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
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WfInstances Epigenomics Experiment (PARALLEL)")
    print("=" * 70)
    print(f"Workers: {args.workers}")
    
    # Load data
    print("\n1. Loading Epigenomics workflow...")
    data = load_epigenomics_data()
    
    print(f"   Workflow: {data['name']}")
    print(f"   Tasks: {data['num_tasks']}")
    print(f"   Edges (ground truth): {data['num_edges']}")
    
    # Analyze incomparable pairs
    pairs = incomparable_pairs_from_closure(data['closure'])
    print(f"   Incomparable pairs: {len(pairs)}")
    print(f"   Parallelism: {len(pairs)/(data['num_tasks']*(data['num_tasks']-1)//2)*100:.1f}%")
    
    # Run experiments in PARALLEL
    print(f"\n2. Running {len(args.ip_cov_targets)} experiments in PARALLEL ({args.workers} workers)...")
    print(f"   IP-Cov targets: {args.ip_cov_targets}")
    print(f"   Iterations per experiment: {args.num_iterations:,}")
    print("=" * 70)
    
    all_results = []
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_ip_cov_experiment,
                i, ip_target, data, args.num_iterations, args.seed, output_dir
            ): ip_target
            for i, ip_target in enumerate(args.ip_cov_targets)
        }
        
        for future in as_completed(futures):
            ip_target = futures[future]
            result = future.result()
            all_results.append(result)
            
            if 'error' in result:
                print(f"[IP-Cov={ip_target}] ✗ ERROR: {result['error']}")
    
    # Sort results by IP-Cov target
    all_results.sort(key=lambda x: x.get('ip_cov_target', 0))
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nAll experiments completed in {elapsed/60:.1f} minutes")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nWorkflow: Epigenomics ({data['num_tasks']} tasks, {data['num_edges']} edges, {len(pairs)} incomparable pairs)")
    print(f"MCMC iterations: {args.num_iterations:,}")
    
    print(f"\n{'IP-Cov':<10} {'Traces':<8} {'BHPOP F1':<10} {'AND F1':<10} {'Maj F1':<10} {'SHD':<6}")
    print("-" * 60)
    for r in all_results:
        if 'error' not in r:
            print(f"{r.get('ip_cov_realized', 0):<10.2f} {r.get('num_traces', 0):<8} "
                  f"{r.get('bhpop_f1', 0):<10.3f} {r.get('and_f1', 0):<10.3f} {r.get('majority_f1', 0):<10.3f} "
                  f"{r.get('bhpop_shd', 0):<6}")
        else:
            print(f"{r.get('ip_cov_target', 0):<10.2f} {'ERROR':<8}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'workflow': {
            'name': data['name'],
            'num_tasks': data['num_tasks'],
            'num_edges': data['num_edges'],
            'num_incomparable_pairs': len(pairs),
        },
        'config': {
            'num_iterations': args.num_iterations,
            'seed': args.seed,
            'ip_cov_targets': args.ip_cov_targets,
            'burn_in_fraction': BURN_IN_FRACTION,
        },
        'results': all_results,
    }
    
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    print(f"\n✅ Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
