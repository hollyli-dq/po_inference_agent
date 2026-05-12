#!/usr/bin/env python3
"""
Queue Jump Baseline for Cloud-IaC (Nicholls et al.)
==================================================

This script runs the queue jump model from Nicholls as a baseline comparison
for the Cloud-IaC scenarios.

The queue jump model uses the NLE-based likelihood:
    P(y_j | remaining) = (1-ε) * (NLE_first / NLE_total) + ε / |remaining|

This is the #P-hard baseline that requires linear extension counting.
Compare with BHPOP's frontier-softmax which avoids this computation.

Configuration:
- Fixed epsilon = 0.01 
- IP-Cov targets: 0.5, 0.7, 0.85, 0.95
- 1 million iterations
- Records timing for comparison with BHPOP

Usage:
    python queue_jump_baseline_cloud_iac.py [--quick] [--ip_cov 0.95]
    
    --quick: Run with 50k iterations for testing
    --ip_cov: Specific IP-Cov target (default: all targets)
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Import from systematic experiments for data loading
from training_scripts.systematic_experiments import (
    build_data_dict,
    greedy_subset_indices_to_target,
    posterior_threshold_mean,
    setup_icml_style,
    OUTPUT_DIR,
)

# =========================
# Configuration
# =========================
EPSILON = 0.01  # Fixed noise parameter (same as BHPOP experiments)
IP_COV_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]

# MCMC settings
NUM_ITERATIONS_FULL = 1_000_000  # 1 million iterations
NUM_ITERATIONS_QUICK = 50_000   # Quick test
BURN_IN_FRACTION = 0.5
SEED_BASE = 42

# Use fixed K for better mixing (queue_jump is slow)
USE_FIXED_K = False

# Output
QJ_OUTPUT_DIR = OUTPUT_DIR / "queue_jump_baseline"
QJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Core Experiment Functions
# =========================

def run_queue_jump_experiment(
    scenario_id: str,
    items: List[int],
    choice_sets: List[List[int]],
    observed_orders: List[List[int]],
    true_cover: np.ndarray,
    num_iterations: int,
    seed: int,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """
    Run MCMC with Nicholls-style queue jump likelihood (NLE-based).
    
    This is the BASELINE method that requires #P-hard NLE computation.
    
    Args:
        scenario_id: Scenario identifier
        items: List of item IDs (local indices)
        choice_sets: List of choice sets
        observed_orders: List of observed orderings
        true_cover: Ground truth transitive reduction
        num_iterations: Number of MCMC iterations
        seed: Random seed
        epsilon: Fixed noise probability
        
    Returns:
        Dictionary with timing, metrics, and MCMC results
    """
    n = len(items)
    
    # Fixed K mode
    fixed_K = n // 2 if USE_FIXED_K else None
    
    print(f"  [{scenario_id}] Running Queue Jump: {len(observed_orders)} traces, "
          f"{num_iterations:,} iters, K={'fixed='+str(fixed_K) if fixed_K else 'RJ'}...")
    
    # Time the MCMC run
    start_time = time.time()
    wall_start = datetime.now()
    
    # KEY: noise_option="queue_jump" uses Nicholls NLE-based likelihood
    mcmc_result = mcmc_simulation_po(
        num_iterations=num_iterations,
        items=items,
        choice_sets=choice_sets,
        observed_orders=observed_orders,
        dr=0.5,
        noise_option="queue_jump",  # <<< Nicholls baseline (NLE-based)
        rho_prior=1.0,
        noise_beta_prior=10.0,  # Prior for Beta(1, beta) on epsilon
        K_prior=3,
        fixed_K=fixed_K,
        random_seed=seed,
        cycle_length=500,
        epsilon=epsilon,  # Fixed trembling-hand epsilon
    )
    
    elapsed_seconds = time.time() - start_time
    wall_elapsed = (datetime.now() - wall_start).total_seconds()
    
    # Extract posterior
    H_trace = mcmc_result.get('H_trace', [])
    burn = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn:]
    
    if not post:
        return {
            'scenario_id': scenario_id,
            'error': 'No posterior samples',
            'elapsed_seconds': elapsed_seconds,
        }
    
    # Compute inferred cover
    inferred = posterior_threshold_mean(post, threshold=0.5)
    
    # Evaluate metrics
    p, r = precision_recall(true_cover, inferred)
    f1 = f1_score(p, r)
    shd = structural_hamming_distance(true_cover, inferred)
    
    # Timing metrics
    time_per_iter_ms = (elapsed_seconds / num_iterations) * 1000
    
    return {
        'scenario_id': scenario_id,
        'method': 'queue_jump',
        'num_iterations': num_iterations,
        'num_traces': len(observed_orders),
        'n_items': n,
        'epsilon': epsilon,
        # Timing (KEY for comparison)
        'elapsed_seconds': elapsed_seconds,
        'wall_elapsed_seconds': wall_elapsed,
        'time_per_iter_ms': time_per_iter_ms,
        # Metrics
        'precision': p,
        'recall': r,
        'f1': f1,
        'shd': shd,
        'inferred_edges': int(inferred.sum()),
        'true_edges': int(true_cover.sum()),
        'acceptance_rate': mcmc_result.get('overall_acceptance_rate', 0.0),
        # MCMC output (for diagnostics)
        'mcmc_result': mcmc_result,
        'inferred_cover': inferred,
    }


def run_single_ip_cov_experiment(
    scenario_id: str,
    scenario_data: Dict[str, Any],
    all_orders: List[List[int]],
    ip_cov_target: float,
    num_iterations: int,
    seed: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Run queue jump experiment for a single scenario at a specific IP-Cov target.
    """
    try:
        true_closure = scenario_data["true_closure"]
        true_cover = scenario_data["true_cover"]
        n_items = len(scenario_data["task_ids"])
        items = list(range(n_items))
        
        # Subsample to target IP-Cov
        idxs, realized_cov = greedy_subset_indices_to_target(
            all_orders, true_closure, ip_cov_target, seed=seed
        )
        
        sampled_orders = [all_orders[i] for i in idxs]
        choice_sets = [list(range(n_items)) for _ in sampled_orders]
        
        print(f"  [{scenario_id}] IP-Cov target={ip_cov_target:.2f}, "
              f"realized={realized_cov:.3f}, traces={len(sampled_orders)}")
        
        # Run experiment
        result = run_queue_jump_experiment(
            scenario_id=scenario_id,
            items=items,
            choice_sets=choice_sets,
            observed_orders=sampled_orders,
            true_cover=true_cover,
            num_iterations=num_iterations,
            seed=seed,
            epsilon=epsilon,
        )
        
        result['ip_cov_target'] = ip_cov_target
        result['ip_cov_realized'] = realized_cov
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'scenario_id': scenario_id,
            'ip_cov_target': ip_cov_target,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


# =========================
# Main Experiment Runner
# =========================

def main(quick: bool = False, ip_cov_filter: Optional[float] = None):
    """
    Run the queue jump baseline experiment for the Cloud-IaC dataset.
    """
    print("=" * 70)
    print("QUEUE JUMP BASELINE FOR ALIYUN CLOUD DATA")
    print("=" * 70)
    print(f"Method: Nicholls et al. Queue Jump (NLE-based)")
    print(f"        This requires #P-hard linear extension counting!")
    print()
    
    num_iterations = NUM_ITERATIONS_QUICK if quick else NUM_ITERATIONS_FULL
    ip_targets = [ip_cov_filter] if ip_cov_filter else IP_COV_TARGETS
    
    print(f"Mode: {'QUICK TEST' if quick else 'FULL RUN'}")
    print(f"Iterations: {num_iterations:,}")
    print(f"Fixed epsilon: {EPSILON}")
    print(f"IP-Cov targets: {ip_targets}")
    print(f"K mode: {'FIXED K=n/2' if USE_FIXED_K else 'Reversible Jump'}")
    print()
    
    # Load data
    print("Loading Cloud-IaC data...")
    try:
        data = build_data_dict(PROJECT_ROOT)
    except FileNotFoundError as e:
        print(f"ERROR: Could not load data: {e}")
        print("Make sure data/cloud_iac_dataset/ exists with ground_truth/, expert_traces/, and traces/")
        return None, None
    
    scenario_ids = data["scenario_ids"]
    print(f"Loaded {len(scenario_ids)} scenarios: {scenario_ids}")
    
    # Prepare all experiments
    experiments = []
    for scenario_id in scenario_ids:
        scenario_data = data["scenario_data"][scenario_id]
        all_orders = data["orders_local_by_assessor"].get(scenario_id, [])
        
        if not all_orders:
            print(f"  Scenario {scenario_id}: No traces, skipping")
            continue
        
        for ip_target in ip_targets:
            experiments.append({
                'scenario_id': scenario_id,
                'scenario_data': scenario_data,
                'all_orders': all_orders,
                'ip_cov_target': ip_target,
                'num_iterations': num_iterations,
                'seed': SEED_BASE + hash(scenario_id) % 10000 + int(ip_target * 100),
                'epsilon': EPSILON,
            })
    
    print(f"\nRunning {len(experiments)} experiments in PARALLEL (5 workers)...")
    print("=" * 70)
    
    # Run experiments in parallel
    all_results = []
    total_start = datetime.now()
    max_workers = 5  # Use 5 cores as requested
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_ip_cov_experiment, **exp): exp
            for exp in experiments
        }
        
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                
                if 'error' not in result:
                    print(f"  ✓ [{result['scenario_id']} @ IP-Cov={result['ip_cov_target']:.2f}] "
                          f"F1={result['f1']:.3f}, Time={result['elapsed_seconds']:.1f}s "
                          f"({result['time_per_iter_ms']:.4f} ms/iter)")
                else:
                    print(f"  ✗ [{exp['scenario_id']} @ IP-Cov={exp['ip_cov_target']}] "
                          f"ERROR: {result['error']}")
            except Exception as e:
                print(f"  ✗ [{exp['scenario_id']} @ IP-Cov={exp['ip_cov_target']}] FAILED: {e}")
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    # Compile results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results_rows = []
    for r in all_results:
        if 'error' not in r:
            results_rows.append({
                'scenario': r['scenario_id'],
                'ip_cov_target': r['ip_cov_target'],
                'ip_cov_realized': r['ip_cov_realized'],
                'num_traces': r['num_traces'],
                'n_items': r['n_items'],
                'f1': r['f1'],
                'precision': r['precision'],
                'recall': r['recall'],
                'shd': r['shd'],
                'elapsed_seconds': r['elapsed_seconds'],
                'time_per_iter_ms': r['time_per_iter_ms'],
                'acceptance_rate': r['acceptance_rate'],
            })
    
    df = pd.DataFrame(results_rows)
    
    if len(df) > 0:
        # Print results table
        print(f"\n{'Scenario':<25} {'IP-Cov':<10} {'Traces':<8} {'F1':<8} {'SHD':<6} "
              f"{'Time(s)':<10} {'ms/iter':<10}")
        print("-" * 85)
        for _, row in df.iterrows():
            print(f"{row['scenario']:<25} {row['ip_cov_realized']:<10.3f} {row['num_traces']:<8} "
                  f"{row['f1']:<8.3f} {row['shd']:<6} {row['elapsed_seconds']:<10.1f} "
                  f"{row['time_per_iter_ms']:<10.4f}")
        
        # Aggregate statistics
        print("\n" + "-" * 85)
        print("AGGREGATE STATISTICS:")
        print(f"  Total experiments: {len(df)}")
        print(f"  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"  Mean F1: {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
        print(f"  Mean time per iteration: {df['time_per_iter_ms'].mean():.4f} ms")
        print(f"  Mean experiment time: {df['elapsed_seconds'].mean():.1f}s")
        
        # Save results
        csv_path = QJ_OUTPUT_DIR / "queue_jump_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")
        
        # Save detailed results with MCMC output
        for r in all_results:
            if 'error' not in r and 'mcmc_result' in r:
                exp_dir = QJ_OUTPUT_DIR / f"{r['scenario_id']}_ipcov{r['ip_cov_target']:.2f}".replace('.', '_')
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save H_trace
                with open(exp_dir / "H_trace.pkl", "wb") as f:
                    pickle.dump(r['mcmc_result'].get('H_trace', []), f)
                
                # Save inferred cover
                np.save(exp_dir / "inferred_cover.npy", r['inferred_cover'])
                
                # Save summary
                summary = {k: v for k, v in r.items() 
                          if k not in ['mcmc_result', 'inferred_cover']}
                with open(exp_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=float)
        
        # Save overall summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'method': 'queue_jump',
            'num_iterations': num_iterations,
            'epsilon': EPSILON,
            'ip_cov_targets': ip_targets,
            'total_elapsed_seconds': total_elapsed,
            'scenarios': scenario_ids,
            'results': results_rows,
        }
        with open(QJ_OUTPUT_DIR / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
    
    # Key comparison note
    print("\n" + "=" * 70)
    print("TIMING COMPARISON NOTE")
    print("=" * 70)
    print("Compare these timings with BHPOP (frontier-softmax) results:")
    print("  - Queue Jump requires NLE computation (#P-hard)")
    print("  - BHPOP uses frontier-softmax (polynomial time)")
    print("  - Expected: BHPOP should be significantly faster")
    print(f"\nQueue Jump mean time per iteration: {df['time_per_iter_ms'].mean():.4f} ms" if len(df) > 0 else "")
    
    return df, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue Jump Baseline for Cloud-IaC")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with 50k iterations")
    parser.add_argument("--ip_cov", type=float, default=None,
                       help="Specific IP-Cov target (default: all targets)")
    
    args = parser.parse_args()
    main(quick=args.quick, ip_cov_filter=args.ip_cov)
