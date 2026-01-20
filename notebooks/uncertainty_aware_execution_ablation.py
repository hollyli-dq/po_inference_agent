#!/usr/bin/env python3
"""
Uncertainty-Aware Execution Ablation Experiment
================================================

This script implements:
1. Queue-jump likelihood ablation: log_successors_queue_jump vs queue_jump
2. Uncertainty-aware execution simulation with confidence thresholds
3. Generates the "one small plot" showing speedup vs violation tradeoff

Key insight: Posterior uncertainty becomes a runtime safety knob.
We execute (or parallelize) an action only when its posterior feasibility
P(a ∈ F_t | D) exceeds a confidence threshold δ.

Usage:
    python uncertainty_aware_execution_ablation.py [--quick]
    
    --quick: Run with reduced iterations for testing
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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
from notebooks.systematic_experiments import (
    build_data_dict,
    greedy_subset_indices_to_target,
    posterior_threshold_mean,
    cp_cov,
    setup_icml_style,
    OUTPUT_DIR,
)

# =========================
# Configuration
# =========================
LIKELIHOODS = ["log_successors_queue_jump", "queue_jump"]
CONFIDENCE_THRESHOLDS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

# MCMC settings
NUM_ITERATIONS_FULL = 1_000_000  # Full run
NUM_ITERATIONS_QUICK = 50_000  # Quick test
BURN_IN_FRACTION = 0.5
SEED_BASE = 42
IP_COV_TARGET = 0.9  # Incomparable pairs coverage target

# Output
ABLATION_OUTPUT_DIR = OUTPUT_DIR / "uncertainty_ablation"
ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Core Functions: Frontier Computation
# =========================

def frontier_from_closure(C: np.ndarray, done_mask: np.ndarray) -> np.ndarray:
    """
    Compute the frontier (feasible actions) given a transitive closure matrix
    and a mask of completed actions.
    
    An action 'a' is in the frontier if:
    - It's not yet done
    - All its predecessors (nodes b where C[b,a]=1) are done
    
    Args:
        C: Transitive closure matrix (n x n). C[b,a]=1 means b must precede a.
        done_mask: Boolean array (n,) where True = action completed.
        
    Returns:
        Array of frontier indices.
    """
    n = C.shape[0]
    
    # For each action, check if all predecessors are done
    # predecessors[a] = {b : C[b,a] == 1}
    # a is feasible iff all predecessors are in done_mask
    
    feasible = np.ones(n, dtype=bool)
    for a in range(n):
        if done_mask[a]:
            feasible[a] = False
            continue
        # Check all predecessors
        preds = np.where(C[:, a] == 1)[0]
        if len(preds) > 0 and not np.all(done_mask[preds]):
            feasible[a] = False
    
    return np.where(feasible)[0]


def posterior_frontier_prob(closure_samples: List[np.ndarray], done_mask: np.ndarray) -> np.ndarray:
    """
    Compute the posterior probability that each action is in the frontier.
    
    P(a ∈ F_t | D) = (1/S) Σ_s 1{a ∈ F_t(C^(s), Done_t)}
    
    Args:
        closure_samples: List of transitive closure matrices from posterior samples
        done_mask: Boolean array of completed actions
        
    Returns:
        Array of probabilities for each action
    """
    n = closure_samples[0].shape[0]
    counts = np.zeros(n, dtype=float)
    S = len(closure_samples)
    
    for C in closure_samples:
        F = frontier_from_closure(C, done_mask)
        counts[F] += 1.0
    
    return counts / max(1, S)


def simulate_uncertainty_executor(
    true_closure: np.ndarray,
    closure_samples: List[np.ndarray],
    delta: float,
) -> Tuple[float, float]:
    """
    Simulate uncertainty-aware execution with confidence threshold δ.
    
    At each step:
    1. Compute P(a ∈ F_t | D) for all remaining actions
    2. Execute all actions where P ≥ δ (parallel batch)
    3. If no action meets threshold, fallback to argmax
    4. Track violations (executing an action before its predecessors are done)
    
    Args:
        true_closure: Ground truth transitive closure matrix
        closure_samples: Posterior samples of closure matrices
        delta: Confidence threshold in [0, 1]
        
    Returns:
        speedup: n / #stages (parallelism measure)
        violation_rate: fraction of executed actions that violated precedence
    """
    n = true_closure.shape[0]
    done_mask = np.zeros(n, dtype=bool)
    stages = 0
    violations = 0
    executed = 0
    
    while done_mask.sum() < n:
        p_feas = posterior_frontier_prob(closure_samples, done_mask)
        
        # Choose batch: all actions with P(feasible) >= delta
        cand = np.where((~done_mask) & (p_feas >= delta))[0]
        
        if cand.size == 0:
            # Fallback: execute the most feasible remaining action
            remaining = np.where(~done_mask)[0]
            cand = np.array([remaining[np.argmax(p_feas[remaining])]])
        
        stages += 1
        
        # Execute batch; measure violations under ground truth
        for a in cand:
            # Check if any predecessor is not done
            preds = np.where(true_closure[:, a] == 1)[0]
            if len(preds) > 0 and np.any(~done_mask[preds]):
                violations += 1
            done_mask[a] = True
            executed += 1
    
    speedup = n / max(1, stages)
    violation_rate = violations / max(1, executed)
    return speedup, violation_rate


def threshold_curve(
    true_closure: np.ndarray,
    closure_samples: List[np.ndarray],
    deltas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute speedup and violation rate curves across threshold values.
    
    Args:
        true_closure: Ground truth closure
        closure_samples: Posterior closure samples
        deltas: Array of threshold values
        
    Returns:
        speedups: Array of speedup values
        viols: Array of violation rates
    """
    speedups = []
    viols = []
    for d in deltas:
        s, v = simulate_uncertainty_executor(true_closure, closure_samples, d)
        speedups.append(s)
        viols.append(v)
    return np.array(speedups), np.array(viols)


# =========================
# Extract Closure Samples from H_trace
# =========================

def extract_closure_samples(
    H_trace: List[np.ndarray],
    burn_in_fraction: float = 0.5,
) -> List[np.ndarray]:
    """
    Extract transitive closure samples from H_trace (single partial order model).
    
    Args:
        H_trace: List of partial order matrices
        burn_in_fraction: Fraction of samples to discard as burn-in
        
    Returns:
        List of transitive closure matrices
    """
    burn_in = int(len(H_trace) * burn_in_fraction)
    post_samples = H_trace[burn_in:]
    
    closure_samples = []
    for H in post_samples:
        # Compute transitive closure
        C = BasicUtils.transitive_closure(H.astype(np.int8))
        closure_samples.append(C)
    
    return closure_samples


# =========================
# Run MCMC for a Single Scenario
# =========================

def run_mcmc_for_scenario(
    scenario_id: Any,
    items: List[int],
    choice_sets: List[List[int]],
    observed_orders: List[List[int]],
    likelihood: str,
    num_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run MCMC experiment for a single scenario with specified likelihood.
    
    Args:
        scenario_id: Scenario identifier (string or int)
        items: List of item IDs
        choice_sets: List of choice sets
        observed_orders: List of observed orderings
        likelihood: "log_successors_queue_jump" or "queue_jump"
        num_iterations: Number of MCMC iterations
        seed: Random seed
        
    Returns:
        Dictionary with MCMC results and extracted closure samples
    """
    print(f"  Running scenario {scenario_id} with {likelihood}, {num_iterations} iterations...")
    
    # Determine noise parameters based on likelihood
    if likelihood == "queue_jump":
        noise_option = "queue_jump"
        noise_beta_prior = 10.0
    else:
        noise_option = "log_successors_queue_jump"
        noise_beta_prior = 1.0
    
    mcmc = mcmc_simulation_po(
        num_iterations=num_iterations,
        items=items,
        choice_sets=choice_sets,
        observed_orders=observed_orders,
        dr=0.5,  # Multiplicative step size for rho (reduced for better acceptance)
        noise_option=noise_option,
        rho_prior=1.0,
        noise_beta_prior=noise_beta_prior,
        K_prior=3,
        fixed_K=None,
        random_seed=seed,
        cycle_length=500,
        epsilon=0.01,  # Trembling-hand epsilon
        softmax_beta_prior=(2.0, 1.0),
        softmax_beta_stepsize=0.1,
    )
    
    # Extract closure samples
    H_trace = mcmc["H_trace"]
    closure_samples = extract_closure_samples(H_trace, BURN_IN_FRACTION)
    
    return {
        "scenario_id": scenario_id,
        "likelihood": likelihood,
        "mcmc": mcmc,
        "closure_samples": closure_samples,
    }


# =========================
# Compute Metrics
# =========================

def compute_structure_metrics(
    mcmc_result: Dict[str, Any],
    true_cover: np.ndarray,
) -> Dict[str, float]:
    """
    Compute structure recovery metrics (F1, SHD) for a single MCMC run.
    """
    H_trace = mcmc_result["mcmc"]["H_trace"]
    burn_in = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn_in:]
    
    if not post:
        return {"cover_f1": 0.0, "precision": 0.0, "recall": 0.0, "shd": float('inf')}
    
    # Aggregate posterior
    inferred_cover = posterior_threshold_mean(post, threshold=0.5)
    
    p, r = precision_recall(true_cover, inferred_cover)
    f1 = f1_score(p, r)
    shd = structural_hamming_distance(true_cover, inferred_cover)
    
    return {
        "cover_f1": f1,
        "precision": p,
        "recall": r,
        "shd": shd,
    }


# =========================
# Plotting Functions
# =========================

def plot_uncertainty_threshold_tradeoff(
    results_by_method: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
    scenario_names: Optional[List[str]] = None,
):
    """
    Generate the 1×2 uncertainty threshold tradeoff plot.
    
    Left panel: Speedup (parallelism) vs δ
    Right panel: Violation rate (safety) vs δ
    
    Args:
        results_by_method: Dict mapping method_name -> {scenario -> (speedups, viols)}
        output_path: Path to save the figure
        scenario_names: Optional list of scenario names for averaging
    """
    setup_icml_style()
    
    # ICML double-column figure size
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharex=True)
    
    # Colors for methods
    colors = {
        "BHPOP (log-successors)": "#377eb8",
        "BHPOP (queue-jump)": "#e41a1c",
    }
    
    markers = {
        "BHPOP (log-successors)": "o",
        "BHPOP (queue-jump)": "s",
    }
    
    for method_name, scenario_results in results_by_method.items():
        # Average across scenarios
        all_speedups = []
        all_viols = []
        
        for sid, (speedups, viols) in scenario_results.items():
            all_speedups.append(speedups)
            all_viols.append(viols)
        
        if not all_speedups:
            continue
            
        mean_speedups = np.mean(all_speedups, axis=0)
        std_speedups = np.std(all_speedups, axis=0)
        mean_viols = np.mean(all_viols, axis=0)
        std_viols = np.std(all_viols, axis=0)
        
        color = colors.get(method_name, "#999999")
        marker = markers.get(method_name, "o")
        
        # Left panel: Speedup
        axes[0].plot(
            CONFIDENCE_THRESHOLDS, mean_speedups,
            color=color, marker=marker, markersize=4,
            linewidth=1.2, label=method_name
        )
        if len(all_speedups) > 1:
            axes[0].fill_between(
                CONFIDENCE_THRESHOLDS,
                mean_speedups - std_speedups,
                mean_speedups + std_speedups,
                color=color, alpha=0.2
            )
        
        # Right panel: Violation rate
        axes[1].plot(
            CONFIDENCE_THRESHOLDS, mean_viols,
            color=color, marker=marker, markersize=4,
            linewidth=1.2, label=method_name
        )
        if len(all_viols) > 1:
            axes[1].fill_between(
                CONFIDENCE_THRESHOLDS,
                mean_viols - std_viols,
                mean_viols + std_viols,
                color=color, alpha=0.2
            )
    
    # Axis labels and formatting
    axes[0].set_xlabel(r"Confidence threshold $\delta$")
    axes[0].set_ylabel("Speedup (n / #stages)")
    axes[0].set_title("Parallelism Benefit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.8)
    
    axes[1].set_xlabel(r"Confidence threshold $\delta$")
    axes[1].set_ylabel("Violation rate")
    axes[1].set_title("Safety Risk")
    axes[1].grid(True, alpha=0.3)
    
    # Set x-axis ticks
    axes[0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved uncertainty threshold plot to: {output_path}")


def plot_f1_comparison(
    df: pd.DataFrame,
    output_path: Path,
):
    """
    Plot F1 comparison between likelihoods by scenario.
    """
    setup_icml_style()
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    scenarios = df["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.35
    
    colors = {
        "log_successors_queue_jump": "#377eb8",
        "queue_jump": "#e41a1c",
    }
    labels = {
        "log_successors_queue_jump": "Log-successors",
        "queue_jump": "Queue-jump",
    }
    
    for i, lh in enumerate(LIKELIHOODS):
        df_lh = df[df["likelihood"] == lh]
        f1_vals = [df_lh[df_lh["scenario"] == s]["cover_f1"].values[0] 
                   if len(df_lh[df_lh["scenario"] == s]) > 0 else 0 
                   for s in scenarios]
        ax.bar(x + i * width, f1_vals, width, label=labels[lh], color=colors[lh])
    
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Cover F1")
    ax.set_title("Structure Recovery: Likelihood Ablation")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved F1 comparison plot to: {output_path}")


# =========================
# Main Experiment Runner
# =========================

def run_single_ablation_experiment(
    scenario_id: Any,
    items: List[int],
    choice_sets: List[List[int]],
    observed_orders: List[List[int]],
    true_cover: np.ndarray,
    true_closure: np.ndarray,
    likelihood: str,
    num_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run a single ablation experiment for one scenario and likelihood.
    """
    result = run_mcmc_for_scenario(
        scenario_id=scenario_id,
        items=items,
        choice_sets=choice_sets,
        observed_orders=observed_orders,
        likelihood=likelihood,
        num_iterations=num_iterations,
        seed=seed,
    )
    
    # Compute structure metrics
    metrics = compute_structure_metrics(result, true_cover)
    
    # Compute uncertainty threshold curves
    closure_samples = result["closure_samples"]
    if len(closure_samples) >= 10:
        speedups, viols = threshold_curve(true_closure, closure_samples, CONFIDENCE_THRESHOLDS)
    else:
        speedups, viols = np.zeros(len(CONFIDENCE_THRESHOLDS)), np.ones(len(CONFIDENCE_THRESHOLDS))
    
    return {
        "scenario_id": scenario_id,
        "likelihood": likelihood,
        "H_trace": result["mcmc"]["H_trace"],
        "metrics": metrics,
        "speedups": speedups,
        "viols": viols,
    }


def main(quick: bool = False):
    """
    Run the complete ablation experiment.
    """
    print("=" * 70)
    print("UNCERTAINTY-AWARE EXECUTION ABLATION EXPERIMENT")
    print("=" * 70)
    
    num_iterations = NUM_ITERATIONS_QUICK if quick else NUM_ITERATIONS_FULL
    print(f"Mode: {'QUICK TEST' if quick else 'FULL RUN'}")
    print(f"Iterations: {num_iterations}")
    print(f"Likelihoods: {LIKELIHOODS}")
    print(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
    print()
    
    # Load data
    print("Loading data...")
    data = build_data_dict(PROJECT_ROOT)
    
    # Prepare experiments for all scenarios and likelihoods
    experiments = []
    
    for scenario_id in data["scenario_ids"]:
        scenario_data = data["scenario_data"][scenario_id]
        true_closure = scenario_data["true_closure"]
        true_cover = scenario_data["true_cover"]
        task_ids = scenario_data["task_ids"]
        n_items = len(task_ids)
        items = list(range(n_items))  # Local indices
        
        # Get local orders for this scenario
        all_orders = data["orders_local_by_assessor"].get(scenario_id, [])
        if not all_orders:
            print(f"  Scenario {scenario_id}: No traces, skipping")
            continue
        
        # Subsample to target incomparable pairs coverage
        idxs, cov = greedy_subset_indices_to_target(
            all_orders, true_closure, IP_COV_TARGET, seed=SEED_BASE
        )
        print(f"  Scenario {scenario_id}: {len(idxs)} traces, realized IP-Cov = {cov:.3f}")
        
        sampled_orders = [all_orders[i] for i in idxs]
        # Choice sets: full observation (all items in each order)
        sampled_choice_sets = [list(range(n_items)) for _ in sampled_orders]
        
        for lh in LIKELIHOODS:
            experiments.append({
                "scenario_id": scenario_id,
                "items": items,
                "choice_sets": sampled_choice_sets,
                "observed_orders": sampled_orders,
                "true_cover": true_cover,
                "true_closure": true_closure,
                "likelihood": lh,
                "num_iterations": num_iterations,
                "seed": SEED_BASE + hash(scenario_id) % 10000,  # Use hash for string scenario_id
            })
    
    # Run experiments in parallel
    print("\n" + "=" * 70)
    print(f"RUNNING {len(experiments)} EXPERIMENTS IN PARALLEL")
    print("=" * 70)
    
    all_results = []
    max_workers = min(len(experiments), 6)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_ablation_experiment, **exp): exp
            for exp in experiments
        }
        
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"  ✓ Scenario {result['scenario_id']}, {result['likelihood']}: "
                      f"F1={result['metrics']['cover_f1']:.3f}")
            except Exception as e:
                print(f"  ✗ Scenario {exp['scenario_id']}, {exp['likelihood']} failed: {e}")
    
    # Compile structure metrics
    print("\n" + "=" * 70)
    print("STRUCTURE METRICS")
    print("=" * 70)
    
    metrics_rows = []
    for result in all_results:
        metrics_rows.append({
            "scenario": result["scenario_id"],
            "likelihood": result["likelihood"],
            **result["metrics"],
        })
    
    df_all_metrics = pd.DataFrame(metrics_rows)
    df_all_metrics.to_csv(ABLATION_OUTPUT_DIR / "structure_metrics.csv", index=False)
    
    for lh in LIKELIHOODS:
        df_lh = df_all_metrics[df_all_metrics["likelihood"] == lh]
        print(f"\n{lh}:")
        for _, row in df_lh.iterrows():
            print(f"  Scenario {row['scenario']}: F1={row['cover_f1']:.3f}, SHD={row['shd']}")
    
    # Compile uncertainty-aware execution curves
    print("\n" + "=" * 70)
    print("UNCERTAINTY-AWARE EXECUTION CURVES")
    print("=" * 70)
    
    method_labels = {
        "log_successors_queue_jump": "BHPOP (log-successors)",
        "queue_jump": "BHPOP (queue-jump)",
    }
    
    results_by_method = {label: {} for label in method_labels.values()}
    
    for result in all_results:
        method_name = method_labels[result["likelihood"]]
        sid = result["scenario_id"]
        results_by_method[method_name][sid] = (result["speedups"], result["viols"])
        
        print(f"  {result['likelihood']} / Scenario {sid}:")
        print(f"    δ=0.5: speedup={result['speedups'][0]:.2f}, viol={result['viols'][0]:.3f}")
        print(f"    δ=0.9: speedup={result['speedups'][4]:.2f}, viol={result['viols'][4]:.3f}")
    
    # Save curves data
    curves_data = {
        method: {str(sid): {"speedups": s.tolist(), "viols": v.tolist()} 
                 for sid, (s, v) in scenarios.items()}
        for method, scenarios in results_by_method.items()
    }
    with open(ABLATION_OUTPUT_DIR / "threshold_curves.json", "w") as f:
        json.dump(curves_data, f, indent=2)
    
    # Save H_trace for each experiment
    for result in all_results:
        exp_dir = ABLATION_OUTPUT_DIR / f"scenario_{result['scenario_id']}_{result['likelihood']}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "H_trace.pkl", "wb") as f:
            pickle.dump(result["H_trace"], f)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Main plot: Uncertainty threshold tradeoff
    plot_uncertainty_threshold_tradeoff(
        results_by_method,
        ABLATION_OUTPUT_DIR / "uncertainty_threshold_ablation.pdf",
    )
    
    # F1 comparison plot
    plot_f1_comparison(
        df_all_metrics,
        ABLATION_OUTPUT_DIR / "f1_likelihood_comparison.pdf",
    )
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nStructure Recovery (Cover-F1):")
    summary = df_all_metrics.groupby("likelihood").agg({
        "cover_f1": ["mean", "std"],
        "shd": ["mean", "std"],
    }).round(3)
    print(summary)
    
    print("\nUncertainty-Aware Execution (mean across scenarios):")
    for method_name, scenarios in results_by_method.items():
        if not scenarios:
            continue
        all_s = [s for (s, v) in scenarios.values()]
        all_v = [v for (s, v) in scenarios.values()]
        mean_s = np.mean(all_s, axis=0)
        mean_v = np.mean(all_v, axis=0)
        print(f"\n{method_name}:")
        for i, delta in enumerate(CONFIDENCE_THRESHOLDS):
            print(f"  δ={delta:.2f}: speedup={mean_s[i]:.2f}, violation={mean_v[i]:.4f}")
    
    print(f"\n✅ All results saved to: {ABLATION_OUTPUT_DIR}")
    
    return df_all_metrics, results_by_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty-Aware Execution Ablation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with reduced iterations")
    args = parser.parse_args()
    
    main(quick=args.quick)
