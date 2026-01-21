#!/usr/bin/env python3
"""
Generate JSON summary files for each experiment.

For each experiment directory, creates a JSON file containing:
- Scenario name
- Experiment configuration (epsilon, IP coverage, etc.)
- Average H matrix (posterior mean)
- H items (resource/task names)
- Posterior statistics
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RESULTS_DIR = Path(__file__).parent / "systematic_experiment_results"
BURN_IN_FRACTION = 0.5
THIN = 1

# Experiment grid configuration (must match systematic_experiments.py)
IP_COV_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]
EPS_JUMP_LIST = [0.005, 0.01, 0.02, 0.05]
SCENARIO_ORDER = [
    'dual_zone_ecs_slb',
    'dual_zone_ecs_slb_rds', 
    'eip_slb_ecs',
    'simple_ecs',
    'slb_ecs_rds',
    'slb_ecs_redis'
]
LIKELIHOODS = ['log_successors_queue_jump']


def exp_id_to_config(exp_id: int) -> dict:
    """
    Derive experiment configuration from exp_id.
    
    The experiments are run in nested loops:
        for ip_cov in IP_COV_TARGETS:      # 5 levels
            for scenario in SCENARIO_ORDER: # 6 scenarios
                for eps in EPS_JUMP_LIST:   # 4 values
                    for lh in LIKELIHOODS:  # 1 value
                        exp_id += 1
    
    So exp_id 1-24 = IP-Cov 0.6, exp_id 25-48 = IP-Cov 0.7, etc.
    """
    # exp_id is 1-indexed
    idx = exp_id - 1
    
    n_scenarios = len(SCENARIO_ORDER)
    n_eps = len(EPS_JUMP_LIST)
    n_lh = len(LIKELIHOODS)
    
    # Number of experiments per IP-Cov level
    per_ipcov = n_scenarios * n_eps * n_lh  # 6 * 4 * 1 = 24
    
    ip_cov_idx = idx // per_ipcov
    remainder = idx % per_ipcov
    
    # Within IP-Cov level: scenarios × eps × likelihood
    per_scenario = n_eps * n_lh  # 4
    scenario_idx = remainder // per_scenario
    remainder2 = remainder % per_scenario
    
    eps_idx = remainder2 // n_lh
    lh_idx = remainder2 % n_lh
    
    # Bounds check
    if ip_cov_idx >= len(IP_COV_TARGETS):
        return None
    if scenario_idx >= len(SCENARIO_ORDER):
        return None
    
    return {
        'ip_cov_target': IP_COV_TARGETS[ip_cov_idx],
        'scenario': SCENARIO_ORDER[scenario_idx],
        'eps_jump': EPS_JUMP_LIST[eps_idx],
        'likelihood': LIKELIHOODS[lh_idx],
    }


# Load ip_cov_realized lookup from baseline data
# The baseline rows have correct ip_cov_realized computed during the original run
IP_COV_REALIZED_LOOKUP = {}
for csv_name in ['experiment_summary_t0.30.csv', 'experiment_summary_t0.33.csv', 'experiment_summary_t0.40.csv']:
    csv_path = RESULTS_DIR / csv_name
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            baselines = df[df['method'] == 'majority']
            for _, row in baselines.iterrows():
                key = (row['scenario'], row['ip_cov_target'])
                if key not in IP_COV_REALIZED_LOOKUP:
                    IP_COV_REALIZED_LOOKUP[key] = row['ip_cov_realized']
            print(f"Loaded {len(IP_COV_REALIZED_LOOKUP)} ip_cov_realized values from {csv_name}")
            break
        except Exception as e:
            print(f"Warning: Could not load {csv_name}: {e}")

if not IP_COV_REALIZED_LOOKUP:
    print("Warning: Could not load ip_cov_realized lookup table")

# Load global metadata
metadata_path = RESULTS_DIR / "experiment_metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        global_metadata = json.load(f)
else:
    global_metadata = {}

# Try to load scenario data from aliyun_data if it exists
ALIYUN_DATA_DIR = PROJECT_ROOT / "aliyun_data"
scenario_metadata = {}

if ALIYUN_DATA_DIR.exists() and (ALIYUN_DATA_DIR / "manual_scenarios").exists():
    print("Loading scenario data from aliyun_data/manual_scenarios...")
    for scenario_file in (ALIYUN_DATA_DIR / "manual_scenarios").glob("*.json"):
        try:
            with open(scenario_file) as f:
                scenario_data = json.load(f)
                edges = scenario_data.get("edges", [])
                tasks = sorted({t for edge in edges for t in edge})
                scenario_metadata[scenario_file.stem] = {
                    "task_ids": tasks,
                    "n_tasks": len(tasks)
                }
        except Exception as e:
            print(f"Warning: Could not load {scenario_file}: {e}")
else:
    print("Note: aliyun_data/manual_scenarios not found. Will use generic task IDs.")


def compute_posterior_statistics(H_trace, burn_in_fraction=0.5, thin=1):
    """
    Compute posterior statistics from H_trace.
    
    Args:
        H_trace: List of H matrices (numpy arrays)
        burn_in_fraction: Fraction of samples to discard as burn-in
        thin: Thinning interval
    
    Returns:
        Dictionary with posterior statistics
    """
    if not H_trace or len(H_trace) == 0:
        return {
            "error": "Empty H_trace",
            "n_samples_total": 0,
            "n_samples_post_burnin": 0
        }
    
    n_total = len(H_trace)
    burn_in_idx = int(n_total * burn_in_fraction)
    post_samples = H_trace[burn_in_idx::thin]
    
    if len(post_samples) == 0:
        return {
            "error": "No samples after burn-in",
            "n_samples_total": n_total,
            "n_samples_post_burnin": 0
        }
    
    # Stack samples and compute statistics
    H_stack = np.stack(post_samples, axis=0)  # shape: (n_samples, n, n)
    
    # Average H (posterior mean)
    avg_H = np.mean(H_stack, axis=0)
    
    # Posterior standard deviation (element-wise)
    std_H = np.std(H_stack, axis=0)
    
    # Count of edges (i -> j) with high posterior probability
    edges_confident = np.sum((avg_H > 0.8) | (avg_H < 0.2))
    total_pairs = avg_H.shape[0] * (avg_H.shape[0] - 1)  # exclude diagonal
    
    return {
        "n_samples_total": n_total,
        "n_samples_post_burnin": len(post_samples),
        "burn_in_fraction": burn_in_fraction,
        "thin": thin,
        "avg_H": avg_H.tolist(),  # Convert to list for JSON serialization
        "std_H": std_H.tolist(),
        "avg_H_mean": float(np.mean(avg_H[np.triu_indices_from(avg_H, k=1)])),  # mean of upper triangle
        "avg_H_std": float(np.mean(std_H[np.triu_indices_from(std_H, k=1)])),   # mean std of upper triangle
        "n_edges_confident": int(edges_confident),
        "frac_edges_confident": float(edges_confident / total_pairs) if total_pairs > 0 else 0.0,
    }


def process_experiment(exp_dir: Path):
    """
    Process a single experiment directory and generate JSON summary.
    
    Args:
        exp_dir: Path to experiment directory
    
    Returns:
        Dictionary with experiment summary, or None if processing failed
    """
    exp_name = exp_dir.name
    print(f"Processing {exp_name}...")
    
    # Parse experiment name: exp_{id}_{scenario}
    parts = exp_name.split("_", 2)
    if len(parts) < 3:
        print(f"  Warning: Cannot parse experiment name: {exp_name}")
        return None
    
    exp_id_str = parts[1]
    scenario_id = parts[2]
    
    try:
        exp_id = int(exp_id_str)
    except ValueError:
        print(f"  Warning: Invalid exp_id: {exp_id_str}")
        return None
    
    # Load H_trace
    h_trace_path = exp_dir / "H_trace.pkl"
    if not h_trace_path.exists():
        print(f"  Warning: H_trace.pkl not found in {exp_name}")
        return None
    
    try:
        with open(h_trace_path, "rb") as f:
            H_trace = pickle.load(f)
    except Exception as e:
        print(f"  Error loading H_trace.pkl: {e}")
        return None
    
    # Load parameter traces if available
    param_traces_path = exp_dir / "param_traces.pkl"
    param_traces = {}
    if param_traces_path.exists():
        try:
            with open(param_traces_path, "rb") as f:
                param_traces = pickle.load(f)
        except Exception as e:
            print(f"  Warning: Could not load param_traces.pkl: {e}")
    
    # Get configuration from exp_id (derived from experiment loop structure)
    derived_config = exp_id_to_config(exp_id)
    if derived_config is None:
        print(f"  Warning: Could not derive config for exp_id={exp_id}")
        config = {}
    else:
        # Verify scenario matches
        if derived_config['scenario'] != scenario_id:
            print(f"  Warning: Scenario mismatch for exp_{exp_id}: expected {derived_config['scenario']}, got {scenario_id}")
        
        # Look up actual ip_cov_realized from baseline data
        ip_cov_target = derived_config['ip_cov_target']
        lookup_key = (scenario_id, ip_cov_target)
        ip_cov_realized = IP_COV_REALIZED_LOOKUP.get(lookup_key, ip_cov_target)
        
        config = {
            "ip_cov_target": ip_cov_target,
            "ip_cov_realized": ip_cov_realized,
            "eps_jump": derived_config['eps_jump'],
            "likelihood": derived_config['likelihood'],
        }
    
    # Get scenario metadata
    # If we don't have task names, we'll infer from H_trace dimensions
    if scenario_id in scenario_metadata:
        scenario_tasks = scenario_metadata[scenario_id]["task_ids"]
        n_tasks = len(scenario_tasks)
    elif H_trace and len(H_trace) > 0:
        # Infer from H matrix dimensions
        n_tasks = H_trace[0].shape[0]
        scenario_tasks = [f"task_{i}" for i in range(n_tasks)]
    else:
        scenario_tasks = []
        n_tasks = 0
    
    # Compute posterior statistics
    posterior_stats = compute_posterior_statistics(H_trace, BURN_IN_FRACTION, THIN)
    
    # Build parameter trace summary
    param_summary = {}
    for key, trace in param_traces.items():
        if isinstance(trace, (list, np.ndarray)):
            trace_array = np.asarray(trace)
            if trace_array.size > 0:
                burn_idx = int(len(trace_array) * BURN_IN_FRACTION)
                post_trace = trace_array[burn_idx:]
                param_summary[key] = {
                    "mean": float(np.mean(post_trace)),
                    "std": float(np.std(post_trace)),
                    "median": float(np.median(post_trace)),
                    "min": float(np.min(post_trace)),
                    "max": float(np.max(post_trace)),
                }
    
    # Build final summary
    summary = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "scenario_name": scenario_id,
        "configuration": {
            **config,
            "num_iterations": global_metadata.get("num_iterations", None),
            "burn_in_fraction": BURN_IN_FRACTION,
            "thin": THIN,
        },
        "scenario": {
            "task_ids": scenario_tasks,
            "n_tasks": n_tasks,
        },
        "posterior": posterior_stats,
        "parameters": param_summary,
        "metadata": {
            "model_type": global_metadata.get("model_type", "single_partial_order_per_scenario"),
            "timestamp": global_metadata.get("timestamp", ""),
        }
    }
    
    return summary


def main():
    """Main function to process all experiments."""
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("exp_")])
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print(f"Processing experiments with burn-in={BURN_IN_FRACTION}, thin={THIN}...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for exp_dir in exp_dirs:
        summary = process_experiment(exp_dir)
        
        if summary is not None:
            # Save JSON summary
            output_path = exp_dir / "summary.json"
            try:
                with open(output_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"  ✓ Saved {output_path}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error saving summary: {e}")
                failed_count += 1
        else:
            failed_count += 1
        
        print()
    
    print("=" * 60)
    print(f"Summary generation complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(exp_dirs)}")
    
    # Also create a single consolidated summary file
    all_summaries = []
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                    # Add lightweight version (without full matrices)
                    lightweight = {
                        "experiment_name": summary["experiment_name"],
                        "scenario_name": summary["scenario_name"],
                        "configuration": summary["configuration"],
                        "posterior_stats": {
                            k: v for k, v in summary["posterior"].items() 
                            if k not in ["avg_H", "std_H"]
                        },
                        "parameters": summary.get("parameters", {}),
                    }
                    all_summaries.append(lightweight)
            except Exception as e:
                print(f"Warning: Could not load {summary_path}: {e}")
    
    # Save consolidated summary
    consolidated_path = RESULTS_DIR / "all_experiments_summary.json"
    with open(consolidated_path, "w") as f:
        json.dump({
            "metadata": global_metadata,
            "n_experiments": len(all_summaries),
            "experiments": all_summaries
        }, f, indent=2)
    
    print(f"\n✓ Saved consolidated summary to {consolidated_path}")


if __name__ == "__main__":
    main()
