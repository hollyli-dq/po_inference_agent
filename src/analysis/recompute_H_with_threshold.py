#!/usr/bin/env python3
"""
Recompute final_H graphs with different thresholds and methods.
Pure Python approach - reads avg_H.pkl files and recomputes metrics.

Usage:
    python src/analysis/recompute_H_with_threshold.py --threshold 0.4 --mode single_po
    python src/analysis/recompute_H_with_threshold.py --threshold 0.5 --mode marginal
"""
import sys
import argparse
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance, _incomparable_pairs
from src.utils.cloud_iac_dataset import resolve_cloud_iac_data_root, resolve_cloud_iac_ground_truth_dir
from src.utils.po_fun import BasicUtils
from src.utils.result_paths import CLOUD_IAC_RESULTS_DIR, LEGACY_CLOUD_IAC_RESULTS_DIR, prefer_existing_path


def load_pickle_with_numpy_fix(path):
    """Load pickle file with workaround for numpy version mismatch."""
    import sys
    
    # Preemptively set up aliases for numpy version compatibility
    try:
        import numpy.core
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = numpy.core
            try:
                sys.modules['numpy._core._multiarray_umath'] = numpy.core.multiarray
            except:
                pass
    except ImportError:
        pass
    
    # Try loading
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError) as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f, encoding='latin1')
            except Exception:
                raise
        raise


def load_scenario_data(scenario_name: str, data_root: Path) -> dict:
    """Load full scenario data including edges and observed orders."""
    scenario_path = resolve_cloud_iac_ground_truth_dir(data_root) / f"{scenario_name}.json"
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    
    edges = scenario.get('edges', [])
    tasks = sorted({t for edge in edges for t in edge})
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    
    true_dag = np.zeros((len(tasks), len(tasks)), dtype=np.int8)
    for parent, child in edges:
        if parent in task_to_idx and child in task_to_idx:
            true_dag[task_to_idx[parent], task_to_idx[child]] = 1
    
    true_cover = BasicUtils.transitive_reduction(true_dag)
    true_closure = BasicUtils.transitive_closure(true_cover)
    
    return {
        'true_cover': true_cover,
        'true_closure': true_closure,
        'tasks': tasks,
        'task_to_idx': task_to_idx,
        'scenario': scenario,
    }


def load_true_cover(scenario_name: str, data_root: Path) -> np.ndarray:
    """Load true cover for a scenario (wrapper for backward compatibility)."""
    data = load_scenario_data(scenario_name, data_root)
    return data['true_cover']


def is_linear_extension(order: list, closure: np.ndarray, task_to_idx: dict) -> bool:
    """
    Check if an observed order is a linear extension of the transitive closure.
    
    A trace is a linear extension if for all i < j in the trace,
    there is no edge j -> i in the closure (i.e., j does NOT precede i).
    """
    n = len(order)
    for i in range(n):
        for j in range(i + 1, n):
            # order[i] comes before order[j] in the trace
            # Check if order[j] -> order[i] in closure (would be a violation)
            task_i = order[i]
            task_j = order[j]
            if task_i in task_to_idx and task_j in task_to_idx:
                idx_i = task_to_idx[task_i]
                idx_j = task_to_idx[task_j]
                if closure[idx_j, idx_i] == 1:  # j precedes i in closure
                    return False  # Violation: trace has i before j but closure says j < i
    return True


def compute_feasibility(closure: np.ndarray, observed_orders: list, task_to_idx: dict) -> float:
    """
    Compute feasibility: fraction of observed traces that are linear extensions of TC(Ĝ).
    """
    if not observed_orders:
        return np.nan
    
    n_valid = sum(1 for order in observed_orders 
                  if is_linear_extension(order, closure, task_to_idx))
    return n_valid / len(observed_orders)


def compute_marginal_mode(H_trace) -> np.ndarray:
    """
    Compute marginal mode: for each edge, take mode across MCMC samples.
    
    Args:
        H_trace: shape (n_samples, n, n) - MCMC trace of H matrices (array or list)
    
    Returns:
        H_mode: shape (n, n) - marginal mode (most frequent value for each edge)
    """
    from scipy import stats
    
    # Convert to numpy array if it's a list
    if isinstance(H_trace, list):
        H_trace = np.array(H_trace)
    
    n_samples, n, _ = H_trace.shape
    H_mode = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Get mode (most frequent value) for this edge
                mode_result = stats.mode(H_trace[:, i, j], keepdims=False)
                H_mode[i, j] = int(mode_result.mode)
    
    return H_mode


def process_experiment(exp_dir: Path, scenario: str, scenario_data: dict,
                       threshold: float, mode: str, observed_orders: list = None) -> dict:
    """
    Process a single experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        scenario: Scenario name
        scenario_data: Dict with true_cover, true_closure, tasks, task_to_idx
        threshold: Threshold for single_po mode (ignored for marginal mode)
        mode: 'single_po' or 'marginal'
        observed_orders: List of observed orders for feasibility computation
    
    Returns:
        dict with metrics or None if processing failed
    """
    try:
        true_cover = scenario_data['true_cover']
        true_closure = scenario_data['true_closure']
        task_to_idx = scenario_data['task_to_idx']
        
        if mode == 'marginal':
            # Load H_trace for marginal mode
            h_trace_path = exp_dir / "H_trace.pkl"
            if not h_trace_path.exists():
                return None
            
            H_trace = load_pickle_with_numpy_fix(h_trace_path)
            H_final = compute_marginal_mode(H_trace)
            # H_final from marginal mode is already a binary matrix (mode of edges)
            # Ensure it's transitive closure for TR computation
            H_final_closure = BasicUtils.transitive_closure(H_final)
            
        else:  # single_po
            # Load avg_H for single PO mode
            avg_h_path = exp_dir / "avg_H.pkl"
            if not avg_h_path.exists():
                return None
            
            avg_H = load_pickle_with_numpy_fix(avg_h_path)
            H_final = (avg_H >= threshold).astype(np.int8)
            # Ensure H_final is transitive closure (needed for TR)
            H_final_closure = BasicUtils.transitive_closure(H_final)
        
        # Compute transitive reduction to get cover edges: TR(Ĝ)
        # Paper: Ê = TR(Ĝ) where Ĝ is the inferred graph (closure)
        H_final_cover = BasicUtils.transitive_reduction(H_final_closure)
        
        # ============================================================
        # METRICS COMPUTATION (following paper definitions)
        # ============================================================
        # Paper: Ê = TR(Ĝ), E* = TR(G*)
        # true_cover is already TR(G*) from load_scenario_data()
        # H_final_cover is TR(TC(Ĝ)) = TR(Ĝ)
        
        # Structural recovery: Precision/Recall/F1 for graph edges
        # Compare cover edges: TR(Ĝ) vs TR(G*)
        p_edge, r_edge = precision_recall(true_cover, H_final_cover)
        f1_edge = f1_score(p_edge, r_edge)
        # SHD on cover edges
        shd = structural_hamming_distance(true_cover, H_final_cover)
        
        # Concurrency recovery: Precision/Recall/F1 over incomparable pairs
        # Incomparable pairs are the unordered pairs under TC(·)
        pred_closure = H_final_closure  # TC(Ĝ) - already computed above
        true_ip_pairs = _incomparable_pairs(true_closure)
        pred_ip_pairs = _incomparable_pairs(pred_closure)
        
        # Precision/Recall for incomparable pairs
        tp_ip = len(true_ip_pairs & pred_ip_pairs)
        fp_ip = len(pred_ip_pairs - true_ip_pairs)
        fn_ip = len(true_ip_pairs - pred_ip_pairs)
        p_ip = float(tp_ip / (tp_ip + fp_ip)) if tp_ip + fp_ip > 0 else 0.0
        r_ip = float(tp_ip / (tp_ip + fn_ip)) if tp_ip + fn_ip > 0 else 0.0
        f1_ip = f1_score(p_ip, r_ip)
        
        # Feasibility: fraction of observed traces that are linear extensions
        if observed_orders:
            feas = compute_feasibility(H_final_closure, observed_orders, task_to_idx)
        else:
            feas = np.nan
        
        # Save final_H (save the cover edges, which is what we evaluate)
        output_name = f"final_H_{mode}_t{threshold:.2f}.pkl" if mode == 'single_po' else f"final_H_{mode}.pkl"
        with open(exp_dir / output_name, "wb") as f:
            pickle.dump(H_final_cover, f)  # Save cover edges TR(Ĝ)
        
        # Load summary.json to get parameters (if available)
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            config = summary.get('configuration', {})
        else:
            config = {}
        
        return {
            'scenario': scenario,
            'ip_cov_target': config.get('ip_cov_target', np.nan),
            'ip_cov_realized': config.get('ip_cov_realized', np.nan),
            'eps_jump': config.get('eps_jump', np.nan),
            'likelihood': config.get('likelihood', 'bhpop'),
            # Edge metrics (structural recovery)
            'cover_precision': p_edge,
            'cover_recall': r_edge,
            'cover_f1': f1_edge,
            'shd': shd,
            # Incomparable pair metrics (concurrency recovery)
            'ip_precision': p_ip,
            'ip_recall': r_ip,
            'ip_f1': f1_ip,
            # Execution diagnostics - computed from observed orders
            'feas': feas,
        }
            
    except Exception as e:
        print(f"  ERROR in {exp_dir.name}: {e}")
        return None


def compute_baseline_ip_f1(baseline_cover: np.ndarray, true_cover: np.ndarray) -> dict:
    """
    Compute IP F1 for a baseline method given its cover matrix.
    
    Args:
        baseline_cover: Cover matrix from baseline method (transitive reduction)
        true_cover: True cover matrix (transitive reduction)
    
    Returns:
        dict with ip_precision, ip_recall, ip_f1
    """
    # Compute transitive closures
    baseline_closure = BasicUtils.transitive_closure(baseline_cover)
    true_closure = BasicUtils.transitive_closure(true_cover)
    
    # Get incomparable pairs
    true_ip_pairs = _incomparable_pairs(true_closure)
    pred_ip_pairs = _incomparable_pairs(baseline_closure)
    
    # Compute precision/recall for incomparable pairs
    tp_ip = len(true_ip_pairs & pred_ip_pairs)
    fp_ip = len(pred_ip_pairs - true_ip_pairs)
    fn_ip = len(true_ip_pairs - pred_ip_pairs)
    
    p_ip = float(tp_ip / (tp_ip + fp_ip)) if tp_ip + fp_ip > 0 else 0.0
    r_ip = float(tp_ip / (tp_ip + fn_ip)) if tp_ip + fn_ip > 0 else 0.0
    f1_ip = f1_score(p_ip, r_ip)
    
    return {
        'ip_precision': p_ip,
        'ip_recall': r_ip,
        'ip_f1': f1_ip,
    }


def main():
    parser = argparse.ArgumentParser(description='Recompute H with different thresholds/methods')
    parser.add_argument('--threshold', type=float, default=0.4, 
                        help='Threshold for single_po mode (default: 0.4)')
    parser.add_argument('--mode', choices=['single_po', 'marginal'], default='single_po',
                        help='Method: single_po (threshold on avg_H) or marginal (mode of H_trace)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: auto-generated)')
    parser.add_argument('--compute-baseline-ip', action='store_true',
                        help='Compute IP F1 for baseline methods (requires data access)')
    
    args = parser.parse_args()
    
    results_dir = prefer_existing_path(CLOUD_IAC_RESULTS_DIR, LEGACY_CLOUD_IAC_RESULTS_DIR)
    data_root = resolve_cloud_iac_data_root(PROJECT_ROOT)
    
    print("="*70)
    print(f"RECOMPUTING H WITH: mode={args.mode}, threshold={args.threshold}")
    print("="*70)
    print()
    
    # Load baseline methods from existing CSV
    # Try multiple possible sources for baseline data
    possible_csvs = [
        results_dir / "experiment_summary_t0.30.csv",
        results_dir / "experiment_summary_t0.40.csv",
        results_dir / "experiment_summary_t0.50.csv",
        results_dir / "experiment_summary.csv",
    ]
    
    csv_path = None
    for p in possible_csvs:
        if p.exists():
            csv_path = p
            break
    
    if csv_path is None:
        print("ERROR: No experiment_summary CSV found!")
        print("Run generate_experiment_summaries.py first.")
        return
    
    df_original = pd.read_csv(csv_path)
    df_baselines = df_original[df_original['method'] != 'bhpop_single_po'].copy()
    
    print(f"Loaded {len(df_baselines)} baseline rows from {csv_path.name}")
    
    # Compute IP F1 for baseline methods if requested
    if args.compute_baseline_ip:
        print("\nComputing IP F1 for baseline methods...")
        # This would require loading original data and recomputing baseline cover matrices
        # For now, we'll add placeholder NaN values
        # TODO: Implement full baseline IP F1 computation
        print("  (Baseline IP F1 computation not yet implemented - requires data access)")
        df_baselines['ip_precision'] = np.nan
        df_baselines['ip_recall'] = np.nan
        df_baselines['ip_f1'] = np.nan
    else:
        # Add empty IP F1 columns
        df_baselines['ip_precision'] = np.nan
        df_baselines['ip_recall'] = np.nan
        df_baselines['ip_f1'] = np.nan
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('exp_')])
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print()
    
    # Group by scenario
    exp_by_scenario = defaultdict(list)
    for exp_dir in exp_dirs:
        parts = exp_dir.name.split('_')
        if len(parts) >= 3:
            scenario = '_'.join(parts[2:])
            exp_by_scenario[scenario].append(exp_dir)
    
    # Process experiments
    bhpop_rows = []
    
    for scenario, scenario_dirs in sorted(exp_by_scenario.items()):
        print(f"{scenario}: {len(scenario_dirs)} experiments")
        
        # Load scenario data (true cover, tasks, etc.)
        try:
            scenario_data = load_scenario_data(scenario, data_root)
        except Exception as e:
            print(f"  ERROR: Could not load scenario data: {e}")
            continue
        
        # Mapping from scenario name to trace prefix
        scenario_to_trace_prefix = {
            'simple_ecs': 'T01',
            'slb_ecs_rds': 'T02',
            'slb_ecs_redis': 'T03',
            'eip_slb_ecs': 'T04',
            'dual_zone_ecs_slb': 'T05',
            'dual_zone_ecs_slb_rds': 'T06',
        }
        
        # Load observed orders from traces directory
        traces_dir = data_root / "traces"
        observed_orders = []
        
        trace_prefix = scenario_to_trace_prefix.get(scenario, '')
        if trace_prefix and traces_dir.exists():
            for trace_file in traces_dir.glob(f"trace_{trace_prefix}_*.json"):
                try:
                    with open(trace_file, 'r') as f:
                        trace_data = json.load(f)
                    # Extract action sequence from trace
                    if 'actions' in trace_data:
                        order = [a.get('action_name') for a in trace_data['actions'] if a.get('action_name')]
                        if order:
                            observed_orders.append(order)
                except Exception:
                    pass
        
        if observed_orders:
            print(f"  Loaded {len(observed_orders)} observed orders for feasibility")
        else:
            print(f"  WARNING: No observed orders found for feasibility computation")
        
        # Process each experiment
        for exp_dir in sorted(scenario_dirs):
            result = process_experiment(exp_dir, scenario, scenario_data,
                                       args.threshold, args.mode, observed_orders)
            if result:
                bhpop_rows.append(result)
        
        print(f"  Processed {len([d for d in scenario_dirs if (d / 'summary.json').exists()])} experiments")
    
    print()
    print(f"Total BHPOP results: {len(bhpop_rows)}")
    
    # Create DataFrame
    df_bhpop = pd.DataFrame(bhpop_rows)
    df_bhpop['method'] = 'bhpop_single_po'  # Always use this name for consistency
    
    # DO NOT expand BHPOP across ip_cov_target values!
    # Each experiment has its own ip_cov_target from its configuration.
    # Expanding would incorrectly duplicate results and cause flat lines in plots.
    print(f"\nBHPOP results by ip_cov_target:")
    if len(df_bhpop) > 0:
        ipcov_counts = df_bhpop.groupby('ip_cov_target').size()
        for ipcov, count in ipcov_counts.items():
            print(f"  ip_cov_target={ipcov}: {count} experiments")
    
    # Combine with baselines
    df_combined = pd.concat([df_baselines, df_bhpop], ignore_index=True)
    
    # Generate output filename
    if args.output:
        output_path = results_dir / args.output
    else:
        if args.mode == 'single_po':
            output_path = results_dir / f"experiment_summary_t{args.threshold:.2f}.csv"
        else:
            output_path = results_dir / f"experiment_summary_{args.mode}.csv"
    
    # Save
    df_combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Baseline rows: {len(df_baselines)}")
    print(f"  BHPOP rows: {len(df_bhpop)}")
    
    # Summary statistics
    print()
    print("="*70)
    print("SUMMARY BY SCENARIO")
    print("="*70)
    
    if len(df_bhpop) > 0:
        print(f"{'Scenario':25s} {'Count':>6s} {'Avg F1':>9s} {'Avg SHD':>9s}")
        print("-"*70)
        
        for scenario in sorted(df_bhpop['scenario'].unique()):
            scenario_data = df_bhpop[df_bhpop['scenario'] == scenario]
            avg_f1 = scenario_data['cover_f1'].mean()
            avg_shd = scenario_data['shd'].mean()
            count = len(scenario_data)
            print(f"{scenario:25s} {count:6d} {avg_f1:9.3f} {avg_shd:9.1f}")
    else:
        print("No BHPOP results generated (check if required files exist)")
    
    print()
    print("="*70)
    print(f"NEXT: Plot with plot_experiment_results.py using {output_path.name}")
    print("="*70)


if __name__ == '__main__':
    main()
