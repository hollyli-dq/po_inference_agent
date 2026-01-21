#!/usr/bin/env python3
"""
Recompute final_H graphs with different thresholds and methods.
Pure Python approach - reads avg_H.pkl files and recomputes metrics.

Usage:
    python recompute_H_with_threshold.py --threshold 0.4 --mode single_po
    python recompute_H_with_threshold.py --threshold 0.5 --mode marginal
"""
import sys
import argparse
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance, _incomparable_pairs
from src.utils.po_fun import BasicUtils
from src.utils.po_fun import BasicUtils


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


def load_true_cover(scenario_name: str, data_root: Path) -> np.ndarray:
    """Load true cover for a scenario."""
    scenario_path = data_root / "manual_scenarios" / f"{scenario_name}.json"
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    
    edges = scenario.get('edges', [])
    tasks = sorted({t for edge in edges for t in edge})
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    
    true_dag = np.zeros((len(tasks), len(tasks)), dtype=np.int8)
    for parent, child in edges:
        if parent in task_to_idx and child in task_to_idx:
            true_dag[task_to_idx[parent], task_to_idx[child]] = 1
    
    return BasicUtils.transitive_reduction(true_dag)


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


def process_experiment(exp_dir: Path, scenario: str, true_cover: np.ndarray, 
                       threshold: float, mode: str) -> dict:
    """
    Process a single experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        scenario: Scenario name
        true_cover: True cover matrix
        threshold: Threshold for single_po mode (ignored for marginal mode)
        mode: 'single_po' or 'marginal'
    
    Returns:
        dict with metrics or None if processing failed
    """
    try:
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
        # true_cover is already TR(G*) from load_true_cover()
        # H_final_cover is TR(TC(Ĝ)) = TR(Ĝ)
        
        # Structural recovery: Precision/Recall/F1 for graph edges
        # Compare cover edges: TR(Ĝ) vs TR(G*)
        p_edge, r_edge = precision_recall(true_cover, H_final_cover)
        f1_edge = f1_score(p_edge, r_edge)
        # SHD on cover edges
        shd = structural_hamming_distance(true_cover, H_final_cover)
        
        # Concurrency recovery: Precision/Recall/F1 over incomparable pairs
        # Incomparable pairs are the unordered pairs under TC(·)
        true_closure = BasicUtils.transitive_closure(true_cover)  # TC(TR(G*)) = TC(G*)
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
            results = summary.get('results', {})
        else:
            # No summary.json: use placeholder values
            # These experiments were run without saving config
            config = {}
            results = {}
        
        return {
            'scenario': scenario,
            'ip_cov_target': config.get('ip_cov_target', np.nan),
            'ip_cov_realized': config.get('ip_cov_realized', np.nan),
            'eps_jump': config.get('eps_jump', np.nan),
            'likelihood': config.get('likelihood', 'bhpop'),  # Default for BHPOP
            # Edge metrics (structural recovery)
            'cover_precision': p_edge,
            'cover_recall': r_edge,
            'cover_f1': f1_edge,
            'shd': shd,
            # Incomparable pair metrics (concurrency recovery)
            'ip_precision': p_ip,
            'ip_recall': r_ip,
            'ip_f1': f1_ip,
            # Execution diagnostics
            'feas': results.get('feasibility', np.nan),
        }
            
    except Exception as e:
        print(f"  ERROR in {exp_dir.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Recompute H with different thresholds/methods')
    parser.add_argument('--threshold', type=float, default=0.4, 
                        help='Threshold for single_po mode (default: 0.4)')
    parser.add_argument('--mode', choices=['single_po', 'marginal'], default='single_po',
                        help='Method: single_po (threshold on avg_H) or marginal (mode of H_trace)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    results_dir = Path("systematic_experiment_results")
    data_root = PROJECT_ROOT / "aliyun_data"
    
    print("="*70)
    print(f"RECOMPUTING H WITH: mode={args.mode}, threshold={args.threshold}")
    print("="*70)
    print()
    
    # Load baseline methods from original CSV
    csv_path = results_dir / "experiment_summary_with_marginal_mode.csv"
    if not csv_path.exists():
        csv_path = results_dir / "experiment_summary.csv"
    
    df_original = pd.read_csv(csv_path)
    df_baselines = df_original[df_original['method'] != 'bhpop_single_po'].copy()
    
    print(f"Loaded {len(df_baselines)} baseline rows from {csv_path.name}")
    
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
        
        # Load true cover
        try:
            true_cover = load_true_cover(scenario, data_root)
        except Exception as e:
            print(f"  ERROR: Could not load true cover: {e}")
            continue
        
        # Process each experiment
        for exp_dir in sorted(scenario_dirs):
            result = process_experiment(exp_dir, scenario, true_cover, 
                                       args.threshold, args.mode)
            if result:
                bhpop_rows.append(result)
        
        print(f"  Processed {len([d for d in scenario_dirs if (d / 'summary.json').exists()])} experiments")
    
    print()
    print(f"Total BHPOP results: {len(bhpop_rows)}")
    
    # Create DataFrame
    df_bhpop = pd.DataFrame(bhpop_rows)
    df_bhpop['method'] = f'bhpop_{args.mode}'
    
    # Expand BHPOP across all ip_cov_target values that exist in baselines
    # (baseline methods are duplicated across ip_cov values even though they don't depend on it)
    unique_ip_covs = sorted(df_baselines['ip_cov_target'].dropna().unique())
    
    if len(df_bhpop) > 0 and len(unique_ip_covs) > 1:
        print(f"\nExpanding BHPOP across ip_cov_target values: {unique_ip_covs}")
        bhpop_expanded = []
        for _, row in df_bhpop.iterrows():
            for ip_cov in unique_ip_covs:
                new_row = row.copy()
                new_row['ip_cov_target'] = ip_cov
                bhpop_expanded.append(new_row)
        df_bhpop = pd.DataFrame(bhpop_expanded)
        print(f"Expanded from {len(bhpop_rows)} to {len(df_bhpop)} rows")
    
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
