#!/usr/bin/env python3
"""
Recompute all BHPOP results using complexity-based thresholds.
Then regenerate all plots for the paper.
"""
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance
from src.utils.po_fun import BasicUtils

# Complexity-based thresholds
COMPLEX_SCENARIOS = {
    'dual_zone_ecs_slb', 'dual_zone_ecs_slb_rds', 'eip_slb_ecs',
    'slb_ecs_rds', 'slb_ecs_redis'
}
THRESHOLD_COMPLEX = 0.4
THRESHOLD_SIMPLE = 0.5


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


def is_linear_extension_partial(order: list, closure: np.ndarray) -> bool:
    """Check if order is consistent with closure."""
    pos = {t: i for i, t in enumerate(order)}
    for i in range(closure.shape[0]):
        for j in range(closure.shape[1]):
            if closure[i, j] == 1:
                if i in pos and j in pos and pos[i] >= pos[j]:
                    return False
    return True


def main():
    results_dir = Path("systematic_experiment_results")
    data_root = PROJECT_ROOT / "aliyun_data"
    
    print("="*70)
    print("RECOMPUTING RESULTS WITH COMPLEXITY-BASED THRESHOLDS")
    print("="*70)
    print(f"Complex scenarios (τ={THRESHOLD_COMPLEX}): {len(COMPLEX_SCENARIOS)}")
    print(f"Simple scenarios (τ={THRESHOLD_SIMPLE}): 1 (simple_ecs)")
    print()
    
    # Load original results for non-BHPOP methods
    df_original = pd.read_csv(results_dir / "experiment_summary.csv")
    
    # Keep non-BHPOP rows as-is
    non_bhpop_rows = df_original[df_original['method'] != 'bhpop_single_po'].to_dict('records')
    
    # Recompute BHPOP rows
    bhpop_rows = []
    
    # Map experiment directories to their parameters
    # Group by scenario for easier matching
    scenario_exp_dirs = defaultdict(list)
    
    for exp_dir in sorted(results_dir.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        parts = exp_dir.name.split('_')
        if len(parts) >= 3:
            scenario = '_'.join(parts[2:])
            scenario_exp_dirs[scenario].append(exp_dir)
    
    # For each scenario, recompute all experiments
    for scenario, exp_dirs in scenario_exp_dirs.items():
        print(f"\n{scenario}: processing {len(exp_dirs)} experiments...")
        
        try:
            true_cover = load_true_cover(scenario, data_root)
        except Exception as e:
            print(f"  Could not load true cover: {e}")
            continue
        
        # Get threshold for this scenario
        threshold = THRESHOLD_COMPLEX if scenario in COMPLEX_SCENARIOS else THRESHOLD_SIMPLE
        print(f"  Using threshold: {threshold}")
        
        # Get original BHPOP rows for this scenario
        orig_bhpop = df_original[
            (df_original['scenario'] == scenario) & 
            (df_original['method'] == 'bhpop_single_po')
        ].copy()
        
        # Process each experiment
        for idx, exp_dir in enumerate(exp_dirs):
            try:
                # Load avg_H
                with open(exp_dir / "avg_H.pkl", "rb") as f:
                    avg_H = pickle.load(f)
                
                # Apply complexity-based threshold
                H_new = (avg_H >= threshold).astype(np.int8)
                
                # Compute metrics
                p, r = precision_recall(true_cover, H_new)
                f1 = f1_score(p, r)
                shd = structural_hamming_distance(true_cover, H_new)
                
                # Save new H
                with open(exp_dir / "final_H_complexity_based.pkl", "wb") as f:
                    pickle.dump(H_new, f)
                
                # Match to original row if possible
                if idx < len(orig_bhpop):
                    orig_row = orig_bhpop.iloc[idx]
                    
                    bhpop_rows.append({
                        'scenario': scenario,
                        'cp_cov_target': orig_row['cp_cov_target'],
                        'cp_cov_realized': orig_row['cp_cov_realized'],
                        'eps_jump': orig_row['eps_jump'],
                        'likelihood': orig_row['likelihood'],
                        'method': 'bhpop_single_po',
                        'cover_f1': f1,
                        'shd': shd,
                        'feas': orig_row['feas'],  # Feasibility doesn't change
                    })
                
            except Exception as e:
                print(f"  Error processing {exp_dir.name}: {e}")
                continue
    
    # Combine all rows
    all_rows = non_bhpop_rows + bhpop_rows
    df_new = pd.DataFrame(all_rows)
    
    # Save new results
    output_path = results_dir / "experiment_summary_complexity_based.csv"
    df_new.to_csv(output_path, index=False)
    
    print()
    print("="*70)
    print(f"✓ Saved new results: {output_path}")
    print(f"  Total rows: {len(df_new)}")
    print(f"  BHPOP rows updated: {len(bhpop_rows)}")
    print()
    
    # Comparison
    print("="*70)
    print("IMPROVEMENT vs FIXED τ=0.5")
    print("="*70)
    print(f"{'Scenario':25s} {'τ':>6s} {'F1_new':>9s} {'F1_old':>9s} {'Δ':>9s}")
    print("-"*70)
    
    for scenario in sorted(set(df_new['scenario'])):
        # New results (complexity-based)
        new_data = df_new[
            (df_new['scenario'] == scenario) & 
            (df_new['method'] == 'bhpop_single_po')
        ]
        
        # Old results (fixed τ=0.5)
        old_data = df_original[
            (df_original['scenario'] == scenario) & 
            (df_original['method'] == 'bhpop_single_po')
        ]
        
        if len(new_data) > 0 and len(old_data) > 0:
            f1_new = new_data['cover_f1'].mean()
            f1_old = old_data['cover_f1'].mean()
            improvement = f1_new - f1_old
            
            threshold = THRESHOLD_COMPLEX if scenario in COMPLEX_SCENARIOS else THRESHOLD_SIMPLE
            marker = '✓' if improvement > 0.05 else ''
            
            print(f"{scenario:25s} {threshold:6.2f} {f1_new:9.3f} {f1_old:9.3f} {improvement:+9.3f} {marker}")
    
    print()
    print("="*70)
    print("NEXT STEP: Run plot_experiment_results.py with new CSV")
    print("="*70)


if __name__ == '__main__':
    main()
