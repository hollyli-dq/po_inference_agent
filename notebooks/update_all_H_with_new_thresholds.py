#!/usr/bin/env python3
"""
Recompute ALL final_H graphs using new thresholds WITHOUT re-running MCMC.
- eip_slb_ecs: τ=0.4
- All others: τ=0.4
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

# Thresholds
THRESHOLD_EIP = 0.4
THRESHOLD_DEFAULT = 0.4

# Workaround for numpy version mismatch in pickle files
def load_pickle_with_numpy_fix(path):
    """Load pickle file with workaround for numpy._core vs numpy.core issue."""
    import sys
    
    # Preemptively set up aliases for numpy version compatibility
    try:
        import numpy.core
        # If numpy.core exists but pickle expects numpy._core, create alias
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = numpy.core
            # Also handle submodules that might be referenced
            try:
                sys.modules['numpy._core._multiarray_umath'] = numpy.core.multiarray
            except:
                pass
    except ImportError:
        pass
    
    # Try loading with different strategies
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError) as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            # Try with latin1 encoding which is more forgiving
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


def main():
    results_dir = Path("systematic_experiment_results")
    data_root = PROJECT_ROOT / "aliyun_data"
    
    print("="*70)
    print("RECOMPUTING ALL H WITH NEW THRESHOLDS")
    print("="*70)
    print(f"eip_slb_ecs: τ={THRESHOLD_EIP}")
    print(f"All others:  τ={THRESHOLD_DEFAULT}")
    print()
    
    # Load original CSV
    df_original = pd.read_csv(results_dir / "experiment_summary.csv")
    
    # Keep non-BHPOP rows unchanged
    df_non_bhpop = df_original[df_original['method'] != 'bhpop_single_po'].copy()
    
    # Process all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('exp_')])
    
    print(f"Processing {len(exp_dirs)} experiments...")
    
    # Group by scenario
    exp_by_scenario = defaultdict(list)
    for exp_dir in exp_dirs:
        parts = exp_dir.name.split('_')
        if len(parts) >= 3:
            scenario = '_'.join(parts[2:])
            exp_by_scenario[scenario].append(exp_dir)
    
    # Track updated results
    bhpop_rows_new = []
    
    for scenario, scenario_dirs in sorted(exp_by_scenario.items()):
        print(f"\n{scenario}: {len(scenario_dirs)} experiments")
        
        # Determine threshold
        threshold = THRESHOLD_EIP if scenario == 'eip_slb_ecs' else THRESHOLD_DEFAULT
        print(f"  Using threshold: {threshold}")
        
        # Load true cover
        try:
            true_cover = load_true_cover(scenario, data_root)
        except Exception as e:
            print(f"  ERROR: Could not load true cover: {e}")
            continue
        
        # Get original BHPOP rows for this scenario
        orig_bhpop = df_original[
            (df_original['scenario'] == scenario) & 
            (df_original['method'] == 'bhpop_single_po')
        ].copy().reset_index(drop=True)
        
        print(f"  Original BHPOP rows: {len(orig_bhpop)}")
        
        # Process each experiment
        for exp_idx, exp_dir in enumerate(sorted(scenario_dirs)):
            try:
                # Load avg_H (posterior mean)
                avg_H_path = exp_dir / "avg_H.pkl"
                if not avg_H_path.exists():
                    continue
                
                avg_H = load_pickle_with_numpy_fix(avg_H_path)
                
                # Apply threshold
                H_new = (avg_H >= threshold).astype(np.int8)
                
                # Compute metrics
                p, r = precision_recall(true_cover, H_new)
                f1 = f1_score(p, r)
                shd = structural_hamming_distance(true_cover, H_new)
                
                # Save new final_H
                with open(exp_dir / "final_H.pkl", "wb") as f:
                    pickle.dump(H_new, f)
                
                # Match to original row (by index within scenario)
                if exp_idx < len(orig_bhpop):
                    orig_row = orig_bhpop.iloc[exp_idx]
                    
                    bhpop_rows_new.append({
                        'scenario': scenario,
                        'ip_cov_target': orig_row['ip_cov_target'],
                        'ip_cov_realized': orig_row['ip_cov_realized'],
                        'eps_jump': orig_row['eps_jump'],
                        'likelihood': orig_row['likelihood'],
                        'method': 'bhpop_single_po',
                        'cover_f1': f1,
                        'shd': shd,
                        'feas': orig_row['feas'],
                    })
                
            except Exception as e:
                print(f"  ERROR in {exp_dir.name}: {e}")
                continue
        
        print(f"  Updated {len([d for d in scenario_dirs if (d / 'avg_H.pkl').exists()])} experiments")
    
    # Create new DataFrame
    df_bhpop_new = pd.DataFrame(bhpop_rows_new)
    df_combined = pd.concat([df_non_bhpop, df_bhpop_new], ignore_index=True)
    
    # Save
    output_path = results_dir / "experiment_summary.csv"
    backup_path = results_dir / "experiment_summary_backup.csv"
    
    # Backup original
    df_original.to_csv(backup_path, index=False)
    print(f"\n✓ Backed up original: {backup_path}")
    
    # Save new
    df_combined.to_csv(output_path, index=False)
    print(f"✓ Updated CSV: {output_path}")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  BHPOP rows: {len(df_bhpop_new)}")
    
    # Comparison
    print()
    print("="*70)
    print("CHANGE IN F1 SCORES")
    print("="*70)
    print(f"{'Scenario':25s} {'τ':>6s} {'F1_new':>9s} {'F1_old':>9s} {'Δ':>9s}")
    print("-"*70)
    
    # Only do comparison if we have BHPOP data
    if len(df_bhpop_new) > 0:
        for scenario in sorted(df_combined['scenario'].unique()):
            new_data = df_bhpop_new[df_bhpop_new['scenario'] == scenario]
            old_data = df_original[
                (df_original['scenario'] == scenario) & 
                (df_original['method'] == 'bhpop_single_po')
            ]
            
            if len(new_data) > 0 and len(old_data) > 0:
                f1_new = new_data['cover_f1'].mean()
                f1_old = old_data['cover_f1'].mean()
                improvement = f1_new - f1_old
                
                threshold = THRESHOLD_EIP if scenario == 'eip_slb_ecs' else THRESHOLD_DEFAULT
                marker = '✓✓' if improvement > 0.1 else ('✓' if abs(improvement) > 0.01 else '≈')
                
                print(f"{scenario:25s} {threshold:6.2f} {f1_new:9.3f} {f1_old:9.3f} {improvement:+9.3f} {marker}")
    else:
        print("No BHPOP rows were processed - cannot compare.")
    
    print()
    print("="*70)
    print("NEXT: Regenerate plots with plot_experiment_results.py")
    print("="*70)


if __name__ == '__main__':
    main()
