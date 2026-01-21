#!/usr/bin/env python3
"""Fix missing eps_jump values in recomputed CSV by merging from original CSV."""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Load CSVs
    results_dir = Path("systematic_experiment_results")
    df_orig = pd.read_csv(results_dir / "experiment_summary.csv")
    df_new = pd.read_csv(results_dir / "experiment_summary_t0.40.csv")
    
    print(f"Original CSV: {len(df_orig)} rows")
    print(f"New CSV: {len(df_new)} rows")
    
    # Get BPOP rows from original with eps_jump
    bhpop_orig = df_orig[df_orig['method'] == 'bhpop_single_po'].copy()
    print(f"\nOriginal BPOP rows: {len(bhpop_orig)}")
    
    # Get BPOP rows from new (missing eps_jump)
    bhpop_new = df_new[df_new['method'] == 'bhpop_single_po'].copy()
    print(f"New BPOP rows: {len(bhpop_new)}")
    print(f"New BPOP rows with null eps_jump: {bhpop_new['eps_jump'].isna().sum()}")
    
    # Create a lookup: (scenario, likelihood, ip_cov_target) -> eps_jump
    # But the new CSV might have been expanded, so we need to match differently
    # Actually, let's match by scenario and likelihood, then assign eps_jump values
    # The original has one row per (scenario, eps_jump, likelihood, ip_cov_target)
    # The new has been expanded across ip_cov_target, so we need to match scenario+likelihood
    
    # Get unique combinations from original
    orig_lookup = bhpop_orig.groupby(['scenario', 'likelihood', 'eps_jump']).size().reset_index()
    print(f"\nOriginal unique (scenario, likelihood, eps_jump) combinations: {len(orig_lookup)}")
    
    # For each row in new CSV, try to find matching eps_jump
    # Strategy: match by scenario and likelihood, then assign all possible eps_jump values
    # But wait - the new CSV has been expanded, so each experiment might appear multiple times
    
    # Better approach: read eps_jump from experiment directories directly
    # Or: merge based on scenario + likelihood, and create all combinations
    
    # Actually, let's check if we can infer eps_jump from the experiment directory structure
    # Or read from summary.json files
    
    # For now, let's try a simpler approach: 
    # If the new CSV has the same number of unique experiments as original,
    # we can match them
    
    # Get unique experiments from original (scenario + likelihood + eps_jump)
    orig_experiments = bhpop_orig[['scenario', 'likelihood', 'eps_jump']].drop_duplicates()
    print(f"\nOriginal unique experiments: {len(orig_experiments)}")
    
    # Get unique experiments from new (scenario + likelihood, but missing eps_jump)
    new_experiments = bhpop_new[['scenario', 'likelihood']].drop_duplicates()
    print(f"New unique (scenario, likelihood) combinations: {len(new_experiments)}")
    
    # Match and merge
    # For each (scenario, likelihood) in new, find all possible eps_jump from original
    merged_rows = []
    for _, new_row in bhpop_new.iterrows():
        scenario = new_row['scenario']
        likelihood = new_row['likelihood']
        
        # Find matching eps_jump values from original
        matching = orig_experiments[
            (orig_experiments['scenario'] == scenario) & 
            (orig_experiments['likelihood'] == likelihood)
        ]
        
        if len(matching) > 0:
            # If there are multiple eps_jump values, we need to create multiple rows
            # But the new CSV might already have the right structure
            # Let's just take the first matching eps_jump for now
            # Actually, we should check if the new CSV structure allows us to assign eps_jump properly
            
            # For now, let's assign the most common eps_jump for this scenario+likelihood
            eps_values = matching['eps_jump'].unique()
            if len(eps_values) == 1:
                new_row['eps_jump'] = eps_values[0]
            else:
                # Multiple eps values - this is tricky
                # The new CSV might have been expanded, so we need to figure out which eps_jump
                # corresponds to which row
                # For now, let's use the first one as a placeholder
                new_row['eps_jump'] = eps_values[0]
        else:
            # No match found - keep as NaN
            pass
        
        merged_rows.append(new_row)
    
    # Actually, this approach is too complex. Let's try reading from experiment directories
    print("\nReading eps_jump from experiment directories...")
    
    import json
    exp_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('exp_')])
    
    # Create mapping: experiment_id -> eps_jump
    exp_to_eps = {}
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                exp_id = summary.get('experiment_id')
                eps_jump = summary.get('configuration', {}).get('eps_jump')
                if exp_id and eps_jump is not None:
                    exp_to_eps[exp_id] = eps_jump
            except:
                pass
    
    print(f"Found eps_jump for {len(exp_to_eps)} experiments")
    
    # This is still complex because we need to map experiments to CSV rows
    # Let's try a different approach: update the recompute script to properly read eps_jump
    
    print("\nBetter solution: Re-run recompute_H_with_threshold.py with fixed eps_jump reading")
    print("Or: Manually merge eps_jump from original CSV")
    
    # For a quick fix, let's create a mapping and update the new CSV
    # Match by scenario + likelihood, and assign eps_jump values
    
    # Create expanded mapping: for each (scenario, likelihood) in new CSV,
    # assign all possible eps_jump values from original
    scenario_likelihood_to_eps = {}
    for _, row in orig_experiments.iterrows():
        key = (row['scenario'], row['likelihood'])
        if key not in scenario_likelihood_to_eps:
            scenario_likelihood_to_eps[key] = []
        scenario_likelihood_to_eps[key].append(row['eps_jump'])
    
    # Now update new CSV: for each BPOP row, assign eps_jump based on scenario+likelihood
    # But we need to be careful - if there are multiple eps_jump values, we might need
    # to create multiple rows or use some other logic
    
    # Actually, the simplest fix: just merge the original BPOP data with eps_jump
    # into the new CSV, matching on scenario, likelihood, and ip_cov_target
    
    print("\nMerging eps_jump from original CSV...")
    
    # Merge on scenario, likelihood, ip_cov_target
    merge_cols = ['scenario', 'likelihood', 'ip_cov_target']
    df_merged = df_new.merge(
        bhpop_orig[merge_cols + ['eps_jump']],
        on=merge_cols,
        how='left',
        suffixes=('', '_orig')
    )
    
    # Update eps_jump in merged dataframe
    df_merged.loc[df_merged['method'] == 'bhpop_single_po', 'eps_jump'] = \
        df_merged.loc[df_merged['method'] == 'bhpop_single_po', 'eps_jump_orig']
    
    # Drop the temporary column
    df_merged = df_merged.drop(columns=['eps_jump_orig'], errors='ignore')
    
    # Save
    output_path = results_dir / "experiment_summary_t0.40_fixed.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"Saved fixed CSV: {output_path}")
    print(f"Fixed BPOP rows with eps_jump: {df_merged[df_merged['method'] == 'bhpop_single_po']['eps_jump'].notna().sum()}")

if __name__ == '__main__':
    main()
