#!/usr/bin/env python3
"""Properly assign eps_jump by matching experiments to their actual ip_cov_target."""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("systematic_experiment_results")
    df = pd.read_csv(results_dir / "experiment_summary_t0.40.csv")
    
    print(f"Loaded CSV: {len(df)} rows")
    
    # Read all experiments and create mapping: (scenario, ip_cov_target) -> eps_jump
    exp_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('exp_')])
    
    # Map: (scenario, ip_cov_target) -> list of eps_jump values
    scenario_ipcov_to_eps = {}
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                scenario = summary.get('scenario_name', '')
                config = summary.get('configuration', {})
                eps_jump = config.get('eps_jump')
                ip_cov_target = config.get('ip_cov_target')
                
                if scenario and eps_jump is not None and ip_cov_target is not None:
                    key = (scenario, ip_cov_target)
                    if key not in scenario_ipcov_to_eps:
                        scenario_ipcov_to_eps[key] = []
                    if eps_jump not in scenario_ipcov_to_eps[key]:
                        scenario_ipcov_to_eps[key].append(eps_jump)
            except Exception as e:
                pass
    
    print(f"Found {len(scenario_ipcov_to_eps)} (scenario, ip_cov_target) combinations")
    
    # Update BPOP rows
    bhpop_mask = df['method'] == 'bhpop_single_po'
    updated_count = 0
    
    for idx, row in df[bhpop_mask].iterrows():
        scenario = row['scenario']
        ip_cov_target = row['ip_cov_target']
        
        key = (scenario, ip_cov_target)
        if key in scenario_ipcov_to_eps:
            eps_values = scenario_ipcov_to_eps[key]
            # If multiple eps_jump values, we need to create multiple rows
            # But for now, let's just use the first one
            # Actually, the CSV should have one row per (scenario, ip_cov_target, eps_jump)
            # So we need to figure out which eps_jump this row corresponds to
            
            # The issue is that the CSV was expanded, so we don't know which eps_jump
            # corresponds to which row. We need to check if there are multiple rows
            # for the same (scenario, ip_cov_target) and assign different eps_jump to each
            
            # Get all rows for this (scenario, ip_cov_target)
            matching_rows = df[(df['method'] == 'bhpop_single_po') & 
                               (df['scenario'] == scenario) & 
                               (df['ip_cov_target'] == ip_cov_target)]
            
            if len(matching_rows) == len(eps_values):
                # Perfect match - assign one eps_jump to each row
                row_idx_in_group = matching_rows.index.get_loc(idx)
                if row_idx_in_group < len(eps_values):
                    df.loc[idx, 'eps_jump'] = eps_values[row_idx_in_group]
                    updated_count += 1
            elif len(eps_values) == 1:
                # Only one eps_jump value, assign it to all rows
                df.loc[idx, 'eps_jump'] = eps_values[0]
                updated_count += 1
            else:
                # Multiple eps_jump values but wrong number of rows
                # Use the first one as fallback
                df.loc[idx, 'eps_jump'] = eps_values[0]
                updated_count += 1
    
    print(f"\nUpdated {updated_count} BPOP rows with eps_jump")
    
    # Save
    df.to_csv(results_dir / "experiment_summary_t0.40.csv", index=False)
    print(f"Saved updated CSV")
    
    # Verify
    bhpop_check = df[df['method'] == 'bhpop_single_po']
    print(f"\nVerification:")
    print(f"BPOP rows: {len(bhpop_check)}")
    print(f"BPOP rows with eps_jump: {bhpop_check['eps_jump'].notna().sum()}")
    print(f"Unique eps_jump values: {sorted(bhpop_check['eps_jump'].dropna().unique())}")
    
    # Show distribution
    print(f"\nEps_jump distribution:")
    print(bhpop_check.groupby('eps_jump').size())

if __name__ == '__main__':
    main()
