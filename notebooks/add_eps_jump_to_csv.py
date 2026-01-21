#!/usr/bin/env python3
"""Add missing eps_jump values to recomputed CSV by reading from experiment directories."""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("systematic_experiment_results")
    df = pd.read_csv(results_dir / "experiment_summary_t0.40.csv")
    
    print(f"Loaded CSV: {len(df)} rows")
    print(f"BPOP rows: {len(df[df['method'] == 'bhpop_single_po'])}")
    print(f"BPOP rows with null eps_jump: {df[df['method'] == 'bhpop_single_po']['eps_jump'].isna().sum()}")
    
    # Read all experiment directories
    exp_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('exp_')])
    
    print(f"\nFound {len(exp_dirs)} experiment directories")
    
    # Create mapping: scenario -> list of (eps_jump, ip_cov_target, likelihood)
    # Match by scenario only since likelihood might differ
    exp_info = {}
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
                likelihood = config.get('likelihood', 'bhpop')
                
                if scenario and eps_jump is not None:
                    if scenario not in exp_info:
                        exp_info[scenario] = []
                    exp_info[scenario].append({
                        'eps_jump': eps_jump,
                        'ip_cov_target': ip_cov_target,
                        'likelihood': likelihood,
                    })
            except Exception as e:
                print(f"Error reading {exp_dir.name}: {e}")
    
    print(f"Found experiment info for {len(exp_info)} scenarios")
    
    # Now update the dataframe
    # For each BPOP row, match by scenario, then by ip_cov_target if possible
    bhpop_mask = df['method'] == 'bhpop_single_po'
    bhpop_df = df[bhpop_mask].copy()
    
    updated_count = 0
    for idx, row in bhpop_df.iterrows():
        scenario = row['scenario']
        ip_cov_target = row['ip_cov_target']
        
        if scenario in exp_info:
            # Try to find exact match by ip_cov_target first
            matches = [e for e in exp_info[scenario] 
                      if abs(e.get('ip_cov_target', np.nan) - ip_cov_target) < 0.01]
            
            if matches:
                # Use the first match
                df.loc[idx, 'eps_jump'] = matches[0]['eps_jump']
                updated_count += 1
            elif len(exp_info[scenario]) > 0:
                # No exact match, use the first available eps_jump
                # (This handles the case where the CSV was expanded across ip_cov_target)
                df.loc[idx, 'eps_jump'] = exp_info[scenario][0]['eps_jump']
                updated_count += 1
    
    print(f"\nUpdated {updated_count} BPOP rows with eps_jump")
    print(f"Remaining null eps_jump: {df[bhpop_mask]['eps_jump'].isna().sum()}")
    
    # Save
    output_path = results_dir / "experiment_summary_t0.40.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved updated CSV: {output_path}")
    
    # Verify
    df_check = pd.read_csv(output_path)
    bhpop_check = df_check[df_check['method'] == 'bhpop_single_po']
    print(f"\nVerification:")
    print(f"BPOP rows: {len(bhpop_check)}")
    print(f"BPOP rows with eps_jump: {bhpop_check['eps_jump'].notna().sum()}")
    print(f"Unique eps_jump values: {sorted(bhpop_check['eps_jump'].dropna().unique())}")

if __name__ == '__main__':
    main()
