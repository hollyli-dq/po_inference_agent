#!/usr/bin/env python3
"""
Recompute experiment results with threshold=0.4 instead of 0.5
Uses existing avg_H.pkl files, no need to re-run MCMC
"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance
from src.utils.po_fun import BasicUtils

# Load experiment summary
results_dir = Path("systematic_experiment_results")
df = pd.read_csv(results_dir / "experiment_summary.csv")

# Update only BHPOP results
rows_updated = 0
for idx, row in df[df['method'] == 'bhpop_single_po'].iterrows():
    scenario = row['scenario']
    cp_cov = row['cp_cov_target']
    eps = row['eps_jump']
    
    # Find matching experiment directory
    exp_dirs = list(results_dir.glob(f"exp_*_{scenario}"))
    
    for exp_dir in exp_dirs:
        try:
            # Load avg_H
            with open(exp_dir / "avg_H.pkl", "rb") as f:
                avg_H = pickle.load(f)
            
            # Load true cover (need to reconstruct from scenario data)
            # This is complex, so let's skip for now and just recompute from the CSV
            # Actually, we need the true_cover for each scenario
            break
        except:
            continue

print(f"This script needs access to true_cover for each scenario.")
print(f"Simpler approach: modify systematic_experiments.py and recompute only the metrics.")
