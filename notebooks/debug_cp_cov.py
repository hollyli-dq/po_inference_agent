#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from systematic_experiments.py
exec(open('systematic_experiments.py').read(), globals())

from src.utils.hpo_model_evaluation import critical_pairs_from_closure
import numpy as np

# Load data
print("Loading aliyun data...")
data_dict = load_aliyun_data()
scenario = "dual_zone_ecs_slb"

print(f"Processing scenario: {scenario}")

# Get data for this scenario
scenario_data = data_dict["scenario_data"][scenario]
true_closure_local = scenario_data["true_closure"]
full_orders_local = data_dict["orders_local_by_assessor"][scenario]

print(f"True closure shape: {true_closure_local.shape}")
print(f"Number of traces: {len(full_orders_local)}")

# Check critical pairs
crit_pairs = critical_pairs_from_closure(true_closure_local)
print(f"Critical pairs in true poset: {len(crit_pairs)}")

if crit_pairs:
    print(f"Sample critical pairs: {list(crit_pairs)[:5]}")

# Check valid linear extensions
valid_count = 0
for i, trace in enumerate(full_orders_local):
    if _is_linear_extension_partial(trace, true_closure_local):
        valid_count += 1
    if i < 3:  # Show first few
        print(f"Trace {i}: {trace} - Valid LE: {_is_linear_extension_partial(trace, true_closure_local)}")

print(f"Valid linear extensions: {valid_count}/{len(full_orders_local)}")

# Check CP-Cov for full set
if full_orders_local:
    full_cp_cov = cp_cov(full_orders_local, true_closure_local)
    print(f"Full trace set CP-Cov: {full_cp_cov:.3f}")

# Check CP-Cov for single trace
if full_orders_local:
    single_cp_cov = cp_cov([full_orders_local[0]], true_closure_local)
    print(f"Single trace CP-Cov: {single_cp_cov:.3f}")
    print(f"Single trace: {full_orders_local[0]}")
