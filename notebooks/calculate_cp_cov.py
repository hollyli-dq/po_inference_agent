#!/usr/bin/env python3

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from systematic_experiments.py
exec(open('systematic_experiments.py').read(), globals())

def calculate_cp_cov_for_scenarios():
    """Calculate CP-Cov for each scenario in the Cloud-IaC-6 dataset."""

    # Build data dictionary (but skip the experiment running)
    project_root = Path(__file__).parent.parent

    # Load scenarios directly without running experiments
    data_root = project_root / "aliyun_data"
    scenarios = load_manual_scenarios(data_root / "manual_scenarios")
    scenario_ids = sorted(scenarios.keys())

    scenario_data: Dict[str, Any] = {}

    for sid in scenario_ids:
        true_adj, task_ids, task_to_idx = scenario_to_adj(scenarios[sid])
        true_closure = BasicUtils.transitive_closure(true_adj.astype(np.int8))
        true_cover = BasicUtils.transitive_reduction_optimized(true_closure.astype(np.int8))
        scenario_data[sid] = {
            "task_ids": task_ids,
            "task_to_idx": task_to_idx,
            "true_adj": true_adj,
            "true_closure": true_closure,
            "true_cover": true_cover,
        }

    # Load traces
    expert_traces = load_traces(data_root / "expert_traces")
    model_traces = load_traces(data_root / "traces")

    orders_local_by_assessor: Dict[str, List[List[int]]] = {}

    for scenario_id in scenario_ids:
        task_ids = scenario_data[scenario_id]["task_ids"]
        task_set = set(task_ids)
        local_task_to_idx = scenario_data[scenario_id]["task_to_idx"]

        selected_traces: List[Dict[str, Any]] = []

        # Add expert traces
        for trace in expert_traces:
            if trace.get("intent_type") == scenario_id:
                selected_traces.append(trace)

        # Add model traces if using combined source
        if TRACE_SOURCE in ("combined",):
            for trace in model_traces:
                if trace.get("intent_type") == scenario_id:
                    selected_traces.append(trace)

        orders_local = []
        for trace in selected_traces:
            actions = trace.get("actions_executed", [])
            # Convert action names to indices
            order = []
            for action in actions:
                if action in local_task_to_idx:
                    order.append(local_task_to_idx[action])
            if order:  # Only add if we have actions
                orders_local.append(order)

        orders_local_by_assessor[scenario_id] = orders_local

    data_dict = {
        "scenario_data": scenario_data,
        "orders_local_by_assessor": orders_local_by_assessor
    }

    print("=== Cloud-IaC-6 Critical-Pair Coverage Analysis ===\n")

    # Scenario mapping to match table order
    scenario_mapping = {
        'simple_ecs': 'S1',
        'slb_ecs_rds': 'S2',
        'slb_ecs_redis': 'S3',
        'eip_slb_ecs': 'S4',
        'dual_zone_ecs_slb': 'S5',
        'dual_zone_ecs_slb_rds': 'S6'
    }

    results = []

    for scenario_id in sorted(scenario_mapping.keys(), key=lambda x: scenario_mapping[x]):
        scenario_data = data_dict["scenario_data"][scenario_id]
        true_closure = scenario_data["true_closure"]
        orders_local = data_dict["orders_local_by_assessor"][scenario_id]

        # Calculate critical pairs
        crit_pairs = incomparable_pairs_from_closure(true_closure)
        num_crit_pairs = len(crit_pairs)

        # Calculate CP-Cov
        if orders_local:
            cp_cov_value = cp_cov(orders_local, true_closure)
        else:
            cp_cov_value = 0.0

        # Get scenario statistics
        num_traces = len(orders_local)
        num_nodes = true_closure.shape[0]
        num_edges = np.sum(true_closure) - num_nodes  # subtract self-loops

        scenario_name = scenario_mapping[scenario_id]

        results.append({
            'scenario': scenario_name,
            'id': scenario_id,
            'nodes': num_nodes,
            'edges': num_edges,
            'traces': num_traces,
            'crit_pairs': num_crit_pairs,
            'cp_cov': cp_cov_value
        })

        print(f"{scenario_name} ({scenario_id}):")
        print(f"  |V| = {num_nodes}, |E| = {num_edges}, N = {num_traces}")
        print(".3f")
        print(f"  Critical pairs: {num_crit_pairs}")
        print()

    return results

if __name__ == "__main__":
    results = calculate_cp_cov_for_scenarios()

    # Print summary table
    print("=== Summary Table ===")
    print("Scenario | |V| | |E| | N | CP-Cov")
    print("---------|-----|-----|----|-------")
    for r in results:
        print("8s")
