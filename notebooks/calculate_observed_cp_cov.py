#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

def _critical_pair_coverage(orders, closure):
    """Calculate critical pair coverage from observation orders."""
    n = closure.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if closure[i, j] or closure[j, i]:
                continue
            pairs.add((i, j))
    if not pairs:
        return 0, 0, 1.0

    seen = {}
    for order in orders:
        pos = {t: i for i, t in enumerate(order)}
        items = list(pos.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                pair = (a, b) if a < b else (b, a)
                if pair not in pairs:
                    continue
                u, v = pair
                mask = 1 if pos[u] < pos[v] else 2
                seen[pair] = seen.get(pair, 0) | mask

    covered = sum(1 for pair in pairs if seen.get(pair, 0) == 3)
    return covered, len(pairs), covered / max(1, len(pairs))

def transitive_closure(adj_matrix: np.ndarray) -> np.ndarray:
    """Compute transitive closure using Floyd-Warshall algorithm."""
    n = adj_matrix.shape[0]
    closure = adj_matrix.copy().astype(int)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])

    return closure.astype(int)

def load_manual_scenarios(scenarios_dir: Path) -> Dict[str, Any]:
    scenarios = {}
    for json_file in scenarios_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            scenario = json.load(f)
            scenario_name = json_file.stem  # Remove .json extension
            scenarios[scenario_name] = scenario
    return scenarios

def scenario_to_adj(scenario: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Convert scenario edges to adjacency matrix."""
    edges = scenario["edges"]
    # Extract all unique nodes
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])

    task_ids = sorted(list(nodes))
    task_to_idx = {task: i for i, task in enumerate(task_ids)}

    n = len(task_ids)
    adj = np.zeros((n, n), dtype=int)

    for edge in edges:
        i = task_to_idx[edge[0]]
        j = task_to_idx[edge[1]]
        adj[i, j] = 1

    return adj, task_ids, task_to_idx

def load_traces(traces_dir: Path) -> List[Dict[str, Any]]:
    traces = []
    for json_file in traces_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            trace_data = json.load(f)
            traces.append(trace_data)
    return traces

def calculate_observed_cp_coverage():
    """Calculate observed critical-pair coverage for each Cloud-IaC-6 scenario."""

    project_root = Path(__file__).parent.parent
    data_root = project_root / "aliyun_data"

    # Load scenarios
    scenarios = load_manual_scenarios(data_root / "manual_scenarios")
    scenario_ids = sorted(scenarios.keys())

    # Load traces
    expert_traces = load_traces(data_root / "expert_traces")
    model_traces = load_traces(data_root / "traces")

    print("=== Cloud-IaC-6 Observed Critical-Pair Coverage Analysis ===\n")

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
        # Build scenario data
        scenario = scenarios[scenario_id]
        true_adj, task_ids, task_to_idx = scenario_to_adj(scenario)
        true_closure = transitive_closure(true_adj)

        # Get traces for this scenario
        selected_traces = []

        # Add expert traces
        for trace in expert_traces:
            # Check both direct intent_type and nested intent.intent_type
            intent_type = trace.get("intent_type") or trace.get("intent", {}).get("intent_type")
            if intent_type == scenario_id:
                selected_traces.append(trace)

        # Add model traces (combined source)
        for trace in model_traces:
            intent = trace.get("intent", {})
            if intent.get("intent_type") == scenario_id:
                selected_traces.append(trace)

        # Convert traces to orders
        orders_local = []
        for trace in selected_traces:
            actions = trace.get("actions_executed", [])
            # Convert action names to indices
            order = []
            for action in actions:
                if action in task_to_idx:
                    order.append(task_to_idx[action])
            if order:  # Only add if we have actions
                orders_local.append(order)

        # Calculate critical pair coverage
        covered_pairs, total_pairs, coverage_ratio = _critical_pair_coverage(orders_local, true_closure)

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
            'covered_pairs': covered_pairs,
            'total_pairs': total_pairs,
            'coverage_ratio': coverage_ratio
        })

        print(f"{scenario_name} ({scenario_id}):")
        print(f"  |V| = {num_nodes}, |E| = {num_edges}, N = {num_traces}")
        print(f"  Critical pairs covered: {covered_pairs}/{total_pairs} ({coverage_ratio:.3f})")
        print()

    return results

if __name__ == "__main__":
    results = calculate_observed_cp_coverage()

    # Print summary table
    print("=== Summary Table ===")
    print("Scenario | |V| | |E| | N | Covered/Total CP | Coverage")
    print("---------|-----|-----|----|-----------------|----------")
    for r in results:
        print("8s")
