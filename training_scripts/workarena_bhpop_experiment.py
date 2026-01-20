#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def _slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}


def _select_tasks(task_groups: Dict[str, List], selected: List[str], top_k: int) -> List[str]:
    if selected:
        selected_set = {s.lower() for s in selected}
        matches = []
        for task_key, traces in task_groups.items():
            task_name = traces[0].task_name if traces else task_key
            key_lower = task_key.lower()
            name_lower = task_name.lower()
            if any(sel in key_lower or sel in name_lower for sel in selected_set):
                matches.append(task_key)
        return matches
    ranked = sorted(task_groups.items(), key=lambda item: len(item[1]), reverse=True)
    return [task for task, _ in ranked[:top_k]]


def _summarize_posterior(results: Dict, burn_in_idx: int) -> Dict[str, float]:
    summary = {}
    for key in ("rho_trace", "tau_trace", "K_trace"):
        trace = results.get(key, [])
        if trace:
            summary[key.replace("_trace", "")] = float(np.mean(trace[burn_in_idx:]))
    if "softmax_beta_trace" in results:
        beta_trace = results.get("softmax_beta_trace", [])
        if beta_trace:
            summary["softmax_beta"] = float(np.mean(beta_trace[burn_in_idx:]))
    if "softmax_lambda_trace" in results:
        lambda_trace = results.get("softmax_lambda_trace", [])
        if lambda_trace:
            summary["softmax_lambda"] = float(np.mean(lambda_trace[burn_in_idx:]))
    return summary


def _posterior_graphs(results: Dict, burn_in_idx: int, threshold: float):
    from src.utils.po_fun import BasicUtils

    h_trace = results.get("H_trace", [])
    if not h_trace:
        return {}, {}

    h_trace = h_trace[burn_in_idx:]
    final_h = {}
    edge_lists = {}

    keys = set()
    for h_iter in h_trace:
        keys.update(h_iter.keys())

    for key in sorted(keys):
        mats = [h_iter[key] for h_iter in h_trace if key in h_iter]
        if not mats:
            continue
        mean_mat = np.mean(mats, axis=0)
        bin_mat = (mean_mat >= threshold).astype(int)
        reduced = BasicUtils.transitive_reduction_optimized(bin_mat)
        final_h[key] = reduced
        edges = [(i, j) for i in range(reduced.shape[0]) for j in range(reduced.shape[1]) if reduced[i, j] == 1]
        edge_lists[key] = edges
    return final_h, edge_lists


def _save_graph_visualizations(
    final_h: Dict[int, np.ndarray],
    idx_to_action: Dict[int, str],
    M0: List[int],
    M_a_dict: Dict[int, List[int]],
    output_dir: Path,
    seed: int,
):
    from src.utils.po_fun_plot import PO_plot

    if 0 in final_h:
        h0 = final_h[0]
        labels = [idx_to_action[idx] for idx in M0]
        PO_plot.visualize_partial_order(h0, labels, title="Global SOP (h0)")
        path = output_dir / f"partial_order_global_seed_{seed}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    for assessor_id, h_a in final_h.items():
        if assessor_id == 0:
            continue
        labels = [idx_to_action[idx] for idx in M_a_dict.get(assessor_id, [])]
        if not labels:
            continue
        PO_plot.visualize_partial_order(h_a, labels, title=f"Agent {assessor_id} SOP")
        path = output_dir / f"partial_order_agent_{assessor_id}_seed_{seed}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()


def _compute_consistency_rate(adj: np.ndarray, observed_orders: Dict[int, List[List[int]]]) -> float:
    from src.utils.workarena_execution import is_linear_extension

    total = 0
    ok = 0
    for orders in observed_orders.values():
        for order in orders:
            total += 1
            if is_linear_extension(order, adj):
                ok += 1
    return ok / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHPOP on WorkArena++ tasks.")
    parser.add_argument("--data-path", required=True, help="Path to WorkArena++ trace JSON/JSONL or directory.")
    parser.add_argument("--action-map", default="config/workarena_action_map.yaml", help="YAML action abstraction map.")
    parser.add_argument("--config", default="config/hpo_mcmc_configuration.yaml", help="MCMC config YAML.")
    parser.add_argument("--output-dir", default="notebooks/outputs/workarena_bhpop", help="Output directory.")
    parser.add_argument("--tasks", nargs="*", default=[], help="Task IDs/names to run.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top tasks (by traces) to run if none selected.")
    parser.add_argument("--min-traces-per-agent", type=int, default=2, help="Minimum successful traces per agent.")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override MCMC iterations.")
    parser.add_argument("--k-prior", type=int, default=None, help="Override K prior.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42], help="Random seeds for stability checks.")
    parser.add_argument("--burn-in-frac", type=float, default=0.5, help="Fraction of samples to discard as burn-in.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Posterior edge threshold.")
    parser.add_argument("--parallel-limit", type=int, default=None, help="Max parallel actions per step.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_hpo_k_optim
    from src.utils.workarena_data_processing import build_hpo_inputs_for_task, group_traces_by_task, load_workarena_traces
    from src.utils.workarena_execution import simulate_execution

    config = _load_config(project_root / args.config)
    num_iterations = args.num_iterations or config.get("mcmc", {}).get("num_iterations", 500000)
    k_prior = args.k_prior or config.get("prior", {}).get("k_prior", 3)

    noise_model = "softmax_queue_jump"
    softmax_beta_prior = (2.0, 0.5)
    softmax_beta_stepsize = 0.3
    softmax_lambda = 1.0

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    action_map_path = Path(args.action_map)
    if not action_map_path.is_absolute():
        action_map_path = project_root / action_map_path

    traces = load_workarena_traces(data_path, action_map_path)
    traces = [trace for trace in traces if trace.success]
    grouped = group_traces_by_task(traces)
    selected_tasks = _select_tasks(grouped, args.tasks, args.top_k)

    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    stability_records = defaultdict(list)

    for task_key in selected_tasks:
        task_traces = grouped[task_key]
        task_name = task_traces[0].task_name if task_traces else task_key
        task_slug = _slugify(task_key)
        task_dir = output_root / task_slug
        task_dir.mkdir(parents=True, exist_ok=True)

        (
            M0,
            assessors,
            M_a_dict,
            O_a_i_dict,
            observed_orders,
            action_to_idx,
            idx_to_action,
            action_info,
            assessor_to_agent,
        ) = build_hpo_inputs_for_task(
            task_traces,
            unroll_duplicates=True,
            min_traces_per_agent=args.min_traces_per_agent,
        )

        print(f"Task '{task_key}': {len(M0)} actions, {len(assessors)} agents")

        for seed in args.seeds:
            mcmc_config = {
                "num_iterations": num_iterations,
                "M0": M0,
                "assessors": assessors,
                "M_a_dict": M_a_dict,
                "O_a_i_dict": O_a_i_dict,
                "observed_orders": observed_orders,
                "dr": config.get("rho", {}).get("dr", 0.1),
                "drrt": config.get("rhotau", {}).get("drrt", 0.8),
                "noise_option": noise_model,
                "softmax_lambda": softmax_lambda,
                "softmax_beta_prior": softmax_beta_prior,
                "softmax_beta_stepsize": softmax_beta_stepsize,
                "rho_prior": config.get("prior", {}).get("rho_prior", 1 / 6),
                "noise_beta_prior": config.get("prior", {}).get("noise_beta_prior", 1.0),
                "K_prior": k_prior,
                "random_seed": seed,
                "cycle_length": 500,
            }

            start = time.time()
            results = mcmc_simulation_hpo_k_optim(**mcmc_config)
            elapsed = time.time() - start

            result_path = task_dir / f"mcmc_results_seed_{seed}.pkl"
            with result_path.open("wb") as handle:
                pickle.dump(results, handle)

            burn_in_idx = int(len(results.get("H_trace", [])) * args.burn_in_frac)
            posterior_summary = _summarize_posterior(results, burn_in_idx)
            final_h, edge_lists = _posterior_graphs(results, burn_in_idx, args.threshold)

            graph_path = task_dir / f"inferred_graphs_seed_{seed}.json"
            graph_payload = {
                "task_key": task_key,
                "task_name": task_name,
                "assessor_to_agent": assessor_to_agent,
                "node_labels": idx_to_action,
                "edges": {
                    str(k): [(idx_to_action.get(src, str(src)), idx_to_action.get(dst, str(dst))) for src, dst in edges]
                    for k, edges in edge_lists.items()
                },
            }
            with graph_path.open("w", encoding="utf-8") as handle:
                json.dump(graph_payload, handle, indent=2, ensure_ascii=True)

            if final_h:
                _save_graph_visualizations(final_h, idx_to_action, M0, M_a_dict, task_dir, seed)

            execution_metrics = {}
            if 0 in final_h:
                adj = final_h[0]
                full_history = simulate_execution(adj, action_info, parallel=False, full_history=True)
                structured = simulate_execution(
                    adj,
                    action_info,
                    parallel=True,
                    parallel_limit=args.parallel_limit,
                    full_history=False,
                    include_dependents=False,
                )
                consistency = _compute_consistency_rate(adj, observed_orders)
                execution_metrics = {
                    "full_history": full_history.__dict__,
                    "structured": structured.__dict__,
                    "consistency_rate": consistency,
                }

            metrics_path = task_dir / f"execution_metrics_seed_{seed}.json"
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(execution_metrics, handle, indent=2, ensure_ascii=True)

            summary_rows.append(
                {
                    "task_key": task_key,
                    "seed": seed,
                    "num_actions": len(M0),
                    "num_agents": len(assessors),
                    "elapsed_sec": round(elapsed, 2),
                    "posterior": posterior_summary,
                    "tokens_full_history": execution_metrics.get("full_history", {}).get("tokens_total"),
                    "tokens_structured": execution_metrics.get("structured", {}).get("tokens_total"),
                    "latency_full_ms": execution_metrics.get("full_history", {}).get("latency_ms"),
                    "latency_structured_ms": execution_metrics.get("structured", {}).get("latency_ms"),
                    "consistency_rate": execution_metrics.get("consistency_rate"),
                }
            )

            if 0 in final_h:
                edges = edge_lists.get(0, [])
                stability_records[task_key].append((seed, set(edges)))

        if len(stability_records[task_key]) > 1:
            pairs = []
            for i in range(len(stability_records[task_key])):
                for j in range(i + 1, len(stability_records[task_key])):
                    seed_i, edges_i = stability_records[task_key][i]
                    seed_j, edges_j = stability_records[task_key][j]
                    union = len(edges_i | edges_j)
                    jaccard = len(edges_i & edges_j) / union if union else 1.0
                    pairs.append({"seed_a": seed_i, "seed_b": seed_j, "jaccard": jaccard})
            stability_path = task_dir / "stability_across_seeds.json"
            with stability_path.open("w", encoding="utf-8") as handle:
                json.dump(pairs, handle, indent=2, ensure_ascii=True)

    summary_path = output_root / "workarena_bhpop_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, ensure_ascii=True)

    summary_md = output_root / "experiment_summary.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# WorkArena++ BHPOP Summary\n\n")
        handle.write("This summary is generated from the WorkArena++ runs.\n\n")
        handle.write(
            "Structural learning: inferred SOP DAGs are saved per task and agent; "
            "seed stability scores are in `stability_across_seeds.json` when multiple seeds are run.\n\n"
        )
        tokens_reductions = []
        latency_reductions = []
        for row in summary_rows:
            tokens_full = row.get("tokens_full_history")
            tokens_struct = row.get("tokens_structured")
            latency_full = row.get("latency_full_ms")
            latency_struct = row.get("latency_structured_ms")
            if tokens_full and tokens_struct:
                tokens_reductions.append(1.0 - (tokens_struct / tokens_full))
            if latency_full and latency_struct:
                latency_reductions.append(1.0 - (latency_struct / latency_full))

        if tokens_reductions:
            avg_token_reduction = 100.0 * float(np.mean(tokens_reductions))
            handle.write(f"- Average token reduction: {avg_token_reduction:.1f}%\n")
        if latency_reductions:
            avg_latency_reduction = 100.0 * float(np.mean(latency_reductions))
            handle.write(f"- Average latency reduction: {avg_latency_reduction:.1f}%\n")
        handle.write("\n")

        for row in summary_rows:
            handle.write(f"## Task: {row['task_key']} (seed {row['seed']})\n")
            handle.write(f"- Actions: {row['num_actions']}, Agents: {row['num_agents']}\n")
            handle.write(f"- Elapsed: {row['elapsed_sec']} sec\n")
            handle.write(
                f"- Tokens (full vs structured): {row.get('tokens_full_history')} vs {row.get('tokens_structured')}\n"
            )
            handle.write(
                f"- Latency ms (full vs structured): {row.get('latency_full_ms')} vs {row.get('latency_structured_ms')}\n"
            )
            handle.write(f"- Consistency rate: {row.get('consistency_rate')}\n\n")


if __name__ == "__main__":
    main()
