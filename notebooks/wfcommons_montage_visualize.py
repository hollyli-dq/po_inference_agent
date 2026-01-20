#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.wfcommons_loader import load_workflow_instance_data


def _load_graph(json_path: Path) -> nx.DiGraph:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    instance = load_workflow_instance_data(data, source_path=json_path)
    task_ids = instance.get("task_ids", [])
    parents = instance.get("parents", {})

    graph = nx.DiGraph()
    graph.add_nodes_from(task_ids)
    for child, parent_set in parents.items():
        for parent in parent_set:
            graph.add_edge(parent, child)
    return graph


def _task_type(node_id: str) -> str:
    if "_" in node_id:
        return node_id.split("_", 1)[0]
    prefix = "".join(ch for ch in node_id if not ch.isdigit())
    return prefix or node_id


def _layered_layout(graph: nx.DiGraph) -> Dict[str, tuple]:
    try:
        order = list(nx.topological_sort(graph))
    except Exception:
        return nx.circular_layout(graph)

    levels: Dict[str, int] = {}
    for node in order:
        preds = list(graph.predecessors(node))
        if not preds:
            levels[node] = 0
        else:
            levels[node] = max(levels[p] for p in preds) + 1

    layers: Dict[int, List[str]] = {}
    for node, level in levels.items():
        layers.setdefault(level, []).append(node)

    pos: Dict[str, tuple] = {}
    for level in sorted(layers):
        nodes = sorted(layers[level], key=str)
        count = len(nodes)
        if count == 1:
            xs = [0.0]
        else:
            xs = [i / (count - 1) for i in range(count)]
        for idx, node in enumerate(nodes):
            pos[node] = (xs[idx], -float(level))
    return pos


def _choose_layout(graph: nx.DiGraph, seed: int) -> Dict[str, tuple]:
    try:
        import scipy.sparse as sp

        if hasattr(sp, "coo_array"):
            return nx.spring_layout(graph, seed=seed)
    except Exception:
        pass
    return _layered_layout(graph)


def _export_node_map(node_types: Dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "task_type"])
        for node_id in sorted(node_types):
            writer.writerow([node_id, node_types[node_id]])
    print(f"Saved node map to {output_path}")


def visualize_ground_truth(
    json_file: Path,
    output_path: Path,
    *,
    seed: int,
    node_size: int,
    label_mode: str,
    label_size: int,
    color_by_type: bool,
    show_legend: bool,
    export_nodes: Optional[Path],
) -> None:
    graph = _load_graph(json_file)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    node_types = {str(node): _task_type(str(node)) for node in graph.nodes()}

    plt.figure(figsize=(12, 8))
    pos = _choose_layout(graph, seed)
    node_colors = None
    legend_handles = []
    if color_by_type:
        types = sorted(set(node_types.values()))
        cmap = plt.get_cmap("tab20")
        color_map = {t: cmap(i % 20) for i, t in enumerate(types)}
        node_colors = [color_map[node_types[str(n)]] for n in graph.nodes()]
        if show_legend and len(types) <= 20:
            import matplotlib.patches as mpatches

            legend_handles = [
                mpatches.Patch(color=color_map[t], label=t) for t in types
            ]
        elif show_legend:
            print("Legend skipped: too many task types.")

    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=node_size,
        arrowsize=10,
        node_color=node_colors,
    )
    if label_mode != "none":
        if label_mode == "id":
            labels = {n: str(n) for n in graph.nodes()}
        else:
            labels = {n: node_types[str(n)] for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=label_size)
    plt.title(f"Ground Truth Graph: {json_file.name}")
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=max(label_size - 2, 6),
            frameon=False,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved graph to {output_path}")

    if export_nodes:
        _export_node_map(node_types, export_nodes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a WFCommons montage instance DAG (ground truth).",
    )
    parser.add_argument(
        "--json-path",
        default="data/wfcommons_wfinstances/montage/montage-chameleon-2mass-025d-001.json",
        help="Path to a WfCommons JSON workflow instance.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: notebooks/outputs/wfcommons_montage/ground_truth_<stem>.png).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Layout random seed.")
    parser.add_argument("--node-size", type=int, default=30, help="Node size.")
    parser.add_argument(
        "--label-mode",
        choices=["none", "id", "type"],
        default="none",
        help="Label nodes by full ID or task type.",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=6,
        help="Font size for node labels.",
    )
    parser.add_argument(
        "--color-by-type",
        action="store_true",
        help="Color nodes by task type.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Show legend for task types (up to 20).",
    )
    parser.add_argument(
        "--export-nodes",
        default=None,
        help="Write node_id->task_type CSV (default: notebooks/outputs/wfcommons_montage/ground_truth_<stem>_nodes.csv).",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise SystemExit(f"JSON file not found: {json_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("notebooks/outputs/wfcommons_montage")
        output_path = output_dir / f"ground_truth_{json_path.stem}.png"
    if args.export_nodes:
        export_nodes = Path(args.export_nodes)
    else:
        output_dir = Path("notebooks/outputs/wfcommons_montage")
        export_nodes = output_dir / f"ground_truth_{json_path.stem}_nodes.csv"

    visualize_ground_truth(
        json_path,
        output_path,
        seed=args.seed,
        node_size=args.node_size,
        label_mode=args.label_mode,
        label_size=args.label_size,
        color_by_type=args.color_by_type,
        show_legend=args.legend,
        export_nodes=export_nodes,
    )


if __name__ == "__main__":
    main()
