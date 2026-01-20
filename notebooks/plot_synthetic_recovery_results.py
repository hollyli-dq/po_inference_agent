#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.utils.po_fun_plot import PO_plot
    _PO_PLOT_AVAILABLE = True
    _PO_PLOT_IMPORT_ERROR = None
except Exception as exc:
    _PO_PLOT_AVAILABLE = False
    _PO_PLOT_IMPORT_ERROR = exc


DEFAULT_INPUT = "notebooks/outputs/hpo_synthetic_recovery/synthetic_recovery_results.json"
DEFAULT_OUTPUT = "notebooks/outputs/hpo_synthetic_recovery/summary_metrics.png"
DEFAULT_POSET_DIR = "notebooks/outputs/hpo_synthetic_recovery/graph_compare"
DEFAULT_PO_PLOT_DIR = "notebooks/outputs/hpo_synthetic_recovery/po_plot"


def _load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No results found in {path}")
    return results


def _unique(values: List) -> List:
    return sorted({v for v in values})


def _select_single(values: List, name: str, value):
    unique = _unique(values)
    if value is not None:
        if value not in unique:
            raise ValueError(f"{name}={value} not in available values: {unique}")
        return value
    if len(unique) != 1:
        raise ValueError(f"Multiple {name} values found: {unique}. Pass --{name.replace('_', '-')}.")
    return unique[0]


def _save_current_plot(path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {path}")


class _PlotShowSaver:
    def __init__(self, prefix: Path, *, dpi: int) -> None:
        self.prefix = prefix
        self.dpi = dpi
        self._orig_show = None
        self._count = 0

    def __enter__(self) -> "_PlotShowSaver":
        self._orig_show = plt.show

        def _show(*args, **kwargs):
            self._count += 1
            out_path = self.prefix.with_name(f"{self.prefix.name}_{self._count}.png")
            _save_current_plot(out_path, dpi=self.dpi)

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_show is not None:
            plt.show = self._orig_show


def _load_poset_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _poset_tag(metadata: Dict[str, object]) -> str:
    parts = []
    for key in ("K", "beta", "epsilon", "trace_count"):
        if key in metadata:
            parts.append(f"{key}{metadata[key]}")
    return "_".join(parts) if parts else "poset"


def plot_posets_from_json(poset_path: Path, output_dir: Path, *, dpi: int) -> None:
    payload = _load_poset_payload(poset_path)
    metadata = payload.get("metadata", {})
    tag = _poset_tag(metadata)

    items = payload.get("items", [])
    assessors = [int(a) for a in payload.get("assessors", [])]
    m_a_dict = {int(k): v for k, v in payload.get("M_a_dict", {}).items()}
    h_true = {int(k): np.array(v, dtype=int) for k, v in payload.get("h_true", {}).items()}
    h_inferred = {
        int(k): np.array(v, dtype=int) for k, v in payload.get("h_inferred", {}).items()
    }

    index_to_item = {i: item for i, item in enumerate(items)}
    global_items = [str(index_to_item[i]) for i in sorted(index_to_item)]

    output_dir.mkdir(parents=True, exist_ok=True)

    if 0 in h_true and 0 in h_inferred:
        prefix = output_dir / f"po_plot_global_{tag}"
        with _PlotShowSaver(prefix, dpi=dpi):
            PO_plot.compare_and_visualize_global(
                h_true[0],
                h_inferred[0],
                index_to_item,
                global_items,
                do_transitive_reduction=True,
            )

    for assessor in assessors:
        if assessor not in h_true or assessor not in h_inferred:
            continue
        ma_list = [str(item) for item in m_a_dict.get(assessor, [])]
        index_to_item_local = {
            i: item for i, item in enumerate(m_a_dict.get(assessor, []))
        }
        prefix = output_dir / f"po_plot_assessor{assessor}_{tag}"
        with _PlotShowSaver(prefix, dpi=dpi):
            PO_plot.compare_and_visualize_assessor(
                assessor,
                ma_list,
                h_true[assessor],
                h_inferred[assessor],
                index_to_item_local,
                do_transitive_reduction=True,
            )


def plot_posets_for_results(
    results: List[dict],
    poset_dir: Path,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    if not _PO_PLOT_AVAILABLE:
        print(f"[WARN] PO_plot unavailable: {_PO_PLOT_IMPORT_ERROR}")
        print("       Install networkx to enable PO_plot visualizations.")
        return
    combos = sorted(
        {
            (row["K"], row["beta"], row["epsilon"], row["trace_count"])
            for row in results
        }
    )
    for K, beta, epsilon, trace_count in combos:
        filename = f"inferred_posets_K{K}_beta{beta}_eps{epsilon}_traces{trace_count}.json"
        poset_path = poset_dir / filename
        if not poset_path.exists():
            print(f"[WARN] Missing poset file: {poset_path}")
            continue
        combo_dir = output_dir / f"K{K}_beta{beta}_eps{epsilon}_traces{trace_count}"
        plot_posets_from_json(poset_path, combo_dir, dpi=dpi)


def plot_metrics(
    rows: List[dict],
    output_path: Path,
    *,
    assessor_id: int,
    epsilon: float,
    trace_count: int,
    dpi: int,
) -> None:
    metrics = [
        ("shd", "SHD (lower better)"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
        ("waic_unlabeled", "WAIC (lower better)"),
        ("ece", "ECE (lower better)"),
        ("mean_entropy", "Mean entropy (lower better)"),
    ]

    ks = sorted({row["K"] for row in rows})
    betas = sorted({row["beta"] for row in rows})

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        for beta in betas:
            ys = []
            for k in ks:
                matches = [
                    row[metric]
                    for row in rows
                    if row["K"] == k and row["beta"] == beta
                ]
                ys.append(matches[0] if matches else np.nan)
            ax.plot(ks, ys, marker="o", label=f"beta={beta}")
        ax.set_title(label)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Assessor {assessor_id} | epsilon={epsilon} | traces={trace_count}")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot synthetic recovery summary metrics.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to synthetic_recovery_results.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--poset-dir", default=DEFAULT_POSET_DIR, help="Directory with inferred_posets_*.json")
    parser.add_argument("--poset-output-dir", default=DEFAULT_PO_PLOT_DIR, help="Output dir for PO_plot figures")
    parser.add_argument("--skip-po-plot", action="store_true", help="Skip PO_plot visualizations")
    parser.add_argument("--assessor-id", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--trace-count", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    results = _load_results(Path(args.input))

    assessor_id = _select_single([row["assessor_id"] for row in results], "assessor_id", args.assessor_id)
    epsilon = _select_single([row["epsilon"] for row in results], "epsilon", args.epsilon)
    trace_count = _select_single([row["trace_count"] for row in results], "trace_count", args.trace_count)

    filtered = [
        row
        for row in results
        if row["assessor_id"] == assessor_id
        and row["epsilon"] == epsilon
        and row["trace_count"] == trace_count
    ]
    if not filtered:
        raise ValueError("No rows match the selected assessor/epsilon/trace_count.")

    plot_metrics(
        filtered,
        Path(args.output),
        assessor_id=assessor_id,
        epsilon=epsilon,
        trace_count=trace_count,
        dpi=args.dpi,
    )

    if not args.skip_po_plot:
        poset_dir = Path(args.poset_dir)
        if not poset_dir.exists():
            print(f"[WARN] poset-dir does not exist: {poset_dir}")
            return
        plot_posets_for_results(
            results,
            poset_dir,
            Path(args.poset_output_dir),
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
