#!/usr/bin/env python3
"""
Recompute final_H and plot results for WfInstances experiments (SRASearch, Epigenomics).

Reads avg_H.pkl / H_trace.pkl and true_cover.npy from ip_cov_* directories,
recomputes metrics with configurable threshold, and generates comparison plots.

Usage:
    python src/analysis/recompute_H_wfinstances.py --experiment srasearch --threshold 0.5
    python src/analysis/recompute_H_wfinstances.py --experiment epigenomics --threshold 0.5
    python src/analysis/recompute_H_wfinstances.py --experiment all --threshold 0.5 --no-plot
"""
from __future__ import annotations

import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib must be configured before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hpo_model_evaluation import (
    precision_recall,
    f1_score,
    structural_hamming_distance,
    _incomparable_pairs,
)
from src.utils.po_fun import BasicUtils
from src.utils.result_paths import (
    WFINSTANCES_SRASEARCH_RESULTS_DIR,
    LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
    WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    WFINSTANCES_RESULTS_ROOT,
    prefer_existing_path,
)

# Optional: use ICML style from the training runner
def _setup_style():
    try:
        from training_scripts.systematic_experiments import setup_icml_style
        setup_icml_style()
    except Exception:
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


# -----------------------------------------------------------------------------
# Pickle / data loading (from recompute_H_with_threshold)
# -----------------------------------------------------------------------------

def load_pickle_with_numpy_fix(path: Path) -> Any:
    """Load pickle with workaround for numpy version mismatch."""
    try:
        import numpy.core
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = numpy.core
            try:
                sys.modules["numpy._core._multiarray_umath"] = numpy.core.multiarray
            except Exception:
                pass
    except ImportError:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError) as e:
        if "numpy._core" in str(e) or "numpy.core" in str(e):
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")
        raise


def compute_marginal_mode(H_trace: np.ndarray) -> np.ndarray:
    """Marginal mode over MCMC samples per edge."""
    from scipy import stats
    if isinstance(H_trace, list):
        H_trace = np.array(H_trace)
    n_samples, n, _ = H_trace.shape
    H_mode = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j:
                mode_result = stats.mode(H_trace[:, i, j], keepdims=False)
                H_mode[i, j] = int(mode_result.mode)
    return H_mode


# -----------------------------------------------------------------------------
# WfInstances-specific layout
# -----------------------------------------------------------------------------

RESULTS_BY_EXPERIMENT = {
    "srasearch": prefer_existing_path(
        WFINSTANCES_SRASEARCH_RESULTS_DIR,
        LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
    ),
    "epigenomics": prefer_existing_path(
        WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
        LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    ),
}


def discover_ip_cov_dirs(results_root: Path) -> List[Tuple[float, Path]]:
    """Find ip_cov_* directories. Returns [(ip_cov_target, path), ...]."""
    out = []
    for d in results_root.iterdir():
        if not d.is_dir() or not d.name.startswith("ip_cov_"):
            continue
        suffix = d.name.replace("ip_cov_", "").replace("_", ".")
        try:
            t = float(suffix)
            out.append((t, d))
        except ValueError:
            continue
    return sorted(out, key=lambda x: x[0])


def process_wfinstances_experiment(
    exp_dir: Path,
    threshold: float,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """
    Process one ip_cov_* directory. Uses true_cover.npy from that dir.
    Returns dict with metrics and inferred cover, or None on failure.
    """
    try:
        true_path = exp_dir / "true_cover.npy"
        if not true_path.exists():
            return None
        true_cover = np.load(true_path)

        if mode == "marginal":
            h_path = exp_dir / "H_trace.pkl"
            if not h_path.exists():
                return None
            H_trace = load_pickle_with_numpy_fix(h_path)
            H_final = compute_marginal_mode(H_trace)
            H_final_closure = BasicUtils.transitive_closure(H_final)
        else:
            avg_path = exp_dir / "avg_H.pkl"
            if not avg_path.exists():
                return None
            avg_H = load_pickle_with_numpy_fix(avg_path)
            H_final = (avg_H >= threshold).astype(np.int8)
            H_final_closure = BasicUtils.transitive_closure(H_final)

        H_final_cover = BasicUtils.transitive_reduction(H_final_closure)
        true_closure = BasicUtils.transitive_closure(true_cover)

        p_edge, r_edge = precision_recall(true_cover, H_final_cover)
        f1_edge = f1_score(p_edge, r_edge)
        shd = structural_hamming_distance(true_cover, H_final_cover)

        true_ip = _incomparable_pairs(true_closure)
        pred_ip = _incomparable_pairs(H_final_closure)
        tp = len(true_ip & pred_ip)
        fp = len(pred_ip - true_ip)
        fn = len(true_ip - pred_ip)
        p_ip = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ip = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ip = f1_score(p_ip, r_ip)

        # Parse ip_cov target from dir name, e.g. ip_cov_0_95 -> 0.95
        name = exp_dir.name.replace("ip_cov_", "").replace("_", ".")
        try:
            ip_cov_target = float(name)
        except ValueError:
            ip_cov_target = np.nan

        # Optionally load realized from experiment_summary
        ip_cov_realized = np.nan
        summary_path = exp_dir.parent / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summ = json.load(f)
            for r in summ.get("results", []):
                if abs(r.get("ip_cov_target", -1) - ip_cov_target) < 1e-6:
                    ip_cov_realized = r.get("ip_cov_realized", np.nan)
                    break

        output_name = f"final_H_{mode}_t{threshold:.2f}.pkl" if mode == "single_po" else "final_H_marginal.pkl"
        with open(exp_dir / output_name, "wb") as f:
            pickle.dump(H_final_cover, f)

        return {
            "ip_cov_target": ip_cov_target,
            "ip_cov_realized": ip_cov_realized,
            "cover_precision": p_edge,
            "cover_recall": r_edge,
            "cover_f1": f1_edge,
            "shd": shd,
            "ip_precision": p_ip,
            "ip_recall": r_ip,
            "ip_f1": f1_ip,
            "inferred_cover": H_final_cover,
            "true_cover": true_cover,
            "exp_dir": exp_dir,
        }
    except Exception as e:
        print(f"  ERROR {exp_dir.name}: {e}")
        return None


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_comparison_pages(
    workflow_name: str,
    results_root: Path,
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a PDF with one page per IP-Cov: Ground Truth | BHPOP | Difference."""
    _setup_style()
    with PdfPages(str(output_path)) as pdf:
        for r in rows:
            true_cover = r["true_cover"]
            inferred = r["inferred_cover"]
            ip = r["ip_cov_target"]
            f1 = r["cover_f1"]
            shd = r["shd"]
            realized = r.get("ip_cov_realized", ip)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"{workflow_name} | IP-Cov={realized:.2f} | F1={f1:.3f}")

            axes[0].imshow(true_cover, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            axes[0].set_title(f"Ground Truth ({int(true_cover.sum())} edges)")

            axes[1].imshow(inferred, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
            axes[1].set_title(f"BHPOP ({int(inferred.sum())} edges)")

            diff = inferred.astype(int) - true_cover.astype(int)
            axes[2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
            axes[2].set_title(f"Difference (SHD={shd})")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_f1_vs_ip_cov(
    workflow_name: str,
    df: pd.DataFrame,
    output_base: Path,
    exts: List[str] = (".pdf", ".png"),
) -> None:
    """F1 and SHD vs IP-Cov (realized or target). Saves output_base + each ext."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = df["ip_cov_realized"].fillna(df["ip_cov_target"]).values

    axes[0].plot(x, df["cover_f1"], "o-", color="#377eb8", lw=1.5, markersize=6)
    axes[0].set_xlabel("IP-Cov (realized)")
    axes[0].set_ylabel("Cover F1")
    axes[0].set_title(f"{workflow_name}: F1 vs IP-Cov")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, df["shd"], "s-", color="#e41a1c", lw=1.5, markersize=6)
    axes[1].set_xlabel("IP-Cov (realized)")
    axes[1].set_ylabel("SHD")
    axes[1].set_title(f"{workflow_name}: SHD vs IP-Cov")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # Use stem + ext (avoid with_suffix: stem like "f1_shd_vs_ipcov_t0.50" has suffix .50)
    base = output_base.parent / output_base.stem
    for ext in exts:
        p = base.parent / (base.name + ext)
        plt.savefig(p, bbox_inches="tight", dpi=150 if ext == ".png" else None)
        print(f"  Saved: {p}")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_experiment(
    name: str,
    results_root: Path,
    threshold: float,
    mode: str,
    do_plot: bool,
) -> pd.DataFrame:
    """Process all ip_cov_* dirs for one workflow and optionally plot."""
    print(f"\n{name}: {results_root}")
    ip_dirs = discover_ip_cov_dirs(results_root)
    if not ip_dirs:
        print(f"  No ip_cov_* directories found.")
        return pd.DataFrame()

    rows = []
    for ip_target, exp_dir in ip_dirs:
        r = process_wfinstances_experiment(exp_dir, threshold, mode)
        if r:
            r["workflow"] = name
            rows.append(r)
            print(f"  {exp_dir.name}: F1={r['cover_f1']:.3f} SHD={r['shd']}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{k: v for k, v in x.items() if k not in ("true_cover", "inferred_cover", "exp_dir")} for x in rows])

    if do_plot:
        plots_dir = results_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        suf = f"_t{threshold:.2f}" if mode == "single_po" else "_marginal"
        plot_comparison_pages(
            name,
            results_root,
            rows,
            plots_dir / f"comparison{suf}.pdf",
        )
        plot_f1_vs_ip_cov(
            name,
            df,
            plots_dir / f"f1_shd_vs_ipcov{suf}.pdf",
            exts=[".pdf", ".png"],
        )

    return df


def main():
    ap = argparse.ArgumentParser(description="Recompute H and plot WfInstances results")
    ap.add_argument("--experiment", choices=["srasearch", "epigenomics", "all"], default="all",
                    help="Which workflow(s) to process")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for single_po mode")
    ap.add_argument("--mode", choices=["single_po", "marginal"], default="single_po",
                    help="single_po (threshold on avg_H) or marginal (mode of H_trace)")
    ap.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    ap.add_argument("--output", type=str, default=None,
                    help="Output CSV path (default: results/wfinstances/wfinstances_recompute_summary.csv)")
    args = ap.parse_args()

    experiments = ["srasearch", "epigenomics"] if args.experiment == "all" else [args.experiment]
    do_plot = not args.no_plot

    print("=" * 60)
    print("Recompute H (WfInstances) + Plot")
    print("=" * 60)
    print(f"  mode={args.mode}  threshold={args.threshold}  plot={do_plot}")

    dfs = []
    for name in experiments:
        root = RESULTS_BY_EXPERIMENT.get(name)
        if not root or not root.exists():
            print(f"\n{name}: results dir not found ({root})")
            continue
        df = run_experiment(name, root, args.threshold, args.mode, do_plot)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("\nNo results generated.")
        return

    out_df = pd.concat(dfs, ignore_index=True)
    default_out_path = WFINSTANCES_RESULTS_ROOT / "wfinstances_recompute_summary.csv"
    out_path = Path(args.output) if args.output else default_out_path
    out_path = out_path if out_path.is_absolute() else (PROJECT_ROOT / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSummary CSV: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
