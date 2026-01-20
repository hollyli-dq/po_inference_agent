#!/usr/bin/env python3
"""
VLC Systematic Inference - Using the same pipeline as systematic_experiments.py

This script applies the BHPOP inference pipeline to VLC trace data,
using the same MCMC settings, posterior aggregation, and diagnostics as the
Aliyun systematic experiments.
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Project root
# -------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- BHPOP imports ---
from src.utils.po_fun import BasicUtils
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_hpo_k_optim

# =========================
# Configuration (matching systematic_experiments.py)
# =========================
DEFAULT_NUM_ITERATIONS = 100_000  # Reduced for VLC (can increase)
BURN_IN_FRACTION = 0.5
THIN = 1
SEED_BASE = 42

# Noise options to test
NOISE_OPTIONS = ["log_successors_queue_jump"]

# =========================
# ICML-style plotting
# =========================
def setup_icml_style():
    """Configure matplotlib for ICML-style academic paper plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (3.25, 2.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.prop_cycle': plt.cycler(color=[
            '#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
            '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'
        ]),
        'legend.framealpha': 0.8,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.frameon': True,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'text.usetex': False,
    })


# =========================
# Posterior aggregation (from systematic_experiments.py)
# =========================
def posterior_threshold_mean(matrices: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Simple threshold aggregation:
    1. Compute mean of posterior samples
    2. Threshold at specified value
    3. Apply transitive reduction
    """
    if not matrices:
        raise ValueError("posterior_threshold_mean called with empty list")
    
    mats_arr = np.stack(matrices, axis=0)
    mean_mat = np.mean(mats_arr, axis=0)
    
    agg_mat = (mean_mat >= threshold).astype(np.int8)
    np.fill_diagonal(agg_mat, 0)
    
    cover = BasicUtils.transitive_reduction(agg_mat.astype(int))
    return cover.astype(np.int8)


# =========================
# Feasibility evaluation
# =========================
def is_linear_extension_partial(order: List[int], closure: np.ndarray) -> bool:
    """Check if order is consistent with closure."""
    pos = {t: i for i, t in enumerate(order)}
    rows, cols = np.where(closure == 1)
    for i, j in zip(rows, cols):
        if i in pos and j in pos and pos[i] >= pos[j]:
            return False
    return True


def feasibility_of_orders(orders: List[List[int]], inferred_cover: np.ndarray) -> float:
    """Fraction of orders that are linear extensions of inferred_cover."""
    if not orders:
        return float("nan")
    inferred_closure = BasicUtils.transitive_closure(inferred_cover.astype(np.int8))
    invalid = sum(1 for o in orders if not is_linear_extension_partial(o, inferred_closure))
    return 1.0 - invalid / max(1, len(orders))


# =========================
# Diagnostics PDF generation
# =========================
def save_diagnostics_pdf(
    mcmc_results: dict,
    output_dir: Path,
    task_id: str,
    *,
    burn_in_frac: float = 0.5,
) -> None:
    """Save comprehensive diagnostics PDF (matching systematic_experiments.py style)."""
    
    pdf_path = output_dir / f"{task_id}_diagnostics.pdf"
    
    with PdfPages(str(pdf_path)) as pdf:
        setup_icml_style()
        
        # 1. Log-likelihood trace
        ll = mcmc_results.get("log_likelihood_currents", [])
        if ll:
            ll = np.asarray(ll, dtype=float)
            burn = int(len(ll) * burn_in_frac)
            
            fig = plt.figure(figsize=(7.0, 2.8))
            plt.plot(ll, lw=1.2, color='#377eb8', alpha=0.8)
            if burn > 0:
                plt.axvline(burn, color="#e41a1c", ls="--", lw=1.2, alpha=0.8,
                           label=f'Burn-in ({burn:,})')
            plt.xlabel("Iteration")
            plt.ylabel("Log-likelihood")
            plt.title("Log-likelihood Trace")
            plt.grid(True, alpha=0.3)
            plt.legend(framealpha=0.8, fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # 2. Parameter traces
        param_keys = [
            "rho_trace", "tau_trace", "K_trace", 
            "prob_noise_trace", "softmax_beta_trace", "softmax_lambda_trace"
        ]
        
        for key in param_keys:
            arr = mcmc_results.get(key)
            if arr is None:
                continue
            
            try:
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 1 or arr.size == 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            burn = int(len(arr) * burn_in_frac)
            post = arr[burn:]
            
            if len(post) == 0:
                continue
            
            fig = plt.figure(figsize=(7.0, 2.8))
            
            # Trace plot
            ax1 = fig.add_subplot(1, 2, 1)
            iterations = np.arange(burn, burn + len(post))
            ax1.plot(iterations, post, lw=1.2, color='#377eb8', alpha=0.8)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel(key.replace("_trace", ""))
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Trace")
            
            # Histogram
            ax2 = fig.add_subplot(1, 2, 2)
            if "K" in key or np.allclose(post, np.round(post)):
                vals = np.round(post).astype(int)
                mn, mx = int(vals.min()), int(vals.max())
                bins = np.arange(mn - 0.5, mx + 1.5, 1.0)
                ax2.hist(vals, bins=bins, edgecolor="black", linewidth=0.5,
                        alpha=0.7, color='#4daf4a')
                ax2.set_xticks(np.arange(mn, mx + 1))
            else:
                ax2.hist(post, bins=25, edgecolor="black", linewidth=0.4,
                        alpha=0.7, color='#4daf4a')
            
            mean_val = np.mean(post)
            ax2.axvline(mean_val, ls='--', lw=1.2, color='#e41a1c',
                       label=f'Mean: {mean_val:.3f}')
            ax2.set_xlabel(key.replace("_trace", ""))
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)
            ax2.set_title("Posterior")
            ax2.legend(framealpha=0.8, fontsize=8)
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # 3. Acceptance rate trace
        acc_rates = mcmc_results.get("acceptance_rates", [])
        if acc_rates:
            acc_rates = np.asarray(acc_rates, dtype=float)
            
            fig = plt.figure(figsize=(7.0, 2.8))
            plt.plot(acc_rates * 100, lw=1.2, color='#4daf4a', alpha=0.8)
            plt.axhline(mcmc_results.get("overall_acceptance_rate", 0) * 100, 
                       color="#e41a1c", ls="--", lw=1.2,
                       label=f'Final: {mcmc_results.get("overall_acceptance_rate", 0)*100:.1f}%')
            plt.xlabel("Iteration")
            plt.ylabel("Acceptance Rate (%)")
            plt.title("MCMC Acceptance Rate")
            plt.grid(True, alpha=0.3)
            plt.legend(framealpha=0.8, fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"💾 Saved diagnostics PDF: {pdf_path}")


# =========================
# Main inference function
# =========================
def run_vlc_inference(
    input_file: Path,
    task_id: str,
    output_dir: Path,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    noise_option: str = "log_successors_queue_jump",
    seed: int = SEED_BASE,
) -> Dict[str, Any]:
    """
    Run BHPOP inference on VLC data using the systematic_experiments.py pipeline.
    """
    
    # Load data
    with open(input_file) as f:
        all_data = json.load(f)
    
    if task_id not in all_data:
        raise ValueError(f"Task '{task_id}' not found. Available: {list(all_data.keys())}")
    
    task_data = all_data[task_id]
    
    # Extract HPO inputs
    M0 = task_data["M0"]
    assessors = task_data["assessors"]
    M_a_dict = {int(k): v for k, v in task_data["M_a_dict"].items()}
    observed_orders = {int(k): v for k, v in task_data["observed_orders"].items()}
    O_a_i_dict = {int(k): v for k, v in task_data.get("O_a_i_dict", task_data.get("choice_sets", {})).items()}
    
    # If O_a_i_dict is empty, construct from observed_orders
    if not O_a_i_dict:
        O_a_i_dict = {
            a: [sorted(set(order)) for order in orders]
            for a, orders in observed_orders.items()
        }
    
    # Get action names if available
    action_names = task_data.get("action_names", [f"action_{i}" for i in M0])
    
    print(f"\n{'='*60}")
    print(f"VLC SYSTEMATIC INFERENCE")
    print(f"{'='*60}")
    print(f"Task: {task_id}")
    print(f"Actions: {len(M0)}")
    print(f"Assessors: {assessors}")
    print(f"Traces per assessor: {[len(observed_orders[a]) for a in assessors]}")
    print(f"Iterations: {num_iterations:,}")
    print(f"Noise option: {noise_option}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run MCMC (same parameters as systematic_experiments.py)
    print("Running MCMC simulation...")
    mcmc_result = mcmc_simulation_hpo_k_optim(
        num_iterations=num_iterations,
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        dr=0.95,
        drrt=0.95,
        noise_option=noise_option,
        rho_prior=1.0,
        noise_beta_prior=1.0,
        K_prior=3,
        fixed_K=None,
        random_seed=seed,
        cycle_length=500,
        softmax_lambda=1.0,
        softmax_beta_prior=(2.0, 1.0),
        softmax_beta_stepsize=0.1,
    )
    
    print(f"\nMCMC completed!")
    print(f"Overall acceptance rate: {mcmc_result.get('overall_acceptance_rate', 0)*100:.2f}%")
    
    # Extract posterior samples
    H_trace = mcmc_result["H_trace"]
    burn = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn::THIN]
    
    print(f"Total samples: {len(H_trace)}")
    print(f"Post burn-in samples: {len(post)}")
    
    # Posterior aggregation for each assessor
    results = {}
    for assessor in assessors:
        mats_a = [it[assessor] for it in post if assessor in it]
        if not mats_a:
            print(f"Warning: No samples for assessor {assessor}")
            continue
        
        # Aggregate using threshold mean (same as systematic_experiments.py)
        inferred_cover = posterior_threshold_mean(mats_a, threshold=0.5)
        avg_mat = np.mean(np.stack(mats_a), axis=0)
        
        # Compute feasibility
        orders_a = observed_orders[assessor]
        feas = feasibility_of_orders(orders_a, inferred_cover)
        
        results[assessor] = {
            "inferred_cover": inferred_cover,
            "avg_posterior": avg_mat,
            "feasibility": feas,
            "num_traces": len(orders_a),
            "num_edges": int(inferred_cover.sum()),
        }
        
        print(f"\nAssessor {assessor}:")
        print(f"  Inferred edges: {int(inferred_cover.sum())}")
        print(f"  Feasibility: {feas:.3f}")
    
    # Save artifacts
    print(f"\nSaving artifacts to {output_dir}...")
    
    # Save H_trace
    with open(output_dir / f"{task_id}_H_trace.pkl", "wb") as f:
        pickle.dump(H_trace, f)
    
    # Save parameter traces
    param_traces = {}
    for key in ['rho_trace', 'tau_trace', 'K_trace', 'prob_noise_trace', 
                'softmax_beta_trace', 'softmax_lambda_trace']:
        if key in mcmc_result:
            param_traces[key] = mcmc_result[key]
    
    with open(output_dir / f"{task_id}_param_traces.pkl", "wb") as f:
        pickle.dump(param_traces, f)
    
    # Save inferred covers
    with open(output_dir / f"{task_id}_inferred_covers.pkl", "wb") as f:
        pickle.dump({a: r["inferred_cover"] for a, r in results.items()}, f)
    
    # Save summary JSON
    summary = {
        "task_id": task_id,
        "num_iterations": num_iterations,
        "burn_in_fraction": BURN_IN_FRACTION,
        "noise_option": noise_option,
        "seed": seed,
        "overall_acceptance_rate": mcmc_result.get("overall_acceptance_rate", 0),
        "num_actions": len(M0),
        "action_names": action_names,
        "assessors": assessors,
        "results": {
            str(a): {
                "feasibility": r["feasibility"],
                "num_traces": r["num_traces"],
                "num_edges": r["num_edges"],
            }
            for a, r in results.items()
        },
        "final_params": {
            "rho": float(mcmc_result.get("rho_final") or 0),
            "tau": float(mcmc_result.get("tau_final") or 0),
            "K": int(mcmc_result.get("K_final") or 0),
            "prob_noise": float(mcmc_result.get("prob_noise_final") or 0),
            "softmax_beta": float(mcmc_result.get("softmax_beta_final") or 0),
        },
    }
    
    with open(output_dir / f"{task_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate diagnostics PDF
    save_diagnostics_pdf(mcmc_result, output_dir, task_id, burn_in_frac=BURN_IN_FRACTION)
    
    # Save inferred partial order visualization
    save_partial_order_plot(results, action_names, output_dir, task_id)
    
    print(f"\n✅ Inference complete!")
    print(f"Results saved to: {output_dir}")
    
    return {
        "mcmc_result": mcmc_result,
        "results": results,
        "summary": summary,
    }


def save_partial_order_plot(
    results: Dict[int, Dict[str, Any]],
    action_names: List[str],
    output_dir: Path,
    task_id: str,
) -> None:
    """Save visualization of inferred partial order."""
    
    setup_icml_style()
    
    for assessor, data in results.items():
        cover = data["inferred_cover"]
        avg_mat = data["avg_posterior"]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Inferred cover (binary)
        ax1 = axes[0]
        im1 = ax1.imshow(cover, cmap='Blues', aspect='auto')
        ax1.set_title(f"Inferred Partial Order Cover\n(Assessor {assessor})")
        ax1.set_xlabel("Action Index")
        ax1.set_ylabel("Action Index")
        plt.colorbar(im1, ax=ax1, label="Edge")
        
        # Plot 2: Average posterior probability
        ax2 = axes[1]
        im2 = ax2.imshow(avg_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax2.set_title(f"Posterior Edge Probability\n(Assessor {assessor})")
        ax2.set_xlabel("Action Index")
        ax2.set_ylabel("Action Index")
        plt.colorbar(im2, ax=ax2, label="P(edge)")
        
        plt.tight_layout()
        
        plot_path = output_dir / f"{task_id}_assessor_{assessor}_po.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"💾 Saved PO plot: {plot_path}")
    
    # Also save action frequency plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for assessor, data in results.items():
        cover = data["inferred_cover"]
        out_degree = cover.sum(axis=1)
        in_degree = cover.sum(axis=0)
        
        x = np.arange(len(action_names))
        width = 0.35
        
        ax.bar(x - width/2, out_degree, width, label='Out-degree', alpha=0.7)
        ax.bar(x + width/2, in_degree, width, label='In-degree', alpha=0.7)
        
        ax.set_xlabel('Action')
        ax.set_ylabel('Degree')
        ax.set_title(f'Action Degrees in Inferred Partial Order')
        ax.set_xticks(x)
        ax.set_xticklabels([a[:20] for a in action_names], rotation=45, ha='right')
        ax.legend()
        
    plt.tight_layout()
    
    degree_path = output_dir / f"{task_id}_action_degrees.png"
    plt.savefig(degree_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"💾 Saved degree plot: {degree_path}")


# =========================
# Main entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLC Systematic Inference")
    parser.add_argument("--input_file", type=str,
                       default="./vlc_hpo_inputs.json",
                       help="Path to HPO inputs JSON")
    parser.add_argument("--task_id", type=str, default="vlc",
                       help="Task ID to analyze")
    parser.add_argument("--output_dir", type=str, default="./vlc_systematic_results",
                       help="Output directory for results")
    parser.add_argument("--num_iterations", type=int, default=DEFAULT_NUM_ITERATIONS,
                       help="Number of MCMC iterations")
    parser.add_argument("--noise_option", type=str, default="log_successors_queue_jump",
                       help="Noise model option")
    parser.add_argument("--seed", type=int, default=SEED_BASE,
                       help="Random seed")
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    run_vlc_inference(
        input_file=input_file,
        task_id=args.task_id,
        output_dir=output_dir,
        num_iterations=args.num_iterations,
        noise_option=args.noise_option,
        seed=args.seed,
    )
