#!/usr/bin/env python3
"""
Post-processing script for WfInstances experiment results.

Generates comprehensive diagnostic plots including:
1. H matrix posterior mean heatmap
2. MCMC traces for all parameters (rho, K, noise, softmax_beta, etc.)
3. Posterior histograms
4. Ground truth vs inferred comparison
5. Edge probability evolution

Usage:
    python src/analysis/analyze_wfinstances_results.py --experiment srasearch
    python src/analysis/analyze_wfinstances_results.py --experiment epigenomics
"""

from __future__ import annotations

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.po_fun import BasicUtils
from src.utils.result_paths import (
    WFINSTANCES_SRASEARCH_RESULTS_DIR,
    LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
    WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    prefer_existing_path,
)


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })


def plot_h_matrix_evolution(
    H_trace: List[np.ndarray],
    burn_in_frac: float = 0.5,
    true_cover: Optional[np.ndarray] = None,
    pdf: Optional[PdfPages] = None,
    title_prefix: str = "",
):
    """Plot H matrix posterior mean and evolution."""
    if not H_trace:
        print("  Warning: Empty H_trace")
        return
    
    burn = int(len(H_trace) * burn_in_frac)
    post_H = H_trace[burn:]
    
    if not post_H:
        print("  Warning: No samples after burn-in")
        return
    
    # Compute posterior mean
    H_mean = np.mean(post_H, axis=0)
    n = H_mean.shape[0]
    
    # Figure 1: H matrix posterior mean
    fig, axes = plt.subplots(1, 3 if true_cover is not None else 2, figsize=(14, 5))
    fig.suptitle(f"{title_prefix}Posterior H Matrix Analysis", fontsize=12)
    
    # Posterior mean
    im0 = axes[0].imshow(H_mean, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f"Posterior Mean P(i→j)\n(n={n}, samples={len(post_H)})")
    axes[0].set_xlabel("j (child)")
    axes[0].set_ylabel("i (parent)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Thresholded (0.5)
    H_thresh = (H_mean > 0.5).astype(float)
    im1 = axes[1].imshow(H_thresh, cmap='Oranges', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"Thresholded (>0.5)\n({int(H_thresh.sum())} edges)")
    axes[1].set_xlabel("j (child)")
    axes[1].set_ylabel("i (parent)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Ground truth comparison
    if true_cover is not None:
        diff = H_thresh - true_cover.astype(float)
        im2 = axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        fp = int((diff > 0).sum())
        fn = int((diff < 0).sum())
        axes[2].set_title(f"Difference (Inferred - Truth)\nFP={fp}, FN={fn}")
        axes[2].set_xlabel("j (child)")
        axes[2].set_ylabel("i (parent)")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    
    # Figure 2: Edge probability histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Get off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    probs = H_mean[mask].flatten()
    
    ax.hist(probs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.set_xlabel("Edge Probability P(i→j)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}Edge Probability Distribution")
    ax.legend()
    
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_parameter_traces(
    mcmc_results: Dict[str, Any],
    burn_in_frac: float = 0.5,
    pdf: Optional[PdfPages] = None,
    title_prefix: str = "",
):
    """Plot MCMC traces and posteriors for all parameters."""
    
    # Parameter keys to plot
    param_keys = [
        ('rho_trace', 'ρ (correlation)'),
        ('K_trace', 'K (latent dimensions)'),
        ('prob_noise_trace', 'Noise probability'),
        ('softmax_beta_trace', 'Softmax β'),
        ('epsilon_trace', 'ε (trembling hand)'),
        ('U_trace', 'U (latent positions)'),
    ]
    
    # Filter to available keys
    available = [(k, label) for k, label in param_keys if k in mcmc_results and mcmc_results[k]]
    
    if not available:
        print("  Warning: No parameter traces found")
        return
    
    # Create figure with subplots
    n_params = len(available)
    fig = plt.figure(figsize=(14, 3 * n_params))
    gs = gridspec.GridSpec(n_params, 2, width_ratios=[2, 1])
    
    fig.suptitle(f"{title_prefix}MCMC Parameter Traces and Posteriors", fontsize=12, y=1.02)
    
    for idx, (key, label) in enumerate(available):
        trace = np.array(mcmc_results[key])
        
        # Handle multi-dimensional traces (e.g., U_trace)
        if trace.ndim > 1:
            # For U_trace, just show the first component or summary
            if key == 'U_trace':
                # Show mean of absolute values as a summary
                trace = np.mean(np.abs(trace.reshape(len(trace), -1)), axis=1)
                label = 'U (mean |latent positions|)'
        
        burn = int(len(trace) * burn_in_frac)
        
        # Trace plot
        ax_trace = fig.add_subplot(gs[idx, 0])
        ax_trace.plot(trace, lw=0.5, alpha=0.7, color='steelblue')
        ax_trace.axvline(burn, color='red', linestyle='--', lw=1.5, label=f'Burn-in ({burn:,})')
        ax_trace.set_xlabel('Iteration')
        ax_trace.set_ylabel(label)
        ax_trace.set_title(f'{label} - Trace')
        ax_trace.legend(loc='upper right', fontsize=8)
        ax_trace.grid(True, alpha=0.3)
        
        # Posterior histogram
        ax_hist = fig.add_subplot(gs[idx, 1])
        post_samples = trace[burn:]
        if len(post_samples) > 0:
            ax_hist.hist(post_samples, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax_hist.axvline(np.mean(post_samples), color='red', linestyle='-', lw=2, 
                          label=f'Mean: {np.mean(post_samples):.3f}')
            ax_hist.axvline(np.median(post_samples), color='green', linestyle='--', lw=2,
                          label=f'Median: {np.median(post_samples):.3f}')
        ax_hist.set_xlabel(label)
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'{label} - Posterior')
        ax_hist.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_log_likelihood_trace(
    mcmc_results: Dict[str, Any],
    burn_in_frac: float = 0.5,
    pdf: Optional[PdfPages] = None,
    title_prefix: str = "",
):
    """Plot log-likelihood trace."""
    ll = mcmc_results.get('log_likelihood_currents', [])
    if not ll:
        print("  Warning: No log-likelihood trace found")
        return
    
    ll = np.array(ll)
    burn = int(len(ll) * burn_in_frac)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{title_prefix}Log-Likelihood Analysis", fontsize=12)
    
    # Full trace
    axes[0].plot(ll, lw=0.5, alpha=0.7, color='steelblue')
    axes[0].axvline(burn, color='red', linestyle='--', lw=1.5, label=f'Burn-in ({burn:,})')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Log-Likelihood')
    axes[0].set_title('Full Trace')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Post burn-in
    post_ll = ll[burn:]
    axes[1].plot(post_ll, lw=0.5, alpha=0.7, color='steelblue')
    axes[1].axhline(np.mean(post_ll), color='red', linestyle='-', lw=1.5,
                   label=f'Mean: {np.mean(post_ll):.1f}')
    axes[1].set_xlabel('Iteration (post burn-in)')
    axes[1].set_ylabel('Log-Likelihood')
    axes[1].set_title(f'Post Burn-in (n={len(post_ll):,})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_acceptance_rates(
    mcmc_results: Dict[str, Any],
    pdf: Optional[PdfPages] = None,
    title_prefix: str = "",
):
    """Plot acceptance rate information."""
    overall_rate = mcmc_results.get('overall_acceptance_rate', 0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Try to get detailed acceptance info
    accept_keys = [k for k in mcmc_results.keys() if 'accept' in k.lower()]
    
    if accept_keys:
        rates = {k: mcmc_results[k] for k in accept_keys if isinstance(mcmc_results[k], (int, float))}
        if rates:
            names = list(rates.keys())
            values = [rates[k] * 100 if rates[k] <= 1 else rates[k] for k in names]
            
            bars = ax.barh(names, values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Acceptance Rate (%)')
            ax.set_title(f'{title_prefix}MCMC Acceptance Rates')
            ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='Target: 20%')
            ax.legend()
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=9)
    else:
        ax.bar(['Overall'], [overall_rate * 100], color='steelblue', alpha=0.7)
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title(f'{title_prefix}Overall Acceptance Rate: {overall_rate*100:.1f}%')
    
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)


def generate_comprehensive_diagnostics(
    mcmc_results: Dict[str, Any],
    output_path: Path,
    true_cover: Optional[np.ndarray] = None,
    title_prefix: str = "",
    burn_in_frac: float = 0.5,
):
    """Generate all diagnostic plots in a single PDF."""
    setup_plot_style()
    
    with PdfPages(str(output_path)) as pdf:
        print(f"  Generating diagnostics: {output_path}")
        
        # 1. Log-likelihood trace
        print("    - Log-likelihood trace")
        plot_log_likelihood_trace(mcmc_results, burn_in_frac, pdf, title_prefix)
        
        # 2. Parameter traces
        print("    - Parameter traces")
        plot_parameter_traces(mcmc_results, burn_in_frac, pdf, title_prefix)
        
        # 3. H matrix analysis
        H_trace = mcmc_results.get('H_trace', [])
        if H_trace:
            print("    - H matrix analysis")
            plot_h_matrix_evolution(H_trace, burn_in_frac, true_cover, pdf, title_prefix)
        
        # 4. Acceptance rates
        print("    - Acceptance rates")
        plot_acceptance_rates(mcmc_results, pdf, title_prefix)
    
    print(f"  ✓ Saved: {output_path}")


def load_experiment_results(experiment: str) -> Dict[str, Any]:
    """Load experiment results from disk."""
    if experiment == 'srasearch':
        base_dir = prefer_existing_path(
            WFINSTANCES_SRASEARCH_RESULTS_DIR,
            LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
        )
        data_dir = PROJECT_ROOT / 'data' / 'wfinstances_srasearch'
    elif experiment == 'epigenomics':
        base_dir = prefer_existing_path(
            WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
            LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
        )
        data_dir = PROJECT_ROOT / 'data' / 'wfinstances_epigenomics'
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    # Load summary
    summary_path = base_dir / 'experiment_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze WfInstances experiment results")
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['srasearch', 'epigenomics'],
                       help='Which experiment to analyze')
    parser.add_argument('--burn_in_frac', type=float, default=0.5,
                       help='Burn-in fraction (default: 0.5)')
    
    args = parser.parse_args()
    
    print(f"=" * 70)
    print(f"Analyzing {args.experiment} experiment results")
    print(f"=" * 70)
    
    try:
        data = load_experiment_results(args.experiment)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the experiment has completed first.")
        return
    
    summary = data['summary']
    base_dir = data['base_dir']
    
    print(f"\nWorkflow: {summary['workflow']['name']}")
    print(f"Tasks: {summary['workflow']['num_tasks']}")
    print(f"Edges: {summary['workflow']['num_edges']}")
    print(f"Incomparable pairs: {summary['workflow'].get('num_incomparable_pairs', 'N/A')}")
    print(f"\nResults found: {len(summary['results'])} IP-Cov levels")
    
    # Check for pickle files with full MCMC results
    for result in summary['results']:
        ip_cov = result['ip_cov_realized']
        ip_str = f"{result['ip_cov_target']:.2f}".replace('.', '_')
        exp_dir = base_dir / f"ip_cov_{ip_str}"
        
        print(f"\n--- IP-Cov = {ip_cov:.2f} ---")
        print(f"  BHPOP F1: {result['bhpop_f1']:.3f}")
        print(f"  AND F1: {result['and_f1']:.3f}")
        print(f"  Majority F1: {result['majority_f1']:.3f}")
        
        # Check for existing diagnostics
        diag_path = exp_dir / 'diagnostics.pdf'
        if diag_path.exists():
            print(f"  Diagnostics: {diag_path}")
        
        # Check for pickle file
        pickle_path = exp_dir / 'mcmc_results.pkl'
        if pickle_path.exists():
            print(f"  MCMC results pickle found - generating comprehensive diagnostics...")
            with open(pickle_path, 'rb') as f:
                mcmc_results = pickle.load(f)
            
            output_path = exp_dir / 'comprehensive_diagnostics.pdf'
            generate_comprehensive_diagnostics(
                mcmc_results,
                output_path,
                title_prefix=f"IP-Cov={ip_cov:.2f} | ",
                burn_in_frac=args.burn_in_frac,
            )
        else:
            print(f"  No MCMC pickle found at {pickle_path}")
            print(f"  (Run experiment with --save_mcmc_results to enable detailed analysis)")


if __name__ == "__main__":
    main()
