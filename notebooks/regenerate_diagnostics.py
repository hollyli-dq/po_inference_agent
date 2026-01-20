#!/usr/bin/env python3
"""
Regenerate diagnostics PDFs for all completed experiments with correct burn-in (0.5 proportion).
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project setup
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "notebooks/systematic_experiment_results"
BURN_IN_FRACTION = 0.5


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


def regenerate_experiment_diagnostics(exp_dir: Path):
    """Regenerate diagnostics PDF for a single experiment."""
    
    # Load likelihood trace
    ll_path = exp_dir / "likelihood_trace.npy"
    param_path = exp_dir / "param_traces.pkl"
    
    if not ll_path.exists() and not param_path.exists():
        print(f"  Skipping {exp_dir.name}: no data files found")
        return False
    
    diag_pdf_path = exp_dir / "diagnostics.pdf"
    
    with PdfPages(str(diag_pdf_path)) as pdf:
        # Plot log-likelihood trace
        if ll_path.exists():
            ll = np.load(ll_path)
            burn = int(len(ll) * BURN_IN_FRACTION)
            
            setup_icml_style()
            fig = plt.figure(figsize=(7.0, 2.8))
            
            plt.plot(ll, lw=1.2, color='#377eb8', alpha=0.8)
            plt.axvline(burn, color="#e41a1c", ls="--", lw=1.2, alpha=0.8,
                       label=f'Burn-in ({burn:,} = 50%)')
            plt.xlabel("Iteration")
            plt.ylabel("Log-likelihood")
            plt.title("Log-likelihood Trace")
            plt.grid(True, alpha=0.3)
            plt.legend(framealpha=0.8, fontsize=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Plot parameter traces
        if param_path.exists():
            with open(param_path, 'rb') as f:
                param_traces = pickle.load(f)
            
            for key, arr in param_traces.items():
                if arr is None or len(arr) == 0:
                    continue
                
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 1:
                    continue
                
                burn = int(len(arr) * BURN_IN_FRACTION)
                post = arr[burn:]
                
                if len(post) == 0:
                    continue
                
                setup_icml_style()
                fig = plt.figure(figsize=(7.0, 2.8))
                
                # Trace plot
                ax1 = fig.add_subplot(1, 2, 1)
                iterations = np.arange(len(arr))
                ax1.plot(iterations, arr, lw=1.2, color='#377eb8', alpha=0.8)
                ax1.axvline(burn, color="#e41a1c", ls="--", lw=1.2, alpha=0.8,
                           label=f'Burn-in (50%)')
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel(key.replace('_trace', ''))
                ax1.grid(True, alpha=0.3)
                ax1.set_title("Trace")
                ax1.legend(framealpha=0.8, fontsize=7)
                
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
                ax2.set_xlabel(key.replace('_trace', ''))
                ax2.set_ylabel("Frequency")
                ax2.grid(True, alpha=0.3)
                ax2.set_title("Posterior (post burn-in)")
                ax2.legend(framealpha=0.8, fontsize=7)
                
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    
    print(f"  ✓ Regenerated {diag_pdf_path.name}")
    return True


def main():
    print("=" * 60)
    print("Regenerating Diagnostics PDFs with 50% Burn-in")
    print("=" * 60)
    
    # Find all experiment directories
    exp_dirs = sorted(RESULTS_DIR.glob("exp_*"))
    print(f"Found {len(exp_dirs)} experiment directories\n")
    
    success = 0
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        print(f"Processing {exp_dir.name}...")
        if regenerate_experiment_diagnostics(exp_dir):
            success += 1
    
    print(f"\n✅ Regenerated {success}/{len(exp_dirs)} diagnostics PDFs")
    print(f"Output: {RESULTS_DIR}/exp_*/diagnostics.pdf")


if __name__ == "__main__":
    main()

