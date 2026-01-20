#!/usr/bin/env python3
"""
Generate ICML-formatted plots from experiment_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# =========================
# ICML-style configuration
# =========================
def setup_icml_style():
    """Configure matplotlib for ICML-style academic paper plots."""
    plt.rcParams.update({
        # Font settings - use available fonts
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,

        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Legend settings
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '0.8',
        'legend.frameon': True,

        # Tick settings
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',

        # No LaTeX
        'text.usetex': False,
    })

# Color palette - colorblind friendly (BHPOP gets prominent blue)
COLORS = {
    'bhpop_marginal_mode': '#377eb8',  # Blue (our method - prominent)
    'majority': '#4daf4a',      # Green
    'inductive_miner_imf': '#984ea3',  # Purple
    'heuristics_miner': '#ff7f00',     # Orange
}

METHOD_LABELS = {
    # 'and': 'AND (Intersection)',  # Removed
    'majority': 'Majority',
    'inductive_miner_imf': 'Inductive Miner',
    'heuristics_miner': 'Heuristics Miner',
    'bhpop_marginal_mode': 'BHPOP (Ours)',
}

MARKERS = {
    'bhpop_marginal_mode': '*',  # Star for our method
    'majority': 's',
    'inductive_miner_imf': '^',
    'heuristics_miner': 'D',
}

SCENARIO_LABELS = {
    'simple_ecs': 'S1: simple_ecs',
    'slb_ecs_rds': 'S2: slb_ecs_rds',
    'slb_ecs_redis': 'S3: slb_ecs_redis',
    'eip_slb_ecs': 'S4: eip_slb_ecs',
    'dual_zone_ecs_slb': 'S5: dual_zone_ecs_slb',
    'dual_zone_ecs_slb_rds': 'S6: dual_zone_ecs_slb_rds',
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the experiment data."""
    df = pd.read_csv(csv_path)
    return df


def plot_f1_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average F1 score by method across all scenarios."""
    # Aggregate by method
    method_avg = df.groupby('method')['cover_f1'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=True)
    
    methods = method_avg['method'].tolist()
    means = method_avg['mean'].tolist()
    stds = method_avg['std'].tolist()
    
    colors = [COLORS.get(m, '#999999') for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, 
                   edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Cover F1 Score')
    ax.set_title('Average F1 Score by Method')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.02, i, f'{mean:.3f}', va='center', fontsize=7)


def plot_f1_vs_cpcov(df: pd.DataFrame, ax: plt.Axes, eps_value: float = 0.01):
    """Line plot: F1 score vs CP-Cov target for each method."""
    # Filter to specific epsilon
    df_eps = df[df['eps_jump'] == eps_value].copy()
    
    # Aggregate by method and cp_cov_target
    agg = df_eps.groupby(['method', 'cp_cov_target'])['cover_f1'].mean().reset_index()
    
    for method in df_eps['method'].unique():
        method_data = agg[agg['method'] == method].sort_values('cp_cov_target')
        if len(method_data) > 0:
            ax.plot(method_data['cp_cov_target'], method_data['cover_f1'],
                   marker=MARKERS.get(method, 'o'),
                   color=COLORS.get(method, '#999999'),
                   label=METHOD_LABELS.get(method, method),
                   markersize=6, linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('CP-Cov Target')
    ax.set_ylabel('Cover F1 Score')
    ax.set_title(f'F1 vs CP-Coverage (eps={eps_value})')
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, ncol=1)


def plot_f1_by_scenario(df: pd.DataFrame, ax: plt.Axes):
    """Grouped bar chart: F1 score by scenario and method."""
    # Use only cp_cov_target=1.0 and eps=0.01 for clean comparison
    df_filt = df[(df['cp_cov_target'] == 1.0) & (df['eps_jump'] == 0.01)].copy()
    
    scenarios = sorted(df_filt['scenario'].unique())
    methods = ['majority', 'inductive_miner_imf', 'heuristics_miner', 'bhpop_marginal_mode']
    methods = [m for m in methods if m in df_filt['method'].unique()]
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    for i, method in enumerate(methods):
        method_data = df_filt[df_filt['method'] == method]
        f1_scores = [method_data[method_data['scenario'] == s]['cover_f1'].values[0] 
                     if len(method_data[method_data['scenario'] == s]) > 0 else 0 
                     for s in scenarios]
        
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, 
               label=METHOD_LABELS.get(method, method),
               color=COLORS.get(method, '#999999'),
               edgecolor='black', linewidth=0.3, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=7)
    ax.set_ylabel('Cover F1 Score')
    ax.set_title('F1 Score by Scenario (CP-Cov=1.0, eps=0.01)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)


def plot_shd_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average SHD (lower is better) by method."""
    method_avg = df.groupby('method')['shd'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=False)  # Lower is better
    
    methods = method_avg['method'].tolist()
    means = method_avg['mean'].tolist()
    stds = method_avg['std'].tolist()
    
    colors = [COLORS.get(m, '#999999') for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, means, xerr=stds, color=colors,
            edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Structural Hamming Distance (lower is better)')
    ax.set_title('Average SHD by Method')
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.5, i, f'{mean:.1f}', va='center', fontsize=7)


def plot_feasibility_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average feasibility by method."""
    method_avg = df.groupby('method')['feas'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=True)
    
    methods = method_avg['method'].tolist()
    means = method_avg['mean'].tolist()
    stds = method_avg['std'].tolist()
    
    colors = [COLORS.get(m, '#999999') for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, means, xerr=stds, color=colors,
            edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Feasibility')
    ax.set_title('Average Feasibility by Method')
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(min(mean + std + 0.02, 1.05), i, f'{mean:.3f}', va='center', fontsize=7)


def plot_heatmap_f1(df: pd.DataFrame, ax: plt.Axes, method: str = 'bhpop_marginal_mode'):
    """Heatmap: F1 score by scenario and CP-Cov target for a specific method."""
    df_method = df[(df['method'] == method) & (df['eps_jump'] == 0.01)].copy()
    
    # Pivot to create heatmap data
    pivot = df_method.pivot_table(values='cover_f1', 
                                   index='scenario', 
                                   columns='cp_cov_target',
                                   aggfunc='mean')
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{c:.2f}' for c in pivot.columns], fontsize=7)
    ax.set_yticklabels([s.replace('_', '\n') for s in pivot.index], fontsize=7)
    
    ax.set_xlabel('CP-Cov Target')
    ax.set_ylabel('Scenario')
    ax.set_title(f'F1 Heatmap: {METHOD_LABELS.get(method, method)}')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=6, color=color)
    
    return im


def plot_eps_comparison(df: pd.DataFrame, ax: plt.Axes):
    """Line plot: F1 vs epsilon for BHPOP method."""
    bhpop_data = df[df['method'] == 'bhpop_marginal_mode'].copy()
    
    if len(bhpop_data) == 0:
        ax.text(0.5, 0.5, 'No BHPOP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Aggregate by eps and cp_cov_target
    for cpcov in sorted(bhpop_data['cp_cov_target'].unique()):
        cpcov_data = bhpop_data[bhpop_data['cp_cov_target'] == cpcov]
        eps_agg = cpcov_data.groupby('eps_jump')['cover_f1'].mean().reset_index()
        eps_agg = eps_agg.sort_values('eps_jump')
        
        ax.plot(eps_agg['eps_jump'], eps_agg['cover_f1'],
               marker='o', label=f'CP-Cov={cpcov}', markersize=5, linewidth=1.2)
    
    ax.set_xlabel('Epsilon (eps_jump)')
    ax.set_ylabel('Cover F1 Score')
    ax.set_title('BHPOP: F1 vs Epsilon by CP-Cov Target')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=6, ncol=2)


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table of results."""
    summary = df.groupby('method').agg({
        'cover_f1': ['mean', 'std', 'max'],
        'shd': ['mean', 'std', 'min'],
        'feas': ['mean', 'std']
    }).round(3)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary['method'] = summary['method'].map(lambda x: METHOD_LABELS.get(x, x))
    
    return summary


def main():
    setup_icml_style()
    
    # Load data
    csv_path = Path('systematic_experiment_results/experiment_summary.csv')
    df = load_data(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Scenarios: {df['scenario'].unique()}")
    print(f"CP-Cov targets: {sorted(df['cp_cov_target'].unique())}")
    
    # Create output directory
    output_dir = Path('systematic_experiment_results/plots')
    output_dir.mkdir(exist_ok=True)
    
    # =========================
    # Figure 1: Overview (2x2)
    # =========================
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    
    plot_f1_by_method(df, axes1[0, 0])
    plot_shd_by_method(df, axes1[0, 1])
    plot_feasibility_by_method(df, axes1[1, 0])
    plot_f1_vs_cpcov(df, axes1[1, 1], eps_value=0.01)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / 'overview.pdf')
    fig1.savefig(output_dir / 'overview.png', dpi=300)
    print(f"Saved: {output_dir / 'overview.pdf'}")
    plt.close(fig1)
    
    # =========================
    # Figure 2: F1 by Scenario
    # =========================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_f1_by_scenario(df, ax2)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'f1_by_scenario.pdf')
    fig2.savefig(output_dir / 'f1_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_by_scenario.pdf'}")
    plt.close(fig2)
    
    # =========================
    # Figure 3: Heatmaps for Majority and BHPOP
    # =========================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    
    im1 = plot_heatmap_f1(df, axes3[0], method='majority')
    im2 = plot_heatmap_f1(df, axes3[1], method='bhpop_marginal_mode')
    
    # Add colorbar
    fig3.colorbar(im2, ax=axes3, shrink=0.8, label='F1 Score')
    
    fig3.tight_layout()
    fig3.savefig(output_dir / 'f1_heatmaps.pdf')
    fig3.savefig(output_dir / 'f1_heatmaps.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_heatmaps.pdf'}")
    plt.close(fig3)
    
    # =========================
    # Figure 4: BHPOP epsilon analysis
    # =========================
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plot_eps_comparison(df, ax4)
    fig4.tight_layout()
    fig4.savefig(output_dir / 'bhpop_epsilon_analysis.pdf')
    fig4.savefig(output_dir / 'bhpop_epsilon_analysis.png', dpi=300)
    print(f"Saved: {output_dir / 'bhpop_epsilon_analysis.pdf'}")
    plt.close(fig4)
    
    # =========================
    # Figure 5: F1 vs CP-Cov for different epsilon values
    # =========================
    # Plot F1 vs CP-Cov for different epsilon values (2x3 grid for 5 epsilon values)
    eps_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    fig5, axes5 = plt.subplots(2, 3, figsize=(12, 8))
    
    for ax, eps in zip(axes5.flat[:len(eps_values)], eps_values):
        plot_f1_vs_cpcov(df, ax, eps_value=eps)
    
    # Hide the unused subplot (6th position)
    if len(eps_values) < 6:
        axes5.flat[5].set_visible(False)
    
    fig5.tight_layout()
    fig5.savefig(output_dir / 'f1_vs_cpcov_by_eps.pdf')
    fig5.savefig(output_dir / 'f1_vs_cpcov_by_eps.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_vs_cpcov_by_eps.pdf'}")
    plt.close(fig5)
    
    # =========================
    # Summary statistics
    # =========================
    summary = create_summary_table(df)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary.to_string(index=False))
    
    summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"\nSaved: {output_dir / 'summary_statistics.csv'}")
    
    # Print best method by scenario
    print("\n" + "="*60)
    print("BEST METHOD BY SCENARIO (CP-Cov=1.0, eps=0.01)")
    print("="*60)
    df_best = df[(df['cp_cov_target'] == 1.0) & (df['eps_jump'] == 0.01)]
    for scenario in sorted(df_best['scenario'].unique()):
        scenario_df = df_best[df_best['scenario'] == scenario]
        best_row = scenario_df.loc[scenario_df['cover_f1'].idxmax()]
        print(f"{scenario}: {METHOD_LABELS.get(best_row['method'], best_row['method'])} "
              f"(F1={best_row['cover_f1']:.3f}, SHD={best_row['shd']}, feas={best_row['feas']:.3f})")
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

