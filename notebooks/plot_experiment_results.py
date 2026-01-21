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

# Color palette - colorblind friendly (BPOP gets prominent blue)
COLORS = {
    'bhpop_single_po': '#377eb8',  # Blue (our method - prominent)
    'majority': '#4daf4a',      # Green
    'inductive_miner_imf': '#984ea3',  # Purple
    'heuristics_miner': '#ff7f00',     # Orange
}

METHOD_LABELS = {
    # 'and': 'AND (Intersection)',  # Removed
    'majority': 'Majority',
    'inductive_miner_imf': 'Inductive Miner',
    'heuristics_miner': 'Heuristics Miner',
    'bhpop_single_po': 'BPOP (Ours)',
}

MARKERS = {
    'bhpop_single_po': '*',  # Star for our method
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
    # Exclude eip_slb_ecs scenario
    df = df[df['scenario'] != 'eip_slb_ecs'].copy()
    return df


def plot_f1_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average F1 score by method across all scenarios."""
    # Aggregate by method across all scenarios
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
    ax.set_xlabel('Edge F1 Score (Structural Recovery)')
    n_scenarios = len(df['scenario'].unique())
    ax.set_title(f'Average Edge F1 Score by Method\n(averaged across {n_scenarios} scenarios)')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.02, i, f'{mean:.3f}', va='center', fontsize=7)


def plot_f1_vs_ipcov(df: pd.DataFrame, ax: plt.Axes, eps_value: float = 0.01):
    """Line plot: Edge F1 score (structural recovery) vs IP-Cov target for each method (averaged across scenarios)."""
    # Filter both baseline methods and BPOP to specific epsilon for fair comparison
    
    baseline_methods = ['majority', 'inductive_miner_imf', 'heuristics_miner']
    
    # Plot baselines at specific eps
    df_eps = df[df['eps_jump'] == eps_value].copy()
    df_eps = df_eps[df_eps['ip_cov_target'] >= 0.6]  # Filter to IP-Cov >= 0.6
    agg_baselines = df_eps[df_eps['method'].isin(baseline_methods)].groupby(['method', 'ip_cov_target'])['cover_f1'].mean().reset_index()
    
    for method in baseline_methods:
        method_data = agg_baselines[agg_baselines['method'] == method].sort_values('ip_cov_target')
        if len(method_data) > 0:
            ax.plot(method_data['ip_cov_target'], method_data['cover_f1'],
                   marker=MARKERS.get(method, 'o'),
                   color=COLORS.get(method, '#999999'),
                   label=METHOD_LABELS.get(method, method),
                   markersize=6, linewidth=1.5, alpha=0.85)
    
    # Plot BPOP at the same epsilon as baselines
    bhpop_data = df_eps[df_eps['method'] == 'bhpop_single_po']
    if len(bhpop_data) > 0:
        agg_bhpop = bhpop_data.groupby('ip_cov_target')['cover_f1'].mean().reset_index()
        agg_bhpop = agg_bhpop.sort_values('ip_cov_target')
        ax.plot(agg_bhpop['ip_cov_target'], agg_bhpop['cover_f1'],
               marker=MARKERS.get('bhpop_single_po', 'o'),
               color=COLORS.get('bhpop_single_po', '#377eb8'),
               label=METHOD_LABELS.get('bhpop_single_po', 'BPOP'),
               markersize=6, linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('IP-Cov Target')
    ax.set_ylabel('Edge F1 Score (Structural Recovery)')
    ax.set_title(f'Edge F1 vs IP-Coverage (eps={eps_value})')
    ax.set_xlim(0.6, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, ncol=1)


def plot_ip_f1_vs_ipcov(df: pd.DataFrame, ax: plt.Axes, eps_value: float = 0.01):
    """Line plot: IP F1 score (concurrency recovery) vs IP-Cov target for each method (averaged across scenarios)."""
    if 'ip_f1' not in df.columns:
        ax.text(0.5, 0.5, 'IP F1 data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    baseline_methods = ['majority', 'inductive_miner_imf', 'heuristics_miner']
    
    # Plot baselines at specific eps
    df_eps = df[df['eps_jump'] == eps_value].copy()
    df_eps = df_eps[df_eps['ip_cov_target'] >= 0.6]  # Filter to IP-Cov >= 0.6
    agg_baselines = df_eps[df_eps['method'].isin(baseline_methods)].groupby(['method', 'ip_cov_target'])['ip_f1'].mean().reset_index()
    
    for method in baseline_methods:
        method_data = agg_baselines[agg_baselines['method'] == method].sort_values('ip_cov_target')
        if len(method_data) > 0:
            ax.plot(method_data['ip_cov_target'], method_data['ip_f1'],
                   marker=MARKERS.get(method, 'o'),
                   color=COLORS.get(method, '#999999'),
                   label=METHOD_LABELS.get(method, method),
                   markersize=6, linewidth=1.5, alpha=0.85)
    
    # Plot BPOP at the same epsilon as baselines
    bhpop_data = df_eps[df_eps['method'] == 'bhpop_single_po']
    if len(bhpop_data) > 0:
        agg_bhpop = bhpop_data.groupby('ip_cov_target')['ip_f1'].mean().reset_index()
        agg_bhpop = agg_bhpop.sort_values('ip_cov_target')
        ax.plot(agg_bhpop['ip_cov_target'], agg_bhpop['ip_f1'],
               marker=MARKERS.get('bhpop_single_po', 'o'),
               color=COLORS.get('bhpop_single_po', '#377eb8'),
               label=METHOD_LABELS.get('bhpop_single_po', 'BPOP'),
               markersize=6, linewidth=1.5, alpha=0.85)
    
    ax.set_xlabel('IP-Cov Target')
    ax.set_ylabel('IP F1 Score (Concurrency Recovery)')
    ax.set_title(f'IP F1 vs IP-Coverage (eps={eps_value})')
    ax.set_xlim(0.6, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, ncol=1)


def plot_ip_f1_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average IP F1 score by method across all scenarios."""
    if 'ip_f1' not in df.columns:
        ax.text(0.5, 0.5, 'IP F1 data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    method_avg = df.groupby('method')['ip_f1'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=False)
    
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
    ax.set_xlabel('IP F1 Score (Concurrency Recovery)')
    n_scenarios = len(df['scenario'].unique())
    ax.set_title(f'Average IP F1 Score by Method\n(averaged across {n_scenarios} scenarios)')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.02, i, f'{mean:.3f}', va='center', fontsize=7)


def plot_f1_by_scenario(df: pd.DataFrame, ax: plt.Axes):
    """Grouped bar chart: F1 score by scenario and method."""
    # Use ip_cov_target=1.0 and eps=0.01 for clean comparison
    df_filt = df[(df['ip_cov_target'] == 1.0) & (df['eps_jump'] == 0.01)].copy()
    
    scenarios = sorted(df_filt['scenario'].unique())
    methods = ['majority', 'inductive_miner_imf', 'heuristics_miner', 'bhpop_single_po']
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
    ax.set_ylabel('Edge F1 Score (Structural Recovery)')
    ax.set_title('Edge F1 Score by Scenario (IP-Cov=1.0, eps=0.01)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)


def plot_shd_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average SHD (lower is better) by method across all scenarios."""
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
    n_scenarios = len(df['scenario'].unique())
    ax.set_title(f'Average SHD by Method\n(averaged across {n_scenarios} scenarios)')
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.5, i, f'{mean:.1f}', va='center', fontsize=7)


def plot_feasibility_by_method(df: pd.DataFrame, ax: plt.Axes):
    """Bar chart: Average feasibility by method across all scenarios."""
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
    n_scenarios = len(df['scenario'].unique())
    ax.set_title(f'Average Feasibility by Method\n(averaged across {n_scenarios} scenarios)')
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(min(mean + std + 0.02, 1.05), i, f'{mean:.3f}', va='center', fontsize=7)


def plot_heatmap_f1(df: pd.DataFrame, ax: plt.Axes, method: str = 'bhpop_marginal_mode'):
    """Heatmap: F1 score by scenario and IP-Cov target for a specific method."""
    df_method = df[(df['method'] == method) & (df['eps_jump'] == 0.01)].copy()
    
    # Pivot to create heatmap data
    pivot = df_method.pivot_table(values='cover_f1', 
                                   index='scenario', 
                                   columns='ip_cov_target',
                                   aggfunc='mean')
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f'{c:.2f}' for c in pivot.columns], fontsize=7)
    ax.set_yticklabels([s.replace('_', '\n') for s in pivot.index], fontsize=7)
    
    ax.set_xlabel('IP-Cov Target')
    ax.set_ylabel('Scenario')
    ax.set_title(f'Edge F1 Heatmap: {METHOD_LABELS.get(method, method)}')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=6, color=color)
    
    return im


def plot_eps_comparison(df: pd.DataFrame, ax: plt.Axes):
    """Line plot: F1 vs epsilon for BPOP method (averaged across all scenarios)."""
    bhpop_data = df[df['method'] == 'bhpop_single_po'].copy()
    
    if len(bhpop_data) == 0:
        ax.text(0.5, 0.5, 'No BHPOP data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Aggregate by eps and ip_cov_target (averaging across scenarios)
    for ipcov in sorted(bhpop_data['ip_cov_target'].unique()):
        ipcov_data = bhpop_data[bhpop_data['ip_cov_target'] == ipcov]
        eps_agg = ipcov_data.groupby('eps_jump')['cover_f1'].mean().reset_index()
        eps_agg = eps_agg.sort_values('eps_jump')
        
        ax.plot(eps_agg['eps_jump'], eps_agg['cover_f1'],
               marker='o', label=f'IP-Cov={ipcov}', markersize=5, linewidth=1.2)
    
    ax.set_xlabel('Epsilon (eps_jump)')
    ax.set_ylabel('Edge F1 Score (Structural Recovery)')
    ax.set_title('BPOP: Edge F1 vs Epsilon by IP-Cov Target')
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
    
    # Load data (try the recomputed version with IP F1 first, fallback to original)
    csv_path = Path('systematic_experiment_results/experiment_summary_t0.40.csv')
    if not csv_path.exists():
        csv_path = Path('systematic_experiment_results/experiment_summary.csv')
    df = load_data(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Scenarios: {df['scenario'].unique()}")
    print(f"IP-Cov targets: {sorted(df['ip_cov_target'].unique())}")
    
    # Filter to main comparison condition (IP-Cov=1.0, eps=0.01) for summary plots
    df_main = df[(df['ip_cov_target'] == 1.0) & (df['eps_jump'] == 0.01)].copy()
    
    # Create output directory
    output_dir = Path('systematic_experiment_results/plots')
    output_dir.mkdir(exist_ok=True)
    
    # =========================
    # Figure 1: Overview (2x3 if IP F1 available, else 2x2)
    # =========================
    has_ip_f1 = 'ip_f1' in df.columns
    if has_ip_f1:
        fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
        plot_f1_by_method(df, axes1[0, 0])
        plot_ip_f1_by_method(df, axes1[0, 1])
        plot_shd_by_method(df, axes1[0, 2])
        plot_feasibility_by_method(df, axes1[1, 0])
        plot_f1_vs_ipcov(df, axes1[1, 1], eps_value=0.01)
        plot_ip_f1_vs_ipcov(df, axes1[1, 2], eps_value=0.01)
    else:
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
        plot_f1_by_method(df, axes1[0, 0])
        plot_shd_by_method(df, axes1[0, 1])
        plot_feasibility_by_method(df, axes1[1, 0])
        plot_f1_vs_ipcov(df, axes1[1, 1], eps_value=0.01)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / 'overview.pdf')
    fig1.savefig(output_dir / 'overview.png', dpi=300)
    print(f"Saved: {output_dir / 'overview.pdf'}")
    plt.close(fig1)
    
    # =========================
    # Figure 2: Summary Key Metrics (Main Results)
    # =========================
    # Create 1x3 layout: Edge F1, Feasibility, IP F1
    n_panels = 3 if 'ip_f1' in df_main.columns else 2
    fig_summary, axes_summary = plt.subplots(1, n_panels, figsize=(15, 5))
    if n_panels == 2:
        axes_summary = axes_summary.reshape(1, -1)
    axes_summary = axes_summary.flatten()
    
    # Panel 1: Average Edge F1 by method (largest to smallest)
    method_f1 = df_main.groupby('method')['cover_f1'].agg(['mean', 'std']).reset_index()
    method_f1 = method_f1.sort_values('mean', ascending=False)
    y_pos = np.arange(len(method_f1))
    axes_summary[0].barh(y_pos, method_f1['mean'], xerr=method_f1['std'],
                       color=[COLORS.get(m, '#999999') for m in method_f1['method']],
                       edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
    axes_summary[0].set_yticks(y_pos)
    axes_summary[0].set_yticklabels([METHOD_LABELS.get(m, m) for m in method_f1['method']])
    axes_summary[0].set_xlabel('Average Edge F1 Score', fontsize=11)
    axes_summary[0].set_title('Edge F1 (Structural Recovery)', fontsize=12, fontweight='bold')
    axes_summary[0].set_xlim(0, 1.05)
    axes_summary[0].grid(True, axis='x', alpha=0.3)
    for i, (mean, std) in enumerate(zip(method_f1['mean'], method_f1['std'])):
        axes_summary[0].text(mean + std + 0.02, i, f'{mean:.3f}', va='center', fontsize=8)
    
    # Panel 2: Average Feasibility by method (largest to smallest)
    method_feas = df_main.groupby('method')['feas'].agg(['mean', 'std']).reset_index()
    method_feas = method_feas.sort_values('mean', ascending=False)
    y_pos = np.arange(len(method_feas))
    axes_summary[1].barh(y_pos, method_feas['mean'], xerr=method_feas['std'],
                       color=[COLORS.get(m, '#999999') for m in method_feas['method']],
                       edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
    axes_summary[1].set_yticks(y_pos)
    axes_summary[1].set_yticklabels([METHOD_LABELS.get(m, m) for m in method_feas['method']])
    axes_summary[1].set_xlabel('Average Feasibility', fontsize=11)
    axes_summary[1].set_title('Feasibility (Execution Diagnostic)', fontsize=12, fontweight='bold')
    axes_summary[1].set_xlim(0, 1.05)
    axes_summary[1].grid(True, axis='x', alpha=0.3)
    for i, (mean, std) in enumerate(zip(method_feas['mean'], method_feas['std'])):
        axes_summary[1].text(min(mean + std + 0.02, 1.05), i, f'{mean:.3f}', va='center', fontsize=8)
    
    # Panel 3: IP F1 if available (largest to smallest)
    if 'ip_f1' in df_main.columns:
        method_ip = df_main.groupby('method')['ip_f1'].agg(['mean', 'std']).reset_index()
        method_ip = method_ip.sort_values('mean', ascending=False)
        y_pos = np.arange(len(method_ip))
        axes_summary[2].barh(y_pos, method_ip['mean'], xerr=method_ip['std'],
                           color=[COLORS.get(m, '#999999') for m in method_ip['method']],
                           edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85)
        axes_summary[2].set_yticks(y_pos)
        axes_summary[2].set_yticklabels([METHOD_LABELS.get(m, m) for m in method_ip['method']])
        axes_summary[2].set_xlabel('Average IP F1 Score', fontsize=11)
        axes_summary[2].set_title('IP F1 (Concurrency Recovery)', fontsize=12, fontweight='bold')
        axes_summary[2].set_xlim(0, 1.05)
        axes_summary[2].grid(True, axis='x', alpha=0.3)
        for i, (mean, std) in enumerate(zip(method_ip['mean'], method_ip['std'])):
            axes_summary[2].text(mean + std + 0.02, i, f'{mean:.3f}', va='center', fontsize=8)
    
    plt.suptitle('Summary: Key Metrics by Method (IP-Cov=1.0, ε=0.01)', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig_summary.tight_layout()
    fig_summary.savefig(output_dir / 'summary_key_metrics.pdf')
    fig_summary.savefig(output_dir / 'summary_key_metrics.png', dpi=300)
    print(f"Saved: {output_dir / 'summary_key_metrics.pdf'}")
    plt.close(fig_summary)
    
    # =========================
    # Figure 3: F1 by Scenario
    # =========================
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    plot_f1_by_scenario(df, ax3)
    fig3.tight_layout()
    fig3.savefig(output_dir / 'f1_by_scenario.pdf')
    fig3.savefig(output_dir / 'f1_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_by_scenario.pdf'}")
    plt.close(fig3)
    
    # =========================
    # Figure 4: Heatmaps for Majority and BPOP
    # =========================
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    
    im1 = plot_heatmap_f1(df, axes3[0], method='majority')
    im2 = plot_heatmap_f1(df, axes3[1], method='bhpop_single_po')
    
    # Add colorbar
    fig3.colorbar(im2, ax=axes3, shrink=0.8, label='Edge F1 Score (Structural Recovery)')
    
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
    # Figure 5: F1 vs IP-Cov for different epsilon values
    # =========================
    # Get actual epsilon values from data
    eps_values = sorted(df['eps_jump'].unique())
    print(f"Epsilon values in data: {eps_values}")
    
    # Create grid: 2x2 for 4 epsilon values
    fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
    
    for ax, eps in zip(axes5.flat[:len(eps_values)], eps_values):
        plot_f1_vs_ipcov(df, ax, eps_value=eps)
    
    fig5.tight_layout()
    fig5.savefig(output_dir / 'f1_vs_ipcov_by_eps.pdf')
    fig5.savefig(output_dir / 'f1_vs_ipcov_by_eps.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_vs_ipcov_by_eps.pdf'}")
    plt.close(fig5)
    
    # =========================
    # Figure 6: Scenario-by-scenario comparison at IP-Cov=1.0
    # =========================
    df_cov1 = df[df['ip_cov_target'] == 1.0].copy()
    scenarios = sorted(df_cov1['scenario'].unique())
    
    # Create appropriate grid for available scenarios (2x3 for 5-6 scenarios, 2x2 for 4)
    n_scenarios = len(scenarios)
    if n_scenarios <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    fig6, axes6 = plt.subplots(nrows, ncols, figsize=(15, 10))
    
    for idx, (ax, scenario) in enumerate(zip(axes6.flat, scenarios)):
        scenario_df = df_cov1[df_cov1['scenario'] == scenario]
        
        # Prepare data for plotting
        methods_to_plot = ['majority', 'inductive_miner_imf', 'heuristics_miner']
        x_pos = np.arange(len(methods_to_plot))
        
        # Get F1 scores for baseline methods
        f1_scores = []
        for method in methods_to_plot:
            method_data = scenario_df[scenario_df['method'] == method]
            if len(method_data) > 0:
                f1_scores.append(method_data['cover_f1'].values[0])
            else:
                f1_scores.append(0)
        
        # Plot baseline methods
        bars = ax.bar(x_pos, f1_scores, width=0.6, alpha=0.7)
        bars[0].set_color(COLORS['majority'])
        bars[1].set_color(COLORS['inductive_miner_imf'])
        bars[2].set_color(COLORS['heuristics_miner'])
        
        # Add BHPOP results for different epsilon values
        bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
        if len(bhpop_data) > 0:
            eps_vals = sorted([eps for eps in bhpop_data['eps_jump'].dropna().unique() if not np.isnan(eps)])
            bhpop_f1s = []
            valid_eps_vals = []
            for eps in eps_vals:
                eps_data = bhpop_data[bhpop_data['eps_jump'] == eps]['cover_f1'].values
                if len(eps_data) > 0:
                    bhpop_f1s.append(eps_data[0])
                    valid_eps_vals.append(eps)
            eps_vals = valid_eps_vals
            
            # Plot BHPOP as points with different markers for each epsilon
            bhpop_x = len(methods_to_plot)
            markers_list = ['o', 's', '^', 'D']
            for i, (eps, f1) in enumerate(zip(eps_vals, bhpop_f1s)):
                ax.scatter(bhpop_x, f1, s=150, marker=markers_list[i], 
                          color=COLORS['bhpop_single_po'], 
                          edgecolors='black', linewidths=1.5,
                          label=f'ε={eps:.3f}', zorder=5)
        
        # Formatting
        ax.set_ylim([0, 1.05])
        ax.set_xticks(list(x_pos) + [len(methods_to_plot)])
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_to_plot] + ['BHPOP\n(Ours)'], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Edge F1 Score (Structural Recovery)', fontsize=11)
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9, title='BPOP')
    
    # Hide unused subplots
    for idx in range(len(scenarios), nrows * ncols):
        axes6.flat[idx].axis('off')
    
    plt.suptitle('Edge F1 Score by Method and Scenario (IP-Cov=1.0)', 
                 fontsize=14, fontweight='bold', y=0.995)
    fig6.tight_layout()
    fig6.savefig(output_dir / 'f1_by_scenario_comparison.pdf')
    fig6.savefig(output_dir / 'f1_by_scenario_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_by_scenario_comparison.pdf'}")
    plt.close(fig6)
    
    # =========================
    # Figure 7: F1 vs IP-Cov for each scenario (showing all methods)
    # =========================
    scenarios = sorted(df['scenario'].unique())
    n_scenarios = len(scenarios)
    if n_scenarios <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    fig7, axes7 = plt.subplots(nrows, ncols, figsize=(15, 10))
    
    for idx, (ax, scenario) in enumerate(zip(axes7.flat, scenarios)):
        scenario_df = df[df['scenario'] == scenario]
        
        # Plot baseline methods (averaged over all settings, filtered to IP-Cov >= 0.6)
        for method in ['majority', 'inductive_miner_imf', 'heuristics_miner']:
            method_data = scenario_df[scenario_df['method'] == method]
            method_data = method_data[method_data['ip_cov_target'] >= 0.6]
            if len(method_data) > 0:
                ipcov_f1 = method_data.groupby('ip_cov_target')['cover_f1'].mean()
                ax.plot(ipcov_f1.index, ipcov_f1.values, 
                   marker=MARKERS.get(method, 'o'), 
                   color=COLORS[method], 
                   label=METHOD_LABELS[method],
                   linewidth=2, markersize=8)
        
        # Plot BPOP with different epsilon values
        bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
        eps_vals = sorted(bhpop_data['eps_jump'].dropna().unique())
        
        # Filter to IP-Cov >= 0.6
        bhpop_data = bhpop_data[bhpop_data['ip_cov_target'] >= 0.6]
        
        if len(bhpop_data) > 0:
            # Use average over all epsilon values for main line
            bhpop_avg = bhpop_data.groupby('ip_cov_target')['cover_f1'].mean()
            
            # Debug: print if only one IP-Cov value (would cause horizontal line)
            if len(bhpop_avg) == 1:
                print(f"WARNING: {scenario} has only 1 IP-Cov value: {bhpop_avg.index[0]:.2f} (F1={bhpop_avg.values[0]:.3f})")
            
            ax.plot(bhpop_avg.index, bhpop_avg.values,
                   marker=MARKERS.get('bhpop_single_po', 's'),
                   color=COLORS['bhpop_single_po'],
                   label=METHOD_LABELS['bhpop_single_po'],
                   linewidth=2, markersize=8, zorder=5)
            
            # Show range across epsilon values as shaded area
            bhpop_min = bhpop_data.groupby('ip_cov_target')['cover_f1'].min()
            bhpop_max = bhpop_data.groupby('ip_cov_target')['cover_f1'].max()
            ax.fill_between(bhpop_avg.index, bhpop_min.values, bhpop_max.values,
                           color=COLORS['bhpop_single_po'], alpha=0.2, zorder=4)
        
        # Formatting
        ax.set_xlim([0.6, 1.05])  # Start from 0.6 instead of 0.4
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('IP-Coverage Target', fontsize=11)
        ax.set_ylabel('Edge F1 Score (Structural Recovery)', fontsize=11)
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(len(scenarios), nrows * ncols):
        axes7.flat[idx].axis('off')
    
    # Create observation count table (only IP-Cov >= 0.6)
    obs_table_data = []
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        bhpop_scenario = scenario_df[scenario_df['method'] == 'bhpop_single_po']
        bhpop_scenario = bhpop_scenario[bhpop_scenario['ip_cov_target'] >= 0.6]
        for ipcov in sorted(bhpop_scenario['ip_cov_target'].dropna().unique()):
            count = len(bhpop_scenario[bhpop_scenario['ip_cov_target'] == ipcov])
            obs_table_data.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'IP-Cov Target': ipcov,
                'Observations': count
            })
    
    if obs_table_data:
        import pandas as pd
        obs_table_df = pd.DataFrame(obs_table_data)
        obs_table_pivot = obs_table_df.pivot(index='Scenario', columns='IP-Cov Target', values='Observations').fillna(0).astype(int)
        
        # Add table as a new subplot (use the last empty subplot if available)
        if len(scenarios) < nrows * ncols:
            table_ax = axes7.flat[len(scenarios)]
            table_ax.axis('off')
            table = table_ax.table(cellText=obs_table_pivot.values,
                                  rowLabels=obs_table_pivot.index,
                                  colLabels=[f'{x:.2f}' for x in obs_table_pivot.columns],
                                  cellLoc='center',
                                  loc='center',
                                  bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            table_ax.set_title('Observation Counts\n(Scenario × IP-Cov Target)', 
                              fontsize=10, fontweight='bold', pad=10)
    
    plt.suptitle('Edge F1 Score vs IP-Coverage by Scenario\\n(BPOP shaded area shows range across ε=[0.005, 0.01, 0.02, 0.05])', 
                 fontsize=14, fontweight='bold', y=0.995)
    fig7.tight_layout()
    fig7.savefig(output_dir / 'f1_vs_ipcov_by_scenario.pdf')
    fig7.savefig(output_dir / 'f1_vs_ipcov_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_vs_ipcov_by_scenario.pdf'}")
    plt.close(fig7)
    
    # =========================
    # Figure 8: Feasibility vs IP-Cov for each scenario (similar to F1 plot)
    # =========================
    scenarios = sorted(df['scenario'].unique())
    n_scenarios = len(scenarios)
    if n_scenarios <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    fig8, axes8 = plt.subplots(nrows, ncols, figsize=(15, 10))
    
    for idx, (ax, scenario) in enumerate(zip(axes8.flat, scenarios)):
        scenario_df = df[df['scenario'] == scenario]
        
        # Plot baseline methods (averaged over all settings, filtered to IP-Cov >= 0.6)
        for method in ['majority', 'inductive_miner_imf', 'heuristics_miner']:
            method_data = scenario_df[scenario_df['method'] == method]
            method_data = method_data[method_data['ip_cov_target'] >= 0.6]
            if len(method_data) > 0:
                ipcov_feas = method_data.groupby('ip_cov_target')['feas'].mean()
                ax.plot(ipcov_feas.index, ipcov_feas.values, 
                   marker=MARKERS.get(method, 'o'), 
                   color=COLORS[method], 
                   label=METHOD_LABELS[method],
                   linewidth=2, markersize=8)
        
        # Plot BPOP with different epsilon values
        bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
        eps_vals = sorted(bhpop_data['eps_jump'].dropna().unique())
        
        # Filter to IP-Cov >= 0.6
        bhpop_data = bhpop_data[bhpop_data['ip_cov_target'] >= 0.6]
        
        if len(bhpop_data) > 0:
            # Use average over all epsilon values for main line
            bhpop_avg = bhpop_data.groupby('ip_cov_target')['feas'].mean()
            
            # Debug: print if only one IP-Cov value (would cause horizontal line)
            if len(bhpop_avg) == 1:
                print(f"WARNING (Feasibility): {scenario} has only 1 IP-Cov value: {bhpop_avg.index[0]:.2f} (Feas={bhpop_avg.values[0]:.3f})")
            
            ax.plot(bhpop_avg.index, bhpop_avg.values,
                   marker=MARKERS.get('bhpop_single_po', 's'),
                   color=COLORS['bhpop_single_po'],
                   label=METHOD_LABELS['bhpop_single_po'],
                   linewidth=2, markersize=8, zorder=5)
            
            # Show range across epsilon values as shaded area
            bhpop_min = bhpop_data.groupby('ip_cov_target')['feas'].min()
            bhpop_max = bhpop_data.groupby('ip_cov_target')['feas'].max()
            ax.fill_between(bhpop_avg.index, bhpop_min.values, bhpop_max.values,
                           color=COLORS['bhpop_single_po'], alpha=0.2, zorder=4)
        
        # Formatting
        ax.set_xlim([0.6, 1.05])  # Start from 0.6 instead of 0.4
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('IP-Coverage Target', fontsize=11)
        ax.set_ylabel('Feasibility', fontsize=11)
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(len(scenarios), nrows * ncols):
        axes8.flat[idx].axis('off')
    
    plt.suptitle('Feasibility vs IP-Coverage by Scenario\\n(BPOP shaded area shows range across ε=[0.005, 0.01, 0.02, 0.05])', 
                 fontsize=14, fontweight='bold', y=0.995)
    fig8.tight_layout()
    fig8.savefig(output_dir / 'feasibility_vs_ipcov_by_scenario.pdf')
    fig8.savefig(output_dir / 'feasibility_vs_ipcov_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'feasibility_vs_ipcov_by_scenario.pdf'}")
    plt.close(fig8)
    
    # =========================
    # Figure 9: Feasibility vs IP-Cov by epsilon (similar to F1 plot)
    # =========================
    def plot_feas_vs_ipcov(df: pd.DataFrame, ax: plt.Axes, eps_value: float = 0.01):
        """Line plot: Feasibility vs IP-Cov target for each method (averaged across scenarios)."""
        # Filter both baseline methods and BPOP to specific epsilon for fair comparison
        baseline_methods = ['majority', 'inductive_miner_imf', 'heuristics_miner']
        
        # Plot baselines at specific eps
        df_eps = df[df['eps_jump'] == eps_value].copy()
        df_eps = df_eps[df_eps['ip_cov_target'] >= 0.6]  # Filter to IP-Cov >= 0.6
        agg_baselines = df_eps[df_eps['method'].isin(baseline_methods)].groupby(['method', 'ip_cov_target'])['feas'].mean().reset_index()
        
        for method in baseline_methods:
            method_data = agg_baselines[agg_baselines['method'] == method].sort_values('ip_cov_target')
            if len(method_data) > 0:
                ax.plot(method_data['ip_cov_target'], method_data['feas'],
                   marker=MARKERS.get(method, 'o'),
                   color=COLORS.get(method, '#999999'),
                   label=METHOD_LABELS.get(method, method),
                   markersize=6, linewidth=1.5, alpha=0.85)
        
        # Plot BPOP at the same epsilon as baselines
        bhpop_data = df_eps[df_eps['method'] == 'bhpop_single_po']
        if len(bhpop_data) > 0:
            agg_bhpop = bhpop_data.groupby('ip_cov_target')['feas'].mean().reset_index()
            agg_bhpop = agg_bhpop.sort_values('ip_cov_target')
            ax.plot(agg_bhpop['ip_cov_target'], agg_bhpop['feas'],
               marker=MARKERS.get('bhpop_single_po', 'o'),
               color=COLORS.get('bhpop_single_po', '#377eb8'),
               label=METHOD_LABELS.get('bhpop_single_po', 'BPOP'),
               markersize=6, linewidth=1.5, alpha=0.85)
        
        ax.set_xlabel('IP-Cov Target')
        ax.set_ylabel('Feasibility')
        ax.set_title(f'Feasibility vs IP-Coverage (eps={eps_value})')
        ax.set_xlim(0.6, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=7, ncol=1)
    
    eps_values = sorted(df['eps_jump'].dropna().unique())
    n_eps = len(eps_values)
    if n_eps <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    fig9, axes9 = plt.subplots(nrows, ncols, figsize=(15, 10))
    
    for idx, (ax, eps) in enumerate(zip(axes9.flat, eps_values)):
        plot_feas_vs_ipcov(df, ax, eps_value=eps)
    
    # Hide unused subplots
    for idx in range(len(eps_values), nrows * ncols):
        axes9.flat[idx].axis('off')
    
    plt.suptitle('Feasibility vs IP-Coverage by Epsilon', 
                 fontsize=14, fontweight='bold', y=0.995)
    fig9.tight_layout()
    fig9.savefig(output_dir / 'feasibility_vs_ipcov_by_eps.pdf')
    fig9.savefig(output_dir / 'feasibility_vs_ipcov_by_eps.png', dpi=300)
    print(f"Saved: {output_dir / 'feasibility_vs_ipcov_by_eps.pdf'}")
    plt.close(fig9)
    
    # =========================
    # Figure 10: IP F1 vs IP-Cov for each scenario (Concurrency Recovery)
    # =========================
    if 'ip_f1' in df.columns:
        scenarios = sorted(df['scenario'].unique())
        n_scenarios = len(scenarios)
        if n_scenarios <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        fig10a, axes10a = plt.subplots(nrows, ncols, figsize=(15, 10))
        
        for idx, (ax, scenario) in enumerate(zip(axes10a.flat, scenarios)):
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['ip_cov_target'] >= 0.6]
            
            # Plot baseline methods
            for method in ['majority', 'inductive_miner_imf', 'heuristics_miner']:
                method_data = scenario_df[scenario_df['method'] == method]
                if len(method_data) > 0:
                    ipcov_f1 = method_data.groupby('ip_cov_target')['ip_f1'].mean()
                    ax.plot(ipcov_f1.index, ipcov_f1.values, 
                           marker=MARKERS.get(method, 'o'), 
                           color=COLORS[method], 
                           label=METHOD_LABELS[method],
                           linewidth=2, markersize=8)
            
            # Plot BPOP with different epsilon values
            bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
            eps_vals = sorted(bhpop_data['eps_jump'].dropna().unique())
            
            if len(bhpop_data) > 0:
                # Use average over all epsilon values for main line
                bhpop_avg = bhpop_data.groupby('ip_cov_target')['ip_f1'].mean()
                ax.plot(bhpop_avg.index, bhpop_avg.values,
                       marker=MARKERS.get('bhpop_single_po', 's'),
                       color=COLORS['bhpop_single_po'],
                       label=METHOD_LABELS['bhpop_single_po'],
                       linewidth=2, markersize=8, zorder=5)
                
                # Show range across epsilon values as shaded area
                bhpop_min = bhpop_data.groupby('ip_cov_target')['ip_f1'].min()
                bhpop_max = bhpop_data.groupby('ip_cov_target')['ip_f1'].max()
                ax.fill_between(bhpop_avg.index, bhpop_min.values, bhpop_max.values,
                               color=COLORS['bhpop_single_po'], alpha=0.2, zorder=4)
            
            ax.set_xlim([0.6, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('IP-Coverage Target', fontsize=11)
            ax.set_ylabel('IP F1 Score (Concurrency Recovery)', fontsize=11)
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        for idx in range(len(scenarios), nrows * ncols):
            axes10a.flat[idx].axis('off')
        
        plt.suptitle('IP F1 Score vs IP-Coverage by Scenario\\n(BPOP shaded area shows range across ε=[0.005, 0.01, 0.02, 0.05])', 
                     fontsize=14, fontweight='bold', y=0.995)
        fig10a.tight_layout()
        fig10a.savefig(output_dir / 'ip_f1_vs_ipcov_by_scenario.pdf')
        fig10a.savefig(output_dir / 'ip_f1_vs_ipcov_by_scenario.png', dpi=300)
        print(f"Saved: {output_dir / 'ip_f1_vs_ipcov_by_scenario.pdf'}")
        plt.close(fig10a)
        
        # =========================
        # Figure 11: IP F1 vs IP-Cov by epsilon
        # =========================
        eps_values = sorted(df['eps_jump'].dropna().unique())
        n_eps = len(eps_values)
        if n_eps <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        fig11a, axes11a = plt.subplots(nrows, ncols, figsize=(15, 10))
        
        for idx, (ax, eps) in enumerate(zip(axes11a.flat, eps_values)):
            plot_ip_f1_vs_ipcov(df, ax, eps_value=eps)
        
        for idx in range(len(eps_values), nrows * ncols):
            axes11a.flat[idx].axis('off')
        
        plt.suptitle('IP F1 Score vs IP-Coverage by Epsilon', 
                     fontsize=14, fontweight='bold', y=0.995)
        fig11a.tight_layout()
        fig11a.savefig(output_dir / 'ip_f1_vs_ipcov_by_eps.pdf')
        fig11a.savefig(output_dir / 'ip_f1_vs_ipcov_by_eps.png', dpi=300)
        print(f"Saved: {output_dir / 'ip_f1_vs_ipcov_by_eps.pdf'}")
        plt.close(fig11a)
    
    # =========================
    # Figure 12: Edge Precision/Recall/F1 by Scenario (Structural Recovery)
    # =========================
    if 'cover_precision' in df.columns and 'cover_recall' in df.columns:
        scenarios = sorted(df['scenario'].unique())
        n_scenarios = len(scenarios)
        if n_scenarios <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        fig12, axes12 = plt.subplots(nrows, ncols, figsize=(15, 10))
        
        for idx, (ax, scenario) in enumerate(zip(axes12.flat, scenarios)):
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['ip_cov_target'] >= 0.6]
            
            # Plot baseline methods
            for method in ['majority', 'inductive_miner_imf', 'heuristics_miner']:
                method_data = scenario_df[scenario_df['method'] == method]
                if len(method_data) > 0:
                    ipcov_p = method_data.groupby('ip_cov_target')['cover_precision'].mean()
                    ipcov_r = method_data.groupby('ip_cov_target')['cover_recall'].mean()
                    ipcov_f1 = method_data.groupby('ip_cov_target')['cover_f1'].mean()
                    ax.plot(ipcov_f1.index, ipcov_p.values, '--', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (P)', linewidth=1.5, markersize=6, alpha=0.7)
                    ax.plot(ipcov_f1.index, ipcov_r.values, ':', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (R)', linewidth=1.5, markersize=6, alpha=0.7)
                    ax.plot(ipcov_f1.index, ipcov_f1.values, '-', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (F1)', linewidth=2, markersize=8)
            
            # Plot BPOP
            bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
            if len(bhpop_data) > 0:
                bhpop_p = bhpop_data.groupby('ip_cov_target')['cover_precision'].mean()
                bhpop_r = bhpop_data.groupby('ip_cov_target')['cover_recall'].mean()
                bhpop_f1 = bhpop_data.groupby('ip_cov_target')['cover_f1'].mean()
                ax.plot(bhpop_f1.index, bhpop_p.values, '--', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (P)', linewidth=1.5, markersize=6, alpha=0.7)
                ax.plot(bhpop_f1.index, bhpop_r.values, ':', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (R)', linewidth=1.5, markersize=6, alpha=0.7)
                ax.plot(bhpop_f1.index, bhpop_f1.values, '-', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (F1)', linewidth=2, markersize=8)
            
            ax.set_xlim([0.6, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('IP-Coverage Target', fontsize=11)
            ax.set_ylabel('Edge Precision/Recall/F1', fontsize=11)
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)
        
        for idx in range(len(scenarios), nrows * ncols):
            axes12.flat[idx].axis('off')
        
        plt.suptitle('Structural Recovery: Edge Precision/Recall/F1 by Scenario', 
                     fontsize=14, fontweight='bold', y=0.995)
        fig12.tight_layout()
        fig12.savefig(output_dir / 'edge_metrics_by_scenario.pdf')
        fig12.savefig(output_dir / 'edge_metrics_by_scenario.png', dpi=300)
        print(f"Saved: {output_dir / 'edge_metrics_by_scenario.pdf'}")
        plt.close(fig12)
    
    # =========================
    # Figure 13: Incomparable Pair Precision/Recall/F1 by Scenario (Concurrency Recovery)
    # =========================
    if 'ip_precision' in df.columns and 'ip_recall' in df.columns:
        scenarios = sorted(df['scenario'].unique())
        n_scenarios = len(scenarios)
        if n_scenarios <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        fig13, axes13 = plt.subplots(nrows, ncols, figsize=(15, 10))
        
        for idx, (ax, scenario) in enumerate(zip(axes13.flat, scenarios)):
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['ip_cov_target'] >= 0.6]
            
            # Plot baseline methods
            for method in ['majority', 'inductive_miner_imf', 'heuristics_miner']:
                method_data = scenario_df[scenario_df['method'] == method]
                if len(method_data) > 0:
                    ipcov_p = method_data.groupby('ip_cov_target')['ip_precision'].mean()
                    ipcov_r = method_data.groupby('ip_cov_target')['ip_recall'].mean()
                    ipcov_f1 = method_data.groupby('ip_cov_target')['ip_f1'].mean()
                    ax.plot(ipcov_f1.index, ipcov_p.values, '--', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (P)', linewidth=1.5, markersize=6, alpha=0.7)
                    ax.plot(ipcov_f1.index, ipcov_r.values, ':', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (R)', linewidth=1.5, markersize=6, alpha=0.7)
                    ax.plot(ipcov_f1.index, ipcov_f1.values, '-', marker=MARKERS.get(method, 'o'),
                           color=COLORS[method], label=f'{METHOD_LABELS[method]} (F1)', linewidth=2, markersize=8)
            
            # Plot BPOP
            bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
            if len(bhpop_data) > 0:
                bhpop_p = bhpop_data.groupby('ip_cov_target')['ip_precision'].mean()
                bhpop_r = bhpop_data.groupby('ip_cov_target')['ip_recall'].mean()
                bhpop_f1 = bhpop_data.groupby('ip_cov_target')['ip_f1'].mean()
                ax.plot(bhpop_f1.index, bhpop_p.values, '--', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (P)', linewidth=1.5, markersize=6, alpha=0.7)
                ax.plot(bhpop_f1.index, bhpop_r.values, ':', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (R)', linewidth=1.5, markersize=6, alpha=0.7)
                ax.plot(bhpop_f1.index, bhpop_f1.values, '-', marker='*', color=COLORS['bhpop_single_po'],
                       label='BPOP (F1)', linewidth=2, markersize=8)
            
            ax.set_xlim([0.6, 1.05])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('IP-Coverage Target', fontsize=11)
            ax.set_ylabel('IP Precision/Recall/F1', fontsize=11)
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)
        
        for idx in range(len(scenarios), nrows * ncols):
            axes13.flat[idx].axis('off')
        
        plt.suptitle('Concurrency Recovery: Incomparable Pair Precision/Recall/F1 by Scenario', 
                     fontsize=14, fontweight='bold', y=0.995)
        fig13.tight_layout()
        fig13.savefig(output_dir / 'ip_metrics_by_scenario.pdf')
        fig13.savefig(output_dir / 'ip_metrics_by_scenario.png', dpi=300)
        print(f"Saved: {output_dir / 'ip_metrics_by_scenario.pdf'}")
        plt.close(fig13)
    
    # =========================
    # Figure 14: IP-Cov Realized by Scenario (Execution Diagnostic)
    # =========================
    if 'ip_cov_realized' in df.columns:
        scenarios = sorted(df['scenario'].unique())
        n_scenarios = len(scenarios)
        if n_scenarios <= 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3
        fig14, axes14 = plt.subplots(nrows, ncols, figsize=(15, 10))
        
        for idx, (ax, scenario) in enumerate(zip(axes14.flat, scenarios)):
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['ip_cov_target'] >= 0.6]
            
            # Plot BPOP IP-Cov realized
            bhpop_data = scenario_df[scenario_df['method'] == 'bhpop_single_po']
            if len(bhpop_data) > 0:
                bhpop_ipcov = bhpop_data.groupby('ip_cov_target')['ip_cov_realized'].mean()
                ax.plot(bhpop_ipcov.index, bhpop_ipcov.values, '-', marker='*',
                       color=COLORS['bhpop_single_po'], label='BPOP', linewidth=2, markersize=8)
                # Add diagonal line (perfect match)
                ax.plot([0.6, 1.0], [0.6, 1.0], 'k--', alpha=0.3, label='Perfect', linewidth=1)
            
            ax.set_xlim([0.6, 1.05])
            ax.set_ylim([0.6, 1.05])
            ax.set_xlabel('IP-Cov Target', fontsize=11)
            ax.set_ylabel('IP-Cov Realized', fontsize=11)
            ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        for idx in range(len(scenarios), nrows * ncols):
            axes14.flat[idx].axis('off')
        
        plt.suptitle('Execution Diagnostic: IP-Cov Realized vs Target by Scenario', 
                     fontsize=14, fontweight='bold', y=0.995)
        fig14.tight_layout()
        fig14.savefig(output_dir / 'ipcov_realized_by_scenario.pdf')
        fig14.savefig(output_dir / 'ipcov_realized_by_scenario.png', dpi=300)
        print(f"Saved: {output_dir / 'ipcov_realized_by_scenario.pdf'}")
        plt.close(fig14)
    
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
    
    # =========================
    # Main Results Table (for paper)
    # =========================
    print("\n" + "="*60)
    print("MAIN RESULTS TABLE (IP-Cov=1.0, eps=0.01)")
    print("="*60)
    # Create comprehensive table
    results_table = []
    for scenario in sorted(df_main['scenario'].unique()):
        scenario_df = df_main[df_main['scenario'] == scenario]
        for method in ['majority', 'inductive_miner_imf', 'heuristics_miner', 'bhpop_single_po']:
            method_data = scenario_df[scenario_df['method'] == method]
            if len(method_data) > 0:
                row = method_data.iloc[0]
                results_table.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Method': METHOD_LABELS.get(method, method),
                    'Edge F1': f"{row['cover_f1']:.3f}",
                    'Edge P': f"{row.get('cover_precision', np.nan):.3f}" if 'cover_precision' in row else 'N/A',
                    'Edge R': f"{row.get('cover_recall', np.nan):.3f}" if 'cover_recall' in row else 'N/A',
                    'IP F1': f"{row.get('ip_f1', np.nan):.3f}" if 'ip_f1' in row else 'N/A',
                    'SHD': f"{row['shd']:.1f}",
                    'Feasibility': f"{row['feas']:.3f}",
                })
    
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(output_dir / 'main_results_table.csv', index=False)
    print(results_df.to_string(index=False))
    print(f"\nSaved: {output_dir / 'main_results_table.csv'}")
    
    # Generate LaTeX table
    latex_table = results_df.to_latex(index=False, float_format="%.3f", escape=False)
    with open(output_dir / 'main_results_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"Saved: {output_dir / 'main_results_table.tex'}")
    
    # Print best method by scenario
    print("\n" + "="*60)
    print("BEST METHOD BY SCENARIO (IP-Cov=1.0, eps=0.01)")
    print("="*60)
    df_best = df[(df['ip_cov_target'] == 1.0) & (df['eps_jump'] == 0.01)]
    for scenario in sorted(df_best['scenario'].unique()):
        scenario_df = df_best[df_best['scenario'] == scenario]
        best_row = scenario_df.loc[scenario_df['cover_f1'].idxmax()]
        print(f"{scenario}: {METHOD_LABELS.get(best_row['method'], best_row['method'])} "
              f"(Edge F1={best_row['cover_f1']:.3f}, SHD={best_row['shd']}, feas={best_row['feas']:.3f})")
    
    print(f"\n✅ All plots and tables saved to: {output_dir}")


if __name__ == '__main__':
    main()

