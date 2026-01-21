#!/usr/bin/env python3
"""
Generate publication-quality plots for Aliyun experiments.

Metrics evaluated (per paper):
1. Structural Recovery: Edge Precision/Recall/F1 on TR(Ĝ) vs TR(G*)
2. Concurrency Recovery: IP Precision/Recall/F1 on incomparable pairs under TC(·)
3. Feasibility: fraction of traces that are linear extensions of TC(Ĝ)
4. IP-Cov: fraction of ground-truth IP witnessed in both orientations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Configuration
# =========================
def setup_style():
    """Configure matplotlib for publication plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.95,
        'text.usetex': False,
    })

# Colors - colorblind friendly
COLORS = {
    'bhpop_single_po': '#2171b5',      # Blue (our method)
    'majority': '#238b45',              # Green
    'inductive_miner_imf': '#6a51a3',  # Purple
    'heuristics_miner': '#d94801',     # Orange
}

METHOD_LABELS = {
    'majority': 'Majority',
    'inductive_miner_imf': 'Inductive Miner',
    'heuristics_miner': 'Heuristics Miner',
    'bhpop_single_po': 'BPOP (Ours)',
}

MARKERS = {
    'bhpop_single_po': '*',
    'majority': 's',
    'inductive_miner_imf': '^',
    'heuristics_miner': 'D',
}

SCENARIO_ORDER = [
    'simple_ecs', 'slb_ecs_rds', 'slb_ecs_redis',
    'eip_slb_ecs', 'dual_zone_ecs_slb', 'dual_zone_ecs_slb_rds',
]

SCENARIO_SHORT = {
    'simple_ecs': 'S1', 'slb_ecs_rds': 'S2', 'slb_ecs_redis': 'S3',
    'eip_slb_ecs': 'S4', 'dual_zone_ecs_slb': 'S5', 'dual_zone_ecs_slb_rds': 'S6',
}


def main():
    setup_style()
    
    # Load data
    csv_path = Path('systematic_experiment_results/experiment_summary_t0.33.csv')
    df = pd.read_csv(csv_path)
    
    output_dir = Path('systematic_experiment_results/plots')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Separate data
    df_bpop = df[df['method'] == 'bhpop_single_po'].copy()
    df_baselines = df[df['method'] != 'bhpop_single_po'].copy()
    
    scenarios = [s for s in SCENARIO_ORDER if s in df['scenario'].unique()]
    methods_order = ['bhpop_single_po', 'majority', 'inductive_miner_imf', 'heuristics_miner']
    baseline_methods = ['majority', 'inductive_miner_imf', 'heuristics_miner']
    
    # =====================================================================
    # FIGURE 1: Summary Key Metrics (for main paper)
    # Compare all methods at IP-Cov=1.0 (highest informativeness)
    # =====================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    
    # Filter all methods to IP-Cov=1.0 (highest informativeness)
    df_baselines_10 = df_baselines[df_baselines['ip_cov_target'] == 1.0]
    df_bpop_10 = df_bpop[df_bpop['ip_cov_target'] == 1.0]
    df_baselines_06 = df_baselines[df_baselines['ip_cov_target'] == 0.6]  # For other figures
    
    # Panel 1: Edge F1 (Structural Recovery)
    ax = axes1[0]
    means, stds, colors, labels = [], [], [], []
    
    for method in methods_order:
        if method == 'bhpop_single_po':
            data = df_bpop_10['cover_f1']  # BPOP at IP-Cov=1.0
        else:
            data = df_baselines_10[df_baselines_10['method'] == method]['cover_f1']
        means.append(data.mean())
        stds.append(data.std())
        colors.append(COLORS[method])
        labels.append(METHOD_LABELS[method])
    
    y_pos = np.arange(len(methods_order))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, capsize=4, height=0.6, 
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Edge F1 Score')
    ax.set_title('Structural Recovery', fontweight='bold', fontsize=13)
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=means[0], color=COLORS['bhpop_single_po'], linestyle='--', alpha=0.5, linewidth=1)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 0.02, i, f'{mean:.2f}', va='center', fontsize=10, fontweight='bold')
    
    # Panel 2: Feasibility
    ax = axes1[1]
    means, stds = [], []
    
    for method in methods_order:
        if method == 'bhpop_single_po':
            data = df_bpop_10['feas'].dropna()  # BPOP at IP-Cov=1.0
            means.append(data.mean() if len(data) > 0 else np.nan)
            stds.append(data.std() if len(data) > 0 else 0)
        else:
            data = df_baselines_10[df_baselines_10['method'] == method]['feas'].dropna()
            means.append(data.mean() if len(data) > 0 else 0)
            stds.append(data.std() if len(data) > 0 else 0)
    
    # Replace NaN with 0 for plotting
    means_plot = [0 if np.isnan(m) else m for m in means]
    bars = ax.barh(y_pos, means_plot, xerr=stds, color=colors, capsize=4, height=0.6,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Feasibility')
    ax.set_title('Execution Validity', fontweight='bold', fontsize=13)
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        if np.isnan(mean):
            ax.text(0.02, i, 'N/A', va='center', fontsize=10, color='gray')
        else:
            ax.text(mean + std + 0.02, i, f'{mean:.2f}', va='center', fontsize=10, fontweight='bold')
    
    fig1.suptitle(r'Aggregate Performance Comparison (IP-Cov=1.0, averaged over $\epsilon$ = 0.005, 0.01, 0.02, 0.05)', 
               fontsize=11, fontweight='bold', y=1.02)
    fig1.tight_layout()
    fig1.savefig(output_dir / 'summary_key_metrics.pdf')
    fig1.savefig(output_dir / 'summary_key_metrics.png', dpi=300)
    print(f"Saved: {output_dir / 'summary_key_metrics.pdf'}")
    plt.close(fig1)
    
    # =====================================================================
    # FIGURE 2: Edge F1 vs IP-Coverage (for understanding data informativeness)
    # =====================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    ip_cov_targets = sorted(df_baselines['ip_cov_target'].dropna().unique())
    
    # Plot BPOP across IP-Cov targets (plot first to be behind baselines)
    bpop_grouped = df_bpop.groupby('ip_cov_target')['cover_f1'].agg(['mean', 'std'])
    ax2.errorbar(bpop_grouped.index, bpop_grouped['mean'], yerr=bpop_grouped['std'],
                marker=MARKERS['bhpop_single_po'], color=COLORS['bhpop_single_po'],
                label='BPOP (Ours)', capsize=5, markersize=12, linewidth=2.5, zorder=5)
    
    # Plot baselines across IP-Cov targets
    for method in baseline_methods:
        method_data = df_baselines[df_baselines['method'] == method]
        grouped = method_data.groupby('ip_cov_target')['cover_f1'].agg(['mean', 'std'])
        ax2.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                    marker=MARKERS[method], color=COLORS[method], 
                    label=METHOD_LABELS[method], capsize=3, linewidth=2, markersize=8)
    
    ax2.set_xlabel('IP-Coverage Target', fontsize=12)
    ax2.set_ylabel('Edge F1 Score (Structural Recovery)', fontsize=12)
    ax2.set_title('Edge F1 vs Trace Informativeness', fontweight='bold', fontsize=13)
    ax2.set_xlim([0.55, 1.05])
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig(output_dir / 'f1_vs_ipcov.pdf')
    fig2.savefig(output_dir / 'f1_vs_ipcov.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_vs_ipcov.pdf'}")
    plt.close(fig2)
    
    # =====================================================================
    # FIGURE 2b: Aggregate Feasibility vs IP-Coverage
    # =====================================================================
    fig2b, ax2b = plt.subplots(figsize=(8, 5))
    
    # Plot BPOP across IP-Cov targets (no error bars)
    bpop_feas_grouped = df_bpop.groupby('ip_cov_target')['feas'].mean()
    ax2b.plot(bpop_feas_grouped.index, bpop_feas_grouped.values,
              marker=MARKERS['bhpop_single_po'], color=COLORS['bhpop_single_po'],
              label='BPOP (Ours)', markersize=12, linewidth=2.5, zorder=5)
    
    # Plot baselines across IP-Cov targets (no error bars)
    for method in baseline_methods:
        method_data = df_baselines[df_baselines['method'] == method]
        grouped = method_data.groupby('ip_cov_target')['feas'].mean()
        ax2b.plot(grouped.index, grouped.values,
                  marker=MARKERS[method], color=COLORS[method], 
                  label=METHOD_LABELS[method], linewidth=2, markersize=8)
    
    ax2b.set_xlabel('IP-Coverage Target', fontsize=12)
    ax2b.set_ylabel('Feasibility', fontsize=12)
    ax2b.set_title('Feasibility vs Trace Informativeness (Aggregate)', fontweight='bold', fontsize=13)
    ax2b.set_xlim([0.55, 1.05])
    ax2b.set_ylim([0, 1.1])
    ax2b.legend(loc='center right', fontsize=10, frameon=True)
    ax2b.grid(alpha=0.3)
    
    fig2b.tight_layout()
    fig2b.savefig(output_dir / 'feasibility_vs_ipcov.pdf')
    fig2b.savefig(output_dir / 'feasibility_vs_ipcov.png', dpi=300)
    print(f"Saved: {output_dir / 'feasibility_vs_ipcov.pdf'}")
    plt.close(fig2b)
    
    # =====================================================================
    # FIGURE 3: Edge F1 by Scenario (detailed breakdown)
    # =====================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, method in enumerate(methods_order):
        method_means, method_stds = [], []
        for scenario in scenarios:
            if method == 'bhpop_single_po':
                data = df_bpop[df_bpop['scenario'] == scenario]['cover_f1']
            else:
                data = df_baselines_06[(df_baselines_06['scenario'] == scenario) & 
                                        (df_baselines_06['method'] == method)]['cover_f1']
            method_means.append(data.mean())
            method_stds.append(data.std())
        
        offset = (i - 1.5) * width
        ax3.bar(x + offset, method_means, width, yerr=method_stds,
               label=METHOD_LABELS[method], color=COLORS[method], capsize=2,
               edgecolor='black', linewidth=0.3)
    
    ax3.set_ylabel('Edge F1 Score', fontsize=12)
    ax3.set_xlabel('Scenario', fontsize=12)
    ax3.set_title('Structural Recovery by Scenario (IP-Cov=0.6)', fontweight='bold', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{SCENARIO_SHORT[s]}\n({s.replace("_", " ")})' for s in scenarios], fontsize=8)
    ax3.legend(loc='upper right', ncol=2, fontsize=9)
    ax3.set_ylim([0, 1.15])
    ax3.grid(axis='y', alpha=0.3)
    
    fig3.tight_layout()
    fig3.savefig(output_dir / 'edge_f1_by_scenario.pdf')
    fig3.savefig(output_dir / 'edge_f1_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'edge_f1_by_scenario.pdf'}")
    plt.close(fig3)
    
    # =====================================================================
    # FIGURE 4: IP F1 by Scenario (BPOP only - baselines don't have this)
    # =====================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    
    ip_means, ip_stds = [], []
    for scenario in scenarios:
        data = df_bpop[df_bpop['scenario'] == scenario]['ip_f1'].dropna()
        ip_means.append(data.mean() if len(data) > 0 else 0)
        ip_stds.append(data.std() if len(data) > 0 else 0)
    
    bars = ax4.bar(x, ip_means, yerr=ip_stds, color=COLORS['bhpop_single_po'], 
                   capsize=4, width=0.6, edgecolor='black', linewidth=0.5)
    
    ax4.set_ylabel('IP F1 Score (Concurrency Recovery)', fontsize=12)
    ax4.set_xlabel('Scenario', fontsize=12)
    ax4.set_title('Concurrency Recovery by Scenario (BPOP)', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{SCENARIO_SHORT[s]}' for s in scenarios], fontsize=10)
    ax4.set_ylim([0, 1.15])
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(ip_means, ip_stds)):
        ax4.text(i, mean + std + 0.02, f'{mean:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_ip_f1 = np.mean(ip_means)
    ax4.axhline(y=avg_ip_f1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.text(len(scenarios)-0.5, avg_ip_f1 + 0.02, f'Avg={avg_ip_f1:.2f}', fontsize=9, color='red')
    
    fig4.tight_layout()
    fig4.savefig(output_dir / 'ip_f1_by_scenario.pdf')
    fig4.savefig(output_dir / 'ip_f1_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'ip_f1_by_scenario.pdf'}")
    plt.close(fig4)
    
    # =====================================================================
    # FIGURE 5: F1 vs IP-Cov by Scenario (6 subplots)
    # =====================================================================
    fig5, axes5 = plt.subplots(2, 3, figsize=(14, 8))
    axes5 = axes5.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes5[idx]
        
        # Baselines across IP-Cov (plot first so BPOP is on top)
        for method in baseline_methods:
            data = df_baselines[(df_baselines['scenario'] == scenario) & 
                               (df_baselines['method'] == method)]
            if len(data) > 0:
                grouped = data.groupby('ip_cov_target')['cover_f1'].mean()
                ax.plot(grouped.index, grouped.values, marker=MARKERS[method],
                       color=COLORS[method], label=METHOD_LABELS[method], linewidth=2, markersize=6)
        
        # BPOP across IP-Cov (plot last so it's on top)
        bpop_data = df_bpop[df_bpop['scenario'] == scenario]
        if len(bpop_data) > 0:
            grouped = bpop_data.groupby('ip_cov_target')['cover_f1'].mean()
            # Add small offset for S1 where BPOP and Majority both = 1.0
            y_offset = -0.03 if scenario == 'simple_ecs' else 0
            ax.plot(grouped.index, grouped.values + y_offset, marker=MARKERS['bhpop_single_po'],
                   color=COLORS['bhpop_single_po'], label=METHOD_LABELS['bhpop_single_po'], 
                   linewidth=2.5, markersize=12, zorder=10)
            # Add annotation for S1
            if scenario == 'simple_ecs':
                ax.annotate('BPOP=Majority=1.0', xy=(0.8, 1.0), fontsize=8, 
                           ha='center', va='bottom', color='gray')
        
        ax.set_xlabel('IP-Cov Target')
        ax.set_ylabel('Edge F1')
        ax.set_title(f'{SCENARIO_SHORT[scenario]}: {scenario.replace("_", " ")}', fontweight='bold', fontsize=11)
        ax.set_xlim([0.55, 1.05])
        ax.set_ylim([0, 1.15])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=2)
    
    fig5.suptitle('Edge F1 vs IP-Coverage by Scenario', fontsize=14, fontweight='bold')
    fig5.tight_layout()
    fig5.savefig(output_dir / 'f1_vs_ipcov_by_scenario.pdf')
    fig5.savefig(output_dir / 'f1_vs_ipcov_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'f1_vs_ipcov_by_scenario.pdf'}")
    plt.close(fig5)
    
    # =====================================================================
    # FIGURE 6: Feasibility vs IP-Cov by Scenario
    # =====================================================================
    fig6, axes6 = plt.subplots(2, 3, figsize=(14, 8))
    axes6 = axes6.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes6[idx]
        
        # Get BPOP and baseline data for this scenario
        bpop_data = df_bpop[df_bpop['scenario'] == scenario]
        bpop_feas = bpop_data.groupby('ip_cov_target')['feas'].mean() if len(bpop_data) > 0 else None
        
        # Baselines (with small offsets for overlapping)
        offsets = {'majority': -0.02, 'inductive_miner_imf': 0.02, 'heuristics_miner': 0.0}
        for method in baseline_methods:
            data = df_baselines[(df_baselines['scenario'] == scenario) & 
                               (df_baselines['method'] == method)]
            if len(data) > 0:
                grouped = data.groupby('ip_cov_target')['feas'].mean()
                # Apply offset only when values overlap at 1.0
                y_vals = grouped.values.copy()
                if bpop_feas is not None:
                    for i, ip_cov in enumerate(grouped.index):
                        if ip_cov in bpop_feas.index:
                            if abs(y_vals[i] - bpop_feas[ip_cov]) < 0.05:
                                y_vals[i] += offsets[method]
                ax.plot(grouped.index, y_vals, marker=MARKERS[method],
                       color=COLORS[method], label=METHOD_LABELS[method], linewidth=2, markersize=6)
        
        # BPOP (on top, with slight offset down when all methods = 1.0)
        if bpop_feas is not None and not bpop_feas.isna().all():
            y_vals = bpop_feas.values.copy()
            # For simple_ecs where all are 1.0, shift BPOP down slightly
            if scenario == 'simple_ecs':
                y_vals = y_vals - 0.04
                ax.annotate('BPOP=Maj.=Ind.=1.0', xy=(0.8, 1.0), fontsize=7, 
                           ha='center', va='bottom', color='gray')
            ax.plot(bpop_feas.index, y_vals, marker=MARKERS['bhpop_single_po'],
                   color=COLORS['bhpop_single_po'], label=METHOD_LABELS['bhpop_single_po'], 
                   linewidth=2.5, markersize=12, zorder=10)
        
        ax.set_xlabel('IP-Cov Target')
        ax.set_ylabel('Feasibility')
        ax.set_title(f'{SCENARIO_SHORT[scenario]}: {scenario.replace("_", " ")}', fontweight='bold', fontsize=11)
        ax.set_xlim([0.55, 1.05])
        ax.set_ylim([0, 1.15])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=2)
    
    fig6.suptitle('Feasibility vs IP-Coverage by Scenario', fontsize=14, fontweight='bold')
    fig6.tight_layout()
    fig6.savefig(output_dir / 'feasibility_by_scenario.pdf')
    fig6.savefig(output_dir / 'feasibility_by_scenario.png', dpi=300)
    print(f"Saved: {output_dir / 'feasibility_by_scenario.pdf'}")
    plt.close(fig6)
    
    # =====================================================================
    # Print Summary Tables
    # =====================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLES FOR PAPER")
    print("="*80)
    
    print("\n--- Table 1: Aggregate Performance at IP-Cov=1.0 ---")
    print(f"{'Method':<20} {'Edge F1':>12} {'IP F1':>12} {'SHD':>10} {'Feasibility':>12}")
    print("-"*70)
    
    for method in methods_order:
        if method == 'bhpop_single_po':
            data = df_bpop_10  # IP-Cov=1.0
        else:
            data = df_baselines_10[df_baselines_10['method'] == method]
        
        edge_f1 = f"{data['cover_f1'].mean():.3f}±{data['cover_f1'].std():.3f}"
        ip_f1 = f"{data['ip_f1'].mean():.3f}" if data['ip_f1'].notna().any() else "N/A"
        shd = f"{data['shd'].mean():.1f}±{data['shd'].std():.1f}"
        feas = f"{data['feas'].mean():.3f}" if data['feas'].notna().any() else "N/A"
        
        print(f"{METHOD_LABELS[method]:<20} {edge_f1:>12} {ip_f1:>12} {shd:>10} {feas:>12}")
    
    print("\n--- Table 2: BPOP by Scenario ---")
    print(f"{'Scenario':<25} {'Edge F1':>10} {'IP F1':>10} {'SHD':>8} {'n':>5}")
    print("-"*60)
    
    for scenario in scenarios:
        data = df_bpop[df_bpop['scenario'] == scenario]
        print(f"{scenario:<25} {data['cover_f1'].mean():>10.3f} {data['ip_f1'].mean():>10.3f} {data['shd'].mean():>8.1f} {len(data):>5}")
    
    # Save tables to CSV
    summary_rows = []
    for method in methods_order:
        if method == 'bhpop_single_po':
            data = df_bpop_10  # IP-Cov=1.0
        else:
            data = df_baselines_10[df_baselines_10['method'] == method]
        
        summary_rows.append({
            'Method': METHOD_LABELS[method],
            'Edge_F1_mean': data['cover_f1'].mean(),
            'Edge_F1_std': data['cover_f1'].std(),
            'IP_F1_mean': data['ip_f1'].mean() if data['ip_f1'].notna().any() else np.nan,
            'SHD_mean': data['shd'].mean(),
            'SHD_std': data['shd'].std(),
            'Feasibility_mean': data['feas'].mean() if data['feas'].notna().any() else np.nan,
        })
    
    pd.DataFrame(summary_rows).to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"\nSaved: {output_dir / 'summary_table.csv'}")
    
    print("\n" + "="*80)
    print(f"✅ All plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
