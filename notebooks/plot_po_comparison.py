#!/usr/bin/env python3
"""
Generate partial order comparison plots for ICML paper.
Shows true vs inferred partial orders with differences highlighted.
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import networkx as nx

# Configuration
SCENARIOS = ['simple_ecs', 'slb_ecs_rds', 'slb_ecs_redis', 
             'eip_slb_ecs', 'dual_zone_ecs_slb', 'dual_zone_ecs_slb_rds']

SCENARIO_TITLES = {
    'simple_ecs': 'S1: Simple ECS',
    'slb_ecs_rds': 'S2: SLB+ECS+RDS',
    'slb_ecs_redis': 'S3: SLB+ECS+Redis',
    'eip_slb_ecs': 'S4: EIP+SLB+ECS',
    'dual_zone_ecs_slb': 'S5: Dual Zone ECS+SLB',
    'dual_zone_ecs_slb_rds': 'S6: Dual Zone+RDS'
}

THRESHOLD = 1/3  # Marginal threshold for edge inference


def setup_style():
    """Configure matplotlib for publication plots."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_scenario_data(scenario_name):
    """Load scenario edges and task names."""
    scenario_file = Path(f'../aliyun_data/manual_scenarios/{scenario_name}.json')
    with open(scenario_file) as f:
        data = json.load(f)
    
    edges = data['edges']
    
    # Get unique tasks (sorted for consistent ordering)
    tasks = set()
    for e in edges:
        tasks.add(e[0])
        tasks.add(e[1])
    tasks = sorted(list(tasks))
    
    # Build adjacency matrix for true cover
    n = len(tasks)
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    true_cover = np.zeros((n, n), dtype=int)
    
    for src, dst in edges:
        true_cover[task_to_idx[src], task_to_idx[dst]] = 1
    
    return tasks, true_cover, edges


def load_inferred_H(scenario_name, threshold=THRESHOLD, target_ip_cov=1.0):
    """Load inferred H matrix from best BPOP experiment at specified IP-Cov."""
    exp_dir = Path('systematic_experiment_results')
    
    # Find experiment for this scenario at target IP-Cov
    best_exp = None
    best_f1 = -1
    
    for exp_path in exp_dir.glob(f'exp_*_{scenario_name}'):
        summary_file = exp_path / 'summary.json'
        if not summary_file.exists():
            continue
            
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Check if this is target IP-Cov
        ip_cov = summary.get('configuration', {}).get('ip_cov_target', 0)
        if abs(ip_cov - target_ip_cov) > 0.01:
            continue
        
        # Get F1 score
        f1 = summary.get('posterior', {}).get('cover_f1', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_exp = exp_path
    
    if best_exp is None:
        print(f"Warning: No experiment found for {scenario_name} at IP-Cov={target_ip_cov}")
        return None
    
    # Load avg_H
    avg_H_file = best_exp / 'avg_H.pkl'
    with open(avg_H_file, 'rb') as f:
        avg_H = pickle.load(f)
    
    # Threshold to get inferred cover
    inferred_cover = (np.array(avg_H) >= threshold).astype(int)
    np.fill_diagonal(inferred_cover, 0)  # Remove self-loops
    
    return inferred_cover


def compute_transitive_reduction(adj_matrix):
    """Compute transitive reduction of adjacency matrix."""
    n = adj_matrix.shape[0]
    # First compute transitive closure
    closure = adj_matrix.copy().astype(bool)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
    
    # Then compute reduction
    reduction = closure.copy()
    for i in range(n):
        for j in range(n):
            if closure[i, j]:
                for k in range(n):
                    if k != i and k != j and closure[i, k] and closure[k, j]:
                        reduction[i, j] = False
                        break
    
    return reduction.astype(int)


def shorten_task_name(name):
    """Moderately shorten task names - readable but compact."""
    replacements = {
        'CreateVpc': 'CreateVpc',
        'CreateVSwitch': 'CreateVSwitch',
        'CreateSecurityGroup': 'CreateSG',
        'AuthorizeSecurityGroup': 'AuthorizeSG',
        'RunInstances': 'RunInstances',
        'CreateLoadBalancer': 'CreateSLB',
        'CreateLoadBalancerHTTPListener': 'CreateListener',
        'StartLoadBalancerListener': 'StartListener',
        'AddBackendServers': 'AddBackend',
        'CreateDBInstance': 'CreateRDS',
        'CreateAccount': 'CreateAccount',
        'ModifySecurityIps': 'ModifySecIPs',
        'CreateInstance': 'CreateRedis',
        'DescribeInstanceAttribute': 'DescribeAttr',
        'AllocateEipAddress': 'AllocateEIP',
        'AssociateEipAddress': 'AssociateEIP',
    }
    return replacements.get(name, name)


def draw_partial_order(ax, tasks, true_cover, inferred_cover, title):
    """Draw partial order comparison graph with full task names."""
    n = len(tasks)
    full_names = [shorten_task_name(t) for t in tasks]
    
    # Create graph for layout
    G = nx.DiGraph()
    for i, name in enumerate(full_names):
        G.add_node(i, label=name)
    
    # Add all edges (true and inferred) for layout
    for i in range(n):
        for j in range(n):
            if true_cover[i, j] or inferred_cover[i, j]:
                G.add_edge(i, j)
    
    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Normalize positions
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1
        pos = {k: ((v[0] - x_min) / x_range, (v[1] - y_min) / y_range) for k, v in pos.items()}
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    
    # Draw edges with different colors
    edge_colors = {
        'correct': '#2ca02c',      # Green - correctly identified
        'missed': '#d62728',        # Red - in true but not inferred
        'false_pos': '#ff7f0e',    # Orange - in inferred but not true
    }
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            true_edge = true_cover[i, j] == 1
            inf_edge = inferred_cover[i, j] == 1
            
            if true_edge and inf_edge:
                color = edge_colors['correct']
                style = '-'
                alpha = 0.9
                lw = 1.5
            elif true_edge and not inf_edge:
                color = edge_colors['missed']
                style = '--'
                alpha = 0.8
                lw = 1.5
            elif not true_edge and inf_edge:
                color = edge_colors['false_pos']
                style = ':'
                alpha = 0.8
                lw = 1.5
            else:
                continue
            
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=color, 
                                      linestyle=style, lw=lw, alpha=alpha,
                                      shrinkA=18, shrinkB=18))
    
    # Draw nodes as rounded rectangles with full names
    for i, name in enumerate(full_names):
        x, y = pos[i]
        # Use appropriate font size
        fontsize = 7 if len(name) > 15 else 8
        
        # Draw text with white background box (larger padding)
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='black', linewidth=1.5)
        ax.text(x, y, name, ha='center', va='center', fontsize=fontsize, 
               fontweight='bold', zorder=4, bbox=bbox_props)
    
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.axis('off')
    ax.set_aspect('equal')


def generate_comparison_plot(target_ip_cov, output_dir):
    """Generate comparison plot for a specific IP-Cov level."""
    
    # Create figure with 2x3 subplots (larger to fit full names)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    stats = []
    
    for idx, scenario in enumerate(SCENARIOS):
        print(f"Processing {scenario} at IP-Cov={target_ip_cov}...")
        
        # Load data
        tasks, true_cover, _ = load_scenario_data(scenario)
        inferred_cover = load_inferred_H(scenario, target_ip_cov=target_ip_cov)
        
        if inferred_cover is None:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(SCENARIO_TITLES[scenario])
            continue
        
        # Compute transitive reduction for cleaner visualization
        true_reduction = compute_transitive_reduction(true_cover)
        inferred_reduction = compute_transitive_reduction(inferred_cover)
        
        # Draw comparison
        draw_partial_order(axes[idx], tasks, true_reduction, inferred_reduction, 
                          SCENARIO_TITLES[scenario])
        
        # Collect stats
        true_edges = np.sum(true_reduction)
        inf_edges = np.sum(inferred_reduction)
        tp = np.sum((true_reduction == 1) & (inferred_reduction == 1))
        fp = np.sum((true_reduction == 0) & (inferred_reduction == 1))
        fn = np.sum((true_reduction == 1) & (inferred_reduction == 0))
        stats.append((scenario, true_edges, inf_edges, tp, fp, fn))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#2ca02c', label='Correct (TP)'),
        mpatches.Patch(color='#d62728', label='Missed (FN)'),
        mpatches.Patch(color='#ff7f0e', label='False Pos (FP)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle(f'Inferred vs True Partial Orders (BPOP at IP-Cov={target_ip_cov}, threshold=1/3)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    # Save with IP-Cov in filename
    ip_cov_str = str(target_ip_cov).replace('.', '')
    fig.savefig(output_dir / f'po_comparison_ipcov{ip_cov_str}.pdf')
    fig.savefig(output_dir / f'po_comparison_ipcov{ip_cov_str}.png', dpi=300)
    print(f"\nSaved: {output_dir / f'po_comparison_ipcov{ip_cov_str}.pdf'}")
    plt.close(fig)
    
    return stats


def main():
    setup_style()
    
    output_dir = Path('systematic_experiment_results/plots')
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots for both IP-Cov levels
    for ip_cov in [1.0, 0.6]:
        print(f"\n{'='*60}")
        print(f"GENERATING PLOT FOR IP-Cov={ip_cov}")
        print(f"{'='*60}")
        
        stats = generate_comparison_plot(ip_cov, output_dir)
        
        # Print statistics
        print(f"\nEDGE STATISTICS (IP-Cov={ip_cov})")
        print("-"*60)
        print(f"{'Scenario':<25} {'True':<6} {'Inferred':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
        print("-"*60)
        for scenario, true_edges, inf_edges, tp, fp, fn in stats:
            print(f"{scenario:<25} {true_edges:<6} {inf_edges:<10} {tp:<5} {fp:<5} {fn:<5}")


if __name__ == '__main__':
    main()
