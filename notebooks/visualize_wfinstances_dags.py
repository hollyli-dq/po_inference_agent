#!/usr/bin/env python3
"""
Visualize WfInstances DAGs for Paper

Creates publication-quality DAG visualizations for:
1. SRASearch (22 tasks, 30 edges)
2. Epigenomics (41 tasks, 48 edges)

Output: PDF files suitable for inclusion in papers.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# Project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available")


def setup_paper_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def extract_task_type(name: str) -> str:
    """Extract task type from full task name."""
    # Pattern: tasktype_tasktype_..._IDxxxxxx or tasktype-build_IDxxxxxx
    if '_ID' in name:
        prefix = name.split('_ID')[0]
        # Get first part before underscore (the actual task type)
        parts = prefix.split('_')
        return parts[0]
    return name


def extract_task_id(name: str) -> str:
    """Extract numeric ID from task name."""
    if '_ID' in name:
        id_part = name.split('_ID')[1]
        # Remove leading zeros
        return str(int(id_part))
    return name


def load_srasearch_dag() -> Dict[str, Any]:
    """Load SRASearch DAG."""
    data_dir = PROJECT_ROOT / "data" / "wfinstances_srasearch"
    json_files = sorted(data_dir.glob("*.json"))
    
    with open(json_files[0]) as f:
        data = json.load(f)
    
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    task_ids = [t.get('name', t.get('id')) for t in tasks]
    n = len(task_ids)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    
    # Build adjacency from children
    adj = np.zeros((n, n), dtype=np.int8)
    for t in tasks:
        name = t.get('name', t.get('id'))
        if name not in task_to_idx:
            continue
        src = task_to_idx[name]
        for child in t.get('children', []):
            if child in task_to_idx:
                adj[src, task_to_idx[child]] = 1
    
    # Extract task types and numeric IDs
    task_types = [extract_task_type(t) for t in task_ids]
    numeric_ids = [extract_task_id(t) for t in task_ids]
    
    return {
        'name': 'SRASearch',
        'task_ids': task_ids,
        'task_types': task_types,
        'numeric_ids': numeric_ids,
        'adj': adj,
        'n': n,
        'num_edges': int(adj.sum()),
    }


def load_epigenomics_dag() -> Dict[str, Any]:
    """Load Epigenomics DAG."""
    data_dir = PROJECT_ROOT / "data" / "wfinstances_epigenomics"
    json_files = list(data_dir.glob("*.json"))
    
    with open(json_files[0]) as f:
        data = json.load(f)
    
    workflow = data.get('workflow', data)
    spec = workflow.get('specification', {})
    tasks = spec.get('tasks', [])
    
    task_ids = [t.get('name', t.get('id')) for t in tasks]
    n = len(task_ids)
    task_to_idx = {t: i for i, t in enumerate(task_ids)}
    
    # Build adjacency from children
    adj = np.zeros((n, n), dtype=np.int8)
    for t in tasks:
        name = t.get('name', t.get('id'))
        if name not in task_to_idx:
            continue
        src = task_to_idx[name]
        for child in t.get('children', []):
            if child in task_to_idx:
                adj[src, task_to_idx[child]] = 1
    
    # Extract task types and numeric IDs
    task_types = [extract_task_type(t) for t in task_ids]
    numeric_ids = [extract_task_id(t) for t in task_ids]
    
    return {
        'name': 'Epigenomics',
        'task_ids': task_ids,
        'task_types': task_types,
        'numeric_ids': numeric_ids,
        'adj': adj,
        'n': n,
        'num_edges': int(adj.sum()),
    }


def compute_dag_levels(adj: np.ndarray) -> List[int]:
    """Compute topological levels for each node (longest path from root)."""
    n = adj.shape[0]
    levels = [-1] * n
    
    def get_level(node):
        if levels[node] >= 0:
            return levels[node]
        
        parents = np.where(adj[:, node] > 0)[0]
        if len(parents) == 0:
            levels[node] = 0
        else:
            levels[node] = 1 + max(get_level(p) for p in parents)
        return levels[node]
    
    for i in range(n):
        get_level(i)
    
    return levels


def get_color_palette(n_colors: int) -> List[Tuple[float, ...]]:
    """Get a visually distinct color palette."""
    # Use tab10 or Set3 for distinct colors
    if n_colors <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.Set3
    
    return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]


def visualize_dag_clean(
    data: Dict[str, Any],
    ax: plt.Axes,
    figsize_hint: Tuple[int, int] = (10, 8),
):
    """Create clean DAG visualization with proper labels."""
    if not NETWORKX_AVAILABLE:
        ax.text(0.5, 0.5, "NetworkX not available", ha='center', va='center')
        return
    
    adj = data['adj']
    n = data['n']
    task_types = data['task_types']
    numeric_ids = data['numeric_ids']
    
    # Create graph
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j)
    
    # Compute levels for hierarchical layout
    levels = compute_dag_levels(adj)
    max_level = max(levels)
    
    # Group nodes by level
    level_nodes = {l: [] for l in range(max_level + 1)}
    for i, l in enumerate(levels):
        level_nodes[l].append(i)
    
    # Compute positions - hierarchical from top to bottom
    pos = {}
    for level, nodes in level_nodes.items():
        n_nodes = len(nodes)
        # Sort nodes within level by their numeric ID for consistency
        nodes_sorted = sorted(nodes, key=lambda x: int(numeric_ids[x]))
        for i, node in enumerate(nodes_sorted):
            # Spread nodes horizontally, centered
            x = (i - (n_nodes - 1) / 2) * 1.5
            # Vertical position: top to bottom
            y = -level * 2
            pos[node] = (x, y)
    
    # Set up colors by task type
    unique_types = sorted(set(task_types))
    colors = get_color_palette(len(unique_types))
    type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}
    node_colors = [type_to_color[task_types[i]] for i in range(n)]
    
    # Determine node size based on number of nodes
    if n <= 25:
        node_size = 600
        font_size = 8
    elif n <= 50:
        node_size = 400
        font_size = 7
    else:
        node_size = 300
        font_size = 6
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        arrowsize=12,
        arrowstyle='-|>',
        edge_color='#555555',
        alpha=0.6,
        width=1.0,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=15,
        min_target_margin=15,
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors='black',
        linewidths=1.0,
    )
    
    # Add labels - use numeric IDs for each node
    labels = {i: numeric_ids[i] for i in range(n)}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=font_size,
        font_weight='bold',
    )
    
    # Create legend with task types
    legend_handles = []
    for t in unique_types:
        count = sum(1 for tt in task_types if tt == t)
        handle = mpatches.Patch(
            facecolor=type_to_color[t],
            edgecolor='black',
            linewidth=0.5,
            label=f'{t} ({count})'
        )
        legend_handles.append(handle)
    
    # Position legend outside the graph
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=False,
    )
    
    # Set title
    ax.set_title(
        f"{data['name']} Workflow DAG\n({data['n']} tasks, {data['num_edges']} edges)",
        fontsize=12,
        fontweight='bold',
        pad=10
    )
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Adjust limits to include all nodes with padding
    x_coords = [pos[i][0] for i in range(n)]
    y_coords = [pos[i][1] for i in range(n)]
    x_margin = max(2, (max(x_coords) - min(x_coords)) * 0.15)
    y_margin = max(2, (max(y_coords) - min(y_coords)) * 0.1)
    ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)


def create_srasearch_figure(output_path: Path):
    """Create clean SRASearch DAG figure."""
    setup_paper_style()
    
    print("Loading SRASearch DAG...")
    data = load_srasearch_dag()
    
    # Print task type summary
    unique_types = sorted(set(data['task_types']))
    print(f"  Task types: {unique_types}")
    for t in unique_types:
        count = sum(1 for tt in data['task_types'] if tt == t)
        print(f"    - {t}: {count} tasks")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_dag_clean(data, ax)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_epigenomics_figure(output_path: Path):
    """Create clean Epigenomics DAG figure."""
    setup_paper_style()
    
    print("Loading Epigenomics DAG...")
    data = load_epigenomics_dag()
    
    # Print task type summary
    unique_types = sorted(set(data['task_types']))
    print(f"  Task types: {unique_types}")
    for t in unique_types:
        count = sum(1 for tt in data['task_types'] if tt == t)
        print(f"    - {t}: {count} tasks")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    visualize_dag_clean(data, ax)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_combined_figure(output_path: Path):
    """Create side-by-side DAG figure."""
    setup_paper_style()
    
    print("Loading DAGs...")
    srasearch = load_srasearch_dag()
    epigenomics = load_epigenomics_dag()
    
    # Create figure with two columns
    fig = plt.figure(figsize=(16, 8))
    
    # SRASearch on left
    ax1 = fig.add_subplot(1, 2, 1)
    visualize_dag_clean(srasearch, ax1)
    
    # Epigenomics on right
    ax2 = fig.add_subplot(1, 2, 2)
    visualize_dag_clean(epigenomics, ax2)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_all_figures(output_dir: Path):
    """Create all publication figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual figures
    create_srasearch_figure(output_dir / "srasearch_dag.pdf")
    create_epigenomics_figure(output_dir / "epigenomics_dag.pdf")
    
    # Combined figure
    create_combined_figure(output_dir / "combined_dags.pdf")
    
    # Summary statistics
    print("\nCreating statistics summary...")
    srasearch = load_srasearch_dag()
    epigenomics = load_epigenomics_dag()
    
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create table data
    sra_types = sorted(set(srasearch['task_types']))
    epi_types = sorted(set(epigenomics['task_types']))
    
    table_data = [
        ['Metric', 'SRASearch', 'Epigenomics'],
        ['Tasks', str(srasearch['n']), str(epigenomics['n'])],
        ['Edges', str(srasearch['num_edges']), str(epigenomics['num_edges'])],
        ['Task Types', str(len(sra_types)), str(len(epi_types))],
        ['Types', ', '.join(sra_types), ', '.join(epi_types[:4]) + '...'],
    ]
    
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.3, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title("Workflow Statistics", fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "workflow_stats.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'workflow_stats.pdf'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize WfInstances DAGs")
    parser.add_argument("--output_dir", type=str,
                       default=str(PROJECT_ROOT / "notebooks" / "wfinstances_dag_figures"),
                       help="Output directory for figures")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Visualizing WfInstances DAGs for Paper")
    print("=" * 60)
    
    create_all_figures(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ All figures generated!")
    print(f"   Output: {output_dir}")
    print("=" * 60)
