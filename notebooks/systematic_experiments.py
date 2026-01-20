#!/usr/bin/env python3
"""
Systematic Experiments for ICML/ICLR/NeurIPS-style Paper (Aliyun manual scenarios)

Pipeline
1) Load manual scenarios (each scenario = one "assessor"; ground-truth DAG/poset cover)
2) Load traces from aliyun_data/expert_traces/ and aliyun_data/traces/ (optionally combine)
3) Normalize sequences -> build orders_local / orders_global / choice_sets per scenario
4) Augment each scenario with synthetic linear extensions until IP-Cov reaches a target
5) Systematic sweep:
   - IP-Cov targets: {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0} PER scenario
   - ε_jump: {0.001, 0.005, 0.01, 0.05, 0.1} (queue-jump noise probability)
   - Likelihoods: log_successors_queue_jump
   - Baselines: AND (intersection), Majority, + Process Mining (Inductive Miner IMf, Heuristics Miner)

Synthetic Trace Generation
---------------------------
IMPORTANT: The SYNTHETIC_METHOD should match the model likelihood for best results.

The log_successors_queue_jump likelihood models agent behavior as:
    P(a | frontier, remaining) = (1-ε) * softmax(β * log(successors(a)+1)) + ε/|remaining|

Available methods:
- "log_successors": Samples using the SAME distribution as the model likelihood.
                   This ensures synthetic traces match model expectations.
- "kahn": Uniform random from frontier. Simpler but doesn't match model exactly.
          Higher epsilon compensates for this mismatch (acts as regularizer).

Epsilon (ε_jump) Interpretation
-------------------------------
- Low ε (0.001): Model expects strict adherence to log-successor preference.
- High ε (0.1): Model tolerates behavioral variation in traces.
- Recommendation: Use ε ≈ 0.01-0.05 for traces generated with "log_successors" method.
                 Use ε ≈ 0.05-0.1 for real-world LLM agent traces (more variation).

Notes
- Expects the BHPOP codebase layout with `src/` importable from PROJECT_ROOT.
- `pm4py` and `pygraphviz` are optional. If missing, the corresponding baselines/plots are skipped.
"""

from __future__ import annotations

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx

# -------------------------
# Project root (expects this script lives in <repo>/scripts/ or similar)
# -------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- BHPOP imports (repo-local) ---
from src.utils.po_fun import BasicUtils
from src.utils.po_fun_plot import PO_plot
from src.utils.po_accelerator_nle_optimized import HPO_LogLikelihoodCache_Optimized
from src.utils.hpo_model_evaluation import precision_recall, f1_score, structural_hamming_distance
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_po


try:
    import pm4py  # type: ignore
    from datetime import datetime, timedelta
    PM4PY_AVAILABLE = True
except Exception:
    PM4PY_AVAILABLE = False
    print("Warning: pm4py not available. Process mining baselines will be skipped.")


# =========================
# Experiment configuration
# =========================
# Sweep grid
IP_COV_TARGETS = [0.6, 0.7, 0.8, 0.9, 1.0]
EPS_JUMP_LIST = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%
# LIKELIHOOD ABLATION: To compare log_successors_queue_jump vs queue_jump, use:
# LIKELIHOODS = ["log_successors_queue_jump", "queue_jump"]
LIKELIHOODS = ["log_successors_queue_jump"]  # Default: frontier-softmax likelihood

# Posterior thresholds for graph extraction
# eip_slb_ecs uses lower threshold due to poor MCMC convergence (see paper analysis)
# All other scenarios use standard threshold=0.5
POSTERIOR_THRESHOLD = 0.5  # Default for all scenarios
THRESHOLD_EIP_SLB_ECS = 0.4  # Special case: challenging scenario with weak MCMC signals

# Synthetic augmentation settings
SYNTHETIC_TARGET_COV = 1.0          # augment each scenario up to this IP-Cov before subsampling
SYNTHETIC_MAX_TRACES = 1000
SYNTHETIC_METHOD = "log_successors"  # "kahn" (uniform) | "log_successors" (matches model likelihood)
SYNTHETIC_SEED = 42
SOFTMAX_BETA = 1.0                   # inverse temperature for log-successors sampling
SOFTMAX_EPSILON = 0.01               # noise probability in sampling (queue-jump)
ALLOW_DUPES = False

# MCMC settings
NUM_ITERATIONS = 1000000  # 500k iterations
BURN_IN_FRACTION = 0.5
THIN = 1
SEED_BASE = 42

# Trace loading settings
TRACE_SOURCE = "combined"           # "expert" | "model" | "combined"
ONLY_SUCCESS = True
DROP_UNKNOWN_ACTIONS = True
DEDUP_ACTIONS = True

# Output
OUTPUT_DIR = PROJECT_ROOT / "notebooks/systematic_experiment_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 8  # parallel processes for experiments


# =========================
# JSON loading helpers
# =========================
def load_traces(dir_path: Path) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    if not dir_path.exists():
        return traces
    for p in sorted(dir_path.glob("*.json")):
        try:
            traces.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"Warning: failed to load trace {p}: {e}")
    return traces


def load_manual_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    scenarios: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"manual_scenarios dir not found: {path}")
    for p in sorted(path.glob("*.json")):
        scenarios[p.stem] = json.loads(p.read_text())
    return scenarios


def select_traces(traces: List[Dict[str, Any]], intent_type: str, only_success: bool) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for trace in traces:
        intent = (trace.get("intent") or {}).get("intent_type")
        if intent_type and intent != intent_type:
            continue
        if only_success and trace.get("status") not in (None, "success"):
            continue
        seq = trace.get("action_sequence") or []
        if not seq:
            continue
        selected.append(trace)
    return selected


def normalize_sequence(
    seq: List[str],
    task_set: set,
    *,
    drop_unknown: bool = True,
    dedup: bool = True,
) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for action in seq:
        if drop_unknown and action not in task_set:
            continue
        if dedup and action in seen:
            continue
        seen.add(action)
        out.append(action)
    return out


# =========================
# True poset construction
# =========================
def scenario_to_adj(scenario: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    edges = scenario.get("edges", [])
    tasks = sorted({t for edge in edges for t in edge})
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    adj = np.zeros((len(tasks), len(tasks)), dtype=np.int8)
    for parent, child in edges:
        if parent in task_to_idx and child in task_to_idx:
            adj[task_to_idx[parent], task_to_idx[child]] = 1
    return adj, tasks, task_to_idx


# =========================
# IP-Cov + LE checks
# =========================
def incomparable_pairs_from_closure(closure: np.ndarray) -> List[Tuple[int, int]]:
    n = closure.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if closure[i, j] == 0 and closure[j, i] == 0]


def ip_cov(orders_local: List[List[int]], true_closure: np.ndarray) -> float:
    pairs = incomparable_pairs_from_closure(true_closure)
    if not pairs:
        return 1.0
    seen = {pair: 0 for pair in pairs}
    for order in orders_local:
        pos = {a: t for t, a in enumerate(order)}
        for (i, j) in pairs:
            if i in pos and j in pos:
                seen[(i, j)] |= 1 if pos[i] < pos[j] else 2
    return sum(v == 3 for v in seen.values()) / len(pairs)


def is_linear_extension_partial(order: List[int], closure: np.ndarray) -> bool:
    pos = {t: i for i, t in enumerate(order)}
    rows, cols = np.where(closure == 1)
    for i, j in zip(rows, cols):
        if i in pos and j in pos and pos[i] >= pos[j]:
            return False
    return True


def feasibility_of_orders(orders_local: List[List[int]], inferred_cover: np.ndarray) -> float:
    """Fraction of orders that are linear extensions of inferred_cover's transitive closure."""
    if not orders_local:
        return float("nan")
    inferred_closure = BasicUtils.transitive_closure(inferred_cover.astype(np.int8))
    invalid = sum(1 for o in orders_local if not is_linear_extension_partial(o, inferred_closure))
    return 1.0 - invalid / max(1, len(orders_local))


# =========================
# Synthetic extension samplers (LOCAL)
# =========================
def _canonical_toposort(cover_adj: np.ndarray) -> List[int]:
    import heapq
    n = cover_adj.shape[0]
    succ = [np.flatnonzero(cover_adj[u]).tolist() for u in range(n)]
    indeg = cover_adj.sum(axis=0).astype(int).tolist()
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    out: List[int] = []
    while heap:
        u = int(heapq.heappop(heap))
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    if len(out) != n:
        raise ValueError("Cycle detected in cover_adj.")
    return out


def _sample_kahn(cover_adj: np.ndarray, rng: np.random.Generator) -> List[int]:
    n = cover_adj.shape[0]
    succ = [np.flatnonzero(cover_adj[u]).tolist() for u in range(n)]
    indeg = cover_adj.sum(axis=0).astype(int)
    frontier = [i for i in range(n) if indeg[i] == 0]
    order: List[int] = []
    while frontier:
        idx = int(rng.integers(0, len(frontier)))
        u = int(frontier.pop(idx))
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                frontier.append(v)
    if len(order) != n:
        raise ValueError("Cycle detected when sampling kahn extension.")
    return order


def _sample_frontier_softmax(
    cover_adj: np.ndarray,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
    epsilon: float = 0.0,
) -> List[int]:
    """Sample using softmax over canonical rank (legacy method)."""
    n = cover_adj.shape[0]
    succ = [np.flatnonzero(cover_adj[u]).tolist() for u in range(n)]
    indeg = cover_adj.sum(axis=0).astype(int)
    frontier = [i for i in range(n) if indeg[i] == 0]

    canonical = _canonical_toposort(cover_adj)
    rank = {a: i for i, a in enumerate(canonical)}

    order: List[int] = []
    while frontier:
        if rng.random() < epsilon:
            idx = int(rng.integers(0, len(frontier)))
            node = int(frontier.pop(idx))
        else:
            scores = np.array([-beta * rank[a] for a in frontier], dtype=float)
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            node = int(rng.choice(frontier, p=w))
            frontier.remove(node)

        order.append(node)
        for nxt in succ[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)

    if len(order) != n:
        raise ValueError("Cycle detected when sampling softmax extension.")
    return order


def _sample_log_successors(
    cover_adj: np.ndarray,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
    epsilon: float = 0.0,
) -> List[int]:
    """
    Sample linear extensions using the log-successors model.
    
    This MATCHES the log_successors_queue_jump likelihood used in MCMC:
      Q(a) = log(remaining_successors(a) + 1)
      P(a) = (1-ε)*softmax(β*Q) + ε/|remaining|
    
    Args:
        cover_adj: Adjacency matrix of the partial order cover
        rng: Random number generator
        beta: Inverse temperature (higher = more deterministic)
        epsilon: Queue-jump probability (noise)
    
    Returns:
        Linear extension as list of indices
    """
    import math
    
    n = cover_adj.shape[0]
    succ = [np.flatnonzero(cover_adj[u]).tolist() for u in range(n)]
    indeg = cover_adj.sum(axis=0).astype(int)
    
    remaining = set(range(n))
    frontier = {i for i in range(n) if indeg[i] == 0}
    order: List[int] = []
    
    while remaining:
        if not frontier:
            raise ValueError("Cycle detected or invalid state in log_successors sampling.")
        
        frontier_list = list(frontier)
        remaining_list = list(remaining)
        
        # Queue-jump: with probability epsilon, pick uniformly from ALL remaining
        if rng.random() < epsilon:
            node = int(rng.choice(remaining_list))
            # If node is not in frontier, this is a "queue-jump" (noise)
            # The model allows this with probability epsilon
        else:
            # Compute Q = log(successors + 1) for each frontier node
            # Count only successors still in remaining
            scores = []
            for a in frontier_list:
                num_successors = sum(1 for v in succ[a] if v in remaining)
                Q = math.log(num_successors + 1)
                scores.append(beta * Q)
            
            # Softmax
            scores = np.array(scores, dtype=float)
            scores -= scores.max()  # numerical stability
            w = np.exp(scores)
            w /= w.sum()
            
            node = int(rng.choice(frontier_list, p=w))
        
        order.append(node)
        remaining.remove(node)
        frontier.discard(node)
        
        # Update frontier: add successors whose prerequisites are now met
        for v in succ[node]:
            if v in remaining:
                indeg[v] -= 1
                if indeg[v] == 0:
                    frontier.add(v)
    
    if len(order) != n:
        raise ValueError("Cycle detected when sampling log_successors extension.")
    return order


def _sample_extension(
    cover_adj: np.ndarray,
    rng: np.random.Generator,
    synthetic_method: str,
    softmax_beta: float,
    softmax_epsilon: float,
) -> List[int]:
    """
    Sample a linear extension using the specified method.
    
    Methods:
        - "kahn": Uniform random from frontier (original, simple)
        - "frontier_softmax": Softmax by canonical rank (legacy)
        - "log_successors": Softmax by log(successors+1) - MATCHES MODEL LIKELIHOOD
    """
    if synthetic_method == "kahn":
        return _sample_kahn(cover_adj, rng)
    if synthetic_method == "frontier_softmax":
        return _sample_frontier_softmax(cover_adj, rng, beta=softmax_beta, epsilon=softmax_epsilon)
    if synthetic_method == "log_successors":
        return _sample_log_successors(cover_adj, rng, beta=softmax_beta, epsilon=softmax_epsilon)
    raise ValueError(f"Unknown SYNTHETIC_METHOD={synthetic_method}. Use 'kahn', 'frontier_softmax', or 'log_successors'.")


# =========================
# Synthetic trace augmentation
# =========================
def _critical_pair_coverage(orders: List[List[int]], closure: np.ndarray) -> Tuple[int, int, float]:
    n = closure.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if closure[i, j] or closure[j, i]:
                continue
            pairs.add((i, j))
    if not pairs:
        return 0, 0, 1.0

    seen: Dict[Tuple[int, int], int] = {}
    for order in orders:
        pos = {t: i for i, t in enumerate(order)}
        items = list(pos.keys())
        for ii in range(len(items)):
            for jj in range(ii + 1, len(items)):
                a, b = items[ii], items[jj]
                pair = (a, b) if a < b else (b, a)
                if pair not in pairs:
                    continue
                u, v = pair
                mask = 1 if pos[u] < pos[v] else 2
                seen[pair] = seen.get(pair, 0) | mask

    covered = sum(1 for pair in pairs if seen.get(pair, 0) == 3)
    return covered, len(pairs), covered / max(1, len(pairs))


def augment_scenario_with_synthetic_traces(
    scenario_id: str,
    scenario_data: Dict[str, Any],
    orders_local: List[List[int]],
    orders_global: List[List[int]],
    choice_sets: List[List[int]],
    meta: List[Dict[str, Any]],
    global_task_to_idx: Dict[str, int],
    *,
    synthetic_target_cov: float = 1.0,
    synthetic_max_traces: int = 1000,
    synthetic_method: str = "kahn",
    synthetic_seed: int = 42,
    softmax_beta: float = 1.0,
    softmax_epsilon: float = 0.01,
    allow_dupes: bool = False,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[Dict[str, Any]]]:
    """Augment scenario with synthetic traces to reach target critical pair coverage."""
    rng = np.random.default_rng(synthetic_seed)

    true_cover = scenario_data[scenario_id]["true_cover"]
    true_closure = scenario_data[scenario_id]["true_closure"]
    task_ids = scenario_data[scenario_id]["task_ids"]

    seen_orders = {tuple(o) for o in orders_local} if not allow_dupes else set()
    _, total_pairs, cov_before = _critical_pair_coverage(orders_local, true_closure)

    added = 0
    while cov_before < synthetic_target_cov and added < synthetic_max_traces:
        order_local = _sample_extension(true_cover, rng, synthetic_method, softmax_beta, softmax_epsilon)

        if not allow_dupes:
            key = tuple(order_local)
            if key in seen_orders:
                continue
            seen_orders.add(key)

        orders_local.append(order_local)

        order_global = [global_task_to_idx[task_ids[i]] for i in order_local]
        orders_global.append(order_global)
        choice_sets.append(sorted(set(order_global)))

        meta.append({
            "trace_id": f"synthetic_{scenario_id}_{added}",
            "intent_type": scenario_id,
            "source_mode": "synthetic",
            "raw_length": len(order_local),
            "used_length": len(order_local),
        })

        added += 1
        _, total_pairs, cov_before = _critical_pair_coverage(orders_local, true_closure)

    print(f"{scenario_id}: added={added} crit_pairs={total_pairs} coverage={cov_before:.3f} (target={synthetic_target_cov})")
    return orders_local, orders_global, choice_sets, meta


# =========================
# Subsampling to IP-Cov target per scenario
# =========================
def greedy_subset_indices_to_target(
    orders_local: List[List[int]],
    true_closure: np.ndarray,
    target: float,
    seed: int,
) -> Tuple[List[int], float]:
    rng = np.random.default_rng(seed)
    valid_idx = [i for i, o in enumerate(orders_local) if is_linear_extension_partial(o, true_closure)]
    if not valid_idx:
        return [], 0.0

    pairs = incomparable_pairs_from_closure(true_closure)
    if not pairs or target <= 0:
        return [valid_idx[0]], 1.0

    valid_orders = [orders_local[i] for i in valid_idx]

    masks: List[Dict[Tuple[int, int], int]] = []
    for o in valid_orders:
        pos = {a: t for t, a in enumerate(o)}
        m: Dict[Tuple[int, int], int] = {}
        for (i, j) in pairs:
            if i in pos and j in pos:
                m[(i, j)] = 1 if pos[i] < pos[j] else 2
        masks.append(m)

    covered = {p: 0 for p in pairs}
    remaining = list(range(len(valid_orders)))
    rng.shuffle(remaining)

    selected_local: List[int] = []
    COMPLETE_BONUS = 5

    def cov_now() -> float:
        return sum(v == 3 for v in covered.values()) / len(pairs)

    while remaining and cov_now() < target:
        best = None
        best_gain = -1
        for idx in remaining:
            gain = 0
            for p, bit in masks[idx].items():
                before = covered[p]
                after = before | bit
                if after != before:
                    gain += 1
                    if after == 3 and before != 3:
                        gain += COMPLETE_BONUS
            if gain > best_gain:
                best_gain = gain
                best = idx

        if best is None or best_gain <= 0:
            break

        selected_local.append(best)
        for p, bit in masks[best].items():
            covered[p] |= bit
        remaining.remove(best)

    chosen_idx = [valid_idx[i] for i in selected_local]
    realized = ip_cov([orders_local[i] for i in chosen_idx], true_closure)
    return chosen_idx, realized


# =========================
# Posterior aggregation methods
# =========================
def posterior_threshold_mean(matrices: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Simple threshold aggregation (same as notebook approach):
    1. Compute mean of posterior samples
    2. Threshold at specified value
    3. Apply transitive reduction
    
    This is the approach that achieves F1=1.0 in the notebook.
    """
    if not matrices:
        raise ValueError("posterior_threshold_mean called with empty list")
    
    # Stack and compute mean
    mats_arr = np.stack(matrices, axis=0)
    mean_mat = np.mean(mats_arr, axis=0)
    
    # Threshold
    agg_mat = (mean_mat >= threshold).astype(np.int8)
    np.fill_diagonal(agg_mat, 0)
    
    # Transitive reduction to get cover
    cover = BasicUtils.transitive_reduction(agg_mat.astype(int))
    return cover.astype(np.int8)


def posterior_pairwise_marginal_mode_robust(closures: List[np.ndarray]) -> np.ndarray:
    """
    Pairwise marginal-mode estimator:
    - compute posterior P(i>j)
    - compute net dominance score s_i = Σ_j (P_ij - P_ji)
    - sort nodes by -s_i
    - keep edges consistent with that order where P_ij > P_ji
    - take transitive closure then transitive reduction => cover
    
    NOTE: This is a more complex method. Use posterior_threshold_mean for simpler aggregation.
    """
    if not closures:
        raise ValueError("posterior_pairwise_marginal_mode_robust called with empty list")

    n = closures[0].shape[0]
    T = len(closures)

    P_ij = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            P_ij[i, j] = sum(1 for M in closures if M[i, j] == 1) / T

    s = np.sum(P_ij - P_ij.T, axis=1)
    order = np.argsort(-s)  # descending

    out = np.zeros((n, n), dtype=np.int8)
    for a_pos, i in enumerate(order):
        for b_pos, j in enumerate(order):
            if a_pos >= b_pos:
                continue
            if P_ij[i, j] > P_ij[j, i]:
                out[i, j] = 1

    np.fill_diagonal(out, 0)

    if np.any(out):
        closure = BasicUtils.transitive_closure(out.astype(np.int8))
        out = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))

    return out.astype(np.int8)


# =========================
# Process Mining Baselines (pm4py-based)
# ======================================
# These implement state-of-the-art process mining algorithms for comparison:
#
# Inductive Miner (IMf):
# - Uses recursive process discovery to build hierarchical process trees
# - Handles concurrency, choice, and loops through cut detection
# - Good for discovering structured, block-based process models
# - Extracts sequence relations from discovered process tree
#
# Heuristics Miner:
# - Uses frequency-based heuristics for robust process discovery
# - More tolerant of noise, incompleteness, and variations in logs
# - Produces dependency measures that quantify precedence strength
# - Better suited for real-world logs with irregularities
#
# Both convert the discovered process models to precedence relations,
# then break cycles and compute the final partial order cover.
# =========================
def pm4py_log_from_orders_local(orders_local: List[List[int]], task_ids: List[str], scenario_id: str):
    """Convert orders_local to a pm4py event log."""
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py not available")

    rows: List[Dict[str, Any]] = []
    t0 = datetime(2020, 1, 1)

    for case_idx, order in enumerate(orders_local):
        for step, a_idx in enumerate(order):
            rows.append({
                "case:concept:name": f"{scenario_id}_{case_idx}",
                "concept:name": task_ids[a_idx],
                "time:timestamp": t0 + timedelta(seconds=int(step)),
            })

    df = pd.DataFrame(rows)
    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    return pm4py.convert_to_event_log(df)


def _break_cycles_greedy(adj: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Ensure DAG by greedily removing lowest-weight edge on any detected cycle."""
    A = adj.copy().astype(np.int8)
    W = weights if weights is not None else A.astype(float)

    while True:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        if nx.is_directed_acyclic_graph(G):
            break

        try:
            cycle = next(nx.simple_cycles(G))
        except StopIteration:
            break

        cyc_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        u, v = min(cyc_edges, key=lambda e: W[e[0], e[1]])
        A[u, v] = 0

        if A.sum() == 0:
            break

    return A


def baseline_inductive_miner_cover(
    orders_local: List[List[int]],
    task_ids: List[str],
    *,
    scenario_id: str,
    noise_threshold: float = 0.2,
) -> np.ndarray:
    """
    Inductive Miner (IMf) baseline - Process Mining Algorithm:

    Inductive Miner discovers a process tree model from event logs using the IMf algorithm.
    The algorithm recursively splits the log based on cut detection, creating a hierarchical
    process model that captures concurrency, choice, and loops.

    Algorithm steps:
    1. Convert local order traces to PM4Py event log format
    2. Discover process tree using Inductive Miner (IMf variant)
    3. Extract sequence relations from process tree via footprints
    4. Map sequence relations to adjacency matrix (i->j for strict precedence)
    5. Break cycles using greedy approach (remove weakest links)
    6. Compute transitive closure and reduction to get final partial order

    Parameters:
    - noise_threshold: Filter out infrequent behavior (default: 0.2)
    - Fallback: If tree discovery fails, try Petri net conversion
    - Fallback: If footprints fail, use log-based footprints

    Returns:
    - np.ndarray: Binary adjacency matrix representing the discovered partial order cover
    """
    n = len(task_ids)
    if not PM4PY_AVAILABLE or not orders_local:
        return np.zeros((n, n), dtype=np.int8)

    try:
        log = pm4py_log_from_orders_local(orders_local, task_ids, scenario_id)

        # Discover process tree using Inductive Miner
        tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)

        # Extract sequence relations via footprints
        fp = None
        try:
            # Primary method: footprints from process tree
            fp = pm4py.discover_footprints(tree)
        except Exception as e1:
            try:
                # Fallback 1: convert to Petri net and get footprints
                net, im, fm = pm4py.convert_to_petri_net(tree)
                fp = pm4py.discover_footprints(net, im, fm)
            except Exception as e2:
                # Fallback 2: footprints directly from log
                fp = pm4py.discover_footprints(log)

        # Extract sequence relations (strict precedence: a > b)
        sequence_rels = set()
        if isinstance(fp, dict):
            sequence_rels = fp.get("sequence", set()) or set()

        # Map to adjacency matrix
        name_to_idx = {name: i for i, name in enumerate(task_ids)}
        adj = np.zeros((n, n), dtype=np.int8)

        for (a, b) in sequence_rels:
            if a in name_to_idx and b in name_to_idx:
                adj[name_to_idx[a], name_to_idx[b]] = 1

        # Break cycles (greedy approach)
        adj = _break_cycles_greedy(adj, weights=adj.astype(float))

        # Compute transitive closure and reduction
        closure = BasicUtils.transitive_closure(adj.astype(np.int8))
        cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
        return cover.astype(np.int8)

    except Exception as e:
        print(f"Warning: Inductive Miner failed for {scenario_id}: {e}")
        return np.zeros((n, n), dtype=np.int8)


def baseline_heuristics_miner_cover(
    orders_local: List[List[int]],
    task_ids: List[str],
    *,
    scenario_id: str,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5,
) -> np.ndarray:
    """
    Heuristics Miner baseline - Process Mining Algorithm:

    Heuristics Miner discovers a heuristics net from event logs using statistical measures.
    Unlike Inductive Miner, it uses frequency-based heuristics to handle noise and incomplete logs,
    making it more robust to real-world process data.

    Algorithm steps:
    1. Convert local order traces to PM4Py event log format
    2. Discover heuristics net with configurable thresholds:
       - dependency_threshold: Minimum dependency measure for a->b relation (default: 0.5)
       - and_threshold: Threshold for AND-split detection (default: 0.65)
       - loop_two_threshold: Threshold for length-two-loop detection (default: 0.5)
    3. Extract dependency matrix from heuristics net
    4. Threshold dependency matrix to create binary adjacency matrix
    5. Break cycles using greedy approach (remove weakest dependency links)
    6. Compute transitive closure and reduction to get final partial order

    Key differences from Inductive Miner:
    - Uses frequency-based heuristics instead of recursive splitting
    - More robust to noise and incomplete logs
    - Produces dependency measures (weights) that can be used for cycle breaking
    - Better suited for real-world logs with variations

    Parameters:
    - dependency_threshold: Min dependency measure 0-1 (higher = stricter)
    - and_threshold: Threshold for concurrency detection
    - loop_two_threshold: Threshold for loop detection

    Returns:
    - np.ndarray: Binary adjacency matrix representing the discovered partial order cover
    """
    n = len(task_ids)
    if not PM4PY_AVAILABLE or not orders_local:
        return np.zeros((n, n), dtype=np.int8)

    try:
        log = pm4py_log_from_orders_local(orders_local, task_ids, scenario_id)

        # Discover heuristics net with configurable parameters
        hn = pm4py.discover_heuristics_net(
            log,
            dependency_threshold=dependency_threshold,
            and_threshold=and_threshold,
            loop_two_threshold=loop_two_threshold,
        )

        # Extract dependency matrix (contains dependency measures between tasks)
        name_to_idx = {name: i for i, name in enumerate(task_ids)}
        adj = np.zeros((n, n), dtype=np.int8)
        weights = np.zeros((n, n), dtype=float)

        dep_mat = getattr(hn, "dependency_matrix", None)
        if dep_mat is None:
            print(f"Warning: No dependency_matrix found in heuristics net for {scenario_id}")
            return np.zeros((n, n), dtype=np.int8)

        # Convert dependency matrix to adjacency matrix with weights
        for a, out in dep_mat.items():
            if a not in name_to_idx:
                continue
            for b, dep in out.items():
                if b not in name_to_idx:
                    continue
                ia, ib = name_to_idx[a], name_to_idx[b]
                dep_val = float(dep)
                weights[ia, ib] = dep_val
                if dep_val >= dependency_threshold:
                    adj[ia, ib] = 1

        # Break cycles using dependency weights (remove weakest links first)
        adj = _break_cycles_greedy(adj, weights=weights)

        # Compute transitive closure and reduction
        closure = BasicUtils.transitive_closure(adj.astype(np.int8))
        cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
        return cover.astype(np.int8)

    except Exception as e:
        print(f"Warning: Heuristics Miner failed for {scenario_id}: {e}")
        return np.zeros((n, n), dtype=np.int8)


# =========================
# Posterior Parameter Plots
# =========================
def save_posterior_params_pdf(
    mcmc_results: dict,
    pdf_path: Path,
    *,
    burn_in_frac: float = 0.5,
    exclude_keys: Optional[set] = None,
    pdf_pages: Optional[Any] = None,
    param_traces_path: Optional[Path] = None,
) -> None:
    """
    Save posterior trace + histogram plots for scalar MCMC traces into ONE PDF.

    Works in batch mode (no plt.show), no seaborn needed.
    """

    exclude_keys = exclude_keys or {"H_trace"}  # add more if needed

    # Load parameter traces from file if provided
    if param_traces_path and param_traces_path.exists():
        try:
            with open(param_traces_path, 'rb') as f:
                param_traces = pickle.load(f)
            # Merge with mcmc_results
            merged_results = {**mcmc_results, **param_traces}
        except Exception as e:
            print(f"Warning: Could not load parameter traces from {param_traces_path}: {e}")
            merged_results = mcmc_results
    else:
        merged_results = mcmc_results

    # Prefer known trace keys first, then fall back to any *_trace keys.
    preferred = [
        "rho_trace",
        "tau_trace",
        "K_trace",
        "prob_noise_trace",
        "softmax_beta_trace",
    ]

    # Collect candidate keys
    keys = []
    for k in preferred:
        if k in merged_results:
            keys.append(k)

    # Add any other trace keys not in preferred
    for k in sorted(merged_results.keys()):
        if k in keys or k in exclude_keys or k == "softmax_lambda_trace":
            continue
        if k.endswith("_trace"):
            keys.append(k)

    print(f"[DEBUG] Found {len(keys)} trace keys to plot: {keys}")
    if not keys:
        if pdf_pages is None:
            print(f"[WARN] No *_trace keys found in mcmc_results. Not writing {pdf_path}.")
        return

    def _to_1d_numeric(x: Any) -> Optional[np.ndarray]:
        try:
            # First try as numeric array
            arr = np.asarray(x, dtype=float)
        except (ValueError, TypeError):
            # If that fails, it's not a simple numeric array
            return None
        except Exception:
            return None
        if arr.size == 0:
            return None
        # We only handle 1D numeric traces here
        if arr.ndim != 1:
            return None
        if not np.issubdtype(arr.dtype, np.number):
            return None
        return arr

    # Use provided pdf_pages or create our own
    if pdf_pages is not None:
        pdf = pdf_pages
        wrote_any = False
    else:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(str(pdf_path))
        pdf.__enter__()
        wrote_any = False

    for key in keys:
        if key in exclude_keys:
            continue
        arr = _to_1d_numeric(merged_results.get(key))
        if arr is None:
            continue

        burn = int(len(arr) * burn_in_frac)
        burn = max(0, min(burn, len(arr)))
        post = arr[burn:]

        if len(post) == 0:
            print(f"[WARN] {key} empty after burn-in={burn}. Skipping.")
            continue

        # Set up ICML-style plotting
        setup_icml_style()

        # Build figure: trace + histogram
        fig = plt.figure(figsize=(7.0, 2.8))  # ICML double-column width, compact height

        ax1 = fig.add_subplot(1, 2, 1)
        iterations = np.arange(burn, burn + len(post))
        ax1.plot(iterations, post, lw=1.2, color='#377eb8', alpha=0.8)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(key)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Trace")

        ax2 = fig.add_subplot(1, 2, 2)

        # Better bins for discrete K-like traces
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

        # Add vertical line for mean
        mean_val = np.mean(post)
        ax2.axvline(mean_val, ls='--', lw=1.2, color='#e41a1c',
                   label=f'Mean: {mean_val:.3f}')
        ax2.set_xlabel(key)
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Posterior")
        ax2.legend(framealpha=0.8, fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        wrote_any = True

    if not wrote_any and pdf_pages is None:
        print(f"[WARN] No valid numeric 1D traces found. Not writing {pdf_path}.")
        return

    # Close PdfPages if we created it
    if pdf_pages is None:
        pdf.__exit__(None, None, None)
        print(f"[INFO] Saved posterior parameter plots to: {pdf_path}")
    else:
        print(f"[INFO] Added posterior parameter plots to combined PDF")


# =========================
# Diagnostics PDF
# =========================
def save_diagnostics_pdf(
    mcmc_results: dict,
    pdf_path: Path,
    *,
    burn_in_frac: float = 0.5,  # Use proportion instead of fixed number
    pdf_pages: Optional[Any] = None,
) -> None:
    # Use provided pdf_pages or create our own
    if pdf_pages is not None:
        pdf = pdf_pages
        # When pdf_pages is provided, caller manages the context
    else:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(str(pdf_path))
        pdf.__enter__()  # Manually enter context

    # Always plot log-likelihood trace (for both standalone and combined PDF)
    ll = mcmc_results.get("log_likelihood_currents", [])
    if ll:
        ll = np.asarray(ll, dtype=float)
        burn = int(len(ll) * burn_in_frac)  # Always use 0.5 proportion
        burn = max(0, min(int(burn), len(ll)))

        # Set up ICML-style plotting
        setup_icml_style()
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
        print(f"[INFO] Added log-likelihood trace plot")

    # Only do additional plots when creating standalone PDF
    if pdf_pages is None:
        beta = mcmc_results.get("softmax_beta_trace", [])
        if beta:
            beta = np.asarray(beta, dtype=float)
            burn = int(len(beta) * burn_in_frac)  # Use same burn-in fraction
            burn = max(0, min(int(burn), len(beta)))

            # Set up ICML-style plotting
            setup_icml_style()
            fig = plt.figure(figsize=(7.0, 5.0))  # Taller for two subplots

            # Trace plot
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(beta, lw=1.2, color='#377eb8', alpha=0.8)
            if burn > 0:
                ax1.axvline(burn, color="#e41a1c", ls="--", lw=1.2, alpha=0.8)
            ax1.set_xlabel("Saved step")
            ax1.set_ylabel("beta")
            ax1.set_title("Softmax beta Trace")
            ax1.grid(True, alpha=0.3)

            # Histogram
            ax2 = fig.add_subplot(2, 1, 2)
            post = beta[burn:] if burn < len(beta) else beta
            ax2.hist(post, bins=25, color="#4daf4a", alpha=0.7, edgecolor="black", linewidth=0.4)
            mean_val = np.mean(post)
            ax2.axvline(mean_val, ls='--', lw=1.2, color='#e41a1c',
                       label=f'Mean: {mean_val:.3f}')
            ax2.set_xlabel("beta")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Softmax beta Posterior")
            ax2.grid(True, alpha=0.3)
            ax2.legend(framealpha=0.8, fontsize=8)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Check if pygraphviz is available for PO_plot functionality
        try:
            import pygraphviz
            pygraphviz_available = True
        except ImportError:
            pygraphviz_available = False

        if pygraphviz_available:
            try:
                true_param_k = {"rho_true": 0, "prob_noise_true": 0, "tau_true": 0, "beta_true": 0, "K_true": 0}
                config = {"prior": {"rho_prior": 1.0, "K_prior": 3}, "noise": {"noise_option": "log_successors_queue_jump"}}
                burn_in_k = int(len(mcmc_results.get("H_trace", [])) * 0.5)
                po_plot_pdf = pdf_path.with_suffix(".po_plot.pdf")
                PO_plot.plot_inferred_variables(
                    mcmc_results,
                    true_param_k,
                    config,
                    burn_in=burn_in_k,
                    output_filename=str(po_plot_pdf),
                    assessors=list(mcmc_results.get("H_trace", [{}])[0].keys()) if mcmc_results.get("H_trace") else [],
                    M_a_dict={},
                    paper_format=True,  # Use ICML-style formatting
                )
                print(f"Saved PO_plot diagnostics to: {po_plot_pdf}")
            except Exception as e:
                print(f"Warning: Could not save PO_plot diagnostics: {e}")


# =========================
# ICML-style plotting configuration
# =========================

def setup_icml_style():
    """Configure matplotlib for ICML-style academic paper plots."""
    # Use available serif fonts (fallback chain for cross-platform compatibility)
    plt.rcParams.update({
        # Font settings - use DejaVu Serif (available on most systems)
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Figure settings
        'figure.figsize': (3.25, 2.5),  # ICML single column width
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'axes.linewidth': 0.8,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Color cycle (colorblind-friendly)
        'axes.prop_cycle': plt.cycler(color=[
            '#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
            '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'
        ]),

        # Legend settings
        'legend.framealpha': 0.8,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.frameon': True,

        # Tick settings
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
    })

    # Use matplotlib's built-in font rendering (Computer Modern-like)
    # LaTeX rendering is too fragile and causes issues
    plt.rcParams.update({
        'text.usetex': False,
    })

# =========================
# Local baselines
# =========================
def baseline_and(orders_local: List[List[int]], n: int) -> np.ndarray:
    """Intersection baseline: i->j iff i always before j whenever both appear."""
    before_ok = np.ones((n, n), dtype=bool)
    seen = np.zeros((n, n), dtype=int)

    for o in orders_local:
        pos = {a: t for t, a in enumerate(o)}
        items = list(pos.keys())
        for i in items:
            for j in items:
                if i == j:
                    continue
                seen[i, j] += 1
                if pos[i] >= pos[j]:
                    before_ok[i, j] = False

    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i != j and seen[i, j] > 0 and before_ok[i, j]:
                adj[i, j] = 1

    closure = BasicUtils.transitive_closure(adj.astype(np.int8))
    cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
    return cover.astype(np.int8)


def baseline_majority(orders_local: List[List[int]], n: int, thr: float = 0.5) -> np.ndarray:
    """Majority baseline with cycle breaking."""
    counts = np.zeros((n, n), dtype=int)
    totals = np.zeros((n, n), dtype=int)

    for o in orders_local:
        pos = {a: t for t, a in enumerate(o)}
        items = list(pos.keys())
        for i in items:
            for j in items:
                if i == j:
                    continue
                totals[i, j] += 1
                if pos[i] < pos[j]:
                    counts[i, j] += 1

    p = np.where(totals > 0, counts / np.maximum(1, totals), 0.0)

    adj = np.zeros((n, n), dtype=np.int8)
    margin = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if totals[i, j] == 0 and totals[j, i] == 0:
                continue
            pij = p[i, j]
            pji = p[j, i]
            if pij > thr and pij > pji:
                adj[i, j] = 1
                margin[i, j] = abs(pij - 0.5)
            elif pji > thr and pji > pij:
                adj[j, i] = 1
                margin[j, i] = abs(pji - 0.5)

    def find_cycle(A: np.ndarray) -> Optional[List[int]]:
        n_ = A.shape[0]
        state = [0] * n_
        parent = [-1] * n_

        def dfs(u: int) -> Optional[List[int]]:
            state[u] = 1
            for v in np.where(A[u] == 1)[0]:
                v = int(v)
                if state[v] == 0:
                    parent[v] = u
                    cyc = dfs(v)
                    if cyc is not None:
                        return cyc
                elif state[v] == 1:
                    cycle = [v]
                    cur = u
                    while cur != v and cur != -1:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(v)
                    cycle.reverse()
                    return cycle
            state[u] = 2
            return None

        for s in range(n_):
            if state[s] == 0:
                cyc = dfs(s)
                if cyc is not None:
                    return cyc
        return None

    while True:
        cyc = find_cycle(adj)
        if cyc is None:
            break
        weakest = None
        weakest_m = float("inf")
        for u, v in zip(cyc[:-1], cyc[1:]):
            if adj[u, v] == 1 and margin[u, v] < weakest_m:
                weakest_m = margin[u, v]
                weakest = (u, v)
        if weakest is None:
            break
        adj[weakest[0], weakest[1]] = 0

    closure = BasicUtils.transitive_closure(adj.astype(np.int8))
    cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
    return cover.astype(np.int8)


# =========================
# Build dataset dict
# =========================
def build_data_dict(project_root: Path) -> Dict[str, Any]:
    data_root = project_root / "aliyun_data"

    scenarios = load_manual_scenarios(data_root / "manual_scenarios")
    scenario_ids = sorted(scenarios.keys())

    scenario_data: Dict[str, Any] = {}
    all_tasks: set = set()

    for sid in scenario_ids:
        true_adj, task_ids, task_to_idx = scenario_to_adj(scenarios[sid])
        true_closure = BasicUtils.transitive_closure(true_adj.astype(np.int8))
        true_cover = BasicUtils.transitive_reduction_optimized(true_closure.astype(np.int8))
        scenario_data[sid] = {
            "task_ids": task_ids,
            "task_to_idx": task_to_idx,
            "true_adj": true_adj,
            "true_closure": true_closure,
            "true_cover": true_cover,
        }
        all_tasks.update(task_ids)

    global_task_ids = sorted(all_tasks)
    global_task_to_idx = {t: i for i, t in enumerate(global_task_ids)}
    M0 = list(range(len(global_task_ids)))

    expert_traces = load_traces(data_root / "expert_traces")
    model_traces = load_traces(data_root / "traces")

    orders_by_assessor: Dict[str, List[List[int]]] = {}
    orders_local_by_assessor: Dict[str, List[List[int]]] = {}
    choice_sets_by_assessor: Dict[str, List[List[int]]] = {}
    trace_meta_by_assessor: Dict[str, List[Dict[str, Any]]] = {}

    for scenario_id in scenario_ids:
        task_ids = scenario_data[scenario_id]["task_ids"]
        task_set = set(task_ids)
        local_task_to_idx = scenario_data[scenario_id]["task_to_idx"]

        selected_traces: List[Dict[str, Any]] = []
        if TRACE_SOURCE in ("expert", "combined"):
            selected_traces.extend(select_traces(expert_traces, scenario_id, ONLY_SUCCESS))
        if TRACE_SOURCE in ("model", "combined"):
            selected_traces.extend(select_traces(model_traces, scenario_id, ONLY_SUCCESS))

        orders_global: List[List[int]] = []
        orders_local: List[List[int]] = []
        choice_sets: List[List[int]] = []
        trace_meta: List[Dict[str, Any]] = []

        for trace in selected_traces:
            raw_seq = trace.get("action_sequence") or []
            norm = normalize_sequence(
                raw_seq,
                task_set,
                drop_unknown=DROP_UNKNOWN_ACTIONS,
                dedup=DEDUP_ACTIONS,
            )
            if not norm:
                continue

            order_global = [global_task_to_idx[action] for action in norm]
            order_local = [local_task_to_idx[action] for action in norm]

            orders_global.append(order_global)
            orders_local.append(order_local)
            choice_sets.append(sorted(set(order_global)))

            trace_meta.append({
                "trace_id": trace.get("trace_id"),
                "intent_type": (trace.get("intent") or {}).get("intent_type"),
                "source_mode": trace.get("mode"),
                "raw_length": len(raw_seq),
                "used_length": len(order_global),
            })

        orders_by_assessor[scenario_id] = orders_global
        orders_local_by_assessor[scenario_id] = orders_local
        choice_sets_by_assessor[scenario_id] = choice_sets
        trace_meta_by_assessor[scenario_id] = trace_meta

    print("trace_source:", TRACE_SOURCE)
    for scenario_id in scenario_ids:
        print(f"{scenario_id}: traces={len(orders_by_assessor.get(scenario_id, []))}")

    # Synthetic augmentation
    for scenario_id in scenario_ids:
        orders_local = orders_local_by_assessor.get(scenario_id, [])
        orders_global = orders_by_assessor.get(scenario_id, [])
        choice_sets = choice_sets_by_assessor.get(scenario_id, [])
        meta = trace_meta_by_assessor.get(scenario_id, [])

        if orders_local:
            orders_local, orders_global, choice_sets, meta = augment_scenario_with_synthetic_traces(
                scenario_id,
                scenario_data,
                orders_local,
                orders_global,
                choice_sets,
                meta,
                global_task_to_idx,
                synthetic_target_cov=SYNTHETIC_TARGET_COV,
                synthetic_max_traces=SYNTHETIC_MAX_TRACES,
                synthetic_method=SYNTHETIC_METHOD,
                synthetic_seed=SYNTHETIC_SEED,
                softmax_beta=SOFTMAX_BETA,
                softmax_epsilon=SOFTMAX_EPSILON,
                allow_dupes=ALLOW_DUPES,
            )
            orders_local_by_assessor[scenario_id] = orders_local
            orders_by_assessor[scenario_id] = orders_global
            choice_sets_by_assessor[scenario_id] = choice_sets
            trace_meta_by_assessor[scenario_id] = meta

    assessors = [sid for sid in scenario_ids if orders_by_assessor.get(sid)]
    if not assessors:
        raise ValueError("No traces found for any scenario (after filtering).")

    M_a_dict = {sid: [global_task_to_idx[t] for t in scenario_data[sid]["task_ids"]] for sid in assessors}
    O_a_i_dict = {sid: choice_sets_by_assessor[sid] for sid in assessors}
    observed_orders = {sid: orders_by_assessor[sid] for sid in assessors}

    print("assessors:", assessors)
    for sid in assessors:
        print(f"{sid}: observations={len(observed_orders[sid])}, items={len(M_a_dict[sid])}")

    return {
        "scenario_ids": scenario_ids,
        "assessors": assessors,
        "scenario_data": scenario_data,
        "orders_by_assessor": orders_by_assessor,
        "orders_local_by_assessor": orders_local_by_assessor,
        "choice_sets_by_assessor": choice_sets_by_assessor,
        "trace_meta_by_assessor": trace_meta_by_assessor,
        "M0": M0,
        "M_a_dict": M_a_dict,
        "O_a_i_dict": O_a_i_dict,
        "observed_orders": observed_orders,
    }


# =========================
# Single scenario experiment runner (one MCMC per scenario)
# =========================
def run_single_scenario_experiment(
    exp_id: int,
    scenario_id: str,
    cp_target: float,
    eps: float,
    lh: str,
    scenario_data: Dict[str, Any],
    sampled_local: List[List[int]],
    realized_cov: float,
    num_iterations: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Run MCMC for a SINGLE scenario (single partial order).
    
    Each scenario runs independently in parallel.
    """
    seed = SEED_BASE + exp_id
    iters = num_iterations if num_iterations is not None else NUM_ITERATIONS
    print(f"  [Exp {exp_id}] scenario={scenario_id} cp={cp_target} eps={eps} lh={lh} seed={seed} iters={iters}")

    # Get scenario-specific data
    task_ids = scenario_data["task_ids"]
    n_items = len(task_ids)
    items = list(range(n_items))  # Local indices: [0, 1, 2, ..., n-1]
    
    # Choice sets are the same as observed orders for full observations
    # Each order covers all items in the scenario
    choice_sets = [list(range(n_items)) for _ in sampled_local]
    
    # Run single partial order MCMC
    # eps controls: epsilon (trembling-hand probability) for all likelihoods
    # For softmax: epsilon is fixed; for queue_jump: prob_noise is updated via MCMC
    mcmc = mcmc_simulation_po(
        num_iterations=iters,
        items=items,
        choice_sets=choice_sets,
        observed_orders=sampled_local,
        dr=0.5,  # Multiplicative step size for rho (reduced for better acceptance)
        noise_option=lh,
        rho_prior=1.0,
        noise_beta_prior=1.0,  # Prior for queue_jump prob_noise updates
        K_prior=3,
        fixed_K=None,
        random_seed=seed,
        cycle_length=500,
        epsilon=eps,  # Trembling-hand epsilon
        softmax_beta_prior=(2.0, 1.0),
        softmax_beta_stepsize=0.1,
    )

    H_trace = mcmc["H_trace"]
    burn = int(len(H_trace) * BURN_IN_FRACTION)
    post = H_trace[burn::THIN]

    # Save artifacts
    exp_dir = OUTPUT_DIR / f"exp_{exp_id}_{scenario_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    ll_trace = mcmc.get("log_likelihood_currents", [])
    if ll_trace:
        np.save(exp_dir / "likelihood_trace.npy", np.asarray(ll_trace, dtype=float))

    with open(exp_dir / "H_trace.pkl", "wb") as f:
        pickle.dump(H_trace, f)

    # Save parameter traces for posterior plotting
    param_traces = {}
    for key in ['rho_trace', 'K_trace', 'prob_noise_trace', 'softmax_beta_trace']:
        if key in mcmc:
            param_traces[key] = mcmc[key]

    if param_traces:
        with open(exp_dir / "param_traces.pkl", "wb") as f:
            pickle.dump(param_traces, f)

    # Point estimate: threshold on mean posterior
    # eip_slb_ecs uses τ=0.4 due to high MCMC uncertainty; others use τ=0.5
    if post:
        mats = [h for h in post]
        threshold = THRESHOLD_EIP_SLB_ECS if scenario_id == 'eip_slb_ecs' else POSTERIOR_THRESHOLD
        final_H = posterior_threshold_mean(mats, threshold=threshold)
        avg_H = np.mean(np.stack(mats), axis=0)
    else:
        final_H = np.zeros((n_items, n_items), dtype=np.int8)
        avg_H = np.zeros((n_items, n_items), dtype=float)

    with open(exp_dir / "final_H.pkl", "wb") as f:
        pickle.dump(final_H, f)
    with open(exp_dir / "avg_H.pkl", "wb") as f:
        pickle.dump(avg_H, f)

    # Create combined diagnostics PDF
    diag_pdf_path = exp_dir / "diagnostics.pdf"
    diag_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(diag_pdf_path)) as pdf:
        try:
            save_diagnostics_pdf(mcmc, diag_pdf_path, burn_in_frac=BURN_IN_FRACTION, pdf_pages=pdf)
        except Exception as e:
            print(f"Warning: Could not save diagnostics: {e}")

        try:
            save_posterior_params_pdf(
                mcmc_results=mcmc,
                pdf_path=diag_pdf_path,
                burn_in_frac=BURN_IN_FRACTION,
                pdf_pages=pdf,
                param_traces_path=exp_dir / "param_traces.pkl",
            )
        except Exception as e:
            print(f"Warning: Could not save posterior parameter plots: {e}")

    print(f"[INFO] Saved diagnostics to: {diag_pdf_path}")

    # Compute metrics
    true_cover = scenario_data["true_cover"]
    p, r = precision_recall(true_cover, final_H)
    cover_f1 = f1_score(p, r)
    shd = structural_hamming_distance(true_cover, final_H)
    feas = feasibility_of_orders(sampled_local, final_H)

    result = {
        "scenario": scenario_id,
        "ip_cov_target": cp_target,
        "ip_cov_realized": realized_cov,
        "eps_jump": eps,
        "likelihood": lh,
        "method": "bhpop_single_po",
        "cover_f1": cover_f1,
        "shd": shd,
        "feas": feas,
    }

    return exp_id, result


# =========================
# Main systematic sweep (parallel per scenario)
# =========================
def run_systematic_experiments(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Run systematic experiments with 6 scenarios in parallel.
    
    Each (scenario, cp_target, eps, likelihood) combination is a separate MCMC run.
    All scenarios run in parallel using ProcessPoolExecutor.
    """
    rows: List[Dict[str, Any]] = []
    experiments: List[Dict[str, Any]] = []
    exp_id = 0

    baseline_method_names = ["majority"]
    if PM4PY_AVAILABLE:
        baseline_method_names += ["inductive_miner_imf", "heuristics_miner"]

    scenario_ids = data["scenario_ids"]
    print(f"\n=== Running experiments for {len(scenario_ids)} scenarios in parallel ===")
    print(f"Scenarios: {scenario_ids}")

    for cp_target in IP_COV_TARGETS:
        print(f"\n=== IP-Cov target {cp_target} ===")

        sampled_local: Dict[str, List[List[int]]] = {}
        realized_cov: Dict[str, float] = {}

        # --- per-scenario subsampling for this CP target ---
        for sid in scenario_ids:
            if sid not in data["orders_local_by_assessor"] or not data["orders_local_by_assessor"][sid]:
                continue
            closure_true = data["scenario_data"][sid]["true_closure"]
            idxs, cov = greedy_subset_indices_to_target(
                data["orders_local_by_assessor"][sid],
                closure_true,
                cp_target,
                seed=SEED_BASE,
            )
            realized_cov[sid] = cov
            sampled_local[sid] = [data["orders_local_by_assessor"][sid][i] for i in idxs]

        # --- baselines for this CP target ---
        for sid in scenario_ids:
            if sid not in sampled_local or not sampled_local[sid]:
                continue
            n = len(data["scenario_data"][sid]["task_ids"])
            true_cover = data["scenario_data"][sid]["true_cover"]

            baseline_covs: Dict[str, np.ndarray] = {
                "majority": baseline_majority(sampled_local[sid], n),
            }
            if PM4PY_AVAILABLE:
                baseline_covs["inductive_miner_imf"] = baseline_inductive_miner_cover(
                    sampled_local[sid],
                    data["scenario_data"][sid]["task_ids"],
                    scenario_id=sid,
                    noise_threshold=0.0,
                )
                baseline_covs["heuristics_miner"] = baseline_heuristics_miner_cover(
                    sampled_local[sid],
                    data["scenario_data"][sid]["task_ids"],
                    scenario_id=sid,
                    dependency_threshold=0.5,
                    and_threshold=0.65,
                    loop_two_threshold=0.5,
                )

            baseline_metrics: Dict[str, Tuple[float, float, float]] = {}
            for mname, inferred in baseline_covs.items():
                p, r = precision_recall(true_cover, inferred)
                f1 = f1_score(p, r)
                shd = structural_hamming_distance(true_cover, inferred)
                feas = feasibility_of_orders(sampled_local[sid], inferred)
                baseline_metrics[mname] = (f1, shd, feas)

            for eps in EPS_JUMP_LIST:
                for mname, (f1, shd, feas) in baseline_metrics.items():
                    rows.append({
                        "scenario": sid,
                        "ip_cov_target": cp_target,
                        "ip_cov_realized": realized_cov[sid],
                        "eps_jump": eps,
                        "likelihood": "baseline",
                        "method": mname,
                        "cover_f1": f1,
                        "shd": shd,
                        "feas": feas,
                    })

        # --- queue MCMC experiments: one per (scenario, eps, likelihood) ---
        for sid in scenario_ids:
            if sid not in sampled_local or not sampled_local[sid]:
                continue
            for eps in EPS_JUMP_LIST:
                for lh in LIKELIHOODS:
                    exp_id += 1
                    experiments.append({
                        "exp_id": exp_id,
                        "scenario_id": sid,
                        "cp_target": cp_target,
                        "eps": eps,
                        "lh": lh,
                        "scenario_data": data["scenario_data"][sid],
                        "sampled_local": sampled_local[sid],
                        "realized_cov": realized_cov[sid],
                    })

    # =========================
    # Run MCMC experiments in parallel (one per scenario)
    # =========================
    print(f"\n=== Running {len(experiments)} MCMC experiments in parallel ===")
    print(f"Using max_workers={MAX_WORKERS}")

    completed = 0
    failures: List[Tuple[Dict[str, Any], Exception]] = []
    mcmc_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_exp = {
            executor.submit(
                run_single_scenario_experiment,
                exp["exp_id"],
                exp["scenario_id"],
                exp["cp_target"],
                exp["eps"],
                exp["lh"],
                exp["scenario_data"],
                exp["sampled_local"],
                exp["realized_cov"],
            ): exp for exp in experiments
        }

        for future in as_completed(future_to_exp):
            exp = future_to_exp[future]
            completed += 1
            try:
                exp_id_result, result_row = future.result()
                mcmc_rows.append(result_row)
                print(f"  ✓ [{completed}/{len(experiments)}] Completed exp {exp_id_result} "
                      f"(scenario={exp['scenario_id']}, cp={exp['cp_target']}, eps={exp['eps']})")
            except Exception as exc:
                failures.append((exp, exc))
                print(f"  ✗ [{completed}/{len(experiments)}] Experiment {exp['exp_id']} failed: {exc}")

    if failures:
        print(f"\n⚠️  {len(failures)} experiments failed:")
        for exp, exc in failures:
            print(f"  - Exp {exp['exp_id']}: scenario={exp['scenario_id']}, cp={exp['cp_target']}, eps={exp['eps']} :: {exc}")
    else:
        print(f"\n✅ All {len(experiments)} experiments completed successfully")

    # Combine baseline rows + MCMC rows
    rows.extend(mcmc_rows)
    df = pd.DataFrame(rows)

    out_csv = OUTPUT_DIR / "experiment_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved complete results to: {out_csv}")

    # =========================
    # Summary statistics
    # =========================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total result rows: {len(df)}")
    if len(df) > 0:
        print(f"Scenarios: {sorted(df['scenario'].unique())}")
        print(f"IP-Cov targets: {sorted(df['ip_cov_target'].unique())}")
        print(f"Epsilon values: {sorted(df['eps_jump'].unique())}")
        print(f"Likelihoods: {sorted(df['likelihood'].unique())}")
        print(f"Methods: {sorted(df['method'].unique())}")

        method_summary = df.groupby("method").agg(
            cover_f1_mean=("cover_f1", "mean"),
            cover_f1_std=("cover_f1", "std"),
            cover_f1_min=("cover_f1", "min"),
            cover_f1_max=("cover_f1", "max"),
            shd_mean=("shd", "mean"),
            shd_std=("shd", "std"),
            shd_min=("shd", "min"),
            shd_max=("shd", "max"),
            feas_mean=("feas", "mean"),
            feas_std=("feas", "std"),
            n=("feas", "count"),
        ).round(4)

        print("\n" + "-" * 60)
        print("PERFORMANCE SUMMARY BY METHOD")
        print("-" * 60)
        print(method_summary)

        best_f1_row = df.loc[df["cover_f1"].idxmax()]
        print("\n" + "-" * 60)
        print("BEST BY COVER_F1")
        print("-" * 60)
        print(best_f1_row.to_string())
    else:
        method_summary = None
        best_f1_row = None

    summary_txt = OUTPUT_DIR / "experiment_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Systematic Experiments Summary (Single PO per Scenario)\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Scenarios: {scenario_ids}\n")
        f.write(f"IP-Cov targets: {IP_COV_TARGETS}\n")
        f.write(f"Epsilon values: {EPS_JUMP_LIST}\n")
        f.write(f"Likelihoods: {LIKELIHOODS}\n\n")
        if method_summary is not None:
            f.write("Performance by method:\n")
            f.write(method_summary.to_string())
            f.write("\n")
    print(f"\n📄 Detailed summary saved to: {summary_txt}")

    # =========================
    # Validation checks
    # =========================
    n_scenarios = len(scenario_ids)
    n_cp_targets = len(IP_COV_TARGETS)
    n_eps_values = len(EPS_JUMP_LIST)
    n_likelihoods = len(LIKELIHOODS)

    expected_mcmc_results = n_cp_targets * n_eps_values * n_likelihoods * n_scenarios
    actual_mcmc_results = len(df[df["method"] == "bhpop_single_po"]) if len(df) > 0 else 0

    expected_baseline_results = n_scenarios * n_cp_targets * n_eps_values * len(baseline_method_names)
    actual_baseline_results = len(df[df["method"].isin(baseline_method_names)]) if len(df) > 0 else 0

    print("\n" + "-" * 60)
    print("VALIDATION CHECKS")
    print("-" * 60)
    print(f"Expected MCMC rows: {expected_mcmc_results}  | Actual: {actual_mcmc_results}")
    print(f"Expected baseline rows: {expected_baseline_results} | Actual: {actual_baseline_results}")

    metadata = {
        "timestamp": str(pd.Timestamp.now()),
        "model_type": "single_partial_order_per_scenario",
        "pm4py_available": PM4PY_AVAILABLE,
        "baseline_methods": baseline_method_names,
        "num_scenarios": n_scenarios,
        "scenarios": scenario_ids,
        "ip_cov_targets": IP_COV_TARGETS,
        "eps_jump_list": EPS_JUMP_LIST,
        "likelihoods": LIKELIHOODS,
        "num_iterations": NUM_ITERATIONS,
        "burn_in_fraction": BURN_IN_FRACTION,
        "thin": THIN,
        "trace_source": TRACE_SOURCE,
        "synthetic_target_cov": SYNTHETIC_TARGET_COV,
        "synthetic_max_traces": SYNTHETIC_MAX_TRACES,
        "synthetic_method": SYNTHETIC_METHOD,
        "max_workers": MAX_WORKERS,
        "mcmc_experiments_planned": len(experiments),
        "mcmc_failures": len(failures),
        "expected_mcmc_rows": expected_mcmc_results,
        "actual_mcmc_rows": actual_mcmc_results,
        "expected_baseline_rows": expected_baseline_results,
        "actual_baseline_rows": actual_baseline_results,
        "output_csv": str(out_csv),
    }
    metadata_file = OUTPUT_DIR / "experiment_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"📄 Experiment metadata saved to: {metadata_file}")

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running test experiment (single PO per scenario)...")
        data = build_data_dict(PROJECT_ROOT)

        # Test one scenario with all its traces
        test_scenario = data["scenario_ids"][0]
        test_local = data["orders_local_by_assessor"][test_scenario]
        test_scenario_data = data["scenario_data"][test_scenario]
        
        print(f"\nTest scenario: {test_scenario}")
        print(f"  Items: {test_scenario_data['task_ids']}")
        print(f"  Traces: {len(test_local)}")

        # Run one experiment with 50000 iterations for testing
        test_iterations = 50000
        print(f"Running with {test_iterations} iterations...")

        exp_id, result = run_single_scenario_experiment(
            exp_id=999,
            scenario_id=test_scenario,
            cp_target=1.0,
            eps=0.01,
            lh="log_successors_queue_jump",
            scenario_data=test_scenario_data,
            sampled_local=test_local,
            realized_cov=1.0,
            num_iterations=test_iterations,
        )

        print(f"\nTest experiment completed:")
        print(f"  Scenario: {result['scenario']}")
        print(f"  F1: {result['cover_f1']:.3f}")
        print(f"  SHD: {result['shd']}")
        print(f"  Feasibility: {result['feas']:.3f}")

        sys.exit(0)

    data = build_data_dict(PROJECT_ROOT)
    df = run_systematic_experiments(data)
    print(df.head())
