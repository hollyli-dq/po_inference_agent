#!/usr/bin/env python3
"""
WFCommons systematic benchmark for BHPOP (action-precedence inference)

This script treats WFCommons as "execution traces from a scheduler/runtime", not an agent.
Per (workflow-family × size) group, we:

1) Load workflow instances from WFCommons JSON files
2) Build ground-truth DAG -> poset (closure + cover)
3) Extract execution-derived traces (linear extensions) from each instance
4) Control trace diversity using IP-Cov targets via subsampling (and optional synthetic augmentation)
5) Run BHPOP MCMC under two likelihoods (Q_rank vs Q_succ) and eps_jump sweep
6) Evaluate structural recovery (Cover F1, SHD, Feasibility). No token/LLM latency metrics.

Outputs:
- CSV summary: <output_dir>/wfcommons_summary.csv
- Per-run diagnostics PDF: <output_dir>/diagnostics/<group>/cp=<...>/eps=<...>/lh=<...>.pdf
"""

from __future__ import annotations

import sys
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

# ---- matplotlib backend ----
import matplotlib
if "ipykernel" in sys.modules:
    try:
        import matplotlib_inline  # noqa: F401
        matplotlib.use("module://matplotlib_inline.backend_inline")
    except Exception:
        pass
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ───────────────────────────────────────────────────────────────
# Project root bootstrap (keeps imports stable for script + notebook)
# ───────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
# If you place this file under <repo>/scripts/, parents[1] is repo root.
# If you place under <repo>/notebooks/, parents[1] is also repo root.
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure src is a package (helps some environments)
(src_dir := PROJECT_ROOT / "src").mkdir(parents=True, exist_ok=True)
(utils_dir := src_dir / "utils").mkdir(parents=True, exist_ok=True)
for d in (src_dir, utils_dir):
    init = d / "__init__.py"
    if not init.exists():
        init.touch()

# ───────────────────────────────────────────────────────────────
# Repo imports (must exist in your project)
# ───────────────────────────────────────────────────────────────
from src.utils.wfcommons_loader import (  # type: ignore
    iter_workflow_instances,
    build_hpo_inputs_from_instances,
    build_adjacency_matrix,
)
from src.utils.po_fun import BasicUtils  # type: ignore
from src.utils.hpo_model_evaluation import (  # type: ignore
    precision_recall,
    f1_score,
    structural_hamming_distance,
)
from src.utils.po_accelerator_nle_optimized import HPO_LogLikelihoodCache_Optimized  # type: ignore
from src.mcmc.hpo_po_hm_mcmc_k_optim import mcmc_simulation_hpo_k_optim  # type: ignore
from src.utils.result_paths import WFCOMMONS_MONTAGE_RECOVERY_RESULTS_DIR


# =========================
# IP-Cov + feasibility
# =========================
def incomparable_pairs_from_closure(closure: np.ndarray) -> List[Tuple[int, int]]:
    n = closure.shape[0]
    # incomparable iff neither i<j nor j<i
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if closure[i, j] == 0 and closure[j, i] == 0]


def cp_cov(orders_local: List[List[int]], true_closure: np.ndarray) -> float:
    pairs = incomparable_pairs_from_closure(true_closure)
    if not pairs:
        return 1.0
    seen = {pair: 0 for pair in pairs}  # 1 means i<j seen, 2 means j<i seen, 3 means both
    for order in orders_local:
        pos = {a: t for t, a in enumerate(order)}
        for (i, j) in pairs:
            if i in pos and j in pos:
                seen[(i, j)] |= 1 if pos[i] < pos[j] else 2
    return sum(v == 3 for v in seen.values()) / len(pairs)


def is_linear_extension(order: List[int], closure: np.ndarray) -> bool:
    """order is a full permutation (or subset); closure[i,j]=1 means i must be before j."""
    pos = {t: i for i, t in enumerate(order)}
    rows, cols = np.where(closure == 1)
    for i, j in zip(rows, cols):
        if i in pos and j in pos and pos[i] >= pos[j]:
            return False
    return True


# =========================
# IP-Cov subsampling (greedy)
# =========================
def greedy_subset_indices_to_target(
    orders_local: List[List[int]],
    true_closure: np.ndarray,
    target: float,
    seed: int,
) -> Tuple[List[int], float]:
    """
    Greedy select a subset of traces whose IP-Cov reaches `target` (as much as possible).
    Assumes orders_local are valid LEs (or filters invalid ones).
    """
    rng = np.random.default_rng(seed)

    valid_idx = [i for i, o in enumerate(orders_local) if is_linear_extension(o, true_closure)]
    valid_orders = [orders_local[i] for i in valid_idx]
    if not valid_orders:
        return [], 0.0

    pairs = incomparable_pairs_from_closure(true_closure)
    if not pairs or target <= 0:
        # trivial: if no incomparable pairs, IP-Cov=1 automatically
        return [valid_idx[0]], 1.0

    # For each order, precompute which way it orients each incomparable pair
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
    COMPLETE_BONUS = 5  # reward completing a pair to "both orientations"

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
    realized = cp_cov([orders_local[i] for i in chosen_idx], true_closure)
    return chosen_idx, realized


# =========================
# Baselines (local space)
# =========================
def baseline_intersection_cover(orders_local: List[List[int]], n: int) -> np.ndarray:
    """i->j iff i precedes j in *every* trace (when both appear); then return cover."""
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
    try:
        cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
    except AttributeError:
        cover = BasicUtils.transitive_reduction(closure.astype(int))
    return cover.astype(np.int8)


def baseline_majority_cover(orders_local: List[List[int]], n: int, thr: float = 0.5) -> np.ndarray:
    """Majority pairwise precedence + simple cycle-breaking; return cover."""
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

    p = np.zeros((n, n), dtype=float)
    p = np.where(totals > 0, counts / np.maximum(1, totals), 0.0)

    adj = np.zeros((n, n), dtype=np.int8)
    margin = np.zeros((n, n), dtype=float)

    # choose orientation for each unordered pair
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

    # cycle-break by removing weakest edge in any found cycle
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
    try:
        cover = BasicUtils.transitive_reduction_optimized(closure.astype(np.int8))
    except AttributeError:
        cover = BasicUtils.transitive_reduction(closure.astype(int))
    return cover.astype(np.int8)


# =========================
# Posterior aggregation: pairwise marginal mode
# =========================
def posterior_pairwise_marginal_mode(mats: List[np.ndarray]) -> np.ndarray:
    """
    Vote per unordered pair among {incomparable, i->j, j->i}.
    Returns an adjacency matrix in the same index space as mats[*].
    """
    if not mats:
        raise ValueError("posterior_pairwise_marginal_mode got empty mats.")
    n = mats[0].shape[0]
    T = len(mats)
    out = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            c_ij = sum(1 for M in mats if M[i, j] == 1)
            c_ji = sum(1 for M in mats if M[j, i] == 1)
            c_inc = T - c_ij - c_ji
            if c_ij >= c_ji and c_ij >= c_inc:
                out[i, j] = 1
            elif c_ji >= c_ij and c_ji >= c_inc:
                out[j, i] = 1
    np.fill_diagonal(out, 0)
    return out


# =========================
# Diagnostics PDF
# =========================
def save_diagnostics_pdf(
    mcmc_results: dict,
    pdf_path: Path,
    *,
    burn_in_ll: Optional[int] = None,     # if None uses 50%
    burn_frac_beta: float = 0.5,
    title_prefix: str = "",
) -> None:
    """
    Save diagnostics plots into a single PDF:
      1) log_likelihood_currents trace
      2) softmax_beta trace (if present)
      3) softmax_beta posterior histogram (post burn)
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # 1) Log-likelihood
        ll = mcmc_results.get("log_likelihood_currents", [])
        if ll:
            ll = np.asarray(ll, dtype=float)
            burn = int(len(ll) * 0.5) if burn_in_ll is None else int(burn_in_ll)
            burn = max(0, min(burn, len(ll)))

            fig = plt.figure(figsize=(10, 3))
            plt.plot(ll, lw=0.8)
            if burn > 0:
                plt.axvline(burn, color="r", ls="--", lw=1)
            plt.title(f"{title_prefix}log_likelihood_currents".strip())
            plt.xlabel("iteration")
            plt.ylabel("log-likelihood")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) softmax_beta trace + hist
        beta = mcmc_results.get("softmax_beta_trace", [])
        if beta:
            beta = np.asarray(beta, dtype=float)
            burn = int(len(beta) * burn_frac_beta)
            burn = max(0, min(burn, len(beta)))

            fig = plt.figure(figsize=(10, 3))
            plt.plot(beta, lw=0.8)
            if burn > 0:
                plt.axvline(burn, color="r", ls="--", lw=1)
            plt.title(f"{title_prefix}softmax_beta trace".strip())
            plt.xlabel("saved-step (trace)")
            plt.ylabel("beta")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            post = beta[burn:] if burn < len(beta) else beta
            fig = plt.figure(figsize=(6, 3))
            plt.hist(post, bins=30, color="gray")
            plt.title(f"{title_prefix}softmax_beta posterior".strip())
            plt.xlabel("beta")
            plt.ylabel("count")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# =========================
# Grouping WFCommons instances
# =========================
def _infer_group_name_from_instance(inst: Dict[str, Any]) -> str:
    """
    Best-effort grouping by filename convention: <group>-<runid>.json
    Works for names like: montage-chameleon-2mass-005d-001.json  -> group=montage-chameleon-2mass-005d
    """
    meta = inst.get("metadata", {}) if isinstance(inst.get("metadata", {}), dict) else {}
    src = meta.get("source_path") or meta.get("instance_name") or meta.get("path") or meta.get("file")
    if src is None:
        # fallback: maybe inst has "source_path" directly
        src = inst.get("source_path") or inst.get("instance_name") or "unknown"

    stem = Path(str(src)).stem
    # handle "prefix::name"
    if "::" in stem:
        stem = stem.split("::")[-1]
    # drop trailing -001 or _001
    stem = re.sub(r"[-_]\d+$", "", stem)
    return stem


def group_instances(instances: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        g = _infer_group_name_from_instance(inst)
        groups.setdefault(g, []).append(inst)
    return groups


# =========================
# Main WFCommons experiment runner
# =========================
def run_wfcommons_systematic(
    *,
    data_dir: Path,
    groups_to_run: Optional[List[str]],
    cp_cov_targets: List[float],
    eps_jump_list: List[float],
    likelihoods: List[str],
    num_iterations: int,
    burn_in_fraction: float,
    thin: int,
    seed_base: int,
    cycle_length: int,
    fixed_k: Optional[int],
    output_dir: Path,
    prefer_execution_order: bool,
) -> pd.DataFrame:
    """
    Run systematic sweep for WFCommons groups.
    Produces a long-form results table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load instances ----
    instances = list(iter_workflow_instances(data_dir))
    if not instances:
        raise RuntimeError(f"No WFCommons instances found under: {data_dir}")

    grouped = group_instances(instances)

    # filter groups
    group_names = sorted(grouped.keys())
    if groups_to_run:
        keep = set(groups_to_run)
        group_names = [g for g in group_names if g in keep]
    if not group_names:
        raise RuntimeError("No groups selected. Check --groups arguments.")

    rows: List[Dict[str, Any]] = []
    exp_id = 0

    for group_name in group_names:
        inst_list = grouped[group_name]
        print(f"\n==============================")
        print(f"WFCommons group: {group_name}")
        print(f"instances: {len(inst_list)}")
        print(f"==============================")

        # ---- Build BHPOP inputs from this group ----
        inputs = build_hpo_inputs_from_instances(inst_list, prefer_execution_order=prefer_execution_order)

        # For this script we assume 1 assessor per group
        assessor_id = inputs.assessors[0]
        M0 = list(inputs.M0)
        n = len(M0)

        # Observed traces: list of permutations (in index space 0..n-1)
        observed_orders_all: List[List[int]] = list(inputs.observed_orders[assessor_id])

        # Choice sets per trace: usually full set, but we compute robustly
        choice_sets_all: List[List[int]] = [sorted(set(o)) for o in observed_orders_all]

        # ---- Ground-truth adjacency from parents_subset ----
        idx_to_task = dict(inputs.idx_to_task)
        task_ids = [idx_to_task[i] for i in range(len(idx_to_task))]

        parent_sets = {k: set(v) for k, v in inputs.parents_subset.items()}
        true_adj = np.array(build_adjacency_matrix(task_ids, parent_sets), dtype=np.int8)
        true_closure = BasicUtils.transitive_closure(true_adj.astype(np.int8))
        try:
            true_cover = BasicUtils.transitive_reduction_optimized(true_closure.astype(np.int8)).astype(np.int8)
        except AttributeError:
            true_cover = BasicUtils.transitive_reduction(true_closure.astype(int)).astype(np.int8)

        # local orders == global orders in this per-group setting
        orders_local_all = observed_orders_all

        base_cov = cp_cov(orders_local_all, true_closure)
        num_traces = len(orders_local_all)
        num_unique = len({tuple(o) for o in orders_local_all})
        print(f"traces={num_traces} unique={num_unique} base IP-Cov={base_cov:.3f}")

        # If we can't even do 1 trace, skip
        if num_traces == 0:
            continue

        # ---- Sweep IP targets (subsampling) ----
        for cp_target in cp_cov_targets:
            idxs, realized_cov = greedy_subset_indices_to_target(
                orders_local_all, true_closure, cp_target, seed=seed_base
            )
            if not idxs:
                print(f"  IP target {cp_target}: no valid traces -> skip")
                continue

            sampled_orders = [observed_orders_all[i] for i in idxs]
            sampled_choice_sets = [choice_sets_all[i] for i in idxs]
            sampled_local = [orders_local_all[i] for i in idxs]

            print(f"  IP target {cp_target}: selected {len(idxs)} traces, realized IP-Cov={realized_cov:.3f}")

            # ---- Baselines (structural only) ----
            inter_cover = baseline_intersection_cover(sampled_local, n)
            maj_cover = baseline_majority_cover(sampled_local, n, thr=0.5)

            for method_name, pred_cover in [("intersection", inter_cover), ("majority", maj_cover)]:
                p, r = precision_recall(true_cover, pred_cover)
                rows.append({
                    "dataset": "wfcommons",
                    "group": group_name,
                    "assessor": assessor_id,
                    "cp_cov_target": cp_target,
                    "cp_cov_realized": realized_cov,
                    "num_traces_total": num_traces,
                    "num_traces_used": len(idxs),
                    "eps_jump": np.nan,
                    "likelihood": "baseline",
                    "method": method_name,
                    "cover_f1": f1_score(p, r),
                    "precision": p,
                    "recall": r,
                    "shd": structural_hamming_distance(true_cover, pred_cover),
                    "feas": np.nan,
                    "diagnostics_pdf": "",
                })

            # ---- MCMC sweeps ----
            for eps in eps_jump_list:
                # Set epsilon for log_successors_queue_jump (must be set before MCMC)
                HPO_LogLikelihoodCache_Optimized.SOFTMAX_FRONTIER_EPS = float(eps)

                for lh in likelihoods:
                    exp_id += 1
                    seed = seed_base + exp_id
                    print(f"    MCMC: eps={eps} lh={lh} seed={seed} traces={len(idxs)} n={n}")

                    # Convert single-PO parameters to hierarchical format (single assessor)
                    assessor_id = inputs.assessors[0]
                    mcmc = mcmc_simulation_hpo_k_optim(
                        num_iterations=num_iterations,
                        M_a_dict={assessor_id: list(M0)},
                        O_a_i_dict={assessor_id: sampled_choice_sets},
                        observed_orders={assessor_id: sampled_orders},
                        dr=0.95,
                        noise_option=lh,
                        rho_prior=1.0,
                        noise_beta_prior=1.0,
                        K_prior=2 if fixed_k is None else fixed_k,
                        fixed_K=fixed_k,
                        random_seed=seed,
                        cycle_length=cycle_length,
                        softmax_beta_prior=(2.0, 1.0),
                        softmax_beta_stepsize=0.1,
                    )

                    # --- Posterior aggregation from H_trace ---
                    H_trace = mcmc.get("H_trace", [])
                    burn = int(len(H_trace) * burn_in_fraction)
                    post = H_trace[burn::max(1, thin)]

                    # H_trace is a list of dicts: [{assessor_id: matrix}, ...]
                    # Extract matrices for our single assessor
                    mats = [it[assessor_id] for it in post if assessor_id in it]
                    if not mats:
                        print("      (no post-burn H samples; skipping eval)")
                        continue

                    marg = posterior_pairwise_marginal_mode(mats)

                    # Enforce DAG / cover
                    marg_closure = BasicUtils.transitive_closure(marg.astype(np.int8))
                    try:
                        inferred_cover = BasicUtils.transitive_reduction_optimized(marg_closure.astype(np.int8)).astype(np.int8)
                    except AttributeError:
                        inferred_cover = BasicUtils.transitive_reduction(marg_closure.astype(int)).astype(np.int8)

                    # --- Structural metrics ---
                    p, r = precision_recall(true_cover, inferred_cover)
                    shd = structural_hamming_distance(true_cover, inferred_cover)

                    # --- Feasibility: do inferred constraints admit the observed traces used? ---
                    inferred_closure = BasicUtils.transitive_closure(inferred_cover.astype(np.int8))
                    feas = float(np.mean([is_linear_extension(o, inferred_closure) for o in sampled_local]))

                    # --- Diagnostics PDF ---
                    pdf_path = diag_dir / group_name / f"cp={cp_target:.2f}" / f"eps={eps:.3g}" / f"lh={lh}.pdf"
                    save_diagnostics_pdf(
                        mcmc,
                        pdf_path,
                        burn_in_ll=None,  # default 50% for ll plot
                        burn_frac_beta=0.5,
                        title_prefix=f"{group_name} | cp={cp_target:.2f} | eps={eps:.3g} | {lh} | ",
                    )

                    rows.append({
                        "dataset": "wfcommons",
                        "group": group_name,
                        "assessor": assessor_id,
                        "cp_cov_target": cp_target,
                        "cp_cov_realized": realized_cov,
                        "num_traces_total": num_traces,
                        "num_traces_used": len(idxs),
                        "eps_jump": eps,
                        "likelihood": lh,
                        "method": "bhpop_marginal_mode",
                        "cover_f1": f1_score(p, r),
                        "precision": p,
                        "recall": r,
                        "shd": shd,
                        "feas": feas,
                        "diagnostics_pdf": str(pdf_path.relative_to(output_dir)),
                    })

    df = pd.DataFrame(rows)
    out_csv = output_dir / "wfcommons_summary.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="WFCommons instances directory, e.g. data/wfcommons_wfinstances/montage")
    p.add_argument("--output_dir", type=str, default=str(WFCOMMONS_MONTAGE_RECOVERY_RESULTS_DIR))
    p.add_argument("--groups", type=str, nargs="*", default=None,
                   help="Explicit WFCommons group names to run. If omitted, run all groups found.")
    p.add_argument("--prefer_execution_order", action="store_true",
                   help="Use execution-derived order if available in instances (recommended).")
    p.add_argument("--num_iterations", type=int, default=20000)
    p.add_argument("--burn_in_fraction", type=float, default=0.5)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--cycle_length", type=int, default=500)
    p.add_argument("--fixed_k", type=int, default=None,
                   help="Fix latent dimension K (disables RJMCMC). If omitted, uses RJMCMC.")
    p.add_argument("--cp_targets", type=float, nargs="*", default=[0.5, 0.8, 0.95, 1.0])
    p.add_argument("--eps_list", type=float, nargs="*", default=[0.005, 0.01, 0.05, 0.1])
    p.add_argument("--likelihoods", type=str, nargs="*", default=["log_successors_queue_jump", "softmax_queue_jump"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = run_wfcommons_systematic(
        data_dir=Path(args.data_dir),
        groups_to_run=args.groups,
        cp_cov_targets=list(args.cp_targets),
        eps_jump_list=list(args.eps_list),
        likelihoods=list(args.likelihoods),
        num_iterations=int(args.num_iterations),
        burn_in_fraction=float(args.burn_in_fraction),
        thin=int(args.thin),
        seed_base=int(args.seed_base),
        cycle_length=int(args.cycle_length),
        fixed_k=args.fixed_k,
        output_dir=Path(args.output_dir),
        prefer_execution_order=bool(args.prefer_execution_order),
    )
    print(df.head())


if __name__ == "__main__":
    main()
