import numpy as np
import pandas as pd
import os, sys, random, math, time, json, yaml, logging
import heapq
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
from tqdm import tqdm
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
# Project‑local imports
from src.utils.po_fun import BasicUtils, StatisticalUtils
from src.utils.po_accelerator_eta_accelerated import (
    get_ultra_fast_eta_matrix
)

import threading
from concurrent.futures import ThreadPoolExecutor
from src.utils.linext_direct import get_linext_direct  
from src.utils.po_fun import BasicUtils  # Fallback when C++ not available
###############################################################################
#                               LogLikelihoodCache
###############################################################################
class LogLikelihoodCache:
    """ Thread‑safe, cached NLE computations for *flat* (single‑layer) models. """

    # --- shared state -----------------------------------------------------------------
    _nle_cache: Dict[bytes, int] = {}
    _nle_first_cache: Dict[Tuple[bytes, int], int] = {}
    _cache_lock = threading.Lock()
    _pool = ThreadPoolExecutor(max_workers=16)

    # --- helpers ----------------------------------------------------------------------
    @staticmethod
    def _mat_key(mat: np.ndarray) -> bytes:
        return mat.tobytes()

    # --- public API -------------------------------------------------------------------
    @classmethod
    def nle(cls, mat: np.ndarray) -> int:
        """Cached BasicUtils.nle (executed in the thread‑pool)."""
        key = cls._mat_key(mat)
        with cls._cache_lock:
            if key in cls._nle_cache:
                return cls._nle_cache[key]
        val = cls._pool.submit(BasicUtils.nle, mat).result()
        with cls._cache_lock:
            cls._nle_cache[key] = val
        return val

    @classmethod
    def nle_first(cls, mat: np.ndarray, idx: int) -> int:
        key = (cls._mat_key(mat), idx)
        with cls._cache_lock:
            if key in cls._nle_first_cache:
                return cls._nle_first_cache[key]
        val = cls._pool.submit(BasicUtils.num_extensions_with_first, mat, idx).result()
        with cls._cache_lock:
            cls._nle_first_cache[key] = val
        return val

    # --- small utility ----------------------------------------------------------------
    @staticmethod
    def bar_eta_vectorised(indices: List[int], U0: np.ndarray) -> np.ndarray:
        """Mean of latent vector for a *set* of global indices (fully vectorised)."""
        if not indices:
            return np.empty(0)
        subset = U0[np.array(indices)]  # (|set|, K)
        return subset.mean(axis=1)      # (|set|,)

###############################################################################
#                           HPO_LogLikelihoodCache
###############################################################################
class HPO_LogLikelihoodCache_Optimized:
    """ Like `LogLikelihoodCache`, but for hierarchical‑partial‑order (HPO) models. """

    _nle_cache: Dict[bytes, int] = {}
    _nle_first_cache: Dict[Tuple[bytes, int], int] = {}
    _cache_lock = threading.Lock()
    _pool = ThreadPoolExecutor(max_workers=4)
    # Removed: _linext (subprocess overhead was too high)
    SOFTMAX_FRONTIER_EPS = 0.001

    @classmethod
    def nle(cls, mat: np.ndarray) -> int:
        """⭐ FAST: LinextDirect C++ acceleration (falls back to BasicUtils.nle if C++ not available)."""
        key = cls._mat_key(mat)
        
        # Check cache without lock
        if key in cls._nle_cache:
            return cls._nle_cache[key]
        
        # G2 CACHING: Use C++ library directly (debug output disabled in C++ source)
        linext = get_linext_direct(quiet=False)
        val = linext.nle(mat)

        
        # Store result
        with cls._cache_lock:
            cls._nle_cache[key] = val
        
        return val

    @classmethod
    def nle_first(cls, mat: np.ndarray, idx: int) -> int:
        """⭐ FIXED: Proper num_extensions_with_first calculation using C++ on reduced matrix."""
        key = (cls._mat_key(mat), idx)
        
        # Check cache without lock
        if key in cls._nle_first_cache:
            return cls._nle_first_cache[key]
        
        # G2 CACHING: C++ accelerated num_extensions_with_first
        tops = BasicUtils.find_tops(mat)
        if idx not in tops:
            val = 0
        else:
            # Remove the row and column for the first item to get reduced matrix
            mat_reduced = np.delete(np.delete(mat, idx, axis=0), idx, axis=1)
            
            # Calculate NLE of the reduced matrix using C++ directly
            linext = get_linext_direct(quiet=False)
            val = linext.nle(mat_reduced)
        
        # Store result
        cls._nle_first_cache[key] = val
        return val

    @classmethod
    def nle_linext(cls, mat: np.ndarray) -> int:
        return cls.nle(mat)  # Redirect to the pure Python nle method

    # -------------------------------------------------------------------------
    @staticmethod
    def _mat_key(mat: np.ndarray) -> bytes:
        return mat.tobytes()

    # -------------------------------------------------------------------------
    @staticmethod
    def bar_eta_vectorised(indices: List[int], U0: np.ndarray) -> np.ndarray:
        """Vectorised bar‑eta; identical to flat case but kept separate for clarity."""
        if not indices:
            return np.empty(0)
        return U0[np.array(indices)].mean(axis=1)


    # -------------------------------------------------------------------------
    @classmethod
    def _adjacency_from_U0(cls, U0: np.ndarray, items: List[int]) -> np.ndarray:
        """Fast adjacency construction for a set of *global* items using only U0."""
        if not items:
            return np.empty((0, 0), dtype=int)
        subset = U0[np.array(items)]
        return BasicUtils.generate_partial_order(subset)

    # -------------------------------------------------------------------------
    @classmethod
    def calculate_log_likelihood_po_optimized(
        cls,
        U: np.ndarray,
        h: np.ndarray,
        observed_orders: List[List[int]],
        choice_sets: List[List[int]],
        items: List[int],
        item_to_index: Dict[int, int],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump",
    ) -> float:
        """
        Log-likelihood calculation for a SINGLE partial order.
        
        Args:
            U: Latent utility matrix (n_items x K)
            h: Partial order adjacency matrix (n_items x n_items)
            observed_orders: List of observed total orders
            choice_sets: List of choice sets corresponding to each observed order
            items: List of item IDs
            item_to_index: Mapping from item ID to matrix index
            prob_noise: Noise probability (epsilon for softmax)
            softmax_params: Optional softmax parameters {"beta": ..., "epsilon": ...}
            noise_option: "queue_jump", "weighted_queue_jump", or "log_successors_queue_jump"
            
        Returns:
            Log-likelihood value
        """
        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump"}:
            raise ValueError(f"Unsupported noise_option: {noise_option}")

        total_ll = 0.0

        # Create local index mapping for items
        g2l = {g: i for i, g in enumerate(items)}

        for task_idx, order in enumerate(observed_orders):
            choice_set = choice_sets[task_idx]
            # Filter choice_set to only items in g2l and create local mapping
            choice_set_filtered = [g for g in choice_set if g in g2l]
            g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
            local_ids = [g2l[g] for g in choice_set_filtered]
            if not local_ids:
                continue

            # Sub-matrix extraction (indices into U/h)
            H_sub = h[np.ix_(local_ids, local_ids)]
            # bar_eta expects indices into U, not item IDs
            bar_eta = cls.bar_eta_vectorised(local_ids, U)
            # obs_local must be indices into H_sub/bar_eta (same as choice_set_filtered order)
            obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
            if not obs_local:
                continue

            # Compute likelihood
            if noise_option == "queue_jump":
                total_ll += cls._queue_jump_ll(obs_local, H_sub, prob_noise)
            elif noise_option == "weighted_queue_jump":
                total_ll += cls._wqueue_jump_ll(obs_local, H_sub, prob_noise, bar_eta)
            elif noise_option == "log_successors_queue_jump":
                if isinstance(softmax_params, dict):
                    beta_val = softmax_params.get("beta", 1.0)
                    epsilon_val = softmax_params.get("epsilon", 0.01)
                else:
                    beta_val = 1.0 / max(prob_noise, 0.01)
                    epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                total_ll += cls._log_successors_queue_jump_ll(
                    obs_local, H_sub, beta=beta_val, epsilon=epsilon_val
                )
        
        return total_ll

    @classmethod
    def calculate_log_likelihood_hpo_optimized(
        cls,
        U: Dict[str, Any],                     # {"U0": …, "U_a_dict": …}
        h_U: Dict[int, np.ndarray],            # assessor → local partial order
        observed_orders: Dict[int, List[List[int]]],
        M_a_dict: Dict[int, List[int]],        # assessor → list of global items
        O_a_i_dict: Dict[int, List[List[int]]],# assessor → list of choice‑sets
        item_to_index: Dict[int, int],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump"
    ) -> float:

        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump"}:
            raise ValueError(f"Unsupported noise_option: {noise_option}")

        total_ll = 0.0
        # Penalty for traces that cannot be mapped to the model (data mismatch).
        # This prevents silent drops that otherwise add 0.0 to the log-likelihood.
        MISSING_DATA_PENALTY = -1e10
        
        # Progress tracking for large datasets
        total_assessors = len(observed_orders)
        processed_assessors = 0

        for assessor, orders_a in observed_orders.items():
            processed_assessors += 1
            Ma = M_a_dict[assessor]

            H_a = h_U[assessor]
            U_a_dict = U.get("U_a_dict", {})
            Ua = U_a_dict[assessor]
            # Map global→local for fast lookup
            g2l = {g: i for i, g in enumerate(Ma)}

            # Precompute eta and bar-eta for entire assessor (reuse across tasks)
            eta_full = get_ultra_fast_eta_matrix(
                assessor=assessor,
                Ua=Ua,
                Ma=Ma,
            )
            bar_eta_full = eta_full.mean(axis=1)  # shape (len(Ma),)

            for task_idx, order in enumerate(orders_a):
                choice_set = O_a_i_dict[assessor][task_idx]
                choice_set_filtered = [g for g in choice_set if g in g2l]
                g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
                local_ids = [g2l[g] for g in choice_set_filtered]
                if not local_ids:
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Sub‑matrix for this choice‑set (now safe because H_a is transitive closure)
                idx_array = np.array(local_ids)
                H_sub = H_a[idx_array[:, None], idx_array]

                # Compute bar-eta for this choice set by indexing the precomputed vector
                bar_eta = bar_eta_full[np.array(local_ids)]


                # Observed order as indices within choice_set
                obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
                if not obs_local:
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Likelihood contribution
                if noise_option == "queue_jump":
                    total_ll += cls._queue_jump_ll(obs_local, H_sub, prob_noise)
                elif noise_option == "weighted_queue_jump":
                    total_ll += cls._wqueue_jump_ll(obs_local, H_sub, prob_noise, bar_eta)
                elif noise_option == "log_successors_queue_jump":
                    if isinstance(softmax_params, dict):
                        beta_val = softmax_params.get("beta", 1.0)
                        epsilon_val = softmax_params.get("epsilon", 0.01)
                    else:
                        beta_val = 1.0 / max(prob_noise, 0.01)
                        epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01

                    total_ll += cls._log_successors_queue_jump_ll(
                        obs_local,
                        H_sub,
                        beta=beta_val,
                        epsilon=epsilon_val
                    )
        return total_ll

    @classmethod
    def calculate_log_likelihood_hpo_with_pointwise(
        cls,
        U: Dict[str, Any],                     # {"U0": …, "U_a_dict": …}
        h_U: Dict[int, np.ndarray],            # assessor → local partial order
        observed_orders: Dict[int, List[List[int]]],
        M_a_dict: Dict[int, List[int]],        # assessor → list of global items
        O_a_i_dict: Dict[int, List[List[int]]],# assessor → list of choice‑sets
        item_to_index: Dict[int, int],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump"
    ) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """
        Calculate log-likelihood with pointwise contributions for WAIC.
        
        Returns:
            total_ll: Total log-likelihood (sum of all pointwise contributions)
            pointwise_ll: Dict mapping (assessor_id, order_idx) -> log-likelihood contribution
        """
        from typing import Dict, Tuple
        
        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump"}:
            raise ValueError(f"Unknown noise_option: {noise_option}")

        total_ll = 0.0
        pointwise_ll = {}
        MISSING_DATA_PENALTY = -1e10
        
        # If h_U is empty, build it using the accelerator
        if not h_U:
            from .po_fun import build_hierarchical_partial_orders_optimized
            h_U = build_hierarchical_partial_orders_optimized(
                M0=list(item_to_index.keys()),
                assessors=list(observed_orders.keys()),
                M_a_dict=M_a_dict,
                U0=U["U0"],
                U_a_dict=U["U_a_dict"],
            )
        
        for assessor, orders_a in observed_orders.items():
            if assessor not in h_U:
                continue
                
            H_a = h_U[assessor]
            g2l = {g: i for i, g in enumerate(M_a_dict[assessor])}
            bar_eta_full = None
            if noise_option == "weighted_queue_jump":
                Ua = U["U_a_dict"][assessor]
                eta_full = get_ultra_fast_eta_matrix(
                    assessor=assessor,
                    Ua=Ua,
                    Ma=M_a_dict[assessor],
                )
                bar_eta_full = eta_full.mean(axis=1)

            for order_idx, order in enumerate(orders_a):
                order_ll = 0.0  # Likelihood for this specific order
                
                choice_set = O_a_i_dict[assessor][order_idx]
                choice_set_filtered = [g for g in choice_set if g in g2l]
                g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
                local_ids = [g2l[g] for g in choice_set_filtered]
                if not local_ids:
                    pointwise_ll[(assessor, order_idx)] = MISSING_DATA_PENALTY
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Sub‑matrix for this choice‑set
                idx_array = np.array(local_ids)
                H_sub = H_a[idx_array[:, None], idx_array]

                # Compute bar-eta for this choice set
                bar_eta = bar_eta_full[np.array(local_ids)] if bar_eta_full is not None else None

                # Observed order as indices within choice_set
                obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
                if not obs_local:
                    pointwise_ll[(assessor, order_idx)] = MISSING_DATA_PENALTY
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Likelihood contribution for this order
                if noise_option == "queue_jump":
                    order_ll = cls._queue_jump_ll(obs_local, H_sub, prob_noise)
                elif noise_option == "weighted_queue_jump":
                    order_ll = cls._wqueue_jump_ll(obs_local, H_sub, prob_noise, bar_eta)
                elif noise_option == "log_successors_queue_jump":
                    if isinstance(softmax_params, dict):
                        beta_val = softmax_params.get("beta", 1.0)
                        epsilon_val = softmax_params.get("epsilon", 0.01)
                    else:
                        beta_val = 1.0 / max(prob_noise, 0.01)
                        epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                    order_ll = cls._log_successors_queue_jump_ll(
                        obs_local,
                        H_sub,
                        beta=beta_val,
                        epsilon=epsilon_val
                    )
                
                pointwise_ll[(assessor, order_idx)] = order_ll
                total_ll += order_ll
                
        return total_ll, pointwise_ll

    # -------------------------------------------------------------------------
    @classmethod
    def _queue_jump_ll(cls, order_idx: List[int], H: np.ndarray, p_noise: float) -> float:
        """Optimized queue jump likelihood calculation with reduced NumPy operations."""
        ll = 0.0
        m = len(order_idx)
        
        # Pre-allocate arrays to avoid repeated list slicing
        order_array = np.array(order_idx, dtype=int)
        
        for j, idx_j in enumerate(order_idx):
            # Use advanced indexing instead of ix_ for better performance
            rem_indices = order_array[j:]
            
            
            Hrem = H[rem_indices[:, None], rem_indices]  # More efficient than ix_
            
            # Input validation and C++ NLE computation
            if Hrem.size == 0 or Hrem.shape[0] <= 1:
                n_le = 1
                n_first = 1
            else:
                # Use C++ accelerated NLE (no artificial size limit)
                n_le = cls.nle(Hrem)
                n_first = cls.nle_first(Hrem, 0)
            
            p_no = (1 - p_noise) * (n_first / n_le) if n_le else 0.0
            p_j  = p_noise / (m - j)
            ll += math.log(max(p_no + p_j, 1e-20))
        return ll



    @classmethod
    def _wqueue_jump_ll(cls, order_idx: List[int], H: np.ndarray, p_noise: float, bar_eta: np.ndarray) -> float:
        """Optimized weighted queue jump likelihood calculation."""
        ll = 0.0
        m = len(order_idx)
        
        # Pre-allocate arrays to avoid repeated list slicing
        order_array = np.array(order_idx, dtype=int)
        
        for j, idx_j in enumerate(order_idx):
            # Use advanced indexing instead of ix_ for better performance
            rem_indices = order_array[j:]
            Hrem = H[rem_indices[:, None], rem_indices]  # More efficient than ix_
            
            # Input validation and C++ NLE computation
            if Hrem.size == 0 or Hrem.shape[0] <= 1:
                n_le = 1
                n_first = 1
            else:
                # Use C++ accelerated NLE (no artificial size limit)
                n_le = cls.nle(Hrem)
                n_first = cls.nle_first(Hrem, 0)
            
            p_no = (1 - p_noise) * (n_first / n_le) if n_le else 0.0
            
            # Optimize weight calculation
            weights = np.exp(bar_eta[rem_indices])
            p_j  = p_noise * (weights[0] / weights.sum())  # weights[0] is the first item (idx_j)
            ll += math.log(max(p_no + p_j, 1e-20))
        return ll

    # -------------------------------------------------------------------------
    # Softmax Queue-Jumping Likelihood (Frontier-Softmax)
    # -------------------------------------------------------------------------
    # Deterministic canonical plan L* (fixed tie-break topological sort) and
    # frontier-softmax choice rule (no linear-extension sampling).
    # -------------------------------------------------------------------------
    SOFTMAX_MC_SAMPLES = 20  # Deprecated (kept for backward compatibility)
    
    # Flag to control Numba usage (set to False if issues arise)
    USE_NUMBA = True
    _numba_checked = False
    _numba_available = False
    
    @classmethod
    def _check_numba(cls):
        """Check if Numba is available (only once)."""
        if not cls._numba_checked:
            try:
                from src.utils.numba_accelerated import NUMBA_AVAILABLE
                cls._numba_available = NUMBA_AVAILABLE and cls.USE_NUMBA
            except ImportError:
                cls._numba_available = False
            cls._numba_checked = True
        return cls._numba_available

    @staticmethod
    def _canonical_toposort(num_nodes: int, successors: List[List[int]], indegrees: List[int]) -> List[int]:
        """Deterministic Kahn topo sort with tie-break on node id."""
        heap = [i for i in range(num_nodes) if indegrees[i] == 0]
        heapq.heapify(heap)
        ordering = []

        while heap:
            node = heapq.heappop(heap)
            ordering.append(node)
            for nxt in successors[node]:
                indegrees[nxt] -= 1
                if indegrees[nxt] == 0:
                    heapq.heappush(heap, nxt)

        if len(ordering) != num_nodes:
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")
        return ordering

    @staticmethod
    def _logsumexp(values: List[float]) -> float:
        if not values:
            return -math.inf
        max_v = max(values)
        if max_v == -math.inf:
            return -math.inf
        return max_v + math.log(sum(math.exp(v - max_v) for v in values))
    
    @classmethod
    def _softmax_queue_jump_ll(
        cls,
        order_idx: List[int],
        H: np.ndarray,
        beta: float = 1.0,
        epsilon: float = 0.01,
    ) -> float:
        """
        Canonical-rank frontier-softmax likelihood with trembling-hand mixture (Eq. 2, 4 in paper):
          Q(a) = -β * rank_L(a)  where rank is position in canonical linearization L*
          P(y_t) = (1-ε)*π_β(y_t) + ε/|R_t|   if y_t in frontier
                 = ε/|R_t|                     otherwise
        
        where π_β is the Boltzmann policy: π_β(a) = exp(Q(a)) / Σ exp(Q(a'))

        Args:
            order_idx: Observed trace as indices [y_1, y_2, ..., y_T]
            H: Partial order matrix (transitive closure)
            beta: Inverse temperature (higher = more deterministic rank preference)
            epsilon: Trembling-hand ε ∈ (0,1), probability of uniform slip

        Returns:
            Log-likelihood log P(Y | H, β, ε)
        """
        m = len(order_idx)
        if m <= 1:
            return 0.0
        if beta <= 0:
            return -1e10

        n = H.shape[0]
        if n == 0:
            return 0.0
        if any(y < 0 or y >= n for y in order_idx):
            return -1e10

        H_bool = H.astype(bool)
        successors = [np.flatnonzero(H_bool[u]).tolist() for u in range(n)]
        indegrees = H_bool.sum(axis=0).astype(int).tolist()

        try:
            L_star = cls._canonical_toposort(n, successors, indegrees.copy())
        except ValueError:
            return -1e10

        # Trembling-hand epsilon
        eps = max(min(epsilon, 0.99), 1e-10)  # Clamp to (0, 1)

        remaining_order = list(L_star)
        remaining = set(remaining_order)
        unmet = indegrees[:]  # unmet prerequisites among remaining nodes
        frontier = {i for i in remaining if unmet[i] == 0}

        logp = 0.0

        for y_t in order_idx:
            if y_t not in remaining:
                return -1e10
            if not frontier:
                return -1e10

            rank = {a: i for i, a in enumerate(remaining_order)}
            # Q(a) = -β * rank(a), prefers actions with lower rank (earlier in canonical order)
            scores = [-beta * rank[a] for a in frontier]
            denom = cls._logsumexp(scores)
            if denom == -math.inf:
                return -1e10

            p_noise = eps / float(len(remaining))
            p_total = p_noise

            if y_t in frontier:
                score_y = -beta * rank[y_t]
                p_rational = math.exp(score_y - denom)
                p_total += (1.0 - eps) * p_rational

            logp += math.log(max(p_total, 1e-300))

            remaining.remove(y_t)
            frontier.discard(y_t)
            remaining_order.remove(y_t)

            for v in successors[y_t]:
                if v in remaining:
                    unmet[v] -= 1
                    if unmet[v] == 0:
                        frontier.add(v)

        return logp
    
    @classmethod
    def _log_successors_queue_jump_ll(
        cls,
        order_idx: List[int],
        H: np.ndarray,
        beta: float = 1.0,
        epsilon: float = 0.01,
    ) -> float:
        """
        Log-successors-based frontier-softmax likelihood (Eq. 3-4 in paper):
          Q(a) = log(1 + S_t(a))  where S_t(a) = |{b ∈ R_t \ {a} : a ≻ b}|
          P(y_t) = (1-ε)·exp(β·Q(y_t)) / Σ exp(β·Q(a)) + ε/|R_t|   if y_t in F_t
                 = ε/|R_t|                                         otherwise
        
        NO canonical ordering L* is used - this is a plan-free likelihood that
        maximizes future choice by preferring actions with more remaining successors.
        
        Args:
            order_idx: Observed trace as indices [y_1, y_2, ..., y_T]
            H: Partial order matrix (transitive closure)
            beta: Inverse temperature (higher = more deterministic)
            epsilon: Trembling-hand ε ∈ (0,1), probability of uniform slip
        
        Returns:
            Log-likelihood log P(Y | H, β, ε)
        """
        m = len(order_idx)
        if m <= 1:
            return 0.0
        if beta <= 0:
            return -1e10

        n = H.shape[0]
        if n == 0:
            return 0.0
        if any(y < 0 or y >= n for y in order_idx):
            return -1e10

        H_bool = H.astype(bool)
        successors = [np.flatnonzero(H_bool[u]).tolist() for u in range(n)]
        indegrees = H_bool.sum(axis=0).astype(int).tolist()

        # Trembling-hand epsilon
        eps = max(min(epsilon, 0.99), 1e-10)  # Clamp to (0, 1)

        # Initialize: all nodes remaining, frontier = nodes with no unmet predecessors
        remaining = set(range(n))
        unmet = indegrees[:]  # unmet prerequisites among remaining nodes
        frontier = {i for i in remaining if unmet[i] == 0}

        if not frontier:
            return -1e10  # No valid starting point

        logp = 0.0

        for y_t in order_idx:
            if y_t not in remaining:
                return -1e10
            if not frontier:
                return -1e10

            # Compute Q = log(1 + successors) for each node in frontier
            # S_t(a) = count of successors of 'a' that are still in remaining
            scores = []
            for a in frontier:
                num_successors = sum(1 for v in successors[a] if v in remaining and v != a)
                Q = math.log(num_successors + 1)  # Q = log(1 + S_t(a))
                scores.append(beta * Q)
            
            denom = cls._logsumexp(scores)
            if denom == -math.inf:
                return -1e10

            # Trembling-hand mixture
            p_noise = eps / float(len(remaining))
            p_total = p_noise

            if y_t in frontier:
                # Compute Q for y_t
                num_successors_y = sum(1 for v in successors[y_t] if v in remaining and v != y_t)
                Q_y = math.log(num_successors_y + 1)
                score_y = beta * Q_y
                p_rational = math.exp(score_y - denom)
                p_total += (1.0 - eps) * p_rational

            logp += math.log(max(p_total, 1e-300))

            # Update state after executing y_t
            remaining.remove(y_t)
            frontier.discard(y_t)

            # Add new frontier nodes (successors of y_t whose prerequisites are now all met)
            for v in successors[y_t]:
                if v in remaining:
                    unmet[v] -= 1
                    if unmet[v] == 0:
                        frontier.add(v)

        return logp
    
    @classmethod
    def _compute_softmax_ll_given_plan(cls, order_idx: List[int], L: List[int], tc: np.ndarray, theta: float) -> float:
        """
        Compute log P(Y | L, θ) for the softmax queue-jumping model.
        
        OPTIMIZED v3: Pure Python for small arrays (10-20x faster than numpy for n<30).
        NumPy overhead dominates for small array operations.
        
        At each step t, the agent picks from the frontier (feasible actions) with:
            score(a) = -θ * (rank_L(a; R_t) - 1)
            P(a) = softmax(scores)
        """
        m = len(order_idx)
        n = len(L)
        
        if m == 0 or n == 0:
            return 0.0
        
        # Create position dict (faster than array for small n)
        pos = {a: i for i, a in enumerate(L)}
        
        # Check all observed actions are in L
        for y_t in order_idx:
            if y_t not in pos:
                return -1e10
        
        ll = 0.0
        remaining = set(L)  # Use set for O(1) removal
        
        for y_t in order_idx:
            if not remaining:
                break
            
            if y_t not in remaining:
                return -1e10
            
            # Build rank map over remaining actions (rank in restricted L)
            remaining_by_pos = sorted(remaining, key=lambda a: pos[a])
            rank_map = {a: idx + 1 for idx, a in enumerate(remaining_by_pos)}

            # Frontier: actions with no predecessors in remaining
            frontier = []
            for a in remaining:
                has_pred = False
                for b in remaining:
                    if b != a and tc[b, a]:
                        has_pred = True
                        break
                if not has_pred:
                    frontier.append(a)

            if not frontier:
                return -1e10

            # Compute scores and log-softmax in pure Python over the frontier
            scores = [-theta * (rank_map[a] - 1) for a in frontier]
            max_score = max(scores)
            
            # Numerically stable log-sum-exp
            sum_exp = sum(math.exp(s - max_score) for s in scores)
            log_sum_exp = max_score + math.log(sum_exp)
            
            epsilon = cls.SOFTMAX_FRONTIER_EPS
            prob_valid = 1.0 - epsilon
            if y_t in frontier:
                y_t_score = -theta * (rank_map[y_t] - 1)
                ll += math.log(prob_valid) + (y_t_score - log_sum_exp)
            else:
                invalid_count = len(remaining) - len(frontier)
                if invalid_count <= 0:
                    return -1e10
                ll += math.log(epsilon) - math.log(invalid_count)
            
            # Remove y_t from remaining
            remaining.remove(y_t)
        
        return ll

    # -------------------------------------------------------------------------
    # TRUE INCREMENTAL LIKELIHOOD: Calculate Delta Change Only
    # -------------------------------------------------------------------------
    @classmethod
    def calculate_incremental_likelihood_delta(
        cls,
        assessor: int,
        row_idx: int,  # Which row of Ua is being updated
        old_row: np.ndarray,  # Previous utility values for this row
        new_row: np.ndarray,  # New utility values for this row
        U_a_dict: Dict[int, np.ndarray],  # Current state
        observed_orders: Dict[int, List[List[int]]],
        M_a_dict: Dict[int, List[int]],
        O_a_i_dict: Dict[int, List[List[int]]],
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump",
        h_U: Optional[Dict[int, np.ndarray]] = None  # ADDED: HU as input parameter
    ) -> float:
        """
        Calculate ONLY the incremental change in likelihood when one row of Ua changes.
        
        COMPLETE REWRITE: Ensure proper state management and accurate incremental calculation.
        
        Args:
            h_U: Dictionary mapping assessor IDs to their hierarchical partial order matrices.
                 If None, will be built internally (for backward compatibility).
        """
        
        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump"}:
            raise ValueError(f"Unsupported noise_option: {noise_option}")
        
        # For all noise options, use full recalculation approach: ll_modified - ll_current
        # Calculate current state likelihood (only for this assessor)
        current_U_a_dict = {assessor: U_a_dict[assessor]}
        current_orders = {assessor: observed_orders[assessor]}
        current_M_a_dict = {assessor: M_a_dict[assessor]}
        current_O_a_i_dict = {assessor: O_a_i_dict[assessor]}
        
        # Use provided h_U or build it internally
        current_h_U = {}
        if h_U is not None and assessor in h_U:
            current_h_U = {assessor: h_U[assessor]}
        else:
            # Build h_U for current state (vectorized eta computation)
            Ua_current = current_U_a_dict[assessor]
            Ma = current_M_a_dict[assessor]
            eta_a_current = get_ultra_fast_eta_matrix(
                assessor=assessor,
                Ua=Ua_current,
                Ma=Ma,
            )
            H_a_current = BasicUtils.generate_partial_order(eta_a_current)
            current_h_U = {assessor: H_a_current}
        
        ll_current = cls.calculate_log_likelihood_hpo_optimized(
            U={"U0": np.zeros((0, U_a_dict[assessor].shape[1])),
               "U_a_dict": current_U_a_dict},
            h_U=current_h_U,
            observed_orders=current_orders,
            M_a_dict=current_M_a_dict,
            O_a_i_dict=current_O_a_i_dict,
            prob_noise=prob_noise,
            softmax_params=softmax_params,
            noise_option=noise_option
        )
        
        # Calculate modified state likelihood (only for this assessor)
        modified_U_a_dict = {assessor: U_a_dict[assessor].copy()}
        modified_U_a_dict[assessor][row_idx, :] = new_row
        
        # Rebuild h_U for modified assessor using vectorized eta
        Ua_modified = modified_U_a_dict[assessor]
        Ma = current_M_a_dict[assessor]
        eta_a = get_ultra_fast_eta_matrix(
            assessor=assessor,
            Ua=Ua_modified,
            Ma=Ma,
        )
        H_a_modified = BasicUtils.generate_partial_order(eta_a)
        
        modified_h_U = {assessor: H_a_modified}
        
        ll_modified = cls.calculate_log_likelihood_hpo_optimized(
            U={"U0": np.zeros((0, U_a_dict[assessor].shape[1])),
               "U_a_dict": modified_U_a_dict},
            h_U=modified_h_U,
            observed_orders=current_orders,
            M_a_dict=current_M_a_dict,
            O_a_i_dict=current_O_a_i_dict,
            prob_noise=prob_noise,
            softmax_params=softmax_params,
            noise_option=noise_option
        )
        
        return ll_modified - ll_current
    
 
    # -------------------------------------------------------------------------
    @classmethod
    def calculate_log_likelihood_hpo_noise_accelerated(
        cls,
        h_U: Dict[int, np.ndarray],            # assessor → local partial order (REUSED)
        observed_orders: Dict[int, List[List[int]]],
        M_a_dict: Dict[int, List[int]],        # assessor → list of global items
        O_a_i_dict: Dict[int, List[List[int]]],# assessor → list of choice‑sets
        prob_noise: float,
        softmax_params: Optional[Dict[str, float]],
        noise_option: str = "queue_jump",
        U: Optional[Dict[str, Any]] = None    # Required for weighted_queue_jump
    ) -> float:
        """
        ACCELERATED: Calculate likelihood for noise-only updates.
        
        Key optimizations:
        1. Reuses existing h_U (no hierarchical structure recomputation)
        2. Reuses existing U0, U_a_dict (no latent variable changes)
        3. Accelerates eta computation using vectorized functions
        4. Computes bar_eta correctly from accelerated eta values
        """
        if noise_option not in {"queue_jump", "weighted_queue_jump", "log_successors_queue_jump"}:
            raise ValueError(f"Unsupported noise_option: {noise_option}")

        total_ll = 0.0
        MISSING_DATA_PENALTY = -1e10

        for assessor, orders_a in observed_orders.items():
            Ma = M_a_dict[assessor]
            H_a = h_U[assessor]  # Reuse existing partial order
            
            # Map global→local for fast lookup
            g2l = {g: i for i, g in enumerate(Ma)}

            # ACCELERATION: Precompute eta matrix for entire assessor using vectorized functions
            if noise_option == "weighted_queue_jump":
                if U is None:
                    raise ValueError("U required for weighted_queue_jump")
                
                U_a_dict = U.get("U_a_dict", {})
                Ua = U_a_dict[assessor]
                
                # Use accelerated eta computation
                from src.utils.po_accelerator_eta_accelerated import get_ultra_fast_eta_matrix
                eta_full = get_ultra_fast_eta_matrix(
                    assessor=assessor,
                    Ua=Ua,
                    Ma=Ma,
                )
                bar_eta_full = eta_full.mean(axis=1)  # shape (len(Ma),)

            for task_idx, order in enumerate(orders_a):
                choice_set = O_a_i_dict[assessor][task_idx]
                choice_set_filtered = [g for g in choice_set if g in g2l]
                g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
                local_ids = [g2l[g] for g in choice_set_filtered]
                if not local_ids:
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Sub‑matrix for this choice‑set (reuse existing H_a)
                idx_array = np.array(local_ids)
                H_sub = H_a[idx_array[:, None], idx_array]

                # Observed order as indices within choice_set
                obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
                if not obs_local:
                    total_ll += MISSING_DATA_PENALTY
                    continue

                # Likelihood contribution
                if noise_option == "queue_jump":
                    total_ll += cls._queue_jump_ll(obs_local, H_sub, prob_noise)
                elif noise_option == "weighted_queue_jump":
                    # ACCELERATION: Use precomputed bar_eta_full
                    bar_eta = bar_eta_full[np.array(local_ids)]
                    total_ll += cls._wqueue_jump_ll(obs_local, H_sub, prob_noise, bar_eta)
                elif noise_option == "log_successors_queue_jump":
                    if isinstance(softmax_params, dict):
                        beta_val = softmax_params.get("beta", 1.0)
                        epsilon_val = softmax_params.get("epsilon", 0.01)
                    else:
                        beta_val = 1.0 / max(prob_noise, 0.01)
                        epsilon_val = softmax_params if isinstance(softmax_params, (int, float)) else 0.01
                    total_ll += cls._log_successors_queue_jump_ll(
                        obs_local,
                        H_sub,
                        beta=beta_val,
                        epsilon=epsilon_val
                    )
        return total_ll 
