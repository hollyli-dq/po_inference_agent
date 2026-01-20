"""
Debug script to investigate low acceptance rates with few traces.

Key questions:
1. Are traces being skipped (contributing 0.0 to likelihood)?
2. Are log-likelihood values reasonable?
3. Are proposal changes too large relative to current likelihood?
"""

import numpy as np
from src.utils.po_accelerator_nle_optimized import HPO_LogLikelihoodCache_Optimized
from src.utils.po_fun import BasicUtils, StatisticalUtils

def debug_likelihood_calculation(
    observed_orders: list,
    choice_sets: list,
    items: list,
    U: np.ndarray,
    h: np.ndarray,
    beta: float = 1.0,
    epsilon: float = 0.01,
):
    """
    Debug version that shows per-trace contributions.
    """
    print(f"\n{'='*70}")
    print(f"DEBUGGING LIKELIHOOD CALCULATION")
    print(f"{'='*70}")
    print(f"Number of traces: {len(observed_orders)}")
    print(f"Number of items: {len(items)}")
    print(f"U shape: {U.shape}")
    print(f"h shape: {h.shape}, edges: {h.sum()}")
    
    total_ll = 0.0
    item_to_index = {item: i for i, item in enumerate(items)}
    g2l = {g: i for i, g in enumerate(items)}
    
    skipped_traces = 0
    trace_contributions = []
    
    for task_idx, order in enumerate(observed_orders):
        choice_set = choice_sets[task_idx]
        choice_set_filtered = [g for g in choice_set if g in g2l]
        g_to_local_idx = {g: i for i, g in enumerate(choice_set_filtered)}
        local_ids = [g2l[g] for g in choice_set_filtered]
        
        if not local_ids:
            print(f"  ⚠️  Trace {task_idx}: SKIPPED (empty local_ids after filtering)")
            skipped_traces += 1
            continue
        
        H_sub = h[np.ix_(local_ids, local_ids)]
        obs_local = [g_to_local_idx[g] for g in order if g in g_to_local_idx]
        
        if not obs_local:
            print(f"  ⚠️  Trace {task_idx}: SKIPPED (empty obs_local after filtering)")
            skipped_traces += 1
            continue
        
        # Compute likelihood for this trace
        trace_ll = HPO_LogLikelihoodCache_Optimized._log_successors_queue_jump_ll(
            obs_local, H_sub, beta=beta, epsilon=epsilon
        )
        
        trace_contributions.append(trace_ll)
        total_ll += trace_ll
        
        if task_idx < 5:  # Show first 5 traces
            print(f"  Trace {task_idx}: len={len(order)}, local_len={len(obs_local)}, "
                  f"ll={trace_ll:.3f}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Total traces processed: {len(observed_orders) - skipped_traces}/{len(observed_orders)}")
    print(f"  Skipped traces: {skipped_traces}")
    print(f"  Total log-likelihood: {total_ll:.3f}")
    if trace_contributions:
        print(f"  Mean per-trace ll: {np.mean(trace_contributions):.3f}")
        print(f"  Std per-trace ll: {np.std(trace_contributions):.3f}")
        print(f"  Min per-trace ll: {np.min(trace_contributions):.3f}")
        print(f"  Max per-trace ll: {np.max(trace_contributions):.3f}")
    print(f"{'='*70}\n")
    
    return total_ll, skipped_traces, trace_contributions


def compare_likelihoods_different_trace_counts():
    """
    Compare log-likelihoods with different numbers of traces to see scaling.
    """
    print("\n" + "="*70)
    print("COMPARING LIKELIHOOD SCALING")
    print("="*70)
    print("With 12 traces, total_ll should be roughly (12/50) = 0.24x of 50 traces")
    print("This means proposals need to be proportionally smaller to maintain similar acceptance rates.\n")


if __name__ == "__main__":
    print("Run this in your notebook after setting up U, h, observed_orders, etc.")
    print("Call: debug_likelihood_calculation(observed_orders_list, choice_sets_list, list(M0), U, h)")
