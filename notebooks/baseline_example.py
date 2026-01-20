"""
Example script showing how to use baseline methods for comparison.

Run this after running the aliyun_manual_step_by_step.ipynb notebook
to get the data variables (observed_orders, M_a_dict, true_cover_global, etc.)
"""

import sys
sys.path.append('../src')

from src.utils.hpo_model_evaluation import (
    evaluate_baselines,
    baseline_always_before_intersection,
    baseline_majority_projection,
    precision_recall,
    f1_score,
    structural_hamming_distance
)

def main():
    """
    Example usage of baseline methods.

    This assumes you have run the aliyun_manual_step_by_step.ipynb notebook
    and have the following variables in memory:
    - observed_orders: Dict[assessor_id -> list of traces]
    - M_a_dict: Dict[assessor_id -> list of items]
    - true_cover_global: Dict[assessor_id -> true cover matrix]
    - orders_local_by_assessor: Dict[assessor_id -> list of local traces]
    - scenario_data: Dict[scenario_id -> scenario info]
    """

    print("="*80)
    print("BASELINE METHODS USAGE EXAMPLES")
    print("="*80)

    # Example 1: High-level evaluation across all scenarios
    print("\n1. High-level evaluation across all scenarios:")
    print("-" * 50)

    # Assuming you have these variables from the notebook:
    # baseline_results = evaluate_baselines(
    #     observed_orders=observed_orders,
    #     M_a_dict=M_a_dict,
    #     true_cover=true_cover_global,
    #     baseline_method="both",
    #     majority_threshold=0.5
    # )

    # For this example, let's show the API:
    print("""
    baseline_results = evaluate_baselines(
        observed_orders=observed_orders,      # From notebook
        M_a_dict=M_a_dict,                     # From notebook
        true_cover=true_cover_global,          # From notebook
        baseline_method="both",                # "always_before", "majority", or "both"
        majority_threshold=0.5                 # Threshold for majority baseline
    )

    # Results format:
    # baseline_results[assessor_id]["always_before"]["f1"] - Always-Before F1 score
    # baseline_results[assessor_id]["majority"]["f1"] - Majority F1 score
    """)

    # Example 2: Manual evaluation for one scenario
    print("\n2. Manual evaluation for one scenario:")
    print("-" * 50)

    print("""
    # Pick a scenario
    scenario_id = "dual_zone_ecs_slb"

    # Get local data for this scenario
    orders_local = orders_local_by_assessor[scenario_id]  # Local indices
    true_cover = scenario_data[scenario_id]["true_cover"]  # True cover matrix
    n_items = len(scenario_data[scenario_id]["task_ids"])  # Number of items

    # Baseline A: Always-Before (Intersection)
    baseline_a = baseline_always_before_intersection(orders_local, n_items)

    # Evaluate against true cover
    prec_a, rec_a = precision_recall(true_cover, baseline_a)
    f1_a = f1_score(prec_a, rec_a)
    shd_a = structural_hamming_distance(true_cover, baseline_a)

    print(f"Always-Before: F1={f1_a:.3f}, P={prec_a:.3f}, R={rec_a:.3f}, SHD={shd_a}")

    # Baseline B: Majority (with different thresholds)
    for threshold in [0.3, 0.5, 0.7]:
        baseline_b = baseline_majority_projection(orders_local, n_items, threshold=threshold)
        prec_b, rec_b = precision_recall(true_cover, baseline_b)
        f1_b = f1_score(prec_b, rec_b)
        shd_b = structural_hamming_distance(true_cover, baseline_b)
        print(f"Majority (thr={threshold}): F1={f1_b:.3f}, P={prec_b:.3f}, R={rec_b:.3f}, SHD={shd_b}")
    """)

    # Example 3: Understanding what the baselines do
    print("\n3. Understanding what the baselines do:")
    print("-" * 50)

    print("""
    ALWAYS-BEFORE (INTERSECTION):
    - Adds edge i→j ONLY if i precedes j in ALL traces
    - Very conservative - requires unanimous agreement
    - Often results in very few edges (sparse graph)
    - Good for high-precision scenarios

    MAJORITY (PROJECTION):
    - Adds edge i→j if p(i before j) > threshold (default 0.5)
    - Based on majority voting across traces
    - Resolves conflicts and ensures acyclicity
    - More edges than Always-Before, potentially more recall

    EVALUATION METRICS:
    - Precision: Fraction of predicted edges that are correct
    - Recall: Fraction of true edges that are recovered
    - F1: Harmonic mean of precision and recall
    - SHD (Structural Hamming Distance): Total edge errors (missing + extra)
    """)

    # Example 4: For paper tables
    print("\n4. For paper tables - comparing to your method:")
    print("-" * 50)

    print("""
    # After running your MCMC, compare:
    # Your method vs Always-Before vs Majority

    results_table = []
    for scenario in ["dual_zone_ecs_slb", "dual_zone_ecs_slb_rds", ...]:
        # Your MCMC results (from final_H_mode)
        # your_f1 = compute_f1_for_scenario(scenario, final_H_mode[scenario])

        # Baseline results
        # baseline_a_f1 = baseline_results[scenario]["always_before"]["f1"]
        # baseline_b_f1 = baseline_results[scenario]["majority"]["f1"]

        results_table.append({
            "scenario": scenario,
            "your_method": your_f1,
            "always_before": baseline_a_f1,
            "majority": baseline_b_f1
        })

    # Show table with different coverage levels
    # coverage=1.0 (full traces), coverage~0.5 (partial traces)
    """)

    print("\n" + "="*80)
    print("Run the cells in aliyun_manual_step_by_step.ipynb to see actual results!")
    print("="*80)

if __name__ == "__main__":
    main()
