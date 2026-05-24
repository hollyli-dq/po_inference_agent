# Simulation Workspace (Execution Experiments)

This directory hosts the experiment drivers for the Tri-Modal CloudOps agent
(§4 and §5.4.2 of the paper). It depends on the simulator and agent in
`../execution/`.

## Files

| File | Used in paper | Description |
|---|---|---|
| `mode_comparison_experiment.py` | Table 4 | Drives Expert / Hybrid / Explore modes on the 6 Cloud-IaC scenarios. |
| `hpo_batch_experiment.py` | Table 3 | IP-Cov sweep ($IP\text{-}Cov\in\{0.6,\dots,1.0\}$). |
| `expert_trace_gen.py` | Tri-Modal | Pure POSET execution, no LLM call. |
| `hybrid_trace_gen.py` | Tri-Modal | POSET + LLM fallback. |
| `explore_trace_gen.py` / `explore_trace_gen_parallel.py` | Tri-Modal | Pure ReAct loop. |
| `diverse_trace_gen.py` | App. trace_acquisition | Heterogeneous LLM trace generator. |
| `detailed_trace_experiment.py` | App. likelihood example | Full prompt/response logger. |
| `compare_modes_analyzer.py` | Table 4 post-hoc | Aggregates trace JSONs into report. |
| `diversity_analyzer.py` | App. trace diversity | IP-Cov estimation across traces. |
| `best_poset_selector.py` + `best_posets.json` | §5.4.2 | Loads IP-Cov=1.0 inferred posets. |
| `manual_scenarios/*.json` | §5.1 | Ground-truth posets for the 6 scenarios. |

## Reproducing the experiments

See [`../AI_REPRODUCE_GUIDE.md`](../AI_REPRODUCE_GUIDE.md) for the canonical,
self-contained reproduction recipe.
