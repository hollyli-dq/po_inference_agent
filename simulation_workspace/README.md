# Simulation Workspace (Execution Experiments)

This directory hosts the experiment drivers for the Tri-Modal CloudOps agent
(paper §6.6.2 / `tab:exec_modes_comparison` and `tab:ipcov_effect`). It depends
on the simulator and agent in `../execution/`, and on `HPO_scenarios/` (BPOP
posterior posets produced by `training_scripts/systematic_experiments.py`).

## Files

| File | Used in paper | Description |
|---|---|---|
| `mode_comparison_experiment.py` | **Table `tab:exec_modes_comparison`** | Drives Expert / Hybrid / Explore / Explore+CoT on the 6 Cloud-IaC scenarios using **BPOP-inferred posets at IP-Cov=1.0** (loaded via `best_posets.json`). |
| `hpo_batch_experiment.py` | **Table `tab:ipcov_effect`** | IP-Cov sweep over the 120 BPOP posteriors in `HPO_scenarios/` (Hybrid mode only). |
| `compare_modes_analyzer.py` | Table 4 post-hoc | Aggregates trace JSONs from `mode_comparison_experiment.py` into a Markdown report. |
| `best_poset_selector.py` + `best_posets.json` | §6.6.2 prerequisite | Selects the IP-Cov=1.0 highest-F1 BPOP poset per scenario. **Must be re-run after regenerating `HPO_scenarios/`** to refresh absolute paths. |
| `manual_scenarios/*.json` | App. ground-truth | Hand-authored ground-truth posets for the 6 scenarios. **Used only by the auxiliary scripts below**, not by the paper tables. |
| `expert_trace_gen.py` | Auxiliary baseline | Generates "ideal expert" traces by running Expert mode on the **ground-truth** poset. NOT the Table 4 Expert column. |
| `hybrid_trace_gen.py` | Auxiliary baseline | Same as above but with Hybrid mode. NOT the Table 4 Hybrid column. |
| `explore_trace_gen.py` / `explore_trace_gen_parallel.py` | Auxiliary baseline | Pure ReAct loop without any poset (intermediate dev tool). |
| `diverse_trace_gen.py` | App. trace_acquisition | Heterogeneous LLM trace generator. |
| `detailed_trace_experiment.py` | App. likelihood example | Full prompt/response logger. |
| `diversity_analyzer.py` | App. trace diversity | IP-Cov estimation across traces. |

## Execution chains (run in this order)

### Chain A — §6.6.1 structural recovery (no LLM key)
```bash
python ../training_scripts/systematic_experiments.py        # produces HPO_scenarios/
```

### Chain B — Table `tab:ipcov_effect` (needs LLM key)
```bash
# Prereq: HPO_scenarios/ + experiment_summary.csv exist
cd simulation_workspace
python hpo_batch_experiment.py
```

### Chain C — Table `tab:exec_modes_comparison` (needs LLM key)
```bash
# Prereq: HPO_scenarios/ exists
cd simulation_workspace
python best_poset_selector.py        # rewrites best_posets.json with absolute paths
python mode_comparison_experiment.py --mode expert
python mode_comparison_experiment.py --mode hybrid
python mode_comparison_experiment.py --mode explore --thinking off
python mode_comparison_experiment.py --mode explore --thinking on
python compare_modes_analyzer.py     # aggregate report
```

> The auxiliary `expert_trace_gen.py` / `hybrid_trace_gen.py` use
> `manual_scenarios/` (ground truth) and are **not** the source of any number
> in the paper tables. Use them only for sanity-checking the simulator/agent
> stack.

## Reproducing the experiments

See the **Execution chains** section above for the canonical, self-contained
reproduction recipes (Chain A / B / C).
