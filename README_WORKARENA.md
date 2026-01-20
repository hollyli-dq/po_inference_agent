# WorkArena++ BHPOP Integration

This document describes the WorkArena++ integration added in this task. It covers
data modeling, abstraction, inference, and evaluation outputs for BHPOP.

## Conceptual Mapping

- Task: A single WorkArena task template (one latent SOP / poset h).
- Agent (assessor m): A distinct execution policy that produces multiple successful
  traces for the same task (prompt/seed/model variants).
- Atomic actions (A): High-level semantic steps abstracted from raw UI actions.
- Trace Y^(m,i): A successful execution trajectory for agent m on task i,
  represented as a sequence of atomic action labels.

## Data Requirements

The loader accepts JSON/JSONL with a list of records. Each record should include:
- task_id or task_name (template identifier)
- agent_id (policy identifier)
- success (boolean)
- actions / steps / trajectory / trace / events (list of action objects)

Minimal example:

```json
{
  "task_id": "create_incident",
  "task_name": "Create and resolve an incident",
  "agent_id": "prompt_v1_seed42",
  "success": true,
  "actions": [
    {"action": "open incident form", "observation": "form loaded"},
    {"action": "fill short description", "fields": {"short_description": "VPN down"}},
    {"action": "assign priority", "fields": {"priority": "2"}},
    {"action": "submit", "result": "INC001234 created"}
  ]
}
```

## Action Abstraction

Action abstraction is driven by `config/workarena_action_map.yaml`:
- `task_patterns`: task-specific regex patterns -> action labels.
- `global_patterns`: fallback rules shared across tasks.
- `ignore_patterns`: drop low-level actions (click/scroll/etc.).
- `artifact_fields`: text fields to include as action artifacts.

Edit this file to align with your WorkArena++ templates and naming conventions.

## Trace Unrolling

Repeated actions are treated as distinct event instances by suffixing:
- `Label` -> `Label#2`, `Label#3`, ...
This matches BHPOP's trace-unrolling assumption and allows repeated steps in a trace.

## Pipeline Overview

1. Load traces and apply action abstraction.
2. Build shared action vocabulary across agents for a task.
3. Convert traces to BHPOP inputs (M0, assessors, observed_orders, etc.).
4. Run BHPOP with frontier-softmax likelihood and canonical plan L*(h).
5. Extract inferred SOPs and posterior summaries.
6. Evaluate structure-aware execution vs full-history baseline.

## Running the Experiment

Script:
- `training_scripts/workarena_bhpop_experiment.py`

Example:

```bash
python training_scripts/workarena_bhpop_experiment.py \
  --data-path data/workarena \
  --tasks "Create Incident" "Provision Service" \
  --top-k 3 \
  --seeds 42 43
```

Key arguments:
- `--data-path`: JSON/JSONL file or directory of traces.
- `--action-map`: action abstraction map YAML.
- `--tasks`: optional list of tasks to run (by id or name substring).
- `--top-k`: fallback to top-k tasks by trace count.
- `--seeds`: multiple seeds for stability checks.

## Outputs

All outputs are written under `notebooks/outputs/workarena_bhpop/`:

- `*/mcmc_results_seed_*.pkl`: raw MCMC results per task/seed
- `*/inferred_graphs_seed_*.json`: inferred SOP edges with action labels
- `*/partial_order_*.png`: SOP visualizations (global + agent-specific)
- `*/execution_metrics_seed_*.json`: token/latency metrics
- `*/stability_across_seeds.json`: seed stability (Jaccard over edges)
- `workarena_bhpop_summary.json`: merged metrics
- `experiment_summary.md`: short paper-ready summary

## Execution Efficiency Evaluation

The execution engine uses inferred SOPs to:
- compute frontier (parallelizable actions),
- optionally execute all frontier actions in parallel,
- prune context to direct prerequisites (and optional dependents),
and compares to a baseline that conditions on full history.

Reported metrics:
- total tokens (estimated from action context artifacts)
- latency (sum of step durations with parallel max)
- success proxy (valid linear extension check)

## Notes / Assumptions

- The integration focuses on correct data modeling and experimental alignment,
  not UI automation.
- The likelihood is frontier-softmax with deterministic canonical plan L*(h),
  lambda fixed to 1 and beta learned (per BHPOP formulation).
- You do not need ground-truth DAGs; BHPOP infers SOPs from traces.

