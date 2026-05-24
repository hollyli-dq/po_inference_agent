# De-Linearizing Agent Traces with Bayesian Partial Orders

This repository contains the code, datasets, experiment drivers, and paper assets for
**De-Linearizing Agent Traces: Bayesian Inference of Latent Partial Orders for Efficient Execution**
([arXiv:2602.02806](https://arxiv.org/abs/2602.02806)).

The core workflow is:

1. infer a latent partial order from observed execution traces,
2. aggregate the posterior into a usable graph,
3. evaluate either structural recovery or downstream execution efficiency.

The repo still contains some older partial-order code, but the current organization is centered on
the paper’s two main experiment families:

- Cloud-IaC execution experiments
- WFCommons structural recoverability experiments

## Main Entry Points

Use `training_scripts/` for experiment runs. Do not treat `notebooks/` as the main execution path.

### Cloud-IaC / execution-efficiency experiments

- `training_scripts/systematic_experiments.py`
  Main paper driver for Cloud-IaC experiments.
- `training_scripts/queue_jump_baseline_cloud_iac.py`
  Queue-jump baseline runs on the Cloud-IaC dataset.
- `training_scripts/queue_jump_ablation.py`
  Likelihood/noise ablations for the Cloud-IaC setting.
- `training_scripts/uncertainty_aware_execution_ablation.py`
  Execution ablations using posterior uncertainty.

### WFCommons / recoverability experiments

- `training_scripts/wfcommons_montage_recovery.py`
  Main recoverability sweep over WFCommons instances.
- `training_scripts/wfinstances_srasearch_experiments.py`
- `training_scripts/wfinstances_epigenomics_experiments.py`
- `training_scripts/wfinstances_montage_experiments.py`

### WorkArena++ integration

- `training_scripts/workarena_bhpop_experiment.py`

This is an auxiliary integration path, not the main paper artifact.

### Execution experiments (§4 Tri-Modal, §5.4.2 Efficient Execution)

The execution-efficiency experiments require both the simulator and the
Tri-Modal CloudOps agent (newly added in this release):

- `execution/aliyun_gym/`     — mock Aliyun cloud simulator with virtual clock
- `execution/cloudops_agent/` — Tri-Modal agent (Expert / Hybrid / Explore + GEE)
- `simulation_workspace/`     — experiment drivers + 6 Cloud-IaC ground-truth
                                scenarios (`manual_scenarios/*.json`) and the
                                IP-Cov=1.0 inferred posets (`best_posets.json`).

Main entry points:

- `simulation_workspace/mode_comparison_experiment.py` — Table 4 driver
- `simulation_workspace/hpo_batch_experiment.py`       — Table 3 IP-Cov sweep
- `simulation_workspace/expert_trace_gen.py`           — Expert mode (no LLM)
- `simulation_workspace/hybrid_trace_gen.py`           — Hybrid mode
- `simulation_workspace/explore_trace_gen_parallel.py` — Explore mode
- `simulation_workspace/diverse_trace_gen.py`          — heterogeneous-LLM
                                                         trace acquisition
                                                         (App. trace_acquisition)

LLM API keys (used by Hybrid / Explore modes only; Expert mode requires no
keys) should be placed in `.env` (template in `.env.example`).

For a complete, self-contained reproduction guide aimed at a fresh reader,
see the per-experiment READMEs under `simulation_workspace/` and
`training_scripts/`.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate hpo_inference
```

Optional: build the linear-extension helper library:

```bash
cd linext
make -f Makefile.shared
```

For headless plotting and multiprocessing, it helps to set:

```bash
export MPLCONFIGDIR=/private/tmp/mplconfig
mkdir -p "$MPLCONFIGDIR"
```

## Quick Start

### 1. Cloud-IaC smoke test

```bash
python training_scripts/systematic_experiments.py --test
```

### 2. Full Cloud-IaC experiment sweep

```bash
python training_scripts/systematic_experiments.py
```

### 3. WFCommons recoverability run

```bash
python training_scripts/wfcommons_montage_recovery.py \
  --data_dir data/wfcommons_wfinstances/montage \
  --prefer_execution_order \
  --num_iterations 20000
```

### 4. WorkArena experiment

```bash
python training_scripts/workarena_bhpop_experiment.py \
  --data-path data/workarena \
  --tasks "Create Incident" "Provision Service" \
  --top-k 3
```

## Postprocessing And Analysis

Postprocessing utilities now live under `src/analysis/`.

Examples:

```bash
python -m src.analysis.generate_experiment_summaries
python -m src.analysis.recompute_H_with_threshold
python -m src.analysis.recompute_H_wfinstances --help
python -m src.analysis.cloud_iac_cp_cov --help
```

## Data And Code Layout

```text
po_inference_agent/
├── config/                  # experiment configs and action maps
├── data/                    # Cloud-IaC and WFCommons data/assets
├── linext/                  # optional C++ linear-extension helper
├── paper/                   # paper text, figure inputs, figure scripts
├── results/                 # generated experiment outputs and diagnostics
├── src/                     # BPOP inference (MCMC + analysis + utilities)
│   ├── analysis/            # postprocessing and evaluation scripts
│   ├── mcmc/                # MCMC samplers
│   └── utils/               # loaders, plotting, likelihood helpers
├── training_scripts/        # main inference experiment entry points
├── execution/               # NEW: simulator + Tri-Modal agent (§4, §5.4.2)
│   ├── aliyun_gym/          # mock Aliyun cloud environment
│   └── cloudops_agent/      # Expert / Hybrid / Explore agent + GEE
├── simulation_workspace/    # NEW: execution-experiment drivers (§5.4.2)
│   └── manual_scenarios/    # 6 Cloud-IaC ground-truth posets
├── .env.example             # LLM API key template
└── environment.yml
```

## Dependencies

The current environment file is aligned to the active paper workflow.

- `pm4py` is required for the process-mining baselines used by the Cloud-IaC pipeline.
- `openpyxl` is required for legacy spreadsheet-based Cloud-IaC ingestion utilities.
- `pygraphviz` is useful for graph rendering, but some scripts can still run with reduced
  visualization support if it is unavailable.

If plotting is slow or noisy on shared machines, set `MPLCONFIGDIR` as shown above.

## Notes

- `notebooks/` is no longer the main home for runnable experiment drivers.
- generated experiment outputs now live under `results/`, not `notebooks/`.
- Shared figure logic is centralized in `src/utils/icml_visual.py`.
- The sampler implementation used by the current single-partial-order workflow is
  `src/mcmc/hpo_po_hm_mcmc_k_optim.py`.

## Reference

Chuxuan Jiang and Geoff K. Nicholls. *De-Linearizing Agent Traces: Bayesian Inference of Latent
Partial Orders for Efficient Execution*. arXiv:2602.02806.
