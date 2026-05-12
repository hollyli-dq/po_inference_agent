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
├── config/              # experiment configs and action maps
├── data/                # Cloud-IaC and WFCommons data/assets
├── linext/              # optional C++ linear-extension helper
├── paper/               # paper text, figure inputs, figure scripts
├── results/             # generated experiment outputs and diagnostics
├── src/
│   ├── analysis/        # postprocessing and evaluation scripts
│   ├── mcmc/            # MCMC samplers
│   └── utils/           # loaders, plotting, likelihood helpers, utilities
├── training_scripts/    # main experiment entry points
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
