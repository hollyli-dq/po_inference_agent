# Hierarchical Partial Orders (HPO) Inference

A Bayesian framework for inferring hierarchical partial orders from ranking data, based on the paper **"Partial Order Hierarchies and Rank-Order Data"**.

## Background

In rank-order data, assessors give preference orders over choice sets. An order is registered as a list giving the elements of its choice set in order from best to worst. Well known parametric models for list-data include the Mallows model and the Plackett-Luce model. These models seek a total order which is "central" to the lists provided by the assessors. Extensions model the list-data as realisations of a mixture of distributions each centred on a total order.

Recent work has relaxed the requirement that the centering order be a total order and instead centre the random lists on a partial order. Lists are random linear extensions of a partial order or linear extensions observed with noise. We give a new hierarchical model for partial orders to handle list data which come in labeled groups. The model reduces to a Plackett-Luce model when the partial order dimension is set equal one and can be used to cluster unlabeled list data.

We carry out Bayesian inference for the poset hierarchy using MCMC. Evaluation of the likelihood costs #P so applications are restricted to choice sets of up to 20 elements.

**Keywords**: Bayesian Inference, Partial Orders, Linear Extensions, Hierarchical Model, Clustering.

## Model

### Hierarchical Prior (BHPOP)

We model a shared global Standard Operating Procedure (SOP) with agent-specific perturbations. Each action instance is a node in a DAG (trace unrolling), so repeated actions are treated as distinct temporal occurrences.

#### Latent Gaussian Hierarchy

$$
U_{j,:}^{(0)} \sim \mathcal N\bigl(\mathbf 0,\Sigma_\rho\bigr), \quad j\in M_0,
$$

$$
U_{j,:}^{(m)} \mid U_{j,:}^{(0)} \sim \mathcal N\!\Bigl(\tau\,U_{j,:}^{(0)},\,(1-\tau^{2})\Sigma_\rho\Bigr), \quad m\in \{1,\dots,M\},\; j\in M_m,
$$

$$
a_i \succ_h a_j \;\Longleftrightarrow\; U_{i,k}^{(m)} > U_{j,k}^{(m)} \;\; \forall k \in \{1,\dots,K\},
$$

$$
h^{(m)} = \mathrm{Dom}\bigl(U^{(m)}\bigr).
$$

#### Linearization and Noisy Execution

Given a poset $h^{(m)}$, we define a deterministic canonical linearization
$L^\star(h^{(m)})$ via a fixed tie-break topological sort (no sampling over linear
extensions). At time $t$, let $R_t$ be the remaining actions after the prefix
$y_{1:t-1}$, and let the frontier (feasible actions) be

$$
F_t(h^{(m)}) = \{a \in R_t : \nexists\, b \in R_t \ \text{s.t.}\ b \succ_{h^{(m)}} a\}.
$$

For a feasible action, define the queue-jump score as the position of $a$ in the
restriction of $L^\star(h^{(m)})$ to $R_t$:

$$
Q(a; h^{(m)}, t) =
\begin{cases}
-(\mathrm{rank}_{L^\star}(a; R_t) - 1), & a \in F_t(h^{(m)}) \\
-\infty, & a \notin F_t(h^{(m)}).
\end{cases}
$$

The next action is drawn from the frontier-softmax policy:

$$
P(y_t \mid y_{1:t-1}, h^{(m)}, \beta) =
\frac{\exp\{\beta Q(y_t; h^{(m)},t)\}}{\sum_{a' \in F_t(h^{(m)})} \exp\{\beta Q(a'; h^{(m)},t)\}}.
$$

The trace likelihood is evaluated exactly by maintaining the frontier as actions are removed,
avoiding linear-extension counting or Monte Carlo integration.

## Datasets

This package includes analysis for two real-world datasets:

| Dataset | Description | Notebook |
|---------|-------------|----------|
| **3D Sound** | Spatial audio preference rankings | `notebooks/sound_data_analyze_mcmc.ipynb` |
| **Potato (Ghana)** | Agricultural variety preference data from Ghana | `notebooks/potato_data_analyze_mcmc.ipynb` |
| **WFCommons WfInstances (v1.5)** | Workflow DAGs + execution data for graph-recovery benchmarking | `src/utils/wfcommons_loader.py` |

For WFCommons WfInstances, download the dataset from Zenodo (DOI: 10.5281/zenodo.12510982)
and use the loader utilities in `src/utils/wfcommons_loader.py`. The loader can read the
zip file directly and derives a topologically valid execution order using the execution
task list as a tie-break (v1.5 does not include per-task start timestamps).

## WorkArena++ Benchmark (Agent Traces)

This repo includes an end-to-end WorkArena++ integration for BHPOP structure learning.
The workflow abstracts raw UI actions into a shared atomic action vocabulary and runs
frontier-softmax BHPOP inference per task template.

Key files:
- Action abstraction map: `config/workarena_action_map.yaml`
- Experiment script: `training_scripts/workarena_bhpop_experiment.py`

Example run:

```bash
python training_scripts/workarena_bhpop_experiment.py \
  --data-path data/workarena \
  --tasks "Create Incident" "Provision Service" \
  --top-k 3
```

Minimal trace schema (JSON/JSONL):

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

Outputs are written to `notebooks/outputs/workarena_bhpop/` including:
inferred SOP graphs, execution-efficiency metrics, and a paper-ready summary.

## Synthetic Recovery Experiments

Run the synthetic recovery experiment script to generate a ground-truth poset,
sample frontier-softmax traces (beta, epsilon), and measure recovery vs #traces:

```bash
python notebooks/hpo_synthetic_recovery_experiment.py \
  --poset-type random_dag \
  --num-items 8 \
  --betas 0.8,1.5 \
  --epsilons 0.01,0.05 \
  --Ks 2,3 \
  --trace-counts 10,20,50,100 \
  --num-iterations 20000
```

Outputs (metrics JSON + plots) are saved under
`notebooks/outputs/hpo_synthetic_recovery/` by default.

## Project Structure

```
hpo_inference/
├── src/                          # Core source code
│   ├── mcmc/                     # MCMC implementations
│   │   └── hpo_po_hm_mcmc_k_optim.py     # Optimized reversible-jump MCMC
│   └── utils/                    # Utility functions
│       ├── po_fun.py                     # Partial order functions
│       ├── po_fun_plot.py                # Visualization utilities
│       ├── linext_direct.py              # C++ library interface
│       ├── linext_accelerator.py         # Linear extension acceleration
│       ├── po_accelerator_nle.py         # NLE acceleration
│       ├── po_accelerator_nle_optimized.py  # Optimized NLE acceleration
│       ├── po_accelerator_nle_enhanced.py   # Adaptive NLE acceleration
│       ├── po_accelerator_eta_accelerated.py # Eta acceleration
│       ├── clustering_inference.py       # Clustering utilities
│       └── hpo_model_evaluation.py       # Model evaluation
│
├── linext/                       # C++ linear extension library
│   ├── src/                      # C++ source files
│   ├── liblinext.dll             # Windows compiled library
│   ├── liblinext.dylib           # macOS compiled library
│   ├── liblinext.so              # Linux compiled library
│   ├── Makefile.shared           # Build configuration
│   └── README.md                 # Library documentation
│
├── notebooks/                    # Jupyter notebooks
│   ├── hpo_mcmc_simulation_rj.ipynb      # Quick start: Reversible-jump MCMC
│   ├── hpo_mcmc_simulation_assessor_cl.ipynb  # Assessor clustering
│   ├── hpo_mcmc_simulation_list_cl.ipynb # List clustering
│   ├── sound_data_analyze_mcmc.ipynb     # 3D Sound dataset analysis
│   └── potato_data_analyze_mcmc.ipynb    # Potato dataset analysis
│
├── config/                       # Configuration files
│   ├── hpo_mcmc_configuration.yaml       # General MCMC configuration
│   ├── ghana_config.py                   # Ghana dataset configuration
│   └── sound_3d_hpo_configuration.yaml   # 3D sound configuration
│
├── data/                         # Data files
│   ├── 3d_data/                  # 3D sound dataset
│   │   └── Sounds.RData
│   ├── ghana_data/               # Ghana potato dataset
│   │   ├── men_tricot_data.csv
│   │   ├── women_tricot_data.csv
│   │   └── covariates.csv
│   └── generated_data/           # Synthetic data for testing
│
├── environment.yml               # Conda environment specification
├── LINEXT_CPP_INSTALLATION_GUIDE.md  # C++ library installation guide
└── README.md
```

### MCMC Implementation

The main sampler is `src/mcmc/hpo_po_hm_mcmc_k_optim.py`. It supports reversible-jump moves over K and multiple noise models (queue-jump, weighted queue-jump, Mallows, and softmax queue-jump).

## Installation

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate hpo_inference
```

### Step 2: C++ Library (Recommended)

The C++ LinExt library provides **100x speedup** for counting linear extensions. Pre-compiled for Windows (`liblinext.dll`). For macOS/Linux, build from source:

```bash
cd linext
make -f Makefile.shared
```

For detailed instructions, see **[LINEXT_CPP_INSTALLATION_GUIDE.md](LINEXT_CPP_INSTALLATION_GUIDE.md)**.

## Quick Start

The fastest way to get started is with the **Reversible-Jump MCMC simulation notebook**:

```bash
jupyter notebook notebooks/hpo_mcmc_simulation_rj.ipynb
```

This notebook demonstrates:
1. Generating synthetic hierarchical partial orders
2. Creating observed total orders with noise
3. Running reversible-jump MCMC inference (variable K)
4. Analyzing posterior distributions
5. Comparing inferred vs. true partial orders

### Dataset Analyses

```bash
# 3D Sound dataset
jupyter notebook notebooks/sound_data_analyze_mcmc.ipynb

# Potato (Ghana) dataset
jupyter notebook notebooks/potato_data_analyze_mcmc.ipynb
```

## Training Scripts

Training scripts are not included in this repo snapshot; use the notebooks in `notebooks/` for running experiments and analyses.

## Configuration

Model parameters are specified in YAML configuration files in `config/`:

| Parameter | Description |
|-----------|-------------|
| `K` | Latent dimension |
| `num_iterations` | MCMC iterations |
| `rho_prior` | Prior for correlation parameter ρ |
| `noise_option` | Noise model (`queue_jump`, `weighted_queue_jump`, `softmax_queue_jump`) |

## Features

- **Hierarchical Partial Order Generation**: Create synthetic hierarchical partial order structures
- **MCMC Inference**: Reversible-jump sampling over latent dimension K
- **Flexible Noise Models**: Queue-jump, weighted queue-jump, and softmax queue-jump
- **C++ Acceleration**: 100x speedup for linear extension counting
- **Posterior Analysis**: Hyperparameter distributions and structural recovery

## References

1. Chuxuan Jiang and Geoff K. Nicholls. *Partial Order Hierarchies and Rank-Order Data*.

2. Chuxuan Jiang and Geoff K. Nicholls. Bayesian inference for partial orders with ties from ranking data with a Plackett-Luce distribution centred on random linear extensions, 2021.

3. Chuxuan Jiang, Geoff K. Nicholls, and Jeong Eun Lee. Bayesian inference for vertex-series-parallel partial orders. In *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence*, volume 216 of Proceedings of Machine Learning Research, pages 995–1004. PMLR, 2023.
