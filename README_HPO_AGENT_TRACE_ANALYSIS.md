# HPO Agent Trace Analysis

This README shows how to run the agent trace analysis script and where to place configuration.

## Run Command

From the repo root:

```bash
python notebooks/hpo_agent_trace_analysis.py --output-dir test1 --matrix-json h_posteriors.json --num-iterations 100000
```

This will:
- run 100,000 iterations,
- save plots, profiles, and JSON outputs under `test1/`,
- write the posterior matrices to `test1/h_posteriors.json`.

## Data File

The script reads the input data file from:

```
data/iac_cloud.csv
```

If you want to use a different dataset, update the `data_path` in
`notebooks/hpo_agent_trace_analysis.py`.

## Configuration

Put the configuration file here:

```
config/hpo_agent_trace_analysis.yaml
```

The script reads this file automatically and uses it as the default settings.
CLI flags (like `--num-iterations` and `--output-dir`) override the YAML file values.

## Conda Environment

From the repo root:

```bash
conda env create -f environment.yml
conda activate hpo_inference
```
