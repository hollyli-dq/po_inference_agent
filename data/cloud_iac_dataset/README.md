# Cloud-IaC-6: A Benchmark Dataset for DAG Recovery from Execution Traces

## Overview

**Cloud-IaC-6** is a benchmark dataset for evaluating partial order (DAG) inference algorithms from execution trace data. The dataset contains execution traces collected from AI agents performing cloud infrastructure deployment tasks, where the ground truth task dependency graph is known.

This dataset accompanies the paper: *"Bayesian Hierarchical Partial Order Inference from Agent Execution Traces"* (ICML 2026).

## Task Description

Cloud infrastructure deployment requires executing a sequence of API calls with dependency constraints. For example:
- A VPC must be created before a VSwitch can be created inside it
- A Security Group must be created and configured before an ECS instance can use it
- A Load Balancer must exist before backend servers can be attached

These dependencies form a Directed Acyclic Graph (DAG). Different execution strategies (topological sorts of the DAG) are valid, but the underlying dependency structure is fixed. The task is to **recover the ground truth DAG from observing execution traces**.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Number of Scenarios | 6 |
| Total Expert Traces | 6 |
| Total LLM Agent Traces | 61 |
| Actions per Scenario | 5-15 |
| Unique Action Types | 16 |

## Directory Structure

```text
data/cloud_iac_dataset/
├── README.md                    # This file
├── ground_truth/                # Ground truth DAG definitions
│   ├── simple_ecs.json
│   ├── slb_ecs_rds.json
│   ├── slb_ecs_redis.json
│   ├── eip_slb_ecs.json
│   ├── dual_zone_ecs_slb.json
│   └── dual_zone_ecs_slb_rds.json
├── expert_traces/               # Traces from expert (topological sort) policy
│   └── *.json
└── traces/                      # Traces from LLM-based agents
    └── *.json
```

## Scenario Descriptions

### 1. Simple ECS (`simple_ecs`)
**Description:** Deploys a VPC network and a single ECS instance.

**Actions:** CreateVpc → CreateVSwitch, CreateSecurityGroup → AuthorizeSecurityGroup → RunInstances

**Graph:**
```
CreateVpc
├── CreateVSwitch ──────────┐
└── CreateSecurityGroup     │
         │                  │
         v                  v
    AuthorizeSecurityGroup──┴── RunInstances
```

### 2. SLB + ECS + RDS (`slb_ecs_rds`)
**Description:** Full-stack web application with load balancer, compute, and database.

**Actions:** 12 nodes including VPC setup, SLB configuration, ECS deployment, and RDS provisioning with listener and backend attachment.

### 3. SLB + ECS + Redis (`slb_ecs_redis`)
**Description:** Web application with caching layer.

**Actions:** 10 nodes including VPC, SLB, ECS, and Redis instance setup.

### 4. EIP + SLB + ECS (`eip_slb_ecs`)
**Description:** Public-facing web application with elastic IP.

**Actions:** 10 nodes including EIP allocation and association with SLB.

### 5. Dual Zone ECS + SLB (`dual_zone_ecs_slb`)
**Description:** High-availability deployment across two availability zones.

**Actions:** 8 nodes with ECS instances in multiple zones behind a load balancer.

### 6. Dual Zone ECS + SLB + RDS (`dual_zone_ecs_slb_rds`)
**Description:** Production-grade HA architecture with dual-zone compute and database.

**Actions:** 10 nodes with full HA setup including RDS high-availability cluster.

## Data Format

### Ground Truth Format (`ground_truth/*.json`)

```json
{
  "name": "Simple ECS",
  "description": "Deploys a VPC network and a single ECS instance.",
  "edges": [
    ["CreateVpc", "CreateVSwitch"],
    ["CreateVpc", "CreateSecurityGroup"],
    ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
    ["CreateVSwitch", "RunInstances"],
    ["AuthorizeSecurityGroup", "RunInstances"]
  ]
}
```

The `edges` field contains a list of directed edges `[source, target]` representing the dependency relation: `source` must complete before `target` can begin.

### Trace Format (`traces/*.json`, `expert_traces/*.json`)

```json
{
  "trace_id": "trace_T01_model_timestamp_hash",
  "task": "Task description in natural language",
  "intent": {
    "intent_type": "simple_ecs",  // Maps to ground truth scenario
    ...
  },
  "mode": "expert" | "explore",   // Execution mode
  "status": "success",
  "action_sequence": [            // Observed execution order (linear extension)
    "CreateVpc",
    "CreateVSwitch",
    "CreateSecurityGroup",
    "AuthorizeSecurityGroup",
    "RunInstances"
  ],
  "actions": [...]                // Detailed action metadata (optional)
}
```

The key field for DAG recovery is `action_sequence`, which represents the observed total order (a valid topological sort of the ground truth DAG, possibly with noise).

## Action Vocabulary

The dataset uses the following cloud API actions:

| Action | Description | Service |
|--------|-------------|---------|
| CreateVpc | Create Virtual Private Cloud | VPC |
| CreateVSwitch | Create subnet within VPC | VPC |
| CreateSecurityGroup | Create network security group | ECS |
| AuthorizeSecurityGroup | Configure security rules | ECS |
| RunInstances | Launch ECS compute instances | ECS |
| CreateLoadBalancer | Create Server Load Balancer | SLB |
| CreateLoadBalancerHTTPListener | Configure HTTP listener | SLB |
| StartLoadBalancerListener | Activate listener | SLB |
| AddBackendServers | Attach servers to load balancer | SLB |
| CreateDBInstance | Create RDS database instance | RDS |
| CreateAccount | Create database user account | RDS |
| ModifySecurityIps | Configure database access whitelist | RDS |
| CreateInstance | Create Redis cache instance | Redis |
| DescribeInstanceAttribute | Query Redis instance status | Redis |
| AllocateEipAddress | Allocate Elastic IP | EIP |
| AssociateEipAddress | Bind EIP to resource | EIP |

## Usage Examples

### Loading the Dataset (Python)

```python
import json
from pathlib import Path

def load_ground_truth(data_dir: Path) -> dict:
    """Load all ground truth DAGs."""
    gt_dir = data_dir / "ground_truth"
    scenarios = {}
    for f in gt_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            scenarios[f.stem] = data
    return scenarios

def load_traces(data_dir: Path) -> list:
    """Load all execution traces."""
    traces = []
    for trace_dir in ["expert_traces", "traces"]:
        for f in (data_dir / trace_dir).glob("*.json"):
            with open(f) as fp:
                traces.append(json.load(fp))
    return traces

def get_traces_for_scenario(traces: list, scenario_id: str) -> list:
    """Filter traces for a specific scenario."""
    return [t for t in traces if t.get("intent", {}).get("intent_type") == scenario_id]

# Example usage
data_dir = Path("data/cloud_iac_dataset")
ground_truth = load_ground_truth(data_dir)
traces = load_traces(data_dir)

# Get traces for "slb_ecs_rds" scenario
slb_traces = get_traces_for_scenario(traces, "slb_ecs_rds")
print(f"Found {len(slb_traces)} traces for slb_ecs_rds")
```

### Converting to Adjacency Matrix

```python
import numpy as np

def edges_to_adjacency(edges: list, nodes: list) -> np.ndarray:
    """Convert edge list to adjacency matrix."""
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=np.int8)
    for src, tgt in edges:
        if src in node_to_idx and tgt in node_to_idx:
            adj[node_to_idx[src], node_to_idx[tgt]] = 1
    return adj

# Example
scenario = ground_truth["simple_ecs"]
nodes = ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", 
         "AuthorizeSecurityGroup", "RunInstances"]
adj = edges_to_adjacency(scenario["edges"], nodes)
```

## Evaluation Metrics

We recommend the following metrics for evaluating DAG recovery:

1. **F1 Score**: Harmonic mean of precision and recall for edge recovery
2. **Structural Hamming Distance (SHD)**: Number of edge additions, deletions, and reversals needed to match ground truth
3. **Feasibility**: Fraction of traces that are valid topological sorts of the inferred DAG

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{author2026bhpop,
  title={Bayesian Hierarchical Partial Order Inference from Agent Execution Traces},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This dataset is released under the CC BY 4.0 license for academic research purposes.

## Acknowledgments

The execution traces were collected using a simulated cloud environment that mirrors the API structure of real cloud infrastructure services. No actual cloud resources were created or personal data collected.
