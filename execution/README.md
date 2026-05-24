# Execution Module (`execution/`)

Mock cloud simulator and Tri-Modal CloudOps agent used by the execution
experiments in §4 and §5.4.2 of the paper.

## Layout

```
execution/
├── aliyun_gym/        # Mock Aliyun environment (VPC / ECS / SLB / RDS / Redis / CMS / OOS)
│   ├── core/          # Action router, validators, resource manager, virtual clock
│   ├── handlers/      # Per-product handlers
│   ├── mock_clients/  # Simulated SDK clients
│   ├── factory.py     # `create_gym_env()` entry point
│   └── config.py      # `LATENCY_CONFIG` for virtual-time simulation
└── cloudops_agent/    # Tri-Modal agent driving the simulator
    ├── agent.py       # CloudOpsAgent main loop
    ├── controller/    # `intent_parser.py`, `mode_selector.py`
    ├── planning/      # `poset_planner.py` (GEE), `react_planner.py`
    ├── memory/        # `blackboard.py`, `trace_store.py`
    ├── knowledge/     # `io_registry.py` (action IO contract)
    └── tools/         # `gym_adapter.py` bridging agent and simulator
```

## Three modes

| Mode | Module | LLM calls | Used in paper |
|---|---|---|---|
| Expert  | `planning.poset_planner.PosetPlanner` | only intent parsing | Table 4 row "Expert" |
| Hybrid  | `agent.CloudOpsAgent` w/ `ExecutionMode.HYBRID` | intent parsing + fallback ReAct | Table 4 row "Hybrid" |
| Explore | `planning.react_planner.ReActPlanner` | full ReAct loop | Table 4 row "Explore" |

## Imports

All internal imports use the `execution.<sub>` prefix, e.g.:

```python
from execution.aliyun_gym.factory import create_gym_env
from execution.cloudops_agent.agent import CloudOpsAgent
```

## Usage

This package is not meant to be invoked directly. See
`../simulation_workspace/` for experiment drivers, and
[`../AI_REPRODUCE_GUIDE.md`](../AI_REPRODUCE_GUIDE.md) for the reproduction
recipe.
