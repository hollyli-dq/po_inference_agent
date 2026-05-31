# HPO 后验偏序数据集

本目录包含 120 组 HPO (Hierarchical Partial Order) 后验推断实验结果，用于验证偏序图质量与 Agent 执行效果的关系。

## 场景分类

共 6 个云资源编排场景，按复杂度递增排列：

| 场景 | 任务描述 | 动作数 | Cover-F1 范围 |
|------|----------|--------|---------------|
| `simple_ecs` | 创建单台 ECS 实例 | 5 | 0.727 (固定) |
| `dual_zone_ecs_slb` | 双可用区 ECS + SLB 负载均衡 | 7 | 0.471 - 1.000 |
| `eip_slb_ecs` | EIP + SLB + ECS 公网架构 | 9 | 0.000 - 1.000 |
| `slb_ecs_redis` | SLB + ECS + Redis 缓存架构 | 9 | 0.105 - 0.952 |
| `slb_ecs_rds` | SLB + ECS + RDS 数据库架构 | 12 | 0.000 - 0.923 |
| `dual_zone_ecs_slb_rds` | 双可用区高可用生产环境 | 10 | 0.462 - 0.880 |

### 场景动作列表

**simple_ecs**
```
CreateVpc → CreateVSwitch → CreateSecurityGroup → AuthorizeSecurityGroup → RunInstances
```

**dual_zone_ecs_slb**
```
CreateVpc, CreateVSwitch, CreateSecurityGroup, AuthorizeSecurityGroup,
CreateLoadBalancer, RunInstances, AddBackendServers
```

**dual_zone_ecs_slb_rds**
```
CreateVpc, CreateVSwitch, CreateSecurityGroup, AuthorizeSecurityGroup,
CreateLoadBalancer, CreateDBInstance, RunInstances, CreateAccount,
AddBackendServers, ModifySecurityIps
```

## 数据分布

### 实验配置矩阵

| 维度 | 取值 | 说明 |
|------|------|------|
| 场景 | 6 种 | 见上表 |
| IP-Cov | 0.6, 0.7, 0.8, 0.9, 1.0 | 不可比对覆盖率 |
| eps_jump | 0.005, 0.01, 0.02, 0.05 | MCMC 跳跃参数 |

**总计**: 6 场景 × 5 IP-Cov × 4 eps = **120 组实验**

### MCMC 推断参数

| 参数 | 值 |
|------|-----|
| 迭代次数 | 1,000,000 |
| Burn-in | 50% |
| 似然函数 | log_successors_queue_jump |

### Cover-F1 分布统计

| 场景 | F1 均值 | F1 最小 | F1 最大 |
|------|---------|---------|---------|
| simple_ecs | 0.727 | 0.727 | 0.727 |
| dual_zone_ecs_slb | 0.729 | 0.471 | 1.000 |
| dual_zone_ecs_slb_rds | 0.682 | 0.462 | 0.880 |
| slb_ecs_redis | 0.486 | 0.105 | 0.952 |
| eip_slb_ecs | 0.391 | 0.000 | 1.000 |
| slb_ecs_rds | 0.369 | 0.000 | 0.923 |

## 目录结构

```
HPO_scenarios/
├── experiment_metadata.json      # 实验元数据
├── experiment_summary.csv        # 所有实验的 Cover-F1 汇总
├── exp_1_dual_zone_ecs_slb/      # 实验目录
│   └── summary.json              # 后验结果
├── exp_2_dual_zone_ecs_slb/
│   └── summary.json
└── ...                           # 共 120 个实验目录
```

### summary.json 结构

```json
{
  "experiment_id": "exp_1",
  "experiment_name": "exp_1_dual_zone_ecs_slb",
  "scenario_name": "dual_zone_ecs_slb",
  "configuration": {
    "ip_cov_target": 0.6,
    "eps_jump": 0.005
  },
  "scenario": {
    "task_ids": ["AddBackendServers", "AuthorizeSecurityGroup", ...],
    "n_tasks": 7
  },
  "posterior": {
    "avg_H": [[0.0, 0.0, ...], ...]  // 后验概率矩阵
  }
}
```

**avg_H 矩阵说明**:
- 维度: n_tasks × n_tasks
- `avg_H[i][j]` 表示动作 i 先于动作 j 的后验概率
- 值域: [0, 1]

## 使用方法

### 1. 加载后验偏序图

```python
from execution.cloudops_agent.planning.poset_planner import PosetGraph

# 指定 summary.json 路径和边阈值
summary_path = "HPO_scenarios/exp_1_dual_zone_ecs_slb/summary.json"
edge_threshold = 0.4  # avg_H[i][j] >= 0.4 时添加边

poset = PosetGraph.load_from_hpo_posterior(summary_path, edge_threshold=edge_threshold)
```

### 2. 批量实验

```bash
cd simulation_workspace

# 运行单个场景
python hpo_batch_experiment.py --scenario simple_ecs

# 运行全部 120 个实验
python hpo_batch_experiment.py

# 限制运行数量
python hpo_batch_experiment.py --limit 10
```

### 3. 配置文件

**experiment_config.yaml** - 实验参数:
```yaml
experiment:
  edge_threshold: 0.4   # 边概率阈值

output:
  results_dir: "hpo_batch_results"
```

**.env** - LLM 配置:
```
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-max
```

### 4. 在 Agent 中使用

```python
from execution.cloudops_agent.config import AgentConfig

config = AgentConfig()
config.switches.poset_edge_threshold = 0.4  # 配置边阈值
config.switches.poset_path = "HPO_scenarios/exp_1_dual_zone_ecs_slb"

agent = CloudOpsAgent(config, ...)
result = agent.run("创建高可用架构...")
```

## 实验结果输出

运行批量实验后，结果保存在 `hpo_batch_results/` 目录：

```
hpo_batch_results/
├── hpo_experiment_results.csv    # 结果汇总表
├── hpo_experiment_details.json   # 详细结果
├── hpo_experiment_report.md      # 分析报告
└── traces/                       # 执行 trace
```

## 关键指标说明

| 指标 | 说明 |
|------|------|
| Cover-F1 | 偏序图结构恢复准确度 (与 Ground Truth 对比) |
| IP-Cov | Incomparable Pair Coverage，trace 多样性度量 |
| eps_jump | MCMC 状态转移参数，控制探索-利用平衡 |
| edge_threshold | 边概率阈值，控制偏序图稀疏度 |
