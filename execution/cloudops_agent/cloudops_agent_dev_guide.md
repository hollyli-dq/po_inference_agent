# CloudOps ReAct Agent

CloudOps Agent 是一个融合了 **System 1 (直觉/专家模式)** 和 **System 2 (推理/探索模式)** 的混合架构智能体，旨在实现云资源编排的高效性与灵活性平衡。

本文档旨在帮助开发者（及大模型）理解 Agent 的架构、代码结构及扩展方式，以便进行后续的功能开发和实验。

## 最新变更 (v2.0)

### 架构重构
- **偏序图外部注入**: 移除 Agent 内部加载逻辑，改由外部脚本注入 HPO 后验偏序图
- **并行执行统计**: Expert 模式支持分层并行执行和提效分析
- **IO Guard 可配置**: 支持关闭预防性参数检查，依赖 API 报错触发降级
- **LLM 错误分级**: 区分致命配置错误和可恢复网络错误

## 1. 核心架构 (Architecture)

本系统基于 `System 1 + System 2` 混合认知架构设计：

*   **System 1 (Expert Mode)**: 基于偏序图 (Poset Graph) 驱动。对于已知、固定的任务模式，直接按照预定义的图结构并发执行，零 LLM 调用，速度极快且成本低。
*   **System 2 (Explore Mode)**: 基于 ReAct (Reasoning + Acting) 推理。对于未知或复杂任务，通过 LLM 逐步推理并调用工具，具有高度灵活性但成本较高。
*   **Hybrid Mode (混合模式)**: 结合两者优势。优先尝试 System 1，在遇到异常或未知情况时，动态降级 (Fallback) 到 System 2 进行错误恢复或继续执行。

### 模块分层

```
src/cloudops_agent/
├── controller/       # 控制层：负责意图解析、模式选择
├── planning/         # 规划层：包含偏序图规划器 (System 1) 和 ReAct 规划器 (System 2)
├── execution/        # 执行层：(逻辑在 agent.py 中，工具适配在 tools/)
├── memory/           # 记忆层：Blackboard (参数黑板), TraceStore (轨迹记录)
├── knowledge/        # 知识层：IO Registry (IO规则), RAG (设计中)
└── tools/            # 工具层：适配 Aliyun-Gym 或 真实 SDK
```

## 2. 快速开始 (Quick Start)

### 环境准备

确保已安装项目依赖（参考根目录 `pyproject.toml`）。

### 编程式调用

目前 Agent 主要通过 Python API 进行调用：

```python
from execution.cloudops_agent.agent import create_agent, AgentConfig
from execution.cloudops_agent.config import ExecutionMode

# 1. 创建 Agent (使用预设配置)
# presets: "production", "trace_collection", "poset_validation", "hybrid_benchmark"
agent = create_agent(preset="production", llm_client=your_llm_client)

# 或者自定义配置
config = AgentConfig(execution_mode=ExecutionMode.HYBRID)
config.switches.verbose = True
agent = create_agent(config=config)

# 2. 执行任务
task = "在杭州区创建一个VPC，并启动2台ECS实例"
result = agent.run(task)

# 3. 查看结果
print(f"Status: {result.status}")
print(f"Resources: {result.resources_created}")
print(f"Tokens Used: {result.total_tokens}")
```

## 3. 代码结构详解 (Code Structure)

为了便于大模型理解和修改代码，以下是关键模块的详细说明：

### 3.1 `agent.py` (Main Entry)
- **`CloudOpsAgent`**: 核心类，维护状态机和生命周期。
- **`run()`**: 执行入口，串联 `IntentParser` -> `ModeSelector` -> `Planner` -> `Execution` 流程。
- **`_execute_expert_mode()`**: 偏序图执行引擎，支持**分层并行执行**：
    - 同层动作模拟并行，使用虚拟时钟记录每层最大耗时
    - 统计 `parallel_time_ms`（并行后耗时）和 `sequential_time_ms`（顺序累加耗时）
    - 受 `config.switches.poset_io_guard_enabled` 控制是否启用 IO 守卫
- **`_execute_explore_mode()`**: ReAct 循环执行引擎。
- **`_execute_fallback()`**: 降级逻辑，当 Expert 模式失败时，携带上下文切换到 ReAct。

**偏序图加载（重要变更）**:

Agent 不再内部加载偏序图，必须由外部注入：

```python
# 推荐方式
config.switches.poset_enabled = False  # 先禁用
agent = CloudOpsAgent(config=config, llm_client=client)
poset = PosetGraph.load_from_hpo_posterior(path, edge_threshold=0.8)
agent.poset_planner.set_poset(poset)
agent.mode_selector.set_poset_graph(poset.to_dict())
agent.config.switches.poset_enabled = True  # 启用
```

参考实现：`simulation_workspace/hpo_batch_experiment.py::create_agent_with_hpo_poset()`

### 3.2 `config.py` (Configuration)
- **`AgentConfig`**: 总配置类。
- **`SwitchConfig`**: 细粒度开关：
    - `poset_enabled`: 是否启用偏序图执行
    - `poset_io_guard_enabled`: 是否启用 IO 守卫（预防性参数检查）
    - `poset_edge_threshold`: HPO 后验边置信度阈值（默认 0.8）
    - `trace_enabled`, `dry_run` 等
- **`LLMConfig`**: LLM 连接配置，支持 OpenAI 兼容协议。
- **Presets**: `preset_trace_collection` 等静态方法提供了不同实验阶段的默认配置。

### 3.3 `controller/`
- **`intent_parser.py`**: 使用 LLM 将自然语言解析为结构化意图 (`ParsedIntent`)。
    - **Token 统计**: `ParsedIntent.llm_tokens` 字段记录意图识别消耗的 LLM Token 数量。
    - **双模式支持**: `parse(text, use_llm=True/False)` 支持 LLM 解析或规则匹配。
- **`mode_selector.py`**: 根据意图和偏序图的覆盖率，决定使用 Expert、Hybrid 还是 Explore 模式。

### 3.4 `planning/`
- **`poset_planner.py`**: 管理偏序图 (DAG)。
    - `get_next_actions(blackboard, io_guard_enabled=True)`: 获取下一批可执行动作（Frontier）
    - `_find_hpo_poset_file()`: 查找 `HPO_scenarios/` 目录下的后验偏序图
    - `_find_manual_poset_file()`: 查找手工定义的偏序图
    - 支持 `use_confidence_filter` 边置信度过滤
- **`react_planner.py`**: 标准 ReAct 规划器，生成 Thought 和 Action。
    - **Discovery First 原则**: 强调在创建资源前必须调用 `Describe*` 接口进行资源发现。
    - **Prompt 优化**: 增强了 System Prompt，明确了“查-改-查”的运维闭环逻辑。
    - **LLM 错误分级处理**:
        - 致命错误（404/401/403/配置错误）向上抛出 `RuntimeError`
        - 可恢复错误（网络超时）记录并返回 None
    - **Verification & Retry**: 支持资源完整性校验和自动重试。

### 3.5 `knowledge/io_registry.py` (Critical)
- **`IORegistry`**: 定义了 API 之间的**数据流依赖**。
- **作用**: 即使偏序图决定了执行顺序，`IORegistry` 确保了动作执行前所需的参数（如 `VpcId`）已从 Blackboard 中获取。
- **开发注意**: 新增 Tool 时，必须在此注册其 Inputs/Outputs 规则。
- **最近更新**: 全面补充了 ECS/RDS/SLB/Redis/CMS/OOS 等产品的查询类接口注册，支持完整的资源发现能力。

### 3.6 `memory/`
- **`blackboard.py`**: 全局共享内存，采用三层架构设计：
    1.  **Global Scope**: 全局通用参数（如 `RegionId`）。
    2.  **Namespace Scope**: 按产品隔离的命名空间（如 `ECS.InstanceId`, `RDS.InstanceId`），解决不同产品同名参数冲突。
    3.  **Resource Registry**: 资源实例注册表，跟踪已创建资源的详细状态。
- **`trace_store.py`**: 用于记录完整的执行轨迹，支持 JSON/CSV/TXT 多格式导出，便于后续分析或离线强化学习训练。

## 4. 扩展指南 (Extension Guide)

### 如何添加新工具 (Add New Tool)
1. 在 `src/aliyun_gym/` 中实现仿真逻辑或 SDK 调用。
2. 在 `src/cloudops_agent/tools/gym_adapter.py` 中适配该工具。
3. **重要**: 在 `src/cloudops_agent/knowledge/io_registry.py` 中注册该工具的输入输出参数。

### 如何添加新意图 (Add New Intent)

当前系统支持6个固定的仿真场景意图：

| IntentType | 场景说明 | 核心资源 |
|------------|---------|----------|
| `SIMPLE_ECS` | 简单 ECS 创建 | VPC + VSwitch + SG + ECS |
| `SLB_ECS_RDS` | SLB + ECS + RDS | + SLB + RDS |
| `SLB_ECS_REDIS` | SLB + ECS + Redis | + SLB + Redis |
| `EIP_SLB_ECS` | EIP + SLB + ECS | + EIP + SLB |
| `DUAL_ZONE_ECS_SLB` | 双可用区 ECS×2 + SLB | 跨可用区高可用 |
| `DUAL_ZONE_ECS_SLB_RDS` | 双可用区 ECS×2 + SLB + RDS 主备 | 全栈高可用 |

如需扩展新场景：
1. 在 `src/cloudops_agent/controller/intent_parser.py` 的 `IntentType` 枚举中添加类型。
2. 在 `INTENT_PATTERNS` 中添加匹配规则。
3. 在 `_infer_flags_from_intent()` 中添加场景标志位映射。
4. 在 `src/cloudops_agent/controller/mode_selector.py` 的 `INTENT_ACTION_TEMPLATES` 中定义 API 序列。
5. (可选) 如果希望支持 Expert 模式，需要在 `poset/` 目录下添加对应的偏序图定义文件。

### 如何修改偏序图 (Modify Poset)
偏序图通常由 `trace_collection` 阶段收集的数据挖掘而来，存储为 JSON 格式。
- 路径配置: `config.switches.poset_path`
- 结构: 包含节点 (Actions) 和边 (Dependencies)。

## 5. 实验与评测 (Experiments)

设计文档中定义了三个实验阶段，可以通过 `create_agent(preset=...)` 快速切换：

1.  **Trace Collection (`trace_collection`)**: 纯 ReAct 模式，用于收集数据。
2.  **Poset Validation (`poset_validation`)**: 纯偏序图模式（严格模式），用于验证图的正确性。
3.  **Hybrid Benchmark (`hybrid_benchmark`)**: 混合模式，用于评估相对于纯 ReAct 的提效（Token 节省、速度提升）。

### 三模式效果对比

使用 `simulation_workspace/trace_analyzer.py` 可生成三模式的横向对比报告：

```bash
python simulation_workspace/trace_analyzer.py
```

典型对比结果：

| 指标 | Expert | Hybrid | Explore |
|------|--------|--------|---------||
| 平均耗时 | ~9s | ~16s | ~105s |
| 平均 LLM Tokens | ~580 | ~710 | ~63,000 |
| Token 节省率 | 99% | 98.9% | 基线 |

## 6. 测试 (Testing)

单元测试位于 `tests/cloudops_agent/`：

```bash
# 运行 agent 相关测试
pytest tests/cloudops_agent/
```

---

## 7. HPO 后验偏序图集成

本 Agent 支持加载由 BHPOP MCMC 推断产生的后验偏序图：

```python
from execution.cloudops_agent.planning.poset_planner import PosetGraph

# 加载 HPO 后验（带边置信度阈值）
poset = PosetGraph.load_from_hpo_posterior(
    "simulation_workspace/HPO_scenarios/T01_simple_ecs/h_posteriors.json",
    edge_threshold=0.8  # 过滤低置信度边
)

# 注入到 Agent
agent.poset_planner.set_poset(poset)
agent.mode_selector.set_poset_graph(poset.to_dict())
```

### IO Guard 配置

当使用 HPO 后验时，可关闭 IO Guard 以提高鲁棒性：

```python
# 关闭 IO 守卫，依赖 API 报错触发降级
config.switches.poset_io_guard_enabled = False
```

---
*Generated for BHPOP CloudOps Agent Development.*
