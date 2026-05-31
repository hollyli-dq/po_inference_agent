# Aliyun-Gym 仿真器开发指南

本文档详细说明了 Aliyun-Gym 仿真器的架构设计、实现细节及开发流程，旨在帮助开发者（包括大模型 Agent）快速理解上下文，进行功能扩展和维护。

## 1. 项目概述

Aliyun-Gym 是一个轻量级的阿里云 API 仿真器，用于模拟云资源的创建、查询、修改和删除操作。它主要用于：
- 大模型 Agent 的训练与评估环境。
- 无需真实云账号的 API 调用测试。
- 故障注入与混沌工程模拟。

## 2. 核心架构与原理

仿真器采用模块化设计，主要包含以下核心组件：

### 2.1 组件交互流程

```mermaid
graph LR
    User[用户/Agent] --> ActionRouter[ActionRouter (路由分发)]
    ActionRouter --> ApiValidator[ApiValidator (参数校验)]
    ApiValidator --> KnowledgeBase[(Knowledge Base (API元数据))]
    ActionRouter --> Handlers[Product Handlers (业务逻辑)]
    Handlers --> StateStore[StateStore (内存状态存储)]
    Handlers --> TimeKeeper[TimeKeeper (虚拟时钟)]
```

### 2.2 核心模块说明

- **ActionRouter (`src/aliyun_gym/core/action_router.py`)**:
    - **职责**: 仿真器的入口，负责接收 API 请求（Action, Parameters, Product），进行参数校验，并路由到对应的 Handler 函数。
    - **特性**: 支持基于产品的动态路由（如 `VPC` 路由到 `vpc_handlers.py`）。
    - **时延模拟**: 内置时延模拟机制，可模拟 API 调用的耗时。

- **ApiValidator (`src/aliyun_gym/core/api_validator.py`)**:
    - **职责**: 基于知识库对请求参数进行严格校验。
    - **校验层级**:
        1. **基础校验**: 必填参数、参数类型（String/Int/Bool）、枚举值（Enum）。
        2. **约束校验**: 数值范围（Min/Max）、字符串长度、正则表达式（Pattern）。
        3. **自定义校验**: 通过 `custom_validators.py` 处理复杂的参数组合逻辑（如互斥参数）。

- **StateStore (`src/aliyun_gym/core/state_store.py`)**:
    - **职责**: 内存数据库，存储所有模拟云资源的状态（如实例列表、VPC 拓扑等）。
    - **特性**: 提供简单的 CRUD 接口，支持资源关联查询。

- **ChaosInjector (`src/aliyun_gym/core/chaos_injector.py`)**:
    - **职责**: 混沌注入器，用于模拟 API 调用失败场景。
    - **用法**: 创建时指定 `failure_rate`（如0.1 表示 10% 概率失败）。
    - **场景**: 测试 Agent 的容错和重试能力。

- **VirtualClock (`src/aliyun_gym/core/time_keeper.py`)**:
    - **职责**: 虚拟时钟，用于模拟时间流逝（如实例启动需要 5 分钟）。
    - **特性**: 不使用真实 `time.sleep()`，而是通过 `clock.tick(seconds)` 快进虚拟时间。
    - **作用**: 
        1. 实现状态机转换（Pending → Running）而无需等待真实时间。
        2. **并行执行分析**: Agent 通过记录每个 API 调用前后的虚拟时间，计算模拟耗时，进而分析偏序图的并行提效（加速比）。

- **TraceRecorder (`src/aliyun_gym/core/trace_recorder.py`)**:
    - **职责**: 记录 Agent 的操作轨迹，用于训练数据生成和调试。
    - **功能**: 记录每个 API 调用的输入/输出，导出为 JSON 文件。

- **Handlers (`src/aliyun_gym/handlers/*.py`)**:
    - **职责**: 实现具体的 API 业务逻辑。
    - **规范**: 每个 API 对应一个 `handle_<ActionName>` 函数。

- **Knowledge Base (`src/aliyun_gym/knowledge/`)**:
    - **职责**: 存储 API 的元数据（Schema），是校验逻辑的数据源。
    - **结构**: `api_docs/<product>/full_meta.json`。

## 3. 快速开始

### 3.1 使用 Factory 创建仿真环境

仿真器提供了 `create_gym_env()` 工厂函数，一键创建完整的仿真环境：

```python
from execution.aliyun_gym.factory import create_gym_env

# 创建仿真环境
env = create_gym_env(
    failure_rate=0.0,      # 混沌注入概率（0.1 = 10% 失败率）
    enable_latency=True,   # 是否启用时延模拟
    use_real_latency=False # 是否使用真实 time.sleep()
)

# 使用 Mock Clients 调用 API
vpc_result = env.vpc_client.create_vpc(CreateVpcRequest())
ecs_result = env.ecs_client.run_instances(RunInstancesRequest())

# 访问共享状态
print(env.state_store.dump())  # 查看所有资源
print(env.clock.now())         # 查看虚拟时间

# 记录操作轨迹
env.trace_recorder.start_trace("创建 VPC")
# ... 执行操作 ...
env.trace_recorder.end_trace("Success")
```

**AliyunGymEnv 包含的组件**：

| 组件 | 说明 |
| --- | --- |
| `state_store` | 内存状态存储 |
| `chaos_injector` | 混沌注入器 |
| `clock` | 虚拟时钟 |
| `action_router` | API 路由器 |
| `trace_recorder` | 轨迹记录器 |
| `vpc_client` / `ecs_client` / ... | 各产品的 Mock Client |

### 3.2 添加新 API 的步骤

1. **检查元数据**: 确认 `src/aliyun_gym/knowledge/api_docs/<product>/full_meta.json` 中包含该 API 的定义。
2. **实现 Handler**: 在对应的 `src/aliyun_gym/handlers/<product>_handlers.py` 中编写处理函数。
3. **注册 API**: 在 `src/aliyun_gym/knowledge/implemented_apis.json` 中添加该 API 名称（可选，用于统计）。
4. **添加测试**: 在 `tests/aliyun_gym/` 下编写单元测试。

### 3.3 Handler 编程规范

- **命名**: 函数名必须为 `handle_<ActionName>`（区分大小写，通常 ActionName 为 PascalCase，如 `handle_RunInstances`）。
- **参数**: 固定顺序为 `(query, state, chaos, clock)`：
    - `query`: 请求参数字典
    - `state`: StateStore 实例
    - `chaos`: ChaosInjector 实例
    - `clock`: VirtualClock 实例
- **返回值**: 返回符合阿里云 API 响应格式的字典（通常包含 `RequestId` 和业务字段）。
- **错误处理**: 遇到业务错误时，返回包含 `Code` 和 `Message` 的错误字典。

**代码示例**:
```python
# src/aliyun_gym/handlers/ecs_handlers.py

def handle_RunInstances(query, state, chaos, clock):
    # 1. 混沌注入检查
    if chaos.should_fail():
        return chaos.generate_error("RunInstances")
    
    # 2. 提取参数 (ApiValidator 已确保必填参数存在)
    vswitch_id = query.get("VSwitchId")
    sg_id = query.get("SecurityGroupId")
    
    # 3. 业务逻辑
    instance_id = f"i-{uuid.uuid4().hex[:8]}"
    state.put(instance_id, {
        "type": "ECS",
        "status": "Pending",
        "created_at": clock.now(),
        "boot_time": 300  # 虚拟 5 分钟启动时间
    })
    
    # 4. 记录虚拟时间流逝
    clock.tick(1.0)
    
    # 5. 返回响应
    return {
        "RequestId": str(uuid.uuid4()),
        "InstanceIdSets": {"InstanceIdSet": [instance_id]}
    }
```

### 3.4 参数校验扩展

如果标准 Schema 无法满足校验需求（如参数 A 和 B 不能同时存在），请在 `src/aliyun_gym/core/custom_validators.py` 中添加规则：

```python
# src/aliyun_gym/core/custom_validators.py

CUSTOM_RULES = {
    "RunInstances": [
        lambda q: CustomValidators.check_mutex(q, ["ImageId", "ImageFamily"]),
    ]
}
```

## 4. 知识工程与静态资源同步

本项目采用“混合仿真”策略，即：
1. **动态资源**（如创建的实例）：存储在内存中（StateStore），由用户行为驱动。
2. **静态元数据**（如地域列表、实例规格）：**通过真实 AK/SK 同步自阿里云**，并缓存为 JSON 文件。这确保了 Agent 在决策时（如选择可用区、选择机型）能够基于真实的云上约束，避免产生幻觉。

### 4.1 静态资源缓存机制

静态资源存储在 `src/aliyun_gym/knowledge/static_resources/` 目录下：
- **regions.json**: 阿里云全量地域列表（RegionId, Endpoint）。
- **zones.json**: 各地域下的可用区信息（ZoneId）。
- **instance_types.json**: ECS 实例规格列表（ecs.g6.large, cpu, memory 等）。
- **images.json**: 常见系统镜像列表（Ubuntu, CentOS 等）。

代码通过 `ResourceManager` 单例类（`src/aliyun_gym/core/resource_manager.py`）统一加载这些资源，并提供查询接口。

### 4.2 同步真实数据

为了更新静态资源，我们提供了一个同步脚本 `src/aliyun_gym/knowledge/fetch_real_resources.py`。该脚本需要真实的阿里云 AccessKey 才能运行。

**配置步骤：**

1. 复制配置文件模板：
   ```bash
   cp src/aliyun_gym/knowledge/aliyun_config.ini.template src/aliyun_gym/knowledge/aliyun_config.ini
   ```

2. 编辑 `aliyun_config.ini`，填入你的真实 AK/SK（注意：此文件已被 `.gitignore` 忽略，不会提交到代码库）：
   ```ini
   [aliyun]
   access_key_id = YOUR_REAL_ACCESS_KEY
   access_key_secret = YOUR_REAL_SECRET
   ```

3. 运行同步脚本：
   ```bash
   python src/aliyun_gym/knowledge/fetch_real_resources.py
   ```
   脚本运行完成后，`static_resources` 目录下的 JSON 文件将被更新为最新数据。

### 4.3 Describe 接口的双重实现

在 Handler 开发中，需区分两种 Describe 接口：

| 接口类型 | 数据来源 | 示例 API | 实现方式 |
| --- | --- | --- | --- |
| **静态元数据接口** | `static_resources/*.json` | `DescribeRegions`<br>`DescribeZones`<br>`DescribeInstanceTypes` | 调用 `ResourceManager` 查询缓存数据 |
| **动态资源接口** | `StateStore` (内存) | `DescribeInstances`<br>`DescribeVpcs`<br>`DescribeSecurityGroups` | 查询内存中的模拟资源列表 |

**开发建议**：如果需要增加新的“查询类”接口，请先判断该数据是否属于“静态元数据”。如果是（如查询价格、查询磁盘类型），建议先扩展 `fetch_real_resources.py` 进行抓取，再通过 `ResourceManager` 提供给 Handler，而不是在 Handler 中硬编码。

## 5. API 元数据管理

仿真器的参数校验依赖于阿里云 OpenAPI 的官方 Schema（元数据）。这些数据从阿里云 OpenAPI 门户 (`api.aliyun.com`) 抓取并缓存到本地。

### 5.1 元数据存储结构

```
src/aliyun_gym/knowledge/api_docs/
├── ecs/
│   ├── full_meta.json      # 完整的 API 元数据（含参数 Schema）
│   └── io_summary.json     # 精简的输入输出摘要
├── vpc/
│   ├── full_meta.json
│   └── io_summary.json
├── rds/
│   └── ...
└── index.json              # 服务索引
```

**文件说明**：
- **full_meta.json**: 包含每个 API 的完整定义，供 `ApiValidator` 用于参数校验。
- **io_summary.json**: 提炼的输入/输出信息，便于 Agent 理解 API 语义。

### 5.2 元数据采集工具

使用 `api_meta_fetcher.py` 从阿里云 OpenAPI 门户抓取元数据（**无需 AK/SK**，使用公开接口）。

**运行方式**：

```bash
# 方式一：抓取所有支持的服务（增量模式，跳过已有 API）
python -m execution.aliyun_gym.knowledge.api_meta_fetcher

# 方式二：仅抓取 implemented_apis.json 中声明的 API（推荐）
python -m execution.aliyun_gym.knowledge.api_meta_fetcher \
    --api-list src/aliyun_gym/knowledge/implemented_apis.json

# 方式三：指定服务
python -m execution.aliyun_gym.knowledge.api_meta_fetcher -s ecs rds
```

**`implemented_apis.json` 的作用**：

该文件是仿真器实现的 API 清单，同时用于：
1. 统计仿真器的 API 覆盖范围。
2. 指导元数据采集（只抓取需要的 API，减少无用数据）。
3. 对比真实 SDK，识别增量任务。

### 5.3 元数据采集流程（内部原理）

```mermaid
graph LR
    A[implemented_apis.json] --> B[ApiMetaFetcher]
    B --> C["api.aliyun.com/meta/v1"]
    C --> D["products/{service}/versions/{version}/apis/{api}/api.json"]
    D --> E[full_meta.json]
    E --> F[io_summary.json]
```

**增量抓取**：默认跳过已存在的 API，只抓取新增的。如需强制全量抓取，可删除对应目录后重新运行。

### 5.4 参数校验规则说明

`ApiValidator` 基于 `full_meta.json` 中的 Schema 进行以下校验：

| 校验类型 | Schema 字段 | 校验逻辑 |
| --- | --- | --- |
| 必填参数 | `required: true` | 参数缺失时返回 `MissingParameter` |
| 参数类型 | `type: string/integer/boolean` | 类型不匹配返回 `InvalidParameter` |
| 枚举值 | `enum: ["a", "b"]` | 值不在枚举范围返回错误 |
| 数值范围 | `minimum/maximum` | 超出范围返回错误 |
| 字符串长度 | `minLength/maxLength` | 长度不符返回错误 |
| 正则匹配 | `pattern` | 格式不符返回错误 |
| 自定义规则 | `custom_validators.py` | 互斥参数、依赖关系等复杂逻辑 |

## 6. 测试指南

使用 `pytest` 运行测试：

```bash
# 运行所有测试
pytest tests/aliyun_gym/

# 运行特定测试文件
pytest tests/aliyun_gym/test_validator_enhanced.py

# 运行特定测试用例
pytest tests/aliyun_gym/test_validator_enhanced.py::TestApiValidatorEnhanced::test_validate_numeric_min_max
```

## 7. 当前支持的 API

完整的已实现 API 列表请参考文件：
[implemented_apis.json](../../execution/aliyun_gym/knowledge/implemented_apis.json)

主要覆盖产品：
- **ECS**: 实例生命周期管理、安全组、镜像等。
- **VPC**: 专有网络、交换机、EIP。
- **RDS**: 数据库实例管理、白名单等。
- **SLB**: 负载均衡实例、监听器、后端服务器。
- **Redis**: 缓存实例管理。
- **CMS**: 云监控指标查询。

## 8. 近期变更记录 (Recent Changes)

### 8.1 仿真器行为严格化
- **SLB Handler**: `handle_AddBackendServers` 现严格遵循 API 定义，要求 `BackendServers` 参数必须为 JSON String，且进行严格的类型检查，不再接受 Python List 对象。
- **参数校验**: 增强了对 API 输入参数的校验逻辑，确保与线上真实 API 行为一致。

### 8.2 全面覆盖查询接口
- **接口补全**: 在 `gym_adapter.py` 和 `io_registry.py` 中补全了大量查询类接口（`DescribeInstanceTypes`, `DescribeImages`, `DescribeAvailableZones` 等）。
- **目的**: 支持 Agent 的“探索优先”策略，使其能够通过查询获取真实的资源规格和库存信息，而不是依赖硬编码的默认值。

### 8.3 默认值移除
- **IntentParser**: 移除了 `vpc_cidr`, `ecs_type` 等参数的默认值。
- **动态映射**: 新增了规格解析逻辑，仅当用户明确指定（如 "2C4G"）时才进行映射，否则强制 Agent 通过 `Describe` 接口去发现可用资源。

### 8.4 并行轨迹生成工具
- **Parallel Trace Generator**: 新增 `simulation_workspace/explore_trace_gen_parallel.py`，支持多进程并行生成仿真轨迹。
- **Console Manager**: 引入 `ConsoleManager` 管理多进程环境下的控制台输出，解决进度条冲突问题。
- **场景覆盖**: 支持全部 6 个标准仿真场景的批量生成。

### 8.5 多格式轨迹导出
- **TraceStore 增强**: `TraceRecorder` 现支持导出为多种格式：
    - **CSV**: 轻量级格式，仅包含 Time, Product, API, Success/Error。
    - **TXT**: 紧凑文本格式，便于 LLM 阅读和理解。
    - **JSON**: 完整的结构化数据（保留原有格式）。

### 8.6 并行执行分析支持
- **GymToolAdapter 扩展**: 新增 `get_virtual_time()` 和 `reset_virtual_clock()` 方法，支持 Agent 获取虚拟时钟用于并行分析。
- **TraceStore 并行字段**: `ActionRecord` 新增 `layer`, `batch_id`, `virtual_start_time`, `virtual_end_time`, `simulated_duration_ms` 字段；`TraceRecord` 新增 `parallelism_stats` 统计。
- **Expert 模式增强**: `_execute_expert_mode` 现自动计算并行统计（加速比、每层耗时等），用于量化偏序图的并行提效。
- **对比报告扩展**: `compare_modes_analyzer.py` 新增"并行执行提效分析"章节，展示顺序/并行时间对比及加速比。

---
*Generated for Aliyun-Gym Contributors & AI Agents*
