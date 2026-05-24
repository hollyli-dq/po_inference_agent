"""
混合模式 Trace 生成脚本
公平对比版本：LLM 意图识别 + LLM 参数推理 + 偏序图执行

设计原则（来自 Memory）：
- 专家模式不应完全绕过 LLM
- 保留意图识别能力以正确选择偏序图
- 保留参数推理能力，但利用偏序图预知参数需求，一次性完成推理
- 失败时降级到探索模式

核心流程：
1. LLM 意图识别 → 选择对应偏序图
2. 从偏序图 + IORegistry 提取需要推理的参数
3. LLM 一次性推理所有参数
4. 使用 Hybrid 模式执行（Expert + Fallback to Explore）
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openai import OpenAI
    from execution.cloudops_agent.config import AgentConfig, ExecutionMode
    from execution.cloudops_agent.agent import CloudOpsAgent, AgentStatus
    from execution.cloudops_agent.planning.poset_planner import PosetGraph, PosetPlanner
    from execution.cloudops_agent.controller.intent_parser import IntentType, IntentParser
    from execution.cloudops_agent.knowledge.io_registry import IORegistry, get_io_registry
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("请确保已安装依赖并在项目根目录下运行。")
    sys.exit(1)


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    """
    从 trace_tasks.json 加载任务列表
    
    Args:
        tasks_path: 任务文件路径
        
    Returns:
        任务列表，每个任务包含 query 和 repeat 字段
    """
    if not os.path.exists(tasks_path):
        print(f"Error: 任务文件未找到: {tasks_path}")
        sys.exit(1)
    
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    return tasks


def load_env(env_path: str) -> Dict[str, str]:
    """简单的 .env 文件解析器"""
    env_vars = {}
    if not os.path.exists(env_path):
        print(f"Warning: .env file not found at {env_path}")
        return env_vars
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


def extract_required_params_from_poset(
    poset: PosetGraph, 
    io_registry: IORegistry
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    从偏序图中提取需要 LLM 推理的参数
    
    逻辑：
    1. 收集所有 Action 需要的输入参数
    2. 收集所有 Action 会产出的输出参数
    3. 需要推理的 = 输入参数 - 输出参数（排除动态生成的）
    
    Args:
        poset: 偏序图
        io_registry: IO 注册表
        
    Returns:
        (需要推理的参数集合, {action: [需要的输入参数]})
    """
    all_inputs: Set[str] = set()
    all_outputs: Set[str] = set()
    action_inputs: Dict[str, List[str]] = {}
    
    # 遍历所有 Action
    for action in poset.nodes.keys():
        spec = io_registry.get_spec(action)
        if spec:
            # 必须输入参数
            required = list(spec.inputs)
            # 可选输入参数中的常用参数也收集
            optional_common = [p for p in spec.optional_inputs 
                             if p in {"Amount", "InstanceName", "VpcName", "VSwitchName",
                                     "SecurityGroupName", "LoadBalancerName", "DBInstanceDescription"}]
            action_inputs[action] = required + optional_common
            all_inputs.update(required)
            all_inputs.update(optional_common)
            
            # 输出参数（这些会由执行过程动态产出）
            all_outputs.update(spec.outputs)
    
    # 需要推理的参数 = 输入参数 - 输出参数
    params_to_infer = all_inputs - all_outputs
    
    return params_to_infer, action_inputs


def build_param_inference_prompt(
    query: str, 
    intent_type: IntentType,
    params_to_infer: Set[str],
    action_inputs: Dict[str, List[str]]
) -> str:
    """
    构建参数推理 Prompt
    
    让 LLM 一次性推理出偏序图执行所需的所有参数
    """
    # 按类别组织参数
    param_categories = {
        "基础设施参数": ["RegionId", "ZoneId", "ZoneIdSecondary", "CidrBlock"],
        "ECS 参数": ["InstanceType", "ImageId", "Amount", "SystemDiskCategory", "SystemDiskSize"],
        "安全组参数": ["IpProtocol", "PortRange", "SourceCidrIp"],
        "SLB 参数": ["AddressType", "LoadBalancerSpec", "ListenerPort", "BackendServerPort", 
                    "ListenerProtocol", "HealthCheck", "Bandwidth"],
        "RDS 参数": ["Engine", "EngineVersion", "DBInstanceClass", "DBInstanceStorage",
                    "DBInstanceNetType", "SecurityIPList", "PayType", "DBInstanceStorageType",
                    "AccountName", "AccountPassword"],
        "Redis 参数": ["InstanceClass", "ChargeType", "NetworkType"],
        "EIP 参数": ["InternetChargeType"],
    }
    
    # 过滤出需要推理的参数（按类别）
    categorized_params = {}
    for category, params in param_categories.items():
        relevant = [p for p in params if p in params_to_infer]
        if relevant:
            categorized_params[category] = relevant
    
    # 构建 Prompt
    prompt = f"""你是阿里云资源配置专家。请根据用户需求，推理出执行云资源创建所需的所有参数。

## 用户需求
{query}

## 识别的场景类型
{intent_type.value}

## 需要推理的参数

"""
    
    for category, params in categorized_params.items():
        prompt += f"### {category}\n"
        for param in params:
            prompt += f"- {param}\n"
        prompt += "\n"
    
    prompt += """## 参数推理规则

1. **RegionId**: 根据用户提到的地区推断，默认 "cn-hangzhou"
2. **ZoneId**: 根据 RegionId 选择合适的可用区，杭州默认 "cn-hangzhou-h"
3. **CidrBlock**: VPC 网段，默认 "172.16.0.0/12"
4. **InstanceType**: 根据用户提到的规格(如2C4G)映射：
   - 1C2G -> ecs.t6.small
   - 2C4G -> ecs.c6.large  
   - 4C8G -> ecs.c6.xlarge
5. **ImageId**: 默认使用 CentOS 7.9 镜像 "centos_7_9_x64_20G_alibase_20230816.vhd"
6. **Amount**: ECS 数量，从用户需求提取，默认 1
7. **安全组**: IpProtocol="tcp", PortRange="80/80", SourceCidrIp="0.0.0.0/0"
8. **SLB**: AddressType="intranet", ListenerPort=80, BackendServerPort=80
9. **RDS**: Engine="MySQL", EngineVersion="8.0", PayType="Postpaid"
10. **双可用区**: 如果需要高可用，提供 ZoneIdSecondary

## 输出格式

请返回 JSON 格式，只包含需要推理的参数：
```json
{
    "RegionId": "cn-hangzhou",
    "ZoneId": "cn-hangzhou-h",
    ...
}
```

注意：只返回 JSON，不要其他内容。
"""
    return prompt


def infer_params_with_llm(
    client: OpenAI,
    model: str,
    query: str, 
    intent_type: IntentType,
    params_to_infer: Set[str],
    action_inputs: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], int]:
    """
    使用 LLM 一次性推理所有参数
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        query: 用户查询
        intent_type: 意图类型
        params_to_infer: 需要推理的参数集合
        action_inputs: 每个 Action 需要的输入参数
        
    Returns:
        (推理出的参数字典, token消耗)
    """
    prompt = build_param_inference_prompt(query, intent_type, params_to_infer, action_inputs)
    tokens_used = 0
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个阿里云资源配置专家，精通各种云产品的参数配置。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # 统计 token 消耗
        if response.usage:
            tokens_used = response.usage.total_tokens
        
        content = response.choices[0].message.content
        # 解析 JSON
        params = json.loads(content)
        return params, tokens_used
        
    except Exception as e:
        print(f"LLM 参数推理失败: {e}")
        # 返回默认参数
        return get_default_params(intent_type), 0


def get_default_params(intent_type: IntentType) -> Dict[str, Any]:
    """获取默认参数（LLM 推理失败时的备选）"""
    base_params = {
        "RegionId": "cn-hangzhou",
        "ZoneId": "cn-hangzhou-h",
        "CidrBlock": "172.16.0.0/12",
        "InstanceType": "ecs.c6.large",
        "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
        "Amount": 1,
        "IpProtocol": "tcp",
        "PortRange": "80/80",
        "SourceCidrIp": "0.0.0.0/0",
    }
    
    # 根据意图类型添加额外参数
    if intent_type in (IntentType.DUAL_ZONE_ECS_SLB, IntentType.DUAL_ZONE_ECS_SLB_RDS):
        base_params["ZoneIdSecondary"] = "cn-hangzhou-i"
        base_params["Amount"] = 2
    
    if intent_type in (IntentType.SLB_ECS_RDS, IntentType.SLB_ECS_REDIS, 
                       IntentType.EIP_SLB_ECS, IntentType.DUAL_ZONE_ECS_SLB,
                       IntentType.DUAL_ZONE_ECS_SLB_RDS):
        base_params.update({
            "AddressType": "intranet",
            "LoadBalancerSpec": "slb.s1.small",
            "ListenerPort": 80,
            "BackendServerPort": 80,
            "ListenerProtocol": "http",
            "HealthCheck": "on",
        })
    
    if intent_type in (IntentType.SLB_ECS_RDS, IntentType.DUAL_ZONE_ECS_SLB_RDS):
        base_params.update({
            "Engine": "MySQL",
            "EngineVersion": "8.0",
            "DBInstanceClass": "mysql.n2.medium.2c",
            "DBInstanceStorage": 20,
            "DBInstanceNetType": "Intranet",
            "SecurityIPList": "172.16.0.0/12",
            "PayType": "Postpaid",
            "DBInstanceStorageType": "cloud_essd",
        })
    
    if intent_type == IntentType.SLB_ECS_REDIS:
        base_params.update({
            "InstanceClass": "redis.master.mid.default",
            "ChargeType": "PostPaid",
            "NetworkType": "VPC",
        })
    
    if intent_type == IntentType.EIP_SLB_ECS:
        base_params.update({
            "Bandwidth": "5",
            "InternetChargeType": "PayByTraffic",
        })
    
    return base_params


def create_hybrid_agent(traces_dir: Path, llm_client=None, model: str = "qwen3-max") -> CloudOpsAgent:
    """
    创建 Hybrid 模式 Agent
    
    特点：
    - 使用 HYBRID 执行模式（Expert + Fallback）
    - Agent 内部根据意图自动选择偏序图
    - 启用降级机制
    """
    config = AgentConfig.preset_hybrid_benchmark()
    config.switches.verbose = True
    config.switches.trace_enabled = True
    config.switches.trace_output_path = str(traces_dir)
    
    # 先禁用偏序图自动加载（避免加载默认路径报错）
    config.switches.poset_enabled = False
    
    # 创建 Agent
    agent = CloudOpsAgent(config=config, llm_client=llm_client)
    
    # 设置偏序图目录（让 PosetPlanner 能找到偏序图文件）
    poset_dir = Path(current_dir) / "manual_scenarios"
    agent.poset_planner.set_poset_dir(str(poset_dir))
    
    # 现在启用偏序图（Agent.run() 时会根据意图自动加载）
    agent.config.switches.poset_enabled = True
    
    # 设置 LLM 模型
    if llm_client:
        agent.intent_parser.llm_client = llm_client
        agent.react_planner.llm_client = llm_client
        agent.react_planner.model_name = model
    
    # 设置 trace 输出路径
    agent.trace_store.output_path = traces_dir
    
    return agent


def run_hybrid_scenario(
    client: OpenAI,
    model: str,
    query: str,
    description: str,
    traces_dir: Path,
    scenario_idx: int,
    total_scenarios: int,
    io_registry: IORegistry
) -> Dict:
    """
    执行单个混合模式场景
    
    流程：
    1. LLM 意图识别（用于提取参数需求）
    2. 从偏序图提取参数需求
    3. LLM 一次性参数推理
    4. Agent 执行（内部自动识别意图并选择偏序图）
    
    Args:
        scenario_idx: 任务索引（用于 trace 命名）
    """
    print(f"\n{'='*60}")
    print(f"[{scenario_idx}/{total_scenarios}] {description}")
    print(f"{'='*60}")
    print(f"任务: {query}")
    
    # Step 1: 意图识别（用于提取参数需求）
    print(f"\n--- Step 1: 意图识别 ---")
    intent_parser = IntentParser(llm_client=client)
    intent = intent_parser.parse(query, use_llm=True)
    
    print(f"识别意图: {intent.intent_type.value}")
    print(f"置信度: {intent.confidence}")
    
    # Step 2: 从偏序图提取参数需求
    print(f"\n--- Step 2: 提取参数需求 ---")
    
    # 创建临时 PosetPlanner 加载偏序图以提取参数需求
    temp_planner = PosetPlanner(io_registry=io_registry)
    temp_planner.set_poset_dir(str(Path(current_dir) / "manual_scenarios"))
    
    has_poset = temp_planner.load_poset_for_intent(intent.intent_type.value)
    
    if has_poset and temp_planner.poset:
        params_to_infer, action_inputs = extract_required_params_from_poset(temp_planner.poset, io_registry)
        print(f"偏序图包含 {len(temp_planner.poset.nodes)} 个 Action")
        print(f"需要推理 {len(params_to_infer)} 个参数: {sorted(params_to_infer)[:10]}...")
    else:
        print(f"⚠️ 意图 {intent.intent_type.value} 没有对应的偏序图，使用默认参数")
        params_to_infer = set()
        action_inputs = {}
    
    # Step 3: LLM 一次性参数推理
    print(f"\n--- Step 3: LLM 参数推理 ---")
    inference_tokens = 0
    if params_to_infer:
        inferred_params, inference_tokens = infer_params_with_llm(
            client, model, query, intent.intent_type, params_to_infer, action_inputs
        )
    else:
        inferred_params = get_default_params(intent.intent_type)
    
    print(f"推理出 {len(inferred_params)} 个参数 (tokens: {inference_tokens})")
    print(f"  RegionId: {inferred_params.get('RegionId')}")
    print(f"  ZoneId: {inferred_params.get('ZoneId')}")
    print(f"  InstanceType: {inferred_params.get('InstanceType')}")
    
    # Step 4: 创建 Agent 并执行
    print(f"\n--- Step 4: Hybrid 模式执行 ---")
    
    # 创建 Agent（内部会根据意图自动选择偏序图）
    agent = create_hybrid_agent(traces_dir, llm_client=client, model=model)
    
    # 设置任务索引，用于 trace 命名（便于横向比较）
    agent.trace_store.set_task_index(scenario_idx)
    
    # 设置预设参数（会在 blackboard 初始化后自动注入）
    agent.set_preset_params(inferred_params)
    
    # 执行
    start_time = time.time()
    try:
        result = agent.run(query)
        duration = time.time() - start_time
        
        status_symbol = "✅" if result.status == AgentStatus.SUCCESS else "❌"
        print(f"\n结果: {status_symbol} {result.status.value}")
        print(f"执行模式: {result.mode_used.value}")
        print(f"降级次数: {result.fallback_count}")
        print(f"执行动作: {result.actions_executed}")
        print(f"LLM Tokens: {result.total_tokens}")
        print(f"Trace ID: {result.trace_id}")
        print(f"耗时: {duration:.2f}s")
        
        if result.error:
            print(f"错误: {result.error}")
        
        # 累加外部 LLM 调用的 tokens（意图识别 + 参数推理）
        total_llm_tokens = result.total_tokens + inference_tokens
        
        return {
            "scenario": description,
            "intent_type": intent.intent_type.value,
            "query": query,
            "status": result.status.value,
            "mode_used": result.mode_used.value,
            "fallback_count": result.fallback_count,
            "actions_executed": result.actions_executed,
            "actions_count": len(result.actions_executed),
            "resources_created": result.resources_created,
            "llm_tokens": total_llm_tokens,
            "trace_id": result.trace_id,
            "duration_s": duration,
            "inferred_params_count": len(inferred_params),
            "inference_tokens": inference_tokens,
            "error": result.error
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n执行异常: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "scenario": description,
            "intent_type": intent.intent_type.value,
            "status": "exception",
            "error": str(e),
            "duration_s": duration
        }


def main():
    print("=" * 60)
    print("  混合模式 Trace 生成 - 简化版本")
    print("=" * 60)
    print("特性:")
    print("  - Agent 内部自动意图识别并选择偏序图")
    print("  - LLM 参数推理（利用偏序图预知需求，一次性完成）")
    print("  - 偏序图执行（省略中间推理步骤）")
    print("  - 失败时降级到 Explore 模式")
    
    # 1. 加载环境变量
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    api_key = os.environ.get("LLM_API_KEY") or env_vars.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or env_vars.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL") or env_vars.get("LLM_MODEL") or "qwen3-max"
    
    if not api_key or "your_api_key_here" in api_key:
        print("\nError: LLM_API_KEY 未配置。请在 .env 文件中设置或通过环境变量传入。")
        sys.exit(1)
    
    print(f"\nUsing LLM: {model} at {base_url}")
    
    # 2. 初始化 LLM Client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 3. 读取任务列表
    tasks_path = os.path.join(current_dir, 'trace_tasks.json')
    tasks = load_tasks(tasks_path)
    print(f"加载了 {len(tasks)} 个任务")
    
    # 4. 创建输出目录
    traces_dir = Path(current_dir) / "hybrid_traces"
    traces_dir.mkdir(exist_ok=True)
    print(f"输出目录: {traces_dir}")
    
    # 5. 初始化 IO Registry
    io_registry = get_io_registry()
    
    # 6. 执行所有场景
    print(f"\n--- 开始执行场景 ---")
    total_start = time.time()
    results = []
    
    total_runs = sum(t.get('repeat', 1) for t in tasks)
    current_run = 0
    
    for task in tasks:
        query = task.get('query')
        repeat = task.get('repeat', 1)
        
        for i in range(repeat):
            current_run += 1
            result = run_hybrid_scenario(
                client=client,
                model=model,
                query=query,
                description=f"任务 {current_run}",
                traces_dir=traces_dir,
                scenario_idx=current_run,
                total_scenarios=total_runs,
                io_registry=io_registry
            )
            results.append(result)
            
            # 简单间隔
            time.sleep(0.5)
    
    # 7. 总结报告
    total_duration = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("  执行总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    fallback_count = sum(1 for r in results if r.get("status") == "fallback")
    exception_count = sum(1 for r in results if r.get("status") == "exception")
    
    total_tokens = sum(r.get("llm_tokens", 0) for r in results)
    total_actions = sum(r.get("actions_count", 0) for r in results)
    total_fallbacks = sum(r.get("fallback_count", 0) for r in results)
    
    print(f"总场景数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"降级完成: {fallback_count}")
    print(f"异常: {exception_count}")
    print(f"总耗时: {total_duration:.2f}s")
    print(f"总 LLM Tokens: {total_tokens}")
    print(f"总执行动作数: {total_actions}")
    print(f"总降级次数: {total_fallbacks}")
    
    print("\n场景详情:")
    for r in results:
        status = r.get("status", "unknown")
        if status == "success":
            status_symbol = "✅"
        elif status == "fallback":
            status_symbol = "🔄"
        elif status in ("fallback_to_explore", "skipped"):
            status_symbol = "⏭️"
        else:
            status_symbol = "❌"
        
        actions_count = r.get("actions_count", 0)
        tokens = r.get("llm_tokens", 0)
        fallbacks = r.get("fallback_count", 0)
        print(f"  {status_symbol} {r['scenario']}: {status} ({actions_count} actions, {tokens} tokens, {fallbacks} fallbacks)")
        if r.get("error") and status not in ("success", "fallback_to_explore"):
            error_msg = r['error'][:60] + "..." if len(r.get('error', '')) > 60 else r.get('error', '')
            print(f"      Error: {error_msg}")
    
    # 8. 保存结果摘要
    summary_file = traces_dir / "execution_summary.json"
    summary = {
        "run_time": datetime.now().isoformat(),
        "mode": "hybrid",
        "features": ["llm_intent_recognition", "llm_param_inference", "poset_execution", "fallback_to_explore"],
        "model": model,
        "total_scenarios": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "fallback_count": fallback_count,
        "exception_count": exception_count,
        "total_llm_tokens": total_tokens,
        "total_actions": total_actions,
        "total_fallbacks": total_fallbacks,
        "total_duration_s": total_duration,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果摘要已保存: {summary_file}")
    print(f"Traces 保存目录: {traces_dir}")


if __name__ == "__main__":
    main()
