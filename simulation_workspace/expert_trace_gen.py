"""专家模式 Trace 生成脚本
基于人工编辑的偏序图进行任务测试。

功能：
1. Agent 内部根据意图识别自动选择 manual_scenarios/ 下的偏序图
2. 使用 Expert 模式，偏序图选择逻辑在 Agent 内部完成
3. 填充完整的标准化运维参数，确保专家模式直接执行成功
4. 生成 Trace 文件到 ./expert_traces/ 目录
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
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
    from execution.cloudops_agent.controller.intent_parser import IntentParser
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("请确保已安装依赖并在项目根目录下运行。")
    sys.exit(1)


# ============================================================
# 标准化运维参数 - 确保专家模式直接执行成功
# ============================================================
STANDARD_PARAMS = {
    # 全局参数
    "RegionId": "cn-hangzhou",
    "ZoneId": "cn-hangzhou-h",
    "ZoneIdSecondary": "cn-hangzhou-i",  # 双可用区场景第二可用区
    
    # VPC/VSwitch
    "CidrBlock": "172.16.0.0/12",
    "VSwitchCidrBlock": "172.16.0.0/24",
    "VSwitchCidrBlockSecondary": "172.16.1.0/24",
    
    # ECS
    "InstanceType": "ecs.c6.large",
    "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
    "SystemDiskCategory": "cloud_essd",
    "SystemDiskSize": 40,
    "InternetMaxBandwidthOut": 5,
    
    # 安全组授权
    "IpProtocol": "tcp",
    "PortRange": "80/80",
    "SourceCidrIp": "0.0.0.0/0",
    
    # SLB Listener
    "ListenerPort": 80,
    "BackendServerPort": 80,
    "ListenerProtocol": "http",
    
    # RDS
    "DBInstanceClass": "mysql.n2.medium.2c",
    "DBInstanceStorage": 20,
    "Engine": "MySQL",
    "EngineVersion": "8.0",
    "DBInstanceNetType": "Intranet",
    "SecurityIPList": "172.16.0.0/12",
    "PayType": "Postpaid",
    "DBInstanceStorageType": "cloud_essd",
    # RDS Account
    "AccountName": "admin",
    "AccountPassword": "Admin@123456",
    "SecurityIps": "172.16.0.0/12",
    
    # Redis
    "InstanceClass": "redis.master.mid.default",
    "RedisEngine": "Redis",
    "RedisEngineVersion": "7.0",
    "ChargeType": "PostPaid",
    "NetworkType": "VPC",
    
    # SLB
    "AddressType": "intranet",
    "LoadBalancerSpec": "slb.s1.small",
    
    # EIP
    "Bandwidth": "5",
    "InternetChargeType": "PayByTraffic",
}


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


def create_expert_agent(traces_dir: Path, llm_client=None, model: str = "qwen3-max") -> CloudOpsAgent:
    """
    创建专家模式 Agent
    Agent 内部根据意图识别自动选择偏序图
    
    Args:
        traces_dir: traces 输出目录
        llm_client: 可选的 LLM 客户端
        model: 模型名称
        
    Returns:
        配置好的 CloudOpsAgent
    """
    # 使用 poset_validation 预设（Expert 模式，不降级）
    config = AgentConfig.preset_poset_validation()
    config.switches.verbose = True
    config.switches.trace_enabled = True
    config.switches.trace_output_path = str(traces_dir)
    config.switches.poset_enabled = True  # 启用偏序图，Agent会根据意图自动加载
    
    # 创建 Agent
    agent = CloudOpsAgent(config=config, llm_client=llm_client)
    
    # 设置偏序图目录，让 Agent 能找到偏序图文件
    poset_dir = Path(current_dir) / "manual_scenarios"
    agent.poset_planner.set_poset_dir(str(poset_dir))
    
    # 设置 trace 输出路径
    agent.trace_store.output_path = traces_dir
    
    return agent


def inject_standard_params(agent: CloudOpsAgent, intent_type_value: str) -> None:
    """
    向 Agent 的 Blackboard 注入标准化运维参数
    确保专家模式能直接执行成功
    
    Args:
        agent: CloudOpsAgent 实例
        intent_type_value: 意图类型值（字符串）
    """
    bb = agent.blackboard
    
    # 注入全局参数
    bb.set_global("RegionId", STANDARD_PARAMS["RegionId"])
    bb.set_global("ZoneId", STANDARD_PARAMS["ZoneId"])
    bb.set_global("CidrBlock", STANDARD_PARAMS["CidrBlock"])
    
    # ECS 参数
    bb.set_global("InstanceType", STANDARD_PARAMS["InstanceType"])
    bb.set_global("ImageId", STANDARD_PARAMS["ImageId"])
    
    # 安全组授权参数
    bb.set_global("IpProtocol", STANDARD_PARAMS["IpProtocol"])
    bb.set_global("PortRange", STANDARD_PARAMS["PortRange"])
    bb.set_global("SourceCidrIp", STANDARD_PARAMS["SourceCidrIp"])
    
    # 双可用区场景特殊处理
    if intent_type_value in ("dual_zone_ecs_slb", "dual_zone_ecs_slb_rds"):
        bb.set_global("Amount", 2)  # 双可用区 2 台 ECS
        bb.set_global("ZoneIdSecondary", STANDARD_PARAMS["ZoneIdSecondary"])
    else:
        bb.set_global("Amount", 1)
    
    # 产品命名空间参数
    bb.set_ns("vpc", "CidrBlock", STANDARD_PARAMS["CidrBlock"])
    
    bb.set_ns("ecs", "InstanceType", STANDARD_PARAMS["InstanceType"])
    bb.set_ns("ecs", "ImageId", STANDARD_PARAMS["ImageId"])
    bb.set_ns("ecs", "SystemDiskCategory", STANDARD_PARAMS["SystemDiskCategory"])
    bb.set_ns("ecs", "SystemDiskSize", STANDARD_PARAMS["SystemDiskSize"])
    
    bb.set_ns("slb", "AddressType", STANDARD_PARAMS["AddressType"])
    
    # RDS 参数（场景需要时）
    if intent_type_value in ("slb_ecs_rds", "dual_zone_ecs_slb_rds"):
        bb.set_ns("rds", "DBInstanceClass", STANDARD_PARAMS["DBInstanceClass"])
        bb.set_ns("rds", "DBInstanceStorage", STANDARD_PARAMS["DBInstanceStorage"])
        bb.set_ns("rds", "Engine", STANDARD_PARAMS["Engine"])
        bb.set_ns("rds", "EngineVersion", STANDARD_PARAMS["EngineVersion"])
        bb.set_ns("rds", "DBInstanceNetType", STANDARD_PARAMS["DBInstanceNetType"])
        bb.set_ns("rds", "SecurityIPList", STANDARD_PARAMS["SecurityIPList"])
        bb.set_ns("rds", "PayType", STANDARD_PARAMS["PayType"])
        bb.set_ns("rds", "DBInstanceStorageType", STANDARD_PARAMS["DBInstanceStorageType"])
    
    # Redis 参数（场景需要时）
    if intent_type_value == "slb_ecs_redis":
        bb.set_ns("redis", "InstanceClass", STANDARD_PARAMS["InstanceClass"])
        bb.set_ns("redis", "Engine", STANDARD_PARAMS["RedisEngine"])
        bb.set_ns("redis", "EngineVersion", STANDARD_PARAMS["RedisEngineVersion"])
        bb.set_ns("redis", "ChargeType", STANDARD_PARAMS["ChargeType"])
        bb.set_ns("redis", "NetworkType", STANDARD_PARAMS["NetworkType"])
    
    # EIP 参数（场景需要时）
    if intent_type_value == "eip_slb_ecs":
        bb.set_ns("eip", "Bandwidth", STANDARD_PARAMS["Bandwidth"])
        bb.set_ns("eip", "InternetChargeType", STANDARD_PARAMS["InternetChargeType"])


def run_expert_scenario(
    agent: CloudOpsAgent,
    query: str, 
    description: str,
    scenario_idx: int,
    total_scenarios: int,
    llm_client=None
) -> Dict:
    """
    运行单个专家模式场景测试
    
    流程：
    1. 使用 IntentParser 识别意图（用于参数注入）
    2. 注入标准化运维参数
    3. 执行任务（Agent 内部自动识别意图并选择偏序图）
    
    Args:
        agent: CloudOpsAgent 实例
        query: 任务查询
        description: 场景描述
        scenario_idx: 任务索引（用于 trace 命名）
        total_scenarios: 总场景数
        llm_client: LLM 客户端（用于意图识别）
        
    Returns:
        执行结果摘要
    """
    print(f"\n{'='*60}")
    print(f"[{scenario_idx}/{total_scenarios}] {description}")
    print(f"{'='*60}")
    print(f"任务: {query}")
    
    # 设置任务索引，用于 trace 命名（便于横向比较）
    agent.trace_store.set_task_index(scenario_idx)
    
    # Step 1: 意图识别（使用 LLM，用于参数注入）
    intent_parser = IntentParser(llm_client=llm_client)
    use_llm = llm_client is not None
    intent = intent_parser.parse(query, use_llm=use_llm)
    intent_type_value = intent.intent_type.value
    intent_llm_tokens = intent.llm_tokens  # LLM token统计
    print(f"\n识别到意图: {intent_type_value} ({'LLM' if use_llm else '规则匹配'})")
    if intent_llm_tokens > 0:
        print(f"意图识别消耗 Tokens: {intent_llm_tokens}")
    
    # Step 2: 注入标准化运维参数（关键：确保专家模式能直接执行）
    inject_standard_params(agent, intent_type_value)
    
    print(f"已注入标准化参数:")
    print(f"  RegionId: {agent.blackboard.get_global('RegionId')}")
    print(f"  ZoneId: {agent.blackboard.get_global('ZoneId')}")
    print(f"  InstanceType: {agent.blackboard.get_global('InstanceType')}")
    
    # Step 3: 执行任务（Agent 内部自动识别意图并选择偏序图）
    start_time = time.time()
    try:
        result = agent.run(query)
        duration = time.time() - start_time
        
        status_symbol = "✅" if result.status == AgentStatus.SUCCESS else "❌"
        print(f"\n结果: {status_symbol} {result.status.value}")
        print(f"执行模式: {result.mode_used.value}")
        print(f"执行动作: {result.actions_executed}")
        print(f"创建资源数: {len(result.resources_created)}")
        print(f"Trace ID: {result.trace_id}")
        print(f"耗时: {duration:.2f}s")
        
        if result.error:
            print(f"错误: {result.error}")
        
        return {
            "scenario": description,
            "intent_type": intent_type_value,
            "query": query,
            "status": result.status.value,
            "mode_used": result.mode_used.value,
            "actions_executed": result.actions_executed,
            "actions_count": len(result.actions_executed),
            "resources_created": result.resources_created,
            "trace_id": result.trace_id,
            "duration_s": duration,
            "llm_tokens": intent_llm_tokens,  # 意图识别LLM token消耗
            "error": result.error
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n执行异常: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "scenario": description,
            "intent_type": intent_type_value,
            "status": "exception",
            "error": str(e),
            "duration_s": duration,
            "llm_tokens": intent_llm_tokens
        }


def main():
    print("=" * 60)
    print("  专家模式 Trace 生成 - 人工偏序图测试")
    print("=" * 60)
    print(f"模式: Expert (poset_validation)")
    print(f"特性: LLM 意图识别 + Agent 自动偏序图选择 + 标准化参数注入")
    
    # 1. 加载环境变量
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    api_key = os.environ.get("LLM_API_KEY") or env_vars.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or env_vars.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL") or env_vars.get("LLM_MODEL") or "qwen3-max"
    
    llm_client = None
    if api_key and "your_api_key_here" not in api_key:
        llm_client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"\nUsing LLM: {model} at {base_url}")
    else:
        print("\nWarning: LLM_API_KEY 未配置，将使用规则匹配进行意图识别")
    
    # 2. 读取任务列表
    tasks_path = os.path.join(current_dir, 'trace_tasks.json')
    tasks = load_tasks(tasks_path)
    print(f"加载了 {len(tasks)} 个任务")
    
    # 3. 创建 expert_traces 目录
    traces_dir = Path(current_dir) / "expert_traces"
    traces_dir.mkdir(exist_ok=True)
    print(f"输出目录: {traces_dir}")
    
    # 4. 创建 Expert Agent（只创建一次，复用执行多个场景）
    print(f"\n--- 创建 Expert Agent ---")
    agent = create_expert_agent(traces_dir, llm_client=llm_client, model=model)
    print(f"Agent 配置完成，偏序图目录: {Path(current_dir) / 'manual_scenarios'}")
    
    # ============================================================
    # 运行所有场景
    # ============================================================
    total_start = time.time()
    results = []
    
    total_runs = sum(t.get('repeat', 1) for t in tasks)
    current_run = 0
    
    for task in tasks:
        query = task.get('query')
        repeat = task.get('repeat', 1)
        
        for i in range(repeat):
            current_run += 1
            result = run_expert_scenario(
                agent=agent,
                query=query,
                description=f"任务 {current_run}",
                scenario_idx=current_run,
                total_scenarios=total_runs,
                llm_client=llm_client
            )
            results.append(result)
            
            # 简单间隔
            time.sleep(0.5)
    
    # ============================================================
    # 总结报告
    # ============================================================
    total_duration = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("  执行总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    exception_count = sum(1 for r in results if r.get("status") == "exception")
    
    print(f"总场景数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"异常: {exception_count}")
    print(f"总耗时: {total_duration:.2f}s")
    
    # 统计执行的动作数
    total_actions = sum(r.get("actions_count", 0) for r in results)
    print(f"总执行动作数: {total_actions}")
    
    # 统计LLM token消耗
    total_llm_tokens = sum(r.get("llm_tokens", 0) for r in results)
    print(f"总LLM Tokens: {total_llm_tokens}")
    
    print("\n场景详情:")
    for r in results:
        status = r.get("status", "unknown")
        if status == "success":
            status_symbol = "✅"
        elif status == "exception":
            status_symbol = "⚠️"
        else:
            status_symbol = "❌"
        
        actions_count = r.get("actions_count", 0)
        print(f"  {status_symbol} {r['scenario']}: {status} ({actions_count} actions)")
        if r.get("error") and status != "success":
            error_msg = r['error'][:60] + "..." if len(r.get('error', '')) > 60 else r.get('error', '')
            print(f"      Error: {error_msg}")
    
    # 保存结果摘要
    summary_file = traces_dir / "execution_summary.json"
    summary = {
        "run_time": datetime.now().isoformat(),
        "mode": "expert",
        "features": ["auto_intent_detection", "auto_poset_selection", "standard_params"],
        "total_scenarios": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "exception_count": exception_count,
        "total_actions": total_actions,
        "total_duration_s": total_duration,
        "total_llm_tokens": total_llm_tokens,  # 添加总LLM token统计
        "standard_params": STANDARD_PARAMS,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果摘要已保存: {summary_file}")
    print(f"Traces 保存目录: {traces_dir}")


if __name__ == "__main__":
    main()
