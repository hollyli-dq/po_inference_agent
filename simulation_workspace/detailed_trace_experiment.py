#!/usr/bin/env python3
"""
详细 LLM 交互 Trace 实验脚本

功能：
- 传入场景名字，执行 4 种模式（expert, hybrid, explore_thinking_off, explore_thinking_on）
- 详细记录每次 LLM 交互内容：
  - 当前模式
  - 当前迭代次数
  - Agent 的问句（prompt）
  - 大模型的思考返回（reasoning）
  - 大模型的回答返回（content）
  - Agent 的动作
  - Gym 的结果

用法:
    # 执行单个场景的 4 种模式
    python detailed_trace_experiment.py --scenario simple_ecs
    
    # 执行指定模式
    python detailed_trace_experiment.py --scenario simple_ecs --mode explore_thinking_on
    
    # 列出所有可用场景
    python detailed_trace_experiment.py --list
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openai import OpenAI
    from execution.cloudops_agent.config import AgentConfig, ExecutionMode
    from execution.cloudops_agent.agent import CloudOpsAgent, AgentStatus
    from execution.cloudops_agent.planning.poset_planner import PosetGraph
    from best_poset_selector import get_best_posets, SCENARIO_QUERIES
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("请确保已安装依赖并在项目根目录下运行。")
    sys.exit(1)


# ============================================================
# 标准化运维参数
# ============================================================
STANDARD_PARAMS = {
    "RegionId": "cn-hangzhou",
    "ZoneId": "cn-hangzhou-h",
    "ZoneIdSecondary": "cn-hangzhou-i",
    "CidrBlock": "172.16.0.0/12",
    "InstanceType": "ecs.c6.large",
    "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
    "SystemDiskCategory": "cloud_essd",
    "SystemDiskSize": 40,
    "IpProtocol": "tcp",
    "PortRange": "80/80",
    "SourceCidrIp": "0.0.0.0/0",
    "ListenerPort": 80,
    "BackendServerPort": 80,
    "ListenerProtocol": "http",
    "AddressType": "intranet",
    "LoadBalancerSpec": "slb.s1.small",
    "DBInstanceClass": "mysql.n2.medium.2c",
    "DBInstanceStorage": 20,
    "Engine": "MySQL",
    "EngineVersion": "8.0",
    "DBInstanceNetType": "Intranet",
    "SecurityIPList": "172.16.0.0/12",
    "PayType": "Postpaid",
    "DBInstanceStorageType": "cloud_essd",
    "InstanceClass": "redis.master.mid.default",
    "ChargeType": "PostPaid",
    "NetworkType": "VPC",
    "Bandwidth": "5",
    "InternetChargeType": "PayByTraffic",
}


# ============================================================
# LLM 交互记录数据结构
# ============================================================
@dataclass
class LLMInteraction:
    """单次 LLM 交互记录"""
    iteration: int
    timestamp: str
    # 请求
    prompt_system: str
    prompt_user: str
    # 响应
    reasoning: Optional[str]  # 思考内容（如果开启 thinking）
    content: str  # 回答内容
    # Token 统计
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # 耗时
    latency_ms: float


@dataclass
class ActionRecord:
    """动作执行记录"""
    iteration: int
    action_name: str
    action_params: Dict[str, Any]
    gym_result_success: bool
    gym_result_data: Optional[Dict[str, Any]]
    gym_result_error: Optional[str]


@dataclass
class ModeTrace:
    """单个模式的详细 Trace"""
    mode: str
    thinking_enabled: bool
    scenario_name: str
    query: str
    
    # 执行结果
    success: bool
    status: str
    error: Optional[str]
    
    # 统计
    total_iterations: int
    total_llm_calls: int
    total_tokens: int
    total_duration_s: float
    
    # 详细记录
    llm_interactions: List[LLMInteraction] = field(default_factory=list)
    actions: List[ActionRecord] = field(default_factory=list)
    
    # 降级信息（仅 hybrid 模式）
    fallback_triggered: bool = False
    fallback_at_iteration: int = -1


# ============================================================
# LLM 调用包装器 - 记录详细交互
# ============================================================
class LLMCallRecorder:
    """LLM 调用记录器，包装真实的 OpenAI 客户端"""
    
    def __init__(self, real_client: OpenAI, verbose: bool = True):
        self.real_client = real_client
        self.records: List[LLMInteraction] = []
        self.iteration = 0
        self.verbose = verbose
        
    def reset(self):
        """重置记录"""
        self.records = []
        self.iteration = 0
    
    def get_records(self) -> List[LLMInteraction]:
        """获取所有记录"""
        return self.records
    
    @property
    def chat(self):
        """返回包装的 chat 对象"""
        return _ChatWrapper(self)


class _ChatWrapper:
    """Chat API 包装器"""
    def __init__(self, recorder: LLMCallRecorder):
        self.recorder = recorder
    
    @property
    def completions(self):
        return _CompletionsWrapper(self.recorder)


class _CompletionsWrapper:
    """Completions API 包装器"""
    def __init__(self, recorder: LLMCallRecorder):
        self.recorder = recorder
    
    def create(self, **kwargs) -> Any:
        """包装 create 调用，记录请求和响应"""
        self.recorder.iteration += 1
        iteration = self.recorder.iteration
        
        # 提取请求信息
        messages = kwargs.get("messages", [])
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content", "")
        
        # 记录请求
        if self.recorder.verbose:
            print(f"\n{'='*60}")
            print(f"[LLM Call #{iteration}]")
            print(f"{'='*60}")
            print(f"\n--- User Prompt (截取前2000字) ---")
            print(user_prompt[:2000])
            if len(user_prompt) > 2000:
                print(f"... (共 {len(user_prompt)} 字符)")
        
        # 调用真实 API
        start_time = time.time()
        response = self.recorder.real_client.chat.completions.create(**kwargs)
        latency = (time.time() - start_time) * 1000
        
        # 提取响应信息
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice else ""
        
        # 提取 reasoning（思考内容）- qwen3 特有
        reasoning = None
        if choice and hasattr(choice.message, 'reasoning_content'):
            reasoning = choice.message.reasoning_content
        
        # Token 统计
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        # 记录响应
        if self.recorder.verbose:
            if reasoning:
                print(f"\n--- Reasoning (思考，截取前1500字) ---")
                print(reasoning[:1500])
                if len(reasoning) > 1500:
                    print(f"... (共 {len(reasoning)} 字符)")
            
            print(f"\n--- Content (回答，截取前1500字) ---")
            print(content[:1500])
            if len(content) > 1500:
                print(f"... (共 {len(content)} 字符)")
            
            print(f"\n--- Stats ---")
            print(f"Latency: {latency:.0f}ms, Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
        
        # 保存记录
        record = LLMInteraction(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            reasoning=reasoning,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency
        )
        self.recorder.records.append(record)
        
        return response


# ============================================================
# Agent 执行包装器 - 记录动作和 Gym 结果
# ============================================================
class ActionRecorder:
    """动作执行记录器"""
    
    def __init__(self):
        self.records: List[ActionRecord] = []
        self.iteration = 0
    
    def reset(self):
        self.records = []
        self.iteration = 0
    
    def record_action(self, action_name: str, params: Dict[str, Any], 
                      success: bool, result: Optional[Dict], error: Optional[str]):
        """记录动作执行"""
        self.iteration += 1
        record = ActionRecord(
            iteration=self.iteration,
            action_name=action_name,
            action_params=params,
            gym_result_success=success,
            gym_result_data=result,
            gym_result_error=error
        )
        self.records.append(record)
        return record


# ============================================================
# 实验执行器
# ============================================================
def load_env(env_path: str) -> Dict[str, str]:
    """加载 .env 文件"""
    env_vars = {}
    if not os.path.exists(env_path):
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


def get_scenario_params(scenario_name: str) -> Dict[str, Any]:
    """获取场景特定参数"""
    params = dict(STANDARD_PARAMS)
    if scenario_name in ("dual_zone_ecs_slb", "dual_zone_ecs_slb_rds"):
        params["Amount"] = 2
    else:
        params["Amount"] = 1
    return params


def create_agent_for_mode(
    mode_name: str,
    poset_info: Dict[str, Any],
    traces_dir: Path,
    llm_recorder: LLMCallRecorder,
    edge_threshold: float = 0.333
) -> Tuple[CloudOpsAgent, Optional[PosetGraph]]:
    """根据模式创建 Agent"""
    
    # 解析模式
    if mode_name == "expert":
        mode = ExecutionMode.EXPERT
        enable_thinking = False
    elif mode_name == "hybrid":
        mode = ExecutionMode.HYBRID
        enable_thinking = False
    elif mode_name == "explore_thinking_off":
        mode = ExecutionMode.EXPLORE
        enable_thinking = False
    elif mode_name == "explore_thinking_on":
        mode = ExecutionMode.EXPLORE
        enable_thinking = True
    else:
        raise ValueError(f"Unknown mode: {mode_name}")
    
    # 创建配置
    if mode == ExecutionMode.EXPERT:
        config = AgentConfig.preset_poset_validation()
    elif mode == ExecutionMode.HYBRID:
        config = AgentConfig.preset_hybrid_benchmark()
    else:
        config = AgentConfig.preset_trace_collection()
    
    config.switches.verbose = True
    config.switches.trace_enabled = True
    config.switches.trace_output_path = str(traces_dir)
    config.llm.react_reasoning.enable_thinking = enable_thinking
    
    # 创建 Agent，使用 LLM 记录器
    agent = CloudOpsAgent(config=config, llm_client=llm_recorder)
    
    # 设置模型和思考开关
    agent.intent_parser.llm_client = llm_recorder
    agent.react_planner.llm_client = llm_recorder
    agent.react_planner.model_name = "qwen3-max"
    agent.react_planner.enable_thinking = enable_thinking
    
    agent.trace_store.output_path = traces_dir
    
    # Expert 和 Hybrid 模式加载偏序图
    poset = None
    if mode in (ExecutionMode.EXPERT, ExecutionMode.HYBRID):
        summary_path = poset_info.get("summary_path", "")
        if summary_path and os.path.exists(summary_path):
            config.switches.poset_enabled = False
            poset = PosetGraph.load_from_hpo_posterior(summary_path, edge_threshold=edge_threshold)
            agent.poset_planner.set_poset(poset)
            agent.config.switches.poset_enabled = True
            agent.mode_selector.set_poset_graph(poset.to_dict())
    
    return agent, poset


def run_single_mode(
    scenario_name: str,
    poset_info: Dict[str, Any],
    mode_name: str,
    real_llm_client: OpenAI,
    traces_dir: Path,
    verbose: bool = True
) -> ModeTrace:
    """执行单个模式的实验"""
    
    query = poset_info["query"]
    
    print(f"\n{'#'*70}")
    print(f"# 场景: {scenario_name}")
    print(f"# 模式: {mode_name}")
    print(f"# 任务: {query}")
    print(f"{'#'*70}")
    
    # 创建 LLM 记录器
    llm_recorder = LLMCallRecorder(real_llm_client, verbose=verbose)
    
    # 确定是否开启 thinking
    enable_thinking = "thinking_on" in mode_name
    
    try:
        # 创建 Agent
        agent, poset = create_agent_for_mode(
            mode_name, poset_info, traces_dir, llm_recorder
        )
        
        # 设置预设参数
        preset_params = get_scenario_params(scenario_name)
        agent.set_preset_params(preset_params)
        
        # 执行
        start_time = time.time()
        result = agent.run(query)
        duration = time.time() - start_time
        
        # 打印结果
        status_symbol = "✅" if result.status == AgentStatus.SUCCESS else "❌"
        print(f"\n{'='*60}")
        print(f"执行结果: {status_symbol} {result.status.value}")
        print(f"执行动作: {result.actions_executed}")
        print(f"降级次数: {result.fallback_count}")
        print(f"总耗时: {duration:.2f}s")
        print(f"{'='*60}")
        
        # 构建 ModeTrace
        llm_records = llm_recorder.get_records()
        
        # 从保存的 trace 文件中读取动作记录
        action_records = []
        trace_id = result.trace_id
        if trace_id:
            # 查找保存的 trace JSON 文件
            trace_files = list(traces_dir.glob(f"{trace_id}.json"))
            if trace_files:
                try:
                    with open(trace_files[0], 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                    for idx, action in enumerate(trace_data.get("actions", []), 1):
                        action_records.append(ActionRecord(
                            iteration=idx,
                            action_name=action.get("action_name", ""),
                            action_params=action.get("params", {}),
                            gym_result_success=action.get("status", "") == "success",
                            gym_result_data=action.get("result"),
                            gym_result_error=action.get("error")
                        ))
                except Exception as e:
                    print(f"Warning: 读取 trace 文件失败: {e}")
        
        mode_trace = ModeTrace(
            mode=mode_name,
            thinking_enabled=enable_thinking,
            scenario_name=scenario_name,
            query=query,
            success=result.status == AgentStatus.SUCCESS,
            status=result.status.value,
            error=result.error,
            total_iterations=len(llm_records),
            total_llm_calls=len(llm_records),
            total_tokens=sum(r.total_tokens for r in llm_records),
            total_duration_s=duration,
            llm_interactions=llm_records,
            actions=action_records,
            fallback_triggered=result.fallback_count > 0,
            fallback_at_iteration=result.fallback_count if result.fallback_count > 0 else -1
        )
        
        return mode_trace
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return ModeTrace(
            mode=mode_name,
            thinking_enabled=enable_thinking,
            scenario_name=scenario_name,
            query=query,
            success=False,
            status="exception",
            error=str(e),
            total_iterations=0,
            total_llm_calls=0,
            total_tokens=0,
            total_duration_s=0,
            llm_interactions=[],
            actions=[]
        )


def save_detailed_trace(traces: List[ModeTrace], output_path: Path):
    """保存详细 trace 到文件"""
    
    # 转换为可序列化的字典
    data = []
    for trace in traces:
        trace_dict = asdict(trace)
        # 转换 LLMInteraction 和 ActionRecord
        trace_dict["llm_interactions"] = [asdict(r) for r in trace.llm_interactions]
        trace_dict["actions"] = [asdict(a) for a in trace.actions]
        data.append(trace_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细 Trace 已保存到: {output_path}")


def print_summary(traces: List[ModeTrace]):
    """打印摘要"""
    print(f"\n{'='*70}")
    print("实验摘要")
    print(f"{'='*70}")
    
    for trace in traces:
        status = "✅" if trace.success else "❌"
        thinking = "ON" if trace.thinking_enabled else "OFF"
        print(f"\n{status} {trace.mode} (thinking={thinking})")
        print(f"   状态: {trace.status}")
        print(f"   LLM调用次数: {trace.total_llm_calls}")
        print(f"   总Tokens: {trace.total_tokens}")
        print(f"   动作数: {len(trace.actions)}")
        print(f"   耗时: {trace.total_duration_s:.2f}s")
        if trace.fallback_triggered:
            print(f"   降级: 是 (第{trace.fallback_at_iteration}次)")
        if trace.error:
            print(f"   错误: {trace.error[:100]}...")


# ============================================================
# 主函数
# ============================================================
ALL_MODES = ["expert", "hybrid", "explore_thinking_off", "explore_thinking_on"]


def main():
    parser = argparse.ArgumentParser(description='详细 LLM 交互 Trace 实验')
    parser.add_argument('--scenario', type=str, help='场景名称')
    parser.add_argument('--mode', type=str, choices=ALL_MODES + ["all"], default="all",
                        help='执行模式 (默认执行所有4种)')
    parser.add_argument('--list', action='store_true', help='列出所有可用场景')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    args = parser.parse_args()
    
    # 加载最优偏序图配置
    best_posets = get_best_posets()
    
    # 列出场景
    if args.list:
        print("可用场景:")
        for name, info in sorted(best_posets.items()):
            print(f"  - {name}: {info['query'][:50]}...")
        return
    
    # 检查场景
    if not args.scenario:
        print("Error: 请指定 --scenario 参数")
        print("使用 --list 查看所有可用场景")
        sys.exit(1)
    
    if args.scenario not in best_posets:
        print(f"Error: 未知场景 '{args.scenario}'")
        print("使用 --list 查看所有可用场景")
        sys.exit(1)
    
    # 加载 LLM 配置
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    api_key = os.environ.get("LLM_API_KEY") or env_vars.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or env_vars.get("LLM_BASE_URL")
    
    if not api_key or "your_api_key_here" in api_key:
        print("Error: LLM_API_KEY 未配置")
        sys.exit(1)
    
    # 创建真实 LLM 客户端
    real_client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 确定要执行的模式
    modes_to_run = ALL_MODES if args.mode == "all" else [args.mode]
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(current_dir) / "detailed_traces" / f"{args.scenario}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n场景: {args.scenario}")
    print(f"模式: {modes_to_run}")
    print(f"输出目录: {output_dir}")
    
    # 执行实验
    traces = []
    poset_info = best_posets[args.scenario]
    
    for mode_name in modes_to_run:
        trace = run_single_mode(
            scenario_name=args.scenario,
            poset_info=poset_info,
            mode_name=mode_name,
            real_llm_client=real_client,
            traces_dir=output_dir,
            verbose=not args.quiet
        )
        traces.append(trace)
        
        # 模式间间隔
        if mode_name != modes_to_run[-1]:
            print("\n等待 2 秒后执行下一个模式...")
            time.sleep(2)
    
    # 保存详细 trace
    output_file = output_dir / "detailed_trace.json"
    save_detailed_trace(traces, output_file)
    
    # 打印摘要
    print_summary(traces)
    
    print(f"\n实验完成！")


if __name__ == "__main__":
    main()
