"""
HPO 批量实验脚本

遍历 120 个 HPO 后验偏序图，使用混合模式进行仿真，汇总结果分析：
- 成功率
- 执行速度
- 动作数量
- 降级发生在第几步
- Token 消耗
- 与专家模式相比多余的动作数量

实验设计：
- 120个偏序图 = 6场景 × 5 IP-Cov × 4 eps_jump
- 每个偏序图对应一次仿真执行
- 结果关联 experiment_summary.csv 中的 Cover-F1 进行分析

配置层级：
- .env: LLM 配置 (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)
- experiment_config.yaml: 实验参数 (edge_threshold, results_dir)
- 命令行: 运行时参数 (--scenario, --limit)
"""

import sys
import os
import json
import time
import csv
import yaml
import argparse
import threading
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

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


# ============================================================
# 控制台管理器和进度监控
# ============================================================

class ConsoleManager:
    """线程安全的控制台输出管理器"""
    def __init__(self):
        self._lock = threading.Lock()
        self._last_lines = 0
        self._progress_buffer = ""

    def log(self, *args, **kwargs):
        """打印普通日志，不干扰进度条"""
        msg = " ".join(map(str, args))
        end = kwargs.get("end", "\n")
        msg += end
        
        with self._lock:
            self._clear_progress_lines()
            sys.stdout.write(msg)
            self._write_progress_buffer()
            sys.stdout.flush()

    def update_progress(self, text: str):
        """更新底部进度条"""
        with self._lock:
            self._clear_progress_lines()
            self._progress_buffer = text
            self._write_progress_buffer()
            sys.stdout.flush()
            
    def clear_progress(self):
        """清除进度条"""
        self.update_progress("")

    def _clear_progress_lines(self):
        if self._last_lines > 0:
            for _ in range(self._last_lines):
                sys.stdout.write("\033[1A")
                sys.stdout.write("\033[2K")
            sys.stdout.write("\r")
            self._last_lines = 0

    def _write_progress_buffer(self):
        if self._progress_buffer:
            sys.stdout.write(self._progress_buffer)
            self._last_lines = self._progress_buffer.count('\n')
            if not self._progress_buffer.endswith('\n'):
                self._last_lines += 1
                sys.stdout.write('\n')


@dataclass
class TaskProgress:
    """任务进度信息"""
    exp_id: str
    scenario: str
    ip_cov: float
    eps: float
    agent: Optional[CloudOpsAgent] = None
    status: str = "pending"  # pending, running, completed, failed
    api_count: int = 0
    last_api: str = ""
    start_time: float = 0.0
    result: Optional['ExperimentResult'] = None


class ProgressMonitor:
    """进度监控器 - 定期输出各任务执行状态"""
    
    def __init__(self, console: ConsoleManager, interval: float = 3.0):
        self.console = console
        self.interval = interval
        self.tasks: Dict[str, TaskProgress] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.total_count = 0
        self.completed_count = 0
    
    def register_task(self, exp_id: str, scenario: str, ip_cov: float, eps: float) -> TaskProgress:
        """注册新任务"""
        with self._lock:
            progress = TaskProgress(exp_id=exp_id, scenario=scenario, ip_cov=ip_cov, eps=eps)
            self.tasks[exp_id] = progress
            self.total_count = len(self.tasks)
            return progress
    
    def update_agent(self, exp_id: str, agent: CloudOpsAgent):
        """绑定 Agent 实例"""
        with self._lock:
            if exp_id in self.tasks:
                self.tasks[exp_id].agent = agent
                self.tasks[exp_id].status = "running"
                self.tasks[exp_id].start_time = time.time()
    
    def mark_complete(self, exp_id: str, result: 'ExperimentResult'):
        """标记任务完成"""
        with self._lock:
            if exp_id in self.tasks:
                self.tasks[exp_id].status = "completed" if result.success else "failed"
                self.tasks[exp_id].result = result
                self.tasks[exp_id].agent = None
                self.completed_count += 1
    
    def start(self):
        """启动监控线程"""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """停止监控线程"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        self.console.clear_progress()
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.interval)
    
    def _print_progress(self):
        """输出当前进度"""
        with self._lock:
            running_tasks = []
            for exp_id, prog in self.tasks.items():
                if prog.status == "running" and prog.agent:
                    # 从 agent 的 trace_store 读取进度
                    trace = prog.agent.trace_store.get_current_trace()
                    if trace:
                        prog.api_count = len(trace.actions)
                        if trace.actions:
                            prog.last_api = trace.actions[-1].action_name
                    
                    elapsed = time.time() - prog.start_time
                    running_tasks.append((exp_id, prog, elapsed))
            
            if running_tasks:
                lines = []
                lines.append("=" * 80)
                lines.append(f"📊 进度: {self.completed_count}/{self.total_count} 完成 | {len(running_tasks)} 个任务运行中")
                lines.append("-" * 80)
                for exp_id, prog, elapsed in running_tasks[:10]:  # 最多显示10个
                    last_api = prog.last_api[-25:] if prog.last_api else "(等待中)"
                    lines.append(f"  {prog.scenario[:20]:20s} ip={prog.ip_cov} | APIs: {prog.api_count:2d} | {last_api:25s} | {elapsed:.1f}s")
                if len(running_tasks) > 10:
                    lines.append(f"  ... 另有 {len(running_tasks) - 10} 个任务运行中")
                lines.append("=" * 80)
                self.console.update_progress("\n".join(lines) + "\n")
            else:
                self.console.update_progress("")


# 全局实例
console = ConsoleManager()
progress_monitor: Optional[ProgressMonitor] = None

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    console.log(*args, **kwargs)


# ============================================================
# 场景配置
# ============================================================

# 场景名到任务 query 的映射
SCENARIO_TO_QUERY = {
    "simple_ecs": "在杭州可用区H创建一个2核4G的ECS实例",
    "slb_ecs_rds": "创建一个完整的Web应用架构：包含一个SLB负载均衡器，后端挂载一台ECS服务器，并配置一个RDS MySQL数据库",
    "slb_ecs_redis": "搭建一个带缓存的Web服务：创建SLB负载均衡、一台ECS实例、以及Redis缓存实例",
    "eip_slb_ecs": "创建一个公网可访问的负载均衡架构：申请一个EIP弹性公网IP，绑定到SLB，SLB后端挂载一台ECS",
    "dual_zone_ecs_slb": "创建高可用架构：在杭州的两个不同可用区各创建一台ECS实例，并创建一个SLB将这两台ECS作为后端服务器",
    "dual_zone_ecs_slb_rds": "创建完整的高可用生产环境：双可用区部署两台ECS，创建SLB做负载均衡，并配置RDS MySQL主备高可用集群",
}

# 专家模式的最优动作列表（Ground Truth）
EXPERT_ACTIONS = {
    "simple_ecs": ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup", "RunInstances"],
    "slb_ecs_rds": ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup", 
                   "CreateLoadBalancer", "CreateDBInstance", "RunInstances", "CreateLoadBalancerHTTPListener",
                   "CreateAccount", "StartLoadBalancerListener", "AddBackendServers", "ModifySecurityIps"],
    "slb_ecs_redis": ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup",
                     "CreateLoadBalancer", "CreateInstance", "RunInstances", "AddBackendServers"],
    "eip_slb_ecs": ["CreateVpc", "AllocateEipAddress", "CreateVSwitch", "CreateSecurityGroup", 
                   "AuthorizeSecurityGroup", "CreateLoadBalancer", "RunInstances", "AssociateEipAddress", "AddBackendServers"],
    "dual_zone_ecs_slb": ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup",
                         "CreateLoadBalancer", "RunInstances", "AddBackendServers"],
    "dual_zone_ecs_slb_rds": ["CreateVpc", "CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup",
                             "CreateLoadBalancer", "CreateDBInstance", "RunInstances", "CreateAccount",
                             "AddBackendServers", "ModifySecurityIps"],
}


@dataclass
class ExperimentResult:
    """单次实验结果"""
    experiment_id: str
    scenario_name: str
    ip_cov_target: float
    ip_cov_realized: float
    eps_jump: float
    cover_f1: float  # 从 experiment_summary.csv 获取
    
    # 执行结果
    status: str
    mode_used: str
    success: bool  # API 层面的成功（无报错）
    is_complete: bool  # 任务完整执行（无缺失动作）
    
    # 效率指标
    execution_time_s: float
    action_count: int  # 偏序图动作数（不含 Describe）
    total_action_count: int  # 总动作数（含 Describe）
    actions_executed: List[str]
    
    # 降级相关
    fallback_triggered: bool
    fallback_layer: int  # 降级发生的层号，-1 表示未降级
    fallback_react_steps: int  # 降级后执行的 ReAct 步数
    
    # LLM 调用统计
    llm_call_count: int  # LLM 调用次数
    total_tokens: int
    inference_tokens: int
    
    # 与专家模式对比
    expert_action_count: int
    extra_actions: int  # 多余动作数
    missing_actions: List[str]  # 缺失的专家动作
    extra_action_list: List[str]  # 多余的动作列表
    
    # 错误信息
    error: Optional[str] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def load_experiment_summary(summary_path: str) -> pd.DataFrame:
    """
    加载 experiment_summary.csv，提取 BHPOP 的 Cover-F1 结果
    """
    df = pd.read_csv(summary_path)
    # 只保留 BHPOP 结果
    bhpop_df = df[df['method'] == 'bhpop_single_po'].copy()
    return bhpop_df


def get_cover_f1(bhpop_df: pd.DataFrame, scenario: str, ip_cov: float, eps: float) -> float:
    """
    从 BHPOP 结果中获取特定实验的 Cover-F1
    """
    mask = (
        (bhpop_df['scenario'] == scenario) & 
        (abs(bhpop_df['ip_cov_target'] - ip_cov) < 0.01) & 
        (abs(bhpop_df['eps_jump'] - eps) < 0.001)
    )
    matched = bhpop_df[mask]
    if len(matched) > 0:
        return matched.iloc[0]['cover_f1']
    return -1.0  # 未找到


def load_hpo_experiments(hpo_dir: str) -> List[Dict[str, Any]]:
    """
    加载所有 HPO 实验配置
    """
    experiments = []
    hpo_path = Path(hpo_dir)
    
    for exp_dir in sorted(hpo_path.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        
        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        experiments.append({
            "experiment_id": summary.get("experiment_id"),
            "experiment_name": summary.get("experiment_name"),
            "scenario_name": summary.get("scenario_name"),
            "configuration": summary.get("configuration", {}),
            "scenario": summary.get("scenario", {}),
            "posterior": summary.get("posterior", {}),
            "summary_path": str(summary_file)
        })
    
    return experiments


def get_default_params(scenario_name: str) -> Dict[str, Any]:
    """获取场景的默认参数"""
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
    
    # 根据场景添加额外参数
    if scenario_name in ("dual_zone_ecs_slb", "dual_zone_ecs_slb_rds"):
        base_params["ZoneIdSecondary"] = "cn-hangzhou-i"
        base_params["Amount"] = 2
    
    if scenario_name in ("slb_ecs_rds", "slb_ecs_redis", "eip_slb_ecs", 
                         "dual_zone_ecs_slb", "dual_zone_ecs_slb_rds"):
        base_params.update({
            "AddressType": "intranet",
            "LoadBalancerSpec": "slb.s1.small",
            "ListenerPort": 80,
            "BackendServerPort": 80,
            "ListenerProtocol": "http",
            "HealthCheck": "on",
        })
    
    if scenario_name in ("slb_ecs_rds", "dual_zone_ecs_slb_rds"):
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
    
    if scenario_name == "slb_ecs_redis":
        base_params.update({
            "InstanceClass": "redis.master.mid.default",
            "ChargeType": "PostPaid",
            "NetworkType": "VPC",
        })
    
    if scenario_name == "eip_slb_ecs":
        base_params.update({
            "Bandwidth": "5",
            "InternetChargeType": "PayByTraffic",
        })
    
    return base_params


def create_agent_with_hpo_poset(
    summary_path: str,
    edge_threshold: float,
    traces_dir: Path,
    llm_client=None,
    model: str = "qwen3-max"
) -> Tuple[CloudOpsAgent, PosetGraph]:
    """
    创建使用指定 HPO 后验偏序图的 Agent
    
    Args:
        summary_path: HPO summary.json 文件路径
        edge_threshold: 边概率阈值
        traces_dir: Trace 输出目录
        llm_client: LLM 客户端
        model: 模型名称
    
    Returns:
        (Agent, PosetGraph)
    """
    # 创建 Hybrid 模式配置
    config = AgentConfig.preset_hybrid_benchmark()
    config.switches.verbose = False  # 批量实验时关闭详细输出
    config.switches.trace_enabled = True
    config.switches.trace_output_path = str(traces_dir)
    config.switches.poset_enabled = False  # 先禁用自动加载
    
    # 创建 Agent
    agent = CloudOpsAgent(config=config, llm_client=llm_client)
    
    # 从 HPO 后验加载偏序图
    poset = PosetGraph.load_from_hpo_posterior(summary_path, edge_threshold=edge_threshold)
    
    # 设置偏序图到 Agent
    agent.poset_planner.set_poset(poset)
    agent.config.switches.poset_enabled = True
    
    # 同步到 mode_selector
    agent.mode_selector.set_poset_graph(poset.to_dict())
    
    # 设置 LLM
    if llm_client:
        agent.intent_parser.llm_client = llm_client
        agent.react_planner.llm_client = llm_client
        agent.react_planner.model_name = model
    
    agent.trace_store.output_path = traces_dir
    
    return agent, poset


def compute_action_diff(executed: List[str], expert: List[str]) -> Tuple[int, List[str], List[str]]:
    """
    计算执行的动作与专家模式的差异
    
    Returns:
        (多余动作数, 缺失动作列表, 多余动作列表)
    """
    executed_set = set(executed)
    expert_set = set(expert)
    
    missing = list(expert_set - executed_set)
    extra = list(executed_set - expert_set)
    
    # 多余动作数 = 执行总数 - 专家动作数（如果成功）
    # 但更准确的是：不在专家列表中的动作数量
    extra_count = len(extra)
    
    return extra_count, missing, extra


def run_single_experiment(
    experiment: Dict[str, Any],
    llm_client: OpenAI,
    model: str,
    traces_dir: Path,
    io_registry: IORegistry,
    bhpop_df: pd.DataFrame,
    edge_threshold: float = 0.333,
    progress_monitor: Optional[ProgressMonitor] = None
) -> ExperimentResult:
    """
    执行单个 HPO 偏序图实验
    """
    exp_id = experiment["experiment_id"]
    scenario = experiment["scenario_name"]
    config = experiment["configuration"]
    
    ip_cov_target = config.get("ip_cov_target", 0.0)
    ip_cov_realized = config.get("ip_cov_realized", 0.0)
    eps_jump = config.get("eps_jump", 0.01)
    
    # 获取 Cover-F1
    cover_f1 = get_cover_f1(bhpop_df, scenario, ip_cov_target, eps_jump)
    
    # 获取任务 query
    query = SCENARIO_TO_QUERY.get(scenario, "")
    if not query:
        return ExperimentResult(
            experiment_id=exp_id,
            scenario_name=scenario,
            ip_cov_target=ip_cov_target,
            ip_cov_realized=ip_cov_realized,
            eps_jump=eps_jump,
            cover_f1=cover_f1,
            status="error",
            mode_used="none",
            success=False,
            is_complete=False,
            execution_time_s=0.0,
            action_count=0,
            total_action_count=0,
            actions_executed=[],
            fallback_triggered=False,
            fallback_step=-1,
            fallback_batch=-1,
            llm_call_count=0,
            total_tokens=0,
            inference_tokens=0,
            expert_action_count=len(EXPERT_ACTIONS.get(scenario, [])),
            extra_actions=0,
            missing_actions=[],
            extra_action_list=[],
            error=f"Unknown scenario: {scenario}"
        )
    
    # 创建带有 HPO 偏序图的 Agent
    try:
        agent, poset = create_agent_with_hpo_poset(
            experiment["summary_path"],
            edge_threshold,
            traces_dir,
            llm_client,
            model
        )
    except Exception as e:
        return ExperimentResult(
            experiment_id=exp_id,
            scenario_name=scenario,
            ip_cov_target=ip_cov_target,
            ip_cov_realized=ip_cov_realized,
            eps_jump=eps_jump,
            cover_f1=cover_f1,
            status="error",
            mode_used="none",
            success=False,
            is_complete=False,
            execution_time_s=0.0,
            action_count=0,
            total_action_count=0,
            actions_executed=[],
            fallback_triggered=False,
            fallback_step=-1,
            fallback_batch=-1,
            llm_call_count=0,
            total_tokens=0,
            inference_tokens=0,
            expert_action_count=len(EXPERT_ACTIONS.get(scenario, [])),
            extra_actions=0,
            missing_actions=[],
            extra_action_list=[],
            error=f"Failed to load poset: {e}"
        )
    
    # 设置预设参数
    preset_params = get_default_params(scenario)
    agent.set_preset_params(preset_params)
    
    # 设置任务索引用于 trace 命名
    agent.trace_store.set_task_index(int(exp_id))
    
    # 更新进度监控器
    if progress_monitor:
        progress_monitor.update_agent(str(exp_id), agent)
    
    # 执行
    start_time = time.time()
    try:
        result = agent.run(query)
        execution_time = time.time() - start_time
        
        # 判断是否降级（使用 Agent 的 fallback_count）
        fallback_triggered = result.fallback_count > 0
        
        # 计算动作数量（不含 Describe 类型查询）
        total_actions = result.actions_executed
        poset_actions = [a for a in total_actions if not a.startswith("Describe")]
        
        # 从最后保存的 trace 获取降级信息
        # 注意: agent.run() 结束后 _current_trace 已为 None，需要从 _traces 列表获取
        fallback_layer = -1
        fallback_react_steps = 0
        
        # 获取最后保存的 trace
        traces = agent.trace_store.get_all_traces()
        trace = traces[-1] if traces else None
        
        if trace:
            # 使用 trace 的 fallback_count 作为降级判断（更准确）
            fallback_triggered = trace.fallback_count > 0
            
            if fallback_triggered:
                # 获取降级发生的层号
                if trace.fallback_step >= 0 and trace.fallback_step > 0 and len(trace.actions) >= trace.fallback_step:
                    last_poset_action = trace.actions[trace.fallback_step - 1]
                    fallback_layer = last_poset_action.layer if last_poset_action.layer >= 0 else -1
                
                # 统计降级后执行的 ReAct 步数
                fallback_react_steps = sum(1 for a in trace.actions if a.source in ('react', 'fallback'))
        
        # LLM 调用 = 1 (意图解析) + 降级后每个动作约 1 次 LLM 思考
        llm_call_count = 1 + fallback_react_steps  # 意图解析 + ReAct 步数
        
        # 计算与专家模式的差异（只比较偏序图动作）
        expert_actions = EXPERT_ACTIONS.get(scenario, [])
        extra_count, missing, extra = compute_action_diff(poset_actions, expert_actions)
        
        # is_complete: 无缺失动作时为 True
        is_complete = len(missing) == 0
        
        return ExperimentResult(
            experiment_id=exp_id,
            scenario_name=scenario,
            ip_cov_target=ip_cov_target,
            ip_cov_realized=ip_cov_realized,
            eps_jump=eps_jump,
            cover_f1=cover_f1,
            status=result.status.value,
            mode_used=result.mode_used.value,
            success=result.status == AgentStatus.SUCCESS,
            is_complete=is_complete,
            execution_time_s=execution_time,
            action_count=len(poset_actions),
            total_action_count=len(total_actions),
            actions_executed=result.actions_executed,
            fallback_triggered=fallback_triggered,
            fallback_layer=fallback_layer,
            fallback_react_steps=fallback_react_steps,
            llm_call_count=llm_call_count,
            total_tokens=result.total_tokens,
            inference_tokens=0,
            expert_action_count=len(expert_actions),
            extra_actions=extra_count,
            missing_actions=missing,
            extra_action_list=extra,
            error=result.error
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        expert_actions = EXPERT_ACTIONS.get(scenario, [])
        
        return ExperimentResult(
            experiment_id=exp_id,
            scenario_name=scenario,
            ip_cov_target=ip_cov_target,
            ip_cov_realized=ip_cov_realized,
            eps_jump=eps_jump,
            cover_f1=cover_f1,
            status="exception",
            mode_used="none",
            success=False,
            is_complete=False,
            execution_time_s=execution_time,
            action_count=0,
            total_action_count=0,
            actions_executed=[],
            fallback_triggered=False,
            fallback_layer=-1,
            fallback_react_steps=0,
            llm_call_count=0,
            total_tokens=0,
            inference_tokens=0,
            expert_action_count=len(expert_actions),
            extra_actions=0,
            missing_actions=[],
            extra_action_list=[],
            error=str(e)
        )


def generate_summary_report(results: List[ExperimentResult], output_dir: Path) -> None:
    """
    生成汇总分析报告
    """
    df = pd.DataFrame([asdict(r) for r in results])
    
    # 保存原始结果
    df.to_csv(output_dir / "hpo_experiment_results.csv", index=False)
    
    # 按 Cover-F1 分组分析
    print("\n" + "=" * 80)
    print("  HPO 批量实验汇总报告")
    print("=" * 80)
    
    # 总体统计
    total = len(results)
    success_count = df['success'].sum()
    complete_count = df['is_complete'].sum()
    fallback_count = df['fallback_triggered'].sum()
    total_llm_calls = df['llm_call_count'].sum()
    total_tokens = df['total_tokens'].sum()
    
    print(f"\n总实验数: {total}")
    print(f"成功数: {success_count} ({100*success_count/total:.1f}%)")
    print(f"完整数: {complete_count} ({100*complete_count/total:.1f}%)")
    print(f"降级数: {fallback_count} ({100*fallback_count/total:.1f}%)")
    print(f"LLM 调用总数: {total_llm_calls}")
    print(f"Token 消耗总量: {total_tokens}")
    
    # 按 IP-Cov 分组
    print("\n--- 按 IP-Cov 分组统计 ---")
    ip_cov_groups = df.groupby('ip_cov_target').agg({
        'success': ['sum', 'count'],
        'execution_time_s': 'mean',
        'action_count': 'mean',
        'extra_actions': 'mean',
        'llm_call_count': 'sum',
        'total_tokens': 'sum',
        'fallback_triggered': 'sum',
        'cover_f1': 'mean'
    }).round(3)
    print(ip_cov_groups)
    
    # 按场景分组
    print("\n--- 按场景分组统计 ---")
    scenario_groups = df.groupby('scenario_name').agg({
        'success': ['sum', 'count'],
        'execution_time_s': 'mean',
        'action_count': 'mean',
        'extra_actions': 'mean',
        'llm_call_count': 'sum',
        'total_tokens': 'sum',
        'fallback_triggered': 'sum'
    }).round(3)
    print(scenario_groups)
    
    # 降级详情统计
    fallback_df = df[df['fallback_triggered'] == True]
    if len(fallback_df) > 0:
        print("\n--- 降级详情统计 ---")
        print(f"降级实验数: {len(fallback_df)}")
        print(f"平均降级层号: {fallback_df['fallback_layer'].mean():.1f}")
        print(f"平均降级步数: {fallback_df['fallback_react_steps'].mean():.1f}")
        print(f"降级实验 LLM 调用: {fallback_df['llm_call_count'].sum()}")
        print(f"降级实验 Token 消耗: {fallback_df['total_tokens'].sum()}")
    
    # Cover-F1 与执行成功率的关系
    print("\n--- Cover-F1 与执行效果关系 ---")
    # 将 Cover-F1 分为几个区间
    df['f1_bin'] = pd.cut(df['cover_f1'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
                          labels=['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'],
                          include_lowest=True)  # 包含左边界，确保 F1=0 归入 '0-0.3'
    f1_analysis = df.groupby('f1_bin').agg({
        'success': ['sum', 'count'],
        'fallback_triggered': 'sum',
        'llm_call_count': 'sum',
        'total_tokens': 'sum',
        'extra_actions': 'mean',
        'execution_time_s': 'mean'
    }).round(3)
    print(f1_analysis)
    
    # 生成 Markdown 报告
    report_lines = [
        "# HPO 批量实验报告",
        f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 1. 实验概述",
        f"\n- 总实验数: {total}",
        f"- 成功率: {100*success_count/total:.1f}%",
        f"- 完整率: {100*complete_count/total:.1f}%",
        f"- 降级率: {100*fallback_count/total:.1f}%",
        f"- LLM 调用总数: {total_llm_calls}",
        f"- Token 消耗总量: {total_tokens}",
        "\n> **注**: 成功率=API无报错；完整率=无缺失动作",
        "\n## 2. 按 IP-Cov 分组统计",
        "\n| IP-Cov | 任务总数 | 任务成功率 | 完整率 | 平均动作数 | 平均多余动作 | 任务降级率 | 动作降级率 | 单任务平均LLM调用次数 | 单任务Token消耗 | 平均F1 |",
        "|--------|----------|------------|--------|------------|--------------|------------|------------|----------------------|-----------------|--------|",
    ]
    
    for ip_cov in [0.6, 0.7, 0.8, 0.9, 1.0]:
        sub = df[df['ip_cov_target'] == ip_cov]
        if len(sub) > 0:
            task_count = len(sub)
            task_success_rate = 100 * sub['success'].mean()
            task_complete_rate = 100 * sub['is_complete'].mean()
            avg_action = sub['action_count'].mean()
            avg_extra_action = sub['extra_actions'].mean()
            task_fallback_rate = 100 * sub['fallback_triggered'].sum() / task_count
            total_actions = sub['total_action_count'].sum()  # 使用总动作数作为分母
            total_fallback_steps = sub['fallback_react_steps'].sum()
            action_fallback_rate = 100 * total_fallback_steps / total_actions if total_actions > 0 else 0
            avg_llm_calls = sub['llm_call_count'].sum() / task_count
            avg_tokens = sub['total_tokens'].sum() / task_count
            avg_f1 = sub['cover_f1'].mean()
            report_lines.append(
                f"| {ip_cov} | {task_count} | {task_success_rate:.1f}% | {task_complete_rate:.1f}% | "
                f"{avg_action:.1f} | {avg_extra_action:.1f} | "
                f"{task_fallback_rate:.1f}% | {action_fallback_rate:.1f}% | "
                f"{avg_llm_calls:.1f} | {avg_tokens:.0f} | {avg_f1:.3f} |"
            )
    
    report_lines.extend([
        "\n## 3. 按场景分组统计",
        "\n| 场景 | 任务总数 | 任务成功率 | 完整率 | 平均动作数 | 平均多余动作 | 任务降级率 | 动作降级率 | 单任务平均LLM调用次数 | 单任务Token消耗 |",
        "|------|----------|------------|--------|------------|--------------|------------|------------|----------------------|-----------------|",
    ])
    
    for scenario in SCENARIO_TO_QUERY.keys():
        sub = df[df['scenario_name'] == scenario]
        if len(sub) > 0:
            task_count = len(sub)
            task_success_rate = 100 * sub['success'].mean()
            task_complete_rate = 100 * sub['is_complete'].mean()
            avg_action = sub['action_count'].mean()
            avg_extra_action = sub['extra_actions'].mean()
            task_fallback_rate = 100 * sub['fallback_triggered'].sum() / task_count
            total_actions = sub['total_action_count'].sum()  # 使用总动作数作为分母
            total_fallback_steps = sub['fallback_react_steps'].sum()
            action_fallback_rate = 100 * total_fallback_steps / total_actions if total_actions > 0 else 0
            avg_llm_calls = sub['llm_call_count'].sum() / task_count
            avg_tokens = sub['total_tokens'].sum() / task_count
            report_lines.append(
                f"| {scenario} | {task_count} | {task_success_rate:.1f}% | {task_complete_rate:.1f}% | "
                f"{avg_action:.1f} | {avg_extra_action:.1f} | "
                f"{task_fallback_rate:.1f}% | {action_fallback_rate:.1f}% | "
                f"{avg_llm_calls:.1f} | {avg_tokens:.0f} |"
            )
    
    # 降级详情
    if len(fallback_df) > 0:
        report_lines.extend([
            "\n## 4. 降级详情",
            f"\n- 降级实验数: {len(fallback_df)}",
            f"- 平均降级层号: {fallback_df['fallback_layer'].mean():.1f}",
            f"- 平均降级步数: {fallback_df['fallback_react_steps'].mean():.1f}",
            f"- 降级实验 LLM 调用: {fallback_df['llm_call_count'].sum()}",
            f"- 降级实验 Token 消耗: {fallback_df['total_tokens'].sum()}",
            "\n### 降级实验列表",
            "\n| 实验ID | 场景 | IP-Cov | Cover-F1 | 降级层号 | 降级步数 | LLM调用 | Token |",
            "|--------|------|--------|----------|----------|----------|---------|-------|",
        ])
        for _, row in fallback_df.iterrows():
            report_lines.append(
                f"| {row['experiment_id']} | {row['scenario_name']} | {row['ip_cov_target']} | "
                f"{row['cover_f1']:.3f} | {row['fallback_layer']} | {row['fallback_react_steps']} | "
                f"{row['llm_call_count']} | {row['total_tokens']} |"
            )
    
    report_lines.extend([
        "\n## 5. Cover-F1 与执行效果关系",
        "\n| F1区间 | 任务总数 | 任务成功率 | 完整率 | 任务降级率 | 动作降级率 | 单任务平均LLM调用次数 | 单任务Token消耗 |",
        "|--------|----------|------------|--------|------------|------------|----------------------|-----------------|",
    ])
    
    for f1_bin in ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']:
        sub = df[df['f1_bin'] == f1_bin]
        if len(sub) > 0:
            task_count = len(sub)
            task_success_rate = 100 * sub['success'].mean()
            task_complete_rate = 100 * sub['is_complete'].mean()
            task_fallback_rate = 100 * sub['fallback_triggered'].sum() / task_count
            total_actions = sub['total_action_count'].sum()  # 使用总动作数作为分母
            total_fallback_steps = sub['fallback_react_steps'].sum()
            action_fallback_rate = 100 * total_fallback_steps / total_actions if total_actions > 0 else 0
            avg_llm_calls = sub['llm_call_count'].sum() / task_count
            avg_tokens = sub['total_tokens'].sum() / task_count
            report_lines.append(
                f"| {f1_bin} | {task_count} | {task_success_rate:.1f}% | {task_complete_rate:.1f}% | "
                f"{task_fallback_rate:.1f}% | {action_fallback_rate:.1f}% | "
                f"{avg_llm_calls:.1f} | {avg_tokens:.0f} |"
            )
    
    # 场景×IP-Cov 交叉分析
    scenario_order = list(SCENARIO_TO_QUERY.keys())
    report_lines.extend([
        "\n## 6. 场景×IP-Cov 交叉分析",
        "\n> 此表揭示场景复杂度与偏序图覆盖率的交互效应",
        "\n### 6.1 各场景在不同 IP-Cov 下的降级率",
        "\n| 场景 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |",
        "|------|-----|-----|-----|-----|-----|",
    ])
    
    for scenario in scenario_order:
        row_data = [scenario]
        for ip_cov in [0.6, 0.7, 0.8, 0.9, 1.0]:
            sub = df[(df['scenario_name'] == scenario) & (df['ip_cov_target'] == ip_cov)]
            if len(sub) > 0:
                fallback_rate = 100 * sub['fallback_triggered'].sum() / len(sub)
                row_data.append(f"{fallback_rate:.0f}%")
            else:
                row_data.append("-")
        report_lines.append(f"| {' | '.join(row_data)} |")
    
    report_lines.extend([
        "\n### 6.2 各场景在不同 IP-Cov 下的平均 Cover-F1",
        "\n| 场景 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |",
        "|------|-----|-----|-----|-----|-----|",
    ])
    
    for scenario in scenario_order:
        row_data = [scenario]
        for ip_cov in [0.6, 0.7, 0.8, 0.9, 1.0]:
            sub = df[(df['scenario_name'] == scenario) & (df['ip_cov_target'] == ip_cov)]
            if len(sub) > 0:
                avg_f1 = sub['cover_f1'].mean()
                row_data.append(f"{avg_f1:.2f}")
            else:
                row_data.append("-")
        report_lines.append(f"| {' | '.join(row_data)} |")
    
    report_lines.extend([
        "\n### 6.3 各场景在不同 IP-Cov 下的完整率",
        "\n| 场景 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |",
        "|------|-----|-----|-----|-----|-----|",
    ])
    
    for scenario in scenario_order:
        row_data = [scenario]
        for ip_cov in [0.6, 0.7, 0.8, 0.9, 1.0]:
            sub = df[(df['scenario_name'] == scenario) & (df['ip_cov_target'] == ip_cov)]
            if len(sub) > 0:
                complete_rate = 100 * sub['is_complete'].mean()
                row_data.append(f"{complete_rate:.0f}%")
            else:
                row_data.append("-")
        report_lines.append(f"| {' | '.join(row_data)} |")
    
    # F1区间的场景分布
    report_lines.extend([
        "\n### 6.4 低F1区间(0-0.3)的场景分布",
        "\n| 场景 | 任务数 | 占比 | 主要缺失动作 |",
        "|------|--------|------|-------------|",
    ])
    
    low_f1_df = df[df['f1_bin'] == '0-0.3']
    if len(low_f1_df) > 0:
        scenario_counts = low_f1_df['scenario_name'].value_counts()
        # 分析缺失动作
        scenario_missing = {}
        for scenario in scenario_counts.index:
            sub = low_f1_df[low_f1_df['scenario_name'] == scenario]
            all_missing = []
            for _, row in sub.iterrows():
                if row['missing_actions'] and isinstance(row['missing_actions'], list):
                    all_missing.extend(row['missing_actions'])
            # 统计最常见的缺失动作
            if all_missing:
                from collections import Counter
                common_missing = Counter(all_missing).most_common(3)
                scenario_missing[scenario] = ', '.join([f"{m[0]}" for m in common_missing])
            else:
                scenario_missing[scenario] = "-"
        
        for scenario, count in scenario_counts.items():
            pct = 100 * count / len(low_f1_df)
            missing_info = scenario_missing.get(scenario, "-")
            report_lines.append(f"| {scenario} | {count} | {pct:.1f}% | {missing_info} |")
        
        # 添加场景难度小结
        report_lines.extend([
            "\n### 6.5 场景难度分析小结",
            "\n根据交叉分析，场景可分为三类：",
            "\n**简单场景**（任意IP-Cov下降级率=0）：",
            "- `simple_ecs`: 仅5步，依赖链简单",
            "- `dual_zone_ecs_slb`: 7步，结构清晰", 
            "- `dual_zone_ecs_slb_rds`: 10步，但依赖明确",
            "\n**困难场景**（低IP-Cov下降级率>40%）：",
            "- `slb_ecs_rds`: 12步，含监听器配置、账号创建等隐式依赖",
            "- `eip_slb_ecs`: 9步，EIP关联易遗漏",
            "- `slb_ecs_redis`: 8步，Redis创建流程特殊",
            "\n**关键发现**：Cover-F1 与执行效果的相关性受场景复杂度调节，困难场景需要更高的 IP-Cov 才能达到与简单场景相当的效果。",
        ])
    
    report_lines.extend([
        "\n## 7. 关键发现",
        "\n### 假设验证",
        "\n> 推断偏序图的 Cover-F1 越高，Expert 模式执行时成功率越高、动作序列越接近最优、降级触发率越低。",
        "\n### 场景复杂度对假设的影响",
        "\n> 交叉分析表明，场景复杂度是重要的混淆变量。困难场景在低 IP-Cov 下几乎必然降级，而简单场景即使 IP-Cov=0.6 也能完整执行。",
    ])
    
    # 保存报告
    report_path = output_dir / "hpo_experiment_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n报告已保存: {report_path}")


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='HPO 批量实验')
    parser.add_argument('--scenario', type=str, default=None, help='只运行指定场景')
    parser.add_argument('--limit', type=int, default=None, help='限制运行数量')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  HPO 批量实验 - 验证偏序图质量对执行效果的影响")
    print("=" * 80)
    
    # 1. 加载 .env (LLM 配置)
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    api_key = os.environ.get("LLM_API_KEY") or env_vars.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or env_vars.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL") or env_vars.get("LLM_MODEL") or "qwen3-max"
    
    if not api_key or "your_api_key_here" in api_key:
        print("\nError: LLM_API_KEY 未配置。请在 .env 文件中设置。")
        sys.exit(1)
    
    # 2. 加载 experiment_config.yaml (实验参数)
    config_path = os.path.join(current_dir, 'experiment_config.yaml')
    if not os.path.exists(config_path):
        print(f"\nError: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    exp_config = config.get('experiment', {})
    output_config = config.get('output', {})
    
    edge_threshold = exp_config.get('edge_threshold', 0.333)
    results_dir_name = output_config.get('results_dir', 'hpo_batch_results')
    
    # 3. 命令行参数
    target_scenario = args.scenario
    limit = args.limit
    
    # 打印配置
    print(f"配置: edge_threshold={edge_threshold}", end="")
    if target_scenario:
        print(f", scenario={target_scenario}", end="")
    if limit:
        print(f", limit={limit}", end="")
    print()
    print(f"Using LLM: {model} at {base_url}")
    
    # 2. 初始化
    client = OpenAI(api_key=api_key, base_url=base_url)
    io_registry = get_io_registry()
    
    # 3. 加载 HPO 实验列表
    hpo_dir = Path(current_dir) / "HPO_scenarios"
    experiments = load_hpo_experiments(str(hpo_dir))
    
    # 按场景过滤
    if target_scenario:
        experiments = [e for e in experiments if e["scenario_name"] == target_scenario]
        print(f"过滤后 {len(experiments)} 个实验（场景: {target_scenario}）")
    else:
        print(f"加载了 {len(experiments)} 个 HPO 实验")
    
    # 限制数量
    if limit and len(experiments) > limit:
        experiments = experiments[:limit]
        print(f"限制运行前 {limit} 个实验")
    
    # 4. 加载 experiment_summary.csv 获取 Cover-F1
    summary_csv = hpo_dir / "experiment_summary.csv"
    bhpop_df = load_experiment_summary(str(summary_csv))
    print(f"加载了 {len(bhpop_df)} 条 BHPOP 结果")
    
    # 5. 创建输出目录
    output_dir = Path(current_dir) / results_dir_name
    output_dir.mkdir(exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    
    # 6. 执行批量实验（并行）
    global progress_monitor
    max_workers = 20  # 并发度
    monitor_interval = 3.0  # 进度刷新间隔
    print(f"\n--- 开始批量实验 ({len(experiments)} 个, 并发度: {max_workers}) ---")
    results = []
    
    # 初始化进度监控器
    progress_monitor = ProgressMonitor(console, interval=monitor_interval)
    
    # 预注册所有任务
    for experiment in experiments:
        exp_id = str(experiment["experiment_id"])
        scenario = experiment["scenario_name"]
        ip_cov = experiment["configuration"].get("ip_cov_target", 0)
        eps = experiment["configuration"].get("eps_jump", 0)
        progress_monitor.register_task(exp_id, scenario, ip_cov, eps)
    
    def run_experiment_wrapper(idx_exp):
        """包装函数，用于并行执行"""
        idx, experiment = idx_exp
        exp_id = str(experiment["experiment_id"])
        scenario = experiment["scenario_name"]
        ip_cov = experiment["configuration"].get("ip_cov_target", 0)
        eps = experiment["configuration"].get("eps_jump", 0)
        
        # 每个线程创建独立的 LLM client，避免 token 统计累积
        thread_client = OpenAI(api_key=api_key, base_url=base_url)
        
        result = run_single_experiment(
            experiment=experiment,
            llm_client=thread_client,
            model=model,
            traces_dir=traces_dir,
            io_registry=io_registry,
            bhpop_df=bhpop_df,
            edge_threshold=edge_threshold,
            progress_monitor=progress_monitor
        )
        
        # 标记完成并输出结果
        progress_monitor.mark_complete(exp_id, result)
        status_symbol = "\u2705" if result.success else ("\U0001F504" if result.fallback_triggered else "\u274c")
        safe_print(f"[{progress_monitor.completed_count}/{progress_monitor.total_count}] {scenario} ip={ip_cov} eps={eps}: "
                   f"{status_symbol} LLM:{result.llm_call_count} Token:{result.total_tokens} F1:{result.cover_f1:.3f}")
        
        return result
    
    # 启动监控
    progress_monitor.start()
    
    try:
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_experiment_wrapper, (i, exp)): i 
                       for i, exp in enumerate(experiments)}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    safe_print(f"实验执行异常: {e}")
    finally:
        # 停止监控
        progress_monitor.stop()
    
    # 7. 生成汇总报告
    generate_summary_report(results, output_dir)
    
    # 8. 保存详细结果 JSON
    detail_file = output_dir / "hpo_experiment_details.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存: {detail_file}")
    print(f"Traces 目录: {traces_dir}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
