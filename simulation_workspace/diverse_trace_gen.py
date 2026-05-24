"""多样化 Trace 生成器
用于 CloudOps Agent 的偏序推导训练数据收集。

核心策略：
- 多模型采样：使用不同基础模型产生不同决策风格
- 多温度采样：同一模型不同温度产生路径变体
- 输出到 traces/ 目录，用于偏序推导训练

用法：
  python diverse_trace_gen.py                   # 默认运行
  python diverse_trace_gen.py --workers 8       # 指定并行数
  python diverse_trace_gen.py --models qwen-turbo,qwen-flash  # 只用特定模型

说明：
- 如需横向对比评测，请使用 explore_trace_gen_parallel.py
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openai import OpenAI
    from execution.cloudops_agent.agent import create_agent, AgentStatus, CloudOpsAgent
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("请确保已安装依赖并在项目根目录下运行。")
    sys.exit(1)


# ============================================================
# 配置区 - 可根据需要调整
# ============================================================

# 模型配置列表（每场景 10 条变体）
MODEL_CONFIG = [
    {"model": "deepseek-v3.2",     "temperature": 0.0},
    {"model": "glm-4.7",           "temperature": 0.3},
    {"model": "kimi-k2-thinking",  "temperature": 0.0},
    {"model": "qwen3-max",         "temperature": 0.0},
    {"model": "qwen-plus",         "temperature": 0.0},
    {"model": "qwen-plus",         "temperature": 0.3},
    {"model": "qwen-turbo",        "temperature": 0.3},
    {"model": "qwen-turbo",        "temperature": 0.5},
    {"model": "qwen-turbo",        "temperature": 0.7},
    {"model": "qwen-flash",        "temperature": 0.5},
]

# 各模型的 API 配置
# 所有模型统一使用 DashScope，只需配置 LLM_API_KEY 和 LLM_BASE_URL
DASHSCOPE_DEFAULT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ============================================================
# 工具函数
# ============================================================

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


def get_model_client(model_name: str, env_vars: Dict[str, str]) -> Optional[OpenAI]:
    """
    获取指定模型的 OpenAI 客户端
    
    所有模型统一使用 DashScope，只需配置 LLM_API_KEY 和 LLM_BASE_URL
    
    Args:
        model_name: 模型名称
        env_vars: 环境变量字典
        
    Returns:
        OpenAI 客户端，如果配置缺失则返回 None
    """
    # 获取 API Key
    api_key = (
        os.environ.get("LLM_API_KEY") or
        env_vars.get("LLM_API_KEY")
    )
    if not api_key:
        print(f"Warning: LLM_API_KEY 未配置")
        return None
    
    # 获取 Base URL
    base_url = (
        os.environ.get("LLM_BASE_URL") or
        env_vars.get("LLM_BASE_URL") or
        DASHSCOPE_DEFAULT_URL
    )
    
    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# 控制台管理（线程安全 + 进度条）
# ============================================================

class ConsoleManager:
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


console = ConsoleManager()


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    console.log(*args, **kwargs)


@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: int
    model: str
    temperature: float
    query: str
    agent: Optional['CloudOpsAgent'] = None
    status: str = "pending"  # pending, running, completed, failed
    api_count: int = 0
    last_api: str = ""
    start_time: float = 0.0


class ProgressMonitor:
    """进度监控器 - 定期输出各任务执行状态"""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.tasks: Dict[int, TaskProgress] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.total_tasks = 0
    
    def register_task(self, task_id: int, model: str, temperature: float, query: str) -> TaskProgress:
        """注册新任务"""
        with self._lock:
            progress = TaskProgress(task_id=task_id, model=model, temperature=temperature, query=query)
            self.tasks[task_id] = progress
            return progress
    
    def update_agent(self, task_id: int, agent: 'CloudOpsAgent'):
        """绑定 Agent 实例，用于读取进度"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].agent = agent
                self.tasks[task_id].status = "running"
                self.tasks[task_id].start_time = time.time()
    
    def mark_complete(self, task_id: int, success: bool):
        """标记任务完成"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = "completed" if success else "failed"
                self.tasks[task_id].agent = None
    
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
        console.clear_progress()
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.interval)
    
    def _print_progress(self):
        """输出当前进度"""
        with self._lock:
            running_tasks = []
            completed = sum(1 for p in self.tasks.values() if p.status in ("completed", "failed"))
            
            for task_id, prog in self.tasks.items():
                if prog.status == "running" and prog.agent:
                    trace = prog.agent.trace_store.get_current_trace()
                    if trace:
                        prog.api_count = len(trace.actions)
                        if trace.actions:
                            prog.last_api = trace.actions[-1].action_name
                    
                    elapsed = time.time() - prog.start_time
                    running_tasks.append((task_id, prog, elapsed))
            
            if running_tasks:
                lines = []
                lines.append("=" * 80)
                lines.append(f"📊 进度: {completed}/{self.total_tasks} | 运行中: {len(running_tasks)}")
                lines.append("-" * 80)
                for task_id, prog, elapsed in running_tasks:
                    model_temp = f"{prog.model} t={prog.temperature}"
                    last_api = prog.last_api or "(等待中)"
                    lines.append(f"  Task {task_id:3d} | {model_temp:25s} | APIs: {prog.api_count:2d} | {last_api:30s} | {elapsed:.0f}s")
                lines.append("=" * 80)
                console.update_progress("\n".join(lines) + "\n")
            else:
                console.update_progress("")


# 全局进度监控器
progress_monitor: Optional[ProgressMonitor] = None


# ============================================================
# 核心生成逻辑
# ============================================================

@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: int
    scenario_id: int
    model: str
    temperature: float
    trace_id: str
    success: bool
    error: str
    duration_s: float
    actions_count: int
    tokens: int


def run_single_variant(
    task_id: int,
    scenario_id: int,
    query: str,
    model_name: str,
    temperature: float,
    env_vars: Dict[str, str],
    traces_dir: str
) -> TaskResult:
    """
    执行单个变体任务
    """
    global progress_monitor
    start_time = time.time()
    
    safe_print(f"[Task {task_id:03d}] 开始: {model_name} t={temperature}")
    
    try:
        # 获取模型客户端
        client = get_model_client(model_name, env_vars)
        if not client:
            if progress_monitor:
                progress_monitor.mark_complete(task_id, False)
            return TaskResult(
                task_id=task_id,
                scenario_id=scenario_id,
                model=model_name,
                temperature=temperature,
                trace_id="",
                success=False,
                error=f"LLM_API_KEY 未配置",
                duration_s=time.time() - start_time,
                actions_count=0,
                tokens=0
            )
        
        # 创建 Agent
        agent = create_agent(preset="trace_collection", llm_client=client)
        
        # 配置模型和温度
        agent.react_planner.model_name = model_name
        agent.react_planner.temperature = temperature
        agent.config.llm.react_reasoning.model = model_name
        agent.config.llm.intent_parse.model = model_name
        
        # 配置输出路径
        agent.trace_store.output_path = Path(traces_dir)
        agent.config.switches.trace_output_path = traces_dir
        agent.config.switches.verbose = False
        
        # 设置自定义 trace 命名（包含模型和温度信息）
        agent.trace_store.set_task_index(scenario_id)
        agent.trace_store.set_custom_suffix(f"{model_name}_t{temperature}")
        
        # 设置生成配置元数据（用于后续分析）
        agent.trace_store.set_generation_config(
            generator="diverse_trace_gen",
            model=model_name,
            temperature=temperature,
            scenario_id=scenario_id,
            task_id=task_id,
            query=query
        )
        
        # 注册到进度监控器
        if progress_monitor:
            progress_monitor.update_agent(task_id, agent)
        
        # 执行任务
        result = agent.run(query)
        duration = time.time() - start_time
        
        success = result.status == AgentStatus.SUCCESS
        status_symbol = "✅" if success else "❌"
        safe_print(f"[Task {task_id:03d}] {status_symbol} {model_name} t={temperature} | "
                   f"APIs: {len(result.actions_executed)} | {duration:.1f}s")
        
        # 标记完成
        if progress_monitor:
            progress_monitor.mark_complete(task_id, success)
        
        return TaskResult(
            task_id=task_id,
            scenario_id=scenario_id,
            model=model_name,
            temperature=temperature,
            trace_id=result.trace_id,
            success=success,
            error=result.error or "",
            duration_s=duration,
            actions_count=len(result.actions_executed),
            tokens=result.total_tokens
        )
        
    except Exception as e:
        duration = time.time() - start_time
        safe_print(f"[Task {task_id:03d}] ❌ 异常: {model_name} t={temperature} | {e}")
        if progress_monitor:
            progress_monitor.mark_complete(task_id, False)
        return TaskResult(
            task_id=task_id,
            scenario_id=scenario_id,
            model=model_name,
            temperature=temperature,
            trace_id="",
            success=False,
            error=str(e),
            duration_s=duration,
            actions_count=0,
            tokens=0
        )


def generate_summary(results: List[TaskResult], traces_dir: str, duration_s: float) -> None:
    """生成执行汇总文件"""
    # 按模型统计
    by_model: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r.model not in by_model:
            by_model[r.model] = {"total": 0, "success": 0, "tokens": 0}
        by_model[r.model]["total"] += 1
        if r.success:
            by_model[r.model]["success"] += 1
        by_model[r.model]["tokens"] += r.tokens
    
    for model in by_model:
        total = by_model[model]["total"]
        success = by_model[model]["success"]
        by_model[model]["rate"] = success / total if total > 0 else 0
    
    # 按场景统计
    by_scenario: Dict[int, Dict[str, int]] = {}
    for r in results:
        if r.scenario_id not in by_scenario:
            by_scenario[r.scenario_id] = {"total": 0, "success": 0}
        by_scenario[r.scenario_id]["total"] += 1
        if r.success:
            by_scenario[r.scenario_id]["success"] += 1
    
    # 汇总
    total_count = len(results)
    success_count = sum(1 for r in results if r.success)
    total_tokens = sum(r.tokens for r in results)
    
    summary = {
        "run_time": datetime.now().isoformat(),
        "total_traces": total_count,
        "success_count": success_count,
        "fail_count": total_count - success_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
        "total_tokens": total_tokens,
        "total_duration_s": duration_s,
        "by_model": by_model,
        "by_scenario": {str(k): v for k, v in by_scenario.items()},
        "results": [
            {
                "task_id": r.task_id,
                "scenario_id": r.scenario_id,
                "model": r.model,
                "temperature": r.temperature,
                "trace_id": r.trace_id,
                "success": r.success,
                "error": r.error,
                "duration_s": r.duration_s,
                "actions_count": r.actions_count,
                "tokens": r.tokens
            }
            for r in results
        ]
    }
    
    summary_path = os.path.join(traces_dir, "generation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总已保存: {summary_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    global progress_monitor
    
    parser = argparse.ArgumentParser(description="多样化 Trace 生成器")
    parser.add_argument("--workers", type=int, default=8, help="并行数 (默认: 8)")
    parser.add_argument("--models", type=str, default=None, 
                        help="只使用指定模型，逗号分隔 (如: qwen-turbo,qwen-flash)")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="只运行指定场景，逗号分隔 (如: 1,2,3)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="进度监控间隔秒 (默认: 5.0)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🚀 多样化 Trace 生成器")
    print("  用途: 偏序推导训练数据收集")
    print("=" * 60)
    
    # 1. 加载环境变量
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    # 2. 过滤模型配置
    model_configs = MODEL_CONFIG
    if args.models:
        allowed_models = set(args.models.split(","))
        model_configs = [c for c in MODEL_CONFIG if c["model"] in allowed_models]
        print(f"已过滤模型: {allowed_models}")
    
    if not model_configs:
        print("Error: 没有可用的模型配置")
        sys.exit(1)
    
    # 3. 读取任务列表
    tasks_path = os.path.join(current_dir, 'trace_tasks.json')
    if not os.path.exists(tasks_path):
        print(f"Error: 任务文件未找到: {tasks_path}")
        sys.exit(1)
    
    with open(tasks_path, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    # 过滤场景
    if args.scenarios:
        allowed_scenarios = set(int(s) for s in args.scenarios.split(","))
        scenarios = [s for i, s in enumerate(scenarios, 1) if i in allowed_scenarios]
        print(f"已过滤场景: {allowed_scenarios}")
    
    print(f"\n配置摘要:")
    print(f"  场景数: {len(scenarios)}")
    print(f"  模型变体: {len(model_configs)}")
    print(f"  预计生成: {len(scenarios) * len(model_configs)} 条 trace")
    print(f"  并行数: {args.workers}")
    print(f"  监控间隔: {args.interval}s")
    
    # 4. 创建输出目录
    traces_dir = os.path.join(current_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    print(f"  输出目录: {traces_dir}")
    
    # 5. 构建任务列表
    all_tasks: List[Tuple[int, int, str, str, float]] = []
    task_id = 0
    for scenario_id, scenario in enumerate(scenarios, 1):
        query = scenario.get('query')
        for config in model_configs:
            task_id += 1
            all_tasks.append((task_id, scenario_id, query, config["model"], config["temperature"]))
    
    # 6. 初始化进度监控器
    progress_monitor = ProgressMonitor(interval=args.interval)
    progress_monitor.total_tasks = len(all_tasks)
    
    # 预注册所有任务
    for task_id, scenario_id, query, model, temp in all_tasks:
        progress_monitor.register_task(task_id, model, temp, query)
    
    print(f"\n开始执行 {len(all_tasks)} 个任务...\n")
    
    # 启动监控
    progress_monitor.start()
    
    # 7. 并行执行
    start_time = time.time()
    results: List[TaskResult] = []
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_variant,
                    task_id, scenario_id, query, model, temp,
                    env_vars, traces_dir
                ): task_id
                for task_id, scenario_id, query, model, temp in all_tasks
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task_id = futures[future]
                    safe_print(f"[Task {task_id}] 执行失败: {e}")
    finally:
        progress_monitor.stop()
    
    total_duration = time.time() - start_time
    
    # 8. 统计结果
    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count
    total_tokens = sum(r.tokens for r in results)
    
    print("\n" + "=" * 60)
    print("  执行完成")
    print("=" * 60)
    print(f"总任务数: {len(results)}")
    print(f"成功: {success_count} ✅")
    print(f"失败: {fail_count} ❌")
    print(f"成功率: {success_count/len(results)*100:.1f}%")
    print(f"总耗时: {total_duration:.1f}s")
    print(f"总 Tokens: {total_tokens}")
    print(f"Traces 保存至: {traces_dir}")
    
    # 按模型统计
    print("\n按模型统计:")
    model_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        if r.model not in model_stats:
            model_stats[r.model] = {"total": 0, "success": 0}
        model_stats[r.model]["total"] += 1
        if r.success:
            model_stats[r.model]["success"] += 1
    
    for model, stats in sorted(model_stats.items()):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {model}: {stats['success']}/{stats['total']} ({rate:.0f}%)")
    
    # 打印失败任务
    if fail_count > 0:
        print("\n失败任务:")
        for r in results:
            if not r.success:
                print(f"  - Task {r.task_id} ({r.model} t={r.temperature}): {r.error[:50]}")
    
    # 9. 生成汇总文件
    generate_summary(results, traces_dir, total_duration)


if __name__ == "__main__":
    main()
