"""并行 Trace 生成脚本 (Compare 模式)
用于 CloudOps Agent 的横向对比评测。

功能：
- 输出到 explore_traces/ 目录，与 expert/hybrid 模式进行横向对比
- 固定使用 .env 中配置的模型，保持与其他模式一致
- 用于比较：耗时、正确率、token消耗等指标

用法：
  python explore_trace_gen_parallel.py [并行数] [监控间隔]

说明：
- 如需生成多样化 trace 用于偏序推导训练，请使用 diverse_trace_gen.py
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

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
            # Move cursor up and clear line
            for _ in range(self._last_lines):
                sys.stdout.write("\033[1A") # Move up 1 line
                sys.stdout.write("\033[2K") # Clear entire line
            sys.stdout.write("\r") # Move to start of line
            self._last_lines = 0

    def _write_progress_buffer(self):
        if self._progress_buffer:
            sys.stdout.write(self._progress_buffer)
            # Calculate lines
            self._last_lines = self._progress_buffer.count('\n')
            if not self._progress_buffer.endswith('\n'):
                self._last_lines += 1
                sys.stdout.write('\n')

# 全局控制台管理器
console = ConsoleManager()

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    console.log(*args, **kwargs)


@dataclass
class TaskProgress:
    """任务进度信息"""
    task_id: int
    query: str
    agent: Optional[CloudOpsAgent] = None
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
    
    def register_task(self, task_id: int, query: str) -> TaskProgress:
        """注册新任务"""
        with self._lock:
            progress = TaskProgress(task_id=task_id, query=query)
            self.tasks[task_id] = progress
            return progress
    
    def update_agent(self, task_id: int, agent: CloudOpsAgent):
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
                self.tasks[task_id].agent = None  # 释放引用
    
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
        """监控循环 - 定期输出进度"""
        while not self._stop_event.is_set():
            self._print_progress()
            self._stop_event.wait(self.interval)
    
    def _print_progress(self):
        """输出当前进度"""
        with self._lock:
            running_tasks = []
            for task_id, prog in self.tasks.items():
                if prog.status == "running" and prog.agent:
                    # 从 agent 的 trace_store 读取进度
                    trace = prog.agent.trace_store.get_current_trace()
                    if trace:
                        prog.api_count = len(trace.actions)
                        if trace.actions:
                            prog.last_api = trace.actions[-1].action_name
                    
                    elapsed = time.time() - prog.start_time
                    running_tasks.append((task_id, prog, elapsed))
            
            if running_tasks:
                lines = []
                lines.append("=" * 70)
                lines.append(f"📊 进度报告 ({len(running_tasks)} 个任务运行中)")
                lines.append("-" * 70)
                for task_id, prog, elapsed in running_tasks:
                    query_short = prog.query[:35] + "..." if len(prog.query) > 35 else prog.query
                    last_api = prog.last_api or "(等待中)"
                    lines.append(f"  Task {task_id:2d} | APIs: {prog.api_count:3d} | 上一API: {last_api:30s} | {elapsed:.1f}s")
                    lines.append(f"           └─ {query_short}")
                lines.append("=" * 70)
                console.update_progress("\n".join(lines) + "\n")
            else:
                console.update_progress("")


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


# 全局进度监控器
progress_monitor: Optional[ProgressMonitor] = None


def run_single_task(
    task_id: int,
    query: str,
    run_index: int,
    api_key: str,
    base_url: str,
    model: str,
    traces_dir: str
) -> Tuple[int, str, bool, str]:
    """
    执行单个任务
    
    Returns:
        Tuple[task_id, trace_id, success, error_msg]
    """
    global progress_monitor
    
    safe_print(f"[Task {task_id}] 开始执行: {query[:50]}...")
    
    try:
        # 每个线程创建独立的 OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 每次重新创建 Agent 以确保状态隔离
        agent = create_agent(preset="trace_collection", llm_client=client)
        
        # 更新 LLM 模型配置
        agent.react_planner.model_name = model
        agent.config.llm.react_reasoning.model = model
        agent.config.llm.intent_parse.model = model
        
        # 覆盖默认的 trace 路径
        agent.trace_store.output_path = Path(traces_dir)
        agent.config.switches.trace_output_path = traces_dir
        agent.config.switches.verbose = False  # 并行时关闭详细输出避免混乱
        
        # 设置任务索引，用于 trace 命名
        agent.trace_store.set_task_index(task_id)
        
        # 注册到进度监控器
        if progress_monitor:
            progress_monitor.update_agent(task_id, agent)
        
        result = agent.run(query)
        
        success = result.status == AgentStatus.SUCCESS
        status_symbol = "✅" if success else "❌"
        safe_print(f"[Task {task_id}] {status_symbol} 完成 | APIs: {len(result.actions_executed)} | Trace: {result.trace_id}")
        
        # 标记完成
        if progress_monitor:
            progress_monitor.mark_complete(task_id, success)
        
        return (task_id, result.trace_id, success, result.error or "")
        
    except Exception as e:
        safe_print(f"[Task {task_id}] ❌ 异常: {e}")
        if progress_monitor:
            progress_monitor.mark_complete(task_id, False)
        return (task_id, "", False, str(e))


def main():
    global progress_monitor
    
    # 默认参数
    max_workers = 6
    monitor_interval = 5.0  # 秒
    
    # 解析命令行参数: python explore_trace_gen_parallel.py [并行数] [监控间隔秒]
    if len(sys.argv) > 1:
        try:
            max_workers = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python explore_trace_gen_parallel.py [并行数] [监控间隔]")
            print(f"  并行数: 同时执行的任务数，默认 6")
            print(f"  监控间隔: 进度刷新间隔秒数，默认 5.0")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            monitor_interval = float(sys.argv[2])
        except ValueError:
            print(f"Warning: 无效的监控间隔参数 '{sys.argv[2]}'，使用默认值 {monitor_interval}s")
    
    # 固定输出到 explore_traces 目录（用于横向对比）
    traces_dir_name = "explore_traces"
    mode_desc = "Compare 模式 (与 expert/hybrid 横向对比)"
    
    # 1. 加载环境变量
    env_path = os.path.join(current_dir, '.env')
    env_vars = load_env(env_path)
    
    # 优先使用环境变量，其次使用 .env 文件
    api_key = os.environ.get("LLM_API_KEY") or env_vars.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL") or env_vars.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL") or env_vars.get("LLM_MODEL") or "qwen3-max"
    
    if not api_key or "your_api_key_here" in api_key:
        print("Error: LLM_API_KEY 未配置。请在 .env 文件中设置或通过环境变量传入。")
        sys.exit(1)
        
    print(f"运行模式: {mode_desc}")
    print(f"Using LLM: {model} at {base_url}")
    print(f"并行度: {max_workers}")
    print(f"进度监控间隔: {monitor_interval}s")
    
    # 2. 读取任务列表
    tasks_path = os.path.join(current_dir, 'trace_tasks.json')
    if not os.path.exists(tasks_path):
        print(f"Error: 任务文件未找到: {tasks_path}")
        sys.exit(1)
        
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"Loaded {len(tasks)} task definitions.")
    
    # 3. 展开所有任务（处理 repeat）
    all_runs: List[Tuple[int, str, int]] = []  # (task_id, query, run_index)
    task_id = 0
    for task_def in tasks:
        query = task_def.get('query')
        repeat = task_def.get('repeat', 1)
        
        for run_idx in range(repeat):
            task_id += 1
            all_runs.append((task_id, query, run_idx + 1))
    
    total_runs = len(all_runs)
    print(f"总共 {total_runs} 个任务实例待执行\n")
    
    # 创建输出目录
    traces_dir = os.path.join(current_dir, traces_dir_name)
    os.makedirs(traces_dir, exist_ok=True)
    
    # 初始化进度监控器
    progress_monitor = ProgressMonitor(interval=monitor_interval)
    
    # 预注册所有任务
    for task_id, query, run_idx in all_runs:
        progress_monitor.register_task(task_id, query)
    
    # 启动监控
    progress_monitor.start()
    
    # 4. 并行执行任务
    start_time = time.time()
    results = []
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    run_single_task,
                    task_id,
                    query,
                    run_idx,
                    api_key,
                    base_url,
                    model,
                    traces_dir
                ): (task_id, query)
                for task_id, query, run_idx in all_runs
            }
        
            # 收集结果
            for future in as_completed(futures):
                task_id, query = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    safe_print(f"[Task {task_id}] 执行失败: {e}")
                    results.append((task_id, "", False, str(e)))
    finally:
        # 停止监控
        progress_monitor.stop()
    
    # 5. 统计结果
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r[2])
    fail_count = len(results) - success_count
    
    print("\n" + "="*60)
    print("批量生成完成")
    print("="*60)
    print(f"总任务数: {total_runs}")
    print(f"成功: {success_count} ✅")
    print(f"失败: {fail_count} ❌")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"平均每任务: {elapsed/total_runs:.2f}s")
    print(f"Traces 保存至: {traces_dir}")
    
    # 打印失败的任务
    if fail_count > 0:
        print("\n失败任务:")
        for task_id, trace_id, success, error in results:
            if not success:
                print(f"  - Task {task_id}: {error}")


if __name__ == "__main__":
    main()
