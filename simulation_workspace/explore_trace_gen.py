"""
批量 Trace 生成脚本
用于 CloudOps Agent 的数据收集阶段。

功能：
1. 读取 .env 配置 (LLM Key)
2. 读取 trace_tasks.json 任务列表
3. 调用 CloudOpsAgent (Explore模式) 批量执行任务
4. 自动生成 Trace 文件到 ./traces/ 目录
"""

import sys
import os
import json
import time
from typing import Dict
from pathlib import Path

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openai import OpenAI
    from execution.cloudops_agent.agent import create_agent, AgentStatus
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("请确保已安装依赖并在项目根目录下运行。")
    sys.exit(1)

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

def main():
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
        
    print(f"Using LLM: {model} at {base_url}")
    
    # 2. 初始化 LLM Client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 3. 读取任务列表
    tasks_path = os.path.join(current_dir, 'trace_tasks.json')
    if not os.path.exists(tasks_path):
        print(f"Error: 任务文件未找到: {tasks_path}")
        sys.exit(1)
        
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    print(f"Loaded {len(tasks)} task definitions.")
    
    # 4. 执行任务
    total_runs = sum(t.get('repeat', 1) for t in tasks)
    current_run = 0
    
    # 创建 traces 目录
    traces_dir = os.path.join(current_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    
    for task_def in tasks:
        query = task_def.get('query')
        repeat = task_def.get('repeat', 1)
        
        print(f"\nProcessing Task: {query}")
        print(f"Repeats: {repeat}")
        
        for i in range(repeat):
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Starting run {i+1}...")
            
            # 每次重新创建 Agent 以确保状态隔离
            # 使用 'trace_collection' 预设：关闭偏序图，开启 Trace，使用 ReAct
            agent = create_agent(preset="trace_collection", llm_client=client)
            
            # 更新 LLM 模型配置
            agent.react_planner.model_name = model
            agent.config.llm.react_reasoning.model = model
            agent.config.llm.intent_parse.model = model
            
            # 覆盖默认的 trace 路径，确保在当前目录下
            agent.trace_store.output_path = Path(traces_dir)
            agent.config.switches.trace_output_path = traces_dir
            agent.config.switches.verbose = True
            
            # 设置任务索引，用于 trace 命名（便于横向比较）
            agent.trace_store.set_task_index(current_run)
            
            try:
                result = agent.run(query)
                
                status_symbol = "✅" if result.status == AgentStatus.SUCCESS else "❌"
                print(f"Run finished: {status_symbol} {result.status.value}")
                print(f"Trace ID: {result.trace_id}")
                print(f"Duration: {result.duration_ms:.2f}ms")
                if result.error:
                    print(f"Error: {result.error}")
                    
            except Exception as e:
                print(f"Run failed with exception: {e}")
                
            # 简单的间隔，避免速率限制
            time.sleep(1)

    print("\nBatch generation completed.")
    print(f"Traces saved to: {traces_dir}")

if __name__ == "__main__":
    main()
