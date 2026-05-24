"""
CloudOps ReAct Agent - System 1 + System 2 混合架构

支持三种运行模式：
- Expert Mode: 偏序图驱动，零 LLM 调用
- Hybrid Mode: 偏序图 + 动态降级
- Explore Mode: 纯 ReAct + RAG
"""

from execution.cloudops_agent.config import AgentConfig, ExecutionMode

__version__ = "0.1.0"
__all__ = ["AgentConfig", "ExecutionMode"]
