"""
模式选择器

根据意图和偏序图覆盖率，自动选择执行模式
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set
from enum import Enum

from execution.cloudops_agent.config import ExecutionMode, AgentConfig
from execution.cloudops_agent.controller.intent_parser import ParsedIntent, IntentType


@dataclass
class ModeDecision:
    """模式决策结果"""
    mode: ExecutionMode
    reason: str
    coverage: float = 0.0
    available_actions: List[str] = None
    missing_actions: List[str] = None
    
    def __post_init__(self):
        if self.available_actions is None:
            self.available_actions = []
        if self.missing_actions is None:
            self.missing_actions = []


class ModeSelector:
    """
    模式选择器
    
    根据以下因素决定执行模式：
    1. 配置的强制模式
    2. 偏序图是否可用
    3. 意图与偏序图的覆盖率
    """
    
    # 意图类型对应的标准动作序列 - 对应6个仿真场景
    INTENT_ACTION_TEMPLATES = {
        # 场景1: 简单 ECS 创建
        IntentType.SIMPLE_ECS: [
            "CreateVpc",
            "CreateVSwitch",
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",
        ],
        # 场景2: SLB + ECS + RDS
        IntentType.SLB_ECS_RDS: [
            "CreateVpc",
            "CreateVSwitch",
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",
            "CreateLoadBalancer",
            "AddBackendServers",
            "CreateDBInstance",
        ],
        # 场景3: SLB + ECS + Redis
        IntentType.SLB_ECS_REDIS: [
            "CreateVpc",
            "CreateVSwitch",
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",
            "CreateLoadBalancer",
            "AddBackendServers",
            "CreateInstance",  # Redis CreateInstance
        ],
        # 场景4: EIP + SLB + ECS
        IntentType.EIP_SLB_ECS: [
            "CreateVpc",
            "CreateVSwitch",
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",
            "CreateLoadBalancer",
            "AddBackendServers",
            "AllocateEipAddress",
            "AssociateEipAddress",
        ],
        # 场景5: 双可用区 ECS×2 + SLB
        IntentType.DUAL_ZONE_ECS_SLB: [
            "CreateVpc",
            "CreateVSwitch",           # 可用区A VSwitch
            "CreateVSwitch",           # 可用区B VSwitch
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",             # 可用区A ECS
            "RunInstances",             # 可用区B ECS
            "CreateLoadBalancer",
            "AddBackendServers",
        ],
        # 场景6: 双可用区 ECS×2 + SLB + RDS 主备
        IntentType.DUAL_ZONE_ECS_SLB_RDS: [
            "CreateVpc",
            "CreateVSwitch",           # 可用区A VSwitch
            "CreateVSwitch",           # 可用区B VSwitch
            "CreateSecurityGroup",
            "AuthorizeSecurityGroup",
            "RunInstances",             # 可用区A ECS
            "RunInstances",             # 可用区B ECS
            "CreateLoadBalancer",
            "AddBackendServers",
            "CreateDBInstance",         # RDS 主实例 (高可用版)
        ],
    }
    
    def __init__(self, config: AgentConfig, poset_graph: Optional[Dict] = None):
        """
        初始化选择器
        
        Args:
            config: Agent 配置
            poset_graph: 偏序图数据（如果已加载）
        """
        self.config = config
        self.poset_graph = poset_graph
        self._poset_actions: Set[str] = set()
        
        if poset_graph:
            self._extract_poset_actions()
    
    def _extract_poset_actions(self) -> None:
        """从偏序图中提取所有 action"""
        if not self.poset_graph:
            return
        
        # 假设偏序图格式为 {action: {predecessors: [...], successors: [...]}}
        # 或者是邻接矩阵格式
        if isinstance(self.poset_graph, dict):
            if 'nodes' in self.poset_graph:
                self._poset_actions = set(self.poset_graph['nodes'])
            elif 'actions' in self.poset_graph:
                self._poset_actions = set(self.poset_graph['actions'])
            else:
                # 假设 key 就是 action
                self._poset_actions = set(self.poset_graph.keys())
    
    def set_poset_graph(self, poset_graph: Dict) -> None:
        """设置偏序图"""
        self.poset_graph = poset_graph
        self._extract_poset_actions()
    
    def select_mode(self, intent: ParsedIntent) -> ModeDecision:
        """
        选择执行模式
        
        Args:
            intent: 解析后的意图
            
        Returns:
            ModeDecision
        """
        # 1. 检查是否强制使用某种模式
        if self.config.execution_mode == ExecutionMode.EXPLORE:
            return ModeDecision(
                mode=ExecutionMode.EXPLORE,
                reason="配置强制使用 Explore 模式"
            )
        
        # 2. 检查偏序图是否启用和可用
        if not self.config.switches.poset_enabled:
            return ModeDecision(
                mode=ExecutionMode.EXPLORE,
                reason="偏序图未启用"
            )
        
        if not self.poset_graph:
            return ModeDecision(
                mode=ExecutionMode.EXPLORE,
                reason="偏序图未加载"
            )
        
        # 3. 计算覆盖率
        required_actions = self._get_required_actions(intent)
        coverage, available, missing = self._compute_coverage(required_actions)
        
        # 4. 根据覆盖率选择模式
        if coverage >= 0.95:
            mode = ExecutionMode.EXPERT
            reason = f"偏序图覆盖率 {coverage:.0%}，使用 Expert 模式"
        elif coverage >= 0.5:
            if self.config.execution_mode == ExecutionMode.EXPERT:
                # 配置要求 Expert，但覆盖率不足
                if self.config.switches.poset_strict_mode:
                    mode = ExecutionMode.EXPERT
                    reason = f"严格模式，强制使用 Expert（覆盖率 {coverage:.0%}）"
                else:
                    mode = ExecutionMode.HYBRID
                    reason = f"覆盖率 {coverage:.0%}，降级到 Hybrid 模式"
            else:
                mode = ExecutionMode.HYBRID
                reason = f"偏序图覆盖率 {coverage:.0%}，使用 Hybrid 模式"
        else:
            if self.config.switches.poset_strict_mode:
                mode = ExecutionMode.EXPERT
                reason = f"严格模式，强制使用 Expert（覆盖率 {coverage:.0%}，可能失败）"
            else:
                mode = ExecutionMode.EXPLORE
                reason = f"偏序图覆盖率 {coverage:.0%} 过低，使用 Explore 模式"
        
        return ModeDecision(
            mode=mode,
            reason=reason,
            coverage=coverage,
            available_actions=available,
            missing_actions=missing
        )
    
    def _get_required_actions(self, intent: ParsedIntent) -> List[str]:
        """获取意图所需的动作列表"""
        template = self.INTENT_ACTION_TEMPLATES.get(intent.intent_type, [])
        actions = list(template)
        
        # 6个场景的标志位已由 intent_type 自动推断，此处无需额外调整
        # 保留扩展接口以便未来自定义
        
        return actions
    
    def _compute_coverage(self, required_actions: List[str]) -> tuple[float, List[str], List[str]]:
        """
        计算偏序图对所需动作的覆盖率
        
        Returns:
            (coverage, available_actions, missing_actions)
        """
        if not required_actions:
            return 1.0, [], []
        
        available = [a for a in required_actions if a in self._poset_actions]
        missing = [a for a in required_actions if a not in self._poset_actions]
        
        coverage = len(available) / len(required_actions)
        return coverage, available, missing
    
    def should_fallback(self, error: str, retry_count: int) -> bool:
        """
        判断是否应该降级到 ReAct
        
        简单逻辑：API 报错重试 3 次后降级
        
        Args:
            error: 错误信息
            retry_count: 已重试次数
            
        Returns:
            是否应该降级
        """
        # 严格模式不降级
        if self.config.switches.poset_strict_mode:
            return False
        
        # 不允许降级
        if not self.config.switches.poset_fallback_enabled:
            return False
        
        # 有错误且重试次数达到阈值，降级
        if error and retry_count >= self.config.max_retries:
            return True
        
        return False
