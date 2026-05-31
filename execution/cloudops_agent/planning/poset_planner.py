"""
偏序图规划器

基于偏序图进行任务规划和调度，实现 System 1 快速执行
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
import json
from pathlib import Path

from execution.cloudops_agent.knowledge.io_registry import IORegistry, get_io_registry
from execution.cloudops_agent.memory.blackboard import Blackboard

# 延迟导入避免循环依赖
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from execution.cloudops_agent.controller.intent_parser import IntentType


class ActionState(Enum):
    """动作状态"""
    PENDING = "pending"
    READY = "ready"      # IO 就绪，可执行
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # IO 未就绪


@dataclass
class ActionNode:
    """偏序图中的动作节点"""
    action: str
    predecessors: Set[str] = field(default_factory=set)  # 前置动作
    successors: Set[str] = field(default_factory=set)    # 后置动作
    state: ActionState = ActionState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    layer: int = -1  # 拓扑层级


class PosetGraph:
    """
    偏序图数据结构
    
    支持从 JSON 加载和手动构建
    
    边权重 (edge_weights) 用于存储后验边概率，支持基于置信度的执行控制。
    """
    
    # TODO: 后验边概率阈值 - 用于过滤低置信度边
    # 当边概率 < threshold 时，该边不参与执行（视为不可靠依赖）
    # 论文中建议阈值为 0.8，可根据场景调整
    EDGE_CONFIDENCE_THRESHOLD: float = 0.8
    
    def __init__(self):
        self.nodes: Dict[str, ActionNode] = {}
        self._layers: List[List[str]] = []
        # edge_weights: 存储边的后验概率 P(a → b)
        # 来源：BHPOP 贝叶斯推断的后验分布
        self.edge_weights: Dict[Tuple[str, str], float] = {}
    
    def add_action(self, action: str, predecessors: Optional[List[str]] = None) -> None:
        """添加动作节点"""
        if action not in self.nodes:
            self.nodes[action] = ActionNode(action=action)
        
        if predecessors:
            for pred in predecessors:
                self.add_edge(pred, action)
    
    def add_edge(self, from_action: str, to_action: str, weight: float = 1.0) -> None:
        """
        添加依赖边
        
        Args:
            from_action: 前置动作
            to_action: 后置动作
            weight: 边权重/概率 (0.0 - 1.0)
        """
        if from_action not in self.nodes:
            self.nodes[from_action] = ActionNode(action=from_action)
        if to_action not in self.nodes:
            self.nodes[to_action] = ActionNode(action=to_action)
        
        self.nodes[from_action].successors.add(to_action)
        self.nodes[to_action].predecessors.add(from_action)
        self.edge_weights[(from_action, to_action)] = weight
    
    @classmethod
    def load_from_hpo_matrix(cls, file_path: str, assessor_id: str = "0") -> 'PosetGraph':
        """
        从 HPO 挖掘结果 (Adjacency Matrix JSON) 加载偏序图
        
        Structure expected:
        {
            "assessors": {
                "0": {
                    "labels": ["Action1", "Action2", ...],
                    "h_mode": [[0, 1, ...], ...],  # Binary Adjacency Matrix
                    "prob_gt": [[0.1, 0.9, ...], ...] # Probability Matrix
                }
            }
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        if "assessors" not in data or assessor_id not in data["assessors"]:
            raise ValueError(f"Invalid HPO data structure or missing assessor {assessor_id}")
            
        assessor_data = data["assessors"][assessor_id]
        labels = assessor_data.get("labels", [])
        h_mode = assessor_data.get("h_mode", [])
        prob_gt = assessor_data.get("prob_gt", [])
        
        if not labels or not h_mode:
            raise ValueError("Missing 'labels' or 'h_mode' in assessor data")
            
        graph = cls()
        
        # 1. Add all nodes first
        for label in labels:
            graph.add_action(label)
            
        # 2. Add edges based on h_mode (adjacency matrix)
        num_nodes = len(labels)
        for i in range(num_nodes):
            for j in range(num_nodes):
                # h_mode[i][j] == 1 implies edge from labels[i] to labels[j]
                if h_mode[i][j] == 1:
                    source = labels[i]
                    target = labels[j]
                    
                    # Get probability if available
                    weight = 1.0
                    if prob_gt and len(prob_gt) > i and len(prob_gt[i]) > j:
                        weight = prob_gt[i][j]
                        
                    graph.add_edge(source, target, weight=weight)
                    
        return graph

    @classmethod
    def load_from_hpo_posterior(cls, file_path: str, edge_threshold: float) -> 'PosetGraph':
        """
        从 HPO 后验推断结果加载偏序图
        
        Structure expected (HPO_scenarios 格式):
        {
            "scenario": {
                "task_ids": ["Action1", "Action2", ...],
                "n_tasks": 7
            },
            "posterior": {
                "avg_H": [[0.0, 0.004, ...], ...],  # 后验边概率矩阵
                "std_H": [[0.0, 0.063, ...], ...]   # 后验边概率标准差
            }
        }
        
        Args:
            file_path: HPO 后验结果文件路径
            edge_threshold: 边概率阈值，avg_H[i][j] >= threshold 时添加边
        
        Returns:
            PosetGraph 实例
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
        
        # 验证数据结构
        if "scenario" not in data or "posterior" not in data:
            raise ValueError("Invalid HPO posterior format: missing 'scenario' or 'posterior'")
            
        scenario = data["scenario"]
        posterior = data["posterior"]
        
        task_ids = scenario.get("task_ids", [])
        avg_H = posterior.get("avg_H", [])
        
        if not task_ids or not avg_H:
            raise ValueError("Missing 'task_ids' or 'avg_H' in HPO posterior data")
        
        num_tasks = len(task_ids)
        if len(avg_H) != num_tasks:
            raise ValueError(f"Matrix size mismatch: {len(avg_H)} != {num_tasks}")
            
        graph = cls()
        
        # 1. 添加所有节点
        for task_id in task_ids:
            graph.add_action(task_id)
        
        # 2. 根据后验概率矩阵添加边
        # avg_H[i][j] 表示从 task_ids[i] 到 task_ids[j] 的边概率
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue  # 跳过自环
                    
                prob = avg_H[i][j]
                if prob >= edge_threshold:
                    source = task_ids[i]
                    target = task_ids[j]
                    graph.add_edge(source, target, weight=prob)
        
        graph.compute_layers()
        return graph

    def compute_layers(self) -> List[List[str]]:
        """
        计算拓扑层级
        
        Layer 0: 入度为 0 的节点
        Layer N: 所有前置都在 Layer < N 的节点
        """
        self._layers = []
        remaining = set(self.nodes.keys())
        completed = set()
        
        while remaining:
            # 找出所有前置已完成的节点
            ready = []
            for action in remaining:
                node = self.nodes[action]
                if node.predecessors.issubset(completed):
                    ready.append(action)
            
            if not ready:
                # 有环或死锁
                break
            
            # 设置层级
            layer_idx = len(self._layers)
            for action in ready:
                self.nodes[action].layer = layer_idx
                remaining.remove(action)
                completed.add(action)
            
            self._layers.append(ready)
        
        return self._layers
    
    def get_ready_actions(self, completed_actions: Set[str], 
                          blackboard: Blackboard,
                          io_registry: IORegistry,
                          use_confidence_filter: bool = False,
                          io_guard_enabled: bool = True) -> List[str]:
        """
        获取当前可执行的动作 (Frontier 计算)
            
        条件：
        1. 所有前置动作已完成
        2. IO 参数在 blackboard 中就绪（当 io_guard_enabled=True 时）
        3. 状态为 PENDING 或 BLOCKED（避免重复返回）
            
        Args:
            completed_actions: 已完成的动作集合
            blackboard: 参数黑板
            io_registry: IO 注册表
            use_confidence_filter: 是否启用边置信度过滤
                                   启用后，低于 EDGE_CONFIDENCE_THRESHOLD 的边将被忽略
            io_guard_enabled: 是否启用 IO Guard（预防性参数检查）
                              False 时只依赖偏序图前置依赖，让 API 执行时自己报错
        """
        ready = []
        for action, node in self.nodes.items():
            # 跳过已完成、执行中、已就绪的动作
            if node.state in (ActionState.COMPLETED, ActionState.EXECUTING, ActionState.READY):
                continue
                
            # 检查前置动作
            # TODO: 基于边置信度的前置过滤
            # 当 use_confidence_filter=True 时，只考虑高置信度边 (prob >= threshold)
            # 低置信度边视为“可选依赖”，不阻塞后续动作执行
            effective_predecessors = node.predecessors
            if use_confidence_filter:
                effective_predecessors = self._filter_by_confidence(action, node.predecessors)
                
            if not effective_predecessors.issubset(completed_actions):
                continue
                
            # IO Guard：可选的预防性参数检查
            if io_guard_enabled:
                is_ready, missing = io_registry.check_inputs_ready(action, blackboard)
                if is_ready:
                    ready.append(action)
                    node.state = ActionState.READY
                else:
                    node.state = ActionState.BLOCKED
            else:
                # 禁用 IO Guard：只依赖偏序图，让 API 执行时自己报错触发降级
                ready.append(action)
                node.state = ActionState.READY
            
        return ready
    
    def _filter_by_confidence(self, action: str, predecessors: Set[str]) -> Set[str]:
        """
        基于边置信度过滤前置依赖
        
        只保留置信度 >= EDGE_CONFIDENCE_THRESHOLD 的边
        这允许在后验不确定时跳过低置信度依赖，提高执行鲁棒性
        
        Args:
            action: 当前动作
            predecessors: 原始前置动作集合
            
        Returns:
            过滤后的高置信度前置动作集合
        """
        filtered = set()
        for pred in predecessors:
            edge_prob = self.edge_weights.get((pred, action), 1.0)  # 默认为确定性边
            if edge_prob >= self.EDGE_CONFIDENCE_THRESHOLD:
                filtered.add(pred)
            # TODO: 可记录被过滤的低置信度边，用于调试或降级决策
        return filtered
    
    def mark_completed(self, action: str, result: Dict[str, Any]) -> None:
        """标记动作完成"""
        if action in self.nodes:
            self.nodes[action].state = ActionState.COMPLETED
            self.nodes[action].result = result
    
    def mark_failed(self, action: str, error: str) -> None:
        """标记动作失败"""
        if action in self.nodes:
            self.nodes[action].state = ActionState.FAILED
            self.nodes[action].error = error
    
    def get_completed_actions(self) -> Set[str]:
        """获取已完成的动作"""
        return {a for a, n in self.nodes.items() if n.state == ActionState.COMPLETED}
    
    def is_complete(self) -> bool:
        """检查是否所有动作都已完成"""
        return all(n.state == ActionState.COMPLETED for n in self.nodes.values())
    
    def has_failed(self) -> bool:
        """检查是否有失败的动作"""
        return any(n.state == ActionState.FAILED for n in self.nodes.values())
    
    def has_blocked_by_io(self) -> bool:
        """检查是否有动作被 IO 阻塞（前置完成但参数未就绪）"""
        return any(n.state == ActionState.BLOCKED for n in self.nodes.values())
    
    def get_blocked_actions(self) -> List[str]:
        """获取被阻塞的动作列表"""
        return [a for a, n in self.nodes.items() if n.state == ActionState.BLOCKED]
    
    def get_remaining_subgraph(self) -> Dict[str, List[str]]:
        """获取剩余未完成的子图"""
        remaining = {}
        for action, node in self.nodes.items():
            if node.state not in (ActionState.COMPLETED, ActionState.EXECUTING):
                remaining[action] = list(node.predecessors - self.get_completed_actions())
        return remaining
    
    @classmethod
    def from_json(cls, path: str) -> "PosetGraph":
        """从 JSON 文件加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PosetGraph":
        """从字典构建"""
        graph = cls()
        
        # 支持多种格式
        if 'edges' in data:
            # 边列表格式: {"edges": [["A", "B"], ...]}
            for from_action, to_action in data['edges']:
                graph.add_edge(from_action, to_action)
        elif 'adjacency' in data:
            # 邻接表格式: {"adjacency": {"A": ["B", "C"], ...}}
            for from_action, to_actions in data['adjacency'].items():
                graph.add_action(from_action)
                for to_action in to_actions:
                    graph.add_edge(from_action, to_action)
        else:
            # 默认格式: {"A": {"predecessors": [...], "successors": [...]}, ...}
            for action, spec in data.items():
                if isinstance(spec, dict):
                    preds = spec.get('predecessors', [])
                    graph.add_action(action, preds)
        
        graph.compute_layers()
        return graph

    @classmethod
    def load_from_manual_config(cls, file_path: str) -> 'PosetGraph':
        """
        从简化的手动配置 JSON 加载偏序图
        
        Format:
        {
          "name": "Scenario Name",
          "description": "...",
          "edges": [
            ["FromAction", "ToAction"],
            ["FromAction", "ToAction", 0.9]  # Optional probability
          ]
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        graph = cls()
        
        # Add edges
        if "edges" in data:
            for edge in data["edges"]:
                if len(edge) >= 2:
                    source = edge[0]
                    target = edge[1]
                    weight = edge[2] if len(edge) > 2 else 1.0
                    graph.add_edge(source, target, weight=weight)
        
        # Fallback for "adjacency" format if reused
        elif "adjacency" in data:
             for from_action, to_actions in data['adjacency'].items():
                graph.add_action(from_action)
                for to_action in to_actions:
                    graph.add_edge(from_action, to_action)
                    
        graph.compute_layers()
        return graph

    def export_mermaid(self, title: str = "API Dependency Graph") -> str:
        """
        导出 Mermaid 流程图格式
        """
        lines = ["graph TD"]
        lines.append(f"    %% {title}")
        
        edges_added = set()
        for action, node in self.nodes.items():
            for successor in node.successors:
                weight = self.edge_weights.get((action, successor), 1.0)
                # 显示概率权重，如果不是1.0
                if weight < 0.99 or weight > 1.01:
                    edge_str = f"    {action} -- {weight:.2f} --> {successor}"
                else:
                    edge_str = f"    {action} --> {successor}"
                    
                if edge_str not in edges_added:
                    lines.append(edge_str)
                    edges_added.add(edge_str)
                    
        return "\n".join(lines)

    def export_layers_text(self) -> str:
        """
        导出分层文本表示
        """
        if not self._layers:
            self.compute_layers()
            
        lines = ["Execution Layers (Parallel Groups):"]
        for idx, layer in enumerate(self._layers):
            lines.append(f"Layer {idx}: {', '.join(layer)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            action: {
                "predecessors": list(node.predecessors),
                "successors": list(node.successors),
                "layer": node.layer,
                "state": node.state.value
            }
            for action, node in self.nodes.items()
        }


class PosetPlanner:
    """
    偏序图规划器
    
    实现 System 1 快速执行：
    1. 加载偏序图
    2. 按拓扑顺序调度
    3. 支持并行执行同层动作
    
    后验不确定性支持（规划中）：
    - 边权重来自 BHPOP 贝叶斯推断的后验分布
    - 可通过 EDGE_CONFIDENCE_THRESHOLD 阈值过滤低置信度边
    - 低于阈值的边视为"不可靠依赖"，不阻塞执行
    - 这实现了论文中"后验不确定性驱动安全边界"的设计
    """
    
    # ============================================================
    # 意图类型 -> 偏序图文件名映射
    # 6个仿真场景对应的偏序图配置文件
    # ============================================================
    INTENT_TO_POSET_FILE: Dict[str, str] = {
        "simple_ecs": "simple_ecs.json",
        "slb_ecs_rds": "slb_ecs_rds.json",
        "slb_ecs_redis": "slb_ecs_redis.json",
        "eip_slb_ecs": "eip_slb_ecs.json",
        "dual_zone_ecs_slb": "dual_zone_ecs_slb.json",
        "dual_zone_ecs_slb_rds": "dual_zone_ecs_slb_rds.json",
    }
    
    # 核心链路的默认偏序图
    DEFAULT_WEB_STACK_POSET = {
        "CreateVpc": {"predecessors": []},
        "CreateVSwitch": {"predecessors": ["CreateVpc"]},
        "CreateSecurityGroup": {"predecessors": ["CreateVpc"]},
        "AuthorizeSecurityGroup": {"predecessors": ["CreateSecurityGroup"]},
        "RunInstances": {"predecessors": ["CreateVSwitch", "CreateSecurityGroup", "AuthorizeSecurityGroup"]},
        "CreateLoadBalancer": {"predecessors": ["CreateVSwitch"]},
        "AddBackendServers": {"predecessors": ["CreateLoadBalancer", "RunInstances"]},
    }
    
    def __init__(self, io_registry: Optional[IORegistry] = None, poset_dir: Optional[str] = None):
        """初始化规划器
        
        Args:
            io_registry: IO 注册表
            poset_dir: 偏序图文件目录（默认为 manual_scenarios）
        """
        self.io_registry = io_registry or get_io_registry()
        self.poset: Optional[PosetGraph] = None
        self._poset_dir = poset_dir  # 偏序图文件目录
    
    def set_poset_dir(self, poset_dir: str) -> None:
        """设置偏序图文件目录"""
        self._poset_dir = poset_dir
    
    def load_poset(self, path: str, silent: bool = False) -> bool:
        """从文件加载偏序图
        
        Args:
            path: 偏序图文件路径
            silent: 是否静默模式（不打印错误）
        """
        try:
            self.poset = PosetGraph.from_json(path)
            return True
        except Exception as e:
            if not silent:
                print(f"Failed to load poset: {e}")
            return False
    
    def load_default_poset(self, intent_type: str = "web_stack") -> None:
        """加载默认偏序图"""
        if intent_type == "web_stack":
            self.poset = PosetGraph.from_dict(self.DEFAULT_WEB_STACK_POSET)
        else:
            self.poset = PosetGraph()
    
    def __del__load_poset_for_intent(self, intent_type: str, edge_threshold: float) -> bool:
        """
        这个函数已经不用了，后续在外围加载好偏序图后传入
        根据意图类型加载对应的偏序图
        
        优先级：
        1. HPO_scenarios 目录下的后验推断结果（贝叶斯学习得到）
        2. manual_scenarios 目录下的手工定义图
        3. 默认偏序图
        
        Args:
            intent_type: 意图类型（IntentType.value）
        Returns:
            是否成功加载
        """
        # 获取对应的偏序图文件名
        poset_file = self.INTENT_TO_POSET_FILE.get(intent_type)
        if not poset_file:
            # 无匹配的偏序图，使用默认
            self.load_default_poset("web_stack")
            return False
        
        # 优先尝试加载 HPO 后验结果
        hpo_path = self._find_hpo_poset_file(poset_file)
        if hpo_path:
            try:
                self.poset = PosetGraph.load_from_hpo_posterior(str(hpo_path), edge_threshold=edge_threshold)
                return True
            except Exception as e:
                print(f"Failed to load HPO poset for {intent_type}: {e}, falling back to manual config")
        
        # 回退到手工配置
        manual_path = self._find_manual_poset_file(poset_file)
        if manual_path:
            try:
                self.poset = PosetGraph.load_from_manual_config(str(manual_path))
                return True
            except Exception as e:
                print(f"Failed to load manual poset for intent {intent_type}: {e}")
        
        # 最终回退到默认
        self.load_default_poset("web_stack")
        return False
    
    def _find_hpo_poset_file(self, filename: str) -> Optional[Path]:
        """查找 HPO 后验偏序图文件（优先级最高）"""
        # 优先使用配置的目录
        if self._poset_dir:
            path = Path(self._poset_dir) / filename
            if path.exists():
                return path
        
        # HPO_scenarios 目录
        search_paths = [
            Path(__file__).parent.parent.parent.parent / "simulation_workspace" / "HPO_scenarios",
            Path(".") / "HPO_scenarios",
            Path(".") / "simulation_workspace" / "HPO_scenarios",
        ]
        
        for base_dir in search_paths:
            path = base_dir / filename
            if path.exists():
                return path
        
        return None
    
    def _find_manual_poset_file(self, filename: str) -> Optional[Path]:
        """查找手工定义的偏序图文件"""
        search_paths = [
            Path(__file__).parent.parent.parent.parent / "simulation_workspace" / "manual_scenarios",
            Path(".") / "manual_scenarios",
            Path(".") / "simulation_workspace" / "manual_scenarios",
        ]
        
        for base_dir in search_paths:
            path = base_dir / filename
            if path.exists():
                return path
        
        return None
    
    def _find_poset_file(self, filename: str) -> Optional[Path]:
        """查找偏序图文件（兼容旧接口，优先 HPO 再 Manual）"""
        # 优先 HPO
        hpo_path = self._find_hpo_poset_file(filename)
        if hpo_path:
            return hpo_path
        
        # 回退 Manual
        return self._find_manual_poset_file(filename)
    
    def has_poset_for_intent(self, intent_type: str) -> bool:
        """检查是否有对应意图的偏序图"""
        if intent_type not in self.INTENT_TO_POSET_FILE:
            return False
        poset_file = self.INTENT_TO_POSET_FILE[intent_type]
        return self._find_poset_file(poset_file) is not None
    
    def set_poset(self, poset: PosetGraph) -> None:
        """设置偏序图"""
        self.poset = poset
    
    def get_next_actions(self, blackboard: Blackboard, 
                          use_confidence_filter: bool = False,
                          io_guard_enabled: bool = True) -> List[str]:
        """
        获取下一批可执行的动作 (Frontier)
            
        Args:
            blackboard: 参数黑板
            use_confidence_filter: 是否启用后验边置信度过滤
                启用后，低于 PosetGraph.EDGE_CONFIDENCE_THRESHOLD (默认0.8) 的边
                将被视为“不可靠依赖”，不阻塞后续动作执行。
                这允许在后验不确定时提高执行鲁棒性。
            io_guard_enabled: 是否启用 IO Guard（预防性参数检查）
                False 时只依赖偏序图前置依赖，让 API 执行时自己报错触发降级
            
        Returns:
            可并行执行的动作列表
        """
        if not self.poset:
            return []
            
        completed = self.poset.get_completed_actions()
        return self.poset.get_ready_actions(completed, blackboard, self.io_registry, 
                                            use_confidence_filter=use_confidence_filter,
                                            io_guard_enabled=io_guard_enabled)
    
    def mark_action_done(self, action: str, result: Dict[str, Any]) -> None:
        """标记动作完成"""
        if self.poset:
            self.poset.mark_completed(action, result)
    
    def mark_action_failed(self, action: str, error: str) -> None:
        """标记动作失败"""
        if self.poset:
            self.poset.mark_failed(action, error)
    
    def is_complete(self) -> bool:
        """检查是否所有动作都已完成"""
        return self.poset.is_complete() if self.poset else True
    
    def has_failed(self) -> bool:
        """检查是否有失败的动作"""
        return self.poset.has_failed() if self.poset else False
    
    def get_execution_plan(self) -> List[List[str]]:
        """获取执行计划（按层分组）"""
        if not self.poset:
            return []
        
        self.poset.compute_layers()
        return self.poset._layers
    
    def get_remaining_graph(self) -> Dict[str, List[str]]:
        """获取剩余子图（用于降级时传递）"""
        if not self.poset:
            return {}
        return self.poset.get_remaining_subgraph()
    
    def check_io_guard(self, action: str, blackboard: Blackboard) -> Tuple[bool, List[str]]:
        """
        IO Guard 检查
        
        Returns:
            (is_ready, missing_params)
        """
        return self.io_registry.check_inputs_ready(action, blackboard)
