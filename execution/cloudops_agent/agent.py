"""
CloudOps Agent - 主入口

实现 System 1 + System 2 混合架构的云资源编排 Agent
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from datetime import datetime

from execution.cloudops_agent.config import AgentConfig, ExecutionMode
from execution.cloudops_agent.controller.intent_parser import IntentParser, ParsedIntent
from execution.cloudops_agent.controller.mode_selector import ModeSelector, ModeDecision
from execution.cloudops_agent.planning.poset_planner import PosetPlanner, PosetGraph
from execution.cloudops_agent.planning.react_planner import ReActPlanner
from execution.cloudops_agent.knowledge.io_registry import IORegistry, get_io_registry
from execution.cloudops_agent.memory.blackboard import Blackboard
from execution.cloudops_agent.memory.trace_store import TraceStore, ActionStatus
from execution.cloudops_agent.tools.gym_adapter import GymToolAdapter, ToolResult


class AgentStatus(Enum):
    """Agent 执行状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass
class ExecutionResult:
    """执行结果"""
    status: AgentStatus
    mode_used: ExecutionMode
    fallback_count: int
    total_tokens: int
    actions_executed: List[str]
    resources_created: Dict[str, Any]
    error: Optional[str] = None
    trace_id: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "mode_used": self.mode_used.value,
            "fallback_count": self.fallback_count,
            "total_tokens": self.total_tokens,
            "actions_executed": self.actions_executed,
            "resources_created": self.resources_created,
            "error": self.error,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms
        }


class CloudOpsAgent:
    """
    CloudOps ReAct Agent
    
    支持三种执行模式：
    - Expert Mode: 偏序图驱动，零 LLM 调用
    - Hybrid Mode: 偏序图 + 动态降级
    - Explore Mode: 纯 ReAct + RAG
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, llm_client=None):
        """
        初始化 Agent
        
        Args:
            config: Agent 配置
            llm_client: LLM 客户端（OpenAI 兼容）
        """
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        
        # 初始化组件
        self.io_registry = get_io_registry()
        self.intent_parser = IntentParser(llm_client=llm_client)
        self.mode_selector = ModeSelector(self.config)
        self.poset_planner = PosetPlanner(io_registry=self.io_registry)
        self.react_planner = ReActPlanner(llm_client=llm_client, io_registry=self.io_registry,
                                          max_steps=self.config.max_react_steps,
                                          model_name=self.config.llm.react_reasoning.model)
        self.tool_adapter = GymToolAdapter(io_registry=self.io_registry)
        self.trace_store = TraceStore(output_path=self.config.switches.trace_output_path)
        
        # 执行状态
        self.blackboard = Blackboard()
        self.status = AgentStatus.IDLE
        self._current_intent: Optional[ParsedIntent] = None
        self._fallback_count = 0
        self._total_tokens = 0
        self._actions_executed: List[str] = []
        
        # 预设参数（用于混合模式，在 blackboard 初始化后注入）
        self._preset_params: Dict[str, Any] = {}
        
        # 设置工具描述（供 ReAct 使用）
        self.react_planner.set_tool_descriptions(self.tool_adapter.get_all_tool_descriptions())
        
        # ============================================================
        # 偏序图加载职责边界
        # ============================================================
        # 偏序图的加载已从 Agent 内部移除，必须由外部脚本完成加载后注入。
        # 
        # 外部注入方式（推荐）：
        #   1. config.switches.poset_enabled = False  # 创建 Agent 时先禁用
        #   2. agent = CloudOpsAgent(config=config, llm_client=client)
        #   3. poset = PosetGraph.load_from_hpo_posterior(path, edge_threshold=0.8)
        #   4. agent.poset_planner.set_poset(poset)
        #   5. agent.mode_selector.set_poset_graph(poset.to_dict())
        #   6. agent.config.switches.poset_enabled = True  # 启用偏序图执行
        #
        # 参考实现：simulation_workspace/hpo_batch_experiment.py::create_agent_with_hpo_poset()
        # ============================================================
        
        # [已禁用] 原初始化加载逻辑 - 不支持 HPO 后验格式，且缺少 edge_threshold 传递
        # if self.config.switches.poset_enabled:
        #     self._load_poset()
    
    # ============================================================
    # [已禁用] 以下偏序图加载方法已移除，由外部脚本负责加载
    # ============================================================
    
    # def _load_poset(self) -> bool:
    #     """加载偏序图（从配置路径）
    #     
    #     [已禁用] 此方法不支持 HPO 后验格式，且未传递 edge_threshold 参数。
    #     请使用外部注入方式：PosetGraph.load_from_hpo_posterior() + set_poset()
    #     """
    #     path = self.config.switches.poset_path
    #     try:
    #         success = self.poset_planner.load_poset(path, silent=True)
    #         if success:
    #             if self.poset_planner.poset:
    #                 self.mode_selector.set_poset_graph(self.poset_planner.poset.to_dict())
    #             return True
    #     except Exception as e:
    #         if self.config.switches.verbose:
    #             print(f"Failed to load poset from {path}: {e}")
    #     
    #     self.poset_planner.load_default_poset("web_stack")
    #     if self.poset_planner.poset:
    #         self.mode_selector.set_poset_graph(self.poset_planner.poset.to_dict())
    #     return False
    
    # def _load_poset_for_intent(self, intent: ParsedIntent) -> bool:
    #     """根据意图加载对应的偏序图
    #     
    #     [已禁用] 运行时动态加载已移除，偏序图应在 Agent 创建后由外部注入。
    #     """
    #     intent_type = intent.intent_type.value
    #     edge_threshold = self.config.switches.poset_edge_threshold
    #     success = self.poset_planner.load_poset_for_intent(intent_type, edge_threshold=edge_threshold)
    #     if self.poset_planner.poset:
    #         self.mode_selector.set_poset_graph(self.poset_planner.poset.to_dict())
    #     if self.config.switches.verbose:
    #         if success:
    #             print(f"Loaded poset for intent: {intent_type} (edge_threshold={edge_threshold})")
    #         else:
    #             print(f"No specific poset for intent {intent_type}, using default")
    #     return success
    
    def run(self, task: str) -> ExecutionResult:
        """
        执行任务
        
        Args:
            task: 用户任务描述
            
        Returns:
            ExecutionResult
        """
        start_time = datetime.now()
        self.status = AgentStatus.RUNNING
        self._reset_state()
        
        # 开始 trace
        trace_id = None
        error_msg = None  # 初始化错误消息
        if self.config.switches.trace_enabled:
            trace_id = self.trace_store.start_trace(
                task=task,
                mode=self.config.execution_mode.value,
                initial_blackboard=self.blackboard.snapshot()
            )
        
        try:
            # Step 1: 解析意图（所有模式都使用 LLM，只要 llm_client 存在）
            use_llm = self.llm_client is not None
            intent = self.intent_parser.parse(task, use_llm=use_llm)
            self._current_intent = intent
            
            if self.config.switches.trace_enabled:
                self.trace_store.set_intent(intent.to_dict())
            
            # Step 1.5: 偏序图选择
            # 注意：偏序图应在 Agent 初始化时通过 poset_planner.set_poset() 预先设置
            # 运行时不再重新加载，避免覆盖外部设置的 HPO 后验偏序图
            # 如需根据意图动态选择，应在初始化时加载所有场景的偏序图，run() 时只做选择
            # 
            # 原代码（已禁用）：
            # if self.config.switches.poset_enabled:
            #     self._load_poset_for_intent(intent)
            
            # 初始化 blackboard (分层结构)
            initial_params = self.intent_parser.intent_to_blackboard(intent)
            self.blackboard.from_dict(initial_params)
            
            # 注入预设参数（混合模式 LLM 参数推理结果）
            if self._preset_params:
                self._inject_preset_params()
            
            if self.config.switches.verbose:
                print(f"Intent: {intent.intent_type.value}")
                print(f"Blackboard: {self.blackboard.to_dict()}")
            
            # Step 2: 选择模式
            mode_decision = self.mode_selector.select_mode(intent)
            
            if self.config.switches.verbose:
                print(f"Mode: {mode_decision.mode.value} ({mode_decision.reason})")
            
            # Step 3: 执行
            if mode_decision.mode == ExecutionMode.EXPERT:
                self._execute_expert_mode()
            elif mode_decision.mode == ExecutionMode.HYBRID:
                self._execute_hybrid_mode()
            else:
                self._execute_explore_mode(task)
            
            # 检查结果
            if self.status != AgentStatus.FAILED:
                self.status = AgentStatus.SUCCESS
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            error_msg = str(e)
            if self.config.switches.verbose:
                print(f"Error: {error_msg}")
        
        # 结束 trace
        if self.config.switches.trace_enabled:
            self.trace_store.end_trace(
                status=self.status.value,
                final_blackboard=self.blackboard.snapshot(),
                error_summary=error_msg if self.status == AgentStatus.FAILED else None
            )
        
        # 计算耗时
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExecutionResult(
            status=self.status,
            mode_used=mode_decision.mode,
            fallback_count=self._fallback_count,
            total_tokens=self._total_tokens,
            actions_executed=self._actions_executed,
            resources_created=self._extract_created_resources(),
            error=error_msg if self.status == AgentStatus.FAILED else None,
            trace_id=trace_id,
            duration_ms=duration
        )
    
    def _reset_state(self) -> None:
        """重置执行状态"""
        self.blackboard.clear()
        self.tool_adapter.reset()
        self._fallback_count = 0
        self._total_tokens = 0
        self._actions_executed = []
        
        # 重置偏序图状态（注意：不再在这里加载，而是在 run() 中根据意图加载）
        if self.config.switches.poset_enabled and self.poset_planner.poset:
            # 重置节点状态，保持当前偏序图结构
            for node in self.poset_planner.poset.nodes.values():
                from execution.cloudops_agent.planning.poset_planner import ActionState
                node.state = ActionState.PENDING
                node.result = None
                node.error = None
    
    def set_preset_params(self, params: Dict[str, Any]) -> None:
        """
        设置预设参数（用于混合模式）
        
        这些参数会在 blackboard 初始化后自动注入，
        用于支持外部 LLM 参数推理结果的注入。
        
        Args:
            params: 要预设的参数字典
        """
        self._preset_params = params
    
    def _inject_preset_params(self) -> None:
        """
        将预设参数注入到 blackboard
        
        使用分层结构注入：
        - 全局参数直接写入 global
        - 产品相关参数写入对应 namespace
        """
        bb = self.blackboard
        
        # 全局参数列表
        global_params = ["RegionId", "ZoneId", "ZoneIdSecondary", "Amount", 
                         "InstanceType", "ImageId", "CidrBlock",
                         "IpProtocol", "PortRange", "SourceCidrIp"]
        
        for param, value in self._preset_params.items():
            if param in global_params:
                bb.set_global(param, value)
        
        # 产品命名空间映射
        ns_mapping = {
            "vpc": ["CidrBlock", "VSwitchCidrBlock"],
            "ecs": ["InstanceType", "ImageId", "SystemDiskCategory", "SystemDiskSize"],
            "slb": ["AddressType", "LoadBalancerSpec", "ListenerPort", "BackendServerPort",
                    "ListenerProtocol", "HealthCheck", "Bandwidth"],
            "rds": ["Engine", "EngineVersion", "DBInstanceClass", "DBInstanceStorage",
                    "DBInstanceNetType", "SecurityIPList", "PayType", "DBInstanceStorageType",
                    "AccountName", "AccountPassword", "SecurityIps"],
            "redis": ["InstanceClass", "ChargeType", "NetworkType"],
            "eip": ["Bandwidth", "InternetChargeType"],
        }
        
        for ns, ns_params in ns_mapping.items():
            for param in ns_params:
                if param in self._preset_params:
                    bb.set_ns(ns, param, self._preset_params[param])
        
        if self.config.switches.verbose:
            print(f"Injected {len(self._preset_params)} preset params")
    
    def _execute_expert_mode(self) -> None:
        """执行 Expert 模式（纯偏序图）
        
        并行执行分析：
        - 同一层的动作可以并行执行
        - 使用虚拟时钟模拟并行时间：每层取最大耗时
        - 记录并行统计信息用于提效分析
        """
        layer_idx = 0  # 当前执行层
        
        # 并行统计
        layer_times_ms = []  # 每层执行时间（取该层最大耗时）
        actions_per_layer = []  # 每层动作数
        sequential_time_ms = 0.0  # 顺序执行总时间（累加所有动作）
        
        # 重置虚拟时钟（确保从0开始）
        self.tool_adapter.reset_virtual_clock()
        
        while not self.poset_planner.is_complete():
            # 获取下一批可执行的动作（同层可并行）
            # io_guard_enabled=False 时只依赖偏序图，API 报错时触发降级
            io_guard = self.config.switches.poset_io_guard_enabled
            ready_actions = self.poset_planner.get_next_actions(
                self.blackboard, 
                io_guard_enabled=io_guard
            )
            
            if not ready_actions:
                if self.poset_planner.has_failed():
                    break
                # 无可执行动作，直接降级（偏序图阻塞）
                if not self.config.switches.poset_strict_mode:
                    self._execute_fallback(failed_action=None, error="偏序图执行阻塞，无可执行动作")
                    return
                else:
                    self.status = AgentStatus.FAILED
                    break
            
            # 记录本层信息
            batch_id = f"L{layer_idx}"
            actions_per_layer.append(len(ready_actions))
            layer_max_duration_ms = 0.0
            
            # 并行层的虚拟开始时间（所有动作共享）
            layer_virtual_start = self.tool_adapter.get_virtual_time()
            
            # 执行动作（模拟并行：记录每个动作耗时，层取最大值）
            for action in ready_actions:
                virtual_start = self.tool_adapter.get_virtual_time()
                
                # 对单个动作进行重试
                action_retry_count = 0
                result = None
                
                while action_retry_count < self.config.max_retries:
                    result = self._execute_action_with_parallel_info(
                        action, 
                        source="poset",
                        layer=layer_idx,
                        batch_id=batch_id,
                        virtual_start_time=layer_virtual_start  # 并行起始时间相同
                    )
                    
                    if result.success:
                        break  # 成功则跳出重试循环
                    
                    action_retry_count += 1
                    if self.config.switches.verbose:
                        print(f"  Action {action} failed (retry {action_retry_count}/{self.config.max_retries}): {result.error}")
                
                virtual_end = self.tool_adapter.get_virtual_time()
                action_duration_ms = (virtual_end - virtual_start) * 1000
                
                # 累计顺序时间
                sequential_time_ms += action_duration_ms
                
                # 更新层最大耗时（并行取最大值）
                layer_max_duration_ms = max(layer_max_duration_ms, action_duration_ms)
                
                if result.success:
                    self.poset_planner.mark_action_done(action, result.result or {})
                    # 更新 blackboard (使用新的分层写入)
                    if result.result:
                        self.io_registry.write_outputs_to_blackboard(
                            action, result.result, self.blackboard
                        )
                else:
                    # 重试 3 次仍失败
                    self.poset_planner.mark_action_failed(action, result.error or "Unknown error")
                    
                    if self.config.switches.poset_strict_mode:
                        self.status = AgentStatus.FAILED
                        return
                    
                    # 重试 3 次后降级
                    self._execute_fallback(failed_action=action, error=result.error)
                    return
            
            # 记录本层耗时
            layer_times_ms.append(layer_max_duration_ms)
            layer_idx += 1
        
        # 计算并行统计
        parallel_time_ms = sum(layer_times_ms)  # 并行执行时间 = 各层最大值之和
        speedup = sequential_time_ms / parallel_time_ms if parallel_time_ms > 0 else 1.0
        max_parallel = max(actions_per_layer) if actions_per_layer else 0
        
        # 记录并行统计到 Trace
        if self.config.switches.trace_enabled:
            self.trace_store.set_parallelism_stats({
                "total_layers": layer_idx,
                "max_parallel_actions": max_parallel,
                "actions_per_layer": actions_per_layer,
                "sequential_time_ms": round(sequential_time_ms, 2),
                "parallel_time_ms": round(parallel_time_ms, 2),
                "speedup": round(speedup, 2),
                "layer_times_ms": [round(t, 2) for t in layer_times_ms]
            })
    
    def _execute_hybrid_mode(self) -> None:
        """执行 Hybrid 模式（偏序图 + 降级）"""
        # 先尝试 Expert 模式
        self._execute_expert_mode()
        
        # 如果失败且允许降级，切换到 ReAct
        if self.status == AgentStatus.FAILED and self.config.switches.poset_fallback_enabled:
            self._execute_fallback(failed_action=None, error="Expert模式执行失败")
    
    def _execute_explore_mode(self, task: str, verification_retry: int = 0) -> None:
        """执行 Explore 模式（纯 ReAct）
        
        Args:
            task: 任务描述
            verification_retry: 验证重试次数（内部使用）
        """
        # 传递 intent_type 和工具执行器以支持完成验证
        intent_type = self._current_intent.intent_type.value if self._current_intent else None
        
        if self.config.switches.verbose:
            print(f"[ReAct] Starting explore mode, task: {task[:100]}...")
            print(f"[ReAct] llm_client exists: {self.react_planner.llm_client is not None}")
        
        self.react_planner.start(
            task, 
            self.blackboard,
            intent_type=intent_type,
            tool_executor=self._execute_action_for_verification
        )
        
        observation = None
        step_count = 0
        while not self.react_planner.is_complete():
            action, params = self.react_planner.step(observation)
            
            if self.config.switches.verbose:
                print(f"[ReAct] Step {step_count}: action={action}, is_complete={self.react_planner.is_complete()}")
                if self.react_planner.state:
                    print(f"[ReAct] final_answer: {self.react_planner.state.final_answer[:100] if self.react_planner.state.final_answer else 'None'}")
            
            if action is None:
                if self.config.switches.verbose:
                    print(f"[ReAct] Breaking: action is None")
                break
            
            step_count += 1
            # 执行动作
            result = self._execute_action(action, params=params, source="react")
            observation = result
            
            # 更新 blackboard (使用新的分层写入)
            if result.success and result.result:
                self.io_registry.write_outputs_to_blackboard(
                    action, result.result, self.blackboard
                )
        
        self._total_tokens += self.react_planner.get_total_tokens()
        
        # 将 token 消耗同步到 TraceStore
        if self.config.switches.trace_enabled and self.trace_store.get_current_trace():
             self.trace_store.get_current_trace().total_llm_tokens = self._total_tokens
        
        # 执行意图完成验证，如果失败则重试
        self._verify_task_completion_with_retry(task, verification_retry)
    
    def _execute_action_for_verification(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """为验证提供的动作执行器（不记录到 actions_executed）"""
        if self.config.switches.dry_run:
            return ToolResult(success=True, action=action, 
                             result={"dry_run": True, "action": action})
        return self.tool_adapter.execute(action, params)
    
    def _verify_task_completion_with_retry(self, task: str, retry_count: int = 0) -> None:
        """验证任务是否真正完成，失败时重试
        
        Args:
            task: 原始任务描述
            retry_count: 当前重试次数
        """
        max_verification_retries = self.config.max_verification_retries
        
        if not self._current_intent:
            return
        
        passed, message, verify_action, verify_params = self.react_planner.verify_intent_completion()
        
        if self.config.switches.verbose:
            print(f"Verification (attempt {retry_count + 1}): {message}")
        
        # 记录验证动作到 trace
        if self.config.switches.trace_enabled and verify_action:
            action_id = self.trace_store.record_action_start(
                verify_action, verify_params, "verification", "ECS"  # 假设验证通常是 ECS
            )
            if passed:
                self.trace_store.record_action_success(action_id, {"verification": "passed", "message": message})
            else:
                self.trace_store.record_action_failure(action_id, message, "VERIFICATION_FAILED")
        
        # 根据验证结果更新状态
        if passed:
            return  # 验证通过，直接返回
        
        # 验证失败，检查是否可以重试
        if retry_count < max_verification_retries:
            if self.config.switches.verbose:
                print(f"Verification failed, retrying with feedback to LLM (retry {retry_count + 1}/{max_verification_retries})")
            
            # 构建带有验证失败反馈的任务描述
            retry_task = self._build_retry_task(task, message)
            
            # 重新执行 explore 模式
            self._execute_explore_mode(retry_task, verification_retry=retry_count + 1)
        else:
            # 超过最大重试次数，标记失败
            self.status = AgentStatus.FAILED
            if self.config.switches.verbose:
                print(f"Task verification FAILED after {max_verification_retries} retries: {message}")
    
    def _build_retry_task(self, original_task: str, verification_error: str) -> str:
        """构建带有验证失败反馈的重试任务描述
        
        Args:
            original_task: 原始任务描述
            verification_error: 验证失败的错误信息
            
        Returns:
            包含反馈信息的新任务描述
        """
        # 获取当前已创建的资源
        created_resources = self._extract_created_resources()
        resources_str = ", ".join([f"{k}={v}" for k, v in created_resources.items()]) if created_resources else "无"
        
        retry_task = f"""【任务继续执行 - 之前验证失败】

原始任务: {original_task}

验证失败原因: {verification_error}

已创建的资源: {resources_str}

请继续完成任务中尚未创建的资源。注意检查任务描述中提到的所有资源类型（如 ECS、SLB、EIP 等）是否都已创建。"""
        
        return retry_task
    
    def _execute_fallback(self, failed_action: Optional[str] = None, error: Optional[str] = None) -> None:
        """执行降级到 ReAct
        
        Args:
            failed_action: 触发降级的失败动作（如果有）
            error: 失败的错误信息（如果有）
        """
        self._fallback_count += 1
        self.status = AgentStatus.FALLBACK
        
        # 记录到 TraceStore (确保持久化)
        if self.config.switches.trace_enabled:
            self.trace_store.record_fallback()
        
        if self.config.switches.verbose:
            print(f"Fallback #{self._fallback_count} to ReAct mode")
            if failed_action:
                print(f"  Failed action: {failed_action}")
            if error:
                print(f"  Error: {error}")
        
        # 构建完整的降级上下文
        fallback_task = self._build_fallback_context(failed_action, error)
        
        # 使用 ReAct 完成剩余任务
        self._execute_explore_mode(fallback_task)
        
        if self.react_planner.is_complete():
            self.status = AgentStatus.SUCCESS
    
    def _safe_str(self, s: Any) -> str:
        """安全转换字符串，移除无效 Unicode 字符"""
        if s is None:
            return ""
        text = str(s)
        # 移除 surrogate 字符（会导致 UTF-8 编码失败）
        return text.encode('utf-8', errors='replace').decode('utf-8')
    
    def _build_fallback_context(self, failed_action: Optional[str] = None, error: Optional[str] = None) -> str:
        """构建降级任务的完整上下文
        
        Args:
            failed_action: 触发降级的失败动作
            error: 失败的错误信息
            
        Returns:
            完整的降级任务描述
        """
        # 1. 原始任务目标
        original_task = self._safe_str(self._current_intent.raw_text) if self._current_intent else "未知任务"
        
        # 2. 已执行的动作
        executed_actions = self._actions_executed
        
        # 3. 已创建的资源
        created_resources = self._extract_created_resources()
        resources_str = ", ".join([f"{k}={self._safe_str(v)}" for k, v in created_resources.items()]) if created_resources else "无"
        
        # 4. 剩余待完成的动作
        remaining = self.poset_planner.get_remaining_graph()
        remaining_actions = list(remaining.keys())
        
        # 5. 构建错误上下文（安全处理错误信息）
        error_context = ""
        safe_error = self._safe_str(error) if error else ""
        safe_action = self._safe_str(failed_action) if failed_action else ""
        if safe_action and safe_error:
            error_context = f"\n触发降级的原因: 执行 {safe_action} 时报错 - {safe_error}"
        elif safe_error:
            error_context = f"\n触发降级的原因: {safe_error}"
        
        # 6. 构建完整提示词
        fallback_prompt = f"""【任务降级 - 需要 ReAct 接管】

原始任务: {original_task}

已执行的动作: {executed_actions if executed_actions else '无'}

已创建的资源: {resources_str}
{error_context}

剩余待完成的动作（参考）: {remaining_actions if remaining_actions else '无'}

请分析当前状态，继续完成剩余任务。注意：
1. 已创建的资源无需重复创建
2. 根据错误信息分析问题原因
3. 按依赖顺序完成剩余资源的创建"""
        
        return fallback_prompt
    
    def _execute_action(self, action: str, params: Optional[Dict[str, Any]] = None,
                       source: str = "unknown") -> ToolResult:
        """执行单个动作"""
        # 如果没有指定参数，从 blackboard 填充 (使用分层解析)
        if params is None:
            params = self.io_registry.fill_params_from_blackboard(action, self.blackboard)
        
        if self.config.switches.verbose:
            print(f"Executing: {action} with {params}")
        
        # 获取产品归属 (从 IO Registry)
        spec = self.io_registry.get_spec(action)
        product = spec.product if spec else "unknown"
        
        # 记录 trace
        action_id = None
        if self.config.switches.trace_enabled:
            action_id = self.trace_store.record_action_start(action, params, source, product)
        
        # 干跑模式
        if self.config.switches.dry_run:
            result = ToolResult(success=True, action=action, 
                               result={"dry_run": True, "action": action})
        else:
            result = self.tool_adapter.execute(action, params)
        
        # 更新 trace
        if self.config.switches.trace_enabled and action_id:
            if result.success:
                self.trace_store.record_action_success(action_id, result.result or {})
            else:
                self.trace_store.record_action_failure(action_id, result.error or "",
                                                       result.error_code)
        
        # 记录执行的动作
        if result.success:
            self._actions_executed.append(action)
        
        if self.config.switches.verbose:
            if result.success:
                print(f"  Success: {result.result}")
            else:
                print(f"  Failed: {result.error}")
        
        return result
    
    def _execute_action_with_parallel_info(self, action: str, 
                                           params: Optional[Dict[str, Any]] = None,
                                           source: str = "unknown",
                                           layer: int = -1,
                                           batch_id: Optional[str] = None,
                                           virtual_start_time: float = 0.0) -> ToolResult:
        """执行单个动作，并记录并行执行信息
        
        Args:
            action: API 名称
            params: 参数字典（可选）
            source: 来源标识
            layer: 拓扑层级
            batch_id: 批次标识
            virtual_start_time: 虚拟开始时间
            
        Returns:
            ToolResult
        """
        # 如果没有指定参数，从 blackboard 填充 (使用分层解析)
        if params is None:
            params = self.io_registry.fill_params_from_blackboard(action, self.blackboard)
        
        if self.config.switches.verbose:
            print(f"Executing [L{layer}]: {action} with {params}")
        
        # 获取产品归属 (从 IO Registry)
        spec = self.io_registry.get_spec(action)
        product = spec.product if spec else "unknown"
        
        # 记录虚拟时间（执行前）
        v_start = self.tool_adapter.get_virtual_time()
        
        # 记录 trace（带并行信息）
        action_id = None
        if self.config.switches.trace_enabled:
            action_id = self.trace_store.record_action_start(
                action, params, source, product,
                layer=layer,
                batch_id=batch_id,
                virtual_start_time=virtual_start_time
            )
        
        # 干跑模式
        if self.config.switches.dry_run:
            result = ToolResult(success=True, action=action, 
                               result={"dry_run": True, "action": action})
        else:
            result = self.tool_adapter.execute(action, params)
        
        # 记录虚拟时间（执行后）
        v_end = self.tool_adapter.get_virtual_time()
        simulated_duration_ms = (v_end - v_start) * 1000
        
        # 更新 trace（带并行信息）
        if self.config.switches.trace_enabled and action_id:
            if result.success:
                self.trace_store.record_action_success(
                    action_id, result.result or {},
                    virtual_end_time=v_end,
                    simulated_duration_ms=simulated_duration_ms
                )
            else:
                self.trace_store.record_action_failure(action_id, result.error or "",
                                                       result.error_code)
        
        # 记录执行的动作
        if result.success:
            self._actions_executed.append(action)
        
        if self.config.switches.verbose:
            if result.success:
                print(f"  Success ({simulated_duration_ms:.0f}ms sim): {result.result}")
            else:
                print(f"  Failed: {result.error}")
        
        return result
    
    def _should_fallback(self, error: str, retry_count: int) -> bool:
        """判断是否应该降级"""
        return self.mode_selector.should_fallback(error, retry_count)
    
    def _extract_created_resources(self) -> Dict[str, Any]:
        """提取已创建的资源 (从资源注册表获取)"""
        resources = {}
        
        # 从资源注册表获取所有资源
        all_resources = self.blackboard.list_resources()
        for res in all_resources:
            res_type = res.get("_type", "UNKNOWN")
            res_name = res.get("_name", "unknown")
            # 提取关键 ID
            for id_key in ["VpcId", "VSwitchId", "SecurityGroupId", "InstanceIds",
                          "InstanceId", "LoadBalancerId", "AllocationId", "EipAddress",
                          "DBInstanceId", "GroupId", "ExecutionId"]:
                if id_key in res:
                    # 使用 type.id_key 或直接用 id_key
                    if res_type != "UNKNOWN":
                        key = f"{res_type}.{id_key}"
                    else:
                        key = id_key
                    resources[key] = res[id_key]
        
        # 也从全局参数获取 (向后兼容)
        for id_key in ["VpcId", "VSwitchId", "SecurityGroupId", "InstanceIds",
                       "LoadBalancerId", "AllocationId", "DBInstanceId"]:
            if self.blackboard.has_global(id_key) and id_key not in resources:
                resources[id_key] = self.blackboard.get_global(id_key)
        
        return resources
    
    def get_state_dump(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        return {
            "status": self.status.value,
            "blackboard": self.blackboard.to_dict(),
            "actions_executed": self._actions_executed,
            "fallback_count": self._fallback_count,
            "total_tokens": self._total_tokens,
            "gym_state": self.tool_adapter.get_state_dump()
        }


def create_agent(preset: str = "production", llm_client=None) -> CloudOpsAgent:
    """
    创建 Agent 的工厂函数
    
    Args:
        preset: 预设配置名称
            - trace_collection: 阶段1，积累 trace
            - poset_validation: 阶段2，验证偏序图
            - hybrid_benchmark: 阶段3，混合模式评测
            - production: 生产环境
        llm_client: LLM 客户端
        
    Returns:
        CloudOpsAgent 实例
    """
    presets = {
        "trace_collection": AgentConfig.preset_trace_collection,
        "poset_validation": AgentConfig.preset_poset_validation,
        "hybrid_benchmark": AgentConfig.preset_hybrid_benchmark,
        "production": AgentConfig.preset_production,
    }
    
    config_factory = presets.get(preset, AgentConfig.preset_production)
    config = config_factory()
    
    return CloudOpsAgent(config=config, llm_client=llm_client)
