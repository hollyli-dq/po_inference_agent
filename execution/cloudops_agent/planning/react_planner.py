"""ReAct 规划器

实现 System 2 推理执行:
Thought -> Action -> Observation 循环
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
import json
import re

from execution.cloudops_agent.knowledge.io_registry import IORegistry, get_io_registry
from execution.cloudops_agent.memory.blackboard import Blackboard
from execution.cloudops_agent.tools.gym_adapter import ToolResult


# Intent 完成验证规则：定义每种意图需要验证的资源和反查接口
# 格式: intent_type -> 验证配置
# required_resources: 该意图需要创建的资源类型及其验证信息
#   - skip_api_verify: 为 True 时只从 Blackboard 统计，不调用反查 API（适用于需要统计数量的资源）
#   - product: API 所属产品（用于正确路由）
#
# TODO(端到端验证): 当前验证规则只检查「资源创建」，未检查「配置动作」：
#   - 缺少对 CreateAccount（RDS账号创建）的验证，应调用 DescribeAccounts 反查
#   - 缺少对 AddBackendServers（SLB后端绑定）的验证，应调用 DescribeBackendServers 反查
#   - 缺少对 ModifySecurityIps（RDS白名单配置）的验证，应调用 DescribeDBInstanceIPArrayList 反查
#   后续应扩展 required_resources 或新增 "required_actions" 字段支持配置类动作的验证
#
# TODO(skip_api_verify): 当前所有规则都设置了 skip_api_verify=True，意味着：
#   - 只检查 Blackboard 中是否存在资源 ID，不调用仿真器的 Describe* 接口验证资源状态
#   - 后续应将 skip_api_verify 设为 False，并实现真正的仿真器端到端验证
#   - 这将确保任务成功判定基于仿真器的实际状态，而不仅仅是执行记录
INTENT_VERIFICATION_RULES: Dict[str, Dict[str, Any]] = {
    # === 6个仿真场景的完整验证规则 ===
    
    # 场景1: 简单 ECS 创建
    "simple_ecs": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 1, "product": "ecs", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
    },
    
    # 场景2: SLB + ECS + RDS
    "slb_ecs_rds": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "SLB", "key": "LoadBalancerId", "verify_action": "DescribeLoadBalancers", "result_path": "LoadBalancers.LoadBalancer", "min_count": 1, "product": "slb", "skip_api_verify": True},
            {"type": "RDS", "key": "DBInstanceId", "verify_action": "DescribeDBInstances", "result_path": "Items.DBInstance", "min_count": 1, "product": "rds", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
    },
    
    # 场景3: SLB + ECS + Redis
    "slb_ecs_redis": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "SLB", "key": "LoadBalancerId", "verify_action": "DescribeLoadBalancers", "result_path": "LoadBalancers.LoadBalancer", "min_count": 1, "product": "slb", "skip_api_verify": True},
            {"type": "Redis", "key": "InstanceId", "verify_action": "DescribeInstances", "result_path": "Instances.KVStoreInstance", "min_count": 1, "product": "redis", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
    },
    
    # 场景4: EIP + SLB + ECS
    "eip_slb_ecs": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "SLB", "key": "LoadBalancerId", "verify_action": "DescribeLoadBalancers", "result_path": "LoadBalancers.LoadBalancer", "min_count": 1, "product": "slb", "skip_api_verify": True},
            {"type": "EIP", "key": "AllocationId", "verify_action": "DescribeEipAddresses", "result_path": "EipAddresses.EipAddress", "min_count": 1, "product": "vpc", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
    },
    
    # 场景5: 双可用区 ECS×2 + SLB
    "dual_zone_ecs_slb": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 2, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 2, "product": "ecs", "skip_api_verify": True},
            {"type": "SLB", "key": "LoadBalancerId", "verify_action": "DescribeLoadBalancers", "result_path": "LoadBalancers.LoadBalancer", "min_count": 1, "product": "slb", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
        "expected_ecs_count": 2,
    },
    
    # 场景6: 双可用区 ECS×2 + SLB + RDS 主备
    "dual_zone_ecs_slb_rds": {
        "required_resources": [
            {"type": "VPC", "key": "VpcId", "verify_action": "DescribeVpcs", "result_path": "Vpcs.Vpc", "min_count": 1, "product": "vpc", "skip_api_verify": True},
            {"type": "VSwitch", "key": "VSwitchId", "verify_action": "DescribeVSwitches", "result_path": "VSwitches.VSwitch", "min_count": 2, "product": "vpc", "skip_api_verify": True},
            {"type": "SecurityGroup", "key": "SecurityGroupId", "verify_action": "DescribeSecurityGroups", "result_path": "SecurityGroups.SecurityGroup", "min_count": 1, "product": "ecs", "skip_api_verify": True},
            {"type": "ECS", "key": "InstanceIds", "verify_action": "DescribeInstances", "result_path": "Instances.Instance", "min_count": 2, "product": "ecs", "skip_api_verify": True},
            {"type": "SLB", "key": "LoadBalancerId", "verify_action": "DescribeLoadBalancers", "result_path": "LoadBalancers.LoadBalancer", "min_count": 1, "product": "slb", "skip_api_verify": True},
            {"type": "RDS", "key": "DBInstanceId", "verify_action": "DescribeDBInstances", "result_path": "Items.DBInstance", "min_count": 1, "product": "rds", "skip_api_verify": True},
        ],
        "ecs_count_key": "Amount",
        "expected_ecs_count": 2,
    },
    
}

# Token 估算常量 (按字符数粗估, 1 token ≈ 2-3 中文字符或 4 英文字符)
TOKEN_PER_CHAR = 0.35  # 偏保守估计
MAX_CONTEXT_TOKENS = 100000  # 100K token 阈值
TARGET_CONTEXT_TOKENS = 50000  # 压缩目标 50K


class ReActStep(Enum):
    """ReAct 步骤类型"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


@dataclass
class ReActState:
    """ReAct 执行状态"""
    step_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_thought: str = ""
    current_action: Optional[str] = None
    current_params: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    final_answer: str = ""
    total_tokens: int = 0
    verification_passed: bool = False  # 新增：资源验证是否通过
    verification_message: str = ""  # 新增：验证结果消息
    
    def add_thought(self, thought: str, tokens: int = 0) -> None:
        self.current_thought = thought
        self.history.append({
            "step": self.step_count,
            "type": "thought",
            "content": thought
        })
        self.total_tokens += tokens
    
    def add_action(self, action: str, params: Dict[str, Any]) -> None:
        self.current_action = action
        self.current_params = params
        self.history.append({
            "step": self.step_count,
            "type": "action",
            "action": action,
            "params": params
        })
    
    def add_observation(self, result: ToolResult) -> None:
        # 压缩大型 observation
        compressed_result = self._compress_observation(result.result) if result.success else None
        self.history.append({
            "step": self.step_count,
            "type": "observation",
            "success": result.success,
            "result": compressed_result,
            "error": result.error if not result.success else None
        })
        self.step_count += 1
    
    def _compress_observation(self, result: Any) -> Any:
        """压缩单个 observation 结果"""
        if result is None:
            return None
        
        result_str = json.dumps(result, ensure_ascii=False)
        # 单个 observation 超过 10K 字符时进行压缩
        if len(result_str) > 10000:
            return self._summarize_large_result(result)
        return result
    
    def _summarize_large_result(self, result: Any) -> Dict[str, Any]:
        """对大型结果进行摘要"""
        if not isinstance(result, dict):
            return {"_compressed": True, "_summary": str(result)[:500]}
        
        summary = {"_compressed": True, "RequestId": result.get("RequestId", "")}
        
        # 处理常见的大型列表字段
        list_fields = [
            ("Images", "Image"),
            ("Instances", "Instance"),
            ("Vpcs", "Vpc"),
            ("VSwitches", "VSwitch"),
            ("AvailableZones", "AvailableZone"),
            ("SecurityGroups", "SecurityGroup"),
            ("LoadBalancers", "LoadBalancer"),
        ]
        
        for outer_key, inner_key in list_fields:
            if outer_key in result:
                outer_data = result[outer_key]
                if isinstance(outer_data, dict) and inner_key in outer_data:
                    items = outer_data[inner_key]
                    if isinstance(items, list) and len(items) > 0:
                        # 只保留前 5 条，并提取关键字段
                        compressed_items = []
                        for item in items[:5]:
                            if isinstance(item, dict):
                                # 提取常见的关键字段
                                compressed_item = {}
                                key_fields = ["ImageId", "ImageName", "OSName", "InstanceId", 
                                             "InstanceType", "VpcId", "VSwitchId", "ZoneId",
                                             "SecurityGroupId", "Status", "LoadBalancerId",
                                             "Value", "Available"]
                                for kf in key_fields:
                                    if kf in item:
                                        compressed_item[kf] = item[kf]
                                if compressed_item:
                                    compressed_items.append(compressed_item)
                            else:
                                compressed_items.append(item)
                        
                        summary[f"{outer_key}_count"] = len(items)
                        summary[f"{outer_key}_sample"] = compressed_items
                        summary[f"_note_{outer_key}"] = f"共 {len(items)} 条，已压缩为前 5 条关键字段"
        
        # 保留 TotalCount 如果存在
        if "TotalCount" in result:
            summary["TotalCount"] = result["TotalCount"]
        
        return summary
    
    def set_final(self, answer: str) -> None:
        self.is_complete = True
        self.final_answer = answer
        self.history.append({
            "step": self.step_count,
            "type": "final",
            "answer": answer
        })
    
    def get_formatted_history(self) -> str:
        """格式化历史记录（用于 LLM prompt）"""
        lines = []
        for entry in self.history:
            if entry["type"] == "thought":
                lines.append(f"Thought: {entry['content']}")
            elif entry["type"] == "action":
                lines.append(f"Action: {entry['action']}")
                lines.append(f"Action Input: {json.dumps(entry['params'], ensure_ascii=False)}")
            elif entry["type"] == "observation":
                if entry["success"]:
                    lines.append(f"Observation: {json.dumps(entry['result'], ensure_ascii=False)}")
                else:
                    lines.append(f"Observation: Error - {entry['error']}")
            elif entry["type"] == "final":
                lines.append(f"Final Answer: {entry['answer']}")
        
        result = "\n".join(lines)
        
        # 上下文压缩：如果总长度超过阈值，进行压缩
        estimated_tokens = len(result) * TOKEN_PER_CHAR
        if estimated_tokens > MAX_CONTEXT_TOKENS:
            result = self._compress_history(result)
        
        return result
    
    def _compress_history(self, history_str: str) -> str:
        """压缩历史记录到目标 token 数"""
        target_chars = int(TARGET_CONTEXT_TOKENS / TOKEN_PER_CHAR)
        
        if len(history_str) <= target_chars:
            return history_str
        
        # 策略：保留最近的步骤，压缩早期的 observation
        lines = history_str.split("\n")
        
        # 找到所有 Observation 行并压缩早期的
        new_lines = []
        observation_count = 0
        total_observations = sum(1 for l in lines if l.startswith("Observation:"))
        keep_full_count = min(3, total_observations)  # 保留最近 3 个完整 observation
        
        for line in lines:
            if line.startswith("Observation:"):
                observation_count += 1
                # 早期的 observation 进行压缩
                if observation_count <= total_observations - keep_full_count:
                    # 只保留前 200 字符 + 提示
                    if len(line) > 250:
                        new_lines.append(line[:200] + "... [已压缩]")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        result = "\n".join(new_lines)
        
        # 如果仍然超过目标，进行更激进的截断
        if len(result) > target_chars:
            # 保留最后 target_chars 个字符，并添加提示
            result = "[早期历史已压缩]\n...\n" + result[-target_chars:]
        
        return result


class ReActPlanner:
    """
    ReAct 规划器
    
    实现 Thought-Action-Observation 循环
    """
    
    SYSTEM_PROMPT = """你是一个阿里云资源编排助手。你需要根据用户请求，通过调用 API 完成云资源创建。

可用的 Actions:
{tool_descriptions}

请按以下格式回复：
Thought: 你的思考过程
Action: API 名称
Action Input: {{"param1": "value1", "param2": "value2"}}

或者如果任务完成：
Thought: 任务已完成，总结结果
Final Answer: 创建完成的资源列表

=== 核心原则 ===

1. **探索优先 (Discovery First)**: 
   - 不要假设默认参数（如 ZoneId, InstanceType, ImageId 等）。
   - 创建资源前，**必须**调用查询接口获取可用的资源规格和可用区。
   - 常用查询链：
     - ECS: DescribeAvailableResource -> DescribeImages -> RunInstances
     - RDS: DescribeAvailableZones -> ListClasses -> CreateDBInstance
     - SLB/Redis: DescribeAvailableResource -> Create...

2. **查重 (Idempotency)**: 
   - 在创建资源前，调用 Describe* 接口检查是否已存在同名或符合条件的资源，避免重复创建。

3. **验证 (Verification)**: 
   - 创建资源后，建议调用 Describe* 接口确认资源状态为 Available/Active/Running。

4. **依赖顺序**: 
   - 严格遵守依赖关系：VPC -> VSwitch -> SecurityGroup -> ECS/RDS/SLB。

=== 任务完整性约束 (Task Completeness) - 最高优先级 ===

**[严格规则] 在输出 Final Answer 之前，必须完成以下检查清单：**

1. **资源类型完整性检查**：
   - 仔细分析用户任务中提到的每一个资源类型（ECS、SLB、RDS、Redis、EIP 等）
   - 逐一确认每种资源类型是否已调用对应的 Create*/Run* 接口创建
   - 资源类型对应的创建接口：
     - VPC -> CreateVpc
     - VSwitch -> CreateVSwitch
     - SecurityGroup -> CreateSecurityGroup
     - ECS/实例/服务器 -> RunInstances
     - SLB/负载均衡 -> CreateLoadBalancer + AddBackendServers
     - RDS/数据库/MySQL -> CreateDBInstance
     - Redis/缓存 -> CreateInstance (r-kvstore)
     - EIP/弹性公网IP -> AllocateEipAddress + AssociateEipAddress
   - **如果有任何资源类型未创建，禁止输出 Final Answer，必须继续执行创建操作**

2. **资源数量一致性检查**：
   - 如果任务指定了数量（如"2台ECS"、"双可用区"），必须确保创建的数量与任务要求一致
   - 双可用区场景：至少需要2个VSwitch（分布在不同可用区）+ 对应数量的ECS
   - 检查 RunInstances 的 Amount 参数或多次调用的总数

3. **常见错误警示**：
   - ❌ 只创建了网络基础设施（VPC/VSwitch/SecurityGroup）就结束，忘记创建核心资源（ECS/RDS/SLB）
   - ❌ 查询完可用区/规格信息后直接结束，未执行实际创建操作
   - ❌ 只创建了部分资源就声称完成（如只有ECS没有SLB）
   - ❌ 资源数量不足（如任务要求2台ECS但只创建了1台）

4. **Final Answer 前的自检**：
   在输出 Final Answer 之前，必须在 Thought 中明确列出：
   - 任务要求的资源类型列表
   - 每种资源的创建状态（已创建/未创建）
   - 创建的资源ID汇总
   - 数量是否匹配任务要求
   只有当所有检查项都通过时，才能输出 Final Answer。

=== 示例 ===

示例1 - 创建 ECS：
Thought: 用户需要在杭州创建 ECS。首先查询杭州的可用资源情况，确定哪个可用区有货。
Action: DescribeAvailableResource
Action Input: {{"RegionId": "cn-hangzhou", "DestinationResource": "InstanceType"}}
...
Thought: 可用区 cn-hangzhou-h 支持所需规格。查询该区域的镜像列表。
Action: DescribeImages
Action Input: {{"RegionId": "cn-hangzhou"}}
...
Thought: 确定了可用区、规格和镜像。先创建网络基础设施。
Action: CreateVpc
...
[继续创建 VSwitch, SecurityGroup, 最后 RunInstances]
...
Thought: 任务完整性检查：
- 任务要求：ECS
- VPC: ✓ 已创建 vpc-xxx
- VSwitch: ✓ 已创建 vsw-xxx
- SecurityGroup: ✓ 已创建 sg-xxx
- ECS: ✓ 已创建 i-xxx
所有资源已创建完成。
Final Answer: 创建完成。VPC: vpc-xxx, VSwitch: vsw-xxx, SecurityGroup: sg-xxx, ECS: i-xxx

示例2 - 双可用区 ECS + SLB + RDS（任务完整性检查失败示例）：
Thought: 任务完整性检查：
- 任务要求：双可用区ECS×2、SLB、RDS
- VPC: ✓ 已创建
- VSwitch: ✓ 已创建2个
- SecurityGroup: ✓ 已创建
- ECS: ✓ 已创建2台
- SLB: ✗ 未创建 <-- 缺失！
- RDS: ✗ 未创建 <-- 缺失！
检查未通过，需要继续创建 SLB 和 RDS。
Action: CreateLoadBalancer
...
"""
    
    def __init__(self, llm_client=None, io_registry: Optional[IORegistry] = None,
                 max_steps: int = 20, model_name: str = "qwen3-max",
                 temperature: float = 0.0, enable_thinking: bool = False):
        """
        初始化 ReAct 规划器
        
        Args:
            llm_client: LLM 客户端（OpenAI 兼容）
            io_registry: IO 注册表
            max_steps: 最大步数
            model_name: 模型名称
            temperature: LLM 采样温度（0.0=确定性，>0 增加随机性）
            enable_thinking: 是否启用深度思考模式 (qwen3-max 支持)
        """
        self.llm_client = llm_client
        self.io_registry = io_registry or get_io_registry()
        self.max_steps = max_steps
        self.model_name = model_name
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.state: Optional[ReActState] = None
        self._tool_descriptions: str = ""
        self._intent_type: Optional[str] = None  # 新增：意图类型
        self._tool_executor: Optional[Callable] = None  # 新增：工具执行器回调
    
    def set_llm_client(self, client) -> None:
        """设置 LLM 客户端"""
        self.llm_client = client
    
    def set_tool_descriptions(self, descriptions: str) -> None:
        """设置工具描述"""
        self._tool_descriptions = descriptions
    
    def start(self, task: str, blackboard: Blackboard, 
               intent_type: Optional[str] = None,
               tool_executor: Optional[Callable] = None) -> ReActState:
        """开始新的 ReAct 会话
        
        Args:
            task: 任务描述
            blackboard: 参数黑板
            intent_type: 意图类型（用于完成验证）
            tool_executor: 工具执行器回调函数（用于资源反查）
        """
        self.state = ReActState()
        self._task = task
        self._blackboard = blackboard
        self._intent_type = intent_type
        self._tool_executor = tool_executor
        return self.state
    
    def step(self, observation: Optional[ToolResult] = None) -> tuple[Optional[str], Dict[str, Any]]:
        """
        执行一步 ReAct
        
        Args:
            observation: 上一步的执行结果（如果有）
            
        Returns:
            (action, params) 或 (None, {}) 如果完成
        """
        if not self.state:
            raise RuntimeError("ReAct session not started. Call start() first.")
        
        # 记录上一步的 observation
        if observation:
            self.state.add_observation(observation)
        
        # 检查是否超过最大步数
        if self.state.step_count >= self.max_steps:
            self.state.set_final("达到最大步数限制")
            return None, {}
        
        # 调用 LLM 获取下一步
        if self.llm_client:
            action, params = self._llm_step()
            if action:
                return action, params
            # 如果 LLM 返回 None，可能是 Final Answer，也可能是解析失败
            # 只有在明确解析失败时才提前终止，否则继续循环（如果需要多轮对话）
            if self.state.final_answer == "无法解析 Action":
                return None, {}
            return None, {} # 正常结束
        else:
            # 无 LLM 时使用简单规则
            return self._rule_step()
    
    def _llm_step(self) -> tuple[Optional[str], Dict[str, Any]]:
        """使用 LLM 决定下一步"""
        prompt = self._build_prompt()
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT.format(
                        tool_descriptions=self._tool_descriptions
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                extra_body={"enable_thinking": self.enable_thinking}
            )
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            return self._parse_llm_response(content, tokens)
            
        except Exception as e:
            error_str = str(e)
            # 致命错误（配置错误、认证错误等）应该向上抛出
            fatal_keywords = ['404', '401', '403', 'model', 'api_key', 'authentication', 'not found', 'Invalid']
            is_fatal = any(kw.lower() in error_str.lower() for kw in fatal_keywords)
            
            if is_fatal:
                # 致命错误，向上抛出让调用方处理
                raise RuntimeError(f"LLM 调用失败（配置错误）: {error_str}") from e
            
            # 非致命错误（如网络超时），记录并返回
            self.state.add_thought(f"LLM 调用失败: {error_str}")
            self.state.set_final(f"LLM 错误: {error_str}")
            return None, {}
    
    def _build_prompt(self) -> str:
        """构建 LLM prompt"""
        history = self.state.get_formatted_history()
        # 使用扁平格式便于 LLM 理解
        blackboard_dict = self._blackboard.to_flat_dict()
        # 安全序列化，处理无效 Unicode 字符
        blackboard_str = json.dumps(
            self._sanitize_for_json(blackboard_dict), 
            ensure_ascii=False, 
            indent=2
        )
        
        prompt = f"""任务: {self._task}

当前参数黑板:
{blackboard_str}

执行历史:
{history if history else "(无)"}

请决定下一步动作。"""
        
        return prompt
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """递归清理对象中的无效 Unicode 字符，确保 JSON 序列化安全"""
        if obj is None:
            return None
        elif isinstance(obj, str):
            # 移除 surrogate 字符（会导致 UTF-8 编码失败）
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        elif isinstance(obj, dict):
            return {self._sanitize_for_json(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(item) for item in obj)
        else:
            return obj
    
    def _parse_llm_response(self, content: str, tokens: int) -> tuple[Optional[str], Dict[str, Any]]:
        """解析 LLM 响应"""
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|\Z)', content, re.DOTALL)
        if thought_match:
            self.state.add_thought(thought_match.group(1).strip(), tokens)
        
        # 检查是否是 Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL)
        if final_match:
            self.state.set_final(final_match.group(1).strip())
            return None, {}
        
        # 提取 Action
        action_match = re.search(r'Action:\s*([a-zA-Z0-9_]+)', content)
        if not action_match:
            # 如果没有找到 Action，但 LLM 输出了大量思考，可能是在规划
            # 尝试查找 Final Answer，如果没有则认为无法解析
            self.state.add_thought("LLM 未输出明确的 Action，可能正在思考或输出格式错误")
            self.state.set_final("无法解析 Action")
            return None, {}
        
        action = action_match.group(1)
        
        # 提取 Action Input
        input_match = re.search(r'Action Input:\s*(\{.*\})', content, re.DOTALL)
        params = {}
        if input_match:
            try:
                params = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 合并 blackboard 参数 (使用分层解析)
        bb_params = self.io_registry.fill_params_from_blackboard(action, self._blackboard)
        bb_params.update(params)  # LLM 指定的参数优先
        
        self.state.add_action(action, bb_params)
        return action, bb_params
    
    def _rule_step(self) -> tuple[Optional[str], Dict[str, Any]]:
        """无 LLM 时的规则执行(用于测试)"""
        # 简单的顺序执行
        executed = {e["action"] for e in self.state.history if e["type"] == "action"}
        
        sequence = [
            "CreateVpc",
            "CreateVSwitch",
            "CreateSecurityGroup",
            "RunInstances"
        ]
        
        for action in sequence:
            if action not in executed:
                params = self.io_registry.fill_params_from_blackboard(action, self._blackboard)
                self.state.add_thought(f"执行 {action}")
                self.state.add_action(action, params)
                return action, params
        
        self.state.set_final("所有步骤完成")
        return None, {}
    
    def is_complete(self) -> bool:
        """检查是否完成"""
        return self.state.is_complete if self.state else True
    
    def get_state(self) -> Optional[ReActState]:
        """获取当前状态"""
        return self.state
    
    def get_total_tokens(self) -> int:
        """获取消耗的总 token 数"""
        return self.state.total_tokens if self.state else 0
    
    def verify_intent_completion(self) -> Tuple[bool, str, Optional[str], Dict[str, Any]]:
        """
        验证意图是否真正完成
        
        通用验证规则：
        1. 云产品种类完整性检查 - 所有 required_resources 中定义的资源类型都必须存在
        2. 数量一致性检查 - 创建的资源数量与意图要求匹配
        
        Returns:
            Tuple[bool, str, Optional[str], Dict]:
                - passed: 是否通过验证
                - message: 验证结果消息
                - verify_action: 使用的反查接口（多个时返回第一个失败的）
                - verify_params: 反查参数
        """
        if not self._intent_type:
            return True, "未指定意图类型，跳过验证", None, {}
        
        rule = INTENT_VERIFICATION_RULES.get(self._intent_type)
        if not rule:
            return True, f"意图类型 {self._intent_type} 无验证规则，跳过验证", None, {}
        
        required_resources = rule.get("required_resources", [])
        if not required_resources:
            return True, "无需验证的资源", None, {}
        
        region_id = self._blackboard.get_global("RegionId") or "cn-hangzhou"
        
        # === 通用验证规则 1: 云产品种类完整性检查 ===
        missing_resources = []
        created_resources = []
        first_failed_action = None
        first_failed_params = {}
        
        for res_spec in required_resources:
            res_type = res_spec["type"]
            res_key = res_spec["key"]
            verify_action = res_spec["verify_action"]
            result_path = res_spec["result_path"]
            min_count = res_spec.get("min_count", 1)
            skip_api_verify = res_spec.get("skip_api_verify", False)
            
            # 尝试从 Blackboard 获取资源 ID
            resource_id = self._get_resource_id_from_blackboard(res_type, res_key)
            
            if not resource_id:
                missing_resources.append(f"{res_type}(缺少{res_key})")
                if not first_failed_action:
                    first_failed_action = verify_action
                    first_failed_params = {"RegionId": region_id}
                continue
            
            # 计算 Blackboard 中的资源数量
            bb_count = len(resource_id) if isinstance(resource_id, list) else 1
            
            # 如果 skip_api_verify 为 True，直接使用 Blackboard 的数量
            if skip_api_verify:
                if bb_count >= min_count:
                    created_resources.append(f"{res_type}({bb_count}个)")
                else:
                    missing_resources.append(f"{res_type}(需{min_count}个,实际{bb_count}个)")
                continue
            
            # 如果有执行器且需要 API 验证，进行反查验证
            if self._tool_executor:
                verify_params = {"RegionId": region_id}
                self._add_resource_id_to_params(res_key, resource_id, verify_params)
                
                try:
                    result = self._tool_executor(verify_action, verify_params)
                    if result.success:
                        actual_count = self._count_resources_from_result(result.result, result_path)
                        if actual_count >= min_count:
                            created_resources.append(f"{res_type}({actual_count}个)")
                        else:
                            missing_resources.append(f"{res_type}(需{min_count}个,实际{actual_count}个)")
                            if not first_failed_action:
                                first_failed_action = verify_action
                                first_failed_params = verify_params
                    else:
                        missing_resources.append(f"{res_type}(反查失败)")
                        if not first_failed_action:
                            first_failed_action = verify_action
                            first_failed_params = verify_params
                except Exception as e:
                    missing_resources.append(f"{res_type}(验证异常:{str(e)[:20]})")
            else:
                # 无执行器，仅检查 ID 是否存在
                if bb_count >= min_count:
                    created_resources.append(f"{res_type}({bb_count}个)")
                else:
                    missing_resources.append(f"{res_type}(需{min_count}个,实际{bb_count}个)")
        
        # === 通用验证规则 2: 数量一致性检查 (ECS 数量) ===
        quantity_error = None
        ecs_count_key = rule.get("ecs_count_key")
        expected_ecs_count = rule.get("expected_ecs_count")
        
        if ecs_count_key:
            # 从 Blackboard 获取期望数量
            expected_count = expected_ecs_count or self._blackboard.get_global(ecs_count_key) or 1
            
            # 获取实际创建的 ECS 数量
            ecs_ids = self._get_resource_id_from_blackboard("ECS", "InstanceIds")
            actual_ecs_count = 0
            if ecs_ids:
                actual_ecs_count = len(ecs_ids) if isinstance(ecs_ids, list) else 1
            
            if actual_ecs_count < expected_count:
                quantity_error = f"ECS数量不足(期望{expected_count}台,实际{actual_ecs_count}台)"
        
        # === 汇总验证结果 ===
        if missing_resources or quantity_error:
            error_parts = []
            if missing_resources:
                error_parts.append(f"缺少资源: {', '.join(missing_resources)}")
            if quantity_error:
                error_parts.append(quantity_error)
            
            error_msg = "; ".join(error_parts)
            self.state.verification_passed = False
            self.state.verification_message = error_msg
            return False, f"任务未完成: {error_msg}", first_failed_action, first_failed_params
        
        # 验证通过
        success_msg = f"验证通过: 已创建 {', '.join(created_resources)}"
        self.state.verification_passed = True
        self.state.verification_message = success_msg
        return True, success_msg, None, {}
    
    def _get_resource_id_from_blackboard(self, res_type: str, res_key: str) -> Any:
        """从 Blackboard 获取资源 ID（支持多实例统计）"""
        # 资源类型映射：验证规则中的类型 -> Blackboard 中的类型
        type_map = {
            "ECS": "ECS",
            "VPC": "VPC",
            "VSwitch": "VPC",  # VSwitch 在 Blackboard 中注册为 VPC 类型
            "SecurityGroup": "ECS",  # SecurityGroup 在 Blackboard 中注册为 ECS 类型
            "SLB": "SLB",
            "RDS": "RDS",
            "Redis": "REDIS",  # Redis 在 Blackboard 中可能是 REDIS
            "EIP": "EIP",
        }
        
        bb_type = type_map.get(res_type, res_type)
        
        # 首先尝试从资源注册表获取
        resource_ids = self._blackboard.get_resource_ids(bb_type, res_key)
        if resource_ids:
            return resource_ids
        
        # 特殊处理: VSwitch 可能直接用 VSwitch 类型注册
        if res_type == "VSwitch":
            resource_ids = self._blackboard.get_resource_ids("VSwitch", res_key)
            if resource_ids:
                return resource_ids
        
        # 特殊处理: SecurityGroup 可能直接用 SecurityGroup 类型注册
        if res_type == "SecurityGroup":
            resource_ids = self._blackboard.get_resource_ids("SecurityGroup", res_key)
            if resource_ids:
                return resource_ids
        
        # 尝试从全局参数获取
        resource_id = self._blackboard.get_global(res_key)
        if resource_id:
            return resource_id
        
        # 特殊处理: 某些资源可能有别名
        alias_map = {
            "InstanceId": ["InstanceIds"],  # Redis 的 InstanceId
            "InstanceIds": ["InstanceId"],
            "VSwitchId": [],  # VSwitch 无别名
            "AllocationId": ["EipAddress"],  # EIP 可能用 EipAddress
        }
        
        for alias in alias_map.get(res_key, []):
            resource_ids = self._blackboard.get_resource_ids(bb_type, alias)
            if resource_ids:
                return resource_ids
            resource_id = self._blackboard.get_global(alias)
            if resource_id:
                return resource_id
        
        # 特殊处理: 通过类型统计所有该类型的资源数量
        # 这对于 VSwitch 等需要统计多个实例的场景很重要
        resources = self._blackboard.list_resources(bb_type)
        if resources:
            ids = []
            for r in resources:
                if res_key in r:
                    val = r[res_key]
                    if isinstance(val, list):
                        ids.extend(val)
                    else:
                        ids.append(val)
            if ids:
                return ids
        
        return None
    
    def _add_resource_id_to_params(self, res_key: str, resource_id: Any, params: Dict[str, Any]) -> None:
        """根据资源类型将 ID 添加到参数中"""
        if res_key == "InstanceIds":
            if isinstance(resource_id, list):
                params["InstanceIds"] = resource_id
            else:
                params["InstanceIds"] = [resource_id]
        elif res_key in ["VpcId", "VSwitchId", "SecurityGroupId", "LoadBalancerId", 
                         "DBInstanceId", "AllocationId", "RedisInstanceId"]:
            params[res_key] = resource_id if not isinstance(resource_id, list) else resource_id[0]
    
    def _count_resources_from_result(self, result: Dict[str, Any], result_path: str) -> int:
        """从 API 结果中计算资源数量"""
        if not result:
            return 0
        
        data = result
        for key in result_path.split("."):
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return 0
        
        if isinstance(data, list):
            return len(data)
        elif data:
            return 1
        return 0
    
    def get_verification_action(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        获取需要执行的验证动作(供 Agent 调用)
        
        Returns:
            Optional[Tuple[str, Dict]]: (action_name, params) 或 None
        """
        if not self._intent_type:
            return None
        
        rule = INTENT_VERIFICATION_RULES.get(self._intent_type)
        if not rule:
            return None
        
        required_resources = rule.get("required_resources", [])
        if not required_resources:
            return None
        
        region_id = self._blackboard.get_global("RegionId") or "cn-hangzhou"
        
        # 返回第一个缺少资源的验证动作
        for res_spec in required_resources:
            res_type = res_spec["type"]
            res_key = res_spec["key"]
            verify_action = res_spec["verify_action"]
            
            resource_id = self._get_resource_id_from_blackboard(res_type, res_key)
            if not resource_id:
                return verify_action, {"RegionId": region_id}
        
        # 所有资源都存在，返回第一个资源的验证动作
        first_spec = required_resources[0]
        return first_spec["verify_action"], {"RegionId": region_id}
