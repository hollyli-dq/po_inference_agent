"""
Aliyun-Gym 工具适配器

将 Aliyun-Gym 的 Mock Client 封装为统一的工具接口，供 Agent 调用。
"""
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import json

from execution.aliyun_gym.factory import create_gym_env, AliyunGymEnv, reset_env
from execution.cloudops_agent.knowledge.io_registry import IORegistry, get_io_registry


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    action: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class GymToolAdapter:
    """
    Aliyun-Gym 工具适配器
    
    将 Mock Client 的 API 调用封装为统一接口
    """
    
    # Action 到 Client 和方法的映射
    ACTION_MAP = {
        # VPC
        "CreateVpc": ("vpc_client", "create_vpc"),
        "DescribeVpcs": ("vpc_client", "describe_vpcs"),
        "CreateVSwitch": ("vpc_client", "create_vswitch"),
        "DescribeVSwitches": ("vpc_client", "describe_vswitches"),
        "DescribeZones": ("vpc_client", "describe_zones"),
        "AllocateEipAddress": ("vpc_client", "allocate_eip_address"),
        "AssociateEipAddress": ("vpc_client", "associate_eip_address"),
        
        # ECS
        "CreateSecurityGroup": ("ecs_client", "create_security_group"),
        "AuthorizeSecurityGroup": ("ecs_client", "authorize_security_group"),
        "DescribeSecurityGroups": ("ecs_client", "describe_security_groups"),
        "DescribeSecurityGroupAttribute": ("ecs_client", "describe_security_group_attribute"),
        "RunInstances": ("ecs_client", "run_instances"),
        "DescribeInstances": ("ecs_client", "describe_instances"),
        "StartInstance": ("ecs_client", "start_instance"),
        "StopInstance": ("ecs_client", "stop_instance"),
        "RebootInstance": ("ecs_client", "reboot_instance"),
        "DeleteInstances": ("ecs_client", "delete_instances"),
        "DescribeInstanceTypes": ("ecs_client", "describe_instance_types"),
        "DescribeImages": ("ecs_client", "describe_images"),
        "DescribeAvailableResource": ("ecs_client", "describe_available_resource"),
        "DescribePrice": ("ecs_client", "describe_price"),
        "DescribeRegions": ("ecs_client", "describe_regions"),
        "DescribeAccountAttributes": ("ecs_client", "describe_account_attributes"),
        "ReplaceSystemDisk": ("ecs_client", "replace_system_disk"),
        
        # SLB
        "CreateLoadBalancer": ("slb_client", "create_load_balancer"),
        "AddBackendServers": ("slb_client", "add_backend_servers"),
        "DescribeLoadBalancers": ("slb_client", "describe_load_balancers"),
        "CreateLoadBalancerHTTPListener": ("slb_client", "create_load_balancer_http_listener"),
        "CreateLoadBalancerTCPListener": ("slb_client", "create_load_balancer_tcp_listener"),
        "StartLoadBalancerListener": ("slb_client", "start_load_balancer_listener"),
        "CreateAccessControlList": ("slb_client", "create_access_control_list"),
        "AddAccessControlListEntry": ("slb_client", "add_access_control_list_entry"),
        "DescribeLoadBalancerAttribute": ("slb_client", "describe_load_balancer_attribute"),
        "DescribeLoadBalancerHTTPListenerAttribute": ("slb_client", "describe_load_balancer_http_listener_attribute"),
        "DescribeLoadBalancerTCPListenerAttribute": ("slb_client", "describe_load_balancer_tcp_listener_attribute"),
        "DescribeLoadBalancerListeners": ("slb_client", "describe_load_balancer_listeners"),
        
        # EIP (Moved to VPC client in some versions, but here kept separate if using eip_client mock)
        # Assuming vpc_handlers handles EIP logic as seen in file list, but adapter previously used eip_client.
        # Checking vpc_handlers.py, it has handle_AllocateEipAddress. 
        # So it should be vpc_client.
        # "AllocateEipAddress": ("eip_client", "allocate_eip_address"), 
        # "AssociateEipAddress": ("eip_client", "associate_eip_address"),

        # RDS
        "CreateDBInstance": ("rds_client", "create_db_instance"),
        "DescribeDBInstanceAttribute": ("rds_client", "describe_db_instance_attribute"),
        "AllocateInstancePublicConnection": ("rds_client", "allocate_instance_public_connection"),
        "CreateAccount": ("rds_client", "create_account"),
        "DescribeAvailableZones": ("rds_client", "describe_available_zones"),
        "ListClasses": ("rds_client", "list_classes"),
        "DescribeAccounts": ("rds_client", "describe_accounts"),
        "DescribeDBInstanceIPArrayList": ("rds_client", "describe_db_instance_ip_array_list"),
        "DescribeDBInstanceNetInfoForChannel": ("rds_client", "describe_db_instance_net_info_for_channel"),
        "DescribeDBInstances": ("rds_client", "describe_db_instances"),
        "ModifySecurityIps": ("rds_client", "modify_security_ips"),
        "StartDBInstance": ("rds_client", "start_db_instance"),
        "StopDBInstance": ("rds_client", "stop_db_instance"),
        "RestartDBInstance": ("rds_client", "restart_db_instance"),
        
        # Redis
        "CreateInstance": ("redis_client", "create_instance"),
        "DescribeInstanceAttribute": ("redis_client", "describe_instance_attribute"),
        "DescribeRedisInstances": ("redis_client", "describe_instances"),  # 避免与 ECS DescribeInstances 冲突
        "DescribeDBInstanceNetInfo": ("redis_client", "describe_db_instance_net_info"),
        "DescribeSecurityIps": ("redis_client", "describe_security_ips"),
        
        # CMS
        "CreateMonitorGroup": ("cms_client", "create_monitor_group"),
        "CreateMonitorGroupInstances": ("cms_client", "create_monitor_group_instances"),
        "CreateHostAvailability": ("cms_client", "create_host_availability"),
        "DescribeMetricLast": ("cms_client", "describe_metric_last"),

        # OOS
        "StartExecution": ("oos_client", "start_execution"),
    }
    
    def __init__(self, failure_rate: float = 0.0, 
                 io_registry: Optional[IORegistry] = None,
                 use_real_latency: bool = False):
        """
        初始化适配器
        
        Args:
            failure_rate: 随机失败率（0.0-1.0），用于混沌测试
            io_registry: IO 注册表，用于参数填充
            use_real_latency: 是否使用真实时延（time.sleep）
        """
        self.env = create_gym_env(
            failure_rate=failure_rate, 
            enable_latency=True,
            use_real_latency=use_real_latency
        )
        self.io_registry = io_registry or get_io_registry()
        self._dry_run = False
    
    def set_dry_run(self, enabled: bool) -> None:
        """设置干跑模式"""
        self._dry_run = enabled
    
    def reset(self) -> None:
        """重置环境"""
        reset_env(self.env)
    
    def execute(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """
        执行 API 调用
        
        Args:
            action: API 名称（如 CreateVpc）
            params: 参数字典
            
        Returns:
            ToolResult 包含执行结果或错误信息
        """
        if self._dry_run:
            return ToolResult(
                success=True,
                action=action,
                result={"dry_run": True, "action": action, "params": params}
            )
        
        if action not in self.ACTION_MAP:
            return ToolResult(
                success=False,
                action=action,
                error=f"Unknown action: {action}",
                error_code="UnknownAction"
            )
        
        client_name, method_name = self.ACTION_MAP[action]
        client = getattr(self.env, client_name)
        
        try:
            # 直接调用 call_api 方法
            result = self._call_api_direct(client, action, params)
            
            # 检查是否是业务错误（仿真器返回的错误响应）
            # 识别条件：返回值包含 "Code" 字段，且不包含 "RequestId"（或 Code 不是 "200"）
            # 这保持了仿真器真实性：真实 API 错误返回也是这种格式
            if isinstance(result, dict) and "Code" in result:
                if "RequestId" not in result and result["Code"] != "200":
                    return ToolResult(
                        success=False,
                        action=action,
                        error=result.get("Message", "API business error"),
                        error_code=result["Code"]
                    )
            
            return ToolResult(
                success=True,
                action=action,
                result=result
            )
            
        except Exception as e:
            error_code = getattr(e, 'code', 'UnknownError')
            error_message = str(e)
            
            # 尝试从 TeaException 提取更多信息
            if hasattr(e, 'message'):
                error_message = e.message
            
            return ToolResult(
                success=False,
                action=action,
                error=error_message,
                error_code=error_code
            )
    
    def _call_api_direct(self, client, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        直接调用 API（绕过 SDK 的请求对象构建）
        """
        class MockRequest:
            def __init__(self, query: Dict[str, Any]):
                self.query = query
                self.body = None
        
        class MockParams:
            def __init__(self, action: str):
                self.action = action
        
        mock_params = MockParams(action)
        mock_request = MockRequest(params)
        mock_runtime = None
        
        # 调用 call_api
        response = client.call_api(mock_params, mock_request, mock_runtime)
        
        # 提取结果
        if isinstance(response, dict) and 'body' in response:
            return response['body']
        return response
    
    def execute_with_blackboard(self, action: str, blackboard: "Blackboard") -> ToolResult:
        """
        使用 Blackboard 自动填充参数并执行
        
        Args:
            action: API 名称
            blackboard: 参数黑板
            
        Returns:
            ToolResult
        """
        # 从 blackboard 填充参数
        params = self.io_registry.fill_params_from_blackboard(action, blackboard.to_dict())
        
        # 执行
        result = self.execute(action, params)
        
        # 如果成功，写回输出到 blackboard
        if result.success and result.result:
            self.io_registry.write_outputs_to_blackboard(
                action, result.result, blackboard._data
            )
            # 需要通过 set 方法写入以记录来源
            outputs = self.io_registry.get_outputs(action)
            for output in outputs:
                if output in result.result:
                    blackboard.set(output, result.result[output], 
                                  source="api_output", action=action)
        
        return result
    
    def get_available_actions(self) -> List[str]:
        """获取所有可用的 action"""
        return list(self.ACTION_MAP.keys())
    
    def get_tool_description(self, action: str) -> str:
        """获取工具描述（用于 LLM）"""
        spec = self.io_registry.get_spec(action)
        if not spec:
            return f"{action}: Unknown action"
        
        inputs = ", ".join(spec.inputs)
        outputs = ", ".join(spec.outputs) if spec.outputs else "None"
        optional = ", ".join(spec.optional_inputs) if spec.optional_inputs else "None"
        
        return f"""
{action} ({spec.product})
  Required inputs: {inputs}
  Optional inputs: {optional}
  Outputs: {outputs}
""".strip()
    
    def get_all_tool_descriptions(self) -> str:
        """获取所有工具描述"""
        descriptions = []
        for action in sorted(self.ACTION_MAP.keys()):
            descriptions.append(self.get_tool_description(action))
        return "\n\n".join(descriptions)
    
    def get_state_dump(self) -> Dict[str, Any]:
        """获取当前环境状态快照"""
        return self.env.state_store.dump()
    
    def get_virtual_time(self) -> float:
        """获取当前虚拟时间（秒）
        
        用于计算模拟执行耗时，支持并行提效分析
        """
        return self.env.clock.now()
    
    def reset_virtual_clock(self) -> None:
        """重置虚拟时钟"""
        self.env.clock.reset()


class ToolExecutor:
    """
    工具执行器 - 统一的工具调用入口
    
    支持：
    - Aliyun-Gym (仿真)
    - 未来可扩展真实 SDK
    """
    
    def __init__(self, adapter: Optional[GymToolAdapter] = None):
        self.adapter = adapter or GymToolAdapter()
    
    def execute(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """执行工具"""
        return self.adapter.execute(action, params)
    
    def reset(self) -> None:
        """重置环境"""
        self.adapter.reset()
