"""IO Registry - API 输入输出依赖注册表

定义每个 API 的输入依赖和输出产出,用于:
1. IO Guard: 检查参数是否就绪
2. 自动参数填充(支持分层 Blackboard)
3. 资源注册(创建类 API 自动注册到资源表)
4. 循环展开(列表参数)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from execution.cloudops_agent.memory.blackboard import Blackboard

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class IOSpec:
    """单个 API 的 IO 规格"""
    inputs: List[str] = field(default_factory=list)   # 必须的输入参数
    outputs: List[str] = field(default_factory=list)  # 产出的参数
    optional_inputs: List[str] = field(default_factory=list)  # 可选输入
    product: str = ""  # 所属产品(VPC/ECS/SLB 等)
    is_create: bool = False  # 是否是创建类 API(用于资源注册)
    resource_id_key: str = ""  # 主资源 ID 的 key(如 InstanceId, VpcId)


class IORegistry:
    """IO 注册表 - 核心组件"""
    
    # 默认的核心链路 IO 规则(VPC -> ECS -> SLB)
    DEFAULT_REGISTRY: Dict[str, IOSpec] = {
        # VPC 相关
        "CreateVpc": IOSpec(
            inputs=["RegionId"],
            outputs=["VpcId", "VRouterId", "RouteTableId"],
            optional_inputs=["CidrBlock", "VpcName"],
            product="VPC",
            is_create=True,
            resource_id_key="VpcId"
        ),
        "DescribeVpcs": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["VpcId"],
            product="VPC"
        ),
        "CreateVSwitch": IOSpec(
            inputs=["VpcId", "ZoneId", "CidrBlock"],
            outputs=["VSwitchId"],
            optional_inputs=["VSwitchName"],
            product="VPC",
            is_create=True,
            resource_id_key="VSwitchId"
        ),
        "DescribeVSwitches": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["VpcId", "VSwitchId"],
            product="VPC"
        ),
        "DescribeZones": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            product="VPC"
        ),
        
        # 安全组相关
        "CreateSecurityGroup": IOSpec(
            inputs=["VpcId", "RegionId"],
            outputs=["SecurityGroupId", "SecurityGroupCreated"],  # 添加虚拟标记
            optional_inputs=["SecurityGroupName", "Description"],
            product="ECS",
            is_create=True,
            resource_id_key="SecurityGroupId"
        ),
        "AuthorizeSecurityGroup": IOSpec(
            inputs=["RegionId", "SecurityGroupId", "IpProtocol", "PortRange", "SecurityGroupCreated"],  # 依赖安全组创建完成
            outputs=["SecurityGroupRulesConfigured"],  # 产出规则配置完成标记
            optional_inputs=["SourceCidrIp", "Policy"],
            product="ECS"
        ),
        "DescribeSecurityGroups": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["VpcId", "SecurityGroupId"],
            product="ECS"
        ),
        
        # ECS 相关
        "RunInstances": IOSpec(
            inputs=["RegionId", "VSwitchId", "SecurityGroupId", "InstanceType", "ImageId", 
                    "SecurityGroupRulesConfigured"],  # 依赖安全组授权完成
            outputs=["InstanceIds", "InstancesCreated"],  # 添加虚拟标记
            optional_inputs=["InstanceName", "Amount", "SystemDiskCategory", "SystemDiskSize"],
            product="ECS",
            is_create=True,
            resource_id_key="InstanceIds"
        ),
        "DescribeInstances": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["InstanceIds", "VpcId"],
            product="ECS"
        ),
        "StartInstance": IOSpec(
            inputs=["InstanceId"],
            outputs=[],
            product="ECS"
        ),
        "StopInstance": IOSpec(
            inputs=["InstanceId"],
            outputs=[],
            optional_inputs=["ForceStop"],
            product="ECS"
        ),
        "DescribeInstanceTypes": IOSpec(
            inputs=[],
            outputs=[],
            optional_inputs=["InstanceTypeFamily"],
            product="ECS"
        ),
        "DescribeImages": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["ImageName"],
            product="ECS"
        ),
        "DescribeAvailableResource": IOSpec(
            inputs=["RegionId"],
            outputs=["AvailableZones"],
            optional_inputs=["DestinationResource", "InstanceChargeType"],
            product="ECS"
        ),
        
        # SLB 相关
        "CreateLoadBalancer": IOSpec(
            inputs=["RegionId", "VSwitchId"],
            outputs=["LoadBalancerId", "Address", "LoadBalancerCreated"],  # 添加虚拟标记
            optional_inputs=["LoadBalancerName", "AddressType", "LoadBalancerSpec"],
            product="SLB",
            is_create=True,
            resource_id_key="LoadBalancerId"
        ),
        "AddBackendServers": IOSpec(
            inputs=["LoadBalancerId", "BackendServers", "LoadBalancerCreated", "InstancesCreated"],  # 依赖 SLB 和 ECS 创建完成
            outputs=[],
            product="SLB"
        ),
        "DescribeLoadBalancers": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["LoadBalancerId", "VpcId"],
            product="SLB"
        ),
        "CreateLoadBalancerTCPListener": IOSpec(
            inputs=["LoadBalancerId", "ListenerPort", "BackendServerPort"],
            outputs=[],
            optional_inputs=["Bandwidth", "HealthCheckConnectPort"],
            product="SLB"
        ),
        "CreateLoadBalancerHTTPListener": IOSpec(
            inputs=["LoadBalancerId", "ListenerPort", "BackendServerPort", "HealthCheck", "LoadBalancerCreated"],  # 依赖 SLB 创建
            outputs=["HTTPListenerCreated"],  # 产出 Listener 创建标记
            optional_inputs=["Bandwidth", "StickySession"],
            product="SLB"
        ),
        "StartLoadBalancerListener": IOSpec(
            inputs=["LoadBalancerId", "ListenerPort", "ListenerProtocol", "HTTPListenerCreated"],  # 依赖 Listener 创建
            outputs=[],
            product="SLB"
        ),
        "CreateAccessControlList": IOSpec(
            inputs=["RegionId", "AclName"],
            outputs=["AclId"],
            optional_inputs=["AddressIPVersion"],
            product="SLB",
            is_create=True,
            resource_id_key="AclId"
        ),
        "AddAccessControlListEntry": IOSpec(
            inputs=["AclId", "AclEntrys"],
            outputs=[],
            product="SLB"
        ),
        "DescribeLoadBalancerAttribute": IOSpec(
            inputs=["LoadBalancerId"],
            outputs=[],
            product="SLB"
        ),
        "DescribeLoadBalancerHTTPListenerAttribute": IOSpec(
            inputs=["LoadBalancerId", "ListenerPort"],
            outputs=[],
            product="SLB"
        ),
        "DescribeLoadBalancerTCPListenerAttribute": IOSpec(
            inputs=["LoadBalancerId", "ListenerPort"],
            outputs=[],
            product="SLB"
        ),
        "DescribeLoadBalancerListeners": IOSpec(
            inputs=["LoadBalancerId"],
            outputs=[],
            optional_inputs=["ListenerProtocol"],
            product="SLB"
        ),

        # EIP 相关(P1 扩展)
        "AllocateEipAddress": IOSpec(
            inputs=["RegionId"],
            outputs=["AllocationId", "EipAddress", "EipAllocated"],  # 添加虚拟标记
            optional_inputs=["Bandwidth", "InternetChargeType"],
            product="EIP",
            is_create=True,
            resource_id_key="AllocationId"
        ),
        "AssociateEipAddress": IOSpec(
            inputs=["AllocationId", "InstanceId", "EipAllocated", "LoadBalancerCreated"],  # 依赖 EIP 和 SLB/ECS
            outputs=[],
            optional_inputs=["InstanceType"],
            product="EIP"
        ),

        # RDS 相关
        "CreateDBInstance": IOSpec(
            inputs=["RegionId", "Engine", "EngineVersion", "DBInstanceClass", "DBInstanceStorage", "DBInstanceNetType", "SecurityIPList", "PayType"],
            outputs=["DBInstanceId", "OrderId", "ConnectionString", "Port", "DBInstanceCreated"],  # 添加虚拟标记
            optional_inputs=["VpcId", "VSwitchId", "ZoneId", "DBInstanceDescription"],
            product="RDS",
            is_create=True,
            resource_id_key="DBInstanceId"
        ),
        "CreateAccount": IOSpec(
            inputs=["DBInstanceId", "AccountName", "AccountPassword", "DBInstanceCreated"],  # 依赖 RDS 创建完成
            outputs=[],
            optional_inputs=["AccountType", "AccountDescription"],
            product="RDS"
        ),
        "DescribeAvailableZones": IOSpec(
            inputs=["RegionId"],
            outputs=["AvailableZones"],
            optional_inputs=["Engine", "EngineVersion"],
            product="RDS"
        ),
        "ListClasses": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["Engine", "EngineVersion"],
            product="RDS"
        ),
        "AllocateInstancePublicConnection": IOSpec(
            inputs=["DBInstanceId", "ConnectionStringPrefix", "Port"],
            outputs=["ConnectionString"],
            product="RDS"
        ),
        "DescribeDBInstanceAttribute": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),
        "DescribeAccounts": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),
        "DescribeDBInstanceIPArrayList": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),
        "DescribeDBInstanceNetInfoForChannel": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            optional_inputs=["DBInstanceNetType"],
            product="RDS"
        ),
        "DescribeDBInstances": IOSpec(
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["DBInstanceId", "VpcId"],
            product="RDS"
        ),
        "ModifySecurityIps": IOSpec(
            inputs=["DBInstanceId", "SecurityIps", "DBInstanceCreated", "InstancesCreated"],  # 依赖 RDS 和 ECS 创建完成
            outputs=[],
            product="RDS"
        ),
        "StartDBInstance": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),
        "StopDBInstance": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),
        "RestartDBInstance": IOSpec(
            inputs=["DBInstanceId"],
            outputs=[],
            product="RDS"
        ),

        # Redis
        "CreateInstance": IOSpec(
            inputs=["RegionId", "InstanceClass"],
            outputs=["InstanceId", "ConnectionDomain", "Port", "RedisCreated"],  # 添加虚拟标记
            optional_inputs=["InstanceName", "Password", "VpcId", "VSwitchId", "ZoneId", "ChargeType", "NetworkType", "Token"],
            product="Redis",
            is_create=True,
            resource_id_key="InstanceId"
        ),
        "DescribeInstanceAttribute": IOSpec(
            inputs=["InstanceId", "RedisCreated"],  # 依赖 Redis 创建完成
            outputs=[],
            product="Redis"
        ),
        "DescribeRedisInstances": IOSpec(  # 避免与 ECS DescribeInstances 冲突
            inputs=["RegionId"],
            outputs=[],
            optional_inputs=["InstanceIds", "VpcId"],
            product="Redis"
        ),
        "DescribeDBInstanceNetInfo": IOSpec(
            inputs=["InstanceId"],
            outputs=[],
            product="Redis"
        ),
        "DescribeSecurityIps": IOSpec(
            inputs=["InstanceId"],
            outputs=[],
            product="Redis"
        ),
        
        # CMS
        "CreateMonitorGroup": IOSpec(
            inputs=["GroupName"],
            outputs=["GroupId"],
            optional_inputs=["ContactGroups"],
            product="CMS",
            is_create=True,
            resource_id_key="GroupId"
        ),
        "CreateMonitorGroupInstances": IOSpec(
            inputs=["GroupId", "Instances"],
            outputs=[],
            product="CMS"
        ),
        "CreateHostAvailability": IOSpec(
            inputs=["TaskName", "TaskType", "Instances"],
            outputs=["TaskId"],
            optional_inputs=["GroupId", "AlertConfig"],
            product="CMS",
            is_create=True,
            resource_id_key="TaskId"
        ),
        "DescribeMetricLast": IOSpec(
            inputs=["MetricName", "Namespace"],
            outputs=[],
            optional_inputs=["Dimensions", "Period"],
            product="CMS"
        ),
        
        # OOS
        "StartExecution": IOSpec(
            inputs=["TemplateName"],
            outputs=["ExecutionId"],
            optional_inputs=["Parameters", "RegionId"],
            product="OOS",
            is_create=True,
            resource_id_key="ExecutionId"
        ),
    }
    
    def __init__(self, custom_registry: Optional[Dict[str, IOSpec]] = None):
        """初始化 IO 注册表"""
        self._registry: Dict[str, IOSpec] = dict(self.DEFAULT_REGISTRY)
        if custom_registry:
            self._registry.update(custom_registry)
    
    def get_spec(self, action: str) -> Optional[IOSpec]:
        """获取指定 action 的 IO 规格"""
        return self._registry.get(action)
    
    def get_required_inputs(self, action: str) -> List[str]:
        """获取 action 的必须输入参数"""
        spec = self.get_spec(action)
        return spec.inputs if spec else []
    
    def get_outputs(self, action: str) -> List[str]:
        """获取 action 的输出参数"""
        spec = self.get_spec(action)
        return spec.outputs if spec else []
    
    def get_product(self, action: str) -> str:
        """获取 action 所属产品"""
        spec = self.get_spec(action)
        return spec.product if spec else "UNKNOWN"
    
    def check_inputs_ready(self, action: str, blackboard: "Blackboard") -> tuple[bool, List[str]]:
        """
        检查 action 的输入参数是否在 blackboard 中就绪
        
        Args:
            action: API 名称
            blackboard: 分层参数黑板
            
        Returns:
            (is_ready, missing_params)
        """
        spec = self.get_spec(action)
        if not spec:
            return True, []
        
        product = spec.product.lower()
        missing = []
        
        for param in spec.inputs:
            # 特殊参数检查：某些参数可以通过转换生成
            if self._can_derive_param(action, param, blackboard):
                continue
                
            # 使用智能解析: 全局 -> 命名空间
            value = blackboard.resolve(product, param)
            if value is None:
                missing.append(param)
        
        return len(missing) == 0, missing
    
    def _can_derive_param(self, action: str, param: str, blackboard: "Blackboard") -> bool:
        """
        检查参数是否可以通过转换逻辑生成
        
        某些参数虽然不直接存在于 blackboard，但可以从其他参数转换得到
        """
        # AddBackendServers 的 BackendServers 可以从 InstanceIds 转换
        if action == "AddBackendServers" and param == "BackendServers":
            instance_ids = blackboard.get_global("InstanceIds")
            return instance_ids is not None and len(instance_ids) > 0
        
        # AssociateEipAddress 的 InstanceId 可以使用 LoadBalancerId
        if action == "AssociateEipAddress" and param == "InstanceId":
            lb_id = blackboard.get_global("LoadBalancerId")
            return lb_id is not None
        
        return False
    
    def fill_params_from_blackboard(self, action: str, blackboard: "Blackboard") -> Dict[str, Any]:
        """
        从分层 blackboard 自动填充参数
        
        Args:
            action: API 名称
            blackboard: 分层参数黑板
            
        Returns:
            参数字典
        """
        spec = self.get_spec(action)
        if not spec:
            return {}
        
        product = spec.product.lower()
        params = {}
        all_inputs = spec.inputs + spec.optional_inputs
        
        for param in all_inputs:
            # 使用智能解析: 全局 -> 命名空间
            value = blackboard.resolve(product, param)
            if value is not None:
                params[param] = value
        
        # 特殊参数转换逻辑
        params = self._apply_param_transformations(action, params, blackboard)
        
        return params
    
    def _apply_param_transformations(self, action: str, params: Dict[str, Any], 
                                       blackboard: "Blackboard") -> Dict[str, Any]:
        """
        应用特殊参数转换逻辑
        
        某些 API 需要特殊格式的参数，这里进行自动转换
        """
        import json
        
        # AddBackendServers: 将 InstanceIds 转换为 BackendServers JSON 数组
        if action == "AddBackendServers":
            if "BackendServers" not in params or not params.get("BackendServers"):
                instance_ids = blackboard.get_global("InstanceIds")
                if instance_ids:
                    if isinstance(instance_ids, list):
                        backend_servers = [{"ServerId": iid, "Weight": 100} for iid in instance_ids]
                    else:
                        backend_servers = [{"ServerId": instance_ids, "Weight": 100}]
                    params["BackendServers"] = json.dumps(backend_servers)
        
        # AssociateEipAddress: 将 LoadBalancerId 作为 InstanceId（绑定 EIP 到 SLB）
        if action == "AssociateEipAddress":
            if "InstanceId" not in params or not params.get("InstanceId"):
                # 优先使用 LoadBalancerId（EIP 绑定到 SLB 场景）
                lb_id = blackboard.get_global("LoadBalancerId")
                if lb_id:
                    params["InstanceId"] = lb_id
                    params["InstanceType"] = "SlbInstance"
        
        return params
    
    def write_outputs_to_blackboard(self, action: str, result: Dict[str, Any], 
                                    blackboard: "Blackboard",
                                    resource_name: str = "") -> Optional[str]:
        """
        将 action 的输出写入 blackboard
        
        - 创建类 API: 注册到资源表 + 写入全局
        - 查询类 API: 只写入全局
        
        Args:
            action: API 名称
            result: API 返回结果
            blackboard: 分层参数黑板
            resource_name: 可选的资源名称,空则自动生成
            
        Returns:
            资源名称(如果是创建类 API)
        """
        spec = self.get_spec(action)
        if not spec:
            return None
        
        outputs = spec.outputs
        registered_name = None
        
        # 如果是创建类 API,注册到资源表
        if spec.is_create:
            # 提取输出属性
            attrs = {}
            for output in outputs:
                if output in result:
                    attrs[output] = result[output]
            
            # 特殊处理: RunInstances 返回的是 InstanceIdSets
            if action == "RunInstances" and "InstanceIdSets" in result:
                id_set = result["InstanceIdSets"]
                if isinstance(id_set, dict) and "InstanceIdSet" in id_set:
                    attrs["InstanceIds"] = id_set["InstanceIdSet"]
            
            if attrs:
                registered_name = blackboard.register_resource(
                    name=resource_name,
                    resource_type=spec.product,
                    action=action,
                    **attrs
                )
        
        # 同时写入全局(方便快速访问最近创建的资源)
        for output in outputs:
            if output in result:
                blackboard.set_global(output, result[output])
        
        # 特殊处理: RunInstances
        if action == "RunInstances" and "InstanceIdSets" in result:
            id_set = result["InstanceIdSets"]
            if isinstance(id_set, dict) and "InstanceIdSet" in id_set:
                blackboard.set_global("InstanceIds", id_set["InstanceIdSet"])
        
        return registered_name
    
    def list_actions(self) -> List[str]:
        """列出所有注册的 action"""
        return list(self._registry.keys())
    
    def list_actions_by_product(self, product: str) -> List[str]:
        """列出指定产品的所有 action"""
        return [
            action for action, spec in self._registry.items()
            if spec.product.upper() == product.upper()
        ]
    
    @classmethod
    def from_yaml(cls, path: str) -> "IORegistry":
        """从 YAML 文件加载"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        custom_registry = {}
        for action, spec_data in data.items():
            custom_registry[action] = IOSpec(
                inputs=spec_data.get('inputs', []),
                outputs=spec_data.get('outputs', []),
                optional_inputs=spec_data.get('optional_inputs', []),
                product=spec_data.get('product', '')
            )
        
        return cls(custom_registry)
    
    def to_yaml(self, path: str) -> None:
        """导出为 YAML 文件"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        data = {}
        for action, spec in self._registry.items():
            data[action] = {
                'inputs': spec.inputs,
                'outputs': spec.outputs,
                'optional_inputs': spec.optional_inputs,
                'product': spec.product
            }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# 全局单例
_default_registry: Optional[IORegistry] = None


def get_io_registry() -> IORegistry:
    """获取全局 IO 注册表"""
    global _default_registry
    if _default_registry is None:
        _default_registry = IORegistry()
    return _default_registry
