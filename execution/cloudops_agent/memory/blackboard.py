"""
Blackboard - 分层参数黑板

支持三层结构:
1. 全局参数 (global): 跨产品共享的基础参数，如 RegionId, ZoneId
2. 产品命名空间 (namespace): 产品级默认配置，如 rds.Engine = "MySQL"
3. 资源注册表 (resources): API 返回的资源实例，支持多实例

使用示例:
    bb = Blackboard()
    
    # 全局参数
    bb.set_global("RegionId", "cn-hangzhou")
    bb.get_global("RegionId")  # "cn-hangzhou"
    
    # 产品命名空间
    bb.set_ns("rds", "Engine", "MySQL")
    bb.get_ns("rds", "Engine")  # "MySQL"
    
    # 资源注册
    bb.register_resource("ecs_web_1", "ECS", "RunInstances", InstanceId="i-xxx")
    bb.get_resource("ecs_web_1")  # {"_type": "ECS", "InstanceId": "i-xxx", ...}
    bb.list_resources("ECS")  # [resource1, resource2, ...]
"""
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import copy
import uuid


class ParamSource(Enum):
    """参数来源"""
    INTENT = "intent"       # 从用户意图解析
    DEFAULT = "default"     # 系统默认值
    API_OUTPUT = "api_output"  # API 返回
    USER_INPUT = "user_input"  # 用户直接输入
    DERIVED = "derived"     # 推导得出


@dataclass
class ResourceEntry:
    """资源条目"""
    name: str               # 资源名称 (唯一标识)
    resource_type: str      # 资源类型: ECS, RDS, VPC 等
    action: str             # 创建该资源的 API
    attributes: Dict[str, Any]  # 资源属性
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "_name": self.name,
            "_type": self.resource_type,
            "_action": self.action,
            "_created_at": self.created_at.isoformat(),
            **self.attributes
        }


class Blackboard:
    """
    分层参数黑板
    
    结构:
    - _global: Dict[str, Any]           # 全局参数
    - _namespace: Dict[str, Dict]       # 产品命名空间
    - _resources: Dict[str, ResourceEntry]  # 资源注册表
    """
    
    # 已知产品列表
    KNOWN_PRODUCTS = {"vpc", "ecs", "rds", "redis", "slb", "oss", "cms", "oos"}
    
    def __init__(self):
        self._global: Dict[str, Any] = {}
        self._namespace: Dict[str, Dict[str, Any]] = {
            product: {} for product in self.KNOWN_PRODUCTS
        }
        self._resources: Dict[str, ResourceEntry] = {}
        self._history: List[Dict[str, Any]] = []
        self._resource_counter: Dict[str, int] = {}  # 用于生成资源名
    
    # ==================== 全局参数操作 ====================
    
    def set_global(self, key: str, value: Any, source: ParamSource = ParamSource.DEFAULT) -> None:
        """设置全局参数"""
        self._global[key] = value
        self._record_history("set_global", key=key, value=value, source=source.value)
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """获取全局参数"""
        return self._global.get(key, default)
    
    def has_global(self, key: str) -> bool:
        """检查全局参数是否存在"""
        return key in self._global
    
    def get_all_global(self) -> Dict[str, Any]:
        """获取所有全局参数"""
        return self._global.copy()
    
    # ==================== 命名空间操作 ====================
    
    def set_ns(self, product: str, key: str, value: Any) -> None:
        """设置产品命名空间中的参数"""
        product = product.lower()
        if product not in self._namespace:
            self._namespace[product] = {}
        self._namespace[product][key] = value
        self._record_history("set_ns", product=product, key=key, value=value)
    
    def get_ns(self, product: str, key: str, default: Any = None) -> Any:
        """获取产品命名空间中的参数"""
        product = product.lower()
        return self._namespace.get(product, {}).get(key, default)
    
    def has_ns(self, product: str, key: str) -> bool:
        """检查产品命名空间中的参数是否存在"""
        product = product.lower()
        return key in self._namespace.get(product, {})
    
    def get_all_ns(self, product: str) -> Dict[str, Any]:
        """获取产品的所有命名空间参数"""
        product = product.lower()
        return self._namespace.get(product, {}).copy()
    
    def set_ns_defaults(self, product: str, defaults: Dict[str, Any]) -> None:
        """批量设置产品的默认参数"""
        product = product.lower()
        if product not in self._namespace:
            self._namespace[product] = {}
        self._namespace[product].update(defaults)
        self._record_history("set_ns_defaults", product=product, defaults=defaults)
    
    # ==================== 资源注册表操作 ====================
    
    def register_resource(self, name: str, resource_type: str, action: str, 
                          **attributes) -> str:
        """
        注册资源实例
        
        Args:
            name: 资源名称，如 "ecs_web_1"。如果为空，自动生成
            resource_type: 资源类型，如 "ECS", "RDS"
            action: 创建该资源的 API，如 "RunInstances"
            **attributes: 资源属性，如 InstanceId="i-xxx"
            
        Returns:
            资源名称
        """
        # 自动生成名称
        if not name:
            name = self._generate_resource_name(resource_type)
        
        entry = ResourceEntry(
            name=name,
            resource_type=resource_type.upper(),
            action=action,
            attributes=attributes
        )
        self._resources[name] = entry
        self._record_history("register_resource", name=name, res_type=resource_type,
                            api_action=action, attributes=attributes)
        return name
    
    def _generate_resource_name(self, resource_type: str) -> str:
        """生成资源名称"""
        resource_type = resource_type.lower()
        count = self._resource_counter.get(resource_type, 0) + 1
        self._resource_counter[resource_type] = count
        return f"{resource_type}_{count}"
    
    def get_resource(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定资源"""
        entry = self._resources.get(name)
        return entry.to_dict() if entry else None
    
    def get_resource_attr(self, name: str, attr: str, default: Any = None) -> Any:
        """获取资源的指定属性"""
        entry = self._resources.get(name)
        if not entry:
            return default
        return entry.attributes.get(attr, default)
    
    def has_resource(self, name: str) -> bool:
        """检查资源是否存在"""
        return name in self._resources
    
    def list_resources(self, resource_type: str = None) -> List[Dict[str, Any]]:
        """
        列出资源
        
        Args:
            resource_type: 可选，按类型过滤，如 "ECS"
            
        Returns:
            资源列表
        """
        result = []
        for entry in self._resources.values():
            if resource_type is None or entry.resource_type.upper() == resource_type.upper():
                result.append(entry.to_dict())
        return result
    
    def get_resource_ids(self, resource_type: str, id_attr: str) -> List[str]:
        """
        获取某类型所有资源的 ID 列表
        
        Args:
            resource_type: 资源类型，如 "ECS"
            id_attr: ID 属性名，如 "InstanceId"
            
        Returns:
            ID 列表，如 ["i-xxx1", "i-xxx2"]
        """
        ids = []
        for entry in self._resources.values():
            if entry.resource_type.upper() == resource_type.upper():
                if id_attr in entry.attributes:
                    value = entry.attributes[id_attr]
                    # 处理列表类型的 ID (如 InstanceIdSets)
                    if isinstance(value, list):
                        ids.extend(value)
                    else:
                        ids.append(value)
        return ids
    
    def count_resources(self, resource_type: str = None) -> int:
        """统计资源数量"""
        if resource_type is None:
            return len(self._resources)
        return sum(1 for e in self._resources.values() 
                   if e.resource_type.upper() == resource_type.upper())
    
    # ==================== 智能参数解析 ====================
    
    def resolve(self, product: str, key: str, default: Any = None) -> Any:
        """
        智能解析参数
        
        优先级: 全局 > 产品命名空间 > 默认值
        
        Args:
            product: 产品名称
            key: 参数名
            default: 默认值
            
        Returns:
            参数值
        """
        # 1. 先查全局
        if self.has_global(key):
            return self.get_global(key)
        
        # 2. 再查产品命名空间
        if self.has_ns(product, key):
            return self.get_ns(product, key)
        
        # 3. 返回默认值
        return default
    
    def resolve_for_action(self, product: str, required_keys: List[str]) -> Dict[str, Any]:
        """
        为某个 API 解析所有需要的参数
        
        Args:
            product: 产品名称
            required_keys: 需要的参数列表
            
        Returns:
            参数字典
        """
        result = {}
        for key in required_keys:
            value = self.resolve(product, key)
            if value is not None:
                result[key] = value
        return result
    
    # ==================== 序列化与快照 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为完整字典"""
        return {
            "global": self._global.copy(),
            "namespace": {k: v.copy() for k, v in self._namespace.items() if v},
            "resources": {name: entry.to_dict() for name, entry in self._resources.items()}
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        导出为扁平字典（用于向 LLM 展示）
        
        格式:
        - 全局参数直接展示
        - 命名空间用 product.key 格式
        - 资源用 @resource_name.attr 格式
        """
        result = {}
        
        # 全局参数
        result.update(self._global)
        
        # 命名空间
        for product, params in self._namespace.items():
            for key, value in params.items():
                result[f"{product}.{key}"] = value
        
        # 资源摘要
        for name, entry in self._resources.items():
            # 只展示关键属性
            key_attrs = ["InstanceId", "VpcId", "VSwitchId", "DBInstanceId", 
                        "LoadBalancerId", "SecurityGroupId", "Status"]
            for attr in key_attrs:
                if attr in entry.attributes:
                    result[f"@{name}.{attr}"] = entry.attributes[attr]
        
        return result
    
    def snapshot(self) -> Dict[str, Any]:
        """创建深拷贝快照"""
        return copy.deepcopy(self.to_dict())
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典恢复"""
        if "global" in data:
            self._global = data["global"].copy()
        if "namespace" in data:
            for product, params in data["namespace"].items():
                self._namespace[product.lower()] = params.copy()
        # 注意: resources 暂不支持从字典恢复
    
    def clear(self) -> None:
        """清空所有数据"""
        self._global.clear()
        self._namespace = {product: {} for product in self.KNOWN_PRODUCTS}
        self._resources.clear()
        self._resource_counter.clear()
        self._record_history("clear")
    
    # ==================== 历史记录 ====================
    
    def _record_history(self, action: str, **kwargs) -> None:
        """记录变更历史"""
        self._history.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取变更历史"""
        return self._history.copy()
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self) -> str:
        return (f"Blackboard(global={len(self._global)}, "
                f"ns={sum(len(v) for v in self._namespace.values())}, "
                f"resources={len(self._resources)})")
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
    
    def summary(self) -> str:
        """生成摘要（用于调试）"""
        lines = ["=== Blackboard Summary ==="]
        
        lines.append("\n[Global]")
        for k, v in self._global.items():
            lines.append(f"  {k}: {v}")
        
        lines.append("\n[Namespace]")
        for product, params in self._namespace.items():
            if params:
                lines.append(f"  {product}:")
                for k, v in params.items():
                    lines.append(f"    {k}: {v}")
        
        lines.append("\n[Resources]")
        for name, entry in self._resources.items():
            lines.append(f"  {name} ({entry.resource_type}):")
            for k, v in entry.attributes.items():
                lines.append(f"    {k}: {v}")
        
        return "\n".join(lines)
