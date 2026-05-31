import importlib
import logging
import random
import time
import os
from typing import Dict, Any, List, Optional
from execution.aliyun_gym.core.api_validator import ApiValidator

logger = logging.getLogger(__name__)

# Product to handler module mapping
# 注意: EIP 类 API 属于 VPC 产品，不是独立产品
PRODUCT_HANDLER_MAP = {
    "VPC": "vpc_handlers",      # 包含 EIP 相关 API (AllocateEipAddress, AssociateEipAddress)
    "ECS": "ecs_handlers",
    "SLB": "slb_handlers",
    "RDS": "rds_handlers",
    "REDIS": "redis_handlers",
    "CMS": "cms_handlers",
    "OOS": "oos_handlers",
}

class ActionRouter:
    """
    Dispatches API actions to handler functions.
    Supports product-based routing to avoid naming conflicts.
    """
    
    # 时延配置（秒）
    LATENCY_CONFIG = {
        "describe": 1.0,          # Describe* 接口: 1秒
        "default_min": 3.0,       # 其他接口: 3-5秒
        "default_max": 5.0,
    }
    
    def __init__(self, state_store, chaos_injector, time_keeper, 
                 enable_latency: bool = True, 
                 use_real_latency: bool = False):
        self.state_store = state_store
        self.chaos_injector = chaos_injector
        self.time_keeper = time_keeper
        self.enable_latency = enable_latency      # 是否启用时延模拟
        self.use_real_latency = use_real_latency  # 是否使用真实等待（time.sleep）
        self._handler_cache: Dict[str, Any] = {}  # Cache loaded modules
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gym_dir = os.path.dirname(current_dir)
        kb_path = os.path.join(gym_dir, "knowledge")
        self.validator = ApiValidator(kb_path)
        
        self._preload_handlers()

    def _preload_handlers(self):
        """Pre-load all handler modules."""
        for product, module_name in PRODUCT_HANDLER_MAP.items():
            if module_name not in self._handler_cache:
                try:
                    module = importlib.import_module(f"execution.aliyun_gym.handlers.{module_name}")
                    self._handler_cache[module_name] = module
                except ImportError as e:
                    logger.warning(f"Handler module {module_name} not found: {e}")

    def dispatch(self, action: str, query: Dict[str, Any], product: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatch action to the appropriate handler.
        
        Args:
            action: API action name (e.g., 'CreateInstance')
            query: Request parameters
            product: Product identifier (e.g., 'VPC', 'ECS', 'REDIS')
                     Required for APIs with naming conflicts.
        """
        # 1. Validate Request (Parameter Hallucinations, Invalid Types, etc.)
        if product:
            validation_error = self.validator.validate(action, query, product)
            if validation_error:
                # Add RequestId if missing, although real API adds it on error too
                if "RequestId" not in validation_error:
                    import uuid
                    validation_error["RequestId"] = str(uuid.uuid4())
                return validation_error

        # 模拟时延
        latency = self._get_latency(action)
        if self.enable_latency and latency > 0:
            # 虚拟时间记录
            self.time_keeper.tick(latency)
            # 真实等待（可选）
            if self.use_real_latency:
                time.sleep(latency)
        
        handler_name = f"handle_{action}"
        
        # If product is specified, look in that product's handler module
        if product:
            product_upper = product.upper()
            module_name = PRODUCT_HANDLER_MAP.get(product_upper)
            if module_name and module_name in self._handler_cache:
                module = self._handler_cache[module_name]
                handler = self._find_handler_in_module(module, handler_name)
                if handler:
                    return handler(query, self.state_store, self.chaos_injector, self.time_keeper)
        
        # Fallback: search all modules (for backward compatibility)
        for module_name, module in self._handler_cache.items():
            handler = self._find_handler_in_module(module, handler_name)
            if handler:
                return handler(query, self.state_store, self.chaos_injector, self.time_keeper)
        
        return {
            "Code": "UnsupportedAction",
            "Message": f"Action {action} (product={product}) is not mocked yet."
        }
    
    def _find_handler_in_module(self, module, handler_name: str):
        """Find handler function in module (case-insensitive)."""
        for attr_name in dir(module):
            if attr_name.lower() == handler_name.lower():
                return getattr(module, attr_name)
        return None
    
    def _get_latency(self, action: str) -> float:
        """
        根据 action 名称获取时延（秒）
        
        - Describe* 接口: 1秒
        - 其他接口: 随机 3-5秒
        """
        if action.startswith("Describe"):
            return self.LATENCY_CONFIG["describe"]
        else:
            min_lat = self.LATENCY_CONFIG["default_min"]
            max_lat = self.LATENCY_CONFIG["default_max"]
            return random.uniform(min_lat, max_lat)
    
    def set_latency_enabled(self, enabled: bool) -> None:
        """启用/禁用时延模拟"""
        self.enable_latency = enabled
    
    def get_elapsed_time(self) -> float:
        """获取已进行的虚拟时间（秒）"""
        return self.time_keeper.now()
