import json
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import os

class IOMiner:
    """
    IO 规则挖掘器 (Parameter Dependency Miner)
    专注于数据流向分析：Inputs -> Actions -> Outputs
    """
    def __init__(self):
        # 存储挖掘出的规则: Action -> {inputs, outputs, product}
        # inputs/outputs 使用 set 去重
        self.io_rules: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"inputs": set(), "outputs": set(), "product": "UNKNOWN"}
        )
        
        # 忽略的非业务字段
        self.ignored_output_keys = {"RequestId", "Code", "Message", "Success", "PageNumber", "PageSize", "TotalCount"}

    def load_trace(self, trace_path: str):
        """加载并分析 Trace 文件"""
        try:
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)
            
            actions = trace_data.get("actions", [])
            self._analyze_actions(actions)
        except Exception as e:
            print(f"Error loading trace {trace_path}: {e}")

    def load_traces_from_dir(self, dir_path: str):
        """批量加载目录下所有 JSON Trace"""
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return

        count = 0
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                full_path = os.path.join(dir_path, filename)
                self.load_trace(full_path)
                count += 1
        print(f"Processed {count} trace files from {dir_path}")

    def _analyze_actions(self, actions: List[Dict[str, Any]]):
        """分析 Action 序列中的参数流"""
        for action in actions:
            action_name = action["action_name"]
            params = action.get("params", {})
            result = action.get("result", {})
            product = action.get("product", "UNKNOWN")
            
            # 1. 记录产品信息 (如果有更具体的就更新)
            if product != "UNKNOWN":
                self.io_rules[action_name]["product"] = product
            
            # 2. 分析输入 (Params)
            # 策略：Trace 中出现的所有参数都视为该场景下的必要输入
            for key in params.keys():
                self.io_rules[action_name]["inputs"].add(key)
                
            # 3. 分析输出 (Result)
            self._register_outputs(action_name, result)

    def _register_outputs(self, action_name: str, result: Any, prefix: str = ""):
        """递归注册输出字段"""
        if isinstance(result, dict):
            for k, v in result.items():
                if k in self.ignored_output_keys:
                    continue
                
                # 记录顶层输出 Key (IO Registry 通常只关心顶层 Key)
                if not prefix: 
                    self.io_rules[action_name]["outputs"].add(k)
                
                # 递归处理嵌套结构
                new_key = f"{prefix}.{k}" if prefix else k
                self._register_outputs(action_name, v, new_key)
                
        elif isinstance(result, list):
            for item in result:
                self._register_outputs(action_name, item, prefix)

    def export_io_registry(self) -> Dict[str, Any]:
        """导出为 IORegistry 兼容格式"""
        registry = {}
        for action, rules in self.io_rules.items():
            registry[action] = {
                "inputs": sorted(list(rules["inputs"])),
                "outputs": sorted(list(rules["outputs"])),
                "product": rules["product"]
            }
        return registry
