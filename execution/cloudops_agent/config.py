"""
配置管理模块
支持多模式切换、LLM 配置、开关控制
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ExecutionMode(Enum):
    """执行模式"""
    EXPERT = "expert"      # 纯偏序图，零 LLM
    HYBRID = "hybrid"      # 偏序图 + 降级
    EXPLORE = "explore"    # 纯 ReAct + RAG


@dataclass
class LLMConfig:
    """LLM 配置 - OpenAI 兼容协议"""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "qwen3-max"
    temperature: float = 0.0
    max_tokens: int = 2048
    enable_thinking: bool = False  # 是否启用深度思考模式 (qwen3-max 支持)


@dataclass
class LLMLayerConfig:
    """分层 LLM 配置"""
    intent_parse: LLMConfig = field(default_factory=lambda: LLMConfig(model="qwen3-max"))
    react_reasoning: LLMConfig = field(default_factory=lambda: LLMConfig(model="qwen3-max"))
    error_recovery: LLMConfig = field(default_factory=lambda: LLMConfig(model="qwen3-max"))


@dataclass  
class SwitchConfig:
    """细粒度开关配置"""
    # 偏序图相关
    poset_enabled: bool = True
    poset_path: str = "./poset/h_posteriors.json"
    poset_fallback_enabled: bool = True
    poset_strict_mode: bool = False
    #poset_edge_threshold: float = 0.8    # HPO 后验边概率阈值，avg_H[i][j] >= threshold 时添加边 Agent 已经不加载偏序图了。
    poset_io_guard_enabled: bool = False  # IO Guard：预防性参数检查。False=只依赖偏序图，API报错时降级
    
    # RAG 相关
    rag_enabled: bool = True
    rag_endpoint: str = "http://localhost:8080"
    
    # Trace 采集
    trace_enabled: bool = True
    trace_output_path: str = "./traces/"
    
    # 调试开关
    verbose: bool = False
    dry_run: bool = False


@dataclass
class AgentConfig:
    """主配置"""
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    switches: SwitchConfig = field(default_factory=SwitchConfig)
    llm: LLMLayerConfig = field(default_factory=LLMLayerConfig)
    max_retries: int = 3
    max_react_steps: int = 100
    max_verification_retries: int = 2  # 验证失败后的最大重试次数
    
    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """从 YAML 文件加载配置"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load YAML config. Install with: pip install pyyaml")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "AgentConfig":
        """从 JSON 文件加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """从字典构建配置"""
        config = cls()
        
        if 'execution_mode' in data:
            config.execution_mode = ExecutionMode(data['execution_mode'])
        
        if 'switches' in data:
            for key, value in data['switches'].items():
                if hasattr(config.switches, key):
                    setattr(config.switches, key, value)
        
        if 'llm' in data:
            for layer, llm_data in data['llm'].items():
                if hasattr(config.llm, layer):
                    llm_config = getattr(config.llm, layer)
                    for key, value in llm_data.items():
                        if hasattr(llm_config, key):
                            setattr(llm_config, key, value)
        
        for key in ['max_retries', 'max_react_steps', 'max_verification_retries']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @classmethod
    def preset_trace_collection(cls) -> "AgentConfig":
        """预设：阶段1 - Trace 积累（无偏序图）"""
        config = cls(execution_mode=ExecutionMode.EXPLORE)
        config.switches.poset_enabled = False
        config.switches.rag_enabled = True
        config.switches.trace_enabled = True
        return config
    
    @classmethod
    def preset_poset_validation(cls) -> "AgentConfig":
        """预设：阶段2 - 偏序图验证（不降级）"""
        config = cls(execution_mode=ExecutionMode.EXPERT)
        config.switches.poset_enabled = True
        config.switches.poset_fallback_enabled = False
        config.switches.poset_strict_mode = True
        config.switches.trace_enabled = True
        return config
    
    @classmethod
    def preset_hybrid_benchmark(cls) -> "AgentConfig":
        """预设：阶段3 - 混合模式评测"""
        config = cls(execution_mode=ExecutionMode.HYBRID)
        config.switches.poset_enabled = True
        config.switches.poset_fallback_enabled = True
        config.switches.trace_enabled = True
        return config
    
    @classmethod
    def preset_production(cls) -> "AgentConfig":
        """预设：生产环境"""
        config = cls(execution_mode=ExecutionMode.HYBRID)
        config.switches.poset_enabled = True
        config.switches.poset_fallback_enabled = True
        config.switches.trace_enabled = False
        config.switches.verbose = False
        return config
