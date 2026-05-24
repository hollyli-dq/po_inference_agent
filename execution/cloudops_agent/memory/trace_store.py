"""Trace Store - 执行轨迹存储

记录 Agent 执行过程中的所有动作, 用于:
1. 训练偏序图
2. 调试分析
3. 实验评测

支持多种输出格式:
- JSON: 完整详细数据
- CSV: 简洁的 API 调用记录
- TXT: LLM 友好的紧凑格式
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path
import json
import uuid
import csv


class ActionStatus(Enum):
    """动作执行状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionRecord:
    """单个动作记录"""
    action_id: str
    action_name: str
    params: Dict[str, Any]
    status: ActionStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    llm_tokens: int = 0  # 该动作消耗的 token
    source: str = "unknown"  # poset / react / fallback
    product: Optional[str] = None # 关联的云产品 (ECS/VPC/RDS...)
    # 并行执行相关字段
    layer: int = -1  # 拓扑层级（同层可并行）
    batch_id: Optional[str] = None  # 批次标识
    virtual_start_time: float = 0.0  # 虚拟开始时间（秒）
    virtual_end_time: float = 0.0  # 虚拟结束时间（秒）
    simulated_duration_ms: float = 0.0  # 模拟执行耗时（毫秒）
    
    @property
    def duration_ms(self) -> Optional[float]:
        """执行耗时（毫秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "params": self.params,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "llm_tokens": self.llm_tokens,
            "source": self.source,
            "product": self.product,
            # 并行相关
            "layer": self.layer,
            "batch_id": self.batch_id,
            "virtual_start_time": self.virtual_start_time,
            "virtual_end_time": self.virtual_end_time,
            "simulated_duration_ms": self.simulated_duration_ms
        }


@dataclass
class TraceRecord:
    """完整的执行轨迹"""
    trace_id: str
    task: str  # 用户任务描述
    intent: Optional[Dict[str, Any]] = None  # 解析出的意图
    mode: str = "unknown"  # expert / hybrid / explore
    actions: List[ActionRecord] = field(default_factory=list)
    status: str = "running"  # running / success / failed
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    blackboard_initial: Dict[str, Any] = field(default_factory=dict)
    blackboard_final: Dict[str, Any] = field(default_factory=dict)
    total_llm_tokens: int = 0
    fallback_count: int = 0  # 降级次数
    fallback_step: int = -1  # 降级发生时已执行的动作数（-1 表示未降级）
    error_summary: Optional[str] = None
    # 生成配置元数据（用于训练数据分析）
    generation_config: Dict[str, Any] = field(default_factory=dict)
    # 并行执行统计
    parallelism_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, record: ActionRecord) -> None:
        """添加动作记录"""
        self.actions.append(record)
        self.total_llm_tokens += record.llm_tokens
    
    def get_action_sequence(self) -> List[str]:
        """获取动作序列"""
        return [a.action_name for a in self.actions if a.status == ActionStatus.SUCCESS]
    
    def get_io_mappings(self) -> List[Dict[str, Any]]:
        """提取 IO 映射（用于训练）"""
        mappings = []
        for action in self.actions:
            if action.status == ActionStatus.SUCCESS and action.result:
                mappings.append({
                    "action": action.action_name,
                    "inputs": action.params,
                    "outputs": action.result
                })
        return mappings
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "intent": self.intent,
            "mode": self.mode,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "total_llm_tokens": self.total_llm_tokens,
            "fallback_count": self.fallback_count,
            "action_count": len(self.actions),
            "action_sequence": self.get_action_sequence(),
            "actions": [a.to_dict() for a in self.actions],
            "blackboard_initial": self.blackboard_initial,
            "blackboard_final": self.blackboard_final,
            "error_summary": self.error_summary,
            "generation_config": self.generation_config,
            "parallelism_stats": self.parallelism_stats
        }


class TraceStore:
    """
    轨迹存储管理器
    
    支持多种输出格式：
    - JSON: 完整详细数据
    - CSV: 简洁的 API 调用记录 (调用时间, 产品, API名, 结果)
    - TXT: LLM 友好的紧凑格式
    """
    
    def __init__(self, output_path: str = "./traces/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._current_trace: Optional[TraceRecord] = None
        self._traces: List[TraceRecord] = []
        self._task_index: Optional[int] = None  # 任务索引位置
        self._custom_suffix: Optional[str] = None  # 自定义后缀（如模型名_温度）
        self._generation_config: Dict[str, Any] = {}  # 生成配置元数据
    
    def set_task_index(self, index: int):
        """设置当前任务在批量任务中的索引"""
        self._task_index = index
    
    def set_custom_suffix(self, suffix: str):
        """设置自定义 trace 命名后缀（如模型名和温度）"""
        self._custom_suffix = suffix
    
    def set_generation_config(self, **kwargs):
        """设置生成配置元数据
        
        用于记录 trace 生成时的配置信息，便于后续分析和比较。
        
        常用参数:
            model: 基础模型名称 (str)
            temperature: 温度参数 (float)
            scenario_id: 场景编号 (int)
            generator: 生成器名称 (str)
        """
        self._generation_config = kwargs
    
    def start_trace(self, task: str, mode: str = "unknown", 
                    initial_blackboard: Optional[Dict[str, Any]] = None) -> str:
        """开始新的轨迹记录"""
        # trace_id 格式: trace_T{index}_{custom_suffix}_{timestamp}_{random}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = uuid.uuid4().hex[:6]
        
        # 构建 trace_id
        parts = ["trace"]
        if self._task_index is not None:
            parts.append(f"T{self._task_index:02d}")
        if self._custom_suffix:
            parts.append(self._custom_suffix)
        parts.append(timestamp)
        parts.append(random_suffix)
        
        trace_id = "_".join(parts)
            
        self._current_trace = TraceRecord(
            trace_id=trace_id,
            task=task,
            mode=mode,
            blackboard_initial=initial_blackboard or {},
            generation_config=self._generation_config.copy()  # 注入生成配置
        )
        # 重置临时状态
        self._custom_suffix = None
        self._generation_config = {}
        
        return trace_id
    
    def set_intent(self, intent: Dict[str, Any]) -> None:
        """设置解析出的意图"""
        if self._current_trace:
            self._current_trace.intent = intent
    
    def record_action_start(self, action_name: str, params: Dict[str, Any],
                           source: str = "unknown", product: Optional[str] = None,
                           layer: int = -1, batch_id: Optional[str] = None,
                           virtual_start_time: float = 0.0) -> str:
        """记录动作开始
        
        Args:
            action_name: 动作名称
            params: 参数字典
            source: 来源 (poset/react/fallback)
            product: 产品类型
            layer: 拓扑层级（用于并行分析）
            batch_id: 批次标识
            virtual_start_time: 虚拟开始时间（秒）
        """
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        record = ActionRecord(
            action_id=action_id,
            action_name=action_name,
            params=params,
            status=ActionStatus.EXECUTING,
            start_time=datetime.now(),
            source=source,
            product=product,
            layer=layer,
            batch_id=batch_id,
            virtual_start_time=virtual_start_time
        )
        if self._current_trace:
            self._current_trace.add_action(record)
        return action_id
    
    def record_action_success(self, action_id: str, result: Dict[str, Any],
                             llm_tokens: int = 0,
                             virtual_end_time: float = 0.0,
                             simulated_duration_ms: float = 0.0) -> None:
        """记录动作成功
        
        Args:
            action_id: 动作ID
            result: 执行结果
            llm_tokens: 消耗的 token 数
            virtual_end_time: 虚拟结束时间（秒）
            simulated_duration_ms: 模拟执行耗时（毫秒）
        """
        if self._current_trace:
            for action in self._current_trace.actions:
                if action.action_id == action_id:
                    action.status = ActionStatus.SUCCESS
                    action.result = result
                    action.end_time = datetime.now()
                    action.llm_tokens = llm_tokens
                    action.virtual_end_time = virtual_end_time
                    action.simulated_duration_ms = simulated_duration_ms
                    break
    
    def record_action_failure(self, action_id: str, error: str,
                             error_code: Optional[str] = None,
                             llm_tokens: int = 0) -> None:
        """记录动作失败"""
        if self._current_trace:
            for action in self._current_trace.actions:
                if action.action_id == action_id:
                    action.status = ActionStatus.FAILED
                    action.error = error
                    action.error_code = error_code
                    action.end_time = datetime.now()
                    action.llm_tokens = llm_tokens
                    break
    
    def record_fallback(self) -> None:
        """记录降级事件
        
        同时记录降级发生时已执行的动作数，便于分析降级点
        """
        if self._current_trace:
            self._current_trace.fallback_count += 1
            # 记录降级发生时的步骤位置（已执行的动作数）
            if self._current_trace.fallback_step < 0:  # 只记录第一次降级
                self._current_trace.fallback_step = len(self._current_trace.actions)
    
    def set_parallelism_stats(self, stats: Dict[str, Any]) -> None:
        """设置并行执行统计信息
        
        Args:
            stats: 统计字典，包含:
                - total_layers: 总层数（关键路径长度）
                - max_parallel_actions: 单层最大并行动作数
                - actions_per_layer: 每层动作数列表
                - sequential_time_ms: 顺序执行总时间
                - parallel_time_ms: 并行执行总时间
                - speedup: 加速比
                - layer_times_ms: 每层执行时间列表
        """
        if self._current_trace:
            self._current_trace.parallelism_stats = stats
    
    def end_trace(self, status: str, final_blackboard: Optional[Dict[str, Any]] = None,
                  error_summary: Optional[str] = None) -> Optional[TraceRecord]:
        """结束轨迹记录"""
        if self._current_trace:
            self._current_trace.status = status
            self._current_trace.end_time = datetime.now()
            self._current_trace.blackboard_final = final_blackboard or {}
            self._current_trace.error_summary = error_summary
            
            # 保存到列表
            self._traces.append(self._current_trace)
            
            # 保存到文件
            self._save_trace(self._current_trace)
            
            trace = self._current_trace
            self._current_trace = None
            return trace
        return None
    
    def _save_trace(self, trace: TraceRecord) -> None:
        """保存轨迹到文件（多格式）"""
        # 1. 保存 JSON 格式（完整数据）
        json_path = self.output_path / f"{trace.trace_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # 使用安全序列化，处理无效 Unicode 字符
            json.dump(self._sanitize_for_json(trace.to_dict()), f, ensure_ascii=False, indent=2)
        
        # 2. 保存 CSV 格式（简洁 API 调用记录）
        self._save_trace_csv(trace)
        
        # 3. 保存 TXT 格式（LLM 友好）
        self._save_trace_txt(trace)
    
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
    
    def _save_trace_csv(self, trace: TraceRecord) -> None:
        """保存 CSV 格式 - 只包含核心调用信息"""
        csv_path = self.output_path / f"{trace.trace_id}.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['time', 'product', 'api', 'success', 'error'])
            
            for action in trace.actions:
                time_str = action.start_time.strftime('%H:%M:%S') if action.start_time else ''
                success = 1 if action.status == ActionStatus.SUCCESS else 0
                error_brief = (action.error[:50] + '...') if action.error and len(action.error) > 50 else (action.error or '')
                
                writer.writerow([
                    time_str,
                    action.product or 'unknown',
                    action.action_name,
                    success,
                    error_brief
                ])
    
    def _save_trace_txt(self, trace: TraceRecord) -> None:
        """
        保存 TXT 格式 - LLM 友好的紧凑格式
        
        格式设计原则:
        - 去除冗余 JSON 符号
        - 使用简洁的键值对
        - 分层结构清晰
        - 参数和结果用单行紧凑表示
        """
        txt_path = self.output_path / f"{trace.trace_id}.txt"
        lines = []
        
        # 头部摘要
        lines.append(f"=== TRACE: {trace.trace_id} ===")
        lines.append(f"TASK: {trace.task}")
        lines.append(f"STATUS: {trace.status} | MODE: {trace.mode} | ACTIONS: {len(trace.actions)}")
        lines.append(f"DURATION: {trace.duration_ms:.0f}ms | TOKENS: {trace.total_llm_tokens}")
        if trace.error_summary:
            lines.append(f"ERROR: {trace.error_summary}")
        lines.append("")
        
        # Action 序列
        lines.append("--- ACTION SEQUENCE ---")
        for i, action in enumerate(trace.actions, 1):
            status_mark = "✓" if action.status == ActionStatus.SUCCESS else "✗"
            time_str = action.start_time.strftime('%H:%M:%S') if action.start_time else '??:??:??'
            
            # 第一行: 序号 状态 时间 产品.API
            lines.append(f"[{i:02d}] {status_mark} {time_str} {action.product}.{action.action_name}")
            
            # 第二行: 参数 (紧凑表示)
            params_str = self._compact_dict(action.params)
            lines.append(f"     IN: {params_str}")
            
            # 第三行: 结果或错误
            if action.status == ActionStatus.SUCCESS:
                out_str = self._compact_dict(action.result, max_len=150)
                lines.append(f"     OUT: {out_str}")
            else:
                lines.append(f"     ERR: {action.error}")
        
        lines.append("")
        
        # 创建的资源摘要
        lines.append("--- RESOURCES CREATED ---")
        resources = trace.blackboard_final.get('resources', {})
        if resources:
            for name, res in resources.items():
                res_type = res.get('_type', 'UNKNOWN')
                # 提取关键 ID
                id_keys = ['VpcId', 'VSwitchId', 'SecurityGroupId', 'InstanceIds', 
                          'LoadBalancerId', 'DBInstanceId', 'AllocationId', 'EipAddress']
                ids = {k: res[k] for k in id_keys if k in res}
                ids_str = ', '.join(f"{k}={v}" for k, v in ids.items())
                lines.append(f"  {res_type}: {ids_str}")
        else:
            lines.append("  (none)")
        
        lines.append("")
        lines.append("=== END ===")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _compact_dict(self, d: Optional[Dict], max_len: int = 200) -> str:
        """将字典转换为紧凑字符串"""
        if not d:
            return '{}'
        
        # 简化常见的大对象
        simplified = {}
        for k, v in d.items():
            if k in ['Images', 'Zones', 'AvailableZones', 'SecurityGroups']:
                # 大列表只保留数量
                if isinstance(v, dict) and any(isinstance(vv, list) for vv in v.values()):
                    for kk, vv in v.items():
                        if isinstance(vv, list):
                            simplified[k] = f"[{len(vv)} items]"
                            break
                elif isinstance(v, list):
                    simplified[k] = f"[{len(v)} items]"
                else:
                    simplified[k] = v
            elif isinstance(v, dict) and len(str(v)) > 100:
                # 复杂字典简化
                keys = list(v.keys())[:3]
                simplified[k] = f"{{{', '.join(keys)}...}}"
            elif isinstance(v, list) and len(v) > 5:
                simplified[k] = f"[{len(v)} items]"
            else:
                simplified[k] = v
        
        # 转为紧凑字符串
        result = ', '.join(f"{k}={v}" for k, v in simplified.items())
        if len(result) > max_len:
            result = result[:max_len-3] + '...'
        return result
    
    def get_current_trace(self) -> Optional[TraceRecord]:
        """获取当前轨迹"""
        return self._current_trace
    
    def get_all_traces(self) -> List[TraceRecord]:
        """获取所有轨迹"""
        return self._traces.copy()
    
    def load_traces_from_disk(self) -> List[TraceRecord]:
        """从磁盘加载所有轨迹"""
        traces = []
        for filepath in self.output_path.glob("trace_*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 简化处理，这里不完全还原对象
                traces.append(data)
        return traces
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._traces)
        success = sum(1 for t in self._traces if t.status == "success")
        failed = sum(1 for t in self._traces if t.status == "failed")
        total_tokens = sum(t.total_llm_tokens for t in self._traces)
        total_actions = sum(len(t.actions) for t in self._traces)
        fallbacks = sum(t.fallback_count for t in self._traces)
        
        return {
            "total_traces": total,
            "success_count": success,
            "failed_count": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_llm_tokens": total_tokens,
            "avg_tokens_per_trace": total_tokens / total if total > 0 else 0,
            "total_actions": total_actions,
            "avg_actions_per_trace": total_actions / total if total > 0 else 0,
            "total_fallbacks": fallbacks
        }
