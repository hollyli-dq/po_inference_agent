"""
TraceRecorder: Records Agent execution traces for BHPOP training.
Outputs JSON format compatible with the BHPOP algorithm.
"""
import json
import uuid
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Step:
    """A single step in an execution trace."""
    step_id: int
    action: str
    product: str
    params: Dict[str, Any]
    result: str  # "Success" or "Failed"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    io_fingerprint: Dict[str, List[str]] = field(default_factory=lambda: {"in": [], "out": []})
    simulated_time_ms: float = 0.0
    wall_time_ms: float = 0.0
    cost_tokens: int = 0  # For LLM token counting


@dataclass 
class Trace:
    """A complete execution trace."""
    trace_id: str
    intent: str  # Natural language description of the task
    intent_cluster: str = ""  # Cluster label (filled by clustering algorithm)
    steps: List[Step] = field(default_factory=list)
    total_simulated_time_ms: float = 0.0
    total_wall_time_ms: float = 0.0
    total_tokens: int = 0
    final_status: str = "Pending"  # Success, Failed, Pending
    created_at: str = ""


class TraceRecorder:
    """
    Records execution traces for training BHPOP.
    
    Usage:
        recorder = TraceRecorder()
        recorder.start_trace("Create a web stack with 3 ECS instances")
        
        # For each API call:
        recorder.record_step(
            action="CreateVpc",
            product="VPC",
            params={"CidrBlock": "10.0.0.0/8"},
            result="Success",
            io_fingerprint={"in": [], "out": ["vpc-xxx"]}
        )
        
        recorder.end_trace("Success")
        recorder.save("traces.json")
    """
    
    def __init__(self):
        self._current_trace: Optional[Trace] = None
        self._traces: List[Trace] = []
        self._step_counter: int = 0
        self._trace_start_time: float = 0.0
        self._step_start_time: float = 0.0
    
    def start_trace(self, intent: str) -> str:
        """Start recording a new trace."""
        trace_id = f"t-{uuid.uuid4().hex[:8]}"
        self._current_trace = Trace(
            trace_id=trace_id,
            intent=intent,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self._step_counter = 0
        self._trace_start_time = time.time()
        return trace_id
    
    def start_step(self):
        """Mark the start of a step (for wall time measurement)."""
        self._step_start_time = time.time()
    
    def record_step(
        self,
        action: str,
        product: str,
        params: Dict[str, Any],
        result: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        io_fingerprint: Optional[Dict[str, List[str]]] = None,
        simulated_time_ms: float = 0.0,
        cost_tokens: int = 0
    ) -> Step:
        """Record a single step in the current trace."""
        if not self._current_trace:
            raise RuntimeError("No active trace. Call start_trace() first.")
        
        self._step_counter += 1
        wall_time_ms = (time.time() - self._step_start_time) * 1000 if self._step_start_time else 0.0
        
        step = Step(
            step_id=self._step_counter,
            action=action,
            product=product,
            params=self._sanitize_params(params),
            result=result,
            error_code=error_code,
            error_message=error_message,
            io_fingerprint=io_fingerprint or {"in": [], "out": []},
            simulated_time_ms=simulated_time_ms,
            wall_time_ms=wall_time_ms,
            cost_tokens=cost_tokens
        )
        
        self._current_trace.steps.append(step)
        self._current_trace.total_simulated_time_ms += simulated_time_ms
        self._current_trace.total_wall_time_ms += wall_time_ms
        self._current_trace.total_tokens += cost_tokens
        
        return step
    
    def end_trace(self, final_status: str = "Success") -> Trace:
        """End the current trace and add it to the collection."""
        if not self._current_trace:
            raise RuntimeError("No active trace to end.")
        
        self._current_trace.final_status = final_status
        self._current_trace.total_wall_time_ms = (time.time() - self._trace_start_time) * 1000
        
        trace = self._current_trace
        self._traces.append(trace)
        self._current_trace = None
        
        return trace
    
    def get_current_trace(self) -> Optional[Trace]:
        """Get the current active trace."""
        return self._current_trace
    
    def get_all_traces(self) -> List[Trace]:
        """Get all recorded traces."""
        return self._traces
    
    def save(self, filepath: str, append: bool = False) -> None:
        """Save traces to a JSON file."""
        path = Path(filepath)
        
        if append and path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    existing.extend([asdict(t) for t in self._traces])
                    data = existing
                else:
                    data = [asdict(t) for t in self._traces]
        else:
            data = [asdict(t) for t in self._traces]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str) -> List[Trace]:
        """Load traces from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        traces = []
        for item in data:
            steps = [Step(**s) for s in item.pop('steps', [])]
            trace = Trace(**item, steps=steps)
            traces.append(trace)
        
        self._traces = traces
        return traces
    
    def clear(self) -> None:
        """Clear all recorded traces."""
        self._traces = []
        self._current_trace = None
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for logging.
        Remove sensitive info but keep ID references for IO mining.
        """
        sanitized = {}
        sensitive_keys = {'Password', 'AccessKeyId', 'AccessKeySecret', 'SecurityToken'}
        
        for key, value in params.items():
            if key in sensitive_keys:
                sanitized[key] = "***"
            elif isinstance(value, str) and len(value) > 200:
                # Truncate very long strings
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        
        return sanitized


def extract_io_fingerprint(response: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract ID fingerprints from API response for IO mining.
    Looks for fields containing 'Id' or 'ID' patterns.
    """
    outputs = []
    
    # Patterns that indicate ID fields (case-insensitive contains check)
    id_keywords = ['id', 'Id', 'ID']
    
    def is_id_key(key: str) -> bool:
        """Check if key looks like an ID field."""
        # Match: VpcId, InstanceId, InstanceIdSet, InstanceIdSets, SecurityGroupIds, etc.
        for kw in id_keywords:
            if kw in key:
                return True
        return False
    
    def extract_ids(obj, parent_key=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if is_id_key(key):
                    if isinstance(value, str) and value:
                        outputs.append(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, str) and v:
                                outputs.append(v)
                            elif isinstance(v, dict):
                                # Recursively extract from dict items in list
                                extract_ids(v, key)
                    elif isinstance(value, dict):
                        # Nested dict under ID key (e.g., InstanceIdSets -> InstanceIdSet)
                        extract_ids(value, key)
                else:
                    extract_ids(value, key)
        elif isinstance(obj, list):
            for item in obj:
                extract_ids(item, parent_key)
    
    extract_ids(response)
    
    return {"in": [], "out": list(set(outputs))}
