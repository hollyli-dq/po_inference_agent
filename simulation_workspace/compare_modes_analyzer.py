#!/usr/bin/env python3
"""
Trace Analyzer - 解析并对比 explore、hybrid、expert 三种模式的执行效果

支持两种目录结构:
1. 旧结构: expert_traces/, hybrid_traces/, explore_traces/
2. 新结构: mode_traces/expert/, mode_traces/hybrid/, mode_traces/explore_thinking_off/, mode_traces/explore_thinking_on/
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 配置
WORKSPACE_DIR = Path(__file__).parent

# 新结构目录 (优先使用)
MODE_TRACES_DIR = WORKSPACE_DIR / "mode_traces"

# 旧结构目录 (向后兼容)
EXPERT_DIR = WORKSPACE_DIR / "expert_traces"
HYBRID_DIR = WORKSPACE_DIR / "hybrid_traces"
EXPLORE_DIR = WORKSPACE_DIR / "explore_traces"

OUTPUT_FILE = WORKSPACE_DIR / "mode_comparison_report.md"


def load_json(file_path: Path) -> Optional[Dict]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def parse_expert_traces() -> Dict[str, Dict]:
    """解析expert模式的traces"""
    summary_file = EXPERT_DIR / "execution_summary.json"
    summary = load_json(summary_file)
    if not summary:
        return {}
    
    results = {}
    total_duration_s_core = 0.0
    
    # 并行统计汇总
    total_sequential_time_ms = 0.0
    total_parallel_time_ms = 0.0
    total_layers = 0
    max_parallel_ever = 0
    
    for result in summary.get("results", []):
        task_id = result.get("intent_type", "unknown")
        duration_s = result.get("duration_s", 0)
        total_duration_s_core += duration_s or 0
        
        # 尝试从对应的 trace 文件读取并行统计
        trace_id = result.get("trace_id", "")
        parallelism_stats = {}
        if trace_id:
            trace_file = EXPERT_DIR / f"{trace_id}.json"
            trace_data = load_json(trace_file)
            if trace_data:
                parallelism_stats = trace_data.get("parallelism_stats", {})
                # 汇总并行统计
                total_sequential_time_ms += parallelism_stats.get("sequential_time_ms", 0)
                total_parallel_time_ms += parallelism_stats.get("parallel_time_ms", 0)
                total_layers += parallelism_stats.get("total_layers", 0)
                max_parallel_ever = max(max_parallel_ever, parallelism_stats.get("max_parallel_actions", 0))
        
        results[task_id] = {
            "mode": "expert",
            "scenario": result.get("scenario", ""),
            "query": result.get("query", ""),
            "status": result.get("status", "unknown"),
            "duration_s": duration_s,
            "duration_ms": (duration_s or 0) * 1000,
            "actions_count": result.get("actions_count", 0),
            "actions_executed": result.get("actions_executed", []),
            "llm_tokens": result.get("llm_tokens", 0),  # 意图识别LLM token
            "trace_id": result.get("trace_id", ""),
            "error": result.get("error"),
            # 并行统计
            "parallelism_stats": parallelism_stats,
        }
    
    # 计算整体加速比
    overall_speedup = total_sequential_time_ms / total_parallel_time_ms if total_parallel_time_ms > 0 else 1.0
    
    # 添加汇总信息
    results["_summary"] = {
        "mode": "expert",
        "total_scenarios": summary.get("total_scenarios", 0),
        "success_count": summary.get("success_count", 0),
        "failed_count": summary.get("failed_count", 0),
        "total_actions": summary.get("total_actions", 0),
        "total_duration_s": summary.get("total_duration_s", 0),
        "total_duration_s_core": total_duration_s_core,
        "total_llm_tokens": summary.get("total_llm_tokens", 0),  # 读取总LLM token
        # 并行统计汇总
        "parallelism": {
            "total_sequential_time_ms": round(total_sequential_time_ms, 2),
            "total_parallel_time_ms": round(total_parallel_time_ms, 2),
            "overall_speedup": round(overall_speedup, 2),
            "total_layers": total_layers,
            "max_parallel_actions": max_parallel_ever,
        }
    }
    
    return results


def parse_hybrid_traces() -> Dict[str, Dict]:
    """解析hybrid模式的traces"""
    summary_file = HYBRID_DIR / "execution_summary.json"
    summary = load_json(summary_file)
    if not summary:
        return {}
    
    results = {}
    total_duration_s_core = 0.0
    for result in summary.get("results", []):
        task_id = result.get("intent_type", "unknown")
        duration_s = result.get("duration_s", 0)
        total_duration_s_core += duration_s or 0
        results[task_id] = {
            "mode": "hybrid",
            "scenario": result.get("scenario", ""),
            "query": result.get("query", ""),
            "status": result.get("status", "unknown"),
            "duration_s": duration_s,
            "duration_ms": (duration_s or 0) * 1000,
            "actions_count": result.get("actions_count", 0),
            "actions_executed": result.get("actions_executed", []),
            "llm_tokens": result.get("llm_tokens", 0),
            "trace_id": result.get("trace_id", ""),
            "fallback_count": result.get("fallback_count", 0),
            "error": result.get("error"),
        }
    
    # 添加汇总信息
    results["_summary"] = {
        "mode": "hybrid",
        "model": summary.get("model", ""),
        "total_scenarios": summary.get("total_scenarios", 0),
        "success_count": summary.get("success_count", 0),
        "failed_count": summary.get("failed_count", 0),
        "total_actions": summary.get("total_actions", 0),
        "total_duration_s": summary.get("total_duration_s", 0),
        "total_duration_s_core": total_duration_s_core,
        "total_llm_tokens": summary.get("total_llm_tokens", 0),
        "total_fallbacks": summary.get("total_fallbacks", 0),
    }
    
    return results


def parse_explore_traces() -> Dict[str, Dict]:
    """解析explore模式的traces (从单个trace文件解析)"""
    results = {}
    total_actions = 0
    total_duration_ms = 0
    total_llm_tokens = 0
    success_count = 0
    failed_count = 0
    
    # 遍历所有trace JSON文件
    trace_files = sorted(EXPLORE_DIR.glob("trace_*.json"))
    
    for trace_file in trace_files:
        trace_data = load_json(trace_file)
        if not trace_data:
            continue
        
        task_id = trace_data.get("intent", {}).get("intent_type", "unknown")
        status = trace_data.get("status", "unknown")
        duration_ms = trace_data.get("duration_ms", 0)
        action_count = trace_data.get("action_count", 0)
        llm_tokens = trace_data.get("total_llm_tokens", 0)
        
        # 提取场景编号
        trace_id = trace_data.get("trace_id", "")
        scenario_match = trace_id.split("_")[1] if trace_id else ""
        scenario_num = int(scenario_match[1:]) if scenario_match.startswith("T") else 0
        
        results[task_id] = {
            "mode": "explore",
            "scenario": f"任务 {scenario_num}",
            "query": trace_data.get("task", ""),
            "status": status,
            "duration_s": duration_ms / 1000,
            "duration_ms": duration_ms,
            "actions_count": action_count,
            "actions_executed": trace_data.get("action_sequence", []),
            "llm_tokens": llm_tokens,
            "trace_id": trace_id,
            "fallback_count": trace_data.get("fallback_count", 0),
            "error": trace_data.get("error_summary"),
        }
        
        # 汇总统计
        total_actions += action_count
        total_duration_ms += duration_ms
        total_llm_tokens += llm_tokens
        if status == "success":
            success_count += 1
        else:
            failed_count += 1
    
    # 添加汇总信息
    results["_summary"] = {
        "mode": "explore",
        "total_scenarios": len(trace_files),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_actions": total_actions,
        "total_duration_s": total_duration_ms / 1000,
        "total_llm_tokens": total_llm_tokens,
    }
    
    return results


def parse_new_format_traces(traces_dir: Path, mode_name: str) -> Dict[str, Dict]:
    """
    解析新格式的 traces (来自 mode_comparison_experiment.py)
    
    Args:
        traces_dir: traces 目录 (mode_traces/expert 等)
        mode_name: 模式名称 (expert, hybrid, explore)
    """
    if not traces_dir.exists():
        return {}
    
    summary_file = traces_dir / "execution_summary.json"
    summary = load_json(summary_file)
    if not summary:
        return {}
    
    results = {}
    total_duration_s_core = 0.0
    
    for result in summary.get("results", []):
        # 新格式使用 scenario_name 作为 task_id
        task_id = result.get("scenario_name", "unknown")
        duration_s = result.get("execution_time_s", 0)
        total_duration_s_core += duration_s or 0
        
        results[task_id] = {
            "mode": mode_name,
            "scenario": task_id,
            "query": result.get("query", ""),
            "status": "success" if result.get("success", False) else "failed",
            "duration_s": duration_s,
            "duration_ms": (duration_s or 0) * 1000,
            "actions_count": result.get("action_count", 0),
            "actions_executed": result.get("actions_executed", []),
            "llm_tokens": result.get("total_tokens", 0),
            "trace_id": result.get("trace_id", ""),
            "fallback_count": result.get("fallback_count", 0),
            "error": result.get("error"),
            "cover_f1": result.get("cover_f1", 0),
            "thinking_enabled": result.get("thinking_enabled", False),
        }
    
    # 添加汇总信息
    results["_summary"] = {
        "mode": mode_name,
        "thinking_enabled": summary.get("thinking_enabled", False),
        "total_scenarios": summary.get("total_scenarios", 0),
        "success_count": summary.get("success_count", 0),
        "failed_count": summary.get("failed_count", 0),
        "total_actions": summary.get("total_actions", 0),
        "total_duration_s": summary.get("total_duration_s", 0),
        "total_duration_s_core": total_duration_s_core,
        "total_llm_tokens": summary.get("total_tokens", 0),
        "total_fallbacks": sum(r.get("fallback_count", 0) for r in summary.get("results", [])),
    }
    
    return results


def auto_detect_and_parse() -> Dict[str, Dict[str, Dict]]:
    """
    自动检测目录结构并解析所有模式的 traces
    
    Returns:
        {
            "expert": {...},
            "hybrid": {...},
            "explore": {...},
            "explore_thinking_on": {...},  # 如果存在
        }
    """
    all_data = {}
    
    # 优先使用新结构
    if MODE_TRACES_DIR.exists():
        print(f"检测到新目录结构: {MODE_TRACES_DIR}")
        
        # Expert
        expert_dir = MODE_TRACES_DIR / "expert"
        if expert_dir.exists():
            all_data["expert"] = parse_new_format_traces(expert_dir, "expert")
            print(f"  - expert: {len(all_data['expert']) - 1} 个场景")
        
        # Hybrid
        hybrid_dir = MODE_TRACES_DIR / "hybrid"
        if hybrid_dir.exists():
            all_data["hybrid"] = parse_new_format_traces(hybrid_dir, "hybrid")
            print(f"  - hybrid: {len(all_data['hybrid']) - 1} 个场景")
        
        # Explore (关闭思考)
        explore_off_dir = MODE_TRACES_DIR / "explore_thinking_off"
        if explore_off_dir.exists():
            all_data["explore"] = parse_new_format_traces(explore_off_dir, "explore")
            print(f"  - explore (thinking off): {len(all_data['explore']) - 1} 个场景")
        
        # Explore (开启思考)
        explore_on_dir = MODE_TRACES_DIR / "explore_thinking_on"
        if explore_on_dir.exists():
            all_data["explore_thinking_on"] = parse_new_format_traces(explore_on_dir, "explore_thinking_on")
            print(f"  - explore (thinking on): {len(all_data['explore_thinking_on']) - 1} 个场景")
    
    # 如果新结构没有数据，回退到旧结构
    if not all_data:
        print("使用旧目录结构...")
        
        expert_data = parse_expert_traces()
        if expert_data:
            all_data["expert"] = expert_data
            print(f"  - expert: {len(expert_data) - 1} 个场景")
        
        hybrid_data = parse_hybrid_traces()
        if hybrid_data:
            all_data["hybrid"] = hybrid_data
            print(f"  - hybrid: {len(hybrid_data) - 1} 个场景")
        
        explore_data = parse_explore_traces()
        if explore_data:
            all_data["explore"] = explore_data
            print(f"  - explore: {len(explore_data) - 1} 个场景")
    
    return all_data


def load_best_posets() -> Dict[str, Any]:
    """加载最优偏序图配置"""
    best_posets_file = WORKSPACE_DIR / "best_posets.json"
    if best_posets_file.exists():
        return load_json(best_posets_file) or {}
    return {}


def generate_markdown_report(
    expert: Dict, 
    hybrid: Dict, 
    explore: Dict,
    explore_thinking_on: Dict = None,
    best_posets: Dict = None
) -> str:
    """生成Markdown格式的对比报告（支持4种模式对比）"""
    lines = []
    
    # 是否有4种模式
    has_explore_on = explore_thinking_on and len(explore_thinking_on) > 1
    
    # 标题
    lines.append("# Trace 执行效果对比报告")
    lines.append("")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 最优偏序图配置总结
    if best_posets:
        lines.append("## 0. 最优偏序图配置 (HPO 结果)")
        lines.append("")
        lines.append("以下是通过 HPO 贝叶斯优化选出的各场景 F1 最高的偏序图配置：")
        lines.append("")
        lines.append("| 场景 | Cover-F1 | IP-Cov Target | eps_jump | 任务描述 |")
        lines.append("|------|----------|---------------|----------|----------|")
        for scenario, info in sorted(best_posets.items()):
            f1 = info.get('cover_f1', 0)
            ip_cov = info.get('ip_cov_target', 0)
            eps = info.get('eps_jump', 0)
            query = info.get('query', '')[:40] + '...' if len(info.get('query', '')) > 40 else info.get('query', '')
            lines.append(f"| {scenario} | **{f1:.4f}** | {ip_cov} | {eps} | {query} |")
        lines.append("")
        avg_f1 = sum(info.get('cover_f1', 0) for info in best_posets.values()) / len(best_posets) if best_posets else 0
        lines.append(f"> **平均 Cover-F1**: {avg_f1:.4f}")
        lines.append("")
    
    # 概述
    lines.append("## 1. 模式概述")
    lines.append("")
    lines.append("| 模式 | 描述 |")
    lines.append("|------|------|")
    lines.append("| **Expert** | 基于预定义的POSET(偏序执行计划)，使用标准参数，无LLM调用 |")
    lines.append("| **Hybrid** | 使用LLM进行意图识别和参数推断，结合POSET执行，支持降级到ReAct |")
    lines.append("| **Explore (思考关闭)** | 纯LLM驱动的ReAct探索模式，关闭深度思考 |")
    if has_explore_on:
        lines.append("| **Explore (思考开启)** | 纯LLM驱动的ReAct探索模式，开启深度思考 |")
    lines.append("")
    
    # 汇总对比表
    lines.append("## 2. 总体性能对比")
    lines.append("")
    
    expert_sum = expert.get("_summary", {})
    hybrid_sum = hybrid.get("_summary", {})
    explore_sum = explore.get("_summary", {})
    explore_on_sum = explore_thinking_on.get("_summary", {}) if explore_thinking_on else {}
    
    if has_explore_on:
        lines.append("| 指标 | Expert | Hybrid | Explore (思考关) | Explore (思考开) |")
        lines.append("|------|--------|--------|------------------|------------------|")
        lines.append(f"| 场景数量 | {expert_sum.get('total_scenarios', 0)} | {hybrid_sum.get('total_scenarios', 0)} | {explore_sum.get('total_scenarios', 0)} | {explore_on_sum.get('total_scenarios', 0)} |")
        lines.append(f"| 成功数量 | {expert_sum.get('success_count', 0)} | {hybrid_sum.get('success_count', 0)} | {explore_sum.get('success_count', 0)} | {explore_on_sum.get('success_count', 0)} |")
        lines.append(f"| 失败数量 | {expert_sum.get('failed_count', 0)} | {hybrid_sum.get('failed_count', 0)} | {explore_sum.get('failed_count', 0)} | {explore_on_sum.get('failed_count', 0)} |")
    else:
        lines.append("| 指标 | Expert | Hybrid | Explore |")
        lines.append("|------|--------|--------|---------|")
        lines.append(f"| 场景数量 | {expert_sum.get('total_scenarios', 0)} | {hybrid_sum.get('total_scenarios', 0)} | {explore_sum.get('total_scenarios', 0)} |")
        lines.append(f"| 成功数量 | {expert_sum.get('success_count', 0)} | {hybrid_sum.get('success_count', 0)} | {explore_sum.get('success_count', 0)} |")
        lines.append(f"| 失败数量 | {expert_sum.get('failed_count', 0)} | {hybrid_sum.get('failed_count', 0)} | {explore_sum.get('failed_count', 0)} |")
    
    # 成功率
    expert_rate = (expert_sum.get('success_count', 0) / max(expert_sum.get('total_scenarios', 1), 1)) * 100
    hybrid_rate = (hybrid_sum.get('success_count', 0) / max(hybrid_sum.get('total_scenarios', 1), 1)) * 100
    explore_rate = (explore_sum.get('success_count', 0) / max(explore_sum.get('total_scenarios', 1), 1)) * 100
    explore_on_rate = (explore_on_sum.get('success_count', 0) / max(explore_on_sum.get('total_scenarios', 1), 1)) * 100 if has_explore_on else 0
    
    if has_explore_on:
        lines.append(f"| **成功率** | **{expert_rate:.1f}%** | **{hybrid_rate:.1f}%** | **{explore_rate:.1f}%** | **{explore_on_rate:.1f}%** |")
        lines.append(f"| 总操作数 | {expert_sum.get('total_actions', 0)} | {hybrid_sum.get('total_actions', 0)} | {explore_sum.get('total_actions', 0)} | {explore_on_sum.get('total_actions', 0)} |")
        lines.append(f"| 总耗时(秒) | {expert_sum.get('total_duration_s', 0):.2f} | {hybrid_sum.get('total_duration_s', 0):.2f} | {explore_sum.get('total_duration_s', 0):.2f} | {explore_on_sum.get('total_duration_s', 0):.2f} |")
        lines.append(f"| LLM Tokens | {expert_sum.get('total_llm_tokens', 0):,} | {hybrid_sum.get('total_llm_tokens', 0):,} | {explore_sum.get('total_llm_tokens', 0):,} | {explore_on_sum.get('total_llm_tokens', 0):,} |")
    else:
        lines.append(f"| **成功率** | **{expert_rate:.1f}%** | **{hybrid_rate:.1f}%** | **{explore_rate:.1f}%** |")
        lines.append(f"| 总操作数 | {expert_sum.get('total_actions', 0)} | {hybrid_sum.get('total_actions', 0)} | {explore_sum.get('total_actions', 0)} |")
        lines.append(f"| 总耗时(秒) | {expert_sum.get('total_duration_s', 0):.2f} | {hybrid_sum.get('total_duration_s', 0):.2f} | {explore_sum.get('total_duration_s', 0):.2f} |")
        lines.append(f"| LLM Tokens | {expert_sum.get('total_llm_tokens', 0):,} | {hybrid_sum.get('total_llm_tokens', 0):,} | {explore_sum.get('total_llm_tokens', 0):,} |")
    
    if hybrid_sum.get('total_fallbacks'):
        if has_explore_on:
            lines.append(f"| Fallback次数 | - | {hybrid_sum.get('total_fallbacks', 0)} | - | - |")
        else:
            lines.append(f"| Fallback次数 | - | {hybrid_sum.get('total_fallbacks', 0)} | - |")
    
    lines.append("")
    
    # 效率对比
    lines.append("## 3. 效率分析")
    lines.append("")
    
    # 计算平均值
    expert_avg_time = expert_sum.get('total_duration_s', 0) / max(expert_sum.get('total_scenarios', 1), 1)
    hybrid_avg_time = hybrid_sum.get('total_duration_s', 0) / max(hybrid_sum.get('total_scenarios', 1), 1)
    explore_avg_time = explore_sum.get('total_duration_s', 0) / max(explore_sum.get('total_scenarios', 1), 1)
    explore_on_avg_time = explore_on_sum.get('total_duration_s', 0) / max(explore_on_sum.get('total_scenarios', 1), 1) if has_explore_on else 0
    
    expert_avg_actions = expert_sum.get('total_actions', 0) / max(expert_sum.get('total_scenarios', 1), 1)
    hybrid_avg_actions = hybrid_sum.get('total_actions', 0) / max(hybrid_sum.get('total_scenarios', 1), 1)
    explore_avg_actions = explore_sum.get('total_actions', 0) / max(explore_sum.get('total_scenarios', 1), 1)
    explore_on_avg_actions = explore_on_sum.get('total_actions', 0) / max(explore_on_sum.get('total_scenarios', 1), 1) if has_explore_on else 0
    
    expert_avg_tokens = expert_sum.get('total_llm_tokens', 0) / max(expert_sum.get('total_scenarios', 1), 1)
    hybrid_avg_tokens = hybrid_sum.get('total_llm_tokens', 0) / max(hybrid_sum.get('total_scenarios', 1), 1)
    explore_avg_tokens = explore_sum.get('total_llm_tokens', 0) / max(explore_sum.get('total_scenarios', 1), 1)
    explore_on_avg_tokens = explore_on_sum.get('total_llm_tokens', 0) / max(explore_on_sum.get('total_scenarios', 1), 1) if has_explore_on else 0
    
    if has_explore_on:
        lines.append("| 平均指标 | Expert | Hybrid | Explore (思考关) | Explore (思考开) |")
        lines.append("|----------|--------|--------|------------------|------------------|")
        lines.append(f"| 平均耗时(秒) | {expert_avg_time:.2f} | {hybrid_avg_time:.2f} | {explore_avg_time:.2f} | {explore_on_avg_time:.2f} |")
        lines.append(f"| 平均操作数 | {expert_avg_actions:.1f} | {hybrid_avg_actions:.1f} | {explore_avg_actions:.1f} | {explore_on_avg_actions:.1f} |")
        lines.append(f"| 平均LLM Tokens | {expert_avg_tokens:.0f} | {hybrid_avg_tokens:.0f} | {explore_avg_tokens:.0f} | {explore_on_avg_tokens:.0f} |")
    else:
        lines.append("| 平均指标 | Expert | Hybrid | Explore |")
        lines.append("|----------|--------|--------|---------|")
        lines.append(f"| 平均耗时(秒) | {expert_avg_time:.2f} | {hybrid_avg_time:.2f} | {explore_avg_time:.2f} |")
        lines.append(f"| 平均操作数 | {expert_avg_actions:.1f} | {hybrid_avg_actions:.1f} | {explore_avg_actions:.1f} |")
        lines.append(f"| 平均LLM Tokens | {expert_avg_tokens:.0f} | {hybrid_avg_tokens:.0f} | {explore_avg_tokens:.0f} |")
    lines.append("")
    
    # 速度对比
    lines.append("### 速度对比")
    lines.append("")
    if expert_avg_time > 0:
        lines.append(f"- **Hybrid vs Expert**: Hybrid 耗时是 Expert 的 **{hybrid_avg_time/expert_avg_time:.1f}x**")
        lines.append(f"- **Explore(思考关) vs Expert**: Explore 耗时是 Expert 的 **{explore_avg_time/expert_avg_time:.1f}x**")
        if has_explore_on:
            lines.append(f"- **Explore(思考开) vs Expert**: Explore 耗时是 Expert 的 **{explore_on_avg_time/expert_avg_time:.1f}x**")
    if hybrid_avg_time > 0:
        lines.append(f"- **Explore(思考关) vs Hybrid**: Explore 耗时是 Hybrid 的 **{explore_avg_time/hybrid_avg_time:.1f}x**")
    if has_explore_on and explore_avg_time > 0:
        lines.append(f"- **Explore(思考开) vs Explore(思考关)**: 思考开启耗时是关闭的 **{explore_on_avg_time/explore_avg_time:.1f}x**")
    lines.append("")
    
    # Token消耗对比
    lines.append("### LLM Token 消耗对比")
    lines.append("")
    if hybrid_avg_tokens > 0:
        lines.append(f"- Explore(思考关) 平均消耗 Tokens 是 Hybrid 的 **{explore_avg_tokens/hybrid_avg_tokens:.1f}x**")
        if has_explore_on:
            lines.append(f"- Explore(思考开) 平均消耗 Tokens 是 Hybrid 的 **{explore_on_avg_tokens/hybrid_avg_tokens:.1f}x**")
    if has_explore_on and explore_avg_tokens > 0:
        lines.append(f"- 开启思考后 Token 消耗增加 **{(explore_on_avg_tokens - explore_avg_tokens)/explore_avg_tokens*100:.1f}%**")
    lines.append("")
    
    # 分场景详细对比
    lines.append("## 4. 分场景详细对比")
    lines.append("")
    
    # 获取所有场景类型
    all_tasks = set()
    all_data_sources = [expert, hybrid, explore]
    if has_explore_on:
        all_data_sources.append(explore_thinking_on)
    for data in all_data_sources:
        all_tasks.update(k for k in data.keys() if not k.startswith("_"))
    
    # 按场景名排序
    sorted_tasks = sorted(all_tasks)
    
    for task_id in sorted_tasks:
        expert_task = expert.get(task_id, {})
        hybrid_task = hybrid.get(task_id, {})
        explore_task = explore.get(task_id, {})
        explore_on_task = explore_thinking_on.get(task_id, {}) if has_explore_on else {}
        
        # 获取任务描述和 F1 信息
        query = expert_task.get("query") or hybrid_task.get("query") or explore_task.get("query") or "未知任务"
        scenario = expert_task.get("scenario") or hybrid_task.get("scenario") or explore_task.get("scenario") or ""
        cover_f1 = expert_task.get("cover_f1") or hybrid_task.get("cover_f1") or explore_task.get("cover_f1") or 0
        
        lines.append(f"### {scenario}: {task_id}")
        lines.append("")
        lines.append(f"> **任务描述**: {query}")
        if cover_f1 > 0:
            lines.append(f">")
            lines.append(f"> **偏序图 Cover-F1**: {cover_f1:.4f}")
        lines.append("")
        
        if has_explore_on:
            lines.append("| 指标 | Expert | Hybrid | Explore (思考关) | Explore (思考开) |")
            lines.append("|------|--------|--------|------------------|------------------|")
        else:
            lines.append("| 指标 | Expert | Hybrid | Explore |")
            lines.append("|------|--------|--------|---------|")
        
        # 状态
        expert_status = expert_task.get("status", "-")
        hybrid_status = hybrid_task.get("status", "-")
        explore_status = explore_task.get("status", "-")
        explore_on_status = explore_on_task.get("status", "-") if has_explore_on else "-"
        if has_explore_on:
            lines.append(f"| 状态 | {_format_status(expert_status)} | {_format_status(hybrid_status)} | {_format_status(explore_status)} | {_format_status(explore_on_status)} |")
        else:
            lines.append(f"| 状态 | {_format_status(expert_status)} | {_format_status(hybrid_status)} | {_format_status(explore_status)} |")
        
        # 耗时
        expert_time = expert_task.get("duration_s", 0)
        hybrid_time = hybrid_task.get("duration_s", 0)
        explore_time = explore_task.get("duration_s", 0)
        explore_on_time = explore_on_task.get("duration_s", 0) if has_explore_on else 0
        if has_explore_on:
            lines.append(f"| 耗时(秒) | {expert_time:.2f} | {hybrid_time:.2f} | {explore_time:.2f} | {explore_on_time:.2f} |")
        else:
            lines.append(f"| 耗时(秒) | {expert_time:.2f} | {hybrid_time:.2f} | {explore_time:.2f} |")
        
        # 操作数
        expert_actions = expert_task.get("actions_count", 0)
        hybrid_actions = hybrid_task.get("actions_count", 0)
        explore_actions = explore_task.get("actions_count", 0)
        explore_on_actions = explore_on_task.get("actions_count", 0) if has_explore_on else 0
        if has_explore_on:
            lines.append(f"| 操作数 | {expert_actions} | {hybrid_actions} | {explore_actions} | {explore_on_actions} |")
        else:
            lines.append(f"| 操作数 | {expert_actions} | {hybrid_actions} | {explore_actions} |")
        
        # LLM Tokens
        expert_tokens = expert_task.get("llm_tokens", 0)
        hybrid_tokens = hybrid_task.get("llm_tokens", 0)
        explore_tokens = explore_task.get("llm_tokens", 0)
        explore_on_tokens = explore_on_task.get("llm_tokens", 0) if has_explore_on else 0
        if has_explore_on:
            lines.append(f"| LLM Tokens | {expert_tokens} | {hybrid_tokens:,} | {explore_tokens:,} | {explore_on_tokens:,} |")
        else:
            lines.append(f"| LLM Tokens | {expert_tokens} | {hybrid_tokens:,} | {explore_tokens:,} |")
        
        lines.append("")
        
        # 操作序列对比
        lines.append("**操作序列对比:**")
        lines.append("")
        
        expert_seq = expert_task.get("actions_executed", [])
        hybrid_seq = hybrid_task.get("actions_executed", [])
        explore_seq = explore_task.get("actions_executed", [])
        explore_on_seq = explore_on_task.get("actions_executed", []) if has_explore_on else []
        
        if has_explore_on:
            lines.append("| Expert | Hybrid | Explore (思考关) | Explore (思考开) |")
            lines.append("|--------|--------|------------------|------------------|")
            max_len = max(len(expert_seq), len(hybrid_seq), len(explore_seq), len(explore_on_seq))
            for i in range(max_len):
                e_action = expert_seq[i] if i < len(expert_seq) else "-"
                h_action = hybrid_seq[i] if i < len(hybrid_seq) else "-"
                x_action = explore_seq[i] if i < len(explore_seq) else "-"
                xo_action = explore_on_seq[i] if i < len(explore_on_seq) else "-"
                lines.append(f"| {e_action} | {h_action} | {x_action} | {xo_action} |")
        else:
            lines.append("| Expert | Hybrid | Explore |")
            lines.append("|--------|--------|---------|")
            max_len = max(len(expert_seq), len(hybrid_seq), len(explore_seq))
            for i in range(max_len):
                e_action = expert_seq[i] if i < len(expert_seq) else "-"
                h_action = hybrid_seq[i] if i < len(hybrid_seq) else "-"
                x_action = explore_seq[i] if i < len(explore_seq) else "-"
                lines.append(f"| {e_action} | {h_action} | {x_action} |")
        
        lines.append("")
    
    # 结论
    lines.append("## 5. 结论与分析")
    lines.append("")
    
    lines.append("### 各模式特点总结")
    lines.append("")
    lines.append("#### Expert 模式")
    lines.append("- **优势**: 执行速度最快，无LLM开销，结果稳定可预测，支持并行执行")
    lines.append("- **局限**: 依赖预定义的执行计划，灵活性较低，参数填充依赖预设")
    lines.append("")
    
    lines.append("#### Hybrid 模式")
    lines.append("- **优势**: 结合LLM智能与预定义计划，平衡效率和灵活性，支持降级")
    lines.append("- **局限**: 需要LLM调用进行意图识别，有一定延迟")
    lines.append("")
    
    lines.append("#### Explore 模式 (思考关闭)")
    lines.append("- **优势**: 完全自主探索，可处理未知场景，灵活性最高")
    lines.append("- **局限**: 耗时较长，Token消耗较大，可能产生冗余操作")
    lines.append("")
    
    if has_explore_on:
        lines.append("#### Explore 模式 (思考开启)")
        lines.append("- **优势**: 深度推理能力更强，复杂场景决策质量更高")
        lines.append("- **局限**: 耗时最长，Token消耗最大，思考过程增加开销")
        lines.append("")
    
    lines.append("### 性能建议")
    lines.append("")
    lines.append("1. **已知场景优先使用 Expert 模式** - 速度快、成本低、结果稳定")
    lines.append("2. **新场景或参数不确定时使用 Hybrid 模式** - 平衡效率和智能")
    lines.append("3. **完全陌生领域使用 Explore 模式** - 牺牲效率换取探索能力")
    if has_explore_on:
        lines.append("4. **复杂推理场景考虑开启思考** - 提升决策质量但增加成本")
    lines.append("")
    
    return "\n".join(lines)


def _format_status(status: str) -> str:
    """格式化状态显示"""
    if status == "success":
        return "✅ 成功"
    elif status == "failed" or status == "error":
        return "❌ 失败"
    elif status == "-":
        return "-"
    else:
        return status


def main():
    """主函数"""
    print("=" * 60)
    print("Trace 执行效果分析器")
    print("=" * 60)
    
    # 自动检测并解析所有模式
    print("\n[检测] 扫描目录结构...")
    all_data = auto_detect_and_parse()
    
    if not all_data:
        print("\nError: 未找到任何 traces 数据")
        print("请先运行实验脚本生成 traces:")
        print("  python mode_comparison_experiment.py --mode expert")
        print("  python mode_comparison_experiment.py --mode hybrid")
        print("  python mode_comparison_experiment.py --mode explore --thinking off")
        print("  python mode_comparison_experiment.py --mode explore --thinking on")
        return
    
    # 提取数据
    expert_data = all_data.get("expert", {})
    hybrid_data = all_data.get("hybrid", {})
    explore_data = all_data.get("explore", {})
    explore_thinking_on = all_data.get("explore_thinking_on", {})
    
    # 加载最优偏序图配置
    print("\n[加载] 最优偏序图配置...")
    best_posets = load_best_posets()
    if best_posets:
        print(f"  找到 {len(best_posets)} 个场景的最优偏序图配置")
    else:
        print("  未找到 best_posets.json，跳过偏序图总结")
    
    # 生成报告
    print("\n[生成] 创建对比报告...")
    report = generate_markdown_report(
        expert_data, 
        hybrid_data, 
        explore_data,
        explore_thinking_on=explore_thinking_on,
        best_posets=best_posets
    )
    
    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    legacy_output_file = WORKSPACE_DIR / "trace_comparison_report.md"
    if legacy_output_file != OUTPUT_FILE:
        with open(legacy_output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    print(f"\n✅ 报告已生成: {OUTPUT_FILE}")
    print("=" * 60)
    
    # 打印摘要
    print("\n📊 快速摘要:")
    expert_sum = expert_data.get("_summary", {})
    hybrid_sum = hybrid_data.get("_summary", {})
    explore_sum = explore_data.get("_summary", {})
    explore_on_sum = explore_thinking_on.get("_summary", {})
    
    if expert_sum:
        print(f"\n   Expert:  {expert_sum.get('success_count', 0)}/{expert_sum.get('total_scenarios', 0)} 成功, "
              f"耗时 {expert_sum.get('total_duration_s', 0):.1f}s, "
              f"Tokens: {expert_sum.get('total_llm_tokens', 0)}")
    
    if hybrid_sum:
        print(f"   Hybrid:  {hybrid_sum.get('success_count', 0)}/{hybrid_sum.get('total_scenarios', 0)} 成功, "
              f"耗时 {hybrid_sum.get('total_duration_s', 0):.1f}s, "
              f"Tokens: {hybrid_sum.get('total_llm_tokens', 0):,}")
    
    if explore_sum:
        print(f"   Explore (thinking off): {explore_sum.get('success_count', 0)}/{explore_sum.get('total_scenarios', 0)} 成功, "
              f"耗时 {explore_sum.get('total_duration_s', 0):.1f}s, "
              f"Tokens: {explore_sum.get('total_llm_tokens', 0):,}")
    
    if explore_on_sum:
        print(f"   Explore (thinking on):  {explore_on_sum.get('success_count', 0)}/{explore_on_sum.get('total_scenarios', 0)} 成功, "
              f"耗时 {explore_on_sum.get('total_duration_s', 0):.1f}s, "
              f"Tokens: {explore_on_sum.get('total_llm_tokens', 0):,}")


if __name__ == "__main__":
    main()
