"""Trace 多样性分析器

分析多样化 trace 生成的效果，评估是否满足偏序推导需求。

分析维度：
1. 基础统计：按场景/模型/温度的成功率、耗时、token消耗
2. 序列多样性：同场景下不同执行序列的数量和差异度
3. 偏序覆盖：从 trace 推导的偏序关系 vs 手工偏序图的覆盖率

用法：
  python trace_diversity_analyzer.py [traces_dir]
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TraceInfo:
    """解析后的 Trace 信息"""
    trace_id: str
    scenario_id: int
    intent_type: str
    model: str
    temperature: float
    status: str
    action_sequence: List[str]  # 成功执行的 API 序列
    all_actions: List[str]      # 所有 API（包括失败的）
    duration_ms: float
    tokens: int
    query: str


@dataclass
class ScenarioStats:
    """场景统计"""
    scenario_id: int
    intent_type: str
    total: int = 0
    success: int = 0
    unique_sequences: Set[Tuple[str, ...]] = field(default_factory=set)
    traces: List[TraceInfo] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0
    
    @property
    def diversity_score(self) -> float:
        """多样性得分 = 唯一序列数 / 成功 trace 数"""
        return len(self.unique_sequences) / self.success if self.success > 0 else 0


@dataclass 
class ModelStats:
    """模型统计"""
    model: str
    temperature: float
    total: int = 0
    success: int = 0
    total_duration_ms: float = 0
    total_tokens: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0
    
    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.total if self.total > 0 else 0
    
    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total if self.total > 0 else 0


# ============================================================
# 偏序分析工具
# ============================================================

def extract_edges_from_sequence(sequence: List[str]) -> Set[Tuple[str, str]]:
    """从执行序列中提取所有偏序边（前->后依赖）"""
    edges = set()
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            edges.add((sequence[i], sequence[j]))
    return edges


def extract_must_before_edges(sequences: List[List[str]]) -> Set[Tuple[str, str]]:
    """
    从多条序列中提取"必须在前"的偏序约束
    
    如果 A 在所有序列中都在 B 之前，则 A->B 是必须的偏序
    """
    if not sequences:
        return set()
    
    # 收集所有出现的操作
    all_actions = set()
    for seq in sequences:
        all_actions.update(seq)
    
    # 检查每对操作
    must_before = set()
    for a, b in combinations(all_actions, 2):
        a_always_before_b = True
        b_always_before_a = True
        
        for seq in sequences:
            if a in seq and b in seq:
                idx_a = seq.index(a)
                idx_b = seq.index(b)
                if idx_a >= idx_b:
                    a_always_before_b = False
                if idx_b >= idx_a:
                    b_always_before_a = False
        
        if a_always_before_b and not b_always_before_a:
            must_before.add((a, b))
        elif b_always_before_a and not a_always_before_b:
            must_before.add((b, a))
    
    return must_before


def compute_sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个序列的相似度 (0-1)
    使用 Jaccard 相似度 + 顺序惩罚
    """
    if not seq1 or not seq2:
        return 0.0
    
    set1, set2 = set(seq1), set(seq2)
    jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
    
    # 顺序相似度：相同元素的相对顺序
    common = list(set1 & set2)
    if len(common) < 2:
        return jaccard
    
    order_same = 0
    order_total = 0
    for i, a in enumerate(common):
        for b in common[i+1:]:
            order_total += 1
            idx1_a = seq1.index(a) if a in seq1 else -1
            idx1_b = seq1.index(b) if b in seq1 else -1
            idx2_a = seq2.index(a) if a in seq2 else -1
            idx2_b = seq2.index(b) if b in seq2 else -1
            
            if idx1_a >= 0 and idx1_b >= 0 and idx2_a >= 0 and idx2_b >= 0:
                if (idx1_a < idx1_b) == (idx2_a < idx2_b):
                    order_same += 1
    
    order_sim = order_same / order_total if order_total > 0 else 1.0
    
    return 0.5 * jaccard + 0.5 * order_sim


# ============================================================
# 加载数据
# ============================================================

def load_traces(traces_dir: str) -> List[TraceInfo]:
    """加载所有 trace JSON 文件"""
    traces = []
    traces_path = Path(traces_dir)
    
    for json_file in traces_path.glob("*.json"):
        if json_file.name == "generation_summary.json":
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            gen_config = data.get("generation_config", {})
            intent = data.get("intent", {})
            
            trace = TraceInfo(
                trace_id=data.get("trace_id", ""),
                scenario_id=gen_config.get("scenario_id", 0),
                intent_type=intent.get("intent_type", "unknown"),
                model=gen_config.get("model", "unknown"),
                temperature=gen_config.get("temperature", 0.0),
                status=data.get("status", "unknown"),
                action_sequence=data.get("action_sequence", []),
                all_actions=[a.get("action_name", "") for a in data.get("actions", [])],
                duration_ms=data.get("duration_ms", 0),
                tokens=data.get("total_llm_tokens", 0),
                query=gen_config.get("query", "")
            )
            traces.append(trace)
        except Exception as e:
            print(f"Warning: 无法解析 {json_file.name}: {e}")
    
    return traces


def load_manual_posets(manual_dir: str) -> Dict[str, Set[Tuple[str, str]]]:
    """加载手工偏序图"""
    posets = {}
    manual_path = Path(manual_dir)
    
    for json_file in manual_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            name = json_file.stem  # 文件名作为场景名
            edges = set()
            for edge in data.get("edges", []):
                if len(edge) == 2:
                    edges.add((edge[0], edge[1]))
            
            posets[name] = edges
        except Exception as e:
            print(f"Warning: 无法解析 {json_file.name}: {e}")
    
    return posets


# ============================================================
# 分析逻辑
# ============================================================

def analyze_traces(traces: List[TraceInfo], manual_posets: Dict[str, Set[Tuple[str, str]]]) -> Dict[str, Any]:
    """执行完整分析"""
    
    # 1. 按场景分组
    by_scenario: Dict[int, ScenarioStats] = {}
    for t in traces:
        if t.scenario_id not in by_scenario:
            by_scenario[t.scenario_id] = ScenarioStats(
                scenario_id=t.scenario_id,
                intent_type=t.intent_type
            )
        
        stats = by_scenario[t.scenario_id]
        stats.total += 1
        stats.traces.append(t)
        
        if t.status == "success":
            stats.success += 1
            seq_tuple = tuple(t.action_sequence)
            stats.unique_sequences.add(seq_tuple)
    
    # 2. 按模型/温度分组
    by_model: Dict[str, ModelStats] = {}
    for t in traces:
        key = f"{t.model}_t{t.temperature}"
        if key not in by_model:
            by_model[key] = ModelStats(model=t.model, temperature=t.temperature)
        
        stats = by_model[key]
        stats.total += 1
        stats.total_duration_ms += t.duration_ms
        stats.total_tokens += t.tokens
        
        if t.status == "success":
            stats.success += 1
    
    # 3. 偏序分析（按场景）
    poset_analysis = {}
    for scenario_id, stats in by_scenario.items():
        success_sequences = [
            t.action_sequence for t in stats.traces 
            if t.status == "success" and t.action_sequence
        ]
        
        if not success_sequences:
            continue
        
        # 从 trace 推导的必须偏序
        inferred_edges = extract_must_before_edges(success_sequences)
        
        # 找对应的手工偏序图
        intent_type = stats.intent_type
        manual_edges = manual_posets.get(intent_type, set())
        
        # 计算覆盖率
        if manual_edges:
            covered = inferred_edges & manual_edges
            coverage = len(covered) / len(manual_edges)
            extra = inferred_edges - manual_edges
        else:
            covered = set()
            coverage = 0
            extra = inferred_edges
        
        poset_analysis[scenario_id] = {
            "intent_type": intent_type,
            "inferred_edges": inferred_edges,
            "manual_edges": manual_edges,
            "covered": covered,
            "coverage": coverage,
            "extra_edges": extra,
            "num_sequences": len(success_sequences),
            "unique_sequences": len(stats.unique_sequences)
        }
    
    # 4. 序列间相似度分析
    similarity_analysis = {}
    for scenario_id, stats in by_scenario.items():
        success_sequences = [
            t.action_sequence for t in stats.traces 
            if t.status == "success" and t.action_sequence
        ]
        
        if len(success_sequences) < 2:
            continue
        
        # 计算所有序列对的相似度
        similarities = []
        for i, seq1 in enumerate(success_sequences):
            for seq2 in success_sequences[i+1:]:
                sim = compute_sequence_similarity(seq1, seq2)
                similarities.append(sim)
        
        avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
        
        similarity_analysis[scenario_id] = {
            "avg_similarity": avg_sim,
            "diversity_score": 1 - avg_sim,  # 多样性 = 1 - 相似度
            "num_pairs": len(similarities),
            "min_sim": min(similarities) if similarities else 0,
            "max_sim": max(similarities) if similarities else 0
        }
    
    return {
        "by_scenario": by_scenario,
        "by_model": by_model,
        "poset_analysis": poset_analysis,
        "similarity_analysis": similarity_analysis
    }


# ============================================================
# 报告生成
# ============================================================

def print_report(analysis: Dict[str, Any], manual_posets: Dict[str, Set[Tuple[str, str]]]):
    """打印分析报告"""
    
    by_scenario = analysis["by_scenario"]
    by_model = analysis["by_model"]
    poset_analysis = analysis["poset_analysis"]
    similarity_analysis = analysis["similarity_analysis"]
    
    print("=" * 80)
    print("  📊 Trace 多样性分析报告")
    print("=" * 80)
    
    # ---- 总体统计 ----
    total_traces = sum(s.total for s in by_scenario.values())
    total_success = sum(s.success for s in by_scenario.values())
    total_unique_seqs = sum(len(s.unique_sequences) for s in by_scenario.values())
    
    print(f"\n【总体统计】")
    print(f"  总 Trace 数: {total_traces}")
    print(f"  成功数: {total_success} ({total_success/total_traces*100:.1f}%)")
    print(f"  唯一执行序列数: {total_unique_seqs}")
    
    # ---- 按场景统计 ----
    print(f"\n【按场景统计】")
    print("-" * 80)
    print(f"{'场景ID':^8} {'意图类型':^20} {'总数':^6} {'成功':^6} {'成功率':^8} {'唯一序列':^10} {'多样性':^8}")
    print("-" * 80)
    
    for scenario_id in sorted(by_scenario.keys()):
        stats = by_scenario[scenario_id]
        print(f"{scenario_id:^8} {stats.intent_type:^20} {stats.total:^6} {stats.success:^6} "
              f"{stats.success_rate*100:^7.1f}% {len(stats.unique_sequences):^10} {stats.diversity_score:^8.2f}")
    
    # ---- 按模型统计 ----
    print(f"\n【按模型/温度统计】")
    print("-" * 80)
    print(f"{'模型':^20} {'温度':^6} {'总数':^6} {'成功':^6} {'成功率':^8} {'平均耗时':^12} {'平均Token':^10}")
    print("-" * 80)
    
    for key in sorted(by_model.keys()):
        stats = by_model[key]
        print(f"{stats.model:^20} {stats.temperature:^6.1f} {stats.total:^6} {stats.success:^6} "
              f"{stats.success_rate*100:^7.1f}% {stats.avg_duration_ms/1000:^11.1f}s {stats.avg_tokens:^10.0f}")
    
    # ---- 序列多样性分析 ----
    print(f"\n【序列多样性分析】")
    print("-" * 80)
    print(f"{'场景ID':^8} {'唯一序列':^10} {'序列对数':^10} {'平均相似度':^12} {'多样性得分':^12}")
    print("-" * 80)
    
    for scenario_id in sorted(similarity_analysis.keys()):
        sim = similarity_analysis[scenario_id]
        stats = by_scenario[scenario_id]
        print(f"{scenario_id:^8} {len(stats.unique_sequences):^10} {sim['num_pairs']:^10} "
              f"{sim['avg_similarity']:^12.3f} {sim['diversity_score']:^12.3f}")
    
    # ---- 偏序覆盖分析 ----
    print(f"\n【偏序覆盖分析】")
    print("-" * 80)
    
    for scenario_id in sorted(poset_analysis.keys()):
        pa = poset_analysis[scenario_id]
        intent_type = pa["intent_type"]
        
        print(f"\n场景 {scenario_id} ({intent_type}):")
        print(f"  成功序列数: {pa['num_sequences']}, 唯一序列数: {pa['unique_sequences']}")
        print(f"  推导偏序边数: {len(pa['inferred_edges'])}")
        print(f"  手工偏序边数: {len(pa['manual_edges'])}")
        
        if pa['manual_edges']:
            print(f"  覆盖率: {pa['coverage']*100:.1f}%")
            print(f"  已覆盖: {pa['covered']}")
            uncovered = pa['manual_edges'] - pa['covered']
            if uncovered:
                print(f"  未覆盖: {uncovered}")
        
        if pa['extra_edges']:
            print(f"  额外推导(非必须): {pa['extra_edges']}")
    
    # ---- 执行序列详情 ----
    print(f"\n【各场景唯一执行序列】")
    print("-" * 80)
    
    for scenario_id in sorted(by_scenario.keys()):
        stats = by_scenario[scenario_id]
        print(f"\n场景 {scenario_id} ({stats.intent_type}): {len(stats.unique_sequences)} 种序列")
        
        for i, seq in enumerate(sorted(stats.unique_sequences), 1):
            seq_str = " -> ".join(seq)
            print(f"  [{i}] {seq_str}")
    
    # ---- 结论 ----
    print(f"\n" + "=" * 80)
    print("【分析结论】")
    print("=" * 80)
    
    # 多样性评估
    avg_diversity = sum(s.diversity_score for s in by_scenario.values()) / len(by_scenario) if by_scenario else 0
    
    if avg_diversity > 0.5:
        diversity_verdict = "✅ 良好 - 多数场景产生了多样化的执行序列"
    elif avg_diversity > 0.2:
        diversity_verdict = "⚠️ 一般 - 部分场景序列较单一"
    else:
        diversity_verdict = "❌ 不足 - 序列高度相似，需增加扰动"
    
    print(f"\n1. 多样性评估: {diversity_verdict}")
    print(f"   平均多样性得分: {avg_diversity:.2f}")
    
    # 偏序覆盖评估
    coverages = [pa["coverage"] for pa in poset_analysis.values() if pa["manual_edges"]]
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        if avg_coverage > 0.8:
            coverage_verdict = "✅ 良好 - 大部分必须偏序已覆盖"
        elif avg_coverage > 0.5:
            coverage_verdict = "⚠️ 部分覆盖 - 需更多 trace 样本"
        else:
            coverage_verdict = "❌ 覆盖不足 - trace 数量或多样性需提升"
        print(f"\n2. 偏序覆盖: {coverage_verdict}")
        print(f"   平均覆盖率: {avg_coverage*100:.1f}%")
    else:
        print(f"\n2. 偏序覆盖: 无手工偏序图可对比")
    
    # 模型表现
    best_model = max(by_model.values(), key=lambda x: x.success_rate)
    print(f"\n3. 最佳模型: {best_model.model} (t={best_model.temperature}) - 成功率 {best_model.success_rate*100:.1f}%")
    
    # 建议
    print(f"\n4. 建议:")
    if avg_diversity < 0.3:
        print("   - 增加更多模型变体或提高温度参数")
    if coverages and avg_coverage < 0.7:
        print("   - 增加 trace 生成数量以覆盖更多偏序边")
    
    low_success_models = [k for k, v in by_model.items() if v.success_rate < 0.5]
    if low_success_models:
        print(f"   - 考虑移除低成功率模型: {low_success_models}")


def save_analysis_markdown(analysis: Dict[str, Any], output_path: str):
    """保存分析结果为 Markdown 文档"""
    
    by_scenario = analysis["by_scenario"]
    by_model = analysis["by_model"]
    poset_analysis = analysis["poset_analysis"]
    similarity_analysis = analysis["similarity_analysis"]
    
    # 计算总体指标
    total_traces = sum(s.total for s in by_scenario.values())
    total_success = sum(s.success for s in by_scenario.values())
    total_unique_seqs = sum(len(s.unique_sequences) for s in by_scenario.values())
    avg_diversity = sum(s.diversity_score for s in by_scenario.values()) / len(by_scenario) if by_scenario else 0
    
    coverages = [pa["coverage"] for pa in poset_analysis.values() if pa["manual_edges"]]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0
    
    lines = []
    lines.append("# Trace 多样性分析报告")
    lines.append("")
    lines.append(f"> 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 摘要
    lines.append("## 1. 摘要")
    lines.append("")
    lines.append("| 指标 | 结果 | 评价 |")
    lines.append("|------|------|------|")
    lines.append(f"| 总 Trace 数 | {total_traces} | - |")
    lines.append(f"| 成功率 | {total_success/total_traces*100:.1f}% ({total_success}/{total_traces}) | {'✅ 优秀' if total_success/total_traces > 0.9 else '⚠️ 一般'} |")
    lines.append(f"| 唯一序列数 | {total_unique_seqs} | - |")
    lines.append(f"| 平均多样性得分 | {avg_diversity:.2f} | {'✅ 良好' if avg_diversity > 0.5 else '⚠️ 不足'} |")
    lines.append(f"| 偏序覆盖率 | {avg_coverage*100:.1f}% | {'✅ 良好' if avg_coverage > 0.8 else '⚠️ 需提升'} |")
    lines.append("")
    
    # 按场景统计
    lines.append("## 2. 按场景统计")
    lines.append("")
    lines.append("| 场景ID | 意图类型 | 总数 | 成功 | 成功率 | 唯一序列 | 多样性得分 |")
    lines.append("|--------|----------|------|------|--------|----------|------------|")
    
    for scenario_id in sorted(by_scenario.keys()):
        stats = by_scenario[scenario_id]
        lines.append(f"| {scenario_id} | {stats.intent_type} | {stats.total} | {stats.success} | "
                     f"{stats.success_rate*100:.1f}% | {len(stats.unique_sequences)} | {stats.diversity_score:.2f} |")
    lines.append("")
    
    # 按模型统计
    lines.append("## 3. 按模型/温度统计")
    lines.append("")
    lines.append("| 模型 | 温度 | 总数 | 成功 | 成功率 | 平均耗时 | 平均Token |")
    lines.append("|------|------|------|------|--------|----------|-----------|")
    
    for key in sorted(by_model.keys()):
        stats = by_model[key]
        lines.append(f"| {stats.model} | {stats.temperature:.1f} | {stats.total} | {stats.success} | "
                     f"{stats.success_rate*100:.1f}% | {stats.avg_duration_ms/1000:.1f}s | {stats.avg_tokens:.0f} |")
    lines.append("")
    
    # 序列多样性分析
    lines.append("## 4. 序列多样性分析")
    lines.append("")
    lines.append("| 场景ID | 唯一序列 | 序列对数 | 平均相似度 | 多样性得分 |")
    lines.append("|--------|----------|----------|------------|------------|")
    
    for scenario_id in sorted(similarity_analysis.keys()):
        sim = similarity_analysis[scenario_id]
        stats = by_scenario[scenario_id]
        lines.append(f"| {scenario_id} | {len(stats.unique_sequences)} | {sim['num_pairs']} | "
                     f"{sim['avg_similarity']:.3f} | {sim['diversity_score']:.3f} |")
    lines.append("")
    
    # 偏序覆盖分析
    lines.append("## 5. 偏序覆盖分析")
    lines.append("")
    
    for scenario_id in sorted(poset_analysis.keys()):
        pa = poset_analysis[scenario_id]
        intent_type = pa["intent_type"]
        
        lines.append(f"### 场景 {scenario_id}: {intent_type}")
        lines.append("")
        lines.append(f"- 成功序列数: {pa['num_sequences']}")
        lines.append(f"- 唯一序列数: {pa['unique_sequences']}")
        lines.append(f"- 推导偏序边数: {len(pa['inferred_edges'])}")
        lines.append(f"- 手工偏序边数: {len(pa['manual_edges'])}")
        
        if pa['manual_edges']:
            lines.append(f"- **覆盖率: {pa['coverage']*100:.1f}%**")
            lines.append("")
            lines.append("**已覆盖的偏序边:**")
            lines.append("```")
            for edge in sorted(pa['covered']):
                lines.append(f"  {edge[0]} -> {edge[1]}")
            lines.append("```")
            
            uncovered = pa['manual_edges'] - pa['covered']
            if uncovered:
                lines.append("")
                lines.append("**未覆盖的偏序边:**")
                lines.append("```")
                for edge in sorted(uncovered):
                    lines.append(f"  {edge[0]} -> {edge[1]}")
                lines.append("```")
        lines.append("")
    
    # 各场景唯一执行序列
    lines.append("## 6. 各场景唯一执行序列")
    lines.append("")
    
    for scenario_id in sorted(by_scenario.keys()):
        stats = by_scenario[scenario_id]
        lines.append(f"### 场景 {scenario_id} ({stats.intent_type}): {len(stats.unique_sequences)} 种序列")
        lines.append("")
        
        for i, seq in enumerate(sorted(stats.unique_sequences), 1):
            seq_str = " → ".join(seq)
            lines.append(f"{i}. `{seq_str}`")
        lines.append("")
    
    # 结论与建议
    lines.append("## 7. 结论与建议")
    lines.append("")
    
    # 多样性评估
    if avg_diversity > 0.5:
        lines.append("### 多样性评估: ✅ 良好")
        lines.append("多数场景产生了多样化的执行序列，满足偏序推导需求。")
    elif avg_diversity > 0.2:
        lines.append("### 多样性评估: ⚠️ 一般")
        lines.append("部分场景序列较单一，建议增加模型变体或提高温度参数。")
    else:
        lines.append("### 多样性评估: ❌ 不足")
        lines.append("序列高度相似，需大幅增加扰动。")
    lines.append("")
    
    # 偏序覆盖评估
    if coverages:
        if avg_coverage > 0.8:
            lines.append("### 偏序覆盖: ✅ 良好")
            lines.append("大部分必须偏序已被 trace 数据覆盖。")
        elif avg_coverage > 0.5:
            lines.append("### 偏序覆盖: ⚠️ 部分覆盖")
            lines.append("需更多 trace 样本以覆盖全部偏序边。")
        else:
            lines.append("### 偏序覆盖: ❌ 覆盖不足")
            lines.append("trace 数量或多样性需提升。")
    lines.append("")
    
    # 最佳模型
    best_model = max(by_model.values(), key=lambda x: (x.success_rate, -x.avg_duration_ms))
    lines.append(f"### 最佳模型推荐")
    lines.append(f"**{best_model.model}** (temperature={best_model.temperature})")
    lines.append(f"- 成功率: {best_model.success_rate*100:.1f}%")
    lines.append(f"- 平均耗时: {best_model.avg_duration_ms/1000:.1f}s")
    lines.append(f"- 平均Token: {best_model.avg_tokens:.0f}")
    lines.append("")
    
    # 建议
    lines.append("### 优化建议")
    suggestions = []
    if avg_diversity < 0.3:
        suggestions.append("- 增加更多模型变体或提高温度参数")
    if coverages and avg_coverage < 0.7:
        suggestions.append("- 增加 trace 生成数量以覆盖更多偏序边")
    
    low_success_models = [f"{v.model} (t={v.temperature})" for k, v in by_model.items() if v.success_rate < 0.8]
    if low_success_models:
        suggestions.append(f"- 考虑移除低成功率模型: {', '.join(low_success_models)}")
    
    if not suggestions:
        suggestions.append("- 当前配置已满足偏序推导需求，无需调整")
    
    for s in suggestions:
        lines.append(s)
    lines.append("")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\n分析报告已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    # 默认路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_traces_dir = os.path.join(script_dir, "traces")
    default_manual_dir = os.path.join(script_dir, "manual_scenarios")
    
    # 解析参数
    traces_dir = sys.argv[1] if len(sys.argv) > 1 else default_traces_dir
    manual_dir = sys.argv[2] if len(sys.argv) > 2 else default_manual_dir
    
    print(f"Traces 目录: {traces_dir}")
    print(f"手工偏序图目录: {manual_dir}")
    
    # 加载数据
    traces = load_traces(traces_dir)
    if not traces:
        print("Error: 未找到任何 trace 文件")
        sys.exit(1)
    
    print(f"已加载 {len(traces)} 条 trace")
    
    manual_posets = load_manual_posets(manual_dir)
    print(f"已加载 {len(manual_posets)} 个手工偏序图: {list(manual_posets.keys())}")
    
    # 执行分析
    analysis = analyze_traces(traces, manual_posets)
    
    # 打印报告
    print_report(analysis, manual_posets)
    
    # 保存 Markdown 结果（输出到仿真目录）
    output_path = os.path.join(script_dir, "diversity_report.md")
    save_analysis_markdown(analysis, output_path)


if __name__ == "__main__":
    main()
