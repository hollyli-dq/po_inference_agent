#!/usr/bin/env python3
"""
最优偏序图选择器

从 experiment_summary.csv 中为每个场景选择 cover_f1 最高的 BHPOP 偏序图配置。
用于三种模式对比实验。

用法:
    # 直接运行查看选择结果
    python best_poset_selector.py
    
    # 在其他脚本中使用
    from best_poset_selector import get_best_posets
    best_posets = get_best_posets()
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


# 场景列表
SCENARIOS = [
    "simple_ecs",
    "slb_ecs_rds",
    "slb_ecs_redis",
    "eip_slb_ecs",
    "dual_zone_ecs_slb",
    "dual_zone_ecs_slb_rds",
]

# 场景对应的任务 query
SCENARIO_QUERIES = {
    "simple_ecs": "在杭州可用区H创建一个2核4G的ECS实例",
    "slb_ecs_rds": "创建一个完整的Web应用架构：包含一个SLB负载均衡器，后端挂载一台ECS服务器，并配置一个RDS MySQL数据库",
    "slb_ecs_redis": "搭建一个带缓存的Web服务：创建SLB负载均衡、一台ECS实例、以及Redis缓存实例",
    "eip_slb_ecs": "创建一个公网可访问的负载均衡架构：申请一个EIP弹性公网IP，绑定到SLB，SLB后端挂载一台ECS",
    "dual_zone_ecs_slb": "创建高可用架构：在杭州的两个不同可用区各创建一台ECS实例，并创建一个SLB将这两台ECS作为后端服务器",
    "dual_zone_ecs_slb_rds": "创建完整的高可用生产环境：双可用区部署两台ECS，创建SLB做负载均衡，并配置RDS MySQL主备高可用集群",
}


def get_best_posets(
    summary_csv: Optional[str] = None,
    hpo_dir: Optional[str] = None,
    method: str = "bhpop_single_po"
) -> Dict[str, Dict[str, Any]]:
    """
    为每个场景选择 cover_f1 最高的偏序图配置
    
    Args:
        summary_csv: experiment_summary.csv 路径，默认自动查找
        hpo_dir: HPO_scenarios 目录路径，默认自动查找
        method: 使用的方法，默认 bhpop_single_po
        
    Returns:
        {scenario_name: {
            "ip_cov_target": float,
            "eps_jump": float,
            "cover_f1": float,
            "exp_dir": str,        # 实验目录路径
            "summary_path": str,   # summary.json 路径
            "query": str,          # 任务描述
        }}
    """
    # 默认路径
    current_dir = Path(__file__).parent
    if summary_csv is None:
        summary_csv = current_dir / "HPO_scenarios" / "experiment_summary.csv"
    if hpo_dir is None:
        hpo_dir = current_dir / "HPO_scenarios"
    
    summary_csv = Path(summary_csv)
    hpo_dir = Path(hpo_dir)
    
    if not summary_csv.exists():
        raise FileNotFoundError(f"找不到 experiment_summary.csv: {summary_csv}")
    
    # 加载 CSV
    df = pd.read_csv(summary_csv)
    
    # 过滤指定方法
    method_df = df[df['method'] == method].copy()
    
    if len(method_df) == 0:
        raise ValueError(f"CSV 中没有方法 {method} 的数据")
    
    # 为每个场景选择 F1 最高的配置
    best_posets = {}
    
    for scenario in SCENARIOS:
        scenario_df = method_df[method_df['scenario'] == scenario]
        
        if len(scenario_df) == 0:
            print(f"Warning: 场景 {scenario} 无 {method} 数据")
            continue
        
        # 选择 cover_f1 最高的
        best_row = scenario_df.loc[scenario_df['cover_f1'].idxmax()]
        
        ip_cov_target = best_row['ip_cov_target']
        eps_jump = best_row['eps_jump']
        cover_f1 = best_row['cover_f1']
        
        # 查找对应的实验目录
        exp_dir = find_experiment_dir(hpo_dir, scenario, ip_cov_target, eps_jump)
        
        if exp_dir is None:
            print(f"Warning: 找不到场景 {scenario} ip_cov={ip_cov_target} eps={eps_jump} 的实验目录")
            continue
        
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            print(f"Warning: 找不到 summary.json: {summary_path}")
            continue
        
        best_posets[scenario] = {
            "ip_cov_target": ip_cov_target,
            "eps_jump": eps_jump,
            "cover_f1": cover_f1,
            "exp_dir": str(exp_dir),
            "summary_path": str(summary_path),
            "query": SCENARIO_QUERIES.get(scenario, ""),
        }
    
    return best_posets


def find_experiment_dir(
    hpo_dir: Path, 
    scenario: str, 
    ip_cov_target: float, 
    eps_jump: float
) -> Optional[Path]:
    """
    查找指定配置的实验目录
    
    实验目录命名格式: exp_{id}_{scenario}
    通过读取 summary.json 匹配 ip_cov_target 和 eps_jump
    """
    for exp_dir in hpo_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.startswith("exp_"):
            continue
        if scenario not in exp_dir.name:
            continue
        
        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            continue
        
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            config = summary.get("configuration", {})
            cfg_ip_cov = config.get("ip_cov_target", 0)
            cfg_eps = config.get("eps_jump", 0)
            
            # 浮点数比较，允许微小误差
            if abs(cfg_ip_cov - ip_cov_target) < 0.001 and abs(cfg_eps - eps_jump) < 0.0001:
                return exp_dir
                
        except Exception:
            continue
    
    return None


def print_best_posets(best_posets: Dict[str, Dict[str, Any]]) -> None:
    """打印最优偏序图选择结果"""
    print("=" * 80)
    print("  最优偏序图选择结果 (每场景 F1 最高的 BHPOP 配置)")
    print("=" * 80)
    print()
    
    print(f"{'场景':<25} {'IP-Cov':>8} {'eps_jump':>10} {'Cover-F1':>10}")
    print("-" * 60)
    
    for scenario, info in sorted(best_posets.items()):
        print(f"{scenario:<25} {info['ip_cov_target']:>8.2f} {info['eps_jump']:>10.3f} {info['cover_f1']:>10.4f}")
    
    print()
    print(f"共选择了 {len(best_posets)} 个场景的最优偏序图")


def save_best_posets(best_posets: Dict[str, Dict[str, Any]], output_path: str = None) -> str:
    """保存最优偏序图选择结果到 JSON 文件"""
    if output_path is None:
        current_dir = Path(__file__).parent
        output_path = current_dir / "best_posets.json"
    
    output_path = Path(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_posets, f, ensure_ascii=False, indent=2)
    
    return str(output_path)


def main():
    """主函数"""
    try:
        best_posets = get_best_posets()
        print_best_posets(best_posets)
        
        # 保存结果
        output_path = save_best_posets(best_posets)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
