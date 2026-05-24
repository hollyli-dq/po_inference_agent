import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from execution.cloudops_agent.mining.io_miner import IOMiner

def main():
    miner = IOMiner()
    
    # 目标目录：simulation_workspace/traces
    # 假设脚本在 src/cloudops_agent/mining/ 目录下
    # ../../../simulation_workspace/traces
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../../"))
    traces_dir = os.path.join(project_root, "simulation_workspace", "traces")
    
    print(f"Mining IO Rules from all traces in: {traces_dir}")
    
    if not os.path.exists(traces_dir):
        print(f"Error: Traces directory not found at {traces_dir}")
        return

    miner.load_traces_from_dir(traces_dir)
    
    io_rules = miner.export_io_registry()
    
    # 输出目录：simulation_workspace/mined_artifacts
    output_dir = os.path.join(project_root, "simulation_workspace", "mined_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "io_registry_rules.json")
    with open(output_path, 'w') as f:
        json.dump(io_rules, f, indent=2)
        
    print(f"\nMined rules saved to: {output_path}")
    print(f"Total actions mined: {len(io_rules)}")
    
    # 打印一些统计信息
    print("\nMined Actions:")
    for action in sorted(io_rules.keys()):
        print(f"  - {action}: {len(io_rules[action]['inputs'])} inputs, {len(io_rules[action]['outputs'])} outputs")

if __name__ == "__main__":
    main()
