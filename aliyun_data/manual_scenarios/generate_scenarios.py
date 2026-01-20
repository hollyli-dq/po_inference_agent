import json
import os
import sys
import glob

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cloudops_agent.planning.poset_planner import PosetGraph

# Scenarios aligned with intent_parser.py and trace_tasks.json
scenarios_config = [
    {
        "filename": "simple_ecs.json",
        "name": "Simple ECS",
        "description": "Deploys a VPC network and a single ECS instance.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"]
        ]
    },
    {
        "filename": "slb_ecs_rds.json",
        "name": "SLB + ECS + RDS",
        "description": "Full Stack: SLB load balancer, ECS application, and RDS database.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"],
            ["CreateVSwitch", "CreateLoadBalancer"],
            ["CreateLoadBalancer", "CreateLoadBalancerHTTPListener"],
            ["CreateLoadBalancerHTTPListener", "StartLoadBalancerListener"],
            ["RunInstances", "AddBackendServers"],
            ["CreateLoadBalancer", "AddBackendServers"],
            ["CreateVSwitch", "CreateDBInstance"],
            ["CreateDBInstance", "CreateAccount"],
            ["CreateDBInstance", "ModifySecurityIps"],
            ["RunInstances", "ModifySecurityIps"]
        ]
    },
    {
        "filename": "slb_ecs_redis.json",
        "name": "SLB + ECS + Redis",
        "description": "Web App with Cache: SLB, ECS, and Redis.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"],
            ["CreateVSwitch", "CreateLoadBalancer"],
            ["RunInstances", "AddBackendServers"],
            ["CreateLoadBalancer", "AddBackendServers"],
            ["CreateVSwitch", "CreateInstance"],
            ["CreateInstance", "DescribeInstanceAttribute"]
        ]
    },
    {
        "filename": "eip_slb_ecs.json",
        "name": "EIP + SLB + ECS",
        "description": "Public Facing Web: EIP bound to SLB with ECS backend.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"],
            ["CreateVSwitch", "CreateLoadBalancer"],
            ["RunInstances", "AddBackendServers"],
            ["CreateLoadBalancer", "AddBackendServers"],
            ["CreateLoadBalancer", "AssociateEipAddress"],
            ["AllocateEipAddress", "AssociateEipAddress"]
        ]
    },
    {
        "filename": "dual_zone_ecs_slb.json",
        "name": "Dual Zone ECS + SLB",
        "description": "High Availability: ECS instances in two zones with SLB.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"],
            ["CreateVSwitch", "CreateLoadBalancer"],
            ["RunInstances", "AddBackendServers"],
            ["CreateLoadBalancer", "AddBackendServers"]
        ]
    },
    {
        "filename": "dual_zone_ecs_slb_rds.json",
        "name": "Dual Zone ECS + SLB + RDS",
        "description": "Production HA: Dual zone ECS, SLB, and RDS HA.",
        "edges": [
            ["CreateVpc", "CreateVSwitch"],
            ["CreateVpc", "CreateSecurityGroup"],
            ["CreateSecurityGroup", "AuthorizeSecurityGroup"],
            ["CreateVSwitch", "RunInstances"],
            ["AuthorizeSecurityGroup", "RunInstances"],
            ["CreateVSwitch", "CreateLoadBalancer"],
            ["RunInstances", "AddBackendServers"],
            ["CreateLoadBalancer", "AddBackendServers"],
            ["CreateVSwitch", "CreateDBInstance"],
            ["CreateDBInstance", "CreateAccount"],
            ["CreateDBInstance", "ModifySecurityIps"],
            ["RunInstances", "ModifySecurityIps"]
        ]
    }
]

def generate_default_files():
    """Restores default scenario JSON files (Warning: Overwrites existing files)"""
    print("Warning: Regenerating default scenarios will overwrite existing files.")
    for s in scenarios_config:
        # Create a copy to avoid modifying the template
        data = s.copy()
        filename = data.pop("filename")
        filepath = os.path.join(current_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated: {filepath}")

def generate_review_report():
    """Scans current directory for JSON scenarios and generates a review markdown"""
    scenarios_dir = current_dir
    report_path = os.path.join(scenarios_dir, "scenarios_review.md")
    
    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(scenarios_dir, "*.json")))
    
    print(f"Scanning directory: {scenarios_dir}")
    print(f"Found {len(json_files)} JSON files.")
    
    with open(report_path, 'w') as report:
        report.write("# Generated Poset Scenarios for Review\n\n")
        report.write("This document visualizes the Partial Order Sets (Posets) defined in the JSON files in this directory.\n")
        report.write("Modify the JSON files directly and run `python generate_scenarios.py` to update this report.\n\n")
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            
            # Skip non-scenario files (like package metadata if any, or temporary files)
            if filename.startswith("_") or filename == "package.json":
                continue
                
            print(f"Processing {filename}...")
            
            try:
                # Load using the manual config method
                graph = PosetGraph.load_from_manual_config(json_file)
                
                # Extract metadata for display
                with open(json_file, 'r') as f:
                    meta = json.load(f)
                    
                name = meta.get('name', filename)
                description = meta.get('description', 'No description provided.')
                
                report.write(f"## {name}\n")
                report.write(f"**File:** `{filename}`\n\n")
                report.write(f"**Description:** {description}\n\n")
                
                report.write("### Mermaid Diagram\n")
                report.write("```mermaid\n")
                report.write(graph.export_mermaid(title=name))
                report.write("\n```\n\n")
                
                report.write("### Execution Layers\n")
                report.write("```text\n")
                report.write(graph.export_layers_text())
                report.write("\n```\n\n")
                report.write("---\n\n")
                
            except Exception as e:
                report.write(f"## Error processing {filename}\n")
                report.write(f"**Error:** {str(e)}\n\n")
                report.write("---\n\n")
                print(f"Error processing {filename}: {e}")

    print(f"\nReview report generated at: {report_path}")

if __name__ == "__main__":
    # Check if arguments provided
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        generate_default_files()
    else:
        # Default behavior: Generate review from existing files
        # This preserves manual edits.
        generate_review_report()
        print("\nNote: To reset JSON files to defaults, run with '--reset' argument.")
