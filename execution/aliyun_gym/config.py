"""
Configuration for Aliyun-Gym simulator.
Adjust these parameters to control simulation behavior.
"""

# ============================================================
# Chaos Injection Configuration
# ============================================================

CHAOS_CONFIG = {
    # Global failure rate (0.0 = no failures, 1.0 = always fail)
    "global_failure_rate": 0.0,
    
    # Individual error type rates (when failure occurs)
    "error_distribution": {
        "network_timeout": 0.3,      # Sdk.ReadTimeout
        "service_unavailable": 0.2,  # ServiceUnavailable
        "stock_out": 0.25,           # OperationDenied.NoStock
        "throttling": 0.15,          # Throttling.User
        "quota_exceeded": 0.1,       # Forbidden.QuotaExceeded
    }
}

# ============================================================
# Simulated Latency Configuration (in milliseconds)
# ============================================================

LATENCY_CONFIG = {
    # Format: {"api_ms": (min, max), "boot_ms": (min, max)}
    # api_ms: Time for API call to return
    # boot_ms: Time for resource to become Ready/Running
    
    "VPC": {
        "CreateVpc": {"api_ms": (2000, 5000), "boot_ms": (0, 0)},
        "CreateVSwitch": {"api_ms": (1000, 3000), "boot_ms": (0, 0)},
        "DescribeVpcs": {"api_ms": (100, 300), "boot_ms": (0, 0)},
        "DescribeVSwitches": {"api_ms": (100, 300), "boot_ms": (0, 0)},
    },
    
    "ECS": {
        "CreateSecurityGroup": {"api_ms": (500, 1500), "boot_ms": (0, 0)},
        "RunInstances": {"api_ms": (5000, 10000), "boot_ms": (60000, 180000)},  # 1-3 min boot
        "DescribeInstances": {"api_ms": (100, 500), "boot_ms": (0, 0)},
    },
    
    "SLB": {
        "CreateLoadBalancer": {"api_ms": (3000, 8000), "boot_ms": (30000, 60000)},
        "AddBackendServers": {"api_ms": (2000, 5000), "boot_ms": (0, 0)},
    },
    
    "RDS": {
        "CreateDBInstance": {"api_ms": (10000, 30000), "boot_ms": (300000, 600000)},  # 5-10 min
        "CreateAccount": {"api_ms": (2000, 5000), "boot_ms": (0, 0)},
    },
    
    "REDIS": {
        "CreateInstance": {"api_ms": (5000, 15000), "boot_ms": (60000, 120000)},  # 1-2 min
        "DescribeInstances": {"api_ms": (100, 500), "boot_ms": (0, 0)},
    },
    
    "EIP": {
        "AllocateEipAddress": {"api_ms": (1000, 3000), "boot_ms": (0, 0)},
        "AssociateEipAddress": {"api_ms": (2000, 5000), "boot_ms": (0, 0)},
    },
    
    "CMS": {
        "CreateMonitorGroup": {"api_ms": (500, 1500), "boot_ms": (0, 0)},
        "CreateMonitorGroupInstances": {"api_ms": (1000, 3000), "boot_ms": (0, 0)},
    },
    
    "OOS": {
        "StartExecution": {"api_ms": (5000, 15000), "boot_ms": (30000, 120000)},
    }
}

# ============================================================
# Quota Limits
# ============================================================

QUOTA_LIMITS = {
    "VPC": 5,           # Max VPCs per account
    "VSwitch": 20,      # Max VSwitches per VPC
    "ECS": 50,          # Max ECS instances
    "SLB": 10,          # Max SLB instances  
    "RDS": 5,           # Max RDS instances
    "REDIS": 10,        # Max Redis instances
    "EIP": 20,          # Max EIPs
}

# ============================================================
# Default Region and Zones
# ============================================================

DEFAULT_REGION = "cn-hangzhou"

AVAILABLE_ZONES = [
    "cn-hangzhou-b",
    "cn-hangzhou-h", 
    "cn-hangzhou-i",
    "cn-hangzhou-j",
    "cn-hangzhou-k",
]

# ============================================================
# Trace Recording Configuration
# ============================================================

TRACE_CONFIG = {
    "output_dir": "./traces",
    "filename_pattern": "trace_{timestamp}.json",
    "auto_save_interval": 100,  # Save every N traces
}
