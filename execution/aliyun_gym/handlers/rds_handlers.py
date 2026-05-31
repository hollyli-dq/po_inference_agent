import uuid
import time
import random

def handle_CreateDBInstance(query, state, chaos, clock):
    vpc_id = query.get('VpcId')
    vswitch_id = query.get('VSwitchId')
    
    if vpc_id and not state.exists(vpc_id):
        return {"Code": "InvalidVpcId.NotFound", "Message": "VPC not found"}
    if vswitch_id and not state.exists(vswitch_id):
        return {"Code": "InvalidVSwitchId.NotFound", "Message": "VSwitch not found"}

    if chaos.should_fail():
        return chaos.generate_error("CreateDBInstance")

    db_id = f"rm-{uuid.uuid4().hex[:8]}"
    
    state.put(db_id, {
        "type": "RDS",
        "status": "Creating", # Initial status
        "vpc_id": vpc_id,
        "vswitch_id": vswitch_id,
        "engine": query.get('Engine'),
        "engine_version": query.get('EngineVersion'),
        "created_at": clock.now(),
        "boot_time": 0, # Simulation: RDS immediately available (was 900s)
        "connection_string": None,
        "accounts": []
    })
    
    clock.tick(1.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "DBInstanceId": db_id,
        "OrderId": str(uuid.uuid4()),
        "ConnectionString": "", # Not available yet
        "Port": "3306"
    }

def handle_CreateAccount(query, state, chaos, clock):
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "RDS instance not found"}
        
    db_data = state.get(db_id)
    
    # 增强验证：检查资源类型
    if db_data.get('type') != 'RDS':
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not an RDS instance"}
    
    # 增强验证：RDS 必须处于 Running 状态才能创建账户
    # 检查时间推进是否使 RDS 进入 Running 状态
    current_time = clock.now()
    if db_data['status'] == "Creating":
        elapsed = current_time - db_data.get('created_at', 0)
        if elapsed >= db_data.get('boot_time', 900):
            db_data['status'] = "Running"
    
    if db_data['status'] != "Running":
        return {
            "Code": "IncorrectDBInstanceState",
            "Message": f"RDS instance status is {db_data['status']}, expected Running. Please wait for RDS creation to complete."
        }
            
    account = {
        "AccountName": query.get('AccountName'),
        "AccountStatus": "Available"
    }
    db_data['accounts'].append(account)
    
    clock.tick(0.5)
    return {"RequestId": str(uuid.uuid4())}

def handle_AllocateInstancePublicConnection(query, state, chaos, clock):
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    
    # Generate connection string
    conn_str = f"{db_id}.mysql.rds.aliyuncs.com"
    db_data['connection_string'] = conn_str
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "ConnectionString": conn_str,
        "DbInstanceName": db_id
    }

def handle_DescribeDBInstanceAttribute(query, state, chaos, clock):
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    data = state.get(db_id)
    current_time = clock.now()
    
    # State Machine
    if data['status'] == "Creating":
        elapsed = current_time - data['created_at']
        if elapsed >= data['boot_time']:
            data['status'] = "Running"
            
    # Construct response structure (simplified)
    # SDK expects Items -> DBInstanceAttribute -> list
    attr = {
        "DBInstanceId": db_id,
        "DBInstanceStatus": data['status'],
        "Engine": data.get('engine'),
        "EngineVersion": data.get('engine_version'),
        "ConnectionString": data.get('connection_string', ''),
        "Port": "3306",
        "VpcId": data.get('vpc_id'),
        "VSwitchId": data.get('vswitch_id')
    }
    
    # 每次状态查询推进 60 秒虚拟时间，模拟真实轮询间隔
    # RDS 创建需要 300 秒，这样约 5 次轮询即可等到 Running 状态
    clock.tick(60.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Items": {"DBInstanceAttribute": [attr]}
    }


def handle_DescribeAccounts(query, state, chaos, clock):
    """Describe RDS instance accounts."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    accounts = db_data.get('accounts', [])
    
    account_list = []
    for acc in accounts:
        account_list.append({
            "AccountName": acc.get('AccountName'),
            "AccountStatus": acc.get('AccountStatus', 'Available'),
            "AccountType": "Normal",
            "DBInstanceId": db_id
        })
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Accounts": {"DBInstanceAccount": account_list}
    }


def handle_DescribeAvailableZones(query, state, chaos, clock):
    """Describe available zones for RDS."""
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AvailableZones": [
            {
                "ZoneId": "cn-hangzhou-b",
                "RegionId": "cn-hangzhou",
                "NetworkTypes": ["VPC"],
                "SupportedEngines": [
                    {
                        "Engine": "MySQL",
                        "SupportedEngineVersions": [
                            {"Version": "5.7"},
                            {"Version": "8.0"}
                        ]
                    },
                    {
                        "Engine": "PostgreSQL",
                        "SupportedEngineVersions": [
                            {"Version": "13.0"},
                            {"Version": "14.0"}
                        ]
                    }
                ]
            },
            {
                "ZoneId": "cn-hangzhou-h",
                "RegionId": "cn-hangzhou",
                "NetworkTypes": ["VPC"]
            }
        ]
    }


def handle_DescribeDBInstanceIPArrayList(query, state, chaos, clock):
    """Describe RDS instance IP whitelist."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    security_ips = db_data.get('security_ips', ['127.0.0.1'])
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Items": {
            "DBInstanceIPArray": [
                {
                    "DBInstanceIPArrayName": "default",
                    "DBInstanceIPArrayAttribute": "hidden",
                    "SecurityIPList": ",".join(security_ips),
                    "SecurityIPType": "IPv4"
                }
            ]
        }
    }


def handle_DescribeDBInstanceNetInfoForChannel(query, state, chaos, clock):
    """Describe RDS instance network information for channel."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    conn_str = db_data.get('connection_string', '')
    
    clock.tick(0.1)
    
    net_info_items = []
    if conn_str:
        net_info_items.append({
            "ConnectionString": conn_str,
            "IPAddress": "192.168.1.100",
            "Port": "3306",
            "VPCId": db_data.get('vpc_id', ''),
            "VSwitchId": db_data.get('vswitch_id', ''),
            "IPType": "Private",
            "ConnectionStringType": "Normal"
        })
    
    return {
        "RequestId": str(uuid.uuid4()),
        "DBInstanceNetInfos": {"DBInstanceNetInfo": net_info_items}
    }


def handle_DescribePrice(query, state, chaos, clock):
    """Describe RDS pricing information."""
    clock.tick(0.1)
    
    db_instance_class = query.get('DBInstanceClass', 'rds.mysql.s1.small')
    db_instance_storage = int(query.get('DBInstanceStorage', 20))
    
    # Mock pricing
    class_prices = {
        "rds.mysql.s1.small": 0.2,
        "rds.mysql.s2.large": 0.4,
        "rds.mysql.m1.medium": 0.6
    }
    
    base_price = class_prices.get(db_instance_class, 0.2)
    storage_price = db_instance_storage * 0.001  # 0.001 CNY per GB per hour
    original_price = base_price + storage_price
    trade_price = original_price * 0.9
    
    return {
        "RequestId": str(uuid.uuid4()),
        "PriceInfo": {
            "Price": {
                "OriginalPrice": original_price,
                "TradePrice": trade_price,
                "DiscountPrice": original_price - trade_price,
                "Currency": "CNY"
            },
            "Rules": {
                "Rule": [
                    {"RuleId": 1, "Description": "Mock RDS pricing rule"}
                ]
            }
        }
    }


def handle_ListClasses(query, state, chaos, clock):
    """List available RDS instance classes."""
    clock.tick(0.1)
    
    engine = query.get('Engine', 'MySQL')
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Items": [
            {
                "DBInstanceClass": "rds.mysql.s1.small",
                "StorageRange": "5-2000",
                "Cpu": "1",
                "MemoryClass": "1024",
                "MaxConnections": "300",
                "MaxIops": "600"
            },
            {
                "DBInstanceClass": "rds.mysql.s2.large",
                "StorageRange": "5-2000",
                "Cpu": "2",
                "MemoryClass": "4096",
                "MaxConnections": "600",
                "MaxIops": "1200"
            },
            {
                "DBInstanceClass": "rds.mysql.m1.medium",
                "StorageRange": "10-3000",
                "Cpu": "4",
                "MemoryClass": "8192",
                "MaxConnections": "1200",
                "MaxIops": "2400"
            }
        ],
        "RegionId": "cn-hangzhou"
    }


def handle_ModifySecurityIps(query, state, chaos, clock):
    """Modify RDS instance IP whitelist."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "RDS instance not found"}
    
    db_data = state.get(db_id)
    
    # 增强验证：检查资源类型
    if db_data.get('type') != 'RDS':
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not an RDS instance"}
    
    # 增强验证：RDS 必须处于 Running 状态才能修改白名单
    current_time = clock.now()
    if db_data['status'] == "Creating":
        elapsed = current_time - db_data.get('created_at', 0)
        if elapsed >= db_data.get('boot_time', 900):
            db_data['status'] = "Running"
    
    if db_data['status'] not in ["Running", "Active"]:
        return {
            "Code": "IncorrectDBInstanceState",
            "Message": f"RDS instance status is {db_data['status']}, expected Running. Cannot modify security IPs."
        }
        
    security_ips = query.get('SecurityIps', '127.0.0.1')
    db_data['security_ips'] = security_ips.split(',')
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_DescribeDBInstances(query, state, chaos, clock):
    """Describe RDS instances list."""
    instances = state.list_by_type("RDS")
    inst_list = []
    
    target_id = query.get('DBInstanceId')
    engine = query.get('Engine')
    db_instance_status = query.get('DBInstanceStatus')
    
    current_time = clock.now()
    
    for db_id, data in instances.items():
        if target_id and db_id != target_id:
            continue
        if engine and data.get('engine') != engine:
            continue
            
        # State Machine: Creating -> Running
        if data['status'] == "Creating":
            elapsed = current_time - data['created_at']
            if elapsed >= data['boot_time']:
                data['status'] = "Running"
        
        if db_instance_status and data['status'] != db_instance_status:
            continue
            
        inst_list.append({
            "DBInstanceId": db_id,
            "DBInstanceDescription": data.get('description', ''),
            "DBInstanceStatus": data['status'],
            "Engine": data.get('engine'),
            "EngineVersion": data.get('engine_version'),
            "DBInstanceNetType": "Intranet",
            "ConnectionMode": "Standard",
            "VpcId": data.get('vpc_id', ''),
            "VSwitchId": data.get('vswitch_id', ''),
            "RegionId": "cn-hangzhou",
            "CreateTime": "2023-01-01T00:00:00Z"
        })
        
    # 每次状态查询推进 60 秒虚拟时间，模拟真实轮询间隔
    clock.tick(60.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalRecordCount": len(inst_list),
        "PageRecordCount": len(inst_list),
        "PageNumber": 1,
        "Items": {"DBInstance": inst_list}
    }


def handle_StartDBInstance(query, state, chaos, clock):
    """Start a stopped RDS instance."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    
    if db_data['status'] not in ["Stopped", "Checking"]:
        return {
            "Code": "IncorrectDBInstanceState",
            "Message": f"Current status {db_data['status']} does not support this operation."
        }
    
    if chaos.should_fail():
        return chaos.generate_error("StartDBInstance")
    
    db_data['status'] = "Starting"
    # Schedule transition to Running after some time
    db_data['start_time'] = clock.now()
    db_data['boot_time'] = 120  # 2 minutes to start
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TaskId": str(uuid.uuid4())
    }


def handle_StopDBInstance(query, state, chaos, clock):
    """Stop a running RDS instance."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    
    if db_data['status'] != "Running":
        return {
            "Code": "IncorrectDBInstanceState",
            "Message": f"Current status {db_data['status']} does not support this operation."
        }
    
    if chaos.should_fail():
        return chaos.generate_error("StopDBInstance")
    
    db_data['status'] = "Stopping"
    # For simplicity, immediately transition to Stopped
    db_data['status'] = "Stopped"
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TaskId": str(uuid.uuid4())
    }


def handle_RestartDBInstance(query, state, chaos, clock):
    """Restart a running RDS instance."""
    db_id = query.get('DBInstanceId')
    if not db_id or not state.exists(db_id):
        return {"Code": "InvalidDBInstanceId.NotFound", "Message": "Not found"}
        
    db_data = state.get(db_id)
    
    if db_data['status'] not in ["Running", "Stopped"]:
        return {
            "Code": "IncorrectDBInstanceState",
            "Message": f"Current status {db_data['status']} does not support this operation."
        }
    
    if chaos.should_fail():
        return chaos.generate_error("RestartDBInstance")
    
    # Simulate restart: temporarily mark as restarting then back to Running
    db_data['status'] = "Restarting"
    db_data['restart_time'] = clock.now()
    db_data['restart_duration'] = 60  # 1 minute restart
    
    # For simplicity in simulation, immediately set to Running after tick
    clock.tick(1.0)
    db_data['status'] = "Running"
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TaskId": str(uuid.uuid4())
    }
