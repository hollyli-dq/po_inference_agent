import uuid
import time
import random

def handle_CreateInstance(query, state, chaos, clock):
    # Redis CreateInstance
    vpc_id = query.get('VpcId')
    vswitch_id = query.get('VSwitchId')
    
    if vpc_id and not state.exists(vpc_id):
        return {"Code": "InvalidVpcId.NotFound", "Message": "VPC not found"}
    if vswitch_id and not state.exists(vswitch_id):
        return {"Code": "InvalidVSwitchId.NotFound", "Message": "VSwitch not found"}

    if chaos.should_fail():
        return chaos.generate_error("CreateInstance")

    redis_id = f"r-kvstore-{uuid.uuid4().hex[:8]}"
    
    state.put(redis_id, {
        "type": "Redis",
        "status": "Creating",
        "vpc_id": vpc_id,
        "vswitch_id": vswitch_id,
        "instance_name": query.get('InstanceName'),
        "created_at": clock.now(),
        "boot_time": 300, # 5 minutes
        "connection_string": None
    })
    
    clock.tick(1.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "InstanceId": redis_id,
        "InstanceName": query.get('InstanceName'),
        "ConnectionDomain": "", # Not ready
        "Port": 6379
    }

def handle_AllocateInstancePublicConnection(query, state, chaos, clock):
    redis_id = query.get('InstanceId')
    if not redis_id or not state.exists(redis_id):
        return {"Code": "InvalidInstanceId.NotFound", "Message": "Not found"}
        
    data = state.get(redis_id)
    conn_str = f"{redis_id}.redis.rds.aliyuncs.com"
    data['connection_string'] = conn_str
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "ConnectionString": conn_str
    }

def handle_DescribeInstances(query, state, chaos, clock):
    instances = state.list_by_type("Redis")
    inst_list = []
    
    target_id = query.get('InstanceIds') # Simplified check
    
    current_time = clock.now()
    
    for rid, data in instances.items():
        if target_id and rid not in target_id:
            continue
            
        if data['status'] == "Creating":
            if current_time - data['created_at'] >= data['boot_time']:
                data['status'] = "Normal"
                
        inst_list.append({
            "InstanceId": rid,
            "InstanceName": data.get('instance_name', ''),
            "InstanceStatus": data['status'],
            "ConnectionDomain": data.get('connection_string', ''),
            "Port": 6379,
            "VpcId": data.get('vpc_id'),
            "VSwitchId": data.get('vswitch_id')
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(inst_list),
        "Instances": {"KVStoreInstance": inst_list}
    }


def handle_DescribeAvailableResource(query, state, chaos, clock):
    """Describe available Redis resources in zones."""
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AvailableZones": {
            "AvailableZone": [
                {
                    "ZoneId": "cn-hangzhou-b",
                    "RegionId": "cn-hangzhou",
                    "SupportedEngines": {
                        "SupportedEngine": [
                            {
                                "Engine": "Redis",
                                "SupportedEditionTypes": {
                                    "SupportedEditionType": [
                                        {
                                            "EditionType": "Community",
                                            "SupportedSeriesTypes": {
                                                "SupportedSeriesType": [
                                                    {"SeriesType": "enhanced_performance_type"}
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "ZoneId": "cn-hangzhou-h",
                    "RegionId": "cn-hangzhou"
                }
            ]
        }
    }


def handle_DescribeDBInstanceNetInfo(query, state, chaos, clock):
    """Describe Redis instance network information."""
    redis_id = query.get('InstanceId')
    if not redis_id or not state.exists(redis_id):
        return {"Code": "InvalidInstanceId.NotFound", "Message": "Not found"}
        
    data = state.get(redis_id)
    conn_str = data.get('connection_string', '')
    
    clock.tick(0.1)
    
    net_info_items = []
    if conn_str:
        net_info_items.append({
            "ConnectionString": conn_str,
            "IPAddress": "192.168.1.200",
            "Port": "6379",
            "VPCId": data.get('vpc_id', ''),
            "VSwitchId": data.get('vswitch_id', ''),
            "IPType": "Private",
            "DBInstanceNetType": "0"
        })
    
    return {
        "RequestId": str(uuid.uuid4()),
        "NetInfoItems": {"InstanceNetInfo": net_info_items}
    }


def handle_DescribePrice(query, state, chaos, clock):
    """Describe Redis pricing information."""
    clock.tick(0.1)
    
    instance_class = query.get('InstanceClass', 'redis.master.small.default')
    
    # Mock pricing
    class_prices = {
        "redis.master.small.default": 0.1,
        "redis.master.mid.default": 0.2,
        "redis.master.large.default": 0.4
    }
    
    original_price = class_prices.get(instance_class, 0.1)
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
                    {"RuleId": 1, "Description": "Mock Redis pricing rule"}
                ]
            }
        }
    }


def handle_DescribeSecurityIps(query, state, chaos, clock):
    """Describe Redis instance IP whitelist."""
    redis_id = query.get('InstanceId')
    if not redis_id or not state.exists(redis_id):
        return {"Code": "InvalidInstanceId.NotFound", "Message": "Not found"}
        
    data = state.get(redis_id)
    security_ips = data.get('security_ips', ['127.0.0.1'])
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "SecurityIpGroups": {
            "SecurityIpGroup": [
                {
                    "SecurityIpGroupName": "default",
                    "SecurityIpGroupAttribute": "hidden",
                    "SecurityIpList": ",".join(security_ips)
                }
            ]
        }
    }


def handle_ModifySecurityIps(query, state, chaos, clock):
    """Modify Redis instance IP whitelist."""
    redis_id = query.get('InstanceId')
    if not redis_id or not state.exists(redis_id):
        return {"Code": "InvalidInstanceId.NotFound", "Message": "Not found"}
        
    security_ips = query.get('SecurityIps', '127.0.0.1')
    
    data = state.get(redis_id)
    data['security_ips'] = security_ips.split(',')
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_DescribeInstanceAttribute(query, state, chaos, clock):
    """Describe Redis instance attribute (details)."""
    redis_id = query.get('InstanceId')
    if not redis_id or not state.exists(redis_id):
        return {"Code": "InvalidInstanceId.NotFound", "Message": "Redis instance not found"}
        
    data = state.get(redis_id)
    current_time = clock.now()
    
    # 检查是否已启动完成
    if data['status'] == "Creating":
        if current_time - data['created_at'] >= data['boot_time']:
            data['status'] = "Normal"
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "InstanceId": redis_id,
        "InstanceName": data.get('instance_name', ''),
        "InstanceStatus": data['status'],
        "InstanceType": "Redis",
        "Engine": "Redis",
        "EngineVersion": "7.0",
        "ConnectionDomain": data.get('connection_string', ''),
        "Port": 6379,
        "VpcId": data.get('vpc_id', ''),
        "VSwitchId": data.get('vswitch_id', ''),
        "PrivateIp": "192.168.1.200",
        "InstanceClass": query.get('InstanceClass', 'redis.master.mid.default'),
        "CreateTime": data.get('created_at', 0)
    }
