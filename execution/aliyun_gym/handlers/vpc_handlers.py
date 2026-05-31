import uuid
import time
import random

def handle_CreateVpc(query, state, chaos, clock):
    if chaos.should_fail():
        return chaos.generate_error("CreateVpc")

    vpc_id = f"vpc-{uuid.uuid4().hex[:8]}"
    vrouter_id = f"vrt-{uuid.uuid4().hex[:8]}"
    route_table_id = f"vtb-{uuid.uuid4().hex[:8]}"
    
    state.put(vpc_id, {
        "type": "VPC",
        "status": "Available",
        "region_id": query.get('RegionId', 'cn-hangzhou'),
        "cidr_block": query.get('CidrBlock'),
        "vpc_name": query.get('VpcName'),
        "created_at": clock.now(),
        "vrouter_id": vrouter_id,
        "route_table_id": route_table_id
    })
    
    clock.tick(0.5) # Simulate API latency
    
    return {
        "RequestId": str(uuid.uuid4()),
        "VpcId": vpc_id,
        "VRouterId": vrouter_id,
        "RouteTableId": route_table_id,
        "ResourceGroupId": query.get('ResourceGroupId', '')
    }

def handle_CreateVSwitch(query, state, chaos, clock):
    vpc_id = query.get('VpcId')
    if not vpc_id or not state.exists(vpc_id):
        return {
            "Code": "InvalidVpcId.NotFound",
            "Message": "The specified VPC does not exist."
        }

    if chaos.should_fail():
        return chaos.generate_error("CreateVSwitch")

    vswitch_id = f"vsw-{uuid.uuid4().hex[:8]}"
    
    state.put(vswitch_id, {
        "type": "VSwitch",
        "status": "Available",
        "vpc_id": vpc_id,
        "zone_id": query.get('ZoneId'),
        "cidr_block": query.get('CidrBlock'),
        "vswitch_name": query.get('VSwitchName'),
        "created_at": clock.now()
    })
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "VSwitchId": vswitch_id
    }

def handle_DescribeVpcs(query, state, chaos, clock):
    vpcs = state.list_by_type("VPC")
    vpc_list = []
    
    target_vpc_id = query.get('VpcId')
    
    for vid, data in vpcs.items():
        if target_vpc_id and vid != target_vpc_id:
            continue
            
        vpc_list.append({
            "VpcId": vid,
            "Status": data['status'],
            "VpcName": data.get('vpc_name', ''),
            "CidrBlock": data.get('cidr_block', ''),
            "RegionId": data.get('region_id', ''),
            "VRouterId": data.get('vrouter_id', ''),
            "RouteTableIds": {"RouteTableId": [data.get('route_table_id', '')]}
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(vpc_list),
        "Vpcs": {"Vpc": vpc_list}
    }

def handle_DescribeVSwitches(query, state, chaos, clock):
    vswitches = state.list_by_type("VSwitch")
    vsw_list = []
    
    target_vpc_id = query.get('VpcId')
    target_vsw_id = query.get('VSwitchId')
    
    for vid, data in vswitches.items():
        if target_vpc_id and data.get('vpc_id') != target_vpc_id:
            continue
        if target_vsw_id and vid != target_vsw_id:
            continue
            
        vsw_list.append({
            "VSwitchId": vid,
            "Status": data['status'],
            "VSwitchName": data.get('vswitch_name', ''),
            "CidrBlock": data.get('cidr_block', ''),
            "ZoneId": data.get('zone_id', ''),
            "VpcId": data.get('vpc_id', '')
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(vsw_list),
        "VSwitches": {"VSwitch": vsw_list}
    }

def handle_DescribeZones(query, state, chaos, clock):
    clock.tick(0.1)
    return {
        "RequestId": str(uuid.uuid4()),
        "Zones": {
            "Zone": [
                {"ZoneId": "cn-hangzhou-b", "LocalName": "Zone B"},
                {"ZoneId": "cn-hangzhou-h", "LocalName": "Zone H"},
                {"ZoneId": "cn-hangzhou-i", "LocalName": "Zone I"},
                {"ZoneId": "cn-hangzhou-j", "LocalName": "Zone J"}
            ]
        }
    }


# ============================================================
# EIP 相关 API (属于 VPC 产品: Vpc/2016-04-28)
# ============================================================

def handle_AllocateEipAddress(query, state, chaos, clock):
    """申请弹性公网IP (EIP)。
    
    产品归属: VPC (专有网络)
    API文档: https://next.api.aliyun.com/document/Vpc/2016-04-28/AllocateEipAddress
    """
    if chaos.should_fail():
        return chaos.generate_error("AllocateEipAddress")

    eip_id = f"eip-{uuid.uuid4().hex[:8]}"
    # 生成模拟公网IP
    ip_address = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    
    state.put(eip_id, {
        "type": "EIP",
        "status": "Available",
        "ip_address": ip_address,
        "bandwidth": query.get('Bandwidth', '5'),
        "internet_charge_type": query.get('InternetChargeType', 'PayByTraffic'),
        "region_id": query.get('RegionId', 'cn-hangzhou'),
        "created_at": clock.now(),
        "instance_id": None,
        "instance_type": None
    })
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AllocationId": eip_id,
        "EipAddress": ip_address
    }


def handle_AssociateEipAddress(query, state, chaos, clock):
    """将弹性公网IP绑定到云产品实例上。
    
    产品归属: VPC (专有网络)
    API文档: https://next.api.aliyun.com/document/Vpc/2016-04-28/AssociateEipAddress
    
    增强验证：检查目标实例状态（业务前置条件）
    """
    eip_id = query.get('AllocationId')
    instance_id = query.get('InstanceId')
    instance_type = query.get('InstanceType', 'EcsInstance')  # EcsInstance, SlbInstance, Nat, etc.
    
    if not eip_id or not state.exists(eip_id):
        return {
            "Code": "InvalidAllocationId.NotFound",
            "Message": "The specified EIP does not exist."
        }
    
    if not instance_id or not state.exists(instance_id):
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": "The specified instance does not exist."
        }
        
    eip_data = state.get(eip_id)
    
    if eip_data['status'] != 'Available':
        return {
            "Code": "IncorrectEipStatus",
            "Message": f"EIP status is {eip_data['status']}, cannot bind new instance."
        }
    
    if eip_data['instance_id']:
        return {
            "Code": "IncorrectEipStatus",
            "Message": "EIP is already bound to another instance."
        }
    
    # 增强验证：检查目标实例状态（业务前置条件）
    target_data = state.get(instance_id)
    if target_data:
        target_type = target_data.get('type')
        target_status = target_data.get('status')
        
        # 如果绑定到 SLB，检查 SLB 状态
        if target_type == 'LoadBalancer':
            if target_status != 'active':
                return {
                    "Code": "IncorrectLoadBalancerStatus",
                    "Message": f"Load balancer status is {target_status}, expected active. Please wait for SLB creation to complete."
                }
        # 如果绑定到 ECS 实例，检查实例状态
        elif target_type == 'Instance':
            if target_status not in ['Running', 'Stopped']:
                return {
                    "Code": "IncorrectInstanceStatus",
                    "Message": f"ECS instance status is {target_status}, expected Running or Stopped. Please wait for instance to be ready."
                }
        
    eip_data['instance_id'] = instance_id
    eip_data['instance_type'] = instance_type
    eip_data['status'] = "InUse"
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4())
    }
