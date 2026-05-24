import uuid
import time
import random
from execution.aliyun_gym.core.resource_manager import ResourceManager

# Helper to get resource manager
def get_resource_manager():
    return ResourceManager.get_instance()

def handle_CreateSecurityGroup(query, state, chaos, clock):
    vpc_id = query.get('VpcId')
    if vpc_id and not state.exists(vpc_id):
        return {
            "Code": "InvalidVpcId.NotFound",
            "Message": "The specified VPC does not exist."
        }

    if chaos.should_fail():
        return chaos.generate_error("CreateSecurityGroup")

    sg_id = f"sg-{uuid.uuid4().hex[:8]}"
    
    state.put(sg_id, {
        "type": "SecurityGroup",
        "status": "Available",
        "vpc_id": vpc_id,
        "sg_name": query.get('SecurityGroupName'),
        "description": query.get('Description'),
        "created_at": clock.now(),
        "rules": []
    })
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "SecurityGroupId": sg_id
    }

def handle_AuthorizeSecurityGroup(query, state, chaos, clock):
    sg_id = query.get('SecurityGroupId')
    if not sg_id or not state.exists(sg_id):
        return {
            "Code": "InvalidSecurityGroupId.NotFound",
            "Message": "The specified SecurityGroup does not exist."
        }
        
    sg_data = state.get(sg_id)
    rule = {
        "IpProtocol": query.get('IpProtocol'),
        "PortRange": query.get('PortRange'),
        "SourceCidrIp": query.get('SourceCidrIp'),
        "Policy": query.get('Policy', 'Accept'),
        "Priority": query.get('Priority', 1)
    }
    sg_data['rules'].append(rule)
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4())
    }

def handle_DescribeSecurityGroups(query, state, chaos, clock):
    sgs = state.list_by_type("SecurityGroup")
    sg_list = []
    
    target_vpc_id = query.get('VpcId')
    target_sg_id = query.get('SecurityGroupId')
    
    for sgid, data in sgs.items():
        if target_vpc_id and data.get('vpc_id') != target_vpc_id:
            continue
        if target_sg_id and sgid != target_sg_id:
            continue
            
        sg_list.append({
            "SecurityGroupId": sgid,
            "SecurityGroupName": data.get('sg_name', ''),
            "Description": data.get('description', ''),
            "VpcId": data.get('vpc_id', ''),
            # Use a fixed base time for formatting, or just use ISO string
            "CreationTime": "2023-01-01T00:00:00Z" 
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(sg_list),
        "SecurityGroups": {"SecurityGroup": sg_list}
    }

def handle_RunInstances(query, state, chaos, clock):
    # Check dependencies
    vswitch_id = query.get('VSwitchId')
    sg_id = query.get('SecurityGroupId')
    
    if vswitch_id and not state.exists(vswitch_id):
        return {
            "Code": "InvalidVSwitchId.NotFound",
            "Message": "The specified VSwitch does not exist."
        }
        
    if sg_id and not state.exists(sg_id):
        return {
            "Code": "InvalidSecurityGroupId.NotFound",
            "Message": "The specified SecurityGroup does not exist."
        }

    # 增强验证：检查安全组是否已配置规则（业务前置条件）
    if sg_id:
        sg_data = state.get(sg_id)
        if sg_data and sg_data.get('type') == 'SecurityGroup':
            rules = sg_data.get('rules', [])
            if not rules:
                return {
                    "Code": "SecurityGroupRulesNotConfigured",
                    "Message": "Security group has no ingress/egress rules configured. Please authorize security group first."
                }

    if chaos.should_fail():
        return chaos.generate_error("RunInstances")

    # Create instances
    amount = int(query.get('Amount', 1))
    instance_ids = []
    
    for _ in range(amount):
        instance_id = f"i-{uuid.uuid4().hex[:8]}"
        instance_ids.append(instance_id)
        
        state.put(instance_id, {
            "type": "ECS",
            "status": "Pending", # Initial status
            "vswitch_id": vswitch_id,
            "sg_id": sg_id,
            "instance_type": query.get('InstanceType'),
            "image_id": query.get('ImageId'),
            "instance_name": query.get('InstanceName'),
            "created_at": clock.now(),
            "boot_time": 300 # Simulate 5 minutes boot time (virtual time)
        })
        
    clock.tick(1.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "InstanceIdSets": {"InstanceIdSet": instance_ids}
    }

def handle_DescribeInstances(query, state, chaos, clock):
    instances = state.list_by_type("ECS")
    inst_list = []
    
    # Handle InstanceIds filtering
    # SDK usually sends JSON array string for InstanceIds
    target_ids = query.get('InstanceIds')
    if isinstance(target_ids, str) and target_ids.startswith('['):
        import json
        try:
            target_ids = json.loads(target_ids)
        except:
            target_ids = None
            
    current_time = clock.now()
    
    for iid, data in instances.items():
        if target_ids and iid not in target_ids:
            continue
            
        # State Machine Logic
        if data['status'] == "Pending":
            elapsed = current_time - data['created_at']
            if elapsed >= data['boot_time']:
                data['status'] = "Running"
        
        inst_list.append({
            "InstanceId": iid,
            "InstanceName": data.get('instance_name', ''),
            "Status": data['status'],
            "VpcAttributes": {"VSwitchId": data.get('vswitch_id', '')},
            "SecurityGroupIds": {"SecurityGroupId": [data.get('sg_id', '')]},
            "InstanceType": data.get('instance_type', ''),
            "ImageId": data.get('image_id', ''),
            "RegionId": "cn-hangzhou",
            "ZoneId": "cn-hangzhou-b"
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(inst_list),
        "Instances": {"Instance": inst_list}
    }


def handle_DescribeSecurityGroupAttribute(query, state, chaos, clock):
    """Describe security group rules/attributes."""
    sg_id = query.get('SecurityGroupId')
    if not sg_id or not state.exists(sg_id):
        return {
            "Code": "InvalidSecurityGroupId.NotFound",
            "Message": "The specified SecurityGroup does not exist."
        }
    
    sg_data = state.get(sg_id)
    permissions = []
    
    for rule in sg_data.get('rules', []):
        permissions.append({
            "IpProtocol": rule.get('IpProtocol', 'tcp'),
            "PortRange": rule.get('PortRange', ''),
            "SourceCidrIp": rule.get('SourceCidrIp', ''),
            "Policy": rule.get('Policy', 'Accept'),
            "Priority": rule.get('Priority', 1),
            "Direction": "ingress"
        })
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "SecurityGroupId": sg_id,
        "SecurityGroupName": sg_data.get('sg_name', ''),
        "Description": sg_data.get('description', ''),
        "VpcId": sg_data.get('vpc_id', ''),
        "Permissions": {"Permission": permissions}
    }


def handle_DescribeAvailableResource(query, state, chaos, clock):
    """Describe available ECS resources in zones."""
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AvailableZones": {
            "AvailableZone": [
                {
                    "ZoneId": "cn-hangzhou-b",
                    "Status": "Available",
                    "StatusCategory": "WithStock",
                    "AvailableResources": {
                        "AvailableResource": [
                            {
                                "Type": "InstanceType",
                                "SupportedResources": {
                                    "SupportedResource": [
                                        {"Value": "ecs.g6.large", "Status": "Available"},
                                        {"Value": "ecs.g6.xlarge", "Status": "Available"},
                                        {"Value": "ecs.c6.large", "Status": "Available"}
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "ZoneId": "cn-hangzhou-h",
                    "Status": "Available",
                    "StatusCategory": "WithStock"
                }
            ]
        }
    }


def handle_DescribeImages(query, state, chaos, clock):
    """Describe available ECS images."""
    clock.tick(0.1)
    
    image_id = query.get('ImageId')
    image_name = query.get('ImageName')
    
    rm = get_resource_manager()
    images = rm.get_images(image_id=image_id, image_name=image_name)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(images),
        "Images": {"Image": images}
    }


def handle_DescribeInstanceTypes(query, state, chaos, clock):
    """Describe available ECS instance types."""
    clock.tick(0.1)
    
    family = query.get('InstanceTypeFamily')
    
    rm = get_resource_manager()
    instance_types = rm.get_instance_types(family=family)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "InstanceTypes": {"InstanceType": instance_types}
    }


def handle_DescribePrice(query, state, chaos, clock):
    """Describe ECS pricing information."""
    clock.tick(0.1)
    
    instance_type = query.get('InstanceType', 'ecs.g6.large')
    
    # Mock pricing based on instance type
    base_prices = {
        "ecs.g6.large": 0.5,
        "ecs.g6.xlarge": 1.0,
        "ecs.c6.large": 0.4,
        "ecs.c6.xlarge": 0.8
    }
    
    original_price = base_prices.get(instance_type, 0.5)
    trade_price = original_price * 0.9  # 10% discount
    
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
                    {"RuleId": 1, "Description": "Mock pricing rule"}
                ]
            }
        }
    }


def handle_DescribeZones(query, state, chaos, clock):
    """Describe available zones for ECS."""
    clock.tick(0.1)
    
    region_id = query.get('RegionId')
    
    rm = get_resource_manager()
    zones = rm.get_zones(region_id=region_id)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Zones": {
            "Zone": zones
        }
    }


def handle_DescribeRegions(query, state, chaos, clock):
    """Describe available regions for ECS."""
    clock.tick(0.1)
    
    rm = get_resource_manager()
    regions = rm.get_regions()
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Regions": {
            "Region": regions
        }
    }


def handle_DescribeAccountAttributes(query, state, chaos, clock):
    """Describe account attributes and quotas for ECS."""
    clock.tick(0.1)
    
    attribute_names = query.get('AttributeName')  # Could be list or string
    
    # Default attributes
    attributes = [
        {
            "AttributeName": "max-security-groups",
            "AttributeValues": {
                "ValueItem": [{"Value": "100", "Count": 100}]
            }
        },
        {
            "AttributeName": "max-elastic-network-interfaces",
            "AttributeValues": {
                "ValueItem": [{"Value": "100", "Count": 100}]
            }
        },
        {
            "AttributeName": "instance-type-quota",
            "AttributeValues": {
                "ValueItem": [{"Value": "100", "Count": 100}]
            }
        },
        {
            "AttributeName": "max-instances",
            "AttributeValues": {
                "ValueItem": [{"Value": "100", "Count": 100}]
            }
        },
        {
            "AttributeName": "used-instances",
            "AttributeValues": {
                "ValueItem": [{"Value": "5", "Count": 5}]
            }
        }
    ]
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AccountAttributeItems": {
            "AccountAttributeItem": attributes
        }
    }


def handle_DeleteInstances(query, state, chaos, clock):
    """Delete one or more ECS instances."""
    instance_ids = query.get('InstanceId')  # Can be list or comma-separated
    
    if not instance_ids:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter InstanceId is not supplied."
        }
    
    # Parse instance IDs
    if isinstance(instance_ids, str):
        if instance_ids.startswith('['):
            import json
            try:
                instance_ids = json.loads(instance_ids)
            except:
                instance_ids = [instance_ids]
        else:
            instance_ids = [instance_ids]
    
    if chaos.should_fail():
        return chaos.generate_error("DeleteInstances")
    
    deleted_ids = []
    for iid in instance_ids:
        if state.exists(iid):
            inst_data = state.get(iid)
            if inst_data.get('type') == 'ECS':
                # Check if instance can be deleted (usually needs to be stopped)
                if inst_data.get('status') not in ['Stopped', 'Running', 'Pending']:
                    return {
                        "Code": "IncorrectInstanceStatus",
                        "Message": f"Instance {iid} status does not allow deletion."
                    }
                state.delete(iid)
                deleted_ids.append(iid)
        else:
            return {
                "Code": "InvalidInstanceId.NotFound",
                "Message": f"Instance {iid} does not exist."
            }
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_StartInstance(query, state, chaos, clock):
    """Start a stopped ECS instance."""
    instance_id = query.get('InstanceId')
    
    if not instance_id:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter InstanceId is not supplied."
        }
    
    if not state.exists(instance_id):
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} does not exist."
        }
    
    inst_data = state.get(instance_id)
    
    if inst_data.get('type') != 'ECS':
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} is not an ECS instance."
        }
    
    if inst_data['status'] != 'Stopped':
        return {
            "Code": "IncorrectInstanceStatus",
            "Message": f"Instance status must be Stopped to start. Current: {inst_data['status']}"
        }
    
    if chaos.should_fail():
        return chaos.generate_error("StartInstance")
    
    inst_data['status'] = 'Starting'
    inst_data['start_time'] = clock.now()
    inst_data['boot_time'] = 60  # 1 minute to start
    
    clock.tick(0.5)
    
    # For simulation, immediately transition to Running
    inst_data['status'] = 'Running'
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_StopInstance(query, state, chaos, clock):
    """Stop a running ECS instance."""
    instance_id = query.get('InstanceId')
    force_stop = query.get('ForceStop', 'false').lower() == 'true'
    
    if not instance_id:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter InstanceId is not supplied."
        }
    
    if not state.exists(instance_id):
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} does not exist."
        }
    
    inst_data = state.get(instance_id)
    
    if inst_data.get('type') != 'ECS':
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} is not an ECS instance."
        }
    
    if inst_data['status'] != 'Running':
        return {
            "Code": "IncorrectInstanceStatus",
            "Message": f"Instance status must be Running to stop. Current: {inst_data['status']}"
        }
    
    if chaos.should_fail():
        return chaos.generate_error("StopInstance")
    
    inst_data['status'] = 'Stopping'
    
    clock.tick(0.5)
    
    # For simulation, immediately transition to Stopped
    inst_data['status'] = 'Stopped'
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_RebootInstance(query, state, chaos, clock):
    """Reboot an ECS instance."""
    instance_id = query.get('InstanceId')
    force_stop = query.get('ForceStop', 'false').lower() == 'true'
    
    if not instance_id:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter InstanceId is not supplied."
        }
    
    if not state.exists(instance_id):
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} does not exist."
        }
    
    inst_data = state.get(instance_id)
    
    if inst_data.get('type') != 'ECS':
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} is not an ECS instance."
        }
    
    if inst_data['status'] != 'Running':
        return {
            "Code": "IncorrectInstanceStatus",
            "Message": f"Instance status must be Running to reboot. Current: {inst_data['status']}"
        }
    
    if chaos.should_fail():
        return chaos.generate_error("RebootInstance")
    
    # Simulate reboot: briefly show Stopping/Starting
    inst_data['status'] = 'Stopping'
    clock.tick(0.5)
    inst_data['status'] = 'Starting'
    clock.tick(0.5)
    inst_data['status'] = 'Running'
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_ReplaceSystemDisk(query, state, chaos, clock):
    """Replace the system disk of an ECS instance."""
    instance_id = query.get('InstanceId')
    image_id = query.get('ImageId')
    
    if not instance_id:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter InstanceId is not supplied."
        }
    
    if not image_id:
        return {
            "Code": "MissingParameter",
            "Message": "The required parameter ImageId is not supplied."
        }
    
    if not state.exists(instance_id):
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} does not exist."
        }
    
    inst_data = state.get(instance_id)
    
    if inst_data.get('type') != 'ECS':
        return {
            "Code": "InvalidInstanceId.NotFound",
            "Message": f"Instance {instance_id} is not an ECS instance."
        }
    
    if inst_data['status'] != 'Stopped':
        return {
            "Code": "IncorrectInstanceStatus",
            "Message": f"Instance must be stopped to replace system disk. Current: {inst_data['status']}"
        }
    
    if chaos.should_fail():
        return chaos.generate_error("ReplaceSystemDisk")
    
    # Generate new disk ID
    new_disk_id = f"d-{uuid.uuid4().hex[:8]}"
    
    # Update instance with new image
    inst_data['image_id'] = image_id
    inst_data['system_disk_id'] = new_disk_id
    
    clock.tick(1.0)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "DiskId": new_disk_id
    }
