import uuid
import time
import random

def handle_CreateLoadBalancer(query, state, chaos, clock):
    vpc_id = query.get('VpcId')
    if vpc_id and not state.exists(vpc_id):
        return {
            "Code": "InvalidVpcId.NotFound",
            "Message": "The specified VPC does not exist."
        }

    if chaos.should_fail():
        return chaos.generate_error("CreateLoadBalancer")

    lb_id = f"lb-{uuid.uuid4().hex[:8]}"
    
    state.put(lb_id, {
        "type": "SLB",
        "status": "Active", # SLB is usually active immediately
        "vpc_id": vpc_id,
        "lb_name": query.get('LoadBalancerName'),
        "address_type": query.get('AddressType', 'intranet'),
        "created_at": clock.now(),
        "listeners": [],
        "backend_servers": []
    })
    
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "LoadBalancerId": lb_id,
        "Address": "192.168.1.100", # Mock IP
        "LoadBalancerName": query.get('LoadBalancerName'),
        "VpcId": vpc_id
    }

def handle_CreateLoadBalancerHTTPListener(query, state, chaos, clock):
    lb_id = query.get('LoadBalancerId')
    if not lb_id or not state.exists(lb_id):
        return {
            "Code": "InvalidLoadBalancerId.NotFound",
            "Message": "The specified LoadBalancer does not exist."
        }
        
    lb_data = state.get(lb_id)
    listener = {
        "protocol": "http",
        "port": query.get('ListenerPort'),
        "backend_port": query.get('BackendServerPort'),
        "status": "Stopped" # Initially stopped
    }
    lb_data['listeners'].append(listener)
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4())
    }

def handle_StartLoadBalancerListener(query, state, chaos, clock):
    lb_id = query.get('LoadBalancerId')
    port = query.get('ListenerPort')
    protocol = query.get('ListenerProtocol', 'tcp').lower()
    
    if not lb_id or not state.exists(lb_id):
        return {"Code": "InvalidLoadBalancerId.NotFound", "Message": "LoadBalancer not found"}
    
    lb_data = state.get(lb_id)
    
    # 增强验证：检查 SLB 类型和状态
    if lb_data.get('type') != 'SLB':
        return {"Code": "InvalidLoadBalancerId.NotFound", "Message": "Not a LoadBalancer"}
    if lb_data.get('status') != 'Active':
        return {"Code": "LoadBalancerNotActive", "Message": f"LoadBalancer is {lb_data.get('status')}, expected Active"}
    
    # 查找监听器
    found = False
    for l in lb_data.get('listeners', []):
        if str(l['port']) == str(port):
            # 增强验证：检查监听器协议匹配
            if l.get('protocol', 'tcp') != protocol:
                return {"Code": "ListenerProtocolMismatch", "Message": f"Listener protocol is {l.get('protocol')}, not {protocol}"}
            l['status'] = "Running"
            found = True
            break
            
    if not found:
        return {"Code": "InvalidListenerPort.NotFound", "Message": f"Listener on port {port} not found. Please create listener first."}
        
    clock.tick(0.2)
    return {"RequestId": str(uuid.uuid4())}

def handle_AddBackendServers(query, state, chaos, clock):
    lb_id = query.get('LoadBalancerId')
    backend_servers = query.get('BackendServers') # JSON string
    
    if not lb_id or not state.exists(lb_id):
        return {"Code": "InvalidLoadBalancerId.NotFound", "Message": "The specified LoadBalancer does not exist."}
    
    # 增强验证：检查 SLB 状态
    lb_data = state.get(lb_id)
    if lb_data.get('type') != 'SLB':
        return {"Code": "InvalidLoadBalancerId.NotFound", "Message": "The specified LoadBalancer does not exist."}
    if lb_data.get('status') != 'Active':
        return {"Code": "LoadBalancerNotActive", "Message": f"LoadBalancer status is {lb_data.get('status')}, expected Active."}
        
    if not backend_servers:
        # If parameter is missing, do nothing (as per API behavior usually)
        return {
            "RequestId": str(uuid.uuid4()),
            "LoadBalancerId": lb_id,
            "BackendServers": {"BackendServer": []}
        }

    # Strict type check: BackendServers must be a string
    if not isinstance(backend_servers, str):
        return {
            "Code": "InvalidParameter",
            "Message": "The parameter BackendServers must be a JSON string."
        }

    import json
    try:
        servers = json.loads(backend_servers)
    except json.JSONDecodeError:
        return {
            "Code": "InvalidParameter",
            "Message": "The parameter BackendServers is not valid JSON."
        }
        
    if not isinstance(servers, list):
         return {
            "Code": "InvalidParameter", 
            "Message": "The parameter BackendServers must be a JSON list."
        }
        
    # 增强验证：检查每个后端实例是否存在且状态正确
    for s in servers:
        sid = s.get('ServerId')
        if not state.exists(sid):
            return {"Code": "InvalidServerId.NotFound", "Message": f"The specified instance {sid} does not exist."}
        
        inst_data = state.get(sid)
        if inst_data.get('type') != 'ECS':
            return {"Code": "InvalidServerId.NotFound", "Message": f"The specified instance {sid} is not an ECS instance."}
        
        # 检查实例状态
        current_time = clock.now()
        if inst_data.get('status') == 'Pending':
            elapsed = current_time - inst_data.get('created_at', 0)
            if elapsed >= inst_data.get('boot_time', 300):
                inst_data['status'] = 'Running'
        
        if inst_data.get('status') not in ['Running', 'Pending']:
            return {"Code": "IncorrectInstanceStatus", "Message": f"Instance {sid} status is {inst_data.get('status')}, cannot add to LoadBalancer."}
        
        lb_data['backend_servers'].append(s)
        
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "LoadBalancerId": lb_id,
        "BackendServers": {"BackendServer": servers}
    }

def handle_DescribeLoadBalancers(query, state, chaos, clock):
    lbs = state.list_by_type("SLB")
    lb_list = []
    
    target_id = query.get('LoadBalancerId')
    
    for lbid, data in lbs.items():
        if target_id and lbid != target_id:
            continue
            
        lb_list.append({
            "LoadBalancerId": lbid,
            "LoadBalancerName": data.get('lb_name', ''),
            "LoadBalancerStatus": data['status'],
            "Address": "192.168.1.100",
            "VpcId": data.get('vpc_id', '')
        })
        
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(lb_list),
        "LoadBalancers": {"LoadBalancer": lb_list}
    }


def handle_CreateLoadBalancerTCPListener(query, state, chaos, clock):
    """Create a TCP listener for load balancer."""
    lb_id = query.get('LoadBalancerId')
    if not lb_id or not state.exists(lb_id):
        return {
            "Code": "InvalidLoadBalancerId.NotFound",
            "Message": "The specified LoadBalancer does not exist."
        }
        
    lb_data = state.get(lb_id)
    listener = {
        "protocol": "tcp",
        "port": query.get('ListenerPort'),
        "backend_port": query.get('BackendServerPort'),
        "bandwidth": query.get('Bandwidth', -1),
        "status": "Stopped"
    }
    lb_data['listeners'].append(listener)
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_CreateAccessControlList(query, state, chaos, clock):
    """Create an access control list for SLB."""
    if chaos.should_fail():
        return chaos.generate_error("CreateAccessControlList")

    acl_id = f"acl-{uuid.uuid4().hex[:8]}"
    
    state.put(acl_id, {
        "type": "ACL",
        "acl_name": query.get('AclName'),
        "address_ip_version": query.get('AddressIPVersion', 'ipv4'),
        "created_at": clock.now(),
        "entries": []
    })
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AclId": acl_id
    }


def handle_AddAccessControlListEntry(query, state, chaos, clock):
    """Add entries to access control list."""
    acl_id = query.get('AclId')
    if not acl_id or not state.exists(acl_id):
        return {
            "Code": "InvalidAclId.NotFound",
            "Message": "The specified ACL does not exist."
        }
    
    acl_entries = query.get('AclEntrys')  # JSON string
    
    import json
    try:
        entries = json.loads(acl_entries)
    except:
        entries = []
    
    acl_data = state.get(acl_id)
    for entry in entries:
        acl_data['entries'].append({
            "entry": entry.get('entry'),
            "comment": entry.get('comment', '')
        })
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4())
    }


def handle_DescribeAvailableResource(query, state, chaos, clock):
    """Describe available SLB resources in zones."""
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "AvailableResources": {
            "AvailableResource": [
                {
                    "MasterZoneId": "cn-hangzhou-b",
                    "SlaveZoneId": "cn-hangzhou-h",
                    "SupportedLoadBalancerSpec": {
                        "LoadBalancerSpec": [
                            "slb.s1.small",
                            "slb.s2.small",
                            "slb.s2.medium",
                            "slb.s3.small"
                        ]
                    }
                },
                {
                    "MasterZoneId": "cn-hangzhou-h",
                    "SlaveZoneId": "cn-hangzhou-b"
                }
            ]
        }
    }


def handle_DescribeLoadBalancerAttribute(query, state, chaos, clock):
    """Describe load balancer attributes in detail."""
    lb_id = query.get('LoadBalancerId')
    if not lb_id or not state.exists(lb_id):
        return {
            "Code": "InvalidLoadBalancerId.NotFound",
            "Message": "The specified LoadBalancer does not exist."
        }
    
    data = state.get(lb_id)
    
    # Build listener ports info
    listener_ports = [l.get('port') for l in data.get('listeners', [])]
    listener_ports_and_protocol = [
        {"ListenerPort": l.get('port'), "ListenerProtocol": l.get('protocol', 'tcp').upper()}
        for l in data.get('listeners', [])
    ]
    
    # Build backend servers
    backend_servers = [
        {"ServerId": s.get('ServerId'), "Weight": s.get('Weight', 100)}
        for s in data.get('backend_servers', [])
    ]
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "LoadBalancerId": lb_id,
        "LoadBalancerName": data.get('lb_name', ''),
        "LoadBalancerStatus": data.get('status', 'Active'),
        "Address": "192.168.1.100",
        "AddressType": data.get('address_type', 'intranet'),
        "VpcId": data.get('vpc_id', ''),
        "CreateTime": "2023-01-01T00:00:00Z",
        "ListenerPorts": {"ListenerPort": listener_ports},
        "ListenerPortsAndProtocol": {"ListenerPortAndProtocol": listener_ports_and_protocol},
        "BackendServers": {"BackendServer": backend_servers}
    }


def handle_DescribeLoadBalancerHTTPListenerAttribute(query, state, chaos, clock):
    """Describe HTTP listener attributes."""
    lb_id = query.get('LoadBalancerId')
    port = query.get('ListenerPort')
    
    if not lb_id or not state.exists(lb_id):
        return {
            "Code": "InvalidLoadBalancerId.NotFound",
            "Message": "The specified LoadBalancer does not exist."
        }
    
    lb_data = state.get(lb_id)
    listener = None
    
    for l in lb_data.get('listeners', []):
        if str(l.get('port')) == str(port) and l.get('protocol') == 'http':
            listener = l
            break
    
    if not listener:
        return {
            "Code": "InvalidListenerPort.NotFound",
            "Message": "The specified listener does not exist."
        }
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "ListenerPort": listener.get('port'),
        "BackendServerPort": listener.get('backend_port'),
        "Status": listener.get('status', 'Stopped'),
        "Bandwidth": -1,
        "HealthCheck": "on",
        "StickySession": "off"
    }


def handle_DescribeLoadBalancerTCPListenerAttribute(query, state, chaos, clock):
    """Describe TCP listener attributes."""
    lb_id = query.get('LoadBalancerId')
    port = query.get('ListenerPort')
    
    if not lb_id or not state.exists(lb_id):
        return {
            "Code": "InvalidLoadBalancerId.NotFound",
            "Message": "The specified LoadBalancer does not exist."
        }
    
    lb_data = state.get(lb_id)
    listener = None
    
    for l in lb_data.get('listeners', []):
        if str(l.get('port')) == str(port) and l.get('protocol') == 'tcp':
            listener = l
            break
    
    if not listener:
        return {
            "Code": "InvalidListenerPort.NotFound",
            "Message": "The specified listener does not exist."
        }
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "ListenerPort": listener.get('port'),
        "BackendServerPort": listener.get('backend_port'),
        "Status": listener.get('status', 'Stopped'),
        "Bandwidth": listener.get('bandwidth', -1),
        "HealthCheck": "on",
        "Scheduler": "wrr"
    }


def handle_DescribeLoadBalancerListeners(query, state, chaos, clock):
    """Describe all listeners of a load balancer."""
    lb_ids = query.get('LoadBalancerId')
    
    listeners_list = []
    
    if lb_ids:
        # Normalize to list
        if isinstance(lb_ids, str):
            import json
            try:
                target_ids = json.loads(lb_ids)
                if not isinstance(target_ids, list):
                    target_ids = [lb_ids]
            except:
                target_ids = [lb_ids]
        elif isinstance(lb_ids, list):
             target_ids = lb_ids
        else:
             target_ids = []

        for lb_id in target_ids:
            if not state.exists(lb_id):
                # If checking multiple, maybe ignore missing? 
                # But if checking ONE and it's missing, return error.
                # Aliyun usually returns empty for missing if list provided?
                # Or checks all?
                # For simplicity, if ANY missing, return error if it's the only one.
                # If list, usually filter.
                # Let's assume strict check for now as per previous logic.
                if len(target_ids) == 1:
                     return {
                        "Code": "InvalidLoadBalancerId.NotFound",
                        "Message": "The specified LoadBalancer does not exist."
                    }
                continue
            
            lb_data = state.get(lb_id)
            for l in lb_data.get('listeners', []):
                listeners_list.append({
                    "LoadBalancerId": lb_id,
                    "ListenerPort": l.get('port'),
                    "ListenerProtocol": l.get('protocol', 'tcp').upper(),
                    "BackendServerPort": l.get('backend_port'),
                    "Status": l.get('status', 'Stopped')
                })
    else:
        # List all listeners from all SLBs
        lbs = state.list_by_type("SLB")
        for lbid, data in lbs.items():
            for l in data.get('listeners', []):
                listeners_list.append({
                    "LoadBalancerId": lbid,
                    "ListenerPort": l.get('port'),
                    "ListenerProtocol": l.get('protocol', 'tcp').upper(),
                    "BackendServerPort": l.get('backend_port'),
                    "Status": l.get('status', 'Stopped')
                })
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TotalCount": len(listeners_list),
        "Listeners": listeners_list
    }



