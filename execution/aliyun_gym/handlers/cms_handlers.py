import uuid
import time
import random

def handle_CreateMonitorGroup(query, state, chaos, clock):
    if chaos.should_fail():
        return chaos.generate_error("CreateMonitorGroup")

    group_id = f"gid-{uuid.uuid4().hex[:8]}"
    
    state.put(group_id, {
        "type": "MonitorGroup",
        "status": "Active",
        "group_name": query.get('GroupName'),
        "created_at": clock.now(),
        "instances": []
    })
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "GroupId": group_id,
        "GroupName": query.get('GroupName')
    }

def handle_CreateMonitorGroupInstances(query, state, chaos, clock):
    group_id = query.get('GroupId')
    instances = query.get('Instances') # JSON string usually
    
    if not group_id or not state.exists(group_id):
        return {"Code": "InvalidGroupId.NotFound", "Message": "Not found"}
        
    # Parse instances
    import json
    try:
        inst_list = json.loads(instances)
    except:
        inst_list = []
        
    group_data = state.get(group_id)
    
    # Weak dependency check: warn but don't fail if instance doesn't exist?
    # Or fail? CMS usually fails if instance ID is invalid.
    for inst in inst_list:
        iid = inst.get('InstanceId')
        if not state.exists(iid):
             return {"Code": "InvalidInstanceId.NotFound", "Message": f"Instance {iid} not found"}
        group_data['instances'].append(inst)
        
    clock.tick(0.5)
    
    return {
        "RequestId": str(uuid.uuid4())
    }

def handle_CreateHostAvailability(query, state, chaos, clock):
    group_id = query.get('GroupId')
    if not group_id or not state.exists(group_id):
        return {"Code": "InvalidGroupId.NotFound", "Message": "Not found"}
        
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    
    clock.tick(0.2)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "TaskId": task_id
    }


def handle_DescribeMetricLast(query, state, chaos, clock):
    """
    Describe the latest metric data for cloud resources.
    Returns mock monitoring data.
    """
    namespace = query.get('Namespace', 'acs_ecs_dashboard')  # e.g., acs_ecs_dashboard, acs_rds_dashboard
    metric_name = query.get('MetricName', 'CPUUtilization')
    dimensions = query.get('Dimensions')  # JSON string like [{"instanceId":"i-xxx"}]
    
    # Parse dimensions if provided
    import json
    instance_ids = []
    if dimensions:
        try:
            dims = json.loads(dimensions)
            if isinstance(dims, list):
                for d in dims:
                    if isinstance(d, dict) and 'instanceId' in d:
                        instance_ids.append(d['instanceId'])
        except:
            pass
    
    # Generate mock metric data points
    datapoints = []
    
    # Common metrics and their mock values
    mock_metrics = {
        'CPUUtilization': {'Average': 25.5, 'Maximum': 45.0, 'Minimum': 5.0},
        'MemoryUtilization': {'Average': 60.0, 'Maximum': 85.0, 'Minimum': 40.0},
        'DiskReadIOPS': {'Average': 100, 'Maximum': 500, 'Minimum': 10},
        'DiskWriteIOPS': {'Average': 80, 'Maximum': 400, 'Minimum': 5},
        'InternetInRate': {'Average': 1024, 'Maximum': 5120, 'Minimum': 100},
        'InternetOutRate': {'Average': 512, 'Maximum': 2048, 'Minimum': 50},
        'MySQL_IOPS': {'Average': 50, 'Maximum': 200, 'Minimum': 10},
        'ConnectionUsage': {'Average': 30.0, 'Maximum': 60.0, 'Minimum': 5.0},
    }
    
    metric_values = mock_metrics.get(metric_name, {'Average': 50.0, 'Maximum': 80.0, 'Minimum': 20.0})
    
    # If specific instances are requested, generate data for each
    if instance_ids:
        for iid in instance_ids:
            import random
            datapoints.append({
                'instanceId': iid,
                'timestamp': int(clock.now() * 1000),  # Milliseconds
                'Average': metric_values['Average'] + random.uniform(-5, 5),
                'Maximum': metric_values['Maximum'],
                'Minimum': metric_values['Minimum'],
            })
    else:
        # Return a single mock data point
        import random
        datapoints.append({
            'timestamp': int(clock.now() * 1000),
            'Average': metric_values['Average'] + random.uniform(-5, 5),
            'Maximum': metric_values['Maximum'],
            'Minimum': metric_values['Minimum'],
        })
    
    clock.tick(0.1)
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Code": "200",
        "Success": True,
        "Period": "60",
        "Datapoints": json.dumps(datapoints)
    }
