import uuid
import time
import random
import json

def handle_StartExecution(query, state, chaos, clock):
    # OOS StartExecution
    # Usually takes 'Parameters' which contains 'instanceId'
    
    params_str = query.get('Parameters')
    instance_id = None
    
    if params_str:
        try:
            params = json.loads(params_str)
            instance_id = params.get('instanceId')
        except:
            pass
            
    # If instanceId is found, check its status
    if instance_id:
        if not state.exists(instance_id):
            return {"Code": "InvalidInstanceId.NotFound", "Message": "Instance not found"}
            
        inst_data = state.get(instance_id)
        
        # Check if Running
        # We need to update status based on time first
        current_time = clock.now()
        if inst_data['status'] == "Pending":
            if current_time - inst_data['created_at'] >= inst_data['boot_time']:
                inst_data['status'] = "Running"
                
        if inst_data['status'] != "Running":
            return {
                "Code": "InstanceNotRunning", 
                "Message": f"Instance {instance_id} is not running (Status: {inst_data['status']})."
            }

    if chaos.should_fail():
        return chaos.generate_error("StartExecution")

    execution_id = f"exec-{uuid.uuid4().hex[:8]}"
    
    clock.tick(1.0) # OOS execution takes time
    
    return {
        "RequestId": str(uuid.uuid4()),
        "Execution": {
            "ExecutionId": execution_id,
            "Status": "Started"
        }
    }
