from typing import Any, Dict, Type

def make_response(response_class: Type, data: Dict[str, Any]) -> Any:
    """
    Convert a dictionary to an Alibaba Cloud SDK Response object.
    
    Args:
        response_class: The class of the response (e.g., CreateVpcResponse)
        data: The dictionary containing the response body data
        
    Returns:
        An instance of response_class with the body populated
    """
    response = response_class()
    
    # Most SDK responses have a 'body' attribute which is another model
    if hasattr(response, 'body'):
        # Try to find the class for the body
        # In Tea models, type hints are usually available in __annotations__
        # or we can try to infer it from the module
        
        body_type = None
        if hasattr(response_class, '__annotations__') and 'body' in response_class.__annotations__:
            body_type = response_class.__annotations__['body']
        
        if body_type:
            # Instantiate the body class
            body_instance = body_type()
            # Use from_map if available (Tea Model standard)
            if hasattr(body_instance, 'from_map'):
                body_instance.from_map(data)
            else:
                # Fallback: set attributes directly
                for k, v in data.items():
                    if hasattr(body_instance, k):
                        setattr(body_instance, k, v)
            
            response.body = body_instance
            
    return response
