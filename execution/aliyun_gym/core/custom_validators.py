from typing import Dict, Any, Optional, List

class CustomValidators:
    @staticmethod
    def check_mutex(query: Dict[str, Any], params: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check that at most one of the parameters is present.
        """
        present = [p for p in params if p in query]
        if len(present) > 1:
            return {
                "Code": "InvalidParameter",
                "Message": f"The parameters {present} are mutually exclusive and cannot be specified together."
            }
        return None

    @staticmethod
    def check_required_if_value(query: Dict[str, Any], 
                              trigger_param: str, 
                              trigger_value: str, 
                              required_params: List[str]) -> Optional[Dict[str, Any]]:
        """
        If trigger_param has trigger_value, then required_params must be present.
        """
        if query.get(trigger_param) == trigger_value:
            missing = [p for p in required_params if p not in query]
            if missing:
                return {
                    "Code": "MissingParameter",
                    "Message": f"When {trigger_param} is {trigger_value}, the following parameters are required: {missing}"
                }
        return None

# Map Action -> List of validator functions
# Each function takes (query) and returns None or error dict
CUSTOM_RULES = {
    "RunInstances": [
        lambda q: CustomValidators.check_mutex(q, ["ImageId", "ImageFamily"]),
        lambda q: CustomValidators.check_mutex(q, ["SecurityGroupId", "SecurityGroupIds"]),
        # Add more rules as needed
    ]
}

def validate_custom_rules(action: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rules = CUSTOM_RULES.get(action, [])
    for rule in rules:
        error = rule(query)
        if error:
            return error
    return None
