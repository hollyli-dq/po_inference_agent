import random
from typing import Dict, Optional

class ChaosInjector:
    """
    Fault injection engine for Aliyun-Gym.
    Simulates random cloud failures.
    """
    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate

    def should_fail(self, error_type: str = "random") -> bool:
        """Check if a failure should occur."""
        return random.random() < self.failure_rate

    def generate_error(self, action: str) -> Dict[str, str]:
        """Generate a realistic error response."""
        scenarios = [
            # Network errors (Retryable)
            {"Code": "Sdk.ReadTimeout", "Message": "Read timed out"},
            {"Code": "ServiceUnavailable", "Message": "The service is temporarily unavailable."},
            
            # Resource errors (Need change strategy)
            {"Code": "OperationDenied.NoStock", "Message": "The specified instance type is out of stock."},
            
            # Throttling
            {"Code": "Throttling.User", "Message": "Request was denied due to user flow control."}
        ]
        return random.choice(scenarios)
