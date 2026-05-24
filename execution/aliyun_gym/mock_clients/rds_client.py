from alibabacloud_rds20140815.client import Client as RdsClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockRdsClient(MockClientBase, RdsClient):
    """
    Mock Client for RDS.
    """
    PRODUCT_ID = "RDS"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
