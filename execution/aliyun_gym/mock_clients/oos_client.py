from alibabacloud_oos20190601.client import Client as OosClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockOosClient(MockClientBase, OosClient):
    """
    Mock Client for OOS (Operation Orchestration Service).
    """
    PRODUCT_ID = "OOS"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
