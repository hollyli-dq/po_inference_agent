from alibabacloud_slb20140515.client import Client as SlbClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockSlbClient(MockClientBase, SlbClient):
    """
    Mock Client for SLB.
    """
    PRODUCT_ID = "SLB"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
