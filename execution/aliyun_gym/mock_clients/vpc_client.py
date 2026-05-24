from alibabacloud_vpc20160428.client import Client as VpcClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockVpcClient(MockClientBase, VpcClient):
    """
    Mock Client for VPC.
    Inherits from both MockClientBase (for interception) and VpcClient (for type signatures).
    """
    PRODUCT_ID = "VPC"
    
    def __init__(self, state_store, chaos_injector, action_router):
        # Initialize MockClientBase
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
        # We do NOT call VpcClient.__init__ because it requires real config
