from alibabacloud_vpc20160428.client import Client as VpcClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockEipClient(MockClientBase, VpcClient):
    """
    Mock Client for EIP.
    EIP APIs are part of VPC SDK, but we use a separate PRODUCT_ID
    to route to eip_handlers if needed.
    """
    PRODUCT_ID = "EIP"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
