from alibabacloud_ecs20140526.client import Client as EcsClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockEcsClient(MockClientBase, EcsClient):
    """
    Mock Client for ECS.
    """
    PRODUCT_ID = "ECS"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
