from alibabacloud_r_kvstore20150101.client import Client as RedisClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockRedisClient(MockClientBase, RedisClient):
    """
    Mock Client for Redis (R-KVStore).
    """
    PRODUCT_ID = "REDIS"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
