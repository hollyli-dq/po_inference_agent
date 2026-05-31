from alibabacloud_cms20190101.client import Client as CmsClient
from execution.aliyun_gym.mock_clients.base import MockClientBase

class MockCmsClient(MockClientBase, CmsClient):
    """
    Mock Client for CMS (Cloud Monitor Service).
    """
    PRODUCT_ID = "CMS"
    
    def __init__(self, state_store, chaos_injector, action_router):
        MockClientBase.__init__(self, state_store, chaos_injector, action_router)
