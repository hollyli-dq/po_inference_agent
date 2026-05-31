# Mock Clients for Aliyun-Gym
# These clients inherit from official Alibaba Cloud SDKs but intercept API calls

from execution.aliyun_gym.mock_clients.vpc_client import MockVpcClient
from execution.aliyun_gym.mock_clients.ecs_client import MockEcsClient
from execution.aliyun_gym.mock_clients.slb_client import MockSlbClient
from execution.aliyun_gym.mock_clients.rds_client import MockRdsClient
from execution.aliyun_gym.mock_clients.redis_client import MockRedisClient
from execution.aliyun_gym.mock_clients.cms_client import MockCmsClient
from execution.aliyun_gym.mock_clients.oos_client import MockOosClient
from execution.aliyun_gym.mock_clients.eip_client import MockEipClient

__all__ = [
    "MockVpcClient",
    "MockEcsClient",
    "MockSlbClient",
    "MockRdsClient",
    "MockRedisClient",
    "MockCmsClient",
    "MockOosClient",
    "MockEipClient",
]
