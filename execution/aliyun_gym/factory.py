"""
Factory functions for creating Aliyun-Gym environments.
Provides a convenient way to set up mock clients with shared state.
"""
from typing import Dict, Optional, NamedTuple

from execution.aliyun_gym.core.state_store import StateStore
from execution.aliyun_gym.core.chaos_injector import ChaosInjector
from execution.aliyun_gym.core.time_keeper import VirtualClock
from execution.aliyun_gym.core.action_router import ActionRouter
from execution.aliyun_gym.core.trace_recorder import TraceRecorder

from execution.aliyun_gym.mock_clients.vpc_client import MockVpcClient
from execution.aliyun_gym.mock_clients.ecs_client import MockEcsClient
from execution.aliyun_gym.mock_clients.slb_client import MockSlbClient
from execution.aliyun_gym.mock_clients.rds_client import MockRdsClient
from execution.aliyun_gym.mock_clients.redis_client import MockRedisClient
from execution.aliyun_gym.mock_clients.cms_client import MockCmsClient
from execution.aliyun_gym.mock_clients.oos_client import MockOosClient
from execution.aliyun_gym.mock_clients.eip_client import MockEipClient


class AliyunGymEnv(NamedTuple):
    """Container for all Aliyun-Gym components."""
    state_store: StateStore
    chaos_injector: ChaosInjector
    clock: VirtualClock
    action_router: ActionRouter
    trace_recorder: TraceRecorder
    
    # Mock clients
    vpc_client: MockVpcClient
    ecs_client: MockEcsClient
    slb_client: MockSlbClient
    rds_client: MockRdsClient
    redis_client: MockRedisClient
    cms_client: MockCmsClient
    oos_client: MockOosClient
    eip_client: MockEipClient


def create_gym_env(failure_rate: float = 0.0, 
                   enable_latency: bool = True,
                   use_real_latency: bool = False) -> AliyunGymEnv:
    """
    Create a complete Aliyun-Gym environment with all mock clients.
    
    Args:
        failure_rate: Probability of random failures (0.0 to 1.0)
        enable_latency: Whether to simulate API latency (Describe*: 1s, others: 3-5s)
        use_real_latency: Whether to use real time.sleep() (slow but realistic)
    
    Returns:
        AliyunGymEnv with all components initialized and sharing state.
    
    Example:
        env = create_gym_env(failure_rate=0.1)
        
        # Use clients
        response = env.vpc_client.create_vpc(request)
        
        # Access shared state
        print(env.state_store.dump())
        
        # Record traces
        env.trace_recorder.start_trace("Create VPC")
        ...
        env.trace_recorder.end_trace("Success")
    """
    # Create core components
    state_store = StateStore()
    chaos_injector = ChaosInjector(failure_rate=failure_rate)
    clock = VirtualClock()
    action_router = ActionRouter(
        state_store, chaos_injector, clock, 
        enable_latency=enable_latency,
        use_real_latency=use_real_latency
    )
    trace_recorder = TraceRecorder()
    
    # Create mock clients (all share the same state)
    vpc_client = MockVpcClient(state_store, chaos_injector, action_router)
    ecs_client = MockEcsClient(state_store, chaos_injector, action_router)
    slb_client = MockSlbClient(state_store, chaos_injector, action_router)
    rds_client = MockRdsClient(state_store, chaos_injector, action_router)
    redis_client = MockRedisClient(state_store, chaos_injector, action_router)
    cms_client = MockCmsClient(state_store, chaos_injector, action_router)
    oos_client = MockOosClient(state_store, chaos_injector, action_router)
    eip_client = MockEipClient(state_store, chaos_injector, action_router)
    
    return AliyunGymEnv(
        state_store=state_store,
        chaos_injector=chaos_injector,
        clock=clock,
        action_router=action_router,
        trace_recorder=trace_recorder,
        vpc_client=vpc_client,
        ecs_client=ecs_client,
        slb_client=slb_client,
        rds_client=rds_client,
        redis_client=redis_client,
        cms_client=cms_client,
        oos_client=oos_client,
        eip_client=eip_client,
    )


def reset_env(env: AliyunGymEnv) -> None:
    """
    Reset the environment to initial state.
    Clears all resources and resets the clock.
    
    Args:
        env: The AliyunGymEnv to reset
    """
    env.state_store.clear()
    env.clock.reset()
    env.trace_recorder.clear()


def get_client_by_product(env: AliyunGymEnv, product: str):
    """
    Get the appropriate mock client for a product.
    
    Args:
        env: The AliyunGymEnv
        product: Product name (e.g., "VPC", "ECS", "RDS")
    
    Returns:
        The corresponding mock client
    """
    product_upper = product.upper()
    clients = {
        "VPC": env.vpc_client,
        "ECS": env.ecs_client,
        "SLB": env.slb_client,
        "RDS": env.rds_client,
        "REDIS": env.redis_client,
        "CMS": env.cms_client,
        "OOS": env.oos_client,
        "EIP": env.eip_client,
    }
    
    if product_upper not in clients:
        raise ValueError(f"Unknown product: {product}. Available: {list(clients.keys())}")
    
    return clients[product_upper]
