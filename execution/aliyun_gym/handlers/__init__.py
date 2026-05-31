# API Handlers for Aliyun-Gym
# Each handler simulates one Alibaba Cloud API

# Handlers are loaded dynamically by ActionRouter
# This file just documents available handler modules

AVAILABLE_HANDLERS = [
    "vpc_handlers",     # VPC, VSwitch, DescribeZones
    "ecs_handlers",     # SecurityGroup, RunInstances, DescribeInstances
    "slb_handlers",     # LoadBalancer, Listener, BackendServers
    "rds_handlers",     # DBInstance, Account, Connection
    "redis_handlers",   # Redis CreateInstance, DescribeInstances
    "eip_handlers",     # AllocateEipAddress, AssociateEipAddress
    "cms_handlers",     # MonitorGroup, Availability
    "oos_handlers",     # StartExecution
]
