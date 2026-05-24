"""
意图解析模块

将用户自然语言输入解析为结构化的意图对象
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import re


class IntentType(Enum):
    """
    意图类型 - 对应6个仿真场景
    
    场景设计:
    1. SIMPLE_ECS: 简单 ECS 创建
    2. SLB_ECS_RDS: SLB + ECS + RDS
    3. SLB_ECS_REDIS: SLB + ECS + Redis
    4. EIP_SLB_ECS: EIP + SLB + ECS
    5. DUAL_ZONE_ECS_SLB: 双可用区 ECS×2 + SLB
    6. DUAL_ZONE_ECS_SLB_RDS: 双可用区 ECS×2 + SLB + RDS 主备
    """
    SIMPLE_ECS = "simple_ecs"                      # 简单 ECS 创建
    SLB_ECS_RDS = "slb_ecs_rds"                    # SLB + ECS + RDS
    SLB_ECS_REDIS = "slb_ecs_redis"                # SLB + ECS + Redis
    EIP_SLB_ECS = "eip_slb_ecs"                    # EIP + SLB + ECS
    DUAL_ZONE_ECS_SLB = "dual_zone_ecs_slb"        # 双可用区 ECS×2 + SLB
    DUAL_ZONE_ECS_SLB_RDS = "dual_zone_ecs_slb_rds"  # 双可用区 ECS×2 + SLB + RDS 主备
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """
    解析后的意图
    
    根据6个仿真场景设计，包含必要的参数字段
    """
    intent_type: IntentType
    raw_text: str
    region_id: str = "cn-hangzhou"
    zone_id: Optional[str] = None
    zone_id_secondary: Optional[str] = None  # 第二可用区（双可用区场景）
    vpc_cidr: Optional[str] = None
    vswitch_cidr: Optional[str] = None
    vswitch_cidr_secondary: Optional[str] = None  # 第二可用区 VSwitch CIDR
    ecs_count: int = 1
    ecs_type: Optional[str] = None
    image_id: Optional[str] = None
    # 资源规格字段
    rds_class: Optional[str] = None
    redis_class: Optional[str] = None
    slb_spec: Optional[str] = None
    # 场景标志位（由 intent_type 自动推断）
    need_slb: bool = False
    need_eip: bool = False
    need_rds: bool = False
    need_redis: bool = False
    dual_zone: bool = False
    rds_ha: bool = False  # RDS 主备模式
    extra_params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    # LLM token 统计
    llm_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type.value,
            "raw_text": self.raw_text,
            "region_id": self.region_id,
            "zone_id": self.zone_id,
            "zone_id_secondary": self.zone_id_secondary,
            "vpc_cidr": self.vpc_cidr,
            "vswitch_cidr": self.vswitch_cidr,
            "vswitch_cidr_secondary": self.vswitch_cidr_secondary,
            "ecs_count": self.ecs_count,
            "ecs_type": self.ecs_type,
            "image_id": self.image_id,
            "rds_class": self.rds_class,
            "redis_class": self.redis_class,
            "slb_spec": self.slb_spec,
            "need_slb": self.need_slb,
            "need_eip": self.need_eip,
            "need_rds": self.need_rds,
            "need_redis": self.need_redis,
            "dual_zone": self.dual_zone,
            "rds_ha": self.rds_ha,
            "extra_params": self.extra_params,
            "confidence": self.confidence,
            "llm_tokens": self.llm_tokens
        }


class IntentParser:
    """
    意图解析器
    
    支持两种模式：
    1. 规则解析（无需 LLM，快速）
    2. LLM 解析（需要 LLM，更灵活）
    """
    
    # 区域映射
    REGION_KEYWORDS = {
        "杭州": "cn-hangzhou",
        "hangzhou": "cn-hangzhou",
        "上海": "cn-shanghai",
        "shanghai": "cn-shanghai",
        "北京": "cn-beijing",
        "beijing": "cn-beijing",
        "深圳": "cn-shenzhen",
        "shenzhen": "cn-shenzhen",
    }
    
    # 意图关键词 - 对应6个仿真场景
    INTENT_PATTERNS = {
        # 场景6: 双可用区 ECS×2 + SLB + RDS 主备 (最复杂，优先匹配)
        IntentType.DUAL_ZONE_ECS_SLB_RDS: [
            r"双可用区.*rds.*主备",
            r"双可用区.*slb.*rds",
            r"双az.*rds",
            r"高可用.*rds.*主备",
            r"dual.*zone.*rds",
        ],
        # 场景5: 双可用区 ECS×2 + SLB
        IntentType.DUAL_ZONE_ECS_SLB: [
            r"双可用区.*ecs.*slb",
            r"双可用区.*slb",
            r"双az.*slb",
            r"dual.*zone.*slb",
            r"高可用.*ecs.*slb",
            r"跨可用区.*slb",
        ],
        # 场景4: EIP + SLB + ECS
        IntentType.EIP_SLB_ECS: [
            r"eip.*slb.*ecs",
            r"公网.*slb.*ecs",
            r"外网.*负载均衡",
            r"公网ip.*slb",
            r"eip.*负载均衡",
        ],
        # 场景3: SLB + ECS + Redis
        IntentType.SLB_ECS_REDIS: [
            r"slb.*ecs.*redis",
            r"负载均衡.*redis",
            r"ecs.*redis.*slb",
            r"redis.*缓存.*slb",
            r"web.*redis",
        ],
        # 场景2: SLB + ECS + RDS
        IntentType.SLB_ECS_RDS: [
            r"slb.*ecs.*rds",
            r"负载均衡.*rds",
            r"ecs.*rds.*slb",
            r"web.*数据库",
            r"应用.*mysql",
            r"slb.*mysql",
        ],
        # 场景1: 简单 ECS 创建 (最简单，最后匹配)
        IntentType.SIMPLE_ECS: [
            r"简单.*ecs",
            r"创建.*ecs",
            r"创建.*服务器",
            r"创建.*实例",
            r"启动.*ecs",
            r"单.*ecs",
        ],
    }
    
    def __init__(self, llm_client=None):
        """
        初始化解析器
        
        Args:
            llm_client: 可选的 LLM 客户端，用于 LLM 解析模式
        """
        self.llm_client = llm_client
    
    def parse(self, text: str, use_llm: bool = False) -> ParsedIntent:
        """
        解析用户输入
        
        Args:
            text: 用户输入文本
            use_llm: 是否使用 LLM 解析
            
        Returns:
            ParsedIntent 对象
        """
        if use_llm and self.llm_client:
            return self._parse_with_llm(text)
        return self._parse_with_rules(text)
    
    def _map_spec_to_resources(self, cpu: int, memory: int) -> Dict[str, str]:
        """
        将 CPU/Memory 规格映射为具体的阿里云资源规格
        
        Args:
            cpu: CPU 核数
            memory: 内存大小 (GB)
            
        Returns:
            Dict 包含 ecs_type, rds_class, redis_class
        """
        # ECS 映射 (简化版)
        if cpu == 2 and memory == 4:
            ecs_type = "ecs.c6.large"      # 2vCPU 4GiB (计算型)
            rds_class = "mysql.n2.medium.2c" # 2核 4GB (通用型)
            redis_class = "redis.master.mid.default" # 4GB (Redis主要看内存，这里映射为4G规格)
        elif cpu == 4 and memory == 8:
            ecs_type = "ecs.c6.xlarge"     # 4vCPU 8GiB
            rds_class = "mysql.n2.large.2c"  # 4核 8GB
            redis_class = "redis.master.large.default" # 8GB
        elif cpu == 1 and memory == 2:
            ecs_type = "ecs.t6.small"      # 1vCPU 2GiB
            rds_class = "mysql.n2.small.1c"  # 1核 2GB
            redis_class = "redis.master.small.default" # 1GB (近似)
        else:
            # 默认回退
            ecs_type = f"ecs.custom.{cpu}c{memory}g"
            rds_class = f"mysql.custom.{cpu}c{memory}g"
            redis_class = "redis.master.small.default"
            
        return {
            "ecs_type": ecs_type,
            "rds_class": rds_class,
            "redis_class": redis_class
        }

    def _parse_with_rules(self, text: str) -> ParsedIntent:
        """使用规则解析"""
        text_lower = text.lower()
        
        # 识别意图类型（按优先级顺序匹配）
        intent_type = IntentType.UNKNOWN
        for itype, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    intent_type = itype
                    break
            if intent_type != IntentType.UNKNOWN:
                break
        
        # 提取区域
        region_id = "cn-hangzhou"  # 默认
        for keyword, region in self.REGION_KEYWORDS.items():
            if keyword in text_lower:
                region_id = region
                break
        
        # 提取 ECS 数量
        ecs_count = 1
        count_match = re.search(r'(\d+)\s*[台个]', text)
        if count_match:
            ecs_count = int(count_match.group(1))
            
        # 提取规格 (如 2C4G, 4c8g)
        ecs_type = None
        rds_class = None
        redis_class = None
        
        spec_match = re.search(r'(\d+)[c|C](\d+)[g|G]', text)
        if spec_match:
            cpu = int(spec_match.group(1))
            memory = int(spec_match.group(2))
            specs = self._map_spec_to_resources(cpu, memory)
            ecs_type = specs["ecs_type"]
            rds_class = specs["rds_class"]
            redis_class = specs["redis_class"]
        
        # 根据 intent_type 自动推断场景标志位
        need_slb, need_eip, need_rds, need_redis, dual_zone, rds_ha = \
            self._infer_flags_from_intent(intent_type)
        
        # 双可用区场景默认 ECS 数量为 2
        if dual_zone and ecs_count == 1:
            ecs_count = 2
        
        return ParsedIntent(
            intent_type=intent_type,
            raw_text=text,
            region_id=region_id,
            ecs_count=ecs_count,
            ecs_type=ecs_type,
            rds_class=rds_class,
            redis_class=redis_class,
            need_slb=need_slb,
            need_eip=need_eip,
            need_rds=need_rds,
            need_redis=need_redis,
            dual_zone=dual_zone,
            rds_ha=rds_ha,
            confidence=0.8 if intent_type != IntentType.UNKNOWN else 0.3
        )
    
    def _infer_flags_from_intent(self, intent_type: IntentType) -> tuple:
        """
        根据意图类型推断场景标志位
        
        Returns:
            (need_slb, need_eip, need_rds, need_redis, dual_zone, rds_ha)
        """
        flags = {
            IntentType.SIMPLE_ECS: (False, False, False, False, False, False),
            IntentType.SLB_ECS_RDS: (True, False, True, False, False, False),
            IntentType.SLB_ECS_REDIS: (True, False, False, True, False, False),
            IntentType.EIP_SLB_ECS: (True, True, False, False, False, False),
            IntentType.DUAL_ZONE_ECS_SLB: (True, False, False, False, True, False),
            IntentType.DUAL_ZONE_ECS_SLB_RDS: (True, False, True, False, True, True),
            IntentType.UNKNOWN: (False, False, False, False, False, False),
        }
        return flags.get(intent_type, (False, False, False, False, False, False))
    
    def _parse_with_llm(self, text: str) -> ParsedIntent:
        """使用 LLM 解析"""
        prompt = self._build_parse_prompt(text)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="qwen3-max", # 确保使用强大的模型进行解析
                messages=[
                    {"role": "system", "content": "你是一个云资源配置解析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # 提取 token 统计
            llm_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                llm_tokens = getattr(response.usage, 'total_tokens', 0)
            
            content = response.choices[0].message.content
            result = json.loads(content)
            intent = self._build_intent_from_llm_result(text, result)
            intent.llm_tokens = llm_tokens
            return intent
            
        except Exception as e:
            # 降级到规则解析
            intent = self._parse_with_rules(text)
            intent.confidence = 0.5
            return intent
    
    def _build_parse_prompt(self, text: str) -> str:
        """构建 LLM 解析 prompt"""
        return f"""
请解析以下云资源创建请求，提取关键参数，并将通用规格(如 2C4G)转换为阿里云具体产品规格。

用户输入: {text}

参考规格映射:
- 2C4G -> ECS: ecs.c6.large, RDS: mysql.n2.medium.2c, Redis: redis.master.mid.default (4G)
- 4C8G -> ECS: ecs.c6.xlarge, RDS: mysql.n2.large.2c, Redis: redis.master.large.default (8G)
- 1C2G -> ECS: ecs.t6.small, RDS: mysql.n2.small.1c, Redis: redis.master.small.default (1G)

意图类型说明:
- simple_ecs: 简单 ECS 创建
- slb_ecs_rds: SLB + ECS + RDS
- slb_ecs_redis: SLB + ECS + Redis  
- eip_slb_ecs: EIP + SLB + ECS
- dual_zone_ecs_slb: 双可用区 ECS×2 + SLB
- dual_zone_ecs_slb_rds: 双可用区 ECS×2 + SLB + RDS 主备

请返回 JSON 格式:
{{
    "intent_type": "simple_ecs|slb_ecs_rds|slb_ecs_redis|eip_slb_ecs|dual_zone_ecs_slb|dual_zone_ecs_slb_rds|unknown",
    "region_id": "cn-hangzhou",
    "ecs_count": 2,
    "ecs_type": "ecs.g6.large",
    "rds_class": "mysql.n2.medium.1",
    "redis_class": "redis.master.small.default",
    "dual_zone": false,
    "vpc_cidr": "172.16.0.0/12",
    "vswitch_cidr": "172.16.0.0/24"
}}
"""
    
    def _build_intent_from_llm_result(self, text: str, result: Dict[str, Any]) -> ParsedIntent:
        """从 LLM 结果构建 Intent"""
        intent_type_str = result.get("intent_type", "unknown")
        try:
            intent_type = IntentType(intent_type_str)
        except ValueError:
            intent_type = IntentType.UNKNOWN
        
        # 根据 intent_type 推断场景标志位
        need_slb, need_eip, need_rds, need_redis, dual_zone, rds_ha = \
            self._infer_flags_from_intent(intent_type)
        
        # LLM 可以覆盖双可用区标志
        if result.get("dual_zone"):
            dual_zone = True
        
        ecs_count = result.get("ecs_count", 1)
        if dual_zone and ecs_count == 1:
            ecs_count = 2
        
        return ParsedIntent(
            intent_type=intent_type,
            raw_text=text,
            region_id=result.get("region_id", "cn-hangzhou"),
            ecs_count=ecs_count,
            ecs_type=result.get("ecs_type"),
            rds_class=result.get("rds_class"),
            redis_class=result.get("redis_class"),
            slb_spec=result.get("slb_spec"),
            need_slb=need_slb,
            need_eip=need_eip,
            need_rds=need_rds,
            need_redis=need_redis,
            dual_zone=dual_zone,
            rds_ha=rds_ha,
            vpc_cidr=result.get("vpc_cidr", "172.16.0.0/12"),
            vswitch_cidr=result.get("vswitch_cidr", "172.16.0.0/24"),
            confidence=0.9
        )
    
    # 标准化运维参数（确保专家模式能直接执行）
    STANDARD_PARAMS = {
        "cn-hangzhou": {
            "ZoneId": "cn-hangzhou-h",
            "ZoneIdSecondary": "cn-hangzhou-i",
            "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
        },
        "cn-shanghai": {
            "ZoneId": "cn-shanghai-b",
            "ZoneIdSecondary": "cn-shanghai-g",
            "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
        },
        "cn-beijing": {
            "ZoneId": "cn-beijing-h",
            "ZoneIdSecondary": "cn-beijing-g",
            "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
        },
        "cn-shenzhen": {
            "ZoneId": "cn-shenzhen-d",
            "ZoneIdSecondary": "cn-shenzhen-e",
            "ImageId": "centos_7_9_x64_20G_alibase_20230816.vhd",
        },
    }
    
    def intent_to_blackboard(self, intent: ParsedIntent) -> Dict[str, Any]:
        """
        将 Intent 转换为 Blackboard 初始参数
        
        返回分层结构:
        - global: 全局参数 (RegionId, ZoneId)
        - namespace: 产品级默认配置
        
        特性:
        - 自动填充标准化运维参数（ZoneId, ImageId, InstanceType 等）
        - 确保专家模式能直接执行成功
        """
        # 获取区域对应的标准参数
        region_params = self.STANDARD_PARAMS.get(
            intent.region_id, 
            self.STANDARD_PARAMS["cn-hangzhou"]  # 默认杭州
        )
        
        # 确定 ZoneId（优先用户指定，否则用标准参数）
        zone_id = intent.zone_id or region_params["ZoneId"]
        zone_id_secondary = intent.zone_id_secondary or region_params.get("ZoneIdSecondary")
        
        # 确定 ImageId（优先用户指定，否则用标准参数）
        image_id = intent.image_id or region_params["ImageId"]
        
        # 确定 InstanceType（优先用户指定，否则用默认值）
        instance_type = intent.ecs_type or "ecs.c6.large"
        
        result = {
            # 全局参数（包含完整的标准化参数）
            "global": {
                "RegionId": intent.region_id,
                "ZoneId": zone_id,
                "Amount": intent.ecs_count,
                # ECS 必要参数
                "InstanceType": instance_type,
                "ImageId": image_id,
                # VSwitch CIDR
                "CidrBlock": intent.vpc_cidr or "172.16.0.0/12",
                # 安全组授权参数
                "IpProtocol": "tcp",
                "PortRange": "80/80",
                "SourceCidrIp": "0.0.0.0/0",
            },
            # 产品命名空间默认配置
            "namespace": {
                "vpc": {
                    "CidrBlock": intent.vpc_cidr or "172.16.0.0/12",
                },
                "ecs": {
                    "InstanceType": instance_type,
                    "ImageId": image_id,
                    "SystemDiskCategory": "cloud_essd",
                    "SystemDiskSize": 40,
                },
                "rds": {
                    "Engine": "MySQL",
                    "EngineVersion": "8.0",
                    "DBInstanceNetType": "Intranet",
                    "SecurityIPList": "172.16.0.0/12",
                    "PayType": "Postpaid",
                    "DBInstanceStorageType": "cloud_essd",
                    "DBInstanceClass": intent.rds_class or "mysql.n2.medium.2c",
                    "DBInstanceStorage": 20,
                    # RDS Account 参数
                    "AccountName": "admin",
                    "AccountPassword": "Admin@123456",
                    "SecurityIps": "172.16.0.0/12",
                },
                "redis": {
                    "Engine": "Redis",
                    "EngineVersion": "7.0",
                    "ChargeType": "PostPaid",
                    "NetworkType": "VPC",
                    "InstanceClass": intent.redis_class or "redis.master.mid.default",
                },
                "slb": {
                    "AddressType": "intranet",
                    "LoadBalancerSpec": intent.slb_spec or "slb.s1.small",
                    # SLB Listener 参数
                    "ListenerPort": 80,
                    "BackendServerPort": 80,
                    "ListenerProtocol": "http",
                    "HealthCheck": "on",
                },
                "eip": {
                    "Bandwidth": "5",
                    "InternetChargeType": "PayByTraffic",
                },
            }
        }
        
        # 双可用区场景：添加第二可用区
        if intent.dual_zone and zone_id_secondary:
            result["global"]["ZoneIdSecondary"] = zone_id_secondary
        
        # VSwitch CIDR
        if intent.vswitch_cidr:
            result["namespace"]["vpc"]["VSwitchCidrBlock"] = intent.vswitch_cidr
        
        return result
