"""
API Meta Fetcher for Aliyun-Gym.
参考 alibaba-cloud-ops-mcp-server 的 api_meta_client.py 实现
从阿里云 OpenAPI 门户抓取 API 元信息并保存到本地
"""

import os
import json
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from configparser import ConfigParser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# API Meta 常量定义
API_META_KEYS = (
    VERSION, RESPONSES, SCHEMA, PROPERTIES, HTTP_SUCCESS_CODE, 
    DEFAULT_VERSION, CODE, REF, APIS, SERVICE_KEY, NAME, 
    IN, PARAMETERS, STYLE, BODY, SUMMARY, DESCRIPTION
) = (
    'version', 'responses', 'schema', 'properties', '200', 
    'defaultVersion', 'code', '$ref', 'apis', 'service', 
    'name', 'in', 'parameters', 'style', 'body', 'summary', 'description'
)

# 仿真器支持的服务列表
SUPPORTED_SERVICES = ['vpc', 'ecs', 'rds', 'slb', 'redis', 'eip', 'cms', 'oos']


class ApiMetaFetcher:
    """
    从阿里云 OpenAPI 门户获取 API 元信息并保存到本地。
    
    用法:
        fetcher = ApiMetaFetcher()
        fetcher.fetch_all_services()  # 抓取所有支持服务的 API 元信息
    """
    
    BASE_URL = 'https://api.aliyun.com/meta/v1'
    
    # API 路径配置
    API_PATHS = {
        'products': 'products.json',
        'overview': 'products/{service}/versions/{version}/overview.json',
        'api_info': 'products/{service}/versions/{version}/apis/{api}/api.json',
        'api_docs': 'products/{service}/versions/{version}/api-docs.json',
    }
    
    def __init__(self, 
                 output_dir: str = None,
                 config_file: str = None,
                 services: List[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        初始化 API 元信息获取器。
        
        Args:
            output_dir: 输出目录，默认为 src/aliyun_gym/knowledge/api_docs
            config_file: 配置文件路径（可选，用于扩展认证场景）
            services: 要抓取的服务列表，默认为 SUPPORTED_SERVICES
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        # 设置输出目录
        if output_dir is None:
            current_dir = Path(__file__).parent
            output_dir = current_dir / 'api_docs'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置（可选）
        self.config = {}
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)
        
        # 设置服务列表
        self.services = services or SUPPORTED_SERVICES
        
        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 创建带重试的 session
        self.session = self._create_retry_session()
        
        # 缓存产品列表
        self._products_cache = None
    
    def _create_retry_session(self) -> requests.Session:
        """创建带自动重试的 requests session"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET']
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session
    
    def _load_config(self, config_file: str):
        """加载配置文件（INI 格式）"""
        parser = ConfigParser()
        parser.read(config_file)
        
        if 'aliyun' in parser:
            self.config['access_key_id'] = parser.get('aliyun', 'access_key_id', fallback=None)
            self.config['access_key_secret'] = parser.get('aliyun', 'access_key_secret', fallback=None)
            self.config['region_id'] = parser.get('aliyun', 'region_id', fallback='cn-hangzhou')
        
        logger.info(f"Loaded config from {config_file}")
    
    def _request(self, path: str, **kwargs) -> Dict:
        """发起 HTTP 请求获取 API 元信息（带重试）"""
        url = f"{self.BASE_URL}/{path.format(**kwargs)}"
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries + 1} attempts: {e}")
        
        raise last_error
    
    def get_products(self) -> List[Dict]:
        """获取所有产品列表"""
        if self._products_cache is None:
            self._products_cache = self._request(self.API_PATHS['products'])
        return self._products_cache
    
    def get_service_version(self, service: str) -> Optional[str]:
        """获取服务的默认版本号"""
        products = self.get_products()
        for item in products:
            if item.get(CODE, '').lower() == service.lower():
                return item.get(DEFAULT_VERSION)
        return None
    
    def get_service_info(self, service: str) -> Optional[Dict]:
        """获取服务的完整信息"""
        products = self.get_products()
        for item in products:
            if item.get(CODE, '').lower() == service.lower():
                return item
        return None
    
    def get_api_overview(self, service: str, version: str) -> Dict:
        """获取服务的 API 概览"""
        return self._request(
            self.API_PATHS['overview'],
            service=service,
            version=version
        )
    
    def get_api_info(self, service: str, version: str, api: str) -> Dict:
        """获取单个 API 的详细信息"""
        return self._request(
            self.API_PATHS['api_info'],
            service=service,
            version=version,
            api=api
        )
    
    def get_api_docs(self, service: str, version: str) -> Dict:
        """获取服务的完整 API 文档"""
        return self._request(
            self.API_PATHS['api_docs'],
            service=service,
            version=version
        )
    
    def extract_api_io_info(self, api_info: Dict) -> Dict:
        """
        提取 API 的输入输出信息（用于构建 IO 映射规则）
        
        Returns:
            {
                'name': API名称,
                'summary': API摘要,
                'parameters': [{'name': ..., 'type': ..., 'required': ..., 'description': ...}],
                'response': {'properties': {...}},
            }
        """
        result = {
            'name': api_info.get(NAME, ''),
            'summary': api_info.get(SUMMARY, ''),
            'description': api_info.get(DESCRIPTION, ''),
            'parameters': [],
            'response_properties': [],
        }
        
        # 提取输入参数
        for param in api_info.get(PARAMETERS, []):
            schema = param.get(SCHEMA, {})
            result['parameters'].append({
                'name': param.get(NAME),
                'in': param.get(IN),
                'required': schema.get('required', False),
                'type': schema.get('type', 'string'),
                'description': schema.get('description', ''),
                'example': schema.get('example', ''),
            })
        
        # 提取输出属性
        responses = api_info.get(RESPONSES, {})
        success_response = responses.get(HTTP_SUCCESS_CODE, {})
        schema = success_response.get(SCHEMA, {})
        properties = schema.get(PROPERTIES, {})
        
        for prop_name, prop_info in properties.items():
            result['response_properties'].append({
                'name': prop_name,
                'type': prop_info.get('type', 'string'),
                'description': prop_info.get('description', ''),
            })
        
        return result
    
    def fetch_service(self, service: str, incremental: bool = True) -> Dict:
        """
        抓取单个服务的所有 API 元信息
        
        Args:
            service: 服务代码
            incremental: 是否增量抓取（跳过已存在的 API）
        
        Returns:
            {
                'service': 服务代码,
                'name': 服务名称,
                'version': 版本号,
                'apis': {api_name: api_info, ...}
            }
        """
        logger.info(f"Fetching API meta for service: {service}")
        
        service_info = self.get_service_info(service)
        if not service_info:
            logger.warning(f"Service not found: {service}")
            return {}
        
        version = service_info.get(DEFAULT_VERSION)
        service_code = service_info.get(CODE)
        
        # 尝试加载已有数据（增量模式）
        existing_data = None
        if incremental:
            existing_data = self.load_service_meta(service_code)
        
        result = {
            'service': service_code,
            'name': service_info.get('name', ''),
            'version': version,
            'style': service_info.get('style', 'RPC'),
            'apis': existing_data.get('apis', {}) if existing_data else {},
            'api_io_summary': existing_data.get('api_io_summary', {}) if existing_data else {},
        }
        
        try:
            # 获取 API 概览
            overview = self.get_api_overview(service_code, version)
            api_names = list(overview.get(APIS, {}).keys())
            
            # 计算需要抓取的 API
            existing_apis = set(result['apis'].keys())
            apis_to_fetch = [a for a in api_names if a not in existing_apis]
            
            logger.info(f"Found {len(api_names)} APIs for {service}, need to fetch {len(apis_to_fetch)} new APIs")
            
            # 获取每个 API 的详细信息
            success_count = 0
            fail_count = 0
            for i, api_name in enumerate(apis_to_fetch):
                try:
                    api_info = self.get_api_info(service_code, version, api_name)
                    result['apis'][api_name] = api_info
                    result['api_io_summary'][api_name] = self.extract_api_io_info(api_info)
                    success_count += 1
                    
                    # 每 50 个 API 保存一次（防止中断丢失）
                    if (i + 1) % 50 == 0:
                        self._save_service_data(service_code, result)
                        logger.info(f"Progress: {i + 1}/{len(apis_to_fetch)} APIs fetched")
                    
                    # 请求间隔，避免请求过快
                    time.sleep(0.1)
                    
                except Exception as e:
                    fail_count += 1
                    logger.warning(f"Failed to fetch API {api_name}: {e}")
            
            logger.info(f"Fetch complete: {success_count} success, {fail_count} failed")
            
            # 最终保存
            self._save_service_data(service_code, result)
            
        except Exception as e:
            logger.error(f"Failed to fetch service {service}: {e}")
        
        return result
    
    def _save_service_data(self, service: str, data: Dict):
        """保存服务数据到本地文件"""
        service_dir = self.output_dir / service.lower()
        service_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的 API 元信息
        full_path = service_dir / 'full_meta.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved full meta to {full_path}")
        
        # 保存简化的 IO 摘要（用于业务规则）
        io_summary_path = service_dir / 'io_summary.json'
        with open(io_summary_path, 'w', encoding='utf-8') as f:
            json.dump(data.get('api_io_summary', {}), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved IO summary to {io_summary_path}")
    
    def fetch_all_services(self) -> Dict[str, Dict]:
        """抓取所有支持服务的 API 元信息"""
        results = {}
        for service in self.services:
            results[service] = self.fetch_service(service)
        
        # 生成汇总索引
        self._generate_index(results)
        
        return results
    
    def _generate_index(self, results: Dict[str, Dict]):
        """生成服务索引文件"""
        index = {
            'services': [],
            'total_apis': 0,
        }
        
        for service, data in results.items():
            if data:
                api_count = len(data.get('apis', {}))
                index['services'].append({
                    'code': data.get('service', service),
                    'name': data.get('name', ''),
                    'version': data.get('version', ''),
                    'api_count': api_count,
                })
                index['total_apis'] += api_count
        
        index_path = self.output_dir / 'index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info(f"Generated index at {index_path}")
    
    def load_service_meta(self, service: str) -> Optional[Dict]:
        """从本地加载服务的 API 元信息"""
        meta_path = self.output_dir / service.lower() / 'full_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def load_io_summary(self, service: str) -> Optional[Dict]:
        """从本地加载服务的 IO 摘要"""
        summary_path = self.output_dir / service.lower() / 'io_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


def fetch_from_api_list(fetcher: ApiMetaFetcher, api_list_file: str):
    """
    从 API 列表文件中读取需要抓取的 API，实现精准抓取
    
    Args:
        fetcher: ApiMetaFetcher 实例
        api_list_file: API 列表文件路径 (JSON 格式)
    """
    with open(api_list_file, 'r', encoding='utf-8') as f:
        api_list = json.load(f)
    
    results = {}
    
    for service, apis in api_list.items():
        # 跳过元数据字段
        if service.startswith('_'):
            continue
            
        logger.info(f"Processing service: {service} ({len(apis)} APIs)")
        
        # 获取服务信息
        service_info = fetcher.get_service_info(service)
        if not service_info:
            logger.warning(f"Service not found: {service}")
            continue
        
        service_code = service_info.get(CODE)
        version = service_info.get(DEFAULT_VERSION)
        
        # 加载已有数据
        existing_data = fetcher.load_service_meta(service_code) or {
            'service': service_code,
            'name': service_info.get('name', ''),
            'version': version,
            'style': service_info.get('style', 'RPC'),
            'apis': {},
            'api_io_summary': {},
        }
        
        # 计算需要抓取的 API
        existing_apis = set(existing_data.get('apis', {}).keys())
        apis_to_fetch = [a for a in apis if a not in existing_apis]
        
        if not apis_to_fetch:
            logger.info(f"  All {len(apis)} APIs already fetched, skipping")
            results[service] = existing_data
            continue
        
        logger.info(f"  Need to fetch {len(apis_to_fetch)} new APIs: {apis_to_fetch}")
        
        success_count = 0
        fail_count = 0
        
        for api_name in apis_to_fetch:
            try:
                api_info = fetcher.get_api_info(service_code, version, api_name)
                existing_data['apis'][api_name] = api_info
                existing_data['api_io_summary'][api_name] = fetcher.extract_api_io_info(api_info)
                success_count += 1
                time.sleep(0.1)  # 请求间隔
            except Exception as e:
                fail_count += 1
                logger.warning(f"  Failed to fetch API {api_name}: {e}")
        
        # 保存
        fetcher._save_service_data(service_code, existing_data)
        results[service] = existing_data
        
        logger.info(f"  Complete: {success_count} success, {fail_count} failed")
    
    return results


def main():
    """命令行入口"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Fetch Aliyun API metadata for Aliyun-Gym simulator'
    )
    parser.add_argument(
        '-s', '--services',
        nargs='+',
        default=None,
        help=f'Services to fetch (default: {SUPPORTED_SERVICES})'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory (default: src/aliyun_gym/knowledge/api_docs)'
    )
    parser.add_argument(
        '-c', '--config',
        default=None,
        help='Config file path (optional, for future auth scenarios)'
    )
    parser.add_argument(
        '-l', '--api-list',
        default=None,
        help='API list JSON file (e.g., implemented_apis.json) for precise fetching'
    )
    
    args = parser.parse_args()
    
    fetcher = ApiMetaFetcher(
        output_dir=args.output,
        config_file=args.config,
        services=args.services or SUPPORTED_SERVICES
    )
    
    # 使用 API 列表文件精准抓取
    if args.api_list:
        print(f"Fetching from API list: {args.api_list}")
        results = fetch_from_api_list(fetcher, args.api_list)
    else:
        print(f"Fetching API metadata for services: {fetcher.services}")
        results = fetcher.fetch_all_services()
    
    print("\n=== Fetch Summary ===")
    for service, data in results.items():
        if data:
            api_count = len(data.get('apis', {}))
            print(f"  {service}: {api_count} APIs")
        else:
            print(f"  {service}: Failed")
    
    print(f"\nOutput directory: {fetcher.output_dir}")


if __name__ == '__main__':
    main()
