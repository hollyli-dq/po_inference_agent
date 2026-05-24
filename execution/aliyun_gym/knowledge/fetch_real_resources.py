import json
import os
import sys
from pathlib import Path
from configparser import ConfigParser

from alibabacloud_ecs20140526.client import Client as EcsClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ecs20140526 import models as ecs_models

def get_config():
    """Load configuration from aliyun_config.ini"""
    current_dir = Path(__file__).parent
    config_path = current_dir / 'aliyun_config.ini'
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please copy aliyun_config.ini.template to aliyun_config.ini and fill in your AK/SK.")
        sys.exit(1)
        
    config = ConfigParser()
    config.read(config_path)
    
    if 'aliyun' not in config:
        print("Section [aliyun] not found in config")
        sys.exit(1)
        
    return config['aliyun']

def create_client(access_key_id, access_key_secret, region_id):
    """Create ECS Client"""
    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        read_timeout=20000,  # 20 seconds
        connect_timeout=10000 # 10 seconds
    )
    config.endpoint = f'ecs.{region_id}.aliyuncs.com'
    return EcsClient(config)

def save_json(data, filename):
    """Save data to JSON file in static_resources directory"""
    current_dir = Path(__file__).parent
    output_dir = current_dir / 'static_resources'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {filename}")

def fetch_regions(client):
    print("Fetching Regions...")
    request = ecs_models.DescribeRegionsRequest()
    response = client.describe_regions(request)
    regions = [r.to_map() for r in response.body.regions.region]
    save_json(regions, 'regions.json')
    return regions

def fetch_zones(client, region_id):
    print(f"Fetching Zones for {region_id}...")
    # Switch endpoint to the specific region for DescribeZones just in case, 
    # though usually the client endpoint is fine if it matches.
    # Actually DescribeZones can be called from any endpoint if RegionId is passed, 
    # but let's stick to the client's region or update it.
    # The client was created with a specific region endpoint.
    
    request = ecs_models.DescribeZonesRequest(region_id=region_id)
    response = client.describe_zones(request)
    zones = [z.to_map() for z in response.body.zones.zone]
    # We might want to merge zones from multiple regions if we iterate, 
    # but for now let's just save what we get or maybe we should structure it differently?
    # The current static_resources/zones.json is a flat list of zones from multiple regions?
    # Let's check the existing structure if possible. 
    # But usually a flat list with RegionId field is fine.
    return zones

def fetch_instance_types(client):
    print("Fetching Instance Types...")
    request = ecs_models.DescribeInstanceTypesRequest()
    response = client.describe_instance_types(request)
    types = [t.to_map() for t in response.body.instance_types.instance_type]
    save_json(types, 'instance_types.json')

def fetch_images(client, region_id):
    print(f"Fetching Images for {region_id}...")
    # Fetch system images (Ubuntu, CentOS, Aliyun Linux)
    request = ecs_models.DescribeImagesRequest(
        region_id=region_id,
        image_owner_alias='system',
        page_size=100  # Limit to 100 to avoid too huge file
    )
    response = client.describe_images(request)
    images = [i.to_map() for i in response.body.images.image]
    save_json(images, 'images.json')

def main():
    config = get_config()
    ak = config.get('access_key_id')
    sk = config.get('access_key_secret')
    default_region = config.get('region_id', 'cn-hangzhou')
    
    if ak.startswith('YOUR') or sk.startswith('YOUR'):
        print("Error: Please fill in real AccessKey ID and Secret in aliyun_config.ini.template")
        sys.exit(1)
        
    client = create_client(ak, sk, default_region)
    
    # 1. Fetch Regions
    regions = fetch_regions(client)
    
    # 2. Fetch Zones (collect from a few key regions)
    target_regions = ['cn-hangzhou', 'cn-shanghai', 'cn-beijing']
    all_zones = []
    
    for rid in target_regions:
        try:
            # We need a client for the specific region or just pass RegionId
            # DescribeZones requires RegionId parameter.
            # Using the same client (endpoint ecs.cn-hangzhou...) usually works for other regions 
            # if we are just querying metadata, but for strictness let's recreate client or just pass param.
            # Aliyun ECS API usually routes correctly.
            zones = fetch_zones(client, rid)
            all_zones.extend(zones)
        except Exception as e:
            print(f"Failed to fetch zones for {rid}: {e}")
            
    save_json(all_zones, 'zones.json')
    
    # 3. Fetch Instance Types (Global or Region specific? usually region specific but types are mostly global definitions)
    # DescribeInstanceTypes is not region specific in some contexts, but let's see. 
    # Actually it often takes no region or valid in current region. 
    # We will use the default client.
    try:
        fetch_instance_types(client)
    except Exception as e:
        print(f"Failed to fetch instance types: {e}")
        
    # 4. Fetch Images (Region specific, but system images are often similar)
    # We will fetch for the default region
    try:
        fetch_images(client, default_region)
    except Exception as e:
        print(f"Failed to fetch images: {e}")

if __name__ == '__main__':
    main()
