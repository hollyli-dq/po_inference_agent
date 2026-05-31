import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages static resources loaded from knowledge base (e.g., Regions, Zones, InstanceTypes).
    Singleton-like access to read-only data.
    """
    _instance = None
    
    def __init__(self):
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._base_path = self._resolve_resource_path()
        self._load_all()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_resource_path(self) -> str:
        """Find the static_resources directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # core -> aliyun_gym -> knowledge -> static_resources
        return os.path.join(os.path.dirname(current_dir), "knowledge", "static_resources")

    def _load_all(self):
        """Load all known JSON files."""
        files = {
            "regions": "regions.json",
            "zones": "zones.json",
            "instance_types": "instance_types.json",
            "images": "images.json"
        }
        
        for key, filename in files.items():
            path = os.path.join(self._base_path, filename)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._cache[key] = json.load(f)
                    # logger.info(f"Loaded {key} from {path}")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
                    self._cache[key] = []
            else:
                logger.warning(f"Resource file not found: {path}")
                self._cache[key] = []

    def get_regions(self) -> List[Dict[str, Any]]:
        return self._cache.get("regions", [])

    def get_zones(self, region_id: Optional[str] = None) -> List[Dict[str, Any]]:
        zones = self._cache.get("zones", [])
        if region_id:
            filtered = []
            for z in zones:
                rid = z.get("RegionId")
                if not rid and "ZoneId" in z:
                    # Infer from ZoneId (e.g. cn-hangzhou-b -> cn-hangzhou)
                    parts = z["ZoneId"].rsplit("-", 1)
                    if len(parts) > 1:
                        rid = parts[0]
                        z["RegionId"] = rid
                
                if rid == region_id:
                    filtered.append(z)
            return filtered
        return zones

    def get_instance_types(self, family: Optional[str] = None) -> List[Dict[str, Any]]:
        types = self._cache.get("instance_types", [])
        if family:
            return [t for t in types if t.get("InstanceTypeFamily") == family]
        return types

    def get_images(self, image_id: Optional[str] = None, 
                   image_name: Optional[str] = None,
                   os_type: Optional[str] = None) -> List[Dict[str, Any]]:
        images = self._cache.get("images", [])
        result = images
        
        if image_id:
            result = [i for i in result if i.get("ImageId") == image_id]
        
        if image_name:
            # Simple partial match (case-insensitive)
            target = image_name.lower()
            result = [i for i in result if target in i.get("ImageName", "").lower()]
            
        if os_type:
             result = [i for i in result if i.get("OSType") == os_type]
             
        return result
