import json
from typing import Dict, Any, Optional

class StateStore:
    """
    In-memory database for Aliyun-Gym resources.
    Stores resources as a dictionary: {resource_id: resource_data}
    """
    def __init__(self):
        self._resources: Dict[str, Any] = {}

    def put(self, resource_id: str, data: Dict[str, Any]) -> None:
        """Create or update a resource."""
        self._resources[resource_id] = data

    def get(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a resource by ID."""
        return self._resources.get(resource_id)

    def exists(self, resource_id: str) -> bool:
        """Check if a resource exists."""
        return resource_id in self._resources

    def delete(self, resource_id: str) -> None:
        """Delete a resource."""
        if resource_id in self._resources:
            del self._resources[resource_id]

    def list_by_type(self, resource_type: str) -> Dict[str, Any]:
        """List all resources of a specific type."""
        return {
            rid: rdata 
            for rid, rdata in self._resources.items() 
            if rdata.get("type") == resource_type
        }

    def dump(self) -> Dict[str, Any]:
        """Dump the entire state for debugging."""
        return self._resources

    def clear(self) -> None:
        """Clear all resources."""
        self._resources.clear()
