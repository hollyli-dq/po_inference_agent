import json
import os
import logging
import re
from typing import Dict, Any, Optional, List, Union
from execution.aliyun_gym.core.custom_validators import validate_custom_rules

logger = logging.getLogger(__name__)

class ApiValidator:
    """
    Validates API requests against the knowledge base (meta-data).
    """
    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self._meta_cache: Dict[str, Any] = {}
        
        # Mapping from product name (upper/camel) to folder name in api_docs
        self.product_folder_map = {
            "ecs": "ecs",
            "vpc": "vpc",
            "rds": "rds",
            "slb": "slb",
            "redis": "r-kvstore",
            "cms": "cms",
            "oos": "oos",
            "oss": "oss"
        }

    def _load_meta(self, product: str) -> Optional[Dict[str, Any]]:
        """Load full_meta.json for the given product."""
        product_key = product.lower()
        if product_key in self._meta_cache:
            return self._meta_cache[product_key]
            
        folder_name = self.product_folder_map.get(product_key, product_key)
        
        meta_path = os.path.join(self.kb_path, "api_docs", folder_name, "full_meta.json")
        # logger.warning(f"Loading meta from {meta_path}")
        if not os.path.exists(meta_path):
            logger.warning(f"Meta data not found for product {product} at {meta_path}")
            return None
            
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._meta_cache[product_key] = data
                return data
        except Exception as e:
            logger.error(f"Failed to load meta data for {product}: {e}")
            return None

    def validate(self, action: str, query: Dict[str, Any], product: str) -> Optional[Dict[str, Any]]:
        """
        Validates the request parameters.
        
        Args:
            action: API Action name.
            query: Dictionary of parameters.
            product: Product name (e.g., 'ECS', 'VPC').
            
        Returns:
            None if valid.
            Dict with 'Code' and 'Message' if invalid.
        """
        if not product:
            return None
            
        meta = self._load_meta(product)
        if not meta:
            # logger.warning(f"No meta found for product {product}")
            return None
            
        # Meta structure: "apis" -> { "ActionName": { "parameters": [ ... ] } }
        apis = meta.get("apis", {})
        api_def = apis.get(action)
        
        if not api_def:
            # logger.warning(f"Action {action} not found in meta for {product}")
            return None
            
        parameters = api_def.get("parameters", [])
        # logger.warning(f"Validating {action} with {len(parameters)} params")
        
        # Create a map for faster lookup if needed, but iteration is fine for small N
        for param_def in parameters:
            name = param_def.get("name")
            schema = param_def.get("schema", {})
            required = schema.get("required", False)
            p_type = schema.get("type", "string")
            enum_values = schema.get("enum")
            
            # Handle flattened parameters (e.g. lists) - simplified for now.
            # If name is "DataDisk", but query has "DataDisk.1.Size", we skip "DataDisk" check.
            # We only check keys that are expected to be present exactly as named.
            
            if name not in query:
                if required:
                    # Check if it's a complex object (array/repeatList) that might be passed as flat keys
                    # If so, we can't easily check 'missing' without parsing all keys.
                    # Assumption: If type is array, we skip 'required' check on the list name itself
                    # unless we see it missing AND no keys start with name + "."
                    if p_type == "array":
                        has_items = any(k.startswith(f"{name}.") for k in query.keys())
                        if not has_items:
                             return {
                                "Code": "MissingParameter",
                                "Message": f"The parameter {name} is mandatory."
                            }
                    else:
                        return {
                            "Code": "MissingParameter",
                            "Message": f"The parameter {name} is mandatory."
                        }
                continue
                
            value = query[name]
            
            # Validate Type
            if not self._validate_type(value, p_type):
                return {
                    "Code": "InvalidParameter",
                    "Message": f"The parameter {name} value '{value}' is not valid type {p_type}."
                }

            # Validate Numeric Constraints (minimum, maximum)
            if p_type in ["integer", "number"]:
                try:
                    num_val = float(value)
                    if "minimum" in schema:
                        min_val = float(schema["minimum"])
                        if num_val < min_val:
                            return {
                                "Code": "InvalidParameter",
                                "Message": f"The parameter {name} ({value}) is less than the minimum value {min_val}."
                            }
                    if "maximum" in schema:
                        max_val = float(schema["maximum"])
                        if num_val > max_val:
                            return {
                                "Code": "InvalidParameter",
                                "Message": f"The parameter {name} ({value}) is greater than the maximum value {max_val}."
                            }
                except (ValueError, TypeError):
                    pass # Handled by type check

            # Validate String Constraints (minLength, maxLength, pattern)
            if p_type == "string":
                val_str = str(value)
                if "minLength" in schema:
                    min_len = int(schema["minLength"])
                    if len(val_str) < min_len:
                        return {
                            "Code": "InvalidParameter",
                            "Message": f"The parameter {name} length ({len(val_str)}) is less than minimum length {min_len}."
                        }
                if "maxLength" in schema:
                    max_len = int(schema["maxLength"])
                    if len(val_str) > max_len:
                        return {
                            "Code": "InvalidParameter",
                            "Message": f"The parameter {name} length ({len(val_str)}) is greater than maximum length {max_len}."
                        }
                if "pattern" in schema:
                    pattern = schema["pattern"]
                    try:
                        if not re.match(pattern, val_str):
                            return {
                                "Code": "InvalidParameter",
                                "Message": f"The parameter {name} ({value}) does not match the required pattern."
                            }
                    except re.error:
                        logger.warning(f"Invalid regex pattern in meta for {name}: {pattern}")

            # Validate Enum
            if enum_values:
                # Convert value to string for comparison if enum contains strings
                # Enum values in JSON are usually strings or numbers.
                if value not in enum_values:
                    # Try string matching if value is not string
                    if str(value) not in [str(e) for e in enum_values]:
                         return {
                            "Code": "InvalidParameter",
                            "Message": f"The specified parameter {name} is not valid. Allowed: {enum_values}"
                        }
                        
        # 3. Custom Logic Validation (Mutex, Dependencies, etc.)
        custom_error = validate_custom_rules(action, query)
        if custom_error:
            return custom_error

        return None

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        if expected_type == "string":
            return True # Everything can be a string
        elif expected_type == "integer":
            try:
                int(value)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == "boolean":
            if isinstance(value, bool):
                return True
            if str(value).lower() in ["true", "false"]:
                return True
            return False
        # For array/object, if the value IS present as a key (not flattened), check type
        elif expected_type == "array":
            return isinstance(value, (list, tuple))
        
        return True
