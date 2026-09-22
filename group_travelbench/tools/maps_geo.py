"""
Geocoding tool for GroupTravelbench.
"""
import json
from typing import Optional
from ..core.tools import SandboxBaseTool


class MapsGeo(SandboxBaseTool):
    """Geocoding tool with sandbox caching support."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "address": {
                    "description": "待解析的结构化地址",
                    "type": "string"
                },
                "city": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "指定查询城市",
                }
            },
            "required": [
                "address"
            ]
        }
    
        super().__init__(
            name="maps_geo",
            description='地理编码：地址转经纬度。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_parameters(self, params: str) -> str:
        """
        Validate parameters.

        Checks:
        1. Each name exists in ``input_schema``
        2. Each value matches the schema type
        3. Every required parameter is present
        """
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False)
        
        if not isinstance(params_dict, dict):
            return json.dumps({"error": "参数必须是一个JSON对象"}, ensure_ascii=False)
        
        schema_properties = self.input_schema.get("properties", {})
        required_params = self.input_schema.get("required", [])
        
        # Reject unknown parameters
        for param_name in params_dict.keys():
            if param_name not in schema_properties:
                return json.dumps({"error": f"未知参数: '{param_name}'，允许的参数为: {list(schema_properties.keys())}"}, ensure_ascii=False)
        
        # Check that all required parameters are present
        for required_param in required_params:
            if required_param not in params_dict:
                return json.dumps({"error": f"缺少必填参数: '{required_param}'"}, ensure_ascii=False)
        
        # Check parameter types
        for param_name, param_value in params_dict.items():
            if param_name not in schema_properties:
                continue
            
            param_schema = schema_properties[param_name]
            expected_type = param_schema.get("type")
            
            # Handle anyOf types (e.g. city may be string or null)
            if "anyOf" in param_schema:
                valid_types = []
                allows_null = False
                for type_option in param_schema["anyOf"]:
                    if type_option.get("type") == "null":
                        allows_null = True
                    else:
                        valid_types.append(type_option.get("type"))
                
                if param_value is None:
                    if not allows_null:
                        return json.dumps({"error": f"参数 '{param_name}' 不允许为 null"}, ensure_ascii=False)
                else:
                    type_valid = False
                    for valid_type in valid_types:
                        if self._check_type(param_value, valid_type):
                            type_valid = True
                            break
                    if not type_valid:
                        return json.dumps({"error": f"参数 '{param_name}' 类型错误，期望类型为: {valid_types}，实际值为: {type(param_value).__name__}"}, ensure_ascii=False)
            elif expected_type and not self._check_type(param_value, expected_type):
                return json.dumps({"error": f"参数 '{param_name}' 类型错误，期望类型为: {expected_type}，实际值为: {type(param_value).__name__}"}, ensure_ascii=False)
        
        # Reject an empty address
        if "address" in params_dict:
            address = params_dict["address"]
            if not address or not address.strip():
                return json.dumps({"error": "address 参数不能为空"}, ensure_ascii=False)
        
        return ''
    
    def _check_type(self, value, expected_type: str) -> bool:
        """Check whether a value matches the expected JSON-schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True
        return isinstance(value, expected_python_type)


maps_geo = MapsGeo()
