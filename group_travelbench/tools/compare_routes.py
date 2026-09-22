"""
Route comparison tool for GroupTravelbench.
"""
import json
from typing import Optional
from ..core.tools import SandboxBaseTool
from .get_adcode import validate_city_name


class CompareRoutes(SandboxBaseTool):
    """Route comparison tool with sandbox caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "origin": {
                    "description": "起点，坐标(经度,纬度)或地名。",
                    "type": "string"
                },
                "destination": {
                    "description": "终点，坐标(经度,纬度)或地名。",
                    "type": "string"
                },
                "city": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "城市名，传入可获取公交方案。"
                }
            },
            "required": [
                "origin",
                "destination"
            ]
        }

        super().__init__(
            name="compare_routes",
            description='同时对比驾车/公交/步行/骑行四种方式的时间、距离、费用。传 city 可获取公交方案。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_parameters(self, params: str) -> str:
        """Validate parameters."""
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

            # Handle anyOf types
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

        # Reject an empty origin; treat coordinates and place names differently
        if "origin" in params_dict:
            origin = params_dict["origin"]
            if not origin or not origin.strip():
                return json.dumps({"error": "参数 'origin' 不能为空"}, ensure_ascii=False)
            origin = origin.strip()
            if len(origin) >= 3 and origin[:3].isdigit():
                coord_error = self._validate_coordinate(origin, "origin")
                if coord_error:
                    return coord_error

        # Reject an empty destination; treat coordinates and place names differently
        if "destination" in params_dict:
            destination = params_dict["destination"]
            if not destination or not destination.strip():
                return json.dumps({"error": "参数 'destination' 不能为空"}, ensure_ascii=False)
            destination = destination.strip()
            if len(destination) >= 3 and destination[:3].isdigit():
                coord_error = self._validate_coordinate(destination, "destination")
                if coord_error:
                    return coord_error

        # Validate the city name
        if "city" in params_dict and params_dict["city"] is not None:
            city = params_dict["city"]
            if city.strip() and not validate_city_name(city):
                return json.dumps({"error": f"参数 'city' 不是合法的城市/行政区名称: '{city}'"}, ensure_ascii=False)

        return ''

    def _validate_coordinate(self, location: str, param_name: str) -> str:
        """Validate a lon,lat coordinate; only a comma-separated pair is accepted."""
        parts = location.split(",")
        if len(parts) != 2:
            return json.dumps({"error": f"参数 '{param_name}' 坐标格式错误，应为 '经度,纬度' 格式（逗号分隔），实际值为: '{location}'"}, ensure_ascii=False)
        try:
            lon = float(parts[0].strip())
            lat = float(parts[1].strip())
        except ValueError:
            return json.dumps({"error": f"参数 '{param_name}' 中的经纬度必须是有效数字，实际值为: '{location}'"}, ensure_ascii=False)
        if not (-180 <= lon <= 180):
            return json.dumps({"error": f"参数 '{param_name}' 经度(lon)超出范围，应在 -180 到 180 之间，实际值为: {lon}"}, ensure_ascii=False)
        if not (-90 <= lat <= 90):
            return json.dumps({"error": f"参数 '{param_name}' 纬度(lat)超出范围，应在 -90 到 90 之间，实际值为: {lat}"}, ensure_ascii=False)
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


compare_routes = CompareRoutes()
