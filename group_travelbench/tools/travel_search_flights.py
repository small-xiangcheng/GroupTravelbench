"""
Flight search tool for GroupTravelbench.
"""
import json
import re
from datetime import datetime
from typing import Optional
from ..core.tools import SandboxBaseTool
from .get_adcode import validate_city_name

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
TIME_PATTERN = re.compile(r'^\d{2}:\d{2}$')

class TravelSearchFlights(SandboxBaseTool):
    """Flight search tool with sandbox caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "origin": {
                    "description": "出发城市，如\"北京市\"。",
                    "type": "string"
                },
                "destination": {
                    "description": "到达城市，如\"上海市\"。",
                    "type": "string"
                },
                "date": {
                    "description": "日期，YYYY-MM-DD。每次查一天。",
                    "type": "string"
                },
                "depart_time_start": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "最早起飞时间，HH:MM。"
                },
                "depart_time_end": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "最晚起飞时间，HH:MM。"
                },
                "sort_by": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "排序：depart_time(默认)/price/duration。"
                },
                "airport": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "机场关键词筛选，如\"虹桥\"、\"大兴\"。"
                }
            },
            "required": [
                "origin",
                "destination",
                "date"
            ]
        }

        super().__init__(
            name="travel_search_flights",
            description='国内航班搜索。建议用时间/排序/机场参数缩小范围。',
            input_schema=input_schema,
            cache_dir=cache_dir
        )

    def _validate_parameters(self, params: str) -> str:
        """Validate parameters."""
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"JSON格式无效: {str(e)}"}, ensure_ascii=False)

        if not isinstance(params_dict, dict):
            return json.dumps({"error": "参数必须是JSON对象"}, ensure_ascii=False)

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

        # origin is required and must be a valid city name
        if "origin" in params_dict:
            origin = params_dict["origin"]
            if not origin or not origin.strip():
                return json.dumps({"error": "参数 'origin' 不能为空"}, ensure_ascii=False)
            if not validate_city_name(origin):
                return json.dumps({"error": f"参数 'origin' 不是合法的城市/行政区名称: '{origin}'"}, ensure_ascii=False)

        # destination is required and must be a valid city name
        if "destination" in params_dict:
            destination = params_dict["destination"]
            if not destination or not destination.strip():
                return json.dumps({"error": "参数 'destination' 不能为空"}, ensure_ascii=False)
            if not validate_city_name(destination):
                return json.dumps({"error": f"参数 'destination' 不是合法的城市/行政区名称: '{destination}'"}, ensure_ascii=False)

        # Validate the date format
        if "date" in params_dict:
            date_str = params_dict["date"]
            if not DATE_PATTERN.match(date_str):
                return json.dumps({"error": f'日期格式错误: "{date_str}"，应为 "YYYY-MM-DD"'}, ensure_ascii=False)
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return json.dumps({"error": f'参数 date 不是有效的日期: "{date_str}"'}, ensure_ascii=False)

        # Validate depart_time_start
        if "depart_time_start" in params_dict and params_dict["depart_time_start"] is not None:
            time_error = self._validate_time_format(params_dict["depart_time_start"], "depart_time_start")
            if time_error:
                return time_error

        # Validate depart_time_end
        if "depart_time_end" in params_dict and params_dict["depart_time_end"] is not None:
            time_error = self._validate_time_format(params_dict["depart_time_end"], "depart_time_end")
            if time_error:
                return time_error

        # Require depart_time_start <= depart_time_end
        if (params_dict.get("depart_time_start") is not None and
                params_dict.get("depart_time_end") is not None):
            if params_dict["depart_time_start"] > params_dict["depart_time_end"]:
                return json.dumps({"error": "参数 'depart_time_start' 不能晚于 'depart_time_end'"}, ensure_ascii=False)

        # Validate sort_by
        if "sort_by" in params_dict and params_dict["sort_by"] is not None:
            valid_sort_values = ["depart_time", "price", "duration"]
            if params_dict["sort_by"] not in valid_sort_values:
                return json.dumps({"error": f"参数 'sort_by' 值错误，应为 {valid_sort_values} 之一，实际值为: '{params_dict['sort_by']}'"}, ensure_ascii=False)

        return ''

    def _validate_time_format(self, time_str: str, param_name: str) -> str:
        """Validate an HH:MM time string."""
        if not TIME_PATTERN.match(time_str):
            return json.dumps({"error": f"参数 '{param_name}' 格式错误，应为 'HH:MM' 格式，实际值为: '{time_str}'"}, ensure_ascii=False)
        try:
            hour, minute = time_str.split(":")
            hour, minute = int(hour), int(minute)
            if not (0 <= hour <= 23):
                return json.dumps({"error": f"参数 '{param_name}' 小时超出范围，应在 00-23 之间，实际值为: '{time_str}'"}, ensure_ascii=False)
            if not (0 <= minute <= 59):
                return json.dumps({"error": f"参数 '{param_name}' 分钟超出范围，应在 00-59 之间，实际值为: '{time_str}'"}, ensure_ascii=False)
        except ValueError:
            return json.dumps({"error": f"参数 '{param_name}' 不是有效的时间: '{time_str}'"}, ensure_ascii=False)
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


travel_search_flights = TravelSearchFlights()
