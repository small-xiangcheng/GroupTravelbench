"""
Weather query tool for GroupTravelbench.
"""
import json
import re
from datetime import datetime
from typing import Optional
from ..core.tools import SandboxBaseTool
from .get_adcode import validate_city_name

DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class MapsWeather(SandboxBaseTool):
    """Weather query tool with sandbox caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "location": {
                    "description": "城市名称",
                    "type": "string"
                },
                "date": {
                    "description": "日期，格式 YYYY-MM-DD",
                    "type": "string"
                },
                "days": {
                    "default": 0,
                    "description": "额外查询天数。返回 date 至 date+days 的天气，总共 days+1 天；仅支持查询未来5天内的天气预报，超出范围的日期将无法提供准确预报数据。",
                    "type": "integer"
                }
            },
            "required": [
                "location",
                "date"
            ]
        }

        super().__init__(
            name="maps_weather",
            description='天气查询：根据城市名和日期查询，支持查询未来5天内的天气预报。超出5天范围的部分将无法提供准确预报。示例日期格式：2023-03-01',
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

            if expected_type and not self._check_type(param_value, expected_type):
                return json.dumps({"error": f"参数 '{param_name}' 类型错误，期望类型为: {expected_type}，实际值为: {type(param_value).__name__}"}, ensure_ascii=False)

        # Reject an empty location
        if "location" in params_dict:
            location = params_dict["location"]
            if not location or not location.strip():
                return json.dumps({"error": "参数 'location' 不能为空"}, ensure_ascii=False)
            # The location must be a valid administrative name
            if not validate_city_name(location):
                return json.dumps({"error": f"参数 'location' 对应的城市/行政区 '{location}' 不存在或无法识别"}, ensure_ascii=False)

        # Validate the date format
        if "date" in params_dict:
            date_str = params_dict["date"]
            if not DATE_PATTERN.match(date_str):
                return json.dumps({"error": f"日期格式错误: \"{date_str}\"，应为 \"YYYY-MM-DD\""}, ensure_ascii=False)
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return json.dumps({"error": f"参数 date 不是有效的日期: \"{date_str}\""}, ensure_ascii=False)

        # Validate the days range
        if "days" in params_dict:
            days = params_dict["days"]
            if days < 0:
                return json.dumps({"error": "参数 'days' 最小值为0"}, ensure_ascii=False)
            if days > 4:
                return json.dumps({"error": f"参数 'days' 超出范围，仅支持查询未来5天内的天气预报（days 最大值为4，总共 days+1 天），实际值为: {days}"}, ensure_ascii=False)

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


maps_weather = MapsWeather()
