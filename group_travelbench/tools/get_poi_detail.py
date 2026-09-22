"""
POI detail tool for GroupTravelbench.
"""
import json
from typing import Optional
from ..core.tools import SandboxBaseTool


class GetPoiDetail(SandboxBaseTool):
    """POI detail tool with sandbox caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "id": {
                    "description": "POI ID，来自 search_poi 结果。",
                    "type": "string"
                }
            },
            "required": [
                "id"
            ]
        }

        super().__init__(
            name="get_poi_detail",
            description='查POI详情（电话/营业时间/设施/政策等）。需要确认具体信息时使用，传search_poi返回的ID。',
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

            if expected_type and not self._check_type(param_value, expected_type):
                return json.dumps({"error": f"参数 '{param_name}' 类型错误，期望类型为: {expected_type}，实际值为: {type(param_value).__name__}"}, ensure_ascii=False)

        # Reject an empty id
        if "id" in params_dict:
            poi_id = params_dict["id"]
            if not poi_id or not poi_id.strip():
                return json.dumps({"error": "参数 'id' 不能为空"}, ensure_ascii=False)

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


get_poi_detail = GetPoiDetail()
