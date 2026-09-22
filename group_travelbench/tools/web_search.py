"""
Web search tool for GroupTravelbench.
"""
import json
from typing import Optional
from ..core.tools import SandboxBaseTool


class WebSearch(SandboxBaseTool):
    """Web search tool with sandbox caching support."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "description": "Web 搜索关键词",
                    "title": "Query",
                    "type": "string"
                }
            },
            "required": [
                "query"
            ]
        }
    
        super().__init__(
            name="web_search",
            description='通用 Web 搜索。',
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

        # Reject an empty query
        if "query" in params_dict:
            if not params_dict["query"] or not params_dict["query"].strip():
                return json.dumps({"error": "参数 'query' 不能为空"}, ensure_ascii=False)

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


web_search = WebSearch()
