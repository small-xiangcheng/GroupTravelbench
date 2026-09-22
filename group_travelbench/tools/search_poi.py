"""
POI search tool for GroupTravelbench.
"""
import json
from typing import Optional
from ..core.tools import SandboxBaseTool
from .get_adcode import validate_city_name


class SearchPoi(SandboxBaseTool):
    """POI search tool with sandbox caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        input_schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "description": "搜索关键词，如\"酒店\"、\"火锅\"、\"景点\"。仅传分类/名称词。",
                    "type": "string"
                },
                "city": {
                    "description": "城市名，如\"北京\"、\"上海\"。",
                    "type": "string"
                },
                "region": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "区/县级行政区划名，如\"朝阳区\"、\"浦东新区\"。仅支持标准行政区划，不可带路名/商圈。"
                },
                "location": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "附近搜索中心点，坐标或地名。"
                },
                "radius_km": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "搜索半径(km)，默认5。"
                },
                "price_min": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "最低价格（元）。建议搭配 region 缩小范围以获得更好的筛选效果。"
                },
                "price_max": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "最高价格（元）。建议搭配 region 缩小范围以获得更好的筛选效果。"
                },
                "sort_by": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": "排序：relevance/distance/rating/price_low/price_high。"
                },
                "page": {
                    "default": 1,
                    "description": "页码。",
                    "type": "integer"
                },
                "page_size": {
                    "default": 10,
                    "description": "每页条数，最大20。",
                    "type": "integer"
                }
            },
            "required": [
                "keywords",
                "city"
            ]
        }

        super().__init__(
            name="search_poi",
            description='搜索POI（景点/酒店/餐厅等）。找地方吃住玩时使用。',
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

        # Reject empty keywords
        if "keywords" in params_dict:
            if not params_dict["keywords"] or not params_dict["keywords"].strip():
                return json.dumps({"error": "参数 'keywords' 不能为空"}, ensure_ascii=False)

        # city is required and must be a valid city name
        if "city" in params_dict:
            if not params_dict["city"] or not params_dict["city"].strip():
                return json.dumps({"error": "参数 'city' 不能为空"}, ensure_ascii=False)
            if not validate_city_name(params_dict["city"]):
                return json.dumps({"error": f"参数 'city' 不是合法的城市/行政区名称: '{params_dict['city']}'"}, ensure_ascii=False)

        # Validate sort_by
        if "sort_by" in params_dict and params_dict["sort_by"] is not None:
            valid_sort_values = ["relevance", "distance", "rating", "price_low", "price_high"]
            if params_dict["sort_by"] not in valid_sort_values:
                return json.dumps({"error": f"参数 'sort_by' 值错误，应为 {valid_sort_values} 之一，实际值为: '{params_dict['sort_by']}'"}, ensure_ascii=False)

        # Validate page_size
        if "page_size" in params_dict:
            page_size = params_dict["page_size"]
            if page_size < 1 or page_size > 20:
                return json.dumps({"error": f"参数 'page_size' 超出范围，应在 1-20 之间，实际值为: {page_size}"}, ensure_ascii=False)

        # Validate page
        if "page" in params_dict:
            page = params_dict["page"]
            if page < 1:
                return json.dumps({"error": f"参数 'page' 最小值为1，实际值为: {page}"}, ensure_ascii=False)

        # Validate radius_km
        if "radius_km" in params_dict and params_dict["radius_km"] is not None:
            radius_km = params_dict["radius_km"]
            if radius_km <= 0:
                return json.dumps({"error": f"参数 'radius_km' 必须大于0，实际值为: {radius_km}"}, ensure_ascii=False)

        # Validate price_min / price_max
        if "price_min" in params_dict and params_dict["price_min"] is not None:
            if params_dict["price_min"] < 0:
                return json.dumps({"error": f"参数 'price_min' 不能为负数，实际值为: {params_dict['price_min']}"}, ensure_ascii=False)

        if "price_max" in params_dict and params_dict["price_max"] is not None:
            if params_dict["price_max"] < 0:
                return json.dumps({"error": f"参数 'price_max' 不能为负数，实际值为: {params_dict['price_max']}"}, ensure_ascii=False)

        if ("price_min" in params_dict and params_dict.get("price_min") is not None and
                "price_max" in params_dict and params_dict.get("price_max") is not None):
            if params_dict["price_min"] > params_dict["price_max"]:
                return json.dumps({"error": f"参数 'price_min' 不能大于 'price_max'"}, ensure_ascii=False)

        # Validate location when it looks like a coordinate
        if "location" in params_dict and params_dict["location"] is not None:
            location = params_dict["location"].strip()
            if not location:
                return json.dumps({"error": "参数 'location' 不能为空"}, ensure_ascii=False)
            # Treat the value as a coordinate when the first three characters are digits
            if len(location) >= 3 and location[:3].isdigit():
                location_error = self._validate_location_coordinate(location)
                if location_error:
                    return location_error

        return ''

    def _validate_location_coordinate(self, location: str) -> str:
        """Validate a ``location`` coordinate (lon,lat); only a comma-separated pair is accepted."""
        parts = location.split(",")

        if len(parts) != 2:
            return json.dumps({"error": f"location 坐标格式错误，应为 '经度,纬度' 格式（逗号分隔），实际值为: '{location}'"}, ensure_ascii=False)

        try:
            lon = float(parts[0].strip())
            lat = float(parts[1].strip())
        except ValueError:
            return json.dumps({"error": f"location 中的经纬度必须是有效数字，实际值为: '{location}'"}, ensure_ascii=False)

        if not (-180 <= lon <= 180):
            return json.dumps({"error": f"location 经度(lon)超出范围，应在 -180 到 180 之间，实际值为: {lon}"}, ensure_ascii=False)

        if not (-90 <= lat <= 90):
            return json.dumps({"error": f"location 纬度(lat)超出范围，应在 -90 到 90 之间，实际值为: {lat}"}, ensure_ascii=False)

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


search_poi = SearchPoi()
