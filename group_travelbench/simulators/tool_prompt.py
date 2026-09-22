"""
Prompts for LLM-based tool simulator.
"""

# System prompt template for tool simulation
TOOL_SIMULATION_SYSTEM_PROMPT = """你是一个工具模拟器，需要模拟 {tool_name} 工具的真实返回结果。

工具定义：
名称：{tool_name}
描述：{tool_description}
参数定义：{tool_parameters}

任务要求：
1. 根据提供的真实示例，理解工具的输出格式和内容特点
2. 基于输入参数生成合理的模拟结果
3. 确保输出格式与示例一致
4. 生成的内容要符合工具的业务逻辑和真实场景
5. 直接返回模拟结果，不要添加任何额外说明、解释或 markdown 格式
6. 不要返回 JSON 格式的包装，直接返回工具本身应该返回的内容
"""

# Template for examples section (when examples are available)
EXAMPLES_SECTION_TEMPLATE = """
以下是 {num_examples} 个真实调用示例供参考：
"""

# Template for single example
SINGLE_EXAMPLE_TEMPLATE = """
示例 {index}：
输入参数：{params}
输出结果：{result}
"""

# Template for no examples case
NO_EXAMPLES_TEMPLATE = """
注意：没有找到 {tool_name} 的历史示例，请根据工具定义和参数生成合理的模拟结果。
"""

# User prompt template for tool simulation
TOOL_SIMULATION_USER_PROMPT = """请为以下参数生成 {tool_name} 工具的模拟返回结果：

参数：{params_json}

要求：
1. 注意参考真实调用示例，部分信息可能直接来源于给你的示例，相似的调用参数要生成相似的模拟结果
2. 内容要符合工具的业务逻辑和真实场景
3. 如果是列表类型的结果，生成若干个合理的条目
4. 数值要在合理范围内
5. 时间、日期等信息要符合参数中的约束
6. 直接返回结果内容，不要添加任何解释或格式包装
"""

TOOLS_SCHEMAS = [{'type': 'function', 'function': {'name': 'maps_geo', 'description': '地理编码：地址转经纬度。', 'parameters': {'type': 'object', 'properties': {'address': {'description': '待解析的结构化地址', 'type': 'string'}, 'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '指定查询城市'}}, 'required': ['address']}}}, {'type': 'function', 'function': {'name': 'maps_weather', 'description': '天气查询：根据城市名和日期查询，支持查询未来5天内的天气预报。超出5天范围的部分将无法提供准确预报。示例日期格式：2023-03-01', 'parameters': {'type': 'object', 'properties': {'location': {'description': '城市名称', 'type': 'string'}, 'date': {'description': '日期，格式 YYYY-MM-DD', 'type': 'string'}, 'days': {'default': 0, 'description': '额外查询天数。返回 date 至 date+days 的天气，总共 days+1 天；仅支持查询未来5天内的天气预报，超出范围的日期将无法提供准确预报数据。', 'type': 'integer'}}, 'required': ['location', 'date']}}}, {'type': 'function', 'function': {'name': 'travel_search_flights', 'description': '国内航班搜索。建议用时间/排序/机场参数缩小范围。', 'parameters': {'type': 'object', 'properties': {'origin': {'description': '出发城市，如"北京市"。', 'type': 'string'}, 'destination': {'description': '到达城市，如"上海市"。', 'type': 'string'}, 'date': {'description': '日期，YYYY-MM-DD。每次查一天。', 'type': 'string'}, 'depart_time_start': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '最早起飞时间，HH:MM。'}, 'depart_time_end': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '最晚起飞时间，HH:MM。'}, 'sort_by': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '排序：depart_time(默认)/price/duration。'}, 'airport': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '机场关键词筛选，如"虹桥"、"大兴"。'}}, 'required': ['origin', 'destination', 'date']}}}, {'type': 'function', 'function': {'name': 'travel_search_trains', 'description': '国内火车/高铁搜索。建议用车次类型/时间/排序参数缩小范围。', 'parameters': {'type': 'object', 'properties': {'origin': {'description': '出发城市名，如"广州市"、"北京市"。', 'type': 'string'}, 'destination': {'description': '到达城市名，如"上海市"、"重庆市"。', 'type': 'string'}, 'date': {'description': '日期，YYYY-MM-DD。每次查一天。', 'type': 'string'}, 'depart_time_start': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '最早发车时间，HH:MM。'}, 'depart_time_end': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '最晚发车时间，HH:MM。'}, 'sort_by': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '排序：depart_time(默认，从早到晚)/depart_time_desc(从晚到早)/duration/price。G/D/C优先。'}, 'train_type_filter': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '车次筛选：high_speed/high_speed_and_emu/regular。'}}, 'required': ['origin', 'destination', 'date']}}}, {'type': 'function', 'function': {'name': 'web_search', 'description': '通用 Web 搜索。', 'parameters': {'type': 'object', 'properties': {'query': {'description': 'Web 搜索关键词', 'title': 'Query', 'type': 'string'}}, 'required': ['query']}}}, {'type': 'function', 'function': {'name': 'search_poi', 'description': '搜索POI（景点/酒店/餐厅等）。找地方吃住玩时使用。', 'parameters': {'type': 'object', 'properties': {'keywords': {'description': '搜索关键词，如"酒店"、"火锅"、"景点"。仅传分类/名称词。', 'type': 'string'}, 'city': {'description': '城市名，如"北京"、"上海"。', 'type': 'string'}, 'region': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '区/县级行政区划名，如"朝阳区"、"浦东新区"。仅支持标准行政区划，不可带路名/商圈。'}, 'location': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '附近搜索中心点，坐标或地名。'}, 'radius_km': {'anyOf': [{'type': 'number'}, {'type': 'null'}], 'default': None, 'description': '搜索半径(km)，默认5。'}, 'price_min': {'anyOf': [{'type': 'number'}, {'type': 'null'}], 'default': None, 'description': '最低价格（元）。建议搭配 region 缩小范围以获得更好的筛选效果。'}, 'price_max': {'anyOf': [{'type': 'number'}, {'type': 'null'}], 'default': None, 'description': '最高价格（元）。建议搭配 region 缩小范围以获得更好的筛选效果。'}, 'sort_by': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '排序：relevance/distance/rating/price_low/price_high。'}, 'page': {'default': 1, 'description': '页码。', 'type': 'integer'}, 'page_size': {'default': 10, 'description': '每页条数，最大20。', 'type': 'integer'}}, 'required': ['keywords', 'city']}}}, {'type': 'function', 'function': {'name': 'get_poi_detail', 'description': '查POI详情（电话/营业时间/设施/政策等）。需要确认具体信息时使用，传search_poi返回的ID。', 'parameters': {'type': 'object', 'properties': {'id': {'description': 'POI ID，来自 search_poi 结果。', 'type': 'string'}}, 'required': ['id']}}}, {'type': 'function', 'function': {'name': 'plan_route', 'description': "路径规划（驾车/步行/骑行/公交）。问'怎么去XX'时使用。支持坐标或地名。", 'parameters': {'type': 'object', 'properties': {'origin': {'description': '起点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'destination': {'description': '终点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'mode': {'default': 'transit', 'description': '交通方式：transit(默认)/driving/walking/bicycling。', 'type': 'string'}, 'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '城市名。transit必传；地名输入时建议传入。'}, 'date': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '出发日期YYYY-MM-DD，仅transit有效。'}, 'time': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '出发时间HH:MM，仅transit有效。'}}, 'required': ['origin', 'destination']}}}, {'type': 'function', 'function': {'name': 'compare_routes', 'description': '同时对比驾车/公交/步行/骑行四种方式的时间、距离、费用。传 city 可获取公交方案。', 'parameters': {'type': 'object', 'properties': {'origin': {'description': '起点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'destination': {'description': '终点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '城市名，传入可获取公交方案。'}}, 'required': ['origin', 'destination']}}}, {'type': 'function', 'function': {'name': 'search_along_route', 'description': '沿路线搜索POI（加油站/餐厅/服务区等）。路途中找吃的、加油时使用。', 'parameters': {'type': 'object', 'properties': {'origin': {'description': '起点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'destination': {'description': '终点，坐标(经度,纬度)或地名。', 'type': 'string'}, 'keywords': {'description': '搜索关键词，如"加油站"、"餐厅"、"服务区"。', 'type': 'string'}, 'mode': {'default': 'driving', 'description': '交通方式：driving(默认)/walking/bicycling。', 'type': 'string'}, 'city': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'description': '城市名，地名输入时建议传入。'}, 'route_range': {'default': 500, 'description': '沿途搜索宽度(米)，默认500，最大3000。', 'type': 'integer'}, 'route_index': {'default': 1, 'description': '路线编号(从1开始)，默认1。', 'type': 'integer'}, 'poi_limit': {'default': 20, 'description': '返回POI上限，默认20，最大50。', 'type': 'integer'}}, 'required': ['origin', 'destination', 'keywords']}}}]
