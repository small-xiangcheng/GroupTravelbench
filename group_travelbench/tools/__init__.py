"""
Tool modules for GroupTravelbench.
This module imports all available tool classes and handles their centralized registration.
"""

# Import sandbox_tool_registry for centralized registration
from ..core.tools import sandbox_tool_registry

from . import maps_geo
from . import maps_weather
from . import travel_search_flights
from . import travel_search_trains
from . import web_search
from . import search_poi
from . import get_poi_detail
from . import plan_route
from . import compare_routes
from . import search_along_route

# Import the tool list for runtime filtering
from .tool_list import TOOL_NAMES

# Re-export specific tool instances for direct access
from .maps_geo import maps_geo
from .maps_weather import maps_weather
from .travel_search_flights import travel_search_flights
from .travel_search_trains import travel_search_trains
from .web_search import web_search
from .search_poi import search_poi
from .get_poi_detail import get_poi_detail
from .plan_route import plan_route
from .compare_routes import compare_routes
from .search_along_route import search_along_route

# Centralized tool registration - register tools here.
#
# IMPORTANT: only tools whose `name` is listed in TOOL_NAMES get registered.
# Anything commented out / removed from tool_list.TOOL_NAMES is silently
# skipped here. This is the single source of truth for "which tools the
# Agent can see" — the Agent (`GroupTravelAgent`) and tool simulators all
# consume `sandbox_tool_registry.get_tools()`, which now reflects the
# whitelist directly. Without this filter, modifying tool_list.py only
# affected the `__main__` startup banner and had no effect on actual
# tool calling.
def register_all_tools():
    """Register only the whitelisted tools (per TOOL_NAMES) with the
    sandbox tool registry."""
    all_tools = [
        maps_geo,
        maps_weather,
        travel_search_flights,
        travel_search_trains,
        web_search,
        search_poi,
        get_poi_detail,
        plan_route,
        compare_routes,
        search_along_route,
    ]

    whitelist = set(TOOL_NAMES)
    for tool in all_tools:
        if tool.name not in whitelist:
            continue
        try:
            sandbox_tool_registry.register(tool)
        except ValueError as e:
            if "already registered" not in str(e):
                raise e  # Re-raise if it's not a duplicate registration error

# Automatically register all tools when this module is imported
register_all_tools()

# Define what gets exported with 'from .tools import *'
__all__ = [
    'TOOL_NAMES',
    'maps_geo',
    'maps_weather',
    'travel_search_flights',
    'travel_search_trains',
    'web_search',
    'search_poi',
    'get_poi_detail',
    'plan_route',
    'compare_routes',
    'search_along_route',
]
