"""
Group-utility evaluation.

Score the Agent-generated travel plan against ``effective_preferences``
using a four-tier preference rule, including a split-group penalty.

Scoring rules:
- Strong tier (must/reject/avg_budget/intensity): +2 if satisfied / -2 if violated
- Weak tier (prefer/avoid/category_pref/hotel_preference): +1 if satisfied / -1 if violated
- Positive preferences (must/prefer): points only when scheduled; no penalty if omitted
- Negative preferences (reject/avoid): penalty only when scheduled; no bonus if omitted
- Split-group penalty: K parallel groups in the same slot → deduct (K-1)

Each activity is scored independently:
- Attraction ``category_pref``: per attraction (not de-duplicated by category)
- Transport preferences: per intercity trip
- Food preferences: per restaurant
- Hotel preferences: per night
- Split-group errors (the same user in two overlapping activities) earn no utility
"""

import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

import sys
from pathlib import Path

# Add the package path so POICategoryResolver can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.eval_util import POICategoryResolver


# =============================================================================
# Plan extraction
# =============================================================================

def extract_plan_from_conversation(conversation_history: List[Dict]) -> Optional[Dict]:
    """
    Extract the final travel-plan JSON from ``conversation_history``.

    Scan backwards for the first Agent message that contains both ``days``
    and ``city_segments``, then parse it.

    The extracted JSON is returned only if it passes the same structural
    check as ``GroupTravelAgent.check_is_travel_plan`` (non-empty ``days``
    list, ``days[0]`` is a dict whose ``city_segments`` is a non-empty list).
    This keeps extraction no looser than the stop condition: otherwise a
    ``city_segments`` substring in earlier natural language, or another JSON
    object that appears before the real plan, could yield a half-formed or
    misaligned object and silently score the wrong payload.
    """
    for msg in reversed(conversation_history):
        if msg.get("role") != "Agent":
            continue
        content = msg.get("content", "")
        if "days" not in content or "city_segments" not in content:
            continue

        # Try extracting from a ```json ``` fenced block
        code_block_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content)
        if code_block_match:
            try:
                parsed = json.loads(code_block_match.group(1))
                if _is_valid_travel_plan(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fall back to matching the outermost JSON object
        # Use brace matching to recover a complete JSON object
        plan_json = _extract_json_with_days(content)
        if plan_json is not None and _is_valid_travel_plan(plan_json):
            return plan_json

    return None

def _is_valid_travel_plan(parsed: Any) -> bool:
    """
    Structural check: whether ``parsed`` is a valid travel plan.

    Same rule as ``GroupTravelAgent.check_is_travel_plan``: top-level dict,
    non-empty ``days`` list, first element a dict with a non-empty
    ``city_segments`` list. Segment internals are left to plan-validity
    scoring; this gate only rejects JSON that is not a plan at all.
    """
    if not isinstance(parsed, dict):
        return False
    days = parsed.get("days")
    if not isinstance(days, list) or not days:
        return False
    first_day = days[0]
    if not isinstance(first_day, dict):
        return False
    city_segments = first_day.get("city_segments")
    return isinstance(city_segments, list) and bool(city_segments)


def _extract_json_with_days(text: str) -> Optional[Dict]:
    """Extract a JSON object that contains a ``days`` field from text."""
    # Find the start of {"days"
    start_patterns = ['"days"', "'days'"]
    for pattern in start_patterns:
        idx = text.find(pattern)
        if idx == -1:
            continue
        # Walk backward to the nearest {
        brace_start = text.rfind("{", 0, idx)
        if brace_start == -1:
            continue
        # Use brace matching to recover a complete JSON object
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# =============================================================================
# Extract activities from the plan
# =============================================================================

def _get_all_activities(plan: Dict) -> List[Dict]:
    """Collect every activity in the plan, including ``intercity_transport``."""
    activities = []
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if segment.get("type") == "intercity_transport":
                activity = dict(segment)
                activity["_day"] = day_num
                activity["_city"] = None
                activities.append(activity)
            elif "activities" in segment:
                city = segment.get("city", "")
                for act in segment["activities"]:
                    activity = dict(act)
                    activity["_day"] = day_num
                    activity["_city"] = city
                    activities.append(activity)
    return activities


def _get_participants(activity: Dict, all_users: List[str]) -> List[str]:
    """Return the activity's participant list."""
    participants = activity.get("participants", [])
    if not participants or participants == ["All"]:
        return all_users
    return participants


def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse an HH:MM time string into a ``datetime`` (date part is a fixed dummy)."""
    try:
        return datetime.strptime(time_str, "%H:%M")
    except (ValueError, TypeError):
        return None


def _time_ranges_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Return whether two time ranges overlap."""
    t1_start = _parse_time(start1)
    t1_end = _parse_time(end1)
    t2_start = _parse_time(start2)
    t2_end = _parse_time(end2)

    if not all([t1_start, t1_end, t2_start, t2_end]):
        return False

    # Overnight ranges (e.g. hotel 20:00-08:00) are excluded from overlap checks
    if t1_end <= t1_start or t2_end <= t2_start:
        return False

    return t1_start < t2_end and t2_start < t1_end

# =============================================================================
# Split-error detection: overlapping activities whose participants intersect
# =============================================================================

def _find_invalid_split_activities(plan: Dict, all_users: List[str]) -> Set[Tuple[int, str, str]]:
    """
    Find split-group errors: the same user appears in overlapping activities.

    Returns a set of ``(day_num, location, start_time)`` triples that identify
    activities which must be excluded from utility scoring.
    """
    invalid_activities: Set[Tuple[int, str, str]] = set()

    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue

            activities = segment["activities"]
            for user in all_users:
                user_acts = []
                for act in activities:
                    if act.get("type", "") == "hotel":
                        continue
                    participants = _get_participants(act, all_users)
                    if user in participants:
                        user_acts.append(act)

                for i in range(len(user_acts)):
                    for j in range(i + 1, len(user_acts)):
                        if _time_ranges_overlap(
                            user_acts[i].get("start_time", ""),
                            user_acts[i].get("end_time", ""),
                            user_acts[j].get("start_time", ""),
                            user_acts[j].get("end_time", ""),
                        ):
                            act_key_i = (
                                day_num,
                                user_acts[i].get("location", user_acts[i].get("to", "")),
                                user_acts[i].get("start_time", ""),
                            )
                            act_key_j = (
                                day_num,
                                user_acts[j].get("location", user_acts[j].get("to", "")),
                                user_acts[j].get("start_time", ""),
                            )
                            invalid_activities.add(act_key_i)
                            invalid_activities.add(act_key_j)

    return invalid_activities

def _is_activity_invalid(
    activity: Dict, day_num: int, invalid_activities: Set[Tuple[int, str, str]]
) -> bool:
    """Return whether the activity is in the invalid-activity set."""
    act_key = (
        day_num,
        activity.get("location", activity.get("to", "")),
        activity.get("start_time", ""),
    )
    return act_key in invalid_activities

# =============================================================================
# Per-category preference scoring

def _evaluate_attraction_preferences(
    plan: Dict,
    user_name: str,
    city_prefs: Dict[str, Any],
    all_users: List[str],
    resolver: POICategoryResolver,
    invalid_activities: Optional[Set[Tuple[int, str, str]]] = None,
) -> List[Dict]:
    """
    Score attraction-preference satisfaction.

    Walk every attraction activity and score its participants independently:
    - must_visit: substring match; +2 if scheduled, no penalty if omitted
    - reject_visit: substring match; -2 if scheduled, no bonus if omitted
    - category_pref.positive: category match; +1 per attraction, no penalty if omitted
    - category_pref.negative: category match; -1 per attraction, no bonus if omitted

    Attractions already matched by must_visit/reject_visit are excluded from
    ``category_pref`` to avoid double counting. Split-group errors
    (``invalid_activities``) are not scored.
    """
    score_details = []
    activities = _get_all_activities(plan)
    if invalid_activities is None:
        invalid_activities = set()

    for city, city_pref in city_prefs.items():
        attractions_pref = city_pref.get("attractions", {})
        must_visit = attractions_pref.get("must_visit", [])
        reject_visit = attractions_pref.get("reject_visit", [])
        category_pref_dict = attractions_pref.get("category_pref", {})
        positive_cats = category_pref_dict.get("positive", [])
        negative_cats = category_pref_dict.get("negative", [])

        # Attractions this user joined in this city (excluding invalid split activities)
        user_attractions = []
        for act in activities:
            if act.get("type") != "attraction":
                continue
            if act.get("_city") != city:
                continue
            participants = _get_participants(act, all_users)
            if user_name not in participants:
                continue
            if _is_activity_invalid(act, act.get("_day", 0), invalid_activities):
                continue
            user_attractions.append(act)

        # must_visit scoring (strong positive, +2)
        # Remember locations already scored by must_visit/reject_visit to avoid double-counting category_pref
        strong_matched_locations: Set[str] = set()

        for must_poi in must_visit:
            found = any(
                must_poi in act.get("location", "")
                for act in user_attractions
            )
            if found:
                score_details.append({
                    "field": "must_visit",
                    "item": must_poi,
                    "city": city,
                    "satisfied": True,
                    "score": 2,
                })
                for act in user_attractions:
                    if must_poi in act.get("location", ""):
                        strong_matched_locations.add(act.get("location", ""))

        # reject_visit scoring (strong negative, -2)
        for reject_poi in reject_visit:
            violated = any(
                reject_poi in act.get("location", "")
                for act in user_attractions
            )
            if violated:
                score_details.append({
                    "field": "reject_visit",
                    "item": reject_poi,
                    "city": city,
                    "satisfied": False,
                    "score": -2,
                })
                for act in user_attractions:
                    if reject_poi in act.get("location", ""):
                        strong_matched_locations.add(act.get("location", ""))

        # category_pref.positive scoring (weak positive, +1 per attraction)
        # Prefer name matching: does the attraction name contain the preference keyword?
        for act in user_attractions:
            location = act.get("location", "")
            if location in strong_matched_locations:
                continue
            # First check via the resolver that the POI exists in the database
            category = resolver.resolve(location, city)
            if category is None:
                # POI is not in the database; skip scoring
                continue
            # Try name matching first
            name_matched = False
            for pref_cat in positive_cats:
                if pref_cat in location:
                    score_details.append({
                        "field": "category_pref.positive",
                        "item": pref_cat,
                        "location": location,
                        "city": city,
                        "satisfied": True,
                        "score": 1,
                    })
                    name_matched = True
                    break
            # Name did not match; fall back to category matching
            if not name_matched:
                if category in positive_cats:
                    score_details.append({
                        "field": "category_pref.positive",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": True,
                        "score": 1,
                    })

        # category_pref.negative scoring (weak negative, -1 per attraction)
        # Likewise prefer name matching
        for act in user_attractions:
            location = act.get("location", "")
            if location in strong_matched_locations:
                continue
            # First check via the resolver that the POI exists in the database
            category = resolver.resolve(location, city)
            if category is None:
                # POI is not in the database; skip scoring
                continue
            # Try name matching first
            name_matched = False
            for pref_cat in negative_cats:
                if pref_cat in location:
                    score_details.append({
                        "field": "category_pref.negative",
                        "item": pref_cat,
                        "location": location,
                        "city": city,
                        "satisfied": False,
                        "score": -1,
                    })
                    name_matched = True
                    break
            # Name did not match; fall back to category matching
            if not name_matched:
                if category in negative_cats:
                    score_details.append({
                        "field": "category_pref.negative",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": False,
                        "score": -1,
                    })

    return score_details


def _evaluate_food_preferences(
    plan: Dict,
    user_name: str,
    city_prefs: Dict[str, Any],
    all_users: List[str],
    resolver: POICategoryResolver,
    invalid_activities: Optional[Set[Tuple[int, str, str]]] = None,
) -> List[Dict]:
    """
    Score food-preference satisfaction.

    Walk every dining activity and score its participants independently:
    - must_eat: category match; +2 per restaurant
    - reject_eat: category match; -2 per restaurant
    - prefer_eat: category match; +1 per restaurant
    - avoid_eat: category match; -1 per restaurant

    Split-group errors (``invalid_activities``) are not scored.
    """
    score_details = []
    activities = _get_all_activities(plan)
    if invalid_activities is None:
        invalid_activities = set()

    for city, city_pref in city_prefs.items():
        food_pref = city_pref.get("food", {})
        must_eat = food_pref.get("must_eat", [])
        reject_eat = food_pref.get("reject_eat", [])
        prefer_eat = food_pref.get("prefer_eat", [])
        avoid_eat = food_pref.get("avoid_eat", [])

        # Food activities this user joined in this city (excluding invalid split activities)
        user_food_acts = []
        for act in activities:
            if act.get("type") != "food":
                continue
            if act.get("_city") != city:
                continue
            participants = _get_participants(act, all_users)
            if user_name not in participants:
                continue
            if _is_activity_invalid(act, act.get("_day", 0), invalid_activities):
                continue
            user_food_acts.append(act)

        # Score each restaurant independently
        for act in user_food_acts:
            location = act.get("location", "")

            # First check via the resolver that the POI exists in the database
            category = resolver.resolve(location, city)
            # POI is not in the database (possibly fabricated); skip scoring
            if category is None:
                continue

            # Prefer name matching: does the restaurant name contain the preference keyword?
            # On a name match, score immediately and skip category logic
            scored = False

            # Name-match priority: must > reject > prefer > avoid
            for pref_item in must_eat:
                if pref_item in location:
                    score_details.append({
                        "field": "must_eat",
                        "item": pref_item,
                        "location": location,
                        "city": city,
                        "satisfied": True,
                        "score": 2,
                    })
                    scored = True
                    break

            if not scored:
                for pref_item in reject_eat:
                    if pref_item in location:
                        score_details.append({
                            "field": "reject_eat",
                            "item": pref_item,
                            "location": location,
                            "city": city,
                            "satisfied": False,
                            "score": -2,
                        })
                        scored = True
                        break

            if not scored:
                for pref_item in prefer_eat:
                    if pref_item in location:
                        score_details.append({
                            "field": "prefer_eat",
                            "item": pref_item,
                            "location": location,
                            "city": city,
                            "satisfied": True,
                            "score": 1,
                        })
                        scored = True
                        break

            if not scored:
                for pref_item in avoid_eat:
                    if pref_item in location:
                        score_details.append({
                            "field": "avoid_eat",
                            "item": pref_item,
                            "location": location,
                            "city": city,
                            "satisfied": False,
                            "score": -1,
                        })
                        scored = True
                        break

            # Name did not match; fall back to category matching
            if not scored:
                if category in must_eat:
                    score_details.append({
                        "field": "must_eat",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": True,
                        "score": 2,
                    })
                    scored = True

                if category in reject_eat:
                    score_details.append({
                        "field": "reject_eat",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": False,
                        "score": -2,
                    })
                    scored = True

                if not scored and category in prefer_eat:
                    score_details.append({
                        "field": "prefer_eat",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": True,
                        "score": 1,
                    })
                    scored = True

                if not scored and category in avoid_eat:
                    score_details.append({
                        "field": "avoid_eat",
                        "item": category,
                        "location": location,
                        "city": city,
                        "satisfied": False,
                        "score": -1,
                    })

    return score_details


def _evaluate_transport_preferences(
    plan: Dict,
    user_prefs: Dict[str, Any],
) -> List[Dict]:
    """
    Score transport-preference satisfaction.

    Score each ``intercity_transport`` independently:
    - must: +2 each time that mode is scheduled
    - reject: -2 each time that mode is scheduled
    - prefer: +1 each time that mode is scheduled
    - avoid: -1 each time that mode is scheduled
    """
    score_details = []
    transport_pref = user_prefs.get("global_constraints", {}).get("transport", {})
    must_modes = transport_pref.get("must", [])
    reject_modes = transport_pref.get("reject", [])
    prefer_modes = transport_pref.get("prefer", [])
    avoid_modes = transport_pref.get("avoid", [])

    # Score each intercity_transport in the plan
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if segment.get("type") != "intercity_transport":
                continue
            mode = segment.get("transport_mode", "")
            if not mode:
                continue

            route = f"{segment.get('from', '')}->{segment.get('to', segment.get('to_city', ''))}"

            # Each trip matches the highest-priority bucket: must > reject > prefer > avoid
            scored = False

            if mode in must_modes:
                score_details.append({
                    "field": "transport.must",
                    "item": mode,
                    "route": route,
                    "day": day_num,
                    "satisfied": True,
                    "score": 2,
                })
                scored = True

            if mode in reject_modes:
                score_details.append({
                    "field": "transport.reject",
                    "item": mode,
                    "route": route,
                    "day": day_num,
                    "satisfied": False,
                    "score": -2,
                })
                scored = True

            if not scored and mode in prefer_modes:
                score_details.append({
                    "field": "transport.prefer",
                    "item": mode,
                    "route": route,
                    "day": day_num,
                    "satisfied": True,
                    "score": 1,
                })
                scored = True

            if not scored and mode in avoid_modes:
                score_details.append({
                    "field": "transport.avoid",
                    "item": mode,
                    "route": route,
                    "day": day_num,
                    "satisfied": False,
                    "score": -1,
                })

    return score_details


def _evaluate_hotel_preferences(
    plan: Dict,
    user_prefs: Dict[str, Any],
    resolver: POICategoryResolver,
) -> List[Dict]:
    """
    Score hotel-preference satisfaction.

    Score each night's hotel independently:
    - prefer: +1 per night for that hotel type
    - avoid: -1 per night for that hotel type

    Hotels are shared by the whole group; participants are not split.
    """
    score_details = []
    hotel_pref = user_prefs.get("global_constraints", {}).get("hotel_preference", {})
    prefer_types = hotel_pref.get("prefer", [])
    avoid_types = hotel_pref.get("avoid", [])

    if not prefer_types and not avoid_types:
        return score_details

    # Score each night's hotel in the plan
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue
            city = segment.get("city", "")
            for act in segment["activities"]:
                if act.get("type") != "hotel":
                    continue
                location = act.get("location", "")
                category = resolver.resolve(location, city)
                if not category:
                    continue

                # Each night matches the highest-priority bucket: prefer > avoid
                if category in prefer_types:
                    score_details.append({
                        "field": "hotel_preference.prefer",
                        "item": category,
                        "location": location,
                        "day": day_num,
                        "satisfied": True,
                        "score": 1,
                    })
                elif category in avoid_types:
                    score_details.append({
                        "field": "hotel_preference.avoid",
                        "item": category,
                        "location": location,
                        "day": day_num,
                        "satisfied": False,
                        "score": -1,
                    })

    return score_details


def _evaluate_intensity_preferences(
    plan: Dict,
    user_name: str,
    user_prefs: Dict[str, Any],
    all_users: List[str],
) -> List[Dict]:
    """
    Score activity-intensity preferences.

    - max_poi_per_day: the user's daily attraction count must not exceed the
      cap (strong tier; -2 on violation)
    - max_active_hours: the user's daily attraction + intracity_transport
      duration must not exceed the cap (strong tier; -2 on violation)
    """
    score_details = []
    intensity = user_prefs.get("global_constraints", {}).get("intensity", {})
    max_poi = intensity.get("max_poi_per_day", 0)
    max_hours = intensity.get("max_active_hours", 0)

    if not max_poi and not max_hours:
        return score_details

    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        day_poi_count = 0
        day_active_minutes = 0

        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue
            for act in segment["activities"]:
                participants = _get_participants(act, all_users)
                if user_name not in participants:
                    continue

                act_type = act.get("type", "")

                if act_type == "attraction":
                    day_poi_count += 1
                    day_active_minutes += _calculate_duration_minutes(act)
                elif act_type == "intracity_transport":
                    day_active_minutes += _calculate_duration_minutes(act)

        # max_poi_per_day check
        if max_poi > 0 and day_poi_count > max_poi:
            score_details.append({
                "field": "intensity.max_poi_per_day",
                "day": day_num,
                "actual": day_poi_count,
                "limit": max_poi,
                "satisfied": False,
                "score": -2,
            })

        # max_active_hours check
        if max_hours > 0:
            actual_hours = day_active_minutes / 60.0
            if actual_hours > max_hours:
                score_details.append({
                    "field": "intensity.max_active_hours",
                    "day": day_num,
                    "actual_hours": round(actual_hours, 2),
                    "limit": max_hours,
                    "satisfied": False,
                    "score": -2,
                })

    return score_details


def _calculate_duration_minutes(activity: Dict) -> int:
    """Compute the activity duration in minutes."""
    start = _parse_time(activity.get("start_time", ""))
    end = _parse_time(activity.get("end_time", ""))
    if not start or not end:
        return 0
    if end <= start:
        return 0
    delta = end - start
    return int(delta.total_seconds() / 60)


def _evaluate_budget_preference(
    plan: Dict,
    user_name: str,
    user_prefs: Dict[str, Any],
    all_users: List[str],
) -> List[Dict]:
    """
    Score budget preference.

    Only hotel + food + intercity_transport ``avg_cost`` values are summed.
    When the group splits, costs stay per-person (``avg_cost`` is already
    per capita) and can be added directly. Exceeding ``avg_budget`` is a
    strong-tier violation (-2).
    """
    score_details = []
    avg_budget = user_prefs.get("global_constraints", {}).get("avg_budget", 0)

    if not avg_budget:
        return score_details

    total_cost = 0.0

    for day in plan.get("days", []):
        for segment in day.get("city_segments", []):
            if segment.get("type") == "intercity_transport":
                # intercity_transport is mandatory for everyone; avg_cost is charged to each person
                cost = segment.get("avg_cost", 0)
                if cost:
                    total_cost += float(cost)
            elif "activities" in segment:
                for act in segment["activities"]:
                    act_type = act.get("type", "")
                    cost = act.get("avg_cost", 0)
                    if not cost:
                        continue

                    if act_type == "hotel":
                        # hotel is mandatory for everyone; avg_cost is charged directly
                        total_cost += float(cost)
                    elif act_type == "food":
                        # food must inspect participants
                        participants = _get_participants(act, all_users)
                        if user_name in participants:
                            total_cost += float(cost)

    if total_cost > avg_budget:
        score_details.append({
            "field": "avg_budget",
            "actual_cost": round(total_cost, 2),
            "budget_limit": avg_budget,
            "satisfied": False,
            "score": -2,
        })

    return score_details


# =============================================================================
# Split-penalty calculation
# =============================================================================

def _calculate_split_penalty(plan: Dict, all_users: List[str]) -> Dict[str, Any]:
    """
    Compute the split-group penalty.

    Rule: K parallel groups in the same slot → deduct (K-1).
    Detection: within the same day's ``city_block``, find activity groups
    whose time ranges overlap and whose participant sets are disjoint.
    """
    total_penalty = 0
    split_events = []

    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue

            # Only activity types that are allowed to split
            splittable_types = {"attraction", "food", "rest", "intracity_transport"}
            splittable_activities = []
            for act in segment["activities"]:
                if act.get("type") in splittable_types:
                    participants = act.get("participants", [])
                    # Only a non-empty, non-All participants list can be a split
                    if participants and participants != ["All"]:
                        splittable_activities.append(act)

            if not splittable_activities:
                continue

            # Group concurrent activities (overlapping intervals, including transitive overlap).
            # Union-Find: add j to a group if it overlaps any existing member.
            visited = set()
            for i in range(len(splittable_activities)):
                if i in visited:
                    continue
                concurrent_group = [splittable_activities[i]]
                visited.add(i)

                # Rescan until no new member is added (transitive overlap)
                changed = True
                while changed:
                    changed = False
                    for j in range(len(splittable_activities)):
                        if j in visited:
                            continue
                        act_j = splittable_activities[j]
                        # Check whether j overlaps any existing member of the group
                        overlaps_with_group = any(
                            _time_ranges_overlap(
                                member.get("start_time", ""),
                                member.get("end_time", ""),
                                act_j.get("start_time", ""),
                                act_j.get("end_time", ""),
                            )
                            for member in concurrent_group
                        )
                        if overlaps_with_group:
                            concurrent_group.append(act_j)
                            visited.add(j)
                            changed = True

                if len(concurrent_group) <= 1:
                    continue

                # Bucket by participants: activities with the same set belong to one team.
                # Use a frozenset key to identify distinct teams.
                team_map: Dict[frozenset, List[Dict]] = {}
                for act in concurrent_group:
                    p_key = frozenset(act.get("participants", []))
                    if p_key not in team_map:
                        team_map[p_key] = []
                    team_map[p_key].append(act)

                if len(team_map) <= 1:
                    # All activities belong to one team (identical participants): not a split
                    continue

                # Verify that teams have disjoint participants
                team_participant_sets = list(team_map.keys())
                is_valid_split = True
                for ti in range(len(team_participant_sets)):
                    for tj in range(ti + 1, len(team_participant_sets)):
                        if team_participant_sets[ti] & team_participant_sets[tj]:
                            is_valid_split = False
                            break
                    if not is_valid_split:
                        break

                if is_valid_split and len(team_map) > 1:
                    num_teams = len(team_map)
                    penalty = num_teams - 1
                    total_penalty += penalty
                    split_events.append({
                        "day": day_num,
                        "num_teams": num_teams,
                        "penalty": penalty,
                        "activities": [
                            {
                                "location": act.get("location", act.get("to", "")),
                                "time": f"{act.get('start_time', '')}-{act.get('end_time', '')}",
                                "participants": act.get("participants", []),
                            }
                            for act in concurrent_group
                        ],
                    })

    return {
        "total_penalty": total_penalty,
        "split_events": split_events,
    }


# =============================================================================
# Main evaluation entry
# =============================================================================

def evaluate_group_utility(
    plan: Dict,
    effective_preferences: Dict[str, Any],
    user_names: List[str],
) -> Dict[str, Any]:
    """
    Evaluate group utility.

    Args:
        plan: Travel-plan JSON (must contain ``days``).
        effective_preferences: Reference preference table.
        user_names: List of user names.

    Returns:
        Result dict with per-user preference details, total score, and
        the split-group penalty.
    """
    resolver = POICategoryResolver.get_instance()

    # Pre-identify invalid splits (the same user in several overlapping activities)
    invalid_activities = _find_invalid_split_activities(plan, user_names)

    per_user_results = {}
    total_score = 0

    for user_name in user_names:
        user_ref = effective_preferences.get(user_name, {})
        user_prefs = user_ref.get("preference", user_ref)
        if "preference" not in user_ref and "global_constraints" not in user_ref:
            continue
        if "preference" in user_ref:
            user_prefs = user_ref["preference"]

        city_prefs = user_prefs.get("city_specific_preferences", {})

        # Per-dimension scores (invalid_activities are excluded)
        attraction_scores = _evaluate_attraction_preferences(
            plan, user_name, city_prefs, user_names, resolver, invalid_activities
        )
        food_scores = _evaluate_food_preferences(
            plan, user_name, city_prefs, user_names, resolver, invalid_activities
        )
        transport_scores = _evaluate_transport_preferences(plan, user_prefs)
        hotel_scores = _evaluate_hotel_preferences(plan, user_prefs, resolver)
        intensity_scores = _evaluate_intensity_preferences(
            plan, user_name, user_prefs, user_names
        )
        budget_scores = _evaluate_budget_preference(
            plan, user_name, user_prefs, user_names
        )

        all_scores = (
            attraction_scores + food_scores + transport_scores
            + hotel_scores + intensity_scores + budget_scores
        )

        user_total = sum(item["score"] for item in all_scores)
        per_user_results[user_name] = {
            "total_score": user_total,
            "details": all_scores,
            "breakdown": {
                "attractions": sum(s["score"] for s in attraction_scores),
                "food": sum(s["score"] for s in food_scores),
                "transport": sum(s["score"] for s in transport_scores),
                "hotel": sum(s["score"] for s in hotel_scores),
                "intensity": sum(s["score"] for s in intensity_scores),
                "budget": sum(s["score"] for s in budget_scores),
            },
        }
        total_score += user_total

    # Split penalty
    split_result = _calculate_split_penalty(plan, user_names)
    total_score -= split_result["total_penalty"]

    # Average over group size so tasks with different group sizes are comparable
    num_users = len(per_user_results) if per_user_results else 1
    per_user_average_score = total_score / num_users

    return {
        "per_user": per_user_results,
        "split_penalty": split_result,
        "total_score_before_penalty": total_score + split_result["total_penalty"],
        "total_score": total_score,
        "num_users": num_users,
        "per_user_average_score": round(per_user_average_score, 2),
    }
