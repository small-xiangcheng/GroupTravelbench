"""
Preference-collection completeness evaluation.

Compare the Agent's preference table (``agent_preference_table``) with
the reference (``effective_preferences``) field by field.

Scoring:
- List fields: counted by element; N reference items are N points, M
  collected items score M
- Scalar fields: exact match scores 1, otherwise 0
- Nested structures: flattened to leaves, then compared one by one
"""

from typing import Dict, Any, List, Tuple


def evaluate_preference_completeness(
    agent_pref_table: Dict[str, Any],
    effective_preferences: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate preference-collection completeness.

    Args:
        agent_pref_table: Agent-maintained table,
            ``{user: {global_constraints, city_specific_preferences}}``.
        effective_preferences: Reference table,
            ``{user: {role, preference: {...}, ...}}``.

    Returns:
        Dict with per-user scores and overall statistics.
    """
    results = {
        "per_user": {},
        "total_collected": 0,
        "total_possible": 0,
        "completeness_ratio": 0.0,
        "precision": 1.0,
        "f1": 0.0,
    }

    total_correct_in_agent = 0
    total_agent_items = 0

    for user_name, ref_data in effective_preferences.items():
        ref_pref = ref_data.get("preference", ref_data)
        if "preference" not in ref_data and "global_constraints" in ref_data:
            ref_pref = ref_data

        agent_pref = agent_pref_table.get(user_name, {})

        user_result = _compare_preferences(agent_pref, ref_pref)
        results["per_user"][user_name] = user_result
        results["total_collected"] += user_result["collected"]
        results["total_possible"] += user_result["possible"]

        # Accumulate precision numerator/denominator from per-user details
        for detail in user_result["details"]:
            if "agent_total" in detail:
                total_agent_items += detail["agent_total"]
                total_correct_in_agent += int(detail["precision"] * detail["agent_total"])
            else:
                # Scalar fields: agent provided a value if it is not None
                if detail.get("agent_value") is not None:
                    total_agent_items += 1
                    total_correct_in_agent += detail["collected"]

    recall = results["total_collected"] / results["total_possible"] if results["total_possible"] > 0 else 1.0
    precision = total_correct_in_agent / total_agent_items if total_agent_items > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    results["completeness_ratio"] = recall
    results["precision"] = precision
    results["f1"] = f1

    return results


def _compare_preferences(
    agent_pref: Dict[str, Any],
    ref_pref: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare Agent preferences with the reference field by field.

    Returns:
        Dict with recall, precision, F1, and per-field diffs.
    """
    details: List[Dict[str, Any]] = []
    total_collected = 0
    total_possible = 0
    total_correct_in_agent = 0
    total_agent_items = 0

    # Compare global_constraints
    ref_global = ref_pref.get("global_constraints") or {}
    agent_global = agent_pref.get("global_constraints") or {}
    _compare_global_constraints(agent_global, ref_global, details)

    # Compare city_specific_preferences
    ref_cities = ref_pref.get("city_specific_preferences") or {}
    agent_cities = agent_pref.get("city_specific_preferences") or {}
    _compare_city_preferences(agent_cities, ref_cities, details)

    for detail in details:
        total_collected += detail["collected"]
        total_possible += detail["possible"]
        # Accumulate precision stats (only list fields have these)
        if "agent_total" in detail:
            total_agent_items += detail["agent_total"]
            total_correct_in_agent += int(detail["precision"] * detail["agent_total"])
        else:
            # Scalar fields: agent provided a value if it is not None
            if detail.get("agent_value") is not None:
                total_agent_items += 1
                total_correct_in_agent += detail["collected"]

    recall = total_collected / total_possible if total_possible > 0 else 1.0
    precision = total_correct_in_agent / total_agent_items if total_agent_items > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "collected": total_collected,
        "possible": total_possible,
        "ratio": recall,
        "precision": precision,
        "f1": f1,
        "details": details,
    }


def _compare_global_constraints(
    agent_global: Dict[str, Any],
    ref_global: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> None:
    """Compare fields under ``global_constraints``."""

    # avg_budget (scalar)
    _compare_scalar(
        agent_global.get("avg_budget", 0),
        ref_global.get("avg_budget", 0),
        "global_constraints.avg_budget",
        details,
    )

    # transport (four list fields)
    ref_transport = ref_global.get("transport") or {}
    agent_transport = agent_global.get("transport") or {}
    for level in ["must", "prefer", "avoid", "reject"]:
        _compare_list(
            agent_transport.get(level) or [],
            ref_transport.get(level) or [],
            f"global_constraints.transport.{level}",
            details,
        )

    # intensity (two scalars)
    ref_intensity = ref_global.get("intensity") or {}
    agent_intensity = agent_global.get("intensity") or {}
    for field in ["max_poi_per_day", "max_active_hours"]:
        _compare_scalar(
            agent_intensity.get(field, 0),
            ref_intensity.get(field, 0),
            f"global_constraints.intensity.{field}",
            details,
        )

    # hotel_preference (two lists)
    ref_hotel = ref_global.get("hotel_preference") or {}
    agent_hotel = agent_global.get("hotel_preference") or {}
    for level in ["prefer", "avoid"]:
        _compare_list(
            agent_hotel.get(level) or [],
            ref_hotel.get(level) or [],
            f"global_constraints.hotel_preference.{level}",
            details,
        )


def _compare_city_preferences(
    agent_cities: Dict[str, Any],
    ref_cities: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> None:
    """Compare per-city fields under ``city_specific_preferences``."""

    for city_name, ref_city_pref in ref_cities.items():
        agent_city_pref = agent_cities.get(city_name) or {}

        # attractions — skip and score 0 if the value is not a dict
        ref_attractions = (ref_city_pref.get("attractions") or {}) if ref_city_pref else {}
        agent_attractions = agent_city_pref.get("attractions") or {}
        if not isinstance(agent_attractions, dict):
            agent_attractions = {}

        for field in ["must_visit", "reject_visit"]:
            _compare_list(
                agent_attractions.get(field) or [],
                ref_attractions.get(field) or [],
                f"city[{city_name}].attractions.{field}",
                details,
            )

        # category_pref — skip and score 0 if the value is not a dict
        ref_cat_pref = ref_attractions.get("category_pref") or {}
        agent_cat_pref = agent_attractions.get("category_pref") or {}
        if not isinstance(agent_cat_pref, dict):
            agent_cat_pref = {}
        for level in ["positive", "negative"]:
            _compare_list(
                agent_cat_pref.get(level) or [],
                ref_cat_pref.get(level) or [],
                f"city[{city_name}].attractions.category_pref.{level}",
                details,
            )

        # food — skip and score 0 if the value is not a dict
        ref_food = (ref_city_pref.get("food") or {}) if ref_city_pref else {}
        agent_food = agent_city_pref.get("food") or {}
        if not isinstance(agent_food, dict):
            agent_food = {}
        for field in ["must_eat", "prefer_eat", "avoid_eat", "reject_eat"]:
            _compare_list(
                agent_food.get(field) or [],
                ref_food.get(field) or [],
                f"city[{city_name}].food.{field}",
                details,
            )


def _compare_scalar(
    agent_value: Any,
    ref_value: Any,
    field_path: str,
    details: List[Dict[str, Any]],
) -> None:
    """Compare a scalar field (exact match). Counted only when the reference is non-zero / non-empty."""
    if not ref_value:
        return

    collected = 1 if agent_value == ref_value else 0
    details.append({
        "field_path": field_path,
        "collected": collected,
        "possible": 1,
        "agent_value": agent_value,
        "ref_value": ref_value,
    })


def _compare_list(
    agent_list: List[Any],
    ref_list: List[Any],
    field_path: str,
    details: List[Dict[str, Any]],
) -> None:
    """
    Compare a list field (counted by element). Counted only when the
    reference list is non-empty.

    Reports both recall (collected/possible) and precision (correct/agent_total)
    to detect cases where the Agent pads the list with spurious entries.
    """
    if not ref_list:
        return

    possible = len(ref_list)
    collected = 0
    matched_items = []
    missed_items = []

    for ref_item in ref_list:
        if ref_item in agent_list:
            collected += 1
            matched_items.append(ref_item)
        else:
            missed_items.append(ref_item)

    # Precision: of what the Agent reported, how many are actually correct
    agent_total = len(agent_list)
    correct_in_agent = sum(1 for item in agent_list if item in ref_list)
    precision = correct_in_agent / agent_total if agent_total > 0 else 1.0
    spurious_items = [item for item in agent_list if item not in ref_list]

    details.append({
        "field_path": field_path,
        "collected": collected,
        "possible": possible,
        "agent_total": agent_total,
        "precision": precision,
        "spurious": spurious_items,
        "matched": matched_items,
        "missed": missed_items,
    })
