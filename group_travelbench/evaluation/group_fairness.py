"""
Group-fairness evaluation.

Compute each user's individual utility (excluding the split-group penalty),
then measure the ratio between the minimum and maximum scores.

Fairness metric = min_score / max_score * 100%.
A larger value is fairer; 100% means every user has the same utility.
"""

from typing import Dict, Any, List

from .group_utility import (
    _evaluate_attraction_preferences,
    _evaluate_food_preferences,
    _evaluate_transport_preferences,
    _evaluate_hotel_preferences,
    _evaluate_intensity_preferences,
    _evaluate_budget_preference,
    _find_invalid_split_activities,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.eval_util import POICategoryResolver


def evaluate_group_fairness(
    plan: Dict,
    effective_preferences: Dict[str, Any],
    user_names: List[str],
) -> Dict[str, Any]:
    """
    Evaluate group fairness.

    Compute each user's individual utility (excluding the split-group penalty),
    then derive the fairness metric.

    Args:
        plan: Travel-plan JSON.
        effective_preferences: Reference preference table.
        user_names: List of user names.

    Returns:
        Result dict with per-user scores and the fairness metric.
    """
    resolver = POICategoryResolver.get_instance()

    # Pre-identify activities that violate the group-split rules
    invalid_activities = _find_invalid_split_activities(plan, user_names)

    per_user_scores: Dict[str, int] = {}

    for user_name in user_names:
        user_ref = effective_preferences.get(user_name, {})
        user_prefs = user_ref.get("preference", user_ref)

        city_prefs = user_prefs.get("city_specific_preferences", {})

        # Per-dimension scores (same logic as group_utility, without the split penalty)
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
        per_user_scores[user_name] = user_total

    # Fairness = min/max * 100%. Higher is fairer; 100% is perfectly fair
    scores = list(per_user_scores.values())
    if not scores:
        return {
            "per_user_scores": per_user_scores,
            "max_score": 0,
            "min_score": 0,
            "fairness_score": 100.0,
        }

    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        # Everyone has the same utility: perfectly fair
        fairness_score = 100.0
    elif max_score > 0 and min_score >= 0:
        # Typical case: min/max * 100%
        fairness_score = (min_score / max_score) * 100.0
    elif max_score > 0 and min_score < 0:
        # A negative min means some user was given a rejected preference: extremely unfair
        fairness_score = 0.0
    else:
        # max_score <= 0 (everyone's utility is negative or zero)
        # Invert: the less-negative score is better; use max/min
        if min_score == 0:
            fairness_score = 0.0
        else:
            fairness_score = (max_score / min_score) * 100.0

    return {
        "per_user_scores": per_user_scores,
        "max_score": max_score,
        "min_score": min_score,
        "fairness_score": round(fairness_score, 2),
    }
