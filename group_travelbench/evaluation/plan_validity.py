"""
Travel-plan validity checks.

Checks:
1. Adjacent cities have an ``intercity_transport``
2. Every night except the last has a hotel
3. Activities in the same ``city_block`` do not overlap for a given participant
4. Activity times are consistent (start < end, except overnight hotels)
5. Attractions are visited during opening hours
6. Within a ``city_block``, each user's consecutive locations are connected by ``intracity_transport``
7. From each user's view, activity times stay monotonic within a day
8. Food, hotel, and intercity transport include ``avg_cost``
9. Each ``participants`` entry is ``"All"`` or a known user, and is not an empty list
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.eval_util import POIOpeningHoursResolver


def evaluate_plan_validity(
    plan: Dict,
    user_names: List[str],
) -> Dict[str, Any]:
    """
    Run validity checks on a travel plan.

    Args:
        plan: Travel-plan JSON.
        user_names: List of user names.

    Returns:
        Dict with the various check results.
    """
    issues: List[Dict[str, Any]] = []

    _check_intercity_transport(plan, issues)
    _check_hotel_every_night(plan, issues)
    _check_time_order(plan, issues)
    _check_time_overlap(plan, user_names, issues)
    _check_opening_hours(plan, issues)
    _check_intracity_transport_continuity(plan, user_names, issues)
    _check_day_timeline_order(plan, user_names, issues)
    _check_avg_cost_present(plan, issues)
    _check_participants_validity(plan, user_names, issues)

    passed = len(issues) == 0
    issue_counts = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    return {
        "valid": passed,
        "total_issues": len(issues),
        "issue_counts": issue_counts,
        "issues": issues,
    }


# =============================================================================
# Check 1: intercity_transport between adjacent cities
# =============================================================================

def _normalize_city_name(city: str) -> str:
    """
    Normalize a city name by stripping common administrative suffixes.

    Examples: "合肥市" -> "合肥", "北京市" -> "北京", "杭州" -> "杭州".
    """
    if not city:
        return city
    # Strip common administrative suffixes
    for suffix in ("市", "区", "县", "自治州", "地区"):
        if city.endswith(suffix) and len(city) > len(suffix):
            return city[: -len(suffix)]
    return city

def _cities_match(city1: str, city2: str) -> bool:
    """
    Return whether two city names refer to the same city.

    Handles common variants such as "合肥" == "合肥市" and "北京" == "北京市".
    """
    if city1 == city2:
        return True
    return _normalize_city_name(city1) == _normalize_city_name(city2)

def _check_intercity_transport(plan: Dict, issues: List[Dict]) -> None:
    """Check that city changes include intercity transport."""
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        segments = day.get("city_segments", [])

        prev_city = None
        for i, segment in enumerate(segments):
            if segment.get("type") == "intercity_transport":
                prev_city = segment.get("to_city", "")
            elif "city" in segment:
                current_city = segment.get("city", "")
                if prev_city is not None and not _cities_match(prev_city, current_city):
                    # City changed but there is no intercity_transport
                    issues.append({
                        "type": "missing_intercity_transport",
                        "day": day_num,
                        "segment_index": i,
                        "from_city": prev_city,
                        "to_city": current_city,
                        "message": f"Day {day_num}: 从 {prev_city} 到 {current_city} 缺少城间交通",
                    })
                prev_city = current_city

    # Cross-day city continuity
    days = plan.get("days", [])
    for i in range(1, len(days)):
        prev_day_segments = days[i - 1].get("city_segments", [])
        curr_day_segments = days[i].get("city_segments", [])

        # Last city of the previous day
        prev_last_city = _get_last_city_of_day(prev_day_segments)
        # First city of the current day
        curr_first_city = _get_first_city_of_day(curr_day_segments)

        if (
            prev_last_city
            and curr_first_city
            and not _cities_match(prev_last_city, curr_first_city)
        ):
            # Cities differ across days: the current day must start with intercity_transport
            first_segment = curr_day_segments[0] if curr_day_segments else None
            if not first_segment or first_segment.get("type") != "intercity_transport":
                issues.append({
                    "type": "missing_intercity_transport",
                    "day": days[i].get("day", 0),
                    "from_city": prev_last_city,
                    "to_city": curr_first_city,
                    "message": (
                        f"Day {days[i].get('day', 0)}: 跨天从 {prev_last_city} "
                        f"到 {curr_first_city} 缺少城间交通"
                    ),
                })


def _get_last_city_of_day(segments: List[Dict]) -> Optional[str]:
    """Return the last city that appears on a given day."""
    for segment in reversed(segments):
        if "city" in segment:
            return segment["city"]
        if segment.get("type") == "intercity_transport":
            return segment.get("to_city")
    return None


def _get_first_city_of_day(segments: List[Dict]) -> Optional[str]:
    """Return the first city that appears on a given day."""
    for segment in segments:
        if segment.get("type") == "intercity_transport":
            return segment.get("to_city")
        if "city" in segment:
            return segment["city"]
    return None


# =============================================================================
# Check 2: a hotel every night
# =============================================================================

def _check_hotel_every_night(plan: Dict, issues: List[Dict]) -> None:
    """
    Check that every day except the last has lodging, with the hotel as
    the last activity of the last ``city_block``.

    A mid-day hotel rest is allowed; the day still needs a hotel activity
    at the end of the last ``city_block`` as the night's stay.
    """
    days = plan.get("days", [])
    if len(days) <= 1:
        return

    for day_data in days[:-1]:  # The last day does not need lodging
        day_num = day_data.get("day", 0)
        segments = day_data.get("city_segments", [])

        # Find the last city_block that has activities
        last_city_block = None
        for segment in reversed(segments):
            if "activities" in segment and segment.get("type") != "intercity_transport":
                last_city_block = segment
                break

        if last_city_block is None:
            issues.append({
                "type": "missing_hotel",
                "day": day_num,
                "message": f"Day {day_num}: 未安排住宿（无 city_block）",
            })
            continue

        # The last activity of the last city_block must be a hotel
        activities = last_city_block.get("activities", [])
        if not activities:
            issues.append({
                "type": "missing_hotel",
                "day": day_num,
                "message": f"Day {day_num}: 未安排住宿（最后一个 city_block 无活动）",
            })
            continue

        last_activity = activities[-1]
        if last_activity.get("type") != "hotel":
            # Whether the day has any hotel (it may sit in a non-final block)
            has_any_hotel = False
            for seg in segments:
                if "activities" not in seg:
                    continue
                for act in seg["activities"]:
                    if act.get("type") == "hotel":
                        has_any_hotel = True
                        break
                if has_any_hotel:
                    break

            if not has_any_hotel:
                issues.append({
                    "type": "missing_hotel",
                    "day": day_num,
                    "message": f"Day {day_num}: 未安排住宿",
                })
            else:
                issues.append({
                    "type": "hotel_not_last_activity",
                    "day": day_num,
                    "message": (
                        f"Day {day_num}: hotel 未安排在当天最后一个 city_block 的最后一个活动位置"
                    ),
                })


# =============================================================================
# Check 3: activity time order is sensible
# =============================================================================

def _check_time_order(plan: Dict, issues: List[Dict]) -> None:
    """Check activity times (start_time < end_time; overnight hotel and intercity_transport are exempt)."""
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if segment.get("type") == "intercity_transport":
                # intercity_transport may span midnight (e.g. a night flight 21:20-00:10)
                _validate_time_range(segment, day_num, "intercity_transport", issues, allow_overnight=True)
            elif "activities" in segment:
                for act in segment["activities"]:
                    act_type = act.get("type", "")
                    # hotel may span midnight (e.g. 20:00-08:00)
                    if act_type == "hotel":
                        continue
                    _validate_time_range(act, day_num, act_type, issues)


def _validate_time_range(
    activity: Dict, day_num: int, act_type: str, issues: List[Dict],
    allow_overnight: bool = False,
) -> None:
    """
    Validate a single activity's time range.

    Args:
        allow_overnight: If True, allow end < start (overnight cases such as night flights).
    """
    start_str = activity.get("start_time", "")
    end_str = activity.get("end_time", "")

    start = _parse_time(start_str)
    end = _parse_time(end_str)

    if not start or not end:
        issues.append({
            "type": "invalid_time_format",
            "day": day_num,
            "activity_type": act_type,
            "location": activity.get("location", activity.get("to", "")),
            "start_time": start_str,
            "end_time": end_str,
            "message": (
                f"Day {day_num}: {act_type} "
                f"'{activity.get('location', activity.get('to', ''))}' "
                f"时间格式无效 ({start_str} - {end_str})"
            ),
        })
        return

    if end <= start:
        # Overnight activities (e.g. a night flight 21:20→00:10) are not flagged
        if allow_overnight:
            return
        issues.append({
            "type": "invalid_time_range",
            "day": day_num,
            "activity_type": act_type,
            "location": activity.get("location", activity.get("to", "")),
            "start_time": start_str,
            "end_time": end_str,
            "message": (
                f"Day {day_num}: {act_type} "
                f"'{activity.get('location', activity.get('to', ''))}' "
                f"结束时间不晚于开始时间 ({start_str} - {end_str})"
            ),
        })


# =============================================================================
# Check 4: overlapping activities for the same user
# =============================================================================

def _check_time_overlap(
    plan: Dict, user_names: List[str], issues: List[Dict]
) -> None:
    """
    Check that a user's activities do not overlap in time.

    Reuses ``group_utility._find_invalid_split_activities`` as the single
    source of detection and converts conflicts into validity issues.
    """
    # Reuse group_utility's overlap detection to collect overlapping pairs
    # Also re-run the check here to attach concrete conflict pairs to the issue
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue

            activities = segment["activities"]

            # Group by user and check each user's activities for overlap
            for user in user_names:
                user_activities = []
                for act in activities:
                    act_type = act.get("type", "")
                    # Overnight hotels are excluded from overlap detection
                    if act_type == "hotel":
                        continue
                    participants = _get_participants(act, user_names)
                    if user in participants:
                        user_activities.append(act)

                # Check pairwise overlap
                for i in range(len(user_activities)):
                    for j in range(i + 1, len(user_activities)):
                        act_i = user_activities[i]
                        act_j = user_activities[j]
                        if _time_ranges_overlap(
                            act_i.get("start_time", ""),
                            act_i.get("end_time", ""),
                            act_j.get("start_time", ""),
                            act_j.get("end_time", ""),
                        ):
                            issues.append({
                                "type": "time_overlap",
                                "day": day_num,
                                "user": user,
                                "activity_1": {
                                    "type": act_i.get("type"),
                                    "location": act_i.get(
                                        "location", act_i.get("to", "")
                                    ),
                                    "time": (
                                        f"{act_i.get('start_time', '')}"
                                        f"-{act_i.get('end_time', '')}"
                                    ),
                                },
                                "activity_2": {
                                    "type": act_j.get("type"),
                                    "location": act_j.get(
                                        "location", act_j.get("to", "")
                                    ),
                                    "time": (
                                        f"{act_j.get('start_time', '')}"
                                        f"-{act_j.get('end_time', '')}"
                                    ),
                                },
                                "message": (
                                    f"Day {day_num}: {user} 的活动时间冲突 - "
                                    f"'{act_i.get('location', act_i.get('to', ''))}' "
                                    f"({act_i.get('start_time', '')}-{act_i.get('end_time', '')}) "
                                    f"与 '{act_j.get('location', act_j.get('to', ''))}' "
                                    f"({act_j.get('start_time', '')}-{act_j.get('end_time', '')})"
                                ),
                            })


# =============================================================================
# Check 5: attraction opening hours
# =============================================================================

def _check_opening_hours(plan: Dict, issues: List[Dict]) -> None:
    """
    Check that attractions are visited during opening hours.

    Logic:
    1. Walk every activity with ``type="attraction"``
    2. Resolve that day's opening windows via ``POIOpeningHoursResolver``
    3. Require ``start_time``–``end_time`` to lie entirely inside a window
    4. Record an issue when it does not

    Skip when:
    - the POI has no opening-hours data (``opening_hours`` is null)
    - the POI is always open (``always_open=true``)
    - the POI cannot be matched
    - the plan has no date
    """
    resolver = POIOpeningHoursResolver.get_instance()

    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        date_str = day.get("date", "")

        if not date_str:
            # No date; skip opening-hours checks for this day
            continue

        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue

            city = segment.get("city", "")
            if not city:
                continue

            for act in segment["activities"]:
                if act.get("type") != "attraction":
                    continue

                location = act.get("location", "")
                start_time = act.get("start_time", "")
                end_time = act.get("end_time", "")

                if not location or not start_time or not end_time:
                    continue

                result = resolver.is_within_opening_hours(
                    location, city, date_str, start_time, end_time
                )

                if result is None:
                    # Cannot decide (no data / open all day); skip
                    continue

                if result is False:
                    # Fetch the concrete opening windows for the error message
                    windows = resolver.get_time_windows(location, city, date_str)
                    if windows is not None and len(windows) == 0:
                        # Closed that day (weekday does not match)
                        issues.append({
                            "type": "outside_opening_hours",
                            "day": day_num,
                            "date": date_str,
                            "location": location,
                            "city": city,
                            "activity_time": f"{start_time}-{end_time}",
                            "opening_hours": "当天不开放",
                            "message": (
                                f"Day {day_num} ({date_str}): "
                                f"'{location}' 当天不开放"
                            ),
                        })
                    else:
                        # Time falls outside the opening window
                        window_str = ", ".join(
                            f"{w['open']}-{w['close']}" for w in (windows or [])
                        )
                        issues.append({
                            "type": "outside_opening_hours",
                            "day": day_num,
                            "date": date_str,
                            "location": location,
                            "city": city,
                            "activity_time": f"{start_time}-{end_time}",
                            "opening_hours": window_str,
                            "message": (
                                f"Day {day_num} ({date_str}): "
                                f"'{location}' 的活动时间 {start_time}-{end_time} "
                                f"不在开放时间 {window_str} 内"
                            ),
                        })


# =============================================================================
# Helpers
# =============================================================================

def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse an HH:MM time string."""
    try:
        return datetime.strptime(time_str, "%H:%M")
    except (ValueError, TypeError):
        return None


def _time_ranges_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Return whether two time ranges overlap (overnight ranges are not handled)."""
    t1_start = _parse_time(start1)
    t1_end = _parse_time(end1)
    t2_start = _parse_time(start2)
    t2_end = _parse_time(end2)

    if not all([t1_start, t1_end, t2_start, t2_end]):
        return False

    # Overnight ranges are excluded from overlap checks
    if t1_end <= t1_start or t2_end <= t2_start:
        return False

    return t1_start < t2_end and t2_start < t1_end


def _get_participants(activity: Dict, all_users: List[str]) -> List[str]:
    """Return the activity's participant list."""
    participants = activity.get("participants", [])
    if not participants or participants == ["All"]:
        return all_users
    return participants


# =============================================================================
# Check 6: intra-city transport continuity (per-user tracking, including splits)
# =============================================================================

def _check_intracity_transport_continuity(
    plan: Dict, user_names: List[str], issues: List[Dict]
) -> None:
    """
    Check intra-city transport continuity: within a ``city_block``,
    each user's consecutive activity locations must connect.

    Core logic:
    - Track each user's current location independently
    - Every activity the user joins has a start location and an end location
    - If the current location ≠ the next activity's start, and no
      ``intracity_transport`` bridges them, record an issue
    - Under split-group, only activities that user joins update their location

    Location rules:
    - attraction / food / hotel: start = end = ``location``
    - intracity_transport: start = ``from``, end = ``to``
    - rest: location unchanged (on-the-spot rest); start = end = previous location
    """
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        segments = day.get("city_segments", [])

        # Track each user's location across segments (intercity_transport resets it)
        user_positions: Dict[str, Optional[str]] = {
            user: None for user in user_names
        }

        for segment in segments:
            if segment.get("type") == "intercity_transport":
                # Intercity transport: reset every user to the arrival hub
                # Location could be tagged as from_city+arrival; we use None to mean
                # "just arrived; the first activity of the next city_block sets the location"
                arrival_point = _get_intercity_arrival_point(segment)
                for user in user_names:
                    user_positions[user] = arrival_point
            elif "activities" in segment:
                _check_continuity_in_city_block(
                    segment, day_num, user_names, user_positions, issues
                )


_INTERCITY_ARRIVAL_SENTINEL = "__intercity_arrival__"

def _get_intercity_arrival_point(segment: Dict) -> Optional[str]:
    """
    Infer an arrival-point marker from an ``intercity_transport``.

    Those segments usually have only ``from_city`` / ``to_city``, not a
    concrete arrival station (e.g. "西安北站"), so a sentinel marks
    "the user just arrived at the new city's transport hub". Later
    continuity checks then require the first activity in the new city
    to be an ``intracity_transport``, or they record a missing-transfer
    issue.
    """
    return _INTERCITY_ARRIVAL_SENTINEL


def _check_continuity_in_city_block(
    segment: Dict,
    day_num: int,
    user_names: List[str],
    user_positions: Dict[str, Optional[str]],
    issues: List[Dict],
) -> None:
    """
    Check location continuity for every user inside one ``city_block``.

    Args:
        segment: The ``city_block`` segment.
        day_num: Day number.
        user_names: All user names.
        user_positions: Mutable map of each user's current location.
        issues: Mutable issue list (appended in place).
    """
    activities = segment.get("activities", [])
    city = segment.get("city", "")

    for activity in activities:
        act_type = activity.get("type", "")
        participants = _get_participants(activity, user_names)

        # Extract the activity's start and end locations
        act_start_location = _get_activity_start_location(activity)
        act_end_location = _get_activity_end_location(activity)

        # Continuity check for every participant of this activity
        for user in participants:
            prev_position = user_positions.get(user)

            # Skip rest (in-place; no location change)
            if act_type == "rest":
                continue

            # Handle the 'just arrived' sentinel: if the first activity is not
            # an intracity_transport, the hub-to-first-destination transfer is missing
            if prev_position == _INTERCITY_ARRIVAL_SENTINEL:
                if act_type != "intracity_transport":
                    issues.append({
                        "type": "missing_intracity_transport",
                        "day": day_num,
                        "city": city,
                        "user": user,
                        "from_location": "交通枢纽(到达点)",
                        "to_location": act_start_location or "(unknown)",
                        "message": (
                            f"Day {day_num} ({city}): {user} 到达新城市后"
                            f" 直接前往 '{act_start_location or '?'}'，"
                            f"缺少从交通枢纽出发的城内交通"
                        ),
                    })
                # Update the location and continue, with or without an explicit transfer
                if act_end_location is not None:
                    user_positions[user] = act_end_location
                continue

            # Skip hotel here (hotel continuity only checks that a transfer leads to it)
            # hotel is special-cased: it has an explicit location
            if act_type == "hotel":
                if (
                    prev_position is not None
                    and act_start_location is not None
                    and prev_position != act_start_location
                ):
                    issues.append({
                        "type": "missing_intracity_transport",
                        "day": day_num,
                        "city": city,
                        "user": user,
                        "from_location": prev_position,
                        "to_location": act_start_location,
                        "message": (
                            f"Day {day_num} ({city}): {user} 从"
                            f" '{prev_position}' 到 '{act_start_location}'"
                            f" 缺少城内交通"
                        ),
                    })
                # After a hotel, the location becomes the hotel location
                if act_end_location:
                    user_positions[user] = act_end_location
                continue

            # For attraction / food / intracity_transport
            if (
                prev_position is not None
                and act_start_location is not None
                and prev_position != act_start_location
                and act_type != "intracity_transport"
            ):
                # Previous location != this activity's start, and this is not a transport activity
                # An intra-city transfer is missing
                issues.append({
                    "type": "missing_intracity_transport",
                    "day": day_num,
                    "city": city,
                    "user": user,
                    "from_location": prev_position,
                    "to_location": act_start_location,
                    "message": (
                        f"Day {day_num} ({city}): {user} 从"
                        f" '{prev_position}' 到 '{act_start_location}'"
                        f" 缺少城内交通"
                    ),
                })

            # If this is an intracity_transport, its from must match the user's current location
            if act_type == "intracity_transport":
                if (
                    prev_position is not None
                    and act_start_location is not None
                    and prev_position != act_start_location
                ):
                    issues.append({
                        "type": "intracity_transport_origin_mismatch",
                        "day": day_num,
                        "city": city,
                        "user": user,
                        "expected_from": prev_position,
                        "actual_from": act_start_location,
                        "message": (
                            f"Day {day_num} ({city}): {user} 的城内交通"
                            f" 起点为 '{act_start_location}'，但用户当前"
                            f" 位于 '{prev_position}'"
                        ),
                    })

            # Update the user's location
            if act_end_location is not None:
                user_positions[user] = act_end_location


def _get_activity_start_location(activity: Dict) -> Optional[str]:
    """
    Return the activity's start location.

    - attraction / food / hotel: ``location``
    - intracity_transport: ``from``
    - rest: ``None`` (location unchanged)
    """
    act_type = activity.get("type", "")
    if act_type in ("attraction", "food", "hotel"):
        return activity.get("location")
    elif act_type == "intracity_transport":
        return activity.get("from")
    return None


def _get_activity_end_location(activity: Dict) -> Optional[str]:
    """
    Return the activity's end location.

    - attraction / food / hotel: ``location``
    - intracity_transport: ``to``
    - rest: ``None`` (location unchanged)
    """
    act_type = activity.get("type", "")
    if act_type in ("attraction", "food", "hotel"):
        return activity.get("location")
    elif act_type == "intracity_transport":
        return activity.get("to")
    return None


# =============================================================================
# Check 7: whole-day timeline (detect time going backward across segments)
# =============================================================================

def _check_day_timeline_order(
    plan: Dict, user_names: List[str], issues: List[Dict]
) -> None:
    """
    Check that, from each user's view, activity times are monotonic
    within a day (no time travel).

    Logic:
    - Flatten every segment of the day (``intercity_transport`` and
      ``city_block`` activities) into a timeline
    - For each user, take their activities in list order and require the
      next ``start_time`` >= the previous ``end_time`` (overnight hotels
      excepted)
    - Time travel across segments is detected as well
    """
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        segments = day.get("city_segments", [])

        # Flatten all activities into an ordered list with start_time / end_time / participants
        timeline_items = _flatten_day_to_timeline(segments, user_names)

        # Check time monotonicity per user
        for user in user_names:
            user_items = [
                item for item in timeline_items if user in item["participants"]
            ]

            prev_end_time = None
            prev_description = None
            for item in user_items:
                # Skip hotel (overnight; end_time may be smaller than start_time)
                if item.get("act_type") == "hotel":
                    # hotel.start_time must still come after the previous activity
                    current_start = _parse_time(item["start_time"])
                    if (
                        prev_end_time is not None
                        and current_start is not None
                        and current_start < prev_end_time
                    ):
                        issues.append({
                            "type": "timeline_order_violation",
                            "day": day_num,
                            "user": user,
                            "prev_activity": prev_description,
                            "curr_activity": item["description"],
                            "prev_end_time": item.get("prev_end_str", ""),
                            "curr_start_time": item["start_time"],
                            "message": (
                                f"Day {day_num}: {user} 的时间线倒流 - "
                                f"'{prev_description}' 结束于 "
                                f"{_format_time(prev_end_time)}，但 "
                                f"'{item['description']}' 开始于 "
                                f"{item['start_time']}"
                            ),
                        })
                    # Do not update prev_end_time for hotel (overnight)
                    continue

                current_start = _parse_time(item["start_time"])
                current_end = _parse_time(item["end_time"])

                if (
                    prev_end_time is not None
                    and current_start is not None
                    and current_start < prev_end_time
                ):
                    issues.append({
                        "type": "timeline_order_violation",
                        "day": day_num,
                        "user": user,
                        "prev_activity": prev_description,
                        "curr_activity": item["description"],
                        "prev_end_time": _format_time(prev_end_time),
                        "curr_start_time": item["start_time"],
                        "message": (
                            f"Day {day_num}: {user} 的时间线倒流 - "
                            f"'{prev_description}' 结束于 "
                            f"{_format_time(prev_end_time)}，但 "
                            f"'{item['description']}' 开始于 "
                            f"{item['start_time']}"
                        ),
                    })

                # Update prev_end_time
                if current_end is not None:
                    # Only update for regular activities where end > start
                    if current_start is None or current_end >= current_start:
                        prev_end_time = current_end
                        prev_description = item["description"]


def _flatten_day_to_timeline(
    segments: List[Dict], all_users: List[str]
) -> List[Dict]:
    """
    Flatten a day's segments into an ordered timeline.

    Each entry contains:
    - start_time, end_time: strings
    - participants: user list
    - act_type: activity type
    - description: text used in error messages
    """
    timeline = []

    for segment in segments:
        if segment.get("type") == "intercity_transport":
            timeline.append({
                "start_time": segment.get("start_time", ""),
                "end_time": segment.get("end_time", ""),
                "participants": all_users,  # intercity_transport is mandatory for everyone
                "act_type": "intercity_transport",
                "description": (
                    f"城际交通 {segment.get('from_city', '')}"
                    f"→{segment.get('to_city', '')}"
                ),
            })
        elif "activities" in segment:
            city = segment.get("city", "")
            for act in segment["activities"]:
                act_type = act.get("type", "")
                participants = _get_participants(act, all_users)

                if act_type == "intracity_transport":
                    description = (
                        f"城内交通 {act.get('from', '')}"
                        f"→{act.get('to', '')}"
                    )
                elif act_type in ("attraction", "food", "hotel"):
                    description = f"{act_type} {act.get('location', '')}"
                elif act_type == "rest":
                    description = "休息"
                else:
                    description = f"{act_type}"

                timeline.append({
                    "start_time": act.get("start_time", ""),
                    "end_time": act.get("end_time", ""),
                    "participants": participants,
                    "act_type": act_type,
                    "description": description,
                })

    return timeline


def _format_time(dt: Optional[datetime]) -> str:
    """Format a ``datetime`` as an HH:MM string."""
    if dt is None:
        return "?"
    return dt.strftime("%H:%M")


# =============================================================================
# Check 8: missing avg_cost
# =============================================================================

def _check_avg_cost_present(plan: Dict, issues: List[Dict]) -> None:
    """
    Check that food / hotel / intercity_transport include ``avg_cost``.

    The prompt requires those three types to fill ``avg_cost``.
    """
    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            # Inspect intercity_transport segments
            if segment.get("type") == "intercity_transport":
                if "avg_cost" not in segment or segment["avg_cost"] is None:
                    issues.append({
                        "type": "missing_avg_cost",
                        "day": day_num,
                        "activity_type": "intercity_transport",
                        "location": (
                            f"{segment.get('from_city', '')}"
                            f"→{segment.get('to_city', '')}"
                        ),
                        "message": (
                            f"Day {day_num}: 城际交通 "
                            f"{segment.get('from_city', '')}→"
                            f"{segment.get('to_city', '')} 缺少 avg_cost"
                        ),
                    })
            # Inspect activities inside city_blocks
            elif "activities" in segment:
                for act in segment["activities"]:
                    act_type = act.get("type", "")
                    if act_type in ("food", "hotel"):
                        if "avg_cost" not in act or act["avg_cost"] is None:
                            issues.append({
                                "type": "missing_avg_cost",
                                "day": day_num,
                                "activity_type": act_type,
                                "location": act.get("location", ""),
                                "message": (
                                    f"Day {day_num}: {act_type} "
                                    f"'{act.get('location', '')}' "
                                    f"缺少 avg_cost"
                                ),
                            })


# =============================================================================
# Check 9: participants validity
# =============================================================================

def _check_participants_validity(
    plan: Dict, user_names: List[str], issues: List[Dict]
) -> None:
    """
    Validate the activity's ``participants`` list:
    - every name must be in ``user_names`` or be ``"All"``
    - ``participants`` must not be an empty list when the field is present
    """
    valid_names = set(user_names) | {"All"}

    for day in plan.get("days", []):
        day_num = day.get("day", 0)
        for segment in day.get("city_segments", []):
            if "activities" not in segment:
                continue

            city = segment.get("city", "")
            for act in segment["activities"]:
                act_type = act.get("type", "")
                # hotel and intercity_transport do not require participants
                if act_type in ("hotel",):
                    continue

                participants = act.get("participants")
                if participants is None:
                    # No participants field; skip
                    # (some types such as hotel do not need it)
                    continue

                if not isinstance(participants, list):
                    issues.append({
                        "type": "invalid_participants",
                        "day": day_num,
                        "city": city,
                        "activity_type": act_type,
                        "location": act.get(
                            "location", act.get("to", "")
                        ),
                        "message": (
                            f"Day {day_num} ({city}): {act_type} "
                            f"'{act.get('location', act.get('to', ''))}' "
                            f"的 participants 不是列表"
                        ),
                    })
                    continue

                if len(participants) == 0:
                    issues.append({
                        "type": "invalid_participants",
                        "day": day_num,
                        "city": city,
                        "activity_type": act_type,
                        "location": act.get(
                            "location", act.get("to", "")
                        ),
                        "message": (
                            f"Day {day_num} ({city}): {act_type} "
                            f"'{act.get('location', act.get('to', ''))}' "
                            f"的 participants 为空列表"
                        ),
                    })
                    continue

                # Check that every name is a valid user
                for name in participants:
                    if name not in valid_names:
                        issues.append({
                            "type": "invalid_participants",
                            "day": day_num,
                            "city": city,
                            "activity_type": act_type,
                            "location": act.get(
                                "location", act.get("to", "")
                            ),
                            "invalid_name": name,
                            "message": (
                                f"Day {day_num} ({city}): {act_type} "
                                f"'{act.get('location', act.get('to', ''))}'"
                                f" 的参与者 '{name}' 不在用户列表中"
                            ),
                        })
