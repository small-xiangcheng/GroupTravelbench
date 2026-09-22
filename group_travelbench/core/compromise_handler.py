"""
Compromise tracking for the multi-user travel planning scenario.

Three preference tables coexist in this benchmark:
  ① original_preferences  — the unmodified input from the dataset
  ② agent_preference_table — what the Agent has inferred (lives in
     `GroupTravelAgent.preference_table`); used to evaluate the Agent's
     information-collection ability.
  ③ effective_preferences  — ① with deterministic compromise edits applied;
     produced and maintained by this module. Acts as the ground truth that
     ② is evaluated against, and (per Q8) also as the preference baseline
     used when scoring the final travel plan.

The user-side prompt requires that whenever a user agrees to compromise,
their reply ends with an implicit marker of the form:

    [<dotted.field.path> : <new_value_in_json>]

Examples:
    [global_constraints.avg_budget : 1800]
    [global_constraints.transport.must : ["飞机"]]
    [city_specific_preferences.西安市.attractions.must_visit : ["兵马俑"]]
    [city_specific_preferences.西安市.food.must_eat : []]

The marker is parsed by `extract_compromise_marker`, applied to ③ via
`CompromiseTracker.apply`, and then stripped from the user's visible
message. The Agent perceives compromise only through:
  (a) the user's natural-language sentence ("好吧，那不去了"), and
  (b) the next preference summary it generates, which is expected to
      reflect the new ground truth in ③.

Constraints (per design):
  - A marker is only honored when the *previous* message was the Agent
    @-mentioning *this* user (i.e. a compromise was actually solicited).
  - Each user may agree to at most `MAX_COMPROMISES_PER_USER` compromises.
    Crossing the limit flips their `compromisable` flag to False and the
    caller should rebuild that simulator's system prompt.
"""

import copy
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from .task_context import task_prefix

logger = logging.getLogger(__name__)

# Maximum number of times a user may *agree* to compromise. Rejections
# do not count. Once exceeded the user becomes non-compromisable for the
# remainder of the conversation.
MAX_COMPROMISES_PER_USER = 2

# Marker syntax:
#   [<path>:<value>]
# Path: dotted segments, each segment may contain CJK chars / digits /
# letters / underscore (e.g. city_specific_preferences.西安市.food.must_eat).
# Value: any JSON literal (list, number, string, bool, null), greedy up to
# the last ']' on the same line so nested lists like ["a","b"] still work.
# NOTE: We use \Z (absolute end of string) instead of $ with re.MULTILINE
# to ensure only the very last line of the message is matched as a marker.
# This prevents false positives from bracket-like text in the message body.
_MARKER_RE = re.compile(
    r"\[\s*([\w\u4e00-\u9fff.]+)\s*:\s*(.+?)\s*\]\s*\Z",
)


def _try_repair_json(raw_value: str) -> Optional[Any]:
    """Attempt to repair common truncated or malformed JSON values.

    LLMs sometimes emit incomplete JSON in compromise markers, e.g.:
      - Lone ``[`` or ``{`` → empty list / dict
      - Unclosed list ``["a", "b"`` → try appending ``]``
      - Unclosed dict ``{"k": "v"`` → try appending ``}``

    Returns the parsed value on success, or None if repair fails.
    """
    stripped = raw_value.strip()

    # Lone bracket → empty container
    if stripped == "[":
        return []
    if stripped == "{":
        return {}

    # Try closing unclosed brackets
    if stripped.startswith("[") and not stripped.endswith("]"):
        try:
            return json.loads(stripped + "]")
        except json.JSONDecodeError:
            pass
    if stripped.startswith("{") and not stripped.endswith("}"):
        try:
            return json.loads(stripped + "}")
        except json.JSONDecodeError:
            pass

    return None


def extract_compromise_marker(text: str) -> Optional[Tuple[str, Any, str]]:
    """Extract the trailing compromise marker from a user message.

    Searches for the *last* line that matches the marker pattern. If found,
    parses the value as JSON. Returns (field_path, parsed_value, raw_match)
    where raw_match is the exact substring to be stripped from the message.

    Returns None if no valid marker is present or the value is not valid
    JSON.
    """
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None

    # Use the last match — the marker is expected at end of message.
    match = matches[-1]
    field_path = match.group(1).strip()
    raw_value = match.group(2).strip()

    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        # Tolerate bare strings without quotes (e.g. "高铁" instead of "\"高铁\"")
        # by treating the raw token as a string only when it looks like one.
        if not raw_value.startswith(("[", "{", "\"", "-")) and not raw_value.replace(
            ".", "", 1
        ).isdigit():
            parsed_value = raw_value
        else:
            # Attempt common truncation repairs before giving up:
            #   - Lone "[" or "{" → treat as empty list / dict
            #   - Unclosed list like '["a", "b"' → try appending ']'
            #   - Unclosed dict like '{"k": "v"' → try appending '}'
            repaired_value = _try_repair_json(raw_value)
            if repaired_value is not None:
                parsed_value = repaired_value
                logger.info(
                    f"[Compromise] repaired malformed JSON value: "
                    f"{raw_value!r} -> {parsed_value!r}"
                )
            else:
                logger.warning(
                    f"{task_prefix()}[Compromise] marker value is not valid "
                    f"JSON: {raw_value!r}; ignoring marker."
                )
                return None

    return field_path, parsed_value, match.group(0)


def strip_compromise_marker(text: str, raw_match: str) -> str:
    """Remove the matched marker substring (and any trailing newlines) from text."""
    stripped = text.replace(raw_match, "", 1)
    # Clean up dangling whitespace / blank trailing line.
    return stripped.rstrip()


def _normalize_field_path(field_path: str) -> str:
    """Normalize a dotted field path emitted by the LLM.

    Handles two common LLM error patterns:

    1. **Adjacent duplicate segments** (stutter):
       - ``global_constraints.hotel_preference.hotel_preference.prefer``
         → ``global_constraints.hotel_preference.prefer``
       - ``city_specific_preferences.city_specific_preferences.万宁市.food.reject_eat``
         → ``city_specific_preferences.万宁市.food.reject_eat``

    2. **Underscore-joined garbled segments**: the LLM concatenates a known
       canonical segment name with extra fragments via underscore instead of
       using a dot separator, producing invalid compound tokens like:
       - ``city_specific_preferences_specific_preferences``
         → should be ``city_specific_preferences``
       - ``global_constraints_constraints``
         → should be ``global_constraints``

       The algorithm checks each segment against a set of known canonical
       names; if a segment *starts with* a canonical name but has a trailing
       underscore-joined suffix, it is replaced by the canonical name.

    Non-adjacent duplicates (which may be legitimate, e.g. a dict whose key
    happens to match a parent key) are left untouched.
    """
    # Known canonical segment names in the preference schema. These are
    # used to detect garbled underscore-joined compounds.
    _CANONICAL_SEGMENTS = {
        "global_constraints",
        "city_specific_preferences",
        "hotel_preference",
        "category_pref",
        "must_visit",
        "reject_visit",
        "must_eat",
        "prefer_eat",
        "avoid_eat",
        "reject_eat",
        "max_poi_per_day",
        "max_active_hours",
        "avg_budget",
    }

    segments = field_path.split(".")
    if not segments:
        return field_path

    # Pass 1: fix underscore-joined garbled segments.
    # e.g. "city_specific_preferences_specific_preferences" → "city_specific_preferences"
    repaired_segments = []
    for seg in segments:
        fixed = seg
        for canonical in _CANONICAL_SEGMENTS:
            if (
                seg.startswith(canonical + "_")
                and seg != canonical
                and len(seg) > len(canonical) + 1
            ):
                # The segment starts with a canonical name + "_" + junk.
                # Replace with just the canonical name.
                fixed = canonical
                break
        repaired_segments.append(fixed)

    # Pass 2: collapse adjacent identical segments (stutter).
    deduped = [repaired_segments[0]]
    for segment in repaired_segments[1:]:
        if segment != deduped[-1]:
            deduped.append(segment)

    normalized = ".".join(deduped)
    if normalized != field_path:
        logger.info(
            f"[Compromise] normalized field path: "
            f"'{field_path}' -> '{normalized}'"
        )
    return normalized


def _set_by_path(
    container: Dict[str, Any],
    field_path: str,
    new_value: Any,
) -> bool:
    """Set ``container[field_path] = new_value`` using a dotted path.

    Path segments traverse nested dicts (or dict-of-dict for
    city_specific_preferences). The leaf is overwritten with new_value.
    Returns True on success, False if any intermediate segment is missing.
    """
    segments = field_path.split(".")
    if not segments:
        return False

    cursor: Any = container
    for seg in segments[:-1]:
        if not isinstance(cursor, dict) or seg not in cursor:
            return False
        cursor = cursor[seg]

    leaf = segments[-1]
    if not isinstance(cursor, dict):
        return False

    cursor[leaf] = new_value
    return True


class CompromiseTracker:
    """Manages effective_preferences (③) and per-user compromise counts.

    Lifecycle:
      1. Construct from the original `user_preferences` dict (deep-copied).
      2. After detecting a compromise marker on a user reply that was
         solicited by Agent (`previous Agent message @-mentioned this user`),
         call `apply(user, field_path, new_value, round_idx)`.
      3. Inspect `is_exhausted(user)` to decide whether to flip a simulator's
         compromisable flag.
      4. After the conversation ends, expose `effective_preferences` and
         `compromise_log` for evaluation.
    """

    def __init__(self, original_user_preferences: Dict[str, Any]):
        # ① snapshot — kept untouched for reference by callers if needed.
        self._original = copy.deepcopy(original_user_preferences)
        # ③ — starts as a deep copy of ① and accumulates compromise edits.
        # We mirror the input shape: {user: {preference: {...}, compromisable: bool, ...}}
        self._effective: Dict[str, Any] = copy.deepcopy(original_user_preferences)
        # Per-user agreed-compromise counter (rejections not counted).
        self._counts: Dict[str, int] = {user: 0 for user in original_user_preferences}
        # Audit trail of all applied compromises.
        self._log: List[Dict[str, Any]] = []

    # -- queries -----------------------------------------------------------

    @property
    def effective_preferences(self) -> Dict[str, Any]:
        """The ③ table after all applied compromises (live reference)."""
        return self._effective

    @property
    def compromise_log(self) -> List[Dict[str, Any]]:
        """Audit log of compromises that were actually applied."""
        return self._log

    def get_count(self, user: str) -> int:
        return self._counts.get(user, 0)

    def is_exhausted(self, user: str) -> bool:
        """True iff `user` has reached the per-user compromise cap."""
        return self.get_count(user) >= MAX_COMPROMISES_PER_USER

    # -- mutation ----------------------------------------------------------

    def apply(
        self,
        user: str,
        field_path: str,
        new_value: Any,
        round_idx: int,
    ) -> bool:
        """Apply a compromise to user's slot in ③.

        The field path is resolved against the *preference* sub-dict
        (i.e. `self._effective[user]["preference"]`), which is where
        global_constraints / city_specific_preferences live.

        Returns True if the field existed and was overwritten; False if the
        path could not be resolved (in which case ③ is left unchanged but
        the attempt is still logged for debugging).
        """
        if user not in self._effective:
            logger.warning(
                f"{task_prefix()}[Compromise] unknown user '{user}'; skipping."
            )
            return False

        # Hard cap: even if the user simulator ignores its
        # `compromisable=False` prompt and still emits a valid compromise
        # marker, refuse to apply it once the per-user quota is reached.
        # This is the data-layer safety net behind the prompt-level soft
        # constraint, guaranteeing ③ can never be mutated more than
        # MAX_COMPROMISES_PER_USER times per user.
        if self.is_exhausted(user):
            logger.warning(
                f"{task_prefix()}[Compromise] {user} has already reached the "
                f"per-user cap ({MAX_COMPROMISES_PER_USER}); rejecting further "
                f"compromise on '{field_path}'."
            )
            return False

        user_slot = self._effective[user]
        # Defensive: original input format is
        #   {"User1": {"preference": {...}, "compromisable": bool, ...}}
        # but tolerate a flatter shape too.
        target_root = (
            user_slot["preference"]
            if isinstance(user_slot, dict) and "preference" in user_slot
            else user_slot
        )

        # Step 0: Normalize the path — collapse consecutive duplicate
        # segments that LLMs occasionally stutter out.
        field_path = _normalize_field_path(field_path)

        ok = _set_by_path(target_root, field_path, new_value)

        # Fallback: models sometimes omit intermediate path segments.
        # Common cases:
        #   - "hotel_preference.prefer" → should be
        #     "global_constraints.hotel_preference.prefer"
        #   - "西安市.attractions.must_visit" → should be
        #     "city_specific_preferences.西安市.attractions.must_visit"
        # Try auto-prepending known parent segments when the direct path fails.
        resolved_path = field_path
        if not ok:
            fallback_prefixes = (
                "global_constraints.",
                "city_specific_preferences.",
            )
            for prefix in fallback_prefixes:
                if not field_path.startswith(prefix):
                    candidate = prefix + field_path
                    if _set_by_path(target_root, candidate, new_value):
                        ok = True
                        resolved_path = candidate
                        logger.info(
                            f"[Compromise] path '{field_path}' not found; "
                            f"resolved via fallback to '{resolved_path}'."
                        )
                        break

        log_entry: Dict[str, Any] = {
            "round": round_idx,
            "user": user,
            "field_path": resolved_path,
            "new_value": new_value,
            "applied": ok,
        }
        self._log.append(log_entry)

        if ok:
            self._counts[user] = self._counts.get(user, 0) + 1
            logger.info(
                f"{task_prefix()}[Compromise] applied: {user}.{resolved_path} "
                f"-> {new_value!r} "
                f"(count={self._counts[user]}/{MAX_COMPROMISES_PER_USER})"
            )
        else:
            logger.warning(
                f"{task_prefix()}[Compromise] field path not found: "
                f"{user}.{field_path}; marker ignored, no state change."
            )

        return ok
