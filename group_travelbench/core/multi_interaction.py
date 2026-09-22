"""
Multi-user travel planning interaction logic.

Orchestrates the full multi-user conversation flow:
  Phase 1: User initial statements (pre-generated, becomes shared context)
  Phase 2: Agent summary + free interaction (polling, @mention, convergence)

Key mechanisms:
  - Shared context: all roles share the same chat history (role: message format)
  - Priority on each loop iteration: @mention preempt > timed convergence
    (only at "round boundary") > polling.
  - Polling: round-robin among User1..UserN, Agent. One *round* = one full
    pass through `polling_order`, i.e. each polling slot has spoken once.
    @-mention preempts do NOT consume a polling slot, so a round may
    contain more than `len(polling_order)` events when preempt chains
    occur.
  - Convergence: every `convergence_interval` rounds (after a round
    boundary is reached), Agent summarizes preferences + conflicts.
  - Termination: Agent emits a travel_plan, or max_turn (in rounds)
    is reached.

Counters:
  - `total_rounds` (a.k.a. round): counts only complete polling laps;
    drives `max_turn` and `convergence_interval`.
  - `total_events` (a.k.a. event_seq): global event counter incremented on
    every speaking event (including User [pass], @-mention preempts, the
    opening convergence message, in-loop convergence messages, and the
    max_turn fallback final plan). Used for fine-grained logging,
    compromise event ordering, and per-round runaway protection.

Pass policy:
  - Only Users may emit [pass]; the Agent has no pass mechanism and is
    expected to always produce a substantive turn (ask, mediate, call
    tools, or emit the final plan).
  - A User [pass] still counts as one event and (if it was a polling
    slot, not a preempt) still consumes that polling slot.

Per-round runaway protection:
  - To bound the cost of long @-mention preempt chains, each round has a
    hard cap on its event count (`MAX_EVENTS_PER_ROUND_FACTOR` ×
    `len(polling_order)`). When hit, the round is force-closed (round
    counter advances, `polling_index` resets to 0).
"""

import copy
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from .config import OpenAIConfig, BenchmarkConfig
from .compromise_handler import (
    CompromiseTracker,
    extract_compromise_marker,
    strip_compromise_marker,
    _MARKER_RE,
)
from .task_context import set_task_context, task_prefix
from ..agents.group_agent import GroupTravelAgent
from ..simulators.multi_user_simulator import MultiUserSimulator

# Import tools to trigger automatic registration with sandbox_tool_registry
# (tools/__init__.py calls register_all_tools() on import)
from ..tools import *  # noqa: F401, F403
from ..tools.tool_list import TOOL_NAMES
from ..utils.multi_util import format_context

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

AGENT_ROLE_NAME = "Agent"

# When a participant is being @-mentioned, they MUST not output [pass].
# If the LLM still outputs [pass], we retry generation up to this many times.
MAX_MENTION_PASS_RETRIES = 2

# Pattern matching one or more trailing `[pass]` tokens (each possibly on its
# own line, possibly surrounded by whitespace) at the very end of a message.
# Used by `_strip_trailing_pass_tokens` to remove residual `[pass]` markers
# that user simulators occasionally append after a substantive reply (an
# observed quirk of LLMs given the dual instruction "answer substantively
# OR output [pass]"). Such residue, if left in `conversation_history`, both
# leaks the pass-token convention into the Agent's view of the chat and
# misrepresents what the user actually said. We strip it conservatively:
# only contiguous trailing `[pass]` tokens are removed; anything earlier
# in the message is left untouched.
_TRAILING_PASS_RE = re.compile(r"(?:\s*\[pass\]\s*)+\Z", re.IGNORECASE)

def _strip_trailing_pass_tokens(text: str) -> str:
    """Strip contiguous trailing `[pass]` tokens from a user message.

    Leaves a message that is *entirely* `[pass]` (after trimming) untouched
    so that `MultiUserSimulator.check_is_pass` still classifies it as a
    pass. Concretely: if removing the trailing pass tokens would leave an
    empty string, the original text is returned unchanged.
    """
    stripped = _TRAILING_PASS_RE.sub("", text)
    if not stripped.strip():
        # Whole message was just `[pass]` (possibly repeated) — preserve
        # the original so the pass-detection path fires normally.
        return text
    return stripped.rstrip()

# Per-round event-count cap (in units of `len(polling_order)`). A round is
# force-closed once its event count reaches
# `MAX_EVENTS_PER_ROUND_FACTOR * len(polling_order)`. This protects against
# pathological @-mention preempt chains turning a single round into an
# unbounded conversation.
MAX_EVENTS_PER_ROUND_FACTOR = 5


def _has_new_visible_messages_since_last_speak(
    speaker: str,
    conversation_history: List[Dict[str, Any]],
) -> bool:
    """Check if there are new messages visible to `speaker` since their last utterance.

    A "visible message" for a User is any entry with:
      - type == "message"
      - has_tool_calls is falsy (not Agent's intermediate reasoning)
      - role != speaker (other people's messages)

    If the speaker has never spoken (no entry found), return True (they
    should get a chance to speak). If they spoke and there are no new
    visible messages after that point, return False (nothing new to react
    to — skip their turn to avoid the model hallucinating on stale context).

    This is ONLY applied to User speakers during polling (not @mention
    preempts, not the Agent). The Agent always speaks when polled.
    """
    # Find the index of this speaker's last message in history
    last_speak_index = -1
    for idx in range(len(conversation_history) - 1, -1, -1):
        entry = conversation_history[idx]
        if entry["type"] == "message" and entry["role"] == speaker:
            last_speak_index = idx
            break

    if last_speak_index < 0:
        # Speaker has never spoken — they should get a chance
        return True

    # Check if any visible message exists after last_speak_index
    for idx in range(last_speak_index + 1, len(conversation_history)):
        entry = conversation_history[idx]
        if entry["type"] != "message":
            continue
        if entry.get("has_tool_calls"):
            continue
        if entry["role"] != speaker:
            return True

    return False


def _generate_speaker_response(
    speaker: str,
    is_being_mentioned: bool,
    agent: GroupTravelAgent,
    user_simulators: Dict[str, "MultiUserSimulator"],
    conversation_history: List[Dict[str, Any]],
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Generate one response from the given speaker.

    Pass semantics:
      - The Agent has no pass mechanism; its output is always treated as a
        substantive message (we never check Agent output for [pass]).
      - For Users, [pass] is a legal default in normal turns. However,
        when a User is being @-mentioned, [pass] violates the contract
        and we retry generation up to `MAX_MENTION_PASS_RETRIES` times
        before giving up.

    Returns:
        (response_text, num_steps, tool_entries)
        tool_entries is always empty for User speakers.
    """
    # Agent path: no pass check, no retry. Agent is forbidden from
    # passing by its own prompt, and the orchestrator never inspects
    # Agent output for [pass] anywhere.
    if speaker == AGENT_ROLE_NAME:
        return agent.generate_response(
            conversation_history=conversation_history,
            is_being_mentioned=is_being_mentioned,
        )

    # User path.
    simulator = user_simulators[speaker]

    # Free-turn (not @-mentioned): single shot, [pass] is a legitimate reply.
    if not is_being_mentioned:
        response_text = simulator.generate_response(
            conversation_history=conversation_history,
            is_being_mentioned=False,
        )
        return response_text, 1, []

    # Mention contract: must answer substantively. Retry on [pass].
    attempts = 1 + MAX_MENTION_PASS_RETRIES
    response_text = ""
    for attempt in range(1, attempts + 1):
        response_text = simulator.generate_response(
            conversation_history=conversation_history,
            is_being_mentioned=True,
        )
        if not MultiUserSimulator.check_is_pass(response_text):
            return response_text, attempt, []
        # Include task_id and the last message that triggered the @mention
        # so operators can diagnose *which* task and *what context* caused
        # the user to pass when they shouldn't have.
        prev_msg = conversation_history[-1] if conversation_history else {}
        prev_role = prev_msg.get("role", "?")
        prev_content = prev_msg.get("content", "")
        logger.warning(
            f"{task_prefix()}[Mention] {speaker} returned [pass] while being "
            f"@-mentioned (attempt {attempt}/{attempts}); regenerating. "
            f"| prev_msg: [{prev_role}] {prev_content!r}"
        )

    # All retries exhausted — return the last (still-pass) reply. The
    # outer loop will treat it as a normal user pass.
    return response_text, attempts, []

# ============================================================================
# Helper functions
# ============================================================================

def build_user_list_description(user_names: List[str]) -> str:
    """Build a human-readable description of all participants."""
    lines = [f"- {name}" for name in user_names]
    return "\n".join(lines)

def detect_mention(text: str, all_role_names: List[str]) -> Optional[str]:
    """
    Detect the first @mention in a message.

    Args:
        text: The message text to scan.
        all_role_names: List of all valid role names (e.g., ["User1", "User2", "Agent"]).

    Returns:
        The mentioned role name, or None if no valid mention found.
    """
    # Match @RoleName pattern (case-sensitive). We can't use \b because the
    # ASCII word-boundary breaks down when the @-mention is followed by a
    # CJK character (e.g. "@User1你怎么想？") — \b only matches at
    # word/non-word transitions and CJK chars are non-word, so depending on
    # the *previous* char the boundary may or may not fire. Replace with an
    # explicit "next char is not a word char" lookahead, which behaves the
    # same for ASCII-on-ASCII boundaries and correctly fires for CJK-after.
    #
    # Also: sort role names by length DESC before joining so that longer
    # names win when one is a prefix of another (e.g. "User1" vs "User10").
    #
    # Search from the END of the message (rightmost match). This ensures
    # that in messages like "@User2 查询结果如下... @User3 你能接受吗？",
    # the *last* @mention (the actual question target) is detected rather
    # than the first (which is often just a reference in a statement).
    sorted_names = sorted(all_role_names, key=len, reverse=True)
    pattern = (
        r"@("
        + "|".join(re.escape(name) for name in sorted_names)
        + r")(?![A-Za-z0-9_])"
    )
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None

def get_last_message_entry(
    history: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Get the last *outward* 'message' type entry from the conversation
    history.

    "Outward" excludes Agent messages with `has_tool_calls=True`, which
    are intermediate reasoning steps emitted between tool calls and not
    visible to users. @mention detection, compromise validation, and
    `truncate_history_before_last_message` all want the last
    user-visible message — they must not see Agent reasoning, otherwise
    e.g. an "@User1 让我先查一下" buried inside a reasoning step would
    be misclassified as an outward @-mention.
    """
    for entry in reversed(history):
        if entry["type"] != "message":
            continue
        if entry.get("has_tool_calls"):
            continue
        return entry
    return None

def truncate_history_before_last_message(
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a copy of `history` with the last message entry and any of its
    trailing tool entries removed.

    Used by the post-plan preference refresh: ② should reflect what the Agent
    knew *before* deciding to emit the travel_plan, so the plan message
    itself (and the tool calls it triggered) must be excluded from the
    refresh prompt.
    """
    # Find index of the last message entry.
    last_msg_idx = -1
    for idx in range(len(history) - 1, -1, -1):
        if history[idx]["type"] == "message":
            last_msg_idx = idx
            break

    if last_msg_idx < 0:
        return list(history)

    # Drop everything from that message onward (tool entries trailing it
    # would otherwise survive without their owning message and confuse
    # format_context).
    return history[:last_msg_idx]

# ============================================================================
# Phase 2: Free Interaction Loop
# ============================================================================

def _maybe_apply_compromise_from_user_reply(
    speaker: str,
    response_text: str,
    conversation_history: List[Dict[str, Any]],
    compromise_tracker: CompromiseTracker,
    user_simulators: Dict[str, MultiUserSimulator],
    event_seq: int,
) -> Tuple[str, bool]:
    """Detect a compromise marker on a user reply, apply it if legitimate,
    and return the message text *with the marker stripped*.

    Legitimacy rules (per Q7):
      - The marker is honored only when the previous message in
        `conversation_history` is from `Agent` AND that message
        @-mentioned the current `speaker` (i.e. compromise was solicited).
      - Anything output by the Agent itself or by users not under such an
        @-mention is treated as an illegal marker: it is still stripped
        from the message (so it won't pollute downstream context), but
        ③ is not modified and the count is not incremented.

    Side effects on success:
      - Apply the compromise to `compromise_tracker` (③ + count + log).
      - If the user has just hit the cap, flip their simulator's
        `compromisable` flag to False (rebuilds that simulator's prompt).

    Returns:
        (cleaned_text, marker_invalid)
        - cleaned_text: message with marker stripped, ready for history.
        - marker_invalid: True when the user's reply contained a compromise
          marker that could NOT be successfully applied (either malformed
          JSON or illegitimate context). The caller should consider
          regenerating the user's reply to avoid polluting history with
          compromise-like natural language that has no matching state change.
    """
    if speaker == AGENT_ROLE_NAME:
        # Agents may not emit compromise markers.
        return response_text, False

    # Pre-check: does the text contain a marker-shaped pattern at all?
    # This lets us distinguish "no marker" (no retry needed) from
    # "marker present but extract_compromise_marker failed" (retry needed).
    has_marker_pattern = bool(_MARKER_RE.search(response_text))

    extracted = extract_compromise_marker(response_text)
    if extracted is None:
        # If there was a marker pattern but extraction failed (e.g.
        # malformed JSON value), signal that the reply is tainted.
        if has_marker_pattern:
            logger.warning(
                f"{task_prefix()}[Compromise] {speaker} emitted a marker-like "
                f"pattern but extraction failed (malformed); signaling retry. "
                f"| user_reply: {response_text!r}"
            )
            return response_text, True
        return response_text, False

    field_path, new_value, raw_match = extracted
    cleaned_text = strip_compromise_marker(response_text, raw_match)

    # The Agent must have just @-mentioned this exact user.
    last_msg = get_last_message_entry(conversation_history)
    if last_msg is None or last_msg.get("role") != AGENT_ROLE_NAME:
        prev_role = last_msg.get("role", "None") if last_msg else "None"
        prev_content = (last_msg.get("content", "")) if last_msg else ""
        logger.warning(
            f"{task_prefix()}[Compromise] {speaker} emitted a marker but "
            f"previous message was not from Agent (was from {prev_role}); "
            f"ignoring marker and signaling retry. "
            f"| prev_msg: [{prev_role}] {prev_content!r} "
            f"| user_reply: {response_text!r}"
        )
        return cleaned_text, True

    mentioned = detect_mention(
        last_msg["content"],
        list(user_simulators.keys()) + [AGENT_ROLE_NAME],
    )
    if mentioned != speaker:
        agent_content = last_msg.get("content", "")
        logger.warning(
            f"{task_prefix()}[Compromise] {speaker} emitted a marker but "
            f"Agent did not @-mention them (mentioned='{mentioned}'); "
            f"ignoring marker and signaling retry. "
            f"| agent_msg: {agent_content!r} "
            f"| user_reply: {response_text!r}"
        )
        return cleaned_text, True

    # Legitimate compromise — apply. We pass the global event sequence
    # number as `round_idx` so the compromise log preserves intra-round
    # ordering (the field is treated as an opaque ordinal by the tracker).
    applied = compromise_tracker.apply(
        user=speaker,
        field_path=field_path,
        new_value=new_value,
        round_idx=event_seq,
    )

    if not applied:
        # Field path could not be resolved (e.g. LLM emitted a garbled
        # path like "city_specific_preferences_specific_preferences.X").
        # Signal retry so the user can regenerate with a correct marker.
        logger.warning(
            f"{task_prefix()}[Compromise] {speaker}'s marker had an "
            f"unresolvable field path '{field_path}'; signaling retry."
        )
        return cleaned_text, True

    # If this user has just exhausted their quota, flip the simulator's flag
    # so subsequent prompts make them non-compromisable.
    if compromise_tracker.is_exhausted(speaker):
        sim = user_simulators.get(speaker)
        if sim is not None and sim.compromisable:
            logger.info(
                f"[Compromise] {speaker} reached the per-user cap; "
                "flipping compromisable=False."
            )
            sim.set_compromisable(False)

    return cleaned_text, False

def _commit_speaker_event(
    actual_speaker: str,
    response_text: str,
    speaker_tool_entries: List[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    compromise_tracker: CompromiseTracker,
    user_simulators: Dict[str, MultiUserSimulator],
    event_seq: int,
    round_idx: int,
    agent: GroupTravelAgent,
    plan_generated: bool,
    completion_reason: str,
) -> Tuple[bool, str, bool, bool]:
    """Commit a single speaker event into shared state.

    Centralises the post-generation logic shared by both the @-mention
    preempt path and the polling path:
      - Detect User [pass] (Agent has no pass mechanism).
      - On non-pass: extract+apply any compromise marker, log, append the
        message and any tool entries, opportunistically refresh ②, and
        check whether the Agent emitted a final travel_plan.
      - If the Agent's reply is a bare preference table (not a travel plan),
        treat it as an internal-only update: refresh ② but do NOT commit
        the raw JSON to conversation_history. Signal the caller to trigger
        a convergence summary so the Agent can re-express its state in
        natural language.

    Returns:
        (plan_generated, completion_reason, needs_convergence, needs_user_retry)
        - plan_generated: True when Agent emitted a complete travel plan.
        - completion_reason: updated reason string.
        - needs_convergence: True when Agent output a bare preference table
          in a non-convergence turn; caller should trigger a convergence
          summary (the preference table output was NOT committed to history).
        - needs_user_retry: True when a User's reply contained an invalid
          compromise marker (malformed JSON or illegitimate context). The
          caller should regenerate the user's reply to avoid polluting
          history with compromise-like language that has no state change.
    """
    # Sanitize residual trailing `[pass]` tokens from user replies before
    # any downstream handling. This addresses a recurring LLM quirk where
    # a user simulator outputs a substantive answer and *then* appends a
    # stray `[pass]` (sometimes on a new line). Without this strip, the
    # token would survive into `conversation_history` and into every
    # subsequent context view shown to the Agent and other users. We do
    # not apply the strip to Agent messages: Agent has no [pass] mechanism,
    # so any `[pass]`-looking substring there is meaningful natural-language
    # content that must not be silently mutated.
    if actual_speaker != AGENT_ROLE_NAME:
        response_text = _strip_trailing_pass_tokens(response_text)

    is_pass = (
        actual_speaker != AGENT_ROLE_NAME
        and MultiUserSimulator.check_is_pass(response_text)
    )

    if is_pass:
        logger.info(
            f"[Round {round_idx} / Event {event_seq}] {actual_speaker}: [pass]"
        )
        # [pass] does NOT enter conversation history.
        return plan_generated, completion_reason, False, False

    # Non-pass path.
    # User-side compromise marker: detect, validate, apply to ③, then
    # strip from the visible message before it enters history. Agent
    # messages are passed through unchanged here (markers from Agent
    # are illegal and the helper no-ops on them).
    response_text, marker_invalid = _maybe_apply_compromise_from_user_reply(
        speaker=actual_speaker,
        response_text=response_text,
        conversation_history=conversation_history,
        compromise_tracker=compromise_tracker,
        user_simulators=user_simulators,
        event_seq=event_seq,
    )

    # If the user emitted an invalid compromise marker, signal the caller
    # to retry generation. Do NOT commit this reply to history — the
    # compromise-like natural language would mislead the Agent.
    if marker_invalid:
        logger.warning(
            f"{task_prefix()}[CompromiseRetry] {actual_speaker}'s reply "
            f"contains an invalid compromise marker (round={round_idx}, "
            f"event={event_seq}); signaling caller to retry user generation."
        )
        return plan_generated, completion_reason, False, True

    logger.info(
        f"[Round {round_idx} / Event {event_seq}] {actual_speaker}: "
        f"{response_text[:200]}..."
    )

    # Agent-specific: detect bare preference table output in a normal
    # (non-convergence) turn. The Agent's system prompt does NOT instruct
    # it to output raw preference JSON during free interaction — only
    # during formal convergence rounds. If the model outputs a preference
    # table here, we treat it as an internal-only update:
    #   - Update ② (agent.preference_table) so the info is preserved.
    #   - Commit any preceding tool_call/tool_result entries to history
    #     (they are valuable context), but strip the final message entry
    #     containing the raw preference JSON.
    #   - Signal the caller to trigger a convergence summary so the Agent
    #     can re-express its understanding in natural language.
    #
    # Result in conversation_history (when tool calls exist):
    #   tool_call1, tool_result1, ..., [convergence natural-language reply]
    # Result (when no tool calls):
    #   [convergence natural-language reply]
    if actual_speaker == AGENT_ROLE_NAME:
        updated = agent.try_update_preference_table_from_text(response_text)
        if updated and not GroupTravelAgent.check_is_travel_plan(response_text):
            # Commit tool_call/tool_result entries (everything except the
            # final message entry) so tool context is preserved in history.
            tool_only_entries = [
                entry for entry in speaker_tool_entries
                if entry.get("type") in ("tool_call", "tool_result")
                or entry.get("has_tool_calls") is True
            ]
            if tool_only_entries:
                conversation_history.extend(tool_only_entries)
            logger.info(
                f"{task_prefix()}[PreferenceTableIntercept] Agent output a "
                f"bare preference table in a normal turn (round={round_idx}, "
                f"event={event_seq}); committed {len(tool_only_entries)} tool "
                f"entries, updating ② internally and triggering convergence "
                f"instead of committing the preference JSON to history."
            )
            return plan_generated, completion_reason, True, False

    # Commit to history.
    #
    # Agent path: `speaker_tool_entries` is the chronological
    # `turn_entries` list returned by `GroupTravelAgent.generate_response`,
    # already containing the final outward message (with
    # `has_tool_calls=False`) plus any preceding intermediate reasoning
    # and tool entries. Extend in one shot. Note: compromise marker /
    # `[pass]` stripping above are User-side mechanisms (the strippers
    # short-circuit on Agent), so the in-list final message content
    # equals `response_text` for Agent turns.
    #
    # User path: `speaker_tool_entries` is always empty (Users don't run
    # tools and produce no intermediate messages); synthesize a single
    # outward message entry from `response_text` (which has already been
    # `[pass]`-stripped and compromise-marker-stripped above).
    if actual_speaker == AGENT_ROLE_NAME:
        conversation_history.extend(speaker_tool_entries)
    else:
        conversation_history.append({
            "type": "message",
            "role": actual_speaker,
            "content": response_text,
            "has_tool_calls": False,
        })
    # Termination: Agent emitted a travel_plan -> end immediately. By
    # design, there is no feedback/revision phase: the plan is a
    # one-shot global decision and users have no chance to revise it.
    if (
        actual_speaker == AGENT_ROLE_NAME
        and GroupTravelAgent.check_is_travel_plan(response_text)
    ):
        plan_generated = True
        completion_reason = "plan_generated"
        logger.info(
            "[Stop] Agent generated a travel plan. Conversation complete."
        )

    return plan_generated, completion_reason, False, False

def run_multi_user_interaction(
    agent: GroupTravelAgent,
    user_simulators: Dict[str, MultiUserSimulator],
    user_names: List[str],
    conversation_history: List[Dict[str, Any]],
    compromise_tracker: CompromiseTracker,
    max_turn: int = 15,
    convergence_interval: int = 3,
) -> Dict[str, Any]:
    """
    Run the full multi-user interaction loop (Phase 2).

    Round vs event semantics:
      - One *round* = one complete pass through `polling_order`, i.e. each
        polling slot has spoken once. `total_rounds` advances only when the
        round closes (polling_index has cycled back to 0, OR the per-round
        runaway cap was hit).
      - `total_events` (event_seq) is a finer-grained counter incremented
        on every speaking event, including @-mention preempts (which do
        NOT consume a polling slot), User [pass]es, the opening
        convergence message, in-loop convergence messages, and the
        max_turn fallback final plan.

    Iteration priority (every `while` iteration):
      1. @mention preempt: if the most recent message contains an @ to a
         valid role, that role speaks next. Preempts do NOT consume a
         polling slot and do NOT advance the round counter on their own
         (but they do count toward the per-round event cap).
      2. Timed convergence: only checked at a *round boundary* (i.e. when
         `polling_index == 0` and `total_rounds > 0`); fires when
         `total_rounds % convergence_interval == 0`.
      3. Polling: speak in `polling_order`, advance `polling_index`, and
         when `polling_index` wraps to 0 the round closes
         (`total_rounds += 1`).

    Termination conditions:
      - The Agent emits a complete travel_plan JSON (immediate stop, no
        feedback/revision phase).
      - `total_rounds` reaches `max_turn`, in which case the Agent is
        forced to produce a final plan as a fallback.

    Pass policy:
      - Only Users may emit [pass]; the Agent has no pass mechanism and
        every Agent turn unconditionally enters the conversation history.
      - A User [pass] is one event but is not appended to the
        conversation history. If it was a polling slot (not a preempt),
        it still consumes that slot.

    Args:
        agent: The GroupTravelAgent instance.
        user_simulators: Dict mapping user_name -> MultiUserSimulator.
        user_names: Ordered list of user names.
        conversation_history: The global structured conversation history (mutated in-place).
        compromise_tracker: Tracker for ③ effective_preferences and
            per-user compromise counts. Mutated as users agree to compromises.
        max_turn: Maximum number of *rounds* (complete polling laps).
        convergence_interval: Trigger an Agent convergence summary every
            this-many rounds (checked at round boundary).

    Returns:
        Result dict containing conversation history, agent trace, statistics, etc.
    """
    if max_turn < 1:
        raise ValueError(f"max_turn must be >= 1, got {max_turn}")
    if convergence_interval < 1:
        raise ValueError(
            f"convergence_interval must be >= 1, got {convergence_interval}"
        )

    start_time = time.time()

    # All valid role names for @mention detection
    all_role_names = user_names + [AGENT_ROLE_NAME]

    # Polling order: User1, User2, ..., UserN, Agent
    polling_order = user_names + [AGENT_ROLE_NAME]
    polling_index = 0  # Current position in polling order

    # Per-round runaway cap: bounds long @-mention preempt chains.
    max_events_per_round = MAX_EVENTS_PER_ROUND_FACTOR * len(polling_order)

    # Counters
    total_rounds = 0   # Completed polling laps; gates max_turn / convergence
    total_events = 0   # Global event sequence (every speaking event)
    events_in_current_round = 0  # Resets on round close
    total_steps = 0    # Total LLM call steps (Agent may use multiple per turn)
    plan_generated = False
    completion_reason = "unknown"

    # Track consumed @mentions to prevent infinite preempt loops.
    # When a mention target exhausts pass-retries (or successfully responds),
    # we record (id(last_msg), target) so the same message won't re-trigger
    # the same preempt on subsequent loop iterations. This is necessary
    # because [pass] does not enter conversation_history, meaning
    # get_last_message_entry() keeps returning the same Agent message.
    consumed_mentions: set = set()

    # Cross-turn repetitive tool-call tracking: if the Agent's
    # `generate_response` flags `_last_turn_repetitive` for
    # `_MAX_CONSECUTIVE_REPETITIVE_TURNS` consecutive Agent turns,
    # terminate the entire task early.
    _MAX_CONSECUTIVE_REPETITIVE_TURNS = 2
    _consecutive_repetitive_agent_turns = 0

    # --- Phase 2 starts: Agent does initial convergence summary ---
    # The Agent has no pass mechanism, so its convergence output is
    # unconditionally appended to conversation_history. This opening
    # summary is a one-off event outside any polling round.
    logger.info("[Phase2] Starting with Agent convergence summary...")
    # Step 1: Agent preference summary (does NOT enter conversation history)
    agent.generate_preference_summary(conversation_history)
    logger.info("[Phase2] Agent preference table updated.")
    # Step 2: Agent convergence summary (DOES enter conversation history)
    agent_response, steps, agent_tool_entries = agent.generate_convergence_summary(
        conversation_history
    )
    total_steps += steps
    total_events += 1
    # `agent_tool_entries` is the chronological `turn_entries` list:
    # it already includes the final outward message entry plus any
    # preceding intermediate reasoning + tool entries. Extend in one shot.
    conversation_history.extend(agent_tool_entries)
    logger.info(
        f"[Event {total_events}] Agent opening summary: "
        f"{agent_response[:200]}..."
    )

    # Opportunistic: if the convergence summary embedded a complete
    # user_preferences JSON, refresh ② for free.
    agent.try_update_preference_table_from_text(agent_response)

    # Edge case: Agent generated a complete plan in the very first
    # convergence summary. Per design, plan emission immediately
    # terminates the conversation — there is no feedback/revision phase.
    if GroupTravelAgent.check_is_travel_plan(agent_response):
        plan_generated = True
        completion_reason = "plan_generated"
        logger.info(
            "[Stop] Agent generated travel plan in opening summary. "
            "Conversation complete."
        )

    # --- Main interaction loop ---
    while not plan_generated and total_rounds < max_turn:
        # ---- Priority 1: @mention preempt ----
        # Detect a still-pending @ from the most recent message. Preempts
        # outrank both timed convergence and polling. They do NOT consume
        # a polling slot, but they DO count toward the per-round event cap
        # (so a runaway preempt chain still gets force-closed).
        last_msg = get_last_message_entry(conversation_history)
        mention_target: Optional[str] = None
        if last_msg is not None:
            # Only Agent messages can trigger @mention preempts. User-to-User
            # @mentions are illegal (prompt forbids them) and must not trigger
            # a preempt — otherwise a User acting like an Agent (e.g. asking
            # another User to compromise) would hijack the conversation flow
            # and cause invalid compromise markers (prev_msg not from Agent).
            if last_msg.get("role") == AGENT_ROLE_NAME:
                candidate = detect_mention(last_msg["content"], all_role_names)
                # Skip if this (message, target) pair was already consumed
                # (e.g. target exhausted pass-retries on a prior iteration).
                # This prevents infinite preempt loops when [pass] doesn't
                # enter history and get_last_message_entry() keeps returning
                # the same Agent message.
                if candidate and (id(last_msg), candidate) not in consumed_mentions:
                    mention_target = candidate

        if mention_target is not None:
            # `mention_target` is only set when `last_msg is not None`,
            # so this assertion is always true; it just narrows the type
            # for the static checker on the `last_msg['role']` access below.
            assert last_msg is not None
            actual_speaker = mention_target
            logger.info(
                f"[Round {total_rounds} / Event {total_events + 1}] "
                f"@mention preempt: {last_msg['role']} -> {actual_speaker}"
            )
            response_text, speaker_steps, speaker_tool_entries = _generate_speaker_response(
                speaker=actual_speaker,
                is_being_mentioned=True,
                agent=agent,
                user_simulators=user_simulators,
                conversation_history=conversation_history,
            )
            total_steps += speaker_steps
            total_events += 1
            events_in_current_round += 1

            # Detect mention-pass-exhaustion BEFORE commit: if the @-ed user
            # is a User and `_generate_speaker_response` exhausted its
            # retries with [pass], the pass is not appended to history,
            # which means `last_msg` remains the soliciting Agent message
            # and the next loop iteration re-triggers the same preempt.
            # That spins until the per-round runaway cap fires, burning a
            # large number of LLM calls silently. Force-advance polling
            # once to break the loop deterministically.
            mention_pass_exhausted = (
                actual_speaker != AGENT_ROLE_NAME
                and MultiUserSimulator.check_is_pass(response_text)
            )

            # Retry loop for invalid compromise markers: if the User's
            # reply contains a malformed/illegitimate marker, regenerate
            # up to _MAX_COMPROMISE_RETRIES total attempts to get a clean
            # reply. On exhaustion, strip the marker and commit to avoid
            # blocking the conversation.
            _MAX_COMPROMISE_RETRIES = 3
            _comp_attempt = 0
            while _comp_attempt < _MAX_COMPROMISE_RETRIES:
                _comp_attempt += 1
                plan_generated, completion_reason, needs_convergence, needs_user_retry = _commit_speaker_event(
                    actual_speaker=actual_speaker,
                    response_text=response_text,
                    speaker_tool_entries=speaker_tool_entries,
                    conversation_history=conversation_history,
                    compromise_tracker=compromise_tracker,
                    user_simulators=user_simulators,
                    event_seq=total_events,
                    round_idx=total_rounds,
                    agent=agent,
                    plan_generated=plan_generated,
                    completion_reason=completion_reason,
                )
                if not needs_user_retry:
                    break
                if _comp_attempt >= _MAX_COMPROMISE_RETRIES:
                    # Exhausted all attempts — the user genuinely intends
                    # to compromise (Agent asked without @mention). Apply
                    # the compromise to ③ to keep state consistent with
                    # what the Agent will perceive from the natural language.
                    logger.warning(
                        f"{task_prefix()}[CompromiseRetry] {actual_speaker} "
                        f"exhausted {_MAX_COMPROMISE_RETRIES} attempts for "
                        f"invalid compromise marker; force-applying compromise "
                        f"and committing stripped reply."
                    )
                    marker_match = _MARKER_RE.search(response_text)
                    committed_text = (
                        strip_compromise_marker(response_text, marker_match.group(0))
                        if marker_match else response_text
                    )
                    # Force-apply the compromise to ③ since the user's
                    # intent is clear even though the @mention check failed.
                    if marker_match:
                        extracted = extract_compromise_marker(response_text)
                        if extracted is not None:
                            field_path, new_value, _raw = extracted
                            applied = compromise_tracker.apply(
                                user=actual_speaker,
                                field_path=field_path,
                                new_value=new_value,
                                round_idx=total_events,
                            )
                            if applied and compromise_tracker.is_exhausted(actual_speaker):
                                sim = user_simulators.get(actual_speaker)
                                if sim is not None and sim.compromisable:
                                    logger.info(
                                        f"[Compromise] {actual_speaker} reached "
                                        "the per-user cap; flipping "
                                        "compromisable=False."
                                    )
                                    sim.set_compromisable(False)
                    conversation_history.append({
                        "type": "message",
                        "role": actual_speaker,
                        "content": committed_text,
                        "has_tool_calls": False,
                    })
                    break
                # Regenerate the user's reply.
                logger.warning(
                    f"{task_prefix()}[CompromiseRetry] Regenerating "
                    f"{actual_speaker}'s reply (attempt {_comp_attempt + 1}/"
                    f"{_MAX_COMPROMISE_RETRIES})."
                )
                response_text, speaker_steps, speaker_tool_entries = _generate_speaker_response(
                    speaker=actual_speaker,
                    is_being_mentioned=True,
                    agent=agent,
                    user_simulators=user_simulators,
                    conversation_history=conversation_history,
                )
                total_steps += speaker_steps

            if plan_generated:
                break

            # Agent output a bare preference table in a normal turn:
            # trigger a convergence summary so it re-expresses in natural
            # language. The preference table was already updated internally
            # but NOT committed to conversation_history.
            if needs_convergence:
                conv_response, steps, conv_tool_entries = agent.generate_convergence_summary(
                    conversation_history
                )
                total_steps += steps
                total_events += 1
                conversation_history.extend(conv_tool_entries)
                agent.try_update_preference_table_from_text(conv_response)
                if GroupTravelAgent.check_is_travel_plan(conv_response):
                    plan_generated = True
                    completion_reason = "plan_generated"
                    logger.info(
                        "[Stop] Agent generated travel plan during "
                        "preference-table-intercept convergence."
                    )
                    break
                continue

            # Cross-turn repetitive tool-call detection for Agent
            if actual_speaker == AGENT_ROLE_NAME:
                if getattr(agent, '_last_turn_repetitive', False):
                    repetitive_reason = getattr(agent, '_repetitive_reason', 'consecutive')

                    # "multiple" means the same tool+args was called 3+ times
                    # across iterations — terminate immediately without
                    # requiring consecutive turns.
                    if repetitive_reason == "multiple":
                        completion_reason = "multiple_repeated_tool_calls"
                        logger.error(
                            "[Stop] Agent called the same tool with identical arguments "
                            "more than 2 times. Terminating task."
                        )
                        break

                    # "consecutive" means back-to-back identical calls within
                    # a single turn. Require 2 consecutive Agent turns to fire.
                    _consecutive_repetitive_agent_turns += 1
                    logger.warning(
                        f"{task_prefix()}[Repetitive] Agent turn flagged as repetitive "
                        f"({_consecutive_repetitive_agent_turns}/{_MAX_CONSECUTIVE_REPETITIVE_TURNS})"
                    )
                    if _consecutive_repetitive_agent_turns >= _MAX_CONSECUTIVE_REPETITIVE_TURNS:
                        completion_reason = "consecutive_repeated_tool_calls"
                        logger.error(
                            f"{task_prefix()}[Stop] Agent hit consecutive repetitive "
                            "tool-call limit. Terminating task."
                        )
                        break
                else:
                    _consecutive_repetitive_agent_turns = 0

            if mention_pass_exhausted:
                # Mark this (message, target) as consumed so subsequent
                # loop iterations won't re-trigger the same preempt.
                # This is the primary fix for the infinite preempt loop:
                # without it, [pass] not entering history means
                # get_last_message_entry() keeps returning the same Agent
                # message and detect_mention() keeps finding the same target.
                consumed_mentions.add((id(last_msg), actual_speaker))
                logger.warning(
                    f"{task_prefix()}[Mention] {actual_speaker} exhausted "
                    f"pass-retries while @-mentioned; marking mention as "
                    f"consumed to prevent re-trigger."
                )
                continue

            # Preempt does not advance polling_index, so the round only
            # closes via the runaway cap.
            if events_in_current_round >= max_events_per_round:
                logger.warning(
                    f"{task_prefix()}[Round {total_rounds}] Per-round event cap "
                    f"({max_events_per_round}) hit during preempt chain; "
                    "force-closing round and resetting polling_index."
                )
                total_rounds += 1
                events_in_current_round = 0
                polling_index = 0
            continue

        # ---- Priority 2: timed convergence (round-boundary only) ----
        # Only checked when we're at a fresh round boundary (polling_index
        # has just wrapped to 0 and total_rounds > 0). This guarantees
        # convergence never half-cuts an in-progress round.
        if (
            polling_index == 0
            and total_rounds > 0
            and total_rounds % convergence_interval == 0
        ):
            logger.info(
                f"[Convergence] Triggered at round {total_rounds}. "
                "Agent performing preference + convergence summary."
            )

            # Step 1: preference summary (internal only, no history append)
            agent.generate_preference_summary(conversation_history)

            # Step 2: convergence summary (enters conversation history).
            conv_response, steps, conv_tool_entries = agent.generate_convergence_summary(
                conversation_history
            )
            total_steps += steps
            total_events += 1

            # `conv_tool_entries` already contains the chronological
            # message + tool sequence (including the final outward
            # convergence message).
            conversation_history.extend(conv_tool_entries)
            agent.try_update_preference_table_from_text(conv_response)

            logger.info(
                f"[Event {total_events}] Convergence summary: "
                f"{conv_response[:200]}..."
            )

            if GroupTravelAgent.check_is_travel_plan(conv_response):
                plan_generated = True
                completion_reason = "plan_generated"
                logger.info(
                    "[Stop] Agent generated travel plan during convergence. "
                    "Conversation complete."
                )
                break

            # Convergence as a "phantom round": since the convergence
            # condition `total_rounds % convergence_interval == 0` would
            # remain true on the next iteration if no round progressed,
            # we'd loop forever firing convergence. To avoid this, we
            # bump `total_rounds` by 1 here to "skip past" this checkpoint.
            # The cost: one round of `max_turn` budget per convergence
            # event, which is a fair price for the simplicity.
            #
            # `events_in_current_round` is reset defensively even though
            # it is guaranteed to already be 0 here (this branch is gated
            # on `polling_index == 0` which only holds right after a
            # round close, which itself resets the counter). The reset
            # makes the round-close invariant uniform across all three
            # branches that advance `total_rounds`.
            total_rounds += 1
            events_in_current_round = 0
            # If the convergence message contained an @ to a user, the
            # next iteration will pick it up via the priority-1 preempt.
            continue

        # ---- Priority 3: polling ----
        actual_speaker = polling_order[polling_index]

        # Skip User speakers who have no new visible messages since their
        # last utterance. This avoids calling the LLM when the User has
        # nothing new to react to (all other participants passed or the
        # User was the most recent speaker). The Agent is never skipped.
        if (
            actual_speaker != AGENT_ROLE_NAME
            and not _has_new_visible_messages_since_last_speak(
                actual_speaker, conversation_history
            )
        ):
            logger.info(
                f"[Round {total_rounds} / Event {total_events}] "
                f"Skipping {actual_speaker} (no new visible messages since last speak)"
            )
            polling_index = (polling_index + 1) % len(polling_order)
            # Still check round closure after advancing
            if polling_index == 0:
                total_rounds += 1
                events_in_current_round = 0
            continue

        response_text, speaker_steps, speaker_tool_entries = _generate_speaker_response(
            speaker=actual_speaker,
            is_being_mentioned=False,
            agent=agent,
            user_simulators=user_simulators,
            conversation_history=conversation_history,
        )
        total_steps += speaker_steps
        total_events += 1
        events_in_current_round += 1

        # Polling consumes a slot; advance polling_index regardless of
        # whether the speaker passed.
        polling_index = (polling_index + 1) % len(polling_order)

        # Retry loop for invalid compromise markers (polling path).
        _MAX_COMPROMISE_RETRIES = 3
        _comp_attempt = 0
        while _comp_attempt < _MAX_COMPROMISE_RETRIES:
            _comp_attempt += 1
            plan_generated, completion_reason, needs_convergence, needs_user_retry = _commit_speaker_event(
                actual_speaker=actual_speaker,
                response_text=response_text,
                speaker_tool_entries=speaker_tool_entries,
                conversation_history=conversation_history,
                compromise_tracker=compromise_tracker,
                user_simulators=user_simulators,
                event_seq=total_events,
                round_idx=total_rounds,
                agent=agent,
                plan_generated=plan_generated,
                completion_reason=completion_reason,
            )
            if not needs_user_retry:
                break
            if _comp_attempt >= _MAX_COMPROMISE_RETRIES:
                # Exhausted all attempts — the user genuinely intends
                # to compromise (Agent asked without @mention). Apply
                # the compromise to ③ to keep state consistent with
                # what the Agent will perceive from the natural language.
                logger.warning(
                    f"{task_prefix()}[CompromiseRetry] {actual_speaker} "
                    f"exhausted {_MAX_COMPROMISE_RETRIES} attempts for "
                    f"invalid compromise marker; force-applying compromise "
                    f"and committing stripped reply."
                )
                marker_match = _MARKER_RE.search(response_text)
                committed_text = (
                    strip_compromise_marker(response_text, marker_match.group(0))
                    if marker_match else response_text
                )
                # Force-apply the compromise to ③ since the user's
                # intent is clear even though the @mention check failed.
                if marker_match:
                    extracted = extract_compromise_marker(response_text)
                    if extracted is not None:
                        field_path, new_value, _raw = extracted
                        applied = compromise_tracker.apply(
                            user=actual_speaker,
                            field_path=field_path,
                            new_value=new_value,
                            round_idx=total_events,
                        )
                        if applied and compromise_tracker.is_exhausted(actual_speaker):
                            sim = user_simulators.get(actual_speaker)
                            if sim is not None and sim.compromisable:
                                logger.info(
                                    f"[Compromise] {actual_speaker} reached "
                                    "the per-user cap; flipping "
                                    "compromisable=False."
                                )
                                sim.set_compromisable(False)
                conversation_history.append({
                    "type": "message",
                    "role": actual_speaker,
                    "content": committed_text,
                    "has_tool_calls": False,
                })
                break
            logger.warning(
                f"{task_prefix()}[CompromiseRetry] Regenerating "
                f"{actual_speaker}'s reply (attempt {_comp_attempt + 1}/"
                f"{_MAX_COMPROMISE_RETRIES})."
            )
            response_text, speaker_steps, speaker_tool_entries = _generate_speaker_response(
                speaker=actual_speaker,
                is_being_mentioned=False,
                agent=agent,
                user_simulators=user_simulators,
                conversation_history=conversation_history,
            )
            total_steps += speaker_steps

        if plan_generated:
            break

        # Agent output a bare preference table in a normal turn:
        # trigger a convergence summary so it re-expresses in natural
        # language. The preference table was already updated internally
        # but NOT committed to conversation_history.
        if needs_convergence:
            conv_response, steps, conv_tool_entries = agent.generate_convergence_summary(
                conversation_history
            )
            total_steps += steps
            total_events += 1
            conversation_history.extend(conv_tool_entries)
            agent.try_update_preference_table_from_text(conv_response)
            if GroupTravelAgent.check_is_travel_plan(conv_response):
                plan_generated = True
                completion_reason = "plan_generated"
                logger.info(
                    "[Stop] Agent generated travel plan during "
                    "preference-table-intercept convergence."
                )
                break
            # After convergence, check round closure and continue the loop.
            if polling_index == 0:
                total_rounds += 1
                events_in_current_round = 0
            continue

        # Cross-turn repetitive tool-call detection for Agent (polling path)
        if actual_speaker == AGENT_ROLE_NAME:
            if getattr(agent, '_last_turn_repetitive', False):
                repetitive_reason = getattr(agent, '_repetitive_reason', 'consecutive')

                # "multiple" means the same tool+args was called 3+ times
                # across iterations — terminate immediately.
                if repetitive_reason == "multiple":
                    completion_reason = "multiple_repeated_tool_calls"
                    logger.error(
                        "[Stop] Agent called the same tool with identical arguments "
                        "more than 2 times. Terminating task."
                    )
                    break

                # "consecutive" means back-to-back identical calls.
                _consecutive_repetitive_agent_turns += 1
                logger.warning(
                    f"{task_prefix()}[Repetitive] Agent turn flagged as repetitive "
                    f"({_consecutive_repetitive_agent_turns}/{_MAX_CONSECUTIVE_REPETITIVE_TURNS})"
                )
                if _consecutive_repetitive_agent_turns >= _MAX_CONSECUTIVE_REPETITIVE_TURNS:
                    completion_reason = "consecutive_repeated_tool_calls"
                    logger.error(
                        f"{task_prefix()}[Stop] Agent hit consecutive repetitive "
                        "tool-call limit. Terminating task."
                    )
                    break
            else:
                _consecutive_repetitive_agent_turns = 0

        # Round closure: either polling has wrapped, or the runaway cap fired.
        if polling_index == 0:
            total_rounds += 1
            events_in_current_round = 0
        elif events_in_current_round >= max_events_per_round:
            logger.warning(
                f"{task_prefix()}[Round {total_rounds}] Per-round event cap "
                f"({max_events_per_round}) hit during polling; "
                "force-closing round and resetting polling_index."
            )
            total_rounds += 1
            events_in_current_round = 0
            polling_index = 0

    # --- Max turn fallback ---
    # Triggers only if we exited the loop because total_rounds >= max_turn
    # without a plan having been emitted.
    # Retry up to _MAX_FALLBACK_RETRIES times if the output is not a valid
    # travel plan JSON (e.g. bracket mismatch, missing fields). Since the
    # conversation has already consumed many rounds, a simple retry is
    # worthwhile before declaring the task as failed.
    _MAX_FALLBACK_RETRIES = 3
    fallback_plan_is_valid = False
    if total_rounds >= max_turn and completion_reason == "unknown":
        logger.info(
            f"[Fallback] Max turn ({max_turn} rounds) reached. "
            "Agent generating final plan."
        )

        for _fallback_attempt in range(1, _MAX_FALLBACK_RETRIES + 1):
            final_plan, steps, final_tool_entries = agent.generate_final_plan(
                conversation_history
            )
            total_steps += steps

            fallback_plan_is_valid = GroupTravelAgent.check_is_travel_plan(final_plan)
            if fallback_plan_is_valid:
                logger.info(
                    f"[Fallback] Final plan passed structure check "
                    f"(attempt {_fallback_attempt}/{_MAX_FALLBACK_RETRIES})."
                )
                break

            logger.warning(
                f"{task_prefix()}[Fallback] Final plan failed structure check "
                f"(attempt {_fallback_attempt}/{_MAX_FALLBACK_RETRIES}); "
                f"{'retrying...' if _fallback_attempt < _MAX_FALLBACK_RETRIES else 'giving up.'}"
            )

        total_events += 1

        # `final_tool_entries` already contains the chronological
        # message + tool sequence (including the final plan message).
        conversation_history.extend(final_tool_entries)
        # `completion_reason` stays as "max_turn_reached" to mark this as a
        # forced-finish — distinguishable from the natural "plan_generated"
        # path where the Agent decided to commit on its own. We do, however,
        # still validate the fallback output: a structurally-valid plan
        # earns a post-plan preference refresh (per Q6 scheme (a)) so that
        # ② evaluation has a comparable "what the Agent knew right before
        # committing" snapshot. An invalid/empty fallback skips the refresh
        # since there is nothing meaningful to slice off.
        completion_reason = "max_turn_reached"
        if fallback_plan_is_valid:
            # A structurally-valid fallback plan IS a real plan: mark
            # `plan_generated` so downstream evaluation scores it on the
            # utility / fairness / validity dimensions instead of treating
            # it as "plan not generated" (which scores 0 on those three).
            # `completion_reason` stays "max_turn_reached" so we can still
            # distinguish a forced finish from a natural commit.
            plan_generated = True
        else:
            logger.warning(
                f"{task_prefix()}[Fallback] Final plan failed travel_plan "
                f"structure check after {_MAX_FALLBACK_RETRIES} attempts; "
                "skipping post-plan preference refresh."
            )

    if completion_reason == "unknown":
        completion_reason = "loop_ended"

    # Post-plan preference refresh (per Q6, scheme (a)):
    # Run when the conversation ended with a *valid* plan as the last
    # message — either because Agent committed naturally
    # (`plan_generated`), or because max_turn fallback produced a
    # structurally-valid plan. Refresh on the history *minus* the plan
    # message and its tool entries. The resulting ② snapshot represents
    # "what the Agent knew right before the plan was emitted". This is
    # the snapshot used to compare against ③ in evaluation.
    if completion_reason == "plan_generated" or fallback_plan_is_valid:
        logger.info(
            "[PostPlanRefresh] Refreshing agent preference table on "
            "history sliced before the final plan message."
        )
        history_for_refresh = truncate_history_before_last_message(
            conversation_history
        )
        agent.generate_preference_summary(
            conversation_history=conversation_history,
            history_override=history_for_refresh,
        )

    duration = time.time() - start_time

    # Strip reasoning_content from conversation_history before saving
    # to reduce output file size (not needed for evaluation).
    for entry in conversation_history:
        entry.pop("reasoning_content", None)

    # Build result
    result = {
        "conversation_history": conversation_history,
        "agent_trace": agent.get_trace(),
        # ② Agent's inferred preference table (post-final-refresh if plan
        # was generated). Stored as a parsed object when possible to avoid
        # double-encoding when serialised by callers.
        "agent_preference_table": _try_parse_json(agent.preference_table),
        # ③ Effective preferences after applied compromises.
        "effective_preferences": compromise_tracker.effective_preferences,
        "compromise_log": compromise_tracker.compromise_log,
        "agent_tool_stats": agent.get_tool_stats(),
        # Number of completed polling laps (one round = each polling slot
        # has spoken once). This is the unit `max_turn` and
        # `convergence_interval` operate in.
        "total_rounds": total_rounds,
        # Global event count (every speaking event: polling turns,
        # @-mention preempts, User [pass]es, in-loop convergence
        # messages, the opening convergence message, and the max_turn
        # fallback final plan if any).
        "total_events": total_events,
        "total_steps": total_steps,
        "duration": duration,
        "completion_reason": completion_reason,
        "plan_generated": plan_generated,
    }

    return result

def _try_parse_json(s: str) -> Any:
    """Best-effort JSON parse; return original string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s

# ============================================================================
# Top-level entry point
# ============================================================================

def run_multi_user_conversation(
    data: Dict[str, Any],
    agent_config: OpenAIConfig,
    user_config: OpenAIConfig,
    max_turn: int = 15,
    convergence_interval: int = 3,
    debug: bool = False,
    trial_id: int = -1,
) -> Dict[str, Any]:
    """
    Top-level entry point for running a multi-user travel planning conversation.

    Expected data format:
    {
        "query": "吉林三天两夜的旅行计划",
        "time": "2025-07-15",                # departure date
        "context": "optional background info",
        # User preferences (dataset format, required). Each value must include:
        #   - preference:    preference-profile dict
        #     - global_constraints: {...}
        #     - city_specific_preferences: {city_name: {...}, ...}  (dict, not a list)
        #   - compromisable: whether the user will yield when the Agent @-asks
        "user_preferences": {
            "User1": {"preference": {...}, "compromisable": false, ...},
            "User2": {"preference": {...}, "compromisable": true, ...}
        },
        # Phase-1 opening statements: required; every task in the dataset has them
        "initial_messages": [
            {"role": "User1", "content": "..."},
            {"role": "User2", "content": "..."},
        ]
    }

    Note: The sandbox state is process-global (set up by SandboxManager
    inside __main__). All conversations in the same process share that state.

    Args:
        data: Input data dict with query, user_preferences, initial_messages, etc.
        agent_config: OpenAI config for the Agent.
        user_config: OpenAI config for user simulators.
        max_turn: Maximum number of rounds (complete polling laps).
        convergence_interval: Number of rounds between convergence summaries.
        debug: Enable debug logging.

    Returns:
        Result dict with full conversation data.

    Raises:
        ValueError: If 'user_preferences' is missing/empty,
            or if 'initial_messages' is missing/empty.
    """
    # Configure project-level logging.
    #
    # Logging policy (matches the batch runner's concurrency policy):
    #   - debug=True  → root level DEBUG. The batch runner forces
    #     max_concurrency=1 in this mode, so the detailed per-event INFO
    #     trace ("[Round X / Event Y] User1: ...") is readable.
    #   - debug=False → root level WARNING. Per-conversation INFO traces
    #     are suppressed because, under multi-threaded batch execution,
    #     interleaved INFO lines from different tasks are unreadable and
    #     misleading. Task-level progress is still printed by the batch
    #     runner (`run_one_conversation` uses `print`, not the logger).
    #     Errors / warnings (e.g. compromise marker rejection, mention
    #     pass-exhaustion) still surface through WARNING.
    #
    # Third-party loggers are pinned to WARNING regardless, otherwise
    # they would dump full HTTP request bodies, TLS handshakes, SSE
    # connection traces, MCP JSON-RPC frames, asyncio selector events,
    # etc. when DEBUG is enabled on the root logger.
    root_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    noisy_loggers = (
        # HTTP clients
        "httpx", "httpcore", "openai", "urllib3", "requests",
        # asyncio internals (e.g. "Using selector: EpollSelector")
        "asyncio", "anyio",
    )
    for noisy_logger in noisy_loggers:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Bind task_id and trial_id to the current thread so that all
    # downstream loggers can include them without explicit parameter threading.
    task_id = data.get("task_id", "")
    set_task_context(task_id, trial_id=trial_id)

    query = data.get("query", "")
    time_str = data.get("time", "")
    context = data.get("context", "")

    # Build simulator inputs from the synthesized "user_preferences" dict.
    # Each entry in user_preferences contains technical fields (role, trace_id,
    # compromisable) plus the actual preference dict. We only inject the
    # preference content into the simulator's user_preference and keep
    # compromisable as a separate flag; technical fields are dropped.
    user_prefs = data.get("user_preferences")
    if not user_prefs:
        raise ValueError(
            "data must contain non-empty 'user_preferences' dict."
        )

    simulator_inputs: List[Dict[str, Any]] = []
    for user_key, user_pref_data in user_prefs.items():
        preference = user_pref_data.get("preference", {})
        simulator_inputs.append({
            "name": user_key,
            "user_preference": json.dumps(preference, ensure_ascii=False),
            "compromisable": bool(user_pref_data.get("compromisable", False)),
        })
    user_names = [u["name"] for u in simulator_inputs]
    user_list_description = build_user_list_description(user_names)

    # Build the compromise tracker from the *original* user_preferences
    # input. ③ effective_preferences starts as a deep copy of ① and gets
    # mutated by the orchestrator whenever a user reply contains a valid
    # compromise marker.
    compromise_tracker = CompromiseTracker(user_prefs)

    # Create user simulators
    user_simulators: Dict[str, MultiUserSimulator] = {}
    for sim_input in simulator_inputs:
        name = sim_input["name"]
        user_simulators[name] = MultiUserSimulator(
            config=user_config,
            user_name=name,
            user_preference=sim_input.get("user_preference", ""),
            query=query,
            time=time_str,
            user_list_description=user_list_description,
            compromisable=sim_input.get("compromisable", False),
        )

    # Create Agent
    agent = GroupTravelAgent(
        config=agent_config,
        query=query,
        time=time_str,
        context=context,
        user_names=user_names,
        user_list_description=user_list_description,
    )

    # Global structured conversation history
    conversation_history: List[Dict[str, Any]] = []

    # Phase 1: Load pre-generated initial messages (required field)
    initial_messages = data.get("initial_messages")
    if not initial_messages:
        raise ValueError(
            "data must contain a non-empty 'initial_messages' field "
            "(one opening statement per user)."
        )

    logger.info(
        f"Phase 1: Loading {len(initial_messages)} pre-provided initial messages."
    )
    for msg in initial_messages:
        conversation_history.append({
            "type": "message",
            "role": msg["role"],
            "content": msg["content"],
        })
    # Phase 2: Free interaction
    logger.info("Phase 2: Starting free interaction...")
    result = run_multi_user_interaction(
        agent=agent,
        user_simulators=user_simulators,
        user_names=user_names,
        conversation_history=conversation_history,
        compromise_tracker=compromise_tracker,
        max_turn=max_turn,
        convergence_interval=convergence_interval,
    )

    # Enrich result with input metadata.
    # ① original_preferences is a deep copy of the dataset input, kept so
    # downstream evaluation has all three preference snapshots in one
    # result record (① vs ② = info-collection accuracy, ② vs ③ = compromise
    # tracking accuracy, ③ used as the baseline when scoring travel_plan).
    result["query"] = query
    result["time"] = time_str
    result["context"] = context
    result["user_names"] = user_names
    result["max_turn"] = max_turn
    result["convergence_interval"] = convergence_interval
    result["original_preferences"] = copy.deepcopy(user_prefs)

    return result

def run_and_save_multi_user_conversation(
    data: Dict[str, Any],
    agent_config: OpenAIConfig,
    user_config: OpenAIConfig,
    output_file: str = "multi_user_results.json",
    max_turn: int = 15,
    convergence_interval: int = 3,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run a multi-user conversation and save results to a JSON file.

    Args:
        data: Input data dict.
        agent_config: OpenAI config for the Agent.
        user_config: OpenAI config for user simulators.
        output_file: Path to save results.
        max_turn: Maximum number of rounds.
        convergence_interval: Rounds between convergence summaries.
        debug: Enable debug logging.

    Returns:
        Result dict.
    """
    result = run_multi_user_conversation(
        data=data,
        agent_config=agent_config,
        user_config=user_config,
        max_turn=max_turn,
        convergence_interval=convergence_interval,
        debug=debug,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")
    print(f"✅ Multi-user conversation completed.")
    print(
        f"   Rounds: {result['total_rounds']}, "
        f"Events: {result['total_events']}, "
        f"Steps: {result['total_steps']}"
    )
    print(f"   Duration: {result['duration']:.1f}s")
    print(f"   Completion: {result['completion_reason']}")
    print(f"   Plan generated: {result['plan_generated']}")
    print(f"   Output: {output_file}")

    return result
