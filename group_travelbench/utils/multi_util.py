"""
Utility functions for multi-user travel planning.

Provides two flavors of conversation history formatting:

1. `format_context` (legacy text-based) — still used by
   `generate_preference_summary` where tool-level detail is irrelevant
   and a flat text representation suffices.

2. `build_agent_messages` / `build_user_messages` — new structured
   formatters that produce a proper OpenAI-style message list with
   native assistant/tool_calls/tool role entries. These prevent the
   model from "imitating" tool call text patterns that appear in flat
   text context.
"""

import json
from typing import List, Dict, Any

from ..core.messages import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)

# Counter for generating unique fake tool_call IDs (conversation_history
# entries do not store the original OpenAI tool_call ID).
_tool_call_counter = 0


def _next_tool_call_id() -> str:
    """Generate a unique fake tool_call_id for reconstructed history."""
    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_hist_{_tool_call_counter}"


# ============================================================================
# Legacy text formatter (retained for preference_summary and backward compat)
# ============================================================================

def format_context(
    history: List[Dict[str, Any]],
    include_tool_info: bool = False,
) -> str:
    """
    Format the structured conversation history into a plain-text string.

    Kept for backward compatibility and for `generate_preference_summary`
    which does not need structured tool entries.

    `include_tool_info=False`:
      Outputs only outward messages (Users + the Agent's final reply).
      Skips tool entries entirely AND skips Agent messages with
      `has_tool_calls=True`.

    `include_tool_info=True`:
      Renders every Agent message with its trailing tool info inlined.

    Args:
        history: The structured conversation history.
        include_tool_info: Whether to include tool call/result info.

    Returns:
        Formatted context string.
    """
    if not include_tool_info:
        lines = []
        for entry in history:
            if entry["type"] != "message":
                continue
            if entry.get("has_tool_calls"):
                continue
            lines.append(f"{entry['role']}: {entry['content']}")
        return "\n".join(lines)

    # include_tool_info=True: inline tool info after Agent messages
    lines: List[str] = []
    index = 0
    while index < len(history):
        entry = history[index]

        if entry["type"] == "message":
            if entry["role"] == "Agent":
                tool_lines: List[str] = []
                next_index = index + 1
                while next_index < len(history) and history[next_index]["type"] in ("tool_call", "tool_result"):
                    tool_entry = history[next_index]
                    if tool_entry["type"] == "tool_call":
                        tool_lines.append(
                            f"tool_call: {tool_entry['name']}({tool_entry['arguments']})"
                        )
                    elif tool_entry["type"] == "tool_result":
                        error_marker = " [ERROR]" if tool_entry.get("error") else ""
                        tool_lines.append(
                            f"tool_result:{error_marker} {tool_entry['result']}"
                        )
                    next_index += 1

                content = entry.get("content", "")
                if tool_lines:
                    lines.append("Agent:")
                    lines.extend(tool_lines)
                    if content:
                        lines.append(content)
                else:
                    lines.append(f"Agent: {content}" if content else "Agent:")

                index = next_index
            else:
                lines.append(f"{entry['role']}: {entry['content']}")
                index += 1
        else:
            index += 1

    return "\n".join(lines)


# ============================================================================
# Structured message builders (new — OpenAI native format)
# ============================================================================

def _ensure_alternation(messages: List[Message]) -> List[Message]:
    """Post-process a message list to guarantee user/assistant alternation.

    OpenAI API requires strict alternation between user and assistant roles
    (tool messages are allowed between an assistant-with-tool-calls and the
    next assistant). This function handles two edge cases:

    1. The list starts with an AssistantMessage (no preceding UserMessage):
       insert a placeholder UserMessage at the front.
    2. Two consecutive AssistantMessages appear without any intervening
       ToolMessage (can happen when all Users pass and their messages don't
       enter history): insert a placeholder UserMessage between them.

    The legitimate sequence ``AssistantMessage(tool_calls)`` →
    ``ToolMessage(s)`` → ``AssistantMessage`` is NOT a violation and is
    left untouched.
    """
    if not messages:
        return messages

    result: List[Message] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            # Determine if this is a legitimate tool-loop continuation:
            # the last entry in result should be a ToolMessage (which means
            # the preceding AssistantMessage had tool_calls). If there's no
            # ToolMessage right before us, check whether the last non-tool
            # entry was a UserMessage (legitimate alternation) or another
            # AssistantMessage (violation).
            if not result:
                # First message is assistant — insert placeholder user
                result.append(UserMessage(content="(对话开始)"))
            elif isinstance(result[-1], ToolMessage):
                # Legitimate: tool result followed by next assistant step
                pass
            elif isinstance(result[-1], AssistantMessage):
                # Two consecutive assistants with no tool in between
                result.append(UserMessage(content="(其他参与者暂无发言)"))
            # If result[-1] is UserMessage, alternation is correct — no action

        result.append(msg)

    return result


def build_agent_messages(
    history: List[Dict[str, Any]],
) -> List[Message]:
    """Build an OpenAI-native message list from the Agent's perspective.

    Design:
      - Agent's own messages → ``AssistantMessage`` (with native tool_calls
        when the entry has ``has_tool_calls=True``).
      - Tool results → ``ToolMessage``.
      - All User messages (any role other than "Agent") between two Agent
        speaking turns are merged into a single ``UserMessage`` with
        "RoleName: content" lines, preserving multi-user role attribution.

    The resulting list strictly alternates user/assistant (with tool messages
    interleaved between assistant-with-tool-calls and the next assistant).
    This ensures the model sees its own prior tool usage in native format
    and won't "imitate" tool_call text patterns in its content output.

    Returns:
        List[Message] ready to be prepended with a SystemMessage and sent
        to the OpenAI API.
    """
    messages: List[Message] = []
    # Buffer collecting non-Agent entries to merge into a single UserMessage
    user_buffer: List[str] = []
    index = 0

    while index < len(history):
        entry = history[index]

        if entry["type"] == "message" and entry["role"] == "Agent":
            # Flush any accumulated user buffer before this Agent turn
            if user_buffer:
                messages.append(UserMessage(content="\n".join(user_buffer)))
                user_buffer = []

            # Determine if this Agent message has tool calls following it
            has_tools = entry.get("has_tool_calls", False)
            content = entry.get("content", "") or ""

            if has_tools:
                # Collect the tool_call and tool_result entries that follow
                tool_calls: List[ToolCall] = []
                tool_messages: List[ToolMessage] = []
                next_index = index + 1

                while next_index < len(history) and history[next_index]["type"] in ("tool_call", "tool_result"):
                    tool_entry = history[next_index]
                    if tool_entry["type"] == "tool_call":
                        call_id = _next_tool_call_id()
                        arguments_raw = tool_entry.get("arguments", "{}")
                        if isinstance(arguments_raw, str):
                            try:
                                arguments_dict = json.loads(arguments_raw)
                            except (json.JSONDecodeError, TypeError):
                                arguments_dict = {"raw_arguments": arguments_raw}
                        else:
                            arguments_dict = arguments_raw
                        tool_calls.append(ToolCall(
                            id=call_id,
                            name=tool_entry["name"],
                            arguments=arguments_dict,
                        ))
                    elif tool_entry["type"] == "tool_result":
                        # Match to the most recent tool_call's id
                        matched_id = tool_calls[-1].id if tool_calls else _next_tool_call_id()
                        tool_messages.append(ToolMessage(
                            tool_call_id=matched_id,
                            name=tool_entry.get("name", "unknown"),
                            content=tool_entry.get("result", ""),
                            error=tool_entry.get("error", False),
                        ))
                    next_index += 1

                # Emit AssistantMessage with tool_calls
                reasoning = entry.get("reasoning_content")
                messages.append(AssistantMessage(
                    content=content if content else None,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls if tool_calls else None,
                ))
                # Emit ToolMessages
                messages.extend(tool_messages)
                index = next_index
            else:
                # Final outward reply (no tool calls)
                reasoning = entry.get("reasoning_content")
                messages.append(AssistantMessage(
                    content=content,
                    reasoning_content=reasoning,
                ))
                index += 1

        elif entry["type"] == "message":
            # Non-Agent message: accumulate into user buffer
            user_buffer.append(f"{entry['role']}: {entry['content']}")
            index += 1
        else:
            # Orphan tool entries (shouldn't normally happen) — skip
            index += 1

    # Flush trailing user buffer (if conversation ends with user messages)
    if user_buffer:
        messages.append(UserMessage(content="\n".join(user_buffer)))

    return _ensure_alternation(messages)


def build_user_messages(
    history: List[Dict[str, Any]],
    self_role: str,
) -> List[Message]:
    """Build an OpenAI-native message list from a specific User's perspective.

    Design:
      - The specified user's own messages → ``AssistantMessage``.
      - All other participants' messages (Agent final replies + other Users)
        between two of this user's speaking turns are merged into a single
        ``UserMessage`` with "RoleName: content" lines.
      - Agent intermediate reasoning (``has_tool_calls=True``) and tool
        entries are skipped entirely — Users never see tool details.

    The resulting list strictly alternates user/assistant, matching the
    standard OpenAI multi-turn format.

    Args:
        history: The structured conversation history.
        self_role: The role name of the user being simulated (e.g. "User1").

    Returns:
        List[Message] ready to be prepended with a SystemMessage.
    """
    messages: List[Message] = []
    # Buffer collecting "others' messages" to merge into UserMessage
    others_buffer: List[str] = []

    for entry in history:
        if entry["type"] != "message":
            # Skip tool_call / tool_result entries entirely for Users
            continue
        if entry.get("has_tool_calls"):
            # Skip Agent's intermediate reasoning
            continue

        role = entry["role"]
        content = entry.get("content", "") or ""

        if role == self_role:
            # This user's own message → AssistantMessage
            # First flush others_buffer
            if others_buffer:
                messages.append(UserMessage(content="\n".join(others_buffer)))
                others_buffer = []
            messages.append(AssistantMessage(content=content))
        else:
            # Someone else's message → accumulate
            others_buffer.append(f"{role}: {content}")

    # Flush trailing others buffer
    if others_buffer:
        messages.append(UserMessage(content="\n".join(others_buffer)))

    return _ensure_alternation(messages)