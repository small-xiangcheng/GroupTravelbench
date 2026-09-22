"""
Multi-user travel planning Agent implementation.
Handles tool calling, preference table maintenance, and plan generation
in a multi-user group chat scenario.
"""

import json
from typing import List, Dict, Any, Optional, Tuple

from ..core.config import OpenAIConfig, ReasoningMode
from ..core.openai_client import OpenAIClient
from ..core.messages import (
    SystemMessage, UserMessage, AssistantMessage, ToolMessage
)
from ..core.tools import sandbox_tool_registry
from .multi_prompt import (
    MULTI_USER_AGENT_SYSTEM_PROMPT,
    MULTI_USER_AGENT_PREFERENCE_SUMMARY_PROMPT,
    MULTI_USER_AGENT_CONVERGENCE_PROMPT,
    MULTI_USER_AGENT_FINAL_PLAN_PROMPT,
    MULTI_USER_AGENT_FORCE_FINISH_INSTRUCTION,
)
from ..utils.multi_util import format_context, build_agent_messages


class GroupTravelAgent:
    """
    Agent for multi-user travel planning scenario.
    
    Maintains:
    - A dynamic system prompt with user preference table
    - A detailed trace (including tool calls/results) for its own reasoning
    - Generates responses that go into the shared context (without tool details)
    """

    def __init__(
        self,
        config: OpenAIConfig,
        query: str,
        time: str = "",
        context: str = "",
        user_names: Optional[List[str]] = None,
        user_list_description: str = "",
    ):
        self.config = config
        self.client = OpenAIClient(config)
        self.query = query
        self.time = time
        self.context = context
        self.user_names = user_names or []
        self.user_list_description = user_list_description

        # Tool management - reuse sandbox tool registry
        self.tool_registry = sandbox_tool_registry
        self.available_tools = self.tool_registry.get_tools()

        # Dynamic preference table (JSON string), updated during convergence
        self.preference_table: str = "{}"

        # Detailed agent trace for saving: includes tool calls and tool results
        # Each entry is a dict with keys: role, content, tool_calls, tool_results, etc.
        self.agent_trace: List[Dict[str, Any]] = []

        # Track tool execution statistics
        self.tool_stats: Dict[str, int] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "tool_errors": 0,
            "tool_calls_executed": 0,
        }

    def _build_system_prompt(self) -> str:
        """Build the current system prompt with the latest preference table
        always injected.

        Rationale: the preference table is the Agent's structured view of
        what it has learned so far. Keeping it in every system prompt
        (free replies, convergence, tool-call loops, final plan, preference
        summary refresh) lets the Agent reliably:
          - identify which fields are still missing (asking targeted @-questions)
          - avoid re-asking what users already stated
          - perform global trade-offs when emitting the final plan
        Token cost is bounded (the table is at most ~1-2k tokens) and is
        prefix-cacheable across the tool-call loop within a single turn.
        """
        preference_table_block = (
            "\n【当前已收集的用户偏好表】\n"
            f"{self.preference_table}\n"
        )

        return MULTI_USER_AGENT_SYSTEM_PROMPT.format(
            user_list_description=self.user_list_description,
            query=self.query,
            context=self.context,
            time=self.time,
            preference_table_block=preference_table_block,
        )

    def generate_response(
        self,
        conversation_history: List[Dict[str, Any]],
        is_being_mentioned: bool = False,
        extra_instruction: Optional[str] = None,
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """
        Generate a response in the multi-user chat.

        Storage model (chronological, mirrors OpenAI's native message order):
          Each tool-calling iteration emits one message entry followed by its
          tool_call/tool_result entries; the final iteration (no tool calls)
          emits a single message entry. All entries are accumulated in
          `turn_entries` IN THE ORDER THEY OCCURRED, so the caller can simply
          `conversation_history.extend(turn_entries)` to preserve the true
          time line.

          Each Agent message entry carries a `has_tool_calls` flag that
          distinguishes intermediate reasoning steps (`True`, includes the
          model's "let me check X first" content interleaved with tool calls)
          from the final outward reply (`False`). Downstream consumers use
          this flag:
            - `format_context(include_tool_info=True)`: renders every message
              + its trailing tool entries as before — works unchanged because
              each message still owns the tools that immediately follow it.
            - `format_context(include_tool_info=False)` (User view, preference
              summary): SKIPS messages with `has_tool_calls=True`, so user
              simulators don't see the Agent's internal "let me query …"
              chatter.
            - @mention detection (`get_last_message_entry`): also skips
              `has_tool_calls=True` entries — only outward messages can be
              the target of an @mention.

        agent_trace remains unchanged and continues to record (new_turn,
        tool_calls, tool_result, assistant_response) in arrival order.

        Args:
            conversation_history: Read-only here; the caller is responsible
                for extending it with the returned `turn_entries`.
            is_being_mentioned: Whether the Agent was @mentioned. The Agent
                has no pass mechanism, so this only adds a reminder to
                answer directly without @-redirecting.
            extra_instruction: Optional extra instruction appended to the
                system prompt.

        Returns:
            Tuple of (final_text, num_steps, turn_entries) where
            turn_entries already contains the final message entry plus all
            intermediate reasoning + tool entries in chronological order.
            The caller does `conversation_history.extend(turn_entries)`
            once — no separate "append message then extend tools" step.
        """
        system_content = self._build_system_prompt()
        if is_being_mentioned:
            # Agent has no pass mechanism; this reminder only enforces
            # "answer the question directly without re-@-ing someone else".
            system_content += (
                "\n\n【重要提醒】你刚刚被 @ 了，你必须直接围绕被问到的问题作答，"
                "不能再 @ 别人转交问题。"
            )
        if extra_instruction:
            system_content += f"\n\n{extra_instruction}"

        # Build the initial message list using native OpenAI structure.
        # Agent's own prior messages are represented as AssistantMessage
        # (with native tool_calls), while all User messages are merged
        # into UserMessage entries. This prevents the model from
        # "imitating" tool_call text patterns that appeared in legacy
        # flat-text context rendering.
        history_messages = build_agent_messages(conversation_history)
        turn_messages = [SystemMessage(content=system_content)] + history_messages

        # Record in trace
        self.agent_trace.append({
            "type": "new_turn",
            "history_length": len(conversation_history),
        })

        # Chronological accumulator: every iteration emits one message entry
        # followed by its tool entries. See class docstring above for schema.
        turn_entries: List[Dict[str, Any]] = []

        max_iterations = 20
        iteration = 0

        # Track recent tool calls to detect repetitive loops.
        # Detection strategy: if two consecutive iterations call any of
        # the same (tool_name, arguments_json) pairs, we consider it a
        # repetitive loop and break immediately.
        _MAX_CONSECUTIVE_REPEATS = 2
        _recent_tool_signatures: List[Any] = []
        # Flag exposed to caller so the outer loop can track cross-turn repeats.
        self._last_turn_repetitive = False
        # Repetitive-loop reason: "consecutive" (same call back-to-back) or
        # "multiple" (same call appeared 3+ times across iterations).
        self._repetitive_reason: str = ""

        # Cross-iteration global tool-call counter: tracks how many times
        # each unique (name, arguments_json) signature has been called.
        # Once any signature exceeds _MAX_SAME_CALL_COUNT, the task is
        # terminated to prevent cyclic tool-call loops that inflate context.
        _MAX_SAME_CALL_COUNT = 2  # allow at most 2 calls; 3rd triggers stop
        if not hasattr(self, '_global_tool_call_counts'):
            self._global_tool_call_counts: Dict[str, int] = {}
        _global_counts = self._global_tool_call_counts

        while iteration < max_iterations:
            iteration += 1
            assistant_message, _ = self.client.generate_response(
                messages=turn_messages,
                tools=self.available_tools,
            )

            # Whether this iteration's content is intermediate reasoning
            # (will be followed by tools) or the final outward reply.
            is_intermediate = assistant_message.has_tool_calls()
            iteration_text = (assistant_message.content or "").strip()

            # Always append the message entry first. For intermediate
            # iterations the LLM may emit no content alongside its
            # tool_calls; we still record an entry so that downstream
            # renderers can attach the trailing tool entries to a
            # well-defined anchor.
            entry: Dict[str, Any] = {
                "type": "message",
                "role": "Agent",
                "content": iteration_text,
                "has_tool_calls": is_intermediate,
            }
            # Persist reasoning_content in conversation_history only when
            # reasoning_mode is ALL_TURNS (needed for cross-turn context).
            if (
                self.config.reasoning_mode == ReasoningMode.ALL_TURNS
                and assistant_message.reasoning_content
            ):
                entry["reasoning_content"] = assistant_message.reasoning_content
            turn_entries.append(entry)

            if not is_intermediate:
                # Final iteration: record in trace and return.
                final_text = iteration_text
                self.agent_trace.append({
                    "type": "assistant_response",
                    "content": final_text,
                    "iteration": iteration,
                })
                return final_text, iteration, turn_entries

            # --- Intermediate iteration: handle tool calls ---
            # Control reasoning_content propagation within the tool-call loop
            # based on reasoning_mode.
            if self.config.reasoning_mode == ReasoningMode.NONE:
                # Strip reasoning before appending to turn_messages
                assistant_message = assistant_message.model_copy(
                    update={"reasoning_content": None}
                )
            # For SINGLE_TURN and ALL_TURNS, keep reasoning_content as-is
            turn_messages.append(assistant_message)

            # Record the model's intermediate reasoning text (if any). LLMs
            # are allowed to emit `assistant.content` alongside `tool_calls`
            # (e.g. "let me check the weather first"), and this text is
            # part of the trace's causal story but otherwise lives only in
            # `conversation_history` as the iteration's message entry.
            # Skip empty content so the trace doesn't get padded with
            # blank entries when the model goes straight to tool calls.
            if iteration_text:
                self.agent_trace.append({
                    "type": "intermediate_reasoning",
                    "content": iteration_text,
                    "iteration": iteration,
                })

            # Record tool calls in trace
            tool_call_records = []
            tool_calls_list: List[Any] = (
                list(assistant_message.tool_calls)
                if assistant_message.tool_calls
                else []
            )
            for tool_call in tool_calls_list:
                tool_call_records.append({
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                })

            self.agent_trace.append({
                "type": "tool_calls",
                "calls": tool_call_records,
                "iteration": iteration,
            })

            # Execute each tool call
            for tool_call in tool_calls_list:
                arguments_str = json.dumps(
                    tool_call.arguments, ensure_ascii=False
                )

                # Short-circuit: if arguments contain "raw_arguments", the
                # original JSON from the model was malformed and could not be
                # parsed. Return a clear parse-error message immediately
                # instead of forwarding garbage to the real tool executor.
                if "raw_arguments" in tool_call.arguments:
                    raw_input = tool_call.arguments["raw_arguments"]
                    # Truncate very long garbage (e.g. repetition loops) to
                    # keep the context window manageable for the model.
                    max_display_len = 500
                    display_raw = (
                        raw_input[:max_display_len] + "..."
                        if len(raw_input) > max_display_len
                        else raw_input
                    )
                    result_content = json.dumps(
                        {"error": f"工具参数JSON解析失败，请检查JSON格式是否合法（如：字符串值是否加了引号、括号是否闭合等）。原始参数: {display_raw}"},
                        ensure_ascii=False,
                    )
                    has_error = True
                    self.tool_stats["tool_errors"] += 1
                    tool_message = ToolMessage(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=result_content,
                        error=True,
                    )
                else:
                    try:
                        tool_result_dict = self.tool_registry.execute_tool(
                            tool_call.name,
                            **tool_call.arguments,
                        )
                        result_content = tool_result_dict.get("result", "")
                        cache_hit = tool_result_dict.get("cache_hit", False)
                        has_error = tool_result_dict.get("has_error", False)

                        if cache_hit:
                            self.tool_stats["cache_hits"] += 1
                        else:
                            self.tool_stats["cache_misses"] += 1
                        if has_error:
                            self.tool_stats["tool_errors"] += 1

                        tool_message = ToolMessage(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=result_content,
                            error=has_error,
                        )
                    except Exception as exc:
                        self.tool_stats["tool_errors"] += 1
                        self.tool_stats["cache_misses"] += 1
                        result_content = f"Error executing tool: {str(exc)}"
                        has_error = True
                        tool_message = ToolMessage(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            content=result_content,
                            error=True,
                        )

                # Append tool result to temporary turn messages (for multi-step)
                turn_messages.append(tool_message)

                # Append tool entries to chronological accumulator
                # (immediately following this iteration's message entry).
                turn_entries.append({
                    "type": "tool_call",
                    "name": tool_call.name,
                    "arguments": arguments_str,
                })
                turn_entries.append({
                    "type": "tool_result",
                    "name": tool_call.name,
                    "result": result_content,
                    "error": has_error,
                })

                # Record tool result in trace
                self.agent_trace.append({
                    "type": "tool_result",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "result": result_content,
                    "error": tool_message.error,
                    "iteration": iteration,
                })

            self.tool_stats["tool_calls_executed"] += len(tool_calls_list)

            # --- Repetitive tool-call detection ---
            # Strategy 1 (consecutive): if two consecutive iterations call
            # any of the same (tool_name, arguments_json) pairs, we consider
            # it a consecutive repetitive loop and break immediately.
            current_signature = frozenset(
                (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
                for tc in tool_calls_list
            )
            _recent_tool_signatures.append(current_signature)
            if len(_recent_tool_signatures) >= _MAX_CONSECUTIVE_REPEATS:
                tail = _recent_tool_signatures[-_MAX_CONSECUTIVE_REPEATS:]
                overlapping_calls = tail[0] & tail[1]
                if overlapping_calls:
                    self._last_turn_repetitive = True
                    self._repetitive_reason = "consecutive"
                    self.agent_trace.append({
                        "type": "repetitive_tool_calls_detected",
                        "reason": "consecutive_repeated_tool_calls",
                        "repeated_calls": [
                            {"name": n, "arguments": a} for n, a in overlapping_calls
                        ],
                        "repeat_count": _MAX_CONSECUTIVE_REPEATS,
                        "iteration": iteration,
                    })
                    break  # Fall through to the forced-finish path below

            # Strategy 2 (global frequency): track every unique (name, args)
            # signature across all iterations. If any signature is called
            # more than _MAX_SAME_CALL_COUNT times, it means the model is
            # cycling through tool calls (possibly with different tools in
            # between) and inflating the context. Terminate immediately.
            _exceeded_signature = None
            for tc in tool_calls_list:
                sig = (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
                sig_key = f"{sig[0]}|{sig[1]}"
                _global_counts[sig_key] = _global_counts.get(sig_key, 0) + 1
                if _global_counts[sig_key] > _MAX_SAME_CALL_COUNT:
                    _exceeded_signature = sig
                    break

            if _exceeded_signature is not None:
                self._last_turn_repetitive = True
                self._repetitive_reason = "multiple"
                self.agent_trace.append({
                    "type": "repetitive_tool_calls_detected",
                    "reason": "multiple_repeated_tool_calls",
                    "repeated_call": {
                        "name": _exceeded_signature[0],
                        "arguments": _exceeded_signature[1],
                    },
                    "call_count": _global_counts[f"{_exceeded_signature[0]}|{_exceeded_signature[1]}"],
                    "max_allowed": _MAX_SAME_CALL_COUNT,
                    "iteration": iteration,
                })
                break  # Fall through to the forced-finish path below

        # Max iterations reached (or repetitive tool-call loop detected) -
        # force one final generation WITHOUT tools, instructing the model
        # to produce a substantive response based on whatever it has
        # gathered so far.
        self.agent_trace.append({
            "type": "max_iterations_reached",
            "iteration": iteration,
        })

        # Rebuild the final messages: keep the same conversation context and
        # prior tool interactions in turn_messages, but append a forcing
        # instruction and call LLM with tools disabled.
        forced_system = SystemMessage(
            content=system_content + MULTI_USER_AGENT_FORCE_FINISH_INSTRUCTION
        )
        forced_messages: List[Any] = [forced_system] + turn_messages[1:]

        try:
            final_assistant_message, _ = self.client.generate_response(
                messages=forced_messages,
                tools=None,
            )
            final_text = (final_assistant_message.content or "").strip()
        except Exception as exc:
            # Fallback only if the forced generation itself fails.
            final_text = (
                f"（系统提示：本轮工具调用已达上限，且强制收尾生成失败：{exc}。"
                "请基于上方已有信息继续讨论。）"
            )

        # Append the forced final message as the closing entry. has_tool_calls
        # is False since this is the outward reply.
        turn_entries.append({
            "type": "message",
            "role": "Agent",
            "content": final_text,
            "has_tool_calls": False,
        })

        self.agent_trace.append({
            "type": "assistant_response",
            "content": final_text,
            "iteration": iteration,
            "forced_finish": True,
        })

        return final_text, iteration, turn_entries

    def generate_preference_summary(
        self,
        conversation_history: List[Dict[str, Any]],
        history_override: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a preference table update (convergence step 1).

        This output is NOT added to conversation history - it only updates the
        Agent's internal preference table in the system prompt.
        Uses format_context with include_tool_info=False because tool call
        records are irrelevant for preference extraction.

        Args:
            conversation_history: The global structured conversation history
                (used as fallback if history_override is None).
            history_override: Optional alternative history slice to feed the
                LLM. The post-plan refresh path passes a truncated history
                (without the final travel_plan message and its tool entries)
                so that the snapshot reflects "what the Agent knew right
                before deciding to emit the plan", not the plan itself.

        Returns:
            The raw preference summary text from the LLM.
        """
        history_for_prompt = (
            history_override if history_override is not None else conversation_history
        )
        # The previous preference table is always part of the system
        # prompt, so the LLM can do an incremental update rather than
        # re-extracting from scratch.
        system_content = self._build_system_prompt()
        shared_context_str = format_context(
            history_for_prompt, include_tool_info=False
        )
        user_content = (
            f"[聊天记录开始]\n{shared_context_str}\n[聊天记录结束]\n\n"
            f"{MULTI_USER_AGENT_PREFERENCE_SUMMARY_PROMPT}"
        )
        messages = [
            SystemMessage(content=system_content),
            UserMessage(content=user_content),
        ]
        response, _ = self.client.generate_response(messages=messages)
        raw_text = (response.content or "").strip()

        # Try to extract and update the preference table
        self._update_preference_table(raw_text)

        self.agent_trace.append({
            "type": "preference_summary",
            "content": raw_text,
        })

        return raw_text

    def generate_convergence_summary(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """
        Generate a convergence summary (convergence step 2).

        This IS a normal response that goes into conversation history.
        The Agent may call tools during this process.
        Uses enriched context (include_tool_info=True).

        Args:
            conversation_history: The global structured conversation history.

        Returns:
            Tuple of (summary_text, num_steps, turn_tool_entries).
        """
        return self.generate_response(
            conversation_history=conversation_history,
            extra_instruction=MULTI_USER_AGENT_CONVERGENCE_PROMPT,
        )

    def generate_final_plan(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> Tuple[str, int, List[Dict[str, Any]]]:
        """
        Generate the final travel plan (max_turn fallback).
        Uses enriched context (include_tool_info=True).

        The final plan instruction is injected both in the system prompt
        (via extra_instruction) AND as the last UserMessage in the
        conversation. This dual injection ensures the model attends to
        the "generate plan NOW" directive even in very long contexts
        where a system-prompt-only append may be overlooked.

        Args:
            conversation_history: The global structured conversation history.

        Returns:
            Tuple of (plan_text, num_steps, turn_tool_entries).
        """
        # Append a synthetic message to conversation_history so that
        # build_agent_messages places it as the last UserMessage in the
        # model's view. We use a temporary copy to avoid mutating the
        # caller's history (the orchestrator will extend history with
        # turn_entries after this call returns).
        # Using a non-Agent role ensures build_agent_messages merges it
        # into a trailing UserMessage, making it the last thing the model
        # sees before generating its reply.
        history_with_directive = list(conversation_history) + [{
            "type": "message",
            "role": "user",
            "content": f"[系统指令] {MULTI_USER_AGENT_FINAL_PLAN_PROMPT}",
            "has_tool_calls": False,
        }]
        return self.generate_response(
            conversation_history=history_with_directive,
            extra_instruction=MULTI_USER_AGENT_FINAL_PLAN_PROMPT,
        )

    def try_update_preference_table_from_text(self, raw_text: str) -> bool:
        """Public wrapper around `_update_preference_table` for opportunistic
        parsing of an Agent's free-form reply.

        The orchestrator calls this after every Agent message: if the message
        happens to embed a preference-table JSON, the internal table is
        refreshed for free. No-op when the text contains no such JSON.
        Returns True iff the table was actually updated.
        """
        before = self.preference_table
        self._update_preference_table(raw_text)
        return self.preference_table != before

    def _update_preference_table(self, raw_text: str) -> None:
        """
        Parse the LLM output and overwrite the internal preference table.

        The expected output is a bare JSON object whose top level is the
        preference table itself, i.e. `{"User1": {...}, "User2": {...}}`,
        where each value is the per-user preference dict (containing
        `global_constraints` and/or `city_specific_preferences`).

        Recognition is structural: the parsed JSON must be a dict whose
        values look like per-user preference dicts (i.e. each value is a
        dict containing at least one of the expected top-level preference
        fields). This both rules out unrelated JSON (e.g. a tool argument
        blob, a travel_plan) and tolerates partial snapshots.

        Rationale for full overwrite (not merge): the preference summary
        prompt requires the LLM to output a complete current snapshot.
        Merging would mask schema-violating outputs and, more importantly,
        would prevent legitimate withdrawals (e.g. a user compromising on
        a `must` field, correctly dropped from the new snapshot but
        resurrected by a merge).

        If parsing or validation fails, the existing preference table is
        left untouched.
        """
        json_str = self._extract_json_from_text(raw_text)
        if json_str is None:
            return

        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return

        if not self._looks_like_preference_table(parsed):
            return

        self.preference_table = json.dumps(
            parsed, ensure_ascii=False, indent=2
        )

    @staticmethod
    def _looks_like_preference_table(obj: Any) -> bool:
        """Heuristic structural check for a bare preference table.

        A valid table is `{user_name: per_user_pref_dict, ...}` where each
        per-user dict contains at least one of the canonical top-level
        preference fields. This is strict enough to reject travel_plan
        JSON (which has `days` / `activities` at the top level) and tool
        argument blobs, while tolerating partial snapshots that omit one
        of the two main sections.
        """
        if not isinstance(obj, dict) or not obj:
            return False
        per_user_fields = {"global_constraints", "city_specific_preferences"}
        for value in obj.values():
            if not isinstance(value, dict):
                return False
            if not (per_user_fields & value.keys()):
                return False
        return True

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract a JSON object string from text that may contain markdown
        code blocks or other surrounding text.
        
        Returns the JSON string if found, None otherwise.
        """
        # Try to find JSON in markdown code block
        import re
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find a top-level JSON object by matching braces
        brace_start = text.find("{")
        if brace_start == -1:
            return None

        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start : i + 1]

        return None

    @staticmethod
    def check_is_travel_plan(text: str) -> bool:
        """
        Check whether the Agent's output contains a complete travel plan.

        Recognition is structural and matches the bare plan shape the
        prompt asks the model to emit:
            {"query": ..., "days": [{"day": 1, "date": ...,
                                     "city_segments": [...]}]}

        Specifically: top-level dict with a non-empty `days` list whose
        first entry is a dict containing a non-empty `city_segments` list.
        We don't probe further into segment internals here — partially
        malformed segments should still let the plan flow into evaluation
        (where format errors are scored), rather than being silently
        misclassified as ongoing chat.
        """
        json_str = GroupTravelAgent._extract_json_from_text(text)
        if json_str is None:
            return False
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return False
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

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the full agent trace including tool calls and results."""
        return self.agent_trace

    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool execution statistics."""
        return self.tool_stats
