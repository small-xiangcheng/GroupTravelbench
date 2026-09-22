"""
Main entry point for the GroupTravelbench framework.

This module provides the CLI interface for running multi-user travel planning
conversations.

Usage:
    # Batch run all entries in a JSONL file with concurrency
    python -m group_travelbench run --file datas/test.jsonl --max-concurrency 4

    # Repeat each data entry 3 times (independent trials)
    python -m group_travelbench run --file datas/test.jsonl --num-trials 3

    # Inline JSON (single conversation)
    python -m group_travelbench run --json '{"query": "...", "user_preferences": {...}, "initial_messages": [...]}'

    python -m group_travelbench tools
    python -m group_travelbench status
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

from .core.config import (
    OpenAIConfig,
    BenchmarkConfig,
    SandboxConfig,
    SandboxMode,
    DEFAULT_CACHE_DIR,
)
from .core.sandbox_manager import SandboxManager
from .core.multi_interaction import run_multi_user_conversation

from .tools import *  # noqa: F401, F403
from .tools.tool_list import TOOL_NAMES


# ============================================================================
# Configuration helpers
# ============================================================================

def parse_llm_args(llm_args_str: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON string of extra LLM arguments into a dict.

    Returns {} for None / empty / whitespace input. Raises a clear
    ``ValueError`` on malformed JSON or non-dict payloads so the CLI can
    surface a friendly error instead of a generic stack trace.
    """
    if not llm_args_str or not llm_args_str.strip():
        return {}
    try:
        parsed = json.loads(llm_args_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for llm-args: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"llm-args must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed

def create_openai_config(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    extra_args: Optional[Dict[str, Any]] = None,
) -> OpenAIConfig:
    """Create an OpenAIConfig from environment variables and provided params.

    ``extra_args`` (parsed from a CLI ``--*-llm-args`` JSON string) overrides
    any of the defaults below — most usefully ``api_base``, ``temperature``,
    ``max_tokens`` for self-deployed models behind custom endpoints.
    """
    config_kwargs: Dict[str, Any] = {
        "model_name": model_name,
        "api_key": os.getenv("OPENAI_API_KEY", "your-api-key"),
        "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_args:
        config_kwargs.update(extra_args)
    return OpenAIConfig(**config_kwargs)


# ============================================================================
# Data loading
# ============================================================================

def load_data_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load one or more data entries from a JSONL file."""
    datas: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                datas.append(json.loads(line))
    return datas


# ============================================================================
# Sandbox setup (process-global side effect)
# ============================================================================

def setup_sandbox(
    sandbox_mode: str,
    cache_dir: str,
    agent_llm: str,
    tool_llm: str,
    agent_llm_args: Optional[Dict[str, Any]] = None,
    tool_llm_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Initialize the process-global sandbox.

    Note: SandboxManager.initialize_sandbox() mutates module-level globals
    (set_sandbox_mode, set_llm_simulator). All subsequent conversations in
    the same process share that state — there is no per-conversation
    sandbox handle to thread through.

    ``tool_llm`` is the model used by the sandbox tool simulator (cache miss
    fallback). It can be the same as ``agent_llm`` or independent — see
    ``--tool-llm`` / ``--tool-llm-args`` on the CLI.
    """
    sandbox_mode_map = {
        "isolated": SandboxMode.ISOLATED,
    }

    agent_config = create_openai_config(
        agent_llm, temperature=0.7, extra_args=agent_llm_args,
    )
    # Tool simulator defaults to temperature=0 for deterministic stub outputs;
    # callers can still override via --tool-llm-args (including enable_thinking).
    # Role requirements: the user simulator must be a strong *thinking* model
    # (thinking ON) — a weak one degrades simulation quality badly — while the
    # tool simulator only fabricates tool payloads and works fine with a
    # non-thinking instruction-following model.
    tool_config = create_openai_config(
        tool_llm, temperature=0.0, extra_args=tool_llm_args,
    )

    benchmark_config = BenchmarkConfig(
        assistant_config=agent_config,
        user_simulator_config=agent_config,
        tool_simulator_config=tool_config,
        sandbox_config=SandboxConfig(
            enabled=True,
            mode=sandbox_mode_map[sandbox_mode],
            cache_dir=cache_dir,
            auto_setup_llm_simulator=True,
        ),
    )
    SandboxManager(benchmark_config).initialize_sandbox()
    print(f"✅ Sandbox initialized ({sandbox_mode} mode, cache: {cache_dir})")

# ============================================================================
# Run command
# ============================================================================

def run_one_conversation(
    data: Dict[str, Any],
    agent_config: OpenAIConfig,
    user_config: OpenAIConfig,
    max_turn: int,
    convergence_interval: int,
    debug: bool,
    task_index: int,
    trial_id: int,
) -> Dict[str, Any]:
    """Run a single multi-user conversation and return the enriched result.

    Validation errors and runtime exceptions are caught and packed into a
    failure result so that batch execution does not abort on a single bad task.
    """
    task_id = data.get("task_id", "")

    # Validate input
    if "query" not in data:
        return {
            "task_index": task_index,
            "task_id": task_id,
            "trial_id": trial_id,
            "success": False,
            "error": "Input data must contain 'query' field",
        }

    if not data.get("user_preferences"):
        return {
            "task_index": task_index,
            "task_id": task_id,
            "trial_id": trial_id,
            "success": False,
            "error": "Input data must contain non-empty 'user_preferences'",
        }

    if not data.get("initial_messages"):
        return {
            "task_index": task_index,
            "task_id": task_id,
            "trial_id": trial_id,
            "success": False,
            "error": "Input data must contain a non-empty 'initial_messages' field "
                     "(one opening statement per user).",
        }

    # In debug mode tasks run serially (max_concurrency forced to 1 by
    # `run_batch`), so a banner before each task cleanly delimits the
    # otherwise interleaved-looking log stream from `run_multi_user_conversation`.
    # In non-debug mode INFO logs are suppressed entirely; we instead emit a
    # single concise progress line per task in `run_batch` (the ✅/❌ lines).
    if debug:
        banner = "=" * 70
        print(
            f"\n{banner}\n"
            f"▶️  task_id={task_id or '<none>'} trial={trial_id} starting\n"
            f"{banner}",
            flush=True,
        )

    # Dynamically adjust max_turn and convergence_interval based on
    # difficulty_type in the data (overrides CLI defaults when present).
    difficulty = data.get("difficulty_type", "")
    DIFFICULTY_SETTINGS = {
        "easy": {"max_turn": 15, "convergence_interval": 3},
        "medium": {"max_turn": 20, "convergence_interval": 4},
        "hard": {"max_turn": 25, "convergence_interval": 5},
    }
    if difficulty in DIFFICULTY_SETTINGS:
        max_turn = DIFFICULTY_SETTINGS[difficulty]["max_turn"]
        convergence_interval = DIFFICULTY_SETTINGS[difficulty]["convergence_interval"]

    try:
        result = run_multi_user_conversation(
            data=data,
            agent_config=agent_config,
            user_config=user_config,
            max_turn=max_turn,
            convergence_interval=convergence_interval,
            debug=debug,
            trial_id=trial_id,
        )
        result["task_index"] = task_index
        result["task_id"] = task_id
        result["trial_id"] = trial_id
        result["success"] = True
        return result

    except Exception as exc:
        if debug:
            traceback.print_exc()
        # Classify the error for more informative completion_reason
        error_str = str(exc)
        if "maximum context length" in error_str or "context_length_exceeded" in error_str:
            failure_reason = "context_overflow"
        elif "Connection error" in error_str or "timeout" in error_str.lower():
            failure_reason = "connection_error"
        elif "rate limit" in error_str.lower() or "429" in error_str:
            failure_reason = "rate_limit"
        else:
            failure_reason = "runtime_error"
        return {
            "task_index": task_index,
            "task_id": task_id,
            "trial_id": trial_id,
            "success": False,
            "error": error_str,
            "completion_reason": failure_reason,
            "plan_generated": False,
        }

def _format_task_result_line(
    result: Dict[str, Any],
    current: int,
    total: int,
) -> str:
    """Format a single concise task progress line for non-debug mode."""
    task_id = result.get("task_id") or "<none>"
    trial_id = result.get("trial_id", 0)
    if result.get("success"):
        rounds = result.get("total_rounds", "?")
        reason = result.get("completion_reason", "?")
        duration = result.get("duration")
        duration_str = f", {duration:.1f}s" if isinstance(duration, (int, float)) else ""
        return (
            f"✅ [{current}/{total}] task_id={task_id} trial={trial_id} | "
            f"rounds={rounds}, reason={reason}{duration_str}"
        )
    err = str(result.get("error", "unknown"))
    if len(err) > 120:
        err = err[:120] + "..."
    return f"❌ [{current}/{total}] task_id={task_id} trial={trial_id} | Error: {err}"

# Transient error types that should be retried on resume.
RETRYABLE_COMPLETION_REASONS = {"rate_limit", "connection_error"}


def _load_completed_tasks(output_file: str) -> Tuple[set, set]:
    """Load task keys from an existing output JSONL file, separating
    permanently-completed tasks from transiently-failed ones.

    Returns:
        A tuple of (completed_keys, retryable_keys):
        - completed_keys: (task_id, trial_id) tuples for tasks that either
          succeeded or failed with a non-transient error (should be skipped).
        - retryable_keys: (task_id, trial_id) tuples for tasks that failed
          due to transient errors (rate_limit, connection_error) and should
          be retried.

    Malformed lines are skipped with a warning.
    """
    completed: set = set()
    retryable: set = set()
    if not os.path.exists(output_file):
        return completed, retryable
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    task_id = result.get("task_id", "")
                    trial_id = result.get("trial_id", 0)
                    key = (task_id, trial_id)
                    # Classify: transient failures go into retryable set
                    if (
                        not result.get("success")
                        and result.get("completion_reason") in RETRYABLE_COMPLETION_REASONS
                    ):
                        retryable.add(key)
                    else:
                        completed.add(key)
                except json.JSONDecodeError:
                    print(
                        f"⚠️  Skipping malformed line {line_num} in {output_file}"
                    )
    except OSError as exc:
        print(f"⚠️  Could not read {output_file}: {exc}")
    return completed, retryable


def _purge_retryable_from_output(output_file: str, retryable_keys: set) -> int:
    """Remove lines corresponding to retryable tasks from the output file.

    This ensures that stale failure records don't pollute the final
    evaluation results. Returns the number of lines removed.
    """
    if not retryable_keys or not os.path.exists(output_file):
        return 0

    kept_lines: List[str] = []
    removed_count = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                result = json.loads(stripped)
                key = (result.get("task_id", ""), result.get("trial_id", 0))
                if key in retryable_keys:
                    removed_count += 1
                    continue
            except json.JSONDecodeError:
                pass  # keep malformed lines as-is to avoid data loss
            kept_lines.append(stripped)

    with open(output_file, "w", encoding="utf-8") as f:
        for kept_line in kept_lines:
            f.write(kept_line + "\n")

    return removed_count


def run_batch(
    datas: List[Dict[str, Any]],
    agent_llm: str,
    user_llm: str,
    max_turn: int,
    convergence_interval: int,
    output_file: str,
    max_concurrency: int,
    num_trials: int,
    debug: bool,
    agent_llm_args: Optional[Dict[str, Any]] = None,
    user_llm_args: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> None:
    """Run a list of multi-user conversations concurrently and save all
    results into a single JSONL file (one result per line).

    Each input data entry is repeated ``num_trials`` times (independent
    trials), producing ``len(datas) * num_trials`` total tasks.

    When ``resume=True``, the output file is scanned for already-completed
    tasks (matched by ``task_id`` + ``trial_id``). Those tasks are skipped,
    and new results are appended to the existing file. This enables
    resuming interrupted batch runs without re-running finished tasks.

    Concurrency policy:
      - debug=True  → force max_concurrency=1. Detailed per-conversation
        INFO logs (round/event traces) are only useful when interleaved
        output won't garble them. Single-thread keeps the log readable.
      - debug=False → honor user-supplied max_concurrency. Per-conversation
        INFO logs are suppressed inside `run_multi_user_conversation`
        (root logger raised to WARNING), so the only output is a single
        concise ✅/❌ progress line per task.
    """
    agent_config = create_openai_config(
        agent_llm, temperature=0.7, extra_args=agent_llm_args,
    )
    user_config = create_openai_config(
        user_llm, temperature=0, extra_args=user_llm_args,
    )

    if debug and max_concurrency != 1:
        print(
            f"⚠️  debug=True forces max_concurrency=1 "
            f"(was {max_concurrency}) to keep per-conversation logs readable."
        )
        max_concurrency = 1

    # Build the full (data_index, trial_id) task list.
    all_tasks: List[Tuple[int, int, Dict[str, Any]]] = [
        (data_idx, trial, data)
        for data_idx, data in enumerate(datas)
        for trial in range(num_trials)
    ]

    # Resume support: skip tasks already present in the output file.
    if resume:
        completed_keys, retryable_keys = _load_completed_tasks(output_file)

        # Purge transient-failure records from the output file so they
        # don't pollute the final evaluation.
        if retryable_keys:
            purged = _purge_retryable_from_output(output_file, retryable_keys)
            print(
                f"🧹 Purged {purged} transient-failure record(s) "
                f"(reasons: {', '.join(RETRYABLE_COMPLETION_REASONS)}) "
                f"from {output_file} — these tasks will be retried."
            )

        if completed_keys or retryable_keys:
            # Only skip permanently-completed tasks; retryable ones are re-run.
            tasks = [
                (data_idx, trial, data)
                for data_idx, trial, data in all_tasks
                if (data.get("task_id", ""), trial) not in completed_keys
            ]
            skipped = len(all_tasks) - len(tasks)
            retried = len([
                t for t in tasks
                if (t[2].get("task_id", ""), t[1]) in retryable_keys
            ])
            print(
                f"🔄 Resume mode: {len(completed_keys)} completed (skipped), "
                f"{len(retryable_keys)} transient-failures (retrying), "
                f"remaining {len(tasks)} to run "
                f"(including {retried} retries)."
            )
        else:
            tasks = all_tasks
            print(f"🔄 Resume mode: no completed tasks found, running all.")
    else:
        tasks = all_tasks

    total_tasks_all = len(all_tasks)
    total_tasks_remaining = len(tasks)

    if total_tasks_remaining == 0:
        print("\n✅ All tasks already completed. Nothing to run.")
        return

    print(
        f"\n🚀 Running {total_tasks_remaining} task(s) "
        f"({total_tasks_all} total = {len(datas)} data × {num_trials} trial"
        f"{f', {total_tasks_all - total_tasks_remaining} skipped' if resume else ''})"
        f" with max_concurrency={max_concurrency}"
        f"{' (debug, verbose logs)' if debug else ' (task-level progress only)'}\n"
    )

    write_lock = Lock()
    start_time = time.time()

    # In resume mode, append to existing file; otherwise truncate.
    if not resume:
        with open(output_file, "w", encoding="utf-8") as _:
            pass

    def submit(data_idx: int, trial_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        return run_one_conversation(
            data=data,
            agent_config=agent_config,
            user_config=user_config,
            max_turn=max_turn,
            convergence_interval=convergence_interval,
            debug=debug,
            task_index=data_idx,
            trial_id=trial_id,
        )

    success_count = 0
    fail_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        future_to_task = {
            executor.submit(submit, data_idx, trial_id, data): (data_idx, trial_id)
            for data_idx, trial_id, data in tasks
        }
        for future in as_completed(future_to_task):
            data_idx, trial_id = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                # Should already be caught inside run_one_conversation, but be safe.
                result = {
                    "task_index": data_idx,
                    "task_id": datas[data_idx].get("task_id", ""),
                    "trial_id": trial_id,
                    "success": False,
                    "error": f"Uncaught exception: {exc}",
                }

            completed += 1
            if result.get("success"):
                success_count += 1
            else:
                fail_count += 1

            print(
                _format_task_result_line(result, completed, total_tasks_remaining),
                flush=True,
            )

            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    duration = time.time() - start_time
    skipped_count = total_tasks_all - total_tasks_remaining
    print("\n" + "=" * 60)
    print(f"✅ Batch completed in {duration:.1f}s")
    print(
        f"   Datas: {len(datas)}, Trials/data: {num_trials}, "
        f"Total tasks: {total_tasks_all}"
        f"{f', Skipped (resumed): {skipped_count}' if skipped_count else ''}"
        f", Ran: {total_tasks_remaining}"
        f", Success: {success_count}, Failed: {fail_count}"
    )
    if total_tasks_remaining:
        print(f"   Success rate (this run): {success_count / total_tasks_remaining:.1%}")
    print(f"   Output (JSONL): {output_file}")
    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    print("🚀 GroupTravelbench (Multi-User)")
    print(f"📦 Available tools: {', '.join(TOOL_NAMES)}")

    from .core.tools import sandbox_tool_registry

    all_tools = sandbox_tool_registry.get_tools()
    filtered_tools = [tool for tool in all_tools if tool.name in TOOL_NAMES]
    print(f"🛠️  Available tools: {len(filtered_tools)} (filtered from {len(all_tools)} total)")

    parser = argparse.ArgumentParser(
        description="GroupTravelbench - Multi-User Conversation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input data format (JSONL, one JSON object per line):
{
    "query": "吉林三天两夜的旅行计划",
    "time": "2025-07-15",                     # departure date
    "context": "出发地：北京",
    "user_preferences": {                     # required: each user needs preference + compromisable
        "User1": {
            "preference": {                   # preference profile:
                "global_constraints": {...},  #   - global_constraints
                "city_specific_preferences": {#   - city_specific_preferences (dict, key=city name)
                    "西安市": {...}
                }
            },
            "compromisable": false            # whether the user may yield (at most 2 yeses; then forced false)
        },
        "User2": {"preference": {...}, "compromisable": true}
    },
    "initial_messages": [                     # required: each user's opening statement (shipped in the dataset)
        {"role": "User1", "content": "我希望预算控制在2000以内..."},
        ...
    ]
}

Output result fields (per task, one JSON per line in --output):
  - task_index                 0-based index of the input row in the JSONL
  - task_id                    task_id from the input (may be empty)
  - trial_id                   trial number (0-based, 0..num_trials-1)
  - success                    whether the task finished successfully
  - conversation_history       full conversation (compromise markers stripped)
  - agent_trace                Agent trace, including tool calls
  - agent_preference_table     (2) Agent-maintained preference table (refreshed
                               from history before the plan was emitted)
  - effective_preferences      (3) original preferences + applied compromises
  - compromise_log             compromise audit log
  - original_preferences       (1) original input preferences (deep copy)
  - completion_reason          plan_generated / max_turn_reached / loop_ended

Notes:
  - User count is dynamic (2, 3, 4, ...); it is the number of keys in user_preferences
  - Keys in user_preferences must be unique (User1, User2, ... recommended)
  - city_specific_preferences must be a dict (key=city name), not a list
  - initial_messages is required; every row in datas/test.jsonl already has it
  - When --num-trials > 1, output lines = input rows × num_trials;
    each line is uniquely identified by (task_index, trial_id)
  - Environment variables: OPENAI_API_KEY, OPENAI_API_BASE
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run command ----
    run_parser = subparsers.add_parser("run", help="Run multi-user conversation(s)")

    # Input source (mutually exclusive)
    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", type=str,
        help="Path to JSONL input file (ALL entries will be run)",
    )
    input_group.add_argument(
        "--json", type=str,
        help="Inline JSON data string (single conversation)",
    )

    # Model configuration
    run_parser.add_argument(
        "--agent-llm", default="gpt-4o",
        help="Model for Agent (default: gpt-4o)",
    )
    run_parser.add_argument(
        "--agent-llm-args", type=str, default=None,
        help='JSON dict overriding Agent OpenAIConfig fields, e.g. '
             '\'{"max_tokens": 8192, "temperature": 0.7, '
             '"api_base": "https://my-endpoint/v1"}\'',
    )
    run_parser.add_argument(
        "--user-llm", default="gpt-4o",
        help="Model for User simulators (default: gpt-4o)",
    )
    run_parser.add_argument(
        "--user-llm-args", type=str, default=None,
        help="JSON dict overriding User-simulator OpenAIConfig fields "
             "(same schema as --agent-llm-args).",
    )
    run_parser.add_argument(
        "--tool-llm", default=None,
        help="Model for the sandbox tool simulator (cache-miss fallback). "
             "Defaults to --agent-llm if omitted. Only used when "
             "--sandbox-mode is set.",
    )
    run_parser.add_argument(
        "--tool-llm-args", type=str, default=None,
        help="JSON dict overriding Tool-simulator OpenAIConfig fields "
             "(same schema as --agent-llm-args).",
    )

    # Interaction parameters
    run_parser.add_argument(
        "--max-turn", type=int, default=50,
        help="Max interaction rounds (default: 50)",
    )
    run_parser.add_argument(
        "--convergence-interval", type=int, default=10,
        help="Rounds between convergence summaries (default: 10)",
    )

    # Concurrency
    run_parser.add_argument(
        "--max-concurrency", type=int, default=10,
        help="Max concurrent conversations when running a JSONL file (default: 10)",
    )

    # Trials
    run_parser.add_argument(
        "--num-trials", type=int, default=1,
        help="Number of trials per data entry; each trial runs independently "
             "with a distinct trial_id (default: 1)",
    )

    # Sandbox / tool cache
    run_parser.add_argument(
        "--sandbox-mode", type=str, default=None,
        choices=["isolated"],
        help="Enable the offline sandbox: replay cached tool results, and fall "
             "back to the LLM tool simulator on a cache miss",
    )
    run_parser.add_argument(
        "--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for sandbox (default: {DEFAULT_CACHE_DIR})",
    )

    # Output
    run_parser.add_argument(
        "--output", default="multi_user_results.jsonl",
        help="Output JSONL file path; one result per line "
             "(default: multi_user_results.jsonl)",
    )

    # Debug
    run_parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )

    # Resume (enabled by default; use --no-resume to force a fresh run)
    run_parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume: truncate the output file and re-run all tasks "
             "from scratch. By default, successfully completed tasks (matched "
             "by task_id + trial_id) in the output file are skipped and new "
             "results are appended.",
    )

    # ---- tools command ----
    subparsers.add_parser("tools", help="List available tools")

    # ---- status command ----
    subparsers.add_parser("status", help="Show cache status")

    args = parser.parse_args()

    if args.command == "run":
        # Load all input data (JSONL = batch; --json = single)
        if args.file:
            datas = load_data_from_jsonl(args.file)
            if not datas:
                print(f"❌ No data found in {args.file}")
                sys.exit(1)
            print(f"✅ Loaded {len(datas)} entries from {args.file}")
        else:
            try:
                datas = [json.loads(args.json)]
            except json.JSONDecodeError as exc:
                print(f"❌ Invalid JSON: {exc}")
                sys.exit(1)
            print("✅ Loaded inline JSON data (1 entry)")

        # Parse per-LLM extra args (JSON dicts) once and reuse.
        try:
            agent_llm_args = parse_llm_args(args.agent_llm_args)
            user_llm_args = parse_llm_args(args.user_llm_args)
            tool_llm_args = parse_llm_args(args.tool_llm_args)
        except ValueError as exc:
            print(f"❌ {exc}")
            sys.exit(1)

        # Initialize sandbox once (process-global side effect).
        # --tool-llm defaults to --agent-llm (model name only) when not
        # specified, preserving the old "tool simulator shares the agent
        # model" behavior. --tool-llm-args, however, is fully independent
        # of --agent-llm-args — matching the open-source travelbench CLI
        # and avoiding accidental routing of tool calls to a custom agent
        # endpoint via inherited api_base.
        if args.sandbox_mode:
            tool_llm = args.tool_llm or args.agent_llm
            setup_sandbox(
                sandbox_mode=args.sandbox_mode,
                cache_dir=args.cache_dir,
                agent_llm=args.agent_llm,
                tool_llm=tool_llm,
                agent_llm_args=agent_llm_args,
                tool_llm_args=tool_llm_args,
            )

        if args.num_trials < 1:
            print(f"❌ --num-trials must be >= 1 (got {args.num_trials})")
            sys.exit(1)

        run_batch(
            datas=datas,
            agent_llm=args.agent_llm,
            user_llm=args.user_llm,
            max_turn=args.max_turn,
            convergence_interval=args.convergence_interval,
            output_file=args.output,
            max_concurrency=args.max_concurrency,
            num_trials=args.num_trials,
            debug=args.debug,
            agent_llm_args=agent_llm_args,
            user_llm_args=user_llm_args,
            resume=not args.no_resume,
        )

    elif args.command == "tools":
        all_tools_openai = sandbox_tool_registry.to_openai_format()
        tools_openai_format = [
            tool for tool in all_tools_openai
            if tool.get("function", {}).get("name") in TOOL_NAMES
        ]

        print("🔧 Available tools:")
        print(f"📦 Tool modules: {', '.join(TOOL_NAMES)}")
        print(f"🛠️  Total tools: {len(tools_openai_format)}")

        for tool_spec in tools_openai_format:
            tool_func = tool_spec["function"]
            print(f"  - {tool_func['name']}: {tool_func['description']}")

        print("\n📋 OpenAI Function Calling Format:")
        print("   Tools are ready for use with OpenAI-style function calling")
        if tools_openai_format:
            print(f"   Example tool spec: {json.dumps(tools_openai_format[0], indent=4, ensure_ascii=False)}")

    elif args.command == "status":
        cache_dir = DEFAULT_CACHE_DIR
        if not os.path.exists(cache_dir):
            print(f"📂 Cache directory {cache_dir} does not exist")
            return

        try:
            from .core.tools import get_cache_stats

            stats = get_cache_stats(cache_dir)
            if not stats:
                print(f"📭 No cache data found in {cache_dir}")
                return

            print(f"📊 Cache Status ({cache_dir}):")
            for tool_name, stat in stats.items():
                print(
                    f"  - {tool_name}: "
                    f"{stat.get('cached_calls', 0)} cached, "
                    f"{stat.get('missed_calls', 0)} missed"
                )

        except Exception as exc:
            print(f"❌ Error reading cache: {exc}")
            traceback.print_exc()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
