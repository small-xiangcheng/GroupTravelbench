"""
Thread-local task context for multi-threaded batch execution.

In high-concurrency scenarios (ThreadPoolExecutor), each conversation runs
in its own thread. This module provides a lightweight mechanism to attach
a `task_id` and `trial_id` to the current thread so that loggers anywhere
in the call stack can include them without explicitly threading the values
through every function signature.

Usage:
    from .task_context import set_task_context, task_prefix

    # At the top of a conversation thread:
    set_task_context("task_001234", trial_id=2)

    # Anywhere deeper in the call stack:
    prefix = task_prefix()  # returns "[task_001234/trial=2] "
"""

import threading

_thread_local = threading.local()


def set_task_context(task_id: str, trial_id: int = -1) -> None:
    """Bind task_id and trial_id to the current thread."""
    _thread_local.task_id = task_id
    _thread_local.trial_id = trial_id


def set_task_id(task_id: str) -> None:
    """Bind a task_id to the current thread (backward-compatible)."""
    _thread_local.task_id = task_id


def get_task_id() -> str:
    """Retrieve the task_id bound to the current thread."""
    return getattr(_thread_local, "task_id", "")


def get_trial_id() -> int:
    """Retrieve the trial_id bound to the current thread (-1 if unset)."""
    return getattr(_thread_local, "trial_id", -1)


def task_prefix() -> str:
    """Return a log-friendly prefix like '[task_001234/trial=2] ' or '' if unset."""
    tid = get_task_id()
    trial = get_trial_id()
    if not tid:
        return ""
    if trial < 0:
        return f"[{tid}] "
    return f"[{tid}/trial={trial}] "