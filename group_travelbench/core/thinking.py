"""
Unified construction and compatibility handling for thinking (reasoning) params.

Background: the same code base talks to several OpenAI-compatible backends, and
each of them switches thinking on in a different way:

  - vLLM (Qwen3 family) : extra_body.chat_template_kwargs.enable_thinking = bool
  - SGLang (DeepSeek-V4) : extra_body.chat_template_kwargs.thinking       = bool
  - Some gateways        : chat_template_kwargs is silently ignored (no error)

The strategy is therefore "write both keys and let the backend decide":

  1. The switch is written as both ``thinking`` and ``enable_thinking`` inside
     ``chat_template_kwargs``. A backend only forwards these keys to the Jinja
     chat template as rendering arguments, and keys a template does not
     reference are ignored silently, so writing both is safe. SGLang normalizes
     reasoning parameters in the same way (it setdefaults both keys:
     ``thinking`` for deepseek-v3/v4 and kimi_k2, ``enable_thinking`` for
     qwen3, glm45, ...).
  2. Some gateways validate the body strictly and answer 400 for these optional
     keys, so ``drop_unsupported_params()`` is provided to remove the offending
     key, retry, and remember the conclusion per endpoint.
"""

import threading
from typing import Any, Dict, Optional


# Keys that may be dropped, ordered by "least compatible first".
_DROPPABLE_KEYS = ("chat_template_kwargs",)

# Per-endpoint set of keys already proven to be rejected, so later requests
# skip the doomed round-trip. key = api_base string, value = set of key names.
_unsupported: Dict[str, set] = {}
_unsupported_lock = threading.Lock()


def build_thinking_extra_body(
    enable_thinking: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Build an extra_body carrying the thinking-control parameters.

    Args:
        enable_thinking: None means "do not interfere at all" (no thinking
            related parameter is sent, the backend default applies); True or
            False explicitly enables / disables thinking.

    Returns:
        The extra_body dict, or None when nothing should be sent.
    """
    if enable_thinking is None:
        return None

    # SGLang understands "thinking", vLLM understands "enable_thinking";
    # send both so each backend picks up the one it knows.
    return {
        "chat_template_kwargs": {
            "thinking": enable_thinking,
            "enable_thinking": enable_thinking,
        }
    }


def strip_known_unsupported(endpoint: Optional[str], extra_body: Optional[Dict[str, Any]]):
    """Pre-remove keys this endpoint is known to reject, saving failed requests."""
    if not extra_body or not endpoint:
        return extra_body

    with _unsupported_lock:
        known = set(_unsupported.get(endpoint, ()))
    if not known:
        return extra_body

    pruned = {k: v for k, v in extra_body.items() if k not in known}
    return pruned or None


def drop_unsupported_params(
    request_params: Dict[str, Any],
    error: Exception,
    endpoint: Optional[str] = None,
) -> bool:
    """Drop one suspicious key when the backend rejects the optional params.

    Args:
        request_params: Parameters to retry with; modified in place.
        error: Exception raised by the previous call.
        endpoint: api_base, used to cache the conclusion.

    Returns:
        True if a key was actually removed and a retry is worthwhile;
        False if there is nothing to remove or the error is unrelated.
    """
    extra_body = request_params.get("extra_body")
    if not extra_body:
        return False

    if not _looks_like_bad_param(error):
        return False

    for key in _DROPPABLE_KEYS:
        if key in extra_body:
            extra_body.pop(key)
            if not extra_body:
                request_params.pop("extra_body", None)
            if endpoint:
                with _unsupported_lock:
                    _unsupported.setdefault(endpoint, set()).add(key)
            return True

    return False


def _looks_like_bad_param(error: Exception) -> bool:
    """Tell whether the error means "request parameter not accepted" rather
    than a network / server-side failure.

    Only parameter errors trigger the downgrade; transient failures are retried
    as usual so that they are not masked.
    """
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (400, 422, "400", "422"):
        return True

    msg = str(error).lower()
    if "400" in msg or "422" in msg:
        return True

    return any(
        kw in msg
        for kw in (
            "unrecognized request argument",
            "extra inputs are not permitted",
            "unexpected keyword argument",
            "invalid_request_error",
            "validation error",
            "chat_template_kwargs",
            "enable_thinking",
        )
    )
