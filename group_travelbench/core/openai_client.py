"""
OpenAI API client for GroupTravelbench.
Handles all interactions with OpenAI API including tool calling.
"""

import json
import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from .config import OpenAIConfig
from .messages import Message, AssistantMessage, ToolCall, to_openai_messages
from .thinking import (
    build_thinking_extra_body,
    drop_unsupported_params,
    strip_known_unsupported,
)
from .tools import Tool


class OpenAIClient:
    """OpenAI API client with tool calling support."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        # Check if using OpenRouter API
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
        self.enable_thinking = config.enable_thinking
    
    def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Tuple[AssistantMessage, Dict[str, Any]]:
        """
        Generate response from OpenAI API.
        
        Args:
            messages: Conversation messages
            tools: Available tools for the assistant
            **kwargs: Additional arguments to override config
            
        Returns:
            Tuple of (AssistantMessage, usage_info)
        """
        # Convert messages to OpenAI format
        openai_messages = to_openai_messages(messages)
        # print("raw messages:", openai_messages)
        # Prepare request parameters
        request_params = {
            "model": self.config.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }
        if self.config.max_tokens is not None:
            request_params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        # Add tools if provided
        if tools:
            request_params["tools"] = [tool.to_openai_format() for tool in tools]
            request_params["tool_choice"] = "auto"
        # Add thinking-control parameters. This is compatible with vLLM
        # (enable_thinking) and SGLang (thinking): both keys are sent and each
        # backend picks up the one it understands, while a chat template simply
        # ignores keys it does not reference. Gateways that validate the body
        # strictly answer 400 for these optional keys; the retry loop below
        # drops the offending key and retries, so an optional parameter never
        # makes the call fail.
        extra_body = build_thinking_extra_body(
            enable_thinking=kwargs.get("enable_thinking", self.enable_thinking),
        )
        extra_body = strip_known_unsupported(self.config.api_base, extra_body)
        if extra_body:
            request_params["extra_body"] = extra_body
        # Retry mechanism
        max_retries = 15
        attempt = 0
        while attempt < max_retries:
            try:
                # Make API call
                stream_response = self.client.chat.completions.create(**request_params)
                response = stream_to_complete_response(stream_response)
                # Extract response data
                choice = response.choices[0]
                message = choice.message
                # Handle content - it might be a string, list, or None
                if message.content is None:
                    content = None
                elif isinstance(message.content, str):
                    content = message.content.strip()
                    # Strip thinking content: prefer text after </think>, fallback to text before it
                    if "</think>" in content:
                        after_think = content.split("</think>", 1)[-1].strip()
                        if after_think:
                            content = after_think
                        else:
                            content = content.split("</think>", 1)[0].strip()
                            # Remove leading <think> tag if present
                            if content.startswith("<think>"):
                                content = content[7:].strip()
                elif isinstance(message.content, list):
                    # Convert list to string (e.g., for structured outputs)
                    content = json.dumps(message.content, ensure_ascii=False)
                else:
                    # Fallback: convert to string
                    content = str(message.content)
                # Extract reasoning_content (field name varies by backend:
                # "reasoning_content" for some providers, "reasoning" for vLLM)
                reasoning_content = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)

                # Create AssistantMessage
                assistant_message = AssistantMessage(
                    content=content,
                    reasoning_content=reasoning_content,
                    usage=response.usage.model_dump() if response.usage else None
                )
                # Handle tool calls if present
                if message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        try:
                            arguments = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            # Handle malformed JSON
                            arguments = {"raw_arguments": tc.function.arguments}
                        
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments
                        ))
                    assistant_message.tool_calls = tool_calls
                
                # Usage information
                usage_info = {
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": choice.finish_reason
                }

                # Guard: empty content with no tool_calls is likely a
                # degenerate response (e.g. thinking consumed all tokens).
                if not content and not message.tool_calls:
                    attempt += 1
                    if attempt < max_retries:
                        retry_delay = attempt * 2
                        logger.warning(
                            f"Model returned empty content without tool_calls "
                            f"(finish_reason={choice.finish_reason}, "
                            f"attempt {attempt}/{max_retries}); "
                            f"retrying in {retry_delay}s."
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(
                            f"Model returned empty content after {max_retries} "
                            f"attempts; returning empty response as-is."
                        )

                return assistant_message, usage_info
                
            except Exception as e:
                # When the backend rejects the optional thinking-control
                # parameter (gateways differ in how strictly they validate the
                # body), drop the offending key and retry right away. Such a
                # failure neither consumes the retry budget nor waits, because
                # the request has changed - it is not the same attempt failing.
                if drop_unsupported_params(request_params, e, self.config.api_base):
                    continue

                attempt += 1
                if attempt < max_retries:
                    # Exponential backoff with full jitter:
                    # Randomizes retry delay within [0, min(base * 2^attempt, max_delay)]
                    # to avoid thundering herd when multiple concurrent requests hit QPS limits.
                    base_delay = 2
                    max_delay = 60
                    exponential = base_delay * (2 ** attempt)
                    retry_delay = random.uniform(0, min(exponential, max_delay))
                    time.sleep(retry_delay)
                else:
                    print("error:", openai_messages)
                    # Last attempt failed, raise exception
                    raise OpenAIClientError(
                        f"OpenAI API call failed after {max_retries} attempts: {str(e)}"
                    ) from e
    
    def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Create embeddings using OpenAI-compatible API.
        
        Args:
            texts: Text or list of texts to embed
            model: Model name (defaults to config model)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (embeddings, usage_info)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Retry mechanism
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=model or self.config.model_name,
                    **kwargs
                )
                
                # Extract embeddings
                embeddings = [item.embedding for item in response.data]
                
                # Usage info
                usage_info = {
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None,
                    "model": response.model
                }
                
                return embeddings, usage_info
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise OpenAIClientError(
                        f"Embeddings API call failed after {max_retries} attempts: {str(e)}"
                    ) from e

class OpenAIClientError(Exception):
    """Exception raised for OpenAI client errors."""
    pass


def stream_to_complete_response(stream) -> ChatCompletion:
    """
    Consume a streaming response and assemble it into a complete ChatCompletion object.
    Handles reasoning_content, content, and tool_calls field concatenation.
    """
    collected_id = None
    collected_model = None
    collected_created = None
    collected_usage = None
    finish_reason = None

    reasoning_content_parts = []
    content_parts = []
    tool_calls_map = {}  # index -> {id, function_name, function_arguments}

    for chunk in stream:
        if chunk.id:
            collected_id = chunk.id
        if chunk.model:
            collected_model = chunk.model
        if chunk.created:
            collected_created = chunk.created
        if chunk.usage:
            collected_usage = chunk.usage

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        if delta is None:
            continue

        # Accumulate reasoning_content (field name varies by backend:
        # "reasoning_content" for some providers, "reasoning" for vLLM)
        reasoning_chunk = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
        if reasoning_chunk:
            reasoning_content_parts.append(reasoning_chunk)

        # Accumulate content
        if delta.content:
            content_parts.append(delta.content)

        # Accumulate tool_calls
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                idx = tool_call_delta.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}

                if tool_call_delta.id:
                    tool_calls_map[idx]["id"] = tool_call_delta.id
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        tool_calls_map[idx]["name"] += tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_calls_map[idx]["arguments"] += tool_call_delta.function.arguments

    # Assemble the final ChatCompletion
    assembled_content = "".join(content_parts) if content_parts else None
    assembled_reasoning = "".join(reasoning_content_parts) if reasoning_content_parts else None

    assembled_tool_calls = None
    if tool_calls_map:
        assembled_tool_calls = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            assembled_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
            )

    message_kwargs = {
        "role": "assistant",
        "content": assembled_content,
        "tool_calls": assembled_tool_calls,
        "refusal": None,
    }
    # Store as both field names so downstream getattr finds it regardless
    # of which name the caller checks ("reasoning" for vLLM,
    # "reasoning_content" for other providers).
    if assembled_reasoning is not None:
        message_kwargs["reasoning"] = assembled_reasoning
        message_kwargs["reasoning_content"] = assembled_reasoning

    assembled_message = ChatCompletionMessage(**message_kwargs)

    assembled_choice = Choice(
        finish_reason=finish_reason or "stop",
        index=0,
        message=assembled_message,
        logprobs=None,
    )

    response = ChatCompletion(
        id=collected_id or "",
        choices=[assembled_choice],
        created=collected_created or 0,
        model=collected_model or "",
        object="chat.completion",
        usage=collected_usage,
    )

    return response