"""
Message types and management for GroupTravelbench.
Simplified version based on tau2 message system.
"""

import json
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().isoformat()


# Role types
SystemRole = Literal["system"]
UserRole = Literal["user"] 
AssistantRole = Literal["assistant"]
ToolRole = Literal["tool"]


class ToolCall(BaseModel):
    """A tool call made by the assistant."""
    
    id: str = Field(description="Unique identifier for the tool call")
    name: str = Field(description="Tool function name")
    arguments: Dict[str, Any] = Field(description="Tool function arguments")
    
    def __str__(self) -> str:
        return f"ToolCall(id={self.id}, name={self.name}, args={json.dumps(self.arguments, indent=2)})"


class SystemMessage(BaseModel):
    """System message containing instructions."""
    
    role: SystemRole = Field(default="system")
    content: str = Field(description="System message content")
    timestamp: str = Field(default_factory=get_timestamp)
    
    def __str__(self) -> str:
        return f"SystemMessage: {self.content[:100]}..."


class UserMessage(BaseModel):
    """Message from user (or user simulator)."""
    
    role: UserRole = Field(default="user")
    content: str = Field(description="User message content")
    timestamp: str = Field(default_factory=get_timestamp)
    turn_idx: Optional[int] = Field(default=None, description="Turn index in conversation")
    
    def __str__(self) -> str:
        return f"UserMessage(turn={self.turn_idx}): {self.content[:100]}..."


class AssistantMessage(BaseModel):
    """Message from assistant."""
    
    role: AssistantRole = Field(default="assistant")
    content: Optional[str] = Field(default=None, description="Assistant message content")
    reasoning_content: Optional[str] = Field(default=None, description="Reasoning/thinking content returned by the model, stored as-is for multi-turn context")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Tool calls made by assistant")
    timestamp: str = Field(default_factory=get_timestamp)
    turn_idx: Optional[int] = Field(default=None, description="Turn index in conversation")
    
    # Token usage and cost tracking
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage information")
    cost: Optional[float] = Field(default=None, description="API call cost")
    
    def has_text_content(self) -> bool:
        """Check if message has text content."""
        return self.content is not None and self.content.strip() != ""
    
    def has_tool_calls(self) -> bool:
        """Check if message has tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def __str__(self) -> str:
        if self.has_text_content():
            return f"AssistantMessage(turn={self.turn_idx}): {self.content[:100]}..."
        elif self.has_tool_calls():
            calls = [f"{tc.name}({tc.id})" for tc in self.tool_calls]
            return f"AssistantMessage(turn={self.turn_idx}): ToolCalls[{', '.join(calls)}]"
        return f"AssistantMessage(turn={self.turn_idx}): [empty]"


class ToolMessage(BaseModel):
    """Message containing tool execution result."""
    
    role: ToolRole = Field(default="tool")
    tool_call_id: str = Field(description="ID of the tool call this responds to")
    name: str = Field(description="Tool name")
    content: str = Field(description="Tool execution result")
    error: bool = Field(default=False, description="Whether tool execution failed")
    timestamp: str = Field(default_factory=get_timestamp)
    turn_idx: Optional[int] = Field(default=None, description="Turn index in conversation")
    
    def __str__(self) -> str:
        status = "ERROR" if self.error else "SUCCESS"
        return f"ToolMessage({self.name}, {status}): {self.content[:100]}..."


# Union type for all message types
Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage

# OpenAI API compatible message format
def to_openai_format(message: Message) -> Dict[str, Any]:
    """Convert message to OpenAI API format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    
    elif isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    
    elif isinstance(message, AssistantMessage):
        result = {"role": "assistant", "content": message.content or ""}
        
        if message.reasoning_content:
            # Both fields for compatibility:
            # - "reasoning_content" for OpenAI-compatible APIs
            # - "reasoning" for vLLM's chat_template rendering
            result["reasoning_content"] = message.reasoning_content
            result["reasoning"] = message.reasoning_content
        
        if message.has_tool_calls():
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in message.tool_calls
            ]
            # DashScope thinking mode requires reasoning_content for all
            # assistant messages that contain tool_calls. If the model
            # didn't produce reasoning, inject empty string to avoid 400.
            if "reasoning_content" not in result:
                result["reasoning_content"] = ""
                result["reasoning"] = ""
        
        return result
    
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content
        }
    
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


def to_openai_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert list of messages to OpenAI API format."""
    return [to_openai_format(msg) for msg in messages]
