"""
Configuration management for GroupTravelbench.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

# Default cache directory for sandbox environment.
#
# Anchor the default to the project root (two levels up from this file:
# core/config.py → group_travelbench/ → <repo_root>) instead of a CWD-relative
# path. Otherwise any process whose cwd happens to land elsewhere (e.g.
# IDE "Run File" with cwd=group_travelbench/tools/) will silently create an
# empty `sandbox_cache/` directory next to itself when tool modules are
# imported (`SandboxBaseTool.__init__` triggers `get_sandbox_cache_manager`,
# which calls `os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)`).
#
# Resolution order:
#   1. SANDBOX_CACHE_DIR env var (honored verbatim, may be relative or absolute)
#   2. <repo_root>/sandbox_cache  (absolute, stable regardless of cwd)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_CACHE_DIR = os.environ.get(
    "SANDBOX_CACHE_DIR",
    os.path.join(_PROJECT_ROOT, "sandbox_cache"),
)

class SandboxMode(Enum):
    """Sandbox operation modes."""
    ISOLATED = "isolated"       # Cache only + LLM simulation for misses

class SandboxConfig(BaseModel):
    """Sandbox environment configuration."""
    
    enabled: bool = Field(default=False, description="Enable sandbox environment")
    mode: SandboxMode = Field(default=SandboxMode.ISOLATED, description="Sandbox operation mode")
    cache_dir: str = Field(default_factory=lambda: DEFAULT_CACHE_DIR, description="Directory for cache storage")
    max_examples: int = Field(default=8, description="Max examples to use for LLM simulation")
    simulation_temperature: float = Field(default=0.7, description="Temperature for LLM simulation")
    auto_setup_llm_simulator: bool = Field(default=True, description="Auto setup LLM simulator for isolated mode")
    use_similarity_retrieval: bool = Field(default=True, description="Use similarity-based retrieval instead of random sampling")
    
    # Remote embedding service configuration (independent of the agent / user-simulator)
    use_remote_embedding: bool = Field(default=False, description="Use remote embedding service instead of local model")
    embedding_service_url: Optional[str] = Field(default=None, description="Remote embedding service URL (e.g., http://localhost:8001/v1)")
    embedding_model_name: Optional[str] = Field(default=None, description="Embedding model name for remote service")


class ReasoningMode(str, Enum):
    """Controls how reasoning_content is preserved and passed back to the API.

    - NONE: Never save or pass back reasoning_content.
    - SINGLE_TURN: Pass reasoning_content within a single tool-call loop
      (same generate_response call), but discard across turns.
    - ALL_TURNS: Persist reasoning_content in conversation_history and
      pass it back across all turns.
    """
    NONE = "none"
    SINGLE_TURN = "single_turn"
    ALL_TURNS = "all_turns"

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    
    model_name: str = Field(default="gpt-4", description="Model name to use")
    api_key: str = Field(description="OpenAI API key")
    api_base: str = Field(default="https://api.openai.com/v1", description="API base URL")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    timeout: Optional[int] = Field(default=None, description="Request timeout in seconds (None = no timeout)")
    enable_thinking: bool = Field(
        default=False,
        description="Send an explicit thinking switch to the backend. Intended per role: "
                    "ON for the Agent and ON for the user simulator (a weak / non-thinking "
                    "user simulator degrades conversation quality badly), OFF for the tool "
                    "simulator (it only fabricates plausible tool payloads)."
    )
    reasoning_mode: ReasoningMode = Field(default=ReasoningMode.ALL_TURNS, description="How to handle reasoning_content: none/single_turn/all_turns")


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration."""
    
    # Role-specific model requirements:
    #   - assistant: the model under test.
    #   - user simulator: MUST be a strong thinking model — it has to play the
    #     users well enough to drive the conversation; a weak model degrades
    #     simulation quality badly (thinking ON).
    #   - tool simulator: easy job (fabricate plausible tool payloads from cached
    #     examples), an instruction-following model is enough (thinking OFF).
    assistant_config: OpenAIConfig = Field(description="Configuration for the assistant (model under test)")
    user_simulator_config: OpenAIConfig = Field(description="Configuration for the user simulator (strong thinking model required)")
    tool_simulator_config: OpenAIConfig = Field(description="Configuration for the tool simulator (instruction-following model is enough, thinking not needed)")
    max_conversation_turns: int = Field(default=20, description="Maximum conversation turns")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Travel domain settings
    supported_languages: List[str] = Field(default=["en"], description="Supported languages")
    
    # Sandbox environment settings
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig, description="Sandbox environment configuration")
