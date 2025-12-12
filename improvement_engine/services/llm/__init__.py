"""LLM layer - prompt rendering and LLM client wrapper.

Provides:
- PromptRenderer: Renders templates (template rendering only)
- LLMClient: Wrapper around shared.llm (API calls only)
"""

from .prompt_renderer import PromptRenderer
from .llm_client import LLMClient

__all__ = [
    "PromptRenderer",
    "LLMClient",
]
