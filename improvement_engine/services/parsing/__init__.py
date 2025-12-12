"""Parsing layer for conversations and scope extraction.

Provides:
- ConversationParser: Parses conversation structure (parsing only)
- ScopeExtractor: Extracts scope content using config (NO hardcoding)
"""

from .conversation_parser import (
    ConversationParser,
    ParsedConversation,
    ParsedMessage,
)
from .scope_extractor import ScopeExtractor

__all__ = [
    "ConversationParser",
    "ParsedConversation",
    "ParsedMessage",
    "ScopeExtractor",
]
