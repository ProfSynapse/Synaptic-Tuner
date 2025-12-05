"""
SelfPlay - Synthetic Dataset Generation via 3-Prompt Pipeline

A sophisticated self-play system for generating training data where a fine-tuned model
generates both sides of the conversation through a three-prompt pipeline:

1. Prompt 1: Generate realistic workspace environment
2. Prompt 2: Generate vague user request
3. Prompt 3: Generate assistant response with tool calls

The system supports both tool-based and behavioral generation modes.
"""

from .id_utils import (
    generate_session_id,
    generate_workspace_id,
    generate_ids,
    validate_session_id,
    validate_workspace_id,
)

__version__ = "1.0.0"
__all__ = [
    "generate_session_id",
    "generate_workspace_id",
    "generate_ids",
    "validate_session_id",
    "validate_workspace_id",
]
