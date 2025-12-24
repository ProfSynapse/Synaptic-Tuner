"""
Shared validation infrastructure for Synaptic-Tuner.

This module provides unified validation across:
- SynthChat: Synthetic data generation validation
- Evaluator: Model response evaluation
- Trainer: Training-time fitness evaluation

Architecture:
    parsing/    - Format-agnostic response parsing (auto-detects Qwen/Mistral/ChatML)
    validators/ - Config-driven content validators (XML, JSON, YAML, regex, code)
    rubric/     - Rubric loading and caching infrastructure
"""

# High-level imports for convenience
from .parsing import (
    parse_response,
    parse_tool_calls,
    ParsedResponse,
    ParsedToolCall,
    extract_tool_name_from_response,
    extract_arguments_from_response,
    has_tool_call,
)
from .validators import (
    StructureValidator,
    CrossScopeValidator,
    ContentValidatorProtocol,
    BaseContentValidator,
)
from .rubric import (
    RubricLoader,
    RubricCache,
    RubricRepository,
)
from .fitness import (
    FitnessEvaluator,
    FitnessResult,
    create_fitness_evaluator,
)

__all__ = [
    # Parsing
    "parse_response",
    "parse_tool_calls",
    "ParsedResponse",
    "ParsedToolCall",
    "extract_tool_name_from_response",
    "extract_arguments_from_response",
    "has_tool_call",
    # Validators
    "StructureValidator",
    "CrossScopeValidator",
    "ContentValidatorProtocol",
    "BaseContentValidator",
    # Rubric
    "RubricLoader",
    "RubricCache",
    "RubricRepository",
    # Fitness
    "FitnessEvaluator",
    "FitnessResult",
    "create_fitness_evaluator",
]
