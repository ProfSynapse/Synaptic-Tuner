"""Shared judge module -- generic LLM-as-judge components.

Location: shared/judge/__init__.py
Summary: Public API for the shared judge module. Provides typed dataclasses,
         rubric loading, schema building, judge execution, and interaction
         logging. Generic and reusable -- no Evaluator-specific or
         SynthChat-specific logic. Consumed by Evaluator (via JudgeValidator)
         and SynthChat (directly or via a future adapter).
"""

from .models import JudgeConfig, JudgeResult, JudgeScore, RubricDef
from .rubric_loader import RubricLoader
from .schema_builder import SchemaBuilder
from .judge_service import JudgeService
from .interaction_logger import InteractionLogger

__all__ = [
    "JudgeConfig",
    "JudgeResult",
    "JudgeScore",
    "RubricDef",
    "RubricLoader",
    "SchemaBuilder",
    "JudgeService",
    "InteractionLogger",
]
