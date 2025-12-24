"""
Data layer - re-exports rubric management from shared.validation.

All rubric implementations have been moved to shared/validation/rubric/
for use across SynthChat, Evaluator, and Trainer modules.
"""

from shared.validation.rubric import (
    RubricLoader,
    RubricCache,
    RubricRepository,
)

# Keep local RubricMetadata if it exists
try:
    from .rubric_repository import RubricMetadata
    __all__ = [
        "RubricLoader",
        "RubricCache",
        "RubricRepository",
        "RubricMetadata",
    ]
except ImportError:
    __all__ = [
        "RubricLoader",
        "RubricCache",
        "RubricRepository",
    ]
