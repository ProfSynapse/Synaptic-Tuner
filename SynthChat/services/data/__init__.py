"""Data access layer for rubrics.

Provides:
- RubricLoader: Loads YAML files (I/O only)
- RubricCache: Caches rubrics in memory (caching only)
- RubricRepository: Facade coordinating loader + cache (query interface)
"""

from .rubric_loader import RubricLoader
from .rubric_cache import RubricCache
from .rubric_repository import RubricRepository, RubricMetadata

__all__ = [
    "RubricLoader",
    "RubricCache",
    "RubricRepository",
    "RubricMetadata",
]
