"""Generic batched-inference engine for the Synaptic Tuner.

This package provides two prompt-in / artifact-out batch verbs with
crash-safe incremental persistence and resume:

  - ``batch-generate``: prompts in, completions out.
  - ``batch-capture``: sequences in, per-layer hidden states at named token
    positions out.

Nothing in this package is project-specific: it takes prompts / sequences and
emits completions / hidden states. Grading, pooling, outcome taxonomies, and
row schemas belong to the consuming project, not here.

The engines live behind a small interface (``tuner.batch.engines``) so new
backends (currently ``hf-batched`` and ``vllm``) can be added without touching
the persistence, checkpoint, or CLI layers.
"""

from tuner.batch.persistence import (
    RunCheckpoint,
    compute_config_hash,
    JsonlAppender,
)

__all__ = [
    "RunCheckpoint",
    "compute_config_hash",
    "JsonlAppender",
]
