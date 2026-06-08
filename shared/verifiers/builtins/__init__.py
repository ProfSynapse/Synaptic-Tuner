"""Built-in verifier implementations.

Importing this package runs each module's ``@register(...)`` decorator as a
side-effect, populating :data:`shared.verifiers.registry.VERIFIER_FACTORIES`
with the built-in verifier types (``substring``, ``structure``, ``llm_judge``,
``args_match``, ``assertions``, ``tool_sequence``).
"""

from __future__ import annotations

from . import (  # noqa: F401  (registration side-effect)
    args_match,
    assertion_verifier,
    llm_judge,
    structure,
    substring,
    tool_sequence,
)

__all__ = [
    "substring",
    "structure",
    "llm_judge",
    "args_match",
    "assertion_verifier",
    "tool_sequence",
]
