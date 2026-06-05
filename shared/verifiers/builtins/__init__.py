"""Built-in verifier implementations.

Importing this package runs each module's ``@register(...)`` decorator as a
side-effect, populating :data:`shared.verifiers.registry.VERIFIER_FACTORIES`
with the built-in verifier types (``substring``, ``structure``, ``llm_judge``).
"""

from __future__ import annotations

from . import llm_judge, structure, substring  # noqa: F401  (registration side-effect)

__all__ = ["substring", "structure", "llm_judge"]
