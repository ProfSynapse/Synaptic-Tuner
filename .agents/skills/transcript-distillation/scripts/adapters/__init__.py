"""Adapter registry. Add a new transcript format here.

To support a new format: implement the Adapter interface in a new module
(see base.py and writing-adapters.md), import it, and add an instance below.
"""
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter

_ADAPTERS = [ClaudeCodeAdapter(), CodexAdapter()]
REGISTRY = {a.name: a for a in _ADAPTERS}


def get_adapter(fmt: str):
    if fmt not in REGISTRY:
        raise KeyError(
            f"unknown transcript format '{fmt}'. Registered: {sorted(REGISTRY)}. "
            f"Add an adapter in adapters/ and register it in adapters/__init__.py."
        )
    return REGISTRY[fmt]
