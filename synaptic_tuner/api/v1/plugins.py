"""Public plug-in protocols and trusted discovery contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tuner.project.context import ProjectContext
from tuner.project.plugins import (
    PLUGIN_ENTRY_POINT_GROUP,
    PLUGIN_ENTRY_POINT_GROUP_PREFIX,
    TRUST_NOTICE,
    PluginBinding,
    PluginConflictError,
    PluginContractError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginReferenceError,
    discover_plugins,
    plugin_entry_point_group,
    resolve_plugin,
)


@dataclass(frozen=True)
class PluginContext:
    """Context passed to a trusted plug-in factory at execution time."""

    project: ProjectContext
    name: str
    kind: str
    api: str
    config: Mapping[str, Any] = field(default_factory=dict)


class Renderer(Protocol):
    def __call__(self, row: Mapping[str, Any]) -> str: ...


class Grader(Protocol):
    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ContentEndResolver(Protocol):
    def __call__(
        self, full_ids: Sequence[int], prompt_len: int, tokenizer: Any
    ) -> int: ...


class PluginFactory(Protocol):
    def __call__(self, context: PluginContext) -> object: ...


__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "PLUGIN_ENTRY_POINT_GROUP_PREFIX",
    "TRUST_NOTICE",
    "ContentEndResolver",
    "Grader",
    "PluginBinding",
    "PluginConflictError",
    "PluginContext",
    "PluginContractError",
    "PluginError",
    "PluginFactory",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginReferenceError",
    "Renderer",
    "discover_plugins",
    "plugin_entry_point_group",
    "resolve_plugin",
]
