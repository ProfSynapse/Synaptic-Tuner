"""Supported Synaptic Tuner API v1.

Only names exported here are part of the narrow v1 compatibility surface.
Internal ``tuner`` and ``shared`` modules are not implicitly public APIs.
"""

from .capabilities import CapabilityDescriptor
from .context import PathRef, ProjectContext
from .events import EventEnvelope, ResultEnvelope
from .plugins import PluginBinding, PluginContext
from .secrets import SecretRef
from .sources import SourceLock

__all__ = [
    "CapabilityDescriptor",
    "EventEnvelope",
    "PathRef",
    "PluginBinding",
    "PluginContext",
    "ProjectContext",
    "ResultEnvelope",
    "SecretRef",
    "SourceLock",
]
