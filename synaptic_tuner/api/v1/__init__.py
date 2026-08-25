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
    "CLOUD_PROVIDERS",
    "CloudSourceContract",
    "CloudTrainingAPI",
    "CloudTrainingPlan",
    "CloudTrainingRequest",
    "CloudTrainingResult",
    "EventEnvelope",
    "PathRef",
    "PluginBinding",
    "PluginContext",
    "ProjectContext",
    "ResultEnvelope",
    "SecretRef",
    "SourceLock",
]

_TRAINING_EXPORTS = {
    "CLOUD_PROVIDERS",
    "CloudSourceContract",
    "CloudTrainingAPI",
    "CloudTrainingPlan",
    "CloudTrainingRequest",
    "CloudTrainingResult",
}


def __getattr__(name: str):
    """Load the cloud-training API only when a caller requests it."""

    if name in _TRAINING_EXPORTS:
        from . import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
