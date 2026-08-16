"""Import-light capability discovery and machine-output contracts."""

from .builtins import builtin_descriptors
from .events import emit_diagnostic, write_event, write_result
from .registry import CapabilityRegistry, builtin_registry
from .schema import validate_descriptor, validate_event, validate_result

__all__ = [
    "CapabilityRegistry",
    "builtin_descriptors",
    "builtin_registry",
    "emit_diagnostic",
    "validate_descriptor",
    "validate_event",
    "validate_result",
    "write_event",
    "write_result",
]
