"""
Schema validator - DEPRECATED module, re-exports from validators/ package.

This file is kept for backward compatibility. Import directly from:
    from SynthChat.services.validators import SchemaValidator
"""

# Re-export from new location for backward compatibility
from .validators import (
    SchemaValidator,
    StructureValidator,
    CrossScopeValidator,
    BaseContentValidator,
    ContentValidatorProtocol,
)

__all__ = [
    "SchemaValidator",
    "StructureValidator",
    "CrossScopeValidator",
    "BaseContentValidator",
    "ContentValidatorProtocol",
]
