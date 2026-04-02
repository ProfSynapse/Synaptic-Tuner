"""
LoRA Surgery package — Strategy pattern decomposition.

Location: shared/evolutionary/surgery/__init__.py
Purpose: Public API for the surgery package. Re-exports key types and
         provides the operation registry interface.
Used by: shared/evolutionary/lora_surgery.py (backward-compat shim),
         tuner/handlers/surgery_handler.py, tests.
"""

from .config import OperationResult, SurgeryConfig, SurgeryResult
from .registry import get_operation, list_operations, register_operation
from .surgeon import LoRASurgeon
from .base import SurgeryOperation

# Import operations to trigger @register_operation decorators
from . import operations  # noqa: F401

__all__ = [
    # Core types
    "LoRASurgeon",
    "SurgeryConfig",
    "SurgeryResult",
    "OperationResult",
    "SurgeryOperation",
    # Registry API
    "get_operation",
    "list_operations",
    "register_operation",
]
