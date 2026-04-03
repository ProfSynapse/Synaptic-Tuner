"""
Backward compatibility shim for LoRA Weight Surgery.

.. deprecated::
    This module is a backward-compatibility shim. New code should import
    directly from ``shared.evolutionary.surgery`` instead::

        from shared.evolutionary.surgery import LoRASurgeon, SurgeryConfig
        from shared.evolutionary.surgery.utils import copy_adapter, load_all_weights

    The underscore-prefixed helper aliases (``_load_all_weights``,
    ``_copy_adapter``, etc.) are deprecated. Use the public names from
    ``shared.evolutionary.surgery.utils`` directly.

Location: shared/evolutionary/lora_surgery.py
Purpose: Re-export all public symbols from the new surgery/ package so
         existing consumers (surgery_handler, tests) continue to work
         without import changes.
Used by: tuner/handlers/surgery_handler.py, tests/test_lora_surgery.py,
         tests/test_karpathy_integration.py
"""

from shared.evolutionary.surgery import (
    LoRASurgeon,
    OperationResult,
    SurgeryConfig,
    SurgeryResult,
)
from shared.evolutionary.surgery.utils import (
    _check_dependencies,
    _copy_adapter,
    _find_lora_pairs,
    _find_safetensor_files,
    _get_layer_indices,
    _get_module_types,
    _is_attention_key,
    _is_mlp_key,
    _load_adapter_config,
    _load_all_weights,
    _save_adapter_config,
    _save_all_weights,
    _softmax,
)

__all__ = [
    "LoRASurgeon",
    "SurgeryConfig",
    "OperationResult",
    "SurgeryResult",
    # Helpers re-exported for backward compat
    "_check_dependencies",
    "_copy_adapter",
    "_find_lora_pairs",
    "_find_safetensor_files",
    "_get_layer_indices",
    "_get_module_types",
    "_is_attention_key",
    "_is_mlp_key",
    "_load_adapter_config",
    "_load_all_weights",
    "_save_adapter_config",
    "_save_all_weights",
    "_softmax",
]
