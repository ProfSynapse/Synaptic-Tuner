"""Intervention engine: forward-hook edits to a decoder layer's output."""

from MechInterp.intervention.hooks import (
    InterventionHook,
    GenerationInterventionController,
    additive_push,
    erase_and_write,
    resolve_final_positions,
    get_decoder_layer,
)
from MechInterp.intervention.equivalence import (
    max_abs_divergence,
    relative_tolerance,
    equivalence_ok,
)

__all__ = [
    "InterventionHook",
    "GenerationInterventionController",
    "additive_push",
    "erase_and_write",
    "resolve_final_positions",
    "get_decoder_layer",
    "max_abs_divergence",
    "relative_tolerance",
    "equivalence_ok",
]
