"""Private broker driver composition; never exported."""
from __future__ import annotations
from .control import StageControlPlane
from ...contracts import EffectObservation
class _ModalEffectDriver:
    def __init__(self,resolve_effect,control:StageControlPlane,private_mutator):self.resolve_effect=resolve_effect;self.control=control;self.mutator=private_mutator
    def execute_once(self,canonical_command:bytes)->EffectObservation:
        effect,receipt=self.resolve_effect(canonical_command);self.control.validate(receipt)
        try:return self.mutator.execute_once(canonical_command)
        except Exception:raise RuntimeError("modal_effect_failed") from None
