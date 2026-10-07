"""Internal single-attempt provider mutation seam."""
from __future__ import annotations
from typing import Protocol
from .contracts import EffectObservation
class _ProviderMutationDriver(Protocol):
    def execute_once(self,canonical_command:bytes)->EffectObservation:...
class _ProviderEffectExecutor:
    def __init__(self,driver:_ProviderMutationDriver):self._driver=driver
    def execute_once(self,canonical_command:bytes)->EffectObservation:return self._driver.execute_once(canonical_command)
