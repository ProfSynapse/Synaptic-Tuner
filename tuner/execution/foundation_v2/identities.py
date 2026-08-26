"""Derivation-only sealed effect identities."""
from enum import Enum
from .canonical import canonical_bytes,domain_digest,parse_canonical_object
from .preparation import CanonicalPreparationV2
from .references import ProviderRunRefV1
class EffectKind(str,Enum):STAGE="stage";SUBMIT="submit";CANCEL="cancel"
_ISSUER=object()
class EffectIdentityV2:
    __slots__=("_raw","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("EffectIdentityV2 is final")
    def __init__(self,raw,*,_issuer):
        if _issuer is not _ISSUER:raise TypeError("effects are derivation-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("effect is immutable")
    def _doc(self):return parse_canonical_object(self._raw,name="effect")
    @property
    def kind(self):return EffectKind(self._doc()["kind"])
    @property
    def provider(self):
        from synaptic_tuner.api.v1.providers import ProviderRef
        return ProviderRef.from_dict(self._doc()["provider"])
    @property
    def scope(self):
        from .references import ExecutionScopeV1
        return ExecutionScopeV1(**self._doc()["scope"])
    @property
    def cancel_target(self):
        raw=self._doc()["cancel_target"];return None if raw is None else ProviderRunRefV1(**raw)
    @property
    def effect_digest(self):return domain_digest(f"synaptic-{self.kind.value}-effect/v2",canonical_bytes({k:v for k,v in self._doc().items() if k not in {"effect_digest","effect_id"}}))
    @property
    def effect_id(self):return f"{self.kind.value}-{self.effect_digest}"
    def __getattr__(self,n):
        if n in {"project_ref","run_id","plan_fingerprint","preparation_digest"}:return self._doc()[n]
        raise AttributeError(n)
    def to_dict(self):return {**self._doc(),"effect_digest":self.effect_digest,"effect_id":self.effect_id}
    def __eq__(self,o):return type(o) is EffectIdentityV2 and self._raw==o._raw
def derive_effect(preparation,kind,*,cancel_target=None):
    prep=CanonicalPreparationV2.parse(preparation.canonical_bytes)
    if type(kind) is not EffectKind:raise TypeError("exact EffectKind required")
    if kind is EffectKind.CANCEL:
        if type(cancel_target) is not ProviderRunRefV1:raise ValueError("cancel requires exact target")
    elif cancel_target is not None:raise ValueError("non-cancel cannot target run")
    basis={"provider":prep.provider.to_dict(),"scope":prep.scope.to_dict(),"project_ref":prep.project_ref,"run_id":prep.run_id,"plan_fingerprint":prep.plan_fingerprint,"preparation_digest":prep.preparation_digest,"kind":kind.value,"cancel_target":None if cancel_target is None else cancel_target.to_dict()}
    return EffectIdentityV2(canonical_bytes(basis),_issuer=_ISSUER)
