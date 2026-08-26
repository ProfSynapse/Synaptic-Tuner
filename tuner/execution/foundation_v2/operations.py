"""Derivation-only operation bindings."""
from .canonical import canonical_bytes,domain_digest,parse_canonical_object,safe_ref
from .preparation import CanonicalPreparationV2
from .identities import derive_effect
_ISSUER=object()
class OperationBindingV2:
    __slots__=("_raw","_prep","_effect","_sealed")
    def __init_subclass__(cls,**kw):raise TypeError("OperationBindingV2 is final")
    def __init__(self,raw,prep,effect,*,_issuer):
        if _issuer is not _ISSUER:raise TypeError("operations are derivation-minted")
        object.__setattr__(self,"_raw",bytes(raw));object.__setattr__(self,"_prep",prep.canonical_bytes);object.__setattr__(self,"_effect",effect);object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):raise AttributeError("operation is immutable")
    @property
    def preparation(self):return CanonicalPreparationV2.parse(self._prep)
    @property
    def effect(self):return derive_effect(self.preparation,self._effect.kind,cancel_target=self._effect.cancel_target)
    @property
    def invocation_nonce(self):return parse_canonical_object(self._raw,name="operation")["invocation_nonce"]
    @property
    def canonical_bytes(self):return bytes(self._raw)
    @property
    def digest(self):return domain_digest("synaptic-operation-binding/v2",self._raw)
    def to_dict(self):return parse_canonical_object(self._raw,name="operation")
def derive_operation(preparation,effect,invocation_nonce):
    prep=CanonicalPreparationV2.parse(preparation.canonical_bytes);safe_ref(invocation_nonce,"invocation_nonce")
    expected=derive_effect(prep,effect.kind,cancel_target=effect.cancel_target)
    if expected!=effect:raise ValueError("effect reconstruction mismatch")
    doc={"schema_version":"synaptic-operation-binding/v2","preparation_digest":prep.preparation_digest,"effect":expected.to_dict(),"invocation_nonce":invocation_nonce,"source_digest":prep.source_digest,"workload_digest":prep.workload_digest,"runtime_digest":prep.runtime_digest,"resource_digest":prep.resource_digest,"artifact_contract_digest":prep.artifact_contract_digest,"quote_digest":prep.quote_digest,"secret_requirements_digest":prep.secret_requirements_digest}
    return OperationBindingV2(canonical_bytes(doc),prep,expected,_issuer=_ISSUER)
