"""Time-, epoch-, and revocation-bound exact reconstructed command authority."""
import hashlib,hmac
from dataclasses import dataclass
from .canonical import canonical_bytes,digest_text,domain_digest,exact_integer,safe_ref
from .commands import parse_exact_command
@dataclass(frozen=True,slots=True)
class GrantContentV2:
    grant_ref:str;command_digest:str;operation_digest:str;effect_id:str;preparation_digest:str;executor_digest:str;provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;effect_kind:str;payload_schema:str;policy_digest:str;requirement_digest:str;not_before_epoch:int;expires_at_epoch:int;authority_epoch:int;revocation_generation:int
    def __post_init__(self):
        for n in ("grant_ref","effect_id","provider_id","profile_ref","account_ref","namespace_ref","effect_kind","payload_schema"):safe_ref(getattr(self,n),n)
        for n in ("command_digest","operation_digest","preparation_digest","executor_digest","policy_digest","requirement_digest"):digest_text(getattr(self,n),n)
        exact_integer(self.not_before_epoch,"not_before_epoch");exact_integer(self.expires_at_epoch,"expires_at_epoch",minimum=1);exact_integer(self.authority_epoch,"authority_epoch",minimum=1);exact_integer(self.revocation_generation,"revocation_generation")
        if self.expires_at_epoch<=self.not_before_epoch:raise ValueError("grant expiry must follow activation")
        matrix={"stage":"stage-payload/v2","submit":"submit-payload/v2","cancel":"cancel-payload/v2"}
        if matrix.get(self.effect_kind)!=self.payload_schema:raise ValueError("grant effect and payload mismatch")
    def to_dict(self):return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @property
    def digest(self):return domain_digest("synaptic-grant-content/v3",canonical_bytes(self.to_dict()))
@dataclass(frozen=True,slots=True)
class AuthenticatedGrantV2:
    content:GrantContentV2;authority_ref:str;tag:str
    def __post_init__(self):
        if type(self.content) is not GrantContentV2:raise TypeError("exact grant content required")
        safe_ref(self.authority_ref,"authority_ref");digest_text(self.tag,"tag")
@dataclass(frozen=True,slots=True)
class ReconciliationGrantContentV1:
    grant_ref:str;command_digest:str;effect_id:str;preparation_digest:str;adapter_digest:str;provider_id:str;profile_ref:str;account_ref:str;namespace_ref:str;owner_ref:str;generation:int;ownership_epoch:int;policy_digest:str;requirement_digest:str;not_before_epoch:int;expires_at_epoch:int;authority_epoch:int;revocation_generation:int
    def __post_init__(self):
        for n in ("grant_ref","effect_id","provider_id","profile_ref","account_ref","namespace_ref","owner_ref"):safe_ref(getattr(self,n),n)
        for n in ("command_digest","preparation_digest","adapter_digest","policy_digest","requirement_digest"):digest_text(getattr(self,n),n)
        exact_integer(self.generation,"generation",minimum=1);exact_integer(self.ownership_epoch,"ownership_epoch",minimum=1);exact_integer(self.not_before_epoch,"not_before_epoch");exact_integer(self.expires_at_epoch,"expires_at_epoch",minimum=1);exact_integer(self.authority_epoch,"authority_epoch",minimum=1);exact_integer(self.revocation_generation,"revocation_generation")
        if self.expires_at_epoch<=self.not_before_epoch:raise ValueError("grant expiry must follow activation")
    def to_dict(self):return {n:getattr(self,n) for n in self.__dataclass_fields__}
    @property
    def digest(self):return domain_digest("synaptic-reconciliation-grant/v2",canonical_bytes(self.to_dict()))
@dataclass(frozen=True,slots=True)
class AuthenticatedReconciliationGrantV1:
    content:ReconciliationGrantContentV1;authority_ref:str;tag:str
    def __post_init__(self):
        if type(self.content) is not ReconciliationGrantContentV1:raise TypeError("exact reconciliation content required")
        safe_ref(self.authority_ref,"authority_ref");digest_text(self.tag,"tag")
class GrantAuthorityV2:
    __slots__=("authority_ref","_key","epoch","revocation_generation","_revoked")
    def __init__(self,authority_ref,key,*,epoch=1,revocation_generation=0):
        self.authority_ref=safe_ref(authority_ref,"authority_ref");self._key=key;self.epoch=epoch;self.revocation_generation=revocation_generation;self._revoked=set()
        if not isinstance(key,bytes) or len(key)<32:raise ValueError("authority key too short")
    def __repr__(self):return f"GrantAuthorityV2(authority_ref={self.authority_ref!r}, key=<redacted>)"
    def _tag(self,domain,digest):return hmac.new(self._key,domain+bytes.fromhex(digest),hashlib.sha256).hexdigest()
    def revoke(self,grant_ref):self._revoked.add(safe_ref(grant_ref,"grant_ref"))
    def issue(self,command_bytes,*,grant_ref,policy_digest,requirement_digest,not_before_epoch,expires_at_epoch):
        c=parse_exact_command(bytes(command_bytes));p=c.preparation;e=c.operation.effect
        content=GrantContentV2(grant_ref,c.digest,c.operation.digest,e.effect_id,p.preparation_digest,c.executor.digest,p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,e.kind.value,c.payload.payload_kind,policy_digest,requirement_digest,not_before_epoch,expires_at_epoch,self.epoch,self.revocation_generation)
        return AuthenticatedGrantV2(content,self.authority_ref,self._tag(b"grant-v3\0",content.digest))
    def verify(self,grant,command_bytes,*,now_epoch):
        try:
            c=parse_exact_command(bytes(command_bytes));p=c.preparation;e=c.operation.effect;x=grant.content
            rebuilt=GrantContentV2(x.grant_ref,c.digest,c.operation.digest,e.effect_id,p.preparation_digest,c.executor.digest,p.provider.provider_id,p.provider.profile_ref,p.scope.account_ref,p.scope.namespace_ref,e.kind.value,c.payload.payload_kind,x.policy_digest,x.requirement_digest,x.not_before_epoch,x.expires_at_epoch,x.authority_epoch,x.revocation_generation)
            return x==rebuilt and grant.authority_ref==self.authority_ref and x.authority_epoch==self.epoch and x.revocation_generation==self.revocation_generation and x.grant_ref not in self._revoked and x.not_before_epoch<=now_epoch<x.expires_at_epoch and hmac.compare_digest(grant.tag,self._tag(b"grant-v3\0",x.digest))
        except Exception:return False
    def issue_reconciliation(self,content):
        if type(content) is not ReconciliationGrantContentV1:raise TypeError("exact reconciliation content required")
        return AuthenticatedReconciliationGrantV1(content,self.authority_ref,self._tag(b"reconcile-v2\0",content.digest))
    def verify_reconciliation(self,grant,*,now_epoch):
        try:
            x=grant.content
            return grant.authority_ref==self.authority_ref and x.authority_epoch==self.epoch and x.revocation_generation==self.revocation_generation and x.grant_ref not in self._revoked and x.not_before_epoch<=now_epoch<x.expires_at_epoch and hmac.compare_digest(grant.tag,self._tag(b"reconcile-v2\0",x.digest))
        except Exception:return False
