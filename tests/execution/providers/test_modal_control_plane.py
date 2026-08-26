from __future__ import annotations
import hashlib,hmac,json
from dataclasses import replace
import pytest
from tuner.execution._effect_executor import _ProviderEffectExecutor
from tuner.execution.contracts import EffectDisposition,EffectIdentity,EffectKind,EffectObservation,ExecutionScope
from tuner.execution.operation import ModalStageTargetV1,OperationBindingV1
from tuner.execution.providers.modal import *
from tuner.execution.providers.modal.contracts import canonical_json,sha

D="a"*64
ROOT="operations/e";OUTPUT=ROOT+"/output";CONTROL=ROOT+"/control";EVIDENCE=ROOT+"/evidence";LOGS=ROOT+"/logs"
class Auth:
    key=b"k"
    def sign(self,p):return hmac.new(self.key,p,hashlib.sha256).digest()
    def verify(self,*args):
        p,t=(args[-3],args[-2]) if len(args)==4 else args
        return hmac.compare_digest(self.sign(p),t)
class Fake:
    def __init__(self,binding):self.binding=binding;self.files={};self.listing=();self.listings={};self.proof=CapabilityProofV1(True,True,True,True,True,True,True)
    def bound_scope(self):return (self.binding.account_ref,self.binding.workspace_ref,self.binding.environment_ref,self.binding.client_ref)
    def capability_proof(self,b):return self.proof
    def read_complete(self,v,p,*,max_bytes):
        data=self.files[(v,p)]
        if len(data)>max_bytes:raise ValueError
        return data
    def list_prefix(self,v,p,*,max_entries):return self.listings.get((v,p),self.listing)
class Store:
    def __init__(self,e):self.e=e
    def load_modal_expectation(self,effect_id):return self.e
def binding(version="1.5.4"):return ModalClientBinding("acct","work","env","client",version)
def operation():
    e=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));return OperationBindingV1("p","r",e,"g",D,D,D,D,D,D,D,D,D,D,D,"nonce",ModalStageTargetV1("slot","cv","av","operations/e/output",1,"key"))
def expectation():
    return StageExpectationV1(operation(),binding(),D,D,6)
def stage_setup():
    ex=expectation();f=Fake(ex.binding);bundle=b"bundle";ex=replace(ex,bundle_digest=sha(bundle));claim=canonical_json({"schema":"synaptic.modal-stage-claim/v1","effect_provider":"modal","effect_account_ref":"acct","effect_namespace_ref":"env","effect_id":"e","effect_kind":"submit","operation_key":"op","operation_binding_digest":ex.operation_binding_digest,"control_volume_id":"cv","artifact_volume_id":"av","bundle_digest":ex.bundle_digest,"bundle_size":6,"plan_digest":D,"invocation_nonce":"nonce","output_prefix":OUTPUT});ex=replace(ex,claim_digest=sha(claim));f.files[("cv",CONTROL+"/stage-claim.v1.json")]=claim;f.files[("cv",CONTROL+"/stage-claim.v1.mac")]=Auth().sign(claim);f.files[("av",ROOT+"/input/bundle.bin")]=bundle;r=StageReceiptV1("e",ex.operation_binding_digest,"cv","av",sha(claim),sha(bundle));return ex,f,r
def test_readiness_exact_version_scope_and_all_capabilities(monkeypatch):
    b=binding();f=Fake(b);assert readiness(b,f) is Readiness.READY
    assert readiness(binding("1.5.3"),f) is Readiness.NOT_READY
    f.proof=replace(f.proof,image_identity=False);assert readiness(b,f) is Readiness.NOT_READY
    monkeypatch.setenv("MODAL_PROFILE","ambient");assert readiness(b,object()) is Readiness.NOT_READY
def test_readiness_closes_scope_and_capability_errors_without_cause():
    b=binding();f=Fake(b)
    f.bound_scope=lambda:(_ for _ in ()).throw(RuntimeError("Bearer raw-secret"))
    report=readiness_report(b,f);assert report==ReadinessReport(Readiness.NOT_READY,"scope_unavailable")
    f=Fake(b);f.capability_proof=lambda unused:(_ for _ in ()).throw(RuntimeError("raw-secret"))
    assert readiness_report(b,f).reason_code=="capability_unavailable"
def test_control_and_artifact_volumes_and_reserved_output_are_disjoint():
    with pytest.raises(ValueError):StageReceiptV1("e",D,"same","same",D,D)
    with pytest.raises(ValueError):CompletionExpectationV1(binding(),"cv","av","logs","key","fc-1","e",D,D,D,"nonce",1)
    with pytest.raises(ValueError):ArtifactMemberV1(ArtifactRole.TOKENIZER,"evidence/tokenizer",1,D,"id")
def test_stage_values_confer_no_authority_and_all_substitutions_fail():
    ex,f,r=stage_setup();plane=StageControlPlane(Store(ex),Auth(),f);assert plane.validate(r)==ex
    for bad in [replace(r,control_volume_id="other"),replace(r,bundle_digest="b"*64),replace(r,claim_digest="b"*64)]:
        with pytest.raises(ValueError):plane.validate(bad)
    f.files[("cv",CONTROL+"/stage-claim.v1.mac")]=b"x"*32
    with pytest.raises(ValueError):plane.validate(r)
@pytest.mark.parametrize("field,bad",[("effect_id","other"),("operation_binding_digest","b"*64),("control_volume_id","other"),("artifact_volume_id","other"),("claim_digest","b"*64),("bundle_digest","b"*64)])
def test_every_stage_receipt_substitution_fails_before_any_facade_or_auth_call(field,bad):
    ex,_,receipt=stage_setup();calls=[]
    class CountingFacade:
        def bound_scope(self):calls.append("scope");return ("acct","work","env","client")
        def capability_proof(self,binding):calls.append("capability");return CapabilityProofV1(True,True,True,True,True,True,True)
        def read_complete(self,*args,**kwargs):calls.append("read");raise AssertionError("must not read")
    class CountingAuth:
        def verify(self,*args):calls.append("auth");raise AssertionError("must not authenticate")
    with pytest.raises(ValueError,match="receipt mismatch"):StageControlPlane(Store(ex),CountingAuth(),CountingFacade()).validate(replace(receipt,**{field:bad}))
    assert calls==[]
def test_valid_stage_receipt_reaches_readiness_authentication_and_reads():
    ex,f,receipt=stage_setup();calls=[]
    original_scope=f.bound_scope;original_capability=f.capability_proof;original_read=f.read_complete
    f.bound_scope=lambda:(calls.append("scope"),original_scope())[1]
    f.capability_proof=lambda value:(calls.append("capability"),original_capability(value))[1]
    f.read_complete=lambda *args,**kwargs:(calls.append("read"),original_read(*args,**kwargs))[1]
    class CountingAuth(Auth):
        def verify(self,*args):calls.append("auth");return super().verify(*args)
    assert StageControlPlane(Store(ex),CountingAuth(),f).validate(receipt)==ex
    assert calls==["scope","capability","read","read","auth","read"]
def test_stage_expectation_rejects_non_modal_scope_and_non_submit_effects():
    ex=expectation()
    for effect in (EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("hf","acct","env")),EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","other","env")),EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","other")),EffectIdentity("e","op",EffectKind.CANCEL,ExecutionScope("modal","acct","env"))):
        with pytest.raises(ValueError):replace(ex,operation=replace(ex.operation,effect=effect))
    with pytest.raises(ValueError):EffectIdentity("e","not canonical key",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"))
@pytest.mark.parametrize("field,bad",[("effect_provider","hf"),("effect_account_ref","other"),("effect_namespace_ref","other"),("effect_id","other"),("effect_kind","cancel"),("operation_key","other")])
def test_authenticated_stage_claim_binds_complete_effect_identity(field,bad):
    ex,f,receipt=stage_setup();claim=json.loads(f.files[("cv",CONTROL+"/stage-claim.v1.json")]);claim[field]=bad;raw=canonical_json(claim);ex=replace(ex,claim_digest=sha(raw));receipt=replace(receipt,claim_digest=sha(raw));f.files[("cv",CONTROL+"/stage-claim.v1.json")]=raw;f.files[("cv",CONTROL+"/stage-claim.v1.mac")]=Auth().sign(raw)
    with pytest.raises(ValueError):StageControlPlane(Store(ex),Auth(),f).validate(receipt)
def member(role,path,data):return ArtifactMemberV1(role,path,len(data),sha(data),"id-"+role.value)
def provider_identity():return {"account_ref":"acct","workspace_ref":"work","environment_ref":"env","client_ref":"client","sdk_version":"1.5.4","control_volume_id":"cv","artifact_volume_id":"av","job_ref":"fc-1","effect_id":"e","command_digest":D,"plan_digest":D,"deployment_attestation_digest":D,"invocation_nonce":"nonce","generation":1}
def manifest_data(members):return canonical_json({"schema":"synaptic.modal-completion/v1","members":[{"role":m.role.value,"path":m.path,"size":m.size,"sha256":m.sha256,"provider_entry_id":m.provider_entry_id} for m in members],**provider_identity(),"terminal_evidence_digest":D,"log_chain_digest":D})
def completion_expectation(*,terminal_digest=D,log_digest=D):return CompletionExpectationV1(binding(),"cv","av",OUTPUT,"key","fc-1","e",D,D,D,"nonce",1)
def manifest_data_bound(members,terminal_digest,log_digest):
    value=json.loads(manifest_data(members));value["terminal_evidence_digest"]=terminal_digest;value["log_chain_digest"]=log_digest;return canonical_json(value)
def terminal_data(*,log_digest=D,artifact_digest=D,status="completed",identity=None):return canonical_json({"schema":"synaptic.modal-terminal/v1","status_code":status,**(provider_identity() if identity is None else identity),"artifact_set_digest":artifact_digest,"log_chain_digest":log_digest})
def log_chunk(*,sequence=0,previous="0"*64,effect="e",generation=1):
    records=[{"code":"completed","message":"done"}];raw=canonical_json({"schema":"synaptic.modal-log-chunk/v1","generation":generation,"sequence":sequence,"previous_digest":previous,"payload_digest":sha(canonical_json(records)),"job_ref":"fc-1","effect_id":effect,"plan_digest":D,"invocation_nonce":"nonce","records":records});return raw,StructuredLogChunkV1.parse(raw)
def log_metadata(raw,chain,identity=None):return canonical_json({"schema":"synaptic.modal-log-metadata/v1",**(provider_identity() if identity is None else identity),"chain_digest":chain,"chunks":[{"path":LOGS+"/chunks/000.json","size":len(raw),"sha256":sha(raw),"provider_entry_id":"log-entry-0"}]})
def test_manifest_exact_runtime_roles_and_artifact_relist_read_hashes():
    roles=list(ArtifactRole);items=[member(r,OUTPUT+"/"+r.value,(r.value).encode()) for r in roles];m=CompletionManifestV1.parse(manifest_data(items));_,f,_=stage_setup();ex=completion_expectation();f.listing=tuple((x.path,x.size,x.provider_entry_id) for x in items)
    for x in items:f.files[("av",x.path)]=x.role.value.encode()
    assert len(verify_artifacts(m,ex,f))==5
    with pytest.raises(ValueError):CompletionManifestV1.parse(manifest_data(items[:-1]))
    with pytest.raises(ValueError):CompletionManifestV1.parse(manifest_data(items[:-1]+[items[0]]))
    f.files[("av",items[0].path)]=b"changed"
    with pytest.raises(ValueError):verify_artifacts(m,ex,f)
def test_manifest_and_actual_listing_reject_duplicate_paths_and_entry_ids_before_mapping():
    items=[member(r,OUTPUT+"/"+r.value,r.value.encode()) for r in ArtifactRole]
    raw=json.loads(manifest_data(items));raw["members"][1]["path"]=raw["members"][0]["path"]
    with pytest.raises(ValueError):CompletionManifestV1.parse(canonical_json(raw))
    raw=json.loads(manifest_data(items));raw["members"][1]["provider_entry_id"]=raw["members"][0]["provider_entry_id"]
    with pytest.raises(ValueError):CompletionManifestV1.parse(canonical_json(raw))
    manifest=CompletionManifestV1.parse(manifest_data(items));_,f,_=stage_setup();ex=completion_expectation()
    for item in items:f.files[("av",item.path)]=item.role.value.encode()
    valid=[(x.path,x.size,x.provider_entry_id) for x in items];valid[1]=(valid[1][0],valid[1][1],valid[0][2]);f.listing=tuple(valid)
    with pytest.raises(ValueError):verify_artifacts(manifest,ex,f)
@pytest.mark.parametrize("field,bad",[("account_ref","other"),("workspace_ref","other"),("environment_ref","other"),("client_ref","other"),("sdk_version","1.5.3"),("control_volume_id","other"),("artifact_volume_id","other"),("job_ref","fc-other"),("effect_id","other"),("command_digest","b"*64),("plan_digest","b"*64),("deployment_attestation_digest","b"*64),("invocation_nonce","other"),("generation",2)])
def test_completion_manifest_rejects_every_cross_plane_identity_substitution(field,bad):
    items=[member(r,OUTPUT+"/"+r.value,r.value.encode()) for r in ArtifactRole];value=json.loads(manifest_data(items));value[field]=bad;manifest=CompletionManifestV1.parse(canonical_json(value));_,f,_=stage_setup()
    with pytest.raises(ValueError):verify_artifacts(manifest,completion_expectation(),f)
def test_host_authenticated_completion_manifest_control():
    items=[member(r,OUTPUT+"/"+r.value,r.value.encode()) for r in ArtifactRole];_,f,_=stage_setup()
    for x in items:f.files[("av",x.path)]=x.role.value.encode()
    log_raw,chunk=log_chunk();chain=chunk.chunk_digest
    artifact_digest=CompletionManifestV1.parse(manifest_data_bound(items,D,chain)).artifact_set_digest
    terminal_raw=terminal_data(log_digest=chain,artifact_digest=artifact_digest);terminal_digest=sha(terminal_raw)
    ex=completion_expectation(terminal_digest=terminal_digest,log_digest=chain);data=manifest_data_bound(items,terminal_digest,chain)
    terminal_ex=TerminalExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",1)
    metadata=log_metadata(log_raw,chain);log_ex=LogExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",1)
    f.files.update({("cv",EVIDENCE+"/completion-manifest.v1.json"):data,("cv",EVIDENCE+"/completion-manifest.v1.mac"):Auth().sign(data),("cv",EVIDENCE+"/terminal-evidence.v1.json"):terminal_raw,("cv",EVIDENCE+"/terminal-evidence.v1.mac"):Auth().sign(terminal_raw),("cv",LOGS+"/log-metadata.v1.json"):metadata,("cv",LOGS+"/log-metadata.v1.mac"):Auth().sign(metadata),("cv",LOGS+"/chunks/000.json"):log_raw})
    f.listings[("av",OUTPUT+"/")]=tuple((x.path,x.size,x.provider_entry_id) for x in items);f.listings[("cv",LOGS+"/chunks/")]=((LOGS+"/chunks/000.json",len(log_raw),"log-entry-0"),)
    class CStore:
        def load_completion_expectation(self,effect_id):return ex
        def load_terminal_expectation(self,effect_id):return terminal_ex
        def load_log_expectation(self,effect_id):return log_ex
    terminal=TerminalControlPlane(CStore(),Auth(),f);logs=LogControlPlane(CStore(),Auth(),f)
    assert CompletionControlPlane(CStore(),Auth(),f,terminal,logs).validate("e").effect_id=="e"
    f.files[("cv",EVIDENCE+"/completion-manifest.v1.mac")]=b"x"*32
    with pytest.raises(ValueError):CompletionControlPlane(CStore(),Auth(),f,terminal,logs).validate("e")
def test_terminal_evidence_is_separate_strict_control_record():
    data=terminal_data();assert TerminalEvidenceV1.parse(data).status_code=="completed"
    with pytest.raises(ValueError):TerminalEvidenceV1.parse(data[:-1]+b" ")
def test_redaction_nested_multiline_and_credentials():
    out=redact({"client secret":"two words stay hidden","nested":{"api_key":"sk-abcdefghijk","note":"Bearer secret\npassword = multi word value\nnext"},"b":["eyJabc.def.ghi","Basic Zm9vOmJhcg==","https://u:p@example.com","token=xyz"]})
    for secret in ("two words stay hidden","sk-abcdefghijk","Bearer secret","multi word value","eyJabc","Zm9v","u:p","xyz"):assert secret not in out
    decoded=json.loads(out);assert isinstance(decoded["nested"],dict) and decoded["client secret"]=="[REDACTED]"
def test_cursor_constant_size_tamper_bomb_expiry_generation():
    a=Auth();now=[10];s=CursorService(a,clock=lambda:now[0]);one=s.issue(generation=1,chunk=2,record=3,expires_at=20,context_digest=D);two=s.issue(generation=1,chunk=9,record=8,expires_at=20,context_digest=D);assert len(one)==len(two)==s.ENCODED_SIZE and s.parse(one,generation=1,context_digest=D)==(2,3)
    with pytest.raises(ValueError):s.parse(one+"A",generation=1,context_digest=D)
    tampered=one[:-2]+("A" if one[-2]!="A" else "B")+one[-1]
    with pytest.raises(ValueError):s.parse(tampered,generation=1,context_digest=D)
    with pytest.raises(ValueError):s.parse(one,generation=2,context_digest=D)
    now[0]=20
    with pytest.raises(ValueError):s.parse(one,generation=1,context_digest=D)
    for kwargs in ({"generation":True,"chunk":2,"record":3,"expires_at":20,"context_digest":D},{"generation":1,"chunk":True,"record":3,"expires_at":20,"context_digest":D},{"generation":1,"chunk":2,"record":True,"expires_at":20,"context_digest":D},{"generation":1,"chunk":2,"record":3,"expires_at":True,"context_digest":D}):
        with pytest.raises(ValueError):s.issue(**kwargs)
def test_structured_log_chunk_closed_codes_and_digest():
    records=[{"code":"progress","message":"step one"}];data=canonical_json({"schema":"synaptic.modal-log-chunk/v1","generation":1,"sequence":0,"previous_digest":"0"*64,"payload_digest":sha(canonical_json(records)),"job_ref":"fc-1","effect_id":"e","plan_digest":D,"invocation_nonce":"nonce","records":records});chunk=StructuredLogChunkV1.parse(data);assert chunk.records[0][0] is LogCode.PROGRESS
    bad=data.replace(b"progress",b"unknownx")
    with pytest.raises(ValueError):StructuredLogChunkV1.parse(bad)
def test_structured_log_chain_binds_generation_identity_sequence_and_previous_digest():
    raw,first=log_chunk();raw2,second=log_chunk(sequence=1,previous=first.chunk_digest)
    assert validate_chain((first,second))==second.chunk_digest
    for bad in (log_chunk(sequence=2,previous=first.chunk_digest)[1],log_chunk(sequence=1,previous="b"*64)[1],log_chunk(sequence=1,previous=first.chunk_digest,effect="other")[1],log_chunk(sequence=1,previous=first.chunk_digest,generation=2)[1]):
        with pytest.raises(ValueError):validate_chain((first,bad))
def test_terminal_control_reauthenticates_exact_effect_and_all_bound_fields():
    raw=terminal_data();ex=TerminalExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",1);_,f,_=stage_setup();f.files[("cv",EVIDENCE+"/terminal-evidence.v1.json")]=raw;f.files[("cv",EVIDENCE+"/terminal-evidence.v1.mac")]=Auth().sign(raw)
    class TStore:
        def load_terminal_expectation(self,effect_id):return ex
    plane=TerminalControlPlane(TStore(),Auth(),f);assert plane.validate("e").evidence.job_ref=="fc-1"
    with pytest.raises(ValueError):plane.validate("other")
    forged=json.loads(raw);forged["deployment_attestation_digest"]="b"*64;forged=canonical_json(forged);f.files[("cv",EVIDENCE+"/terminal-evidence.v1.json")]=forged;f.files[("cv",EVIDENCE+"/terminal-evidence.v1.mac")]=Auth().sign(forged)
    with pytest.raises(ValueError):plane.validate("e")
def test_log_control_reauthenticates_metadata_inventory_content_and_chain():
    raw,chunk=log_chunk();chain=chunk.chunk_digest;metadata=log_metadata(raw,chain);ex=LogExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",1);_,f,_=stage_setup();f.files.update({("cv",LOGS+"/log-metadata.v1.json"):metadata,("cv",LOGS+"/log-metadata.v1.mac"):Auth().sign(metadata),("cv",LOGS+"/chunks/000.json"):raw});f.listings[("cv",LOGS+"/chunks/")]=((LOGS+"/chunks/000.json",len(raw),"log-entry-0"),)
    class LStore:
        def load_log_expectation(self,effect_id):return ex
    plane=LogControlPlane(LStore(),Auth(),f);assert plane.validate("e").chain_digest==chain
    f.listings[("cv",LOGS+"/chunks/")]=((LOGS+"/chunks/000.json",len(raw),"log-entry-0"),(LOGS+"/chunks/000.json",len(raw),"log-entry-0"))
    with pytest.raises(ValueError):plane.validate("e")
def test_all_provider_integer_fields_reject_bool_substitution():
    ex=expectation()
    with pytest.raises(ValueError):replace(ex,bundle_size=True)
    with pytest.raises(ValueError):ArtifactMemberV1(ArtifactRole.TOKENIZER,OUTPUT+"/tokenizer",True,D,"id")
    items=[member(r,OUTPUT+"/"+r.value,r.value.encode()) for r in ArtifactRole]
    value=json.loads(manifest_data(items));value["generation"]=True
    with pytest.raises(ValueError):CompletionManifestV1.parse(canonical_json(value))
    value=json.loads(terminal_data());value["generation"]=True
    with pytest.raises(ValueError):TerminalEvidenceV1.parse(canonical_json(value))
    raw,_=log_chunk();value=json.loads(raw);value["generation"]=True
    with pytest.raises(ValueError):StructuredLogChunkV1.parse(canonical_json(value))
    value=json.loads(raw);value["sequence"]=True
    with pytest.raises(ValueError):StructuredLogChunkV1.parse(canonical_json(value))
    with pytest.raises(ValueError):LogExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",True)
    manifest=CompletionManifestV1.parse(manifest_data(items));_,f,_=stage_setup();f.listing=tuple((x.path,True,x.provider_entry_id) for x in items)
    with pytest.raises(ValueError):verify_artifacts(manifest,completion_expectation(),f)
    ex,f,receipt=stage_setup();claim=json.loads(f.files[("cv",CONTROL+"/stage-claim.v1.json")]);claim["bundle_size"]=True;raw=canonical_json(claim);ex=replace(ex,claim_digest=sha(raw));f.files[("cv",CONTROL+"/stage-claim.v1.json")]=raw;f.files[("cv",CONTROL+"/stage-claim.v1.mac")]=Auth().sign(raw);receipt=replace(receipt,claim_digest=sha(raw))
    with pytest.raises(ValueError):StageControlPlane(Store(ex),Auth(),f).validate(receipt)
def test_cross_plane_identity_exact_equality_includes_host_key_and_binding():
    base=completion_expectation().identity
    changes=(replace(base,binding=binding("1.5.3")),replace(base,control_volume_id="other"),replace(base,artifact_volume_id="other"),replace(base,job_ref="other"),replace(base,effect_id="other"),replace(base,command_digest="b"*64),replace(base,plan_digest="b"*64),replace(base,deployment_attestation_digest="b"*64),replace(base,invocation_nonce="other"),replace(base,generation=2),replace(base,key_ref="other"))
    assert all(type(changed) is CrossPlaneIdentityV1 and changed!=base for changed in changes)
def test_interleaved_terminal_validations_return_isolated_frozen_results():
    identity1=provider_identity();identity2={**provider_identity(),"control_volume_id":"cv2","artifact_volume_id":"av2","job_ref":"fc-2","effect_id":"e2","invocation_nonce":"nonce2","generation":2}
    raw1=terminal_data(identity=identity1);raw2=terminal_data(identity=identity2)
    ex1=TerminalExpectationV1(binding(),"cv","av","key","fc-1","e",D,D,D,"nonce",1);ex2=TerminalExpectationV1(binding(),"cv2","av2","key","fc-2","e2",D,D,D,"nonce2",2)
    f=Fake(binding());f.files.update({("cv",EVIDENCE+"/terminal-evidence.v1.json"):raw1,("cv",EVIDENCE+"/terminal-evidence.v1.mac"):Auth().sign(raw1),("cv2","operations/e2/evidence/terminal-evidence.v1.json"):raw2,("cv2","operations/e2/evidence/terminal-evidence.v1.mac"):Auth().sign(raw2)})
    class TStore:
        def load_terminal_expectation(self,effect_id):return {"e":ex1,"e2":ex2}[effect_id]
    plane=TerminalControlPlane(TStore(),Auth(),f);first=plane.validate("e");second=plane.validate("e2")
    assert first.canonical_digest==sha(raw1) and first.identity==ex1.identity and second.canonical_digest==sha(raw2) and second.identity==ex2.identity and not hasattr(plane,"last_bytes")
    with pytest.raises(Exception):first.canonical_digest=D
def test_public_exports_have_no_mutation_or_private_driver():
    import tuner.execution.providers.modal as modal
    assert not ({"submit","cancel","_ModalEffectDriver","ExplicitModal154ReadFacade"}&set(modal.__all__))
def test_private_effect_driver_composes_with_broker_executor_once():
    from tuner.execution.providers.modal._effects import _ModalEffectDriver
    effect=EffectIdentity("e","op",EffectKind.SUBMIT,ExecutionScope("modal","acct","env"));calls=[]
    class Control:
        def validate(self,receipt):calls.append("validated")
    class Mutator:
        def execute_once(self,raw):calls.append("mutated");return EffectObservation(effect,EffectDisposition.FOUND,"fc-1",D)
    driver=_ModalEffectDriver(lambda raw:(effect,object()),Control(),Mutator());assert _ProviderEffectExecutor(driver).execute_once(b"canonical").provider_job_ref=="fc-1" and calls==["validated","mutated"]
