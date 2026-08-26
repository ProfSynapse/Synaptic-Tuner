from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from tests.execution.providers.test_modal_source_resolution import _deployment
from tests.execution.test_mutation_broker import command
from tuner.execution.broker import MutationCommandV1
from tuner.execution.contracts import EffectDisposition
from tuner.execution.providers.modal.binding import ModalClientBinding, Readiness, readiness
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade, ModalFacadeError,ModalFunctionCallState
from tuner.execution.providers.modal.mutation import _ExplicitModal154FunctionMutator
from tuner.execution.providers.modal.resolution import VerifiedModalDeploymentIdentityV1
from tuner.execution.providers.modal.staging import _ExplicitModal154VolumeWriter, prepare_modal_stage


@dataclass
class Entry:
    path: str
    size: int
    mtime: int = 1
    type: int = 1


class Upload:
    def __init__(self, volume, force): self.volume=volume;self.force=force;self.pending=[]
    def __enter__(self): return self
    def put_file(self, source, path): self.pending.append((path,source.read()))
    def __exit__(self, kind, value, trace):
        if value is None:
            for path,data in self.pending:
                if not self.force and path in self.volume.files: raise RuntimeError("collision")
                self.volume.files[path]=data


class FakeVolume:
    registry={};calls=[]
    def __init__(self,object_id):self.object_id=object_id;self.files={};self.entries=None
    @classmethod
    def from_name(cls,name,**kwargs):cls.calls.append((name,kwargs));return cls.registry[name]
    def read_file(self,path):
        data=self.files[path]
        midpoint=max(1,len(data)//2)
        yield data[:midpoint]
        if midpoint<len(data):yield data[midpoint:]
    def iterdir(self,prefix,recursive=True):
        if self.entries is not None:yield from self.entries;return
        for path,data in sorted(self.files.items()):
            if path.startswith(prefix):yield Entry(path,len(data))
    def batch_upload(self,force=False):return Upload(self,force)


class FakeCall:
    result=TimeoutError()
    def __init__(self,object_id="fc-1"):self.object_id=object_id
    def get(self,timeout=None):
        if isinstance(type(self).result,BaseException):raise type(self).result
        return type(self).result


class FakeFunction:
    calls=[];spawn_calls=[];fail=False
    def __init__(self):self.object_id="fu-1"
    @classmethod
    def from_name(cls,app,name,**kwargs):cls.calls.append((app,name,kwargs));return cls()
    def hydrate(self,client):return self
    def spawn(self,*args):
        type(self).spawn_calls.append(args)
        if type(self).fail:raise RuntimeError("secret provider detail")
        return FakeCall()


class FakeFunctionCall:
    calls=[]
    @classmethod
    def from_id(cls,value,client=None):cls.calls.append((value,client));return FakeCall(value)


class SDK:
    __version__="1.5.4";Volume=FakeVolume;Function=FakeFunction;FunctionCall=FakeFunctionCall


class Auth:
    def sign(self,purpose,payload,key_ref):return b"tag"


def make_facade(*,sdk=SDK,selection=None):
    FakeVolume.calls=[];FakeFunction.calls=[];FakeFunction.spawn_calls=[];FakeFunction.fail=False;FakeCall.result=TimeoutError()
    FakeVolume.registry={"control-name":FakeVolume("cv"),"artifact-name":FakeVolume("av")}
    client=object();binding=ModalClientBinding("acct","workspace","env","client","1.5.4")
    selected=selection or replace(_deployment(),function_version="1")
    facade=ExplicitModal154ReadFacade(
        binding,sdk=sdk,client=client,
        scope_observer=lambda supplied:("acct","workspace","env","client") if supplied is client else (),
        deployment_observer=lambda **kwargs:selected,
        volume_names={"cv":"control-name","av":"artifact-name"},
    )
    return facade,selected


def verified(selection, *, attestation="d" * 64):
    fields={"selection":selection,"issuer_ref":"modal-verifier","evidence_ref":"deployment-proof","audience_ref":"project/run-1","challenge_nonce":"deployment-nonce","verified_at":"2026-08-25T12:01:00Z","expires_at":"2026-08-25T12:10:00Z","key_ref":"deployment-key"}
    unsigned={"schema_version":"synaptic-verified-modal-deployment/v1","selection":selection.to_dict(),**{name:fields[name] for name in ("issuer_ref","evidence_ref","audience_ref","challenge_nonce","verified_at","expires_at","key_ref")}}
    payload=__import__("json").dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    actual=hashlib.sha256(payload).hexdigest()
    selected_digest=actual if attestation=="d"*64 else attestation
    return VerifiedModalDeploymentIdentityV1(**fields,tag_base64="dGFn",attestation_digest=selected_digest)


@pytest.mark.parametrize("version",[None,"1.5.1","1.5.3","1.5.5","garbage"])
def test_exact_sdk_version_is_mandatory(version):
    bad=type("BadSDK",(),{"__version__":version,"Volume":FakeVolume,"Function":FakeFunction})
    binding=ModalClientBinding("acct","workspace","env","client","1.5.4")
    with pytest.raises(ModalFacadeError,match="version_mismatch"):
        ExplicitModal154ReadFacade(binding,sdk=bad,client=object(),scope_observer=lambda _:(),deployment_observer=lambda **_:None,volume_names={"cv":"control"})


def test_read_list_and_deployment_use_only_explicit_client_environment_and_v1():
    facade,selection=make_facade();FakeVolume.registry["control-name"].files["control/x"]=b"abcdef"
    assert readiness(facade.binding,facade) is Readiness.READY
    assert facade.read_complete("cv","control/x",max_bytes=6)==b"abcdef"
    listed=facade.list_prefix("cv","control/",max_entries=1)
    assert listed[0][:2]==("control/x",6) and len(listed[0][2])==64
    assert facade.inspect_deployment(app_name=selection.app_name,function_name=selection.function_name)==selection
    name,kwargs=FakeVolume.calls[0]
    assert name=="control-name" and kwargs=={"environment_name":"env","create_if_missing":False,"version":1,"client":facade.client}
    assert FakeFunction.calls[-1][2]=={"version":1,"environment_name":"env","client":facade.client}


def test_reads_and_listings_fail_closed_on_bounds_duplicates_and_identity_drift():
    facade,_=make_facade();volume=FakeVolume.registry["control-name"];volume.files["control/x"]=b"abcdef"
    with pytest.raises(ModalFacadeError,match="read_failed"):facade.read_complete("cv","control/x",max_bytes=5)
    volume.entries=[Entry("control/x",1),Entry("control/x",1)]
    with pytest.raises(ModalFacadeError,match="list_failed"):facade.list_prefix("cv","control/",max_entries=3)
    volume.entries=[];volume.object_id="wrong"
    with pytest.raises(ModalFacadeError,match="identity_mismatch"):facade.list_prefix("cv","control/",max_entries=3)


def test_prepare_persist_stage_readback_and_no_overwrite():
    facade,_=make_facade();material=prepare_modal_stage(command().operation,facade.binding,b"bundle",Auth())
    # A real host persists material.expectation durably before this call.
    receipt=_ExplicitModal154VolumeWriter(facade).stage_once(material)
    assert receipt.effect_id==command().effect.effect_id
    assert FakeVolume.registry["artifact-name"].files["operations/e/input/bundle.bin"]==b"bundle"
    assert FakeVolume.registry["control-name"].files["operations/e/control/stage-claim.v1.mac"]==b"tag"
    assert all(call[1]["create_if_missing"] is False for call in FakeVolume.calls)
    assert _ExplicitModal154VolumeWriter(facade).stage_once(material)==receipt


def test_stage_resumes_an_exact_one_volume_partial_write_without_overwrite():
    facade,_=make_facade();material=prepare_modal_stage(command().operation,facade.binding,b"bundle",Auth())
    FakeVolume.registry["artifact-name"].files["operations/e/input/bundle.bin"]=b"bundle"
    receipt=_ExplicitModal154VolumeWriter(facade).stage_once(material)
    assert receipt.effect_id=="e"
    assert FakeVolume.registry["control-name"].files["operations/e/control/stage-claim.v1.json"]==material.claim


def test_stage_rejects_changed_partial_content():
    facade,_=make_facade();material=prepare_modal_stage(command().operation,facade.binding,b"bundle",Auth())
    FakeVolume.registry["artifact-name"].files["operations/e/input/bundle.bin"]=b"changed"
    with pytest.raises(ModalFacadeError,match="collision"):
        _ExplicitModal154VolumeWriter(facade).stage_once(material)


def test_mutator_spawns_once_never_remote_and_returns_indeterminate_after_boundary():
    facade,deployment=make_facade();evidence=verified(deployment);mutator=_ExplicitModal154FunctionMutator(facade,evidence)
    bound=MutationCommandV1(replace(command().operation,deployment_attestation_digest=evidence.attestation_digest),command().bundle_digest,command().stage_claim_digest);raw=bound.canonical_bytes
    observation=mutator.execute_once(raw)
    assert observation.disposition is EffectDisposition.FOUND and observation.provider_job_ref=="fc-1"
    assert FakeFunction.spawn_calls==[(raw,)] and not hasattr(FakeFunction,"remote")
    FakeFunction.fail=True
    observation=mutator.execute_once(raw)
    assert observation.disposition is EffectDisposition.INDETERMINATE
    assert FakeFunction.spawn_calls==[(raw,),(raw,)]
    handle=mutator.lookup_handle("fc-1")
    assert handle.object_id=="fc-1" and FakeFunctionCall.calls[-1]==("fc-1",facade.client)


def test_mutator_rejects_deployment_target_substitution_before_spawn():
    facade,deployment=make_facade()
    original=verified(deployment)
    bound=MutationCommandV1(replace(command().operation,deployment_attestation_digest=original.attestation_digest),command().bundle_digest,command().stage_claim_digest)
    substituted=replace(deployment,function_name="substituted-function")
    mutator=_ExplicitModal154FunctionMutator(facade,verified(substituted))
    with pytest.raises(ModalFacadeError,match="binding_mismatch"):
        mutator.execute_once(bound.canonical_bytes)
    assert FakeFunction.spawn_calls==[]


def test_mutation_command_parser_rejects_noncanonical_and_digest_substitution():
    raw=command().canonical_bytes
    assert MutationCommandV1.from_bytes(raw)==command()
    with pytest.raises(ValueError,match="canonical"):MutationCommandV1.from_bytes(raw+b" ")
    changed=raw.replace(command().operation_binding_digest.encode(),b"b"*64,1)
    with pytest.raises(ValueError,match="digest mismatch"):MutationCommandV1.from_bytes(changed)


def test_function_call_observation_is_read_only_bounded_and_non_authoritative():
    facade,_=make_facade()
    assert facade.observe_function_call("fc-1") is ModalFunctionCallState.PENDING
    FakeCall.result={"schema_version":"synaptic-modal-worker-result/v1","effect_id":"e","returncode":0,"status_code":"completed"}
    assert facade.observe_function_call("fc-1") is ModalFunctionCallState.RETURNED
    FakeCall.result={"provider":"untrusted"}
    assert facade.observe_function_call("fc-1") is ModalFunctionCallState.UNKNOWN
    assert FakeFunctionCall.calls[-1]==("fc-1",facade.client)
