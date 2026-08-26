from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.execution.providers.test_modal_bundle import bundle
from tuner.execution.broker import MutationCommandV1
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.remote import ModalRemotePhaseError, MountedModalWorkerV1, ProcessResultV1, admit_remote_invocation, execute_remote_sft
from tuner.execution.providers.modal.producer import MountedCompletionProducerV1
from tuner.execution.providers.modal.mounted_io import read_regular, write_exclusive
from tuner.execution.providers.modal import mounted_io
from tuner.execution.providers.modal.contracts import ArtifactRole, TerminalEvidenceV1, canonical_json
from tuner.execution.providers.modal.manifest import CompletionManifestV1
from tuner.execution.providers.modal.staging import prepare_modal_stage


class Auth:
    def sign(self,purpose,payload,key_ref):return b"tag"
    def verify(self,purpose,payload,tag,key_ref):return tag==b"tag"


def admitted():
    built=bundle();binding=ModalClientBinding("acct","workspace","env","client","1.5.4")
    material=prepare_modal_stage(built.operation,binding,built.transport_base64,Auth())
    command=MutationCommandV1(built.operation,material.expectation.bundle_digest,material.expectation.claim_digest)
    invocation=admit_remote_invocation(command.canonical_bytes,claim=material.claim,claim_tag=material.claim_tag,bundle_transport=material.bundle,verifier=Auth())
    return invocation,command,material


def test_remote_admission_reauthenticates_and_cross_binds_exact_fixed_runtime():
    invocation,command,material=admitted()
    assert invocation.command==command and invocation.bundle.transport_base64==material.bundle
    assert invocation.argv==(invocation.source.python_executable,"/workspace/engine/Trainers/sft/runtime_v1.py","--canonical-workload-stdin")
    assert invocation.cwd==invocation.source.roots["tmp"]
    assert invocation.environment["SYNAPTIC_ENGINE_ROOT"]=="/workspace/engine"


def test_remote_admission_rejects_tag_command_claim_and_bundle_mutations():
    invocation,command,material=admitted()
    with pytest.raises(ValueError,match="authentication failed"):
        admit_remote_invocation(command.canonical_bytes,claim=material.claim,claim_tag=b"wrong",bundle_transport=material.bundle,verifier=Auth())
    with pytest.raises(ValueError,match="digests"):
        admit_remote_invocation(command.canonical_bytes,claim=material.claim,claim_tag=material.claim_tag,bundle_transport=material.bundle+b"A",verifier=Auth())
    changed=replace(command,bundle_digest="b"*64)
    with pytest.raises(ValueError,match="digests"):
        admit_remote_invocation(changed.canonical_bytes,claim=material.claim,claim_tag=material.claim_tag,bundle_transport=material.bundle,verifier=Auth())


def test_remote_execution_has_no_shell_or_runtime_override_surface():
    invocation,_,_=admitted();calls=[]
    class Sources:
        def prepare_and_verify(self,source,deployment):calls.append(("source",source,deployment))
    class Processes:
        def run(self,argv,*,cwd,environment,stdin):
            calls.append(("process",argv,cwd,environment,stdin));return ProcessResultV1(0,b"ok",b"")
    result=execute_remote_sft(invocation,sources=Sources(),processes=Processes())
    assert result.returncode==0 and calls[0][0]=="source" and calls[1][0]=="process"
    assert calls[1][1]==invocation.argv and calls[1][4]==invocation.workload
    assert "shell" not in Processes.run.__annotations__


def test_mounted_worker_reads_only_fixed_two_volume_paths(tmp_path):
    invocation,command,material=admitted();control=tmp_path/"control-volume";artifact=tmp_path/"artifact-volume"
    control_dir=control/"operations"/"effect-1"/"control";input_dir=artifact/"operations"/"effect-1"/"input"
    control_dir.mkdir(parents=True);input_dir.mkdir(parents=True)
    (control_dir/"stage-claim.v1.json").write_bytes(material.claim)
    (control_dir/"stage-claim.v1.mac").write_bytes(material.claim_tag)
    (input_dir/"bundle.bin").write_bytes(material.bundle)
    class Sources:
        def prepare_and_verify(self,source,deployment):pass
    class Processes:
        def run(self,argv,*,cwd,environment,stdin):return ProcessResultV1(0)
    class Completion:
        def finalize(self,invocation,result,*,job_ref):return type("Done",(),{"status_code":"completed"})()
    worker=MountedModalWorkerV1(verifier=Auth(),sources=Sources(),processes=Processes(),completion=Completion(),control_root=str(control),artifact_root=str(artifact))
    assert worker(command.canonical_bytes,"fc-1")=={"schema_version":"synaptic-modal-worker-result/v1","effect_id":invocation.command.effect.effect_id,"returncode":0,"status_code":"completed"}
    (input_dir/"bundle.bin").unlink();(input_dir/"bundle.bin").mkdir()
    with pytest.raises(ValueError,match="unavailable"):worker(command.canonical_bytes,"fc-1")


def test_mounted_worker_preserves_only_closed_remote_phase_diagnostics(tmp_path):
    invocation,command,material=admitted();control=tmp_path/"control-volume";artifact=tmp_path/"artifact-volume"
    control_dir=control/"operations"/"effect-1"/"control";input_dir=artifact/"operations"/"effect-1"/"input"
    control_dir.mkdir(parents=True);input_dir.mkdir(parents=True)
    (control_dir/"stage-claim.v1.json").write_bytes(material.claim)
    (control_dir/"stage-claim.v1.mac").write_bytes(material.claim_tag)
    (input_dir/"bundle.bin").write_bytes(material.bundle)
    class Sources:
        def prepare_and_verify(self,source,deployment):
            raise ModalRemotePhaseError(124,"engine_clone_failed")
    class Processes:
        def run(self,argv,*,cwd,environment,stdin):raise AssertionError
    observed=[]
    class Completion:
        def finalize(self,invocation,result,*,job_ref):
            observed.append(result)
            return type("Done",(),{"status_code":"failed"})()
    worker=MountedModalWorkerV1(verifier=Auth(),sources=Sources(),processes=Processes(),completion=Completion(),control_root=str(control),artifact_root=str(artifact))
    result=worker(command.canonical_bytes,"fc-1")
    assert result["returncode"]==124 and result["status_code"]=="failed"
    assert observed[0].diagnostic_code=="engine_clone_failed"


def test_completion_failure_log_contains_only_closed_diagnostic_code(tmp_path):
    invocation,_,_=admitted();control=tmp_path/"control";artifact=tmp_path/"artifact"
    completed=MountedCompletionProducerV1(Auth(),control_root=str(control),artifact_root=str(artifact)).finalize(
        invocation,ProcessResultV1(121,diagnostic_code="runtime_lock_mismatch"),job_ref="fc-1"
    )
    assert completed.status_code=="failed"
    value=__import__("json").loads((control/"operations"/"effect-1"/"logs"/"chunks"/"000.json").read_bytes())
    assert value["records"]==[{"code":"failed","message":"training failed: runtime_lock_mismatch"}]


def test_completion_producer_publishes_exact_five_and_authenticated_terminal(tmp_path):
    invocation,_,_=admitted();control=tmp_path/"control";volume=tmp_path/"volume"
    roots=dict(invocation.source.roots)
    for name in ("artifacts","state","tracking","cache","tmp"):
        roots[name]=str(volume/invocation.source.run_id/name)
        (volume/invocation.source.run_id/name).mkdir(parents=True)
    environment=dict(invocation.source.environment)
    for name,variable in {"artifacts":"SYNAPTIC_ARTIFACT_ROOT","state":"SYNAPTIC_STATE_ROOT","tracking":"SYNAPTIC_TRACKING_ROOT","cache":"SYNAPTIC_CACHE_ROOT","tmp":"SYNAPTIC_TMP_ROOT"}.items():environment[variable]=roots[name]
    environment["HF_HOME"]=roots["cache"]+"/huggingface";environment["TRANSFORMERS_CACHE"]=roots["cache"]+"/transformers"
    source=replace(invocation.source,roots=roots,writable_capability_root=str(volume),environment=environment);invocation=replace(invocation,source=source)
    records=[]
    for role in ArtifactRole:
        name=role.value+".bin";content=(role.value+"-content").encode()
        (volume/invocation.source.run_id/"artifacts"/name).write_bytes(content)
        import hashlib
        records.append({"role":role.value,"path":name,"sha256":hashlib.sha256(content).hexdigest(),"size":len(content)})
    import hashlib
    workload_fingerprint=hashlib.sha256(b"synaptic-training-workload/v1\0"+invocation.workload).hexdigest()
    (volume/invocation.source.run_id/"state"/"runtime-v1-inventory.json").write_bytes(canonical_json({"schema_version":"synaptic-artifact-inventory/v1","workload_fingerprint":workload_fingerprint,"artifacts":records}))
    class SigningAuth(Auth):
        def sign(self,purpose,payload,key_ref):return b"signed-tag"
    completed=MountedCompletionProducerV1(SigningAuth(),control_root=str(control),artifact_root=str(volume)).finalize(invocation,ProcessResultV1(0),job_ref="fc-1")
    root="operations/effect-1"
    assert completed.status_code=="completed"
    assert len(list((volume/root/"output").iterdir()))==5
    terminal=(control/root/"evidence"/"terminal-evidence.v1.json").read_bytes()
    manifest=(control/root/"evidence"/"completion-manifest.v1.json").read_bytes()
    assert TerminalEvidenceV1.parse(terminal).status_code=="completed"
    assert len(CompletionManifestV1.parse(manifest).members)==5
    assert (control/root/"evidence"/"terminal-evidence.v1.mac").read_bytes()==b"signed-tag"


def test_completion_producer_rejects_inventory_from_another_workload(tmp_path):
    invocation,_,_=admitted();control=tmp_path/"control";volume=tmp_path/"volume"
    roots=dict(invocation.source.roots)
    for name in ("artifacts","state","tracking","cache","tmp"):
        roots[name]=str(volume/invocation.source.run_id/name);(volume/invocation.source.run_id/name).mkdir(parents=True)
    environment=dict(invocation.source.environment)
    for name,variable in {"artifacts":"SYNAPTIC_ARTIFACT_ROOT","state":"SYNAPTIC_STATE_ROOT","tracking":"SYNAPTIC_TRACKING_ROOT","cache":"SYNAPTIC_CACHE_ROOT","tmp":"SYNAPTIC_TMP_ROOT"}.items():environment[variable]=roots[name]
    environment["HF_HOME"]=roots["cache"]+"/huggingface";environment["TRANSFORMERS_CACHE"]=roots["cache"]+"/transformers"
    source=replace(invocation.source,roots=roots,writable_capability_root=str(volume),environment=environment);invocation=replace(invocation,source=source)
    records=[]
    import hashlib
    for role in ArtifactRole:
        name=role.value+".bin";content=b"unrelated"+role.value.encode();(Path(roots["artifacts"])/name).write_bytes(content);records.append({"role":role.value,"path":name,"sha256":hashlib.sha256(content).hexdigest(),"size":len(content)})
    (Path(roots["state"])/"runtime-v1-inventory.json").write_bytes(canonical_json({"schema_version":"synaptic-artifact-inventory/v1","workload_fingerprint":"a"*64,"artifacts":records}))
    with pytest.raises(ValueError,match="workload mismatch"):
        MountedCompletionProducerV1(Auth(),control_root=str(control),artifact_root=str(volume)).finalize(invocation,ProcessResultV1(0),job_ref="fc-1")


def test_mounted_reads_and_writes_reject_symlinked_ancestors(tmp_path):
    root=tmp_path/"mount";outside=tmp_path/"outside";root.mkdir();outside.mkdir()
    link=root/"operations"
    try:
        link.symlink_to(outside,target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    (outside/"member.bin").write_bytes(b"secret")
    with pytest.raises(ValueError,match="unavailable"):
        read_regular(root,link/"member.bin",32)
    with pytest.raises(ValueError,match="trusted directory"):
        write_exclusive(root,link/"new.bin",b"content")
    assert not (outside/"new.bin").exists()


@pytest.mark.skipif(os.name != "posix", reason="Modal runtime uses Linux dirfd traversal")
def test_descriptor_relative_io_defeats_validated_ancestor_substitution(tmp_path,monkeypatch):
    assert mounted_io._SECURE_DIRFD
    root=tmp_path/"mount";ancestor=root/"operations"/"effect"/"output"
    displaced=root/"displaced";outside=tmp_path/"outside"
    ancestor.mkdir(parents=True);outside.mkdir()
    (ancestor/"source.bin").write_bytes(b"trusted")
    (outside/"source.bin").write_bytes(b"substituted")
    original_open=mounted_io._open_leaf;swapped=False
    def substitute(parent,leaf,path,flags,mode=None):
        nonlocal swapped
        if not swapped:
            swapped=True
            ancestor.rename(displaced)
            ancestor.symlink_to(outside,target_is_directory=True)
        return original_open(parent,leaf,path,flags,mode)
    monkeypatch.setattr(mounted_io,"_open_leaf",substitute)
    assert read_regular(root,ancestor/"source.bin",32)==b"trusted"
    assert swapped and not (outside/"new.bin").exists()

    ancestor.unlink();displaced.rename(ancestor);swapped=False
    write_exclusive(root,ancestor/"new.bin",b"published")
    assert swapped and not (outside/"new.bin").exists()
    assert (displaced/"new.bin").read_bytes()==b"published"
