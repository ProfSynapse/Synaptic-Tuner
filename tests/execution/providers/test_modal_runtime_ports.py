from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuner.execution.providers.modal.runtime import (
    EnvironmentHmacAuthenticator,
    GitDualCloneMaterializer,
    SubprocessSftRunner,
    _same_executable,
)
from tuner.execution.providers.modal.remote import ModalRemotePhaseError


def test_runtime_identity_accepts_distinct_paths_to_same_binary(tmp_path):
    binary = tmp_path / "python3.11"
    alias = tmp_path / "python3"
    other = tmp_path / "other-python3"
    binary.write_bytes(b"locked-interpreter")
    alias.hardlink_to(binary)
    other.write_bytes(binary.read_bytes())
    assert _same_executable(str(binary), str(alias)) is True
    assert _same_executable(str(binary), str(other)) is False


def test_environment_hmac_authenticator_requires_exact_base64_key_and_ref(monkeypatch):
    import base64
    monkeypatch.setenv("SYNAPTIC_EVIDENCE_MAC_KEY",base64.b64encode(b"k"*32).decode("ascii"))
    authenticator=EnvironmentHmacAuthenticator(environment_key="SYNAPTIC_EVIDENCE_MAC_KEY",key_ref="evidence-v1")
    tag=authenticator.sign("purpose/v1",b"payload","evidence-v1")
    assert authenticator.verify("purpose/v1",b"payload",tag,"evidence-v1")
    assert not authenticator.verify("purpose/v1",b"changed",tag,"evidence-v1")
    with pytest.raises(ValueError,match="reference"):authenticator.sign("purpose/v1",b"payload","other")
    monkeypatch.setenv("SYNAPTIC_EVIDENCE_MAC_KEY","not-base64")
    with pytest.raises(ValueError,match="invalid"):authenticator.sign("purpose/v1",b"payload","evidence-v1")


def test_subprocess_runner_uses_no_shell_and_never_returns_captured_secret_output(monkeypatch):
    calls=[]
    def run(argv,**kwargs):calls.append((argv,kwargs));return SimpleNamespace(returncode=7,stdout=b"token=secret",stderr=b"Bearer secret")
    monkeypatch.setattr("tuner.execution.providers.modal.runtime.subprocess.run",run)
    monkeypatch.setenv("HF_TOKEN","secret")
    runner=SubprocessSftRunner(secret_keys=("HF_TOKEN",),timeout_seconds=10)
    result=runner.run(("/python","/runtime.py","--canonical-workload-stdin"),cwd="/tmp",environment={"SAFE":"1"},stdin=b"workload")
    assert result.returncode==123 and result.stdout==result.stderr==b""
    assert result.diagnostic_code=="trainer_nonzero"
    argv,kwargs=calls[0]
    assert argv==("/python","/runtime.py","--canonical-workload-stdin")
    assert kwargs["shell"] is False and kwargs["env"]["HF_TOKEN"]=="secret"
    assert kwargs["input"]==b"workload" and kwargs["timeout"]==10


@pytest.mark.parametrize(
    ("runtime_exit", "remote_exit", "diagnostic_code"),
    (
        (2, 121, "runtime_unclassified_rejection"),
        (20, 124, "runtime_workload_rejected"),
        (21, 122, "runtime_artifact_precondition"),
        (22, 124, "runtime_invocation_rejected"),
        (23, 123, "runtime_trainer_failed"),
        (24, 123, "runtime_evidence_rejected"),
        (25, 122, "runtime_artifact_rejected"),
        (30, 124, "runtime_workload_document_rejected"),
        (31, 124, "runtime_workload_engine_rejected"),
        (32, 124, "runtime_workload_schema_rejected"),
        (33, 124, "runtime_workload_reconstruction_rejected"),
        (34, 124, "runtime_workload_fingerprint_rejected"),
        (35, 124, "runtime_workload_roots_rejected"),
    ),
)
def test_subprocess_runner_maps_closed_runtime_stages(
    monkeypatch, runtime_exit, remote_exit, diagnostic_code
):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr(
        "tuner.execution.providers.modal.runtime.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=runtime_exit),
    )
    result = SubprocessSftRunner(secret_keys=("HF_TOKEN",), timeout_seconds=10).run(
        ("/python", "/runtime.py", "--canonical-workload-stdin"),
        cwd="/tmp", environment={}, stdin=b"workload",
    )
    assert (result.returncode, result.diagnostic_code) == (
        remote_exit, diagnostic_code
    )


def test_subprocess_runner_rejects_missing_secret_and_command_override(monkeypatch):
    monkeypatch.delenv("HF_TOKEN",raising=False)
    runner=SubprocessSftRunner(secret_keys=("HF_TOKEN",),timeout_seconds=10)
    with pytest.raises(ValueError,match="command"):
        runner.run(("/python","/runtime.py","--other"),cwd="/tmp",environment={},stdin=b"x")
    with pytest.raises(ModalRemotePhaseError) as failure:
        runner.run(("/python","/runtime.py","--canonical-workload-stdin"),cwd="/tmp",environment={},stdin=b"x")
    assert (failure.value.returncode,failure.value.diagnostic_code)==(120,"credential_unavailable")


def test_remote_git_subprocess_ignores_home_and_global_system_config(monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append((argv,kwargs));return SimpleNamespace(returncode=0,stdout=b"ok")
    monkeypatch.setattr("tuner.execution.providers.modal.runtime.subprocess.run",run)
    assert GitDualCloneMaterializer._subprocess(("git","--version"))==b"ok"
    environment=calls[0][1]["env"]
    assert environment["HOME"]=="/tmp/synaptic-modal-git-home"
    assert environment["GIT_CONFIG_NOSYSTEM"]=="1"
    assert environment["GIT_CONFIG_GLOBAL"]==environment["GIT_CONFIG_SYSTEM"]
    assert environment["GCM_INTERACTIVE"]=="Never"
