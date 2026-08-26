from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from tuner.execution.providers.modal.resolution import (
    ModalDeploymentSelectionV1,
    ModalDualCloneSourceFinalizer,
    VerifiedModalDeploymentIdentityV1,
)
from tuner.execution.providers.modal.deployment_identity import modal_function_name
from tuner.project.context import ProjectContext
from tuner.project.execution_source import AuthenticatedSourceEvidenceV1, ExecutionSourceV1
from tuner.project.source_bundle import SourceLock, SourceLockError
from tuner.execution.evidence import ReplayDisposition


def _source(mode: str = "superproject", path: str = "vendor/engine") -> SourceLock:
    project = {
        "url": "https://github.com/example/product.git", "commit": "a" * 40,
        "dirty": False, "pushed": False,
    }
    engine = {
        "url": "https://github.com/example/engine.git", "commit": "b" * 40,
        "dirty": False, "pushed": False,
        "submodule_path": path, "gitlink_commit": "b" * 40,
    }
    if mode == "standalone":
        engine = dict(project)
    return SourceLock.from_dict({
        "schema_version": "synaptic-source-lock/v1", "run_id": "run-1",
        "created_at": "2026-08-25T12:00:00Z", "mode": mode,
        "sources": {"project": project, "engine": engine},
        "project": {}, "configuration": {}, "plugins": [], "inputs": [],
        "runtime": {}, "outputs": {},
    })


def _deployment() -> ModalDeploymentSelectionV1:
    deployment_ref = "modal-deployment-" + "1" * 32
    return ModalDeploymentSelectionV1(
        account_ref="acct", workspace_ref="workspace", environment_ref="env",
        client_ref="client", app_name="synaptic-training-v1",
        function_name=modal_function_name(deployment_ref),
        deployment_ref=deployment_ref, image_digest="1" * 64,
        dependency_lock_digest="2" * 64, wrapper_digest="3" * 64,
        runtime_digest="4" * 64, python_version="3.12.7",
        python_executable="/usr/local/bin/python3.12", python_executable_digest="5" * 64,
        secret_requirements_digest="6" * 64,
        provider_runtime_requirements_digest="7" * 64,
        runtime_environment={"PATH": "/usr/local/bin", "LANG": "C.UTF-8"},
    )


class LocalPort:
    def __init__(self, source: SourceLock) -> None:
        self.source = source
        self.calls = 0

    def inspect(self, *, context: ProjectContext) -> SourceLock:
        self.calls += 1
        return self.source


class EvidencePort:
    def __init__(self, *, engine_commit: str | None = None,key_ref: str="source-key") -> None:
        self.calls = 0
        self.engine_commit = engine_commit
        self.key_ref=key_ref

    def verify(self, source_lock: SourceLock) -> AuthenticatedSourceEvidenceV1:
        self.calls += 1
        commit = self.engine_commit or source_lock.engine_source.commit
        value=AuthenticatedSourceEvidenceV1(
            project_url=source_lock.project_source.location.canonical_url,
            project_commit=source_lock.project_source.commit,
            engine_url=source_lock.engine_source.location.canonical_url,
            engine_commit=commit,
            engine_submodule_path=source_lock.engine_source.submodule_path,
            gitlink_commit=source_lock.engine_source.gitlink_commit,
            issuer_ref="authenticated-verifier", evidence_ref="proof-1",
            audience_ref="project/run-1", challenge_nonce="source-nonce",
            verified_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:10:00Z",
            key_ref=self.key_ref, tag_base64="dGFn", attestation_digest="e" * 64,
        )
        return replace(value,attestation_digest=hashlib.sha256(value.authenticated_payload).hexdigest())


class DeploymentPort:
    def __init__(self, *, drift: bool = False,key_ref: str="deployment-key") -> None:
        self.calls = 0
        self.drift = drift
        self.key_ref=key_ref

    def verify(self, selection: ModalDeploymentSelectionV1) -> VerifiedModalDeploymentIdentityV1:
        self.calls += 1
        if self.drift:
            deployment_ref = "modal-deployment-" + "2" * 32
            selection = replace(
                selection, deployment_ref=deployment_ref,
                function_name=modal_function_name(deployment_ref),
            )
        fields={"selection":selection,"issuer_ref":"modal-verifier","evidence_ref":"deployment-proof","audience_ref":"project/run-1","challenge_nonce":"deployment-nonce","verified_at":"2026-08-25T12:02:00Z","expires_at":"2026-08-25T12:07:00Z","key_ref":self.key_ref}
        unsigned={"schema_version":"synaptic-verified-modal-deployment/v1","selection":selection.to_dict(),**{name:fields[name] for name in ("issuer_ref","evidence_ref","audience_ref","challenge_nonce","verified_at","expires_at","key_ref")}}
        payload=json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
        return VerifiedModalDeploymentIdentityV1(**fields,tag_base64="dGFn",attestation_digest=hashlib.sha256(payload).hexdigest())


class Auth:
    def sign(self,purpose,payload,key_ref):return b"tag"
    def verify(self,purpose,payload,tag,key_ref):return tag==b"tag"


class Replay:
    def __init__(self):self.values={}
    def admit(self,**value):
        key=(value["purpose"],value["challenge_nonce"]);prior=self.values.get(key)
        if prior is None:self.values[key]=value;return ReplayDisposition.ADMITTED
        return ReplayDisposition.IDEMPOTENT if prior==value else ReplayDisposition.COLLISION


def _context(tmp_path: Path, path: str = "vendor/engine") -> ProjectContext:
    project = tmp_path / "product"
    engine = project.joinpath(*path.split("/"))
    engine.mkdir(parents=True, exist_ok=True)
    return ProjectContext.host(engine_root=engine, project_root=project)


def _finalizer(source: SourceLock, *, local: SourceLock | None = None, drift: bool = False,source_key: str="source-key",deployment_key: str="deployment-key"):
    return ModalDualCloneSourceFinalizer(
        LocalPort(source if local is None else local), EvidencePort(key_ref=source_key), DeploymentPort(drift=drift,key_ref=deployment_key),
        authenticator=Auth(),replay=Replay(),clock=lambda:"2026-08-25T12:03:00Z",
        source_issuer_ref="authenticated-verifier",deployment_issuer_ref="modal-verifier",
        source_key_ref="source-key",deployment_key_ref="deployment-key",
    )


def test_finalizer_derives_the_single_dual_clone_execution_source(tmp_path: Path) -> None:
    source = _source()
    resolution = _finalizer(source).finalize(
        source, context=_context(tmp_path), deployment=_deployment(),audience_ref="project/run-1"
    )
    finalized = resolution.execution_source

    assert isinstance(finalized, ExecutionSourceV1)
    assert finalized.to_dict()["topology"] == {
        "provenance_mode": "superproject", "execution_mode": "dual_clone",
        "engine_submodule_path": "vendor/engine",
    }
    assert finalized.roots == {
        "engine": "/workspace/engine", "project": "/workspace/project",
        "artifacts": "/workspace/run/run-1/artifacts",
        "state": "/workspace/run/run-1/state",
        "tracking": "/workspace/run/run-1/tracking",
        "cache": "/workspace/run/run-1/cache", "tmp": "/workspace/run/run-1/tmp",
    }
    assert finalized.deployment_member_sha256 == __import__("hashlib").sha256(
        json.dumps(resolution.deployment.to_dict(), sort_keys=True, separators=(",", ":"))
        .encode()
    ).hexdigest()
    schema = json.loads(
        (Path(__file__).parents[3] / "schemas" / "synaptic-execution-source-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    validator_for(schema).check_schema(schema)
    validator_for(schema)(schema).validate(finalized.to_dict())
    assert ExecutionSourceV1.from_dict(finalized.to_dict()).canonical_bytes == finalized.canonical_bytes


@pytest.mark.parametrize("mode", ["standalone", "dual_clone"])
def test_finalizer_rejects_non_superproject_provenance(tmp_path: Path, mode: str) -> None:
    source = _source(mode)
    with pytest.raises(SourceLockError, match="superproject provenance"):
        _finalizer(source).finalize(source, context=_context(tmp_path), deployment=_deployment(),audience_ref="project/run-1")


def test_finalizer_rejects_local_head_origin_dirty_or_gitlink_drift(tmp_path: Path) -> None:
    source = _source()
    mutations = (
        replace(source, project_source=replace(source.project_source, commit="c" * 40)),
        replace(source, project_source=replace(
            source.project_source,
            location=source.project_source.location.parse("https://github.com/other/product.git"),
        )),
        replace(source, project_source=replace(source.project_source, dirty=True)),
        replace(source, engine_source=replace(
            source.engine_source, commit="c" * 40, gitlink_commit="c" * 40
        )),
    )
    for inspected in mutations:
        with pytest.raises(SourceLockError, match="current local checkout"):
            _finalizer(source, local=inspected).finalize(
                source, context=_context(tmp_path), deployment=_deployment(),audience_ref="project/run-1"
            )


def test_finalizer_rejects_unbound_pushed_or_deployment_evidence(tmp_path: Path) -> None:
    source = _source()
    with pytest.raises(SourceLockError, match="does not bind both"):
        ModalDualCloneSourceFinalizer(
            LocalPort(source), EvidencePort(engine_commit="c" * 40), DeploymentPort(),authenticator=Auth(),replay=Replay(),clock=lambda:"2026-08-25T12:03:00Z",source_issuer_ref="authenticated-verifier",deployment_issuer_ref="modal-verifier",source_key_ref="source-key",deployment_key_ref="deployment-key"
        ).finalize(source, context=_context(tmp_path), deployment=_deployment(),audience_ref="project/run-1")
    with pytest.raises(SourceLockError, match="does not bind the selection"):
        _finalizer(source, drift=True).finalize(
            source, context=_context(tmp_path), deployment=_deployment(),audience_ref="project/run-1"
        )


def test_finalizer_rejects_wrong_but_authentic_source_and_deployment_keys(tmp_path: Path) -> None:
    source=_source()
    for value in (_finalizer(source,source_key="other-key"),_finalizer(source,deployment_key="other-key")):
        with pytest.raises(SourceLockError,match="issuer, key, or audience"):
            value.finalize(source,context=_context(tmp_path),deployment=_deployment(),audience_ref="project/run-1")


def test_finalizer_rejects_symlink_or_reparse_roots(tmp_path: Path) -> None:
    source = _source()
    real_project = tmp_path / "real"
    (real_project / "vendor" / "engine").mkdir(parents=True)
    linked_project = tmp_path / "linked"
    try:
        linked_project.symlink_to(real_project, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    context = ProjectContext.host(
        project_root=linked_project, engine_root=linked_project / "vendor" / "engine"
    )
    with pytest.raises(SourceLockError, match="reparse"):
        _finalizer(source).finalize(source, context=context, deployment=_deployment(),audience_ref="project/run-1")
