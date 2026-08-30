from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

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
from tuner.project.execution_source import validate_source_lock_provenance_v1
from tuner.execution.evidence import ReplayDisposition


def _source(
    mode: str = "superproject", path: str = "vendor/engine",
    configuration: dict[str, object] | None = None,
) -> SourceLock:
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
        "project": {}, "configuration": configuration or {}, "plugins": [], "inputs": [],
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
    def __init__(self, source: SourceLock, callback=None) -> None:
        self.source = source
        self.calls = 0
        self.callback = callback

    def inspect(self, *, context: ProjectContext) -> SourceLock:
        self.calls += 1
        if self.callback:self.callback()
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
            source_lock_binding=source_lock.binding,
            issuer_ref="authenticated-verifier", evidence_ref="proof-1",
            audience_ref="project/run-1", challenge_nonce="source-nonce",
            verified_at="2026-08-25T12:01:00Z", expires_at="2026-08-25T12:10:00Z",
            key_ref=self.key_ref, tag_base64="dGFn", attestation_digest="e" * 64,
        )
        self.last=replace(value,attestation_digest=hashlib.sha256(value.authenticated_payload).hexdigest())
        return self.last


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
        self.last=VerifiedModalDeploymentIdentityV1(**fields,tag_base64="dGFn",attestation_digest=hashlib.sha256(payload).hexdigest())
        return self.last


class Auth:
    def __init__(self, callback=None):self.callback=callback;self.purposes=[]
    def sign(self,purpose,payload,key_ref):return b"tag"
    def verify(self,purpose,payload,tag,key_ref):
        self.purposes.append(purpose)
        if self.callback:self.callback()
        return tag==b"tag"


class Replay:
    def __init__(self,callback=None):self.values={};self.callback=callback
    def admit(self,**value):
        if self.callback:self.callback()
        key=(value["purpose"],value["challenge_nonce"]);prior=self.values.get(key)
        if prior is None:self.values[key]=value;return ReplayDisposition.ADMITTED
        return ReplayDisposition.IDEMPOTENT if prior==value else ReplayDisposition.COLLISION


def _context(tmp_path: Path, path: str = "vendor/engine") -> ProjectContext:
    project = tmp_path / "product"
    engine = project.joinpath(*path.split("/"))
    engine.mkdir(parents=True, exist_ok=True)
    return ProjectContext.host(engine_root=engine, project_root=project)


def _finalizer(source: SourceLock, *, local: SourceLock | None = None, local_port=None, drift: bool = False,source_key: str="source-key",deployment_key: str="deployment-key",auth=None,pushed=None,deployments=None,replay=None):
    return ModalDualCloneSourceFinalizer(
        local_port or LocalPort(source if local is None else local), pushed or EvidencePort(key_ref=source_key), deployments or DeploymentPort(drift=drift,key_ref=deployment_key),
        authenticator=auth or Auth(),replay=replay or Replay(),clock=lambda:"2026-08-25T12:03:00Z",
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
    assert finalized.writable_capability_root == "/workspace/run"
    assert finalized.to_dict()["runtime"]["capability_roots"] == {
        "writable": "/workspace/run"
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
    assert finalized.source_evidence.source_lock_binding == source.binding


def test_complete_source_lock_binding_changes_execution_and_plan_source_identity(tmp_path: Path) -> None:
    first = _source(configuration={"training_input_digest": "1" * 64})
    second = _source(configuration={"training_input_digest": "2" * 64})
    first_source = _finalizer(first).finalize(
        first, context=_context(tmp_path / "first"), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    second_source = _finalizer(second).finalize(
        second, context=_context(tmp_path / "second"), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    assert first_source.source_evidence.source_lock_binding != second_source.source_evidence.source_lock_binding
    assert first_source.fingerprint != second_source.fingerprint


def test_provider_neutral_training_provenance_view_binds_all_five_keys(tmp_path: Path) -> None:
    projection = {
        "training_input_digest": "1" * 64,
        "training_contract_identity_digest": "2" * 64,
        "training_source_sha256": "3" * 64,
        "training_ingress_digest": "4" * 64,
        "provider_policy_digest": "5" * 64,
    }
    lock = _source(configuration=projection)
    execution = _finalizer(lock).finalize(
        lock, context=_context(tmp_path), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    view = validate_source_lock_provenance_v1(execution, lock, projection)
    assert view.binding == lock.binding
    for key in projection:
        changed = dict(projection)
        changed[key] = "f" * 64
        with pytest.raises(SourceLockError):
            validate_source_lock_provenance_v1(execution, lock, changed)


def test_execution_source_rejects_missing_or_substituted_source_lock_binding(tmp_path: Path) -> None:
    source = _source(configuration={"training_input_digest": "1" * 64})
    execution = _finalizer(source).finalize(
        source, context=_context(tmp_path), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    missing = execution.to_dict()
    missing["source_evidence"].pop("source_lock_binding")
    with pytest.raises(SourceLockError):
        ExecutionSourceV1.from_dict(missing)
    changed = execution.to_dict()
    changed["source_evidence"]["source_lock_binding"]["source_lock_digest"] = "f" * 64
    substituted = ExecutionSourceV1.from_dict(changed)
    assert substituted.fingerprint != execution.fingerprint


def test_finalizer_uses_provider_neutral_purpose_and_rejects_admission_mutation(tmp_path: Path) -> None:
    source = _source(configuration={"training_input_digest": "1" * 64})
    auth = Auth()
    _finalizer(source,auth=auth).finalize(
        source,context=_context(tmp_path / "purpose"),deployment=_deployment(),
        audience_ref="project/run-1",
    )
    assert auth.purposes[0] == "source-lock-evidence/v1"

    source = _source(configuration={"training_input_digest": "1" * 64})
    auth = Auth(lambda:object.__setattr__(source,"configuration",{"changed":True}))
    with pytest.raises(SourceLockError,match="evidence admission failed"):
        _finalizer(source,auth=auth).finalize(
            source,context=_context(tmp_path / "mutation"),deployment=_deployment(),
            audience_ref="project/run-1",
        )


def test_finalizer_rejects_source_evidence_mutation_after_auth_or_replay(tmp_path: Path) -> None:
    for stage in ("authenticate", "replay"):
        source=_source(configuration={"training_input_digest":"1"*64})
        pushed=EvidencePort()
        mutate=lambda:object.__setattr__(pushed.last,"evidence_ref","changed-proof")
        auth=Auth(mutate if stage=="authenticate" else None)
        replay=Replay(mutate if stage=="replay" else None)
        with pytest.raises(SourceLockError,match="evidence admission failed"):
            _finalizer(source,pushed=pushed,auth=auth,replay=replay).finalize(
                source,context=_context(tmp_path/stage),deployment=_deployment(),
                audience_ref="project/run-1",
            )


def test_finalizer_rejects_pushed_and_deployment_callback_lock_mutation(tmp_path: Path) -> None:
    source=_source(configuration={"training_input_digest":"1"*64})

    local_port=LocalPort(
        source,
        lambda:object.__setattr__(source,"configuration",{"changed":True}),
    )
    with pytest.raises(SourceLockError,match="source lock changed"):
        _finalizer(source,local_port=local_port).finalize(
            source,context=_context(tmp_path/"local"),deployment=_deployment(),
            audience_ref="project/run-1",
        )

    source=_source(configuration={"training_input_digest":"1"*64})

    class MutatingPushed(EvidencePort):
        def verify(self, source_lock):
            result=super().verify(source_lock)
            object.__setattr__(source_lock,"configuration",{"changed":True})
            return result

    with pytest.raises(SourceLockError,match="source lock changed"):
        _finalizer(source,pushed=MutatingPushed()).finalize(
            source,context=_context(tmp_path/"pushed"),deployment=_deployment(),
            audience_ref="project/run-1",
        )

    source=_source(configuration={"training_input_digest":"1"*64})

    class MutatingDeployment(DeploymentPort):
        def verify(self, selection):
            result=super().verify(selection)
            object.__setattr__(selection,"accelerator","H100")
            return result

    with pytest.raises(SourceLockError,match="selection changed"):
        _finalizer(source,deployments=MutatingDeployment()).finalize(
            source,context=_context(tmp_path/"deployment"),deployment=_deployment(),
            audience_ref="project/run-1",
        )


def test_finalizer_rechecks_source_evidence_after_deployment_callback(tmp_path: Path) -> None:
    source=_source(configuration={"training_input_digest":"1"*64})
    pushed=EvidencePort()

    class MutatingDeployment(DeploymentPort):
        def verify(self, selection):
            result=super().verify(selection)
            object.__setattr__(pushed.last,"attestation_digest","f"*64)
            return result

    with pytest.raises(SourceLockError,match="source evidence"):
        _finalizer(source,pushed=pushed,deployments=MutatingDeployment()).finalize(
            source,context=_context(tmp_path),deployment=_deployment(),
            audience_ref="project/run-1",
        )


def test_finalizer_rejects_deployment_evidence_mutation_after_auth_or_replay(tmp_path: Path) -> None:
    for stage in ("authenticate", "replay"):
        source=_source();deployments=DeploymentPort()

        class DeploymentAuth(Auth):
            def verify(self,purpose,payload,tag,key_ref):
                result=super().verify(purpose,payload,tag,key_ref)
                if stage=="authenticate" and purpose=="modal-deployment-evidence/v1":
                    object.__setattr__(deployments.last,"evidence_ref","changed-proof")
                return result

        calls={"count":0}

        def replay_mutation():
            calls["count"]+=1
            if stage=="replay" and calls["count"]==2:
                object.__setattr__(deployments.last,"evidence_ref","changed-proof")

        with pytest.raises(SourceLockError,match="evidence admission failed"):
            _finalizer(
                source,deployments=deployments,auth=DeploymentAuth(),
                replay=Replay(replay_mutation),
            ).finalize(
                source,context=_context(tmp_path/stage),deployment=_deployment(),
                audience_ref="project/run-1",
            )


def test_execution_source_rejects_evidence_and_git_source_subclasses(tmp_path: Path) -> None:
    source=_source()
    execution=_finalizer(source).finalize(
        source,context=_context(tmp_path),deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source

    class EvidenceSubclass(AuthenticatedSourceEvidenceV1):
        def binds_sources(self,*_args):
            return True

    forged=EvidenceSubclass.from_dict(execution.source_evidence.to_dict())
    with pytest.raises(TypeError,match="exact Authenticated"):
        replace(execution,source_evidence=forged)

    class GitSourceSubclass(type(execution.project_source)):
        pass

    project=execution.project_source
    forged_project=GitSourceSubclass(
        project.location,project.commit,project.branch,project.dirty,project.pushed,
        project.submodule_path,project.gitlink_commit,
    )
    with pytest.raises(TypeError,match="GitSource"):
        replace(execution,project_source=forged_project)


def test_new_evidence_parsers_reject_mapping_proxies(tmp_path: Path) -> None:
    source=_source()
    evidence=EvidencePort().verify(source)
    with pytest.raises(SourceLockError):
        AuthenticatedSourceEvidenceV1.from_dict(MappingProxyType(evidence.to_dict()))
    execution=_finalizer(source).finalize(
        source,context=_context(tmp_path),deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    with pytest.raises(SourceLockError):
        ExecutionSourceV1.from_dict(MappingProxyType(execution.to_dict()))
    class Text(str):
        pass
    evidence_value=evidence.to_dict();evidence_value["issuer_ref"]=Text("issuer")
    with pytest.raises(SourceLockError):
        AuthenticatedSourceEvidenceV1.from_dict(evidence_value)
    execution_value=execution.to_dict()
    execution_value["runtime"]["environment"]["variables"]["PATH"]=Text("/bin")
    with pytest.raises(SourceLockError):
        ExecutionSourceV1.from_dict(execution_value)


class _HostileExecutionFieldName(str):
    armed = False
    calls = 0

    def __hash__(self):
        if type(self).armed:
            type(self).calls += 1
            raise RuntimeError("private-execution-field")
        return str.__hash__(self)

    def __eq__(self, other):
        if type(self).armed:
            type(self).calls += 1
            raise RuntimeError("private-execution-field")
        return str.__eq__(self, other)


def _hostile_execution_key(value: dict[str, object], name: str) -> None:
    original = value.pop(name)
    value[_HostileExecutionFieldName(name)] = original


@pytest.mark.parametrize(
    "path",
    [
        ("evidence",), ("evidence", "project"), ("evidence", "engine"),
        ("evidence", "binding"), ("execution",), ("execution", "topology"),
        ("execution", "sources"), ("execution", "project"),
        ("execution", "engine"), ("execution", "runtime"),
        ("execution", "roots"), ("execution", "capability_roots"),
        ("execution", "interpreter"), ("execution", "environment"),
        ("execution", "variables"),
    ],
)
def test_execution_parser_field_inventory_rejects_hostile_string_subclass_without_callbacks(
    tmp_path: Path, path: tuple[str, ...]
) -> None:
    source = _source()
    execution = _finalizer(source).finalize(
        source, context=_context(tmp_path), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    if path[0] == "evidence":
        value = execution.source_evidence.to_dict()
        targets = {
            1: (value, "schema_version"),
            "project": (value["project"], "url"),
            "engine": (value["engine"], "url"),
            "binding": (value["source_lock_binding"], "schema_version"),
        }
        target, key = targets[path[1] if len(path) > 1 else 1]
        _hostile_execution_key(target, key)
        parse = lambda: AuthenticatedSourceEvidenceV1.from_dict(value)
    else:
        value = execution.to_dict()
        targets = {
            1: (value, "schema_version"),
            "topology": (value["topology"], "provenance_mode"),
            "sources": (value["sources"], "project"),
            "project": (value["sources"]["project"], "url"),
            "engine": (value["sources"]["engine"], "url"),
            "runtime": (value["runtime"], "schema_version"),
            "roots": (value["runtime"]["roots"], "engine"),
            "capability_roots": (value["runtime"]["capability_roots"], "writable"),
            "interpreter": (value["runtime"]["interpreter"], "implementation"),
            "environment": (value["runtime"]["environment"], "clear_inherited"),
            "variables": (value["runtime"]["environment"]["variables"], "PATH"),
        }
        target, key = targets[path[1] if len(path) > 1 else 1]
        _hostile_execution_key(target, key)
        parse = lambda: ExecutionSourceV1.from_dict(value)
    _HostileExecutionFieldName.calls = 0
    _HostileExecutionFieldName.armed = True
    try:
        with pytest.raises(SourceLockError) as caught:
            parse()
    finally:
        _HostileExecutionFieldName.armed = False
    assert _HostileExecutionFieldName.calls == 0
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private-execution-field" not in str(caught.value)


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ((), "schema_version"),
        (("topology",), "provenance_mode"),
        (("topology",), "execution_mode"),
        (("runtime",), "schema_version"),
        (("runtime", "interpreter"), "implementation"),
        (("runtime", "interpreter"), "version"),
    ],
)
def test_execution_parser_rejects_subclassed_canonical_identity_values(
    tmp_path: Path, path: tuple[str, ...], name: str
) -> None:
    class Text(str):
        pass

    source = _source()
    execution = _finalizer(source).finalize(
        source, context=_context(tmp_path), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    value = execution.to_dict()
    target = value
    for component in path:
        target = target[component]
    target[name] = Text(target[name])
    with pytest.raises(SourceLockError):
        ExecutionSourceV1.from_dict(value)


def test_execution_constructor_rejects_subclassed_schema_and_interpreter_identity(
    tmp_path: Path,
) -> None:
    class Text(str):
        pass

    source = _source()
    execution = _finalizer(source).finalize(
        source, context=_context(tmp_path), deployment=_deployment(),
        audience_ref="project/run-1",
    ).execution_source
    for change in (
        {"schema_version": Text(execution.schema_version)},
        {"python_implementation": Text(execution.python_implementation)},
        {"python_version": Text(execution.python_version)},
    ):
        with pytest.raises(SourceLockError):
            replace(execution, **change)


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

    class WrongBindingEvidence(EvidencePort):
        def verify(self, source_lock):
            value = super().verify(source_lock)
            other = replace(source_lock, configuration={"different": "binding"})
            changed = replace(value, source_lock_binding=other.binding)
            return replace(
                changed,
                attestation_digest=hashlib.sha256(
                    changed.authenticated_payload
                ).hexdigest(),
            )

    with pytest.raises(SourceLockError, match="does not bind both"):
        ModalDualCloneSourceFinalizer(
            LocalPort(source), WrongBindingEvidence(), DeploymentPort(),
            authenticator=Auth(), replay=Replay(),
            clock=lambda:"2026-08-25T12:03:00Z",
            source_issuer_ref="authenticated-verifier",
            deployment_issuer_ref="modal-verifier",
            source_key_ref="source-key", deployment_key_ref="deployment-key",
        ).finalize(
            source, context=_context(tmp_path / "wrong-binding"),
            deployment=_deployment(), audience_ref="project/run-1",
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
