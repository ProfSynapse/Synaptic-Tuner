"""Provider-free host operator and authoritative readback tests."""
from __future__ import annotations

from pathlib import Path

import pytest

import tuner.execution.providers.modal.runtime_release_qualification as qualification
from tuner.cloud.modal_runtime_qualification_operator import (
    ModalRuntimeQualificationOperator,
)
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeReleaseQualificationRoots,
    ModalRuntimeReleaseQualificationWorker,
    build_modal_runtime_release_qualification_dispatch,
)
from tuner.execution.providers.modal.runtime_release_qualification_reader import (
    ModalRuntimeReleaseQualificationReader,
)
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA, local_cpu_result

from tests.execution.providers.test_modal_runtime_release_qualification import (
    Auth, Facts, Observer, _case,
)
from tests.execution.providers.test_modal_sdk154_adapter import (
    FakeFunction, FakeVolume, SDK,
)


class Catalog:
    def __init__(self): self.values = {}; self.publish_calls = 0; self.fail = False
    def resolve(self, digest): return self.values.get(digest)
    def publish_if_absent(self, digest, call_id):
        self.publish_calls += 1
        if self.fail: raise RuntimeError("private catalog failure")
        return self.values.setdefault(digest, call_id) == call_id


@pytest.fixture(autouse=True)
def facts_type(monkeypatch):
    monkeypatch.setattr(qualification, "_qualification_facts_type", lambda: Facts)


def _facade(facts):
    client = object()
    FakeVolume.calls = []
    FakeVolume.registry = {
        "control-name": FakeVolume(facts.control_volume_id),
        "artifact-name": FakeVolume(facts.artifact_volume_id),
    }
    FakeFunction.calls = []; FakeFunction.spawn_calls = []; FakeFunction.fail = False
    facade = ExplicitModal154ReadFacade(
        facts.wrapped.client_binding, sdk=SDK, client=client,
        scope_observer=lambda supplied: (
            facts.wrapped.account_ref, facts.wrapped.workspace_ref,
            facts.wrapped.environment_ref, facts.wrapped.client_ref,
        ) if supplied is client else (),
        deployment_observer=lambda **_: None,
        volume_names={
            facts.control_volume_id: "control-name",
            facts.artifact_volume_id: "artifact-name",
        },
    )
    return facade


def test_operator_stages_fixed_fixture_spawns_once_and_replay_uses_catalog() -> None:
    _, facts = _case(); facade = _facade(facts); auth = Auth(); catalog = Catalog()
    operator = ModalRuntimeQualificationOperator(
        facade=facade, deployment_observer=Observer(facts), verifier=auth,
        call_catalog=catalog,
    )
    receipt = operator.stage_fixture_once(
        effect_id="qualify-runtime", deployment_facts=facts,
    )
    dispatch, _ = _case()
    assert dispatch.fixture == receipt
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    first = operator.submit_once(raw, expected_facts=facts)
    replay = operator.submit_once(raw, expected_facts=facts)
    assert first == replay and first.provider_call_id == "fc-1"
    assert len(FakeFunction.spawn_calls) == 1
    assert catalog.publish_calls == 1


def test_catalog_failure_after_spawn_is_indeterminate_and_never_auto_replayed() -> None:
    dispatch, facts = _case(); facade = _facade(facts); auth = Auth(); catalog = Catalog()
    catalog.fail = True
    operator = ModalRuntimeQualificationOperator(
        facade=facade, deployment_observer=Observer(facts), verifier=auth,
        call_catalog=catalog,
    )
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    outcome = operator.submit_once(raw, expected_facts=facts)
    assert outcome.disposition == "indeterminate"
    assert len(FakeFunction.spawn_calls) == 1
    assert operator.reconcile(dispatch.dispatch_digest).disposition == "indeterminate"
    assert len(FakeFunction.spawn_calls) == 1


def test_reader_authenticates_rehashes_and_relists_one_authoritative_output(monkeypatch, tmp_path: Path) -> None:
    dispatch, facts = _case(); auth = Auth()
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    roots = ModalRuntimeReleaseQualificationRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
    )
    roots.control.mkdir(); roots.artifacts.mkdir()
    staged = roots.artifacts / dispatch.fixture.path
    staged.parent.mkdir(parents=True); staged.write_bytes(LOCAL_CPU_DATA)
    monkeypatch.setattr(
        "tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda release: local_cpu_result(dispatch.runtime_release, "a" * 64),
    )
    worker = ModalRuntimeReleaseQualificationWorker(
        expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-qualification",
        roots=roots,
    )
    assert worker(raw, commit_artifacts=lambda: None, commit_control=lambda: None)["status_code"] == "completed"
    facade = _facade(facts)
    control = FakeVolume.registry["control-name"]
    artifacts = FakeVolume.registry["artifact-name"]
    for path in roots.control.rglob("*"):
        if path.is_file(): control.files[path.relative_to(roots.control).as_posix()] = path.read_bytes()
    for path in roots.artifacts.rglob("*"):
        if path.is_file() and path.name == "evidence.json":
            artifacts.files[path.relative_to(roots.artifacts).as_posix()] = path.read_bytes()
    reader = ModalRuntimeReleaseQualificationReader(
        facade=facade, deployment_observer=Observer(facts), verifier=auth,
    )
    observed = reader.observe(dispatch, provider_call_id="fc-qualification")
    assert observed.receipt.provider_call_id == "fc-qualification"
    assert b'"training_executed":false' in observed.output
    assert b'"gpu_qualified":false' in observed.output


def test_reader_rejects_changed_output_after_authenticated_receipt(monkeypatch, tmp_path: Path) -> None:
    # Exercise the full setup once, then mutate the authoritative artifact.
    dispatch, facts = _case(); auth = Auth()
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    roots = ModalRuntimeReleaseQualificationRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
    )
    roots.control.mkdir(); roots.artifacts.mkdir()
    staged = roots.artifacts / dispatch.fixture.path
    staged.parent.mkdir(parents=True); staged.write_bytes(LOCAL_CPU_DATA)
    monkeypatch.setattr(
        "tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda release: local_cpu_result(dispatch.runtime_release, "a" * 64),
    )
    worker = ModalRuntimeReleaseQualificationWorker(
        expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-qualification",
        roots=roots,
    )
    assert worker(raw, commit_artifacts=lambda: None, commit_control=lambda: None)["status_code"] == "completed"
    facade = _facade(facts)
    for path in roots.control.rglob("*"):
        if path.is_file():
            FakeVolume.registry["control-name"].files[path.relative_to(roots.control).as_posix()] = path.read_bytes()
    output = next(roots.artifacts.rglob("evidence.json"))
    relative = output.relative_to(roots.artifacts).as_posix()
    FakeVolume.registry["artifact-name"].files[relative] = b"substituted"
    reader = ModalRuntimeReleaseQualificationReader(
        facade=facade, deployment_observer=Observer(facts), verifier=auth,
    )
    with pytest.raises(ValueError, match="inventory|content"):
        reader.observe(dispatch, provider_call_id="fc-qualification")
