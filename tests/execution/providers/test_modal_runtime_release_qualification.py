"""Provider-free CPU qualification tests for the Modal runtime release lane."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.canonical import canonical_bytes
import tuner.execution.providers.modal.runtime_release_qualification as qualification
from tuner.execution.providers.modal.packaged_staging import ModalPackagedStageReceipt
from tuner.execution.providers.modal.runtime_release_qualification import (
    CURRENT_LAYOUT_LIMITATION,
    QUALIFICATION_HMAC_ENV_KEY,
    QUALIFICATION_HMAC_KEY_REF,
    ModalRuntimeQualificationHmacAuthenticator,
    ModalRuntimeReleaseFixtureReceiptV1,
    ModalRuntimeReleaseQualificationDispatchV1,
    ModalRuntimeReleaseQualificationPolicyV1,
    ModalRuntimeReleaseQualificationReceiptV1,
    ModalRuntimeReleaseQualificationRoots,
    ModalRuntimeReleaseQualificationWorker,
    build_modal_runtime_release_qualification_dispatch,
    parse_modal_runtime_release_qualification_dispatch,
)
from tuner.execution.providers.modal.runtime_release_deployment import (
    ModalRuntimeReleaseDeploymentFactsV1,
    ModalRuntimeReleaseDeploymentObservationV1,
    ModalRuntimeReleaseFunctionFactV1,
    ModalRuntimeReleaseFunctionSpecV1,
    ModalRuntimeReleaseSecretFactV1,
    ModalRuntimeReleaseSecretSpecV1,
    ModalRuntimeReleaseVolumeFactV1,
    ModalRuntimeReleaseVolumeSpecV1,
)
from tuner.runtime.releases import ProviderRuntimeBindingV1
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA

from tests.execution.providers.test_modal_packaged_binding import _release_and_execution


class Facts:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.artifact_volume_id = wrapped.artifact_volume_id
        self.control_volume_id = wrapped.control_volume_id
        self.app_name = wrapped.app_name
        self.functions = (
            SimpleNamespace(
                spec=SimpleNamespace(role="self_check", name=wrapped.function_name),
                function_id="fu-1",
            ),
        )

    @property
    def canonical_bytes(self): return self.wrapped.canonical_bytes
    @property
    def facts_digest(self): return self.wrapped.facts_digest
    def validate_release(self, release): return self.wrapped.validate_release(release)
    def build_provider_binding(self, release): return self.wrapped.build_provider_binding(release)
    def __eq__(self, other): return type(other) is Facts and self.wrapped == other.wrapped
    @classmethod
    def parse(cls, raw):
        from tuner.execution.providers.modal.packaged_binding import ModalPackagedRuntimeFactsV1
        return cls(ModalPackagedRuntimeFactsV1.parse(raw))


class Auth:
    def __init__(self): self.signed = []; self.verified = []
    def sign(self, purpose, payload, key_ref):
        self.signed.append((purpose, payload, key_ref)); return b"tag"
    def verify(self, purpose, payload, tag, key_ref):
        self.verified.append((purpose, payload, tag, key_ref)); return tag == b"tag"


class Observer:
    def __init__(self, value): self.value = value; self.calls = []
    def observe(self, expected): self.calls.append(expected); return self.value


@pytest.fixture(autouse=True)
def facts_type(monkeypatch):
    monkeypatch.setattr(qualification, "_qualification_facts_type", lambda: Facts)


def _case():
    release, packaged, provider, _, _ = _release_and_execution(LOCAL_CPU_DATA)
    facts = Facts(packaged)
    fixture = ModalRuntimeReleaseFixtureReceiptV1.create(
        effect_id="qualify-runtime", artifact_volume_id=facts.artifact_volume_id,
    )
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        "qualify-runtime", release, provider, facts, fixture,
        ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF,
    )
    return dispatch, facts


def _real_facts(release, client_binding, *, mount_root: Path | None = None):
    control_mount = str((mount_root / "control").resolve()) if mount_root else "/workspace/control"
    artifact_mount = str((mount_root / "artifacts").resolve()) if mount_root else "/workspace/artifacts"
    volumes = (
        ModalRuntimeReleaseVolumeFactV1(
            ModalRuntimeReleaseVolumeSpecV1(
                "control", "control-name", control_mount,
            ), "vo-control",
        ),
        ModalRuntimeReleaseVolumeFactV1(
            ModalRuntimeReleaseVolumeSpecV1(
                "artifacts", "artifact-name", artifact_mount,
            ), "vo-artifact",
        ),
    )
    functions = (
        ModalRuntimeReleaseFunctionFactV1(
            ModalRuntimeReleaseFunctionSpecV1(
                "training", "train", "tuner.execution.providers.modal.packaged_worker",
                "ModalPackagedWorker", ("control", "artifacts"), (), 1000,
                1024, 900, "A10", False,
            ), "fu-training",
        ),
        ModalRuntimeReleaseFunctionFactV1(
            ModalRuntimeReleaseFunctionSpecV1(
                "self_check", "self-check",
                "tuner.runtime.runtime_release_modal_self_check",
                "run_runtime_release_self_check", ("control", "artifacts"),
                ("qualification-auth",),
                1000, 512, 120, None, True,
                restrict_modal_access=False,
            ), "fu-selfcheck",
        ),
    )
    return ModalRuntimeReleaseDeploymentFactsV1(
        "d" * 64, release.manifest_digest, client_binding,
        "runtime-app", "ap-runtime", 7, release.image_ref,
        release.image_digest, "im-runtime", functions, volumes,
        (ModalRuntimeReleaseSecretFactV1(
            ModalRuntimeReleaseSecretSpecV1(
                "qualification-auth", (QUALIFICATION_HMAC_ENV_KEY,),
            ),
            "st-qualification",
        ),),
        release.python_implementation, release.python_version,
        release.python_executable, release.python_executable_digest,
        release.package_digest, release.installed_distributions_digest,
        release.worker_entrypoint, release.worker_closure_digest,
    )


def test_dispatch_binds_release_deployment_fixture_and_fixed_cpu_policy() -> None:
    dispatch, facts = _case()
    document = dispatch.unsigned_dict()
    assert document["deployment_facts_digest"] == facts.facts_digest
    assert document["runtime_release"]["manifest_digest"] == dispatch.runtime_release.manifest_digest
    assert document["fixture"] == dispatch.fixture.to_dict()
    assert document["policy"] == {
        "cpu": 1, "memory_mib": 512, "timeout_seconds": 120,
        "network_access": False, "gpu": False,
    }
    assert document["limitations"] == [CURRENT_LAYOUT_LIMITATION]


def test_dispatch_accepts_exact_release_deployment_facts(monkeypatch) -> None:
    release, packaged, _, _, _ = _release_and_execution(LOCAL_CPU_DATA)
    facts = _real_facts(release, packaged.client_binding)
    provider = ProviderRuntimeBindingV1.build(
        provider_ref="modal", runtime_release=release,
        provider_facts_schema=facts.schema_version,
        provider_facts_digest=facts.facts_digest,
    )
    monkeypatch.setattr(
        qualification, "_qualification_facts_type",
        lambda: ModalRuntimeReleaseDeploymentFactsV1,
    )
    fixture = ModalRuntimeReleaseFixtureReceiptV1.create(
        effect_id="qualify-real", artifact_volume_id="vo-artifact",
    )
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        "qualify-real", release, provider, facts, fixture,
        ModalRuntimeReleaseQualificationPolicyV1(), QUALIFICATION_HMAC_KEY_REF,
    )
    auth = Auth()
    assert parse_modal_runtime_release_qualification_dispatch(
        build_modal_runtime_release_qualification_dispatch(dispatch, auth), auth,
    ) == dispatch
    current = ModalRuntimeReleaseDeploymentObservationV1(
        facts.app_name, facts.client_binding.environment_ref, True,
        facts.app_id, "", facts.generation,
        tuple(sorted((item.spec.name, item.function_id) for item in facts.functions)),
    )
    digest = qualification._observe_current_layout(Observer(current), facts)
    assert len(digest) == 64
    # Current observation has no image field; acknowledged facts remain the
    # only Function→Image relationship evidence.
    assert "image" not in current.to_dict()


@pytest.mark.parametrize("fault", ("missing", "wrong_key"))
def test_dispatch_rejects_missing_or_wrong_qualification_auth_secret(
    monkeypatch, fault: str,
) -> None:
    release, packaged, _, _, _ = _release_and_execution(LOCAL_CPU_DATA)
    facts = _real_facts(release, packaged.client_binding)
    secrets = () if fault == "missing" else (
        ModalRuntimeReleaseSecretFactV1(
            ModalRuntimeReleaseSecretSpecV1(
                "qualification-auth", ("ANOTHER_KEY",),
            ),
            "st-qualification",
        ),
    )
    changed = replace(facts, secrets=secrets)
    provider = ProviderRuntimeBindingV1.build(
        provider_ref="modal", runtime_release=release,
        provider_facts_schema=changed.schema_version,
        provider_facts_digest=changed.facts_digest,
    )
    monkeypatch.setattr(
        qualification, "_qualification_facts_type",
        lambda: ModalRuntimeReleaseDeploymentFactsV1,
    )
    with pytest.raises(ValueError, match="auth Secret"):
        ModalRuntimeReleaseQualificationDispatchV1(
            "qualify-secret", release, provider, changed,
            ModalRuntimeReleaseFixtureReceiptV1.create(
                effect_id="qualify-secret", artifact_volume_id="vo-artifact",
            ),
            ModalRuntimeReleaseQualificationPolicyV1(),
            QUALIFICATION_HMAC_KEY_REF,
        )


def test_signed_dispatch_round_trips_canonically_and_contains_no_secret_or_training_input() -> None:
    dispatch, _ = _case(); auth = Auth()
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    assert parse_modal_runtime_release_qualification_dispatch(raw, auth) == dispatch
    lowered = raw.lower()
    assert b"token" not in lowered and b"password" not in lowered
    assert b"training_lineage" not in lowered and b"final_model" not in lowered
    assert len(auth.signed) == len(auth.verified) == 1


def test_hmac_is_domain_separated_and_key_reference_is_fixed() -> None:
    first = ModalRuntimeQualificationHmacAuthenticator(b"a" * 32)
    second = ModalRuntimeQualificationHmacAuthenticator(b"b" * 32)
    tag = first.sign("dispatch-purpose", b"payload", QUALIFICATION_HMAC_KEY_REF)
    assert first.verify(
        "dispatch-purpose", b"payload", tag, QUALIFICATION_HMAC_KEY_REF,
    ) is True
    assert first.verify(
        "receipt-purpose", b"payload", tag, QUALIFICATION_HMAC_KEY_REF,
    ) is False
    assert second.verify(
        "dispatch-purpose", b"payload", tag, QUALIFICATION_HMAC_KEY_REF,
    ) is False
    with pytest.raises(ValueError, match="key reference"):
        first.sign("dispatch-purpose", b"payload", "caller-selected")
    dispatch, _ = _case()
    with pytest.raises(ValueError, match="key reference"):
        replace(dispatch, key_ref="caller-selected")


@pytest.mark.parametrize("fault", ("release", "facts", "fixture", "policy"))
def test_cross_binding_or_non_cpu_policy_is_rejected(fault: str) -> None:
    dispatch, facts = _case()
    with pytest.raises((TypeError, ValueError)):
        if fault == "release":
            replace(dispatch.runtime_release, package_digest="0" * 64)
        elif fault == "facts":
            changed = Facts(replace(facts.wrapped, deployment_generation=8))
            ModalRuntimeReleaseQualificationDispatchV1(
                dispatch.effect_id, dispatch.runtime_release,
                dispatch.provider_binding, changed, dispatch.fixture,
                dispatch.policy, dispatch.key_ref,
            )
        elif fault == "fixture":
            replace(dispatch.fixture, artifact_volume_id="vo-other")
        else:
            ModalRuntimeReleaseQualificationPolicyV1(gpu=True)


def test_worker_runs_one_self_check_and_commits_artifact_before_control(monkeypatch, tmp_path: Path) -> None:
    dispatch, facts = _case(); auth = Auth()
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    roots = ModalRuntimeReleaseQualificationRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
    )
    roots.control.mkdir(); roots.artifacts.mkdir()
    staged = roots.artifacts / dispatch.fixture.path
    staged.parent.mkdir(parents=True); staged.write_bytes(LOCAL_CPU_DATA)
    calls = []
    monkeypatch.setattr(
        "tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda release: calls.append(release) or {
            "schema_version": "synaptic-installed-child-cpu/v1",
            "training_executed": False, "gpu_qualified": False,
        },
    )
    events = []
    worker = ModalRuntimeReleaseQualificationWorker(
        expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-qualification",
        roots=roots,
    )
    result = worker(
        raw, commit_artifacts=lambda: events.append("artifact"),
        commit_control=lambda: events.append("control"),
    )
    assert result["status_code"] == "completed"
    assert len(calls) == 1 and events == ["artifact", "control"]
    receipt = ModalRuntimeReleaseQualificationReceiptV1.parse(next(
        roots.control.rglob("receipt.json")
    ).read_bytes())
    assert receipt.provider_call_id == "fc-qualification"
    assert receipt.deployment_facts_digest == facts.facts_digest
    output = next(roots.artifacts.rglob("evidence.json")).read_bytes()
    assert hashlib.sha256(output).hexdigest() == receipt.output_sha256


def test_worker_fails_before_self_check_when_fixture_or_current_layout_differs(monkeypatch, tmp_path: Path) -> None:
    dispatch, facts = _case(); auth = Auth()
    raw = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    roots = ModalRuntimeReleaseQualificationRoots(
        (tmp_path / "control").resolve(), (tmp_path / "artifacts").resolve(),
    )
    roots.control.mkdir(); roots.artifacts.mkdir()
    staged = roots.artifacts / dispatch.fixture.path
    staged.parent.mkdir(parents=True); staged.write_bytes(b"substituted")
    called = []
    monkeypatch.setattr(
        "tuner.runtime.packaged_training_worker.qualify_installed_child",
        lambda release: called.append(release),
    )
    worker = ModalRuntimeReleaseQualificationWorker(
        expected_facts=facts, verifier=auth, signer=auth,
        observer=Observer(facts), call_id_provider=lambda: "fc-qualification",
        roots=roots,
    )
    assert worker(raw, commit_artifacts=lambda: None, commit_control=lambda: None)["status_code"] == "failed"
    assert called == []


def test_training_receipt_and_qualification_receipt_are_mutually_rejected() -> None:
    dispatch, _ = _case()
    training = ModalPackagedStageReceipt(
        dispatch.effect_id, "1" * 64, dispatch.fixture.artifact_volume_id,
        operation_path := __import__(
            "tuner.execution.providers.modal.contracts", fromlist=["operation_path"],
        ).operation_path(dispatch.effect_id, "input", "prepared", "2" * 64, "payload.bin"),
        1, "2" * 64,
        __import__(
            "tuner.execution.providers.modal.contracts", fromlist=["provider_entry_identity"],
        ).provider_entry_identity(dispatch.fixture.artifact_volume_id, operation_path, 1),
    )
    with pytest.raises(ValueError):
        ModalRuntimeReleaseQualificationReceiptV1.parse(training.canonical_bytes)
