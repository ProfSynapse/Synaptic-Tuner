"""Provider-free qualification of the protected Modal release authority."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuner.cloud.modal_runtime_release_operator import (
    InstalledModalRuntimeReleaseEntrypoints,
    LocalModalRuntimeReleaseState,
    ModalRuntimeReleaseOperator,
    ModalRuntimeReleaseOperatorError,
    ModalRuntimeReleaseQualificationController,
    run_modal_runtime_release_cli_action,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.runtime_release_deployment import (
    ModalRuntimeReleaseDeployer,
)
from tuner.cloud.modal_runtime_qualification_operator import ModalRuntimeQualificationOutcome
from tuner.execution.providers.modal.runtime_release_qualification import (
    ModalRuntimeQualificationHmacAuthenticator,
    ModalRuntimeReleaseFixtureReceiptV1,
    build_modal_runtime_release_qualification_dispatch,
    parse_modal_runtime_release_qualification_dispatch,
)

from tests.execution.providers.test_modal_runtime_release_deployment import (
    FakeApp,
    Reader,
    SDK,
    _observation,
    _plan,
    packaged_training_entry,
)
from tuner.runtime.runtime_release_modal_self_check import run_runtime_release_self_check
from tuner.execution.providers.modal.binding import ModalClientBinding


class MemoryState:
    def __init__(self):
        self.values: dict[tuple[str, str], bytes] = {}

    def publish_if_absent(self, kind: str, release_ref: str, payload: bytes) -> bool:
        key = kind, release_ref
        if key in self.values:
            return False
        self.values[key] = bytes(payload)
        return True

    def resolve(self, kind: str, release_ref: str) -> bytes | None:
        value = self.values.get((kind, release_ref))
        return None if value is None else bytes(value)


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.value


class Entrypoints:
    def __init__(self):
        self.calls = 0

    def resolve(self, plan):
        self.calls += 1
        return {
            "training": packaged_training_entry,
            "self_check": run_runtime_release_self_check,
        }


def _operator(reader: Reader, state=None):
    client = object()
    deployer = ModalRuntimeReleaseDeployer(
        sdk=SDK, client=client,
        client_binding=ModalClientBinding(
            "workspace", "workspace", "production", "release-client", "1.5.4",
        ),
        reader=reader,
    )
    sources = Entrypoints()
    return ModalRuntimeReleaseOperator(
        state=state or MemoryState(), deployer=deployer,
        entrypoints=sources, clock=Clock(),
    ), sources


@pytest.fixture(autouse=True)
def _reset_app():
    FakeApp.instances.clear()
    FakeApp.fail_deploy = False


def _approve(operator: ModalRuntimeReleaseOperator, ref: str):
    return operator.approve(
        ref,
        authorization_reference="user-approved-release-20260922",
        issued_at="2026-09-22T11:00:00Z",
        expires_at="2026-09-22T13:00:00Z",
    )


def test_full_release_flow_claims_before_one_deploy_and_mints_binding_only_after_ack() -> None:
    plan = _plan()
    current = _observation(1)
    state = MemoryState()
    operator, entrypoints = _operator(
        Reader([None, None, current, current, current]), state,
    )

    preflight = operator.preflight(plan.canonical_bytes)
    assert preflight["ready"] is True
    approval = _approve(operator, plan.deployment_spec_digest)
    assert approval["credentials_included"] is False
    assert "EVIDENCE_KEY" not in json.dumps(approval)

    result = operator.execute(plan.deployment_spec_digest)
    assert result["status"] == "ACKNOWLEDGED"
    assert result["retry_allowed"] is False
    assert result["binding_authorized"] is True
    assert entrypoints.calls == 1
    assert state.resolve("attempt", plan.deployment_spec_digest) is not None
    assert state.resolve("facts", plan.deployment_spec_digest) is not None
    assert len(FakeApp.instances) == 1

    with pytest.raises(ModalRuntimeReleaseOperatorError, match="already_consumed"):
        operator.execute(plan.deployment_spec_digest)
    assert entrypoints.calls == 1
    assert len(FakeApp.instances) == 1

    observed = operator.observe(plan.deployment_spec_digest)
    assert observed["status"] == "CURRENT"
    assert observed["current_state_version_pinned"] is False
    assert observed["function_image_link_readable"] is False
    verified = operator.verify(plan.deployment_spec_digest)
    assert verified["deployment_spec_digest"] == plan.deployment_spec_digest
    assert verified["current_state_version_pinned"] is False


def test_ambiguous_deploy_retains_claim_but_never_facts_binding_or_retry() -> None:
    plan = _plan()
    state = MemoryState()
    operator, entrypoints = _operator(Reader([None, None]), state)
    operator.preflight(plan.canonical_bytes)
    _approve(operator, plan.deployment_spec_digest)
    FakeApp.fail_deploy = True

    result = operator.execute(plan.deployment_spec_digest)

    assert result["status"] == "INDETERMINATE"
    assert result["retry_allowed"] is False
    assert result["binding_authorized"] is False
    assert state.resolve("attempt", plan.deployment_spec_digest) is not None
    assert state.resolve("facts", plan.deployment_spec_digest) is None
    assert state.resolve("verification", plan.deployment_spec_digest) is None
    assert operator.recover(plan.deployment_spec_digest) == {
        "status": "INDETERMINATE", "recovered": False,
        "retry_allowed": False, "binding_authorized": False,
    }
    assert entrypoints.calls == 1
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="already_consumed"):
        operator.execute(plan.deployment_spec_digest)


def test_changed_generation_does_not_authorize_binding() -> None:
    plan = _plan()
    state = MemoryState()
    operator, _ = _operator(
        Reader([None, None, _observation(1), _observation(2), _observation(2)]), state,
    )
    operator.preflight(plan.canonical_bytes)
    _approve(operator, plan.deployment_spec_digest)
    assert operator.execute(plan.deployment_spec_digest)["status"] == "ACKNOWLEDGED"
    assert operator.observe(plan.deployment_spec_digest) == {
        "status": "CHANGED", "retry_allowed": False,
        "binding_authorized": False,
    }
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="not_current"):
        operator.verify(plan.deployment_spec_digest)


def test_local_state_is_exclusive_bounded_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    import tuner.execution.providers.modal.runtime_release_qualification as qualification_module
    from tests.execution.providers.test_modal_runtime_release_qualification import Facts, _case

    state = LocalModalRuntimeReleaseState((tmp_path / "private-state").resolve())
    ref = "a" * 64
    payload = b'{"closed":true}'
    assert state.publish_if_absent("attempt", ref, payload) is True
    assert state.publish_if_absent("attempt", ref, b'{"other":true}') is False
    assert state.resolve("attempt", ref) == payload
    monkeypatch.setattr(qualification_module, "_qualification_facts_type", lambda: Facts)
    dispatch, _ = _case()
    auth = ModalRuntimeQualificationHmacAuthenticator(b"k" * 32)
    signed = build_modal_runtime_release_qualification_dispatch(dispatch, auth)
    assert state.publish_if_absent("qualification-dispatch", ref, signed) is True
    assert parse_modal_runtime_release_qualification_dispatch(
        state.resolve("qualification-dispatch", ref), auth,
    ) == dispatch
    assert state.publish_if_absent("qualification-dispatch", ref, signed) is False
    large_raw = b"x" * (64 * 1024 + 1)
    large_ref = "b" * 64
    assert state.publish_if_absent("qualification-dispatch", large_ref, large_raw) is True
    assert state.resolve("qualification-dispatch", large_ref) == large_raw
    assert state.publish_if_absent("qualification-dispatch", large_ref, large_raw) is False
    with pytest.raises(ValueError, match="oversized"):
        state.publish_if_absent("qualification-dispatch", "c" * 64, b"x" * (512 * 1024 + 1))
    claim = b'{"cpu_call_limit":1}'
    assert state.publish_if_absent("qualification-attempt", ref, claim) is True
    assert state.publish_if_absent("qualification-attempt", ref, claim) is False
    assert state.resolve("qualification-attempt", ref) == claim


def test_installed_entrypoint_resolver_requires_exact_plan_callables() -> None:
    plan = _plan()
    resolved = InstalledModalRuntimeReleaseEntrypoints().resolve(plan)
    assert resolved == {
        "training": packaged_training_entry,
        "self_check": run_runtime_release_self_check,
    }


def test_cli_approval_is_provider_and_credential_free(tmp_path: Path, monkeypatch) -> None:
    plan = _plan()
    base = (tmp_path / "private-release-state").resolve()
    state = LocalModalRuntimeReleaseState(base / "modal-runtime-releases")
    ref = plan.deployment_spec_digest
    assert state.publish_if_absent("plan", ref, plan.canonical_bytes)
    assert state.publish_if_absent("preflight", ref, canonical_bytes({
        "preflight_id": "a" * 64,
    }))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    context = SimpleNamespace(
        engine_root=(tmp_path / "engine").resolve(),
        project_root=(tmp_path / "project").resolve(),
        invocation_cwd=tmp_path.resolve(),
    )
    args = SimpleNamespace(
        base_dir=str(base), release_ref=ref, env_file=None,
        authorization_reference="approved-release", issued_at="2026-09-22T11:00:00Z",
        expires_at="2026-09-22T13:00:00Z",
    )

    result = run_modal_runtime_release_cli_action(
        "approve", args=args, context=context,
    )

    assert result["credentials_included"] is False
    assert result["deployment_spec_digest"] == ref


def test_cpu_qualification_needs_separate_preflight_approval_and_claim_before_staging() -> None:
    plan = _plan()
    current = _observation(1)
    state = MemoryState()
    release, _ = _operator(Reader([None, None] + [current] * 10), state)
    release.preflight(plan.canonical_bytes)
    _approve(release, plan.deployment_spec_digest)
    assert release.execute(plan.deployment_spec_digest)["status"] == "ACKNOWLEDGED"
    release.verify(plan.deployment_spec_digest)
    ref = plan.deployment_spec_digest
    auth = ModalRuntimeQualificationHmacAuthenticator(b"k" * 32)

    class Qualification:
        stages = 0
        submits = 0

        def stage_fixture_once(self, *, effect_id, deployment_facts):
            assert state.resolve("qualification-attempt", ref) is not None
            self.stages += 1
            return ModalRuntimeReleaseFixtureReceiptV1.create(
                effect_id=effect_id, artifact_volume_id=next(
                    item.volume_id for item in deployment_facts.volumes
                    if item.spec.role == "artifacts"
                ),
            )

        def submit_once(self, raw, *, expected_facts):
            assert state.resolve("qualification-dispatch", ref) == raw
            dispatch = parse_modal_runtime_release_qualification_dispatch(raw, auth)
            assert dispatch.deployment_facts == expected_facts
            self.submits += 1
            return ModalRuntimeQualificationOutcome("indeterminate")

    qualification = Qualification()
    controller = ModalRuntimeReleaseQualificationController(
        state=state, clock=Clock(), deployer=release._deployer,
        qualification_operator=qualification, authenticator=auth,
    )
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="preflight_unavailable"):
        controller.approve(ref, authorization_reference="cpu-call", issued_at="2026-09-22T11:00:00Z",
                           expires_at="2026-09-22T13:00:00Z")
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="approval_unavailable"):
        controller.execute(ref)
    preflight = controller.preflight(ref)
    assert preflight["policy"]["gpu"] is False
    approval = controller.approve(
        ref, authorization_reference="separate-cpu-call",
        issued_at="2026-09-22T11:00:00Z", expires_at="2026-09-22T13:00:00Z",
    )
    assert approval["authorization_id"] != json.loads(state.resolve("approval", ref))["authorization_id"]
    outcome = controller.execute(ref)
    assert outcome["status"] == "INDETERMINATE"
    assert outcome["retry_allowed"] is False
    assert state.resolve("qualification-dispatch", ref) is not None
    assert qualification.stages == qualification.submits == 1
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="claim_consumed"):
        controller.execute(ref)
    assert qualification.stages == qualification.submits == 1
    assert controller.recover(ref)["status"] == "INDETERMINATE"
    state.values[("qualification-dispatch", ref)] = b"{}"
    with pytest.raises(ValueError, match="dispatch"):
        controller.recover(ref)


def test_cpu_qualification_rejects_future_issued_approval_before_claim() -> None:
    plan = _plan()
    current = _observation(1)
    state = MemoryState()
    release, _ = _operator(Reader([None, None] + [current] * 10), state)
    release.preflight(plan.canonical_bytes)
    _approve(release, plan.deployment_spec_digest)
    assert release.execute(plan.deployment_spec_digest)["status"] == "ACKNOWLEDGED"
    release.verify(plan.deployment_spec_digest)
    controller = ModalRuntimeReleaseQualificationController(
        state=state, clock=Clock(), deployer=release._deployer,
    )
    ref = plan.deployment_spec_digest
    controller.preflight(ref)
    controller.approve(
        ref, authorization_reference="future-cpu-call",
        issued_at="2026-09-22T12:01:00Z", expires_at="2026-09-22T13:00:00Z",
    )
    with pytest.raises(ModalRuntimeReleaseOperatorError, match="not_yet_valid"):
        controller.execute(ref)
    assert state.resolve("qualification-attempt", ref) is None
