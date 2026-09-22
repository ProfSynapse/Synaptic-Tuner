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
    run_modal_runtime_release_cli_action,
)
from tuner.execution.foundation_v2.canonical import canonical_bytes
from tuner.execution.providers.modal.runtime_release_deployment import (
    ModalRuntimeReleaseDeployer,
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


def test_local_state_is_exclusive_bounded_and_round_trips(tmp_path: Path) -> None:
    state = LocalModalRuntimeReleaseState((tmp_path / "private-state").resolve())
    ref = "a" * 64
    payload = b'{"closed":true}'
    assert state.publish_if_absent("attempt", ref, payload) is True
    assert state.publish_if_absent("attempt", ref, b'{"other":true}') is False
    assert state.resolve("attempt", ref) == payload


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
