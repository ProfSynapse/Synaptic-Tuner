"""Host composition security tests independent of provider I/O."""

from __future__ import annotations

import json

import pytest

from synaptic_tuner.api.v1.training_facade import (
    AuthorizationRequirement,
    TrainingPreflight,
)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2

from examples.modal_chat.authority import ModalChatAuthorityError
from examples.modal_chat.artifacts import ModalChatArtifactVerifier
from examples.modal_chat.host import (
    ModalChatHostError,
    _AuthorizationSlot,
    compose_modal_chat_host,
)
from examples.modal_chat.requests import ModalChatTrainingRequests
from tests.execution.coordinator_v1.test_state_machine import PLAN, intent, prep
from tests.execution.providers import test_modal_coordinator_bundle as bundle_cases
from tests.execution.providers import test_modal_coordinator_consumer as modal_cases
from tests.execution.providers.test_modal_coordinator_transport import Function, Volume
from tuner.execution.providers.modal.coordinator_retention import (
    ModalRetainedPreparation,
)

from tests.examples.test_modal_chat_requests import (
    Catalog,
    _case as request_case,
    _input,
)


class Clock:
    def now(self):
        return "2026-08-27T00:00:00Z"

    def now_iso(self):
        return self.now()

    def now_epoch(self):
        return 1787788800


def _preflight(*, cost=25, checked_at="2026-08-26T23:59:00Z"):
    return TrainingPreflight(
        PLAN.plan_fingerprint,
        True,
        checked_at,
        "2026-08-27T00:10:00Z",
        (AuthorizationRequirement("training.start", True, cost, "USD"),),
    )


def _slot():
    grants = GrantAuthorityV2("grants", b"g" * 32)
    return (
        _AuthorizationSlot(
            PLAN,
            prep(),
            grants,
            Clock(),
            cost=25,
            currency="USD",
            maximum_grant_seconds=300,
        ),
        grants,
    )


def test_slot_binds_first_actual_preflight_then_issues_exact_grant():
    slot, grants = _slot()
    preflight = _preflight()
    digest = slot.commit_preflight(PLAN, preflight)
    assert slot.commit_preflight(PLAN, preflight) == digest
    command = intent("stage").canonical_command_bytes
    grant = slot.issue_effect_grant(
        command, preflight_digest=digest, now_epoch=Clock().now_epoch()
    )
    assert grants.verify(grant, command, now_epoch=Clock().now_epoch())


def test_slot_rejects_cost_substitution_and_changed_preflight():
    slot, _ = _slot()
    with pytest.raises(ModalChatHostError):
        slot.commit_preflight(PLAN, _preflight(cost=26))
    accepted = _preflight()
    slot.commit_preflight(PLAN, accepted)
    with pytest.raises(ModalChatAuthorityError):
        slot.commit_preflight(PLAN, _preflight(checked_at="2026-08-26T23:59:01Z"))


def test_slot_cannot_issue_before_commit():
    slot, _ = _slot()
    with pytest.raises(ModalChatHostError):
        slot.issue_effect_grant(
            intent("stage").canonical_command_bytes,
            preflight_digest="0" * 64,
            now_epoch=Clock().now_epoch(),
        )


class Identities:
    def allocate(self, *, project_ref, request_digest):
        from synaptic_tuner.api.v1.results import TrainingRunRef

        return "request-a", TrainingRunRef("run-1", project_ref)


def test_factory_runs_actual_public_training_path_once(monkeypatch, tmp_path):
    from examples.modal_chat.authority import HMACAuthenticator

    purposes = frozenset(
        {
            "source-lock-evidence/v1",
            "modal-deployment-evidence/v1",
            "modal-stage-claim/v2",
            "modal-launch-claim/v1",
            "provider-run-observation/v1",
            "provider-log-page/v1",
            "modal-quote-evidence/v1",
        }
    )
    evidence = HMACAuthenticator(
        {
            name: b"e" * 32
            for name in (
                "source-key",
                "deployment-key",
                "stage-key",
                "read-key",
                "quote-key",
            )
        },
        allowed_purposes=purposes,
    )
    # Sign fixture source/deployment/quote facts with the actual host algorithm;
    # never weaken the production verifier to accept fixture-only tags.
    monkeypatch.setattr(modal_cases, "evidence_tag", evidence.sign)
    # Own and restore the shared SDK fixture state, including on assertion failure.
    monkeypatch.setattr(Function, "spawns", [])
    monkeypatch.setattr(Function, "resolutions", [])
    monkeypatch.setattr(Function, "fail_spawn", False)
    monkeypatch.setattr(Function, "returned_id", "fc-1")
    monkeypatch.setattr(Volume, "registry", {})
    monkeypatch.setattr(Volume, "calls", [])
    rich_bridge, _, _, _ = request_case(tmp_path)
    request_document = json.loads(_input())
    monkeypatch.setattr(
        modal_cases,
        "_fixture",
        lambda: bundle_cases._fixture(request=request_document),
    )
    template, material, recipes, policy, closure = modal_cases._material(monkeypatch)
    preparation, operational, facade = modal_cases._preflight(
        monkeypatch, template, material
    )
    clock = operational._clock
    preparation._clock = clock
    request_catalog, material_catalog = Catalog(), Catalog()
    material_catalog.values["request-a"] = material.canonical_bytes
    requests = ModalChatTrainingRequests(
        service=rich_bridge._service,
        recipes=recipes,
        project_ref="project-a",
        identities=Identities(),
        request_catalog=request_catalog,
        material_catalog=material_catalog,
    )
    requests.resolve(requests.load(_input()))
    retained = ModalRetainedPreparation(
        preparation.snapshot(),
        template.deployment_bytes,
        material.canonical_bytes,
        recipes,
        policy,
        closure,
        "cv",
        "av",
        "stage-key",
    )
    host = compose_modal_chat_host(
        preparation=preparation,
        operational_preflight=operational,
        facade=facade,
        deployment=template.deployment,
        recipes=recipes,
        requests=requests,
        retained=retained,
        evidence_authenticator=evidence,
        evidence_key_ref="read-key",
        artifact_key=b"a" * 32,
        clock=clock,
        observed_at="2026-08-25T12:02:00Z",
        grant_key=b"g" * 32,
        receipt_key=b"r" * 32,
        invalid_evidence_key=b"i" * 32,
        assessment_key=b"s" * 32,
        cursor_key=b"c" * 32,
        approved_cost_minor_units=125,
    )
    request = host.api.training.load(_input())
    resolved = host.api.training.resolve(request)
    context, _, _ = preparation._snapshot()
    plan = host.api.training.plan(resolved, context.provider)
    preflight = host.api.training.preflight(plan)
    first = host.api.training.start(plan, preflight)
    assert first.accepted is True
    assert len(Function.spawns) == 1
    assert host.api.training.start(plan, preflight) == first
    assert len(Function.spawns) == 1
