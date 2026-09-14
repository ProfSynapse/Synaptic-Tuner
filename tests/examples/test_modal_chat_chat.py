from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.modal_chat.chat import (
    ModalChatRuntimeCompositionError,
    _BindingAuthority,
    _BindingCatalog,
    _ChatGrants,
    compose_modal_run_chat,
)
from examples.modal_chat.host import ModalChatHost
from tests.execution.foundation_v2.helpers import D, prep, stage_command
from tests.execution.providers.test_modal_coordinator_preflight import evidence_tag
from tests.execution.providers.test_modal_inference_preparation import Auth, _document
from tests.execution.providers.test_modal_inference_run_chat import (
    _Clock,
    _runtime,
)
from tests.execution.providers.test_modal_inference_workload import (
    _case as workload_case,
)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.canonical import parse_canonical_object
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.coordinator_preflight import (
    TrustedEvidenceIdentity,
)
from tuner.execution.providers.modal.inference_preparation import (
    CONFIG_EVIDENCE_PURPOSE,
    AuthenticatedModalInferencePreparationConfig,
    ModalInferencePreparationConfig,
)
from tuner.execution.providers.modal.inference_transport import (
    ModalChatLaunchSettings,
    ModalInferenceSdkTransport,
)
from tuner.inference.run_chat import open_run_chat


def test_binding_catalog_is_exact_immutable_and_authenticates(monkeypatch) -> None:
    case = _runtime(monkeypatch)
    with case["runtime"].open(case["runs"], case["run"]):
        pass
    value = next(iter(case["catalog"].values.values()))
    catalog = _BindingCatalog()
    authority = _BindingAuthority(
        catalog, case["configuration"].body_bytes, "chat-session-1", "v1"
    )
    assert catalog.publish_if_absent(value.command_digest, value) is True
    assert catalog.publish_if_absent(value.command_digest, value) is False
    assert authority.authenticate(value) is True
    assert authority.authenticate(object()) is False


def test_composer_rejects_an_independent_clock_before_any_provider_work():
    import inspect

    host = ModalChatHost(*([None] * 10), _Clock())
    arguments = {
        name: None
        for name, parameter in inspect.signature(
            compose_modal_run_chat
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    arguments.update(host=host, clock=_Clock())
    with pytest.raises(ModalChatRuntimeCompositionError, match="clock_mismatch"):
        compose_modal_run_chat(**arguments)


def test_chat_grants_are_command_bound_bounded_and_one_per_phase() -> None:
    authority = GrantAuthorityV2("chat-grants", b"g" * 32)
    grants = _ChatGrants(
        authority,
        _Clock(),
        session_id="run",
        policy_digest=D[0],
        requirement_digest=D[1],
        maximum_seconds=60,
    )
    command = stage_command(prep(run_id="run"))
    grant = grants.grant(command.canonical_bytes, phase="stage")
    assert authority.verify(grant, command.canonical_bytes, now_epoch=150)
    assert grant.content.expires_at_epoch - grant.content.not_before_epoch == 60
    with pytest.raises(ModalChatRuntimeCompositionError, match="already_issued"):
        grants.grant(command.canonical_bytes, phase="stage")
    with pytest.raises(ModalChatRuntimeCompositionError, match="invalid"):
        _ChatGrants(
            authority,
            _Clock(),
            session_id="other-session",
            policy_digest=D[0],
            requirement_digest=D[1],
            maximum_seconds=60,
        ).grant(command.canonical_bytes, phase="stage")


def test_composer_rejects_any_image_other_than_explicit_qualified_identity(
    monkeypatch,
) -> None:
    source, _, values = workload_case(monkeypatch)
    document = _document(source)
    config = ModalInferencePreparationConfig.build(document)
    authenticated = AuthenticatedModalInferencePreparationConfig(
        config.canonical_bytes,
        evidence_tag(CONFIG_EVIDENCE_PURPOSE, config.canonical_bytes, "config-key"),
    )
    clock = _Clock()
    host = ModalChatHost(*([SimpleNamespace()] * 10), clock)
    with pytest.raises(ModalChatRuntimeCompositionError, match="not_qualified"):
        compose_modal_run_chat(
            host=host,
            deployment=SimpleNamespace(),
            launch_source=None,
            recipes=None,
            configuration=authenticated,
            configuration_trust=None,
            quote=None,
            quote_trust=None,
            evidence_authenticator=None,
            clock=clock,
            session_id="chat-session",
            stage_nonce="stage-nonce",
            submit_nonce="submit-nonce",
            executor_version="v1",
            policy_digest=D[0],
            requirement_digest=D[1],
            launch_settings=None,
            launch_signer=None,
            issued_at="2026-09-09T00:00:00Z",
            expires_at="2026-09-09T00:05:00Z",
            evidence_ref="evidence-a",
            profile_ref="chat-a10",
            namespace_ref="namespace-a",
            prequalified_image_id="im-unqualified",
        )


def test_actual_open_run_chat_with_real_adapter_and_simulated_provider(monkeypatch):
    case = _runtime(monkeypatch)
    source_transport, binder = case["source_transport"], case["binder"]
    ports = SimpleNamespace(
        grant_authority=source_transport._foundation.grants,
        receipt_authority=source_transport._foundation.receipts,
        invalid_evidence_authority=source_transport._foundation.invalid,
        assessment_authority=source_transport._assessments,
        foundation_authenticator=source_transport._foundation,
        binding_authority=source_transport._authority,
        stage_authority=source_transport._stage,
        launch_authority=source_transport._launch,
    )
    host = ModalChatHost(
        SimpleNamespace(runs=case["runs"]),
        SimpleNamespace(foundation=binder._foundation),
        binder._reader,
        None,
        ports,
        SimpleNamespace(workflow_store=binder._workflows),
        None,
        None,
        None,
        None,
        case["clock"],
    )

    class Deployment:
        selection = SimpleNamespace(account_ref=case["prior_source"].account_ref)

        def facade(self):
            return source_transport._facade

    simulator = case["transport"]

    def simulated_execute(real_transport, binding, command):
        outcome = simulator.execute_once(binding, command)
        if command.operation.effect.kind.value == "submit":
            lease = simulator.handoff.take(submit_command_digest=command.digest)
            real_transport._handoff.publish(lease)
        return outcome

    monkeypatch.setattr(
        ModalInferenceSdkTransport,
        "execute_once",
        simulated_execute,
    )
    document = parse_canonical_object(case["configuration"].body_bytes, name="config")
    document["image"]["provider_image_id"] = "im-rwqrQYtujjHal3RpL3RYW5"
    parsed = ModalInferencePreparationConfig.build(document)
    configuration = AuthenticatedModalInferencePreparationConfig(
        parsed.canonical_bytes,
        evidence_tag(CONFIG_EVIDENCE_PURPOSE, parsed.canonical_bytes, "config-key"),
    )
    graph = compose_modal_run_chat(
        host=host,
        deployment=Deployment(),
        launch_source=source_transport._launch_source,
        recipes=source_transport._recipes,
        configuration=configuration,
        configuration_trust=TrustedEvidenceIdentity(
            "host-config", "config-key", "chat-session"
        ),
        quote=case["quote"],
        quote_trust=TrustedEvidenceIdentity("host-quoter", "quote-key", "project-run"),
        evidence_authenticator=Auth(),
        clock=case["clock"],
        session_id="chat-session-1",
        stage_nonce="chat-stage-nonce",
        submit_nonce="chat-submit-nonce",
        executor_version="v1",
        policy_digest=D[0],
        requirement_digest=D[1],
        launch_settings=ModalChatLaunchSettings(
            b"config",
            "issuer",
            "audience",
            "key",
            "challenge",
            "/artifacts",
            "/control",
            "/cache",
            "EVIDENCE_KEY",
            None,
        ),
        launch_signer=Auth(),
        issued_at="2026-09-09T12:02:00Z",
        expires_at="2026-09-09T12:03:00Z",
        evidence_ref="chat-evidence",
        profile_ref="chat-a10",
        namespace_ref=case["prior_source"].namespace_ref,
        prequalified_image_id="im-rwqrQYtujjHal3RpL3RYW5",
    )
    with open_run_chat(case["runs"], case["run"], runtime=graph.runtime) as prepared:
        assert prepared.session.chat("hello").message == "one"
    assert simulator.sandbox.terminations == 1
    assert graph.pending_ownership() == ()
