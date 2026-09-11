"""End-to-end provider-free checks for the one-attempt Modal run-chat adapter."""

from __future__ import annotations

import hashlib
from queue import Queue
import time

import pytest
from tests.execution.foundation_v2.helpers import StrongVerifier
from tests.execution.providers.test_modal_coordinator_preflight import (
    body as quote_body,
    evidence_tag,
)
from tests.execution.providers.test_modal_inference_preparation import (
    Auth,
    _document,
)
from tests.execution.providers.test_modal_inference_workload import (
    _case as workload_case,
)
from tuner.execution.foundation_v2.authority import GrantAuthorityV2
from tuner.execution.foundation_v2.broker import EffectBrokerV2
from tuner.execution.foundation_v2.canonical import canonical_bytes, domain_digest
from tuner.execution.foundation_v2.commands import SubmitCommandV2, parse_exact_command
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.foundation_v2.repository import InMemoryEffectRepositoryV2
from tuner.execution.providers.modal.coordinator_effects import ModalEffectOutcome
from tuner.execution.providers.modal.coordinator_preflight import (
    AuthenticatedModalQuote,
    QUOTE_PURPOSE,
    TrustedEvidenceIdentity,
)
from tuner.execution.providers.modal.inference_channel import encode_modal_chat_frame
from tuner.execution.providers.modal.inference_channel_client import (
    ModalInferenceChannelClient,
)
from tuner.execution.providers.modal.inference_commands import (
    ModalInferenceCommandBinding,
)
from tuner.execution.providers.modal.inference_effects import (
    ModalChatEffectExecutor,
    ModalChatExecutorResolver,
)
from tuner.execution.providers.modal.inference_launch import modal_chat_stage_ref
from tuner.execution.providers.modal.inference_preparation import (
    CONFIG_EVIDENCE_PURPOSE,
    AuthenticatedModalInferencePreparationConfig,
    ModalInferencePreparationConfig,
)
from tuner.execution.providers.modal.inference_run_chat import (
    ModalRunChatError,
    ModalRunChatRuntime,
)
from tuner.execution.providers.modal.inference_transport import (
    ModalChatLeaseHandoff,
    ModalChatSandboxLease,
)
from Evaluator.chat_session import ChatSessionError
from tuner.inference.run_chat import PreparedModelIdentity, open_run_chat


class _Clock:
    def now_iso(self):
        return "2026-09-09T12:02:00Z"

    def now_epoch(self):
        return 150


class _Catalog:
    def __init__(self):
        self.values = {}

    def resolve(self, key):
        return self.values.get(key)

    def publish_if_absent(self, key, value):
        self.values.setdefault(key, value)


class _BindingAuthority:
    """Test host policy authorizing only this config/session/executor tuple."""

    def __init__(self, configuration_bytes):
        self.configuration_bytes = configuration_bytes
        self.authenticated = []

    def authenticate(self, binding):
        if type(binding) is not ModalInferenceCommandBinding:
            return False
        snapshot = __import__(
            "tuner.execution.providers.modal.inference_preparation",
            fromlist=["_validate_preparation_snapshot"],
        )._validate_preparation_snapshot(binding.preparation_snapshot)
        command = parse_exact_command(binding.command_bytes)
        accepted = (
            canonical_bytes(snapshot["configuration"]) == self.configuration_bytes
            and command.preparation.run_id == "chat-session-1"
            and command.executor.executor_id == "modal-chat-executor"
            and command.executor.implementation_version == "v1"
        )
        if accepted:
            self.authenticated.append(binding.canonical_bytes)
        return accepted


class _Grants:
    def __init__(self, authority):
        self.authority = authority
        self.calls = []
        self.callback = None

    def grant(self, command_bytes, *, phase):
        command = parse_exact_command(command_bytes)
        self.calls.append((phase, command.digest, bytes(command_bytes)))
        if self.callback is not None:
            self.callback(phase)
        return self.authority.issue(
            command_bytes,
            grant_ref=f"chat-{phase}-grant",
            policy_digest="9" * 64,
            requirement_digest="a" * 64,
            not_before_epoch=100,
            expires_at_epoch=200,
        )


class _Foundation:
    def __init__(self, repository, assessments):
        self.repository = repository
        self.assessments = assessments

    def get(self, effect_id):
        return self.repository.get(effect_id)

    def assess(self, record):
        return self.assessments.assess(record)


class _Input:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def drain(self):
        return None


class _Output:
    def __init__(self, values):
        self.values = Queue()
        for value in values:
            self.values.put(value)

    def __iter__(self):
        return self

    def __next__(self):
        return self.values.get()


class _Sandbox:
    object_id = "sb-run-chat"

    def __init__(self, output, *, unresolved_cleanup=False):
        self._stdin = _Input()
        self._stdout = _Output(output)
        self.terminations = 0
        self.unresolved_cleanup = unresolved_cleanup

    @property
    def stdin(self):
        return self._stdin

    @property
    def stdout(self):
        return self._stdout

    def terminate(self, *, wait):
        assert wait is True
        self.terminations += 1
        return None if self.unresolved_cleanup else 0


def _frame(session_id, launch_digest, model, kind, **extra):
    return encode_modal_chat_frame(
        {
            "schema_version": "synaptic-modal-chat-channel/v1",
            "kind": kind,
            "session_id": session_id,
            "launch_digest": launch_digest,
            **({"model": model} if kind == "ready" else {}),
            **extra,
        },
        65536,
    ).decode()


class _Transport:
    def __init__(self, handoff):
        self.handoff = handoff
        self.calls = []
        self.sandbox = None
        self.fail_submit = False
        self.lease_mismatch = None
        self.after_stage = None
        self.lease_lifetime = 60
        self.expire_before_return = False
        self.unresolved_cleanup = False

    def execute_once(self, binding, command):
        self.calls.append(command)
        if not isinstance(command, SubmitCommandV2):
            result = ModalEffectOutcome(
                ObservationDisposition.FOUND, modal_chat_stage_ref(binding)
            )
            if self.after_stage is not None:
                self.after_stage()
            return result
        snapshot = __import__(
            "tuner.execution.providers.modal.inference_preparation",
            fromlist=["_validate_preparation_snapshot"],
        )._validate_preparation_snapshot(binding.preparation_snapshot)
        chat = snapshot["chat_input"]
        workload = chat["workload"]
        argument = b"authenticated-run-chat-launch"
        launch_digest = hashlib.sha256(argument).hexdigest()
        model = {
            "model_ref": (
                "wrong/model"
                if self.lease_mismatch == "model"
                else workload["model_ref"]
            ),
            "model_revision": workload["model_revision"],
            "tokenizer_revision": workload["tokenizer_revision"],
            "model_kind": "full",
        }
        session_id = chat["session_id"]
        sandbox = _Sandbox(
            [
                _frame(session_id, launch_digest, model, "ready"),
                _frame(
                    session_id,
                    launch_digest,
                    model,
                    "chat",
                    request_id=1,
                    content="one",
                ),
                _frame(
                    session_id,
                    launch_digest,
                    model,
                    "chat",
                    request_id=2,
                    content="two",
                ),
                _frame(session_id, launch_digest, model, "closed", request_id=3),
            ],
            unresolved_cleanup=self.unresolved_cleanup,
        )
        deadline = time.monotonic() + self.lease_lifetime
        channel = ModalInferenceChannelClient(
            sandbox,
            session_id=session_id,
            argument_bytes=argument,
            deadline=deadline,
            startup_timeout_seconds=1,
            request_timeout_seconds=1,
            max_request_bytes=8192,
            max_response_bytes=65536,
        )
        lease = ModalChatSandboxLease(
            command.digest,
            session_id,
            argument,
            deadline,
            (
                b"wrong-snapshot"
                if self.lease_mismatch == "snapshot"
                else binding.preparation_snapshot
            ),
            sandbox,
            channel,
        )
        self.sandbox = sandbox
        self.handoff.publish(lease)
        if self.expire_before_return:
            time.sleep(self.lease_lifetime + 0.01)
        if self.fail_submit:
            raise RuntimeError("private ambiguous broker failure")
        provider_ref = (
            "sb-other" if self.lease_mismatch == "provider_ref" else sandbox.object_id
        )
        return ModalEffectOutcome(ObservationDisposition.FOUND, provider_ref)

    def lookup_once(self, binding, command):
        raise AssertionError("run-chat open never reconciles or retries")


def _runtime(monkeypatch):
    prior_source, source_transport, values = workload_case(monkeypatch)
    binder, runs, run, operations, store = values[:5]
    operations.calls.clear()
    document = _document(prior_source)
    config = ModalInferencePreparationConfig.build(document)
    configuration = AuthenticatedModalInferencePreparationConfig(
        config.canonical_bytes,
        evidence_tag(CONFIG_EVIDENCE_PURPOSE, config.canonical_bytes, "config-key"),
    )
    resource = domain_digest(
        "synaptic-modal-inference-resource/v1",
        canonical_bytes(
            {
                "resources": document["resources"],
                "volumes": document["volumes"],
                "application": document["application"],
            }
        ),
    )
    raw_quote = quote_body(
        provider_id="modal",
        profile_ref="chat-a10",
        account_ref=prior_source.account_ref,
        namespace_ref=prior_source.namespace_ref,
        resource_digest=resource,
    )
    quote = AuthenticatedModalQuote(
        raw_quote, evidence_tag(QUOTE_PURPOSE, raw_quote, "quote-key")
    )
    # The runtime deliberately receives one Foundation evidence authenticator for
    # both source admission and chat effects, so the test host composes the new
    # chat repository from those same real authorities.
    grants = source_transport._foundation.grants
    receipts = source_transport._foundation.receipts
    invalid = source_transport._foundation.invalid
    repository = InMemoryEffectRepositoryV2(
        receipts, invalid, StrongVerifier(), StrongVerifier(), grants
    )
    clock = _Clock()
    assessments = source_transport._assessments
    foundation = _Foundation(repository, assessments)
    catalog = _Catalog()
    authority = _BindingAuthority(config.canonical_bytes)
    handoff = ModalChatLeaseHandoff()
    transport = _Transport(handoff)
    executor = ModalChatEffectExecutor(
        profile_ref="chat-a10",
        account_ref=prior_source.account_ref,
        namespace_ref=prior_source.namespace_ref,
        implementation_version="v1",
        catalog=catalog,
        authority=authority,
        transport=transport,
    )
    broker = EffectBrokerV2(
        repository,
        ModalChatExecutorResolver(executor),
        grants,
        receipts,
        invalid,
    )
    grant_port = _Grants(grants)
    runtime = ModalRunChatRuntime(
        runs=runs,
        source_binder=binder,
        launch_source=source_transport._launch_source,
        foundation_authenticator=source_transport._foundation,
        assessment_authenticator=source_transport._assessments,
        workload_binding_authority=source_transport._authority,
        stage_verifier=source_transport._stage,
        launch_verifier=source_transport._launch,
        recipes=source_transport._recipes,
        configuration=configuration,
        configuration_trust=TrustedEvidenceIdentity(
            "host-config", "config-key", "chat-session"
        ),
        quote=quote,
        quote_trust=TrustedEvidenceIdentity("host-quoter", "quote-key", "project-run"),
        evidence_authenticator=Auth(),
        clock=clock,
        session_id="chat-session-1",
        executor_version="v1",
        stage_nonce="chat-stage-nonce",
        submit_nonce="chat-submit-nonce",
        catalog=catalog,
        chat_binding_authority=authority,
        grants=grant_port,
        broker=broker,
        foundation=foundation,
        handoff=handoff,
    )
    return locals()


def test_success_uses_one_reverify_two_exact_grants_and_closes(monkeypatch):
    case = _runtime(monkeypatch)
    with open_run_chat(case["runs"], case["run"], runtime=case["runtime"]) as prepared:
        assert prepared.session.chat("first").message == "one"
        assert prepared.session.chat("second").message == "two"
        assert prepared.model == PreparedModelIdentity(
            prepared.model.model_ref,
            prepared.model.model_revision,
            prepared.model.tokenizer_revision,
            "full",
        )
    assert case["operations"].calls.count("reverify") == 1
    assert [phase for phase, _, _ in case["grant_port"].calls] == [
        "stage",
        "submit",
    ]
    assert [raw for _, _, raw in case["grant_port"].calls] == [
        command.canonical_bytes for command in case["transport"].calls
    ]
    assert [type(value).__name__ for value in case["transport"].calls] == [
        "StageCommandV2",
        "SubmitCommandV2",
    ]
    assert case["transport"].sandbox.terminations == 1


def test_workflow_drift_in_grant_callback_denies_before_submit_dispatch(monkeypatch):
    case = _runtime(monkeypatch)

    def mutate(phase):
        if phase == "submit":
            workflow = case["store"].value
            object.__setattr__(workflow, "revision", workflow.revision + 1)

    case["grant_port"].callback = mutate
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert [type(value).__name__ for value in case["transport"].calls] == [
        "StageCommandV2"
    ]


def test_workflow_drift_after_stage_denies_before_submit_grant(monkeypatch):
    case = _runtime(monkeypatch)

    def drift_after_stage():
        workflow = case["store"].value
        object.__setattr__(workflow, "revision", workflow.revision + 1)

    case["transport"].after_stage = drift_after_stage
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert [phase for phase, _, _ in case["grant_port"].calls] == ["stage"]


def test_bad_stage_grant_never_reaches_transport_and_attempt_is_one_use(monkeypatch):
    case = _runtime(monkeypatch)
    foreign = GrantAuthorityV2("foreign", b"f" * 32)
    case["grant_port"].authority = foreign
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert case["transport"].calls == []
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_already_attempted$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert case["operations"].calls.count("reverify") == 1


def test_body_exception_is_preserved_while_owned_lease_is_closed(monkeypatch):
    case = _runtime(monkeypatch)
    with pytest.raises(LookupError, match="caller-body"):
        with case["runtime"].open(case["runs"], case["run"]):
            raise LookupError("caller-body")
    assert case["runtime"].owned_lease is not None
    assert case["transport"].sandbox.terminations == 1


def test_submit_failure_after_handoff_publication_peeks_and_closes(monkeypatch):
    case = _runtime(monkeypatch)
    case["transport"].fail_submit = True
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert case["runtime"].owned_lease is not None
    assert case["transport"].sandbox.terminations == 1


def test_interrupt_after_handoff_take_retains_exact_cleanup_owner(monkeypatch):
    case = _runtime(monkeypatch)
    take = ModalChatLeaseHandoff.take

    def take_then_interrupt(self, *, submit_command_digest):
        take(self, submit_command_digest=submit_command_digest)
        raise KeyboardInterrupt

    monkeypatch.setattr(ModalChatLeaseHandoff, "take", take_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        with case["runtime"].open(case["runs"], case["run"]):
            pytest.fail("interrupted handoff yielded a session")
    owner = case["runtime"].owned_lease
    assert owner is not None
    assert (
        case["handoff"].peek(submit_command_digest=owner.submit_command_digest) is owner
    )
    assert case["transport"].sandbox.terminations == 1
    with pytest.raises(RuntimeError):
        take(case["handoff"], submit_command_digest=owner.submit_command_digest)


@pytest.mark.parametrize("mismatch", ("model", "provider_ref", "snapshot"))
def test_ready_lease_identity_mismatch_is_owned_and_closed(monkeypatch, mismatch):
    case = _runtime(monkeypatch)
    case["transport"].lease_mismatch = mismatch
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert case["runtime"].owned_lease is not None
    assert case["transport"].sandbox.terminations == 1


def test_expired_ready_lease_is_rejected_before_yield_and_closed(monkeypatch):
    case = _runtime(monkeypatch)
    case["transport"].lease_lifetime = 0.02
    case["transport"].expire_before_return = True
    yielded = []
    with pytest.raises(ModalRunChatError, match="^modal_run_chat_invalid$"):
        with case["runtime"].open(case["runs"], case["run"]):
            yielded.append(True)
    assert yielded == []
    assert case["runtime"].owned_lease is not None
    assert case["transport"].sandbox.terminations == 1


def test_normal_exit_surfaces_unresolved_cleanup_and_retains_owner(monkeypatch):
    case = _runtime(monkeypatch)
    case["transport"].unresolved_cleanup = True
    with pytest.raises(ChatSessionError, match="runtime cleanup remains unresolved"):
        with case["runtime"].open(case["runs"], case["run"]):
            pass
    assert case["runtime"].owned_lease is not None
    assert case["runtime"].owned_lease.cleanup_pending is True
    assert case["transport"].sandbox.terminations == 1
