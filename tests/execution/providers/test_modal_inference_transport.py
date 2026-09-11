"""Provider-free checks for the pure chat transport boundary."""

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

from tuner.execution.foundation_v2.commands import parse_exact_command
from tuner.execution.foundation_v2.observations import ObservationDisposition
from tuner.execution.providers.modal.binding import ModalClientBinding
from tuner.execution.providers.modal.facade import ExplicitModal154ReadFacade
from tuner.execution.providers.modal.inference_launch import modal_chat_stage_ref
from tuner.execution.providers.modal.inference_channel import (
    decode_modal_chat_frame,
    encode_modal_chat_frame,
)
from tuner.execution.providers.modal.inference_channel_client import (
    ModalInferenceChannelClient as RealModalInferenceChannelClient,
)
from tuner.inference.run_chat import PreparedModelIdentity
from tuner.execution.providers.modal.inference_preparation import (
    _validate_preparation_snapshot,
)
from tuner.execution.providers.modal.inference_transport import (
    ModalChatCleanupOwnership,
    ModalChatLaunchSettings,
    ModalChatLeaseHandoff,
    ModalInferenceSdkTransport,
    ModalInferenceTransportError,
)
from tests.execution.providers.test_modal_inference_commands import Authority, _bindings
from tests.execution.providers.test_modal_inference_launch_integration import (
    _launch_case,
)


def _case(monkeypatch):
    binding = _bindings(monkeypatch)[0]
    command = parse_exact_command(binding.command_bytes)
    document = _validate_preparation_snapshot(binding.preparation_snapshot)[
        "configuration"
    ]
    client = document["client"]
    sdk = SimpleNamespace(__version__="1.5.4")
    explicit = object()
    facade = ExplicitModal154ReadFacade(
        ModalClientBinding(
            client["account_ref"],
            client["workspace_ref"],
            client["environment_ref"],
            client["client_ref"],
            client["sdk_version"],
        ),
        sdk=sdk,
        client=explicit,
        scope_observer=lambda supplied: (
            client["account_ref"],
            client["workspace_ref"],
            client["environment_ref"],
            client["client_ref"],
        ),
        deployment_observer=lambda **kwargs: None,
        volume_names={"volume": "name"},
    )
    authority = Authority([binding.canonical_bytes])
    return binding, command, facade, authority


def test_stage_execute_and_lookup_are_deterministic_and_sdk_free(monkeypatch):
    binding, command, facade, authority = _case(monkeypatch)
    transport = ModalInferenceSdkTransport(
        facade=facade,
        binding_authority=authority,
        handoff=ModalChatLeaseHandoff(),
    )
    executed = transport.execute_once(binding, command)
    looked_up = transport.lookup_once(binding, command)
    assert executed == looked_up
    assert executed.disposition is ObservationDisposition.FOUND
    assert executed.provider_ref == modal_chat_stage_ref(binding)


def test_substituted_command_is_denied(monkeypatch):
    binding, command, facade, authority = _case(monkeypatch)
    other = _bindings(monkeypatch)[1]
    transport = ModalInferenceSdkTransport(
        facade=facade,
        binding_authority=authority,
        handoff=ModalChatLeaseHandoff(),
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(binding, parse_exact_command(other.command_bytes))


def test_binding_authority_cannot_mutate_authenticated_snapshot(monkeypatch):
    binding, command, facade, _ = _case(monkeypatch)

    class MutatingAuthority:
        def authenticate(self, supplied):
            object.__setattr__(supplied, "preparation_snapshot", b"changed")
            return True

    transport = ModalInferenceSdkTransport(
        facade=facade,
        binding_authority=MutatingAuthority(),
        handoff=ModalChatLeaseHandoff(),
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(binding, command)


def test_submit_without_required_dependencies_is_rejected(monkeypatch):
    binding = _bindings(monkeypatch)[1]
    command = parse_exact_command(binding.command_bytes)
    stage, _, facade, _ = _case(monkeypatch)
    authority = Authority([binding.canonical_bytes, stage.canonical_bytes])
    transport = ModalInferenceSdkTransport(
        facade=facade,
        binding_authority=authority,
        handoff=ModalChatLeaseHandoff(),
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(binding, command)


class _Input:
    def __init__(self, *, error=False, gate=None):
        self.values = []
        self.error = error
        self.gate = gate
        self.wrote = threading.Event()

    def write(self, value):
        if self.gate is not None:
            self.gate.wait()
        if self.error:
            raise OSError("synthetic stdin failure")
        self.values.append(value)
        self.wrote.set()

    def drain(self):
        return None


class _Output:
    def __init__(self, stdin, case, *, wrong_model=False, ready_eof=False):
        self.stdin = stdin
        self.case = case
        self.wrong_model = wrong_model
        self.ready_eof = ready_eof

    def __iter__(self):
        return self

    def __next__(self):
        if self.ready_eof:
            raise StopIteration
        envelope = self.case.envelope
        model = self.case.workload
        return encode_modal_chat_frame(
            {
                "schema_version": "synaptic-modal-chat-channel/v1",
                "kind": "ready",
                "session_id": self.case.session_id,
                "launch_digest": hashlib.sha256(envelope.argument_bytes).hexdigest(),
                "model": {
                    "model_ref": "wrong" if self.wrong_model else model["model_ref"],
                    "model_revision": model["model_revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                    "model_kind": "full",
                },
            },
            65536,
        ).decode("utf-8")


class _Sandbox:
    object_id = "sb-chat"
    stdin = None
    stdout = None

    def __init__(
        self,
        case,
        *,
        wrong_model=False,
        stdin_error=False,
        stdin_gate=None,
        ready_eof=False,
        termination_gate=None,
    ):
        self.stdin = _Input(error=stdin_error, gate=stdin_gate)
        self.stdout = _Output(
            self.stdin, case, wrong_model=wrong_model, ready_eof=ready_eof
        )
        self.terminations = []
        self.terminated = threading.Event()
        self.termination_started = threading.Event()
        self.termination_gate = termination_gate

    def terminate(self, *, wait):
        self.terminations.append(wait)
        self.termination_started.set()
        if self.termination_gate is not None:
            self.termination_gate.wait()
        self.terminated.set()
        return 0


class _Channel:
    def __init__(self, sandbox, **kwargs):
        ready = decode_modal_chat_frame(
            next(iter(sandbox.stdout)).encode("utf-8"), kwargs["max_response_bytes"]
        )
        assert ready["session_id"] == kwargs["session_id"]
        assert (
            ready["launch_digest"]
            == hashlib.sha256(kwargs["argument_bytes"]).hexdigest()
        )
        self._sandbox = sandbox
        self._object_id = sandbox.object_id
        self._session_id = kwargs["session_id"]
        self._launch_digest = hashlib.sha256(kwargs["argument_bytes"]).hexdigest()
        self._deadline = kwargs["deadline"]
        self._model = PreparedModelIdentity(**ready["model"])
        self.cleanup_pending = True

    @property
    def model(self):
        return PreparedModelIdentity(
            self._model.model_ref,
            self._model.model_revision,
            self._model.tokenizer_revision,
            self._model.model_kind,
        )

    def close(self, **kwargs):
        self.cleanup_pending = False
        return type(self._sandbox.terminate(wait=True)) is int


class _Volume:
    object_id = ""

    def __init__(self, object_id):
        self.object_id = object_id
        self.is_hydrated = False

    def hydrate(self, client):
        self.is_hydrated = True


class _ImageValue:
    object_id = "im-chat"

    def __init__(self):
        self.is_hydrated = False
        self.hydrate_calls = []

    def hydrate(self, client):
        self.hydrate_calls.append(client)
        self.is_hydrated = True


def _submit_transport(
    monkeypatch,
    *,
    wrong_model=False,
    delayed_create: threading.Event | None = None,
    stdin_error=False,
    stdin_gate=None,
    mismatch=None,
    interrupt_create=False,
    evidence_key="HF_TOKEN",
    real_channel=False,
    ready_eof=False,
    termination_gate=None,
):
    launch = _launch_case(monkeypatch)
    snapshot = _validate_preparation_snapshot(
        launch.submit_binding.preparation_snapshot
    )
    configuration = snapshot["configuration"]
    state = SimpleNamespace(
        creates=[], volumes=[], secrets=[], created=threading.Event()
    )
    channel_case = SimpleNamespace(
        envelope=None,
        workload=snapshot["chat_input"]["workload"],
        session_id=snapshot["chat_input"]["session_id"],
    )

    class Volume:
        @staticmethod
        def from_name(name, **kwargs):
            state.volumes.append((name, kwargs))
            for prefix in ("source_artifact", "chat_control", "model_cache"):
                if configuration["volumes"][prefix + "_volume_ref"] == name:
                    return _Volume(configuration["volumes"][prefix + "_volume_id"])
            raise AssertionError

    class App:
        @staticmethod
        def lookup(name, **kwargs):
            app_id = configuration["application"]["app_ref"]
            return SimpleNamespace(app_id="ap-wrong" if mismatch == "app" else app_id)

    class Image:
        @staticmethod
        def from_id(image_id, **kwargs):
            assert image_id == "im-chat"
            value = _ImageValue()
            if mismatch == "image":
                value.object_id = "im-wrong"
            state.image = value
            return value

    class Secret:
        @staticmethod
        def from_name(name, **kwargs):
            state.secrets.append((name, kwargs))
            return (name, tuple(kwargs["required_keys"]))

    class Sandbox:
        @staticmethod
        def create(*args, **kwargs):
            state.creates.append((args, kwargs))
            if delayed_create is not None:
                delayed_create.wait()
            value = _Sandbox(
                channel_case,
                wrong_model=wrong_model,
                stdin_error=stdin_error,
                stdin_gate=stdin_gate,
                ready_eof=ready_eof,
                termination_gate=termination_gate,
            )
            state.sandbox = value
            state.created.set()
            return value

    sdk = SimpleNamespace(
        __version__="1.5.4",
        Volume=Volume,
        App=App,
        Image=Image,
        Secret=Secret,
        Sandbox=Sandbox,
    )
    client_config = configuration["client"]
    facade = ExplicitModal154ReadFacade(
        ModalClientBinding(
            client_config["account_ref"],
            client_config["workspace_ref"],
            client_config["environment_ref"],
            client_config["client_ref"],
            client_config["sdk_version"],
        ),
        sdk=sdk,
        client=object(),
        scope_observer=lambda supplied: tuple(
            client_config[name]
            for name in (
                "account_ref",
                "workspace_ref",
                "environment_ref",
                "client_ref",
            )
        ),
        deployment_observer=lambda **kwargs: None,
        volume_names={
            configuration["volumes"][prefix + "_volume_id"]: (
                "wrong-volume-name"
                if mismatch == "volume" and prefix == "source_artifact"
                else configuration["volumes"][prefix + "_volume_ref"]
            )
            for prefix in ("source_artifact", "chat_control", "model_cache")
        },
    )

    class Foundation:
        def get(self, effect_id):
            assert effect_id == launch.stage_record.command.operation.effect.effect_id
            return launch.stage_record

        def assess(self, record):
            assert record is launch.stage_record
            return launch.assessment

    launch.case.catalog.values[launch.case.stage.command_digest] = launch.case.stage
    handoff = ModalChatLeaseHandoff()
    settings = ModalChatLaunchSettings(
        configuration_bytes=launch.expectation.configuration_bytes,
        issuer_ref=launch.expectation.issuer_ref,
        audience_ref=launch.expectation.audience_ref,
        key_ref=launch.expectation.key_ref,
        challenge_nonce=launch.expectation.challenge_nonce,
        artifact_root=launch.expectation.artifact_root,
        control_root=launch.expectation.control_root,
        cache_root=launch.expectation.cache_root,
        evidence_environment_key=evidence_key,
        model_token_key=None,
    )
    state.settings = settings
    transport = ModalInferenceSdkTransport(
        facade=facade,
        binding_authority=launch.case.authority,
        handoff=handoff,
        foundation=Foundation(),
        catalog=launch.case.catalog,
        foundation_authenticator=launch.arguments["foundation_authenticator"],
        assessment_authenticator=launch.arguments["assessment_authenticator"],
        signer=launch.authenticator,
        launch_settings=settings,
        clock=launch.clock,
        issued_at=launch.arguments["issued_at"],
        expires_at=launch.arguments["expires_at"],
        evidence_ref=launch.arguments["evidence_ref"],
    )
    import tuner.execution.providers.modal.inference_transport as transport_module

    original_prepare = transport_module.prepare_modal_chat_launch
    original_snapshot = transport_module._validate_preparation_snapshot

    def snapshot_with_provider_image(value):
        checked = original_snapshot(value)
        checked["configuration"]["image"].setdefault("provider_image_id", "im-chat")
        return checked

    def capture_prepare(**kwargs):
        value = original_prepare(**kwargs)
        channel_case.envelope = value
        return value

    monkeypatch.setattr(
        transport_module, "_validate_preparation_snapshot", snapshot_with_provider_image
    )
    if not real_channel:
        monkeypatch.setattr(transport_module, "ModalInferenceChannelClient", _Channel)
    monkeypatch.setattr(transport_module, "prepare_modal_chat_launch", capture_prepare)
    if delayed_create is not None or stdin_gate is not None:
        original_bounded = transport_module._bounded

        def bound_create_quickly(
            operation, *, deadline, late_cleanup=None, ownership=None
        ):
            if (
                interrupt_create
                and len(state.secrets) == len(configuration["secrets"])
                and not state.creates
            ):
                pending = transport_module._PendingCall(
                    operation, late_cleanup, ownership
                )
                if ownership is not None:
                    ownership.pending = pending
                pending.start()
                pending.abandon()
                error = KeyboardInterrupt()
                error.pending_operation = pending
                raise error
            if (
                delayed_create is not None
                and len(state.secrets) == len(configuration["secrets"])
            ) or (stdin_gate is not None and state.creates):
                deadline = min(deadline, time.monotonic() + 0.02)
            return original_bounded(
                operation,
                deadline=deadline,
                late_cleanup=late_cleanup,
                ownership=ownership,
            )

        monkeypatch.setattr(transport_module, "_bounded", bound_create_quickly)
    return launch, transport, handoff, state, configuration


def test_submit_creates_one_exact_sandbox_and_publishes_ready_lease(monkeypatch):
    launch, transport, handoff, state, configuration = _submit_transport(monkeypatch)
    result = transport.execute_once(launch.submit_binding, launch.submit)
    assert result.disposition is ObservationDisposition.FOUND
    assert result.provider_ref == "sb-chat"
    assert len(state.creates) == 1
    args, kwargs = state.creates[0]
    assert args == (
        configuration["runtime"]["python_executable"],
        "-m",
        "tuner.execution.providers.modal.inference_entrypoint",
    )
    assert kwargs["pty"] is False
    assert kwargs["encrypted_ports"] == kwargs["h2_ports"] == []
    assert kwargs["unencrypted_ports"] == []
    assert kwargs["workdir"] == "/workspace/modal-chat"
    assert state.image.is_hydrated is True
    assert len(state.image.hydrate_calls) == 1
    lease = handoff.take(submit_command_digest=launch.submit.digest)
    assert lease.sandbox is state.sandbox
    assert handoff.peek(submit_command_digest=launch.submit.digest) is lease
    with pytest.raises(ModalInferenceTransportError):
        handoff.take(submit_command_digest=launch.submit.digest)
    assert len(state.sandbox.stdin.values) == 1


def test_constructor_owns_launch_settings_copy(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(monkeypatch)
    object.__setattr__(state.settings, "evidence_environment_key", "invalid-key")
    result = transport.execute_once(launch.submit_binding, launch.submit)
    assert result.disposition is ObservationDisposition.FOUND
    handoff.take(submit_command_digest=launch.submit.digest).close()


def test_ready_model_mismatch_cleans_exact_created_sandbox(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, wrong_model=True
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert ownership is not None and ownership.sandbox is state.sandbox


@pytest.mark.parametrize("mismatch", ("volume", "app", "image"))
def test_exact_provider_read_mismatch_is_denied_before_create(monkeypatch, mismatch):
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, mismatch=mismatch
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert state.creates == []
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_failed_stdin_start_is_not_retried_and_cleans(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, stdin_error=True
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert len(state.creates) == 1
    assert state.sandbox.stdin.values == []
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_interrupted_stdin_start_cleans_and_retains_before_propagating(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(monkeypatch)

    def interrupt(_self, _value):
        raise KeyboardInterrupt

    def cleanup_interrupt(self, *, wait):
        self.terminations.append(wait)
        raise SystemExit

    monkeypatch.setattr(_Input, "write", interrupt)
    monkeypatch.setattr(_Sandbox, "terminate", cleanup_interrupt)
    with pytest.raises(KeyboardInterrupt):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert len(state.creates) == 1
    assert state.sandbox.terminations == [True]
    assert (
        transport.pending_cleanup(submit_command_digest=launch.submit.digest).sandbox
        is state.sandbox
    )
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_failed_ready_retains_completed_channel_cleanup_lease(monkeypatch):
    original_start = RealModalInferenceChannelClient._start_termination

    def complete_before_return(channel):
        original_start(channel)
        assert channel._termination_done.wait(1.0)

    monkeypatch.setattr(
        RealModalInferenceChannelClient, "_start_termination", complete_before_return
    )
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch,
        real_channel=True,
        ready_eof=True,
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert type(ownership.channel) is RealModalInferenceChannelClient
    assert ownership.channel._sandbox is state.sandbox
    assert ownership.channel.cleanup_pending is False
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_failed_ready_retains_pending_channel_cleanup_lease(monkeypatch):
    release = threading.Event()
    close_entered = threading.Event()
    original_close = ModalChatCleanupOwnership.close

    def observed_close(ownership, **kwargs):
        if ownership.channel is not None:
            close_entered.set()
        return original_close(ownership, **kwargs)

    monkeypatch.setattr(ModalChatCleanupOwnership, "close", observed_close)
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch,
        real_channel=True,
        ready_eof=True,
        termination_gate=release,
    )
    errors = []

    def execute():
        try:
            transport.execute_once(launch.submit_binding, launch.submit)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    assert close_entered.wait(1.0)
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert type(ownership.channel) is RealModalInferenceChannelClient
    assert ownership.channel.cleanup_pending is True
    assert state.sandbox.terminations == [True]
    release.set()
    worker.join(timeout=1.0)
    assert len(errors) == 1 and type(errors[0]) is ModalInferenceTransportError
    assert ownership.channel.cleanup_pending is False
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_invalid_static_start_is_denied_before_sdk_reads(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, evidence_key="not-a-valid-key"
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert state.volumes == state.secrets == state.creates == []
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_ambiguous_stdin_start_is_not_retried_and_cleans(monkeypatch):
    release = threading.Event()
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, stdin_gate=release
    )
    with pytest.raises(ModalInferenceTransportError):
        transport.execute_once(launch.submit_binding, launch.submit)
    assert len(state.creates) == 1
    assert state.sandbox.terminations == [True]
    release.set()
    assert state.sandbox.stdin.wrote.wait(0.5)
    assert len(state.sandbox.stdin.values) == 1
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_ambiguous_create_retains_late_exact_handle_for_cleanup(monkeypatch):
    release = threading.Event()
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, delayed_create=release
    )
    result = transport.execute_once(launch.submit_binding, launch.submit)
    assert result.disposition is ObservationDisposition.INDETERMINATE
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert ownership is not None and ownership.pending is not None
    assert ownership.close() is False
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None
    release.set()
    assert state.created.wait(0.5)
    assert state.sandbox.terminated.wait(0.5)
    assert state.sandbox.terminations == [True]


def test_interrupted_create_retains_late_exact_handle_for_cleanup(monkeypatch):
    release = threading.Event()
    launch, transport, handoff, state, _ = _submit_transport(
        monkeypatch, delayed_create=release, interrupt_create=True
    )
    with pytest.raises(KeyboardInterrupt):
        transport.execute_once(launch.submit_binding, launch.submit)
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert ownership is not None and ownership.pending is not None
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None
    release.set()
    ownership.pending.worker.join(timeout=1.0)
    assert state.sandbox.terminations == [True]


def test_interrupted_worker_start_retains_precreated_ownership_cell(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(monkeypatch)
    import tuner.execution.providers.modal.inference_transport as transport_module

    original_start = transport_module._PendingCall.start

    def interrupt_after_start(pending):
        original_start(pending)
        if pending._ownership is not None:
            raise KeyboardInterrupt

    monkeypatch.setattr(transport_module._PendingCall, "start", interrupt_after_start)
    with pytest.raises(KeyboardInterrupt):
        transport.execute_once(launch.submit_binding, launch.submit)
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert ownership is not None and ownership.pending is not None
    assert state.created.wait(0.5)
    assert state.sandbox.terminated.wait(0.5)
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_interrupt_after_create_return_uses_deposited_exact_handle(monkeypatch):
    launch, transport, handoff, state, _ = _submit_transport(monkeypatch)
    import tuner.execution.providers.modal.inference_transport as transport_module

    original_bounded = transport_module._bounded

    def interrupt_return(operation, *, deadline, late_cleanup=None, ownership=None):
        value = original_bounded(
            operation,
            deadline=deadline,
            late_cleanup=late_cleanup,
            ownership=ownership,
        )
        if ownership is not None:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(transport_module, "_bounded", interrupt_return)
    with pytest.raises(KeyboardInterrupt):
        transport.execute_once(launch.submit_binding, launch.submit)
    ownership = transport.pending_cleanup(submit_command_digest=launch.submit.digest)
    assert ownership is not None and ownership.sandbox is state.sandbox
    assert state.sandbox.terminations == [True]
    assert handoff.peek(submit_command_digest=launch.submit.digest) is None


def test_cleanup_wait_is_bounded_and_retains_unresolved_attempt():
    release = threading.Event()

    class BlockingSandbox:
        object_id = "sb-cleanup"

        def __init__(self):
            self.terminations = []

        def terminate(self, *, wait):
            self.terminations.append(wait)
            release.wait()
            return 0

    sandbox = BlockingSandbox()
    ownership = ModalChatCleanupOwnership("d" * 64, sandbox=sandbox)
    assert ownership.close(term_timeout=0.01) is False
    assert ownership.pending is not None
    release.set()
    ownership.pending.worker.join(timeout=1.0)
    assert sandbox.terminations == [True]
