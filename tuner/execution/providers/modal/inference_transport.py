"""Explicit-client Modal transport for authenticated chat effects.

Foundation owns authorization and durable consumption. This module only
defines the exact ephemeral Sandbox ownership handoff used after a successful
provider attempt; it is neither a grant nor a persistent registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from inspect import getattr_static
import math
from queue import Empty, Queue
import threading
import time

from tuner.execution.foundation_v2.commands import (
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)
from tuner.execution.foundation_v2.observations import ObservationDisposition

from .coordinator_effects import ModalEffectOutcome
from .facade import ExplicitModal154ReadFacade
from .inference_channel_client import ModalInferenceChannelClient
from .inference_commands import ModalInferenceCommandBinding
from .inference_entrypoint import ModalChatStart, encode_modal_chat_start
from .inference_launch import modal_chat_stage_ref, prepare_modal_chat_launch
from .inference_preparation import _validate_preparation_snapshot
from .inference_retention import load_modal_chat_command
from .inference_wire import ModalChatWorkerExpectation

_CLOSED = "modal_inference_transport_invalid"


class ModalInferenceTransportError(RuntimeError):
    """Closed non-secret transport failure."""


class _PendingCall:
    """One finite host wait with atomic ownership of a late returned handle.

    ``_value`` retains the exact late handle as process-local recovery evidence.
    A discarded cleanup callback result is never treated as proof of shutdown.
    """

    def __init__(self, operation, late_cleanup=None, ownership=None) -> None:
        self._lock = threading.Lock()
        self._abandoned = False
        self._completed = False
        self._claimed = False
        self._successful = False
        self._value = None
        self._late_cleanup = late_cleanup
        self._ownership = ownership
        self.result = Queue(maxsize=1)

        def invoke() -> None:
            try:
                value = operation()
                successful = True
            except BaseException as error:
                value = error
                successful = False
            cleanup = None
            with self._lock:
                self._completed = True
                self._successful = successful
                self._value = value
                if successful and self._ownership is not None:
                    self._ownership.sandbox = value
                if self._abandoned and successful and not self._claimed:
                    self._claimed = True
                    cleanup = value
            self.result.put((successful, value))
            if cleanup is not None and self._late_cleanup is not None:
                try:
                    self._late_cleanup(cleanup)
                except BaseException:
                    pass

        self.worker = threading.Thread(
            target=invoke, name="modal-chat-provider-operation", daemon=True
        )

    def start(self) -> None:
        self.worker.start()

    def abandon(self) -> None:
        cleanup = None
        with self._lock:
            self._abandoned = True
            if self._completed and self._successful and not self._claimed:
                self._claimed = True
                cleanup = self._value
        if cleanup is not None and self._late_cleanup is not None:
            try:
                self._late_cleanup(cleanup)
            except BaseException:
                pass


@dataclass(slots=True)
class ModalChatCleanupOwnership:
    """Non-secret ownership retained when SUBMIT cannot publish a ready lease."""

    submit_command_digest: str
    sandbox: object | None = None
    pending: _PendingCall | None = None
    channel: ModalInferenceChannelClient | None = None
    _close_lock: object = field(default_factory=threading.Lock, repr=False)
    _close_started: bool = field(default=False, init=False, repr=False)

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        if self.channel is not None:
            return self.channel.close(
                term_timeout=term_timeout, kill_timeout=kill_timeout
            )
        if self.sandbox is None:
            return False
        try:
            if (
                type(term_timeout) not in (int, float)
                or type(term_timeout) is bool
                or not math.isfinite(float(term_timeout))
                or term_timeout <= 0
            ):
                raise ValueError("exact positive cleanup timeout required")
            with self._close_lock:
                if self._close_started:
                    return False
                self._close_started = True
            sandbox = self.sandbox
            object_id = sandbox.object_id
            terminate = sandbox.terminate

            def terminate_exact():
                if sandbox.object_id != object_id:
                    raise ValueError
                return terminate(wait=True)

            result = _bounded(
                terminate_exact,
                deadline=time.monotonic() + float(term_timeout),
            )
            return type(result) is int
        except TimeoutError as error:
            self.pending = getattr(error, "pending_operation", self.pending)
            return False
        except Exception:
            return False


def _bounded(operation, *, deadline: float, late_cleanup=None, ownership=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    pending = _PendingCall(operation, late_cleanup, ownership)
    if ownership is not None:
        ownership.pending = pending
    try:
        pending.start()
        successful, value = pending.result.get(timeout=remaining)
    except Empty:
        pending.abandon()
        error = TimeoutError()
        error.pending_operation = pending  # type: ignore[attr-defined]
        raise error from None
    except BaseException as error:
        pending.abandon()
        try:
            error.pending_operation = pending  # type: ignore[attr-defined]
        except Exception:
            pass
        raise
    if successful:
        return value
    if isinstance(value, (KeyboardInterrupt, SystemExit)):
        raise value
    raise ModalInferenceTransportError(_CLOSED) from None


def _utc_seconds(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError
    return parsed.astimezone(timezone.utc).timestamp()


@dataclass(frozen=True, slots=True)
class ModalChatLaunchSettings:
    """Constructor-frozen launch fields that do not depend on the STAGE receipt."""

    configuration_bytes: bytes
    issuer_ref: str
    audience_ref: str
    key_ref: str
    challenge_nonce: str
    artifact_root: str
    control_root: str
    cache_root: str
    evidence_environment_key: str
    model_token_key: str | None

    def __post_init__(self) -> None:
        if type(self.configuration_bytes) is not bytes or not self.configuration_bytes:
            raise TypeError("exact nonempty configuration bytes required")
        for name in (
            "issuer_ref",
            "audience_ref",
            "key_ref",
            "challenge_nonce",
            "artifact_root",
            "control_root",
            "cache_root",
            "evidence_environment_key",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"exact nonempty {name} required")
        if self.model_token_key is not None and (
            type(self.model_token_key) is not str or not self.model_token_key
        ):
            raise TypeError("exact model token key required")

    def expectation_for(self, command: SubmitCommandV2) -> ModalChatWorkerExpectation:
        if type(command) is not SubmitCommandV2:
            raise TypeError("exact SUBMIT command required")
        return ModalChatWorkerExpectation(
            configuration_bytes=self.configuration_bytes,
            submit_command_digest=command.digest,
            executor_version=command.executor.implementation_version,
            issuer_ref=self.issuer_ref,
            audience_ref=self.audience_ref,
            key_ref=self.key_ref,
            challenge_nonce=self.challenge_nonce,
            artifact_root=self.artifact_root,
            control_root=self.control_root,
            cache_root=self.cache_root,
        )


@dataclass(frozen=True, slots=True)
class ModalChatSandboxLease:
    """Exact newly-created Sandbox and its already-bound channel ownership."""

    submit_command_digest: str
    session_id: str
    argument_bytes: bytes
    deadline: float
    preparation_snapshot: bytes
    sandbox: object
    channel: ModalInferenceChannelClient

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.submit_command_digest, self.session_id)
        ):
            raise TypeError("exact nonempty lease references required")
        if any(
            type(value) is not bytes or not value
            for value in (self.argument_bytes, self.preparation_snapshot)
        ):
            raise TypeError("exact nonempty lease bytes required")
        if (
            type(self.deadline) not in (int, float)
            or type(self.deadline) is bool
            or not math.isfinite(float(self.deadline))
            or self.deadline <= 0
        ):
            raise TypeError("exact lease deadline required")
        if type(self.channel) is not ModalInferenceChannelClient:
            raise TypeError("exact Modal channel client required")
        object_id = getattr_static(type(self.sandbox), "object_id", None)
        terminate = getattr_static(type(self.sandbox), "terminate", None)
        if object_id is None or terminate is None or not callable(terminate):
            raise TypeError("exact owned Sandbox surface required")
        if (
            getattr(self.channel, "_sandbox", None) is not self.sandbox
            or getattr(self.channel, "_object_id", None) != self.sandbox.object_id
            or getattr(self.channel, "_session_id", None) != self.session_id
            or getattr(self.channel, "_launch_digest", None)
            != hashlib.sha256(self.argument_bytes).hexdigest()
            or getattr(self.channel, "_deadline", None) != self.deadline
        ):
            raise TypeError("Modal channel lease binding differs")

    @property
    def cleanup_pending(self) -> bool:
        return self.channel.cleanup_pending

    def close(self, *, term_timeout: float = 5.0, kill_timeout: float = 5.0) -> bool:
        return self.channel.close(
            term_timeout=term_timeout,
            kill_timeout=kill_timeout,
        )


class ModalChatLeaseHandoff:
    """Single-slot one-use ownership handoff with non-consuming lookup."""

    __slots__ = ("_lock", "_value", "_taken")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: ModalChatSandboxLease | None = None
        self._taken = False

    def publish(self, lease: ModalChatSandboxLease) -> None:
        if type(lease) is not ModalChatSandboxLease:
            raise TypeError("exact Modal chat lease required")
        with self._lock:
            if self._value is not None:
                raise ModalInferenceTransportError(_CLOSED)
            self._value = lease

    def peek(self, *, submit_command_digest: str) -> ModalChatSandboxLease | None:
        if type(submit_command_digest) is not str or not submit_command_digest:
            raise TypeError("exact submit command digest required")
        with self._lock:
            value = self._value
            if value is None or value.submit_command_digest != submit_command_digest:
                return None
            return value

    def take(self, *, submit_command_digest: str) -> ModalChatSandboxLease:
        if type(submit_command_digest) is not str or not submit_command_digest:
            raise TypeError("exact submit command digest required")
        with self._lock:
            value = self._value
            if (
                value is None
                or self._taken
                or value.submit_command_digest != submit_command_digest
            ):
                raise ModalInferenceTransportError(_CLOSED)
            self._taken = True
            return value


class ModalInferenceSdkTransport:
    """Authenticated one-attempt Modal chat Sandbox transport."""

    __slots__ = (
        "_facade",
        "_authority",
        "_handoff",
        "_scope",
        "_foundation",
        "_catalog",
        "_foundation_authenticator",
        "_assessment_authenticator",
        "_signer",
        "_launch_settings",
        "_clock",
        "_issued_at",
        "_expires_at",
        "_evidence_ref",
        "_cleanup",
        "_lock",
    )

    def __init__(
        self,
        *,
        facade: ExplicitModal154ReadFacade,
        binding_authority: object,
        handoff: ModalChatLeaseHandoff,
        foundation: object | None = None,
        catalog: object | None = None,
        foundation_authenticator: object | None = None,
        assessment_authenticator: object | None = None,
        signer: object | None = None,
        launch_settings: ModalChatLaunchSettings | None = None,
        clock: object | None = None,
        issued_at: str | None = None,
        expires_at: str | None = None,
        evidence_ref: str | None = None,
    ) -> None:
        if type(facade) is not ExplicitModal154ReadFacade:
            raise TypeError("exact explicit Modal facade required")
        if type(handoff) is not ModalChatLeaseHandoff:
            raise TypeError("exact Modal chat lease handoff required")
        if (
            launch_settings is not None
            and type(launch_settings) is not ModalChatLaunchSettings
        ):
            raise TypeError("exact Modal chat launch settings required")
        authenticate = getattr_static(type(binding_authority), "authenticate", None)
        if (
            authenticate is None
            or not callable(authenticate)
            or getattr_static(binding_authority, "authenticate", None)
            is not authenticate
        ):
            raise TypeError("exact static binding authority required")
        binding = facade.binding
        scope = (
            binding.account_ref,
            binding.workspace_ref,
            binding.environment_ref,
            binding.client_ref,
        )
        self._facade = facade
        self._authority = binding_authority
        self._handoff = handoff
        self._scope = scope
        self._foundation = foundation
        self._catalog = catalog
        self._foundation_authenticator = foundation_authenticator
        self._assessment_authenticator = assessment_authenticator
        self._signer = signer
        self._launch_settings = (
            None
            if launch_settings is None
            else ModalChatLaunchSettings(
                **{
                    name: getattr(launch_settings, name)
                    for name in launch_settings.__dataclass_fields__
                }
            )
        )
        self._clock = clock
        self._issued_at = issued_at
        self._expires_at = expires_at
        self._evidence_ref = evidence_ref
        self._cleanup: dict[str, ModalChatCleanupOwnership] = {}
        self._lock = threading.Lock()

    def pending_cleanup(
        self, *, submit_command_digest: str
    ) -> ModalChatCleanupOwnership | None:
        with self._lock:
            return self._cleanup.get(submit_command_digest)

    def _retain_cleanup(self, digest: str, ownership: ModalChatCleanupOwnership):
        with self._lock:
            return self._cleanup.setdefault(digest, ownership)

    def _release_cleanup(
        self, digest: str, ownership: ModalChatCleanupOwnership
    ) -> None:
        with self._lock:
            if self._cleanup.get(digest) is ownership:
                del self._cleanup[digest]

    def _submit(self, owned, parsed) -> ModalEffectOutcome:
        sandbox = None
        channel = None
        deadline = 0.0
        try:
            dependencies = (
                self._foundation,
                self._catalog,
                self._foundation_authenticator,
                self._assessment_authenticator,
                self._signer,
                self._launch_settings,
                self._clock,
            )
            if any(value is None for value in dependencies) or any(
                type(value) is not str or not value
                for value in (
                    self._issued_at,
                    self._expires_at,
                    self._evidence_ref,
                )
            ):
                raise ValueError
            snapshot = _validate_preparation_snapshot(owned.preparation_snapshot)
            configuration = snapshot["configuration"]
            policy = configuration["policy"]
            # Anchor the only provider lifetime before the first live read.
            started_mono = time.monotonic()
            wall_now = _utc_seconds(self._clock.now_iso())
            claim_remaining = _utc_seconds(self._expires_at) - wall_now
            lifetime = min(policy["absolute_lifetime_seconds"], claim_remaining)
            if lifetime <= 0:
                raise ValueError
            deadline = started_mono + lifetime

            stage_record = self._foundation.get(
                parsed.stage_predecessor.stage_effect_id
            )
            if stage_record is None:
                raise ValueError
            stage_command = parse_exact_command(stage_record.command_bytes)
            if type(stage_command) is not StageCommandV2:
                raise ValueError
            stage_binding = load_modal_chat_command(
                stage_command.digest,
                catalog=self._catalog,
                authority=self._authority,
            )
        except Exception:
            raise ModalInferenceTransportError(_CLOSED) from None

        try:
            stage_assessment = self._foundation.assess(stage_record)
            expectation = self._launch_settings.expectation_for(parsed)
            envelope = prepare_modal_chat_launch(
                stage_binding=stage_binding,
                submit_binding=owned,
                stage_record=stage_record,
                stage_assessment=stage_assessment,
                foundation_authenticator=self._foundation_authenticator,
                assessment_authenticator=self._assessment_authenticator,
                binding_authority=self._authority,
                signer=self._signer,
                expectation=expectation,
                clock=self._clock,
                issued_at=self._issued_at,
                expires_at=self._expires_at,
                evidence_ref=self._evidence_ref,
            )
            evidence_key = self._launch_settings.evidence_environment_key
            model_key = self._launch_settings.model_token_key
            declared_keys = [
                key
                for secret in configuration["secrets"]
                for key in secret["required_keys"]
            ]
            expected_keys = [evidence_key] + ([] if model_key is None else [model_key])
            if len(declared_keys) != len(expected_keys) or set(declared_keys) != set(
                expected_keys
            ):
                raise ValueError
            start = encode_modal_chat_start(
                ModalChatStart(
                    expectation,
                    envelope.argument_bytes,
                    evidence_key,
                    model_key,
                )
            )
            if _bounded(self._facade.bound_scope, deadline=deadline) != self._scope:
                raise ValueError
            volumes_config = configuration["volumes"]
            volume_pairs = (
                ("source_artifact", expectation.artifact_root),
                ("chat_control", expectation.control_root),
                ("model_cache", expectation.cache_root),
            )
            mounts = {}
            for prefix, mount in volume_pairs:
                volume_id = volumes_config[prefix + "_volume_id"]
                if (
                    self._facade.volume_name(volume_id)
                    != volumes_config[prefix + "_volume_ref"]
                ):
                    raise ValueError
                mounts[mount] = _bounded(
                    lambda value=volume_id: self._facade._volume(value),
                    deadline=deadline,
                )
            sdk = self._facade.sdk
            client = self._facade.client
            app_config = configuration["application"]
            app = _bounded(
                lambda: sdk.App.lookup(
                    app_config["app_name"],
                    environment_name=self._facade.binding.environment_ref,
                    create_if_missing=False,
                    client=client,
                ),
                deadline=deadline,
            )
            if getattr(app, "app_id", None) != app_config["app_ref"]:
                raise ValueError
            image_config = configuration["image"]
            image_id = image_config["provider_image_id"]
            image = _bounded(
                lambda: sdk.Image.from_id(image_id, client=client),
                deadline=deadline,
            )
            _bounded(lambda: image.hydrate(client), deadline=deadline)
            if (
                getattr(image, "is_hydrated", False) is not True
                or getattr(image, "object_id", None) != image_id
            ):
                raise ValueError
            secrets = []
            for secret in configuration["secrets"]:
                keys = list(secret["required_keys"])
                secrets.append(
                    _bounded(
                        lambda secret=secret, keys=keys: sdk.Secret.from_name(
                            secret["name"],
                            environment_name=self._facade.binding.environment_ref,
                            required_keys=keys,
                            client=client,
                        ),
                        deadline=deadline,
                    )
                )
            resources = configuration["resources"]
            command = (
                configuration["runtime"]["python_executable"],
                "-m",
                "tuner.execution.providers.modal.inference_entrypoint",
            )
            gpu = f'{resources["accelerator"]}:{resources["accelerator_count"]}'
            create = lambda: sdk.Sandbox.create(
                *command,
                app=app,
                image=image,
                secrets=secrets,
                volumes=mounts,
                gpu=gpu,
                cpu=resources["cpu_millicores"] / 1000,
                memory=resources["memory_mb"],
                timeout=resources["provider_timeout_seconds"],
                idle_timeout=resources["provider_idle_timeout_seconds"],
                workdir="/workspace/modal-chat",
                pty=False,
                encrypted_ports=[],
                h2_ports=[],
                unencrypted_ports=[],
                client=client,
            )
            create_ownership = ModalChatCleanupOwnership(parsed.digest)
            self._retain_cleanup(parsed.digest, create_ownership)
            try:
                sandbox = _bounded(
                    create,
                    deadline=deadline,
                    late_cleanup=lambda _handle: create_ownership.close(),
                    ownership=create_ownership,
                )
            except (KeyboardInterrupt, SystemExit) as error:
                raise
            except TimeoutError as error:
                pending = getattr(error, "pending_operation", None)
                if create_ownership.pending is None:
                    create_ownership.pending = pending
                return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)
            object_id = sandbox.object_id
            if type(object_id) is not str or not object_id:
                raise ValueError

            def write_start():
                if sandbox.stdin.write(start) is not None:
                    raise ValueError
                if sandbox.stdin.drain() is not None:
                    raise ValueError

            _bounded(write_start, deadline=deadline)
            try:
                channel = ModalInferenceChannelClient(
                    sandbox,
                    session_id=snapshot["chat_input"]["session_id"],
                    argument_bytes=envelope.argument_bytes,
                    deadline=deadline,
                    startup_timeout_seconds=policy["startup_timeout_seconds"],
                    request_timeout_seconds=policy["request_timeout_seconds"],
                    max_request_bytes=policy["max_request_bytes"],
                    max_response_bytes=policy["max_response_bytes"],
                )
            except BaseException as error:
                cleanup_lease = getattr(error, "cleanup_lease", None)
                if (
                    type(cleanup_lease) is ModalInferenceChannelClient
                    and getattr(cleanup_lease, "_sandbox", None) is sandbox
                    and getattr(cleanup_lease, "_object_id", None) == sandbox.object_id
                ):
                    channel = cleanup_lease
                raise
            workload = snapshot["chat_input"]["workload"]
            model = channel.model
            if (
                model.model_ref != workload["model_ref"]
                or model.model_revision != workload["model_revision"]
                or model.tokenizer_revision != workload["tokenizer_revision"]
            ):
                raise ValueError
            lease = ModalChatSandboxLease(
                parsed.digest,
                snapshot["chat_input"]["session_id"],
                envelope.argument_bytes,
                deadline,
                owned.preparation_snapshot,
                sandbox,
                channel,
            )
            self._handoff.publish(lease)
            self._release_cleanup(parsed.digest, create_ownership)
            return ModalEffectOutcome(ObservationDisposition.FOUND, object_id)
        except (KeyboardInterrupt, SystemExit) as control_error:
            ownership = self.pending_cleanup(submit_command_digest=parsed.digest)
            if ownership is None:
                ownership = ModalChatCleanupOwnership(parsed.digest)
                ownership = self._retain_cleanup(parsed.digest, ownership)
            if sandbox is not None:
                ownership.sandbox = sandbox
            if channel is not None:
                ownership.channel = channel
            if ownership.sandbox is not None:
                try:
                    ownership.close()
                except BaseException:
                    pass
            raise control_error
        except Exception:
            ownership = self.pending_cleanup(submit_command_digest=parsed.digest)
            if ownership is None:
                ownership = ModalChatCleanupOwnership(parsed.digest)
                ownership = self._retain_cleanup(parsed.digest, ownership)
            if sandbox is not None:
                ownership.sandbox = sandbox
            if channel is not None:
                ownership.channel = channel
            if ownership.sandbox is not None:
                ownership.close()
            raise ModalInferenceTransportError(_CLOSED) from None

    def _binding(self, supplied: object, supplied_command: object):
        if type(supplied) is not ModalInferenceCommandBinding:
            raise ModalInferenceTransportError(_CLOSED)
        try:
            supplied_bytes = (supplied.command_bytes, supplied.preparation_snapshot)
            supplied_command_bytes = supplied_command.canonical_bytes
            owned = ModalInferenceCommandBinding(
                *supplied_bytes,
            )
            command = parse_exact_command(owned.command_bytes)
            preparation = command.preparation
            snapshot = _validate_preparation_snapshot(owned.preparation_snapshot)
            configuration = snapshot["configuration"]
            actual_scope = (
                preparation.scope.account_ref,
                self._facade.binding.workspace_ref,
                self._facade.binding.environment_ref,
                self._facade.binding.client_ref,
            )
            authenticated = self._authority.authenticate(owned)
            if (
                type(command) is not type(supplied_command)
                or command.canonical_bytes != supplied_command.canonical_bytes
                or authenticated is not True
                or (supplied.command_bytes, supplied.preparation_snapshot)
                != supplied_bytes
                or (owned.command_bytes, owned.preparation_snapshot) != supplied_bytes
                or supplied_command.canonical_bytes != supplied_command_bytes
                or actual_scope != self._scope
                or preparation.provider.provider_id != "modal"
                or configuration["provider"]["profile_ref"]
                != preparation.provider.profile_ref
                or tuple(
                    configuration["client"][name]
                    for name in (
                        "account_ref",
                        "workspace_ref",
                        "environment_ref",
                        "client_ref",
                    )
                )
                != self._scope
            ):
                raise ValueError
            result = ModalInferenceCommandBinding(*supplied_bytes)
            reparsed = parse_exact_command(result.command_bytes)
            _validate_preparation_snapshot(result.preparation_snapshot)
            if reparsed.canonical_bytes != supplied_command_bytes:
                raise ValueError
            return result, reparsed
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise ModalInferenceTransportError(_CLOSED) from None

    def execute_once(self, binding: object, command: object) -> ModalEffectOutcome:
        owned, parsed = self._binding(binding, command)
        if type(parsed) is StageCommandV2:
            return ModalEffectOutcome(
                ObservationDisposition.FOUND,
                modal_chat_stage_ref(owned),
            )
        if type(parsed) is SubmitCommandV2:
            return self._submit(owned, parsed)
        raise ModalInferenceTransportError(_CLOSED)

    def lookup_once(self, binding: object, command: object) -> ModalEffectOutcome:
        owned, parsed = self._binding(binding, command)
        if type(parsed) is StageCommandV2:
            return ModalEffectOutcome(
                ObservationDisposition.FOUND,
                modal_chat_stage_ref(owned),
            )
        if type(parsed) is SubmitCommandV2:
            lease = self._handoff.peek(submit_command_digest=parsed.digest)
            if lease is not None:
                object_id = lease.sandbox.object_id
                if type(object_id) is str and object_id:
                    return ModalEffectOutcome(ObservationDisposition.FOUND, object_id)
        return ModalEffectOutcome(ObservationDisposition.INDETERMINATE)


__all__: list[str] = []
