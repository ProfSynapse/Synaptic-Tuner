"""Pure remote admission for a host-authenticated Modal chat launch."""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import json
from datetime import timedelta
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from tuner.execution.evidence import (
    DEPLOYMENT_EVIDENCE_POLICY,
    canonical_utc,
    parse_utc,
    validate_evidence_window,
)
from tuner.execution.foundation_v2.canonical import (
    MAX_CANONICAL_BYTES,
    canonical_bytes,
    digest_text,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import (
    StageCommandV2,
    SubmitCommandV2,
    parse_exact_command,
)

from .inference_commands import ModalInferenceCommandBinding
from .inference_preparation import (
    ModalInferencePreparationConfig,
    _validate_preparation_snapshot,
)

_PURPOSE = "modal-inference-launch/v1"
_CLAIM_SCHEMA = "synaptic-modal-inference-launch/v1"
_ARGUMENT_SCHEMA = "synaptic-modal-inference-launch-argument/v1"
_MAX_ARGUMENT_BYTES = 96 * 1024
_CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "stage_ref",
        "stage_binding_digest",
        "stage_command_digest",
        "stage_effect_id",
        "stage_invocation_nonce",
        "submit_binding_digest",
        "submit_command_digest",
        "submit_effect_id",
        "submit_invocation_nonce",
        "preparation_snapshot_sha256",
        "configuration_digest",
        "session_id",
        "stage_record_digest",
        "stage_assessment_digest",
        "stage_foundation_binding_digest",
        "stage_outcome_digest",
        "stage_bound_reference_digest",
        "stage_authenticated_receipt_digest",
        "issued_at",
        "expires_at",
        "evidence_ref",
        "issuer_ref",
        "audience_ref",
        "key_ref",
        "challenge_nonce",
        "artifact_root",
        "control_root",
        "cache_root",
    }
)


class _Verifier(Protocol):
    def verify(
        self, purpose: str, payload: bytes, tag: bytes, key_ref: str
    ) -> bool: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ModalChatWorkerExpectation:
    """Static worker configuration, independent of the submitted launch claim."""

    configuration_bytes: bytes
    submit_command_digest: str
    executor_version: str
    issuer_ref: str
    audience_ref: str
    key_ref: str
    challenge_nonce: str
    artifact_root: str
    control_root: str
    cache_root: str

    def __post_init__(self) -> None:
        if type(self.configuration_bytes) is not bytes:
            raise TypeError("exact inference configuration bytes required")
        ModalInferencePreparationConfig.parse(self.configuration_bytes)
        if type(self.submit_command_digest) is not str:
            raise TypeError("submit_command_digest must be an exact string")
        digest_text(self.submit_command_digest, "submit_command_digest")
        for name, value in (
            ("executor_version", self.executor_version),
            ("issuer_ref", self.issuer_ref),
            ("audience_ref", self.audience_ref),
            ("key_ref", self.key_ref),
            ("challenge_nonce", self.challenge_nonce),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact string")
            safe_ref(value, name)
        roots = tuple(
            _root(value, name)
            for value, name in (
                (self.artifact_root, "artifact_root"),
                (self.control_root, "control_root"),
                (self.cache_root, "cache_root"),
            )
        )
        if any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(roots)
            for b in roots[i + 1 :]
        ):
            raise ValueError(
                "Modal chat mount roots must be distinct and nonoverlapping"
            )

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalChatWorkerExpectation is final")


class ModalChatWorkerAdmission:
    """Factory-minted immutable result of pure launch admission."""

    __slots__ = (
        "_argument_bytes",
        "_claim",
        "_claim_tag",
        "_stage_command_bytes",
        "_submit_command_bytes",
        "_preparation_snapshot",
    )

    def __init__(self, *args, **kwargs):
        raise TypeError("Modal chat worker admissions are factory issued")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalChatWorkerAdmission is final")

    def __setattr__(self, name, value):
        raise AttributeError("Modal chat worker admissions are immutable")

    @classmethod
    def _create(cls, values: tuple[bytes, ...]):
        owned = object.__new__(cls)
        for name, value in zip(cls.__slots__, values, strict=True):
            object.__setattr__(owned, name, bytes(value))
        return owned

    @property
    def argument_bytes(self) -> bytes:
        self._validate()
        return bytes(self._argument_bytes)

    @property
    def claim(self) -> bytes:
        self._validate()
        return bytes(self._claim)

    @property
    def claim_tag(self) -> bytes:
        self._validate()
        return bytes(self._claim_tag)

    @property
    def stage_command_bytes(self) -> bytes:
        self._validate()
        return bytes(self._stage_command_bytes)

    @property
    def submit_command_bytes(self) -> bytes:
        self._validate()
        return bytes(self._submit_command_bytes)

    @property
    def preparation_snapshot(self) -> bytes:
        self._validate()
        return bytes(self._preparation_snapshot)

    @property
    def stage_command(self) -> StageCommandV2:
        self._validate()
        return parse_exact_command(self._stage_command_bytes)

    @property
    def submit_command(self) -> SubmitCommandV2:
        self._validate()
        return parse_exact_command(self._submit_command_bytes)

    @property
    def configuration(self) -> ModalInferencePreparationConfig:
        self._validate()
        snapshot = _validate_preparation_snapshot(self._preparation_snapshot)
        return ModalInferencePreparationConfig.build(snapshot["configuration"])

    def _validate(self) -> None:
        values = tuple(getattr(self, name) for name in self.__slots__)
        if any(type(value) is not bytes or not value for value in values):
            raise TypeError("Modal chat worker admission bytes changed")
        if _encode_argument(*values[1:]) != values[0]:
            raise ValueError("Modal chat worker admission changed")
        ModalInferenceCommandBinding(values[3], values[5])
        ModalInferenceCommandBinding(values[4], values[5])


def _root(value: object, name: str) -> PurePosixPath:
    if (
        type(value) is not str
        or len(value) > 4096
        or value == "/"
        or not value.startswith("/")
        or "//" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{name} must be a canonical absolute POSIX path")
    return path


def _json_bytes(document: dict[str, object]) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _encode_argument(
    claim: bytes,
    claim_tag: bytes,
    stage_command_bytes: bytes,
    submit_command_bytes: bytes,
    preparation_snapshot: bytes,
) -> bytes:
    values = (
        claim,
        claim_tag,
        stage_command_bytes,
        submit_command_bytes,
        preparation_snapshot,
    )
    maxima = (
        MAX_CANONICAL_BYTES,
        128,
        MAX_CANONICAL_BYTES,
        MAX_CANONICAL_BYTES,
        MAX_CANONICAL_BYTES,
    )
    if any(
        type(value) is not bytes or not value or len(value) > maximum
        for value, maximum in zip(values, maxima, strict=True)
    ):
        raise TypeError("exact nonempty launch argument bytes required")
    document = {
        "schema_version": _ARGUMENT_SCHEMA,
        "claim": _b64(claim),
        "claim_tag": _b64(claim_tag),
        "stage_command": _b64(stage_command_bytes),
        "submit_command": _b64(submit_command_bytes),
        "preparation_snapshot": _b64(preparation_snapshot),
    }
    encoded = _json_bytes(document)
    if len(encoded) > _MAX_ARGUMENT_BYTES:
        raise ValueError("Modal chat launch argument exceeds its bound")
    return encoded


def _decode(value: object, name: str, maximum: int) -> bytes:
    if type(value) is not str or not value or len(value) > 4 * maximum // 3 + 8:
        raise ValueError(f"{name} exceeds its bound")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError(f"{name} is not strict base64") from None
    if not decoded or len(decoded) > maximum or _b64(decoded) != value:
        raise ValueError(f"{name} is not canonical bounded base64")
    return decoded


def _clock_now(clock) -> str:
    method = inspect.getattr_static(type(clock), "now_iso", None)
    if (
        method is None
        or not callable(method)
        or inspect.getattr_static(clock, "now_iso", None) is not method
    ):
        raise TypeError("exact static clock now_iso method required")
    value = clock.now_iso()
    if type(value) is not str:
        raise TypeError("clock now_iso must return an exact string")
    return canonical_utc(value, "now")


def admit_modal_chat_launch(
    argument: bytes,
    *,
    expectation: ModalChatWorkerExpectation,
    verifier: _Verifier,
    clock,
) -> ModalChatWorkerAdmission:
    """Admit a signed chat launch without filesystem or provider activity."""
    try:
        if type(expectation) is not ModalChatWorkerExpectation:
            raise TypeError("exact Modal chat worker expectation required")
        expectation_values = tuple(
            getattr(expectation, name) for name in expectation.__dataclass_fields__
        )
        owned_expectation = ModalChatWorkerExpectation(
            *(),
            **dict(
                zip(expectation.__dataclass_fields__, expectation_values, strict=True)
            ),
        )
        if (
            type(argument) is not bytes
            or not argument
            or len(argument) > _MAX_ARGUMENT_BYTES
        ):
            raise ValueError("Modal chat launch argument exceeds its bound")

        def object_hook(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON field")
                result[key] = value
            return result

        document = json.loads(
            argument.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("invalid JSON constant")
            ),
        )
        expected_argument_fields = {
            "schema_version",
            "claim",
            "claim_tag",
            "stage_command",
            "submit_command",
            "preparation_snapshot",
        }
        if (
            type(document) is not dict
            or set(document) != expected_argument_fields
            or document["schema_version"] != _ARGUMENT_SCHEMA
            or _json_bytes(document) != argument
        ):
            raise ValueError("Modal chat launch argument is not exact canonical JSON")
        claim = _decode(document["claim"], "launch claim", MAX_CANONICAL_BYTES)
        claim_tag = _decode(document["claim_tag"], "launch tag", 128)
        stage_bytes = _decode(
            document["stage_command"], "STAGE command", MAX_CANONICAL_BYTES
        )
        submit_bytes = _decode(
            document["submit_command"], "SUBMIT command", MAX_CANONICAL_BYTES
        )
        snapshot_bytes = _decode(
            document["preparation_snapshot"],
            "preparation snapshot",
            MAX_CANONICAL_BYTES,
        )
        claim_document = parse_canonical_object(claim, name="Modal chat launch claim")
        if (
            set(claim_document) != _CLAIM_FIELDS
            or claim_document["schema_version"] != _CLAIM_SCHEMA
        ):
            raise ValueError("Modal chat launch claim fields are invalid")
        for name in _CLAIM_FIELDS - {"schema_version"}:
            if type(claim_document[name]) is not str:
                raise TypeError(f"{name} must be an exact string")
        for name in (
            "stage_binding_digest",
            "stage_command_digest",
            "submit_binding_digest",
            "submit_command_digest",
            "preparation_snapshot_sha256",
            "configuration_digest",
            "stage_record_digest",
            "stage_assessment_digest",
            "stage_foundation_binding_digest",
            "stage_outcome_digest",
            "stage_bound_reference_digest",
            "stage_authenticated_receipt_digest",
        ):
            digest_text(claim_document[name], name)
        for name in (
            "stage_ref",
            "stage_effect_id",
            "stage_invocation_nonce",
            "submit_effect_id",
            "submit_invocation_nonce",
            "session_id",
            "evidence_ref",
            "issuer_ref",
            "audience_ref",
            "key_ref",
            "challenge_nonce",
        ):
            safe_ref(claim_document[name], name)
        roots = tuple(
            _root(claim_document[name], name)
            for name in ("artifact_root", "control_root", "cache_root")
        )
        if any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(roots)
            for b in roots[i + 1 :]
        ):
            raise ValueError("claim mount roots overlap")
        verifier_method = inspect.getattr_static(type(verifier), "verify", None)
        if (
            verifier_method is None
            or not callable(verifier_method)
            or inspect.getattr_static(verifier, "verify", None) is not verifier_method
        ):
            raise TypeError("exact static evidence verifier required")
        if (
            verifier.verify(_PURPOSE, claim, claim_tag, owned_expectation.key_ref)
            is not True
        ):
            raise ValueError("Modal chat launch authentication failed")
        if (
            tuple(
                getattr(expectation, name) for name in expectation.__dataclass_fields__
            )
            != expectation_values
        ):
            raise ValueError("Modal chat worker expectation changed")
        now = _clock_now(clock)
        if (
            tuple(
                getattr(expectation, name) for name in expectation.__dataclass_fields__
            )
            != expectation_values
        ):
            raise ValueError("Modal chat worker expectation changed")
        validate_evidence_window(
            verified_at=claim_document["issued_at"],
            expires_at=claim_document["expires_at"],
            now=now,
            policy=DEPLOYMENT_EVIDENCE_POLICY,
        )
        stage_binding = ModalInferenceCommandBinding(stage_bytes, snapshot_bytes)
        submit_binding = ModalInferenceCommandBinding(submit_bytes, snapshot_bytes)
        stage = parse_exact_command(stage_bytes)
        submit = parse_exact_command(submit_bytes)
        if type(stage) is not StageCommandV2 or type(submit) is not SubmitCommandV2:
            raise ValueError("launch requires exact STAGE and SUBMIT commands")
        snapshot = _validate_preparation_snapshot(snapshot_bytes)
        configuration_bytes = canonical_bytes(snapshot["configuration"])
        configuration = ModalInferencePreparationConfig.parse(configuration_bytes)
        absolute_lifetime = configuration.document["policy"][
            "absolute_lifetime_seconds"
        ]
        if parse_utc(claim_document["expires_at"]) - parse_utc(
            claim_document["issued_at"]
        ) > timedelta(seconds=absolute_lifetime):
            raise ValueError("launch lifetime exceeds chat policy")
        expected = (
            "modal-chat-stage:" + stage_binding.binding_digest,
            stage_binding.binding_digest,
            stage.digest,
            stage.operation.effect.effect_id,
            stage.operation.invocation_nonce,
            submit_binding.binding_digest,
            submit.digest,
            submit.operation.effect.effect_id,
            submit.operation.invocation_nonce,
            hashlib.sha256(snapshot_bytes).hexdigest(),
            snapshot["configuration_digest"],
            snapshot["chat_input"]["session_id"],
            owned_expectation.issuer_ref,
            owned_expectation.audience_ref,
            owned_expectation.key_ref,
            owned_expectation.challenge_nonce,
            owned_expectation.artifact_root,
            owned_expectation.control_root,
            owned_expectation.cache_root,
        )
        actual = (
            claim_document["stage_ref"],
            claim_document["stage_binding_digest"],
            claim_document["stage_command_digest"],
            claim_document["stage_effect_id"],
            claim_document["stage_invocation_nonce"],
            claim_document["submit_binding_digest"],
            claim_document["submit_command_digest"],
            claim_document["submit_effect_id"],
            claim_document["submit_invocation_nonce"],
            claim_document["preparation_snapshot_sha256"],
            claim_document["configuration_digest"],
            claim_document["session_id"],
            claim_document["issuer_ref"],
            claim_document["audience_ref"],
            claim_document["key_ref"],
            claim_document["challenge_nonce"],
            claim_document["artifact_root"],
            claim_document["control_root"],
            claim_document["cache_root"],
        )
        if (
            actual != expected
            or configuration_bytes != owned_expectation.configuration_bytes
            or submit.digest != owned_expectation.submit_command_digest
            or submit.executor.implementation_version
            != owned_expectation.executor_version
            or stage.executor != submit.executor
            or stage.preparation != submit.preparation
        ):
            raise ValueError("Modal chat launch differs from worker expectation")
        predecessor = submit.stage_predecessor
        preparation = submit.preparation
        if tuple(
            getattr(predecessor, name) for name in predecessor.__dataclass_fields__
        ) != (
            preparation.provider.provider_id,
            preparation.provider.profile_ref,
            preparation.scope.account_ref,
            preparation.scope.namespace_ref,
            preparation.project_ref,
            preparation.run_id,
            preparation.plan_fingerprint,
            preparation.preparation_digest,
            preparation.workload_digest,
            stage.operation.effect.effect_id,
            claim_document["stage_authenticated_receipt_digest"],
            claim_document["stage_record_digest"],
        ):
            raise ValueError("Modal chat stage predecessor differs")
        if (
            _encode_argument(
                claim, claim_tag, stage_bytes, submit_bytes, snapshot_bytes
            )
            != argument
        ):
            raise ValueError("Modal chat launch argument changed")
        return ModalChatWorkerAdmission._create(
            (argument, claim, claim_tag, stage_bytes, submit_bytes, snapshot_bytes)
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ValueError("Modal chat launch admission invalid") from None


__all__: list[str] = []
