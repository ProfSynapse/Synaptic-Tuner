"""Closed contracts and remote worker for Modal runtime-release CPU qualification.

This lane qualifies the installed runtime only.  It cannot train, prepare a
model, use a GPU, deploy an app, or infer a Function-to-Image relationship from
current Modal state.  The acknowledged deployment facts remain the authority
for that relationship; current observation can only recheck app generation and
layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
from pathlib import Path
from typing import Protocol

from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    parse_canonical_object,
    safe_ref,
)
from tuner.runtime.packaged_training_worker import LOCAL_CPU_DATA
from tuner.runtime.releases import (
    PackagedTrainingRuntimeReleaseV1,
    ProviderRuntimeBindingV1,
)

from .contracts import operation_path, provider_entry_identity
from .mounted_io import hash_regular, read_regular, write_exclusive


QUALIFICATION_FIXTURE_SCHEMA = "synaptic-modal-runtime-release-fixture/v1"
QUALIFICATION_DISPATCH_SCHEMA = "synaptic-modal-runtime-release-qualification-dispatch/v1"
QUALIFICATION_RECEIPT_SCHEMA = "synaptic-modal-runtime-release-qualification-receipt/v1"
QUALIFICATION_OUTPUT_SCHEMA = "synaptic-modal-runtime-release-qualification-output/v1"
QUALIFICATION_RESULT_SCHEMA = "synaptic-modal-runtime-release-qualification-result/v1"
CURRENT_LAYOUT_LIMITATION = (
    "current_layout_reobserved_function_image_relationship_not_independently_readable"
)
QUALIFICATION_HMAC_ENV_KEY = "SYNAPTIC_MODAL_QUALIFICATION_HMAC_KEY"
QUALIFICATION_HMAC_KEY_REF = "modal-runtime-release-qualification-hmac-v1"
MAX_DISPATCH_BYTES = 512 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024


def _qualification_facts_type():
    from .runtime_release_deployment import ModalRuntimeReleaseDeploymentFactsV1
    return ModalRuntimeReleaseDeploymentFactsV1


def _volume_id(facts, role: str) -> str:
    legacy_name = {
        "artifacts": "artifact_volume_id", "control": "control_volume_id",
    }.get(role, role + "_volume_id")
    legacy = getattr(facts, legacy_name, None)
    if legacy is not None:  # Test adapter for the older fact vector only.
        return safe_ref(legacy, role + "_volume_id")
    matches = tuple(item.volume_id for item in facts.volumes if item.spec.role == role)
    if len(matches) != 1:
        raise ValueError("runtime qualification Volume role is not exact")
    return safe_ref(matches[0], role + "_volume_id")


def _volume_fact(facts, role: str):
    matches = tuple(item for item in facts.volumes if item.spec.role == role)
    if len(matches) != 1:
        raise ValueError("runtime qualification Volume role is not exact")
    return matches[0]


def _qualification_secret_fact(facts):
    self_check = _self_check_fact(facts)
    if len(self_check.spec.secret_names) != 1:
        raise ValueError("runtime qualification auth Secret is not exact")
    name = self_check.spec.secret_names[0]
    matches = tuple(item for item in facts.secrets if item.spec.name == name)
    if len(matches) != 1 \
            or matches[0].spec.required_keys != (QUALIFICATION_HMAC_ENV_KEY,):
        raise ValueError("runtime qualification auth Secret policy differs")
    return matches[0]


def _validate_deployment_release(facts, release) -> None:
    legacy = getattr(facts, "validate_release", None)
    if callable(legacy):  # Test adapter for the older fact vector only.
        legacy(release)
        return
    if (
        facts.runtime_release_digest != release.manifest_digest
        or facts.image_reference != release.image_ref
        or facts.image_digest != release.image_digest
        or facts.python_implementation != release.python_implementation
        or facts.python_version != release.python_version
        or facts.python_executable != release.python_executable
        or facts.python_executable_digest != release.python_executable_digest
        or facts.package_digest != release.package_digest
        or facts.installed_distributions_digest != release.installed_distributions_digest
        or facts.worker_entrypoint != release.worker_entrypoint
        or facts.worker_closure_digest != release.worker_closure_digest
        or facts.current_state_version_pinned is not False
        or facts.function_image_link_readable is not False
    ):
        raise ValueError("runtime qualification deployment differs from release")
    self_checks = tuple(item for item in facts.functions if item.spec.role == "self_check")
    if len(self_checks) != 1:
        raise ValueError("runtime qualification self-check Function is not exact")
    spec = self_checks[0].spec
    if (
        spec.module != "tuner.runtime.runtime_release_modal_self_check"
        or spec.qualname != "run_runtime_release_self_check"
        or spec.volume_roles != ("control", "artifacts")
        or spec.cpu_milli != 1000 or spec.memory_mib != 512
        or spec.timeout_seconds != 120 or spec.gpu is not None
        or spec.block_network is not True or spec.retries != 0
        or spec.restrict_modal_access is not False
        or spec.single_use_containers is not True
        or spec.serialized is not False or spec.include_source is not False
    ):
        raise ValueError("runtime qualification self-check Function policy differs")
    _qualification_secret_fact(facts)


def _self_check_fact(facts):
    matches = tuple(item for item in facts.functions if item.spec.role == "self_check")
    if len(matches) != 1:
        raise ValueError("runtime qualification self-check Function is not exact")
    return matches[0]


def _expected_provider_binding(facts, release):
    legacy = getattr(facts, "build_provider_binding", None)
    if callable(legacy):  # Test adapter for the older fact vector only.
        return legacy(release)
    return ProviderRuntimeBindingV1.build(
        provider_ref="modal", runtime_release=release,
        provider_facts_schema=facts.schema_version,
        provider_facts_digest=facts.facts_digest,
    )


def _observe_current_layout(observer, facts) -> str:
    """Recheck only current app/generation/layout, never Function→Image."""
    current = observer.observe(facts)
    if type(current) is type(facts) and current == facts:  # Legacy test adapter.
        return facts.facts_digest
    from .runtime_release_deployment import ModalRuntimeReleaseDeploymentObservationV1
    if type(current) is not ModalRuntimeReleaseDeploymentObservationV1:
        raise ValueError("runtime qualification current observation is invalid")
    expected_functions = tuple(sorted(
        (item.spec.name, item.function_id) for item in facts.functions
    ))
    if (
        current.deployed is not True or current.app_name != facts.app_name
        or current.app_id != facts.app_id or current.generation != facts.generation
        or current.function_ids != expected_functions or current.class_ids
    ):
        raise ValueError("runtime qualification current layout changed")
    return hashlib.sha256(canonical_bytes(current.to_dict())).hexdigest()


def _secret_free(value: object) -> None:
    # Secret resource names/IDs and required-key names are committed deployment
    # facts.  Credential values and ambient token/password fields are forbidden.
    forbidden = ("token", "password", "api_key", "apikey", "credential", "secret_value")
    if type(value) is dict:
        for key, member in value.items():
            if type(key) is not str or any(part in key.lower() for part in forbidden):
                raise ValueError("qualification document contains a secret-shaped field")
            _secret_free(member)
    elif type(value) is list:
        for member in value:
            _secret_free(member)


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationPolicyV1:
    cpu: int = 1
    memory_mib: int = 512
    timeout_seconds: int = 120
    network_access: bool = False
    gpu: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.cpu) is not int or self.cpu != 1
            or type(self.memory_mib) is not int or self.memory_mib != 512
            or type(self.timeout_seconds) is not int or self.timeout_seconds != 120
            or self.network_access is not False or self.gpu is not False
        ):
            raise ValueError("runtime qualification policy is not the fixed CPU policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu": self.cpu, "memory_mib": self.memory_mib,
            "timeout_seconds": self.timeout_seconds,
            "network_access": self.network_access, "gpu": self.gpu,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]):
        if type(value) is not dict or set(value) != {
            "cpu", "memory_mib", "timeout_seconds", "network_access", "gpu",
        }:
            raise ValueError("runtime qualification policy is invalid")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseFixtureReceiptV1:
    effect_id: str
    artifact_volume_id: str
    path: str
    size_bytes: int
    sha256: str
    provider_entry_id: str
    schema_version: str = QUALIFICATION_FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_FIXTURE_SCHEMA:
            raise ValueError("unsupported qualification fixture receipt")
        safe_ref(self.effect_id, "effect_id")
        safe_ref(self.artifact_volume_id, "artifact_volume_id")
        digest_text(self.sha256, "fixture_sha256")
        safe_ref(self.provider_entry_id, "provider_entry_id")
        if type(self.size_bytes) is not int or self.size_bytes != len(LOCAL_CPU_DATA):
            raise ValueError("qualification fixture size is not exact")
        digest = hashlib.sha256(LOCAL_CPU_DATA).hexdigest()
        expected = operation_path(
            self.effect_id, "runtime-release-qualification", "input", digest,
            "fixture.jsonl",
        )
        if self.sha256 != digest or self.path != expected or self.provider_entry_id != provider_entry_identity(
            self.artifact_volume_id, expected, self.size_bytes,
        ):
            raise ValueError("qualification fixture receipt differs from fixed input")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version, "effect_id": self.effect_id,
            "artifact_volume_id": self.artifact_volume_id, "path": self.path,
            "size_bytes": self.size_bytes, "sha256": self.sha256,
            "provider_entry_id": self.provider_entry_id,
        }

    @classmethod
    def create(cls, *, effect_id: str, artifact_volume_id: str):
        digest = hashlib.sha256(LOCAL_CPU_DATA).hexdigest()
        path = operation_path(
            effect_id, "runtime-release-qualification", "input", digest,
            "fixture.jsonl",
        )
        return cls(
            effect_id, artifact_volume_id, path, len(LOCAL_CPU_DATA), digest,
            provider_entry_identity(artifact_volume_id, path, len(LOCAL_CPU_DATA)),
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]):
        if type(value) is not dict or set(value) != {
            "schema_version", "effect_id", "artifact_volume_id", "path",
            "size_bytes", "sha256", "provider_entry_id",
        }:
            raise ValueError("qualification fixture receipt is invalid")
        return cls(**value)  # type: ignore[arg-type]


class ModalRuntimeQualificationAuthenticator(Protocol):
    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes: ...
    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool: ...


class ModalRuntimeQualificationHmacAuthenticator:
    """Closed HMAC authenticator for the one qualification Secret value."""

    __slots__ = ("_key",)

    def __init__(self, key: bytes) -> None:
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("runtime qualification HMAC key is invalid")
        self._key = key

    def sign(self, purpose: str, payload: bytes, key_ref: str) -> bytes:
        if key_ref != QUALIFICATION_HMAC_KEY_REF:
            raise ValueError("runtime qualification HMAC key reference differs")
        if type(purpose) is not str or not purpose.isascii() or not purpose \
                or type(payload) is not bytes:
            raise TypeError("runtime qualification HMAC input is invalid")
        return hmac.new(
            self._key, purpose.encode("ascii") + b"\0" + payload,
            hashlib.sha256,
        ).digest()

    def verify(self, purpose: str, payload: bytes, tag: bytes, key_ref: str) -> bool:
        if type(tag) is not bytes:
            return False
        try:
            expected = self.sign(purpose, payload, key_ref)
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(expected, tag)


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationDispatchV1:
    effect_id: str
    runtime_release: PackagedTrainingRuntimeReleaseV1
    provider_binding: ProviderRuntimeBindingV1
    deployment_facts: object
    fixture: ModalRuntimeReleaseFixtureReceiptV1
    policy: ModalRuntimeReleaseQualificationPolicyV1
    key_ref: str

    def __post_init__(self) -> None:
        facts_type = _qualification_facts_type()
        if type(self.runtime_release) is not PackagedTrainingRuntimeReleaseV1 \
                or type(self.provider_binding) is not ProviderRuntimeBindingV1 \
                or type(self.deployment_facts) is not facts_type \
                or type(self.fixture) is not ModalRuntimeReleaseFixtureReceiptV1 \
                or type(self.policy) is not ModalRuntimeReleaseQualificationPolicyV1:
            raise TypeError("exact runtime qualification contracts required")
        safe_ref(self.effect_id, "effect_id")
        if self.key_ref != QUALIFICATION_HMAC_KEY_REF:
            raise ValueError("runtime qualification key reference differs")
        if self.fixture.effect_id != self.effect_id:
            raise ValueError("qualification fixture targets another effect")
        _validate_deployment_release(self.deployment_facts, self.runtime_release)
        if self.provider_binding != _expected_provider_binding(
            self.deployment_facts, self.runtime_release,
        ):
            raise ValueError("qualification provider binding differs from deployment facts")
        if self.fixture.artifact_volume_id != _volume_id(
            self.deployment_facts, "artifacts",
        ):
            raise ValueError("qualification fixture targets another artifact Volume")

    def unsigned_dict(self) -> dict[str, object]:
        result = {
            "schema_version": QUALIFICATION_DISPATCH_SCHEMA,
            "effect_id": self.effect_id,
            "runtime_release": self.runtime_release.to_dict(),
            "provider_binding": self.provider_binding.to_dict(),
            "deployment_facts": parse_canonical_object(
                self.deployment_facts.canonical_bytes,
                name="Modal runtime release deployment facts",
            ),
            "deployment_facts_digest": self.deployment_facts.facts_digest,
            "fixture": self.fixture.to_dict(), "policy": self.policy.to_dict(),
            "key_ref": self.key_ref,
            "limitations": [CURRENT_LAYOUT_LIMITATION],
        }
        _secret_free(result)
        return result

    @property
    def unsigned_bytes(self) -> bytes:
        return canonical_bytes(self.unsigned_dict())

    @property
    def dispatch_digest(self) -> str:
        return hashlib.sha256(self.unsigned_bytes).hexdigest()


def build_modal_runtime_release_qualification_dispatch(
    dispatch: ModalRuntimeReleaseQualificationDispatchV1,
    authenticator: ModalRuntimeQualificationAuthenticator,
) -> bytes:
    if type(dispatch) is not ModalRuntimeReleaseQualificationDispatchV1:
        raise TypeError("exact runtime qualification dispatch required")
    payload = dispatch.unsigned_bytes
    tag = authenticator.sign(
        "modal-runtime-release-qualification-dispatch/v1", payload,
        dispatch.key_ref,
    )
    if type(tag) is not bytes or not tag or len(tag) > 128:
        raise ValueError("qualification dispatch authentication is invalid")
    result = canonical_bytes({
        "dispatch": dispatch.unsigned_dict(),
        "authentication": {"tag": base64.b64encode(tag).decode("ascii")},
    })
    if len(result) > MAX_DISPATCH_BYTES:
        raise ValueError("qualification dispatch exceeds its bound")
    return result


def parse_modal_runtime_release_qualification_dispatch(
    payload: bytes, verifier: ModalRuntimeQualificationAuthenticator,
) -> ModalRuntimeReleaseQualificationDispatchV1:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_DISPATCH_BYTES:
        raise ValueError("qualification dispatch bytes are invalid")
    envelope = parse_canonical_object(payload, name="Modal runtime qualification dispatch")
    if set(envelope) != {"dispatch", "authentication"} \
            or type(envelope["dispatch"]) is not dict \
            or type(envelope["authentication"]) is not dict \
            or set(envelope["authentication"]) != {"tag"}:
        raise ValueError("qualification dispatch envelope is invalid")
    document = envelope["dispatch"]
    if set(document) != {
        "schema_version", "effect_id", "runtime_release", "provider_binding",
        "deployment_facts", "deployment_facts_digest", "fixture", "policy",
        "key_ref", "limitations",
    } or document.get("schema_version") != QUALIFICATION_DISPATCH_SCHEMA \
            or document.get("limitations") != [CURRENT_LAYOUT_LIMITATION]:
        raise ValueError("qualification dispatch is invalid")
    facts_type = _qualification_facts_type()
    facts = facts_type.parse(canonical_bytes(document["deployment_facts"]))
    if facts.facts_digest != document["deployment_facts_digest"]:
        raise ValueError("qualification deployment facts digest differs")
    dispatch = ModalRuntimeReleaseQualificationDispatchV1(
        document["effect_id"],
        PackagedTrainingRuntimeReleaseV1.from_dict(document["runtime_release"]),
        ProviderRuntimeBindingV1.from_dict(document["provider_binding"]),
        facts, ModalRuntimeReleaseFixtureReceiptV1.from_dict(document["fixture"]),
        ModalRuntimeReleaseQualificationPolicyV1.from_dict(document["policy"]),
        document["key_ref"],
    )
    try:
        tag = base64.b64decode(
            envelope["authentication"]["tag"], validate=True,
        )
        valid = verifier.verify(
            "modal-runtime-release-qualification-dispatch/v1",
            dispatch.unsigned_bytes, tag, dispatch.key_ref,
        )
    except Exception:
        valid = False
    if valid is not True:
        raise ValueError("qualification dispatch authentication failed")
    return dispatch


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationRoots:
    control: Path
    artifacts: Path

    def __post_init__(self) -> None:
        if any(not isinstance(path, Path) or not path.is_absolute() for path in (
            self.control, self.artifacts,
        )) or self.control == self.artifacts:
            raise ValueError("qualification roots must be distinct absolute paths")


@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationReceiptV1:
    effect_id: str
    dispatch_digest: str
    provider_call_id: str
    deployment_facts_digest: str
    current_observation_digest: str
    output_path: str
    output_size: int
    output_sha256: str
    output_provider_entry_id: str
    limitation: str = CURRENT_LAYOUT_LIMITATION
    schema_version: str = QUALIFICATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_RECEIPT_SCHEMA \
                or self.limitation != CURRENT_LAYOUT_LIMITATION:
            raise ValueError("unsupported qualification receipt")
        safe_ref(self.effect_id, "effect_id")
        safe_ref(self.provider_call_id, "provider_call_id")
        for name in (
            "dispatch_digest", "deployment_facts_digest",
            "current_observation_digest", "output_sha256",
        ):
            digest_text(getattr(self, name), name)
        if type(self.output_size) is not int or not 1 <= self.output_size <= MAX_EVIDENCE_BYTES:
            raise ValueError("qualification output size is invalid")
        expected = operation_path(
            self.effect_id, "runtime-release-qualification", "output",
            "evidence.json",
        )
        if self.output_path != expected:
            raise ValueError("qualification output path is invalid")
        safe_ref(self.output_provider_entry_id, "output_provider_entry_id")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, object]):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("qualification receipt is invalid")
        return cls(**value)  # type: ignore[arg-type]

    @classmethod
    def parse(cls, payload: bytes):
        value = parse_canonical_object(payload, name="Modal qualification receipt")
        result = cls.from_dict(value)
        if result.canonical_bytes != payload:
            raise ValueError("qualification receipt is not canonical")
        return result


class ModalRuntimeReleaseQualificationWorker:
    """One-shot remote wrapper around the installed isolated-child diagnostic."""

    def __init__(self, *, expected_facts, verifier, signer, observer,
                 call_id_provider, roots: ModalRuntimeReleaseQualificationRoots):
        if type(expected_facts) is not _qualification_facts_type() \
                or type(roots) is not ModalRuntimeReleaseQualificationRoots:
            raise TypeError("exact qualification worker facts and roots required")
        for value, member in (
            (verifier, "verify"), (signer, "sign"), (observer, "observe"),
        ):
            if not callable(getattr(value, member, None)):
                raise TypeError("qualification worker collaborator is incomplete")
        if not callable(call_id_provider):
            raise TypeError("qualification call identity provider required")
        self._facts, self._verifier, self._signer = expected_facts, verifier, signer
        self._observer, self._call_id, self._roots = observer, call_id_provider, roots

    def __call__(self, dispatch_bytes: bytes, *, commit_artifacts, commit_control):
        try:
            dispatch = parse_modal_runtime_release_qualification_dispatch(
                dispatch_bytes, self._verifier,
            )
            if dispatch.deployment_facts != self._facts:
                raise ValueError
            current_observation_digest = _observe_current_layout(
                self._observer, self._facts,
            )
            staged = self._roots.artifacts / dispatch.fixture.path
            if hash_regular(
                self._roots.artifacts, staged, dispatch.fixture.size_bytes,
            ) != (dispatch.fixture.size_bytes, dispatch.fixture.sha256) \
                    or read_regular(
                        self._roots.artifacts, staged, dispatch.fixture.size_bytes,
                    ) != LOCAL_CPU_DATA:
                raise ValueError
            from tuner.runtime.packaged_training_worker import qualify_installed_child
            child = qualify_installed_child(dispatch.runtime_release.to_dict())
            output = canonical_bytes({
                "schema_version": QUALIFICATION_OUTPUT_SCHEMA,
                "status": "passed", "effect_id": dispatch.effect_id,
                "dispatch_digest": dispatch.dispatch_digest,
                "runtime_release_digest": dispatch.runtime_release.manifest_digest,
                "deployment_facts_digest": self._facts.facts_digest,
                "fixture_sha256": dispatch.fixture.sha256,
                "child": child, "limitations": [CURRENT_LAYOUT_LIMITATION],
                "training_executed": False, "gpu_qualified": False,
            })
            if len(output) > MAX_EVIDENCE_BYTES:
                raise ValueError
            output_path = operation_path(
                dispatch.effect_id, "runtime-release-qualification", "output",
                "evidence.json",
            )
            write_exclusive(
                self._roots.artifacts, self._roots.artifacts / output_path, output,
            )
            commit_artifacts()
            call_id = safe_ref(self._call_id(), "provider_call_id")
            receipt = ModalRuntimeReleaseQualificationReceiptV1(
                dispatch.effect_id, dispatch.dispatch_digest, call_id,
                self._facts.facts_digest, current_observation_digest, output_path,
                len(output), hashlib.sha256(output).hexdigest(),
                provider_entry_identity(
                    _volume_id(self._facts, "artifacts"), output_path, len(output),
                ),
            )
            tag = self._signer.sign(
                "modal-runtime-release-qualification-receipt/v1",
                receipt.canonical_bytes, dispatch.key_ref,
            )
            if type(tag) is not bytes or not tag or len(tag) > 128:
                raise ValueError
            evidence = self._roots.control / operation_path(
                dispatch.effect_id, "runtime-release-qualification", "receipt",
            )
            write_exclusive(
                self._roots.control, evidence / "receipt.json",
                receipt.canonical_bytes,
            )
            write_exclusive(self._roots.control, evidence / "receipt.mac", tag)
            commit_control()
            return {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "completed",
            }
        except BaseException:
            return {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "failed",
            }


__all__ = [
    "CURRENT_LAYOUT_LIMITATION", "MAX_DISPATCH_BYTES", "MAX_EVIDENCE_BYTES",
    "QUALIFICATION_HMAC_ENV_KEY", "QUALIFICATION_HMAC_KEY_REF",
    "ModalRuntimeQualificationAuthenticator",
    "ModalRuntimeQualificationHmacAuthenticator",
    "ModalRuntimeReleaseFixtureReceiptV1",
    "ModalRuntimeReleaseQualificationDispatchV1",
    "ModalRuntimeReleaseQualificationPolicyV1",
    "ModalRuntimeReleaseQualificationReceiptV1",
    "ModalRuntimeReleaseQualificationRoots",
    "ModalRuntimeReleaseQualificationWorker",
    "build_modal_runtime_release_qualification_dispatch",
    "parse_modal_runtime_release_qualification_dispatch",
]
