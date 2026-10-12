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
import time
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
    PackagedRuntimeRelease, PackagedTrainingRuntimeReleaseV1,
    PackagedTrainingRuntimeReleaseV2, parse_packaged_runtime_release,
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
from tuner.runtime.packaged_sft_execution import RELEASE_INSPECTION_STAGES
_AMBIENT_PROBE_TRIGGER = (
    "PARENT_RUNTIME_INSTALLED_INVENTORY_DUPLICATE_VERSION_MISMATCH_OTHER_CROSS_ROOT"
)
_AMBIENT_PROBE_STAGES = frozenset(
    "PARENT_RUNTIME_AMBIENT_ROOT_" + root + "_CHILD_" + child
    for root in ("MATCH", "MISMATCH", "UNPROVEN")
    for child in ("PASS", "FAIL")
)
PARENT_RUNTIME_FAILURE_STAGES = frozenset(
    "PARENT_RUNTIME_" + stage for stage in RELEASE_INSPECTION_STAGES
) | {"PARENT_RUNTIME_UNAVAILABLE"} | _AMBIENT_PROBE_STAGES


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
        or (type(release) is PackagedTrainingRuntimeReleaseV1 and (
            facts.image_reference != release.image_ref or facts.image_digest != release.image_digest))
        or (type(release) is PackagedTrainingRuntimeReleaseV2 and (
            facts.image_reference is not None or facts.image_digest is not None
            or facts.material_digest != release.material_digest))
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


TRANSPORT_PROBE_SCHEMA = "synaptic-modal-cpu-transport-probe/v2"
TRANSPORT_PROBE_PURPOSE = "modal-runtime-release-transport-probe/v2"


@dataclass(frozen=True, slots=True)
class ModalCPUTransportProbeV1:
    marker_name: str
    marker_sha256: str
    schema_version: str = TRANSPORT_PROBE_SCHEMA
    metric_records: int = 8
    max_leaf_bytes: int = 8192
    max_page_bytes: int = 65536
    max_page_entries: int = 100
    max_polls: int = 180
    max_host_seconds: int = 180
    max_ledger_bytes: int = 1048576
    final_hold_seconds: int = 20
    publication_strategy: str = "modal_sdk_create_only/v1"
    collision_attempts: int = 1
    terminal_receipt_volume_role: str = "control"

    def __post_init__(self):
        if (self.schema_version != TRANSPORT_PROBE_SCHEMA
                or self.publication_strategy != "modal_sdk_create_only/v1"
                or self.terminal_receipt_volume_role != "control"):
            raise ValueError("invalid CPU transport probe schema")
        if (type(self.marker_name) is not str or not self.marker_name.startswith("probe-")
                or not self.marker_name.endswith(".bin") or len(self.marker_name) != 42
                or any(ch not in "0123456789abcdef" for ch in self.marker_name[6:-4])):
            raise ValueError("invalid CPU transport marker name")
        digest_text(self.marker_sha256, "transport_marker_sha256")
        for name, expected in (("metric_records", 8), ("max_leaf_bytes", 8192),
                ("max_page_bytes", 65536), ("max_page_entries", 100),
                ("max_polls", 180), ("max_host_seconds", 180),
                ("max_ledger_bytes", 1048576), ("collision_attempts", 1),
                ("final_hold_seconds", 20)):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError("CPU transport probe bound differs")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, document):
        if type(document) is not dict or set(document) != set(cls.__dataclass_fields__):
            raise ValueError("CPU transport probe configuration is invalid")
        return cls(**document)


def transport_probe_root(dispatch):
    return operation_path(dispatch.effect_id, "runtime-release-transport")


def transport_probe_content(dispatch, call_id, index):
    if type(index) is not int or not 0 <= index <= 8:
        raise ValueError("CPU transport probe sequence invalid")
    return {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
            "dispatch_digest": dispatch.dispatch_digest, "effect_id": dispatch.effect_id,
            "provider_call_id": safe_ref(call_id, "provider_call_id"), "sequence": index,
            "kind": "numeric_fixture" if index < 8 else "fixture_exhausted",
            "value": index if index < 8 else None,
            "training_executed": False, "gpu_qualified": False}


def transport_probe_envelope(content, signer, key_ref):
    tag = signer.sign(TRANSPORT_PROBE_PURPOSE, canonical_bytes(content), key_ref)
    if type(tag) is not bytes or not 0 < len(tag) <= 128:
        raise ValueError("CPU transport probe authentication invalid")
    return canonical_bytes({"content": content, "tag_hex": tag.hex()})


def verify_transport_probe_envelope(raw, verifier, key_ref):
    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise ValueError("CPU transport probe leaf bound invalid")
    document = parse_canonical_object(raw, name="CPU transport probe envelope")
    if set(document) != {"content", "tag_hex"} or type(document["content"]) is not dict:
        raise ValueError("CPU transport probe envelope invalid")
    tag = document["tag_hex"]
    if (type(tag) is not str or not 2 <= len(tag) <= 256 or len(tag) % 2
            or any(ch not in "0123456789abcdef" for ch in tag)):
        raise ValueError("CPU transport probe tag invalid")
    if verifier.verify(TRANSPORT_PROBE_PURPOSE, canonical_bytes(document["content"]),
                       bytes.fromhex(tag), key_ref) is not True:
        raise ValueError("CPU transport probe authentication failed")
    return document["content"]


def _run_transport_probe(dispatch, *, binding, signer, call_id, deadline, publication_factory):
    from .packaged_progress import SDKProgressPublicationError
    if not callable(publication_factory):
        raise ValueError("CPU transport SDK publication unavailable")
    probe, root = dispatch.transport_probe, transport_probe_root(dispatch)
    if time.monotonic() + 40 > deadline:
        raise TimeoutError("CPU transport window unavailable")
    publication = publication_factory(binding, root, deadline)
    def pause(seconds):
        if time.monotonic() + seconds >= deadline:
            raise TimeoutError("CPU transport phase window unavailable")
        time.sleep(seconds)
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU transport phase window exhausted")
    leaves, collision = [], None
    for index in range(9):
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU transport window exhausted")
        content = transport_probe_content(dispatch, call_id, index)
        raw = transport_probe_envelope(content, signer, dispatch.key_ref)
        path = root + f"/final/{index:012d}.json"
        publication.publish_exclusive(path, raw, probe.max_leaf_bytes)
        digest = hashlib.sha256(raw).hexdigest()
        if index == 0:
            # Exactly one separately admitted server collision, never replay the
            # original publisher's consumed uncertain/successful path claim.
            other = publication_factory(binding, root, deadline)
            candidate = transport_probe_envelope({**content, "value": 99}, signer, dispatch.key_ref)
            try:
                other.publish_exclusive(path, candidate, probe.max_leaf_bytes)
            except SDKProgressPublicationError as error:
                if type(error) is not SDKProgressPublicationError or error.code != "SDK_PUBLISH_COLLISION":
                    raise
            else:
                raise ValueError("CPU transport collision was not rejected")
            collision = {"attempts": 1, "rejected": True, "path": path,
                         "original_sha256": digest}
        leaves.append({"sequence": index, "path": path, "size_bytes": len(raw), "sha256": digest})
        pause(2)
    pause(probe.final_hold_seconds)
    return {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
            "publication_strategy": probe.publication_strategy, "root": root,
            "records": leaves, "collision": collision,
            "training_executed": False, "gpu_qualified": False}


def _publish_transport_receipt(dispatch, *, binding, signer, call_id, evidence):
    content = {"schema_version": TRANSPORT_PROBE_SCHEMA, "scope": "synthetic_cpu_transport",
        "effect_id": dispatch.effect_id, "dispatch_digest": dispatch.dispatch_digest,
        "runtime_release_digest": dispatch.runtime_release.manifest_digest,
        "deployment_facts_digest": dispatch.deployment_facts.facts_digest,
        "fixture_sha256": dispatch.fixture.sha256, "provider_call_id": call_id,
        "probe": dispatch.transport_probe.to_dict(), "evidence": evidence,
        "training_executed": False, "gpu_qualified": False}
    raw = transport_probe_envelope(content, signer, dispatch.key_ref)
    prefix = operation_path(dispatch.effect_id, "runtime-release-transport-evidence")
    binding.claim_directory(prefix)
    binding.write_exclusive(prefix + "/receipt.json", raw, 8192)



@dataclass(frozen=True, slots=True)
class ModalRuntimeReleaseQualificationDispatchV1:
    effect_id: str
    runtime_release: PackagedRuntimeRelease
    provider_binding: ProviderRuntimeBindingV1
    deployment_facts: object
    fixture: ModalRuntimeReleaseFixtureReceiptV1
    policy: ModalRuntimeReleaseQualificationPolicyV1
    key_ref: str
    transport_probe: ModalCPUTransportProbeV1 | None = None

    def __post_init__(self) -> None:
        facts_type = _qualification_facts_type()
        if type(self.runtime_release) not in (PackagedTrainingRuntimeReleaseV1, PackagedTrainingRuntimeReleaseV2) \
                or type(self.provider_binding) is not ProviderRuntimeBindingV1 \
                or type(self.deployment_facts) is not facts_type \
                or type(self.fixture) is not ModalRuntimeReleaseFixtureReceiptV1 \
                or type(self.policy) is not ModalRuntimeReleaseQualificationPolicyV1:
            raise TypeError("exact runtime qualification contracts required")
        if self.transport_probe is not None and type(self.transport_probe) is not ModalCPUTransportProbeV1:
            raise TypeError("exact CPU transport probe required")
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
        if self.transport_probe is not None:
            result["transport_probe"] = self.transport_probe.to_dict()
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
    if set(document) - ({"transport_probe"} if "transport_probe" in document else set()) != {
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
        parse_packaged_runtime_release(document["runtime_release"]),
        ProviderRuntimeBindingV1.from_dict(document["provider_binding"]),
        facts, ModalRuntimeReleaseFixtureReceiptV1.from_dict(document["fixture"]),
        ModalRuntimeReleaseQualificationPolicyV1.from_dict(document["policy"]),
        document["key_ref"],
        ModalCPUTransportProbeV1.from_dict(document["transport_probe"]) if "transport_probe" in document else None,
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

    def __call__(self, dispatch_bytes: bytes, *, commit_artifacts, commit_control, publication_factory=None):
        probe_binding = None
        core_published = False
        deadline = time.monotonic() + 110
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
            if dispatch.transport_probe is not None:
                from .volume_root_binding import VolumeRootBinding
                probe_binding = VolumeRootBinding.bind(root_path=str(self._roots.control),
                    volume_id=_volume_id(self._facts, "control"),
                    marker_name=dispatch.transport_probe.marker_name,
                    marker_sha256=dispatch.transport_probe.marker_sha256)
            from tuner.runtime.packaged_training_worker import (
                PackagedLocalCPUStageError, inspect_reviewed_root_inventory,
                qualify_installed_child,
            )
            release_payload = dispatch.runtime_release.to_dict()
            from tuner.runtime.packaged_sft_execution import (
                PackagedReleaseInspectionError, _inspect_parent_release,
            )
            try:
                _inspect_parent_release(dispatch.runtime_release)
            except PackagedReleaseInspectionError as error:
                stage = "PARENT_RUNTIME_" + error.stage if type(error.stage) is str else ""
                if type(error) is not PackagedReleaseInspectionError or stage not in PARENT_RUNTIME_FAILURE_STAGES:
                    stage = "PARENT_RUNTIME_UNAVAILABLE"
                if stage == _AMBIENT_PROBE_TRIGGER:
                    try:
                        measured = inspect_reviewed_root_inventory(dispatch.runtime_release)
                    except BaseException:
                        measured = None
                    root = "UNPROVEN"
                    if (type(measured) is dict and set(measured) == {"digest", "count"}
                            and type(measured["digest"]) is str and len(measured["digest"]) == 64
                            and all(character in "0123456789abcdef" for character in measured["digest"])
                            and type(measured["count"]) is int and 1 <= measured["count"] <= 4096):
                        root = ("MATCH" if measured["digest"] == dispatch.runtime_release.installed_distributions_digest
                                and measured["count"] == dispatch.runtime_release.installed_distribution_count
                                else "MISMATCH")
                    try:
                        qualify_installed_child(release_payload)
                        child = "PASS"
                    except BaseException:
                        child = "FAIL"
                    stage = "PARENT_RUNTIME_AMBIENT_ROOT_" + root + "_CHILD_" + child
                return {
                    "schema_version": QUALIFICATION_RESULT_SCHEMA,
                    "status_code": "failed",
                    "failure_stage": stage,
                }
            except BaseException:
                return {
                    "schema_version": QUALIFICATION_RESULT_SCHEMA,
                    "status_code": "failed",
                    "failure_stage": "PARENT_RUNTIME_UNAVAILABLE",
                }
            try:
                child = qualify_installed_child(release_payload)
            except PackagedLocalCPUStageError as error:
                if error.stage not in {"PARENT_RELEASE", "CHILD_RESULT"}:
                    raise ValueError from None
                return {
                    "schema_version": QUALIFICATION_RESULT_SCHEMA,
                    "status_code": "failed",
                    "failure_stage": error.stage,
                }
            except BaseException:
                return {
                    "schema_version": QUALIFICATION_RESULT_SCHEMA,
                    "status_code": "failed",
                    "failure_stage": "INSTALLED_CHILD",
                }
            output_document = {
                "schema_version": QUALIFICATION_OUTPUT_SCHEMA,
                "status": "passed", "effect_id": dispatch.effect_id,
                "dispatch_digest": dispatch.dispatch_digest,
                "runtime_release_digest": dispatch.runtime_release.manifest_digest,
                "deployment_facts_digest": self._facts.facts_digest,
                "fixture_sha256": dispatch.fixture.sha256,
                "child": child, "limitations": [CURRENT_LAYOUT_LIMITATION],
                "training_executed": False, "gpu_qualified": False,
            }
            output = canonical_bytes(output_document)
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
            core_published = True
            # Core CPU proof is mandatory and already published independently.
            # Optional transport failures do not replace that original receipt.
            if dispatch.transport_probe is not None:
                try:
                    try:
                        probe_evidence = _run_transport_probe(dispatch, binding=probe_binding,
                            signer=self._signer, call_id=call_id, deadline=deadline,
                            publication_factory=publication_factory)
                    except Exception as error:
                        from .packaged_progress import SDKProgressPublicationError, SDK_PUBLICATION_DIAGNOSTICS
                        failure_code = (error.code if type(error) is SDKProgressPublicationError
                            and error.code in SDK_PUBLICATION_DIAGNOSTICS else "TRANSPORT_PROBE_UNAVAILABLE")
                        probe_evidence = {"status": "inconclusive", "failure_code": failure_code}
                    _publish_transport_receipt(dispatch, binding=probe_binding,
                        signer=self._signer, call_id=call_id, evidence=probe_evidence)
                    commit_control()
                except Exception:
                    return {"schema_version": QUALIFICATION_RESULT_SCHEMA,
                            "status_code": "completed", "transport_probe_status": "unavailable"}
            return {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "completed",
            }
        except BaseException:
            return {
                "schema_version": QUALIFICATION_RESULT_SCHEMA,
                "status_code": "failed",
            }
        finally:
            if probe_binding is not None:
                try:
                    probe_binding.close()
                except Exception:
                    if not core_published:
                        raise


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
