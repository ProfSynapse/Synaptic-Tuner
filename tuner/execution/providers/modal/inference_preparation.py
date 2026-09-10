"""Pure Foundation preparation for one separately authorized Modal chat session.

The factory authenticates only the supplied inference configuration and quote.
Its source/workload arguments must be fresh results from the existing trusted
binders.  This module performs no readiness check, grants no mutation, and does
not make a serving grant.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from inspect import getattr_static
import re

from synaptic_tuner.api.v1.providers import ProviderRef
from synaptic_tuner.api.v1.results import TrainingRunRef, VerifiedArtifact
from Evaluator.chat_session import ChatSessionPolicy
from tuner.execution.coordinator_v1.ports import CoordinatorClockPortV1
from tuner.execution.evidence import (
    DEPLOYMENT_EVIDENCE_POLICY,
    EvidenceAuthenticator,
    canonical_utc,
    parse_utc,
    validate_evidence_window,
)
from tuner.execution.foundation_v2.canonical import (
    canonical_bytes,
    digest_text,
    domain_digest,
    exact_fields,
    exact_integer,
    parse_canonical_object,
    safe_ref,
)
from tuner.execution.foundation_v2.commands import (
    CanonicalProviderPayloadV1,
    StageCommandV2,
    SubmitCommandV2,
    build_stage_command,
    build_submit_command,
)
from tuner.execution.foundation_v2.executors import ExecutorDescriptorV1
from tuner.execution.foundation_v2.identities import EffectKind, derive_effect
from tuner.execution.foundation_v2.preparation import CanonicalPreparationV2
from tuner.execution.foundation_v2.references import (
    ExecutionScopeV1,
    StagePredecessorV2,
)
from tuner.training.recipes import CompiledWorkload

from .coordinator_preflight import (
    AuthenticatedModalQuote,
    ModalQuoteBody,
    QUOTE_EVIDENCE_POLICY,
    QUOTE_PURPOSE,
    TrustedEvidenceIdentity,
)
from .config import ModalSecretProfileV1
from .contracts import (
    ArtifactMemberV1,
    ArtifactRole,
    EXACT_ARTIFACT_ROLES,
    provider_entry_identity,
)
from .inference_binding import ModalInferenceSourceBinding
from .inference_workload import ModalInferenceWorkloadBinding, _model

CONFIG_EVIDENCE_PURPOSE = "modal-inference-preparation-evidence/v1"
_SCHEMA = "synaptic-modal-inference-preparation-config/v1"
_TOKEN = object()
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_TOP = frozenset(
    {
        "schema_version",
        "provider",
        "client",
        "application",
        "image",
        "runtime",
        "volumes",
        "resources",
        "policy",
        "secrets",
        "evidence",
    }
)
_NESTED = {
    "provider": frozenset({"provider_id", "profile_ref"}),
    "client": frozenset(
        {
            "account_ref",
            "workspace_ref",
            "environment_ref",
            "client_ref",
            "sdk_version",
        }
    ),
    "application": frozenset(
        {"app_name", "app_ref", "sandbox_entrypoint", "worker_ref"}
    ),
    "image": frozenset({"registry_reference", "image_digest"}),
    "runtime": frozenset(
        {
            "dependency_lock_digest",
            "runtime_lock_digest",
            "source_lock_digest",
            "worker_closure_digest",
            "python_version",
            "python_executable",
            "python_executable_digest",
        }
    ),
    "volumes": frozenset(
        {
            "source_artifact_volume_ref",
            "source_artifact_volume_id",
            "chat_control_volume_ref",
            "chat_control_volume_id",
            "model_cache_volume_ref",
            "model_cache_volume_id",
            "key_ref",
        }
    ),
    "resources": frozenset(
        {
            "accelerator",
            "accelerator_count",
            "cpu_millicores",
            "memory_mb",
            "service_port",
            "provider_timeout_seconds",
            "provider_idle_timeout_seconds",
            "max_retries",
        }
    ),
    "policy": frozenset(
        {
            "startup_timeout_seconds",
            "request_timeout_seconds",
            "idle_timeout_seconds",
            "absolute_lifetime_seconds",
            "max_turns",
            "max_history_bytes",
            "max_request_bytes",
            "max_response_bytes",
        }
    ),
    "evidence": frozenset(
        {
            "issuer_ref",
            "evidence_ref",
            "audience_ref",
            "challenge_nonce",
            "key_ref",
            "verified_at",
            "expires_at",
        }
    ),
}


class ModalInferencePreparationError(RuntimeError):
    """Closed non-secret inference preparation failure."""


class ModalInferencePreparationConfig:
    __slots__ = ("_raw",)

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalInferencePreparationConfig is final")

    def __init__(self, *args, **kwargs):
        raise TypeError("inference preparation configs are parser minted")

    def __setattr__(self, name, value):
        raise AttributeError("inference preparation config is immutable")

    @classmethod
    def build(cls, document: dict[str, object]) -> "ModalInferencePreparationConfig":
        if type(document) is not dict:
            raise TypeError("configuration document must be an exact object")
        return cls.parse(canonical_bytes(document))

    @classmethod
    def parse(cls, raw: bytes) -> "ModalInferencePreparationConfig":
        if type(raw) is not bytes or not raw or len(raw) > 64 * 1024:
            raise ValueError("inference configuration exceeds its bound")
        document = parse_canonical_object(raw, name="inference preparation config")
        exact_fields(document, _TOP, "inference preparation config")
        if document["schema_version"] != _SCHEMA:
            raise ValueError("unsupported inference preparation config")
        for name, fields in _NESTED.items():
            value = document[name]
            if type(value) is not dict:
                raise TypeError(f"{name} must be an exact object")
            exact_fields(value, fields, name)
        cls._validate(document)
        value = object.__new__(cls)
        object.__setattr__(value, "_raw", bytes(raw))
        return value

    @staticmethod
    def _validate(document: dict[str, object]) -> None:
        provider = document["provider"]
        client = document["client"]
        application = document["application"]
        image = document["image"]
        runtime = document["runtime"]
        volumes = document["volumes"]
        resources = document["resources"]
        policy = document["policy"]
        secrets = document["secrets"]
        evidence = document["evidence"]
        if not all(
            type(value) is dict
            for value in (
                provider,
                client,
                application,
                image,
                runtime,
                volumes,
                resources,
                policy,
                evidence,
            )
        ):
            raise TypeError("inference configuration sections must be exact objects")
        for mapping, names in (
            (provider, tuple(provider)),
            (client, tuple(client)),
            (application, tuple(application)),
            (volumes, tuple(volumes)),
            (
                evidence,
                (
                    "issuer_ref",
                    "evidence_ref",
                    "audience_ref",
                    "challenge_nonce",
                    "key_ref",
                ),
            ),
        ):
            for name in names:
                safe_ref(mapping[name], name)
        if client["sdk_version"] != "1.5.4":
            raise ValueError("unsupported Modal SDK selection")
        reference = image["registry_reference"]
        digest = image["image_digest"]
        digest_text(digest, "image_digest")
        if (
            type(reference) is not str
            or not reference.endswith("@sha256:" + digest)
            or reference.startswith("@")
            or any(
                character.isspace() or character in "'\"`$\\" for character in reference
            )
            or "://" in reference
            or "@" in reference.removesuffix("@sha256:" + digest)
        ):
            raise ValueError("image reference is not digest pinned")
        for name in (
            "dependency_lock_digest",
            "runtime_lock_digest",
            "source_lock_digest",
            "worker_closure_digest",
            "python_executable_digest",
        ):
            digest_text(runtime[name], name)
        if (
            type(runtime["python_version"]) is not str
            or re.fullmatch(
                r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)",
                runtime["python_version"],
            )
            is None
            or type(runtime["python_executable"]) is not str
            or not runtime["python_executable"].startswith("/")
            or "//" in runtime["python_executable"]
            or any(
                part in {"", ".", ".."}
                for part in runtime["python_executable"].split("/")[1:]
            )
        ):
            raise ValueError("inference Python selection is invalid")
        if type(secrets) is not list or len(secrets) > 64:
            raise ValueError("secret requirements are invalid")
        secret_profiles = []
        for value in secrets:
            if type(value) is not dict or set(value) != {"name", "required_keys"}:
                raise ValueError("secret requirement is invalid")
            if type(value["required_keys"]) is not list:
                raise TypeError("secret required_keys must be an exact list")
            secret_profiles.append(
                ModalSecretProfileV1(value["name"], tuple(value["required_keys"]))
            )
        if len({item.name for item in secret_profiles}) != len(secret_profiles):
            raise ValueError("secret names must be unique")
        integers = {
            **resources,
            **policy,
        }
        for name, number in integers.items():
            if name == "accelerator":
                safe_ref(number, name)
            elif name == "max_retries":
                if type(number) is not int or number != 0:
                    raise ValueError("Modal chat retries must be disabled")
            elif type(number) is not int or not 1 <= number <= 2**31 - 1:
                raise ValueError(f"invalid {name}")
        if not 1 <= resources["service_port"] <= 65535:
            raise ValueError("service port is invalid")
        if (
            resources["provider_idle_timeout_seconds"]
            > resources["provider_timeout_seconds"]
        ):
            raise ValueError("provider idle timeout exceeds provider timeout")
        if (
            any(
                resources[name] > 24 * 60 * 60
                for name in (
                    "provider_timeout_seconds",
                    "provider_idle_timeout_seconds",
                )
            )
            or policy["startup_timeout_seconds"] > 24 * 60 * 60
        ):
            raise ValueError("inference timeout exceeds 24 hours")
        if policy["startup_timeout_seconds"] > policy["absolute_lifetime_seconds"]:
            raise ValueError("startup timeout exceeds absolute lifetime")
        if policy["idle_timeout_seconds"] > policy["absolute_lifetime_seconds"]:
            raise ValueError("host idle timeout exceeds absolute lifetime")
        ChatSessionPolicy(
            request_timeout_seconds=policy["request_timeout_seconds"],
            idle_timeout_seconds=policy["idle_timeout_seconds"],
            absolute_lifetime_seconds=policy["absolute_lifetime_seconds"],
            max_turns=policy["max_turns"],
            max_history_bytes=policy["max_history_bytes"],
        )
        for name in ("max_request_bytes", "max_response_bytes"):
            if policy[name] > 64 * 1024 * 1024:
                raise ValueError(f"{name} exceeds its bound")
        volume_ids = tuple(
            volumes[name]
            for name in (
                "source_artifact_volume_id",
                "chat_control_volume_id",
                "model_cache_volume_id",
            )
        )
        volume_refs = tuple(
            volumes[name]
            for name in (
                "source_artifact_volume_ref",
                "chat_control_volume_ref",
                "model_cache_volume_ref",
            )
        )
        if len(set(volume_ids)) != 3 or len(set(volume_refs)) != 3:
            raise ValueError("inference volumes must be distinct")
        for name in ("verified_at", "expires_at"):
            canonical_utc(evidence[name], name)

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._raw)

    @property
    def document(self) -> dict[str, object]:
        return parse_canonical_object(self._raw, name="inference preparation config")


@dataclass(frozen=True, slots=True)
class AuthenticatedModalInferencePreparationConfig:
    body_bytes: bytes
    tag: bytes

    def __post_init__(self) -> None:
        if (
            type(self.body_bytes) is not bytes
            or type(self.tag) is not bytes
            or not self.tag
            or len(self.tag) > 128
        ):
            raise TypeError(
                "exact authenticated inference configuration bytes required"
            )
        ModalInferencePreparationConfig.parse(self.body_bytes)


def _source_snapshot(source: ModalInferenceSourceBinding) -> tuple[object, ...]:
    return (
        source.run.to_dict(),
        tuple(item.to_dict() for item in source.artifacts),
        bytes(source.read_request_bytes),
        source.read_request_digest,
        source.source_workflow_record_digest,
        source.source_revision,
        source.manifest_digest,
        source.artifact_source_digest,
        source.provider_id,
        source.profile_ref,
        source.account_ref,
        source.namespace_ref,
        source.provider_job_ref,
        source.effect_id,
        source.provider_run_binding_digest,
        bytes(source.command_binding.command_bytes),
        bytes(source.command_binding.preparation_snapshot),
        bytes(source.command_binding.deployment_bytes),
        source.artifact_volume_id,
        tuple(
            (item.role.value, item.path, item.size, item.sha256, item.provider_entry_id)
            for item in source.native_members
        ),
        bytes(source.native_evidence),
    )


def _source_projection(source: ModalInferenceSourceBinding) -> dict[str, object]:
    payload = canonical_bytes(
        {
            "run": source.run.to_dict(),
            "artifacts": [item.to_dict() for item in source.artifacts],
            "read_request_sha256": hashlib.sha256(
                source.read_request_bytes
            ).hexdigest(),
            "read_request_digest": source.read_request_digest,
            "source_workflow_record_digest": source.source_workflow_record_digest,
            "source_revision": source.source_revision,
            "manifest_digest": source.manifest_digest,
            "artifact_source_digest": source.artifact_source_digest,
            "provider": [
                source.provider_id,
                source.profile_ref,
                source.account_ref,
                source.namespace_ref,
            ],
            "provider_job_ref": source.provider_job_ref,
            "effect_id": source.effect_id,
            "provider_run_binding_digest": source.provider_run_binding_digest,
            "command_binding_digest": source.command_binding.authenticated_binding_digest,
            "artifact_volume_id": source.artifact_volume_id,
            "members": [
                {
                    "role": item.role.value,
                    "path": item.path,
                    "size": item.size,
                    "sha256": item.sha256,
                    "provider_entry_id": item.provider_entry_id,
                }
                for item in source.native_members
            ],
            "native_evidence_sha256": hashlib.sha256(
                source.native_evidence
            ).hexdigest(),
        }
    )
    return parse_canonical_object(payload, name="inference source projection")


def _workload_snapshot(value: ModalInferenceWorkloadBinding) -> tuple[object, ...]:
    return tuple(
        bytes(field) if type(field) is bytes else field
        for name in value.__dataclass_fields__
        if name != "_token"
        for field in (getattr(value, name),)
    )


def _workload_projection(value: ModalInferenceWorkloadBinding) -> dict[str, object]:
    return {
        name: hashlib.sha256(field).hexdigest() if name == "workload_bytes" else field
        for name in value.__dataclass_fields__
        if name != "_token"
        for field in (getattr(value, name),)
    }


def _authenticate_config(
    configuration, trust, authenticator, now
) -> ModalInferencePreparationConfig:
    if (
        type(configuration) is not AuthenticatedModalInferencePreparationConfig
        or type(trust) is not TrustedEvidenceIdentity
    ):
        raise TypeError("exact inference evidence inputs required")
    body = ModalInferencePreparationConfig.parse(configuration.body_bytes)
    evidence = body.document["evidence"]
    if (evidence["issuer_ref"], evidence["key_ref"], evidence["audience_ref"]) != (
        trust.issuer_ref,
        trust.key_ref,
        trust.audience_ref,
    ):
        raise ValueError("inference evidence trust differs")
    validate_evidence_window(
        verified_at=evidence["verified_at"],
        expires_at=evidence["expires_at"],
        now=now,
        policy=DEPLOYMENT_EVIDENCE_POLICY,
    )
    if (
        authenticator.verify(
            CONFIG_EVIDENCE_PURPOSE,
            body.canonical_bytes,
            configuration.tag,
            evidence["key_ref"],
        )
        is not True
    ):
        raise ValueError("inference evidence authentication failed")
    return body


def _authenticate_quote(quote, trust, authenticator, now) -> ModalQuoteBody:
    if (
        type(quote) is not AuthenticatedModalQuote
        or type(trust) is not TrustedEvidenceIdentity
    ):
        raise TypeError("exact quote evidence inputs required")
    body = ModalQuoteBody.parse(quote.body_bytes)
    if (body.issuer_ref, body.key_ref, body.audience_ref) != (
        trust.issuer_ref,
        trust.key_ref,
        trust.audience_ref,
    ):
        raise ValueError("quote trust differs")
    validate_evidence_window(
        verified_at=body.issued_at,
        expires_at=body.expires_at,
        now=now,
        policy=QUOTE_EVIDENCE_POLICY,
    )
    if not parse_utc(body.issued_at) <= parse_utc(now) < parse_utc(body.expires_at):
        raise ValueError("quote is outside validity window")
    if (
        authenticator.verify(
            QUOTE_PURPOSE, body.canonical_bytes, quote.tag, body.key_ref
        )
        is not True
    ):
        raise ValueError("quote authentication failed")
    return body


def _derive_preparation(chat_input: dict[str, object], executor: ExecutorDescriptorV1):
    source = chat_input["source"]
    configuration = chat_input["configuration"]
    quote = ModalQuoteBody.parse(canonical_bytes(chat_input["quote"]))
    session_id = chat_input["session_id"]
    provider = configuration["provider"]
    client = configuration["client"]
    volumes = configuration["volumes"]
    runtime_digest = domain_digest(
        "synaptic-modal-inference-runtime/v1",
        canonical_bytes(
            {"image": configuration["image"], "runtime": configuration["runtime"]}
        ),
    )
    resource_digest = domain_digest(
        "synaptic-modal-inference-resource/v1",
        canonical_bytes(
            {
                "resources": configuration["resources"],
                "volumes": volumes,
                "application": configuration["application"],
            }
        ),
    )
    workload_digest = domain_digest(
        "synaptic-modal-chat-input/v1", canonical_bytes(chat_input)
    )
    artifact_digest = domain_digest(
        "synaptic-modal-chat-artifact-contract/v1",
        canonical_bytes(
            {
                "artifact_volume_id": source["artifact_volume_id"],
                "members": source["members"],
            }
        ),
    )
    execution_digest = domain_digest(
        "synaptic-modal-chat-execution-binding/v1",
        canonical_bytes(
            {
                "client": client,
                "application": configuration["application"],
                "image": configuration["image"],
                "volumes": volumes,
                "executor": executor.to_dict(),
            }
        ),
    )
    plan_fingerprint = domain_digest(
        "synaptic-modal-chat-plan/v1",
        canonical_bytes(
            {
                "source_digest": source["artifact_source_digest"],
                "workload_digest": workload_digest,
                "runtime_digest": runtime_digest,
                "resource_digest": resource_digest,
                "artifact_contract_digest": artifact_digest,
                "quote_digest": quote.quote_digest,
                "policy": configuration["policy"],
                "execution_binding_digest": execution_digest,
            }
        ),
    )
    return CanonicalPreparationV2.build(
        provider=ProviderRef(provider["provider_id"], provider["profile_ref"]),
        scope=ExecutionScopeV1(source["provider"][2], source["provider"][3]),
        project_ref=source["run"]["project_ref"],
        run_id=session_id,
        plan_fingerprint=plan_fingerprint,
        source_digest=source["artifact_source_digest"],
        workload_digest=workload_digest,
        runtime_digest=runtime_digest,
        resource_digest=resource_digest,
        artifact_contract_digest=artifact_digest,
        quote_digest=quote.quote_digest,
        secret_requirements_digest=domain_digest(
            "synaptic-modal-secret-requirements/v1",
            canonical_bytes(configuration["secrets"]),
        ),
        execution_binding_digest=execution_digest,
    )


def _validate_preparation_snapshot(raw: bytes) -> dict[str, object]:
    """Reconstruct retained content, not fresh source or evidence authority.

    Raw workload/native evidence is not duplicated here. Their original hashes
    remain commitments authenticated by the consumer's complete-content catalog.
    """
    if type(raw) is not bytes or not raw:
        raise TypeError("exact inference preparation bytes required")
    document = parse_canonical_object(raw, name="inference preparation")
    exact_fields(
        document,
        frozenset(
            {
                "schema_version",
                "preparation",
                "executor",
                "configuration",
                "configuration_digest",
                "configuration_tag",
                "quote",
                "quote_tag",
                "chat_input",
                "chat_input_sha256",
            }
        ),
        "inference preparation",
    )
    if document["schema_version"] != "synaptic-modal-inference-preparation/v1":
        raise ValueError("unsupported inference preparation")
    preparation = CanonicalPreparationV2.parse(canonical_bytes(document["preparation"]))
    executor = ExecutorDescriptorV1(**document["executor"])
    configuration_bytes = canonical_bytes(document["configuration"])
    configuration = ModalInferencePreparationConfig.parse(configuration_bytes)
    quote = ModalQuoteBody.parse(canonical_bytes(document["quote"]))
    chat_input = document["chat_input"]
    exact_fields(
        chat_input,
        frozenset(
            {
                "schema_version",
                "session_id",
                "source",
                "workload",
                "configuration",
                "quote",
            }
        ),
        "chat input",
    )
    if chat_input["schema_version"] != "synaptic-modal-chat-input/v1":
        raise ValueError("unsupported chat input")
    source = chat_input["source"]
    workload = chat_input["workload"]
    if type(source) is not dict or type(workload) is not dict:
        raise TypeError("chat source and workload must be exact objects")
    exact_fields(
        source,
        frozenset(
            {
                "run",
                "artifacts",
                "read_request_sha256",
                "read_request_digest",
                "source_workflow_record_digest",
                "source_revision",
                "manifest_digest",
                "artifact_source_digest",
                "provider",
                "provider_job_ref",
                "effect_id",
                "provider_run_binding_digest",
                "command_binding_digest",
                "artifact_volume_id",
                "members",
                "native_evidence_sha256",
            }
        ),
        "chat source",
    )
    expected_workload_fields = set(ModalInferenceWorkloadBinding.__dataclass_fields__)
    expected_workload_fields.remove("_token")
    exact_fields(workload, frozenset(expected_workload_fields), "chat workload")
    for name in (
        "read_request_sha256",
        "read_request_digest",
        "source_workflow_record_digest",
        "manifest_digest",
        "artifact_source_digest",
        "provider_run_binding_digest",
        "command_binding_digest",
        "native_evidence_sha256",
    ):
        digest_text(source[name], name)
    exact_integer(source["source_revision"], "source_revision")
    for name in ("provider_job_ref", "effect_id", "artifact_volume_id"):
        safe_ref(source[name], name)
    workload_refs = {
        "run_id",
        "project_ref",
        "provider_job_ref",
        "submit_effect_id",
        "stage_effect_id",
        "control_volume_id",
        "artifact_volume_id",
        "key_ref",
        "model_ref",
    }
    for name in workload_refs:
        safe_ref(workload[name], name)
    for name in (
        expected_workload_fields
        - workload_refs
        - {
            "workload_size",
            "model_revision",
            "tokenizer_revision",
            "load_in_4bit",
        }
    ):
        digest_text(workload[name], name)
    if (
        type(source["run"]) is not dict
        or type(source["provider"]) is not list
        or len(source["provider"]) != 4
        or type(source["members"]) is not list
        or type(source["artifacts"]) is not list
        or any(type(item) is not dict for item in source["members"])
        or any(type(item) is not dict for item in source["artifacts"])
        or workload["run_id"] != source["run"].get("run_id")
        or workload["project_ref"] != source["run"].get("project_ref")
        or workload["read_request_digest"] != source["read_request_digest"]
        or workload["manifest_digest"] != source["manifest_digest"]
        or workload["artifact_source_digest"] != source["artifact_source_digest"]
        or workload["provider_job_ref"] != source["provider_job_ref"]
        or workload["submit_effect_id"] != source["effect_id"]
        or workload["artifact_volume_id"] != source["artifact_volume_id"]
        or workload["native_evidence_sha256"] != source["native_evidence_sha256"]
    ):
        raise ValueError("chat source and workload projections differ")
    for value in source["provider"]:
        safe_ref(value, "source provider identity")
    run = TrainingRunRef.from_dict(source["run"])
    artifacts = tuple(VerifiedArtifact.from_dict(item) for item in source["artifacts"])
    for item in source["members"]:
        exact_fields(
            item,
            frozenset({"role", "path", "size", "sha256", "provider_entry_id"}),
            "native artifact member",
        )
    members = tuple(
        ArtifactMemberV1(
            ArtifactRole(item["role"]),
            item["path"],
            item["size"],
            item["sha256"],
            item["provider_entry_id"],
        )
        for item in source["members"]
    )
    try:
        configuration_tag = bytes.fromhex(document["configuration_tag"])
        quote_tag = bytes.fromhex(document["quote_tag"])
    except (TypeError, ValueError):
        raise ValueError("inference evidence tags are invalid") from None
    if (
        run.to_dict() != source["run"]
        or len(artifacts) != len(EXACT_ARTIFACT_ROLES)
        or {item.role for item in artifacts}
        != {role.value for role in EXACT_ARTIFACT_ROLES}
        or len(members) != len(EXACT_ARTIFACT_ROLES)
        or {item.role for item in members} != EXACT_ARTIFACT_ROLES
        or tuple((item.role.value, item.size, item.sha256) for item in members)
        != tuple((item.role, item.size_bytes, item.sha256) for item in artifacts)
        or not configuration_tag
        or len(configuration_tag) > 128
        or not quote_tag
        or len(quote_tag) > 128
        or configuration_tag.hex() != document["configuration_tag"]
        or quote_tag.hex() != document["quote_tag"]
        or type(workload["workload_size"]) is not int
        or workload["workload_size"] <= 0
        or workload["workload_bytes"] != workload["workload_sha256"]
        or type(workload["load_in_4bit"]) is not bool
        or type(workload["model_revision"]) is not str
        or _REVISION.fullmatch(workload["model_revision"]) is None
        or workload["tokenizer_revision"] != workload["model_revision"]
    ):
        raise ValueError("chat artifact projection is invalid")
    for member in members:
        if (
            member.path
            != f"operations/{source['effect_id']}/output/{member.role.value}"
            or member.provider_entry_id
            != provider_entry_identity(
                source["artifact_volume_id"], member.path, member.size
            )
            or (
                member.role is ArtifactRole.WORKLOAD_RECORD
                and (
                    member.size != workload["workload_size"]
                    or member.sha256 != workload["workload_sha256"]
                )
            )
        ):
            raise ValueError("native artifact placement differs from chat source")
    expected_preparation = _derive_preparation(chat_input, executor)
    provider = configuration.document["provider"]
    client = configuration.document["client"]
    resource_digest = domain_digest(
        "synaptic-modal-inference-resource/v1",
        canonical_bytes(
            {
                "resources": configuration.document["resources"],
                "volumes": configuration.document["volumes"],
                "application": configuration.document["application"],
            }
        ),
    )
    if (
        type(chat_input) is not dict
        or hashlib.sha256(canonical_bytes(chat_input)).hexdigest()
        != document["chat_input_sha256"]
        or domain_digest(CONFIG_EVIDENCE_PURPOSE, configuration_bytes)
        != document["configuration_digest"]
        or chat_input.get("configuration") != configuration.document
        or chat_input.get("quote")
        != parse_canonical_object(quote.canonical_bytes, name="quote")
        or preparation.quote_digest != quote.quote_digest
        or preparation.provider.provider_id != executor.provider_id
        or executor.executor_id != "modal-chat-executor"
        or preparation != expected_preparation
        or source["provider"][0] != provider["provider_id"]
        or configuration.document["volumes"]["source_artifact_volume_id"]
        != source["artifact_volume_id"]
        or (
            quote.provider_id,
            quote.profile_ref,
            quote.account_ref,
            quote.namespace_ref,
            quote.resource_digest,
        )
        != (
            provider["provider_id"],
            provider["profile_ref"],
            source["provider"][2],
            source["provider"][3],
            resource_digest,
        )
        or source["provider"][2] != client["account_ref"]
        or domain_digest(
            "synaptic-modal-namespace/v1",
            canonical_bytes(
                {
                    "workspace_ref": client["workspace_ref"],
                    "environment_ref": client["environment_ref"],
                }
            ),
        )
        != source["provider"][3]
    ):
        raise ValueError("inference preparation projection differs")
    return document


class ModalInferencePreparation:
    __slots__ = ("_snapshot", "_token")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("ModalInferencePreparation is final")

    def __setattr__(self, name, value):
        if hasattr(self, "_token"):
            raise AttributeError("Modal inference preparation is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, snapshot, _token=None):
        if _token is not _TOKEN:
            raise TypeError("Modal inference preparations are factory issued")
        object.__setattr__(self, "_snapshot", bytes(snapshot))
        object.__setattr__(self, "_token", _TOKEN)

    def _document(self) -> dict[str, object]:
        return _validate_preparation_snapshot(self._snapshot)

    @property
    def preparation(self) -> CanonicalPreparationV2:
        document = self._document()
        return CanonicalPreparationV2.parse(canonical_bytes(document["preparation"]))

    @property
    def executor(self) -> ExecutorDescriptorV1:
        document = self._document()
        return ExecutorDescriptorV1(**document["executor"])

    @property
    def canonical_bytes(self) -> bytes:
        self._document()
        return bytes(self._snapshot)

    def stage(self, invocation_nonce: str) -> StageCommandV2:
        payload = CanonicalProviderPayloadV1.build(
            "modal", "stage-payload/v2", self.preparation.workload_digest
        )
        return build_stage_command(
            self.preparation, invocation_nonce, payload, self.executor
        )

    def submit(
        self, invocation_nonce: str, predecessor: StagePredecessorV2
    ) -> SubmitCommandV2:
        if type(predecessor) is not StagePredecessorV2:
            raise TypeError("exact stage predecessor required")
        prep = self.preparation
        stage_effect = derive_effect(prep, EffectKind.STAGE)
        if (
            predecessor.provider_id != prep.provider.provider_id
            or predecessor.profile_ref != prep.provider.profile_ref
            or predecessor.account_ref != prep.scope.account_ref
            or predecessor.namespace_ref != prep.scope.namespace_ref
            or predecessor.project_ref != prep.project_ref
            or predecessor.run_id != prep.run_id
            or predecessor.plan_fingerprint != prep.plan_fingerprint
            or predecessor.preparation_digest != prep.preparation_digest
            or predecessor.workload_digest != prep.workload_digest
            or predecessor.stage_effect_id != stage_effect.effect_id
        ):
            raise ValueError("stage predecessor differs from chat preparation")
        payload = CanonicalProviderPayloadV1.build(
            "modal", "submit-payload/v2", prep.workload_digest
        )
        return build_submit_command(
            prep, invocation_nonce, payload, self.executor, predecessor
        )


def _prepare(
    source,
    workload,
    configuration,
    configuration_tag,
    quote,
    quote_tag,
    session_id,
    executor_version,
):
    if (
        type(source) is not ModalInferenceSourceBinding
        or type(workload) is not ModalInferenceWorkloadBinding
    ):
        raise TypeError("exact fresh inference bindings required")
    safe_ref(session_id, "session_id")
    safe_ref(executor_version, "executor_version")
    source_snapshot = _source_snapshot(source)
    workload_snapshot = _workload_snapshot(workload)
    source_projection = _source_projection(source)
    workload_projection = _workload_projection(workload)
    workload_document = parse_canonical_object(
        workload.workload_bytes, name="inference workload"
    )
    compiled = CompiledWorkload(
        workload_document.get("method"),
        workload_document.get("schema_version"),
        workload_document.get("entrypoint"),
        bytes(workload.workload_bytes),
    )
    workload_member = next(
        (
            item
            for item in source.native_members
            if item.role.value == "workload_record"
        ),
        None,
    )
    if (
        workload.run_id != source.run.run_id
        or workload.project_ref != source.run.project_ref
        or workload.read_request_digest != source.read_request_digest
        or workload.manifest_digest != source.manifest_digest
        or workload.artifact_source_digest != source.artifact_source_digest
        or workload.provider_job_ref != source.provider_job_ref
        or workload.submit_effect_id != source.effect_id
        or workload.submit_command_digest != source.command_digest
        or workload.preparation_digest != source.preparation_digest
        or workload.workload_digest != source.workload_digest
        or workload.artifact_volume_id != source.artifact_volume_id
        or workload.native_evidence_sha256
        != hashlib.sha256(source.native_evidence).hexdigest()
        or workload.workload_size != len(workload.workload_bytes)
        or workload.workload_sha256
        != hashlib.sha256(workload.workload_bytes).hexdigest()
        or workload.workload_digest != compiled.fingerprint
        or _model(compiled)
        != (
            workload.model_ref,
            workload.model_revision,
            workload.tokenizer_revision,
            workload.load_in_4bit,
        )
        or workload_member is None
        or workload_member.path
        != f"operations/{source.effect_id}/output/workload_record"
        or workload_member.size != workload.workload_size
        or workload_member.sha256 != workload.workload_sha256
    ):
        raise ValueError("source and workload bindings differ")
    document = configuration.document
    provider = document["provider"]
    client = document["client"]
    volumes = document["volumes"]
    selection = source.command_binding.deployment.selection
    if (
        provider["provider_id"] != source.provider_id
        or client
        != {
            "account_ref": source.command_binding.client_binding.account_ref,
            "workspace_ref": source.command_binding.client_binding.workspace_ref,
            "environment_ref": source.command_binding.client_binding.environment_ref,
            "client_ref": source.command_binding.client_binding.client_ref,
            "sdk_version": source.command_binding.client_binding.sdk_version,
        }
        or domain_digest(
            "synaptic-modal-namespace/v1",
            canonical_bytes(
                {
                    "workspace_ref": client["workspace_ref"],
                    "environment_ref": client["environment_ref"],
                }
            ),
        )
        != source.namespace_ref
        or volumes["source_artifact_volume_id"] != source.artifact_volume_id
        or document["application"]["app_name"] == selection.app_name
    ):
        raise ValueError("inference configuration scope differs")
    executor = ExecutorDescriptorV1("modal", "modal-chat-executor", executor_version)
    resource_digest = domain_digest(
        "synaptic-modal-inference-resource/v1",
        canonical_bytes(
            {
                "resources": document["resources"],
                "volumes": document["volumes"],
                "application": document["application"],
            }
        ),
    )
    if (
        quote.provider_id,
        quote.profile_ref,
        quote.account_ref,
        quote.namespace_ref,
        quote.resource_digest,
    ) != (
        provider["provider_id"],
        provider["profile_ref"],
        source.account_ref,
        source.namespace_ref,
        resource_digest,
    ):
        raise ValueError("quote does not bind inference resources")
    chat_input = canonical_bytes(
        {
            "schema_version": "synaptic-modal-chat-input/v1",
            "session_id": session_id,
            "source": source_projection,
            "workload": workload_projection,
            "configuration": document,
            "quote": parse_canonical_object(quote.canonical_bytes, name="quote"),
        }
    )
    preparation = _derive_preparation(
        parse_canonical_object(chat_input, name="chat input"), executor
    )
    snapshot = canonical_bytes(
        {
            "schema_version": "synaptic-modal-inference-preparation/v1",
            "preparation": preparation.to_dict(),
            "executor": executor.to_dict(),
            "configuration": document,
            "configuration_digest": domain_digest(
                CONFIG_EVIDENCE_PURPOSE, configuration.canonical_bytes
            ),
            "quote": parse_canonical_object(quote.canonical_bytes, name="quote"),
            "configuration_tag": configuration_tag.hex(),
            "quote_tag": quote_tag.hex(),
            "chat_input": parse_canonical_object(chat_input, name="chat input"),
            "chat_input_sha256": hashlib.sha256(chat_input).hexdigest(),
        }
    )
    if (
        _source_snapshot(source) != source_snapshot
        or _workload_snapshot(workload) != workload_snapshot
    ):
        raise ValueError("inference inputs changed during preparation")
    return ModalInferencePreparation(snapshot=snapshot, _token=_TOKEN)


def prepare_modal_chat(
    source: ModalInferenceSourceBinding,
    workload: ModalInferenceWorkloadBinding,
    *,
    configuration: AuthenticatedModalInferencePreparationConfig,
    configuration_trust: TrustedEvidenceIdentity,
    quote: AuthenticatedModalQuote,
    quote_trust: TrustedEvidenceIdentity,
    evidence_authenticator: EvidenceAuthenticator,
    clock: CoordinatorClockPortV1,
    session_id: str,
    executor_version: str,
) -> ModalInferencePreparation:
    try:
        if (
            type(source) is not ModalInferenceSourceBinding
            or type(workload) is not ModalInferenceWorkloadBinding
            or type(configuration) is not AuthenticatedModalInferencePreparationConfig
            or type(configuration_trust) is not TrustedEvidenceIdentity
            or type(quote) is not AuthenticatedModalQuote
            or type(quote_trust) is not TrustedEvidenceIdentity
        ):
            raise TypeError("exact inference preparation inputs required")
        if not isinstance(evidence_authenticator, EvidenceAuthenticator):
            raise TypeError("evidence authenticator is required")
        now_method = getattr_static(type(clock), "now_iso", None)
        if (
            now_method is None
            or not callable(now_method)
            or getattr_static(clock, "now_iso", None) is not now_method
        ):
            raise TypeError("coordinator clock is required")
        owned_configuration = AuthenticatedModalInferencePreparationConfig(
            bytes(configuration.body_bytes), bytes(configuration.tag)
        )
        owned_quote = AuthenticatedModalQuote(bytes(quote.body_bytes), bytes(quote.tag))
        owned_configuration_trust = TrustedEvidenceIdentity(
            configuration_trust.issuer_ref,
            configuration_trust.key_ref,
            configuration_trust.audience_ref,
        )
        owned_quote_trust = TrustedEvidenceIdentity(
            quote_trust.issuer_ref, quote_trust.key_ref, quote_trust.audience_ref
        )
        initial_source = _source_snapshot(source)
        initial_workload = _workload_snapshot(workload)
        initial_configuration = bytes(configuration.body_bytes)
        initial_configuration_tag = bytes(configuration.tag)
        initial_quote = bytes(quote.body_bytes)
        initial_quote_tag = bytes(quote.tag)
        initial_configuration_trust = (
            configuration_trust.issuer_ref,
            configuration_trust.key_ref,
            configuration_trust.audience_ref,
        )
        initial_quote_trust = (
            quote_trust.issuer_ref,
            quote_trust.key_ref,
            quote_trust.audience_ref,
        )

        def unchanged() -> bool:
            return (
                _source_snapshot(source) == initial_source
                and _workload_snapshot(workload) == initial_workload
                and configuration.body_bytes == initial_configuration
                and configuration.tag == initial_configuration_tag
                and quote.body_bytes == initial_quote
                and quote.tag == initial_quote_tag
                and (
                    configuration_trust.issuer_ref,
                    configuration_trust.key_ref,
                    configuration_trust.audience_ref,
                )
                == initial_configuration_trust
                and (
                    quote_trust.issuer_ref,
                    quote_trust.key_ref,
                    quote_trust.audience_ref,
                )
                == initial_quote_trust
            )

        now = now_method(clock)
        if not unchanged():
            raise ValueError("inference preparation inputs changed during clock read")
        config = _authenticate_config(
            owned_configuration,
            owned_configuration_trust,
            evidence_authenticator,
            now,
        )
        if not unchanged():
            raise ValueError(
                "inference preparation inputs changed during authentication"
            )
        quote_body = _authenticate_quote(
            owned_quote, owned_quote_trust, evidence_authenticator, now
        )
        if not unchanged():
            raise ValueError(
                "inference preparation inputs changed during authentication"
            )
        result = _prepare(
            source,
            workload,
            config,
            owned_configuration.tag,
            quote_body,
            owned_quote.tag,
            session_id,
            executor_version,
        )
        if not unchanged():
            raise ValueError("inference preparation inputs changed during construction")
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise ModalInferencePreparationError(
            "modal_inference_preparation_invalid"
        ) from None


__all__: list[str] = []
